#!/usr/bin/env python3
"""Import an allowlisted, frozen PPO source snapshot into the training package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


FILE_MAP_SCHEMA = "lunar-ppo-file-map/v1"
IMPORT_RESULT_SCHEMA = "lunar-ppo-import-result/v1"
PACKAGE_ROOT = PurePosixPath("training/lunar_policy_training/lunar_policy_training")
FORBIDDEN_DEPENDENCY_PARTS = {
    "workflows",
    "stage",
    "contentref",
    "authority",
    "repair",
    "artifact_registry",
}


class ImportError(RuntimeError):
    """The frozen PPO import boundary was violated."""


def _git(source_git: Path, *arguments: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(source_git), *arguments],
        check=False,
        capture_output=True,
        text=text,
    )
    if result.returncode:
        stderr = result.stderr if text else result.stderr.decode(errors="replace")
        raise ImportError(f"git {' '.join(arguments)} failed: {stderr.strip()}")
    return result.stdout


def _load_document(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ImportError(f"expected object in {path}")
    return document


def _safe_relative_path(raw: object, *, field: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw:
        raise ImportError(f"unsafe {field}: {raw}")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
        raise ImportError(f"unsafe {field}: {raw}")
    return path


def _resolve_inside(root: Path, relative: PurePosixPath, *, field: str) -> Path:
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ImportError(f"{field} escapes repository root: {relative}")
    return resolved


def _legacy_module(source: PurePosixPath) -> str:
    prefix = PurePosixPath("src/lunar_exploration_ppo")
    return ".".join(source.relative_to(prefix).with_suffix("").parts)


def _relative_import(target: PurePosixPath, imported_target: PurePosixPath, symbol: str) -> str:
    target_parent = target.parent
    imported_module = imported_target.with_suffix("")
    common = 0
    while common < min(len(target_parent.parts), len(imported_module.parts)) and target_parent.parts[common] == imported_module.parts[common]:
        common += 1
    upwards = len(target_parent.parts) - common
    module_tail = imported_module.parts[common:]
    dots = "." * (upwards + 1)
    module = ".".join(module_tail)
    return f"from {dots}{module} import {symbol}"


def _validate_dependencies(
    *, source_path: PurePosixPath,
    target_path: PurePosixPath,
    text: str,
    legacy_targets: dict[str, PurePosixPath],
    allow_dependencies: set[str],
    type_checking_dependencies: set[str],
) -> str:
    try:
        tree = ast.parse(text, filename=source_path.as_posix())
    except SyntaxError as error:
        raise ImportError(f"invalid Python source: {source_path}: {error.msg}") from error
    replacements: dict[str, str] = {}
    needs_type_checking = False
    importlib_names = {"importlib"}
    dynamic_import_names = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        dynamic_import_names.add(alias.asname or alias.name)
            if node.module == "builtins":
                for alias in node.names:
                    if alias.name == "__import__":
                        dynamic_import_names.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _validate_module(alias.name, source_path, legacy_targets, allow_dependencies)
                if alias.name.startswith("lunar_exploration_ppo."):
                    raise ImportError(
                        f"absolute legacy import in {source_path}: {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if node.level or module is None:
                continue
            if module in type_checking_dependencies:
                _validate_type_checking_dependency(module, source_path)
                original = ast.get_source_segment(text, node)
                if original is None:
                    raise ImportError(f"cannot rewrite type-only dependency in {source_path}")
                aliases = "\n".join(
                    f"{' ' * node.col_offset}    from typing import Any as {alias.asname or alias.name}"
                    for alias in node.names
                )
                replacements[original] = f"if TYPE_CHECKING:\n{aliases}"
                needs_type_checking = True
                continue
            _validate_module(module, source_path, legacy_targets, allow_dependencies)
            if module.removeprefix("lunar_exploration_ppo.") in legacy_targets:
                symbols = ", ".join(alias.name + (f" as {alias.asname}" if alias.asname else "") for alias in node.names)
                original = ast.get_source_segment(text, node)
                if original is None:
                    raise ImportError(f"cannot rewrite legacy import in {source_path}")
                replacements[original] = _relative_import(
                    target_path,
                    legacy_targets[module.removeprefix("lunar_exploration_ppo.")],
                    symbols,
                )
        elif isinstance(node, ast.Call):
            module = _dynamic_import_module_name(
                node, importlib_names, dynamic_import_names, source_path
            )
            if module is None:
                continue
            _validate_module(module, source_path, legacy_targets, allow_dependencies)
            if module.startswith("lunar_exploration_ppo."):
                raise ImportError(f"dynamic legacy import in {source_path}: {module}")
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    if needs_type_checking:
        lines = text.splitlines(keepends=True)
        insertion = 0
        body = list(tree.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            insertion = body.pop(0).end_lineno or 0
        while body and isinstance(body[0], ast.ImportFrom) and body[0].module == "__future__":
            insertion = body.pop(0).end_lineno or insertion
        lines.insert(insertion, "from typing import TYPE_CHECKING\n")
        text = "".join(lines)
    return text


def _dynamic_import_module_name(
    node: ast.Call,
    importlib_names: set[str],
    dynamic_import_names: set[str],
    source_path: PurePosixPath,
) -> str | None:
    is_dynamic_import = (
        isinstance(node.func, ast.Name) and node.func.id in dynamic_import_names
    ) or (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in importlib_names
    )
    if not is_dynamic_import:
        return None
    if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
        raise ImportError(f"dynamic import in {source_path} must use a constant module name")
    return node.args[0].value


def _validate_type_checking_dependency(module: str, source_path: PurePosixPath) -> None:
    parts = {part.lower() for part in module.split(".")}
    if parts & FORBIDDEN_DEPENDENCY_PARTS:
        raise ImportError(f"forbidden dependency in {source_path}: {module}")
    if not module.startswith("lunar_exploration_ppo."):
        raise ImportError(f"type-only dependency must be legacy module in {source_path}: {module}")


def _validate_module(
    module: str,
    source_path: PurePosixPath,
    legacy_targets: dict[str, PurePosixPath],
    allow_dependencies: set[str],
) -> None:
    parts = {part.lower() for part in module.split(".")}
    if parts & FORBIDDEN_DEPENDENCY_PARTS:
        raise ImportError(f"forbidden dependency in {source_path}: {module}")
    if module.startswith("lunar_exploration_ppo."):
        legacy = module.removeprefix("lunar_exploration_ppo.")
        if legacy not in legacy_targets:
            raise ImportError(f"unmapped dependency in {source_path}: {module}")
        return
    root = module.split(".", maxsplit=1)[0]
    if root not in allow_dependencies and root not in sys.stdlib_module_names:
        raise ImportError(f"unlisted dependency in {source_path}: {module}")


def import_snapshot(
    *,
    source_git: Path,
    repository_root: Path,
    source_inventory_path: Path,
    file_map_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    inventory = _load_document(source_inventory_path)
    file_map = _load_document(file_map_path)
    if file_map.get("schema_version") != FILE_MAP_SCHEMA:
        raise ImportError("unsupported PPO file-map schema")
    source_contract = file_map.get("source")
    if not isinstance(source_contract, dict):
        raise ImportError("file map is missing source contract")
    repository = source_contract.get("repository")
    commit = source_contract.get("commit")
    inventory_repository = inventory.get("repositories", {}).get(repository, {})
    if commit != inventory_repository.get("commit"):
        raise ImportError("file-map commit does not match source inventory")
    actual_head = str(_git(source_git, "rev-parse", "HEAD")).strip()
    if actual_head != commit:
        raise ImportError(f"source HEAD mismatch: expected {commit}, got {actual_head}")
    actual_origin = str(_git(source_git, "remote", "get-url", "origin")).strip()
    if actual_origin != inventory_repository.get("origin"):
        raise ImportError("source origin mismatch: expected " + str(inventory_repository.get("origin")) + ", got " + actual_origin)

    target_root = _safe_relative_path(file_map.get("target_root"), field="target_root")
    if target_root != PACKAGE_ROOT:
        raise ImportError("target_root must be the lunar_policy_training package root")
    repository_root = repository_root.resolve()
    if not result_path.resolve().is_relative_to(repository_root):
        raise ImportError("result path must be inside repository root")
    groups = file_map.get("groups")
    if not isinstance(groups, dict):
        raise ImportError("file map is missing groups")
    allowed = file_map.get("allow_dependencies")
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise ImportError("file map is missing allow_dependencies")
    type_checking = file_map.get("type_checking_dependencies", [])
    if not isinstance(type_checking, list) or not all(isinstance(item, str) for item in type_checking):
        raise ImportError("type_checking_dependencies must be a list of module names")
    inventory_entries = {entry.get("path"): entry for entry in inventory.get("files", []) if entry.get("repository") == repository}

    mappings: list[tuple[str, PurePosixPath, PurePosixPath]] = []
    seen_sources: set[PurePosixPath] = set()
    seen_targets: set[PurePosixPath] = set()
    for group, entries in groups.items():
        if not isinstance(entries, list):
            raise ImportError(f"group {group} must be a list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ImportError(f"invalid mapping in {group}")
            source = _safe_relative_path(entry.get("source"), field="source")
            target = _safe_relative_path(entry.get("target"), field="target")
            if source in seen_sources or target in seen_targets:
                raise ImportError(f"duplicate source or target mapping: {source} -> {target}")
            seen_sources.add(source)
            seen_targets.add(target)
            if source.as_posix() not in inventory_entries:
                raise ImportError(f"source is not selected by inventory: {source}")
            mappings.append((str(group), source, target))
    legacy_targets = {_legacy_module(source): target for _, source, target in mappings}

    pending: list[tuple[str, PurePosixPath, bytes]] = []
    files: list[dict[str, Any]] = []
    for group, source, target in mappings:
        tree_line = str(_git(source_git, "ls-tree", str(commit), "--", source.as_posix())).strip()
        if not tree_line:
            raise ImportError(f"source is absent from frozen commit: {source}")
        metadata, found_path = tree_line.split("\t", maxsplit=1)
        mode, kind, _object_id = metadata.split()
        if found_path != source.as_posix() or kind != "blob" or mode not in {"100644", "100755"}:
            raise ImportError(f"source is not a regular blob: {source}")
        payload = bytes(_git(source_git, "show", f"{commit}:{source.as_posix()}", text=False))
        entry = inventory_entries[source.as_posix()]
        if len(payload) != entry.get("size_bytes"):
            raise ImportError(f"source size mismatch: {source}")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry.get("sha256"):
            raise ImportError(f"source SHA-256 mismatch: {source}")
        try:
            transformed = _validate_dependencies(
                source_path=source,
                target_path=target,
                text=payload.decode("utf-8"),
                legacy_targets=legacy_targets,
                allow_dependencies=set(allowed),
                type_checking_dependencies=set(type_checking),
            )
        except UnicodeDecodeError as error:
            raise ImportError(f"source is not UTF-8 text: {source}") from error
        pending.append((group, target, transformed.encode("utf-8")))
        files.append({"group": group, "source": source.as_posix(), "target": (target_root / target).as_posix(), "sha256": digest, "size_bytes": len(payload)})

    for _group, target, payload in pending:
        destination = _resolve_inside(repository_root, target_root / target, field="target")
        if destination.exists() and (not destination.is_file() or destination.read_bytes() != payload):
            raise ImportError(f"existing target differs from frozen blob: {target_root / target}")
    for _group, target, payload in pending:
        destination = _resolve_inside(repository_root, target_root / target, field="target")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    result = {"schema_version": IMPORT_RESULT_SCHEMA, "source_repository": repository, "source_commit": commit, "files": files}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = import_snapshot(source_git=arguments.source_git, repository_root=arguments.repository_root, source_inventory_path=arguments.inventory, file_map_path=arguments.map, result_path=arguments.result)
    except (ImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"PPO import error: {error}", file=sys.stderr)
        return 1
    print(f"imported {len(result['files'])} frozen PPO files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
