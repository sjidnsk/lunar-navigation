#!/usr/bin/env python3
"""Import the frozen planner v3 snapshot from verified Git blobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


FILE_MAP_SCHEMA = "lunar-v3-file-map/v1"
IMPORT_RESULT_SCHEMA = "lunar-v3-import-result/v1"
PACKAGE_ROOT = PurePosixPath("ros2_ws/src/lunar_planner_core")
FORBIDDEN_SOURCE_SUFFIXES = {
    ".a",
    ".dll",
    ".dylib",
    ".exe",
    ".lib",
    ".o",
    ".obj",
    ".so",
}


class ImportError(RuntimeError):
    """Raised when the controlled import boundary is violated."""


def _git(source_git: Path, *arguments: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ["git", "-C", str(source_git), *arguments],
        check=False,
        capture_output=True,
        text=text,
    )
    if completed.returncode != 0:
        stderr = completed.stderr if text else completed.stderr.decode(errors="replace")
        raise ImportError(f"git {' '.join(arguments)} failed: {stderr.strip()}")
    return completed.stdout


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ImportError(f"expected object in {path}")
    return document


def _load_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ImportError(f"expected object in {path}")
    return document


def _safe_relative_path(raw: str, *, field: str) -> PurePosixPath:
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


def import_snapshot(
    *,
    source_git: Path,
    repository_root: Path,
    source_inventory_path: Path,
    file_map_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    inventory = _load_json(source_inventory_path)
    file_map = _load_yaml(file_map_path)
    if file_map.get("schema_version") != FILE_MAP_SCHEMA:
        raise ImportError("unsupported v3 file-map schema")

    source_contract = file_map.get("source")
    if not isinstance(source_contract, dict):
        raise ImportError("file map is missing source contract")
    source_repository = source_contract.get("repository")
    source_commit = source_contract.get("commit")
    repositories = inventory.get("repositories", {})
    inventory_repository = repositories.get(source_repository, {})
    if source_commit != inventory_repository.get("commit"):
        raise ImportError("file-map commit does not match source inventory")

    actual_head = str(_git(source_git, "rev-parse", "HEAD")).strip()
    if actual_head != source_commit:
        raise ImportError(
            f"source HEAD mismatch: expected {source_commit}, got {actual_head}"
        )
    actual_origin = str(_git(source_git, "remote", "get-url", "origin")).strip()
    if actual_origin != inventory_repository.get("origin"):
        raise ImportError(
            f"source origin mismatch: expected {inventory_repository.get('origin')}, "
            f"got {actual_origin}"
        )

    inventory_entries = {
        entry["path"]: entry
        for entry in inventory.get("files", [])
        if entry.get("repository") == source_repository
    }
    target_root = _safe_relative_path(
        str(file_map.get("target_root", "")), field="target_root"
    )
    if not target_root.is_relative_to(PACKAGE_ROOT):
        raise ImportError("target_root must be inside lunar_planner_core")
    resolved_repository_root = repository_root.resolve()
    resolved_result = result_path.resolve()
    if not resolved_result.is_relative_to(resolved_repository_root):
        raise ImportError("result path must be inside repository root")

    pending: list[tuple[str, str, PurePosixPath, bytes, int]] = []
    result_files: list[dict[str, Any]] = []
    groups = file_map.get("groups")
    if not isinstance(groups, dict):
        raise ImportError("file map is missing groups")
    excluded_roots = file_map.get("exclude_roots")
    if not isinstance(excluded_roots, list) or not all(
        isinstance(value, str) and value for value in excluded_roots
    ):
        raise ImportError("file map is missing exclude_roots")
    seen_sources: set[PurePosixPath] = set()
    seen_targets: set[PurePosixPath] = set()
    for group, entries in groups.items():
        if not isinstance(entries, list):
            raise ImportError(f"group {group} must be a list")
        for entry in entries:
            source_path = _safe_relative_path(entry["source"], field="source")
            target_path = _safe_relative_path(entry["target"], field="target")
            if source_path in seen_sources or target_path in seen_targets:
                raise ImportError(
                    f"duplicate source or target mapping: {source_path} -> {target_path}"
                )
            seen_sources.add(source_path)
            seen_targets.add(target_path)
            if source_path.suffix.lower() in FORBIDDEN_SOURCE_SUFFIXES:
                raise ImportError(f"forbidden source suffix: {source_path}")
            if any(root in source_path.parts for root in excluded_roots):
                raise ImportError(f"excluded source root: {source_path}")
            if not target_path.is_relative_to(target_root):
                raise ImportError(f"target is outside target_root: {target_path}")
            inventory_entry = inventory_entries.get(source_path.as_posix())
            if inventory_entry is None:
                raise ImportError(f"source is not selected by inventory: {source_path}")

            tree_line = str(
                _git(source_git, "ls-tree", source_commit, "--", source_path.as_posix())
            ).strip()
            if not tree_line:
                raise ImportError(f"source is absent from frozen commit: {source_path}")
            metadata, actual_path = tree_line.split("\t", maxsplit=1)
            mode, object_type, _object_id = metadata.split()
            if actual_path != source_path.as_posix() or object_type != "blob":
                raise ImportError(f"source is not a regular blob: {source_path}")
            if mode not in {"100644", "100755"}:
                raise ImportError(f"source mode is not a regular file: {source_path} ({mode})")

            payload = bytes(
                _git(
                    source_git,
                    "show",
                    f"{source_commit}:{source_path.as_posix()}",
                    text=False,
                )
            )
            digest = hashlib.sha256(payload).hexdigest()
            if len(payload) != inventory_entry.get("size_bytes"):
                raise ImportError(f"source size mismatch: {source_path}")
            if digest != inventory_entry.get("sha256"):
                raise ImportError(f"source SHA-256 mismatch: {source_path}")
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ImportError(f"source is not UTF-8 text: {source_path}") from error
            if b"\x00" in payload or b"\r" in payload:
                raise ImportError(f"source is not canonical LF text: {source_path}")

            pending.append(
                (group, source_path.as_posix(), target_path, payload, int(mode, 8))
            )
            result_files.append(
                {
                    "group": group,
                    "source": source_path.as_posix(),
                    "target": target_path.as_posix(),
                    "sha256": digest,
                    "size_bytes": len(payload),
                }
            )

    mapped_targets = {target_path for _, _, target_path, _, _ in pending}
    resolved_target_root = _resolve_inside(
        repository_root, target_root, field="target_root"
    )
    if resolved_target_root.exists():
        for existing in resolved_target_root.rglob("*"):
            if not (existing.is_file() or existing.is_symlink()):
                continue
            relative = PurePosixPath(
                existing.relative_to(repository_root.resolve()).as_posix()
            )
            if relative not in mapped_targets:
                raise ImportError(f"unlisted target file: {relative}")

    for _group, _source, target_path, payload, _mode in pending:
        destination = _resolve_inside(repository_root, target_path, field="target")
        if destination.is_symlink():
            raise ImportError(f"mapped target is a symbolic link: {target_path}")
        if destination.exists() and (
            not destination.is_file() or destination.read_bytes() != payload
        ):
            raise ImportError(
                f"existing target differs from frozen blob: {target_path}"
            )

    for _group, _source, target_path, payload, mode in pending:
        destination = _resolve_inside(repository_root, target_path, field="target")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        os.chmod(destination, mode)

    result = {
        "schema_version": IMPORT_RESULT_SCHEMA,
        "source_repository": source_repository,
        "source_commit": source_commit,
        "files": result_files,
    }
    resolved_result.parent.mkdir(parents=True, exist_ok=True)
    resolved_result.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--source-inventory", required=True, type=Path)
    parser.add_argument("--file-map", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = import_snapshot(
            source_git=arguments.source_git,
            repository_root=arguments.repository_root,
            source_inventory_path=arguments.source_inventory,
            file_map_path=arguments.file_map,
            result_path=arguments.result,
        )
    except (ImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"v3 import error: {error}", file=sys.stderr)
        return 1
    print(f"imported {len(result['files'])} frozen planner v3 files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
