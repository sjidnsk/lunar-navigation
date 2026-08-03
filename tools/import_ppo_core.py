#!/usr/bin/env python3
"""Import an allowlisted, frozen PPO source snapshot into the training package."""

from __future__ import annotations

import argparse
import ast
import builtins
import copy
import hashlib
import json
import keyword
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


FILE_MAP_SCHEMA = "lunar-ppo-file-map/v1"
IMPORT_RESULT_SCHEMA = "lunar-ppo-import-result/v1"
PACKAGE_ROOT = PurePosixPath("training/lunar_policy_training/lunar_policy_training")
FORBIDDEN_DEPENDENCY_PARTS = {
    "workflows",
    "contentref",
    "authority",
    "repair",
    "artifact",
    "durable",
}
GLOBAL_FORBIDDEN_IMPORT_PARTS = {
    "workflows",
    "contentref",
    "authority",
    "repair",
    "durable",
}
FORBIDDEN_OLD_SYMBOLS = {
    "CrossAttentionFrontierPolicy",
    "PolicyBatch",
    "PolicyForwardOutput",
}
ALLOWED_MAPPING_MODES = {"extract", "manual_thin_adapter"}
BUILTIN_NAMES = frozenset(dir(builtins))
PURE_BUILTIN_CONSTANT_CALLS = {"frozenset"}
PURE_MODULE_CONSTANT_CALLS = {
    "math": {"expm1", "log"},
    "numpy": {"float32"},
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


def _relative_import_parts(
    target: PurePosixPath, imported_target: PurePosixPath
) -> tuple[int, str | None]:
    target_parent = target.parent
    imported_module = imported_target.with_suffix("")
    common = 0
    while common < min(len(target_parent.parts), len(imported_module.parts)) and target_parent.parts[common] == imported_module.parts[common]:
        common += 1
    upwards = len(target_parent.parts) - common
    module_tail = imported_module.parts[common:]
    return upwards + 1, ".".join(module_tail) or None


@dataclass(frozen=True)
class _StringReplacement:
    old: str
    new: str
    count: int


@dataclass(frozen=True)
class _MappingSpec:
    group: str
    source: PurePosixPath
    target: PurePosixPath
    sha256: str
    allow_symbols: tuple[str, ...]
    rename: dict[str, str]
    forbid_imports: tuple[str, ...]
    omit_imports: tuple[str, ...]
    string_replacements: tuple[_StringReplacement, ...]
    mode: str
    adaptation: dict[str, Any]


@dataclass(frozen=True)
class _ImportBinding:
    binding: str
    node: ast.Import | ast.ImportFrom
    alias: ast.alias
    module: str
    legacy_spec: _MappingSpec | None = None


@dataclass(frozen=True)
class _Scope:
    kind: str
    bindings: frozenset[str]
    globals: frozenset[str]
    nonlocals: frozenset[str]


class _BindingCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.bindings: set[str] = set()
        self.globals: set[str] = set()
        self.nonlocals: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            self.bindings.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bindings.add(alias.asname or alias.name.split(".", maxsplit=1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.bindings.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bindings.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.bindings.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bindings.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ListComp(self, node: ast.ListComp) -> None:
        return

    def visit_SetComp(self, node: ast.SetComp) -> None:
        return

    def visit_DictComp(self, node: ast.DictComp) -> None:
        return

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        return

    def visit_Global(self, node: ast.Global) -> None:
        self.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocals.update(node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.bindings.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self.bindings.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self.bindings.add(node.name)


def _argument_names(arguments: ast.arguments) -> set[str]:
    names = {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return names


def _scope_for_body(
    kind: str, body: list[ast.stmt], arguments: ast.arguments | None = None
) -> _Scope:
    collector = _BindingCollector()
    for statement in body:
        collector.visit(statement)
    bindings = collector.bindings | (_argument_names(arguments) if arguments else set())
    bindings -= collector.globals
    return _Scope(
        kind=kind,
        bindings=frozenset(bindings),
        globals=frozenset(collector.globals),
        nonlocals=frozenset(collector.nonlocals),
    )


class _GlobalLoadCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()
        self.scopes: list[_Scope] = []

    def _is_module_global(self, name: str) -> bool:
        for scope in reversed(self.scopes):
            if name in scope.globals:
                return True
            if name in scope.bindings or name in scope.nonlocals:
                return False
        return True

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and self._is_module_global(node.id):
            self.names.add(node.id)

    def _visit_arguments_in_outer_scope(self, arguments: ast.arguments) -> None:
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if arguments.vararg and arguments.vararg.annotation:
            self.visit(arguments.vararg.annotation)
        if arguments.kwarg and arguments.kwarg.annotation:
            self.visit(arguments.kwarg.annotation)
        for default in (*arguments.defaults, *arguments.kw_defaults):
            if default is not None:
                self.visit(default)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_arguments_in_outer_scope(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        outer_scopes = self.scopes
        self.scopes = [scope for scope in outer_scopes if scope.kind != "class"]
        self.scopes.append(_scope_for_body("function", node.body, node.args))
        for statement in node.body:
            self.visit(statement)
        self.scopes = outer_scopes

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_arguments_in_outer_scope(node.args)
        outer_scopes = self.scopes
        self.scopes = [scope for scope in outer_scopes if scope.kind != "class"]
        self.scopes.append(_scope_for_body("function", [], node.args))
        self.visit(node.body)
        self.scopes = outer_scopes

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword_node in node.keywords:
            self.visit(keyword_node.value)
        outer_scopes = self.scopes
        self.scopes = [scope for scope in outer_scopes if scope.kind != "class"]
        self.scopes.append(_scope_for_body("class", node.body))
        for statement in node.body:
            self.visit(statement)
        self.scopes = outer_scopes

    def _visit_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        if not node.generators:
            return
        self.visit(node.generators[0].iter)
        bindings: set[str] = set()
        for generator in node.generators:
            bindings.update(
                child.id
                for child in ast.walk(generator.target)
                if isinstance(child, ast.Name)
            )
        outer_scopes = self.scopes
        self.scopes = [scope for scope in outer_scopes if scope.kind != "class"]
        self.scopes.append(
            _Scope("function", frozenset(bindings), frozenset(), frozenset())
        )
        for index, generator in enumerate(node.generators):
            if index:
                self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.scopes = outer_scopes

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)


def _definition_names(node: ast.stmt) -> tuple[str, ...]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return (node.name,)
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return ()
        return (node.targets[0].id,)
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return (node.target.id,)
    return ()


def _definition_dependencies(node: ast.stmt) -> set[str]:
    collector = _GlobalLoadCollector()
    if isinstance(node, ast.Assign):
        collector.visit(node.value)
    elif isinstance(node, ast.AnnAssign):
        collector.visit(node.annotation)
        if node.value is not None:
            collector.visit(node.value)
    else:
        collector.visit(node)
    return collector.names


def _is_pure_constant_call(
    function: ast.AST,
    import_bindings: dict[str, _ImportBinding],
    definition_names: set[str],
) -> bool:
    if isinstance(function, ast.Name):
        if (
            function.id in PURE_BUILTIN_CONSTANT_CALLS
            and function.id not in import_bindings
            and function.id not in definition_names
        ):
            return True
        binding = import_bindings.get(function.id)
        return bool(
            binding
            and binding.legacy_spec is None
            and isinstance(binding.node, ast.ImportFrom)
            and binding.alias.name
            in PURE_MODULE_CONSTANT_CALLS.get(binding.module, set())
        )
    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
        binding = import_bindings.get(function.value.id)
        return bool(
            binding
            and binding.legacy_spec is None
            and isinstance(binding.node, ast.Import)
            and function.attr
            in PURE_MODULE_CONSTANT_CALLS.get(binding.module, set())
        )
    return False


def _is_safe_constant_expression(
    node: ast.AST | None,
    import_bindings: dict[str, _ImportBinding],
    definition_names: set[str],
) -> bool:
    if node is None:
        return True
    forbidden = (
        ast.Await,
        ast.GeneratorExp,
        ast.Lambda,
        ast.ListComp,
        ast.NamedExpr,
        ast.SetComp,
        ast.DictComp,
        ast.Yield,
        ast.YieldFrom,
    )
    for child in ast.walk(node):
        if isinstance(child, forbidden):
            return False
        if isinstance(child, ast.Call) and not _is_pure_constant_call(
            child.func, import_bindings, definition_names
        ):
            return False
    return True


def _dynamic_function_expression(
    node: ast.AST,
    importlib_names: set[str],
    builtins_names: set[str],
    dynamic_names: set[str],
    *,
    getattr_available: bool,
) -> bool:
    if isinstance(node, ast.Name):
        return node.id in dynamic_names
    if isinstance(node, ast.Attribute):
        if node.attr == "__import__":
            return True
        return (
            node.attr == "import_module"
            and isinstance(node.value, ast.Name)
            and node.value.id in importlib_names
        )
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and getattr_available
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value in {"__import__", "import_module"}
        and (
            not isinstance(node.args[0], ast.Name)
            or node.args[0].id in importlib_names | builtins_names
        )
    )


class _DynamicScopeCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: list[ast.Import | ast.ImportFrom] = []
        self.assignments: list[tuple[tuple[str, ...], ast.AST]] = []
        self.calls: list[ast.Call] = []
        self.children: list[
            ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
        ] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.append(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        names = tuple(
            target.id for target in node.targets if isinstance(target, ast.Name)
        )
        if names:
            self.assignments.append((names, node.value))
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value is not None:
            self.assignments.append(((node.target.id,), node.value))
            self.visit(node.value)
        self.visit(node.annotation)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        if isinstance(node.target, ast.Name):
            self.assignments.append(((node.target.id,), node.value))
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)

    def _visit_function_outer(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
    ) -> None:
        if not isinstance(node, ast.Lambda):
            for decorator in node.decorator_list:
                self.visit(decorator)
            if node.returns is not None:
                self.visit(node.returns)
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.args.vararg and node.args.vararg.annotation:
            self.visit(node.args.vararg.annotation)
        if node.args.kwarg and node.args.kwarg.annotation:
            self.visit(node.args.kwarg.annotation)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        self.children.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_outer(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_outer(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_function_outer(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword_node in node.keywords:
            self.visit(keyword_node.value)
        self.children.append(node)


def _inherited_dynamic_names(names: set[str], scope: _Scope) -> set[str]:
    return {
        name
        for name in names
        if name not in scope.bindings
        or name in scope.globals
        or name in scope.nonlocals
    }


def _reject_dynamic_execution(
    tree: ast.Module,
    source_path: PurePosixPath,
    legacy_specs: dict[str, _MappingSpec],
    allow_dependencies: set[str],
    forbid_imports: tuple[str, ...],
) -> None:
    def analyze_scope(
        body: list[ast.stmt],
        arguments: ast.arguments | None,
        parent_importlib: set[str],
        parent_builtins: set[str],
        parent_dynamic: set[str],
        *,
        kind: str,
        owner: str,
        expression: ast.expr | None = None,
    ) -> None:
        scope = _scope_for_body(kind, body, arguments)
        importlib_names = _inherited_dynamic_names(parent_importlib, scope)
        builtins_names = _inherited_dynamic_names(parent_builtins, scope)
        dynamic_names = _inherited_dynamic_names(parent_dynamic, scope)
        collector = _DynamicScopeCollector()
        for statement in body:
            collector.visit(statement)
        if expression is not None:
            collector.visit(expression)
        for import_node in collector.imports:
            if isinstance(import_node, ast.Import):
                for alias in import_node.names:
                    if alias.name == "importlib":
                        importlib_names.add(alias.asname or "importlib")
                    if alias.name == "builtins":
                        builtins_names.add(alias.asname or "builtins")
            elif import_node.module == "importlib":
                dynamic_names.update(
                    alias.asname or alias.name
                    for alias in import_node.names
                    if alias.name == "import_module"
                )
            elif import_node.module == "builtins":
                dynamic_names.update(
                    alias.asname or alias.name
                    for alias in import_node.names
                    if alias.name == "__import__"
                )
        getattr_available = (
            "getattr" not in scope.bindings
            or "getattr" in scope.globals
            or "getattr" in scope.nonlocals
        )
        changed = True
        while changed:
            changed = False
            for targets, value in collector.assignments:
                if _dynamic_function_expression(
                    value,
                    importlib_names,
                    builtins_names,
                    dynamic_names,
                    getattr_available=getattr_available,
                ):
                    for target in targets:
                        if target not in dynamic_names:
                            dynamic_names.add(target)
                            changed = True
                if isinstance(value, ast.Name) and value.id in importlib_names:
                    for target in targets:
                        if target not in importlib_names:
                            importlib_names.add(target)
                            changed = True
                if isinstance(value, ast.Name) and value.id in builtins_names:
                    for target in targets:
                        if target not in builtins_names:
                            builtins_names.add(target)
                            changed = True
        for call in collector.calls:
            if isinstance(call.func, ast.Name) and call.func.id in {"eval", "exec"}:
                raise ImportError(
                    f"{source_path}:{owner} -> forbidden dynamic execution: "
                    f"{call.func.id}"
                )
            if not _dynamic_function_expression(
                call.func,
                importlib_names,
                builtins_names,
                dynamic_names,
                getattr_available=getattr_available,
            ):
                continue
            if (
                call.args
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
            ):
                try:
                    _validate_module(
                        call.args[0].value,
                        source_path,
                        legacy_specs,
                        allow_dependencies,
                        forbid_imports,
                    )
                except ImportError as error:
                    raise ImportError(
                        f"dynamic import in {source_path}:{owner}; {error}"
                    ) from error
            raise ImportError(f"dynamic import in {source_path}:{owner}")
        for child in collector.children:
            child_owner = (
                child.name
                if owner == "<module>" and not isinstance(child, ast.Lambda)
                else owner
            )
            if isinstance(child, ast.Lambda):
                analyze_scope(
                    [],
                    child.args,
                    importlib_names,
                    builtins_names,
                    dynamic_names,
                    kind="function",
                    owner=child_owner,
                    expression=child.body,
                )
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                analyze_scope(
                    child.body,
                    child.args,
                    importlib_names,
                    builtins_names,
                    dynamic_names,
                    kind="function",
                    owner=child_owner,
                )
            else:
                analyze_scope(
                    child.body,
                    None,
                    importlib_names,
                    builtins_names,
                    dynamic_names,
                    kind="class",
                    owner=child_owner,
                )

    analyze_scope(
        tree.body,
        None,
        set(),
        set(),
        {"__import__"},
        kind="module",
        owner="<module>",
    )


def _forbidden_module_part(module: str) -> str | None:
    parts = tuple(part for part in re.split(r"[.\-_/]", module.lower()) if part)
    for part in parts:
        if part in GLOBAL_FORBIDDEN_IMPORT_PARTS or re.fullmatch(r"stage\d*", part):
            return part
    normalized_parts = tuple(re.sub(r"[^a-z0-9]", "", part) for part in parts)
    if "artifactregistry" in normalized_parts or any(
        left in {"artifact", "artifacts"} and right == "registry"
        for left, right in zip(normalized_parts, normalized_parts[1:])
    ):
        return "artifact-registry"
    return None


def _reject_forbidden_module(
    module: str,
    source_path: PurePosixPath,
    forbid_imports: tuple[str, ...],
) -> None:
    if any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in forbid_imports
    ):
        raise ImportError(f"forbidden dependency in {source_path}: {module}")
    if _forbidden_module_part(module):
        raise ImportError(f"forbidden dependency in {source_path}: {module}")


def _validate_module(
    module: str,
    source_path: PurePosixPath,
    legacy_specs: dict[str, _MappingSpec],
    allow_dependencies: set[str],
    forbid_imports: tuple[str, ...],
) -> _MappingSpec | None:
    _reject_forbidden_module(module, source_path, forbid_imports)
    if module.startswith("lunar_exploration_ppo."):
        legacy = module.removeprefix("lunar_exploration_ppo.")
        if legacy not in legacy_specs:
            raise ImportError(f"unmapped dependency in {source_path}: {module}")
        return legacy_specs[legacy]
    root = module.split(".", maxsplit=1)[0]
    if root not in allow_dependencies and root not in sys.stdlib_module_names:
        raise ImportError(f"unlisted dependency in {source_path}: {module}")
    return None


def _collect_import_bindings(
    *,
    tree: ast.Module,
    source_path: PurePosixPath,
    legacy_specs: dict[str, _MappingSpec],
    allow_dependencies: set[str],
    forbid_imports: tuple[str, ...],
    omit_imports: tuple[str, ...],
) -> tuple[dict[str, _ImportBinding], list[ast.ImportFrom], dict[str, str]]:
    bindings: dict[str, _ImportBinding] = {}
    omitted_bindings: dict[str, str] = {}
    omitted_modules: set[str] = set()
    future_imports: list[ast.ImportFrom] = []

    def register_omitted(binding: str, module: str) -> None:
        if binding in bindings or binding in omitted_bindings:
            raise ImportError(f"duplicate import binding in {source_path}: {binding}")
        omitted_bindings[binding] = module
        omitted_modules.add(module)

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                _reject_forbidden_module(alias.name, source_path, forbid_imports)
                if alias.name in omit_imports:
                    register_omitted(
                        alias.asname or alias.name.split(".", maxsplit=1)[0],
                        alias.name,
                    )
                    continue
                legacy_spec = _validate_module(
                    alias.name,
                    source_path,
                    legacy_specs,
                    allow_dependencies,
                    forbid_imports,
                )
                if legacy_spec is not None:
                    raise ImportError(
                        f"absolute legacy import in {source_path}: {alias.name}"
                    )
                binding = alias.asname or alias.name.split(".", maxsplit=1)[0]
                if binding in bindings:
                    raise ImportError(f"duplicate import binding in {source_path}: {binding}")
                bindings[binding] = _ImportBinding(
                    binding, node, alias, alias.name, None
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ImportError(f"relative source import in {source_path} is not supported")
            if node.module == "__future__":
                if any(alias.name != "annotations" for alias in node.names):
                    raise ImportError(
                        f"unsupported future import in {source_path}: "
                        + ", ".join(alias.name for alias in node.names)
                    )
                future_imports.append(node)
                continue
            if node.module is None:
                raise ImportError(f"unresolved import in {source_path}")
            _reject_forbidden_module(node.module, source_path, forbid_imports)
            if node.module in omit_imports:
                for alias in node.names:
                    if alias.name == "*":
                        raise ImportError(f"star import in {source_path}: {node.module}")
                    register_omitted(alias.asname or alias.name, node.module)
                continue
            legacy_spec = _validate_module(
                node.module,
                source_path,
                legacy_specs,
                allow_dependencies,
                forbid_imports,
            )
            for alias in node.names:
                if alias.name == "*":
                    raise ImportError(f"star import in {source_path}: {node.module}")
                if legacy_spec and alias.name not in legacy_spec.allow_symbols:
                    raise ImportError(
                        f"unallowlisted legacy import in {source_path}: "
                        f"{node.module}.{alias.name}"
                    )
                binding = alias.asname or alias.name
                if binding in bindings:
                    raise ImportError(f"duplicate import binding in {source_path}: {binding}")
                bindings[binding] = _ImportBinding(
                    binding, node, alias, node.module, legacy_spec
                )
    missing_omissions = sorted(set(omit_imports) - omitted_modules)
    if missing_omissions:
        raise ImportError(
            f"omit_imports not found in {source_path}: {missing_omissions[0]}"
        )
    return bindings, future_imports, omitted_bindings


class _RenameGlobals(ast.NodeTransformer):
    def __init__(self, rename: dict[str, str]) -> None:
        self.rename = rename
        self.scopes: list[_Scope] = []

    def _is_module_global(self, name: str) -> bool:
        for scope in reversed(self.scopes):
            if name in scope.globals:
                return True
            if name in scope.bindings or name in scope.nonlocals:
                return False
        return True

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.rename and self._is_module_global(node.id):
            node.id = self.rename[node.id]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.name in self.rename and self._is_module_global(node.name):
            node.name = self.rename[node.name]
        return self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        if node.name in self.rename and self._is_module_global(node.name):
            node.name = self.rename[node.name]
        return self._visit_function(node)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> ast.AST:
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        node.args = self.visit(node.args)
        if node.returns is not None:
            node.returns = self.visit(node.returns)
        outer_scopes = self.scopes
        self.scopes = [scope for scope in outer_scopes if scope.kind != "class"]
        self.scopes.append(_scope_for_body("function", node.body, node.args))
        node.body = [self.visit(statement) for statement in node.body]
        self.scopes = outer_scopes
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        if node.name in self.rename and self._is_module_global(node.name):
            node.name = self.rename[node.name]
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        node.bases = [self.visit(item) for item in node.bases]
        node.keywords = [self.visit(item) for item in node.keywords]
        outer_scopes = self.scopes
        self.scopes = [scope for scope in outer_scopes if scope.kind != "class"]
        self.scopes.append(_scope_for_body("class", node.body))
        node.body = [self.visit(statement) for statement in node.body]
        self.scopes = outer_scopes
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        node.args = self.visit(node.args)
        outer_scopes = self.scopes
        self.scopes = [scope for scope in outer_scopes if scope.kind != "class"]
        self.scopes.append(_scope_for_body("function", [], node.args))
        node.body = self.visit(node.body)
        self.scopes = outer_scopes
        return node

    def _visit_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> ast.AST:
        if not node.generators:
            return node
        node.generators[0].iter = self.visit(node.generators[0].iter)
        bindings = {
            child.id
            for generator in node.generators
            for child in ast.walk(generator.target)
            if isinstance(child, ast.Name)
        }
        outer_scopes = self.scopes
        self.scopes = [scope for scope in outer_scopes if scope.kind != "class"]
        self.scopes.append(
            _Scope("function", frozenset(bindings), frozenset(), frozenset())
        )
        for index, generator in enumerate(node.generators):
            generator.target = self.visit(generator.target)
            if index:
                generator.iter = self.visit(generator.iter)
            generator.ifs = [self.visit(item) for item in generator.ifs]
        if isinstance(node, ast.DictComp):
            node.key = self.visit(node.key)
            node.value = self.visit(node.value)
        else:
            node.elt = self.visit(node.elt)
        self.scopes = outer_scopes
        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        return self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        return self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        return self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        return self._visit_comprehension(node)

    def visit_Global(self, node: ast.Global) -> ast.Global:
        node.names = [self.rename.get(name, name) for name in node.names]
        return node


class _ReplaceExactStrings(ast.NodeTransformer):
    def __init__(self, replacements: tuple[_StringReplacement, ...]) -> None:
        self.replacements = {replacement.old: replacement for replacement in replacements}
        self.counts = {replacement.old: 0 for replacement in replacements}

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if isinstance(node.value, str) and node.value in self.replacements:
            replacement = self.replacements[node.value]
            self.counts[replacement.old] += 1
            return ast.copy_location(ast.Constant(value=replacement.new), node)
        return node


def _forbidden_output_token(value: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    for part in (*FORBIDDEN_DEPENDENCY_PARTS, "stage"):
        if part in normalized:
            return value
    for old_symbol in FORBIDDEN_OLD_SYMBOLS:
        if re.sub(r"[^a-z0-9]", "", old_symbol.lower()) in normalized:
            return value
    return None


def _scan_output(
    tree: ast.Module, source_path: PurePosixPath
) -> None:
    for top_level in tree.body:
        owner_names = _definition_names(top_level)
        owner = owner_names[0] if owner_names else "<module>"
        for node in ast.walk(top_level):
            values: list[str] = []
            if isinstance(node, ast.Name):
                values.append(node.id)
            elif isinstance(node, ast.arg):
                values.append(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                values.append(node.name)
            elif isinstance(node, ast.Attribute):
                values.append(node.attr)
            elif isinstance(node, ast.alias):
                values.extend(value for value in (node.name, node.asname) if value)
            elif isinstance(node, ast.ImportFrom) and node.module:
                values.append(node.module)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                values.append(node.value)
            for value in values:
                forbidden = _forbidden_output_token(value)
                if forbidden is not None:
                    raise ImportError(
                        f"{source_path}:{owner} -> forbidden name/content: {forbidden}"
                    )


def _manual_import_module(
    node: ast.ImportFrom, target: PurePosixPath
) -> str:
    if node.level == 0:
        if node.module is None:
            raise ImportError(f"unresolved import in manual target {target}")
        return node.module
    package_parts = ("lunar_policy_training", *target.parent.parts)
    if node.level > len(package_parts):
        raise ImportError(f"relative import escapes package in manual target {target}")
    base_parts = package_parts[: len(package_parts) - node.level + 1]
    module_parts = tuple(node.module.split(".")) if node.module else ()
    return ".".join((*base_parts, *module_parts))


def _manual_import_bindings(
    *,
    tree: ast.Module,
    spec: _MappingSpec,
    allow_dependencies: set[str],
) -> dict[str, _ImportBinding]:
    bindings: dict[str, _ImportBinding] = {}
    allowed = set(allow_dependencies) | {"lunar_policy_training"}

    def register(
        binding: str,
        node: ast.Import | ast.ImportFrom,
        alias: ast.alias,
        module: str,
    ) -> None:
        if binding in bindings:
            raise ImportError(
                f"duplicate import binding in manual target {spec.target}: {binding}"
            )
        bindings[binding] = _ImportBinding(binding, node, alias, module, None)

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                _validate_module(
                    alias.name,
                    spec.target,
                    {},
                    allowed,
                    spec.forbid_imports,
                )
                register(
                    alias.asname or alias.name.split(".", maxsplit=1)[0],
                    node,
                    alias,
                    alias.name,
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module == "__future__":
                if any(alias.name != "annotations" for alias in node.names):
                    raise ImportError(
                        f"unsupported future import in manual target {spec.target}"
                    )
                continue
            module = _manual_import_module(node, spec.target)
            _validate_module(
                module,
                spec.target,
                {},
                allowed,
                spec.forbid_imports,
            )
            for alias in node.names:
                if alias.name == "*":
                    raise ImportError(
                        f"star import in manual target {spec.target}: {module}"
                    )
                register(alias.asname or alias.name, node, alias, module)
    return bindings


def _is_safe_manual_decorator(
    node: ast.expr,
    import_bindings: dict[str, _ImportBinding],
    definition_names: set[str],
    *,
    method: bool,
) -> bool:
    expression = node.func if isinstance(node, ast.Call) else node
    if not isinstance(expression, ast.Name):
        return False
    if method and expression.id in {"classmethod", "staticmethod", "property"}:
        return (
            expression.id not in import_bindings
            and expression.id not in definition_names
            and not isinstance(node, ast.Call)
        )
    binding = import_bindings.get(expression.id)
    if not (
        binding
        and isinstance(binding.node, ast.ImportFrom)
        and binding.module == "dataclasses"
        and binding.alias.name == "dataclass"
    ):
        return False
    if not isinstance(node, ast.Call):
        return True
    return all(
        _is_safe_constant_expression(value, import_bindings, definition_names)
        for value in (
            *node.args,
            *(keyword.value for keyword in node.keywords),
        )
    )


def _manual_definition_is_safe(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    import_bindings: dict[str, _ImportBinding],
    definition_names: set[str],
    *,
    method: bool = False,
) -> bool:
    if not all(
        _is_safe_manual_decorator(
            decorator,
            import_bindings,
            definition_names,
            method=method,
        )
        for decorator in node.decorator_list
    ):
        return False
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        expressions: list[ast.expr | None] = [
            *node.args.defaults,
            *node.args.kw_defaults,
            node.returns,
            *(argument.annotation for argument in node.args.posonlyargs),
            *(argument.annotation for argument in node.args.args),
            *(argument.annotation for argument in node.args.kwonlyargs),
        ]
        if node.args.vararg is not None:
            expressions.append(node.args.vararg.annotation)
        if node.args.kwarg is not None:
            expressions.append(node.args.kwarg.annotation)
        return all(
            _is_safe_constant_expression(
                expression,
                import_bindings,
                definition_names,
            )
            for expression in expressions
        )
    if not all(
        _is_safe_constant_expression(
            expression,
            import_bindings,
            definition_names,
        )
        for expression in (
            *node.bases,
            *(keyword.value for keyword in node.keywords),
        )
    ):
        return False
    for statement in node.body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if isinstance(statement, ast.Assign):
            if not _is_safe_constant_expression(
                statement.value,
                import_bindings,
                definition_names,
            ):
                return False
            continue
        if isinstance(statement, ast.AnnAssign):
            if not _is_safe_constant_expression(
                statement.value,
                import_bindings,
                definition_names,
            ):
                return False
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _manual_definition_is_safe(
                statement,
                import_bindings,
                definition_names,
                method=True,
            ):
                return False
            continue
        if isinstance(statement, ast.ClassDef):
            if not _manual_definition_is_safe(
                statement,
                import_bindings,
                definition_names,
            ):
                return False
            continue
        return False
    return True


def _scan_manual_target(
    tree: ast.Module,
    spec: _MappingSpec,
    import_bindings: dict[str, _ImportBinding],
) -> None:
    current_observation_symbols = {
        binding
        for binding, imported in import_bindings.items()
        if imported.module == "lunar_policy_training.policy.observation"
        and binding in {"PolicyBatch", "validate_policy_batch"}
    }
    defined_names = {
        name for node in tree.body for name in _definition_names(node)
    }
    if "PolicyBatch" in defined_names:
        raise ImportError(
            f"{spec.target}:PolicyBatch -> forbidden old policy symbol: PolicyBatch"
        )
    for top_level in tree.body:
        owner_names = _definition_names(top_level)
        owner = owner_names[0] if owner_names else "<module>"
        for node in ast.walk(top_level):
            if isinstance(node, ast.Pass) or (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and node.value.value is Ellipsis
            ):
                raise ImportError(
                    f"{spec.target}:{owner} -> empty stub is forbidden"
                )
            values: list[str] = []
            if isinstance(node, ast.Name):
                values.append(node.id)
            elif isinstance(node, ast.arg):
                values.append(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                values.append(node.name)
            elif isinstance(node, ast.Attribute):
                values.append(node.attr)
            elif isinstance(node, ast.alias):
                values.extend(value for value in (node.name, node.asname) if value)
            elif isinstance(node, ast.ImportFrom) and node.module:
                values.append(node.module)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                values.append(node.value)
            for value in values:
                normalized = re.sub(r"[^a-z0-9]", "", value.lower())
                if normalized in {"any", "typechecking"} or "stub" in normalized:
                    raise ImportError(
                        f"{spec.target}:{owner} -> forbidden name/content: {value}"
                    )
                forbidden = _forbidden_output_token(value)
                if forbidden is None:
                    continue
                if value in current_observation_symbols and value not in defined_names:
                    continue
                raise ImportError(
                    f"{spec.target}:{owner} -> forbidden name/content: {forbidden}"
                )


def _validate_manual_target(
    *,
    repository_root: Path,
    target_root: PurePosixPath,
    spec: _MappingSpec,
    allow_dependencies: set[str],
) -> dict[str, Any]:
    target_relative = target_root / spec.target
    destination = _resolve_inside(repository_root, target_relative, field="target")
    if not destination.exists():
        raise ImportError(f"manual target is missing: {target_relative}")
    if destination.is_symlink() or not destination.is_file():
        raise ImportError(f"manual target is not a regular file: {target_relative}")
    payload = destination.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ImportError(f"manual target is not UTF-8: {target_relative}") from error
    try:
        tree = ast.parse(text, filename=target_relative.as_posix())
        compile(tree, target_relative.as_posix(), "exec")
    except (SyntaxError, TypeError, ValueError) as error:
        raise ImportError(f"manual target does not compile: {target_relative}: {error}") from error
    _reject_dynamic_execution(
        tree,
        spec.target,
        {},
        set(allow_dependencies) | {"lunar_policy_training"},
        spec.forbid_imports,
    )
    import_bindings = _manual_import_bindings(
        tree=tree,
        spec=spec,
        allow_dependencies=allow_dependencies,
    )
    definition_names = {
        name for node in tree.body for name in _definition_names(node)
    }
    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        names = _definition_names(node)
        if not names:
            raise ImportError(
                f"top-level execution in manual target {spec.target}: "
                f"{type(node).__name__}"
            )
        if isinstance(node, ast.Assign) and not _is_safe_constant_expression(
            node.value, import_bindings, definition_names
        ):
            raise ImportError(
                f"top-level constant execution in manual target {spec.target}:{names[0]}"
            )
        if isinstance(node, ast.AnnAssign) and not _is_safe_constant_expression(
            node.value, import_bindings, definition_names
        ):
            raise ImportError(
                f"top-level constant execution in manual target {spec.target}:{names[0]}"
            )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not _manual_definition_is_safe(
                node,
                import_bindings,
                definition_names,
            ):
                raise ImportError(
                    f"definition-time execution in manual target "
                    f"{spec.target}:{names[0]}"
                )
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    raise ImportError(
                        f"nested import in manual target {spec.target}:{names[0]}"
                    )
    _scan_manual_target(tree, spec, import_bindings)
    return {
        "target_sha256": hashlib.sha256(payload).hexdigest(),
        "target_size_bytes": len(payload),
        "target_validation": {
            "ast": "parsed-and-compiled",
            "imports": "static-allowlist-and-no-dynamic-imports",
            "kind": "manual-python-adapter",
        },
    }


def _extract_symbols(
    *,
    spec: _MappingSpec,
    text: str,
    legacy_specs: dict[str, _MappingSpec],
    allow_dependencies: set[str],
) -> tuple[str, list[str]]:
    try:
        tree = ast.parse(text, filename=spec.source.as_posix())
    except SyntaxError as error:
        raise ImportError(f"invalid Python source: {spec.source}: {error.msg}") from error
    _reject_dynamic_execution(
        tree,
        spec.source,
        legacy_specs,
        allow_dependencies,
        spec.forbid_imports,
    )
    import_bindings, future_imports, omitted_bindings = _collect_import_bindings(
        tree=tree,
        source_path=spec.source,
        legacy_specs=legacy_specs,
        allow_dependencies=allow_dependencies,
        forbid_imports=spec.forbid_imports,
        omit_imports=spec.omit_imports,
    )
    definition_names = {
        name for node in tree.body for name in _definition_names(node)
    }
    definitions: dict[str, ast.stmt] = {}
    module_docstring: ast.Expr | None = None
    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            module_docstring = node
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        names = _definition_names(node)
        if not names:
            raise ImportError(
                f"top-level execution in {spec.source}: {type(node).__name__}"
            )
        if isinstance(node, ast.Assign) and not _is_safe_constant_expression(
            node.value, import_bindings, definition_names
        ):
            raise ImportError(f"top-level constant execution in {spec.source}:{names[0]}")
        if isinstance(node, ast.AnnAssign) and not _is_safe_constant_expression(
            node.value, import_bindings, definition_names
        ):
            raise ImportError(f"top-level constant execution in {spec.source}:{names[0]}")
        if any(name in definitions for name in names):
            raise ImportError(f"duplicate top-level definition in {spec.source}: {names[0]}")
        definitions.update((name, node) for name in names)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    raise ImportError(f"nested import in {spec.source}:{names[0]}")

    for symbol in spec.allow_symbols:
        if symbol not in definitions:
            raise ImportError(f"{spec.source}:{symbol} -> symbol is not defined")

    selected_nodes = {definitions[symbol] for symbol in spec.allow_symbols}
    required_imports: set[str] = set()
    for symbol in spec.allow_symbols:
        for dependency in sorted(_definition_dependencies(definitions[symbol])):
            if dependency in definitions:
                if dependency not in spec.allow_symbols:
                    raise ImportError(
                        f"{spec.source}:{symbol} -> unallowlisted dependency: {dependency}"
                    )
            elif dependency in import_bindings:
                required_imports.add(dependency)
            elif dependency in omitted_bindings:
                raise ImportError(
                    f"{spec.source}:{symbol} -> omitted import is required: {dependency} "
                    f"from {omitted_bindings[dependency]}"
                )
            elif dependency not in BUILTIN_NAMES:
                raise ImportError(
                    f"{spec.source}:{symbol} -> unresolved dependency: {dependency}"
                )

    output_items: list[tuple[int, ast.stmt]] = []
    if module_docstring is not None:
        output_items.append((module_docstring.lineno, copy.deepcopy(module_docstring)))
    needs_future_annotations = any(
        isinstance(child, ast.AnnAssign)
        or (
            isinstance(child, ast.arg)
            and child.annotation is not None
        )
        or (
            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.returns is not None
        )
        for selected_node in selected_nodes
        for child in ast.walk(selected_node)
    )
    if needs_future_annotations:
        for future_import in future_imports:
            output_items.append((future_import.lineno, copy.deepcopy(future_import)))
    dependency_rename: dict[str, str] = {}
    grouped_bindings: dict[int, list[_ImportBinding]] = {}
    for binding_name in sorted(required_imports):
        binding = import_bindings[binding_name]
        grouped_bindings.setdefault(id(binding.node), []).append(binding)
    for node in tree.body:
        selected_bindings = grouped_bindings.get(id(node), [])
        if not selected_bindings:
            continue
        aliases: list[ast.alias] = []
        import_from_module = node.module if isinstance(node, ast.ImportFrom) else None
        import_from_level = node.level if isinstance(node, ast.ImportFrom) else 0
        for binding in sorted(
            selected_bindings, key=lambda item: (item.alias.lineno, item.alias.col_offset)
        ):
            alias = copy.deepcopy(binding.alias)
            if binding.legacy_spec is not None:
                remote_name = binding.legacy_spec.rename.get(alias.name, alias.name)
                if alias.asname is None and remote_name != alias.name:
                    dependency_rename[alias.name] = remote_name
                alias.name = remote_name
                import_from_level, import_from_module = _relative_import_parts(
                    spec.target, binding.legacy_spec.target
                )
            aliases.append(alias)
        if isinstance(node, ast.Import):
            output_node: ast.stmt = ast.Import(names=aliases)
        else:
            output_node = ast.ImportFrom(
                module=import_from_module,
                names=aliases,
                level=import_from_level,
            )
        output_items.append((node.lineno, ast.copy_location(output_node, node)))

    renamer = _RenameGlobals({**dependency_rename, **spec.rename})
    for node in tree.body:
        if node in selected_nodes:
            output_items.append(
                (
                    node.lineno,
                    renamer.visit(copy.deepcopy(node)),
                )
            )
    output_items.sort(key=lambda item: item[0])
    output_tree = ast.Module(body=[item[1] for item in output_items], type_ignores=[])
    ast.fix_missing_locations(output_tree)
    string_replacer = _ReplaceExactStrings(spec.string_replacements)
    output_tree = string_replacer.visit(output_tree)
    for replacement in spec.string_replacements:
        actual_count = string_replacer.counts[replacement.old]
        if actual_count != replacement.count:
            raise ImportError(
                f"string replacement count in {spec.source}: expected "
                f"{replacement.count}, got {actual_count} for {replacement.old!r}"
            )
    ast.fix_missing_locations(output_tree)
    _scan_output(output_tree, spec.source)
    try:
        compile(output_tree, spec.target.as_posix(), "exec")
    except (SyntaxError, TypeError, ValueError) as error:
        raise ImportError(f"generated module does not compile: {spec.source}: {error}") from error
    output = ast.unparse(output_tree).rstrip() + "\n"
    try:
        parsed_output = ast.parse(output, filename=spec.target.as_posix())
        compile(parsed_output, spec.target.as_posix(), "exec")
    except (SyntaxError, TypeError, ValueError) as error:
        raise ImportError(f"generated text does not compile: {spec.source}: {error}") from error
    _scan_output(parsed_output, spec.source)
    resolved = [
        name
        for node in tree.body
        for name in _definition_names(node)
        if name in spec.allow_symbols
    ]
    return output, resolved


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
    if "type_checking_dependencies" in file_map:
        raise ImportError(
            "type_checking_dependencies is forbidden; dependencies must resolve at runtime"
        )
    inventory_files = inventory.get("files")
    if not isinstance(inventory_files, list):
        raise ImportError("source inventory is missing files")
    inventory_entries = {
        entry.get("path"): entry
        for entry in inventory_files
        if isinstance(entry, dict) and entry.get("repository") == repository
    }

    mappings: list[_MappingSpec] = []
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
            expected_sha256 = entry.get("sha256")
            if not isinstance(expected_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", expected_sha256
            ):
                raise ImportError(f"invalid sha256 for {source}")
            allow_symbols = entry.get("allow_symbols")
            if (
                not isinstance(allow_symbols, list)
                or not all(
                    isinstance(symbol, str)
                    and symbol.isidentifier()
                    and not keyword.iskeyword(symbol)
                    for symbol in allow_symbols
                )
                or len(allow_symbols) != len(set(allow_symbols))
            ):
                raise ImportError(f"invalid allow_symbols for {source}")
            forbidden_old_symbols = [
                symbol for symbol in allow_symbols if symbol in FORBIDDEN_OLD_SYMBOLS
            ]
            if forbidden_old_symbols:
                raise ImportError(
                    f"{source}:{forbidden_old_symbols[0]} -> forbidden old policy symbol: "
                    f"{forbidden_old_symbols[0]}"
                )
            rename = entry.get("rename", {})
            if (
                not isinstance(rename, dict)
                or not all(
                    isinstance(old, str)
                    and old in allow_symbols
                    and isinstance(new, str)
                    and new.isidentifier()
                    and not keyword.iskeyword(new)
                    for old, new in rename.items()
                )
                or len(set(rename.values())) != len(rename)
            ):
                raise ImportError(f"invalid rename for {source}")
            final_names = [rename.get(symbol, symbol) for symbol in allow_symbols]
            if len(final_names) != len(set(final_names)):
                raise ImportError(f"rename collision for {source}")
            forbid_imports = entry.get("forbid_imports", [])
            if (
                not isinstance(forbid_imports, list)
                or not all(
                    isinstance(module, str) and module for module in forbid_imports
                )
                or len(forbid_imports) != len(set(forbid_imports))
            ):
                raise ImportError(f"invalid forbid_imports for {source}")
            omit_imports = entry.get("omit_imports", [])
            if (
                not isinstance(omit_imports, list)
                or not all(isinstance(module, str) and module for module in omit_imports)
                or len(omit_imports) != len(set(omit_imports))
            ):
                raise ImportError(f"invalid omit_imports for {source}")
            for omitted_module in omit_imports:
                if _forbidden_module_part(omitted_module) is not None:
                    raise ImportError(
                        f"omit_imports contains globally forbidden dependency for "
                        f"{source}: {omitted_module}"
                    )
                if any(
                    omitted_module == forbidden_module
                    or omitted_module.startswith(forbidden_module + ".")
                    or forbidden_module.startswith(omitted_module + ".")
                    for forbidden_module in forbid_imports
                ):
                    raise ImportError(
                        f"forbid_imports overlaps omit_imports for {source}: "
                        f"{omitted_module}"
                    )
            raw_replacements = entry.get("string_replacements", [])
            if not isinstance(raw_replacements, list):
                raise ImportError(f"invalid string_replacements for {source}")
            string_replacements: list[_StringReplacement] = []
            replacement_sources: set[str] = set()
            for replacement in raw_replacements:
                if (
                    not isinstance(replacement, dict)
                    or set(replacement) != {"old", "new", "count"}
                    or not isinstance(replacement.get("old"), str)
                    or not replacement["old"]
                    or not isinstance(replacement.get("new"), str)
                    or not replacement["new"]
                    or replacement["old"] == replacement["new"]
                    or type(replacement.get("count")) is not int
                    or replacement["count"] <= 0
                    or replacement["old"] in replacement_sources
                ):
                    raise ImportError(f"invalid string_replacements for {source}")
                replacement_sources.add(replacement["old"])
                string_replacements.append(
                    _StringReplacement(
                        old=replacement["old"],
                        new=replacement["new"],
                        count=replacement["count"],
                    )
                )
            mode = entry.get("mode", "extract")
            if mode not in ALLOWED_MAPPING_MODES:
                raise ImportError(f"invalid extraction mode for {source}: {mode}")
            if mode == "extract" and not allow_symbols:
                raise ImportError(f"allow_symbols must not be empty for extracted source: {source}")
            if mode == "manual_thin_adapter" and (allow_symbols or rename):
                raise ImportError(
                    f"manual_thin_adapter must not extract or rename symbols: {source}"
                )
            adaptation = entry.get("adaptation", {})
            if not isinstance(adaptation, dict):
                raise ImportError(f"invalid adaptation metadata for {source}")
            if source in seen_sources or target in seen_targets:
                raise ImportError(f"duplicate source or target mapping: {source} -> {target}")
            seen_sources.add(source)
            seen_targets.add(target)
            if source.as_posix() not in inventory_entries:
                raise ImportError(f"source is not selected by inventory: {source}")
            mappings.append(
                _MappingSpec(
                    group=str(group),
                    source=source,
                    target=target,
                    sha256=expected_sha256,
                    allow_symbols=tuple(allow_symbols),
                    rename=dict(rename),
                    forbid_imports=tuple(forbid_imports),
                    omit_imports=tuple(omit_imports),
                    string_replacements=tuple(string_replacements),
                    mode=mode,
                    adaptation=copy.deepcopy(adaptation),
                )
            )
    legacy_specs = {_legacy_module(spec.source): spec for spec in mappings}

    pending: list[tuple[_MappingSpec, bytes]] = []
    files: list[dict[str, Any]] = []
    for spec in mappings:
        tree_line = str(
            _git(source_git, "ls-tree", str(commit), "--", spec.source.as_posix())
        ).strip()
        if not tree_line:
            raise ImportError(f"source is absent from frozen commit: {spec.source}")
        metadata, found_path = tree_line.split("\t", maxsplit=1)
        git_mode, kind, object_id = metadata.split()
        if (
            found_path != spec.source.as_posix()
            or kind != "blob"
            or git_mode not in {"100644", "100755"}
        ):
            raise ImportError(f"source is not a regular blob: {spec.source}")
        payload = bytes(
            _git(
                source_git,
                "show",
                f"{commit}:{spec.source.as_posix()}",
                text=False,
            )
        )
        entry = inventory_entries[spec.source.as_posix()]
        if len(payload) != entry.get("size_bytes"):
            raise ImportError(f"source size mismatch: {spec.source}")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry.get("sha256"):
            raise ImportError(f"source SHA-256 mismatch: {spec.source}")
        if digest != spec.sha256:
            raise ImportError(f"file-map SHA-256 mismatch: {spec.source}")
        try:
            source_text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ImportError(f"source is not UTF-8 text: {spec.source}") from error
        resolved_symbols: list[str] = []
        if spec.mode == "extract":
            transformed, resolved_symbols = _extract_symbols(
                spec=spec,
                text=source_text,
                legacy_specs=legacy_specs,
                allow_dependencies=set(allowed),
            )
            pending.append((spec, transformed.encode("utf-8")))
        target_proof: dict[str, Any] = {}
        if spec.mode == "manual_thin_adapter":
            target_proof = _validate_manual_target(
                repository_root=repository_root,
                target_root=target_root,
                spec=spec,
                allow_dependencies=set(allowed),
            )
        file_record = {
                "adaptation": spec.adaptation,
                "group": spec.group,
                "mode": spec.mode,
                "provenance": {
                    "commit": commit,
                    "origin": actual_origin,
                    "repository": repository,
                },
                "rename": spec.rename,
                "omit_imports": list(spec.omit_imports),
                "requested_symbols": list(spec.allow_symbols),
                "resolved_symbols": resolved_symbols,
                "sha256": digest,
                "size_bytes": len(payload),
                "source": spec.source.as_posix(),
                "source_blob_oid": object_id,
                "source_sha256": digest,
                "source_size_bytes": len(payload),
                "string_replacements": [
                    {
                        "count": replacement.count,
                        "new": replacement.new,
                        "old": replacement.old,
                    }
                    for replacement in spec.string_replacements
                ],
                "target": (target_root / spec.target).as_posix(),
            }
        file_record.update(target_proof)
        files.append(file_record)

    for spec, payload in pending:
        destination = _resolve_inside(
            repository_root, target_root / spec.target, field="target"
        )
        if destination.exists() and (not destination.is_file() or destination.read_bytes() != payload):
            raise ImportError(
                f"existing target differs from extracted output: {target_root / spec.target}"
            )
    for spec, payload in pending:
        destination = _resolve_inside(
            repository_root, target_root / spec.target, field="target"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    result = {
        "files": files,
        "schema_version": IMPORT_RESULT_SCHEMA,
        "source_commit": commit,
        "source_origin": actual_origin,
        "source_repository": repository,
    }
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
