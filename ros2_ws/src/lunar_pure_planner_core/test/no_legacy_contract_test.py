from __future__ import annotations

import sys
import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".h",
    ".hpp",
    ".json",
    ".py",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
FORBIDDEN_TOKENS = (
    "Content" + "Ref",
    "ContractObject" + "Registry",
    "Reference" + "Bundle",
    "json" + "_codec",
    "schema" + "_codec",
    "registry" + "_handle",
    "legacy" + "_v3_adapter",
    "GLOBAL_SEARCH_" + "RESOURCE_LIMIT",
    "HOPPER_GLOBAL_ROUTE_" + "RESOURCE_LIMIT",
)
FORBIDDEN_PRODUCTION_SOURCES = {
    "ara_star.cpp",
    "commitment_state_machine.cpp",
    "hopper_planner.cpp",
    "legged_planner.cpp",
    "primitive_reachability_graph.cpp",
    "projection_cache.cpp",
    "reachability_projection.cpp",
    "route_continuation.cpp",
    "safe_projection.cpp",
    "traversability_projection.cpp",
    "wheel_planner.cpp",
}
FORBIDDEN_PRODUCTION_TOKENS = (
    "Planner" + "Input",
    "Planner" + "Output",
    "Provisional" + "GlobalRoute",
    "Provisional" + "RouteObserver",
    "Route" + "Continuation",
    "Execution" + "Context",
    "Execution" + "Feedback",
    "Candidate" + "Disposition",
    "GLOBAL_MAP_" + "STALE",
    "LOCAL_MAP_" + "STALE",
    "mission_" + "revision",
    "map_" + "revision",
    "execution_" + "feedback",
    "provisional_" + "observer",
    "st" + "ale",
)


def production_sources() -> list[Path]:
    cmake = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(
        r"set\(LUNAR_ANYTIME_PRODUCTION_SOURCES\s+(.*?)\)", cmake, re.DOTALL
    )
    assert match is not None, "production source closure is not explicit"
    return [PACKAGE_ROOT / token for token in re.findall(r"src/\S+\.cpp", match.group(1))]


def production_include_closure() -> set[Path]:
    pending = production_sources()
    closure: set[Path] = set()
    include_roots = (PACKAGE_ROOT / "src", PACKAGE_ROOT / "include")
    while pending:
        path = pending.pop()
        path = path.resolve()
        if path in closure:
            continue
        assert path.is_file(), f"production closure file is absent: {path}"
        closure.add(path)
        text = path.read_text(encoding="utf-8")
        for include in re.findall(r'^\s*#\s*include\s+"([^"]+)"', text, re.MULTILINE):
            candidates = (path.parent / include, *(root / include for root in include_roots))
            resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if resolved is not None and PACKAGE_ROOT.resolve() in resolved.parents:
                pending.append(resolved)
    return closure


def legacy_contract_hits() -> list[str]:
    hits: list[str] = []
    this_file = Path(__file__).resolve()
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if not path.is_file() or path.resolve() == this_file:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "CMakeLists.txt":
            continue
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            hits.append(f"{relative}: non-UTF-8 file")
            continue
        for token in FORBIDDEN_TOKENS:
            if token in relative or token in text:
                hits.append(f"{relative}: {token}")
    return hits


def test_core_contains_no_legacy_contract_surface() -> None:
    assert legacy_contract_hits() == []


def test_production_source_closure_contains_only_anytime_backends() -> None:
    sources = production_sources()
    forbidden = sorted(path.name for path in sources if path.name in FORBIDDEN_PRODUCTION_SOURCES)
    assert forbidden == []

    closure = production_include_closure()
    closure_text = "\n".join(path.read_text(encoding="utf-8") for path in closure)
    assert "legacy copied core only" not in closure_text
    assert not re.search(
        r"MapSnapshot::Create\(\s*const\s+GridMap&\s+\w+\s*\)",
        closure_text,
    )
    for token in FORBIDDEN_PRODUCTION_TOKENS:
        assert token not in closure_text, f"legacy production token remains: {token}"


def test_cmake_executes_python_boundaries_through_pytest() -> None:
    cmake = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    for test_file in ("no_legacy_contract_test.py", "minimal_map_entrypoint_test.py"):
        escaped_file = re.escape(test_file)
        command = re.search(
            rf"add_test\(\s*NAME\s+\S+\s+COMMAND\s+(.*?){escaped_file}.*?\)",
            cmake,
            re.DOTALL,
        )
        assert command is not None, f"CTest registration absent for {test_file}"
        assert re.search(r"Python3::Interpreter\s+-m\s+pytest\s+-q", command.group(1))


def main() -> int:
    hits = legacy_contract_hits()
    if hits:
        print("legacy contract surfaces remain:", file=sys.stderr)
        for hit in hits:
            print(f"- {hit}", file=sys.stderr)
        return 1
    print("no legacy contract surfaces: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
