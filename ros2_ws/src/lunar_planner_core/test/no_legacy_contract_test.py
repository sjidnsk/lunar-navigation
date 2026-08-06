from __future__ import annotations

import sys
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
