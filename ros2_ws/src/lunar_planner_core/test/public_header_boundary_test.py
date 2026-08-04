#!/usr/bin/env python3
"""Reject non-standard dependencies and legacy concepts in installed headers."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


EXPECTED_HEADERS = {
    "lunar_planner_core/planner.hpp",
    "lunar_planner_core/traversability_projection.hpp",
    "lunar_planner_core/types/execution_context.hpp",
    "lunar_planner_core/types/geometry.hpp",
    "lunar_planner_core/types/goal.hpp",
    "lunar_planner_core/types/motion_reference.hpp",
    "lunar_planner_core/types/planner_config.hpp",
    "lunar_planner_core/types/planner_io.hpp",
    "lunar_planner_core/types/platform_capability.hpp",
    "lunar_planner_core/types/world_snapshot.hpp",
}
FORBIDDEN_PATTERNS = {
    "ROS client": re.compile(r"\brclcpp\b", re.IGNORECASE),
    "ROS message": re.compile(r"\bgeometry_msgs\b", re.IGNORECASE),
    "JSON dependency": re.compile(r"\bnlohmann\b", re.IGNORECASE),
    "legacy content identity": re.compile(r"\bContent" + r"Ref\b"),
    "legacy registry": re.compile(r"\bRegistry\b"),
    "runtime schema": re.compile(r"\bschema\b", re.IGNORECASE),
    "TensorRT": re.compile(r"\bTensorRT\b", re.IGNORECASE),
    "Python include": re.compile(r"#\s*include\s*[<\"]Python\.h[>\"]"),
}


def check_headers(include_root: Path, compiler: str) -> list[str]:
    errors: list[str] = []
    if not include_root.is_dir():
        return [f"installed include root is absent: {include_root}"]
    headers = {
        path.relative_to(include_root).as_posix()
        for path in include_root.rglob("*.hpp")
    }
    if headers != EXPECTED_HEADERS:
        errors.append(
            "installed header set differs: "
            f"missing={sorted(EXPECTED_HEADERS - headers)}, "
            f"extra={sorted(headers - EXPECTED_HEADERS)}"
        )
    for relative in sorted(headers):
        path = include_root / relative
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{relative}: contains {label}")
        completed = subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-fsyntax-only",
                "-x",
                "c++",
                "-I",
                str(include_root),
                "-",
            ],
            input=f'#include "{relative}"\nint main() {{ return 0; }}\n',
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            errors.append(f"{relative}: standalone compile failed: {detail}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-root", required=True, type=Path)
    parser.add_argument("--compiler", default="c++")
    arguments = parser.parse_args()
    errors = check_headers(arguments.include_root.resolve(), arguments.compiler)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("public header boundary: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
