#!/usr/bin/env python3
"""Compare deterministic pure-planner probe output with the frozen baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


PLATFORMS = ("wheel", "legged", "hopper")
ABSOLUTE_TOLERANCE = 1.0e-9


def _load(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"missing JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON file {path}: {error}") from error


def _compare(expected: Any, actual: Any, location: str) -> list[str]:
    if isinstance(expected, bool):
        if not isinstance(actual, bool):
            return [
                f"{location}: expected boolean {expected!r}, got {actual!r}"
            ]
        return [] if expected == actual else [
            f"{location}: expected {expected!r}, got {actual!r}"
        ]
    if expected is None or isinstance(expected, str):
        return [] if expected == actual else [f"{location}: expected {expected!r}, got {actual!r}"]
    if isinstance(expected, int):
        if isinstance(actual, bool) or not isinstance(actual, int):
            return [f"{location}: expected integer {expected!r}, got {actual!r}"]
        return [] if expected == actual else [f"{location}: expected {expected!r}, got {actual!r}"]
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return [f"{location}: expected number {expected!r}, got {actual!r}"]
        if not math.isfinite(expected) or not math.isfinite(float(actual)):
            return [] if expected == actual else [f"{location}: non-finite numbers differ"]
        if math.isclose(expected, float(actual), rel_tol=0.0, abs_tol=ABSOLUTE_TOLERANCE):
            return []
        return [
            f"{location}: expected {expected:.17g}, got {float(actual):.17g} "
            f"(absolute tolerance {ABSOLUTE_TOLERANCE:.1e})"
        ]
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{location}: expected array, got {type(actual).__name__}"]
        errors = []
        if len(expected) != len(actual):
            errors.append(f"{location}: expected {len(expected)} items, got {len(actual)}")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            errors.extend(_compare(expected_item, actual_item, f"{location}[{index}]"))
        return errors
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{location}: expected object, got {type(actual).__name__}"]
        errors = []
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            errors.append(f"{location}: missing field {key!r}")
        for key in sorted(actual_keys - expected_keys):
            errors.append(f"{location}: unexpected field {key!r}")
        for key in sorted(expected_keys & actual_keys):
            errors.extend(_compare(expected[key], actual[key], f"{location}.{key}"))
        return errors
    return [f"{location}: unsupported expected type {type(expected).__name__}"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_directories(expected_dir: Path, actual_dir: Path) -> list[str]:
    errors: list[str] = []
    for platform in PLATFORMS:
        expected_path = expected_dir / f"{platform}.json"
        run_paths = [actual_dir / f"{platform}.run-{run}.json" for run in range(1, 4)]
        try:
            expected = _load(expected_path)
        except ValueError as error:
            errors.append(str(error))
            continue

        hashes: list[str] = []
        for run_path in run_paths:
            try:
                actual = _load(run_path)
                hashes.append(_sha256(run_path))
            except ValueError as error:
                errors.append(str(error))
                continue
            errors.extend(_compare(expected, actual, platform))
        if len(hashes) == 3 and len(set(hashes)) != 1:
            errors.append(f"{platform}: three runs are nondeterministic: {', '.join(hashes)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    args = parser.parse_args()
    errors = compare_directories(args.expected, args.actual)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("probe parity passed for wheel, legged, hopper; each output is deterministic across 3 runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
