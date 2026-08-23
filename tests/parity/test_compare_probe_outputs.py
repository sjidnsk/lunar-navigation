"""Focused regression tests for probe JSON shape comparison."""

import importlib.util
from pathlib import Path


COMPARATOR_PATH = Path(__file__).with_name("compare_probe_outputs.py")
SPEC = importlib.util.spec_from_file_location("compare_probe_outputs", COMPARATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
COMPARATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARATOR)


def test_boolean_expected_rejects_integer_actual() -> None:
    """JSON boolean fields must not silently accept Python's equal integer."""
    assert COMPARATOR._compare(True, 1, "root.stable_candidate_order") == [
        "root.stable_candidate_order: expected boolean True, got 1"
    ]
