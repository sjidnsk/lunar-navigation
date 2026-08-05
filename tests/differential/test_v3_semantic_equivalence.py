from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPOSITORY_ROOT / "tests" / "differential" / "fixtures"
LEGACY_EXPECTED_PATH = (
    REPOSITORY_ROOT / "tests" / "differential" / "legacy_expected.json"
)
HIERARCHICAL_COST_EXPECTED_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "differential"
    / "hierarchical_cost_expected.json"
)
RUNNER_ENVIRONMENT_VARIABLE = "LUNAR_PLANNER_FACADE_SUMMARY"
EXPECTED_DIRECTIVE = {
    "NEW_REFERENCE_AVAILABLE": "ACTIVATE_NEW_REFERENCE",
    "NO_KNOWN_SAFE_ROUTE": "NO_SAFE_REFERENCE",
    "GOAL_INFEASIBLE": "HOLD_POSITION",
    "NUMERICAL_FAILURE": "HOLD_POSITION",
}


def _runner() -> Path:
    configured = os.environ.get(RUNNER_ENVIRONMENT_VARIABLE)
    if configured is None:
        pytest.skip(
            f"set {RUNNER_ENVIRONMENT_VARIABLE} to the built facade summary runner"
        )
    runner = Path(configured)
    assert runner.is_file(), f"facade summary runner does not exist: {runner}"
    return runner


@pytest.fixture(scope="module")
def facade_summary(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    output = tmp_path_factory.mktemp("facade-summary") / "summary.json"
    completed = subprocess.run(
        [
            str(_runner()),
            str(output),
            str(FIXTURE_ROOT / "wheel_cases.json"),
            str(FIXTURE_ROOT / "legged_cases.json"),
            str(FIXTURE_ROOT / "hopper_cases.json"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def test_facade_matches_frozen_v3_semantics(
    facade_summary: dict[str, object],
) -> None:
    legacy = json.loads(LEGACY_EXPECTED_PATH.read_text(encoding="utf-8"))
    hierarchical_costs = json.loads(
        HIERARCHICAL_COST_EXPECTED_PATH.read_text(encoding="utf-8")
    )
    assert facade_summary["schema_version"] == "lunar-v3-facade-summary/v1"
    assert facade_summary["cost_semantics"] == "hierarchical_global_route_cost"
    assert (
        hierarchical_costs["schema_version"]
        == "lunar-v3-hierarchical-cost-summary/v1"
    )

    actual_by_id = {
        summary["case_id"]: summary for summary in facade_summary["summaries"]
    }
    expected_by_id = {
        summary["case_id"]: summary for summary in legacy["summaries"]
    }
    assert actual_by_id.keys() == expected_by_id.keys()

    boolean_fields = (
        "reachable",
        "collision_free",
        "goal_reached",
        "platform_constraints_satisfied",
    )
    for case_id, expected in expected_by_id.items():
        actual = actual_by_id[case_id]
        assert actual["planning_outcome"] == expected["planning_outcome"], case_id
        assert actual["execution_directive"] == EXPECTED_DIRECTIVE[
            expected["planning_outcome"]
        ], case_id
        for field in boolean_fields:
            assert actual[field] is expected[field], f"{case_id}:{field}"

        expected_cost = hierarchical_costs["costs"][case_id]
        actual_cost = actual["cost"]
        if expected_cost is None:
            assert actual_cost is None, case_id
        else:
            assert isinstance(actual_cost, (int, float)), case_id
            assert math.isfinite(actual_cost) and actual_cost >= 0.0, case_id
            assert math.isclose(
                actual_cost,
                expected_cost,
                rel_tol=0.25,
                abs_tol=0.25,
            ), case_id


@pytest.mark.parametrize("platform", ["wheel", "legged", "hopper"])
def test_facade_repeat_case_is_deterministic(
    facade_summary: dict[str, object], platform: str
) -> None:
    by_id = {
        summary["case_id"]: summary for summary in facade_summary["summaries"]
    }
    safe = dict(by_id[f"{platform}-safe-corridor"])
    repeated = dict(by_id[f"{platform}-deterministic-repeat"])
    safe.pop("case_id")
    repeated.pop("case_id")
    assert repeated == safe
