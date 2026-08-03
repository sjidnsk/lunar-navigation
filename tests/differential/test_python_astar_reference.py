from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPOSITORY_ROOT / "tests" / "differential" / "fixtures"
LEGACY_ASTAR_PATH = FIXTURE_ROOT / "legacy_astar_61e9fa8.py"
LEGACY_ASTAR_SHA256 = (
    "33bf574cfe0f3eaa0fbfbd59127339d5bc23c9f35baf608567c991bcc66ae701"
)
FACADE_RUNNER_ENVIRONMENT_VARIABLE = "LUNAR_PLANNER_FACADE_SUMMARY"


def _load_legacy_astar() -> ModuleType:
    source = LEGACY_ASTAR_PATH.read_bytes()
    assert hashlib.sha256(source).hexdigest() == LEGACY_ASTAR_SHA256
    specification = importlib.util.spec_from_file_location(
        "lunar_frozen_legacy_astar", LEGACY_ASTAR_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _facade_runner() -> Path:
    configured = os.environ.get(FACADE_RUNNER_ENVIRONMENT_VARIABLE)
    if configured is None:
        pytest.skip(
            f"set {FACADE_RUNNER_ENVIRONMENT_VARIABLE} to the built facade runner"
        )
    runner = Path(configured)
    assert runner.is_file(), f"facade summary runner does not exist: {runner}"
    return runner


@pytest.fixture(scope="module")
def legacy_astar() -> ModuleType:
    return _load_legacy_astar()


@pytest.fixture(scope="module")
def wheel_summaries(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    output = tmp_path_factory.mktemp("python-astar-differential") / "summary.json"
    completed = subprocess.run(
        [
            str(_facade_runner()),
            str(output),
            str(FIXTURE_ROOT / "wheel_cases.json"),
            str(FIXTURE_ROOT / "legged_cases.json"),
            str(FIXTURE_ROOT / "hopper_cases.json"),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    return {
        summary["case_id"]: summary
        for summary in document["summaries"]
        if summary["case_id"].startswith("wheel-")
    }


def _offline_case(case_id: str) -> tuple[np.ndarray, np.ndarray]:
    cost = np.ones((8, 12), dtype=np.float64)
    passable = np.ones((8, 12), dtype=np.bool_)
    if case_id == "wheel-no-safe-route":
        passable[:, 3] = False
    elif case_id == "wheel-goal-infeasible":
        passable[3, 4] = False
    elif case_id != "wheel-safe-corridor":
        raise AssertionError(f"unknown offline A* case: {case_id}")
    return cost, passable


def test_legacy_astar_fixture_matches_frozen_inventory() -> None:
    inventory = json.loads(
        (REPOSITORY_ROOT / "migration" / "source_inventory.yaml").read_text(
            encoding="utf-8"
        )
    )
    astar_entries = [
        entry
        for entry in inventory["files"]
        if entry["repository"] == "dev_platform_constraints"
        and entry["path"]
        == "src/dev_platform_constraints/path_planning/astar.py"
    ]
    assert len(astar_entries) == 1
    assert astar_entries[0]["migration_role"] == "tests/differential_reference"
    assert astar_entries[0]["sha256"] == LEGACY_ASTAR_SHA256
    assert hashlib.sha256(LEGACY_ASTAR_PATH.read_bytes()).hexdigest() == (
        astar_entries[0]["sha256"]
    )


@pytest.mark.parametrize(
    "case_id",
    [
        "wheel-safe-corridor",
        "wheel-no-safe-route",
        "wheel-goal-infeasible",
    ],
)
def test_python_astar_and_cpp_v3_agree_on_safety_semantics(
    legacy_astar: ModuleType,
    wheel_summaries: dict[str, object],
    case_id: str,
) -> None:
    cost, passable = _offline_case(case_id)
    start = (2, 3)
    goal = (4, 3)
    result = legacy_astar.astar_path(
        cost,
        passable,
        start=start,
        goal=goal,
        resolution=1.0,
    )
    cpp = wheel_summaries[case_id]

    path_is_safe = bool(result.reachable) and all(
        bool(passable[y, x]) for x, y in result.path
    )
    reaches_goal = bool(result.path) and result.path[-1] == goal
    assert cpp["reachable"] is bool(result.reachable)
    assert cpp["collision_free"] is path_is_safe
    assert cpp["goal_reached"] is reaches_goal

    if result.reachable:
        assert result.path[0] == start
        assert result.path[-1] == goal
        assert math.isfinite(result.total_cost) and result.total_cost >= 0.0
        cpp_cost = cpp["cost"]
        assert isinstance(cpp_cost, (int, float))
        assert math.isfinite(cpp_cost) and cpp_cost >= 0.0
        cost_ratio = cpp_cost / max(result.total_cost, 1.0e-12)
        assert 0.25 <= cost_ratio <= 4.0
    else:
        assert result.path == ()
        assert math.isinf(result.total_cost)
        assert cpp["cost"] is None
