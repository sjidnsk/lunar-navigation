from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest


RUNNER_ENVIRONMENT_VARIABLE = "LUNAR_HIERARCHICAL_PLANNER_BENCHMARK"
SCHEMA_VERSION = "lunar-hierarchical-benchmark/v1"
CELL_TIERS = (65_536, 262_144, 1_048_576)
FIXTURES = ("open", "fixed-obstacle", "narrow-channel", "no-route")
UBUNTU_GLOBAL_P95_S = {
    "65536": 0.5,
    "262144": 1.0,
    "1048576": 2.0,
}
UBUNTU_COMPLETE_P95_S = {
    "65536": 2.0,
    "262144": 3.0,
    "1048576": 4.0,
}
AGX_GLOBAL_P95_S = {
    "65536": 1.0,
    "262144": 2.0,
    "1048576": 4.0,
}
AGX_COMPLETE_P95_S = {
    "65536": 3.0,
    "262144": 4.0,
    "1048576": 6.0,
}
REQUIRED_RESULT_KEYS = {
    "schema_version",
    "platform",
    "fixture",
    "cells",
    "resolution_m",
    "runs",
    "p50_s",
    "p95_s",
    "maximum_s",
    "expanded_states",
    "open_peak",
    "peak_memory_bytes",
    "route_hash",
    "smoothing_p50_s",
    "smoothing_p95_s",
    "smoothing_maximum_s",
    "landing_field_p50_s",
    "landing_field_p95_s",
    "landing_field_maximum_s",
    "trajectory_mode_counts",
}


def _runner() -> Path:
    configured = os.environ.get(RUNNER_ENVIRONMENT_VARIABLE)
    if configured is None:
        pytest.skip(f"set {RUNNER_ENVIRONMENT_VARIABLE} to the benchmark runner")
    runner = Path(configured)
    assert runner.is_file(), f"benchmark runner does not exist: {runner}"
    return runner


@pytest.fixture(scope="module")
def benchmark_document(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    output = tmp_path_factory.mktemp("hierarchical-benchmark") / "benchmark.json"
    completed = subprocess.run(
        [str(_runner()), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def test_benchmark_contract_covers_all_tiers_and_fixtures(
    benchmark_document: dict[str, object],
) -> None:
    assert benchmark_document["schema_version"] == SCHEMA_VERSION
    assert benchmark_document["build_type"] == "Release"
    assert benchmark_document["warmup_runs"] >= 1
    assert benchmark_document["measured_runs"] == 30
    assert benchmark_document["timing_unit"] == "s"
    assert benchmark_document["memory_semantics"] == (
        "planner_estimated_peak_work_memory_bytes"
    )
    profiles = benchmark_document["threshold_profiles"]
    assert profiles == {
        "ubuntu_amd64": {
            "evaluated": True,
            "global_p95_s": UBUNTU_GLOBAL_P95_S,
            "complete_core_p95_s": UBUNTU_COMPLETE_P95_S,
        },
        "jetson_agx_orin": {
            "evaluated": False,
            "global_p95_s": AGX_GLOBAL_P95_S,
            "complete_core_p95_s": AGX_COMPLETE_P95_S,
        },
    }

    results = benchmark_document["results"]
    assert len(results) == len(CELL_TIERS) * len(FIXTURES)
    assert {(item["cells"], item["fixture"]) for item in results} == {
        (cells, fixture) for cells in CELL_TIERS for fixture in FIXTURES
    }

    for result in results:
        assert REQUIRED_RESULT_KEYS <= set(result)
        assert result["schema_version"] == SCHEMA_VERSION
        assert result["platform"] == "WHEELED"
        assert result["cells"] in CELL_TIERS
        assert result["fixture"] in FIXTURES
        assert result["resolution_m"] == 0.2
        assert result["runs"] == 30
        assert re.fullmatch(r"[0-9a-f]{16}", result["route_hash"])
        for key in (
            "p50_s",
            "p95_s",
            "maximum_s",
            "global_p50_s",
            "global_p95_s",
            "global_maximum_s",
            "smoothing_p50_s",
            "smoothing_p95_s",
            "smoothing_maximum_s",
            "landing_field_p50_s",
            "landing_field_p95_s",
            "landing_field_maximum_s",
        ):
            assert math.isfinite(result[key]) and result[key] >= 0.0
        assert result["p50_s"] <= result["p95_s"] <= result["maximum_s"]
        assert (
            result["global_p50_s"]
            <= result["global_p95_s"]
            <= result["global_maximum_s"]
        )
        for key in ("expanded_states", "open_peak", "peak_memory_bytes"):
            assert isinstance(result[key], int) and result[key] >= 0
        assert result["peak_memory_bytes"] <= 256 * 1024 * 1024
        assert result["trajectory_mode_counts"].keys() == {
            "STATIONARY",
            "OPTIMIZED",
            "DISCRETE_FALLBACK",
            "CERTIFIED_HOP",
            "NONE",
        }
        assert sum(result["trajectory_mode_counts"].values()) == 30


def test_ubuntu_thresholds_are_enforced_and_agx_is_device_only(
    benchmark_document: dict[str, object],
) -> None:
    for result in benchmark_document["results"]:
        tier = str(result["cells"])
        assert result["global_p95_s"] <= UBUNTU_GLOBAL_P95_S[tier]
        assert result["p95_s"] <= UBUNTU_COMPLETE_P95_S[tier]
        assert result["ubuntu_threshold_passed"] is True
    assert benchmark_document["threshold_profiles"]["jetson_agx_orin"][
        "evaluated"
    ] is False


def test_routes_and_failure_classes_are_deterministic(
    benchmark_document: dict[str, object],
) -> None:
    for result in benchmark_document["results"]:
        assert result["deterministic"] is True
        if result["fixture"] == "no-route":
            assert result["planning_outcome"] == "NO_KNOWN_SAFE_ROUTE"
            assert result["reason_code"] == "GLOBAL_NO_KNOWN_SAFE_ROUTE"
        else:
            assert result["planning_outcome"] == "NEW_REFERENCE_AVAILABLE"
            assert result["reason_code"] == "WHEEL_PLAN_AVAILABLE"
