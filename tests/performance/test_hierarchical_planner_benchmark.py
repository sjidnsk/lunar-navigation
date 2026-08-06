from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest


RUNNER_ENVIRONMENT_VARIABLE = "LUNAR_HIERARCHICAL_PLANNER_BENCHMARK"
SCHEMA_VERSION = "lunar-three-platform-qualification/v1"
FREEZE_SHA256 = "60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95"
MEASURED_RUNS = 20
CASES = {
    "wheel_50m_l0": {
        "platform": "WHEELED",
        "capability_version": "wheeled-engineering-baseline-v1",
        "map": (50.0, 50.0, 0.2, 0, 250 * 250),
        "threshold_s": 2.0,
        "outcome": "NEW_REFERENCE_AVAILABLE",
        "reason": "WHEEL_PLAN_AVAILABLE",
    },
    "legged_50m_l0": {
        "platform": "LEGGED",
        "capability_version": "quad48-approved-baseline-v1",
        "map": (50.0, 50.0, 0.2, 0, 250 * 250),
        "threshold_s": 3.0,
        "outcome": "NEW_REFERENCE_AVAILABLE",
        "reason": "LEGGED_BODY_PLAN_AVAILABLE",
    },
    "wheel_1km_l3": {
        "platform": "WHEELED",
        "capability_version": "wheeled-engineering-baseline-v1",
        "map": (1000.0, 1000.0, 1.6, 3, 625 * 625),
        "threshold_s": 3.0,
        "outcome": "NEW_REFERENCE_AVAILABLE",
        "reason": "WHEEL_PLAN_AVAILABLE",
    },
    "legged_1km_l3": {
        "platform": "LEGGED",
        "capability_version": "quad48-approved-baseline-v1",
        "map": (1000.0, 1000.0, 1.6, 3, 625 * 625),
        "threshold_s": 3.0,
        "outcome": "NEW_REFERENCE_AVAILABLE",
        "reason": "LEGGED_BODY_PLAN_AVAILABLE",
    },
    "hopper_direct_100m": {
        "platform": "HOPPER",
        "capability_version": "hopper-engineering-baseline-v1",
        "map": (112.0, 12.0, 0.2, 0, 560 * 60),
        "threshold_s": 1.0,
        "outcome": "NEW_REFERENCE_AVAILABLE",
        "reason": "HOPPER_SINGLE_HOP_AVAILABLE",
    },
    "hopper_alternate_time_100m": {
        "platform": "HOPPER",
        "capability_version": "hopper-engineering-baseline-v1",
        "map": (112.0, 12.0, 0.2, 0, 560 * 60),
        "threshold_s": 2.0,
        "outcome": "NEW_REFERENCE_AVAILABLE",
        "reason": "HOPPER_SINGLE_HOP_AVAILABLE",
    },
    "hopper_complete_blocked_100m": {
        "platform": "HOPPER",
        "capability_version": "hopper-engineering-baseline-v1",
        "map": (112.0, 12.0, 0.2, 0, 560 * 60),
        "threshold_s": 5.0,
        "outcome": "NO_KNOWN_SAFE_ROUTE",
        "reason": "HOPPER_ALL_FLIGHT_TUBES_BLOCKED",
    },
}
STAGES = {
    "global_search",
    "local_planning",
    "landing_field",
    "spatial_index",
    "ballistic_solve",
    "flight_tube_certification",
}
COUNTS = {
    "expanded_states",
    "open_peak",
    "peak_work_memory_bytes",
    "peak_resident_memory_bytes",
    "hopper_certification_attempts",
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
    output = tmp_path_factory.mktemp("three-platform-benchmark") / "benchmark.json"
    completed = subprocess.run(
        [str(_runner()), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    return json.loads(output.read_text(encoding="utf-8"))


def test_release_document_is_bound_to_the_approved_freeze(
    benchmark_document: dict[str, object],
) -> None:
    assert benchmark_document == {
        **benchmark_document,
        "schema_version": SCHEMA_VERSION,
        "build_type": "Release",
        "warmup_runs": 1,
        "measured_runs": MEASURED_RUNS,
        "timing_unit": "s",
        "capability_authority": "approved-engineering-baseline",
        "capability_freeze_sha256": FREEZE_SHA256,
        "ubuntu_amd64_release_evaluated": True,
        "jetson_agx_orin_evaluated": False,
    }
    assert set(benchmark_document["cases"]) == set(CASES)
    assert len(benchmark_document["results"]) == len(CASES)


def test_cases_report_map_level_mode_counts_and_deterministic_timings(
    benchmark_document: dict[str, object],
) -> None:
    for case_name, expected in CASES.items():
        case = benchmark_document["cases"][case_name]
        width, height, resolution, level, cells = expected["map"]
        assert case["case_id"] == case_name
        assert case["platform"] == expected["platform"]
        assert case["capability_version"] == expected["capability_version"]
        assert case["map"] == {
            "width_m": width,
            "height_m": height,
            "resolution_m": resolution,
            "level": level,
            "cells": cells,
        }
        assert case["local_map"]["resolution_m"] == 0.2
        assert case["runs"] == MEASURED_RUNS
        assert re.fullmatch(r"[0-9a-f]{16}", case["reference_hash"])
        assert case["deterministic"] is True
        assert case["mode"] in {
            "OPTIMIZED",
            "DISCRETE_FALLBACK",
            "CERTIFIED_HOP",
            "NO_REFERENCE",
        }
        for key in ("p50_s", "p95_s", "maximum_s"):
            assert math.isfinite(case[key]) and case[key] >= 0.0
        assert case["p50_s"] <= case["p95_s"] <= case["maximum_s"]
        assert set(case["stage_timings_s"]) == STAGES
        for timing in case["stage_timings_s"].values():
            assert set(timing) == {"p50_s", "p95_s", "maximum_s"}
            assert timing["p50_s"] <= timing["p95_s"] <= timing["maximum_s"]
            assert all(math.isfinite(value) and value >= 0.0 for value in timing.values())
        for key in COUNTS:
            assert isinstance(case[key], int) and case[key] >= 0
        if expected["platform"] == "HOPPER" and expected["outcome"] == "NEW_REFERENCE_AVAILABLE":
            assert math.isfinite(case["hop_flight_time_s"])
            assert case["hop_flight_time_s"] > 0.0
        else:
            assert case["hop_flight_time_s"] is None


def test_release_thresholds_and_expected_platform_outcomes(
    benchmark_document: dict[str, object],
) -> None:
    cases = benchmark_document["cases"]
    for case_name, expected in CASES.items():
        case = cases[case_name]
        assert case["planning_outcome"] == expected["outcome"]
        assert case["reason_code"] == expected["reason"]
        assert case["release_p95_threshold_s"] == expected["threshold_s"]
        assert case["p95_s"] <= expected["threshold_s"]
        assert case["release_threshold_passed"] is True

    assert cases["hopper_direct_100m"]["mode"] == "CERTIFIED_HOP"
    assert cases["hopper_alternate_time_100m"]["mode"] == "CERTIFIED_HOP"
    assert cases["hopper_complete_blocked_100m"]["mode"] == "NO_REFERENCE"
    assert (
        cases["hopper_alternate_time_100m"]["hopper_certification_attempts"]
        >= cases["hopper_direct_100m"]["hopper_certification_attempts"]
    )
    assert (
        cases["hopper_alternate_time_100m"]["hop_flight_time_s"]
        > cases["hopper_direct_100m"]["hop_flight_time_s"]
    )


def test_test_only_capability_documents_are_not_runtime_authority() -> None:
    fixture_root = Path(__file__).parents[1] / "fixtures" / "capabilities" / "test-only"
    for platform in ("wheeled", "legged", "hopper"):
        text = (fixture_root / f"{platform}.yaml").read_text(encoding="utf-8")
        assert "authority: test-only/non-authoritative" in text
        assert "runtime_eligible: false" in text
