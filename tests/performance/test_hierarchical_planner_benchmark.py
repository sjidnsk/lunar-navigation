from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest


RUNNER_ENVIRONMENT_VARIABLE = "LUNAR_HIERARCHICAL_PLANNER_BENCHMARK"
SCHEMA_VERSION = "lunar-hierarchical-benchmark/v2"
CASES = {
    "wheel_positive": ("WHEELED", 2.0),
    "legged_positive": ("LEGGED", 2.0),
    "hopper_direct_positive": ("HOPPER", 1.0),
    "hopper_multihop_positive": ("HOPPER", 5.0),
    "hopper_complete_negative": ("HOPPER", 5.0),
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
    "safe_landing_nodes",
    "candidate_edges_evaluated",
    "coarse_edges_rejected",
    "full_edges_certified",
    "full_edges_invalidated",
    "edge_certificate_cache_hits",
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
    assert completed.stdout == ""
    return json.loads(output.read_text(encoding="utf-8"))


def test_fixed_all_platform_release_contract(
    benchmark_document: dict[str, object],
) -> None:
    assert benchmark_document["schema_version"] == SCHEMA_VERSION
    assert benchmark_document["build_type"] == "Release"
    assert benchmark_document["warmup_runs"] >= 1
    assert benchmark_document["measured_runs"] == 30
    assert benchmark_document["timing_unit"] == "s"
    assert benchmark_document["map"] == {
        "width_m": 50.0,
        "height_m": 50.0,
        "resolution_m": 0.2,
        "cells": 250 * 250,
    }
    assert benchmark_document["capability_authority"] == (
        "test-only/non-authoritative"
    )
    assert benchmark_document["ubuntu_amd64_release_evaluated"] is True
    assert benchmark_document["jetson_agx_orin_evaluated"] is False
    assert len(benchmark_document["results"]) == len(CASES)


def test_cases_report_deterministic_counts_and_stage_timings(
    benchmark_document: dict[str, object],
) -> None:
    for case_name, (platform, threshold_s) in CASES.items():
        case = benchmark_document[case_name]
        assert case["schema_version"] == SCHEMA_VERSION
        assert case["platform"] == platform
        assert case["fixture"] == case_name
        assert case["authority"] == "test-only/non-authoritative"
        assert case["runs"] == 30
        assert case["cells"] == 250 * 250
        assert case["width_m"] == 50.0
        assert case["height_m"] == 50.0
        assert case["resolution_m"] == 0.2
        assert re.fullmatch(r"[0-9a-f]{16}", case["route_hash"])
        assert case["deterministic"] is True
        assert case["ubuntu_release_p95_threshold_s"] == threshold_s
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


def test_ubuntu_release_thresholds_and_expected_outcomes(
    benchmark_document: dict[str, object],
) -> None:
    for case_name, (_, threshold_s) in CASES.items():
        case = benchmark_document[case_name]
        assert case["p95_s"] <= threshold_s
        assert case["ubuntu_threshold_passed"] is True
        if case_name == "hopper_complete_negative":
            assert case["planning_outcome"] == "NO_KNOWN_SAFE_ROUTE"
            assert case["reason_code"] == "GLOBAL_NO_KNOWN_SAFE_ROUTE"
        elif case_name.startswith("hopper_"):
            assert case["planning_outcome"] == "NEW_REFERENCE_AVAILABLE"
            assert case["reason_code"] == "HOPPER_FIRST_HOP_AVAILABLE"
            assert case["safe_landing_nodes"] > 0
            assert case["candidate_edges_evaluated"] > 0
        elif case_name == "legged_positive":
            assert case["reason_code"] == "LEGGED_BODY_PLAN_AVAILABLE"
        else:
            assert case["reason_code"] == "WHEEL_PLAN_AVAILABLE"


def test_capability_documents_are_explicitly_non_authoritative() -> None:
    fixture_root = Path(__file__).parents[1] / "fixtures" / "capabilities" / "test-only"
    for platform in ("wheeled", "legged", "hopper"):
        text = (fixture_root / f"{platform}.yaml").read_text(encoding="utf-8")
        assert "authority: test-only/non-authoritative" in text
        assert "runtime_eligible: false" in text
