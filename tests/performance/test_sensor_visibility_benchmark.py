from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from lunar_policy_training.capability_freeze import load_frozen_capability_bundle
from lunar_policy_training.sensor_performance import (
    benchmark_observation_throughput,
)


RUNNER_ENVIRONMENT_VARIABLE = "LUNAR_SENSOR_VISIBILITY_BENCHMARK"
THROUGHPUT_ENVIRONMENT_VARIABLE = "LUNAR_RUN_SENSOR_THROUGHPUT_BENCHMARK"


def _runner() -> Path:
    configured = os.environ.get(RUNNER_ENVIRONMENT_VARIABLE)
    if configured is None:
        pytest.skip(f"set {RUNNER_ENVIRONMENT_VARIABLE} to the Release benchmark")
    runner = Path(configured)
    assert runner.is_file(), f"benchmark runner does not exist: {runner}"
    return runner


def test_release_native_visibility_benchmark_meets_frozen_latency_contract() -> None:
    completed = subprocess.run(
        [str(_runner()), "--warmup", "50", "--samples", "200"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    document = json.loads(completed.stdout)

    assert document["schema_version"] == "native-visibility-benchmark/v1"
    assert document["build_type"] == "Release"
    assert document["warmup_count"] == 50
    assert document["sample_count"] == 200
    assert document["timing_unit"] == "ms"
    assert document["candidate_fixture"] == {
        "height": 601,
        "width": 601,
        "resolution_m": 0.2,
        "range_m": 30.0,
        "candidate_count": 64,
    }
    assert document["reveal_fixture"] == {
        "height": 320,
        "width": 320,
        "resolution_m": 0.2,
        "range_m": 30.0,
    }
    for name, limit in (("candidate_gains", 5.0), ("reveal", 2.0)):
        timing = document["latency_ms"][name]
        assert math.isfinite(timing["p50"])
        assert math.isfinite(timing["p95"])
        assert 0.0 <= timing["p50"] <= timing["p95"] <= limit
    assert isinstance(document["checksum"], int)


def test_current_planner_with_schema_valid_v2_fixture_meets_24_worker_contract(
    tmp_path: Path,
) -> None:
    if os.environ.get(THROUGHPUT_ENVIRONMENT_VARIABLE) != "1":
        pytest.skip(f"set {THROUGHPUT_ENVIRONMENT_VARIABLE}=1 for qualification")
    fixture_path = Path(__file__).resolve().parents[2] / (
        "training/lunar_policy_training/tests/test_capability_freeze.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sensor_performance_capability_fixture", fixture_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    lock = module._write_bundle(tmp_path / "current-capability-v2-fixture")
    bundle = load_frozen_capability_bundle(lock, run_kind="formal")

    throughput = benchmark_observation_throughput(
        workers=24,
        capability_bundle=bundle,
    )

    disabled = throughput["observation_disabled"]
    enabled = throughput["observation_enabled"]
    print(
        json.dumps(
            {
                "observation_disabled": disabled,
                "observation_enabled": enabled,
                "throughput_drop_ratio": max(0.0, 1.0 - enabled / disabled),
            },
            sort_keys=True,
        )
    )
    assert disabled > 0.0
    assert enabled > 0.0
    assert max(0.0, 1.0 - enabled / disabled) <= 0.10
