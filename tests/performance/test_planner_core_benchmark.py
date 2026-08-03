from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest


BENCHMARK_RUNNER_ENVIRONMENT_VARIABLE = "LUNAR_PLANNER_CORE_BENCHMARK"
DEVICE_FINGERPRINT_SHA256 = "a" * 64
GIT_COMMIT = "b" * 40


def _runner() -> Path:
    configured = os.environ.get(BENCHMARK_RUNNER_ENVIRONMENT_VARIABLE)
    if configured is None:
        pytest.skip(
            f"set {BENCHMARK_RUNNER_ENVIRONMENT_VARIABLE} to the benchmark runner"
        )
    runner = Path(configured)
    assert runner.is_file(), f"benchmark runner does not exist: {runner}"
    return runner


def _command(runner: Path, output: Path, fingerprint: str) -> list[str]:
    return [
        str(runner),
        "--output",
        str(output),
        "--device-fingerprint-sha256",
        fingerprint,
        "--git-commit",
        GIT_COMMIT,
    ]


def test_benchmark_writes_fixed_fixture_and_timing_contract(tmp_path: Path) -> None:
    output = tmp_path / "planner-core-benchmark.json"
    completed = subprocess.run(
        _command(_runner(), output, DEVICE_FINGERPRINT_SHA256),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_version"] == "lunar-planner-core-benchmark/v1"
    assert document["fixture_id"] == "wheeled-flat-map-v1"
    assert document["platform"] == "WHEELED"
    assert document["map"] == {
        "frame_id": "map",
        "height": 8,
        "resolution_m": 1.0,
        "width": 12,
    }
    assert document["start_xyz_m"] == [2.5, 3.5, 0.0]
    assert document["goal_xyz_m"] == [4.5, 3.5, 0.0]
    assert document["random_seed"] == 549924309330
    assert document["warmup_count"] == 10
    assert document["count"] == 100
    assert document["device_fingerprint_sha256"] == DEVICE_FINGERPRINT_SHA256
    assert document["git_commit"] == GIT_COMMIT
    assert document["release_gate"] is False
    assert document["timing_unit"] == "ms"
    assert document["planning_outcome"] == "NEW_REFERENCE_AVAILABLE"
    for field in ("median", "p95", "max"):
        assert math.isfinite(document[field]) and document[field] >= 0.0
    assert document["median"] <= document["p95"] <= document["max"]


def test_benchmark_rejects_unverifiable_fingerprint_hash(tmp_path: Path) -> None:
    completed = subprocess.run(
        _command(_runner(), tmp_path / "invalid.json", "not-a-sha256"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "device fingerprint" in completed.stderr.lower()
