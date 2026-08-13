from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from lunar_policy_training.sensor_performance import (
    SensorPerformanceError,
    benchmark_observation_throughput,
    build_sensor_performance_report,
    sensor_source_commit,
    sensor_performance_sha256,
    validate_sensor_performance_report,
    write_sensor_performance_report_atomic,
)
from lunar_policy_training.project_capability import (
    APPROVED_PROJECT_CAPABILITY_SHA256,
)
from training.tools import benchmark_sensor_observation as benchmark_tool


HOST = {
    "hostname": "KAI",
    "system": "Linux",
    "release": "6.8.0-test",
    "machine": "x86_64",
}
SOURCE_COMMIT = "a" * 40
CAPABILITY_SHA256 = "b" * 64
SEMANTICS_SHA256 = "c" * 64


def test_sensor_source_commit_resolves_paths_at_a_historical_revision(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "training-test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Training Test"],
        cwd=tmp_path,
        check=True,
    )
    source = (
        tmp_path
        / "training/lunar_policy_training/lunar_policy_training/marker.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text("first\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "sensor source"], cwd=tmp_path, check=True
    )
    sensor_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (tmp_path / "README.md").write_text("unrelated\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "unrelated"], cwd=tmp_path, check=True
    )
    unrelated_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source.write_text("second\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "new sensor source"], cwd=tmp_path, check=True
    )

    assert sensor_source_commit(tmp_path, revision=unrelated_commit) == sensor_commit
    assert sensor_source_commit(tmp_path) != sensor_commit


def _valid_report() -> dict[str, object]:
    report = {
        "schema_version": "sensor-observation-performance/v1",
        "host": dict(HOST),
        "build": {
            "type": "Release",
            "compiler_id": "GNU",
            "compiler_version": "11.4.0",
        },
        "source_commit": SOURCE_COMMIT,
        "capability_sha256": CAPABILITY_SHA256,
        "training_semantics_sha256": SEMANTICS_SHA256,
        "fixture": {
            "scenario_seed": 4080,
            "workers": 30,
            "planner_workload": "cpp-v3-current-capability-v2",
            "native_warmup_count": 50,
            "native_sample_count": 200,
            "throughput_warmup_steps": 2,
            "throughput_sample_steps": 10,
            "candidate": {
                "height": 601,
                "width": 601,
                "resolution_m": 0.2,
                "range_m": 30.0,
                "candidate_count": 64,
            },
            "reveal": {
                "height": 320,
                "width": 320,
                "resolution_m": 0.2,
                "range_m": 30.0,
            },
        },
        "latency_ms": {
            "candidate_gains": {"p50": 1.0, "p95": 4.0},
            "reveal": {"p50": 0.5, "p95": 1.5},
        },
        "throughput_steps_per_second": {
            "observation_disabled": 100.0,
            "observation_enabled": 95.0,
        },
        "throughput_drop_ratio": 0.05,
        "thresholds": {
            "candidate_p95_ms_max": 5.0,
            "reveal_p95_ms_max": 2.0,
            "throughput_drop_ratio_max": 0.10,
        },
        "passed": True,
    }
    report["report_sha256"] = sensor_performance_sha256(report)
    return report


def _native_benchmark() -> dict[str, object]:
    return {
        "schema_version": "native-visibility-benchmark/v1",
        "build_type": "Release",
        "compiler_id": "GNU",
        "compiler_version": "11.4.0",
        "warmup_count": 50,
        "sample_count": 200,
        "timing_unit": "ms",
        "candidate_fixture": {
            "height": 601,
            "width": 601,
            "resolution_m": 0.2,
            "range_m": 30.0,
            "candidate_count": 64,
        },
        "reveal_fixture": {
            "height": 320,
            "width": 320,
            "resolution_m": 0.2,
            "range_m": 30.0,
        },
        "latency_ms": {
            "candidate_gains": {"p50": 0.1, "p95": 0.2},
            "reveal": {"p50": 1.0, "p95": 1.5},
        },
        "checksum": 123,
    }


def _validate(report: dict[str, object]) -> str:
    return validate_sensor_performance_report(
        report,
        expected_host=HOST,
        expected_source_commit=SOURCE_COMMIT,
        expected_capability_sha256=CAPABILITY_SHA256,
        expected_training_semantics_sha256=SEMANTICS_SHA256,
    )


def _reseal(report: dict[str, object]) -> None:
    report["report_sha256"] = sensor_performance_sha256(report)


def test_valid_release_report_is_identity_bound_and_immutable(tmp_path: Path) -> None:
    report = _valid_report()

    digest = _validate(report)
    target = tmp_path / "sensor-performance.json"
    written = write_sensor_performance_report_atomic(target, report)

    assert digest == report["report_sha256"] == written
    assert json.loads(target.read_text(encoding="utf-8")) == report


def test_report_builder_seals_exact_current_planner_fixture() -> None:
    report = build_sensor_performance_report(
        native_benchmark=_native_benchmark(),
        observation_disabled_steps_per_second=100.0,
        observation_enabled_steps_per_second=95.0,
        capability_sha256=CAPABILITY_SHA256,
        source_commit=SOURCE_COMMIT,
        host=HOST,
    )

    assert report["fixture"]["planner_workload"] == (
        "cpp-v3-current-capability-v2"
    )
    assert report["passed"] is True
    assert report["report_sha256"] == sensor_performance_sha256(report)


def test_formal_benchmark_uses_project_capability_without_external_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "sensor-performance.json"
    monkeypatch.setattr(
        benchmark_tool, "require_clean_sensor_source", lambda _root: None
    )
    monkeypatch.setattr(
        benchmark_tool, "_native_executable", lambda _path: tmp_path / "native"
    )
    monkeypatch.setattr(
        benchmark_tool, "_run_native_benchmark", lambda _path: _native_benchmark()
    )
    monkeypatch.setattr(
        benchmark_tool,
        "benchmark_observation_throughput",
        lambda **_kwargs: {
            "observation_disabled": 100.0,
            "observation_enabled": 95.0,
        },
    )
    monkeypatch.setattr(
        benchmark_tool, "sensor_source_commit", lambda _root: SOURCE_COMMIT
    )

    assert benchmark_tool.main(["--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["capability_sha256"] == APPROVED_PROJECT_CAPABILITY_SHA256
    parsed = benchmark_tool.build_parser().parse_args(["--output", str(output)])
    assert not hasattr(parsed, "capability_lock")


def test_formal_throughput_rejects_missing_current_capability_or_worker_fallback() -> None:
    with pytest.raises(SensorPerformanceError, match="formal capability"):
        benchmark_observation_throughput(workers=30)
    with pytest.raises(SensorPerformanceError, match="30 workers"):
        benchmark_observation_throughput(workers=24, diagnostic_only=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda report: report.pop("fixture"), "fields"),
        (lambda report: report["build"].__setitem__("type", "Debug"), "Release"),
        (
            lambda report: report["host"].__setitem__("hostname", "OTHER"),
            "host",
        ),
        (lambda report: report.__setitem__("source_commit", "d" * 40), "source"),
        (
            lambda report: report.__setitem__("capability_sha256", "e" * 64),
            "capability",
        ),
        (
            lambda report: report["fixture"].__setitem__(
                "native_sample_count", 199
            ),
            "fixture",
        ),
        (
            lambda report: report["fixture"].__setitem__("workers", 24),
            "fixture",
        ),
        (
            lambda report: report["latency_ms"]["candidate_gains"].__setitem__(
                "p95", 5.001
            ),
            "candidate",
        ),
        (
            lambda report: report["latency_ms"]["reveal"].__setitem__(
                "p95", 2.001
            ),
            "reveal",
        ),
    ),
)
def test_report_rejects_missing_stale_debug_or_failing_evidence(
    mutation,
    message: str,
) -> None:
    report = deepcopy(_valid_report())
    mutation(report)
    _reseal(report)

    with pytest.raises(SensorPerformanceError, match=message):
        _validate(report)


def test_report_rejects_more_than_ten_percent_throughput_drop() -> None:
    report = deepcopy(_valid_report())
    report["throughput_steps_per_second"]["observation_enabled"] = 89.0
    report["throughput_drop_ratio"] = 0.11
    report["passed"] = False
    _reseal(report)

    with pytest.raises(SensorPerformanceError, match="throughput"):
        _validate(report)


def test_report_rejects_hash_or_nonfinite_json() -> None:
    report = _valid_report()
    report["report_sha256"] = "0" * 64
    with pytest.raises(SensorPerformanceError, match="hash"):
        _validate(report)

    report = _valid_report()
    report["latency_ms"]["reveal"]["p50"] = float("nan")
    with pytest.raises(SensorPerformanceError, match="finite"):
        _validate(report)
