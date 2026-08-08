#!/usr/bin/env python3
"""Collect the Release sensor-observation gate or a non-authorizing diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "lunar_policy_training"
sys.path.insert(0, str(PACKAGE_ROOT))

from lunar_policy_training.capability_freeze import CapabilityFreezeError  # noqa: E402
from lunar_policy_training.project_capability import (  # noqa: E402
    load_project_formal_capability,
)
from lunar_policy_training.sensor_performance import (  # noqa: E402
    NATIVE_SAMPLE_COUNT,
    NATIVE_WARMUP_COUNT,
    REQUIRED_WORKERS,
    SensorPerformanceError,
    benchmark_observation_throughput,
    build_sensor_performance_report,
    current_host_identity,
    require_clean_sensor_source,
    sensor_source_commit,
    validate_sensor_performance_report,
    write_sensor_performance_report_atomic,
)
from lunar_policy_training.training_semantics import (  # noqa: E402
    training_semantics_sha256,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native-benchmark", type=Path)
    parser.add_argument("--workers", type=int, default=REQUIRED_WORKERS)
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="measure throughput without writing formal evidence",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    try:
        if arguments.workers != REQUIRED_WORKERS:
            raise SensorPerformanceError(
                "sensor performance qualification requires exactly 24 workers"
            )
        if arguments.diagnostic_only:
            if arguments.output is not None:
                raise SensorPerformanceError(
                    "diagnostic-only does not accept formal output"
                )
            throughput = benchmark_observation_throughput(
                workers=arguments.workers,
                diagnostic_only=True,
            )
            print(
                json.dumps(
                    {
                        "schema_version": (
                            "sensor-observation-throughput-diagnostic/v1"
                        ),
                        "workers": arguments.workers,
                        "throughput_steps_per_second": throughput,
                        "formal_eligible": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if arguments.output is None:
            raise SensorPerformanceError("formal benchmark requires --output")
        output = _external_output(arguments.output, repository_root)
        require_clean_sensor_source(repository_root)
        try:
            capability = load_project_formal_capability(repository_root)
        except CapabilityFreezeError as error:
            raise SensorPerformanceError(
                f"project formal capability is invalid: {error}"
            ) from error
        native = _run_native_benchmark(_native_executable(arguments.native_benchmark))
        throughput = benchmark_observation_throughput(
            workers=arguments.workers,
            capability_bundle=capability,
        )
        source_commit = sensor_source_commit(repository_root)
        report = build_sensor_performance_report(
            native_benchmark=native,
            observation_disabled_steps_per_second=throughput[
                "observation_disabled"
            ],
            observation_enabled_steps_per_second=throughput[
                "observation_enabled"
            ],
            capability_sha256=capability.bundle_sha256,
            source_commit=source_commit,
        )
        written = write_sensor_performance_report_atomic(output, report)
        if report["passed"] is not True:
            print(
                json.dumps(
                    {
                        "output": str(output),
                        "report_sha256": written,
                        "passed": False,
                    },
                    sort_keys=True,
                )
            )
            return 2
        digest = validate_sensor_performance_report(
            report,
            expected_host=current_host_identity(),
            expected_source_commit=source_commit,
            expected_capability_sha256=capability.bundle_sha256,
            expected_training_semantics_sha256=training_semantics_sha256(),
        )
        if written != digest:
            raise SensorPerformanceError("written sensor report digest changed")
        print(
            json.dumps(
                {
                    "output": str(output),
                    "report_sha256": digest,
                    "passed": True,
                },
                sort_keys=True,
            )
        )
    except SensorPerformanceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def _native_executable(configured: Path | None) -> Path:
    value = configured
    if value is None:
        environment = os.environ.get("LUNAR_SENSOR_VISIBILITY_BENCHMARK")
        discovered = shutil.which("lunar_training_visibility_benchmark")
        value = Path(environment or discovered) if environment or discovered else None
    if value is None:
        raise SensorPerformanceError("Release native visibility benchmark is missing")
    target = value.resolve()
    if value.is_symlink() or not target.is_file() or not os.access(target, os.X_OK):
        raise SensorPerformanceError("native visibility benchmark is unsafe")
    return target


def _run_native_benchmark(executable: Path) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [
                str(executable),
                "--warmup",
                str(NATIVE_WARMUP_COUNT),
                "--samples",
                str(NATIVE_SAMPLE_COUNT),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        document = json.loads(
            completed.stdout,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, subprocess.SubprocessError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise SensorPerformanceError("native visibility benchmark failed") from error
    if not isinstance(document, dict):
        raise SensorPerformanceError("native visibility benchmark output is invalid")
    return document


def _external_output(path: Path, repository_root: Path) -> Path:
    if not path.is_absolute():
        raise SensorPerformanceError("sensor performance output must be absolute")
    target = path
    if target.is_symlink() or not target.parent.is_dir():
        raise SensorPerformanceError("sensor performance output is unsafe")
    resolved_parent = target.parent.resolve(strict=True)
    repository = repository_root.resolve(strict=True)
    if resolved_parent == repository or repository in resolved_parent.parents:
        raise SensorPerformanceError(
            "sensor performance output must remain outside the repository"
        )
    return resolved_parent / target.name


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
