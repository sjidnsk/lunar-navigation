#!/usr/bin/env python3
"""Repeat one planner GoogleTest and write its emitted metrics as JSON."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


METRICS_PREFIX = "[planner-metrics] "
REQUIRED_KEYS = {
    "scenario",
    "success",
    "failed_segment",
    "elapsed_ms",
    "expanded_states",
    "edge_evaluations",
    "state_labels",
    "sweep_cell_checks",
}
NUMERIC_KEYS = REQUIRED_KEYS - {"scenario", "success"}
TARGET_LIMIT_MS = 1000.0
SLA_FAILURE_MS = 2000.0
HARD_LIMIT_MS = 3000.0
MAXIMUM_BASELINE_REGRESSION_FRACTION = 0.10


def parse_metrics_line(output: str) -> dict[str, Any]:
    """Return the sole complete metrics record emitted by one test process."""
    metric_lines = [line[len(METRICS_PREFIX) :] for line in output.splitlines()
                    if line.startswith(METRICS_PREFIX)]
    if len(metric_lines) != 1:
        raise ValueError(f"expected exactly one planner metrics line, found {len(metric_lines)}")
    try:
        metrics = json.loads(metric_lines[0])
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed planner metrics JSON: {error}") from error
    if not isinstance(metrics, dict):
        raise ValueError("planner metrics JSON must be an object")
    missing = REQUIRED_KEYS - metrics.keys()
    unexpected = metrics.keys() - REQUIRED_KEYS
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unexpected:
            details.append("unexpected " + ", ".join(sorted(unexpected)))
        raise ValueError("planner metrics keys: " + "; ".join(details))
    if not isinstance(metrics["scenario"], str):
        raise ValueError("scenario must be a string")
    if not isinstance(metrics["success"], bool):
        raise ValueError("success must be a boolean")
    for key in NUMERIC_KEYS:
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    return metrics


def collect_sample_report(
    binary: str, gtest_filter: str, repetitions: int
) -> dict[str, list[dict[str, Any]]]:
    """Run every repetition and retain valid samples plus raw child errors."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    command = [str(binary), f"--gtest_filter={gtest_filter}"]
    for repetition in range(repetitions):
        try:
            completed = subprocess.run(
                command, check=False, text=True, capture_output=True
            )
        except OSError as error:
            errors.append({
                "repetition": repetition + 1,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": str(error),
                "message": f"repetition {repetition + 1} could not start: {error}",
            })
            break
        try:
            metrics = parse_metrics_line(completed.stdout)
        except ValueError as error:
            metrics = None
            metrics_error = error
        if completed.returncode != 0:
            if metrics is not None:
                samples.append(metrics)
            errors.append({
                "repetition": repetition + 1,
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "message": (
                    f"repetition {repetition + 1} exited with exit code "
                    f"{completed.returncode}"
                ),
            })
            continue
        if metrics is None:
            errors.append({
                "repetition": repetition + 1,
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "message": f"repetition {repetition + 1}: {metrics_error}",
            })
            continue
        samples.append(metrics)
    return {"samples": samples, "errors": errors}


def collect_samples(binary: str, gtest_filter: str, repetitions: int) -> list[dict[str, Any]]:
    """Run one exact GoogleTest filter per repetition and return valid records."""
    collection = collect_sample_report(binary, gtest_filter, repetitions)
    if collection["errors"]:
        error = collection["errors"][0]
        detail = str(error["stderr"]).strip() or str(error["stdout"]).strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(str(error["message"]) + suffix)
    return collection["samples"]


def _nearest_rank(values: list[float | int], percentile: float) -> float | int:
    sorted_samples = sorted(values)
    return sorted_samples[math.ceil(percentile * len(sorted_samples)) - 1]


def summarize(samples: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Compute nearest-rank p50/p95 and max for each recorded numeric metric."""
    if not samples:
        raise ValueError("cannot summarize zero samples")
    summary: dict[str, dict[str, float | int]] = {}
    for key in sorted(NUMERIC_KEYS):
        values = [sample[key] for sample in samples]
        summary[key] = {
            "p50": _nearest_rank(values, 0.50),
            "p95": _nearest_rank(values, 0.95),
            "max": max(values),
        }
    return summary


def evaluate_acceptance(
    samples: list[dict[str, Any]],
    *,
    expected_samples: int | None = None,
    minimum_samples: int | None = None,
    p95_limit_ms: float | None = None,
    baseline_p95_ms: float | None = None,
    maximum_baseline_regression_fraction: float = (
        MAXIMUM_BASELINE_REGRESSION_FRACTION
    ),
) -> dict[str, Any]:
    """Evaluate functional success and the common planner timing boundaries."""
    if expected_samples is not None and expected_samples < 1:
        raise ValueError("expected_samples must be positive")
    if minimum_samples is not None and minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    for name, value in (
        ("p95_limit_ms", p95_limit_ms),
        ("baseline_p95_ms", baseline_p95_ms),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if (not math.isfinite(maximum_baseline_regression_fraction) or
            maximum_baseline_regression_fraction < 0.0):
        raise ValueError(
            "maximum_baseline_regression_fraction must be finite and nonnegative"
        )

    errors: list[str] = []
    sample_count = len(samples)
    successful_samples = sum(sample["success"] is True for sample in samples)
    unsuccessful_sample_indices = [
        index for index, sample in enumerate(samples) if sample["success"] is not True
    ]
    slow_sample_indices = [
        index for index, sample in enumerate(samples)
        if sample["elapsed_ms"] >= SLA_FAILURE_MS
    ]
    hard_limit_sample_indices = [
        index for index, sample in enumerate(samples)
        if sample["elapsed_ms"] >= HARD_LIMIT_MS
    ]
    elapsed_ms_p95 = (
        _nearest_rank([sample["elapsed_ms"] for sample in samples], 0.95)
        if samples else None
    )
    baseline_limit_ms = (
        baseline_p95_ms * (1.0 + maximum_baseline_regression_fraction)
        if baseline_p95_ms is not None else None
    )

    if expected_samples is not None and sample_count != expected_samples:
        errors.append(
            f"expected {expected_samples} samples, collected {sample_count}"
        )
    if minimum_samples is not None and sample_count < minimum_samples:
        errors.append(
            f"expected at least {minimum_samples} samples, collected {sample_count}"
        )
    if unsuccessful_sample_indices:
        errors.append(
            "planner reported failure for samples " +
            ", ".join(str(index) for index in unsuccessful_sample_indices)
        )
    if slow_sample_indices:
        errors.append(
            f"samples at or above {SLA_FAILURE_MS:g} ms: " +
            ", ".join(str(index) for index in slow_sample_indices)
        )
    if hard_limit_sample_indices:
        errors.append(
            f"samples at or above hard {HARD_LIMIT_MS:g} ms limit: " +
            ", ".join(str(index) for index in hard_limit_sample_indices)
        )
    if (p95_limit_ms is not None and elapsed_ms_p95 is not None and
            elapsed_ms_p95 >= p95_limit_ms):
        errors.append(
            f"elapsed_ms p95 {elapsed_ms_p95:g} is not below "
            f"{p95_limit_ms:g} ms"
        )
    if (baseline_limit_ms is not None and elapsed_ms_p95 is not None and
            elapsed_ms_p95 > baseline_limit_ms):
        errors.append(
            f"elapsed_ms p95 {elapsed_ms_p95:g} exceeds baseline limit "
            f"{baseline_limit_ms:g} ms"
        )

    return {
        "passed": not errors,
        "sample_count": sample_count,
        "successful_samples": successful_samples,
        "expected_samples": expected_samples,
        "minimum_samples": minimum_samples,
        "slow_sample_indices": slow_sample_indices,
        "hard_limit_sample_indices": hard_limit_sample_indices,
        "elapsed_ms_p95": elapsed_ms_p95,
        "p95_limit_ms": p95_limit_ms,
        "baseline_p95_ms": baseline_p95_ms,
        "baseline_limit_ms": baseline_limit_ms,
        "maximum_baseline_regression_fraction": (
            maximum_baseline_regression_fraction
        ),
        "errors": errors,
    }


def evaluate_750m_acceptance(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Require the declared ten-out-of-ten 750 m acceptance run."""
    return evaluate_acceptance(samples, expected_samples=10)


def evaluate_simple_acceptance(
    samples: list[dict[str, Any]], baseline_p95_ms: float
) -> dict[str, Any]:
    """Require 30 simple samples, subsecond p95 and <=10% baseline regression."""
    return evaluate_acceptance(
        samples,
        minimum_samples=30,
        p95_limit_ms=TARGET_LIMIT_MS,
        baseline_p95_ms=baseline_p95_ms,
    )


def _cli_acceptance(
    samples: list[dict[str, Any]], repetitions: int
) -> dict[str, Any]:
    scenarios = {sample["scenario"] for sample in samples}
    if scenarios == {"750m"}:
        return evaluate_750m_acceptance(samples)
    return evaluate_acceptance(
        samples,
        expected_samples=repetitions,
        p95_limit_ms=TARGET_LIMIT_MS if repetitions >= 30 else None,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, help="GoogleTest binary to execute")
    parser.add_argument("--gtest-filter", required=True, help="single GoogleTest filter")
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        collection = collect_sample_report(
            arguments.binary, arguments.gtest_filter, arguments.repetitions
        )
        samples = collection["samples"]
        acceptance = _cli_acceptance(samples, arguments.repetitions)
        if collection["errors"]:
            acceptance["passed"] = False
            acceptance["errors"] = [
                *acceptance["errors"],
                *(error["message"] for error in collection["errors"]),
            ]
        report = {
            "gtest_filter": arguments.gtest_filter,
            "repetitions": arguments.repetitions,
            "samples": samples,
            "errors": collection["errors"],
            "summary": summarize(samples) if samples else {},
            "acceptance": acceptance,
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"measure_planner_performance: {error}", file=sys.stderr)
        return 1
    if not acceptance["passed"]:
        for error in acceptance["errors"]:
            print(f"measure_planner_performance: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
