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


def collect_samples(binary: str, gtest_filter: str, repetitions: int) -> list[dict[str, Any]]:
    """Run one exact GoogleTest filter per repetition and retain every record."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    samples: list[dict[str, Any]] = []
    command = [str(binary), f"--gtest_filter={gtest_filter}"]
    for repetition in range(repetitions):
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"repetition {repetition + 1} exited with exit code {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )
        samples.append(parse_metrics_line(completed.stdout))
    return samples


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
        samples = collect_samples(arguments.binary, arguments.gtest_filter, arguments.repetitions)
        report = {
            "gtest_filter": arguments.gtest_filter,
            "repetitions": arguments.repetitions,
            "samples": samples,
            "summary": summarize(samples),
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"measure_planner_performance: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
