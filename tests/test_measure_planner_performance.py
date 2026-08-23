import json

import pytest

from tools import measure_planner_performance as recorder


METRICS = {
    "scenario": "750m",
    "success": False,
    "failed_segment": 21,
    "elapsed_ms": 1001.5,
    "expanded_states": 9151,
    "edge_evaluations": 24089,
    "state_labels": 9151,
    "sweep_cell_checks": 30117568,
}


def metrics_line(metrics: dict[str, object] = METRICS) -> str:
    return "[planner-metrics] " + json.dumps(metrics)


def test_parse_metrics_line_returns_the_complete_metric_record() -> None:
    assert recorder.parse_metrics_line("ordinary output\n" + metrics_line()) == METRICS


def test_parse_metrics_line_rejects_a_missing_required_key() -> None:
    incomplete = dict(METRICS)
    del incomplete["state_labels"]

    with pytest.raises(ValueError, match="state_labels"):
        recorder.parse_metrics_line(metrics_line(incomplete))


def test_parse_metrics_line_rejects_malformed_json() -> None:
    with pytest.raises(ValueError, match="malformed"):
        recorder.parse_metrics_line("[planner-metrics] {not-json}")


def test_collect_samples_rejects_a_nonzero_child_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    class Completed:
        returncode = 1
        stdout = metrics_line()
        stderr = "known planner failure"

    monkeypatch.setattr(recorder.subprocess, "run", lambda *args, **kwargs: Completed())

    with pytest.raises(RuntimeError, match="exit code 1"):
        recorder.collect_samples("wheel-test", "WheelPlanner.LongRange", 1)


def test_collect_samples_keeps_all_ten_raw_samples_and_uses_the_exact_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    class Completed:
        returncode = 0
        stderr = ""

        def __init__(self, elapsed_ms: float) -> None:
            metrics = dict(METRICS)
            metrics["elapsed_ms"] = elapsed_ms
            self.stdout = metrics_line(metrics)

    def run(command: list[str], **kwargs: object) -> Completed:
        commands.append(command)
        assert kwargs == {"check": False, "text": True, "capture_output": True}
        return Completed(float(len(commands)))

    monkeypatch.setattr(recorder.subprocess, "run", run)

    samples = recorder.collect_samples("wheel-test", "WheelPlanner.LongRange", 10)

    assert len(samples) == 10
    assert [sample["elapsed_ms"] for sample in samples] == list(range(1, 11))
    assert commands == [
        ["wheel-test", "--gtest_filter=WheelPlanner.LongRange"] for _ in range(10)
    ]


def test_summarize_uses_nearest_rank_for_p50_p95_and_max() -> None:
    samples = [
        {**METRICS, "elapsed_ms": value}
        for value in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0)
    ]

    summary = recorder.summarize(samples)

    assert summary["elapsed_ms"] == {"p50": 50.0, "p95": 100.0, "max": 100.0}
