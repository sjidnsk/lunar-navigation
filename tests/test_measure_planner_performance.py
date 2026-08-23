import json
from pathlib import Path
from types import SimpleNamespace

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


def passing_sample(elapsed_ms: float = 100.0) -> dict[str, object]:
    return {**METRICS, "success": True, "failed_segment": -1,
            "elapsed_ms": elapsed_ms}


def passing_750m_sample(
    elapsed_ms: float = 50000.0, max_cycle_elapsed_ms: float = 100.0
) -> dict[str, object]:
    return {
        **passing_sample(elapsed_ms),
        "max_cycle_elapsed_ms": max_cycle_elapsed_ms,
    }


def test_750m_acceptance_requires_exactly_ten_successful_samples() -> None:
    accepted = recorder.evaluate_750m_acceptance(
        [passing_750m_sample() for _ in range(10)]
    )
    one_failed = [passing_750m_sample() for _ in range(10)]
    one_failed[6] = {**passing_750m_sample(), "success": False}

    assert accepted["passed"] is True
    assert accepted["successful_samples"] == 10
    assert recorder.evaluate_750m_acceptance(one_failed)["passed"] is False
    assert recorder.evaluate_750m_acceptance(one_failed[:9])["passed"] is False


def test_750m_acceptance_does_not_treat_total_task_time_as_request_latency() -> None:
    acceptance = recorder.evaluate_750m_acceptance(
        [passing_750m_sample(elapsed_ms=60000.0) for _ in range(10)]
    )

    assert acceptance["passed"] is True
    assert acceptance["latency_metric"] == "max_cycle_elapsed_ms"
    assert acceptance["slow_sample_indices"] == []


def test_750m_acceptance_fails_explicitly_when_cycle_latency_is_missing() -> None:
    samples = [passing_750m_sample() for _ in range(10)]
    del samples[4]["max_cycle_elapsed_ms"]

    acceptance = recorder.evaluate_750m_acceptance(samples)

    assert acceptance["passed"] is False
    assert acceptance["missing_latency_sample_indices"] == [4]
    assert any("max_cycle_elapsed_ms" in error for error in acceptance["errors"])


@pytest.mark.parametrize("max_cycle_elapsed_ms", [2000.0, 2999.999, 3000.0])
def test_750m_acceptance_uses_cycle_latency_for_two_and_three_second_gates(
    max_cycle_elapsed_ms: float,
) -> None:
    samples = [passing_750m_sample() for _ in range(10)]
    samples[7]["max_cycle_elapsed_ms"] = max_cycle_elapsed_ms

    acceptance = recorder.evaluate_750m_acceptance(samples)

    assert acceptance["passed"] is False
    assert acceptance["slow_sample_indices"] == [7]
    assert acceptance["hard_limit_sample_indices"] == (
        [7] if max_cycle_elapsed_ms >= 3000.0 else []
    )


def test_parse_metrics_line_preserves_unbound_cycle_classification_fields() -> None:
    extended = {
        **passing_750m_sample(),
        "cycle_target_met_count": 91,
        "cycle_target_missed_count": 3,
    }

    assert recorder.parse_metrics_line(metrics_line(extended)) == extended


def test_parse_metrics_line_rejects_non_numeric_max_cycle_elapsed() -> None:
    invalid = {**passing_750m_sample(), "max_cycle_elapsed_ms": "fast"}

    with pytest.raises(ValueError, match="max_cycle_elapsed_ms must be numeric"):
        recorder.parse_metrics_line(metrics_line(invalid))


def test_summarize_includes_max_cycle_elapsed_nearest_rank_statistics() -> None:
    samples = [
        passing_750m_sample(max_cycle_elapsed_ms=value)
        for value in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
                      100.0)
    ]

    assert recorder.summarize(samples)["max_cycle_elapsed_ms"] == {
        "p50": 50.0,
        "p95": 100.0,
        "max": 100.0,
    }


@pytest.mark.parametrize("elapsed_ms", [2000.0, 2999.999, 3000.0])
def test_acceptance_rejects_every_sample_at_or_above_two_seconds(
    elapsed_ms: float,
) -> None:
    acceptance = recorder.evaluate_acceptance(
        [passing_sample(elapsed_ms)], expected_samples=1
    )

    assert acceptance["passed"] is False
    assert acceptance["slow_sample_indices"] == [0]
    assert acceptance["hard_limit_sample_indices"] == (
        [0] if elapsed_ms >= 3000.0 else []
    )


def test_acceptance_keeps_values_below_two_seconds_eligible() -> None:
    acceptance = recorder.evaluate_acceptance(
        [passing_sample(1999.999)], expected_samples=1
    )

    assert acceptance["passed"] is True
    assert acceptance["slow_sample_indices"] == []
    assert acceptance["hard_limit_sample_indices"] == []


def test_simple_acceptance_enforces_nearest_rank_p95_and_baseline_regression() -> None:
    samples = [passing_sample(900.0) for _ in range(28)]
    samples.extend([passing_sample(999.0), passing_sample(1200.0)])

    accepted = recorder.evaluate_simple_acceptance(samples, baseline_p95_ms=910.0)
    regressed = recorder.evaluate_simple_acceptance(samples, baseline_p95_ms=900.0)

    assert accepted["elapsed_ms_p95"] == 999.0
    assert accepted["passed"] is True
    assert regressed["passed"] is False
    assert regressed["baseline_limit_ms"] == pytest.approx(990.0)


def test_simple_acceptance_requires_thirty_samples_and_strictly_subsecond_p95() -> None:
    too_few = recorder.evaluate_simple_acceptance(
        [passing_sample(500.0) for _ in range(29)], baseline_p95_ms=500.0
    )
    at_boundary = recorder.evaluate_simple_acceptance(
        [passing_sample(1000.0) for _ in range(30)], baseline_p95_ms=1000.0
    )

    assert too_few["passed"] is False
    assert at_boundary["passed"] is False


def test_collection_report_preserves_partial_samples_and_raw_child_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = [
        SimpleNamespace(returncode=0, stdout=metrics_line(passing_sample()), stderr=""),
        SimpleNamespace(returncode=7, stdout="partial stdout", stderr="planner crashed"),
    ]
    monkeypatch.setattr(
        recorder.subprocess, "run", lambda *args, **kwargs: outcomes.pop(0)
    )

    collection = recorder.collect_sample_report("wheel-test", "WheelPlanner.Case", 2)

    assert collection["samples"] == [passing_sample()]
    assert collection["errors"] == [{
        "repetition": 2,
        "command": ["wheel-test", "--gtest_filter=WheelPlanner.Case"],
        "exit_code": 7,
        "stdout": "partial stdout",
        "stderr": "planner crashed",
        "message": "repetition 2 exited with exit code 7",
    }]


def test_collection_report_keeps_metrics_emitted_by_a_failing_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_metrics = dict(METRICS)
    completed = SimpleNamespace(
        returncode=1,
        stdout="gtest failure\n" + metrics_line(failed_metrics),
        stderr="assertion failed",
    )
    monkeypatch.setattr(
        recorder.subprocess, "run", lambda *args, **kwargs: completed
    )

    collection = recorder.collect_sample_report("wheel-test", "WheelPlanner.Case", 1)

    assert collection["samples"] == [failed_metrics]
    assert collection["errors"][0]["exit_code"] == 1


def test_main_writes_samples_and_errors_when_a_child_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "report.json"
    outcomes = [
        SimpleNamespace(returncode=0, stdout=metrics_line(passing_sample()), stderr=""),
        SimpleNamespace(returncode=9, stdout="child output", stderr="child error"),
    ]
    monkeypatch.setattr(
        recorder,
        "parse_arguments",
        lambda: SimpleNamespace(
            binary="wheel-test",
            gtest_filter="WheelPlanner.Case",
            repetitions=2,
            output=output,
        ),
    )
    monkeypatch.setattr(
        recorder.subprocess, "run", lambda *args, **kwargs: outcomes.pop(0)
    )

    assert recorder.main() == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["samples"] == [passing_sample()]
    assert report["errors"][0]["exit_code"] == 9
    assert report["errors"][0]["stdout"] == "child output"
    assert report["errors"][0]["stderr"] == "child error"
    assert report["acceptance"]["passed"] is False
