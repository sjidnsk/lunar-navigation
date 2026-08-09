from __future__ import annotations

import json

import numpy as np
import pytest
from lunar_planner_training_bridge import PlanningOutcome

from lunar_policy_training.ppo.trainer import PPOUpdateMetrics
from lunar_policy_training.training_metrics import (
    TrainingMetricsError,
    TrainingMetricsJournal,
    build_training_update_record,
)


def _ppo_metrics() -> PPOUpdateMetrics:
    return PPOUpdateMetrics(
        total_loss=1.5,
        policy_loss=-0.25,
        value_loss=3.0,
        frontier_entropy=0.75,
        theta_entropy=0.5,
        approx_kl=0.0125,
        gradient_norm=0.8,
        clipped_gradient_norm=0.5,
        parameter_change_l2=0.025,
        optimizer_steps=4,
        epochs_completed=4,
        target_kl_early_stopped=False,
    )


def _record(global_step: int) -> dict[str, object]:
    return build_training_update_record(
        global_step=global_step,
        timestamp_utc="2026-08-09T02:00:00Z",
        curriculum_phase="warmup_legged",
        platform_allocation={"LEGGED": 2},
        worker_platforms=("LEGGED", "LEGGED"),
        rollout_horizon=2,
        raw_rewards=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        dones=np.asarray([[False, True], [True, False]], dtype=np.bool_),
        start_coverage=np.asarray([0.1, 0.2], dtype=np.float32),
        end_coverage=np.asarray([0.3, 0.5], dtype=np.float32),
        success_first_crossings=(False, True, False, True),
        planning_outcomes=(
            PlanningOutcome.NEW_REFERENCE_AVAILABLE,
            PlanningOutcome.RESOURCE_EXHAUSTED,
            PlanningOutcome.NEW_REFERENCE_AVAILABLE,
            PlanningOutcome.INVALID_REQUEST,
        ),
        ppo_metrics=_ppo_metrics(),
        collect_wall_seconds=5.0,
        update_wall_seconds=0.25,
        worker_wait_seconds=4.0,
    )


def test_build_training_update_record_preserves_learning_and_rollout_facts() -> None:
    record = _record(119)

    assert record["schema_version"] == "lunar-training-update-metrics/v1"
    assert record["global_step"] == 119
    assert record["transition_count"] == 4
    assert record["reward"]["mean"] == pytest.approx(2.5)
    assert record["reward"]["std"] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0]))
    assert record["reward"]["per_platform_mean"] == {
        "LEGGED": pytest.approx(2.5)
    }
    assert record["coverage"] == {
        "start_mean": pytest.approx(0.15),
        "end_mean": pytest.approx(0.4),
        "delta_mean": pytest.approx(0.25),
    }
    assert record["terminal_count"] == 2
    assert record["success_first_crossing_count"] == 2
    assert record["planner"]["outcome_counts"] == {
        "INVALID_REQUEST": 1,
        "NEW_REFERENCE_AVAILABLE": 2,
        "RESOURCE_EXHAUSTED": 1,
    }
    assert record["planner"]["success_rate"] == pytest.approx(0.5)
    assert record["ppo"]["approx_kl"] == pytest.approx(0.0125)
    assert record["ppo"]["gradient_norm"] == pytest.approx(0.8)
    assert record["timing"] == {
        "collect_wall_seconds": pytest.approx(5.0),
        "update_wall_seconds": pytest.approx(0.25),
        "worker_wait_seconds": pytest.approx(4.0),
    }


def test_training_metrics_journal_appends_exact_resume_sequence(tmp_path) -> None:
    path = tmp_path / "metrics" / "train.jsonl"
    journal = TrainingMetricsJournal(path, resume_global_step=118)
    journal.append(_record(119))

    resumed = TrainingMetricsJournal(path, resume_global_step=119)
    resumed.append(_record(120))

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["global_step"] for row in rows] == [119, 120]
    assert all(
        row["schema_version"] == "lunar-training-update-metrics/v1"
        for row in rows
    )


def test_training_metrics_journal_rejects_missing_or_duplicate_step(tmp_path) -> None:
    journal = TrainingMetricsJournal(
        tmp_path / "metrics" / "train.jsonl", resume_global_step=118
    )

    with pytest.raises(TrainingMetricsError, match="next global step"):
        journal.append(_record(120))

    journal.append(_record(119))
    with pytest.raises(TrainingMetricsError, match="next global step"):
        journal.append(_record(119))


def test_training_metrics_journal_rejects_nonfinite_metric(tmp_path) -> None:
    journal = TrainingMetricsJournal(
        tmp_path / "metrics" / "train.jsonl", resume_global_step=118
    )
    record = _record(119)
    record["ppo"]["value_loss"] = float("inf")

    with pytest.raises(TrainingMetricsError, match="finite"):
        journal.append(record)


def test_training_metrics_journal_rejects_rows_ahead_of_checkpoint(tmp_path) -> None:
    path = tmp_path / "metrics" / "train.jsonl"
    journal = TrainingMetricsJournal(path, resume_global_step=118)
    journal.append(_record(119))

    with pytest.raises(TrainingMetricsError, match="ahead of checkpoint"):
        TrainingMetricsJournal(path, resume_global_step=118)


def test_training_metrics_journal_reports_checkpoint_summary(tmp_path) -> None:
    path = tmp_path / "metrics" / "train.jsonl"
    journal = TrainingMetricsJournal(path, resume_global_step=118)
    journal.append(_record(119))

    summary = journal.checkpoint_summary(artifact_root=tmp_path)

    assert summary["schema_version"] == "lunar-training-metrics-summary/v1"
    assert summary["path"] == "metrics/train.jsonl"
    assert summary["last_global_step"] == 119
    assert len(summary["sha256"]) == 64
