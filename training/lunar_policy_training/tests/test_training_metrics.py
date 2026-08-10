from __future__ import annotations

import json

import numpy as np
import pytest
from lunar_planner_training_bridge import PlanningOutcome

from lunar_policy_training.environment.candidate_builder import CandidateDiagnostics
from lunar_policy_training.environment.macro_step import (
    TerminalAudit,
    TerminalReason,
)
from lunar_policy_training.ppo.trainer import PPOUpdateMetrics
from lunar_policy_training.training_metrics import (
    TrainingMetricsError,
    TrainingMetricsJournal,
    build_training_update_record,
)


_ZERO_PRIMITIVE_DIAGNOSTICS = {
    "primitive_state_count": 0,
    "forward_reachable_state_count": 0,
    "returnable_state_count": 0,
    "recoverable_observation_state_count": 0,
    "frontier_hint_count": 0,
    "positive_gain_state_count": 0,
    "transit_state_count": 0,
    "invalidated_edge_count": 0,
    "revalidated_edge_count": 0,
}


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
        success_first_crossings=(False, False, True, False),
        planning_outcomes=(
            PlanningOutcome.NEW_REFERENCE_AVAILABLE,
            PlanningOutcome.RESOURCE_EXHAUSTED,
            PlanningOutcome.NEW_REFERENCE_AVAILABLE,
            PlanningOutcome.INVALID_REQUEST,
        ),
        candidate_diagnostics=(
            CandidateDiagnostics(
                frontier_anchor_count=5,
                platform_unreachable_count=1,
                emitted_count=4,
            ),
            CandidateDiagnostics(
                frontier_anchor_count=6,
                platform_unreachable_count=2,
                emitted_count=4,
            ),
            CandidateDiagnostics(
                frontier_anchor_count=7,
                platform_unreachable_count=3,
                emitted_count=4,
            ),
            CandidateDiagnostics(
                frontier_anchor_count=8,
                platform_unreachable_count=4,
                emitted_count=4,
            ),
        ),
        no_candidate_terminations=(False, True, False, False),
        terminal_audits=(
            None,
            TerminalAudit(
                reason=TerminalReason.HARD_FAILURE,
                oracle_opportunity_count=0,
                candidate_diagnostics=CandidateDiagnostics(
                    frontier_anchor_count=6,
                    platform_unreachable_count=2,
                    emitted_count=4,
                ),
                remaining_coverable_detail_cell_count=90,
            ),
            TerminalAudit(
                reason=TerminalReason.SUCCESS,
                oracle_opportunity_count=0,
                candidate_diagnostics=CandidateDiagnostics(
                    frontier_anchor_count=7,
                    platform_unreachable_count=3,
                    emitted_count=4,
                ),
                remaining_coverable_detail_cell_count=0,
            ),
            None,
        ),
        ppo_metrics=_ppo_metrics(),
        collect_wall_seconds=5.0,
        update_wall_seconds=0.25,
        worker_wait_seconds=4.0,
    )


def test_build_training_update_record_preserves_learning_and_rollout_facts() -> None:
    record = _record(119)

    assert record["schema_version"] == "lunar-training-update-metrics/v3"
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
    assert record["success_first_crossing_count"] == 1
    assert record["planner"]["outcome_counts"] == {
        "INVALID_REQUEST": 1,
        "NEW_REFERENCE_AVAILABLE": 2,
        "RESOURCE_EXHAUSTED": 1,
    }
    assert record["planner"]["success_rate"] == pytest.approx(0.5)
    assert record["planner"]["outcome_counts_by_platform"] == {
        "LEGGED": {
            "INVALID_REQUEST": 1,
            "NEW_REFERENCE_AVAILABLE": 2,
            "RESOURCE_EXHAUSTED": 1,
        }
    }
    assert record["candidate"]["by_platform"] == {
        "LEGGED": {
            "frontier_anchor_count": 26,
            "visited_excluded_count": 0,
            "static_infeasible_count": 0,
            "platform_unreachable_count": 10,
            "zero_gain_count": 0,
            "emitted_count": 16,
            "planner_rejected_count": 0,
            **_ZERO_PRIMITIVE_DIAGNOSTICS,
            "no_candidate_termination_count": 1,
            "planner_rejected_exhaustion_count": 0,
        }
    }
    assert record["terminal"] == {
        "reason_counts": {"HARD_FAILURE": 1, "SUCCESS": 1},
        "reason_counts_by_platform": {
            "LEGGED": {"HARD_FAILURE": 1, "SUCCESS": 1}
        },
        "candidate_diagnostics_by_platform": {
            "LEGGED": {
                "frontier_anchor_count": 13,
                "visited_excluded_count": 0,
                "static_infeasible_count": 0,
                "platform_unreachable_count": 5,
                "zero_gain_count": 0,
                "emitted_count": 8,
                "planner_rejected_count": 0,
                **_ZERO_PRIMITIVE_DIAGNOSTICS,
            }
        },
        "oracle_opportunity_count": 0,
        "oracle_contradiction_count": 0,
        "remaining_coverable_detail_cell_count": {
            "count": 2,
            "min": 0,
            "max": 90,
        },
    }
    assert record["ppo"]["approx_kl"] == pytest.approx(0.0125)
    assert record["ppo"]["gradient_norm"] == pytest.approx(0.8)
    assert record["timing"] == {
        "collect_wall_seconds": pytest.approx(5.0),
        "update_wall_seconds": pytest.approx(0.25),
        "worker_wait_seconds": pytest.approx(4.0),
    }


def test_candidate_and_planner_diagnostics_preserve_worker_platform_alignment() -> None:
    record = build_training_update_record(
        global_step=1,
        timestamp_utc="2026-08-09T02:00:00Z",
        curriculum_phase="joint",
        platform_allocation={"WHEELED": 1, "HOPPER": 1},
        worker_platforms=("WHEELED", "HOPPER"),
        rollout_horizon=1,
        raw_rewards=np.asarray([[1.0, -1.0]], dtype=np.float32),
        dones=np.asarray([[False, True]], dtype=np.bool_),
        start_coverage=np.asarray([0.1, 0.2], dtype=np.float32),
        end_coverage=np.asarray([0.2, 0.2], dtype=np.float32),
        success_first_crossings=(False, False),
        planning_outcomes=(
            PlanningOutcome.NEW_REFERENCE_AVAILABLE,
            PlanningOutcome.NO_KNOWN_SAFE_ROUTE,
        ),
        candidate_diagnostics=(
            CandidateDiagnostics(
                frontier_anchor_count=3,
                platform_unreachable_count=1,
                emitted_count=2,
                primitive_state_count=5,
                forward_reachable_state_count=4,
                returnable_state_count=4,
                recoverable_observation_state_count=3,
                frontier_hint_count=2,
                positive_gain_state_count=2,
                invalidated_edge_count=1,
                revalidated_edge_count=8,
            ),
            CandidateDiagnostics(
                frontier_anchor_count=11,
                platform_unreachable_count=4,
                emitted_count=7,
                planner_rejected_count=7,
                primitive_state_count=20,
                forward_reachable_state_count=15,
                returnable_state_count=12,
                recoverable_observation_state_count=11,
                frontier_hint_count=5,
                positive_gain_state_count=7,
                invalidated_edge_count=3,
                revalidated_edge_count=30,
            ),
        ),
        no_candidate_terminations=(False, True),
        terminal_audits=(
            None,
            TerminalAudit(
                reason=TerminalReason.PLANNER_REJECTED_ALL,
                oracle_opportunity_count=0,
                candidate_diagnostics=CandidateDiagnostics(
                    frontier_anchor_count=11,
                    platform_unreachable_count=4,
                    emitted_count=7,
                    planner_rejected_count=7,
                    primitive_state_count=20,
                    forward_reachable_state_count=15,
                    returnable_state_count=12,
                    recoverable_observation_state_count=11,
                    frontier_hint_count=5,
                    positive_gain_state_count=7,
                    invalidated_edge_count=3,
                    revalidated_edge_count=30,
                ),
                remaining_coverable_detail_cell_count=123,
            ),
        ),
        ppo_metrics=_ppo_metrics(),
        collect_wall_seconds=1.0,
        update_wall_seconds=0.5,
        worker_wait_seconds=0.25,
    )

    assert record["candidate"]["by_platform"]["WHEELED"] == {
        "frontier_anchor_count": 3,
        "visited_excluded_count": 0,
        "static_infeasible_count": 0,
        "platform_unreachable_count": 1,
        "zero_gain_count": 0,
        "emitted_count": 2,
        "planner_rejected_count": 0,
        "primitive_state_count": 5,
        "forward_reachable_state_count": 4,
        "returnable_state_count": 4,
        "recoverable_observation_state_count": 3,
        "frontier_hint_count": 2,
        "positive_gain_state_count": 2,
        "transit_state_count": 0,
        "invalidated_edge_count": 1,
        "revalidated_edge_count": 8,
        "no_candidate_termination_count": 0,
        "planner_rejected_exhaustion_count": 0,
    }
    assert record["candidate"]["by_platform"]["HOPPER"] == {
        "frontier_anchor_count": 11,
        "visited_excluded_count": 0,
        "static_infeasible_count": 0,
        "platform_unreachable_count": 4,
        "zero_gain_count": 0,
        "emitted_count": 7,
        "planner_rejected_count": 7,
        "primitive_state_count": 20,
        "forward_reachable_state_count": 15,
        "returnable_state_count": 12,
        "recoverable_observation_state_count": 11,
        "frontier_hint_count": 5,
        "positive_gain_state_count": 7,
        "transit_state_count": 0,
        "invalidated_edge_count": 3,
        "revalidated_edge_count": 30,
        "no_candidate_termination_count": 1,
        "planner_rejected_exhaustion_count": 1,
    }
    assert record["terminal"]["reason_counts_by_platform"] == {
        "HOPPER": {"PLANNER_REJECTED_ALL": 1},
        "WHEELED": {},
    }
    assert record["planner"]["outcome_counts_by_platform"] == {
        "HOPPER": {"NO_KNOWN_SAFE_ROUTE": 1},
        "WHEELED": {"NEW_REFERENCE_AVAILABLE": 1},
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
        row["schema_version"] == "lunar-training-update-metrics/v3"
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


@pytest.mark.parametrize("bad_value", (None, -1))
def test_training_metrics_journal_rejects_missing_or_negative_graph_diagnostic(
    tmp_path, bad_value
) -> None:
    journal = TrainingMetricsJournal(
        tmp_path / "metrics" / "train.jsonl", resume_global_step=118
    )
    record = _record(119)
    diagnostics = record["candidate"]["by_platform"]["LEGGED"]
    if bad_value is None:
        diagnostics.pop("primitive_state_count")
    else:
        diagnostics["primitive_state_count"] = bad_value

    with pytest.raises(TrainingMetricsError, match="diagnostics"):
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
