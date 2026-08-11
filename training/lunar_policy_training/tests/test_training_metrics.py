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


def _diagnostics(
    *,
    snapshot: str,
    universe: int,
    selected: int,
    available: int,
    failed: int,
    reserve: int = 0,
    zero_gain: int = 0,
    visited: int = 0,
    unreachable: int = 0,
) -> CandidateDiagnostics:
    return CandidateDiagnostics(
        physical_snapshot_id=snapshot * 64,
        physical_reachability_algorithm_id="test/reachability-v1",
        physical_candidate_universe_count=universe,
        selected_policy_candidate_count=selected,
        available_candidate_count=available,
        untried_reserve_count=reserve,
        planner_failed_current_snapshot_count=failed,
        zero_gain_count=zero_gain,
        visited_excluded_count=visited,
        physical_unreachable_count=unreachable,
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
        success_first_crossings=(False, False, True, False),
        planning_outcomes=(
            PlanningOutcome.NEW_REFERENCE_AVAILABLE,
            PlanningOutcome.RESOURCE_EXHAUSTED,
            PlanningOutcome.NEW_REFERENCE_AVAILABLE,
            PlanningOutcome.INVALID_REQUEST,
        ),
        candidate_diagnostics=(
            _diagnostics(snapshot="1", universe=5, selected=4, available=4, failed=1, unreachable=1),
            _diagnostics(snapshot="2", universe=6, selected=4, available=4, failed=2, unreachable=2),
            _diagnostics(snapshot="3", universe=7, selected=4, available=4, failed=3, unreachable=3),
            _diagnostics(snapshot="4", universe=8, selected=4, available=4, failed=4, unreachable=4),
        ),
        no_candidate_terminations=(False, True, False, False),
        terminal_audits=(
            None,
            TerminalAudit(
                reason=TerminalReason.HARD_FAILURE,
                oracle_opportunity_count=0,
                candidate_diagnostics=_diagnostics(
                    snapshot="2", universe=6, selected=4, available=4,
                    failed=2, unreachable=2,
                ),
                remaining_coverable_detail_cell_count=90,
            ),
            TerminalAudit(
                reason=TerminalReason.SUCCESS,
                oracle_opportunity_count=0,
                candidate_diagnostics=_diagnostics(
                    snapshot="3", universe=7, selected=4, available=4,
                    failed=3, unreachable=3,
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

    assert record["schema_version"] == "lunar-training-update-metrics/v4"
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
            "physical_candidate_universe_count": 26,
            "selected_policy_candidate_count": 16,
            "available_candidate_count": 16,
            "untried_reserve_count": 0,
            "planner_failed_current_snapshot_count": 10,
            "zero_gain_count": 0,
            "visited_excluded_count": 0,
            "physical_unreachable_count": 10,
            "no_candidate_termination_count": 1,
            "physical_exhaustion_count": 0,
            "planner_blocked_count": 0,
        }
    }
    assert record["candidate"]["identity_by_platform"] == {
        "LEGGED": {
            "physical_snapshot_ids": [character * 64 for character in "1234"],
            "physical_reachability_algorithm_ids": ["test/reachability-v1"],
        }
    }
    assert record["terminal"] == {
        "reason_counts": {"HARD_FAILURE": 1, "SUCCESS": 1},
        "reason_counts_by_platform": {
            "LEGGED": {"HARD_FAILURE": 1, "SUCCESS": 1}
        },
        "candidate_diagnostics_by_platform": {
            "LEGGED": {
                "physical_candidate_universe_count": 13,
                "selected_policy_candidate_count": 8,
                "available_candidate_count": 8,
                "untried_reserve_count": 0,
                "planner_failed_current_snapshot_count": 5,
                "zero_gain_count": 0,
                "visited_excluded_count": 0,
                "physical_unreachable_count": 5,
            }
        },
        "candidate_identity_by_platform": {
            "LEGGED": {
                "physical_snapshot_ids": ["2" * 64, "3" * 64],
                "physical_reachability_algorithm_ids": ["test/reachability-v1"],
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
    wheeled = _diagnostics(
        snapshot="a", universe=3, selected=2, available=2, failed=1,
        unreachable=1,
    )
    blocked = _diagnostics(
        snapshot="b", universe=7, selected=0, available=0, failed=7,
        unreachable=4,
    )
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
        candidate_diagnostics=(wheeled, blocked),
        no_candidate_terminations=(False, True),
        terminal_audits=(
            None,
            TerminalAudit(
                reason=TerminalReason.PLANNER_BLOCKED_WITH_OPPORTUNITY,
                oracle_opportunity_count=146,
                candidate_diagnostics=blocked,
                remaining_coverable_detail_cell_count=123,
            ),
        ),
        ppo_metrics=_ppo_metrics(),
        collect_wall_seconds=1.0,
        update_wall_seconds=0.5,
        worker_wait_seconds=0.25,
    )

    assert record["candidate"]["by_platform"]["WHEELED"] == {
        "physical_candidate_universe_count": 3,
        "selected_policy_candidate_count": 2,
        "available_candidate_count": 2,
        "untried_reserve_count": 0,
        "planner_failed_current_snapshot_count": 1,
        "zero_gain_count": 0,
        "visited_excluded_count": 0,
        "physical_unreachable_count": 1,
        "no_candidate_termination_count": 0,
        "physical_exhaustion_count": 0,
        "planner_blocked_count": 0,
    }
    assert record["candidate"]["by_platform"]["HOPPER"] == {
        "physical_candidate_universe_count": 7,
        "selected_policy_candidate_count": 0,
        "available_candidate_count": 0,
        "untried_reserve_count": 0,
        "planner_failed_current_snapshot_count": 7,
        "zero_gain_count": 0,
        "visited_excluded_count": 0,
        "physical_unreachable_count": 4,
        "no_candidate_termination_count": 1,
        "physical_exhaustion_count": 0,
        "planner_blocked_count": 1,
    }
    assert record["terminal"]["reason_counts_by_platform"] == {
        "HOPPER": {"PLANNER_BLOCKED_WITH_OPPORTUNITY": 1},
        "WHEELED": {},
    }
    assert record["terminal"]["oracle_opportunity_count"] == 146
    assert record["terminal"]["oracle_contradiction_count"] == 0
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
        row["schema_version"] == "lunar-training-update-metrics/v4"
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
def test_training_metrics_journal_rejects_missing_or_negative_physical_diagnostic(
    tmp_path, bad_value
) -> None:
    journal = TrainingMetricsJournal(
        tmp_path / "metrics" / "train.jsonl", resume_global_step=118
    )
    record = _record(119)
    diagnostics = record["candidate"]["by_platform"]["LEGGED"]
    if bad_value is None:
        diagnostics.pop("physical_candidate_universe_count")
    else:
        diagnostics["physical_candidate_universe_count"] = bad_value

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
