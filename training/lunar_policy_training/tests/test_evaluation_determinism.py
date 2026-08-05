from __future__ import annotations

import pytest
import torch
from lunar_planner_training_bridge import ExecutionDirective, PlanningOutcome

from lunar_policy_training.curriculum import CurriculumSchedule

from lunar_policy_training.evaluation.report import (
    CandidateEvaluation,
    EvaluationReport,
    MethodEvaluation,
    PlatformMetrics,
    _ScenarioEvidence,
    _aggregate_platform_metrics,
    _evaluation_reward,
    report_sha256,
    select_best_candidate,
)
from lunar_policy_training.evaluation.report import evaluate_proxy_policy
from lunar_policy_training.policy.cross_attention import CrossAttentionPolicy
from lunar_policy_training.environment.macro_step import (
    ExecutionEvents,
    PlannerTransition,
)
from lunar_policy_training.proxy_scenario import proxy_observation
from lunar_policy_training.reward import InvalidTransition


def _metrics(coverage: float, *, failures: float = 0.0, seconds: float = 5.0):
    return PlatformMetrics(
        scenario_seeds=(101, 102, 103),
        success_coverage_rate=coverage,
        safety_violation_count=0,
        invalid_action_count=0,
        output_finite_rate=1.0,
        platform_reference_mismatch_count=0,
        hopper_commitment_violation_count=0,
        selected_action_observed_safe_rate=1.0,
        deterministic_repeat_match_rate=1.0,
        planner_failure_rate=failures,
        completion_time_s=seconds,
    )


def _report(coverages: tuple[float, float, float]) -> EvaluationReport:
    platforms = {
        name: _metrics(coverage)
        for name, coverage in zip(
            ("WHEELED", "LEGGED", "HOPPER"), coverages
        )
    }
    return EvaluationReport(
        proxy=True,
        scenario_schedule_id="proxy-scenario-schedule-v1:abc",
        reward_hash="a" * 64,
        checkpoint_sha256="b" * 64,
        methods=tuple(
            MethodEvaluation(method=method, per_platform=platforms)
            for method in (
                "ppo_policy",
                "nearest_frontier",
                "gain_over_cost_frontier",
            )
        ),
    )


def test_same_scenario_report_has_identical_canonical_hash() -> None:
    """Would fail if map order, reruns, or timestamps changed report identity."""
    first = _report((0.95, 0.96, 0.97))
    second = _report((0.95, 0.96, 0.97))

    assert first.schema_version == "lunar-policy-release-evaluation/v1"
    assert report_sha256(first) == report_sha256(second)
    assert first.to_dict() == second.to_dict()


def test_platform_metrics_aggregate_observed_execution_events() -> None:
    """Would fail if release metrics were constants instead of worker evidence."""
    metrics = _aggregate_platform_metrics(
        (
            _ScenarioEvidence(
                scenario_seed=10,
                final_coverage=0.96,
                safety_violation_count=1,
                invalid_action_count=0,
                output_finite=True,
                platform_reference_mismatch_count=2,
                hopper_commitment_violation_count=3,
                selected_action_observed_safe_count=1,
                deterministic_match_count=1,
                planner_failure_count=0,
                executed_step_count=2,
                completion_step_count=2,
            ),
            _ScenarioEvidence(
                scenario_seed=11,
                final_coverage=0.50,
                safety_violation_count=4,
                invalid_action_count=5,
                output_finite=False,
                platform_reference_mismatch_count=6,
                hopper_commitment_violation_count=7,
                selected_action_observed_safe_count=1,
                deterministic_match_count=0,
                planner_failure_count=2,
                executed_step_count=2,
                completion_step_count=3,
            ),
        )
    )

    assert metrics.scenario_seeds == (10, 11)
    assert metrics.success_coverage_rate == 0.5
    assert metrics.safety_violation_count == 5
    assert metrics.invalid_action_count == 5
    assert metrics.output_finite_rate == 0.5
    assert metrics.platform_reference_mismatch_count == 8
    assert metrics.hopper_commitment_violation_count == 10
    assert metrics.selected_action_observed_safe_rate == 0.5
    assert metrics.deterministic_repeat_match_rate == 0.25
    assert metrics.planner_failure_rate == 0.5
    assert metrics.completion_time_s == 2.5


@pytest.mark.parametrize(
    "outcome",
    (
        PlanningOutcome.INVALID_REQUEST,
        PlanningOutcome.STALE_INPUT,
        PlanningOutcome.NUMERICAL_FAILURE,
    ),
)
def test_evaluation_reward_fails_closed_on_invalid_transition(
    outcome: PlanningOutcome,
) -> None:
    """Would fail if evaluation converted an invalid transition to reward zero."""
    transition = PlannerTransition(
        next_observation=proxy_observation(0, "WHEELED", step=0),
        mission_observed_delta=0.0,
        priority_observed_delta=0.0,
        normalized_plan_or_execution_cost=0.0,
        normalized_macro_step_time=0.0,
        executed_without_new_coverage=False,
        success_first_crossing=False,
        episode_ended_without_success=False,
        hard_safety_violation=False,
        cancellation_expected=False,
        cpp_exception=None,
        planning_outcome=outcome,
        execution_directive=ExecutionDirective.NO_SAFE_REFERENCE,
        reason_code=outcome.name,
        terminated=False,
        execution_events=ExecutionEvents(),
    )

    with pytest.raises(InvalidTransition):
        _evaluation_reward(transition)


def test_best_checkpoint_maximizes_minimum_platform_coverage() -> None:
    """Would fail if a high mean hid the lowest platform during selection."""
    high_mean_low_hopper = CandidateEvaluation(
        checkpoint="mean.pt", report=_report((1.0, 1.0, 0.90))
    )
    balanced = CandidateEvaluation(
        checkpoint="balanced.pt", report=_report((0.95, 0.95, 0.95))
    )

    assert select_best_candidate(
        (high_mean_low_hopper, balanced)
    ).checkpoint == "balanced.pt"


def test_candidate_ties_use_failure_rate_then_completion_time() -> None:
    """Would fail if candidate ordering ignored the approved tie breakers."""
    base = _report((0.95, 0.95, 0.95))
    slower = CandidateEvaluation(checkpoint="slow.pt", report=base)
    better_metrics = {
        name: metrics.replace(planner_failure_rate=0.0, completion_time_s=4.0)
        for name, metrics in base.method("ppo_policy").per_platform.items()
    }
    better = CandidateEvaluation(
        checkpoint="better.pt",
        report=base.replace_method(
            MethodEvaluation(method="ppo_policy", per_platform=better_metrics)
        ),
    )

    assert select_best_candidate((slower, better)).checkpoint == "better.pt"


def test_real_proxy_evaluation_is_deterministic_and_compares_three_methods() -> None:
    """Would fail if evaluation used precomputed scores or different schedules."""
    torch.manual_seed(4080)
    policy = CrossAttentionPolicy()
    first = evaluate_proxy_policy(
        policy,
        device="cpu",
        checkpoint_sha256="b" * 64,
        schedule=CurriculumSchedule(),
    )
    second = evaluate_proxy_policy(
        policy,
        device="cpu",
        checkpoint_sha256="b" * 64,
        schedule=CurriculumSchedule(),
    )

    assert tuple(method.method for method in first.methods) == (
        "ppo_policy",
        "nearest_frontier",
        "gain_over_cost_frontier",
    )
    assert report_sha256(first) == report_sha256(second)
    expected_seeds = {
        "WHEELED": (10000, 10001, 10002),
        "LEGGED": (11000, 11001, 11002),
        "HOPPER": (12000, 12001, 12002),
    }
    for method in first.methods:
        for platform, metrics in method.per_platform.items():
            assert metrics.scenario_seeds == expected_seeds[platform]
    gain_over_cost = first.method("gain_over_cost_frontier")
    assert all(
        metrics.success_coverage_rate == 1.0
        and metrics.completion_time_s == 2.0
        for metrics in gain_over_cost.per_platform.values()
    )


@pytest.mark.cuda
def test_cuda_marker_is_reserved_for_real_public_evaluation_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    report = evaluate_proxy_policy(
        CrossAttentionPolicy(),
        device="cuda",
        checkpoint_sha256="b" * 64,
        schedule=CurriculumSchedule(),
    )

    assert report.proxy is True
    assert report_sha256(report) == report_sha256(report)
