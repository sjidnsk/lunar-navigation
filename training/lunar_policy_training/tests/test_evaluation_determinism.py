from __future__ import annotations

import pytest
import torch

from lunar_policy_training.curriculum import CurriculumSchedule

from lunar_policy_training.evaluation.report import (
    CandidateEvaluation,
    EvaluationReport,
    MethodEvaluation,
    PlatformMetrics,
    report_sha256,
    select_best_candidate,
)
from lunar_policy_training.evaluation.report import evaluate_proxy_policy
from lunar_policy_training.policy.cross_attention import CrossAttentionPolicy


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
    for method in first.methods:
        for platform, metrics in method.per_platform.items():
            assert metrics.scenario_seeds == (
                CurriculumSchedule()
                .scenario_for(platform_type=platform, scenario_index=0)
                .scenario_seed,
            )
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
