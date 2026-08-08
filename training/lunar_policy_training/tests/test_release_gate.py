from __future__ import annotations

from pathlib import Path

from lunar_policy_training.checkpoint import RunIdentity
from lunar_policy_training.training_semantics import training_semantics_sha256

from lunar_policy_training.evaluation.release_gate import (
    evaluate_release_gate,
    load_gate_rules,
)
from lunar_policy_training.evaluation.report import (
    EvaluationReport,
    MethodEvaluation,
    PlatformMetrics,
    _ScenarioEvidence,
    _aggregate_platform_metrics,
)


ROOT = Path(__file__).resolve().parents[3]


def _platform_metrics(coverage: float) -> PlatformMetrics:
    return PlatformMetrics(
        scenario_seeds=(101, 102),
        success_coverage_rate=coverage,
        safety_violation_count=0,
        invalid_action_count=0,
        output_finite_rate=1.0,
        platform_reference_mismatch_count=0,
        hopper_commitment_violation_count=0,
        selected_action_observed_safe_rate=1.0,
        deterministic_repeat_match_rate=1.0,
        planner_failure_rate=0.0,
        completion_time_s=10.0,
    )


def _report(
    *, wheeled: float, legged: float, hopper: float, proxy: bool = False
) -> EvaluationReport:
    platforms = {
        "WHEELED": _platform_metrics(wheeled),
        "LEGGED": _platform_metrics(legged),
        "HOPPER": _platform_metrics(hopper),
    }
    methods = tuple(
        MethodEvaluation(method=method, per_platform=platforms)
        for method in (
            "ppo_policy",
            "nearest_frontier",
            "gain_over_cost_frontier",
        )
    )
    return EvaluationReport(
        proxy=proxy,
        scenario_schedule_id=("proxy-schedule" if proxy else "formal-schedule"),
        run_identity=RunIdentity(
            run_kind=("development-smoke" if proxy else "formal"),
            data_sha256="1" * 64,
            split_sha256="2" * 64,
            generator_sha256="3" * 64,
            capability_sha256="4" * 64,
            reward_sha256="a" * 64,
            v3_sha256="6" * 64,
            training_semantics_sha256=training_semantics_sha256(),
        ),
        reward_hash="a" * 64,
        checkpoint_sha256="b" * 64,
        methods=methods,
    )


def test_gate_fails_when_only_hopper_is_below_95_percent() -> None:
    """Would fail if a three-platform mean could mask one failing platform."""
    result = evaluate_release_gate(
        _report(wheeled=0.97, legged=0.95, hopper=0.94),
        load_gate_rules(ROOT / "training/configs/candidate_gate_v1.yaml"),
    )

    assert result.passed is False
    assert result.failed_rules == ("HOPPER.success_coverage_rate_min",)
    assert result.formal_candidate_eligible is True


def test_proxy_high_scores_never_pass_formal_gate() -> None:
    """Would fail if development proxy metrics could authorize a release."""
    result = evaluate_release_gate(
        _report(wheeled=1.0, legged=1.0, hopper=1.0, proxy=True),
        load_gate_rules(ROOT / "training/configs/release_gate_v1.yaml"),
    )

    assert result.passed is False
    assert result.formal_candidate_eligible is False
    assert result.run_kind == "development-smoke"
    assert result.proxy is True
    assert result.failed_rules == ("formal_candidate_eligible",)


def test_release_gate_adds_action_safety_and_repeat_rules() -> None:
    """Would fail if the release configuration silently used candidate rules."""
    report = _report(wheeled=0.97, legged=0.96, hopper=0.95)
    hopper = report.method("ppo_policy").per_platform["HOPPER"]
    changed = report.replace_platform_metrics(
        method="ppo_policy",
        platform_type="HOPPER",
        metrics=hopper.replace(deterministic_repeat_match_rate=0.5),
    )

    result = evaluate_release_gate(
        changed,
        load_gate_rules(ROOT / "training/configs/release_gate_v1.yaml"),
    )

    assert result.failed_rules == (
        "HOPPER.deterministic_repeat_match_rate_min",
    )


def test_candidate_gate_rejects_event_derived_execution_violations() -> None:
    """Would fail if observed worker violations were replaced with zeroes."""
    report = _report(wheeled=0.97, legged=0.96, hopper=0.95)
    observed = _aggregate_platform_metrics(
        (
            _ScenarioEvidence(
                scenario_seed=12000,
                final_coverage=1.0,
                safety_violation_count=1,
                invalid_action_count=1,
                output_finite=True,
                platform_reference_mismatch_count=1,
                hopper_commitment_violation_count=1,
                selected_action_observed_safe_count=1,
                deterministic_match_count=1,
                planner_failure_count=0,
                executed_step_count=1,
                completion_step_count=1,
            ),
        )
    )
    changed = report.replace_platform_metrics(
        method="ppo_policy",
        platform_type="HOPPER",
        metrics=observed,
    )

    result = evaluate_release_gate(
        changed,
        load_gate_rules(ROOT / "training/configs/candidate_gate_v1.yaml"),
    )

    assert result.failed_rules == (
        "HOPPER.safety_violation_count_max",
        "HOPPER.invalid_action_count_max",
        "HOPPER.platform_reference_mismatch_count_max",
        "HOPPER.hopper_commitment_violation_count_max",
    )


def test_baselines_are_required_but_do_not_need_to_beat_ppo() -> None:
    """Would fail if diagnostics were turned into a PPO superiority gate."""
    report = _report(wheeled=0.95, legged=0.95, hopper=0.95)
    result = evaluate_release_gate(
        report,
        load_gate_rules(ROOT / "training/configs/candidate_gate_v1.yaml"),
    )

    assert result.passed is True
    assert result.failed_rules == ()
