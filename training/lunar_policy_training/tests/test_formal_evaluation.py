from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "model_contract"))

import lunar_policy_training.evaluation.report as report_module  # noqa: E402
from lunar_policy_training.checkpoint import RunIdentity  # noqa: E402
from lunar_policy_training.evaluation.report import (  # noqa: E402
    FormalEvaluationBatch,
    REQUIRED_METHODS,
    evaluate_formal_policy,
)
from lunar_policy_training.policy.cross_attention import (  # noqa: E402
    CrossAttentionPolicy,
)
from lunar_policy_training.proxy_scenario import proxy_observation  # noqa: E402
from lunar_policy_training.reward import reward_weights_sha256  # noqa: E402


def _identity() -> RunIdentity:
    return RunIdentity(
        run_kind="formal",
        data_sha256="1" * 64,
        split_sha256="2" * 64,
        generator_sha256="3" * 64,
        capability_sha256="4" * 64,
        reward_sha256=reward_weights_sha256(),
        v3_sha256="6" * 64,
        training_semantics_sha256="7" * 64,
    )


def test_formal_evaluation_covers_three_non_train_splits_without_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def evaluate_batch(
        policy,
        *,
        method,
        device,
        batch,
    ):
        calls.append((method, batch.split))
        return {
            platform: tuple(
                report_module._ScenarioEvidence(
                    scenario_seed=scenario_seed,
                    final_coverage=0.5,
                    safety_violation_count=0,
                    invalid_action_count=0,
                    output_finite=True,
                    platform_reference_mismatch_count=0,
                    hopper_commitment_violation_count=0,
                    selected_action_observed_safe_count=1,
                    deterministic_match_count=1,
                    planner_failure_count=0,
                    executed_step_count=1,
                    completion_step_count=1,
                    selected_thetas_rad=(0.0,),
                )
                for scenario_seed in batch.scenario_seeds
            )
            for platform in ("WHEELED", "LEGGED", "HOPPER")
        }

    monkeypatch.setattr(
        report_module, "_evaluate_formal_batch", evaluate_batch
    )
    template = proxy_observation(0, "WHEELED", step=0)
    batches = tuple(
        FormalEvaluationBatch(
            split=split,
            factory=SimpleNamespace(scenario_schedule_id=f"cache/{split}/v3"),
            observation_template=template,
            scenario_seeds=(seed,),
        )
        for split, seed in (
            ("validation", 11),
            ("test", 22),
            ("holdout", 33),
        )
    )

    report = evaluate_formal_policy(
        CrossAttentionPolicy(),
        device="cpu",
        checkpoint_sha256="8" * 64,
        run_identity=_identity(),
        batches=batches,
    )

    assert report.proxy is False
    assert report.run_identity.run_kind == "formal"
    assert tuple(method.method for method in report.methods) == REQUIRED_METHODS
    assert all(
        metrics.scenario_seeds == (11, 22, 33)
        for method in report.methods
        for metrics in method.per_platform.values()
    )
    assert len(calls) == 9
    assert {call[1] for call in calls} == {"validation", "test", "holdout"}


def test_formal_evaluation_chunk_cursor_preserves_cache_order() -> None:
    assert report_module._formal_chunk_episode_cursor(0, 3) == 0
    assert report_module._formal_chunk_episode_cursor(3, 3) == 1
    assert report_module._formal_chunk_episode_cursor(95, 1) == 95
    with pytest.raises(ValueError, match="cache order"):
        report_module._formal_chunk_episode_cursor(3, 2)
