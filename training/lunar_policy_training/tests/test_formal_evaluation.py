from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lunar_planner_training_bridge import PlanningOutcome


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "model_contract"))

import lunar_policy_training.evaluation.report as report_module  # noqa: E402
from lunar_policy_training.checkpoint import RunIdentity  # noqa: E402
from lunar_policy_training.evaluation.report import (  # noqa: E402
    FormalEvaluationBatch,
    FormalEvaluationIncomplete,
    REQUIRED_METHODS,
    evaluate_formal_policy,
    mission_coverage_ratio,
)
from lunar_policy_training.environment.macro_step import ExecutionEvents  # noqa: E402
from lunar_policy_training.policy.observation import PolicyBatch  # noqa: E402
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
                    success_first_crossing=False,
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
    assert all(
        set(method.per_split) == {"validation", "test", "holdout"}
        for method in report.methods
    )
    assert all(
        method.per_split[split][platform].scenario_seeds == (seed,)
        for method in report.methods
        for split, seed in (("validation", 11), ("test", 22), ("holdout", 33))
        for platform in ("WHEELED", "LEGGED", "HOPPER")
    )
    assert len(calls) == 9
    assert {call[1] for call in calls} == {"validation", "test", "holdout"}


def test_formal_evaluation_chunk_cursor_preserves_cache_order() -> None:
    assert report_module._formal_chunk_episode_cursor(0, 3) == 0
    assert report_module._formal_chunk_episode_cursor(3, 3) == 1
    assert report_module._formal_chunk_episode_cursor(95, 1) == 95
    with pytest.raises(ValueError, match="cache order"):
        report_module._formal_chunk_episode_cursor(3, 2)


def _repeat_batch(template: PolicyBatch, rows: int) -> PolicyBatch:
    return PolicyBatch(
        **{
            name: getattr(template, name).repeat(
                rows, *([1] * (getattr(template, name).ndim - 1))
            )
            for name in template.input_names
        }
    )


class _VariableLengthFormalPool:
    terminal_step: int | None = 5

    def __init__(self, **kwargs) -> None:
        rows = sum(kwargs["allocation"].values())
        self.observations = _repeat_batch(kwargs["observation_template"], rows)
        self.step_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def reset(self):
        return SimpleNamespace(observations=self.observations)

    def prepare_decision_boundaries(self, *, policy_version: int):
        return SimpleNamespace(
            observations=self.observations,
            dones=torch.zeros(
                self.observations.prior_channels.shape[0], dtype=torch.bool
            ),
        )

    def step(self, actions, *, policy_version: int):
        self.step_count += 1
        coverage = min(0.95, 0.19 * self.step_count)
        self.observations.pose_features[:, 4] = coverage
        rows = self.observations.prior_channels.shape[0]
        done = self.terminal_step is not None and self.step_count >= self.terminal_step
        return SimpleNamespace(
            observations=self.observations,
            rewards=torch.zeros(rows, dtype=torch.float32),
            dones=torch.full((rows,), done, dtype=torch.bool),
            execution_events=tuple(
                ExecutionEvents(selected_action_observed_safe=True)
                for _ in range(rows)
            ),
            planning_outcomes=tuple(
                PlanningOutcome.NEW_REFERENCE_AVAILABLE for _ in range(rows)
            ),
            success_first_crossings=torch.full(
                (rows,), done and coverage >= 0.95, dtype=torch.bool
            ),
        )

    def reset_terminated_workers(self, worker_indices, *, policy_version: int):
        return SimpleNamespace(observations=self.observations)


def _one_scenario_batch() -> FormalEvaluationBatch:
    return FormalEvaluationBatch(
        split="validation",
        factory=SimpleNamespace(scenario_schedule_id="cache/validation/v3"),
        observation_template=proxy_observation(0, "WHEELED", step=0),
        scenario_seeds=(11,),
    )


def test_formal_evaluation_runs_past_three_to_the_natural_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(report_module, "ParallelEnvPool", _VariableLengthFormalPool)

    result = report_module._evaluate_formal_chunk(
        CrossAttentionPolicy(),
        method="nearest_frontier",
        device=torch.device("cpu"),
        batch=_one_scenario_batch(),
        scenario_offset=0,
        scenario_seeds=(11,),
        watchdog_max_steps=20,
    )

    assert all(values[0].completion_step_count == 5 for values in result.values())
    assert all(values[0].executed_step_count == 5 for values in result.values())
    assert all(values[0].final_coverage == pytest.approx(0.95) for values in result.values())


def test_formal_success_rate_uses_crossing_not_float32_coverage_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RoundedFailurePool(_VariableLengthFormalPool):
        terminal_step = 1

        def step(self, actions, *, policy_version: int):
            result = super().step(actions, policy_version=policy_version)
            result.observations.pose_features[:, 4] = 0.9499995
            result.success_first_crossings.zero_()
            return result

    monkeypatch.setattr(report_module, "ParallelEnvPool", RoundedFailurePool)
    result = report_module._evaluate_formal_chunk(
        CrossAttentionPolicy(),
        method="nearest_frontier",
        device=torch.device("cpu"),
        batch=_one_scenario_batch(),
        scenario_offset=0,
        scenario_seeds=(11,),
        watchdog_max_steps=2,
    )

    metrics = report_module._aggregate_platform_metrics(
        result["WHEELED"], theta_active=True
    )
    assert result["WHEELED"][0].final_coverage == pytest.approx(0.9499995)
    assert metrics.success_coverage_rate == 0.0


def test_formal_evaluation_watchdog_never_becomes_a_failed_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NeverTerminalPool(_VariableLengthFormalPool):
        terminal_step = None

    monkeypatch.setattr(report_module, "ParallelEnvPool", NeverTerminalPool)

    with pytest.raises(FormalEvaluationIncomplete, match="EVALUATION_INCOMPLETE"):
        report_module._evaluate_formal_chunk(
            CrossAttentionPolicy(),
            method="nearest_frontier",
            device=torch.device("cpu"),
            batch=_one_scenario_batch(),
            scenario_offset=0,
            scenario_seeds=(11,),
            watchdog_max_steps=4,
        )


def test_mission_coverage_uses_the_authoritative_roi_ratio_not_canvas_mean() -> None:
    batch = proxy_observation(0, "WHEELED", step=0)
    batch.coverage_summary.zero_()
    batch.coverage_summary[:, 0, :8, :8] = 1.0
    batch.coverage_summary[:, 1, :8, :8] = 1.0
    batch.pose_features[:, 4] = 0.95

    np.testing.assert_allclose(
        mission_coverage_ratio(batch),
        np.asarray([0.95], dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )
