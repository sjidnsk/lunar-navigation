from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lunar_planner_training_bridge import PlanningOutcome

import lunar_policy_training.evaluation.reward_v4_runtime as runtime_module
from lunar_policy_training.environment.candidate_builder import (
    CandidateDiagnostics,
)
from lunar_policy_training.environment.macro_step import (
    ExecutionEvents,
    TerminalAudit,
    TerminalReason,
)
from lunar_policy_training.environment.parallel_pool import (
    CompletedWorkerTransition,
    PreparedWorkerBoundary,
)
from lunar_policy_training.evaluation.report import FormalEvaluationBatch
from lunar_policy_training.evaluation.reward_v4_runtime import (
    RewardV4RuntimeEvaluationError,
    evaluate_reward_v4_fixed_grid,
)
from lunar_policy_training.evaluation.reward_v4_schedule import (
    RewardV4EvaluationTier,
)
from lunar_policy_training.policy.cross_attention import CrossAttentionPolicy
from lunar_policy_training.proxy_scenario import proxy_observation
from lunar_policy_training.reward import RewardComponentsV4, RewardInputsV4
from lunar_policy_training.reward_contract import (
    RewardTerminalClass,
    TaskScaleBucket,
)
from lunar_policy_training.reward_curriculum import PlatformType
from lunar_policy_training.reward_evaluation import (
    RewardEvaluationTask,
    build_reward_v4_evaluation_manifest,
)


_PAYLOAD_SHA = "c" * 64


class _Policy(CrossAttentionPolicy):
    def forward(self, _batch: object) -> object:
        return object()


class _FakePool:
    tasks: tuple[RewardEvaluationTask, ...] = ()
    created = 0

    def __init__(self, *, allocation: object, initial_episode_cursors: object, **_: object) -> None:
        type(self).created += 1
        self.worker_count = len(self.tasks)
        assert sum(dict(allocation).values()) == self.worker_count
        assert tuple(initial_episode_cursors) == tuple(
            task.evaluation_seed for task in self.tasks
        )
        self._queue: list[tuple[object, ...]] = []
        self._macro_counts = [0] * self.worker_count

    def __enter__(self) -> "_FakePool":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def reset(self) -> None:
        return None

    def prepare_workers(
        self, workers: tuple[int, ...], *, policy_version: int, reset: bool
    ) -> None:
        assert policy_version == 0
        assert reset is False
        self._queue.append(
            tuple(self._prepared(worker) for worker in workers)
        )

    def await_completed_workers(self) -> tuple[object, ...]:
        return self._queue.pop(0)

    def submit_actions(
        self,
        actions: dict[int, object],
        *,
        policy_version: int,
        reward_stage: object,
        reward_weights: object,
    ) -> None:
        assert policy_version == 0
        assert tuple(sorted(actions)) == tuple(actions)
        self._queue.append(
            tuple(self._completed(worker) for worker in actions)
        )

    def _state(self, worker: int) -> dict[str, object]:
        task = self.tasks[worker]
        return {
            "episode_cursor": task.evaluation_seed,
            "scale_bucket": task.scale_bucket.value,
            "priority_coverable_detail_cell_count": 0,
            "task_span_cells": 25,
        }

    def _observation(self, worker: int, *, step: int):
        return proxy_observation(
            worker,
            "WHEELED",
            step=step,
            coverage=0.1 * step,
        )

    def _prepared(self, worker: int) -> PreparedWorkerBoundary:
        return PreparedWorkerBoundary(
            worker_index=worker,
            observation=self._observation(
                worker, step=self._macro_counts[worker]
            ),
            policy_version=0,
            worker_state=self._state(worker),
            candidate_diagnostics=CandidateDiagnostics(),
            no_candidate_termination=False,
            terminal_audit=None,
            buffer_index=0,
        )

    def _completed(self, worker: int) -> CompletedWorkerTransition:
        before_step = self._macro_counts[worker]
        after_step = before_step + 1
        self._macro_counts[worker] = after_step
        done = after_step == 2
        diagnostics = CandidateDiagnostics()
        terminal_audit = (
            TerminalAudit(
                reason=TerminalReason.ZERO_EXPECTED_GAIN,
                candidate_diagnostics=diagnostics,
                remaining_coverable_detail_cell_count=10,
            )
            if done
            else None
        )
        return CompletedWorkerTransition(
            worker_index=worker,
            observation=self._observation(worker, step=after_step),
            policy_version=0,
            worker_state=self._state(worker),
            reward_components=RewardComponentsV4(
                coverage=0.0,
                success=0.0,
                terminal_gap=0.0,
                priority=0.0,
                path=0.0,
                total=0.0,
            ),
            reward_inputs=RewardInputsV4(
                platform_type="WHEELED",
                coverage_before=0.1 * before_step,
                coverage_after=0.1 * after_step,
                priority_before=0.0,
                priority_after=0.0,
                path_before_m=float(before_step),
                path_after_m=float(after_step),
                task_scale_m=100.0,
                terminal_class=(
                    RewardTerminalClass.VALID_INCOMPLETE_TERMINAL
                    if done
                    else RewardTerminalClass.CONTINUE
                ),
                success_first_crossing=False,
            ),
            terminal_class=(
                RewardTerminalClass.VALID_INCOMPLETE_TERMINAL
                if done
                else RewardTerminalClass.CONTINUE
            ),
            done=done,
            planning_outcome=PlanningOutcome.NEW_REFERENCE_AVAILABLE,
            reason_code="OK",
            execution_events=ExecutionEvents(),
            candidate_diagnostics=diagnostics,
            terminal_audit=terminal_audit,
            policy_decisions_consumed=1,
            success_first_crossing=False,
            buffer_index=0,
        )


def _manifest():
    return build_reward_v4_evaluation_manifest(
        platforms=(PlatformType.WHEELED,),
        scale_buckets=tuple(TaskScaleBucket),
        evaluation_seeds=(4081,),
    )


def _batch():
    return FormalEvaluationBatch(
        split="validation",
        factory=SimpleNamespace(scenario_schedule_id="test-schedule"),
        observation_template=proxy_observation(0, "WHEELED", step=0),
        scenario_seeds=(4081,),
    )


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakePool.tasks = _manifest().tasks
    _FakePool.created = 0
    monkeypatch.setattr(runtime_module, "ParallelEnvPool", _FakePool)
    monkeypatch.setattr(
        runtime_module,
        "sample_action",
        lambda *_args, **_kwargs: SimpleNamespace(
            selected_frontier_index=torch.zeros(4, dtype=torch.int64),
            selected_theta=torch.zeros(4, dtype=torch.float32),
        ),
    )


def test_sentinel_finishes_after_one_complete_macro_action_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fakes(monkeypatch)
    progress = (tmp_path / "sentinel-progress").resolve()
    kwargs = {
        "device": "cpu",
        "checkpoint_payload_sha256": _PAYLOAD_SHA,
        "manifest": _manifest(),
        "active_platforms": (PlatformType.WHEELED,),
        "batch": _batch(),
        "bootstrap_seed": 7,
        "bootstrap_resample_count": 1,
        "max_macro_actions_per_task": 1,
        "progress_directory": progress,
        "evaluation_tier": RewardV4EvaluationTier.SENTINEL,
    }

    first = evaluate_reward_v4_fixed_grid(_Policy(), **kwargs)
    second = evaluate_reward_v4_fixed_grid(_Policy(), **kwargs)

    assert tuple(row.macro_action_count for row in first.episodes) == (1,) * 4
    assert tuple(row.final_coverage for row in first.episodes) == pytest.approx(
        (0.1,) * 4
    )
    assert second.to_dict() == first.to_dict()
    assert _FakePool.created == 1
    assert len(tuple(progress.glob("*.json"))) == 4

    progress_file = sorted(progress.glob("*.json"))[0]
    payload = json.loads(progress_file.read_text(encoding="utf-8"))
    payload["checkpoint_payload_sha256"] = "d" * 64
    progress_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(
        RewardV4RuntimeEvaluationError,
        match="progress checkpoint identity differs",
    ):
        evaluate_reward_v4_fixed_grid(_Policy(), **kwargs)


def test_full_evaluation_keeps_running_until_natural_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fakes(monkeypatch)

    report = evaluate_reward_v4_fixed_grid(
        _Policy(),
        device="cpu",
        checkpoint_payload_sha256=_PAYLOAD_SHA,
        manifest=_manifest(),
        active_platforms=(PlatformType.WHEELED,),
        batch=_batch(),
        bootstrap_seed=7,
        bootstrap_resample_count=1,
        max_macro_actions_per_task=None,
        progress_directory=(tmp_path / "full-progress").resolve(),
        evaluation_tier=RewardV4EvaluationTier.FULL,
    )

    assert tuple(row.macro_action_count for row in report.episodes) == (2,) * 4
    assert tuple(row.final_coverage for row in report.episodes) == pytest.approx(
        (0.2,) * 4
    )


def test_remaining_evaluation_task_keeps_its_original_scale_lane() -> None:
    tasks = _manifest().tasks
    remaining = (tasks[2],)

    assert runtime_module._remaining_worker_topology(tasks, remaining) == ((2, 4),)
