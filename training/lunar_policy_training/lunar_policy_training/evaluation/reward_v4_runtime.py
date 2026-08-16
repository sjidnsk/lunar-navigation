"""Deterministic fixed-task Reward V4 runtime evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping

import torch

from ..config import FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS
from ..environment.macro_step import PlannerTransition, PolicyAction
from ..environment.parallel_pool import (
    CompletedWorkerTransition,
    ParallelEnvPool,
    PreparedWorkerBoundary,
    RejectedWorkerAction,
)
from ..polar_data.raster import GLOBAL_GEOMETRY
from ..policy.cross_attention import CrossAttentionPolicy, sample_action
from ..policy.observation import PolicyBatch, validate_policy_batch
from ..reward import DEFAULT_REWARD_WEIGHTS
from ..reward_contract import RewardStage, TaskScaleBucket
from ..reward_curriculum import PlatformType
from ..reward_evaluation import (
    RewardEpisodeMetrics,
    RewardEvaluationManifest,
    RewardEvaluationTask,
    build_reward_episode_metrics,
)
from .report import (
    FormalEvaluationBatch,
    RewardV4EvaluationReport,
    build_reward_v4_evaluation_report,
    mission_coverage_ratio,
)


_PLATFORM_INDEX = {
    PlatformType.WHEELED: 0,
    PlatformType.LEGGED: 1,
    PlatformType.HOPPER: 2,
}


class RewardV4RuntimeEvaluationError(RuntimeError):
    """Fixed evaluation input or worker evidence violated its contract."""


@dataclass(slots=True)
class _EpisodeState:
    task: RewardEvaluationTask
    coverage_curve: list[float]
    priority_curve: list[float] | None
    path_curve: list[float]
    task_scale_m: float | None
    priority_denominator_present: bool | None
    hard_error_count: int = 0


def evaluate_reward_v4_fixed_grid(
    policy: CrossAttentionPolicy,
    *,
    device: torch.device | str,
    checkpoint_payload_sha256: str,
    manifest: RewardEvaluationManifest,
    active_platforms: tuple[PlatformType, ...],
    batch: FormalEvaluationBatch,
    bootstrap_seed: int,
    bootstrap_resample_count: int,
    enabled_r2_platforms: tuple[PlatformType, ...] | None = None,
) -> RewardV4EvaluationReport:
    """Evaluate one policy on the active subset of a frozen task manifest."""
    if not isinstance(policy, CrossAttentionPolicy):
        raise TypeError("Reward V4 runtime evaluation requires policy")
    if not isinstance(manifest, RewardEvaluationManifest):
        raise TypeError("Reward V4 runtime evaluation manifest is invalid")
    if not isinstance(batch, FormalEvaluationBatch):
        raise TypeError("Reward V4 runtime evaluation batch is invalid")
    active = _canonical_active_platforms(active_platforms)
    enabled_r2 = (
        active
        if enabled_r2_platforms is None
        else _canonical_enabled_r2_platforms(
            enabled_r2_platforms, active=active
        )
    )
    tasks = tuple(task for task in manifest.tasks if task.platform in active)
    filtered_manifest = RewardEvaluationManifest(tasks=tasks)
    _validate_task_grid(tasks, active=active, batch=batch)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RewardV4RuntimeEvaluationError(
            "CUDA runtime evaluation requested without CUDA"
        )
    allocation = {
        platform.value: sum(task.platform is platform for task in tasks)
        for platform in active
    }
    cursors = tuple(task.evaluation_seed for task in tasks)
    states = {
        worker: _EpisodeState(
            task=task,
            coverage_curve=[],
            priority_curve=None,
            path_curve=[],
            task_scale_m=None,
            priority_denominator_present=None,
        )
        for worker, task in enumerate(tasks)
    }
    completed: dict[int, RewardEpisodeMetrics] = {}
    pending: dict[int, PreparedWorkerBoundary] = {}
    was_training = policy.training
    policy.to(target_device).eval()
    try:
        with ParallelEnvPool(
            allocation=allocation,
            observation_template=batch.observation_template,
            environment_factory=batch.factory,
            reward_fn=_evaluation_reward,
            worker_timeout_seconds=FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS,
            auto_reset=False,
            initial_episode_cursors=cursors,
        ) as pool:
            if pool.worker_count != len(tasks):
                raise RewardV4RuntimeEvaluationError(
                    "runtime evaluation worker grid differs"
                )
            pool.reset()
            pool.prepare_workers(
                tuple(range(len(tasks))), policy_version=0, reset=False
            )
            while len(completed) != len(tasks):
                events = pool.await_completed_workers()
                if not isinstance(events, tuple) or not events:
                    raise RewardV4RuntimeEvaluationError(
                        "runtime evaluation returned no worker event"
                    )
                actionable: list[PreparedWorkerBoundary] = []
                for event in events:
                    if not isinstance(
                        event,
                        (
                            PreparedWorkerBoundary,
                            CompletedWorkerTransition,
                            RejectedWorkerAction,
                        ),
                    ):
                        raise RewardV4RuntimeEvaluationError(
                            "runtime evaluation event type is invalid"
                        )
                    worker = event.worker_index
                    if worker not in states or worker in completed:
                        raise RewardV4RuntimeEvaluationError(
                            "runtime evaluation worker identity is invalid"
                        )
                    if event.policy_version != 0:
                        raise RewardV4RuntimeEvaluationError(
                            "runtime evaluation policy version differs"
                        )
                    state = states[worker]
                    _validate_worker_task(event.worker_state, state.task)
                    if isinstance(event, PreparedWorkerBoundary):
                        if worker in pending:
                            raise RewardV4RuntimeEvaluationError(
                                "runtime worker prepared with pending action"
                            )
                        _validate_observation_platform(
                            event.observation, state.task.platform
                        )
                        _initialize_task_facts(
                            state,
                            worker_state=event.worker_state,
                        )
                        if not event.actionable:
                            state.hard_error_count = int(
                                event.invalid_task_audit is not None
                            )
                            _initialize_no_action_curve(
                                state, event.observation
                            )
                            completed[worker] = _finish_episode(state)
                            continue
                        if not bool(event.observation.candidate_mask.any()):
                            raise RewardV4RuntimeEvaluationError(
                                "actionable evaluation worker has no candidate"
                            )
                        actionable.append(event)
                        continue
                    if isinstance(event, RejectedWorkerAction):
                        prepared = pending.pop(worker, None)
                        if prepared is None:
                            raise RewardV4RuntimeEvaluationError(
                                "runtime worker rejected without pending action"
                            )
                        if state.task.platform is PlatformType.HOPPER:
                            raise RewardV4RuntimeEvaluationError(
                                "HOPPER cannot emit ground reselection"
                            )
                        if (
                            event.candidate_diagnostics.physical_snapshot_id
                            != prepared.candidate_diagnostics.physical_snapshot_id
                        ):
                            raise RewardV4RuntimeEvaluationError(
                                "runtime rejection changed physical snapshot"
                            )
                        pool.prepare_workers(
                            (worker,), policy_version=0, reset=False
                        )
                        continue
                    prepared = pending.pop(worker, None)
                    if prepared is None:
                        raise RewardV4RuntimeEvaluationError(
                            "runtime worker completed without pending action"
                        )
                    _append_transition(state, event)
                    if event.done:
                        completed[worker] = _finish_episode(state)
                    else:
                        pool.prepare_workers(
                            (worker,), policy_version=0, reset=False
                        )
                if actionable:
                    observations = _concatenate_observations(
                        tuple(item.observation for item in actionable),
                        device=target_device,
                    )
                    with torch.no_grad():
                        output = policy(observations)
                        sampled = sample_action(
                            output,
                            observations.candidate_mask,
                            observations.platform_context,
                            deterministic=True,
                        )
                    actions: dict[int, PolicyAction] = {}
                    for row, prepared in enumerate(actionable):
                        worker = prepared.worker_index
                        pending[worker] = prepared
                        actions[worker] = PolicyAction(
                            frontier_index=int(
                                sampled.selected_frontier_index[row]
                                .detach()
                                .cpu()
                                .item()
                            ),
                            theta_rad=float(
                                sampled.selected_theta[row]
                                .detach()
                                .cpu()
                                .item()
                            ),
                        )
                    pool.submit_actions(
                        actions,
                        policy_version=0,
                        reward_stage=RewardStage.R1,
                        reward_weights=DEFAULT_REWARD_WEIGHTS,
                    )
            if pending:
                raise RewardV4RuntimeEvaluationError(
                    "runtime evaluation ended with pending actions"
                )
    finally:
        if was_training:
            policy.train()
    episodes = tuple(completed[index] for index in range(len(tasks)))
    return build_reward_v4_evaluation_report(
        manifest=filtered_manifest,
        episodes=episodes,
        checkpoint_payload_sha256=checkpoint_payload_sha256,
        enabled_r2_platforms=enabled_r2,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resample_count=bootstrap_resample_count,
    )


def _canonical_active_platforms(
    value: tuple[PlatformType, ...],
) -> tuple[PlatformType, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, PlatformType) for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("Reward V4 active platforms are invalid")
    expected = tuple(platform for platform in PlatformType if platform in value)
    if value != expected:
        raise ValueError("Reward V4 active platforms are not canonical")
    return value


def _canonical_enabled_r2_platforms(
    value: tuple[PlatformType, ...],
    *,
    active: tuple[PlatformType, ...],
) -> tuple[PlatformType, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, PlatformType) for item in value
    ):
        raise ValueError("Reward V4 enabled R2 platforms are invalid")
    expected = tuple(platform for platform in active if platform in value)
    if value != expected or len(set(value)) != len(value):
        raise ValueError("Reward V4 enabled R2 platforms are not canonical")
    return value


def _validate_task_grid(
    tasks: tuple[RewardEvaluationTask, ...],
    *,
    active: tuple[PlatformType, ...],
    batch: FormalEvaluationBatch,
) -> None:
    seeds = tuple(sorted(batch.scenario_seeds))
    for platform in active:
        platform_tasks = tuple(
            task for task in tasks if task.platform is platform
        )
        if not platform_tasks:
            raise ValueError("Reward V4 active platform has no tasks")
        by_bucket = {
            bucket: tuple(
                task.evaluation_seed
                for task in platform_tasks
                if task.scale_bucket is bucket
            )
            for bucket in TaskScaleBucket
        }
        if any(tuple(sorted(values)) != seeds for values in by_bucket.values()):
            raise ValueError("Reward V4 fixed task grid differs from batch")


def _validate_worker_task(
    worker_state: Mapping[str, object], task: RewardEvaluationTask
) -> None:
    if not isinstance(worker_state, Mapping):
        raise RewardV4RuntimeEvaluationError(
            "runtime evaluation worker state is invalid"
        )
    if (
        worker_state.get("episode_cursor") != task.evaluation_seed
        or worker_state.get("scale_bucket") != task.scale_bucket.value
    ):
        raise RewardV4RuntimeEvaluationError(
            "runtime evaluation task identity differs"
        )


def _validate_observation_platform(
    observation: PolicyBatch, platform: PlatformType
) -> None:
    expected = torch.zeros((1, 3), dtype=torch.float32)
    expected[0, _PLATFORM_INDEX[platform]] = 1.0
    if not torch.equal(observation.platform_context.detach().cpu(), expected):
        raise RewardV4RuntimeEvaluationError(
            "runtime evaluation platform observation differs"
        )


def _initialize_task_facts(
    state: _EpisodeState, *, worker_state: Mapping[str, object]
) -> None:
    priority_count = worker_state.get("priority_coverable_detail_cell_count")
    if type(priority_count) is not int or priority_count < 0:
        raise RewardV4RuntimeEvaluationError(
            "runtime priority denominator is invalid"
        )
    present = priority_count > 0
    if state.priority_denominator_present is None:
        state.priority_denominator_present = present
    elif state.priority_denominator_present is not present:
        raise RewardV4RuntimeEvaluationError(
            "runtime priority denominator changed"
        )
    if state.task_scale_m is None:
        span = worker_state.get("task_span_cells")
        if type(span) is int and span > 0:
            state.task_scale_m = span * GLOBAL_GEOMETRY.resolution_m


def _initialize_no_action_curve(
    state: _EpisodeState, observation: PolicyBatch
) -> None:
    coverage = float(mission_coverage_ratio(observation)[0])
    state.coverage_curve[:] = [coverage]
    state.path_curve[:] = [0.0]
    if state.priority_denominator_present:
        state.priority_curve = [0.0]
    else:
        state.priority_curve = None
    if state.task_scale_m is None:
        state.task_scale_m = 1.0


def _append_transition(
    state: _EpisodeState, event: CompletedWorkerTransition
) -> None:
    inputs = event.reward_inputs
    if (
        inputs.platform_type != state.task.platform.value
        or event.policy_decisions_consumed != 1
    ):
        raise RewardV4RuntimeEvaluationError(
            "runtime transition identity differs"
        )
    if state.task_scale_m is None:
        state.task_scale_m = float(inputs.task_scale_m)
    elif not math.isclose(
        state.task_scale_m,
        float(inputs.task_scale_m),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RewardV4RuntimeEvaluationError(
            "runtime task scale changed"
        )
    if not state.coverage_curve:
        state.coverage_curve.append(float(inputs.coverage_before))
        state.path_curve.append(float(inputs.path_before_m))
        if state.priority_denominator_present:
            state.priority_curve = [float(inputs.priority_before)]
    else:
        if not math.isclose(
            state.coverage_curve[-1],
            float(inputs.coverage_before),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ) or not math.isclose(
            state.path_curve[-1],
            float(inputs.path_before_m),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RewardV4RuntimeEvaluationError(
                "runtime transition curve is discontinuous"
            )
        if state.priority_denominator_present and (
            state.priority_curve is None
            or not math.isclose(
                state.priority_curve[-1],
                float(inputs.priority_before),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise RewardV4RuntimeEvaluationError(
                "runtime priority curve is discontinuous"
            )
    state.coverage_curve.append(float(inputs.coverage_after))
    state.path_curve.append(float(inputs.path_after_m))
    if state.priority_denominator_present:
        if state.priority_curve is None:
            raise RewardV4RuntimeEvaluationError(
                "runtime priority curve is missing"
            )
        state.priority_curve.append(float(inputs.priority_after))
    if event.done != (event.terminal_audit is not None):
        raise RewardV4RuntimeEvaluationError(
            "runtime terminal evidence disagrees"
        )


def _finish_episode(state: _EpisodeState) -> RewardEpisodeMetrics:
    if (
        not state.coverage_curve
        or not state.path_curve
        or state.task_scale_m is None
        or state.priority_denominator_present is None
    ):
        raise RewardV4RuntimeEvaluationError(
            "runtime evaluation episode is incomplete"
        )
    return build_reward_episode_metrics(
        platform=state.task.platform,
        scale_bucket=state.task.scale_bucket,
        evaluation_seed=state.task.evaluation_seed,
        coverage_curve=state.coverage_curve,
        priority_coverage_curve=state.priority_curve,
        cumulative_executed_path_m=state.path_curve,
        task_scale_m=state.task_scale_m,
        priority_denominator_present=state.priority_denominator_present,
        hard_error_count=state.hard_error_count,
    )


def _concatenate_observations(
    batches: tuple[PolicyBatch, ...], *, device: torch.device
) -> PolicyBatch:
    combined = PolicyBatch(
        **{
            name: torch.cat(
                [getattr(batch, name).to(device) for batch in batches], dim=0
            )
            for name in batches[0].input_names
        },
        observation_identities=tuple(
            batch.observation_identities[0] for batch in batches
        ),
    )
    try:
        validate_policy_batch(combined)
    except ValueError as error:
        raise RewardV4RuntimeEvaluationError(
            "runtime evaluation observation batch is invalid"
        ) from error
    return combined


def _evaluation_reward(_transition: PlannerTransition) -> float:
    return 0.0


__all__ = [
    "RewardV4RuntimeEvaluationError",
    "evaluate_reward_v4_fixed_grid",
]
