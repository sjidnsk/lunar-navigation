"""Deterministic fixed-task Reward V4 runtime evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
import tempfile

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
    reward_episode_metrics_from_mapping,
)
from .report import (
    FormalEvaluationBatch,
    RewardV4EvaluationReport,
    build_reward_v4_evaluation_report,
    mission_coverage_ratio,
)
from .reward_v4_schedule import RewardV4EvaluationTier


_PLATFORM_INDEX = {
    PlatformType.WHEELED: 0,
    PlatformType.LEGGED: 1,
    PlatformType.HOPPER: 2,
}
_EVALUATION_PROGRESS_SCHEMA = "lunar-reward-v4-evaluation-progress/v1"


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
    max_macro_actions_per_task: int | None = None,
    progress_directory: Path | None = None,
    evaluation_tier: RewardV4EvaluationTier = RewardV4EvaluationTier.FULL,
) -> RewardV4EvaluationReport:
    """Evaluate one policy on the active subset of a frozen task manifest."""
    if not isinstance(policy, CrossAttentionPolicy):
        raise TypeError("Reward V4 runtime evaluation requires policy")
    if not isinstance(manifest, RewardEvaluationManifest):
        raise TypeError("Reward V4 runtime evaluation manifest is invalid")
    if not isinstance(batch, FormalEvaluationBatch):
        raise TypeError("Reward V4 runtime evaluation batch is invalid")
    _validate_runtime_evaluation_options(
        checkpoint_payload_sha256=checkpoint_payload_sha256,
        max_macro_actions_per_task=max_macro_actions_per_task,
        progress_directory=progress_directory,
        evaluation_tier=evaluation_tier,
    )
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
    completed = _load_evaluation_progress(
        progress_directory,
        tasks=tasks,
        checkpoint_payload_sha256=checkpoint_payload_sha256,
        evaluation_tier=evaluation_tier,
    )
    remaining_tasks = tuple(
        task for task in tasks if _task_key(task) not in completed
    )
    worker_topology = _remaining_worker_topology(tasks, remaining_tasks)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RewardV4RuntimeEvaluationError(
            "CUDA runtime evaluation requested without CUDA"
        )
    allocation = {
        platform.value: sum(
            task.platform is platform for task in remaining_tasks
        )
        for platform in active
    }
    cursors = tuple(task.evaluation_seed for task in remaining_tasks)
    states = {
        worker: _EpisodeState(
            task=task,
            coverage_curve=[],
            priority_curve=None,
            path_curve=[],
            task_scale_m=None,
            priority_denominator_present=None,
        )
        for worker, task in enumerate(remaining_tasks)
    }
    completed_workers: set[int] = set()
    pending: dict[int, PreparedWorkerBoundary] = {}
    was_training = policy.training
    policy.to(target_device).eval()
    try:
        if remaining_tasks:
            with ParallelEnvPool(
                allocation=allocation,
                observation_template=batch.observation_template,
                environment_factory=batch.factory,
                reward_fn=_evaluation_reward,
                worker_timeout_seconds=FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS,
                auto_reset=False,
                initial_episode_cursors=cursors,
                worker_topology=worker_topology,
            ) as pool:
                if pool.worker_count != len(remaining_tasks):
                    raise RewardV4RuntimeEvaluationError(
                        "runtime evaluation worker grid differs"
                    )
                pool.reset()
                pool.prepare_workers(
                    tuple(range(len(remaining_tasks))),
                    policy_version=0,
                    reset=False,
                )
                while len(completed_workers) != len(remaining_tasks):
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
                        if (
                            worker not in states
                            or worker in completed_workers
                        ):
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
                                episode = _finish_episode(state)
                                completed[_task_key(state.task)] = episode
                                completed_workers.add(worker)
                                _write_evaluation_progress(
                                    progress_directory,
                                    task=state.task,
                                    episode=episode,
                                    checkpoint_payload_sha256=(
                                        checkpoint_payload_sha256
                                    ),
                                    evaluation_tier=evaluation_tier,
                                )
                                continue
                            if not bool(
                                event.observation.candidate_mask.any()
                            ):
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
                        macro_action_count = len(state.coverage_curve) - 1
                        capped = (
                            max_macro_actions_per_task is not None
                            and macro_action_count
                            >= max_macro_actions_per_task
                        )
                        if event.done or capped:
                            episode = _finish_episode(state)
                            completed[_task_key(state.task)] = episode
                            completed_workers.add(worker)
                            _write_evaluation_progress(
                                progress_directory,
                                task=state.task,
                                episode=episode,
                                checkpoint_payload_sha256=(
                                    checkpoint_payload_sha256
                                ),
                                evaluation_tier=evaluation_tier,
                            )
                        else:
                            pool.prepare_workers(
                                (worker,), policy_version=0, reset=False
                            )
                    if actionable:
                        observations = _concatenate_observations(
                            tuple(
                                item.observation for item in actionable
                            ),
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
    episodes = tuple(completed[_task_key(task)] for task in tasks)
    return build_reward_v4_evaluation_report(
        manifest=filtered_manifest,
        episodes=episodes,
        checkpoint_payload_sha256=checkpoint_payload_sha256,
        enabled_r2_platforms=enabled_r2,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resample_count=bootstrap_resample_count,
    )


def _validate_runtime_evaluation_options(
    *,
    checkpoint_payload_sha256: str,
    max_macro_actions_per_task: int | None,
    progress_directory: Path | None,
    evaluation_tier: RewardV4EvaluationTier,
) -> None:
    if (
        not isinstance(checkpoint_payload_sha256, str)
        or len(checkpoint_payload_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in checkpoint_payload_sha256
        )
    ):
        raise ValueError("Reward V4 evaluation checkpoint identity is invalid")
    if not isinstance(evaluation_tier, RewardV4EvaluationTier):
        raise TypeError("Reward V4 evaluation tier is invalid")
    if evaluation_tier is RewardV4EvaluationTier.SENTINEL:
        if max_macro_actions_per_task != 1:
            raise ValueError(
                "Reward V4 sentinel requires one macro action per task"
            )
    elif max_macro_actions_per_task is not None:
        raise ValueError("Reward V4 full evaluation cannot be action-capped")
    if progress_directory is not None and (
        not isinstance(progress_directory, Path)
        or not progress_directory.is_absolute()
    ):
        raise ValueError("Reward V4 evaluation progress path is invalid")


def _task_key(
    task: RewardEvaluationTask,
) -> tuple[PlatformType, TaskScaleBucket, int]:
    return (task.platform, task.scale_bucket, task.evaluation_seed)


def _remaining_worker_topology(
    tasks: tuple[RewardEvaluationTask, ...],
    remaining_tasks: tuple[RewardEvaluationTask, ...],
) -> tuple[tuple[int, int], ...]:
    """Keep resumed task lanes tied to the original fixed evaluation grid."""
    topology: list[tuple[int, int]] = []
    for task in remaining_tasks:
        platform_tasks = tuple(
            candidate for candidate in tasks if candidate.platform is task.platform
        )
        if not platform_tasks:
            raise RewardV4RuntimeEvaluationError(
                "runtime evaluation task platform is unavailable"
            )
        try:
            lane = platform_tasks.index(task)
        except ValueError as error:
            raise RewardV4RuntimeEvaluationError(
                "runtime evaluation task is outside fixed grid"
            ) from error
        topology.append((lane, len(platform_tasks)))
    return tuple(topology)


def _evaluation_progress_path(
    directory: Path, task: RewardEvaluationTask
) -> Path:
    return directory / (
        f"{task.platform.value.lower()}-"
        f"{task.scale_bucket.value}-{task.evaluation_seed}.json"
    )


def _load_evaluation_progress(
    directory: Path | None,
    *,
    tasks: tuple[RewardEvaluationTask, ...],
    checkpoint_payload_sha256: str,
    evaluation_tier: RewardV4EvaluationTier,
) -> dict[
    tuple[PlatformType, TaskScaleBucket, int], RewardEpisodeMetrics
]:
    if directory is None or not directory.exists():
        return {}
    if not directory.is_dir():
        raise RewardV4RuntimeEvaluationError(
            "runtime evaluation progress path is not a directory"
        )
    completed: dict[
        tuple[PlatformType, TaskScaleBucket, int], RewardEpisodeMetrics
    ] = {}
    expected_paths = {
        _evaluation_progress_path(directory, task): task for task in tasks
    }
    unexpected = tuple(
        path
        for path in directory.glob("*.json")
        if path not in expected_paths
    )
    if unexpected:
        raise RewardV4RuntimeEvaluationError(
            "runtime evaluation progress contains an unknown task"
        )
    for path, task in expected_paths.items():
        if not path.exists():
            continue
        episode = _read_evaluation_progress(
            path,
            task=task,
            checkpoint_payload_sha256=checkpoint_payload_sha256,
            evaluation_tier=evaluation_tier,
        )
        completed[_task_key(task)] = episode
    return completed


def _write_evaluation_progress(
    directory: Path | None,
    *,
    task: RewardEvaluationTask,
    episode: RewardEpisodeMetrics,
    checkpoint_payload_sha256: str,
    evaluation_tier: RewardV4EvaluationTier,
) -> None:
    if directory is None:
        return
    path = _evaluation_progress_path(directory, task)
    if path.exists():
        existing = _read_evaluation_progress(
            path,
            task=task,
            checkpoint_payload_sha256=checkpoint_payload_sha256,
            evaluation_tier=evaluation_tier,
        )
        if existing != episode:
            raise RewardV4RuntimeEvaluationError(
                "runtime evaluation progress episode differs"
            )
        return
    body: dict[str, object] = {
        "schema_version": _EVALUATION_PROGRESS_SCHEMA,
        "checkpoint_payload_sha256": checkpoint_payload_sha256,
        "evaluation_tier": evaluation_tier.value,
        "task": task.to_dict(),
        "episode": episode.to_dict(),
    }
    payload = {**body, "progress_sha256": _progress_sha256(body)}
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_evaluation_progress(
    path: Path,
    *,
    task: RewardEvaluationTask,
    checkpoint_payload_sha256: str,
    evaluation_tier: RewardV4EvaluationTier,
) -> RewardEpisodeMetrics:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RewardV4RuntimeEvaluationError(
            "runtime evaluation progress is unreadable"
        ) from error
    fields = {
        "schema_version",
        "checkpoint_payload_sha256",
        "evaluation_tier",
        "task",
        "episode",
        "progress_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise RewardV4RuntimeEvaluationError(
            "runtime evaluation progress structure differs"
        )
    if payload["checkpoint_payload_sha256"] != checkpoint_payload_sha256:
        raise RewardV4RuntimeEvaluationError(
            "runtime evaluation progress checkpoint identity differs"
        )
    if payload["evaluation_tier"] != evaluation_tier.value:
        raise RewardV4RuntimeEvaluationError(
            "runtime evaluation progress tier differs"
        )
    if payload["task"] != task.to_dict():
        raise RewardV4RuntimeEvaluationError(
            "runtime evaluation progress task identity differs"
        )
    body = {key: payload[key] for key in fields if key != "progress_sha256"}
    if payload["progress_sha256"] != _progress_sha256(body):
        raise RewardV4RuntimeEvaluationError(
            "runtime evaluation progress hash differs"
        )
    try:
        episode = reward_episode_metrics_from_mapping(payload["episode"])
    except ValueError as error:
        raise RewardV4RuntimeEvaluationError(
            "runtime evaluation progress episode differs"
        ) from error
    if _task_key(task) != (
        episode.platform,
        episode.scale_bucket,
        episode.evaluation_seed,
    ):
        raise RewardV4RuntimeEvaluationError(
            "runtime evaluation progress episode identity differs"
        )
    return episode


def _progress_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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
