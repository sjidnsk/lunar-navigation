"""In-process rollout collection for the shared seven-input policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from time import perf_counter
from typing import Protocol

import numpy as np
import torch
from lunar_model_contract import ObservationContractV4

from ..policy.cross_attention import (
    ActionSample,
    CrossAttentionPolicy,
    PolicyOutput,
    recompute_action_log_probs,
    sample_action,
)
from ..policy.observation import PolicyBatch, validate_policy_batch
from ..environment.macro_step import PolicyAction, TerminalAudit
from ..environment.parallel_pool import (
    CompletedWorkerTransition,
    InvalidTaskAudit,
    PreparedWorkerBoundary,
    RejectedWorkerAction,
)
from ..environment.task_area import WorkerStratum
from ..recovery.transition_journal import (
    CommittedTransition,
    MacroTransitionPayload,
    TransitionJournal,
    build_pre_worker_audit,
)
from ..reward import DEFAULT_REWARD_WEIGHTS, compute_reward_components
from ..reward_contract import RewardStage, RewardWeightsV4
from .rollout import (
    PLATFORM_ID_BY_TYPE,
    SCALE_BUCKET_ID_BY_BUCKET,
    RolloutBatch,
    compute_gae,
    compute_stratified_gae,
)


class CollectorError(ValueError):
    """An environment transition cannot enter an in-process rollout."""


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    horizon: int
    deterministic: bool

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or self.horizon <= 0:
            raise CollectorError("horizon must be a positive integer")
        if type(self.deterministic) is not bool:
            raise CollectorError("deterministic must be boolean")


@dataclass(slots=True)
class EnvStep:
    observations: PolicyBatch
    rewards: np.ndarray
    dones: np.ndarray


class VectorEnv(Protocol):
    env_count: int

    def reset(self) -> PolicyBatch:
        raise NotImplementedError

    def step(
        self, candidate_indices: np.ndarray, thetas: np.ndarray
    ) -> EnvStep:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class CollectedRollout:
    rollout: RolloutBatch
    rewards: np.ndarray
    dones: np.ndarray


@dataclass(frozen=True, slots=True)
class TaskOutcomeAudit:
    """A legal no-action task terminal that does not consume a PPO slot."""

    worker_index: int
    episode_id: str
    terminal_audit: TerminalAudit
    post_worker_state: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CommittedMacroRollout:
    """One sealed update ordered as strict [slot, worker] time series."""

    update_id: int
    policy_version: int
    actions_per_worker: int
    committed: tuple[CommittedTransition, ...]
    rollout: RolloutBatch
    task_outcomes: tuple[TaskOutcomeAudit, ...]
    journal_sha256: str
    invalid_tasks: tuple[InvalidTaskAudit, ...] = ()
    rejected_actions: tuple[RejectedWorkerAction, ...] = ()
    macro_action_elapsed_s_by_worker: tuple[float, ...] = ()
    planner_call_count_by_worker: tuple[int, ...] = ()
    planner_elapsed_s_by_worker: tuple[float, ...] = ()
    policy_inference_elapsed_s: float = 0.0

    @property
    def slot_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for item in self.committed:
            counts[item.worker_index] = counts.get(item.worker_index, 0) + 1
        return counts


def collect_rollout(
    environment: VectorEnv,
    policy: CrossAttentionPolicy,
    config: CollectorConfig,
    *,
    device: torch.device | str,
) -> CollectedRollout:
    """Collect one fixed-length rollout and compute terminal-aware advantages."""
    if not isinstance(config, CollectorConfig):
        raise CollectorError("config must use CollectorConfig")
    if not isinstance(policy, CrossAttentionPolicy):
        raise CollectorError("policy must use CrossAttentionPolicy")
    env_count = getattr(environment, "env_count", None)
    if type(env_count) is not int or env_count <= 0:
        raise CollectorError("environment count must be a positive integer")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise CollectorError("requested CUDA device is unavailable")
    policy = policy.to(target_device)
    observations, _ = _prepare_policy_observations(
        environment,
        _validated_observations(
            environment.reset(),
            env_count=env_count,
            device=target_device,
            allow_no_candidates=True,
        ),
        env_count=env_count,
        device=target_device,
    )

    observation_rows: dict[str, list[np.ndarray]] = {
        name: [] for name in ObservationContractV4.input_names
    }
    selected_indices: list[np.ndarray] = []
    selected_thetas: list[np.ndarray] = []
    old_log_probs: list[np.ndarray] = []
    old_values: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    dones: list[np.ndarray] = []

    for _ in range(config.horizon):
        stored_observation = _batch_numpy(observations)
        for name, value in stored_observation.items():
            observation_rows[name].append(value)
        with torch.no_grad():
            output = policy(observations)
            sampled = sample_action(
                output,
                observations.candidate_mask,
                observations.platform_context,
                deterministic=config.deterministic,
            )
            recomputed = recompute_action_log_probs(
                output,
                observations.candidate_mask,
                sampled.selected_frontier_index,
                sampled.selected_theta,
                observations.platform_context,
            )
        if not torch.equal(sampled.log_prob_total, recomputed.log_prob_total):
            raise CollectorError("sample and recomputed joint log probability differ")
        action_indices = _tensor_numpy(sampled.selected_frontier_index, np.int64)
        action_thetas = _tensor_numpy(sampled.selected_theta, np.float32)
        transition = environment.step(action_indices, action_thetas)
        observations, step_rewards, step_dones = _validated_transition(
            transition,
            env_count=env_count,
            device=target_device,
        )
        observations, boundary_dones = _prepare_policy_observations(
            environment,
            observations,
            env_count=env_count,
            device=target_device,
        )
        step_dones |= boundary_dones
        selected_indices.append(action_indices)
        selected_thetas.append(action_thetas)
        old_log_probs.append(_tensor_numpy(recomputed.log_prob_total, np.float32))
        old_values.append(_tensor_numpy(output.value, np.float32))
        rewards.append(step_rewards)
        dones.append(step_dones)

    with torch.no_grad():
        last_values = _tensor_numpy(policy(observations).value, np.float32)
    reward_matrix = np.stack(rewards, axis=0)
    done_matrix = np.stack(dones, axis=0)
    value_matrix = np.stack(old_values, axis=0)
    gae = compute_gae(
        rewards=reward_matrix,
        values=value_matrix,
        dones=done_matrix,
        last_values=last_values,
    )
    rollout = RolloutBatch(
        prior_channels=_flatten_time_env(observation_rows["prior_channels"]),
        coverage_summary=_flatten_time_env(observation_rows["coverage_summary"]),
        local_crop=_flatten_time_env(observation_rows["local_crop"]),
        frontier_features=_flatten_time_env(observation_rows["frontier_features"]),
        pose_features=_flatten_time_env(observation_rows["pose_features"]),
        candidate_mask=_flatten_time_env(observation_rows["candidate_mask"]),
        platform_context=_flatten_time_env(observation_rows["platform_context"]),
        selected_frontier_indices=_flatten_time_env(selected_indices),
        selected_thetas=_flatten_time_env(selected_thetas),
        old_log_prob_total=_flatten_time_env(old_log_probs),
        old_values=value_matrix.reshape(-1),
        raw_advantages=gae.raw_advantages.reshape(-1),
        advantages=gae.normalized_advantages.reshape(-1),
        returns=gae.returns.reshape(-1),
        platform_ids=np.argmax(
            _flatten_time_env(observation_rows["platform_context"]), axis=1
        ).astype(np.int64, copy=False),
        scale_bucket_ids=np.zeros(
            (config.horizon * env_count,), dtype=np.int64
        ),
    )
    return CollectedRollout(
        rollout=rollout,
        rewards=reward_matrix,
        dones=done_matrix,
    )


@dataclass(frozen=True, slots=True)
class _PendingMacroAction:
    prepared: PreparedWorkerBoundary
    action: PolicyAction
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    reward_stage: RewardStage
    reward_weights: RewardWeightsV4
    episode_transition_index: int
    transition_id: str
    started_s: float


def collect_committed_macro_rollout(
    *,
    pool: object,
    policy: CrossAttentionPolicy,
    journal: TransitionJournal,
    worker_strata: Sequence[WorkerStratum],
    actions_per_worker: int,
    update_id: int,
    policy_version: int,
    reward_stage: RewardStage | Mapping[str, RewardStage],
    deterministic: bool,
    device: torch.device | str,
    reward_weights: (
        RewardWeightsV4 | Mapping[str, RewardWeightsV4]
    ) = DEFAULT_REWARD_WEIGHTS,
) -> CommittedMacroRollout:
    """Let every worker independently commit its own fixed number of macros."""
    worker_count = getattr(pool, "worker_count", None)
    if type(worker_count) is not int or worker_count <= 0:
        raise CollectorError("committed collector worker count is invalid")
    if not isinstance(policy, CrossAttentionPolicy):
        raise CollectorError("policy must use CrossAttentionPolicy")
    if not isinstance(journal, TransitionJournal):
        raise CollectorError("committed collector requires TransitionJournal")
    if type(actions_per_worker) is not int or actions_per_worker <= 0:
        raise CollectorError("actions per worker must be a positive integer")
    if journal.worker_count != worker_count or (
        journal.slots_per_worker != actions_per_worker
    ):
        raise CollectorError("journal grid differs from committed rollout")
    if type(update_id) is not int or update_id < 0:
        raise CollectorError("committed rollout update ID is invalid")
    if type(policy_version) is not int or policy_version < 0:
        raise CollectorError("committed rollout policy version is invalid")
    if type(deterministic) is not bool:
        raise CollectorError("committed rollout deterministic flag is invalid")
    strata = _validated_worker_strata(worker_strata, worker_count=worker_count)
    stages = _reward_stages_by_platform(reward_stage, strata=strata)
    weights = _reward_weights_by_platform(reward_weights, strata=strata)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise CollectorError("requested CUDA device is unavailable")
    policy = policy.to(target_device)

    loaded = journal.load_update(update_id)
    if any(
        item.policy_version != policy_version
        for item in loaded.committed.values()
    ):
        raise CollectorError("partial journal contains another policy version")
    if loaded.state in ("SEALED", "APPLIED"):
        return _sealed_macro_rollout(
            loaded,
            worker_count=worker_count,
            actions_per_worker=actions_per_worker,
            policy_version=policy_version,
            pool=pool,
            policy=policy,
            strata=strata,
            device=target_device,
            task_outcomes=(),
            invalid_tasks=(),
            rejected_actions=(),
            macro_action_elapsed_s_by_worker=(0.0,) * worker_count,
            planner_call_count_by_worker=(0,) * worker_count,
            planner_elapsed_s_by_worker=(0.0,) * worker_count,
            policy_inference_elapsed_s=0.0,
        )

    recovered = journal.recover_worker_boundaries(update_id=update_id)
    next_slot = {
        worker: recovered[worker].next_slot if worker in recovered else 0
        for worker in range(worker_count)
    }
    last_episode_id = {
        worker: recovered[worker].episode_id
        for worker in recovered
    }
    last_episode_index = {
        worker: recovered[worker].episode_transition_index
        for worker in recovered
    }
    if any(slot > actions_per_worker for slot in next_slot.values()):
        raise CollectorError("partial journal cursor exceeds rollout length")

    prepare_workers = getattr(pool, "prepare_workers", None)
    submit_actions = getattr(pool, "submit_actions", None)
    await_completed = getattr(pool, "await_completed_workers", None)
    if not all(callable(item) for item in (prepare_workers, submit_actions, await_completed)):
        raise CollectorError("pool does not implement independent worker collection")
    terminal_worker_indices = getattr(pool, "terminal_worker_indices", ())
    if (
        not isinstance(terminal_worker_indices, tuple)
        or len(set(terminal_worker_indices)) != len(terminal_worker_indices)
        or any(
            type(worker) is not int or worker not in range(worker_count)
            for worker in terminal_worker_indices
        )
    ):
        raise CollectorError("pool terminal worker state is invalid")
    terminal_workers = set(terminal_worker_indices)

    active_workers = tuple(
        worker
        for worker in range(worker_count)
        if next_slot[worker] < actions_per_worker
    )
    for reset in (False, True):
        selected = tuple(
            worker
            for worker in active_workers
            if (
                worker in terminal_workers
                or (worker in recovered and recovered[worker].done)
            )
            is reset
        )
        if selected:
            prepare_workers(
                selected,
                policy_version=policy_version,
                reset=reset,
            )

    pending: dict[int, _PendingMacroAction] = {}
    task_outcomes: list[TaskOutcomeAudit] = []
    invalid_tasks: list[InvalidTaskAudit] = []
    rejected_actions: list[RejectedWorkerAction] = []
    macro_action_elapsed_s_by_worker = [0.0] * worker_count
    planner_call_count_by_worker = [0] * worker_count
    planner_elapsed_s_by_worker = [0.0] * worker_count
    policy_inference_elapsed_s = 0.0
    while any(slot < actions_per_worker for slot in next_slot.values()):
        events = await_completed()
        if (
            not isinstance(events, tuple)
            or not events
            or any(
                not isinstance(
                    event,
                    (
                        PreparedWorkerBoundary,
                        CompletedWorkerTransition,
                        RejectedWorkerAction,
                    ),
                )
                for event in events
            )
        ):
            raise CollectorError("pool returned invalid independent completion")
        actionable: list[PreparedWorkerBoundary] = []
        for event in events:
            worker = event.worker_index
            if worker not in strata or next_slot[worker] >= actions_per_worker:
                raise CollectorError("pool completed an inactive worker")
            if event.policy_version != policy_version:
                raise CollectorError("pool completed a stale policy version")
            if isinstance(event, PreparedWorkerBoundary):
                if worker in pending:
                    raise CollectorError("worker prepared while an action is pending")
                if not event.actionable:
                    if event.invalid_task_audit is not None:
                        if event.terminal_audit is not None:
                            raise CollectorError(
                                "invalid task cannot also be a task terminal"
                            )
                        invalid_tasks.append(event.invalid_task_audit)
                        prepare_workers(
                            (worker,),
                            policy_version=policy_version,
                            reset=True,
                        )
                        continue
                    if event.terminal_audit is None:
                        raise CollectorError("no-action task has no terminal audit")
                    identity = _single_identity(event.observation)
                    task_outcomes.append(
                        TaskOutcomeAudit(
                            worker_index=worker,
                            episode_id=identity.episode_id,
                            terminal_audit=event.terminal_audit,
                            post_worker_state=event.worker_state,
                        )
                    )
                    prepare_workers(
                        (worker,),
                        policy_version=policy_version,
                        reset=True,
                    )
                    continue
                if not bool(event.observation.candidate_mask.any()):
                    raise CollectorError(
                        "actionable worker has no selectable candidate"
                    )
                _validate_prepared_stratum(event, strata[worker])
                actionable.append(event)
                continue

            if isinstance(event, RejectedWorkerAction):
                current = pending.pop(worker, None)
                if current is None:
                    raise CollectorError(
                        "worker rejected without a pending action"
                    )
                post_identity = _single_identity(event.observation)
                pre_identity = _single_identity(current.prepared.observation)
                if post_identity.episode_id != pre_identity.episode_id:
                    raise CollectorError(
                        "rejected action changed episode identity"
                    )
                if (
                    event.candidate_diagnostics.physical_snapshot_id
                    != current.prepared.candidate_diagnostics.physical_snapshot_id
                ):
                    raise CollectorError(
                        "rejected action changed physical snapshot"
                    )
                _validate_observation_stratum(
                    event.observation, strata[worker]
                )
                rejected_actions.append(event)
                prepare_workers(
                    (worker,),
                    policy_version=policy_version,
                    reset=False,
                )
                continue

            current = pending.pop(worker, None)
            if current is None:
                raise CollectorError("worker completed without a pending action")
            elapsed_s = perf_counter() - current.started_s
            if not np.isfinite(elapsed_s) or elapsed_s < 0.0:
                raise CollectorError("macro action elapsed time is invalid")
            macro_action_elapsed_s_by_worker[worker] += float(elapsed_s)
            planner_events = event.execution_events
            planner_call_count_by_worker[worker] += (
                planner_events.planner_call_count
            )
            planner_elapsed_s_by_worker[worker] += float(
                planner_events.planner_elapsed_s
            )
            post_identity = _single_identity(event.observation)
            pre_identity = _single_identity(current.prepared.observation)
            if post_identity.episode_id != pre_identity.episode_id:
                raise CollectorError("macro action changed episode before commit")
            if event.policy_decisions_consumed != 1:
                raise CollectorError("macro action did not consume exactly one decision")
            try:
                expected_components = compute_reward_components(
                    event.reward_inputs,
                    current.reward_stage,
                    current.reward_weights,
                )
            except (TypeError, ValueError, RuntimeError) as error:
                raise CollectorError(
                    "worker returned invalid Reward V4 inputs"
                ) from error
            if event.reward_components != expected_components:
                raise CollectorError(
                    "worker reward differs from submitted Reward V4 weights"
                )
            payload = MacroTransitionPayload(
                update_id=update_id,
                worker_index=worker,
                slot_index=next_slot[worker],
                policy_version=policy_version,
                platform_type=strata[worker].platform_type,
                scale_bucket=strata[worker].scale_bucket,
                episode_id=pre_identity.episode_id,
                episode_transition_index=current.episode_transition_index,
                transition_id=current.transition_id,
                observation=_policy_batch_row_mapping(current.prepared.observation),
                action={
                    "candidate_index": torch.tensor(
                        current.action.frontier_index, dtype=torch.int64
                    ),
                    "theta": torch.tensor(
                        current.action.theta_rad, dtype=torch.float32
                    ),
                },
                old_log_prob=current.old_log_prob,
                old_value=current.old_value,
                reward_stage=current.reward_stage,
                reward_weights=current.reward_weights,
                reward_components=event.reward_components,
                reward_inputs=event.reward_inputs,
                done=event.done,
                terminal_class=event.terminal_class,
                pre_worker_audit=build_pre_worker_audit(
                    current.prepared.worker_state
                ),
                post_worker_state=event.worker_state,
            )
            committed = journal.commit(payload)
            if committed.transition_id != current.transition_id:
                raise CollectorError("journal committed another transition identity")
            next_slot[worker] += 1
            last_episode_id[worker] = pre_identity.episode_id
            last_episode_index[worker] = current.episode_transition_index
            if next_slot[worker] < actions_per_worker:
                prepare_workers(
                    (worker,),
                    policy_version=policy_version,
                    reset=event.done,
                )

        if actionable:
            policy_started_s = perf_counter()
            observations = _concatenate_policy_batches(
                tuple(item.observation for item in actionable),
                device=target_device,
            )
            action_identities: list[tuple[int, str]] = []
            for prepared in actionable:
                worker = prepared.worker_index
                identity = _single_identity(prepared.observation)
                episode_index = (
                    last_episode_index[worker] + 1
                    if last_episode_id.get(worker) == identity.episode_id
                    else 0
                )
                action_identities.append(
                    (
                        episode_index,
                        _macro_transition_id(
                            run_id=journal.run_id,
                            update_id=update_id,
                            worker_index=worker,
                            slot_index=next_slot[worker],
                            policy_version=policy_version,
                            identity=identity,
                        ),
                    )
                )
            with torch.no_grad():
                output = policy(observations)
                sampled = _sample_transition_keyed_actions(
                    output,
                    observations.candidate_mask,
                    observations.platform_context,
                    deterministic=deterministic,
                    transition_ids=tuple(
                        transition_id
                        for _, transition_id in action_identities
                    ),
                )
                recomputed = recompute_action_log_probs(
                    output,
                    observations.candidate_mask,
                    sampled.selected_frontier_index,
                    sampled.selected_theta,
                    observations.platform_context,
                )
            if not torch.equal(sampled.log_prob_total, recomputed.log_prob_total):
                raise CollectorError(
                    "sample and recomputed joint log probability differ"
                )
            actions_by_reward: dict[
                tuple[RewardStage, RewardWeightsV4], dict[int, PolicyAction]
            ] = {}
            for row, prepared in enumerate(actionable):
                worker = prepared.worker_index
                episode_index, transition_id = action_identities[row]
                action = PolicyAction(
                    frontier_index=int(
                        sampled.selected_frontier_index[row].detach().cpu().item()
                    ),
                    theta_rad=float(
                        sampled.selected_theta[row].detach().cpu().item()
                    ),
                )
                stage = stages[strata[worker].platform_type]
                platform_weights = weights[strata[worker].platform_type]
                pending[worker] = _PendingMacroAction(
                    prepared=prepared,
                    action=action,
                    old_log_prob=(
                        recomputed.log_prob_total[row]
                        .detach()
                        .cpu()
                        .to(torch.float32)
                        .reshape(())
                    ),
                    old_value=(
                        output.value[row]
                        .detach()
                        .cpu()
                        .to(torch.float32)
                        .reshape(())
                    ),
                    reward_stage=stage,
                    reward_weights=platform_weights,
                    episode_transition_index=episode_index,
                    transition_id=transition_id,
                    started_s=perf_counter(),
                )
                actions_by_reward.setdefault(
                    (stage, platform_weights), {}
                )[worker] = action
            policy_elapsed_s = perf_counter() - policy_started_s
            if not np.isfinite(policy_elapsed_s) or policy_elapsed_s < 0.0:
                raise CollectorError("policy inference elapsed time is invalid")
            policy_inference_elapsed_s += float(policy_elapsed_s)
            for (stage, platform_weights), actions in actions_by_reward.items():
                submit_actions(
                    actions,
                    policy_version=policy_version,
                    reward_stage=stage,
                    reward_weights=platform_weights,
                )

    if pending:
        raise CollectorError("committed rollout ended with pending actions")
    sealed = journal.seal_update(
        update_id=update_id,
        expected_slots=journal.expected_slots,
    )
    result = _sealed_macro_rollout(
        sealed,
        worker_count=worker_count,
        actions_per_worker=actions_per_worker,
        policy_version=policy_version,
        pool=pool,
        policy=policy,
        strata=strata,
        device=target_device,
        task_outcomes=tuple(task_outcomes),
        invalid_tasks=tuple(invalid_tasks),
        rejected_actions=tuple(rejected_actions),
        macro_action_elapsed_s_by_worker=tuple(
            macro_action_elapsed_s_by_worker
        ),
        planner_call_count_by_worker=tuple(planner_call_count_by_worker),
        planner_elapsed_s_by_worker=tuple(planner_elapsed_s_by_worker),
        policy_inference_elapsed_s=policy_inference_elapsed_s,
    )
    complete_update = getattr(pool, "complete_policy_update", None)
    if callable(complete_update):
        complete_update(policy_version=policy_version)
    return result


def _validated_worker_strata(
    value: Sequence[WorkerStratum], *, worker_count: int
) -> dict[int, WorkerStratum]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CollectorError("worker strata are invalid")
    strata: dict[int, WorkerStratum] = {}
    for item in value:
        if not isinstance(item, WorkerStratum) or item.worker_index in strata:
            raise CollectorError("worker strata are invalid")
        strata[item.worker_index] = item
    if set(strata) != set(range(worker_count)):
        raise CollectorError("worker strata are incomplete")
    return strata


def _reward_stages_by_platform(
    value: RewardStage | Mapping[str, RewardStage],
    *,
    strata: Mapping[int, WorkerStratum],
) -> dict[str, RewardStage]:
    platforms = {item.platform_type for item in strata.values()}
    if isinstance(value, RewardStage):
        return {platform: value for platform in platforms}
    if (
        not isinstance(value, Mapping)
        or set(value) != platforms
        or any(not isinstance(stage, RewardStage) for stage in value.values())
    ):
        raise CollectorError("platform reward stages are invalid")
    return dict(value)


def _reward_weights_by_platform(
    value: RewardWeightsV4 | Mapping[str, RewardWeightsV4],
    *,
    strata: Mapping[int, WorkerStratum],
) -> dict[str, RewardWeightsV4]:
    platforms = {item.platform_type for item in strata.values()}
    if isinstance(value, RewardWeightsV4):
        return {platform: value for platform in platforms}
    if (
        not isinstance(value, Mapping)
        or set(value) != platforms
        or any(
            not isinstance(weights, RewardWeightsV4)
            for weights in value.values()
        )
    ):
        raise CollectorError("platform reward weights are invalid")
    return dict(value)


def _single_identity(batch: PolicyBatch):
    identities = batch.observation_identities
    if identities is None or len(identities) != 1:
        raise CollectorError("worker observation identity is missing")
    return identities[0]


def _validate_prepared_stratum(
    prepared: PreparedWorkerBoundary, stratum: WorkerStratum
) -> None:
    _validate_observation_stratum(prepared.observation, stratum)


def _validate_observation_stratum(
    observation: PolicyBatch, stratum: WorkerStratum
) -> None:
    platform_index = {"WHEELED": 0, "LEGGED": 1, "HOPPER": 2}[
        stratum.platform_type
    ]
    expected = torch.zeros((1, 3), dtype=torch.float32)
    expected[0, platform_index] = 1.0
    if not torch.equal(observation.platform_context.cpu(), expected):
        raise CollectorError("worker observation platform stratum differs")


def _concatenate_policy_batches(
    batches: tuple[PolicyBatch, ...], *, device: torch.device
) -> PolicyBatch:
    if not batches:
        raise CollectorError("cannot concatenate an empty worker batch")
    combined = PolicyBatch(
        **{
            name: torch.cat(
                [getattr(batch, name).to(device) for batch in batches], dim=0
            )
            for name in batches[0].input_names
        },
        observation_identities=tuple(
            _single_identity(batch) for batch in batches
        ),
    )
    try:
        validate_policy_batch(combined)
    except ValueError as error:
        raise CollectorError("prepared worker observation batch is invalid") from error
    return combined


def _policy_batch_row_mapping(batch: PolicyBatch) -> dict[str, torch.Tensor]:
    if batch.prior_channels.shape[0] != 1:
        raise CollectorError("journal observation must contain one worker row")
    return {
        name: getattr(batch, name)[0].detach().cpu().clone()
        for name in batch.input_names
    }


def _macro_transition_id(
    *,
    run_id: str,
    update_id: int,
    worker_index: int,
    slot_index: int,
    policy_version: int,
    identity: object,
) -> str:
    fields = (
        run_id,
        str(update_id),
        str(worker_index),
        str(slot_index),
        str(policy_version),
        str(getattr(identity, "episode_id", "")),
        str(getattr(identity, "mission_revision", "")),
        str(getattr(identity, "map_snapshot_id", "")),
        str(getattr(identity, "robot_state_id", "")),
        str(getattr(identity, "state_time_ns", "")),
        str(getattr(identity, "candidate_set_id", "")),
    )
    return hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()


def _sample_transition_keyed_actions(
    output: PolicyOutput,
    candidate_mask: torch.Tensor,
    platform_context: torch.Tensor,
    *,
    deterministic: bool,
    transition_ids: tuple[str, ...],
) -> ActionSample:
    """Sample each stochastic macro from its immutable transition identity."""
    batch_size = candidate_mask.shape[0]
    if (
        not isinstance(output, PolicyOutput)
        or not isinstance(transition_ids, tuple)
        or len(transition_ids) != batch_size
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in transition_ids
        )
    ):
        raise CollectorError("transition-keyed sampling identity is invalid")
    if deterministic:
        return sample_action(
            output,
            candidate_mask,
            platform_context,
            deterministic=True,
        )
    device = candidate_mask.device
    cuda_devices = (
        []
        if device.type != "cuda"
        else [
            torch.cuda.current_device()
            if device.index is None
            else device.index
        ]
    )
    rows: list[ActionSample] = []
    for row, transition_id in enumerate(transition_ids):
        digest = hashlib.sha256(
            (
                "lunar-reward-v4-action-sample/v1\x1f"
                + transition_id
            ).encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
        row_output = PolicyOutput(
            frontier_logits=output.frontier_logits[row : row + 1],
            theta_mu=output.theta_mu[row : row + 1],
            theta_kappa=output.theta_kappa[row : row + 1],
            value=output.value[row : row + 1],
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            rows.append(
                sample_action(
                    row_output,
                    candidate_mask[row : row + 1],
                    platform_context[row : row + 1],
                    deterministic=False,
                )
            )
    return ActionSample(
        selected_frontier_index=torch.cat(
            [item.selected_frontier_index for item in rows]
        ),
        selected_theta=torch.cat([item.selected_theta for item in rows]),
        log_prob_frontier=torch.cat(
            [item.log_prob_frontier for item in rows]
        ),
        log_prob_theta=torch.cat([item.log_prob_theta for item in rows]),
        log_prob_total=torch.cat([item.log_prob_total for item in rows]),
        frontier_entropy=torch.cat(
            [item.frontier_entropy for item in rows]
        ),
        theta_entropy=torch.cat([item.theta_entropy for item in rows]),
        theta_active=torch.cat([item.theta_active for item in rows]),
        value=torch.cat([item.value for item in rows]),
    )


def _sealed_macro_rollout(
    loaded: object,
    *,
    worker_count: int,
    actions_per_worker: int,
    policy_version: int,
    pool: object,
    policy: CrossAttentionPolicy,
    strata: Mapping[int, WorkerStratum],
    device: torch.device,
    task_outcomes: tuple[TaskOutcomeAudit, ...],
    invalid_tasks: tuple[InvalidTaskAudit, ...],
    rejected_actions: tuple[RejectedWorkerAction, ...],
    macro_action_elapsed_s_by_worker: tuple[float, ...],
    planner_call_count_by_worker: tuple[int, ...],
    planner_elapsed_s_by_worker: tuple[float, ...],
    policy_inference_elapsed_s: float,
) -> CommittedMacroRollout:
    if (
        getattr(loaded, "state", None) not in ("SEALED", "APPLIED")
        or getattr(loaded, "policy_version", None) != policy_version
        or not isinstance(getattr(loaded, "journal_sha256", None), str)
    ):
        raise CollectorError("committed rollout journal is not sealed")
    committed_map = loaded.committed
    expected = {
        (worker, slot)
        for worker in range(worker_count)
        for slot in range(actions_per_worker)
    }
    if set(committed_map) != expected:
        raise CollectorError("sealed rollout has an incomplete worker grid")
    ordered = tuple(
        committed_map[(worker, slot)]
        for slot in range(actions_per_worker)
        for worker in range(worker_count)
    )
    rollout = _rollout_from_committed_grid(
        committed_map,
        pool=pool,
        policy=policy,
        strata=strata,
        worker_count=worker_count,
        actions_per_worker=actions_per_worker,
        device=device,
    )
    return CommittedMacroRollout(
        update_id=loaded.update_id,
        policy_version=policy_version,
        actions_per_worker=actions_per_worker,
        committed=ordered,
        rollout=rollout,
        task_outcomes=task_outcomes,
        journal_sha256=loaded.journal_sha256,
        invalid_tasks=invalid_tasks,
        rejected_actions=rejected_actions,
        macro_action_elapsed_s_by_worker=macro_action_elapsed_s_by_worker,
        planner_call_count_by_worker=planner_call_count_by_worker,
        planner_elapsed_s_by_worker=planner_elapsed_s_by_worker,
        policy_inference_elapsed_s=policy_inference_elapsed_s,
    )


def _rollout_from_committed_grid(
    committed: Mapping[tuple[int, int], CommittedTransition],
    *,
    pool: object,
    policy: CrossAttentionPolicy,
    strata: Mapping[int, WorkerStratum],
    worker_count: int,
    actions_per_worker: int,
    device: torch.device,
) -> RolloutBatch:
    payload_grid = tuple(
        tuple(
            committed[(worker, slot)].payload
            for worker in range(worker_count)
        )
        for slot in range(actions_per_worker)
    )
    rewards = np.asarray(
        [
            [payload.reward_components.total for payload in row]
            for row in payload_grid
        ],
        dtype=np.float32,
    )
    values = np.asarray(
        [
            [float(payload.old_value.item()) for payload in row]
            for row in payload_grid
        ],
        dtype=np.float32,
    )
    dones = np.asarray(
        [[payload.done for payload in row] for row in payload_grid],
        dtype=np.bool_,
    )
    last_values = _bootstrap_worker_values(
        pool=pool,
        policy=policy,
        last_payloads=payload_grid[-1],
        strata=strata,
        device=device,
    )
    ordered_strata = tuple(strata[worker] for worker in range(worker_count))
    gae = compute_stratified_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_values=last_values,
        worker_strata=ordered_strata,
    )
    flattened_payloads = tuple(
        payload for row in payload_grid for payload in row
    )
    observation_arrays = {
        name: _stack_payload_tensors(
            tuple(payload.observation[name] for payload in flattened_payloads)
        )
        for name in ObservationContractV4.input_names
    }
    platform_ids = np.asarray(
        [
            PLATFORM_ID_BY_TYPE[strata[worker].platform_type]
            for _slot in range(actions_per_worker)
            for worker in range(worker_count)
        ],
        dtype=np.int64,
    )
    scale_bucket_ids = np.asarray(
        [
            SCALE_BUCKET_ID_BY_BUCKET[strata[worker].scale_bucket]
            for _slot in range(actions_per_worker)
            for worker in range(worker_count)
        ],
        dtype=np.int64,
    )
    return RolloutBatch(
        **observation_arrays,
        selected_frontier_indices=np.asarray(
            [
                int(payload.action["candidate_index"].item())
                for payload in flattened_payloads
            ],
            dtype=np.int64,
        ),
        selected_thetas=np.asarray(
            [
                float(payload.action["theta"].item())
                for payload in flattened_payloads
            ],
            dtype=np.float32,
        ),
        old_log_prob_total=np.asarray(
            [float(payload.old_log_prob.item()) for payload in flattened_payloads],
            dtype=np.float32,
        ),
        old_values=values.reshape(-1),
        raw_advantages=gae.raw_advantages.reshape(-1),
        advantages=gae.normalized_advantages.reshape(-1),
        returns=gae.returns.reshape(-1),
        platform_ids=platform_ids,
        scale_bucket_ids=scale_bucket_ids,
    )


def _bootstrap_worker_values(
    *,
    pool: object,
    policy: CrossAttentionPolicy,
    last_payloads: tuple[MacroTransitionPayload, ...],
    strata: Mapping[int, WorkerStratum],
    device: torch.device,
) -> np.ndarray:
    worker_count = len(last_payloads)
    result = np.zeros((worker_count,), dtype=np.float32)
    live_workers = tuple(
        worker
        for worker, payload in enumerate(last_payloads)
        if not payload.done
    )
    if not live_workers:
        return result
    current_observations = getattr(pool, "current_worker_observations", None)
    if not callable(current_observations):
        raise CollectorError("pool cannot publish bootstrap observations")
    batches = current_observations(live_workers)
    if (
        not isinstance(batches, tuple)
        or len(batches) != len(live_workers)
        or any(not isinstance(batch, PolicyBatch) for batch in batches)
    ):
        raise CollectorError("pool bootstrap observations are invalid")
    for worker, batch in zip(live_workers, batches):
        _validate_observation_stratum(batch, strata[worker])
        identity = _single_identity(batch)
        if identity.episode_id != last_payloads[worker].episode_id:
            raise CollectorError("bootstrap observation episode differs")
        if not bool(batch.candidate_mask.any()):
            raise CollectorError("live bootstrap observation has no candidate")
    combined = _concatenate_policy_batches(batches, device=device)
    with torch.no_grad():
        values = policy(combined).value.detach().cpu().to(torch.float32).numpy()
    if values.shape != (len(live_workers),) or not np.isfinite(values).all():
        raise CollectorError("bootstrap policy values are invalid")
    for row, worker in enumerate(live_workers):
        result[worker] = values[row]
    return result


def _stack_payload_tensors(values: tuple[object, ...]) -> np.ndarray:
    if not values or any(not isinstance(value, torch.Tensor) for value in values):
        raise CollectorError("journal observation tensor is invalid")
    try:
        return torch.stack(
            [value.detach().cpu() for value in values], dim=0
        ).numpy().copy()
    except RuntimeError as error:
        raise CollectorError("journal observation tensor shapes differ") from error


def _validated_observations(
    value: object,
    *,
    env_count: int,
    device: torch.device,
    allow_no_candidates: bool = False,
) -> PolicyBatch:
    if not isinstance(value, PolicyBatch):
        raise CollectorError("environment must return a seven-input observation batch")
    moved = PolicyBatch(
        prior_channels=value.prior_channels.to(device),
        coverage_summary=value.coverage_summary.to(device),
        local_crop=value.local_crop.to(device),
        frontier_features=value.frontier_features.to(device),
        pose_features=value.pose_features.to(device),
        candidate_mask=value.candidate_mask.to(device),
        platform_context=value.platform_context.to(device),
        observation_identities=value.observation_identities,
    )
    try:
        validate_policy_batch(moved)
    except ValueError as error:
        raise CollectorError(str(error)) from error
    if moved.prior_channels.shape[0] != env_count:
        raise CollectorError("observation batch size does not match environment count")
    if (
        not allow_no_candidates
        and not bool(moved.candidate_mask.any(dim=1).all())
    ):
        raise CollectorError("all-false candidate rows must bypass rollout collection")
    return moved


def _prepare_policy_observations(
    environment: VectorEnv,
    observations: PolicyBatch,
    *,
    env_count: int,
    device: torch.device,
) -> tuple[PolicyBatch, np.ndarray]:
    preparer = getattr(environment, "prepare_decision_boundaries", None)
    if callable(preparer):
        prepared, _, boundary_dones = _validated_transition(
            preparer(),
            env_count=env_count,
            device=device,
        )
        if bool(_unavailable_decision_rows(prepared).any()):
            raise CollectorError(
                "decision-boundary preparation must return actionable observations"
            )
        return prepared, boundary_dones
    unavailable = _unavailable_decision_rows(observations)
    if bool(unavailable.any()):
        raise CollectorError(
            "all-false candidate rows must bypass rollout collection"
        )
    return observations, np.zeros((env_count,), dtype=np.bool_)


def _unavailable_decision_rows(observations: PolicyBatch) -> torch.Tensor:
    return ~observations.candidate_mask.any(dim=1)


def _validated_transition(
    value: object,
    *,
    env_count: int,
    device: torch.device,
) -> tuple[PolicyBatch, np.ndarray, np.ndarray]:
    if not isinstance(value, EnvStep):
        raise CollectorError("environment step must return EnvStep")
    if (
        not isinstance(value.rewards, np.ndarray)
        or value.rewards.dtype != np.float32
    ):
        raise CollectorError("rewards must be float32")
    if value.rewards.shape != (env_count,):
        raise CollectorError("rewards must have one value per environment")
    if not np.isfinite(value.rewards).all():
        raise CollectorError("rewards must be finite")
    if not isinstance(value.dones, np.ndarray) or value.dones.dtype != np.bool_:
        raise CollectorError("dones must be boolean")
    if value.dones.shape != (env_count,):
        raise CollectorError("dones must have one value per environment")
    observations = _validated_observations(
        value.observations,
        env_count=env_count,
        device=device,
        allow_no_candidates=True,
    )
    return observations, value.rewards.copy(), value.dones.copy()


def _batch_numpy(batch: PolicyBatch) -> dict[str, np.ndarray]:
    return {
        name: _tensor_numpy(getattr(batch, name), None)
        for name in batch.input_names
    }


def _tensor_numpy(
    value: torch.Tensor, dtype: type[np.generic] | None
) -> np.ndarray:
    array = value.detach().cpu().numpy().copy()
    return array.astype(dtype, copy=False) if dtype is not None else array


def _flatten_time_env(values: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(values, axis=0)
    return stacked.reshape((-1, *stacked.shape[2:]))


__all__ = [
    "CollectedRollout",
    "CommittedMacroRollout",
    "CollectorConfig",
    "CollectorError",
    "EnvStep",
    "TaskOutcomeAudit",
    "VectorEnv",
    "collect_committed_macro_rollout",
    "collect_rollout",
]
