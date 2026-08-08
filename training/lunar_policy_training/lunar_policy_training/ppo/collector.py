"""In-process rollout collection for the shared seven-input policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from lunar_model_contract import ObservationContractV2

from ..policy.cross_attention import (
    CrossAttentionPolicy,
    recompute_action_log_probs,
    sample_action,
)
from ..policy.observation import PolicyBatch, validate_policy_batch
from .rollout import RolloutBatch, compute_gae


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
        name: [] for name in ObservationContractV2.input_names
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
                deterministic=config.deterministic,
            )
            recomputed = recompute_action_log_probs(
                output,
                observations.candidate_mask,
                sampled.selected_frontier_index,
                sampled.selected_theta,
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
        advantages=gae.normalized_advantages.reshape(-1),
        returns=gae.returns.reshape(-1),
    )
    return CollectedRollout(
        rollout=rollout,
        rewards=reward_matrix,
        dones=done_matrix,
    )


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
        if bool((~observations.candidate_mask.any(dim=1)).any()):
            raise CollectorError(
                "all-false candidate rows must bypass rollout collection"
            )
        raise CollectorError(
            "exhausted-budget rows must bypass rollout collection"
        )
    return observations, np.zeros((env_count,), dtype=np.bool_)


def _unavailable_decision_rows(observations: PolicyBatch) -> torch.Tensor:
    has_candidate = observations.candidate_mask.any(dim=1)
    has_budget = observations.pose_features[:, 5] > 0.0
    return ~(has_candidate & has_budget)


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
    "CollectorConfig",
    "CollectorError",
    "EnvStep",
    "VectorEnv",
    "collect_rollout",
]
