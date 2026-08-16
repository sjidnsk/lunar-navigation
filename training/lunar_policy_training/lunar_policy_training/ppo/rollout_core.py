"""Rollout generalized advantage estimation mathematical core."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Final

import numpy as np

from ..environment.task_area import WorkerStratum
from ..reward_contract import DEFAULT_REWARD_CONFIG


GAMMA: Final = np.float32(0.995)
GAE_LAMBDA: Final = np.float32(0.95)


class RolloutContractError(ValueError):
    """Rollout inputs violate the generalized advantage estimation contract."""


@dataclass(frozen=True, slots=True)
class GAEResult:
    raw_advantages: np.ndarray
    normalized_advantages: np.ndarray
    returns: np.ndarray

    def __post_init__(self) -> None:
        for value in (
            self.raw_advantages,
            self.normalized_advantages,
            self.returns,
        ):
            value.setflags(write=False)


def compute_gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
) -> GAEResult:
    """Compute the legacy globally normalized fixed gamma/lambda GAE."""
    _validate_gae_inputs(rewards, values, dones, last_values)
    raw = _compute_raw_gae(rewards, values, dones, last_values)
    returns = np.float32(raw + values)
    normalized = _normalize_advantages(raw, allow_constant=False)
    _validate_gae_outputs(raw, normalized, returns)
    return GAEResult(
        raw_advantages=raw,
        normalized_advantages=normalized,
        returns=returns,
    )


def compute_stratified_gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    worker_strata: Sequence[WorkerStratum],
) -> GAEResult:
    """Compute worker-time GAE, then normalize within platform/scale strata."""
    _validate_gae_inputs(rewards, values, dones, last_values)
    strata = _validate_worker_strata(
        worker_strata,
        worker_count=rewards.shape[1],
    )
    raw = _compute_raw_gae(rewards, values, dones, last_values)
    returns = np.float32(raw + values)
    normalized = np.zeros_like(raw, dtype=np.float32)
    ordered_keys = tuple(
        dict.fromkeys(
            (stratum.platform_type, stratum.scale_bucket)
            for stratum in strata
        )
    )
    for key in ordered_keys:
        columns = np.asarray(
            [
                index
                for index, stratum in enumerate(strata)
                if (stratum.platform_type, stratum.scale_bucket) == key
            ],
            dtype=np.int64,
        )
        normalized[:, columns] = _normalize_advantages(
            raw[:, columns],
            allow_constant=True,
        )
    _validate_gae_outputs(raw, normalized, returns)
    return GAEResult(
        raw_advantages=raw,
        normalized_advantages=normalized,
        returns=returns,
    )


def _validate_gae_inputs(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
) -> None:
    arrays = (rewards, values, last_values)
    if any(
        not isinstance(value, np.ndarray) or value.dtype != np.float32
        for value in arrays
    ):
        raise RolloutContractError(
            "GAE rewards, values, and last_values must be FP32"
        )
    if not isinstance(dones, np.ndarray) or dones.dtype != np.bool_:
        raise RolloutContractError("GAE dones must use boolean dtype")
    if (
        rewards.ndim != 2
        or rewards.shape[0] == 0
        or rewards.shape[1] == 0
        or values.shape != rewards.shape
        or dones.shape != rewards.shape
        or last_values.shape != (rewards.shape[1],)
    ):
        raise RolloutContractError("GAE array shape mismatch")
    if any(not np.isfinite(value).all() for value in arrays):
        raise RolloutContractError("GAE inputs must be finite")


def _validate_worker_strata(
    value: Sequence[WorkerStratum], *, worker_count: int
) -> tuple[WorkerStratum, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != worker_count
    ):
        raise RolloutContractError("GAE worker strata are invalid")
    strata = tuple(value)
    if any(
        not isinstance(stratum, WorkerStratum)
        or stratum.worker_index != worker_index
        for worker_index, stratum in enumerate(strata)
    ):
        raise RolloutContractError("GAE worker strata do not match columns")
    return strata


def _compute_raw_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
) -> np.ndarray:
    time_steps, env_count = rewards.shape
    raw = np.zeros((time_steps, env_count), dtype=np.float32)
    next_advantage = np.zeros((env_count,), dtype=np.float32)
    for time_index in range(time_steps - 1, -1, -1):
        next_value = (
            last_values
            if time_index == time_steps - 1
            else values[time_index + 1]
        )
        not_done = np.logical_not(dones[time_index]).astype(np.float32)
        delta = np.float32(
            rewards[time_index]
            + GAMMA * next_value * not_done
            - values[time_index]
        )
        next_advantage = np.float32(
            delta + GAMMA * GAE_LAMBDA * not_done * next_advantage
        )
        raw[time_index] = next_advantage
    return raw


def _normalize_advantages(
    values: np.ndarray, *, allow_constant: bool
) -> np.ndarray:
    mean = np.mean(values, dtype=np.float32)
    centered = np.float32(values - mean)
    variance = np.mean(np.float32(centered * centered), dtype=np.float32)
    standard_deviation = np.sqrt(variance, dtype=np.float32)
    floor = np.float32(DEFAULT_REWARD_CONFIG.advantage_std_floor)
    if not math.isfinite(float(standard_deviation)):
        raise RolloutContractError("GAE normalization is non-finite")
    if standard_deviation <= floor:
        if allow_constant:
            return np.zeros_like(values, dtype=np.float32)
        raise RolloutContractError("GAE normalization is degenerate")
    return np.float32(centered / standard_deviation)


def _validate_gae_outputs(
    raw: np.ndarray, normalized: np.ndarray, returns: np.ndarray
) -> None:
    if any(
        not np.isfinite(value).all()
        for value in (raw, normalized, returns)
    ):
        raise RolloutContractError("GAE outputs must be finite")


__all__ = [
    "GAEResult",
    "GAE_LAMBDA",
    "GAMMA",
    "RolloutContractError",
    "compute_gae",
    "compute_stratified_gae",
]
