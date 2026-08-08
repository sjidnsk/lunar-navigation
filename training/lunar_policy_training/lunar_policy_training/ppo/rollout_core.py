"""Rollout generalized advantage estimation mathematical core."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Final
import numpy as np
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
        for value in (self.raw_advantages, self.normalized_advantages, self.returns):
            value.setflags(write=False)

def compute_gae(*, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_values: np.ndarray) -> GAEResult:
    """计算固定 gamma/lambda 的 `[T,E]` FP32 GAE。"""
    arrays = (rewards, values, last_values)
    if any((not isinstance(value, np.ndarray) or value.dtype != np.float32 for value in arrays)):
        raise RolloutContractError('GAE rewards, values, and last_values must be FP32')
    if not isinstance(dones, np.ndarray) or dones.dtype != np.bool_:
        raise RolloutContractError('GAE dones must use boolean dtype')
    if rewards.ndim != 2 or rewards.shape[0] == 0 or rewards.shape[1] == 0 or (values.shape != rewards.shape) or (dones.shape != rewards.shape) or (last_values.shape != (rewards.shape[1],)):
        raise RolloutContractError('GAE array shape mismatch')
    if any((not np.isfinite(value).all() for value in arrays)):
        raise RolloutContractError('GAE inputs must be finite')
    (time_steps, env_count) = rewards.shape
    raw = np.zeros((time_steps, env_count), dtype=np.float32)
    next_advantage = np.zeros((env_count,), dtype=np.float32)
    for time_index in range(time_steps - 1, -1, -1):
        next_value = last_values if time_index == time_steps - 1 else values[time_index + 1]
        not_done = np.logical_not(dones[time_index]).astype(np.float32)
        delta = np.float32(rewards[time_index] + GAMMA * next_value * not_done - values[time_index])
        next_advantage = np.float32(delta + GAMMA * GAE_LAMBDA * not_done * next_advantage)
        raw[time_index] = next_advantage
    returns = np.float32(raw + values)
    mean = np.mean(raw, dtype=np.float32)
    centered = np.float32(raw - mean)
    variance = np.mean(np.float32(centered * centered), dtype=np.float32)
    standard_deviation = np.sqrt(variance, dtype=np.float32)
    if not math.isfinite(float(standard_deviation)) or standard_deviation <= np.float32(1e-08):
        raise RolloutContractError('GAE normalization is degenerate')
    normalized = np.float32(centered / standard_deviation)
    if any((not np.isfinite(value).all() for value in (raw, normalized, returns))):
        raise RolloutContractError('GAE outputs must be finite')
    return GAEResult(raw_advantages=raw, normalized_advantages=normalized, returns=returns)
