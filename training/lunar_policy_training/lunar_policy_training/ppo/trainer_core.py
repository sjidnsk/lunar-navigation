"""Strict FP32 PPO update mathematical core."""
from __future__ import annotations
import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Mapping
import numpy as np
import torch
from torch import nn

class PPOTrainingError(RuntimeError):
    """PPO 数学、batch 或 optimizer 状态不满足冻结合同。"""

@dataclass(frozen=True, slots=True)
class PPOLossTerms:
    ratio: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    frontier_entropy: torch.Tensor
    total_loss: torch.Tensor
    approx_kl: torch.Tensor

def compute_ppo_loss_terms(*, new_log_prob_total: torch.Tensor, old_log_prob_total: torch.Tensor, normalized_advantage: torch.Tensor, new_value: torch.Tensor, old_value: torch.Tensor, returns: torch.Tensor, frontier_entropy: torch.Tensor) -> PPOLossTerms:
    """计算 joint-action clipped PPO 与 clipped value 的 batch mean。"""
    tensors = (new_log_prob_total, old_log_prob_total, normalized_advantage, new_value, old_value, returns, frontier_entropy)
    if any((not isinstance(value, torch.Tensor) or value.dtype != torch.float32 for value in tensors)):
        raise PPOTrainingError('PPO loss inputs must be FP32 tensors')
    shape = new_log_prob_total.shape
    if len(shape) != 1 or shape[0] == 0 or any((value.shape != shape for value in tensors)):
        raise PPOTrainingError('PPO loss input shapes must match [N]')
    devices = {value.device for value in tensors}
    if len(devices) != 1:
        raise PPOTrainingError('PPO loss inputs must share one device')
    device = new_log_prob_total.device
    try:
        autocast_enabled = torch.is_autocast_enabled(device.type)
    except TypeError:
        autocast_enabled = torch.is_autocast_enabled()
    if autocast_enabled:
        raise PPOTrainingError('AMP/autocast is forbidden for PPO')
    if any((not bool(torch.isfinite(value).all()) for value in tensors)):
        raise PPOTrainingError('PPO loss inputs must be finite')
    ratio = torch.exp(new_log_prob_total - old_log_prob_total)
    unclipped = ratio * normalized_advantage
    clipped = torch.clamp(ratio, 0.8, 1.2) * normalized_advantage
    policy_loss = -torch.minimum(unclipped, clipped).mean(dtype=torch.float32)
    value_clipped = old_value + torch.clamp(new_value - old_value, -0.2, 0.2)
    value_loss = 0.5 * torch.maximum((new_value - returns).square(), (value_clipped - returns).square()).mean(dtype=torch.float32)
    entropy_mean = frontier_entropy.mean(dtype=torch.float32)
    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_mean
    approx_kl = (old_log_prob_total - new_log_prob_total).mean(dtype=torch.float32)
    outputs = (ratio, policy_loss, value_loss, entropy_mean, total_loss, approx_kl)
    if any((value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in outputs)):
        raise PPOTrainingError('PPO loss outputs must be finite FP32')
    return PPOLossTerms(ratio=ratio, policy_loss=policy_loss, value_loss=value_loss, frontier_entropy=entropy_mean, total_loss=total_loss, approx_kl=approx_kl)

def policy_state_sha256(policy: nn.Module) -> str:
    """对 state_dict 的名称、dtype、shape 和原始 bytes 做稳定 SHA-256。"""
    if not isinstance(policy, nn.Module):
        raise PPOTrainingError('policy hash requires a torch module')
    digest = hashlib.sha256()
    state = policy.state_dict()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise PPOTrainingError('policy state contains a non-tensor value')
        contiguous = tensor.detach().contiguous().cpu()
        raw = contiguous.numpy().tobytes(order='C')
        for value in (name.encode('utf-8'), str(contiguous.dtype).encode('ascii'), np.asarray(tuple(contiguous.shape), dtype='<i8').tobytes(), raw):
            digest.update(struct.pack('<Q', len(value)))
            digest.update(value)
    return digest.hexdigest()

def physical_microbatch_slices(sample_count: int) -> tuple[tuple[int, int], ...]:
    if type(sample_count) is not int or sample_count <= 0:
        raise PPOTrainingError('physical microbatch sample count must be positive')
    return tuple(((start, min(start + 32, sample_count)) for start in range(0, sample_count, 32)))

def _gradient_norm(parameters) -> float:
    total = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            return math.inf
        total += float(gradient.double().square().sum().cpu())
    return math.sqrt(total) if found else 0.0

def _parameter_change_l2(policy: nn.Module, before: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for (name, parameter) in policy.named_parameters():
        if name not in before:
            raise PPOTrainingError('policy parameter set changed during update')
        difference = parameter.detach().cpu().double() - before[name].double()
        total += float(difference.square().sum())
    if set(before) != {name for (name, _) in policy.named_parameters()}:
        raise PPOTrainingError('policy parameter set changed during update')
    return math.sqrt(total)
