"""Training tensors that consume the shared observation contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lunar_model_contract.observation import ObservationContractV1, validate_platform_context

from .observation_core import (
    COVERAGE_SUMMARY_CHANNELS,
    FRONTIER_FEATURE_FIELDS,
    GLOBAL_PRIOR_CHANNELS,
    LOCAL_CROP_CHANNELS,
    POSE_FEATURE_FIELDS,
)


@dataclass
class PolicyBatch:
    """Seven tensors in exactly ``ObservationContractV1.input_names`` order."""

    prior_channels: torch.Tensor
    coverage_summary: torch.Tensor
    local_crop: torch.Tensor
    frontier_features: torch.Tensor
    pose_features: torch.Tensor
    candidate_mask: torch.Tensor
    platform_context: torch.Tensor

    @property
    def input_names(self) -> tuple[str, ...]:
        return ObservationContractV1.input_names


def validate_policy_batch(batch: PolicyBatch) -> None:
    """Reject tensors that cannot be passed to the shared PPO architecture."""
    tensors = (
        batch.prior_channels,
        batch.coverage_summary,
        batch.local_crop,
        batch.frontier_features,
        batch.pose_features,
        batch.platform_context,
    )
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 for value in tensors):
        raise ValueError("policy float inputs must be float32 tensors")
    if not isinstance(batch.candidate_mask, torch.Tensor) or batch.candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be bool")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("policy float inputs must be finite")
    batch_size = batch.prior_channels.shape[0]
    if (
        batch.prior_channels.ndim != 4
        or batch.prior_channels.shape[1] != len(GLOBAL_PRIOR_CHANNELS)
    ):
        raise ValueError("prior_channels must be float32 [B,7,H,W]")
    if (
        batch.coverage_summary.shape[:2]
        != (batch_size, len(COVERAGE_SUMMARY_CHANNELS))
        or batch.coverage_summary.ndim != 4
    ):
        raise ValueError("coverage_summary must be float32 [B,8,H,W]")
    if (
        batch.local_crop.shape[:2] != (batch_size, len(LOCAL_CROP_CHANNELS))
        or batch.local_crop.ndim != 4
    ):
        raise ValueError("local_crop must be float32 [B,8,H,W]")
    if (
        batch.frontier_features.ndim != 3
        or batch.frontier_features.shape[:2] != batch.candidate_mask.shape
        or batch.frontier_features.shape[2] != len(FRONTIER_FEATURE_FIELDS)
    ):
        raise ValueError("frontier_features must be float32 [B,M,22]")
    if batch.pose_features.shape != (batch_size, len(POSE_FEATURE_FIELDS)):
        raise ValueError("pose_features must be float32 [B,6]")
    if batch.platform_context.shape != (batch_size, 3):
        raise ValueError("platform_context must be float32 [B,3]")
    if not bool(batch.candidate_mask.any(dim=1).all()):
        raise ValueError("candidate_mask must contain a valid candidate per row")
    devices = {value.device for value in (*tensors, batch.candidate_mask)}
    if len(devices) != 1:
        raise ValueError("policy tensors must share a device")
    validate_platform_context(batch.platform_context.detach().cpu().numpy())
