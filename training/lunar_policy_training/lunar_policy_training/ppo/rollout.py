"""Ephemeral seven-input rollout batch and frozen GAE math."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..policy.observation import PolicyBatch, validate_policy_batch
from .rollout_core import (
    GAEResult,
    GAE_LAMBDA,
    GAMMA,
    RolloutContractError,
    compute_gae,
)


@dataclass(slots=True)
class RolloutBatch:
    """Short-lived PPO arrays used only by an in-process optimizer update."""

    prior_channels: np.ndarray
    coverage_summary: np.ndarray
    local_crop: np.ndarray
    frontier_features: np.ndarray
    pose_features: np.ndarray
    candidate_mask: np.ndarray
    platform_context: np.ndarray
    selected_frontier_indices: np.ndarray
    selected_thetas: np.ndarray
    old_log_prob_total: np.ndarray
    old_values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray

    def __post_init__(self) -> None:
        float_arrays = (
            self.prior_channels,
            self.coverage_summary,
            self.local_crop,
            self.frontier_features,
            self.pose_features,
            self.platform_context,
            self.selected_thetas,
            self.old_log_prob_total,
            self.old_values,
            self.advantages,
            self.returns,
        )
        if any(
            not isinstance(value, np.ndarray) or value.dtype != np.float32
            for value in float_arrays
        ):
            raise RolloutContractError("rollout float arrays must use float32")
        if (
            not isinstance(self.candidate_mask, np.ndarray)
            or self.candidate_mask.dtype != np.bool_
        ):
            raise RolloutContractError("candidate_mask must use boolean dtype")
        if (
            not isinstance(self.selected_frontier_indices, np.ndarray)
            or self.selected_frontier_indices.dtype != np.int64
        ):
            raise RolloutContractError("selected frontier indices must use int64")
        sample_count = self.prior_channels.shape[0]
        if sample_count <= 0 or self.prior_channels.ndim != 4 or self.prior_channels.shape[1] != 7:
            raise RolloutContractError("prior channels must use [N,7,H,W]")
        if self.coverage_summary.ndim != 4 or self.coverage_summary.shape[:2] != (sample_count, 8):
            raise RolloutContractError("coverage summary must use [N,8,H,W]")
        if self.local_crop.ndim != 4 or self.local_crop.shape[:2] != (sample_count, 8):
            raise RolloutContractError("local crop must use [N,8,H,W]")
        if (
            self.frontier_features.ndim != 3
            or self.frontier_features.shape[0] != sample_count
            or self.frontier_features.shape[2] != 22
            or self.candidate_mask.shape != self.frontier_features.shape[:2]
        ):
            raise RolloutContractError("frontier features or mask shape is invalid")
        if self.pose_features.shape != (sample_count, 6):
            raise RolloutContractError("pose features must use [N,6]")
        if self.platform_context.shape != (sample_count, 3):
            raise RolloutContractError("platform context must use [N,3]")
        vectors = (
            self.selected_frontier_indices,
            self.selected_thetas,
            self.old_log_prob_total,
            self.old_values,
            self.advantages,
            self.returns,
        )
        if any(value.shape != (sample_count,) for value in vectors):
            raise RolloutContractError("rollout action and training vectors must use [N]")
        if any(not np.isfinite(value).all() for value in float_arrays):
            raise RolloutContractError("rollout float arrays must be finite")
        if not self.candidate_mask.any(axis=1).all():
            raise RolloutContractError("each rollout row must have a valid candidate")
        if (
            (self.selected_frontier_indices < 0).any()
            or (self.selected_frontier_indices >= self.candidate_mask.shape[1]).any()
        ):
            raise RolloutContractError("selected frontier index is out of range")
        selected_valid = self.candidate_mask[
            np.arange(sample_count), self.selected_frontier_indices
        ]
        if not selected_valid.all():
            raise RolloutContractError("selected frontier index is masked")

    def __len__(self) -> int:
        return int(self.prior_channels.shape[0])

    def select(
        self, indices: np.ndarray, *, device: torch.device | str
    ) -> tuple[
        PolicyBatch,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Materialize selected rows on one training device."""
        if (
            not isinstance(indices, np.ndarray)
            or indices.dtype != np.int64
            or indices.ndim != 1
            or indices.size == 0
            or (indices < 0).any()
            or (indices >= len(self)).any()
        ):
            raise RolloutContractError("selection indices must be valid int64 [K]")
        target = torch.device(device)

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(value[indices]).to(target)

        policy_batch = PolicyBatch(
            prior_channels=tensor(self.prior_channels),
            coverage_summary=tensor(self.coverage_summary),
            local_crop=tensor(self.local_crop),
            frontier_features=tensor(self.frontier_features),
            pose_features=tensor(self.pose_features),
            candidate_mask=tensor(self.candidate_mask),
            platform_context=tensor(self.platform_context),
        )
        validate_policy_batch(policy_batch)
        return (
            policy_batch,
            tensor(self.selected_frontier_indices),
            tensor(self.selected_thetas),
            tensor(self.old_log_prob_total),
            tensor(self.old_values),
            tensor(self.advantages),
            tensor(self.returns),
        )


__all__ = [
    "GAEResult",
    "GAE_LAMBDA",
    "GAMMA",
    "RolloutBatch",
    "RolloutContractError",
    "compute_gae",
]
