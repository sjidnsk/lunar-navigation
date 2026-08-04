"""Ephemeral seven-input rollout batch and frozen GAE math."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from lunar_model_contract import ObservationContractV2, validate_observation_inputs

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
        try:
            validate_observation_inputs(
                {
                    name: getattr(self, name)
                    for name in ObservationContractV2.input_names
                }
            )
        except ValueError as error:
            raise RolloutContractError(str(error)) from error
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
