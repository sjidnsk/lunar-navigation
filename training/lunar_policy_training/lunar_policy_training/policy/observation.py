"""Training tensors that consume the shared observation contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lunar_model_contract import ObservationContractV2, validate_observation_inputs


@dataclass
class PolicyBatch:
    """Seven tensors in exactly ``ObservationContractV2.input_names`` order."""

    prior_channels: torch.Tensor
    coverage_summary: torch.Tensor
    local_crop: torch.Tensor
    frontier_features: torch.Tensor
    pose_features: torch.Tensor
    candidate_mask: torch.Tensor
    platform_context: torch.Tensor

    @property
    def input_names(self) -> tuple[str, ...]:
        return ObservationContractV2.input_names


def validate_policy_batch(batch: PolicyBatch) -> None:
    """Reject tensors that cannot be passed to the shared PPO architecture."""
    if not isinstance(batch, PolicyBatch):
        raise ValueError("policy batch must use PolicyBatch")
    values = {name: getattr(batch, name) for name in ObservationContractV2.input_names}
    if any(not isinstance(value, torch.Tensor) for value in values.values()):
        raise ValueError("policy inputs must be tensors")
    devices = {value.device for value in values.values()}
    if len(devices) != 1:
        raise ValueError("policy tensors must share a device")
    try:
        validate_observation_inputs(
            {name: value.detach().cpu().numpy() for name, value in values.items()}
        )
    except ValueError as error:
        raise ValueError(str(error)) from error
