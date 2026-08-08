"""Training tensors that consume the shared observation contract."""

from __future__ import annotations

from dataclasses import InitVar, dataclass

import torch

from lunar_model_contract import ObservationContractV3, validate_observation_inputs


@dataclass(frozen=True, slots=True)
class ObservationIdentity:
    """Producer-owned decision-boundary identity; never a network input."""

    episode_id: str
    mission_revision: int
    map_snapshot_id: str
    robot_state_id: str
    state_time_ns: int
    execution_state: str
    candidate_set_id: str

    def __post_init__(self) -> None:
        text_values = (
            self.episode_id,
            self.map_snapshot_id,
            self.robot_state_id,
            self.execution_state,
            self.candidate_set_id,
        )
        if any(not isinstance(value, str) or not value for value in text_values):
            raise ValueError("observation identity text fields must be non-empty")
        if type(self.mission_revision) is not int or self.mission_revision < 0:
            raise ValueError("mission revision must be a non-negative integer")
        if type(self.state_time_ns) is not int or self.state_time_ns < 0:
            raise ValueError("state time must be a non-negative integer")


@dataclass
class PolicyBatch:
    """Seven tensors in exactly ``ObservationContractV3.input_names`` order."""

    prior_channels: torch.Tensor
    coverage_summary: torch.Tensor
    local_crop: torch.Tensor
    frontier_features: torch.Tensor
    pose_features: torch.Tensor
    candidate_mask: torch.Tensor
    platform_context: torch.Tensor
    observation_identities: InitVar[tuple[ObservationIdentity, ...] | None] = None

    def __post_init__(
        self,
        observation_identities: tuple[ObservationIdentity, ...] | None,
    ) -> None:
        # InitVar keeps identity outside the frozen seven-field network schema.
        self.observation_identities = observation_identities

    @property
    def input_names(self) -> tuple[str, ...]:
        return ObservationContractV3.input_names


def validate_policy_batch(batch: PolicyBatch) -> None:
    """Reject tensors that cannot be passed to the shared PPO architecture."""
    if not isinstance(batch, PolicyBatch):
        raise ValueError("policy batch must use PolicyBatch")
    values = {name: getattr(batch, name) for name in ObservationContractV3.input_names}
    if any(not isinstance(value, torch.Tensor) for value in values.values()):
        raise ValueError("policy inputs must be tensors")
    devices = {value.device for value in values.values()}
    if len(devices) != 1:
        raise ValueError("policy tensors must share a device")
    if batch.observation_identities is not None:
        if (
            not isinstance(batch.observation_identities, tuple)
            or len(batch.observation_identities) != batch.prior_channels.shape[0]
            or any(
                not isinstance(identity, ObservationIdentity)
                for identity in batch.observation_identities
            )
        ):
            raise ValueError(
                "observation identities must contain one immutable identity per row"
            )
    try:
        validate_observation_inputs(
            {name: value.detach().cpu().numpy() for name, value in values.items()}
        )
    except ValueError as error:
        raise ValueError(str(error)) from error


__all__ = ["ObservationIdentity", "PolicyBatch", "validate_policy_batch"]
