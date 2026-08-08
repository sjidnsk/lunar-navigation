"""Exploration decision boundaries backed by actual conservative observations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable

import torch

from lunar_model_contract.observation import PLATFORM_CONTEXTS

from ..policy.observation import (
    ObservationIdentity,
    PolicyBatch,
    validate_policy_batch,
)
from .observation_builder import Pose2
from .sensor_observation import SensorObservationState, TrainingObservedGrid


_PLATFORMS = frozenset(PLATFORM_CONTEXTS)
_GROUND_PLATFORMS = frozenset({"WHEELED", "LEGGED"})
_HOPPER_NO_OBSERVATION_STATES = frozenset({"JUMP_COMMITTED", "IN_FLIGHT"})


@dataclass(frozen=True, slots=True)
class SensorBoundaryEvidence:
    """Physical pose and elapsed macro-step time supplied by execution."""

    pose_map: Pose2
    elapsed_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.pose_map, Pose2) or self.pose_map.frame_id != "map":
            raise ValueError("sensor boundary pose must be a map-frame Pose2")
        pose_values = (
            self.pose_map.x_m,
            self.pose_map.y_m,
            self.pose_map.yaw_rad,
            self.pose_map.elevation_m,
        )
        if any(not math.isfinite(float(value)) for value in pose_values):
            raise ValueError("sensor boundary pose must be finite")
        if (
            not isinstance(self.elapsed_s, (int, float))
            or isinstance(self.elapsed_s, bool)
            or not math.isfinite(float(self.elapsed_s))
            or self.elapsed_s < 0.0
        ):
            raise ValueError("sensor boundary elapsed time is invalid")


@dataclass(frozen=True, slots=True)
class BoundaryObservationResult:
    """One authoritative policy observation and its physical coverage reward."""

    next_observation: PolicyBatch
    mission_observed_delta: float
    priority_observed_delta: float
    updated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.next_observation, PolicyBatch):
            raise ValueError("boundary result requires a PolicyBatch")
        if (
            self.next_observation.observation_identities is None
            or len(self.next_observation.observation_identities) != 1
        ):
            raise ValueError("boundary result requires one observation identity")
        for name, value in (
            ("mission", self.mission_observed_delta),
            ("priority", self.priority_observed_delta),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"boundary {name} delta must be in [0,1]")
        if type(self.updated) is not bool:
            raise ValueError("boundary updated flag must be boolean")


PolicyObservationBuilder = Callable[
    [TrainingObservedGrid, Pose2], PolicyBatch
]


class ObservationBoundaryController:
    """Own initial reveal and every later exploration observation revision."""

    def __init__(
        self,
        *,
        platform_type: str,
        sensor_state: SensorObservationState,
        policy_observation_builder: PolicyObservationBuilder,
        episode_id: str,
        mission_revision: int,
    ) -> None:
        if platform_type not in _PLATFORMS:
            raise ValueError("sensor boundary platform is invalid")
        if not isinstance(sensor_state, SensorObservationState):
            raise TypeError("sensor boundary requires SensorObservationState")
        if not callable(policy_observation_builder):
            raise TypeError("sensor boundary requires a policy observation builder")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("sensor boundary episode identity is missing")
        if type(mission_revision) is not int or mission_revision < 0:
            raise ValueError("sensor boundary mission revision is invalid")
        self._platform_type = platform_type
        self._sensor_state = sensor_state
        self._policy_observation_builder = policy_observation_builder
        self._episode_id = episode_id
        self._mission_revision = mission_revision
        self._observation_revision = 0
        self._state_time_ns = 0
        self._current_observation: PolicyBatch | None = None

        resolution = sensor_state.truth.canvas.geometry.resolution_m
        cell_area_m2 = resolution * resolution
        self._mission_area_m2 = float(
            sensor_state.mission_roi_ratio.sum(dtype="float64") * cell_area_m2
        )
        self._priority_area_m2 = float(
            (
                sensor_state.mission_priority
                * sensor_state.mission_roi_ratio
            ).sum(dtype="float64")
            * cell_area_m2
        )

    @property
    def platform_type(self) -> str:
        return self._platform_type

    @property
    def sensor_state(self) -> SensorObservationState:
        return self._sensor_state

    @property
    def initialized(self) -> bool:
        return self._current_observation is not None

    @property
    def current_observation(self) -> PolicyBatch:
        if self._current_observation is None:
            raise RuntimeError("sensor boundary has not been reset")
        return _clone_policy_batch(self._current_observation)

    def reset(self, pose_map: Pose2) -> BoundaryObservationResult:
        """Perform the mandatory initial reveal without awarding action reward."""
        if self._current_observation is not None:
            raise RuntimeError("sensor boundary reset may only be called once")
        evidence = SensorBoundaryEvidence(pose_map, 0.0)
        execution_state = (
            "GROUND_HOLD"
            if self._platform_type == "HOPPER"
            else "DECISION_BOUNDARY"
        )
        self._observe(evidence, execution_state)
        return BoundaryObservationResult(
            next_observation=self.current_observation,
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            updated=True,
        )

    def after_execution(
        self,
        *,
        platform_type: str,
        execution_state: str,
        evidence: SensorBoundaryEvidence | None,
    ) -> BoundaryObservationResult:
        """Reveal only at an exploration boundary, never at a hop sub-state."""
        if self._current_observation is None:
            raise RuntimeError("sensor boundary must be reset before execution")
        if platform_type != self._platform_type:
            raise ValueError("sensor boundary platform mismatch")
        if platform_type == "HOPPER" and execution_state in (
            _HOPPER_NO_OBSERVATION_STATES
        ):
            if evidence is not None:
                raise ValueError("in-flight hopper feedback must not reveal sensors")
            return BoundaryObservationResult(
                next_observation=self.current_observation,
                mission_observed_delta=0.0,
                priority_observed_delta=0.0,
                updated=False,
            )
        expected_state = (
            "LANDED_HOLD" if platform_type == "HOPPER" else "DECISION_BOUNDARY"
        )
        if execution_state != expected_state:
            raise ValueError("execution state is not an exploration boundary")
        if not isinstance(evidence, SensorBoundaryEvidence):
            raise ValueError("sensor boundary evidence is required")
        delta = self._observe(evidence, execution_state)
        mission_delta = (
            delta.mission_observed_delta_m2 / self._mission_area_m2
            if self._mission_area_m2 > 0.0
            else 0.0
        )
        priority_delta = (
            delta.priority_observed_delta_m2 / self._priority_area_m2
            if self._priority_area_m2 > 0.0
            else 0.0
        )
        return BoundaryObservationResult(
            next_observation=self.current_observation,
            mission_observed_delta=mission_delta,
            priority_observed_delta=priority_delta,
            updated=True,
        )

    def _observe(
        self, evidence: SensorBoundaryEvidence, execution_state: str
    ):
        canvas = self._sensor_state.truth.canvas
        try:
            pose_cell = canvas.world_to_grid(
                evidence.pose_map.x_m, evidence.pose_map.y_m
            )
        except ValueError as error:
            raise ValueError("sensor boundary pose is outside the grid") from error
        elapsed_ns = int(round(float(evidence.elapsed_s) * 1_000_000_000.0))
        if elapsed_ns < 0 or elapsed_ns > (1 << 63) - 1 - self._state_time_ns:
            raise ValueError("sensor boundary elapsed time is out of range")
        delta = self._sensor_state.observe(
            pose_cell, elapsed_s=float(evidence.elapsed_s)
        )
        self._state_time_ns += elapsed_ns
        self._observation_revision += 1
        observation = self._policy_observation_builder(
            self._sensor_state.observed.copy(), evidence.pose_map
        )
        if not isinstance(observation, PolicyBatch):
            raise ValueError("policy observation builder returned invalid data")
        if observation.observation_identities is not None:
            raise ValueError("sensor boundary owns observation identities")
        try:
            validate_policy_batch(observation)
        except ValueError as error:
            raise ValueError("policy observation builder violated the contract") from error
        expected_context = torch.tensor(
            [PLATFORM_CONTEXTS[self._platform_type]],
            dtype=torch.float32,
            device=observation.platform_context.device,
        )
        if not torch.equal(observation.platform_context, expected_context):
            raise ValueError("policy observation platform does not match controller")
        identity = self._make_identity(observation, evidence.pose_map, execution_state)
        self._current_observation = _clone_policy_batch(
            observation, identity=identity
        )
        return delta

    def _make_identity(
        self,
        observation: PolicyBatch,
        pose_map: Pose2,
        execution_state: str,
    ) -> ObservationIdentity:
        canvas_id = self._sensor_state.truth.canvas.identity
        revision = self._observation_revision
        valid_mask = self._sensor_state.observed.valid_mask
        map_snapshot_id = _digest(
            canvas_id.encode("ascii"),
            revision.to_bytes(8, "big"),
            valid_mask.tobytes(order="C"),
        )
        robot_state_id = _digest(
            repr(
                (
                    pose_map.x_m,
                    pose_map.y_m,
                    pose_map.yaw_rad,
                    pose_map.elevation_m,
                    revision,
                )
            ).encode("ascii")
        )
        candidate_set_id = _digest(
            observation.candidate_mask.detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes(),
            observation.frontier_features.detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes(),
        )
        return ObservationIdentity(
            episode_id=self._episode_id,
            mission_revision=self._mission_revision,
            map_snapshot_id=map_snapshot_id,
            robot_state_id=robot_state_id,
            state_time_ns=self._state_time_ns,
            execution_state=execution_state,
            candidate_set_id=candidate_set_id,
        )


def _digest(*chunks: bytes) -> str:
    value = hashlib.sha256()
    for chunk in chunks:
        value.update(chunk)
    return value.hexdigest()


def _clone_policy_batch(
    observation: PolicyBatch,
    *,
    identity: ObservationIdentity | None = None,
) -> PolicyBatch:
    identities = (
        (identity,)
        if identity is not None
        else observation.observation_identities
    )
    return PolicyBatch(
        prior_channels=observation.prior_channels.clone(),
        coverage_summary=observation.coverage_summary.clone(),
        local_crop=observation.local_crop.clone(),
        frontier_features=observation.frontier_features.clone(),
        pose_features=observation.pose_features.clone(),
        candidate_mask=observation.candidate_mask.clone(),
        platform_context=observation.platform_context.clone(),
        observation_identities=identities,
    )


__all__ = [
    "BoundaryObservationResult",
    "ObservationBoundaryController",
    "SensorBoundaryEvidence",
]
