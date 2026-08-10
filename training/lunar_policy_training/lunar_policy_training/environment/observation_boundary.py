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
from ..training_semantics import FORMAL_SUCCESS_COVERAGE_RATIO
from .observation_builder import Pose2
from .sensor_observation import (
    ObservationDelta,
    SensorObservationState,
    TrainingObservedGrid,
)


_PLATFORMS = frozenset(PLATFORM_CONTEXTS)
_GROUND_PLATFORMS = frozenset({"WHEELED", "LEGGED"})
_HOPPER_NO_OBSERVATION_STATES = frozenset({"JUMP_COMMITTED", "IN_FLIGHT"})


@dataclass(frozen=True, slots=True)
class SensorPathSample:
    """One physical sensor pose and elapsed time since the previous sample."""

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
class SensorBoundaryEvidence:
    """Final pose, total elapsed time, and optional ground-path samples."""

    pose_map: Pose2
    elapsed_s: float
    path_samples: tuple[SensorPathSample, ...] = ()

    def __post_init__(self) -> None:
        SensorPathSample(self.pose_map, self.elapsed_s)
        if not isinstance(self.path_samples, tuple) or any(
            not isinstance(sample, SensorPathSample)
            for sample in self.path_samples
        ):
            raise ValueError("sensor boundary path samples are invalid")
        if not self.path_samples:
            return
        if self.path_samples[-1].pose_map != self.pose_map:
            raise ValueError("sensor boundary final path sample differs from pose")
        elapsed_ns = int(round(float(self.elapsed_s) * 1_000_000_000.0))
        sample_elapsed_ns = sum(
            int(round(float(sample.elapsed_s) * 1_000_000_000.0))
            for sample in self.path_samples
        )
        if sample_elapsed_ns != elapsed_ns:
            raise ValueError("sensor boundary path elapsed time differs from total")


@dataclass(frozen=True, slots=True)
class BoundaryObservationResult:
    """One authoritative policy observation and its physical coverage reward."""

    next_observation: PolicyBatch
    mission_observed_delta: float
    priority_observed_delta: float
    mission_observed_ratio: float
    success_first_crossing: bool
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
            ("mission observed ratio", self.mission_observed_ratio),
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
        if type(self.success_first_crossing) is not bool:
            raise ValueError("boundary success crossing flag must be boolean")


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
        initial_state_time_ns: int = 0,
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
        if type(initial_state_time_ns) is not int or initial_state_time_ns < 0:
            raise ValueError("sensor boundary initial state time is invalid")
        self._platform_type = platform_type
        self._sensor_state = sensor_state
        self._policy_observation_builder = policy_observation_builder
        self._episode_id = episode_id
        self._mission_revision = mission_revision
        self._observation_revision = 0
        self._state_time_ns = initial_state_time_ns
        self._current_observation: PolicyBatch | None = None
        self._mission_observed_area_m2 = 0.0

        resolution = sensor_state.truth.canvas.geometry.resolution_m
        cell_area_m2 = resolution * resolution
        exact_mission_area = getattr(sensor_state, "mission_area_m2", None)
        exact_priority_area = getattr(sensor_state, "priority_area_m2", None)
        self._mission_area_m2 = (
            float(exact_mission_area)
            if exact_mission_area is not None
            else float(
                sensor_state.mission_roi_ratio.sum(dtype="float64")
                * cell_area_m2
            )
        )
        self._priority_area_m2 = (
            float(exact_priority_area)
            if exact_priority_area is not None
            else float(
                (
                    sensor_state.mission_priority
                    * sensor_state.mission_roi_ratio
                ).sum(dtype="float64")
                * cell_area_m2
            )
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
        _, mission_ratio, crossing = self._observe(evidence, execution_state)
        return BoundaryObservationResult(
            next_observation=self.current_observation,
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            mission_observed_ratio=mission_ratio,
            success_first_crossing=crossing,
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
                mission_observed_ratio=self._mission_observed_ratio(),
                success_first_crossing=False,
                updated=False,
            )
        expected_state = (
            "LANDED_HOLD" if platform_type == "HOPPER" else "DECISION_BOUNDARY"
        )
        if execution_state != expected_state:
            raise ValueError("execution state is not an exploration boundary")
        if not isinstance(evidence, SensorBoundaryEvidence):
            raise ValueError("sensor boundary evidence is required")
        if platform_type == "HOPPER" and evidence.path_samples:
            raise ValueError("hopper landing evidence must not contain path samples")
        delta, mission_ratio, crossing = self._observe(
            evidence, execution_state
        )
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
            mission_observed_ratio=mission_ratio,
            success_first_crossing=crossing,
            updated=True,
        )

    def _observe(
        self, evidence: SensorBoundaryEvidence, execution_state: str
    ):
        canvas = self._sensor_state.truth.canvas
        elapsed_ns = int(round(float(evidence.elapsed_s) * 1_000_000_000.0))
        if elapsed_ns < 0 or elapsed_ns > (1 << 63) - 1 - self._state_time_ns:
            raise ValueError("sensor boundary elapsed time is out of range")
        previous_ratio = self._mission_observed_ratio()
        samples = evidence.path_samples or (
            SensorPathSample(evidence.pose_map, evidence.elapsed_s),
        )
        visible_cells = 0
        newly_observed_cells = 0
        mission_observed_delta_m2 = 0.0
        priority_observed_delta_m2 = 0.0
        for sample in samples:
            try:
                canvas.world_to_grid(
                    sample.pose_map.x_m, sample.pose_map.y_m
                )
            except ValueError as error:
                raise ValueError(
                    "sensor boundary pose is outside the grid"
                ) from error
            sample_delta = self._sensor_state.observe_world(
                sample.pose_map, elapsed_s=float(sample.elapsed_s)
            )
            visible_cells += int(sample_delta.visible_cells)
            newly_observed_cells += int(sample_delta.newly_observed_cells)
            mission_observed_delta_m2 += float(
                sample_delta.mission_observed_delta_m2
            )
            priority_observed_delta_m2 += float(
                sample_delta.priority_observed_delta_m2
            )
        delta = ObservationDelta(
            visible_cells=visible_cells,
            newly_observed_cells=newly_observed_cells,
            mission_observed_delta_m2=mission_observed_delta_m2,
            priority_observed_delta_m2=priority_observed_delta_m2,
        )
        self._mission_observed_area_m2 = min(
            self._mission_area_m2,
            self._mission_observed_area_m2
            + float(delta.mission_observed_delta_m2),
        )
        mission_ratio = self._mission_observed_ratio()
        crossing = (
            previous_ratio < FORMAL_SUCCESS_COVERAGE_RATIO <= mission_ratio
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
        if observation.pose_features.shape != (1, 5):
            raise ValueError("policy observation pose features must be [1,5]")
        observation.pose_features[0, 4] = mission_ratio
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
        return delta, mission_ratio, crossing

    def _mission_observed_ratio(self) -> float:
        if self._mission_area_m2 <= 0.0:
            return 0.0
        return min(
            1.0,
            max(0.0, self._mission_observed_area_m2 / self._mission_area_m2),
        )

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
    "SensorPathSample",
]
