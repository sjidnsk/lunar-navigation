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
from ..training_semantics import formal_success_first_crossing
from .macro_step import HopperObservationCommitment, HopperTrajectoryBuffer
from .multires_observation import (
    HopperTrajectoryPoint,
    MultiresSensorObservationState,
)
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


def executed_polyline_length_m(
    start: Pose2,
    samples: tuple[SensorPathSample, ...],
) -> float:
    """Return the 3-D length of only the sensor poses actually consumed."""
    if not isinstance(start, Pose2) or start.frame_id != "map":
        raise ValueError("executed path start must be a map-frame Pose2")
    if any(
        not math.isfinite(float(value))
        for value in (start.x_m, start.y_m, start.elevation_m)
    ):
        raise ValueError("executed path start must be finite")
    if not isinstance(samples, tuple) or any(
        not isinstance(sample, SensorPathSample) for sample in samples
    ):
        raise ValueError("executed path samples are invalid")
    points = (start, *(sample.pose_map for sample in samples))
    return math.fsum(
        math.dist(
            (left.x_m, left.y_m, left.elevation_m),
            (right.x_m, right.y_m, right.elevation_m),
        )
        for left, right in zip(points, points[1:])
    )


@dataclass(frozen=True, slots=True)
class BoundaryObservationResult:
    """One authoritative policy observation and its physical coverage reward."""

    next_observation: PolicyBatch
    mission_observed_delta: float
    priority_observed_delta: float
    mission_observed_ratio: float
    success_first_crossing: bool
    updated: bool
    coverage_before: float = 0.0
    coverage_after: float = 0.0
    priority_before: float = 0.0
    priority_after: float = 0.0
    executed_path_length_m: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.next_observation, PolicyBatch):
            raise ValueError("boundary result requires a PolicyBatch")
        if (
            self.next_observation.observation_identities is None
            or len(self.next_observation.observation_identities) != 1
        ):
            raise ValueError("boundary result requires one observation identity")
        for name, value in (
            ("mission delta", self.mission_observed_delta),
            ("priority delta", self.priority_observed_delta),
            ("mission observed ratio", self.mission_observed_ratio),
            ("coverage before", self.coverage_before),
            ("coverage after", self.coverage_after),
            ("priority before", self.priority_before),
            ("priority after", self.priority_after),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"boundary {name} must be in [0,1]")
        if self.coverage_after < self.coverage_before:
            raise ValueError("boundary coverage regressed")
        if self.priority_after < self.priority_before:
            raise ValueError("boundary priority coverage regressed")
        if (
            not isinstance(self.executed_path_length_m, (int, float))
            or isinstance(self.executed_path_length_m, bool)
            or not math.isfinite(float(self.executed_path_length_m))
            or self.executed_path_length_m < 0.0
        ):
            raise ValueError("boundary executed path length is invalid")
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
        self._priority_observed_area_m2 = 0.0
        self._current_pose: Pose2 | None = None
        self._hopper_trajectory_buffer: HopperTrajectoryBuffer | None = None
        self._last_hopper_commitment_id: str | None = None
        self._last_hopper_boundary_result: BoundaryObservationResult | None = None
        self._last_hopper_landing_evidence: SensorBoundaryEvidence | None = None

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

    @property
    def coverage_ratio(self) -> float:
        return self._mission_observed_ratio()

    @property
    def priority_ratio(self) -> float:
        return self._priority_observed_ratio()

    @property
    def current_pose(self) -> Pose2 | None:
        return self._current_pose

    @property
    def hopper_commitment_active(self) -> bool:
        return self._hopper_trajectory_buffer is not None

    def begin_hopper_trajectory(
        self, commitment: HopperObservationCommitment
    ) -> None:
        if self._platform_type != "HOPPER":
            raise ValueError("hopper trajectory requires a HOPPER controller")
        if self._current_pose is None or self._current_observation is None:
            raise RuntimeError("hopper trajectory requires an initialized boundary")
        if self._hopper_trajectory_buffer is not None:
            raise ValueError("hopper trajectory commitment is already active")
        if not isinstance(commitment, HopperObservationCommitment):
            raise ValueError("hopper trajectory commitment is invalid")
        if math.dist(
            (
                self._current_pose.x_m,
                self._current_pose.y_m,
                self._current_pose.elevation_m,
            ),
            (
                commitment.takeoff_pose.x_m,
                commitment.takeoff_pose.y_m,
                commitment.takeoff_pose.elevation_m,
            ),
        ) > 1.0e-6:
            raise ValueError("hopper trajectory takeoff differs from boundary")
        self._hopper_trajectory_buffer = HopperTrajectoryBuffer(commitment)

    def append_hopper_trajectory(
        self,
        commitment_id: str,
        points: tuple[HopperTrajectoryPoint, ...],
        *,
        within_certified_flight_tube: bool,
    ) -> None:
        buffer = self._hopper_trajectory_buffer
        if buffer is None:
            raise ValueError("hopper trajectory has no active commitment")
        buffer.append(
            commitment_id,
            points,
            within_certified_flight_tube=within_certified_flight_tube,
        )

    def reset(self, pose_map: Pose2) -> BoundaryObservationResult:
        """Perform the mandatory initial reveal without awarding action reward."""
        if self._current_observation is not None:
            raise RuntimeError("sensor boundary reset may only be called once")
        if (
            self._platform_type == "HOPPER"
            and isinstance(self._sensor_state, MultiresSensorObservationState)
        ):
            sensor_state = self._sensor_state
            rollback = sensor_state._capture_hopper_observation_rollback_state()
            initial_state_time_ns = self._state_time_ns
            try:
                patch = sensor_state.prepare_hopper_trajectory_observation(
                    (HopperTrajectoryPoint(pose_map, 0.0),)
                )
                delta = sensor_state.commit_hopper_trajectory_observation(
                    patch, elapsed_s=0.0
                )
                self._mission_observed_area_m2 = min(
                    self._mission_area_m2,
                    float(delta.mission_observed_delta_m2),
                )
                self._priority_observed_area_m2 = min(
                    self._priority_area_m2,
                    float(delta.priority_observed_delta_m2),
                )
                mission_ratio = self._mission_observed_ratio()
                priority_ratio = self._priority_observed_ratio()
                self._build_observation(pose_map, "GROUND_HOLD", mission_ratio)
            except Exception:
                sensor_state._restore_hopper_observation_rollback_state(rollback)
                self._mission_observed_area_m2 = 0.0
                self._priority_observed_area_m2 = 0.0
                self._observation_revision = 0
                self._state_time_ns = initial_state_time_ns
                self._current_observation = None
                raise
            self._current_pose = pose_map
            return BoundaryObservationResult(
                next_observation=self.current_observation,
                mission_observed_delta=0.0,
                priority_observed_delta=0.0,
                mission_observed_ratio=mission_ratio,
                success_first_crossing=False,
                updated=True,
                coverage_before=mission_ratio,
                coverage_after=mission_ratio,
                priority_before=priority_ratio,
                priority_after=priority_ratio,
                executed_path_length_m=0.0,
            )
        evidence = SensorBoundaryEvidence(pose_map, 0.0)
        execution_state = (
            "GROUND_HOLD"
            if self._platform_type == "HOPPER"
            else "DECISION_BOUNDARY"
        )
        _, mission_ratio, priority_ratio, _ = self._observe(
            evidence, execution_state
        )
        self._current_pose = pose_map
        return BoundaryObservationResult(
            next_observation=self.current_observation,
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            mission_observed_ratio=mission_ratio,
            success_first_crossing=False,
            updated=True,
            coverage_before=mission_ratio,
            coverage_after=mission_ratio,
            priority_before=priority_ratio,
            priority_after=priority_ratio,
            executed_path_length_m=0.0,
        )

    def replay_hopper_trajectory(
        self, evidence: SensorBoundaryEvidence
    ) -> BoundaryObservationResult:
        """Replay one previously committed trajectory at a stable boundary."""
        if self._platform_type != "HOPPER" or self._current_pose is None:
            raise ValueError("hopper trajectory replay requires a stable HOPPER")
        if not isinstance(evidence, SensorBoundaryEvidence) or not evidence.path_samples:
            raise ValueError("hopper trajectory replay requires path samples")
        token = hashlib.sha256(
            repr(
                (
                    self._episode_id,
                    self._mission_revision,
                    self._state_time_ns,
                    self._current_pose,
                    evidence,
                )
            ).encode("utf-8")
        ).hexdigest()
        commitment = HopperObservationCommitment(
            commitment_id=token,
            takeoff_pose=self._current_pose,
            expected_landing_pose=evidence.pose_map,
            flight_time_s=float(evidence.elapsed_s),
            flight_tube_radius_m=1.0,
        )
        self.begin_hopper_trajectory(commitment)
        cumulative = 0.0
        points: list[HopperTrajectoryPoint] = []
        for sample in evidence.path_samples:
            cumulative += float(sample.elapsed_s)
            points.append(HopperTrajectoryPoint(sample.pose_map, cumulative))
        self.append_hopper_trajectory(
            token,
            tuple(points),
            within_certified_flight_tube=True,
        )
        replay_evidence = SensorBoundaryEvidence(
            evidence.pose_map, evidence.elapsed_s
        )
        return self._commit_hopper_trajectory(token, replay_evidence)

    def after_execution(
        self,
        *,
        platform_type: str,
        execution_state: str,
        evidence: SensorBoundaryEvidence | None,
        hopper_commitment_id: str | None = None,
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
            coverage = self._mission_observed_ratio()
            priority = self._priority_observed_ratio()
            return BoundaryObservationResult(
                next_observation=self.current_observation,
                mission_observed_delta=0.0,
                priority_observed_delta=0.0,
                mission_observed_ratio=coverage,
                success_first_crossing=False,
                updated=False,
                coverage_before=coverage,
                coverage_after=coverage,
                priority_before=priority,
                priority_after=priority,
                executed_path_length_m=0.0,
            )
        expected_state = (
            "LANDED_HOLD" if platform_type == "HOPPER" else "DECISION_BOUNDARY"
        )
        if execution_state != expected_state:
            raise ValueError("execution state is not an exploration boundary")
        if not isinstance(evidence, SensorBoundaryEvidence):
            raise ValueError("sensor boundary evidence is required")
        if platform_type == "HOPPER":
            if evidence.path_samples:
                raise ValueError("hopper landing evidence must not contain path samples")
            if self._hopper_trajectory_buffer is not None:
                if not isinstance(hopper_commitment_id, str):
                    raise ValueError("hopper landing commitment identity is required")
                return self._commit_hopper_trajectory(
                    hopper_commitment_id, evidence
                )
            if (
                hopper_commitment_id is not None
                and hopper_commitment_id == self._last_hopper_commitment_id
                and self._last_hopper_boundary_result is not None
                and evidence == self._last_hopper_landing_evidence
            ):
                return self._last_hopper_boundary_result
            if hopper_commitment_id is not None:
                raise ValueError("hopper landing has no matching commitment")
        if self._current_pose is None:
            raise RuntimeError("sensor boundary current pose is unavailable")
        coverage_before = self._mission_observed_ratio()
        priority_before = self._priority_observed_ratio()
        executed_samples = evidence.path_samples or (
            SensorPathSample(evidence.pose_map, evidence.elapsed_s),
        )
        executed_path_length = executed_polyline_length_m(
            self._current_pose, executed_samples
        )
        delta, mission_ratio, priority_ratio, crossing = self._observe(
            evidence, execution_state
        )
        self._current_pose = evidence.pose_map
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
            coverage_before=coverage_before,
            coverage_after=mission_ratio,
            priority_before=priority_before,
            priority_after=priority_ratio,
            executed_path_length_m=executed_path_length,
        )

    def _commit_hopper_trajectory(
        self,
        commitment_id: str,
        evidence: SensorBoundaryEvidence,
    ) -> BoundaryObservationResult:
        buffer = self._hopper_trajectory_buffer
        if buffer is None:
            raise ValueError("hopper trajectory has no active commitment")
        sensor_state = self._sensor_state
        if not isinstance(sensor_state, MultiresSensorObservationState):
            raise ValueError("hopper trajectory requires multires observation state")
        points = buffer.finalize(
            commitment_id,
            landing_pose=evidence.pose_map,
            elapsed_s=float(evidence.elapsed_s),
        )
        patch = sensor_state.prepare_hopper_trajectory_observation(points)
        sensor_rollback = sensor_state._capture_hopper_observation_rollback_state()
        previous_observation = self._current_observation
        previous_pose = self._current_pose
        previous_mission_area = self._mission_observed_area_m2
        previous_priority_area = self._priority_observed_area_m2
        previous_revision = self._observation_revision
        previous_state_time_ns = self._state_time_ns
        coverage_before = self._mission_observed_ratio()
        priority_before = self._priority_observed_ratio()
        elapsed_ns = int(round(float(evidence.elapsed_s) * 1_000_000_000.0))
        if elapsed_ns < 0 or elapsed_ns > (1 << 63) - 1 - self._state_time_ns:
            raise ValueError("sensor boundary elapsed time is out of range")
        try:
            delta = sensor_state.commit_hopper_trajectory_observation(
                patch, elapsed_s=float(evidence.elapsed_s)
            )
            self._mission_observed_area_m2 = min(
                self._mission_area_m2,
                self._mission_observed_area_m2
                + float(delta.mission_observed_delta_m2),
            )
            self._priority_observed_area_m2 = min(
                self._priority_area_m2,
                self._priority_observed_area_m2
                + float(delta.priority_observed_delta_m2),
            )
            self._state_time_ns += elapsed_ns
            mission_ratio = self._mission_observed_ratio()
            priority_ratio = self._priority_observed_ratio()
            crossing = formal_success_first_crossing(
                coverage_before, mission_ratio
            )
            self._build_observation(
                evidence.pose_map, "LANDED_HOLD", mission_ratio
            )
            path_samples = tuple(
                SensorPathSample(
                    point.pose_map,
                    (
                        float(point.time_s)
                        - float(points[index - 1].time_s)
                        if index > 0
                        else float(point.time_s)
                    ),
                )
                for index, point in enumerate(points[1:], start=1)
            )
            executed_path_length = executed_polyline_length_m(
                points[0].pose_map, path_samples
            )
            result = BoundaryObservationResult(
                next_observation=self.current_observation,
                mission_observed_delta=(
                    float(delta.mission_observed_delta_m2)
                    / self._mission_area_m2
                    if self._mission_area_m2 > 0.0
                    else 0.0
                ),
                priority_observed_delta=(
                    float(delta.priority_observed_delta_m2)
                    / self._priority_area_m2
                    if self._priority_area_m2 > 0.0
                    else 0.0
                ),
                mission_observed_ratio=mission_ratio,
                success_first_crossing=crossing,
                updated=True,
                coverage_before=coverage_before,
                coverage_after=mission_ratio,
                priority_before=priority_before,
                priority_after=priority_ratio,
                executed_path_length_m=executed_path_length,
            )
        except Exception:
            sensor_state._restore_hopper_observation_rollback_state(
                sensor_rollback
            )
            self._current_observation = previous_observation
            self._current_pose = previous_pose
            self._mission_observed_area_m2 = previous_mission_area
            self._priority_observed_area_m2 = previous_priority_area
            self._observation_revision = previous_revision
            self._state_time_ns = previous_state_time_ns
            raise
        self._current_pose = evidence.pose_map
        self._hopper_trajectory_buffer = None
        self._last_hopper_commitment_id = commitment_id
        self._last_hopper_boundary_result = result
        self._last_hopper_landing_evidence = evidence
        return result

    def rebuild_without_sensor_update(
        self,
        *,
        pose_map: Pose2,
        execution_state: str,
    ) -> BoundaryObservationResult:
        """Rebuild policy candidates while preserving all physical evidence."""
        if self._current_observation is None:
            raise RuntimeError("sensor boundary must be reset before rebuild")
        if not isinstance(pose_map, Pose2) or pose_map.frame_id != "map":
            raise ValueError("sensor boundary rebuild pose must be map-frame Pose2")
        stable_states = (
            {"GROUND_HOLD", "LANDED_HOLD"}
            if self._platform_type == "HOPPER"
            else {"DECISION_BOUNDARY"}
        )
        if execution_state not in stable_states:
            raise ValueError("sensor boundary rebuild requires a stable state")
        mission_ratio = self._mission_observed_ratio()
        priority_ratio = self._priority_observed_ratio()
        self._build_observation(pose_map, execution_state, mission_ratio)
        return BoundaryObservationResult(
            next_observation=self.current_observation,
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            mission_observed_ratio=mission_ratio,
            success_first_crossing=False,
            updated=True,
            coverage_before=mission_ratio,
            coverage_after=mission_ratio,
            priority_before=priority_ratio,
            priority_after=priority_ratio,
            executed_path_length_m=0.0,
        )

    def _observe(
        self, evidence: SensorBoundaryEvidence, execution_state: str
    ):
        elapsed_ns = int(round(float(evidence.elapsed_s) * 1_000_000_000.0))
        if elapsed_ns < 0 or elapsed_ns > (1 << 63) - 1 - self._state_time_ns:
            raise ValueError("sensor boundary elapsed time is out of range")
        previous_ratio = self._mission_observed_ratio()
        samples = evidence.path_samples or (
            SensorPathSample(evidence.pose_map, evidence.elapsed_s),
        )
        sample_cells: list[tuple[int, int]] = []
        for sample in samples:
            try:
                sample_cells.append(
                    self._sensor_state.observation_cell_world(sample.pose_map)
                )
            except ValueError as error:
                raise ValueError(
                    "sensor boundary pose is outside the grid"
                ) from error
        visible_cells = 0
        newly_observed_cells = 0
        mission_observed_delta_m2 = 0.0
        priority_observed_delta_m2 = 0.0
        sample_deltas: list[ObservationDelta] = []
        if self._platform_type in _GROUND_PLATFORMS:
            observe_world_path = getattr(
                self._sensor_state, "observe_world_path", None
            )
            if callable(observe_world_path):
                sample_deltas.append(
                    observe_world_path(
                        tuple(
                            (sample.pose_map, float(sample.elapsed_s))
                            for sample in samples
                        )
                    )
                )
            else:
                group_start = 0
                while group_start < len(samples):
                    group_end = group_start + 1
                    while (
                        group_end < len(samples)
                        and sample_cells[group_end] == sample_cells[group_start]
                    ):
                        group_end += 1
                    sample_deltas.append(
                        self._sensor_state.observe_world_repeated(
                            samples[group_start].pose_map,
                            elapsed_steps_s=tuple(
                                float(sample.elapsed_s)
                                for sample in samples[group_start:group_end]
                            ),
                        )
                    )
                    group_start = group_end
        else:
            sample_deltas.extend(
                self._sensor_state.observe_world(
                    sample.pose_map, elapsed_s=float(sample.elapsed_s)
                )
                for sample in samples
            )
        for sample_delta in sample_deltas:
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
        self._priority_observed_area_m2 = min(
            self._priority_area_m2,
            self._priority_observed_area_m2
            + float(delta.priority_observed_delta_m2),
        )
        priority_ratio = self._priority_observed_ratio()
        crossing = formal_success_first_crossing(previous_ratio, mission_ratio)
        self._state_time_ns += elapsed_ns
        self._build_observation(evidence.pose_map, execution_state, mission_ratio)
        return delta, mission_ratio, priority_ratio, crossing

    def _build_observation(
        self,
        pose_map: Pose2,
        execution_state: str,
        mission_ratio: float,
    ) -> None:
        self._observation_revision += 1
        observation = self._policy_observation_builder(
            self._sensor_state.observed.copy(), pose_map
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
        identity = self._make_identity(observation, pose_map, execution_state)
        self._current_observation = _clone_policy_batch(
            observation, identity=identity
        )

    def _mission_observed_ratio(self) -> float:
        if self._mission_area_m2 <= 0.0:
            return 0.0
        return min(
            1.0,
            max(0.0, self._mission_observed_area_m2 / self._mission_area_m2),
        )

    def _priority_observed_ratio(self) -> float:
        if self._priority_area_m2 <= 0.0:
            return 0.0
        return min(
            1.0,
            max(0.0, self._priority_observed_area_m2 / self._priority_area_m2),
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
    "executed_polyline_length_m",
]
