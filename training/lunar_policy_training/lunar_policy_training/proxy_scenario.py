"""Small deterministic proxy scenarios that exercise the real C++ v3 planner."""

from __future__ import annotations

import math
from datetime import timedelta
from dataclasses import dataclass

import numpy as np
import torch
import lunar_planner_training_bridge as bridge_api

from .curriculum import CurriculumSchedule, PLATFORMS
from .environment.macro_step import ExecutionEvents, PolicyAction
from .environment.parallel_pool import ParallelEnvironmentWorker
from .environment.v3_environment import (
    CommittedHopExecutionFeedback,
    PreparedPlanRequest,
    ReferenceExecutionResult,
    create_v3_environment,
)
from .policy.observation import ObservationIdentity, PolicyBatch


_PLATFORM_INDEX = {platform: index for index, platform in enumerate(PLATFORMS)}


def _vec2(x: float, y: float) -> bridge_api.Vec2:
    value = bridge_api.Vec2()
    value.x = x
    value.y = y
    return value


def _vec3(x: float, y: float, z: float) -> bridge_api.Vec3:
    value = bridge_api.Vec3()
    value.x = x
    value.y = y
    value.z = z
    return value


def _yaw_quaternion(yaw: float) -> bridge_api.Quaternion:
    value = bridge_api.Quaternion()
    value.w = math.cos(yaw / 2.0)
    value.z = math.sin(yaw / 2.0)
    return value


def _pose(x: float, y: float, z: float, *, yaw: float = 0.0) -> bridge_api.Pose3:
    value = bridge_api.Pose3()
    value.position_m = _vec3(x, y, z)
    value.orientation = _yaw_quaternion(yaw)
    return value


def _flat_proxy_map(
    frame_id: str,
    *,
    platform_type: str,
    terrain_id: str,
    stamp_ns: int,
) -> bridge_api.GridMap:
    if platform_type == "HOPPER":
        width, height, resolution = 16, 12, 0.5
    else:
        width, height, resolution = 12, 8, 1.0
    count = width * height
    terrain_amplitude = {
        "flat_sparse": 0.0,
        "rolling_sparse": 0.002,
        "full_proxy": 0.004,
    }[terrain_id]
    elevation = np.asarray(
        [terrain_amplitude * math.sin(index * 0.2) for index in range(count)],
        dtype=np.float32,
    )
    grid = bridge_api.GridMap()
    grid.frame_id = frame_id
    grid.stamp.nanoseconds_since_epoch = stamp_ns
    grid.width = width
    grid.height = height
    grid.resolution_m = resolution
    grid.layers = {
        "elevation": bridge_api.GridLayer(elevation),
        "valid_mask": bridge_api.GridLayer(np.ones(count, dtype=np.uint8)),
        "obstacle": bridge_api.GridLayer(np.zeros(count, dtype=np.uint8)),
        "obstacle_height": bridge_api.GridLayer(
            np.zeros(count, dtype=np.float32)
        ),
        "observation_age_s": bridge_api.GridLayer(
            np.zeros(count, dtype=np.float32)
        ),
        "observation_quality": bridge_api.GridLayer(
            np.ones(count, dtype=np.float32)
        ),
        "elevation_variance": bridge_api.GridLayer(
            np.zeros(count, dtype=np.float32)
        ),
        "obstacle_variance": bridge_api.GridLayer(
            np.zeros(count, dtype=np.float32)
        ),
        "observation_count": bridge_api.GridLayer(
            np.ones(count, dtype=np.uint32)
        ),
        "forbidden": bridge_api.GridLayer(np.zeros(count, dtype=np.uint8)),
    }
    return grid


def _wheel_capability() -> bridge_api.WheeledCapability:
    capability = bridge_api.WheeledCapability()
    capability.footprint_xy_m = [
        _vec2(-0.2, -0.2),
        _vec2(0.2, -0.2),
        _vec2(0.2, 0.2),
        _vec2(-0.2, 0.2),
    ]
    capability.minimum_body_z_m = -0.1
    capability.maximum_body_z_m = 0.5
    capability.maximum_forward_speed_mps = 1.0
    capability.maximum_reverse_speed_mps = 0.8
    capability.maximum_spin_rate_radps = 1.0
    capability.maximum_acceleration_mps2 = 1.0
    capability.maximum_braking_deceleration_mps2 = 1.0
    capability.maximum_yaw_acceleration_radps2 = 1.0
    capability.maximum_lateral_acceleration_mps2 = 1.0
    capability.maximum_curvature_per_m = 1.0
    capability.maximum_slope_rad = 0.5
    capability.minimum_clearance_m = 0.0
    yaw_bin_step = 2.0 * math.pi / 64.0
    primitives = []
    for primitive_id, kind, x, yaw, duration in (
        ("forward", bridge_api.WheelPrimitiveKind.FORWARD, 1.0, 0.0, 1.0),
        ("reverse", bridge_api.WheelPrimitiveKind.REVERSE, -1.0, 0.0, 1.0),
        (
            "forward-arc",
            bridge_api.WheelPrimitiveKind.FORWARD_ARC,
            1.0,
            math.pi / 2.0,
            1.0,
        ),
        (
            "reverse-arc",
            bridge_api.WheelPrimitiveKind.REVERSE_ARC,
            -1.0,
            -math.pi / 2.0,
            1.0,
        ),
        (
            "spin-left",
            bridge_api.WheelPrimitiveKind.SPIN_COUNTERCLOCKWISE,
            0.0,
            yaw_bin_step,
            0.5,
        ),
        (
            "spin-right",
            bridge_api.WheelPrimitiveKind.SPIN_CLOCKWISE,
            0.0,
            -yaw_bin_step,
            0.5,
        ),
    ):
        primitive = bridge_api.WheelMotionPrimitive()
        primitive.primitive_id = primitive_id
        primitive.kind = kind
        primitive.relative_end_pose = _pose(x, 0.0, 0.0, yaw=yaw)
        primitive.nominal_duration = timedelta(seconds=duration)
        primitives.append(primitive)
    capability.motion_primitives = primitives
    return capability


def _legged_capability() -> bridge_api.LeggedCapability:
    capability = bridge_api.LeggedCapability()
    capability.body_half_extent_m = _vec3(0.2, 0.2, 0.3)
    capability.maximum_slope_rad = 0.4
    capability.maximum_roughness_m = 0.2
    capability.maximum_step_height_m = 0.3
    capability.maximum_gap_width_m = 0.4
    capability.minimum_confidence = 0.8
    capability.minimum_body_clearance_m = 0.1
    capability.body_height_m.lower = 0.4
    capability.body_height_m.upper = 0.6
    capability.forward_speed_mps.lower = -0.5
    capability.forward_speed_mps.upper = 0.5
    capability.lateral_speed_mps.lower = -0.5
    capability.lateral_speed_mps.upper = 0.5
    capability.vertical_speed_mps.lower = -0.1
    capability.vertical_speed_mps.upper = 0.1
    capability.yaw_rate_radps.lower = -1.0
    capability.yaw_rate_radps.upper = 1.0
    capability.maximum_linear_acceleration_mps2 = 0.5
    capability.maximum_yaw_acceleration_radps2 = 1.0
    primitives = []
    for primitive_id, kind, x, y in (
        ("forward", bridge_api.LeggedPrimitiveKind.FORWARD, 1.0, 0.0),
        ("backward", bridge_api.LeggedPrimitiveKind.BACKWARD, -1.0, 0.0),
        ("left", bridge_api.LeggedPrimitiveKind.LATERAL_LEFT, 0.0, 1.0),
        ("right", bridge_api.LeggedPrimitiveKind.LATERAL_RIGHT, 0.0, -1.0),
    ):
        primitive = bridge_api.LeggedBodyPrimitive()
        primitive.primitive_id = primitive_id
        primitive.kind = kind
        primitive.body_frame_displacement_m = _vec3(x, y, 0.0)
        primitive.nominal_duration = timedelta(seconds=2)
        primitives.append(primitive)
    yaw_bin_step = 2.0 * math.pi / 64.0
    for primitive_id, yaw_change in (
        ("spin-left", yaw_bin_step),
        ("spin-right", -yaw_bin_step),
    ):
        primitive = bridge_api.LeggedBodyPrimitive()
        primitive.primitive_id = primitive_id
        primitive.kind = bridge_api.LeggedPrimitiveKind.SPIN
        primitive.yaw_change_rad = yaw_change
        primitive.nominal_duration = timedelta(milliseconds=250)
        primitives.append(primitive)
    capability.motion_primitives = primitives
    return capability


def _hopper_capability() -> bridge_api.HopperCapability:
    capability = bridge_api.HopperCapability()
    capability.body_half_extent_m = _vec3(0.35, 0.25, 0.5)
    capability.platform_mass_kg = 10.0
    capability.gravity_mps2 = _vec3(0.0, 0.0, -1.62)
    capability.maximum_landing_slope_rad = 0.4
    capability.maximum_landing_roughness_m = 0.1
    capability.maximum_plane_residual_m = 0.05
    capability.minimum_overhead_clearance_m = 0.0
    capability.minimum_lateral_clearance_m = 0.0
    capability.minimum_landing_region_area_m2 = 0.2
    capability.maximum_launch_speed_mps = 8.0
    capability.maximum_launch_impulse_newton_seconds = 100.0
    capability.minimum_flight_time = timedelta(milliseconds=500)
    capability.maximum_flight_time = timedelta(seconds=10)
    capability.maximum_landing_speed_mps = 8.0
    capability.minimum_downward_impact_speed_mps = 0.1
    capability.minimum_landing_clearance_m = 0.0
    capability.maximum_angular_speed_radps = 2.0
    capability.maximum_angular_acceleration_radps2 = 4.0
    capability.maximum_initial_angular_speed_radps = 0.2
    capability.minimum_settle_guard = timedelta(milliseconds=100)
    return capability


def proxy_observation(
    worker_index: int,
    platform_type: str,
    *,
    step: int,
    position: tuple[float, float] | None = None,
    coverage: float | None = None,
    visited: frozenset[tuple[float, float]] = frozenset(),
    episode_id: str | None = None,
    mission_revision: int = 0,
    map_snapshot_id: str | None = None,
    execution_state: str = "DECISION_BOUNDARY",
) -> PolicyBatch:
    """Build a seven-input proxy observation with real changed episode state."""
    if platform_type not in PLATFORMS:
        raise ValueError("unknown proxy platform")
    if position is None:
        position = (3.0, 3.0) if platform_type == "HOPPER" else (2.5, 3.5)
    if coverage is None:
        coverage = min(0.05 + 0.475 * step, 1.0)
    prior = torch.zeros((1, 4, 256, 256), dtype=torch.float32)
    prior[0, 0].fill_(float(_PLATFORM_INDEX[platform_type]) / 2.0)
    prior[0, 3, min(step, 255), worker_index % 256] = 1.0
    coverage_summary = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    coverage_summary[0, 0].fill_(coverage)
    coverage_summary[0, 1].fill_(1.0 - coverage)
    coverage_summary[0, 2].fill_(1.0 - coverage)
    local_crop = torch.zeros((1, 4, 32, 32), dtype=torch.float32)
    local_crop[0, 1].fill_(coverage)
    local_crop[0, 3, min(step, 31), worker_index % 32] = 1.0
    frontier = torch.zeros((1, 64, 12), dtype=torch.float32)
    targets = _targets(platform_type)
    for index, (target_x, target_y) in enumerate(targets):
        was_visited = (target_x, target_y) in visited
        dx = target_x - position[0]
        dy = target_y - position[1]
        distance = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        frontier[0, index, 0] = target_x / 12.0
        frontier[0, index, 1] = target_y / 8.0
        frontier[0, index, 2] = distance / 10.0
        frontier[0, index, 3] = math.sin(theta)
        frontier[0, index, 4] = math.cos(theta)
        frontier[0, index, 5] = 0.0 if was_visited else 0.475
        frontier[0, index, 6] = 0.0 if was_visited else 1.0 - coverage
        frontier[0, index, 7] = math.sin(theta)
        frontier[0, index, 8] = math.cos(theta)
        frontier[0, index, 9] = 1.0
        frontier[0, index, 10] = 1.0
        frontier[0, index, 11] = 0.0 if was_visited else 1.0 - coverage
    pose = torch.tensor(
        [[
            position[0] / 12.0,
            position[1] / 8.0,
            0.0,
            1.0,
            coverage,
            max(0.0, 1.0 - step / 8.0),
        ]],
        dtype=torch.float32,
    )
    platform_context = torch.zeros((1, 3), dtype=torch.float32)
    platform_context[0, _PLATFORM_INDEX[platform_type]] = 1.0
    state_time_ns = 1_000_000_000 + step * 1_000_000
    candidate_set = ",".join(
        f"{x:.3f}:{y:.3f}:{int((x, y) in visited)}"
        for x, y in targets
    )
    identity = ObservationIdentity(
        episode_id=episode_id or f"proxy-{platform_type}-{worker_index}",
        mission_revision=mission_revision,
        map_snapshot_id=map_snapshot_id or f"proxy-map-{step}",
        robot_state_id=f"{position[0]:.6f}:{position[1]:.6f}",
        state_time_ns=state_time_ns,
        execution_state=execution_state,
        candidate_set_id=candidate_set,
    )
    return PolicyBatch(
        prior_channels=prior,
        coverage_summary=coverage_summary,
        local_crop=local_crop,
        frontier_features=frontier,
        pose_features=pose,
        candidate_mask=torch.tensor([[True, True, True] + [False] * 61], dtype=torch.bool),
        platform_context=platform_context,
        observation_identities=(identity,),
    )


def _targets(platform_type: str) -> tuple[tuple[float, float], ...]:
    if platform_type == "HOPPER":
        return ((4.0, 3.0), (3.0, 4.0), (3.0, 3.0))
    return ((4.5, 3.5), (2.5, 5.5), (2.5, 3.5))


def _finite_positions(positions: tuple[tuple[float, float], ...]) -> bool:
    return all(
        math.isfinite(coordinate)
        for position in positions
        for coordinate in position
    )


def _inside_proxy_map(position: tuple[float, float]) -> bool:
    return 0.0 <= position[0] <= 12.0 and 0.0 <= position[1] <= 8.0


class _ProxyEpisode:
    def __init__(
        self, worker_index: int, platform_type: str, *, scenario_index: int
    ) -> None:
        self.worker_index = worker_index
        self.platform_type = platform_type
        self.scenario = CurriculumSchedule().scenario_for(
            platform_type=platform_type,
            scenario_index=scenario_index,
        )
        self.step = 0
        self.position = (
            (3.0, 3.0) if platform_type == "HOPPER" else (2.5, 3.5)
        )
        self.coverage = 0.05
        self.visited = {self.position}
        self.pending_target = self.position
        self.execution_state = (
            "GROUND_HOLD" if platform_type == "HOPPER" else "DECISION_BOUNDARY"
        )

    @property
    def observation(self) -> PolicyBatch:
        return proxy_observation(
            self.worker_index,
            self.platform_type,
            step=self.step,
            position=self.position,
            coverage=self.coverage,
            visited=frozenset(self.visited),
            episode_id=f"proxy-{self.scenario.scenario_seed}-{self.worker_index}",
            mission_revision=1,
            map_snapshot_id=(
                f"{self.scenario.terrain_id}-{self.step}"
            ),
            execution_state=self.execution_state,
        )

    def produce_observation(self) -> PolicyBatch:
        """Produce the latest immutable-boundary input for every ground decision."""
        return self.observation

    def build_request(
        self,
        action: PolicyAction,
        expected_identity: ObservationIdentity,
    ) -> PreparedPlanRequest:
        if not 0 <= action.frontier_index < 3:
            raise ValueError("proxy action frontier is outside the candidate set")
        current_identity = self.observation.observation_identities[0]
        if current_identity != expected_identity:
            raise ValueError("proxy request snapshot changed after policy preparation")
        target_index = action.frontier_index
        self.pending_target = _targets(self.platform_type)[target_index]
        stamp_ns = 1_000_000_000 + self.step * 1_000_000
        request = bridge_api.TrainingPlanRequest()
        request.request_id = (
            f"proxy-{self.scenario.scenario_seed}-{self.step}-{target_index}"
        )
        request.state_time.nanoseconds_since_epoch = stamp_ns
        goal = bridge_api.PointGoal()
        goal.position_m = _vec3(
            self.pending_target[0], self.pending_target[1], 0.0
        )
        goal.tolerance_m = 0.5 if self.platform_type == "HOPPER" else 0.2
        request.goal.goal_id = f"proxy-frontier-{target_index}"
        request.goal.target = goal
        request.goal.yaw_rad = float(action.theta_rad)
        request.goal.yaw_tolerance_rad = math.pi / 24.0
        if self.platform_type == "WHEELED":
            state = bridge_api.WheeledState()
            state.pose = _pose(*self.position, 0.0)
            request.current_state = state
            request.capability = _wheel_capability()
        elif self.platform_type == "LEGGED":
            state = bridge_api.LeggedState()
            state.body_pose = _pose(*self.position, 0.5)
            request.current_state = state
            request.capability = _legged_capability()
        else:
            state = bridge_api.HopperState()
            state.pose = _pose(*self.position, 0.5)
            request.current_state = state
            request.capability = _hopper_capability()
        request.world.global_map = _flat_proxy_map(
            "map",
            platform_type=self.platform_type,
            terrain_id=self.scenario.terrain_id,
            stamp_ns=stamp_ns,
        )
        request.world.local_map = _flat_proxy_map(
            "odom",
            platform_type=self.platform_type,
            terrain_id=self.scenario.terrain_id,
            stamp_ns=stamp_ns,
        )
        request.world.map_from_odom.parent_frame = "map"
        request.world.map_from_odom.child_frame = "odom"
        request.world.map_from_odom.stamp.nanoseconds_since_epoch = stamp_ns
        request.config.wheel.xy_resolution_m = 1.0
        request.config.wheel.yaw_bin_count = 64
        request.config.legged.xy_resolution_m = 1.0
        request.config.legged.yaw_bin_count = 64
        return PreparedPlanRequest(
            request=request,
            identity=expected_identity,
        )

    def execute_reference(
        self, reference: bridge_api.MotionReference
    ) -> ReferenceExecutionResult:
        if not isinstance(reference, bridge_api.MotionReference):
            return self._execution_failure(safety_violation_count=1)
        if reference.platform_type != self.platform_type:
            return self._execution_failure(platform_reference_mismatch_count=1)
        if self.platform_type == "HOPPER":
            return self._execute_hop_reference(reference)
        return self._execute_trajectory_reference(reference)

    def _execute_trajectory_reference(
        self, reference: bridge_api.MotionReference
    ) -> ReferenceExecutionResult:
        data = reference.data
        expected_semantics = (
            bridge_api.TrajectorySemantics.WHEELED_BASE
            if self.platform_type == "WHEELED"
            else bridge_api.TrajectorySemantics.LEGGED_BODY_REFERENCE
        )
        if not isinstance(data, bridge_api.TrajectoryReference):
            return self._execution_failure(platform_reference_mismatch_count=1)
        points = tuple(data.points)
        if data.semantics != expected_semantics:
            return self._execution_failure(platform_reference_mismatch_count=1)
        if not reference.plan_id or len(points) < 2:
            return self._execution_failure(safety_violation_count=1)
        positions = tuple(
            (point.pose.position_m.x, point.pose.position_m.y)
            for point in points
        )
        timestamps = tuple(point.time_from_start.total_seconds() for point in points)
        if (
            not _finite_positions(positions)
            or not all(math.isfinite(value) for value in timestamps)
            or timestamps[0] < 0.0
            or timestamps[-1] <= timestamps[0]
            or any(right < left for left, right in zip(timestamps, timestamps[1:]))
            or math.dist(positions[0], self.position) > 0.25
            or not all(_inside_proxy_map(position) for position in positions)
        ):
            return self._execution_failure(safety_violation_count=1)
        return self._finish_execution(
            executed_position=positions[-1],
            reference_samples_consumed=len(points),
            execution_state="DECISION_BOUNDARY",
        )

    def _execute_hop_reference(
        self, reference: bridge_api.MotionReference
    ) -> ReferenceExecutionResult:
        data = reference.data
        if not isinstance(data, bridge_api.HopReference):
            return self._execution_failure(
                platform_reference_mismatch_count=1,
                hopper_commitment_violation_count=1,
            )
        segments = tuple(data.segments)
        if not reference.plan_id or not segments:
            return self._execution_failure(
                safety_violation_count=1,
                hopper_commitment_violation_count=1,
            )
        previous_landing = self.position
        final_landing = self.position
        commitment_states = ["JUMP_COMMITTED"]
        for segment in segments:
            commitment_states.append("IN_FLIGHT")
            launch = (
                segment.launch_pose.position_m.x,
                segment.launch_pose.position_m.y,
            )
            boundary = tuple(
                (point.x, point.y) for point in segment.landing_region_boundary_m
            )
            velocity = segment.launch_velocity_mps
            if (
                not segment.segment_id
                or len(boundary) < 3
                or not _finite_positions((launch, *boundary))
                or math.dist(launch, previous_landing) > 0.25
                or not all(_inside_proxy_map(point) for point in boundary)
                or segment.flight_time.total_seconds() <= 0.0
                or not all(
                    math.isfinite(value)
                    for value in (velocity.x, velocity.y, velocity.z)
                )
                or not math.isfinite(segment.flight_tube_radius_m)
                or segment.flight_tube_radius_m <= 0.0
            ):
                return self._execution_failure(
                    safety_violation_count=1,
                    hopper_commitment_violation_count=1,
                )
            final_landing = (
                sum(point[0] for point in boundary) / len(boundary),
                sum(point[1] for point in boundary) / len(boundary),
            )
            previous_landing = final_landing
        commitment_states.append("LANDED_HOLD")
        return self._finish_execution(
            executed_position=final_landing,
            reference_samples_consumed=len(segments),
            execution_state="LANDED_HOLD",
            hopper_commitment_states=tuple(commitment_states),
        )

    def _finish_execution(
        self,
        *,
        executed_position: tuple[float, float],
        reference_samples_consumed: int,
        execution_state: str,
        hopper_commitment_states: tuple[str, ...] = (),
    ) -> ReferenceExecutionResult:
        tolerance = 0.5 if self.platform_type == "HOPPER" else 0.2
        if math.dist(executed_position, self.pending_target) > tolerance:
            return self._execution_failure(safety_violation_count=1)
        repeated = self.pending_target in self.visited
        previous_coverage = self.coverage
        self.position = executed_position
        self.visited.add(self.pending_target)
        self.step += 1
        gain = 0.0 if repeated else min(0.475, 1.0 - self.coverage)
        self.coverage = min(1.0, self.coverage + gain)
        self.execution_state = execution_state
        success_first_crossing = previous_coverage < 0.95 <= self.coverage
        return ReferenceExecutionResult(
            next_observation=self.observation,
            mission_observed_delta=gain,
            priority_observed_delta=0.0,
            normalized_execution_cost_contribution=0.0,
            normalized_execution_time_contribution=0.0,
            executed_without_new_coverage=gain == 0.0,
            success_first_crossing=success_first_crossing,
            episode_ended_without_success=False,
            hard_safety_violation=False,
            terminated=success_first_crossing,
            execution_state=execution_state,
            execution_events=ExecutionEvents(
                reference_samples_consumed=reference_samples_consumed,
                selected_action_observed_safe=True,
                hopper_commitment_states=hopper_commitment_states,
            ),
        )

    def _execution_failure(
        self,
        *,
        safety_violation_count: int = 0,
        platform_reference_mismatch_count: int = 0,
        hopper_commitment_violation_count: int = 0,
    ) -> ReferenceExecutionResult:
        self.step += 1
        self.execution_state = (
            "GROUND_HOLD"
            if self.platform_type == "HOPPER"
            else "DECISION_BOUNDARY"
        )
        return ReferenceExecutionResult(
            next_observation=self.observation,
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            normalized_execution_cost_contribution=0.0,
            normalized_execution_time_contribution=0.0,
            executed_without_new_coverage=True,
            success_first_crossing=False,
            episode_ended_without_success=True,
            hard_safety_violation=True,
            terminated=True,
            execution_state=(
                "GROUND_HOLD"
                if self.platform_type == "HOPPER"
                else "DECISION_BOUNDARY"
            ),
            execution_events=ExecutionEvents(
                safety_violation_count=safety_violation_count,
                platform_reference_mismatch_count=(
                    platform_reference_mismatch_count
                ),
                hopper_commitment_violation_count=(
                    hopper_commitment_violation_count
                ),
                execution_failure_count=1,
            ),
        )

    def committed_hop_feedback(self) -> CommittedHopExecutionFeedback:
        return CommittedHopExecutionFeedback(
            execution_state="LANDED_HOLD",
            next_observation=self.observation,
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            normalized_execution_cost_contribution=0.0,
            normalized_execution_time_contribution=0.0,
            executed_without_new_coverage=True,
            success_first_crossing=False,
            episode_ended_without_success=True,
            hard_safety_violation=True,
            terminated=True,
            execution_events=ExecutionEvents(
                hopper_commitment_violation_count=1,
                execution_failure_count=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class ProxyEnvironmentFactory:
    """Picklable factory with an optional fixed evaluation scenario index."""

    scenario_index: int | None = None

    def __call__(
        self, worker_index: int, platform_type: str
    ) -> ParallelEnvironmentWorker:
        selected_index = (
            worker_index % 3 if self.scenario_index is None else self.scenario_index
        )
        return _create_proxy_environment(
            worker_index, platform_type, scenario_index=selected_index
        )


def _create_proxy_environment(
    worker_index: int, platform_type: str, *, scenario_index: int
) -> ParallelEnvironmentWorker:
    episode = _ProxyEpisode(
        worker_index, platform_type, scenario_index=scenario_index
    )
    initial = episode.observation
    environment = create_v3_environment(
        platform_type=platform_type,
        request_builder=episode.build_request,
        initial_observation=initial,
        observation_provider=episode.produce_observation,
        reference_executor=episode.execute_reference,
        committed_hop_executor=(
            episode.committed_hop_feedback if platform_type == "HOPPER" else None
        ),
        plan_cost_scale=100.0,
        planner_elapsed_scale_s=1.0,
    )
    return ParallelEnvironmentWorker(
        environment=environment,
        initial_observation=initial,
    )


def proxy_environment_factory(
    worker_index: int, platform_type: str
) -> ParallelEnvironmentWorker:
    """Create one process-local deterministic proxy episode and real v3 bridge."""
    return _create_proxy_environment(
        worker_index, platform_type, scenario_index=worker_index % 3
    )


__all__ = [
    "ProxyEnvironmentFactory",
    "proxy_environment_factory",
    "proxy_observation",
]
