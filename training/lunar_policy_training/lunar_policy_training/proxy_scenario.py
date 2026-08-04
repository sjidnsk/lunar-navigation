"""Small deterministic proxy scenarios that exercise the real C++ v3 planner."""

from __future__ import annotations

import math
from datetime import timedelta
from dataclasses import dataclass

import numpy as np
import torch
import lunar_planner_training_bridge as bridge_api

from .curriculum import CurriculumSchedule, PLATFORMS
from .environment.macro_step import PolicyAction
from .environment.parallel_pool import ParallelEnvironmentWorker
from .environment.v3_environment import (
    CommittedHopExecutionFeedback,
    ReferenceExecutionResult,
    create_v3_environment,
)
from .policy.observation import PolicyBatch


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
            math.pi / 2.0,
            0.5,
        ),
        (
            "spin-right",
            bridge_api.WheelPrimitiveKind.SPIN_CLOCKWISE,
            0.0,
            -math.pi / 2.0,
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
) -> PolicyBatch:
    """Build a seven-input proxy observation with real changed episode state."""
    if platform_type not in PLATFORMS:
        raise ValueError("unknown proxy platform")
    if position is None:
        position = (3.0, 3.0) if platform_type == "HOPPER" else (2.5, 3.5)
    if coverage is None:
        coverage = min(0.05 + 0.475 * step, 1.0)
    prior = torch.zeros((1, 7, 8, 8), dtype=torch.float32)
    prior[0, 0].fill_(float(_PLATFORM_INDEX[platform_type]) / 2.0)
    prior[0, 4, min(step, 7), worker_index % 8] = 1.0
    coverage_summary = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
    coverage_summary[0, 0].fill_(coverage)
    coverage_summary[0, 1].fill_(1.0 - coverage)
    local_crop = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
    local_crop[0, 1].fill_(coverage)
    local_crop[0, 5, min(step, 7), worker_index % 8] = 1.0
    frontier = torch.zeros((1, 3, 22), dtype=torch.float32)
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
        frontier[0, index, 14] = math.sin(theta)
        frontier[0, index, 15] = math.cos(theta)
        frontier[0, index, 16] = 1.0
        frontier[0, index, 17] = 1.0
        frontier[0, index, 18] = distance / 10.0
        frontier[0, index, 19] = float(was_visited)
        frontier[0, index, 20] = 0.0 if was_visited else 1.0 - coverage
        frontier[0, index, 21] = 1.0
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
    return PolicyBatch(
        prior_channels=prior,
        coverage_summary=coverage_summary,
        local_crop=local_crop,
        frontier_features=frontier,
        pose_features=pose,
        candidate_mask=torch.ones((1, 3), dtype=torch.bool),
        platform_context=platform_context,
    )


def _targets(platform_type: str) -> tuple[tuple[float, float], ...]:
    if platform_type == "HOPPER":
        return ((4.0, 3.0), (3.0, 4.0), (3.0, 3.0))
    return ((4.5, 3.5), (2.5, 5.5), (2.5, 3.5))


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

    @property
    def observation(self) -> PolicyBatch:
        return proxy_observation(
            self.worker_index,
            self.platform_type,
            step=self.step,
            position=self.position,
            coverage=self.coverage,
            visited=frozenset(self.visited),
        )

    def build_request(self, action: PolicyAction) -> bridge_api.TrainingPlanRequest:
        if not 0 <= action.frontier_index < 3:
            raise ValueError("proxy action frontier is outside the candidate set")
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
        request.goal.yaw_tolerance_rad = math.pi
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
        request.config.legged.xy_resolution_m = 1.0
        return request

    def execute_reference(
        self, reference: bridge_api.MotionReference
    ) -> ReferenceExecutionResult:
        repeated = self.pending_target in self.visited
        previous = self.position
        self.position = self.pending_target
        self.visited.add(self.position)
        self.step += 1
        gain = 0.0 if repeated else min(0.475, 1.0 - self.coverage)
        self.coverage = min(1.0, self.coverage + gain)
        return ReferenceExecutionResult(
            next_observation=self.observation,
            coverage_delta=gain,
            goal_progress=math.hypot(
                self.position[0] - previous[0], self.position[1] - previous[1]
            )
            / 4.0,
            repeated_visit=repeated,
            terminated=self.coverage >= 0.99,
            execution_state=(
                "GROUND_HOLD"
                if self.platform_type == "HOPPER"
                else "DECISION_BOUNDARY"
            ),
        )

    def committed_hop_feedback(self) -> CommittedHopExecutionFeedback:
        executed = self.execute_reference(bridge_api.MotionReference())
        return CommittedHopExecutionFeedback(
            execution_state="LANDED_HOLD",
            next_observation=executed.next_observation,
            coverage_delta=executed.coverage_delta,
            goal_progress=executed.goal_progress,
            repeated_visit=executed.repeated_visit,
            terminated=executed.terminated,
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
