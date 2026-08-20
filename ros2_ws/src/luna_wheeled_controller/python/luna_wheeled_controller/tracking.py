"""Pure, deterministic WHEELED path tracking primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, pi, sin
from typing import Sequence


PathPose = tuple[float, float, float]


@dataclass(frozen=True)
class TrackingPolicy:
    lookahead_m: float
    max_linear_mps: float
    max_angular_radps: float
    max_cross_track_error_m: float
    goal_position_tolerance_m: float
    goal_yaw_tolerance_rad: float


@dataclass(frozen=True)
class TrackingState:
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class TrackingCommand:
    linear_x_mps: float
    angular_z_radps: float
    complete: bool
    failure_reason: str | None = None


def validate_tracking_policy(policy: TrackingPolicy) -> str | None:
    fields = (
        ("LOOKAHEAD_INVALID", policy.lookahead_m),
        ("MAX_LINEAR_INVALID", policy.max_linear_mps),
        ("MAX_ANGULAR_INVALID", policy.max_angular_radps),
        ("MAX_CROSS_TRACK_INVALID", policy.max_cross_track_error_m),
        ("GOAL_POSITION_TOLERANCE_INVALID", policy.goal_position_tolerance_m),
        ("GOAL_YAW_TOLERANCE_INVALID", policy.goal_yaw_tolerance_rad),
    )
    for reason, value in fields:
        if not isfinite(value) or value <= 0.0:
            return reason
    return None


def _zero(reason: str | None = None) -> TrackingCommand:
    return TrackingCommand(0.0, 0.0, False, reason)


def _normalize_angle(angle_rad: float) -> float:
    return (angle_rad + pi) % (2.0 * pi) - pi


def _finite_path(path: Sequence[PathPose]) -> bool:
    return bool(path) and all(len(pose) == 3 and all(isfinite(value) for value in pose) for pose in path)


def _finite_state(state: TrackingState) -> bool:
    return all(isfinite(value) for value in (state.x_m, state.y_m, state.yaw_rad))


def _nearest_index_and_distance(path: Sequence[PathPose], state: TrackingState) -> tuple[int, float]:
    distances = tuple(hypot(x - state.x_m, y - state.y_m) for x, y, _ in path)
    index = min(range(len(path)), key=distances.__getitem__)
    return index, distances[index]


def _lookahead_pose(path: Sequence[PathPose], start: int, state: TrackingState, lookahead_m: float) -> PathPose:
    for pose in path[start:]:
        if hypot(pose[0] - state.x_m, pose[1] - state.y_m) >= lookahead_m:
            return pose
    return path[-1]


def track_path(
    path_xy_yaw: Sequence[PathPose],
    state: TrackingState,
    policy: TrackingPolicy,
) -> TrackingCommand:
    if validate_tracking_policy(policy) is not None or not _finite_path(path_xy_yaw) or not _finite_state(state):
        return _zero("INVALID_REFERENCE")

    goal_x, goal_y, goal_yaw = path_xy_yaw[-1]
    goal_distance = hypot(goal_x - state.x_m, goal_y - state.y_m)
    if goal_distance <= policy.goal_position_tolerance_m and abs(_normalize_angle(goal_yaw - state.yaw_rad)) <= policy.goal_yaw_tolerance_rad:
        return TrackingCommand(0.0, 0.0, True)

    nearest, cross_track_error = _nearest_index_and_distance(path_xy_yaw, state)
    if cross_track_error > policy.max_cross_track_error_m:
        return _zero("PATH_DEVIATION")

    target_x, target_y, _ = _lookahead_pose(path_xy_yaw, nearest, state, policy.lookahead_m)
    dx = target_x - state.x_m
    dy = target_y - state.y_m
    local_y = -sin(state.yaw_rad) * dx + cos(state.yaw_rad) * dy
    lookahead = max(hypot(dx, dy), policy.lookahead_m)
    curvature = 2.0 * local_y / (lookahead * lookahead)
    linear = min(policy.max_linear_mps, max(0.0, goal_distance))
    angular = max(-policy.max_angular_radps, min(policy.max_angular_radps, linear * curvature))
    if abs(angular) < 1e-12:
        angular = 0.0
    return TrackingCommand(linear, angular, False)
