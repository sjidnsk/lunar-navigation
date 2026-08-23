"""Deterministic, dependency-free pure-pursuit tracking."""

from dataclasses import dataclass
import math
from typing import TypeAlias


PathXYYaw: TypeAlias = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class TrackingState:
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class TrackingPolicy:
    lookahead_m: float = 0.5
    max_linear_mps: float = 0.2
    max_angular_radps: float = 0.5
    goal_position_tolerance_m: float = 0.2
    goal_yaw_tolerance_rad: float = 0.2
    max_cross_track_error_m: float = 1.0


@dataclass(frozen=True)
class TrackingCommand:
    linear_x_mps: float
    angular_z_radps: float
    complete: bool
    failure_reason: str | None


def _stopped(reason: str) -> TrackingCommand:
    return TrackingCommand(0.0, 0.0, False, reason)


def _finite_path(path: PathXYYaw) -> tuple[tuple[float, float, float], ...] | None:
    try:
        points = tuple(tuple(float(value) for value in point) for point in path)
    except (TypeError, ValueError):
        return None
    if not points or any(len(point) != 3 for point in points):
        return None
    if not all(math.isfinite(value) for point in points for value in point):
        return None
    return points


def _finite_state(state: TrackingState) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in (state.x_m, state.y_m, state.yaw_rad))
    except (TypeError, ValueError):
        return False


def _valid_policy(policy: TrackingPolicy) -> bool:
    try:
        values = (
            policy.lookahead_m,
            policy.max_linear_mps,
            policy.max_angular_radps,
            policy.goal_position_tolerance_m,
            policy.goal_yaw_tolerance_rad,
            policy.max_cross_track_error_m,
        )
        return all(math.isfinite(float(value)) and float(value) > 0.0 for value in values)
    except (AttributeError, TypeError, ValueError):
        return False


def _angle_error(target: float, current: float) -> float:
    return abs((target - current + math.pi) % (2.0 * math.pi) - math.pi)


def track_path(path: PathXYYaw, state: TrackingState, policy: TrackingPolicy) -> TrackingCommand:
    """Compute one bounded wheel command, or a stopped terminal result."""
    points = _finite_path(path)
    if points is None or not _finite_state(state) or not _valid_policy(policy):
        return _stopped("INVALID_INPUT")

    x = float(state.x_m)
    y = float(state.y_m)
    yaw = float(state.yaw_rad)
    goal_x, goal_y, goal_yaw = points[-1]
    goal_distance = math.hypot(goal_x - x, goal_y - y)
    if goal_distance <= policy.goal_position_tolerance_m and _angle_error(goal_yaw, yaw) <= policy.goal_yaw_tolerance_rad:
        return TrackingCommand(0.0, 0.0, True, None)

    distances = [math.hypot(point[0] - x, point[1] - y) for point in points]
    nearest_index = min(range(len(points)), key=distances.__getitem__)
    if distances[nearest_index] > policy.max_cross_track_error_m:
        return _stopped("PATH_DEVIATION")

    target_index = len(points) - 1
    for index in range(nearest_index + 1, len(points)):
        if math.hypot(points[index][0] - x, points[index][1] - y) >= policy.lookahead_m:
            target_index = index
            break
    target_x, target_y, _ = points[target_index]
    dx = target_x - x
    dy = target_y - y
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    lookahead = max(math.hypot(dx, dy), policy.lookahead_m)
    curvature = 2.0 * local_y / (lookahead * lookahead)
    linear = min(policy.max_linear_mps, max(0.0, goal_distance))
    angular = max(-policy.max_angular_radps, min(policy.max_angular_radps, linear * curvature))
    return TrackingCommand(linear, angular, False, None)
