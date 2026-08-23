"""Deterministic, dependency-free pure-pursuit tracking."""

from dataclasses import dataclass
import math
from typing import TypeAlias

from lunar_pure_wheeled_controller.reference import TrajectorySample


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
    spin_kp: float = 1.5
    translation_epsilon_m: float = 1.0e-3


@dataclass(frozen=True)
class TrackingCommand:
    linear_x_mps: float
    angular_z_radps: float
    complete: bool
    failure_reason: str | None


@dataclass(frozen=True)
class TrajectoryTrackingResult:
    command: TrackingCommand
    next_cursor: int


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
            policy.spin_kp,
            policy.translation_epsilon_m,
        )
        return all(math.isfinite(float(value)) and float(value) > 0.0 for value in values)
    except (AttributeError, TypeError, ValueError):
        return False


def _angle_error(target: float, current: float) -> float:
    return abs((target - current + math.pi) % (2.0 * math.pi) - math.pi)


def _normalized_yaw_error(target: float, current: float) -> float:
    return (target - current + math.pi) % (2.0 * math.pi) - math.pi


def _finite_samples(samples: tuple[TrajectorySample, ...]) -> tuple[TrajectorySample, ...] | None:
    try:
        finite_samples = tuple(samples)
        if not finite_samples:
            return None
        for sample in finite_samples:
            if not all(math.isfinite(float(value)) for value in (
                sample.x_m,
                sample.y_m,
                sample.yaw_rad,
                sample.signed_speed_mps,
                sample.yaw_rate_radps,
            )):
                return None
    except (AttributeError, TypeError, ValueError):
        return None
    return finite_samples


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _is_spin_target(
    samples: tuple[TrajectorySample, ...],
    target_index: int,
    policy: TrackingPolicy,
) -> bool:
    predecessor = samples[target_index - 1]
    target = samples[target_index]
    return (
        math.hypot(target.x_m - predecessor.x_m, target.y_m - predecessor.y_m)
        <= policy.translation_epsilon_m
        and _angle_error(target.yaw_rad, predecessor.yaw_rad) > policy.goal_yaw_tolerance_rad
    )


def _sample_reached(
    samples: tuple[TrajectorySample, ...],
    sample_index: int,
    state: TrackingState,
    policy: TrackingPolicy,
) -> bool:
    target = samples[sample_index]
    position_error = math.hypot(target.x_m - state.x_m, target.y_m - state.y_m)
    if position_error > policy.goal_position_tolerance_m:
        return False
    if _is_spin_target(samples, sample_index, policy):
        return _angle_error(target.yaw_rad, state.yaw_rad) <= policy.goal_yaw_tolerance_rad
    return True


def _translation_sign(
    samples: tuple[TrajectorySample, ...],
    target_index: int,
    policy: TrackingPolicy,
) -> float | None:
    speed = samples[target_index].signed_speed_mps
    if speed != 0.0:
        return math.copysign(1.0, speed)

    for index in range(target_index + 1, len(samples)):
        if _is_spin_target(samples, index, policy):
            break
        speed = samples[index].signed_speed_mps
        if speed != 0.0:
            return math.copysign(1.0, speed)

    for index in range(target_index - 1, -1, -1):
        if index > 0 and _is_spin_target(samples, index, policy):
            break
        speed = samples[index].signed_speed_mps
        if speed != 0.0:
            return math.copysign(1.0, speed)
    return None


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


def track_trajectory(
    samples: tuple[TrajectorySample, ...],
    state: TrackingState,
    policy: TrackingPolicy,
    cursor: int,
) -> TrajectoryTrackingResult:
    """Track an ordered trajectory with reverse translation and in-place spins."""
    finite_samples = _finite_samples(samples)
    if (
        finite_samples is None
        or not _finite_state(state)
        or not _valid_policy(policy)
        or not isinstance(cursor, int)
        or isinstance(cursor, bool)
        or cursor < 0
        or cursor >= len(finite_samples)
    ):
        return TrajectoryTrackingResult(_stopped("INVALID_INPUT"), cursor)

    x = float(state.x_m)
    y = float(state.y_m)
    yaw = float(state.yaw_rad)
    if min(math.hypot(sample.x_m - x, sample.y_m - y) for sample in finite_samples) > policy.max_cross_track_error_m:
        return TrajectoryTrackingResult(_stopped("PATH_DEVIATION"), cursor)

    next_cursor = cursor
    while next_cursor + 1 < len(finite_samples) and _sample_reached(
        finite_samples,
        next_cursor + 1,
        state,
        policy,
    ):
        next_cursor += 1

    if next_cursor + 1 >= len(finite_samples):
        final_sample = finite_samples[-1]
        final_distance = math.hypot(final_sample.x_m - x, final_sample.y_m - y)
        if (
            final_distance <= policy.goal_position_tolerance_m
            and _angle_error(final_sample.yaw_rad, yaw) <= policy.goal_yaw_tolerance_rad
        ):
            return TrajectoryTrackingResult(TrackingCommand(0.0, 0.0, True, None), next_cursor)
        return TrajectoryTrackingResult(_stopped("INVALID_INPUT"), next_cursor)

    target_index = next_cursor + 1
    target = finite_samples[target_index]
    if (
        _is_spin_target(finite_samples, target_index, policy)
        and _angle_error(target.yaw_rad, yaw) > policy.goal_yaw_tolerance_rad
    ):
        angular = _clip(
            policy.spin_kp * _normalized_yaw_error(target.yaw_rad, yaw),
            policy.max_angular_radps,
        )
        return TrajectoryTrackingResult(TrackingCommand(0.0, angular, False, None), next_cursor)

    direction = _translation_sign(finite_samples, target_index, policy)
    if direction is None:
        return TrajectoryTrackingResult(_stopped("INVALID_INPUT"), next_cursor)

    dx = target.x_m - x
    dy = target.y_m - y
    remaining_distance = math.hypot(dx, dy)
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    lookahead = max(remaining_distance, policy.lookahead_m)
    linear = direction * min(policy.max_linear_mps, remaining_distance)
    angular = _clip(linear * 2.0 * local_y / (lookahead * lookahead), policy.max_angular_radps)
    return TrajectoryTrackingResult(TrackingCommand(linear, angular, False, None), next_cursor)
