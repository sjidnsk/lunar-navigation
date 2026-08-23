import math
from dataclasses import replace

import pytest

from lunar_pure_wheeled_controller.tracking import (
    TrackingPolicy,
    TrackingState,
    track_path,
)


def state(x: float, y: float, yaw: float) -> TrackingState:
    return TrackingState(x_m=x, y_m=y, yaw_rad=yaw)


def policy() -> TrackingPolicy:
    return TrackingPolicy(
        lookahead_m=0.5,
        max_linear_mps=0.2,
        max_angular_radps=0.5,
        goal_position_tolerance_m=0.2,
        goal_yaw_tolerance_rad=0.2,
        max_cross_track_error_m=1.0,
    )


def test_straight_path_moves_forward_without_steering() -> None:
    command = track_path(
        ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
        state(0.0, 0.0, 0.0),
        policy(),
    )

    assert 0.0 < command.linear_x_mps <= 0.2
    assert command.angular_z_radps == 0.0
    assert not command.complete
    assert command.failure_reason is None


def test_left_turn_has_positive_bounded_angular_velocity() -> None:
    command = track_path(
        ((0.0, 0.0, 0.0), (1.0, 1.0, 0.8)),
        state(0.0, 0.0, 0.0),
        policy(),
    )

    assert 0.0 < command.angular_z_radps <= 0.5


def test_right_turn_has_negative_bounded_angular_velocity() -> None:
    command = track_path(
        ((0.0, 0.0, 0.0), (1.0, -1.0, -0.8)),
        state(0.0, 0.0, 0.0),
        policy(),
    )

    assert -0.5 <= command.angular_z_radps < 0.0


def test_speed_is_clamped_to_policy_limit() -> None:
    command = track_path(
        ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
        state(0.0, 0.0, 0.0),
        policy(),
    )

    assert command.linear_x_mps == 0.2


def test_reached_goal_requires_position_and_yaw_tolerances() -> None:
    assert track_path(((1.0, 0.0, 0.0),), state(0.9, 0.1, 0.0), policy()).complete

    not_aligned = track_path(((1.0, 0.0, 0.0),), state(0.9, 0.1, math.pi), policy())
    assert not not_aligned.complete


def test_large_cross_track_error_stops() -> None:
    command = track_path(
        ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
        state(0.0, 1.1, 0.0),
        policy(),
    )

    assert command == type(command)(0.0, 0.0, False, "PATH_DEVIATION")


@pytest.mark.parametrize(
    "path, current_state, reason",
    [
        (((0.0, 0.0, 0.0), (math.nan, 0.0, 0.0)), state(0.0, 0.0, 0.0), "INVALID_INPUT"),
        (((0.0, 0.0, 0.0),), state(math.inf, 0.0, 0.0), "INVALID_INPUT"),
        (((0.0, 0.0, 0.0),), state(0.0, 0.0, math.nan), "INVALID_INPUT"),
    ],
)
def test_non_finite_input_is_rejected(path, current_state, reason) -> None:
    command = track_path(path, current_state, policy())

    assert command == type(command)(0.0, 0.0, False, reason)


def test_path_with_no_points_is_rejected() -> None:
    command = track_path((), state(0.0, 0.0, 0.0), policy())

    assert command.failure_reason == "INVALID_INPUT"


def test_tail_without_a_lookahead_point_steers_toward_terminal_goal() -> None:
    """A fallback mutation that uses the nearest point instead of the goal must fail this test."""
    command = track_path(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.1, 0.1, 0.0)),
        state(1.0, 0.0, 0.0),
        replace(policy(), goal_position_tolerance_m=0.05),
    )

    assert not command.complete
    assert command.angular_z_radps > 0.0


@pytest.mark.parametrize(
    "invalid_policy",
    [
        replace(policy(), lookahead_m=0.0),
        replace(policy(), max_linear_mps=0.0),
        replace(policy(), max_angular_radps=0.0),
        replace(policy(), goal_position_tolerance_m=0.0),
        replace(policy(), goal_yaw_tolerance_rad=0.0),
        replace(policy(), max_cross_track_error_m=0.0),
    ],
)
def test_zero_tracking_policy_values_are_rejected(invalid_policy: TrackingPolicy) -> None:
    """A validation mutation that permits a zero policy bound must fail this test."""
    command = track_path(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), state(0.0, 0.0, 0.0), invalid_policy)

    assert command == type(command)(0.0, 0.0, False, "INVALID_INPUT")
