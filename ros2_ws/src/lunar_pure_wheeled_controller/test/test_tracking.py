import math
from dataclasses import replace

import pytest

from lunar_pure_wheeled_controller.tracking import (
    TrackingCommand,
    TrackingPolicy,
    TrackingState,
    track_path,
    track_trajectory,
)
from lunar_pure_wheeled_controller.reference import TrajectorySample


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


def sample(
    x: float,
    y: float,
    yaw: float,
    speed: float,
    yaw_rate: float = 0.0,
) -> TrajectorySample:
    return TrajectorySample(
        x_m=x,
        y_m=y,
        yaw_rad=yaw,
        signed_speed_mps=speed,
        yaw_rate_radps=yaw_rate,
    )


def producer_spin_samples(
    direction: float,
    yaw_rate: float = 0.15,
) -> tuple[TrajectorySample, ...]:
    """Mirror the wheel producer's initial point plus eight spin samples."""
    return tuple(
        sample(
            0.0,
            0.0,
            direction * math.pi * index / 16.0,
            0.0,
            direction * yaw_rate if 0 < index < 8 else 0.0,
        )
        for index in range(9)
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

    assert 0.0 < command.linear_x_mps <= 0.2
    assert command.linear_x_mps <= policy().max_linear_accel_mps2 * .2


def test_reached_goal_requires_position_and_yaw_tolerances() -> None:
    assert track_path(((1.0, 0.0, 0.0),), state(0.9, 0.1, 0.0), policy()).complete

    not_aligned = track_path(((1.0, 0.0, 0.0),), state(0.9, 0.1, math.pi), policy())
    assert not not_aligned.complete


def test_path_at_terminal_position_spins_until_terminal_yaw_is_aligned() -> None:
    command = track_path(
        ((1.0, 0.0, math.pi / 2.0),),
        state(1.0, 0.0, 0.0),
        policy(),
    )

    assert not command.complete
    assert command.failure_reason is None
    assert command.linear_x_mps == 0.0
    assert 0.0 < command.angular_z_radps <= policy().max_angular_radps


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


def test_reverse_trajectory_commands_negative_linear_velocity() -> None:
    """A direction mutation that discards signed speed must fail this test."""
    reverse_samples = (
        sample(0.0, 0.0, 0.0, -0.2),
        sample(-1.0, 0.0, 0.0, -0.2),
    )

    result = track_trajectory(reverse_samples, state(0.0, 0.0, 0.0), policy(), 0)

    assert result.command.linear_x_mps < 0.0
    assert result.command.angular_z_radps == 0.0


def test_reverse_left_arc_commands_negative_angular_velocity() -> None:
    """A curvature mutation that ignores reverse linear sign must fail this test."""
    reverse_left_arc = (
        sample(0.0, 0.0, 0.0, -0.2),
        sample(-1.0, 1.0, 0.0, -0.2),
    )

    result = track_trajectory(reverse_left_arc, state(0.0, 0.0, 0.0), policy(), 0)

    assert result.command.linear_x_mps < 0.0
    assert result.command.angular_z_radps < 0.0


def test_in_place_left_spin_has_zero_linear_and_positive_angular_velocity() -> None:
    """A translation mutation for coincident yaw samples must fail this test."""
    spin_left = (
        sample(0.0, 0.0, 0.0, 0.2),
        sample(0.0, 0.0, math.pi / 2.0, 0.0),
    )

    spin = track_trajectory(spin_left, state(0.0, 0.0, 0.0), policy(), 0).command

    assert spin.linear_x_mps == 0.0
    assert spin.angular_z_radps > 0.0


def test_in_place_right_spin_has_zero_linear_and_negative_angular_velocity() -> None:
    """A yaw-sign mutation must fail this test."""
    spin_right = (
        sample(0.0, 0.0, 0.0, 0.2),
        sample(0.0, 0.0, -math.pi / 2.0, 0.0),
    )

    spin = track_trajectory(spin_right, state(0.0, 0.0, 0.0), policy(), 0).command

    assert spin.linear_x_mps == 0.0
    assert spin.angular_z_radps < 0.0


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_producer_shaped_dense_spin_is_not_skipped(direction: float) -> None:
    """Do not classify producer spins by arrival tolerance."""
    result = track_trajectory(
        producer_spin_samples(direction),
        state(0.0, 0.0, 0.0),
        policy(),
        0,
    )

    assert result.command.failure_reason is None
    assert result.command.linear_x_mps == 0.0
    assert math.isclose(
        result.command.angular_z_radps,
        direction * 0.15,
        abs_tol=1.0e-12,
    )


@pytest.mark.parametrize(
    "target_yaw, requested_yaw_rate",
    [
        (math.pi, 0.11),
        (-math.pi, -0.12),
    ],
)
def test_spin_yaw_rate_resolves_pi_direction_and_caps_command(
    target_yaw: float,
    requested_yaw_rate: float,
) -> None:
    """Ignoring trajectory yaw rate must fail direction or magnitude at pi."""
    result = track_trajectory(
        (
            sample(0.0, 0.0, 0.0, 0.0),
            sample(0.0, 0.0, target_yaw, 0.0, requested_yaw_rate),
        ),
        state(0.0, 0.0, 0.0),
        policy(),
        0,
    )

    assert result.command.linear_x_mps == 0.0
    assert math.isclose(
        result.command.angular_z_radps,
        requested_yaw_rate,
        abs_tol=1.0e-12,
    )


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_terminal_zero_yaw_rate_inherits_contiguous_spin_authority(
    direction: float,
) -> None:
    """Inherit the producer's spin limit at its zero-rate terminal."""
    requested_yaw_rate = direction * 0.13
    samples = (
        sample(0.0, 0.0, 0.0, 0.0),
        sample(0.0, 0.0, direction * math.pi / 4.0, 0.0, requested_yaw_rate),
        sample(0.0, 0.0, direction * math.pi / 2.0, 0.0, 0.0),
    )

    result = track_trajectory(
        samples,
        state(0.0, 0.0, direction * math.pi / 4.0),
        policy(),
        1,
    )

    assert result.command.linear_x_mps == 0.0
    assert math.isclose(
        result.command.angular_z_radps,
        requested_yaw_rate,
        abs_tol=1.0e-12,
    )


def test_trajectory_cursor_advances_past_reached_intermediate_samples() -> None:
    """A cursor mutation that retargets from the start must fail this test."""
    samples = (
        sample(0.0, 0.0, 0.0, 0.2),
        sample(0.1, 0.0, 0.0, 0.2),
        sample(0.2, 0.0, 0.0, 0.2),
        sample(1.0, 0.0, 0.0, 0.2),
    )

    result = track_trajectory(samples, state(0.2, 0.0, 0.0), policy(), 0)

    assert result.next_cursor == 2
    assert result.command.linear_x_mps > 0.0


def test_trajectory_completes_only_at_final_pose_and_yaw() -> None:
    """A completion mutation that ignores final yaw must fail this test."""
    samples = (
        sample(0.0, 0.0, 0.0, 0.2),
        sample(1.0, 0.0, math.pi / 2.0, 0.0),
    )

    result = track_trajectory(samples, state(1.0, 0.0, math.pi / 2.0), policy(), 0)
    not_aligned = track_trajectory(samples, state(1.0, 0.0, 0.0), policy(), 0)

    assert result.command == TrackingCommand(0.0, 0.0, True, None)
    assert not not_aligned.command.complete


def test_trajectory_does_not_complete_at_a_closed_loop_terminal_before_cursor_progress() -> None:
    """A terminal-pose check before ordered cursor progress must fail this test."""
    samples = (
        sample(0.0, 0.0, 0.0, 0.2),
        sample(1.0, 0.0, 0.0, 0.2),
        sample(0.0, 0.0, 0.0, 0.0),
    )

    result = track_trajectory(samples, state(0.0, 0.0, 0.0), policy(), 0)

    assert not result.command.complete
    assert result.next_cursor == 0
    assert result.command.linear_x_mps > 0.0


def test_short_translation_within_epsilon_and_aligned_yaw_is_not_a_spin() -> None:
    """An XY-only spin classification must fail this test."""
    samples = (
        sample(0.0, 0.0, 0.0, 0.2),
        sample(0.0005, 0.0, 0.1, 0.2),
        sample(1.0, 0.0, 0.1, 0.2),
    )

    result = track_trajectory(
        samples,
        state(0.0, 0.0, 0.0),
        replace(policy(), goal_position_tolerance_m=1.0e-4),
        0,
    )

    assert result.command.linear_x_mps > 0.0
    assert result.command.angular_z_radps == 0.0


def test_same_pose_same_yaw_stop_keeps_translation_continuity() -> None:
    """A same-pose stop must not truncate the enclosing translation segment."""
    samples = (
        sample(0.0, 0.0, 0.0, 0.2),
        sample(1.0, 0.0, 0.0, 0.2),
        sample(1.0, 0.0, 0.0, 0.0),
        sample(2.0, 0.0, 0.0, 0.2),
    )

    result = track_trajectory(samples, state(0.75, 0.0, 0.0), policy(), 1)

    assert result.command.linear_x_mps > 0.0
    assert result.command.angular_z_radps == 0.0


@pytest.mark.parametrize(
    "samples, approach_state, approach_sign, departure_sign",
    [
        (
            (
                sample(0.0, 0.0, 0.0, 0.2),
                sample(0.5, 0.0, 0.0, 0.2),
                sample(1.0, 0.0, 0.0, 0.0),
                sample(1.0, 0.0, 0.0, 0.0),
                sample(0.5, 0.0, 0.0, -0.2),
            ),
            state(0.75, 0.0, 0.0),
            1.0,
            -1.0,
        ),
        (
            (
                sample(2.0, 0.0, 0.0, -0.2),
                sample(1.5, 0.0, 0.0, -0.2),
                sample(1.0, 0.0, 0.0, 0.0),
                sample(1.0, 0.0, 0.0, 0.0),
                sample(1.5, 0.0, 0.0, 0.2),
            ),
            state(1.25, 0.0, 0.0),
            -1.0,
            1.0,
        ),
    ],
)
def test_repeated_zero_stop_keeps_prior_direction_until_cursor_advances(
    samples: tuple[TrajectorySample, ...],
    approach_state: TrackingState,
    approach_sign: float,
    departure_sign: float,
) -> None:
    """Do not switch direction before reaching a repeated zero stop."""
    approaching = track_trajectory(samples, approach_state, policy(), 1)
    departing = track_trajectory(samples, state(1.0, 0.0, 0.0), policy(), 1)

    assert (
        math.copysign(1.0, approaching.command.linear_x_mps)
        == approach_sign
    )
    assert approaching.next_cursor == 1
    assert math.copysign(1.0, departing.command.linear_x_mps) == departure_sign
    assert departing.next_cursor == 3


@pytest.mark.parametrize("cursor", [-1, 2])
def test_trajectory_rejects_cursor_outside_sample_bounds(cursor: int) -> None:
    """A cursor-bound mutation must fail this test."""
    samples = (
        sample(0.0, 0.0, 0.0, 0.2),
        sample(1.0, 0.0, 0.0, 0.2),
    )

    result = track_trajectory(samples, state(0.0, 0.0, 0.0), policy(), cursor)

    assert result.command == TrackingCommand(0.0, 0.0, False, "INVALID_INPUT")
    assert result.next_cursor == cursor


def test_trajectory_path_deviation_stops() -> None:
    """A path-deviation mutation must fail this test."""
    samples = (
        sample(0.0, 0.0, 0.0, 0.2),
        sample(3.0, 0.0, 0.0, 0.2),
    )

    result = track_trajectory(samples, state(0.0, 1.1, 0.0), policy(), 0)

    assert result.command == TrackingCommand(0.0, 0.0, False, "PATH_DEVIATION")


def test_trajectory_positive_and_negative_linear_commands_are_bounded() -> None:
    """A trajectory linear clamp mutation must fail this test."""
    positive = track_trajectory(
        (sample(0.0, 0.0, 0.0, 0.2), sample(10.0, 0.0, 0.0, 0.2)),
        state(0.0, 0.0, 0.0),
        replace(policy(), max_linear_mps=0.1),
        0,
    ).command
    negative = track_trajectory(
        (sample(0.0, 0.0, 0.0, -0.2), sample(-10.0, 0.0, 0.0, -0.2)),
        state(0.0, 0.0, 0.0),
        replace(policy(), max_linear_mps=0.1),
        0,
    ).command

    assert 0.0 < positive.linear_x_mps <= 0.1
    assert -0.1 <= negative.linear_x_mps < 0.0


def test_trajectory_translation_angular_command_is_bounded() -> None:
    """A trajectory curvature clamp mutation must fail this test."""
    command = track_trajectory(
        (sample(0.0, 0.0, 0.0, 0.2), sample(1.0, 1.0, 0.0, 0.2)),
        state(0.0, 0.0, 0.0),
        replace(policy(), max_angular_radps=0.1),
        0,
    ).command

    assert 0.0 < command.angular_z_radps <= 0.1


def test_trajectory_positive_and_negative_spin_commands_are_bounded() -> None:
    """A trajectory spin clamp mutation must fail this test."""
    positive = track_trajectory(
        (sample(0.0, 0.0, 0.0, 0.2), sample(0.0, 0.0, math.pi / 2.0, 0.0)),
        state(0.0, 0.0, 0.0),
        replace(policy(), max_angular_radps=0.1),
        0,
    ).command
    negative = track_trajectory(
        (sample(0.0, 0.0, 0.0, 0.2), sample(0.0, 0.0, -math.pi / 2.0, 0.0)),
        state(0.0, 0.0, 0.0),
        replace(policy(), max_angular_radps=0.1),
        0,
    ).command

    assert positive.linear_x_mps == 0.0
    assert positive.angular_z_radps == 0.1
    assert negative.linear_x_mps == 0.0
    assert negative.angular_z_radps == -0.1


def test_trajectory_command_has_no_lateral_velocity() -> None:
    """Adding a lateral wheel command field must fail this safety-boundary test."""
    command = track_trajectory(
        (sample(0.0, 0.0, 0.0, 0.2), sample(1.0, 0.0, 0.0, 0.2)),
        state(0.0, 0.0, 0.0),
        policy(),
        0,
    ).command

    assert not hasattr(command, "linear_y_mps")


def test_zero_speed_terminal_translation_keeps_reverse_direction_after_spin() -> None:
    """A zero-speed lookup crossing a spin into the forward segment must fail this test."""
    samples = (
        sample(0.0, 0.0, 0.0, 0.2),
        sample(1.0, 0.0, 0.0, 0.2),
        sample(1.0, 0.0, math.pi / 2.0, 0.0),
        sample(1.0, -1.0, math.pi / 2.0, -0.2),
        sample(1.0, -2.0, math.pi / 2.0, 0.0),
    )

    spin = track_trajectory(samples, state(1.0, 0.0, 0.0), policy(), 0)
    reverse = track_trajectory(samples, state(1.0, 0.0, math.pi / 2.0), policy(), spin.next_cursor)
    terminal_translation = track_trajectory(
        samples,
        state(1.0, -1.0, math.pi / 2.0),
        policy(),
        reverse.next_cursor,
    )

    assert spin.command.linear_x_mps == 0.0
    assert spin.command.angular_z_radps > 0.0
    assert reverse.command.linear_x_mps < 0.0
    assert terminal_translation.command.linear_x_mps < 0.0
