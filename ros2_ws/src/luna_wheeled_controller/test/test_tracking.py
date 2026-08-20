"""Behavior tests for the WHEELED pure-pursuit tracking core.

Each test protects one control invariant: a forward straight reference must
drive forward without steering, a reached terminal pose must stop, invalid
geometry must never create motion, and cross-track departure must stop.
"""

from __future__ import annotations

from dataclasses import replace

from luna_wheeled_controller.tracking import (
    TrackingPolicy,
    TrackingState,
    track_path,
    validate_tracking_policy,
)


def _policy() -> TrackingPolicy:
    return TrackingPolicy(
        lookahead_m=1.0,
        max_linear_mps=0.2,
        max_angular_radps=0.5,
        max_cross_track_error_m=1.0,
        goal_position_tolerance_m=0.25,
        goal_yaw_tolerance_rad=0.35,
    )


def test_straight_path_commands_bounded_forward_motion_without_steering() -> None:
    result = track_path(
        ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
        TrackingState(x_m=0.0, y_m=0.0, yaw_rad=0.0),
        _policy(),
    )

    assert result.failure_reason is None
    assert 0.0 < result.linear_x_mps <= 0.2
    assert result.angular_z_radps == 0.0
    assert not result.complete


def test_goal_within_position_and_yaw_tolerances_completes_without_motion() -> None:
    result = track_path(
        ((1.0, 0.0, 0.0),),
        TrackingState(x_m=0.9, y_m=0.1, yaw_rad=0.1),
        _policy(),
    )

    assert result.failure_reason is None
    assert result.complete
    assert (result.linear_x_mps, result.angular_z_radps) == (0.0, 0.0)


def test_cross_track_departure_stops_without_commanding_a_recovery_turn() -> None:
    result = track_path(
        ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
        TrackingState(x_m=0.0, y_m=1.01, yaw_rad=0.0),
        _policy(),
    )

    assert result.failure_reason == "PATH_DEVIATION"
    assert not result.complete
    assert (result.linear_x_mps, result.angular_z_radps) == (0.0, 0.0)


def test_nonfinite_reference_is_rejected_without_motion() -> None:
    result = track_path(
        ((0.0, 0.0, 0.0), (float("nan"), 1.0, 0.0)),
        TrackingState(x_m=0.0, y_m=0.0, yaw_rad=0.0),
        _policy(),
    )

    assert result.failure_reason == "INVALID_REFERENCE"
    assert (result.linear_x_mps, result.angular_z_radps) == (0.0, 0.0)


def test_nonpositive_lookahead_is_rejected_before_control() -> None:
    assert validate_tracking_policy(replace(_policy(), lookahead_m=0.0)) == "LOOKAHEAD_INVALID"
