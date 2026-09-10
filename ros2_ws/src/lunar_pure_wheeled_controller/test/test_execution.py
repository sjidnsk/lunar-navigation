import math
from dataclasses import replace
import pytest
from lunar_pure_wheeled_controller.execution import PathExecutor
from lunar_pure_wheeled_controller.path_geometry import Polyline
from lunar_pure_wheeled_controller.tracking import TrackingPolicy, TrackingState


def test_long_segment_midpoint_is_on_path():
    line = Polyline(((0, 0, 0), (3, 0, 0)))
    assert line.project(1.5, 0).error == 0
    assert line.point_at(2.0) == pytest.approx((2.0, 0.0))
    executor = PathExecutor(TrackingPolicy())
    executor.set_path(((0, 0, 0), (3, 0, 0)))
    assert (
        executor.update(TrackingState(1.5, 0, 0), 0.05).command.failure_reason is None
    )


def test_large_initial_heading_error_rotates_before_translation():
    executor = PathExecutor(TrackingPolicy())
    executor.set_path(((0, 0, -1.85), (3, 0, 0)))
    result = executor.update(TrackingState(0, 0, -1.85), 0.05)
    assert result.phase == "ALIGNING"
    assert result.command.linear_x_mps == 0
    assert result.command.angular_z_radps > 0


def test_reverse_direction_is_latched_and_curvature_is_bounded():
    executor = PathExecutor(TrackingPolicy())
    executor.set_path(((0, 0, 0), (-3, 0, 0)))
    result = executor.update(TrackingState(0, 0, 0), 0.05)
    assert result.direction == -1
    assert result.command.linear_x_mps < 0
    assert abs(result.command.linear_x_mps) <= 0.2


def test_completion_waits_for_measured_stop():
    executor = PathExecutor(TrackingPolicy())
    executor.set_path(((0, 0, 0), (1, 0, 0)))
    result = executor.update(TrackingState(1, 0, 0, linear_mps=0.2), 0.05)
    assert not result.command.complete
    result = executor.update(TrackingState(1, 0, 0), 0.05)
    assert result.command.complete


def test_old_revision_and_duplicate_do_not_reset_execution():
    executor = PathExecutor(TrackingPolicy())
    path = ((0, 0, 0), (3, 0, 0))
    assert executor.set_path(path, identity=("a", 2))
    executor.update(TrackingState(1, 0, 0), 0.05)
    assert not executor.set_path(path, identity=("a", 1))
    assert not executor.set_path(path, identity=("a", 2))
    assert executor.progress_m == pytest.approx(1.0)


def test_corner_target_does_not_cut_across_right_angle():
    executor = PathExecutor(TrackingPolicy())
    executor.set_path(((0, 0, 0), (2, 0, math.pi / 2), (2, 2, math.pi / 2)))
    result = executor.update(TrackingState(1.7, 0, 0), 0.05)
    assert result.command.angular_z_radps == 0
    result = executor.update(TrackingState(2, 0, 0), 0.05)
    assert result.command.linear_x_mps == 0
    assert result.command.angular_z_radps > 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_reverse_mps", -1.0),
        ("max_linear_accel_mps2", float("nan")),
        ("max_curvature_per_m", 0.0),
        ("no_progress_timeout_s", -1.0),
        ("allow_reverse", 1),
    ],
)
def test_invalid_execution_limits_rejected(field, value):
    with pytest.raises(ValueError):
        PathExecutor(replace(TrackingPolicy(), **{field: value}))


def test_final_alignment_stall_requests_failure_only_after_stop():
    executor = PathExecutor(replace(TrackingPolicy(), no_progress_timeout_s=0.3))
    executor.set_path(((0, 0, 0), (0, 0, 1.0)))
    for _ in range(10):
        result = executor.update(TrackingState(0, 0, 0), 0.1)
    assert result.phase == "FAILED"
    assert result.command.failure_reason == "ALIGNMENT_STALLED"


def test_deviation_waits_for_measured_stop_before_failure():
    executor = PathExecutor(TrackingPolicy())
    executor.set_path(((0, 0, 0), (3, 0, 0)))
    assert executor.update(TrackingState(1, 2, 0, 0.2), 0.05).phase == "BRAKING"
    assert executor.update(TrackingState(1, 2, 0), 0.05).phase == "FAILED"


@pytest.mark.parametrize("curvature", [-1.0, -0.1, 0.0, 0.1, 1.0])
@pytest.mark.parametrize("yaw_rate", [-0.2, -0.01, 0.0, 0.01, 0.2])
def test_joint_speed_limits_preserve_curvature_and_slew(curvature, yaw_rate):
    from lunar_pure_wheeled_controller.limits import tracking_speed

    p = TrackingPolicy()
    speed = tracking_speed(curvature, 0.2, 0.1, yaw_rate, 0.05, p)
    if speed is not None:
        assert abs(speed - 0.1) <= 0.025 + 1e-9
        assert abs(speed * curvature - yaw_rate) <= 0.025 + 1e-9
        assert 0 <= speed <= 0.2


def test_non_final_reference_still_executes_its_terminal_heading():
    executor=PathExecutor(TrackingPolicy())
    executor.set_path(((0,0,0),(0,0,1.)),final=False)
    result=executor.update(TrackingState(0,0,0),.05)
    assert result.phase == 'FINAL_ALIGN'
    assert result.command.angular_z_radps > 0.


@pytest.mark.parametrize("final", [True, False])
@pytest.mark.parametrize("yaw_rate", [0.0, 0.1])
def test_final_alignment_losing_endpoint_position_brakes_then_fails(final, yaw_rate):
    executor = PathExecutor(TrackingPolicy())
    executor.set_path(((0, 0, 0), (1, 0, 1)), final=final)
    assert executor.update(TrackingState(1, 0, 0), 0.05).phase == "FINAL_ALIGN"
    # An along-path localization correction has zero cross-track error but
    # invalidates the endpoint acceptance used to enter final alignment.
    result = executor.update(TrackingState(0.5, 0, 1, 0, yaw_rate), 0.05)
    assert result.phase == "BRAKING"
    assert not result.command.complete
    assert result.command.linear_x_mps == 0
    assert result.command.angular_z_radps == 0
    result = executor.update(TrackingState(0.5, 0, 1), 0.05)
    assert result.phase == "FAILED"
    assert result.command.failure_reason == "GOAL_POSITION_LOST"
