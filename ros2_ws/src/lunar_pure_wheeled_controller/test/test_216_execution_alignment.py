"""Transport-neutral execution regressions preserved by the 216 deployment."""
from dataclasses import replace
from lunar_pure_wheeled_controller.execution import PathExecutor
from lunar_pure_wheeled_controller.tracking import TrackingPolicy, TrackingState


def test_rotation_stops_when_measured_response_exceeds_command():
    policy = replace(TrackingPolicy(), max_angular_radps=.15, max_angular_accel_radps2=.1)
    executor = PathExecutor(policy)
    executor.set_path(((0, 0, 0), (0, 0, .5235987756)))
    yaw = rate = previous = 0.
    for _ in range(400):
        result = executor.update(TrackingState(0, 0, yaw, angular_radps=rate), .05)
        command = result.command.angular_z_radps
        assert abs(command - previous) <= .005 + 1e-9 or result.phase in ('BRAKING', 'COMPLETED', 'FAILED')
        previous = command
        rate = 1.2 * command
        yaw += rate * .05
        if result.phase == 'COMPLETED':
            break
    assert result.phase == 'COMPLETED'


def test_lateral_motion_prevents_actual_executor_completion():
    executor = PathExecutor(TrackingPolicy())
    executor.set_path(((0, 0, 0), (1, 0, 0)))
    result = executor.update(TrackingState(1, 0, 0, lateral_mps=.2), .05)
    assert not result.command.complete


def test_feedback_spike_does_not_reset_command_ramp():
    executor = PathExecutor(TrackingPolicy())
    executor.set_path(((0, 0, 0), (10, 0, 0)))
    previous = executor.update(TrackingState(0, 0, 0), .05).command.linear_x_mps
    for measured in (.3, .01, .25, .0, .3):
        result = executor.update(TrackingState(.1, 0, 0, measured, .02), .05)
        assert result.phase == 'TRACKING'
        assert 0 < result.command.linear_x_mps <= .2
        assert abs(result.command.linear_x_mps - previous) <= .025 + 1e-9
        previous = result.command.linear_x_mps
