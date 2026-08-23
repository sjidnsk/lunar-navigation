"""ROS-boundary behavior for the isolated pure wheeled controller."""

from __future__ import annotations

import time
import math

import pytest
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from lunar_planning_msgs.msg import MotionReference
from nav_msgs.msg import Odometry, Path
from rclpy.executors import SingleThreadedExecutor

from lunar_pure_wheeled_controller.node import PureWheeledControllerNode, _yaw


def make_reference(*, goal_x: float = 2.0) -> MotionReference:
    reference = MotionReference()
    reference.plan_id = "wheel-plan"
    reference.platform_type = MotionReference.WHEELED
    reference.path_preview = Path()
    for x in (0.0, goal_x):
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.orientation.w = 1.0
        reference.path_preview.poses.append(pose)
    return reference


def make_odometry(*, x: float, y: float = 0.0, yaw: float = 0.0) -> Odometry:
    odometry = Odometry()
    odometry.pose.pose.position.x = x
    odometry.pose.pose.position.y = y
    odometry.pose.pose.orientation.z = math.sin(yaw / 2.0)
    odometry.pose.pose.orientation.w = math.cos(yaw / 2.0)
    return odometry


def wait_for_twists(
    controller: PureWheeledControllerNode,
    observer: rclpy.node.Node,
    received: list[Twist],
    *,
    count: int = 1,
) -> None:
    deadline = time.monotonic() + 1.0
    while len(received) < count and time.monotonic() < deadline:
        rclpy.spin_once(controller, timeout_sec=0.01)
        rclpy.spin_once(observer, timeout_sec=0.01)
    assert len(received) >= count


def wait_for_observed_twists(observer: rclpy.node.Node, received: list[Twist], count: int) -> None:
    deadline = time.monotonic() + 1.0
    while len(received) < count and time.monotonic() < deadline:
        rclpy.spin_once(observer, timeout_sec=0.01)
    assert len(received) >= count


@pytest.fixture
def controller_with_observer():
    rclpy.init()
    controller = PureWheeledControllerNode()
    observer = rclpy.create_node("pure_wheeled_controller_command_observer")
    received: list[Twist] = []
    observer.create_subscription(Twist, "/Car/T5/Car_Cmd_Vel", received.append, 10)
    controller_alive = True

    def destroy_controller() -> None:
        nonlocal controller_alive
        if controller_alive:
            controller.destroy_node()
            controller_alive = False

    try:
        yield controller, observer, received, destroy_controller
    finally:
        observer.destroy_node()
        destroy_controller()
        rclpy.shutdown()


def test_reference_and_odometry_publish_forward_twist(controller_with_observer) -> None:
    """A tracker mutation that drops positive linear output must fail this test."""
    controller, observer, received, _ = controller_with_observer

    controller._on_reference(make_reference())
    controller._on_odometry(make_odometry(x=0.0))
    controller._tick()
    wait_for_twists(controller, observer, received)

    assert received[-1].linear.x > 0.0
    assert received[-1].angular.z == 0.0


def test_invalid_replacement_publishes_zero_and_clears_active_reference(controller_with_observer) -> None:
    """A mutation that retains a prior path after invalid input must fail this test."""
    controller, observer, received, _ = controller_with_observer

    controller._on_reference(make_reference())
    controller._on_odometry(make_odometry(x=0.0))
    controller._on_reference(MotionReference())
    wait_for_twists(controller, observer, received)
    controller._tick()
    wait_for_twists(controller, observer, received, count=2)

    assert received[-2].linear.x == 0.0
    assert received[-2].angular.z == 0.0
    assert received[-1].linear.x == 0.0
    assert received[-1].angular.z == 0.0


def test_goal_completion_publishes_zero_and_clears_active_reference(controller_with_observer) -> None:
    """A mutation that keeps commanding after a reached goal must fail this test."""
    controller, observer, received, _ = controller_with_observer

    controller._on_reference(make_reference(goal_x=0.1))
    controller._on_odometry(make_odometry(x=0.0))
    controller._tick()
    wait_for_twists(controller, observer, received)
    controller._tick()
    wait_for_twists(controller, observer, received, count=2)

    assert received[-2].linear.x == 0.0
    assert received[-2].angular.z == 0.0
    assert received[-1].linear.x == 0.0
    assert received[-1].angular.z == 0.0


def test_path_deviation_publishes_zero_and_clears_active_reference(controller_with_observer) -> None:
    """A mutation that drives despite excessive cross-track error must fail this test."""
    controller, observer, received, _ = controller_with_observer

    controller._on_reference(make_reference())
    controller._on_odometry(make_odometry(x=0.0, y=1.1))
    controller._tick()
    wait_for_twists(controller, observer, received)
    controller._tick()
    wait_for_twists(controller, observer, received, count=2)

    assert received[-2].linear.x == 0.0
    assert received[-2].angular.z == 0.0
    assert received[-1].linear.x == 0.0
    assert received[-1].angular.z == 0.0


def test_destroy_node_publishes_zero_twist(controller_with_observer) -> None:
    """A shutdown mutation that omits its stop command must fail this test."""
    controller, observer, received, destroy_controller = controller_with_observer

    controller._publish_twist(linear=0.1, angular=0.2)
    destroy_controller()
    wait_for_observed_twists(observer, received, count=2)

    assert received[-1].linear.x == 0.0
    assert received[-1].angular.z == 0.0


def test_zero_quaternion_odometry_publishes_zero_and_clears_active_odometry(controller_with_observer) -> None:
    """An odometry-input mutation that accepts a zero quaternion must fail this test."""
    controller, observer, received, _ = controller_with_observer
    invalid_odometry = make_odometry(x=0.0)
    invalid_odometry.pose.pose.orientation.x = 0.0
    invalid_odometry.pose.pose.orientation.y = 0.0
    invalid_odometry.pose.pose.orientation.z = 0.0
    invalid_odometry.pose.pose.orientation.w = 0.0

    controller._on_reference(make_reference())
    controller._on_odometry(invalid_odometry)
    wait_for_twists(controller, observer, received)
    controller._tick()
    wait_for_twists(controller, observer, received, count=2)

    assert received[-2].linear.x == 0.0
    assert received[-2].angular.z == 0.0
    assert received[-1].linear.x == 0.0
    assert received[-1].angular.z == 0.0


def test_default_topic_publishers_drive_bounded_twist() -> None:
    """A topic/remapping mutation that disconnects T4/T3 from T5 must fail this test."""
    rclpy.init()
    controller = PureWheeledControllerNode()
    inputs = rclpy.create_node("pure_wheeled_controller_integration_inputs")
    observer = rclpy.create_node("pure_wheeled_controller_integration_observer")
    executor = SingleThreadedExecutor()
    reference_publisher = inputs.create_publisher(
        MotionReference, "/Car/T4/planning/wheeled_reference", 10
    )
    odometry_publisher = inputs.create_publisher(Odometry, "/Car/T3/localization/odometry", 10)
    received: list[Twist] = []
    observer.create_subscription(Twist, "/Car/T5/Car_Cmd_Vel", received.append, 10)
    for node in (controller, inputs, observer):
        executor.add_node(node)
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not any(command.linear.x > 0.0 for command in received):
            reference_publisher.publish(make_reference())
            odometry_publisher.publish(make_odometry(x=0.0))
            executor.spin_once(timeout_sec=0.01)

        command = next(command for command in received if command.linear.x > 0.0)
        assert 0.0 < command.linear.x <= 0.2
        assert abs(command.angular.z) <= 0.5
    finally:
        executor.shutdown()
        observer.destroy_node()
        inputs.destroy_node()
        controller.destroy_node()
        rclpy.shutdown()


def test_non_positive_control_rate_is_rejected() -> None:
    """A constructor mutation that accepts a zero control rate must fail this test."""
    rclpy.init(args=["--ros-args", "-p", "control_rate_hz:=0.0"])
    try:
        with pytest.raises(ValueError, match="control_rate_hz"):
            PureWheeledControllerNode()
    finally:
        rclpy.shutdown()


def test_odometry_yaw_normalizes_a_non_unit_quaternion() -> None:
    """A yaw-extraction mutation that skips quaternion normalization must fail this test."""
    odometry = make_odometry(x=0.0, yaw=math.pi / 2.0)
    odometry.pose.pose.orientation.z *= 4.0
    odometry.pose.pose.orientation.w *= 4.0

    assert math.isclose(_yaw(odometry), math.pi / 2.0, abs_tol=1e-12)


def test_node_uses_spec_goal_tolerance_parameter_names() -> None:
    """A public-parameter mutation that restores legacy tolerance names must fail this test."""
    rclpy.init(args=[
        "--ros-args",
        "-p", "goal_position_tolerance_m:=0.31",
        "-p", "goal_yaw_tolerance_rad:=0.41",
    ])
    node = None
    try:
        node = PureWheeledControllerNode()
        assert node.get_parameter("goal_position_tolerance_m").value == 0.31
        assert node.get_parameter("goal_yaw_tolerance_rad").value == 0.41
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize(
    "parameter_override, parameter_name",
    [
        ("reference_topic:=relative_reference", "reference_topic"),
        ("odometry_topic:=relative_odometry", "odometry_topic"),
        ("command_topic:=relative_command", "command_topic"),
    ],
)
def test_relative_topic_parameter_is_rejected(parameter_override: str, parameter_name: str) -> None:
    """A topic-validation mutation that accepts relative public topics must fail this test."""
    rclpy.init(args=["--ros-args", "-p", parameter_override])
    node = None
    try:
        with pytest.raises(ValueError, match=parameter_name):
            node = PureWheeledControllerNode()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
