"""Fail-closed ROS adaptation of WHEELED Pure Pursuit tracking."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2

from geometry_msgs.msg import Twist
from lunar_navigation_msgs.msg import MotionExecutionFeedback
from lunar_planning_msgs.msg import MotionReference
from nav_msgs.msg import Odometry
from rclpy.node import Node

from .reference_protocol import make_feedback_identity, parse_wheeled_reference
from .tracking import TrackingPolicy, TrackingState, track_path, validate_tracking_policy


@dataclass(frozen=True)
class _ActiveReference:
    plan_id: str
    path_xy_yaw: tuple[tuple[float, float, float], ...]
    received_ns: int


def _yaw(odometry: Odometry) -> float:
    q = odometry.pose.pose.orientation
    return atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class WheeledControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("luna_wheeled_controller")
        defaults = {
            "reference_topic": "/execution/wheeled_reference",
            "odometry_topic": "/Car/T3/semantic/current_pose",
            "command_topic": "/Car/T5/Car_Cmd_Vel",
            "feedback_topic": "/execution/motion_feedback",
            "control_rate_hz": 20.0,
            "lookahead_m": 1.0,
            "max_linear_mps": 0.2,
            "max_angular_radps": 0.5,
            "max_cross_track_error_m": 1.0,
            "goal_position_tolerance_m": 0.25,
            "goal_yaw_tolerance_rad": 0.35,
            "reference_max_age_s": 1.0,
            "odometry_max_age_s": 0.5,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self._policy = TrackingPolicy(**{
            key: float(self.get_parameter(key).value)
            for key in (
                "lookahead_m", "max_linear_mps", "max_angular_radps",
                "max_cross_track_error_m", "goal_position_tolerance_m", "goal_yaw_tolerance_rad",
            )
        })
        if validate_tracking_policy(self._policy) is not None:
            raise ValueError("invalid wheeled controller tracking policy")
        self._reference_max_age_ns = int(float(self.get_parameter("reference_max_age_s").value) * 1_000_000_000)
        self._odometry_max_age_ns = int(float(self.get_parameter("odometry_max_age_s").value) * 1_000_000_000)
        self._active: _ActiveReference | None = None
        self._odometry: Odometry | None = None
        self._sequence = 0
        self._commands = self.create_publisher(Twist, str(self.get_parameter("command_topic").value), 10)
        self._feedback = self.create_publisher(MotionExecutionFeedback, str(self.get_parameter("feedback_topic").value), 10)
        self.create_subscription(MotionReference, str(self.get_parameter("reference_topic").value), self._on_reference, 10)
        self.create_subscription(Odometry, str(self.get_parameter("odometry_topic").value), self._on_odometry, 10)
        self.create_timer(1.0 / float(self.get_parameter("control_rate_hz").value), self._tick)

    def _on_reference(self, reference: MotionReference) -> None:
        if not reference.plan_id:
            self._cancel("REFERENCE_WITHDRAWN")
            return
        parsed = parse_wheeled_reference(reference)
        if parsed.reason is not None:
            self._active = _ActiveReference(
                reference.plan_id, (), self.get_clock().now().nanoseconds
            )
            self._sequence = 0
            self._fail(parsed.reason)
            return
        self._active = _ActiveReference(parsed.plan_id, parsed.path_xy_yaw, self.get_clock().now().nanoseconds)
        self._sequence = 0
        self._publish_feedback(MotionExecutionFeedback.ACCEPTED)

    def _on_odometry(self, odometry: Odometry) -> None:
        self._odometry = odometry

    def _tick(self) -> None:
        if self._active is None:
            return
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._active.received_ns > self._reference_max_age_ns or self._odometry is None:
            self._fail("STALE_INPUT")
            return
        stamp = self._odometry.header.stamp
        odometry_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if odometry_ns <= 0 or now_ns - odometry_ns > self._odometry_max_age_ns:
            self._fail("STALE_INPUT")
            return
        pose = self._odometry.pose.pose.position
        command = track_path(self._active.path_xy_yaw, TrackingState(pose.x, pose.y, _yaw(self._odometry)), self._policy)
        if command.failure_reason is not None:
            self._fail(command.failure_reason)
            return
        self._publish_twist(command.linear_x_mps, command.angular_z_radps)
        if command.complete:
            self._publish_feedback(MotionExecutionFeedback.SEGMENT_COMPLETE)
            self._active = None
        else:
            self._publish_feedback(MotionExecutionFeedback.EXECUTING)

    def _publish_twist(self, linear: float = 0.0, angular: float = 0.0) -> None:
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self._commands.publish(msg)

    def _publish_feedback(self, state: int, reason: str = "") -> None:
        if self._active is None:
            return
        identity = make_feedback_identity(self._active.plan_id)
        msg = MotionExecutionFeedback()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        self._sequence += 1
        msg.sequence = self._sequence
        msg.platform_type = MotionExecutionFeedback.WHEELED
        msg.plan_id = identity.plan_id
        msg.segment_id = identity.segment_id
        msg.state = state
        msg.reason_code = reason
        self._feedback.publish(msg)

    def _fail(self, reason: str) -> None:
        self._publish_twist()
        self._publish_feedback(MotionExecutionFeedback.FAILED, reason)
        self._active = None

    def _cancel(self, reason: str) -> None:
        self._publish_twist()
        self._publish_feedback(MotionExecutionFeedback.CANCELED, reason)
        self._active = None

    def destroy_node(self) -> bool:
        self._publish_twist()
        return super().destroy_node()


def main() -> None:
    import rclpy

    rclpy.init()
    node = WheeledControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
