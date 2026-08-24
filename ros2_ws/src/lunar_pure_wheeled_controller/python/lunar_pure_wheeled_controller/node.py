"""ROS adapter for the isolated pure wheeled path tracker."""

from __future__ import annotations

from math import atan2, isfinite, sqrt

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node

from .reference import ParsedReference, parse_path
from .tracking import TrackingPolicy, TrackingState, track_path


def _yaw(odometry: Odometry) -> float | None:
    position = odometry.pose.pose.position
    orientation = odometry.pose.pose.orientation
    if not all(isfinite(value) for value in (
        position.x,
        position.y,
        position.z,
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    )):
        return None
    quaternion_norm = sqrt(
        orientation.x * orientation.x
        + orientation.y * orientation.y
        + orientation.z * orientation.z
        + orientation.w * orientation.w
    )
    if not isfinite(quaternion_norm) or quaternion_norm == 0.0:
        return None
    x = orientation.x / quaternion_norm
    y = orientation.y / quaternion_norm
    z = orientation.z / quaternion_norm
    w = orientation.w / quaternion_norm
    yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw if isfinite(yaw) else None


class PureWheeledControllerNode(Node):
    """Publishes bounded wheel commands for the latest valid path."""

    def __init__(self) -> None:
        super().__init__("lunar_pure_wheeled_controller")
        defaults = {
            "path_topic": "/Car/T4/planning/wheeled_path",
            "odometry_topic": "/Car/T3/localization/odometry",
            "command_topic": "/Car/T5/Car_Cmd_Vel",
            "control_rate_hz": 20.0,
            "lookahead_m": 0.5,
            "max_linear_mps": 0.2,
            "max_angular_radps": 0.5,
            "goal_position_tolerance_m": 0.2,
            "goal_yaw_tolerance_rad": 0.2,
            "max_cross_track_error_m": 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        topics = {
            name: self._absolute_topic_name(name, self.get_parameter(name).value)
            for name in ("path_topic", "odometry_topic", "command_topic")
        }
        self._policy = TrackingPolicy(**{
            name: float(self.get_parameter(name).value)
            for name in (
                "lookahead_m",
                "max_linear_mps",
                "max_angular_radps",
                "goal_position_tolerance_m",
                "goal_yaw_tolerance_rad",
                "max_cross_track_error_m",
            )
        })
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        if not isfinite(control_rate_hz) or control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be finite and greater than zero")
        self._active: ParsedReference | None = None
        self._odometry: Odometry | None = None
        self._commands = self.create_publisher(
            Twist, topics["command_topic"], 10
        )
        self.create_subscription(
            Path,
            topics["path_topic"],
            self._on_path,
            10,
        )
        self.create_subscription(
            Odometry,
            topics["odometry_topic"],
            self._on_odometry,
            10,
        )
        self.create_timer(1.0 / control_rate_hz, self._tick)

    @staticmethod
    def _absolute_topic_name(parameter_name: str, value: object) -> str:
        if not isinstance(value, str) or len(value) < 2 or not value.startswith("/"):
            raise ValueError(f"{parameter_name} must be an absolute topic name")
        return value

    def _on_path(self, path: Path) -> None:
        parsed = parse_path(path)
        if parsed.reason is not None:
            self._active = None
            self._publish_twist()
            return
        self._active = parsed

    def _on_odometry(self, odometry: Odometry) -> None:
        if _yaw(odometry) is None:
            self._odometry = None
            self._publish_twist()
            return
        self._odometry = odometry

    def _tick(self) -> None:
        if self._active is None or self._odometry is None:
            self._publish_twist()
            return
        yaw = _yaw(self._odometry)
        if yaw is None:
            self._odometry = None
            self._publish_twist()
            return
        position = self._odometry.pose.pose.position
        command = track_path(
            self._active.path_xy_yaw,
            TrackingState(position.x, position.y, yaw),
            self._policy,
        )
        self._publish_twist(command.linear_x_mps, command.angular_z_radps)
        if command.complete or command.failure_reason is not None:
            self._active = None

    def _publish_twist(self, linear: float = 0.0, angular: float = 0.0) -> None:
        message = Twist()
        message.linear.x = linear
        message.angular.z = angular
        self._commands.publish(message)

    def destroy_node(self) -> bool:
        self._publish_twist()
        return super().destroy_node()


def main() -> None:
    import rclpy

    rclpy.init()
    node = PureWheeledControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
