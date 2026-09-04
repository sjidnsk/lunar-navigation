"""ROS adapter for the isolated pure wheeled path tracker."""

from __future__ import annotations

from math import atan2, isfinite, sqrt

from geometry_msgs.msg import Twist
from lunar_planning_msgs.msg import MotionReference
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from .frame import MapFromOdom, map_tracking_state, parse_map_from_odom
from .incremental_path import ParsedIncrementalPath, parse_incremental_path
from .reference import ParsedReference, parse_reference
from .tracking import TrackingPolicy, TrackingState, track_path, track_trajectory


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


def incremental_path_qos() -> QoSProfile:
    """Match the incremental navigator's durable local-path contract."""
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class PureWheeledControllerNode(Node):
    """Publishes bounded wheel commands for the latest valid reference."""

    def __init__(self) -> None:
        super().__init__("lunar_pure_wheeled_controller")
        defaults = {
            "input_mode": "motion_reference",
            "reference_topic": "/Car/T4/planning/wheeled_reference",
            "path_topic": "/Car/T4/planning/local_path",
            "odometry_topic": "/Car/T3/localization/odometry",
            "command_topic": "/Car/T5/Car_Cmd_Vel",
            "execution_cancel_topic": "/Car/T4/execution/cancel",
            "tf_topic": "/tf",
            "control_rate_hz": 20.0,
            "lookahead_m": 0.5,
            "max_linear_mps": 0.2,
            "max_angular_radps": 0.5,
            "goal_position_tolerance_m": 0.2,
            "goal_yaw_tolerance_rad": 0.2,
            "max_cross_track_error_m": 1.0,
            "spin_kp": 1.5,
            "translation_epsilon_m": 1.0e-3,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        input_mode = self.get_parameter("input_mode").value
        if input_mode not in {"motion_reference", "incremental_path"}:
            raise ValueError(
                "input_mode must be motion_reference or incremental_path"
            )
        self._input_mode = input_mode
        topic_parameters = ["odometry_topic", "command_topic"]
        if self._input_mode == "motion_reference":
            topic_parameters.extend(["reference_topic", "execution_cancel_topic"])
        else:
            topic_parameters.extend(["path_topic", "tf_topic"])
        topics = {
            name: self._absolute_topic_name(name, self.get_parameter(name).value)
            for name in topic_parameters
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
                "spin_kp",
                "translation_epsilon_m",
            )
        })
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        if not isfinite(control_rate_hz) or control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be finite and greater than zero")
        self._active: ParsedReference | ParsedIncrementalPath | None = None
        self._trajectory_cursor = 0
        self._odometry: Odometry | None = None
        self._map_from_odom: MapFromOdom | None = None
        self._commands = self.create_publisher(
            Twist, topics["command_topic"], 10
        )
        self.create_subscription(
            Odometry,
            topics["odometry_topic"],
            self._on_odometry,
            10,
        )
        if self._input_mode == "motion_reference":
            self.create_subscription(
                MotionReference,
                topics["reference_topic"],
                self._on_reference,
                10,
            )
            self.create_subscription(
                String,
                topics["execution_cancel_topic"],
                self._on_cancel,
                10,
            )
        else:
            self.create_subscription(
                Path,
                topics["path_topic"],
                self._on_path,
                incremental_path_qos(),
            )
            self.create_subscription(
                TFMessage,
                topics["tf_topic"],
                self._on_tf,
                10,
            )
        self.create_timer(1.0 / control_rate_hz, self._tick)

    @staticmethod
    def _absolute_topic_name(parameter_name: str, value: object) -> str:
        if not isinstance(value, str) or len(value) < 2 or not value.startswith("/"):
            raise ValueError(f"{parameter_name} must be an absolute topic name")
        return value

    def _on_reference(self, reference: MotionReference) -> None:
        parsed = parse_reference(reference)
        if parsed.reason is not None:
            self._clear_active_and_stop()
            return
        self._active = parsed
        self._trajectory_cursor = 0

    def _on_cancel(self, message: String) -> None:
        if (
            self._input_mode == "motion_reference"
            and isinstance(self._active, ParsedReference)
            and message.data == self._active.plan_id
        ):
            self._clear_active_and_stop()

    def _on_path(self, path: Path) -> None:
        parsed = parse_incremental_path(path)
        if parsed.clear or parsed.reason is not None:
            self._clear_active_and_stop()
            return
        self._active = parsed
        self._trajectory_cursor = 0

    def _on_tf(self, message: TFMessage) -> None:
        update = parse_map_from_odom(message)
        if update.found:
            self._map_from_odom = update.transform
            if update.transform is None:
                self._publish_twist()

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
        if self._input_mode == "incremental_path":
            self._tick_incremental_path()
            return
        self._tick_motion_reference()

    def _tick_incremental_path(self) -> None:
        if (
            not isinstance(self._active, ParsedIncrementalPath)
            or self._odometry is None
            or self._map_from_odom is None
        ):
            self._publish_twist()
            return
        state = map_tracking_state(self._odometry, self._map_from_odom)
        if state is None:
            self._odometry = None
            self._publish_twist()
            return
        command = track_path(self._active.path_xy_yaw, state, self._policy)
        self._publish_twist(command.linear_x_mps, command.angular_z_radps)
        if command.complete or command.failure_reason is not None:
            self._clear_active()

    def _tick_motion_reference(self) -> None:
        if not isinstance(self._active, ParsedReference) or self._odometry is None:
            self._publish_twist()
            return
        yaw = _yaw(self._odometry)
        if yaw is None:
            self._odometry = None
            self._publish_twist()
            return
        position = self._odometry.pose.pose.position
        state = TrackingState(position.x, position.y, yaw)
        if self._active.trajectory_samples:
            result = track_trajectory(
                self._active.trajectory_samples,
                state,
                self._policy,
                self._trajectory_cursor,
            )
            self._trajectory_cursor = result.next_cursor
            command = result.command
        else:
            command = track_path(
                self._active.path_xy_yaw,
                state,
                self._policy,
            )
        self._publish_twist(command.linear_x_mps, command.angular_z_radps)
        if command.complete or command.failure_reason is not None:
            self._clear_active()

    def _clear_active(self) -> None:
        self._active = None
        self._trajectory_cursor = 0

    def _clear_active_and_stop(self) -> None:
        self._clear_active()
        self._publish_twist()

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
