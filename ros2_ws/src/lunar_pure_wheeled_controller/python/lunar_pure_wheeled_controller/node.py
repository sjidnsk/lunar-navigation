"""ROS adapter for the isolated pure wheeled path tracker."""

from __future__ import annotations

from math import atan2, isfinite, sqrt
from dataclasses import fields
from collections import deque
from pathlib import Path as FilePath
import yaml
from time import monotonic

from geometry_msgs.msg import Twist
from lunar_planning_msgs.msg import MotionReference, PathReference, TrackingStatus
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from .frame import MapFromOdom, map_tracking_state, parse_map_from_odom
from .incremental_path import ParsedIncrementalPath, parse_incremental_path
from .reference import ParsedReference, parse_reference
from .tracking import TrackingPolicy, TrackingState, track_trajectory
from .execution import PathExecutor


def _yaw(odometry: Odometry) -> float | None:
    position = odometry.pose.pose.position
    orientation = odometry.pose.pose.orientation
    if not all(
        isfinite(value)
        for value in (
            position.x,
            position.y,
            position.z,
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
    ):
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


def state_input_qos() -> QoSProfile:
    """Accept the navigator's best-effort odometry and TF input contract."""
    return QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


class PureWheeledControllerNode(Node):
    """Publishes bounded wheel commands for the latest valid reference."""

    def __init__(self) -> None:
        super().__init__("lunar_pure_wheeled_controller")
        defaults = {
            "input_mode": "motion_reference",
            "platform_config": "",
            "path_reference_topic": "/Car/T4/planning/path_reference",
            "tracking_status_topic": "/Car/T4/control/tracking_status",
            "reference_topic": "/Car/T4/planning/wheeled_reference",
            "path_topic": "/Car/T4/planning/local_path",
            "odometry_topic": "/Car/T3/localization/odometry",
            "command_topic": "/Car/T5/Car_Cmd_Vel",
            "execution_cancel_topic": "/Car/T4/execution/cancel",
            "tf_topic": "/tf",
            "control_rate_hz": 20.0,
            "odometry_timeout_s": 0.5,
            "tf_timeout_s": 0.5,
            "lookahead_m": 0.5,
            "max_linear_mps": 0.2,
            "max_angular_radps": 0.5,
            "goal_position_tolerance_m": 0.2,
            "goal_yaw_tolerance_rad": 0.2,
            "max_cross_track_error_m": 1.0,
            "spin_kp": 1.5,
            "translation_epsilon_m": 1.0e-3,
        }
        for field in fields(TrackingPolicy):
            defaults.setdefault(field.name, getattr(TrackingPolicy(), field.name))
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        input_mode = self.get_parameter("input_mode").value
        if input_mode not in {
            "motion_reference",
            "incremental_path",
            "incremental_reference",
        }:
            raise ValueError(
                "input_mode must be motion_reference, incremental_path or incremental_reference"
            )
        self._input_mode = input_mode
        topic_parameters = ["odometry_topic", "command_topic"]
        if self._input_mode == "motion_reference":
            topic_parameters.extend(["reference_topic", "execution_cancel_topic"])
        else:
            topic_parameters.extend(["path_topic", "tf_topic"])
            if self._input_mode == "incremental_reference":
                topic_parameters.extend(
                    ["path_reference_topic", "tracking_status_topic"]
                )
        topics = {
            name: self._absolute_topic_name(name, self.get_parameter(name).value)
            for name in topic_parameters
        }
        policy_values = {
            field.name: self.get_parameter(field.name).value
            for field in fields(TrackingPolicy)
        }
        platform_path = self.get_parameter("platform_config").value
        if platform_path:
            capability = yaml.safe_load(
                FilePath(platform_path).read_text(encoding="utf-8")
            )["capability"]
            names = dict(
                max_linear_mps="maximum_forward_speed_mps",
                max_reverse_mps="maximum_reverse_speed_mps",
                max_angular_radps="maximum_spin_rate_radps",
                max_linear_accel_mps2="maximum_acceleration_mps2",
                max_linear_decel_mps2="maximum_braking_deceleration_mps2",
                max_angular_accel_radps2="maximum_yaw_acceleration_radps2",
                max_curvature_per_m="maximum_curvature_per_m",
                max_lateral_accel_mps2="maximum_lateral_acceleration_mps2",
            )
            for key, source in names.items():
                policy_values[key] = min(
                    float(policy_values[key]), float(capability[source])
                )
        self._policy = TrackingPolicy(**policy_values)
        self._executor = PathExecutor(self._policy)
        self._reference_identity = None
        self._retired_sessions = deque(maxlen=64)
        self._invalidated_revision = -1
        self._path_sequence = 0
        self._last_control_ns = None
        self._tracking_status = (
            self.create_publisher(TrackingStatus, topics["tracking_status_topic"], 10)
            if self._input_mode == "incremental_reference"
            else None
        )
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        if not isfinite(control_rate_hz) or control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be finite and greater than zero")
        self._odometry_timeout_s = float(self.get_parameter("odometry_timeout_s").value)
        self._tf_timeout_s = float(self.get_parameter("tf_timeout_s").value)
        if not isfinite(self._odometry_timeout_s) or self._odometry_timeout_s <= 0.0:
            raise ValueError("odometry_timeout_s must be finite and greater than zero")
        if not isfinite(self._tf_timeout_s) or self._tf_timeout_s <= 0.0:
            raise ValueError("tf_timeout_s must be finite and greater than zero")
        self._active: ParsedReference | ParsedIncrementalPath | None = None
        self._trajectory_cursor = 0
        self._odometry: Odometry | None = None
        self._odometry_received_at: float | None = None
        self._map_from_odom: MapFromOdom | None = None
        self._map_from_odom_received_at: float | None = None
        self._commands = self.create_publisher(Twist, topics["command_topic"], 10)
        self.create_subscription(
            Odometry,
            topics["odometry_topic"],
            self._on_odometry,
            state_input_qos(),
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
            if self._input_mode == "incremental_reference":
                self.create_subscription(
                    PathReference,
                    topics["path_reference_topic"],
                    self._on_path_reference,
                    incremental_path_qos(),
                )
            else:
                self.create_subscription(
                    Path, topics["path_topic"], self._on_path, incremental_path_qos()
                )
            self.create_subscription(
                TFMessage,
                topics["tf_topic"],
                self._on_tf,
                state_input_qos(),
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
        if not parsed.trajectory_samples:
            self._path_sequence += 1
            self._executor.set_path(
                parsed.path_xy_yaw, identity=(parsed.plan_id, self._path_sequence)
            )

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
        self._path_sequence += 1
        self._executor.set_path(
            parsed.path_xy_yaw, identity=("plain", self._path_sequence)
        )

    def _on_path_reference(self, reference: PathReference) -> None:
        identity = (bytes(reference.session_id.uuid), int(reference.segment_revision))
        current = self._reference_identity
        if identity[0] in self._retired_sessions:
            return
        if (
            current is not None
            and identity[0] == current[0]
            and identity[1] < current[1]
        ):
            return
        if reference.state == PathReference.INVALIDATED:
            if (
                current is None
                or identity[0] == current[0]
                and identity[1] >= current[1]
            ):
                self._reference_identity = identity
                self._invalidated_revision = identity[1]
                self._clear_active_and_stop()
            return
        if reference.state != PathReference.ACTIVE:
            self._clear_active_and_stop()
            return
        if (
            current is not None
            and identity[0] == current[0]
            and identity[1] <= self._invalidated_revision
        ):
            return
        parsed = parse_incremental_path(reference.path)
        if parsed.clear or parsed.reason is not None or reference.segment_revision == 0:
            self._clear_active_and_stop()
            return
        if self._executor.set_path(
            parsed.path_xy_yaw, identity=identity, final=reference.reaches_final_goal
        ):
            if current is not None and current[0] != identity[0]:
                self._retired_sessions.append(current[0])
                self._invalidated_revision = -1
            self._reference_identity = identity
            self._active = parsed

    def _on_tf(self, message: TFMessage) -> None:
        update = parse_map_from_odom(message)
        if update.found:
            self._map_from_odom = update.transform
            self._map_from_odom_received_at = (
                monotonic() if update.transform is not None else None
            )
            if update.transform is None:
                self._publish_twist()

    def _on_odometry(self, odometry: Odometry) -> None:
        if _yaw(odometry) is None:
            self._odometry = None
            self._odometry_received_at = None
            self._publish_twist()
            return
        self._odometry = odometry
        self._odometry_received_at = monotonic()

    def _tick(self) -> None:
        if (
            self._active is None
            or self._odometry is None
            or self._input_is_stale(
                self._odometry_received_at,
                self._odometry_timeout_s,
            )
        ):
            self._publish_twist()
            return
        if self._input_mode in {"incremental_path", "incremental_reference"}:
            self._tick_incremental_path()
            return
        self._tick_motion_reference()

    def _tick_incremental_path(self) -> None:
        if (
            not isinstance(self._active, ParsedIncrementalPath)
            or self._odometry is None
            or self._map_from_odom is None
            or self._input_is_stale(
                self._map_from_odom_received_at,
                self._tf_timeout_s,
            )
        ):
            self._publish_twist()
            return
        state = map_tracking_state(self._odometry, self._map_from_odom)
        if state is None:
            self._odometry = None
            self._publish_twist()
            return
        now_ns = self.get_clock().now().nanoseconds
        dt = (
            0.05
            if self._last_control_ns is None
            else (now_ns - self._last_control_ns) / 1e9
        )
        self._last_control_ns = now_ns
        if dt <= 0:
            self._publish_twist()
            return
        result = self._executor.update(state, dt)
        command = result.command
        self._publish_twist(command.linear_x_mps, command.angular_z_radps)
        if self._tracking_status is not None and self._reference_identity is not None:
            message = TrackingStatus()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "map"
            message.session_id.uuid = list(self._reference_identity[0])
            message.segment_revision = self._reference_identity[1]
            message.state = getattr(TrackingStatus, result.phase)
            message.direction = result.direction
            message.progress_m, message.cross_track_m = (
                result.progress_m,
                result.cross_track_m,
            )
            message.heading_error_rad = result.heading_error_rad
            message.linear_speed_mps, message.angular_speed_radps = (
                state.linear_mps,
                state.angular_radps,
            )
            message.reason = command.failure_reason or result.phase
            self._tracking_status.publish(message)
        # Hold the terminal executor until a new reference arrives. Repeated
        # terminal feedback is safe because navigation deduplicates revisions.

    @staticmethod
    def _input_is_stale(received_at: float | None, timeout_s: float) -> bool:
        return received_at is None or monotonic() - received_at > timeout_s

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
        state = TrackingState(
            position.x,
            position.y,
            yaw,
            self._odometry.twist.twist.linear.x,
            self._odometry.twist.twist.angular.z,
        )
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
            now_ns = self.get_clock().now().nanoseconds
            dt = (
                0.05
                if self._last_control_ns is None
                else (now_ns - self._last_control_ns) / 1e9
            )
            self._last_control_ns = now_ns
            if dt <= 0:
                self._publish_twist()
                return
            command = self._executor.update(state, dt).command
        self._publish_twist(command.linear_x_mps, command.angular_z_radps)
        if command.complete or command.failure_reason is not None:
            self._clear_active()

    def _clear_active(self) -> None:
        self._executor.clear()
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
