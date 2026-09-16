"""ROS 2 bridge between Twist/Odometry and the bounded Unreal TCP protocol."""

from typing import Callable

import rclpy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster

from .coordinates import convert_feedback, PoseRates
from .kinematics import FourWheelSteering, load_wheel_geometry, STEERING_CONFIG_FIELDS
from .configuration import SimulationConfig
from .transport import ReceivedFeedback, TcpVehicleTransport


TransportFactory = Callable[..., TcpVehicleTransport]


class TcpVehicleBridgeNode(Node):
    def __init__(
        self,
        *,
        transport_factory: TransportFactory = TcpVehicleTransport,
        **node_kwargs,
    ) -> None:
        super().__init__("tcp_vehicle_bridge", **node_kwargs)
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 6668)
        self.declare_parameter("vehicle_id", 0)
        self.declare_parameter("platform_config", "config/wheel.yaml")
        self.declare_parameter("prefix", "/lunar_sim")
        self.declare_parameter("odometry_topic", "")
        self.declare_parameter("cmd_vel_topic", "")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("send_rate_hz", 20.0)
        self.declare_parameter("command_timeout_s", 0.5)
        self.declare_parameter("feedback_timeout_s", 0.5)
        self.declare_parameter("connect_timeout_s", 5.0)
        self.declare_parameter("max_wheel_speed_radps", 10.0)

        for name in STEERING_CONFIG_FIELDS:
            default = SimulationConfig.__dataclass_fields__[name].default
            self.declare_parameter(name, list(default) if isinstance(default, tuple) else default)

        if self.get_parameter("use_sim_time").value:
            raise ValueError("tcp_vehicle_bridge requires use_sim_time=false")

        prefix = str(self.get_parameter("prefix").value).rstrip("/")
        odometry_topic = str(self.get_parameter("odometry_topic").value) or f"{prefix}/odometry"
        cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value) or f"{prefix}/cmd_vel"
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._odom_frame = str(self.get_parameter("odom_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._max_wheel_speed = float(self.get_parameter("max_wheel_speed_radps").value)
        self._steering = FourWheelSteering(
            *load_wheel_geometry(str(self.get_parameter("platform_config").value)),
            max_wheel_speed_radps=self._max_wheel_speed,
            **{name:self.get_parameter(name).value for name in STEERING_CONFIG_FIELDS})
        self._turn_angles = None

        state_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._odometry_publisher = self.create_publisher(Odometry, odometry_topic, state_qos)
        self._pose_rates = PoseRates()
        self.get_logger().warning("Twist: all body velocities from ROS pose differences at receive time; raw velocity axes are uncalibrated")
        self._tf_broadcaster = TransformBroadcaster(self, qos=state_qos)
        self._command_subscription = self.create_subscription(
            Twist, cmd_vel_topic, self._on_command, state_qos
        )
        self._transport = transport_factory(
            str(self.get_parameter("host").value),
            int(self.get_parameter("port").value),
            vehicle_id=int(self.get_parameter("vehicle_id").value),
            command_timeout_s=float(self.get_parameter("command_timeout_s").value),
            feedback_timeout_s=float(self.get_parameter("feedback_timeout_s").value),
            send_rate_hz=float(self.get_parameter("send_rate_hz").value),
            connect_timeout_s=float(self.get_parameter("connect_timeout_s").value),
            on_feedback=self._on_feedback,
            on_error=self._on_transport_error,
        )
        self._transport.start()

    def _on_command(self, message: Twist) -> None:
        try:
            result = self._steering.solve(message.linear.x, message.angular.z)
        except ValueError as error:
            self.get_logger().error(f"invalid cmd_vel; applying zero wheel command: {error}")
            self._transport.set_wheel_command((0.0, 0.0, 0.0, 0.0))
            return
        if result.saturated:
            self.get_logger().warning(
                "wheel command saturated: "
                f"scale={result.scale:.3f}, "
                f"limit={self._max_wheel_speed:.3f}",
                throttle_duration_sec=1.0,
            )
        aligned = self._steering.aligned(result, self._turn_angles)
        if not aligned and any(result.wheel_speeds):
            self.get_logger().info("waiting for wheel steering alignment", throttle_duration_sec=2.0)
        self._transport.set_wheel_command(
            result.wheel_speeds if aligned else (0.,)*4, result.turn_angles)

    def _on_transport_error(self, error: BaseException) -> None:
        self.get_logger().error(f"TCP vehicle transport stopped: {error}")

    def _on_feedback(self, received: ReceivedFeedback) -> None:
        # The protocol has no remote timestamp. Stamp exactly here, once per
        # newly decoded feedback body, using the local ROS receive clock.
        stamp = self.get_clock().now().to_msg()
        self._turn_angles = received.feedback.turn_angles
        feedback = convert_feedback(received.feedback)
        rates = self._pose_rates.update(feedback.position_m, feedback.orientation_xyzw,
                                        received.received_monotonic)
        # A missing derivative must not masquerade as measured standstill.
        unknown = (float("nan"),) * 3
        linear, angular = rates if rates is not None else (unknown, unknown)
        odometry = Odometry()
        # No derivative is available on the first sample or after a timing gap.
        if rates is None:
            for axis in range(6):
                odometry.twist.covariance[axis*7] = 1e6
        odometry.header.stamp = stamp
        odometry.header.frame_id = self._odom_frame
        odometry.child_frame_id = self._base_frame
        (
            odometry.pose.pose.position.x,
            odometry.pose.pose.position.y,
            odometry.pose.pose.position.z,
        ) = feedback.position_m
        (
            odometry.pose.pose.orientation.x,
            odometry.pose.pose.orientation.y,
            odometry.pose.pose.orientation.z,
            odometry.pose.pose.orientation.w,
        ) = feedback.orientation_xyzw
        (
            odometry.twist.twist.linear.x,
            odometry.twist.twist.linear.y,
            odometry.twist.twist.linear.z,
        ) = tuple(float(v) for v in linear)
        (
            odometry.twist.twist.angular.x,
            odometry.twist.twist.angular.y,
            odometry.twist.twist.angular.z,
        ) = tuple(float(v) for v in angular)
        self._odometry_publisher.publish(odometry)

        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = stamp
        map_to_odom.header.frame_id = self._map_frame
        map_to_odom.child_frame_id = self._odom_frame
        map_to_odom.transform.rotation.w = 1.0

        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = stamp
        odom_to_base.header.frame_id = self._odom_frame
        odom_to_base.child_frame_id = self._base_frame
        (
            odom_to_base.transform.translation.x,
            odom_to_base.transform.translation.y,
            odom_to_base.transform.translation.z,
        ) = feedback.position_m
        (
            odom_to_base.transform.rotation.x,
            odom_to_base.transform.rotation.y,
            odom_to_base.transform.rotation.z,
            odom_to_base.transform.rotation.w,
        ) = feedback.orientation_xyzw
        self._tf_broadcaster.sendTransform([map_to_odom, odom_to_base])

    def destroy_node(self) -> bool:
        self._transport.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = TcpVehicleBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
