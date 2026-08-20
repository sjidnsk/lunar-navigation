"""ROS wrapper that validates direct Task3 body odometry."""

from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from lunar_navigation_msgs.msg import LocalizationStatus
from nav_msgs.msg import Odometry

from .localization_status_adapter import LocalizationStatusPolicy, make_localization_status
from .t3_map_adapter_node import shutdown_if_running


class LocalizationStatusAdapterNode(Node):
    """Publish status for Task3 odometry without rewriting its localization."""

    def __init__(self) -> None:
        super().__init__("luna_t3_localization_status_adapter")
        self.declare_parameter("odometry_topic", "/Car/T3/semantic/current_pose")
        self.declare_parameter("status_topic", "/localization/status")
        self.declare_parameter("max_age_s", 1.0)
        self.declare_parameter("valid_position_variance_max", 0.1)
        self.declare_parameter("degraded_position_variance_max", 0.5)
        self._policy = LocalizationStatusPolicy(
            max_age_s=float(self.get_parameter("max_age_s").value),
            valid_position_variance_max=float(
                self.get_parameter("valid_position_variance_max").value
            ),
            degraded_position_variance_max=float(
                self.get_parameter("degraded_position_variance_max").value
            ),
        )
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._publisher = self.create_publisher(
            LocalizationStatus, str(self.get_parameter("status_topic").value), qos
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odometry_topic").value),
            self._on_odometry,
            qos,
        )

    def _on_odometry(self, message: Odometry) -> None:
        self._publisher.publish(make_localization_status(
            message, now_ns=self.get_clock().now().nanoseconds, policy=self._policy
        ))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: LocalizationStatusAdapterNode | None = None
    try:
        node = LocalizationStatusAdapterNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        shutdown_if_running(rclpy.ok, rclpy.shutdown)
