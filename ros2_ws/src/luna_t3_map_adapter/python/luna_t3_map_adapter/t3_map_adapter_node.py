"""ROS 2 coordinator publishing bounded Task3 global planning maps."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from grid_map_msgs.msg import GridMap
from lunar_navigation_msgs.msg import ExplorationTask
from std_msgs.msg import UInt64

from .global_map_cache import GlobalMapCache
from .global_map_coordinator import GlobalMapCoordinator
from .t3_sqlite_reader import Task3MapError


def shutdown_if_running(is_running: Callable[[], bool], shutdown: Callable[[], None]) -> None:
    if is_running():
        shutdown()


class Task3MapAdapterNode(Node):
    """Publish a global map only from a frozen active task and map revision."""

    def __init__(self) -> None:
        super().__init__("luna_t3_map_adapter")
        self.declare_parameter("sqlite_path", "")
        self.declare_parameter("max_cached_tiles", 128)
        self.declare_parameter("exploration_task_topic", "/mission/exploration_task")
        self.declare_parameter("global_map_revision_topic", "/Car/T3/mapping/global_map_revision")
        self.declare_parameter("global_map_output_topic", "/environment/map_global")

        sqlite_path = str(self.get_parameter("sqlite_path").value)
        cached_tiles = int(self.get_parameter("max_cached_tiles").value)
        if not sqlite_path or not Path(sqlite_path).is_absolute():
            raise RuntimeError("TASK3_SQLITE_PATH_MISSING")
        self._coordinator = GlobalMapCoordinator(
            GlobalMapCache(sqlite_path, max_cached_tiles=cached_tiles)
        )
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(
            GridMap, str(self.get_parameter("global_map_output_topic").value), qos
        )
        self.create_subscription(
            ExplorationTask,
            str(self.get_parameter("exploration_task_topic").value),
            self._on_task,
            qos,
        )
        self.create_subscription(
            UInt64,
            str(self.get_parameter("global_map_revision_topic").value),
            self._on_revision,
            qos,
        )

    def _on_task(self, message: ExplorationTask) -> None:
        try:
            self._publish(self._coordinator.accept_task(
                message.mission_id,
                int(message.revision),
                int(message.desired_state),
                (
                    message.roi_min_x_m,
                    message.roi_min_y_m,
                    message.roi_max_x_m,
                    message.roi_max_y_m,
                ),
            ))
        except Task3MapError as error:
            self.get_logger().error(str(error))

    def _on_revision(self, message: UInt64) -> None:
        try:
            self._publish(self._coordinator.accept_revision(int(message.data)))
        except Task3MapError as error:
            self.get_logger().error(str(error))

    def _publish(self, message: GridMap | None) -> None:
        if message is None:
            return
        message.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: Task3MapAdapterNode | None = None
    try:
        node = Task3MapAdapterNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        shutdown_if_running(rclpy.ok, rclpy.shutdown)
