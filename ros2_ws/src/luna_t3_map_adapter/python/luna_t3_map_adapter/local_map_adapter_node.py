"""ROS adapter for direct Task3 odom-local GridMap evidence."""

from __future__ import annotations

import math
from pathlib import Path

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from grid_map_msgs.msg import GridMap
from std_msgs.msg import UInt64
from tf2_ros import Buffer, TransformException, TransformListener

from .grid_map_codec import encode_grid_map
from .local_grid_conversion import (
    LocalMapPolicy,
    ObservationLedger,
    convert_local_grid,
    local_source_from_grid,
)
from .local_tile_evidence import PlanarTransform, tile_evidence_for_local_grid
from .t3_map_adapter_node import shutdown_if_running
from .t3_sqlite_reader import Task3MapError, Task3MapRead, read_roi_at_revision


def adapt_local_grid_message(
    message: GridMap,
    snapshot: Task3MapRead,
    map_from_odom: PlanarTransform,
    policy: LocalMapPolicy,
    ledger: ObservationLedger,
) -> GridMap:
    """Preserve local geometry while deriving planner layers from L0 evidence."""

    source = local_source_from_grid(message)
    evidence = tile_evidence_for_local_grid(source, snapshot, map_from_odom)
    canonical = convert_local_grid(source, evidence, policy, ledger)
    output = encode_grid_map(
        frame_id="odom",
        resolution_m=canonical.resolution_m,
        origin_x_m=canonical.origin_x_m,
        origin_y_m=canonical.origin_y_m,
        layers=canonical.layers,
        basic_layers=("elevation", "valid_mask"),
    )
    output.header.stamp = message.header.stamp
    return output


def _map_from_odom(transform: object) -> PlanarTransform:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    if not (
        math.isclose(rotation.x, 0.0, abs_tol=1.0e-6)
        and math.isclose(rotation.y, 0.0, abs_tol=1.0e-6)
        and math.isclose(rotation.z * rotation.z + rotation.w * rotation.w, 1.0, abs_tol=1.0e-6)
    ):
        raise ValueError("LOCAL_MAP_TF_GEOMETRY_INVALID")
    return PlanarTransform(
        float(translation.x), float(translation.y),
        math.atan2(2.0 * rotation.w * rotation.z, 1.0 - 2.0 * rotation.z * rotation.z),
    )


def _map_bounds(source: object, transform: PlanarTransform) -> tuple[float, float, float, float]:
    corners = (
        (source.origin_x_m, source.origin_y_m),
        (source.origin_x_m + source.occupancy.shape[1] * source.resolution_m, source.origin_y_m),
        (source.origin_x_m, source.origin_y_m + source.occupancy.shape[0] * source.resolution_m),
        (source.origin_x_m + source.occupancy.shape[1] * source.resolution_m,
         source.origin_y_m + source.occupancy.shape[0] * source.resolution_m),
    )
    mapped = tuple(transform.apply(x, y) for x, y in corners)
    return (
        min(x for x, _ in mapped), min(y for _, y in mapped),
        math.nextafter(max(x for x, _ in mapped), math.inf),
        math.nextafter(max(y for _, y in mapped), math.inf),
    )


class LocalMapAdapterNode(Node):
    """Publish canonical local evidence only when all source contracts hold."""

    def __init__(self) -> None:
        super().__init__("luna_t3_local_map_adapter")
        self.declare_parameter("sqlite_path", "")
        self.declare_parameter("source_topic", "/Car/T3/mapping/grid_map")
        self.declare_parameter("map_revision_topic", "/Car/T3/mapping/global_map_revision")
        self.declare_parameter("output_topic", "/environment/map_local")
        self.declare_parameter("occupancy_obstacle_threshold", 50)
        self.declare_parameter("semantic_obstacle_ids", [3])
        self.declare_parameter("semantic_forbidden_ids", [4])
        self.declare_parameter("max_age_s", 30.0)
        sqlite_path = Path(str(self.get_parameter("sqlite_path").value))
        if not sqlite_path.is_absolute():
            raise RuntimeError("TASK3_SQLITE_PATH_MISSING")
        self._sqlite_path = sqlite_path
        self._revision: int | None = None
        self._ledger = ObservationLedger()
        self._policy = LocalMapPolicy(
            occupancy_obstacle_threshold=int(self.get_parameter("occupancy_obstacle_threshold").value),
            semantic_obstacle_ids=frozenset(int(item) for item in self.get_parameter("semantic_obstacle_ids").value),
            semantic_forbidden_ids=frozenset(int(item) for item in self.get_parameter("semantic_forbidden_ids").value),
            max_age_s=float(self.get_parameter("max_age_s").value),
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._publisher = self.create_publisher(GridMap, str(self.get_parameter("output_topic").value), qos)
        self.create_subscription(GridMap, str(self.get_parameter("source_topic").value), self._on_grid, qos)
        self.create_subscription(UInt64, str(self.get_parameter("map_revision_topic").value), self._on_revision, qos)

    def _on_revision(self, message: UInt64) -> None:
        self._revision = int(message.data)

    def _on_grid(self, message: GridMap) -> None:
        if self._revision is None:
            self.get_logger().error("LOCAL_MAP_REVISION_MISSING")
            return
        try:
            source = local_source_from_grid(message)
            transform = self._tf_buffer.lookup_transform(
                "map", "odom", Time.from_msg(message.header.stamp), timeout=Duration(seconds=0.0)
            )
            map_from_odom = _map_from_odom(transform)
            snapshot = read_roi_at_revision(self._sqlite_path, _map_bounds(source, map_from_odom), self._revision)
            self._publisher.publish(adapt_local_grid_message(
                message, snapshot, map_from_odom, self._policy, self._ledger
            ))
        except (Task3MapError, TransformException, ValueError) as error:
            self.get_logger().error(str(error))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: LocalMapAdapterNode | None = None
    try:
        node = LocalMapAdapterNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        shutdown_if_running(rclpy.ok, rclpy.shutdown)
