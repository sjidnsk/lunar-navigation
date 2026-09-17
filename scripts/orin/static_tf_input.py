"""Relay static map->odom to P4's private, receipt-monitored TF input.

The original transform and source timestamp are preserved. Repetition is adapter
liveness, not a new localization measurement. Odometry keeps its own timeout.
Never publishes to shared TF, odometry, or vehicle command topics.
"""
import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage


class StaticInput(Node):
    def __init__(self):
        super().__init__("p4_static_tf_input")
        self.transform = None
        self.publisher = self.create_publisher(TFMessage, "/P4/input/map_to_odom", 10)
        self.create_subscription(
            TFMessage, "/tf_static", self.receive,
            QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_timer(0.2, self.publish)

    def receive(self, message):
        for transform in message.transforms:
            if transform.header.frame_id == "map" and transform.child_frame_id == "odom":
                self.transform = copy.deepcopy(transform)
                self.get_logger().info("Received static map->odom; preserving source timestamp")

    def publish(self):
        if self.transform is not None:
            self.publisher.publish(TFMessage(transforms=[self.transform]))


def main():
    rclpy.init()
    node = StaticInput()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
