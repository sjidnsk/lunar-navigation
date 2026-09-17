"""Real Humble/DDS regression tests; run in an isolated ROS domain only."""
import importlib.util
from pathlib import Path
import time
import unittest

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from tf2_msgs.msg import TFMessage

ROOT = Path(__file__).resolve().parents[1]


class StaticTfInputTest(unittest.TestCase):
    def setUp(self):
        source = ROOT / "scripts/orin/static_tf_input.py"
        self.assertTrue(source.is_file(), "P4 static TF adapter is not implemented")
        spec = importlib.util.spec_from_file_location("p4_static_tf_input", source)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        rclpy.init()
        self.executor = SingleThreadedExecutor()
        self.nodes = []
        self.producer = self.add(rclpy.create_node("static_tf_test_source", enable_rosout=False))
        self.qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.producer.create_publisher(TFMessage, "/tf_static", self.qos)
        self.received = []
        self.producer.create_subscription(TFMessage, "/P4/input/map_to_odom", self.received.append, 10)

    def add(self, node):
        self.nodes.append(node)
        self.executor.add_node(node)
        return node

    def spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.executor.spin_once(timeout_sec=0.02)

    def tearDown(self):
        if not hasattr(self, "executor"):
            return
        for node in reversed(self.nodes):
            self.executor.remove_node(node)
            node.destroy_node()
        self.executor.shutdown()
        rclpy.shutdown()

    @staticmethod
    def transform(parent="map", child="odom", x=1.25):
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = parent, child
        t.header.stamp.sec, t.header.stamp.nanosec = 123, 456789
        t.transform.translation.x = x
        t.transform.rotation.w = 1.0
        return t

    def test_no_identity_is_invented_and_other_edges_are_ignored(self):
        self.add(self.module.StaticInput())
        self.spin(0.4)
        self.assertEqual(self.received, [])
        self.publisher.publish(TFMessage(transforms=[self.transform("odom", "base_link")]))
        self.spin(0.8)
        self.assertEqual(self.received, [])

    def test_late_joiner_repeats_only_map_odom_without_restamping(self):
        expected = self.transform()
        self.publisher.publish(TFMessage(transforms=[self.transform("base_link", "lidar"), expected]))
        self.spin(0.3)
        self.add(self.module.StaticInput())
        self.spin(2.0)
        self.assertGreaterEqual(len(self.received), 4)
        for msg in self.received:
            self.assertEqual(msg.transforms, [expected])

    def test_replacement_is_forwarded_and_no_shared_outputs_are_created(self):
        self.add(self.module.StaticInput())
        self.publisher.publish(TFMessage(transforms=[self.transform()]))
        self.spin(0.8)
        self.assertTrue(self.received)
        replacement = self.transform(x=-2.5)
        self.publisher.publish(TFMessage(transforms=[replacement]))
        self.spin(0.8)
        self.assertEqual(self.received[-1].transforms, [replacement])
        pubs = dict(self.producer.get_publisher_names_and_types_by_node("p4_static_tf_input", "/"))
        self.assertIn("/P4/input/map_to_odom", pubs)
        for topic in ("/tf", "/tf_static", "/Car/T3/localization/odometry", "/Car/T5/Car_Cmd_Vel"):
            self.assertNotIn(topic, pubs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
