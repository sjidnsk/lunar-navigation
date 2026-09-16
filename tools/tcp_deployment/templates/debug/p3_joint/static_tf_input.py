"""Adapt P3 static map->odom to P4's existing receipt-based TF input.

The transform and source stamp are preserved. Static transforms are timeless;
republication is not new localization evidence. Odometry retains its own timeout.
Nothing is published to /tf or /tf_static.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy
from tf2_msgs.msg import TFMessage

class StaticInput(Node):
    def __init__(self):
        super().__init__('p4_static_tf_input')
        self.transform=None
        self.publisher=self.create_publisher(TFMessage,'/P4/input/map_to_odom',10)
        self.create_subscription(TFMessage,'/tf_static',self.receive,QoSProfile(depth=100,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_timer(.2,self.publish)
    def receive(self,message):
        for transform in message.transforms:
            if transform.header.frame_id=='map' and transform.child_frame_id=='odom':
                self.transform=transform
                self.get_logger().info('Received P3 static map->odom; preserving transform and source stamp')
    def publish(self):
        if self.transform is not None:self.publisher.publish(TFMessage(transforms=[self.transform]))

def main():
    rclpy.init();node=StaticInput()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
if __name__=='__main__':main()
