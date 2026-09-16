"""Simulation-only scan speed profile using the shared formal PathExecutor."""
from dataclasses import replace
import json
import math
import rclpy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy
from lunar_pure_wheeled_controller.node import PureWheeledControllerNode


def scan_policy(normal, speed, acceleration):
    if not all(math.isfinite(v) and v>0 for v in (speed,acceleration)):
        raise ValueError('scan angular speed and acceleration must be finite and positive')
    return replace(normal,max_angular_radps=min(normal.max_angular_radps,speed),
                   max_angular_accel_radps2=min(normal.max_angular_accel_radps2,acceleration))


class ScanControllerNode(PureWheeledControllerNode):
    def __init__(self):
        super().__init__()
        self.declare_parameter('scan_profile_enabled',False)
        self.declare_parameter('scan_max_angular_radps',.15)
        self.declare_parameter('scan_max_angular_accel_radps2',.1)
        self.declare_parameter('observation_status_topic','/lunar_sim/observation_status')
        self._normal_policy=self._policy
        self._scan_profile_active=bool(self.get_parameter('scan_profile_enabled').value)
        limited=scan_policy(self._normal_policy,
                            float(self.get_parameter('scan_max_angular_radps').value),
                            float(self.get_parameter('scan_max_angular_accel_radps2').value))
        if self._scan_profile_active:
            self._policy=self._executor.policy=limited
        self.create_subscription(String,str(self.get_parameter('observation_status_topic').value),
                                 self.on_scan_status,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def on_scan_status(self,message):
        if not self._scan_profile_active:
            return
        try:
            phase=json.loads(message.data).get('phase')
        except (ValueError,AttributeError):
            return
        if phase in ('SCAN_COMPLETE','EXPLORATION_TASK_STARTED'):
            self._policy=self._executor.policy=self._normal_policy
            self._scan_profile_active=False
            self.get_logger().info('Initial scan complete; restored normal tracking speed profile')


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO);node=None
    try:
        node=ScanControllerNode();rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
