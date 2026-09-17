"""Isolated native ROS check. Domain 183 only; never connects to the vehicle."""
import importlib.util
import os
from pathlib import Path
import time
assert os.environ.get('ROS_DOMAIN_ID') == '183'
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
spec = importlib.util.spec_from_file_location('p4_gui_test', Path(__file__).resolve().parents[1]/'lunar_car_gui_node.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rclpy.init()
gui = module.LunarCarGuiNode()
probe = rclpy.create_node('p4_gui_ownership_test')
modes, commands = [], []
probe.create_subscription(String, '/car/set_mode', lambda m:modes.append(m.data), 10)
probe.create_subscription(Twist, '/car/cmd_vel', commands.append, 10)
pub = probe.create_publisher(Bool, '/car/external_control_lease', 10)
def spin(seconds):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        rclpy.spin_once(gui, timeout_sec=.01)
        rclpy.spin_once(probe, timeout_sec=.01)
try:
    spin(1)
    pub.publish(Bool(data=True))
    spin(.5)
    assert gui.external_control_active
    commands.clear();modes.clear()
    spin(4)
    assert not modes and not commands, (modes, len(commands))
    assert gui.external_control_active
    gui.emergency_stop();spin(.3)
    assert modes == ['park'] and commands[-1].linear.x == 0.
    modes.clear();commands.clear()
    pub.publish(Bool(data=False));spin(.4)
    assert modes == ['park'] and not gui.external_control_active
    assert all(m.linear.x == 0. and m.angular.z == 0. for m in commands)
    print('PASS: 4s without heartbeat, no park/no GUI cmd; manual stop and explicit release work')
finally:
    gui.destroy_node();probe.destroy_node();rclpy.shutdown()
