import time
import math
from dataclasses import replace
from lunar_obj_tcp_sim.coordinates import wire_orientation

import pytest
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import Log
from tf2_msgs.msg import TFMessage

from lunar_obj_tcp_sim.bridge_node import TcpVehicleBridgeNode
from lunar_obj_tcp_sim.protocol import Feedback
from lunar_obj_tcp_sim.transport import ReceivedFeedback


class FakeTransport:
    def __init__(self, _host, _port, **kwargs):
        self.on_feedback = kwargs["on_feedback"]
        self.commands = []
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def set_wheel_command(self, command, angles=None):
        self.commands.append(tuple(command))
        self.angles = angles

    def close(self):
        self.closed = True


def spin_until(nodes, predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.01)
    assert predicate()


def test_bridge_publishes_each_new_feedback_once_with_same_receive_stamp(tmp_path):
    rclpy.init()
    wheel = tmp_path / "wheel.yaml"
    wheel.write_text(
        "capability:\n  wheel_diameter_m: 0.319\n  track_width_m: 0.67\n  wheelbase_m: 0.8175\n",
        encoding="utf-8",
    )
    holder = {}

    def factory(*args, **kwargs):
        holder["transport"] = FakeTransport(*args, **kwargs)
        return holder["transport"]

    bridge = TcpVehicleBridgeNode(
        transport_factory=factory,
        parameter_overrides=[
            Parameter("platform_config", value=str(wheel)),
            Parameter("prefix", value="/bridge_test"),
        ],
    )
    observer = Node("bridge_observer")
    odometry = []
    transforms = []
    reliable = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    observer.create_subscription(Odometry, "/bridge_test/odometry", odometry.append, reliable)
    observer.create_subscription(TFMessage, "/tf", transforms.append, reliable)
    feedback = Feedback(
        wheel_speeds=(0.0,) * 4,
        turn_angles=(0.0,) * 4,
        position_m=(1.0, 2.0, 3.0),
        orientation_xyzw=(0.1, 0.2, 0.3, 0.9),
        linear_velocity=(4.0, 5.0, 6.0),
        angular_velocity=(0.4, 0.5, 0.6),
    )
    holder["transport"].on_feedback(ReceivedFeedback(feedback, time.monotonic()))
    spin_until(
        [observer, bridge],
        lambda: len(odometry) == 1
        and sum(len(batch.transforms) for batch in transforms) >= 2,
    )

    assert odometry[0].header.frame_id == "odom"
    assert odometry[0].child_frame_id == "base_link"
    assert (odometry[0].pose.pose.position.x, odometry[0].pose.pose.position.y) == (1.0, -2.0)
    # No pose derivative exists on the first sample: it cannot prove rest.
    for vector in (odometry[0].twist.twist.linear, odometry[0].twist.twist.angular):
        assert all(math.isnan(getattr(vector, axis)) for axis in ('x', 'y', 'z'))
    matching = [transform for batch in transforms for transform in batch.transforms]
    map_to_odom = next(item for item in matching if item.header.frame_id == "map")
    odom_to_base = next(item for item in matching if item.header.frame_id == "odom")
    assert map_to_odom.child_frame_id == "odom"
    assert map_to_odom.transform.rotation.w == 1.0
    assert odom_to_base.child_frame_id == "base_link"
    assert odom_to_base.header.stamp == odometry[0].header.stamp

    # Spinning without another receive callback must not republish/restamp old state.
    for _ in range(5):
        rclpy.spin_once(bridge, timeout_sec=0.01)
        rclpy.spin_once(observer, timeout_sec=0.01)
    assert len(odometry) == 1
    # Raw X has the wrong sign: pose-derived forward/reverse must prevail.
    t = time.monotonic()
    baseline = replace(feedback, position_m=(0., 0., 0.),
                       orientation_xyzw=wire_orientation((0., 0., 0., 1.)),
                       linear_velocity=(-.013, 0., 0.))
    holder["transport"].on_feedback(ReceivedFeedback(baseline, t))
    for i, (x, expected) in enumerate(((.025, .1), (0., -.1), (0., 0.)), 1):
        holder["transport"].on_feedback(ReceivedFeedback(replace(baseline, position_m=(x, 0., 0.)), t+i*.25))
        spin_until([observer, bridge], lambda: len(odometry) == i+2)
        assert odometry[-1].twist.twist.linear.x == pytest.approx(expected)

    holder["transport"].on_feedback(ReceivedFeedback(baseline, t+2.0))
    spin_until([observer, bridge], lambda: len(odometry) == 6)
    assert math.isnan(odometry[-1].twist.twist.linear.x)
    holder["transport"].on_feedback(ReceivedFeedback(baseline, t+2.25))
    spin_until([observer, bridge], lambda: len(odometry) == 7)
    assert odometry[-1].twist.twist.linear.x == pytest.approx(0.)

    bridge.destroy_node()
    observer.destroy_node()
    assert holder["transport"].closed is True
    rclpy.shutdown()


def test_bridge_converts_sole_cmd_vel_input_with_wheel_yaml(tmp_path):
    rclpy.init()
    wheel = tmp_path / "wheel.yaml"
    wheel.write_text(
        "capability:\n  wheel_diameter_m: 0.319\n  track_width_m: 0.67\n  wheelbase_m: 0.8175\n",
        encoding="utf-8",
    )
    holder = {}

    def factory(*args, **kwargs):
        holder["transport"] = FakeTransport(*args, **kwargs)
        return holder["transport"]

    bridge = TcpVehicleBridgeNode(
        transport_factory=factory,
        parameter_overrides=[
            Parameter("platform_config", value=str(wheel)),
            Parameter("prefix", value="/bridge_cmd_test"),
        ],
    )
    driver = Node("bridge_driver")
    publisher = driver.create_publisher(Twist, "/bridge_cmd_test/cmd_vel", 10)
    bridge._turn_angles = (0.,)*4
    command = Twist()
    command.linear.x = 0.2
    publisher.publish(command)
    spin_until([bridge, driver], lambda: bool(holder["transport"].commands))

    assert holder["transport"].commands[-1] == pytest.approx((1.2539185,) * 4)

    # A spin first sets all four steering targets with zero drive, then uses
    # actual TurnAngle feedback before releasing wheel speeds.
    turn = Twist()
    turn.angular.z = .3
    bridge._on_command(turn)
    assert holder['transport'].commands[-1] == (0.,)*4
    target_angles = holder['transport'].angles
    assert all(abs(a)>.5 for a in target_angles)
    bridge._on_feedback(ReceivedFeedback(Feedback(
        wheel_speeds=(0.,)*4, turn_angles=target_angles, position_m=(0.,)*3,
        orientation_xyzw=(0.,0.,0.,1.), linear_velocity=(0.,)*3,
        angular_velocity=(0.,)*3), time.monotonic()))
    bridge._on_command(turn)
    assert all(abs(v)>.1 for v in holder['transport'].commands[-1])
    bridge._on_command(Twist())
    assert holder['transport'].commands[-1] == (0.,)*4
    assert holder['transport'].angles == target_angles
    bridge._turn_angles = (0.,)*4

    logs = []
    driver.create_subscription(Log, "/rosout", logs.append, 10)
    saturated = Twist()
    saturated.linear.x = 4.0
    publisher.publish(saturated)
    spin_until(
        [bridge, driver],
        lambda: any("wheel command saturated" in item.msg for item in logs),
    )
    warning = next(item.msg for item in logs if "wheel command saturated" in item.msg)
    assert "scale=0.399" in warning
    assert "limit=10.000" in warning

    invalid = Twist()
    invalid.linear.x = float("nan")
    publisher.publish(invalid)
    spin_until(
        [bridge, driver],
        lambda: holder["transport"].commands[-1] == (0.0,) * 4
        and any("invalid cmd_vel" in item.msg for item in logs),
    )
    bridge.destroy_node()
    driver.destroy_node()
    rclpy.shutdown()
