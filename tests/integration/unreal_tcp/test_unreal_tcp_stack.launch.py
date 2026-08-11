from __future__ import annotations

from pathlib import Path
import socket
import sys
import time
import unittest

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_testing
from launch_testing.actions import ReadyToTest
import pytest
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped
from grid_map_msgs.msg import GridMap
from lunar_navigation_msgs.msg import ExplorationTask, MotionExecutionFeedback
from lunar_planning_msgs.msg import MotionReference
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock


HERE = Path(__file__).resolve().parent


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


TCP_PORT = _free_port()
CONTROL_PORT = _free_port()


@pytest.mark.launch_test
def generate_test_description():
    fake_server = ExecuteProcess(
        cmd=[
            sys.executable,
            str(HERE / "fake_unreal_server.py"),
            "--port",
            str(TCP_PORT),
            "--control-port",
            str(CONTROL_PORT),
        ],
        output="screen",
    )
    real_launch = Path(get_package_share_directory("lunar_navigation_config")) / (
        "launch/unreal_tcp_wheeled_path_planning.launch.py"
    )
    stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(real_launch)),
        launch_arguments={
            "server_host": "127.0.0.1",
            "server_port": str(TCP_PORT),
        }.items(),
    )
    return (
        LaunchDescription(
            [
                fake_server,
                TimerAction(period=0.5, actions=[stack]),
                TimerAction(period=1.0, actions=[ReadyToTest()]),
                launch_testing.util.KeepAliveProc(),
            ]
        ),
        {"fake_server": fake_server},
    )


def _diagnostic_value(message: DiagnosticArray, name: str, key: str):
    for status in message.status:
        if status.name != name:
            continue
        for value in status.values:
            if value.key == key:
                return value.value
    return None


class TestUnrealTcpStack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node("unreal_tcp_stack_integration_test")
        reliable = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        transient = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        best_effort = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        cls.clocks = []
        cls.global_maps = []
        cls.local_maps = []
        cls.odometry = []
        cls.bridge_status = []
        cls.coordinator_status = []
        cls.planner_diagnostics = []
        cls.missions = []
        cls.references = []
        cls.feedback = []
        cls.subscriptions = [
            cls.node.create_subscription(Clock, "/clock", cls.clocks.append, best_effort),
            cls.node.create_subscription(
                GridMap, "/environment/map_global", cls.global_maps.append, transient
            ),
            cls.node.create_subscription(
                GridMap, "/environment/map_local", cls.local_maps.append, transient
            ),
            cls.node.create_subscription(
                Odometry,
                "/lunar/unreal/wheeled_odometry",
                cls.odometry.append,
                best_effort,
            ),
            cls.node.create_subscription(
                DiagnosticArray,
                "/lunar/unreal/status",
                cls.bridge_status.append,
                reliable,
            ),
            cls.node.create_subscription(
                DiagnosticArray,
                "/lunar/path_planning/status",
                cls.coordinator_status.append,
                reliable,
            ),
            cls.node.create_subscription(
                DiagnosticArray,
                "/diagnostics",
                cls.planner_diagnostics.append,
                reliable,
            ),
            cls.node.create_subscription(
                ExplorationTask,
                "/mission/exploration_task",
                cls.missions.append,
                transient,
            ),
            cls.node.create_subscription(
                MotionReference,
                "/lunar/motion_reference",
                cls.references.append,
                reliable,
            ),
            cls.node.create_subscription(
                MotionExecutionFeedback,
                "/execution/motion_feedback",
                cls.feedback.append,
                reliable,
            ),
        ]
        cls.goal_publisher = cls.node.create_publisher(
            PoseStamped, "/goal_pose", reliable
        )

    @classmethod
    def tearDownClass(cls):
        cls.control("STOP")
        cls.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    @classmethod
    def spin_until(cls, predicate, timeout=30.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(cls.node, timeout_sec=0.05)
            if predicate():
                return True
        return predicate()

    @classmethod
    def bridge_value(cls, key):
        for message in reversed(cls.bridge_status):
            value = _diagnostic_value(message, "lunar_unreal_tcp_bridge", key)
            if value is not None:
                return value
        return None

    @classmethod
    def control(cls, command):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(command.encode("ascii"), ("127.0.0.1", CONTROL_PORT))

    @classmethod
    def diagnostic_summary(cls, messages, name):
        for message in reversed(messages):
            for status in message.status:
                if status.name == name:
                    values = {item.key: item.value for item in status.values}
                    return {"message": status.message, **values}
        return {}

    @classmethod
    def debug_summary(cls):
        return (
            "goal produced no reference: "
            f"coordinator={cls.diagnostic_summary(cls.coordinator_status, 'lunar_goal_coordinator')}, "
            f"planner={cls.diagnostic_summary(cls.planner_diagnostics, 'lunar_planner_ros')}, "
            f"missions={len(cls.missions)}, references={len(cls.references)}"
        )

    @classmethod
    def ensure_ready(cls):
        return cls.spin_until(
            lambda: cls.bridge_value("session_state") == "READY"
            and bool(cls.clocks)
            and bool(cls.global_maps)
            and bool(cls.local_maps)
            and bool(cls.odometry),
            timeout=30.0,
        )

    @classmethod
    def ensure_reference(cls):
        if cls.references:
            return True
        if not cls.ensure_ready():
            return False
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = cls.global_maps[-1].header.stamp
        goal.pose.position.x = 2.0
        goal.pose.position.y = 0.0
        goal.pose.orientation.w = 1.0
        cls.goal_publisher.publish(goal)
        return cls.spin_until(lambda: bool(cls.references), timeout=30.0)

    @classmethod
    def disconnect_and_wait_for_new_session(cls):
        old_session = cls.bridge_value("session_id")
        cls.control("DISCONNECT")
        return cls.spin_until(
            lambda: cls.bridge_value("session_state") == "READY"
            and cls.bridge_value("session_id") not in {None, "", old_session},
            timeout=20.0,
        )

    def test_handshake_publishes_clock_and_planner_maps(self):
        self.assertTrue(self.ensure_ready())
        self.assertEqual(self.global_maps[-1].header.frame_id, "map")
        self.assertEqual(self.local_maps[-1].header.frame_id, "odom")
        self.assertIn("valid_mask", self.global_maps[-1].layers)

    def test_goal_produces_wheeled_reference_and_matching_feedback(self):
        self.assertTrue(self.ensure_reference(), self.debug_summary())
        reference = self.references[0]
        self.assertEqual(reference.platform_type, MotionReference.WHEELED)
        self.assertTrue(reference.plan_id)
        self.assertTrue(
            self.spin_until(
                lambda: any(item.plan_id == reference.plan_id for item in self.feedback),
                timeout=10.0,
            )
        )

    def test_segment_complete_and_new_snapshot_trigger_second_plan(self):
        self.assertTrue(self.ensure_reference())
        first_plan = self.references[0].plan_id
        self.control("COMPLETE_ACTIVE")
        self.assertTrue(
            self.spin_until(
                lambda: any(
                    item.plan_id == first_plan
                    and item.state == MotionExecutionFeedback.SEGMENT_COMPLETE
                    for item in self.feedback
                ),
                timeout=15.0,
            ),
            self.debug_summary(),
        )
        self.assertTrue(self.spin_until(lambda: len(self.references) >= 2, timeout=45.0))
        self.assertNotEqual(self.references[1].plan_id, first_plan)

    def test_wrong_plan_stale_skew_wrong_session_and_disconnect_fail_closed(self):
        self.assertTrue(self.ensure_ready())
        self.control("WRONG_PLAN")
        self.assertTrue(
            self.spin_until(
                lambda: self.bridge_value("session_state") == "HOLD"
                and self.bridge_value("reason_code")
                == "EXECUTION_FEEDBACK_IDENTITY_INVALID",
                timeout=5.0,
            )
        )
        self.assertTrue(self.disconnect_and_wait_for_new_session())

        self.control("STALE")
        self.assertTrue(
            self.spin_until(
                lambda: self.bridge_value("session_state") == "HOLD"
                and self.bridge_value("reason_code") == "STATE_OR_MAP_STALE",
                timeout=5.0,
            )
        )
        self.assertTrue(self.disconnect_and_wait_for_new_session())

        self.control("SKEW")
        self.assertTrue(
            self.spin_until(
                lambda: self.bridge_value("session_state") == "HOLD"
                and self.bridge_value("reason_code")
                == "STATE_MAP_SNAPSHOT_SKEW",
                timeout=5.0,
            )
        )
        self.assertTrue(self.disconnect_and_wait_for_new_session())

        self.control("WRONG_SESSION")
        self.assertTrue(
            self.spin_until(
                lambda: self.bridge_value("session_state") == "HOLD"
                and self.bridge_value("reason_code") == "SESSION_ID_MISMATCH",
                timeout=5.0,
            )
        )
        self.assertTrue(self.disconnect_and_wait_for_new_session())

    def test_z_reconnect_does_not_replay_old_reference(self):
        self.assertTrue(self.ensure_ready())
        reference_count = len(self.references)
        self.assertTrue(self.disconnect_and_wait_for_new_session())
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.assertEqual(len(self.references), reference_count)


@launch_testing.post_shutdown_test()
class TestProcessesExit(unittest.TestCase):
    def test_no_crash_exit(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])
