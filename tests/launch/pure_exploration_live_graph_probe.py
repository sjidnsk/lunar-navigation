"""Test-only live graph probe for the installed pure-exploration launch."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point32, PoseStamped, Transform, TransformStamped
from lunar_planning_msgs.action import PlanMotion
from lunar_planning_msgs.msg import MotionReference
from lunar_pure_exploration_msgs.msg import PureExplorationStatus, PureExplorationTask
from rclpy.action import ActionServer
from rclpy.action.server import GoalResponse
from rclpy.executors import MultiThreadedExecutor
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint


ACTION_NAME = "/Car/T4/plan_motion"
EXPECTED_INPUT_SUBSCRIPTIONS = {
    "/Car/T3/mapping/global_overview": "nav_msgs/msg/OccupancyGrid",
    "/Car/T3/localization/odometry": "nav_msgs/msg/Odometry",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/Car/T4/exploration/task": "lunar_pure_exploration_msgs/msg/PureExplorationTask",
    "/Car/T4/planning/diagnostics": "diagnostic_msgs/msg/DiagnosticArray",
}
EXPECTED_OUTPUT_PUBLISHERS = {
    "/Car/T4/execution/motion_reference": "lunar_planning_msgs/msg/MotionReference",
    "/Car/T4/execution/cancel": "std_msgs/msg/String",
    "/Car/T4/exploration/status": "lunar_pure_exploration_msgs/msg/PureExplorationStatus",
    "/Car/T4/exploration/current_goal": "geometry_msgs/msg/PoseStamped",
    "/Car/T4/exploration/frontiers": "visualization_msgs/msg/MarkerArray",
    "/Car/T4/exploration/diagnostics": "diagnostic_msgs/msg/DiagnosticArray",
}
SYSTEM_TOPICS = {"/parameter_events", "/rosout"}
LEGACY_WORKLOAD_PATH_MARKERS = ("lunar_policy", "/ppo/", "/ppo_", "libppo")


class FakePlanMotionServer(Node):
    """Test-only Action server plus the real external inputs/outputs it observes."""

    def __init__(self) -> None:
        super().__init__("task14_fake_plan_motion_server")
        self.goal_received = threading.Event()
        self.result_sent = threading.Event()
        self.reference_received = threading.Event()
        self.waiting_for_stop = threading.Event()
        self._allow_result = threading.Event()
        self._goals: list[PlanMotion.Goal] = []
        self._goals_lock = threading.Lock()
        self._status_condition = threading.Condition()
        self._waiting_status_count = 0
        self._server = ActionServer(
            self,
            PlanMotion,
            ACTION_NAME,
            self._execute,
            goal_callback=self._goal_callback,
        )
        self._map_publisher = self.create_publisher(
            OccupancyGrid, "/Car/T3/mapping/global_overview", 10
        )
        self._odometry_publisher = self.create_publisher(
            Odometry, "/Car/T3/localization/odometry", 10
        )
        self._tf_publisher = self.create_publisher(TFMessage, "/tf", 10)
        self._task_publisher = self.create_publisher(
            PureExplorationTask, "/Car/T4/exploration/task", 10
        )
        self._reference_subscription = self.create_subscription(
            MotionReference,
            "/Car/T4/execution/motion_reference",
            lambda _: self.reference_received.set(),
            10,
        )
        self._status_subscription = self.create_subscription(
            PureExplorationStatus,
            "/Car/T4/exploration/status",
            self._observe_status,
            10,
        )

    def inputs_connected(self) -> bool:
        return all(
            publisher.get_subscription_count() == 1
            for publisher in (
                self._map_publisher,
                self._odometry_publisher,
                self._tf_publisher,
                self._task_publisher,
            )
        )

    def publish_start_inputs(self) -> None:
        global_map = OccupancyGrid()
        global_map.header.frame_id = "map"
        global_map.info.width = 20
        global_map.info.height = 20
        global_map.info.resolution = 0.5
        global_map.info.origin.orientation.w = 1.0
        global_map.data = [
            0 if x < global_map.info.width // 2 else -1
            for _ in range(global_map.info.height)
            for x in range(global_map.info.width)
        ]
        odometry = Odometry()
        odometry.header.frame_id = "odom"
        odometry.child_frame_id = "base_link"
        odometry.pose.pose.position.x = 2.0
        odometry.pose.pose.position.y = 2.0
        odometry.pose.pose.orientation.w = 1.0
        odometry.twist.twist.linear.x = 0.2
        map_from_odom = TransformStamped()
        map_from_odom.header.frame_id = "map"
        map_from_odom.child_frame_id = "odom"
        map_from_odom.transform.rotation.w = 1.0
        transforms = TFMessage()
        transforms.transforms.append(map_from_odom)
        task = PureExplorationTask()
        task.header.frame_id = "map"
        task.task_id = "task14-live-handshake"
        task.command = PureExplorationTask.START
        for x, y in ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)):
            point = Point32()
            point.x = x
            point.y = y
            task.boundary.points.append(point)
        self._map_publisher.publish(global_map)
        self._odometry_publisher.publish(odometry)
        self._tf_publisher.publish(transforms)
        self._task_publisher.publish(task)

    def publish_stationary_odometry(self) -> None:
        odometry = Odometry()
        odometry.header.frame_id = "odom"
        odometry.child_frame_id = "base_link"
        odometry.pose.pose.position.x = 2.0
        odometry.pose.pose.position.y = 2.0
        odometry.pose.pose.orientation.w = 1.0
        self._odometry_publisher.publish(odometry)

    def goal_count(self) -> int:
        with self._goals_lock:
            return len(self._goals)

    def waiting_status_count(self) -> int:
        with self._status_condition:
            return self._waiting_status_count

    def wait_for_next_waiting_status(self, previous: int, timeout: float) -> bool:
        with self._status_condition:
            return self._status_condition.wait_for(
                lambda: self._waiting_status_count > previous, timeout=timeout
            )

    def release_result(self) -> None:
        self._allow_result.set()

    def _observe_status(self, status: PureExplorationStatus) -> None:
        if status.reason_code != "WAITING_FOR_STOP":
            return
        with self._status_condition:
            self._waiting_status_count += 1
            self.waiting_for_stop.set()
            self._status_condition.notify_all()

    def first_goal(self) -> PlanMotion.Goal:
        with self._goals_lock:
            assert self._goals
            return self._goals[0]

    def _goal_callback(self, goal: PlanMotion.Goal) -> GoalResponse:
        with self._goals_lock:
            self._goals.append(goal)
        self.goal_received.set()
        return GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        if not self._allow_result.wait(timeout=20.0):
            raise RuntimeError("test did not release the first PlanMotion result")
        result = PlanMotion.Result()
        goal = goal_handle.request
        result.planning_outcome = PlanMotion.Result.NEW_REFERENCE_AVAILABLE
        result.execution_directive = PlanMotion.Result.ACTIVATE_NEW_REFERENCE
        result.reason_code = "PLAN_FOUND"
        result.has_reference = True
        result.reference.header.frame_id = "map"
        result.reference.plan_id = f"test:{goal.request_id}"
        result.reference.platform_type = MotionReference.WHEELED
        result.reference.path_preview.header.frame_id = "map"
        preview = PoseStamped()
        preview.header.frame_id = "map"
        preview.pose.position.x = goal.goal.point.x
        preview.pose.position.y = goal.goal.point.y
        preview.pose.orientation.w = 1.0
        result.reference.path_preview.poses.append(preview)
        trajectory_point = MultiDOFJointTrajectoryPoint()
        transform = Transform()
        transform.translation.x = goal.goal.point.x
        transform.translation.y = goal.goal.point.y
        transform.rotation.w = 1.0
        trajectory_point.transforms.append(transform)
        result.reference.trajectory.points.append(trajectory_point)
        goal_handle.succeed()
        self.result_sent.set()
        return result


def _ros2(*arguments: str) -> str:
    completed = subprocess.run(
        ["ros2", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    return completed.stdout


def _node_info_sections(node_info: str) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in node_info.splitlines():
        stripped = line.strip()
        if line.startswith("  ") and stripped.endswith(":"):
            current = sections.setdefault(stripped[:-1], {})
        elif line.startswith("    ") and current is not None and ": " in stripped:
            name, type_name = stripped.split(": ", maxsplit=1)
            current[name] = type_name
    return sections


def _assert_no_legacy_workload(launch_process: subprocess.Popen[str]) -> None:
    """No PPO/training process or shared object is allowed in this test-only launch."""
    process_tree = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,args="], text=True
    ).lower()
    assert not any(marker in process_tree for marker in LEGACY_WORKLOAD_PATH_MARKERS)
    maps_path = Path(f"/proc/{launch_process.pid}/maps")
    if maps_path.is_file():
        maps = maps_path.read_text(encoding="utf-8", errors="replace").lower()
        assert not any(marker in maps for marker in LEGACY_WORKLOAD_PATH_MARKERS)


def _wait_until(predicate, deadline: float) -> bool:
    """Poll only graph/readiness state; action and output completion use Events."""
    waiter = threading.Event()
    while time.monotonic() < deadline:
        if predicate():
            return True
        waiter.wait(timeout=min(0.1, deadline - time.monotonic()))
    return predicate()


def main() -> None:
    rclpy.init()
    server = FakePlanMotionServer()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(server)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    share = Path(get_package_share_directory("lunar_pure_exploration_ros"))
    assert share.is_relative_to(Path("/workspace/install")), share
    arguments = [
        "ros2",
        "launch",
        "lunar_pure_exploration_ros",
        "pure_exploration.launch.py",
        "platform_selector:=wheel",
        "maximum_position_probes:=64",
        "maximum_candidate_views:=64",
        "maximum_collision_work_units:=4096",
        "maximum_visibility_work_units:=4096",
        "maximum_path_preview_poses:=64",
        "maximum_executable_path_points:=64",
        "maximum_failure_entries:=8",
        "maximum_failure_patch_cells_per_entry:=256",
        "maximum_failure_total_patch_cells:=1024",
    ]
    environment = os.environ.copy()
    launch = subprocess.Popen(
        arguments,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 20.0
    try:
        graph_ready = False
        while time.monotonic() < deadline:
            names = _ros2("node", "list").splitlines()
            if "/pure_exploration" not in names:
                time.sleep(0.1)
                continue
            node_info = _ros2("node", "info", "/pure_exploration")
            action_types = set(_ros2("action", "list", "-t").splitlines())
            sections = _node_info_sections(node_info)
            publishers = sections.get("Publishers", {})
            subscribers = sections.get("Subscribers", {})
            action_clients = sections.get("Action Clients", {})
            application_publishers = {
                topic: type_name
                for topic, type_name in publishers.items()
                if topic not in SYSTEM_TOPICS
            }
            application_subscribers = {
                topic: type_name
                for topic, type_name in subscribers.items()
                if topic not in SYSTEM_TOPICS and not topic.startswith(f"{ACTION_NAME}/_action/")
            }
            if (
                application_subscribers == EXPECTED_INPUT_SUBSCRIPTIONS
                and application_publishers == EXPECTED_OUTPUT_PUBLISHERS
                and set(action_clients) == {ACTION_NAME}
                and f"{ACTION_NAME} [lunar_planning_msgs/action/PlanMotion]" in action_types
            ):
                assert names.count("/pure_exploration") == 1
                assert action_clients[ACTION_NAME] == "lunar_planning_msgs/action/PlanMotion"
                graph_ready = True
                break
            if launch.poll() is not None:
                break
            time.sleep(0.1)
        if not graph_ready:
            output = launch.stdout.read() if launch.stdout else ""
            raise AssertionError(
                "installed pure_exploration launch did not expose the complete graph; "
                f"returncode={launch.poll()} node_info={node_info if 'node_info' in locals() else ''} "
                f"output={output}"
            )
        assert _wait_until(server.inputs_connected, deadline), "explorer did not subscribe to all inputs"
        _assert_no_legacy_workload(launch)
        server.publish_start_inputs()
        assert server.waiting_for_stop.wait(
            timeout=20.0
        ), "explorer did not establish WAITING_FOR_STOP"
        assert server.goal_count() == 0, "Goal arrived before stationary confirmation"
        for sample in range(2):
            previous = server.waiting_status_count()
            server.publish_stationary_odometry()
            assert server.wait_for_next_waiting_status(
                previous, timeout=5.0
            ), f"stationary odometry sample {sample + 1} was not consumed"
            assert server.goal_count() == 0, (
                "Goal arrived before the third stationary odometry sample"
            )
        assert server.goal_count() == 0, "Goal arrived before the third sample"
        server.publish_stationary_odometry()
        assert server.goal_received.wait(timeout=20.0), "fake PlanMotion server received no Goal"
        assert server.goal_count() == 1, "stationary confirmation did not admit exactly the first Goal"
        goal = server.first_goal()
        assert goal.environment_mode == PlanMotion.Goal.LUNAR_SURFACE
        assert goal.replace_active_request is False
        assert goal.request_id
        server.release_result()
        assert server.result_sent.wait(timeout=5.0), "fake PlanMotion server sent no typed Result"
        assert server.reference_received.wait(timeout=20.0), "explorer did not publish the accepted reference"
    finally:
        server.release_result()
        if launch.poll() is None:
            launch.send_signal(signal.SIGTERM)
            try:
                launch.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                launch.kill()
                launch.wait(timeout=10.0)
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        server.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
