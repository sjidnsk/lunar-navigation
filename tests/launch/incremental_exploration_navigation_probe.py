#!/usr/bin/env python3
"""Run the installed incremental_v2 demo and verify its ROS graph and loop."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("wheel", "legged"), required=True)
    parser.add_argument("--fine-resolution", choices=("0.2", "0.1"), required=True)
    parser.add_argument("--task-size", type=float, default=24.0)
    parser.add_argument("--expect-no-path-recovery", action="store_true")
    parser.add_argument("--timeout", type=float, default=55.0)
    return parser.parse_args()


def _pose_key(message) -> tuple[int, int, int]:
    position = message.pose.position
    orientation = message.pose.orientation
    yaw = math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )
    return (round(position.x * 100), round(position.y * 100), round(yaw * 100))


def main() -> int:
    arguments = _arguments()

    import rclpy
    from action_msgs.msg import GoalStatus, GoalStatusArray
    from diagnostic_msgs.msg import DiagnosticArray
    from geometry_msgs.msg import PoseStamped
    from grid_map_msgs.msg import GridMap
    from lunar_planning_msgs.msg import PathReference
    from lunar_pure_exploration_msgs.msg import PureExplorationStatus, PureExplorationTask
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from visualization_msgs.msg import MarkerArray

    log_path = Path(tempfile.mkstemp(prefix="incremental-demo-probe-", suffix=".log")[1])
    log_stream = log_path.open("w", encoding="utf-8")
    command = [
        "ros2",
        "launch",
        "lunar_incremental_navigation_ros",
        "incremental_exploration_navigation_rviz.launch.py",
        f"platform_type:={arguments.platform}",
        f"fine_resolution_m:={arguments.fine_resolution}",
        f"task_size_m:={arguments.task_size}",
        "start_rviz:=false",
        "show_ground_truth:=false",
    ]
    launch_environment = os.environ.copy()
    if arguments.expect_no_path_recovery:
        # The scenario owns a deterministic local obstacle barrier after its
        # first ACTIVE path. This avoids a second GridMap publisher racing the
        # rolling scenario input while preserving the production demo path.
        launch_environment["LUNAR_DEMO_REQUIRE_NO_PATH_RECOVERY"] = "1"
    process = subprocess.Popen(
        command,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=launch_environment,
    )

    rclpy.init()
    node = rclpy.create_node(f"incremental_demo_probe_{os.getpid()}")
    latched = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    observations = {
        "grid_map": False,
        "resolution": False,
        "local_window": False,
        "exploration_map": False,
        "frontiers": False,
        "active_path": False,
        "statuses": [],
        "goals": set(),
        "latest_goal": None,
        "latest_goal_key": None,
        "goals_after_no_path": set(),
        "no_path_seen": False,
        "task_starts_observed": 0,
        "cancel_requested": False,
        "cancel_published": False,
        "cancel_status_seen": False,
        "navigator_canceled_after_cancel": False,
        "path_invalidated_after_cancel": False,
        "navigation_action_active": False,
        "navigation_action_status_after_cancel": False,
        "navigation_action_quiescent_since": None,
        "planning_reasons": [],
    }

    def on_grid(message: GridMap) -> None:
        observations["grid_map"] = (
            message.header.frame_id == "odom"
            and message.layers == ["elevation"]
            and len(message.data) == 1
            and message.info.length_x <= 24.01
            and message.info.length_y <= 24.01
        )
        observations["resolution"] = math.isclose(
            message.info.resolution,
            float(arguments.fine_resolution),
            rel_tol=0.0,
            abs_tol=1e-6,
        )

    def on_map(message: OccupancyGrid) -> None:
        observations["exploration_map"] = (
            message.header.frame_id == "map"
            and bool(message.data)
            and {-1, 0}.issubset(set(message.data))
        )

    def on_frontiers(message: MarkerArray) -> None:
        observations["frontiers"] = observations["frontiers"] or any(
            marker.ns == "frontiers" and marker.action == marker.ADD
            for marker in message.markers
        )

    def on_window(message: MarkerArray) -> None:
        observations["local_window"] = any(len(marker.points) >= 4 for marker in message.markers)

    def on_goal(message: PoseStamped) -> None:
        key = _pose_key(message)
        observations["goals"].add(key)
        observations["latest_goal_key"] = key
        if observations["no_path_seen"]:
            observations["goals_after_no_path"].add(key)
        observations["latest_goal"] = (
            message.pose.position.x,
            message.pose.position.y,
        )

    def on_path(message: PathReference) -> None:
        if message.state == PathReference.ACTIVE and message.path.poses:
            observations["active_path"] = True
        if (
            observations["cancel_requested"]
            and message.state == PathReference.INVALIDATED
        ):
            observations["path_invalidated_after_cancel"] = True

    def on_status(message: PureExplorationStatus) -> None:
        observations["statuses"].append(message.reason_code)
        if (
            observations["cancel_requested"]
            and message.state == PureExplorationStatus.IDLE
            and message.reason_code == "CANCELED"
        ):
            observations["cancel_status_seen"] = True
        if "NO_PATH" in message.reason_code or message.reason_code == "GOAL_NOT_FREE":
            observations["no_path_seen"] = True

    def on_diagnostics(message: DiagnosticArray) -> None:
        for status in message.status:
            values = {item.key: item.value for item in status.values}
            reason = values.get("reason_code", values.get("cycle_result", ""))
            if reason:
                observations["planning_reasons"].append(reason)
                if observations["cancel_requested"] and reason == "CANCELED":
                    observations["navigator_canceled_after_cancel"] = True
                if "NO_PATH" in reason or reason == "GOAL_NOT_FREE":
                    observations["no_path_seen"] = True

    def on_task(message: PureExplorationTask) -> None:
        if message.command == PureExplorationTask.START:
            observations["task_starts_observed"] += 1

    def on_navigation_action_status(message: GoalStatusArray) -> None:
        active_states = {
            GoalStatus.STATUS_ACCEPTED,
            GoalStatus.STATUS_EXECUTING,
            GoalStatus.STATUS_CANCELING,
        }
        active = any(status.status in active_states for status in message.status_list)
        observations["navigation_action_active"] = active
        if observations["cancel_published"]:
            observations["navigation_action_status_after_cancel"] = True
            observations["navigation_action_quiescent_since"] = (
                None
                if active
                else observations["navigation_action_quiescent_since"]
                or time.monotonic()
            )

    node.create_subscription(GridMap, "/planning_demo/grid_map", on_grid, latched)
    node.create_subscription(
        OccupancyGrid,
        "/planning_demo/mapping/exploration_map",
        on_map,
        latched,
    )
    node.create_subscription(
        MarkerArray,
        "/planning_demo/exploration/frontiers",
        on_frontiers,
        latched,
    )
    node.create_subscription(
        MarkerArray,
        "/planning_demo/local_window",
        on_window,
        latched,
    )
    node.create_subscription(
        PoseStamped,
        "/planning_demo/exploration/current_goal",
        on_goal,
        latched,
    )
    node.create_subscription(
        PathReference,
        "/planning_demo/planning/path_reference",
        on_path,
        latched,
    )
    node.create_subscription(
        PureExplorationStatus,
        "/planning_demo/exploration/status",
        on_status,
        latched,
    )
    node.create_subscription(
        DiagnosticArray,
        "/planning_demo/planning/diagnostics",
        on_diagnostics,
        reliable,
    )
    node.create_subscription(
        PureExplorationTask,
        "/planning_demo/exploration/task",
        on_task,
        reliable,
    )
    node.create_subscription(
        GoalStatusArray,
        "/planning_demo/navigation/navigate_to_pose/_action/status",
        on_navigation_action_status,
        latched,
    )
    task_publisher = node.create_publisher(
        PureExplorationTask,
        "/planning_demo/exploration/task",
        reliable,
    )
    blocked_goal = None
    no_path_seen_after_block = False
    candidate_replaced_after_no_path = False
    node_destroyed = False
    launch_stopped = False
    start = time.monotonic()
    try:
        while time.monotonic() - start < arguments.timeout:
            if process.poll() is not None:
                break
            rclpy.spin_once(node, timeout_sec=0.05)
            if (
                arguments.expect_no_path_recovery
                and observations["active_path"]
                and observations["latest_goal"] is not None
                and blocked_goal is None
            ):
                blocked_goal = observations["latest_goal_key"]

            base_ready = all(
                observations[name]
                for name in (
                    "grid_map",
                    "resolution",
                    "local_window",
                    "exploration_map",
                    "frontiers",
                    "active_path",
                )
            ) and bool(observations["goals"])
            no_path = any(
                "NO_PATH" in reason or reason == "GOAL_NOT_FREE"
                for reason in observations["planning_reasons"] + observations["statuses"]
            )
            if blocked_goal is not None and no_path:
                no_path_seen_after_block = True
            if no_path_seen_after_block:
                candidate_replaced_after_no_path = any(
                    goal != blocked_goal
                    for goal in observations["goals_after_no_path"]
                )
            if base_ready and (
                not arguments.expect_no_path_recovery
                or (no_path_seen_after_block and candidate_replaced_after_no_path)
            ):
                break

        topics = dict(node.get_topic_names_and_types())
        global_overview_publishers = len(
            node.get_publishers_info_by_topic("/Car/T3/mapping/global_overview")
        )
        control_topic_present = "/Car/T5/Car_Cmd_Vel" in topics
        ground_truth_algorithm_subscribers = sum(
            info.node_name
            in {"incremental_navigation", "incremental_exploration"}
            for info in node.get_subscriptions_info_by_topic(
                "/planning_demo/ground_truth"
            )
        )
        no_path_observed = any(
            "NO_PATH" in reason or reason == "GOAL_NOT_FREE"
            for reason in observations["planning_reasons"] + observations["statuses"]
        )

        # Do not interrupt the action server and client simultaneously while a
        # goal is active.  Cancel through the public exploration task contract,
        # wait for both ends of the loop to acknowledge it, and only then stop
        # the launch process.
        observations["cancel_requested"] = True
        cancel = PureExplorationTask()
        cancel.header.stamp = node.get_clock().now().to_msg()
        cancel.header.frame_id = "map"
        cancel.task_id = "incremental-rviz-demo"
        cancel.command = PureExplorationTask.CANCEL
        cancel_deadline = time.monotonic() + 5.0
        observations["cancel_published"] = True
        task_publisher.publish(cancel)
        while time.monotonic() < cancel_deadline and process.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.05)
            quiescent_since = observations["navigation_action_quiescent_since"]
            if (
                observations["cancel_status_seen"]
                and observations["navigation_action_status_after_cancel"]
                and quiescent_since is not None
                and time.monotonic() - quiescent_since >= 0.75
            ):
                break

        navigation_action_quiescent = (
            observations["navigation_action_status_after_cancel"]
            and not observations["navigation_action_active"]
            and observations["navigation_action_quiescent_since"] is not None
        )

        node.destroy_node()
        node_destroyed = True
        rclpy.shutdown()
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
        launch_stopped = True
        log_stream.close()
        log_text = log_path.read_text(encoding="utf-8")
        clean_shutdown = (
            observations["cancel_status_seen"]
            and navigation_action_quiescent
            and "Asked to publish result for goal that does not exist" not in log_text
            and "exit code -6" not in log_text
        )
        result = {
            "platform": arguments.platform,
            "fine_resolution_m": float(arguments.fine_resolution),
            "task_starts_observed": observations["task_starts_observed"],
            "grid_map": observations["grid_map"],
            "resolution": observations["resolution"],
            "local_window": observations["local_window"],
            "exploration_map": observations["exploration_map"],
            "frontiers": observations["frontiers"],
            "unique_goals": len(observations["goals"]),
            "active_path": observations["active_path"],
            "status_reasons": observations["statuses"][-12:],
            "planning_reasons": observations["planning_reasons"][-12:],
            "no_path_observed": no_path_observed,
            "candidate_replaced": candidate_replaced_after_no_path,
            "path_invalidated_after_cancel": observations[
                "path_invalidated_after_cancel"
            ],
            "navigator_canceled_after_cancel": observations[
                "navigator_canceled_after_cancel"
            ],
            "navigation_action_quiescent": navigation_action_quiescent,
            "clean_shutdown": clean_shutdown,
            "global_overview_publishers": global_overview_publishers,
            "ground_truth_algorithm_subscribers": ground_truth_algorithm_subscribers,
            "control_topic_present": control_topic_present,
            "launch_log": str(log_path),
        }
        print(json.dumps(result, sort_keys=True))
        required = (
            observations["grid_map"]
            and observations["resolution"]
            and observations["local_window"]
            and observations["exploration_map"]
            and observations["frontiers"]
            and bool(observations["goals"])
            and observations["active_path"]
            and global_overview_publishers == 0
            and ground_truth_algorithm_subscribers == 0
            and not control_topic_present
            and clean_shutdown
        )
        if arguments.expect_no_path_recovery:
            required = (
                required
                and no_path_seen_after_block
                and candidate_replaced_after_no_path
            )
        return 0 if required else 1
    finally:
        if not node_destroyed:
            node.destroy_node()
            rclpy.shutdown()
        if not launch_stopped and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
        if not log_stream.closed:
            log_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
