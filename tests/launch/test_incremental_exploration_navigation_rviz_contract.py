"""Contracts and no-RViz probes for the isolated incremental_v2 demo."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "launch/incremental_exploration_navigation_rviz.launch.py"
RVIZ = ROOT / "rviz/incremental_exploration_navigation.rviz"
PACKAGE = ROOT / "ros2_ws/src/lunar_incremental_navigation_ros"
SCENARIO_HEADER = (
    PACKAGE
    / "include/lunar_incremental_navigation_ros/incremental_demo_scenario.hpp"
)
SCENARIO = PACKAGE / "src/incremental_demo_scenario.cpp"
SCENARIO_NODE = PACKAGE / "src/incremental_demo_scenario_node.cpp"
MOTION_NODE = PACKAGE / "src/incremental_demo_motion_emulator_node.cpp"
CMAKE = PACKAGE / "CMakeLists.txt"
PROBE = Path(__file__).with_name("incremental_exploration_navigation_probe.py")

PUBLIC_ARGUMENTS = {
    "platform_type": "wheel",
    "fine_resolution_m": "0.2",
    "task_size_m": "300.0",
    "sensor_range_m": "10.0",
    "sensor_fov_deg": "120.0",
    "angular_speed_radps": "1.0",
    "start_rviz": "true",
    "show_ground_truth": "false",
}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _launch_arguments() -> dict[str, str]:
    tree = ast.parse(_source(LAUNCH))
    arguments: dict[str, str] = {}
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Name) or call.func.id != "DeclareLaunchArgument":
            continue
        assert call.args and isinstance(call.args[0], ast.Constant)
        name = call.args[0].value
        default = next(
            item.value.value
            for item in call.keywords
            if item.arg == "default_value" and isinstance(item.value, ast.Constant)
        )
        arguments[str(name)] = str(default)
    return arguments


def _node_executables() -> set[str]:
    tree = ast.parse(_source(LAUNCH))
    executables: set[str] = set()
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Name) or call.func.id != "Node":
            continue
        for keyword in call.keywords:
            if keyword.arg == "executable" and isinstance(keyword.value, ast.Constant):
                executables.add(str(keyword.value.value))
    return executables


def test_launch_exposes_the_demo_observation_and_turning_arguments() -> None:
    assert _launch_arguments() == PUBLIC_ARGUMENTS
    assert "publish_global_elevation" not in _source(LAUNCH)


def test_launch_uses_real_incremental_nodes_and_only_demo_topics() -> None:
    text = _source(LAUNCH)
    assert text.count('"--once"') == 1
    assert {
        "incremental_demo_scenario_node",
        "incremental_demo_motion_emulator_node",
        "lunar_incremental_navigation_node",
        "incremental_exploration_node",
        "rviz2",
    } == _node_executables()
    for topic in (
        "/planning_demo/grid_map",
        "/planning_demo/odometry",
        "/planning_demo/exploration/task",
        "/planning_demo/mapping/exploration_map",
        "/planning_demo/navigation/navigate_to_pose",
        "/planning_demo/planning/path_reference",
        "/planning_demo/planning/global_route",
        "/tf",
    ):
        assert topic in text
    for forbidden in (
        "/Car/T5/Car_Cmd_Vel",
        "/Car/T3/mapping/global_overview",
        "publish_global_elevation",
        "global_map_topic",
    ):
        assert forbidden not in text


def test_scenario_publishes_only_rolling_local_elevation_evidence() -> None:
    header = _source(SCENARIO_HEADER)
    scenario = _source(SCENARIO)
    node = _source(SCENARIO_NODE)
    combined = "\n".join((header, scenario, node))
    assert "IncrementalDemoScenario" in header
    assert "MakeLocalObservation" in header
    assert 'layers = {"elevation"}' in scenario
    assert '"/planning_demo/grid_map"' in node
    assert '"/planning_demo/odometry"' in node
    assert "MakeLocalObservation" in node
    assert "map -> odom" in node
    assert '"/planning_demo/local_window"' in node
    assert "fine_resolution_m == 0.2" in scenario
    assert "fine_resolution_m == 0.1" in scenario
    assert "publish_global_elevation" not in combined
    assert "global_overview" not in combined


def test_no_path_probe_uses_the_scenario_owned_recovery_barrier() -> None:
    node = _source(SCENARIO_NODE)
    probe = _source(PROBE)

    assert "LUNAR_DEMO_REQUIRE_NO_PATH_RECOVERY" in node
    assert "ApplyNoPathRecoveryBarrier" in node
    assert "no_path_recovery_barrier_injected_" in node
    assert "no_path_recovery_barrier_cycles_remaining_" in node
    assert 'launch_environment["LUNAR_DEMO_REQUIRE_NO_PATH_RECOVERY"] = "1"' in probe
    assert 'create_publisher(\n        GridMap,\n        "/planning_demo/grid_map"' not in probe


def test_ground_truth_is_visualization_only_and_never_a_production_input() -> None:
    launch = _source(LAUNCH)
    scenario_node = _source(SCENARIO_NODE)
    motion = _source(MOTION_NODE)
    assert '"/planning_demo/ground_truth"' in scenario_node
    assert "/planning_demo/ground_truth" in _source(RVIZ)
    assert 'LaunchConfiguration("show_ground_truth")' in launch
    assert '"show_ground_truth": show_ground_truth_enabled' in launch
    assert "ground_truth" not in motion

    navigator_start = launch.index('executable="lunar_incremental_navigation_node"')
    explorer_start = launch.index('executable="incremental_exploration_node"')
    rviz_start = launch.index('executable="rviz2"')
    assert "ground_truth" not in launch[navigator_start:explorer_start]
    assert "ground_truth" not in launch[explorer_start:rviz_start]


def test_motion_emulator_consumes_only_active_path_reference() -> None:
    text = _source(MOTION_NODE)
    assert text.count("create_subscription") == 1
    assert text.count("create_publisher") == 1
    assert "lunar_planning_msgs::msg::PathReference" in text
    assert "PathReference::ACTIVE" in text
    assert '"/planning_demo/planning/path_reference"' in text
    assert '"/planning_demo/odometry"' in text
    assert '"/planning_demo/robot_trace"' not in text
    assert '"/planning_demo/visualization/active_path"' not in text
    assert "/Car/T5/Car_Cmd_Vel" not in text
    assert "Twist" not in text
    assert "InitialScanPath" in text
    assert "follower_.SetPath(InitialScanPath())" in text

    scenario = _source(SCENARIO_NODE)
    assert '"/planning_demo/robot_trace"' in scenario
    assert '"/planning_demo/visualization/active_path"' in scenario


def test_rviz_shows_the_required_layers_without_search_animation() -> None:
    document = yaml.safe_load(_source(RVIZ))
    manager = document["Visualization Manager"]
    assert manager["Global Options"]["Fixed Frame"] == "map"
    displays = {display["Name"]: display for display in manager["Displays"]}
    expected_topics = {
        "Task boundary": "/planning_demo/exploration/task_boundary",
        "Three-state task map": "/planning_demo/exploration/task_map_markers",
        "Local observation window": "/planning_demo/local_window",
        "Frontiers": "/planning_demo/exploration/frontiers",
        "Current goal": "/planning_demo/exploration/current_goal",
        "Fine traversability": "/planning_demo/planning/fine_traversability",
        "Global route": "/planning_demo/planning/global_route",
        "Active path": "/planning_demo/visualization/active_path",
        "Robot and trace": "/planning_demo/robot_trace",
        "HUD": "/planning_demo/hud",
        "Ground truth (visualization only)": "/planning_demo/ground_truth",
    }
    assert expected_topics.items() <= {
        (name, display["Topic"])
        for name, display in displays.items()
        if "Topic" in display
    }
    assert displays["Current goal"]["Color"] == "255; 0; 255"
    assert displays["Global route"]["Color"] == "60; 120; 255"
    assert displays["Global route"]["Line Width"] < displays["Active path"]["Line Width"]
    assert displays["Active path"]["Color"] == "255; 140; 0"
    assert displays["Ground truth (visualization only)"]["Enabled"] is True
    assert displays["Ground truth (visualization only)"]["Value"] is True
    text = _source(RVIZ).lower()
    assert "open set" not in text
    assert "closed set" not in text
    assert "a* open" not in text
    assert "a* closed" not in text


def test_package_installs_both_demo_executables_launch_and_rviz() -> None:
    text = _source(CMAKE)
    assert "incremental_demo_scenario_node" in text
    assert "incremental_demo_motion_emulator_node" in text
    assert 'PATTERN "incremental_exploration_navigation_rviz.launch.py"' in text
    assert 'PATTERN "incremental_exploration_navigation.rviz"' in text


def test_probe_cancels_the_active_action_before_stopping_launch() -> None:
    text = _source(PROBE)
    assert "task.command = PureExplorationTask.START" not in text
    assert "PureExplorationTask.CANCEL" in text
    assert "GoalStatusArray" in text
    assert '"navigation_action_quiescent"' in text
    assert '"path_invalidated_after_cancel"' in text
    assert '"clean_shutdown"' in text
    assert "Asked to publish result for goal that does not exist" in text
    assert "math.pi / angular_speed" in _source(LAUNCH)


def _live_available() -> bool:
    if shutil.which("ros2") is None:
        return False
    result = subprocess.run(
        ["ros2", "pkg", "executables", "lunar_incremental_navigation_ros"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return (
        result.returncode == 0
        and "incremental_demo_scenario_node" in result.stdout
        and "incremental_demo_motion_emulator_node" in result.stdout
    )


@pytest.mark.parametrize(
    ("platform", "fine_resolution"),
    [("wheel", "0.2"), ("wheel", "0.1"), ("legged", "0.2")],
)
def test_no_rviz_closed_loop(platform: str, fine_resolution: str) -> None:
    if not _live_available():
        pytest.skip("freshly built demo executables are not sourced")
    environment = os.environ.copy()
    environment["ROS_LOCALHOST_ONLY"] = "1"
    environment["ROS_DOMAIN_ID"] = str(80 + os.getpid() % 120)
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--platform",
            platform,
            "--fine-resolution",
            fine_resolution,
            "--task-size",
            "24.0",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=75,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"global_overview_publishers": 0' in result.stdout
    assert '"ground_truth_algorithm_subscribers": 0' in result.stdout
    assert '"control_topic_present": false' in result.stdout


def test_no_path_feedback_replaces_the_candidate() -> None:
    if not _live_available():
        pytest.skip("freshly built demo executables are not sourced")
    environment = os.environ.copy()
    environment["ROS_LOCALHOST_ONLY"] = "1"
    environment["ROS_DOMAIN_ID"] = str(201 + os.getpid() % 25)
    environment["LUNAR_DEMO_REQUIRE_NO_PATH_RECOVERY"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--platform",
            "wheel",
            "--fine-resolution",
            "0.2",
            "--task-size",
            "24.0",
            "--expect-no-path-recovery",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=75,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"no_path_observed": true' in result.stdout
    assert '"candidate_replaced": true' in result.stdout
