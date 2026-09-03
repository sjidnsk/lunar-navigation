"""Static contracts for the isolated incremental exploration-navigation stack."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEGACY_INTERFACES = ROOT / "config" / "external_interfaces.yaml"
INCREMENTAL_INTERFACES = ROOT / "config" / "incremental_navigation_interfaces.yaml"
STACK_CONFIG = ROOT / "config" / "exploration_navigation.yaml"
NAVIGATION_SOURCE = (
    ROOT
    / "ros2_ws/src/lunar_incremental_navigation_ros/src/incremental_navigation_node.cpp"
)
EXPLORATION_SOURCE = (
    ROOT
    / "ros2_ws/src/lunar_pure_exploration_ros/src/incremental_exploration_node.cpp"
)
MESSAGE_ROOT = ROOT / "ros2_ws/src/lunar_planning_msgs"


def _yaml(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def test_legacy_global_overview_contract_remains_an_occupancy_grid() -> None:
    """Reject an accidental legacy ABI migration while adding incremental_v2."""
    legacy = _yaml(LEGACY_INTERFACES)
    global_overview = legacy["topics"]["global_overview"]

    assert global_overview["name"] == "/Car/T3/mapping/global_overview"
    assert global_overview["type"] == "nav_msgs/msg/OccupancyGrid"


def test_incremental_interfaces_are_local_only_and_define_durable_exploration_map() -> None:
    """Reject a global-map dependency or a lossy exploration-map handoff."""
    interfaces = _yaml(INCREMENTAL_INTERFACES)

    assert set(interfaces) == {"schema_version", "inputs", "action", "outputs"}
    assert set(interfaces["inputs"]) == {"grid_map", "odometry", "tf"}
    assert interfaces["inputs"]["grid_map"] == {
        "name": "/Car/T3/mapping/grid_map",
        "type": "grid_map_msgs/msg/GridMap",
        "owner": "external",
        "frame": "odom",
        "required_layers": ["elevation"],
    }
    assert interfaces["inputs"]["odometry"]["type"] == "nav_msgs/msg/Odometry"
    assert interfaces["inputs"]["tf"]["type"] == "tf2_msgs/msg/TFMessage"
    assert interfaces["action"] == {
        "name": "/Car/T4/navigation/navigate_to_pose",
        "type": "lunar_planning_msgs/action/NavigateToPose",
        "owner": "lunar_incremental_navigation_ros",
    }
    assert set(interfaces["outputs"]) == {
        "exploration_map",
        "path_reference",
        "global_route",
        "diagnostics",
    }
    assert interfaces["outputs"]["exploration_map"] == {
        "name": "/Car/T4/mapping/exploration_map",
        "type": "nav_msgs/msg/OccupancyGrid",
        "owner": "lunar_incremental_navigation_ros",
        "frame": "map",
        "values": {"unknown": -1, "free": 0, "occupied": 100},
        "qos": {
            "reliability": "reliable",
            "durability": "transient_local",
            "history": "keep_last",
            "depth": 1,
        },
    }
    text = INCREMENTAL_INTERFACES.read_text(encoding="utf-8").lower()
    assert "global_overview" not in text
    assert "current_pose" not in text


def test_single_stack_config_has_only_shared_mode_platform_and_node_defaults() -> None:
    """Reject scattered overrides, copied physical parameters, or a fine-resolution knob."""
    config = _yaml(STACK_CONFIG)

    assert set(config) == {"stack", "common", "exploration", "navigation"}
    assert config["stack"] == {"mode": "incremental_v2"}
    assert config["common"]["platform_type"] == "wheel"
    assert config["navigation"]["coarse_resolution_m"] == 1.0
    text = STACK_CONFIG.read_text(encoding="utf-8").lower()
    assert "fine_resolution" not in text
    assert "motion_primitives" not in text
    assert "footprint_xy_m" not in text


def test_incremental_source_keeps_navigation_and_exploration_interfaces_separate() -> None:
    """Reject cross-owned GridMap, frontier/task, or reference interfaces."""
    navigation = NAVIGATION_SOURCE.read_text(encoding="utf-8")
    exploration = EXPLORATION_SOURCE.read_text(encoding="utf-8")

    assert "create_subscription<grid_map_msgs::msg::GridMap>" in navigation
    assert "create_subscription<grid_map_msgs::msg::GridMap>" not in exploration
    assert "create_subscription<Task>" not in navigation
    assert "frontier" not in navigation.lower()
    assert "MotionReference" not in exploration
    assert "PathReference" not in exploration
    assert "create_publisher<lunar_planning_msgs::msg::PathReference>" in navigation


def test_incremental_action_is_additive_and_legacy_plan_motion_is_unchanged() -> None:
    """Reject an ABI edit to PlanMotion or a partial NavigateToPose definition."""
    legacy = _lines(MESSAGE_ROOT / "action/PlanMotion.action")
    incremental = _lines(MESSAGE_ROOT / "action/NavigateToPose.action")

    assert "uint8 environment_mode" in legacy
    assert "float64 target_x_m" not in legacy
    assert incremental[:4] == [
        "float64 target_x_m",
        "float64 target_y_m",
        "bool has_target_yaw",
        "float64 target_yaw_rad",
    ]
    assert "uint64 last_segment_revision" in incremental
    assert "uint64 active_segment_revision" in incremental
