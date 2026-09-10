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
EXPLORATION_HEADER = (
    ROOT
    / "ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/incremental_exploration_node.hpp"
)
EXPLORATION_MAP_PROJECTOR = (
    ROOT
    / "ros2_ws/src/lunar_incremental_navigation_ros/src/exploration_map_projector.cpp"
)
EXPLORATION_MAP_PUBLISHER = (
    ROOT
    / "ros2_ws/src/lunar_incremental_navigation_ros/src/incremental_map_publisher.cpp"
)
EXPLORATION_MANIFEST = ROOT / "ros2_ws/src/lunar_pure_exploration_ros/package.xml"
EXPLORATION_CMAKE = ROOT / "ros2_ws/src/lunar_pure_exploration_ros/CMakeLists.txt"
NAVIGATION_CMAKE = ROOT / "ros2_ws/src/lunar_incremental_navigation_ros/CMakeLists.txt"
PLANNER_CMAKE = ROOT / "ros2_ws/src/lunar_pure_planner_ros/CMakeLists.txt"
STATE_ADAPTER_SOURCE = (
    ROOT / "ros2_ws/src/lunar_incremental_navigation_ros/src/state_adapter.cpp"
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
    assert interfaces["inputs"]["odometry"] == {
        "name": "/Car/T3/localization/odometry",
        "type": "nav_msgs/msg/Odometry",
        "owner": "external",
        "frame": "odom",
        "child_frame": "base_link",
    }
    assert interfaces["inputs"]["tf"]["type"] == "tf2_msgs/msg/TFMessage"
    assert interfaces["action"] == {
        "name": "/Car/T4/navigation/navigate_to_pose",
        "type": "lunar_planning_msgs/action/NavigateToPose",
        "owner": "lunar_incremental_navigation_ros",
    }
    assert set(interfaces["outputs"]) == {
        "exploration_map",
        "path_reference",
        "local_path",
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
    assert interfaces["outputs"]["local_path"] == {
        "name": "/Car/T4/planning/local_path",
        "type": "nav_msgs/msg/Path",
        "owner": "lunar_incremental_navigation_ros",
        "frame": "map",
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
    common = config["common"]
    assert common["platform_type"] == "wheel"
    assert common["coarse_resolution_m"] == 1.0
    assert common["navigation_action"] == "/Car/T4/navigation/navigate_to_pose"
    assert common["local_map_topic"] == "/Car/T3/mapping/grid_map"
    assert common["exploration_map_topic"] == "/Car/T4/mapping/exploration_map"
    assert common["odometry_topic"] == "/Car/T3/localization/odometry"
    assert common["tf_topic"] == "/tf"
    assert common["frames"] == {
        "map": "map",
        "odom": "odom",
        "base_link": "base_link",
    }
    assert common["exploration_map_qos"] == {
        "reliability": "reliable",
        "durability": "transient_local",
        "history": "keep_last",
        "depth": 1,
    }
    assert not {"coarse_resolution_m", "local_map_topic", "exploration_map_topic", "action_name", "navigation_action"} & set(config["exploration"])
    assert not {"coarse_resolution_m", "local_map_topic", "exploration_map_topic", "action_name", "navigation_action"} & set(config["navigation"])
    assert config["navigation"]["local_window_size_m"] == 64.0
    assert config["navigation"]["local_path_topic"] == "/Car/T4/planning/local_path"
    assert "occupied_threshold" not in config["exploration"]
    text = STACK_CONFIG.read_text(encoding="utf-8").lower()
    assert "fine_resolution" not in text
    assert "motion_primitives" not in text
    assert "footprint_xy_m" not in text
    navigation_source = NAVIGATION_SOURCE.read_text(encoding="utf-8")
    assert 'declare_parameter<double>("local_window_size_m", 64.0)' in navigation_source


def test_incremental_explorer_consumes_only_the_published_three_state_map() -> None:
    """Reject a threshold-based reinterpretation of the v2 exploration map."""
    config = _yaml(STACK_CONFIG)
    source = EXPLORATION_SOURCE.read_text(encoding="utf-8")
    header = EXPLORATION_HEADER.read_text(encoding="utf-8")

    assert config["exploration"].get("occupied_threshold") is None
    assert "occupied_threshold" not in source
    assert "occupied_threshold" not in header
    assert "-1, 0, or 100" in source


def test_incremental_odometry_uses_configured_base_frame() -> None:
    """Reject a wheel-footprint frame from leaking into the external pose contract."""
    adapter = STATE_ADAPTER_SOURCE.read_text(encoding="utf-8")

    assert 'odometry.child_frame_id != base_frame' in adapter
    header = (STATE_ADAPTER_SOURCE.parent.parent / "include/lunar_incremental_navigation_ros/state_adapter.hpp").read_text(encoding="utf-8")
    assert 'base_frame = "base_link"' in header
    assert "base_footprint" not in adapter


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


def test_stack_config_launch_and_cpp_keep_the_exploration_map_contract_closed() -> None:
    """Reject endpoint, QoS, value, publisher, or consumer drift across stack layers."""
    config = _yaml(STACK_CONFIG)
    interfaces = _yaml(INCREMENTAL_INTERFACES)
    common = config["common"]
    navigation = NAVIGATION_SOURCE.read_text(encoding="utf-8")
    exploration = EXPLORATION_SOURCE.read_text(encoding="utf-8")
    projector = EXPLORATION_MAP_PROJECTOR.read_text(encoding="utf-8")
    publisher = EXPLORATION_MAP_PUBLISHER.read_text(encoding="utf-8")

    assert common["exploration_map_topic"] == interfaces["outputs"]["exploration_map"]["name"]
    assert common["navigation_action"] == interfaces["action"]["name"]
    assert common["odometry_topic"] == interfaces["inputs"]["odometry"]["name"]
    assert common["tf_topic"] == interfaces["inputs"]["tf"]["name"]
    assert common["local_map_topic"] == interfaces["inputs"]["grid_map"]["name"]
    assert common["exploration_map_qos"] == interfaces["outputs"]["exploration_map"]["qos"]
    assert '"/Car/T4/mapping/exploration_map"' in navigation
    assert '"/Car/T4/mapping/exploration_map"' in exploration
    assert '"/Car/T4/navigation/navigate_to_pose"' in navigation
    assert '"/Car/T4/navigation/navigate_to_pose"' in exploration
    assert "std::vector<std::int8_t>(coarse_geometry.CellCount(), -1)" in projector
    assert "result.data[output_index] = 0;" in projector
    assert "result.data[output_index] = 100;" in projector
    assert "rclcpp::KeepLast{1}" in publisher
    assert ".reliable().transient_local()" in publisher
    assert publisher.count("create_publisher<nav_msgs::msg::OccupancyGrid>") == 1
    assert navigation.count(
        "exploration_map_publisher(node, parameters.exploration_map_topic)"
    ) == 1
    assert exploration.count("create_subscription<nav_msgs::msg::OccupancyGrid>(") == 1
    assert "rclcpp::QoS{1}.reliable().transient_local()" in exploration


def test_stack_config_and_launch_exclude_legacy_or_pose_topic_inputs() -> None:
    """Reject an incremental_v2 dependency on a legacy map or standalone pose topic."""
    stack = STACK_CONFIG.read_text(encoding="utf-8").lower()
    launch = (ROOT / "launch/exploration_navigation.launch.py").read_text(encoding="utf-8").lower()

    for forbidden in ("global_overview", "current_pose"):
        assert forbidden not in stack
        assert forbidden not in launch


def test_top_level_launch_keeps_legacy_independent_and_owns_its_default_config() -> None:
    """Reject a rollback that needs the incremental package or its installed YAML."""
    manifest = EXPLORATION_MANIFEST.read_text(encoding="utf-8")
    exploration_cmake = EXPLORATION_CMAKE.read_text(encoding="utf-8")
    navigation_cmake = NAVIGATION_CMAKE.read_text(encoding="utf-8")
    planner_cmake = PLANNER_CMAKE.read_text(encoding="utf-8")

    assert "<exec_depend>lunar_incremental_navigation_ros</exec_depend>" not in manifest
    assert "<exec_depend>python3-yaml</exec_depend>" in manifest
    assert "exploration_navigation.yaml" in exploration_cmake
    assert "exploration_navigation.launch.py" in exploration_cmake
    assert 'PATTERN "exploration_navigation.yaml"' not in navigation_cmake
    assert 'PATTERN "exploration_navigation.launch.py"' not in navigation_cmake
    assert 'PATTERN "exploration_navigation.yaml" EXCLUDE' in planner_cmake
    assert 'PATTERN "exploration_navigation.launch.py" EXCLUDE' in planner_cmake


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
