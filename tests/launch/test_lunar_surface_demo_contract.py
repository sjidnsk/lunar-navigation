"""Source contract checks for the isolated lunar-surface RViz demo."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "launch" / "lunar_surface_rviz_demo.launch.py"
RVIZ = ROOT / "rviz" / "lunar_surface_demo.rviz"
DEMO_NODE = ROOT / "ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_demo_node.cpp"
VISUALIZER_NODE = ROOT / "ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_visualizer_node.cpp"
REPORTER_NODE = ROOT / "ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_reporter.cpp"


def test_demo_launch_isolated_from_production_action() -> None:
    text = LAUNCH.read_text(encoding="utf-8")
    assert '"/lunar_demo/plan_motion"' in text
    assert '"/Car/T4/plan_motion"' not in text
    assert '"/lunar_demo/global_overview"' in text
    assert '"/lunar_demo/grid_map"' in text


def test_rviz_config_exposes_goal_path_and_lunar_layers() -> None:
    text = RVIZ.read_text(encoding="utf-8")
    for topic in (
        "/lunar_demo/start_pose",
        "/lunar_demo/rviz_goal",
        "/lunar_demo/global_path",
        "/lunar_demo/wheeled_path",
        "/lunar_demo/classic_global_map",
        "/lunar_demo/classic_local_map",
        "/lunar_demo/pose_markers",
        "/lunar_demo/local_window",
        "/lunar_demo/rover_markers",
    ):
        assert topic in text


def test_demo_computes_and_displays_wheel_traversability() -> None:
    launch_text = LAUNCH.read_text(encoding="utf-8")
    demo_text = DEMO_NODE.read_text(encoding="utf-8")
    assert 'executable="lunar_local_traversability_node"' in launch_text
    assert launch_text.count('executable="lunar_local_traversability_node"') == 2
    assert '"local_map_topic": "/lunar_demo/grid_map"' in launch_text
    assert '"traversability_topic": "/lunar_demo/traversability"' in launch_text
    assert '"local_map_topic": "/lunar_demo/global_grid_map"' in launch_text
    assert '"traversability_topic": "/lunar_demo/global_traversability"' in launch_text
    assert '"/lunar_demo/global_grid_map"' in demo_text
    assert '"input_qos_reliability": "reliable"' in launch_text
    assert '"input_qos_durability": "volatile"' in launch_text

    rviz_config = yaml.safe_load(RVIZ.read_text(encoding="utf-8"))
    displays = rviz_config["Visualization Manager"]["Displays"]
    traversability = next(
        display for display in displays
        if display.get("Name") == "Local obstacles and traversability"
    )
    assert traversability["Class"] == "rviz_default_plugins/MarkerArray"
    assert traversability["Topic"] == "/lunar_demo/classic_local_map"


def test_demo_can_select_legged_grid_v1_and_publish_rviz_paths() -> None:
    launch_text = LAUNCH.read_text(encoding="utf-8")

    assert 'DeclareLaunchArgument("platform_type", default_value="wheel")' in launch_text
    assert 'platform_type = LaunchConfiguration("platform_type")' in launch_text
    assert '"platform_type": platform_type' in launch_text
    assert launch_text.count('"platform_type": platform_type') == 4
    assert '"legged_global_mode": "grid_traversability_v1"' in launch_text
    assert '"legged_global_path_topic": "/lunar_demo/legged_global_path"' in launch_text
    assert '"legged_local_path_topic": "/lunar_demo/legged_path"' in launch_text


def test_legged_rviz_paths_are_displayed_and_reporter_uses_authorized_segments() -> None:
    rviz_text = RVIZ.read_text(encoding="utf-8")
    demo_text = DEMO_NODE.read_text(encoding="utf-8")
    reporter_text = REPORTER_NODE.read_text(encoding="utf-8")

    assert "/lunar_demo/legged_global_path" in rviz_text
    assert "/lunar_demo/legged_path" in rviz_text
    assert '"/lunar_demo/legged_path"' in demo_text
    assert '"/lunar_demo/legged_global_path"' in reporter_text
    assert '"/lunar_demo/plan_segment"' in reporter_text
    assert "DemoPlanSegment::EXECUTE" in reporter_text
    assert '"/lunar_demo/legged_path"' not in reporter_text


def test_demo_publishes_legged_body_height_in_odometry() -> None:
    launch_text = LAUNCH.read_text(encoding="utf-8")
    demo_text = DEMO_NODE.read_text(encoding="utf-8")

    assert '"platform_type": platform_type' in launch_text
    assert 'declare_parameter<std::string>("platform_type", "wheel")' in demo_text
    assert "constexpr double kLeggedNominalBodyHeightM = 0.33" in demo_text
    assert 'platform_type_ == "legged" ? kLeggedNominalBodyHeightM : 0.0' in demo_text


def test_demo_uses_a_sixty_four_metre_local_map_and_manual_goal_by_default() -> None:
    launch_text = LAUNCH.read_text(encoding="utf-8")
    demo_text = DEMO_NODE.read_text(encoding="utf-8")

    assert 'DeclareLaunchArgument("auto_goal", default_value="false")' in launch_text
    assert '"auto_goal": ParameterValue(auto_goal, value_type=bool)' in launch_text
    assert "constexpr std::size_t kLocalWidth = 320U" in demo_text
    assert "constexpr std::size_t kLocalHeight = 320U" in demo_text
    assert "constexpr double kLocalResolutionM = 0.2" in demo_text


def test_rviz_distinguishes_global_and_local_paths() -> None:
    rviz_config = yaml.safe_load(RVIZ.read_text(encoding="utf-8"))
    displays = rviz_config["Visualization Manager"]["Displays"]
    by_name = {display["Name"]: display for display in displays}

    assert by_name["Global path"]["Topic"] == "/lunar_demo/global_path"
    assert by_name["Local path segment"]["Topic"] == "/lunar_demo/wheeled_path"
    assert by_name["Global path"]["Color"] != by_name["Local path segment"]["Color"]
    assert by_name["Global traversability and obstacles"]["Topic"] == "/lunar_demo/classic_global_map"
    assert by_name["Local obstacles and traversability"]["Topic"] == "/lunar_demo/classic_local_map"


def test_rviz_uses_shader_safe_classic_obstacle_and_traversability_markers() -> None:
    rviz_config = yaml.safe_load(RVIZ.read_text(encoding="utf-8"))
    displays = rviz_config["Visualization Manager"]["Displays"]
    by_name = {display["Name"]: display for display in displays}

    global_obstacles = by_name["Global traversability and obstacles"]
    local_overlay = by_name["Local obstacles and traversability"]

    assert global_obstacles["Class"] == "rviz_default_plugins/MarkerArray"
    assert global_obstacles["Topic"] == "/lunar_demo/classic_global_map"
    assert local_overlay["Class"] == "rviz_default_plugins/MarkerArray"
    assert local_overlay["Topic"] == "/lunar_demo/classic_local_map"
    assert all(display["Class"] != "rviz_default_plugins/Map" for display in displays)
    assert all(display["Class"] != "grid_map_rviz_plugin/GridMap" for display in displays)
    assert all(display["Class"] != "rviz_default_plugins/Odometry" for display in displays)
    assert "Lunar elevation" not in by_name


def test_rviz_opens_on_local_detail_and_keeps_a_one_kilometre_saved_view() -> None:
    rviz_config = yaml.safe_load(RVIZ.read_text(encoding="utf-8"))
    panels = {panel["Name"]: panel for panel in rviz_config["Panels"]}
    views = rviz_config["Visualization Manager"]["Views"]
    current = views["Current"]
    saved = {view["Name"]: view for view in views["Saved"]}

    assert panels["Displays"]["Class"] == "rviz_common/Displays"
    assert panels["Views"]["Class"] == "rviz_common/Views"
    assert current["Scale"] == 8
    assert current["X"] == -349.5
    assert current["Y"] == 0.5
    assert saved["Global 1 km overview"]["Scale"] == 0.6
    assert saved["Global 1 km overview"]["X"] == 0
    assert saved["Global 1 km overview"]["Y"] == 0
    assert saved["Local 64 m detail"]["Scale"] == 8


def test_launch_starts_concise_reporter() -> None:
    text = LAUNCH.read_text(encoding="utf-8")
    assert 'executable="lunar_surface_reporter_node"' in text
    assert 'name="lunar_surface_reporter"' in text


def test_rviz_uses_a_colorblind_safe_academic_palette() -> None:
    rviz_config = yaml.safe_load(RVIZ.read_text(encoding="utf-8"))
    manager = rviz_config["Visualization Manager"]
    displays = manager["Displays"]

    by_name = {display["Name"]: display for display in displays}
    traversability = by_name["Local obstacles and traversability"]
    global_path = by_name["Global path"]
    local_path = by_name["Local path segment"]
    rover = by_name["Rover marker"]

    assert traversability["Class"] == "rviz_default_plugins/MarkerArray"
    assert global_path["Color"] == "230; 159; 0"
    assert local_path["Color"] == "86; 180; 233"
    assert global_path["Line Style"] == "Billboards"
    assert local_path["Line Style"] == "Billboards"
    assert global_path["Color"] != local_path["Color"]
    assert rover["Class"] == "rviz_default_plugins/MarkerArray"
    assert by_name["Global traversability and obstacles"]["Enabled"] is True
    assert manager["Global Options"]["Background Color"] == "32; 34; 37"


def test_rviz_inputs_are_compatible_and_empty_failed_paths_are_not_forwarded() -> None:
    demo_text = DEMO_NODE.read_text(encoding="utf-8")
    visualizer_text = VISUALIZER_NODE.read_text(encoding="utf-8")

    assert "rclcpp::QoS{10}.reliable()" in demo_text
    assert "!message->header.frame_id.empty()" in visualizer_text
    assert "!message->poses.empty()" in visualizer_text


def test_visualizer_labels_the_rover_at_kilometre_map_scale() -> None:
    text = VISUALIZER_NODE.read_text(encoding="utf-8")
    assert 'rover.ns = "lunar_surface_rover"' in text
    assert "visualization_msgs::msg::Marker::CYLINDER" in text
    assert "visualization_msgs::msg::Marker::TEXT_VIEW_FACING" in text
    assert 'rover_label.text = "Current rover position"' in text
    assert text.count("rover.scale.x = 5.0") == 2
    assert "label.scale.z = 2.0" in text
    assert "rover_label.scale.z = 2.0" in text
    assert "marker.scale.x = 6.0" in text
    assert "marker.scale.z = 3.0" in text
