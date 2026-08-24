"""Source contract checks for the isolated lunar-surface RViz demo."""

from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "launch" / "lunar_surface_rviz_demo.launch.py"
RVIZ = ROOT / "rviz" / "lunar_surface_demo.rviz"
DEMO_NODE = ROOT / "ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_demo_node.cpp"
VISUALIZER_NODE = ROOT / "ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_visualizer_node.cpp"
PACKAGE_XML = ROOT / "ros2_ws/src/lunar_pure_planner_ros/package.xml"


def test_demo_launch_isolated_from_production_action() -> None:
    text = LAUNCH.read_text(encoding="utf-8")
    assert '"/lunar_demo/plan_motion"' in text
    assert '"/Car/T4/plan_motion"' not in text
    assert '"/lunar_demo/global_overview"' in text
    assert '"/lunar_demo/grid_map"' in text


def test_rviz_config_exposes_goal_path_and_lunar_layers() -> None:
    text = RVIZ.read_text(encoding="utf-8")
    for topic in (
        "/lunar_demo/rviz_goal",
        "/lunar_demo/path",
        "/lunar_demo/global_overview",
        "/lunar_demo/terrain_markers",
    ):
        assert topic in text


def test_demo_computes_and_displays_wheel_traversability() -> None:
    launch_text = LAUNCH.read_text(encoding="utf-8")
    assert 'executable="lunar_local_traversability_node"' in launch_text
    assert '"local_map_topic": "/lunar_demo/grid_map"' in launch_text
    assert '"traversability_topic": "/lunar_demo/traversability"' in launch_text
    assert '"input_qos_reliability": "reliable"' in launch_text
    assert '"input_qos_durability": "volatile"' in launch_text

    rviz_config = yaml.safe_load(RVIZ.read_text(encoding="utf-8"))
    displays = rviz_config["Visualization Manager"]["Displays"]
    traversability = next(
        display for display in displays if display.get("Name") == "Wheel traversability"
    )
    assert traversability["Class"] == "grid_map_rviz_plugin/GridMap"
    assert traversability["Topic"] == "/lunar_demo/traversability"
    assert traversability["Height Transformer"] == "Flat"
    assert traversability["Color Transformer"] == "IntensityLayer"
    assert traversability["Color Layer"] == "traversability"

    dependencies = ET.parse(PACKAGE_XML).getroot().findall("exec_depend")
    grid_map_dependency = next(
        element for element in dependencies if element.text == "grid_map_rviz_plugin"
    )
    assert grid_map_dependency.attrib["condition"] == "$ROS_DISTRO == 'jazzy'"


def test_rviz_uses_a_colorblind_safe_academic_palette() -> None:
    rviz_config = yaml.safe_load(RVIZ.read_text(encoding="utf-8"))
    manager = rviz_config["Visualization Manager"]
    displays = manager["Displays"]

    by_name = {display["Name"]: display for display in displays}
    traversability = by_name["Wheel traversability"]
    path = by_name["Planned path"]
    rover = by_name["Rover"]
    goal = by_name["Default goal"]

    assert traversability["Max Color"] == "0; 158; 115"
    assert traversability["Min Color"] == "213; 94; 0"
    assert traversability["Alpha"] == 0.72
    assert path["Color"] == "0; 114; 178"
    assert path["Line Width"] == 0.18
    assert path["Color"] != traversability["Max Color"]
    assert rover["Shape"]["Color"] == "204; 121; 167"
    assert goal["Color"] == "230; 159; 0"
    assert by_name["Lunar obstacles"]["Alpha"] == 0.2
    assert manager["Global Options"]["Background Color"] == "32; 34; 37"


def test_rviz_inputs_are_compatible_and_empty_failed_paths_are_not_forwarded() -> None:
    demo_text = DEMO_NODE.read_text(encoding="utf-8")
    visualizer_text = VISUALIZER_NODE.read_text(encoding="utf-8")

    assert "rclcpp::QoS{10}.reliable()" in demo_text
    assert "!message->header.frame_id.empty()" in visualizer_text
    assert "!message->poses.empty()" in visualizer_text
