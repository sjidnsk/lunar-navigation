"""Source contract checks for the isolated lunar-surface RViz demo."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "launch" / "lunar_surface_rviz_demo.launch.py"
RVIZ = ROOT / "rviz" / "lunar_surface_demo.rviz"
DEMO_NODE = ROOT / "ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_demo_node.cpp"
VISUALIZER_NODE = ROOT / "ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_visualizer_node.cpp"


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


def test_rviz_inputs_are_compatible_and_empty_failed_paths_are_not_forwarded() -> None:
    demo_text = DEMO_NODE.read_text(encoding="utf-8")
    visualizer_text = VISUALIZER_NODE.read_text(encoding="utf-8")

    assert "rclcpp::QoS{10}.reliable()" in demo_text
    assert "!message->path_preview.header.frame_id.empty()" in visualizer_text
    assert "!message->path_preview.poses.empty()" in visualizer_text
