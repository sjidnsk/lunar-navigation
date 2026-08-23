"""Static contract for the isolated 300 m Jazzy exploration composition."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "launch" / "jazzy_300m_exploration_sim.launch.py"
RVIZ = ROOT / "rviz" / "jazzy_300m_exploration_sim.rviz"
CMAKE = ROOT / "ros2_ws" / "src" / "lunar_pure_exploration_sim" / "CMakeLists.txt"

EXPECTED_EXECUTABLES = {
    "simulation_node",
    "lunar_pure_planner_node",
    "pure_exploration_node",
    "lunar_pure_wheeled_controller_node.py",
    "run_coordinator",
    "run_recorder",
    "simulation_hud_node",
    "rviz2",
}

EXACT_TOPICS = {
    "/Car/T3/mapping/global_overview",
    "/Car/T3/mapping/grid_map",
    "/Car/T3/localization/odometry",
    "/Car/T4/plan_motion",
    "/Car/T4/planning/diagnostics",
    "/Car/T4/exploration/task",
    "/Car/T4/exploration/status",
    "/Car/T4/exploration/current_goal",
    "/Car/T4/exploration/frontiers",
    "/Car/T4/exploration/diagnostics",
    "/Car/T4/execution/motion_reference",
    "/Car/T4/execution/cancel",
    "/Car/T4/simulation/sensor_fov",
    "/Car/T4/simulation/vehicle_markers",
    "/Car/T4/simulation/local_map_markers",
    "/Car/T4/simulation/actual_path",
    "/Car/T4/simulation/planned_path",
    "/Car/T4/simulation/hud",
    "/Car/T4/simulation/sim_elapsed",
    "/Car/T5/Car_Cmd_Vel",
    "/tf",
}

CAPACITIES = {
    "maximum_position_probes": 8192,
    "maximum_candidate_views": 4096,
    "maximum_collision_work_units": 4194304,
    "maximum_visibility_work_units": 4096,
    "maximum_path_preview_poses": 4096,
    "maximum_executable_path_points": 4096,
    "maximum_failure_entries": 2048,
    "maximum_failure_patch_cells_per_entry": 512,
    "maximum_failure_total_patch_cells": 262144,
}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _keyword_strings(source: str, function_name: str, keyword: str) -> list[str]:
    tree = ast.parse(source)
    values: list[str] = []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Name) or call.func.id != function_name:
            continue
        for item in call.keywords:
            if item.arg == keyword and isinstance(item.value, ast.Constant):
                if isinstance(item.value.value, str):
                    values.append(item.value.value)
    return values


def test_static_launch_declares_required_arguments_and_absolute_output_guard() -> None:
    source = _source(LAUNCH)
    assert 'DeclareLaunchArgument("seed", default_value="20260824")' in source
    assert 'DeclareLaunchArgument("speed_multiplier", default_value="20.0")' in source
    assert 'DeclareLaunchArgument("start_rviz", default_value="true")' in source
    assert 'DeclareLaunchArgument("output_dir")' in source
    assert "Path(output_dir).is_absolute()" in source
    assert "output_dir must be an absolute external path" in source


def test_static_launch_starts_exact_graph_and_conditionally_starts_rviz() -> None:
    source = _source(LAUNCH)
    assert set(_keyword_strings(source, "Node", "executable")) == EXPECTED_EXECUTABLES
    assert "IfCondition(start_rviz)" in source
    assert "LifecycleNode" not in source


def test_static_launch_freezes_interfaces_controller_override_and_capacities() -> None:
    source = _source(LAUNCH)
    for topic in EXACT_TOPICS:
        assert topic in source
    assert '"reference_topic": "/Car/T4/execution/motion_reference"' in source
    assert source.count('"command_topic": "/Car/T5/Car_Cmd_Vel"') == 2
    assert '"platform_type": "wheel"' in source
    assert '"speed_multiplier": ParameterValue(speed_multiplier, value_type=float)' in source
    for name, value in CAPACITIES.items():
        assert f'"{name}": {value}' in source
    assert "/lunar_demo/" not in source


def test_static_rviz_uses_only_standard_displays_for_approved_topics() -> None:
    source = _source(RVIZ)
    assert "Fixed Frame: map" in source
    assert "grid_map" not in source.lower()
    assert "rviz_grid_map_plugins" not in source
    for display_class, topic in (
        ("rviz_default_plugins/Map", "/Car/T3/mapping/global_overview"),
        ("rviz_default_plugins/Path", "/Car/T4/simulation/actual_path"),
        ("rviz_default_plugins/Path", "/Car/T4/simulation/planned_path"),
        ("rviz_default_plugins/MarkerArray", "/Car/T4/exploration/frontiers"),
        ("rviz_default_plugins/Pose", "/Car/T4/exploration/current_goal"),
        ("rviz_default_plugins/MarkerArray", "/Car/T4/simulation/vehicle_markers"),
        ("rviz_default_plugins/Marker", "/Car/T4/simulation/sensor_fov"),
        ("rviz_default_plugins/MarkerArray", "/Car/T4/simulation/local_map_markers"),
        ("rviz_default_plugins/MarkerArray", "/Car/T4/simulation/hud"),
    ):
        assert display_class in source
        assert f"Topic: {topic}" in source
    assert "/lunar_demo/" not in source


def test_static_sim_package_installs_launch_and_rviz_assets() -> None:
    source = _source(CMAKE)
    assert '"${LUNAR_PURE_SIM_LAUNCH_SOURCE_DIR}/jazzy_300m_exploration_sim.launch.py"' in source
    assert 'DESTINATION share/${PROJECT_NAME}/launch' in source
    assert '"${LUNAR_PURE_SIM_RVIZ_SOURCE_DIR}/jazzy_300m_exploration_sim.rviz"' in source
    assert 'DESTINATION share/${PROJECT_NAME}/rviz' in source
