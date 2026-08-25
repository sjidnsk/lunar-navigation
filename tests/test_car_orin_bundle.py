from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "create_car_orin_bundle.py"


def test_generator_creates_minimal_humble_production_bundle(tmp_path: Path) -> None:
    output = tmp_path / "car_orin"
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--source-root",
            str(ROOT),
            "--output",
            str(output),
        ],
        check=True,
    )

    assert {path.name for path in output.iterdir()} == {
        "MANIFEST.sha256",
        "README_DEPLOY.md",
        "config",
        "launch",
        "ros2_ws",
        "scripts",
    }
    assert {path.name for path in (output / "config").iterdir()} == {
        "external_interfaces.yaml",
        "pure_exploration.yaml",
        "pure_planner.yaml",
        "wheel.yaml",
    }
    assert {path.name for path in (output / "launch").iterdir()} == {
        "local_traversability.launch.py",
        "pure_exploration.launch.py",
        "pure_planner.launch.py",
        "rviz_goal_bridge.launch.py",
    }
    assert {path.name for path in (output / "ros2_ws" / "src").iterdir()} == {
        "lunar_planning_msgs",
        "lunar_pure_exploration_core",
        "lunar_pure_exploration_msgs",
        "lunar_pure_exploration_ros",
        "lunar_pure_planner_core",
        "lunar_pure_planner_ros",
        "lunar_pure_wheeled_controller",
    }

    forbidden_components = {
        ".git",
        ".pytest_cache",
        "build",
        "docs",
        "install",
        "log",
        "test",
        "tests",
        "__pycache__",
    }
    for path in output.rglob("*"):
        assert forbidden_components.isdisjoint(path.relative_to(output).parts)
        assert "lunar_pure_exploration_sim" not in path.name
        assert "lunar_surface_demo" not in path.name
        assert "lunar_surface_visualizer" not in path.name
        assert path.suffix not in {".pyc", ".rviz"}

    expected_core_sources = {
        "src/planner.cpp",
        "src/global_goal_feasibility.cpp",
        "src/grid_v1/grid_v1_planner.cpp",
        "src/grid_v1/traversability_map.cpp",
        "src/hierarchical/frame_transform.cpp",
        "src/hierarchical/global_route_planner.cpp",
        "src/hierarchical/reference_composer.cpp",
        "src/hierarchical/surface_portal_set.cpp",
        "src/hierarchical/surface_rolling_session.cpp",
        "src/hierarchical/surface_global_search.cpp",
        "src/hopper/anytime_hopper_planner.cpp",
        "src/hopper/ballistic_envelope.cpp",
        "src/hopper/ballistic_kinematics.cpp",
        "src/legged/anytime_legged_planner.cpp",
        "src/shared/active_planner_cache.cpp",
        "src/shared/anytime_ara_star.cpp",
        "src/shared/cell_area_distance_transform.cpp",
        "src/shared/global_occupancy_projection.cpp",
        "src/shared/goal_distance_field.cpp",
        "src/shared/local_terrain_projection.cpp",
        "src/shared/map_snapshot.cpp",
        "src/shared/obstacle_height_estimator.cpp",
        "src/shared/planning_timing.cpp",
        "src/shared/request_local_start_patch.cpp",
        "src/shared/search_control.cpp",
        "src/wheel/anytime_wheel_planner.cpp",
    }
    core = output / "ros2_ws" / "src" / "lunar_pure_planner_core"
    actual_core_sources = {
        path.relative_to(core).as_posix() for path in core.rglob("*.cpp")
    }
    assert actual_core_sources == expected_core_sources

    ros_package = output / "ros2_ws" / "src" / "lunar_pure_planner_ros"
    for required_source in (
        "src/center_distance_transform.cpp",
        "src/elevation_occupancy.cpp",
        "src/elevation_occupancy_main.cpp",
        "src/elevation_occupancy_node.cpp",
        "src/traversability_input.cpp",
        "src/traversability_qos.cpp",
    ):
        assert (ros_package / required_source).is_file()
    assert not (ros_package / "src" / "lunar_surface_scenario.cpp").exists()
    assert "LUNAR_BUILD_DEMO" in (ros_package / "CMakeLists.txt").read_text()

    exploration_ros = output / "ros2_ws" / "src" / "lunar_pure_exploration_ros"
    for required_source in (
        "src/exploration_node.cpp",
        "src/marker_builder.cpp",
        "src/planner_client.cpp",
        "src/stationary_planning_gate.cpp",
    ):
        assert (exploration_ros / required_source).is_file()
    assert (output / "launch" / "pure_exploration.launch.py").is_file()
    assert (output / "config" / "pure_exploration.yaml").is_file()

    for script_name in ("build.sh", "start_all.sh", "send_goal.sh"):
        script = output / "scripts" / script_name
        assert os.access(script, os.X_OK)
        subprocess.run(["bash", "-n", str(script)], check=True)

    build_script = (output / "scripts" / "build.sh").read_text()
    assert "/opt/ros/humble/setup.bash" in build_script
    assert "-DBUILD_TESTING=OFF" in build_script
    assert "-DLUNAR_BUILD_DEMO=OFF" in build_script
    assert "lunar_pure_exploration_ros" in build_script

    subprocess.run(
        ["sha256sum", "--check", "MANIFEST.sha256"],
        cwd=output,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
