"""Launch exactly one exploration-navigation stack selected before startup."""

from __future__ import annotations

from pathlib import Path

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


LEGACY_CAPACITIES = {
    "maximum_position_probes": 8192,
    "maximum_candidate_views": 4096,
    "maximum_collision_work_units": 4194304,
    "maximum_visibility_work_units": 4096,
    "maximum_task_raster_cells": 1048576,
    "maximum_guidance_grid_cells": 1048576,
    "maximum_guidance_work_units": 8388608,
    "maximum_approach_candidates": 4096,
    "maximum_path_preview_poses": 4096,
    "maximum_executable_path_points": 4096,
    "maximum_failure_entries": 2048,
    "maximum_failure_patch_cells_per_entry": 512,
    "maximum_failure_total_patch_cells": 262144,
}


def _load_config(path: str) -> dict[str, object]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {
        "stack", "common", "exploration", "navigation"
    }:
        raise RuntimeError("exploration_navigation config has an invalid shape")
    return document


def _incremental_parameters(
    config: dict[str, object], platform_type: str, platform_config: str, use_sim_time: str
) -> tuple[dict[str, object], dict[str, object]]:
    common = dict(config["common"])
    common.pop("platform_type", None)
    common.pop("frames", None)
    common["platform_config"] = platform_config
    common["use_sim_time"] = ParameterValue(use_sim_time, value_type=bool)

    navigation = {**common, **dict(config["navigation"]), "platform_type": platform_type}
    exploration = {
        **common,
        **dict(config["exploration"]),
        "platform_selector": platform_type,
    }
    return navigation, exploration


def _compose(context):
    stack_mode = LaunchConfiguration("stack_mode").perform(context).strip()
    platform_type = LaunchConfiguration("platform_type").perform(context).strip()
    config_file = LaunchConfiguration("config_file").perform(context).strip()
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)

    navigation_share = Path(
        get_package_share_directory("lunar_incremental_navigation_ros")
    )
    planner_share = Path(get_package_share_directory("lunar_pure_planner_ros"))
    exploration_share = Path(
        get_package_share_directory("lunar_pure_exploration_ros")
    )
    platform_config = str(navigation_share / "config" / f"{platform_type}.yaml")

    if stack_mode == "incremental_v2":
        navigation, exploration = _incremental_parameters(
            _load_config(config_file), platform_type, platform_config, use_sim_time
        )
        return [
            Node(
                package="lunar_incremental_navigation_ros",
                executable="lunar_incremental_navigation_node",
                name="incremental_navigation",
                parameters=[navigation],
                output="screen",
            ),
            Node(
                package="lunar_pure_exploration_ros",
                executable="incremental_exploration_node",
                name="incremental_exploration",
                parameters=[exploration],
                output="screen",
            ),
        ]

    if stack_mode == "legacy":
        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(planner_share / "launch" / "pure_planner.launch.py")
                ),
                launch_arguments={"platform_type": platform_type}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(exploration_share / "launch" / "pure_exploration.launch.py")
                ),
                launch_arguments={
                    "platform_selector": platform_type,
                    "platform_config": str(
                        planner_share / "config" / f"{platform_type}.yaml"
                    ),
                    "use_sim_time": use_sim_time,
                    **{name: str(value) for name, value in LEGACY_CAPACITIES.items()},
                }.items(),
            ),
        ]

    raise RuntimeError("unsupported stack_mode; expected legacy or incremental_v2")


def generate_launch_description() -> LaunchDescription:
    navigation_share = get_package_share_directory("lunar_incremental_navigation_ros")
    return LaunchDescription(
        [
            DeclareLaunchArgument("stack_mode", default_value="incremental_v2"),
            DeclareLaunchArgument("platform_type", default_value="wheel"),
            DeclareLaunchArgument(
                "config_file",
                default_value=f"{navigation_share}/config/exploration_navigation.yaml",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            OpaqueFunction(function=_compose),
        ]
    )
