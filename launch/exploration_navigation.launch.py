"""Launch exactly one exploration-navigation stack selected before startup."""

from __future__ import annotations

from pathlib import Path

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
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


def _require_common(config: dict[str, object]) -> dict[str, object]:
    common = config["common"]
    if not isinstance(common, dict) or common.get("frames") != {
        "map": "map", "odom": "odom", "base_link": "base_link"
    }:
        raise RuntimeError("exploration_navigation common frame contract is invalid")
    if common.get("local_map_qos") != {
        "reliability": "reliable", "durability": "transient_local"
    } or common.get("exploration_map_qos") != {
        "reliability": "reliable",
        "durability": "transient_local",
        "history": "keep_last",
        "depth": 1,
    }:
        raise RuntimeError("exploration_navigation common QoS contract is invalid")
    return common


def _legacy_common(config: dict[str, object]) -> dict[str, object]:
    common = config["common"]
    if not isinstance(common, dict):
        raise RuntimeError("exploration_navigation common defaults are invalid")
    return common


def _resolved_value(override: str, configured: object, name: str) -> str:
    value = override or configured
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"exploration_navigation {name} is invalid")
    return value


def _resolved_bool(override: str, configured: object) -> str:
    if override:
        return override
    if not isinstance(configured, bool):
        raise RuntimeError("exploration_navigation use_sim_time is invalid")
    return "true" if configured else "false"


def _incremental_parameters(
    config: dict[str, object], common: dict[str, object], platform_type: str,
    platform_config: str, use_sim_time: str,
) -> tuple[dict[str, object], dict[str, object]]:
    navigation = {
        **dict(config["navigation"]),
        "platform_type": platform_type,
        "platform_config": platform_config,
        "coarse_resolution_m": common["coarse_resolution_m"],
        "local_map_topic": common["local_map_topic"],
        "local_map_qos_reliability": common["local_map_qos"]["reliability"],
        "local_map_qos_durability": common["local_map_qos"]["durability"],
        "odometry_topic": common["odometry_topic"],
        "tf_topic": common["tf_topic"],
        "action_name": common["navigation_action"],
        "exploration_map_topic": common["exploration_map_topic"],
        "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
    }
    exploration = {
        **dict(config["exploration"]),
        "platform_selector": platform_type,
        "platform_config": platform_config,
        "odometry_topic": common["odometry_topic"],
        "tf_topic": common["tf_topic"],
        "navigation_action": common["navigation_action"],
        "exploration_map_topic": common["exploration_map_topic"],
        "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
    }
    return navigation, exploration


def _compose(context):
    requested_stack_mode = LaunchConfiguration("stack_mode").perform(context).strip()
    start_navigation = LaunchConfiguration("start_navigation")
    start_exploration = LaunchConfiguration("start_exploration")
    if requested_stack_mode == "legacy":
        # An explicit rollback must not need the optional incremental package
        # or its YAML. The legacy launch files retain their own defaults.
        stack_mode = "legacy"
        platform_type = (
            LaunchConfiguration("platform_type").perform(context).strip() or "wheel"
        )
        use_sim_time = (
            LaunchConfiguration("use_sim_time").perform(context).strip() or "false"
        )
    else:
        config_file = LaunchConfiguration("config_file").perform(context).strip()
        config = _load_config(config_file)
        stack = config["stack"]
        if not isinstance(stack, dict):
            raise RuntimeError("exploration_navigation stack defaults are invalid")
        stack_mode = _resolved_value(
            requested_stack_mode, stack.get("mode"), "stack_mode",
        )
        if stack_mode not in {"legacy", "incremental_v2"}:
            raise RuntimeError("unsupported stack_mode; expected legacy or incremental_v2")

        common = _legacy_common(config)
        platform_type = _resolved_value(
            LaunchConfiguration("platform_type").perform(context).strip(),
            common.get("platform_type"), "platform_type",
        )
        use_sim_time = _resolved_bool(
            LaunchConfiguration("use_sim_time").perform(context).strip(),
            common.get("use_sim_time"),
        )

    if stack_mode == "legacy":
        planner_share = Path(get_package_share_directory("lunar_pure_planner_ros"))
        exploration_share = Path(
            get_package_share_directory("lunar_pure_exploration_ros")
        )
        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(planner_share / "launch" / "pure_planner.launch.py")
                ),
                launch_arguments={"platform_type": platform_type}.items(),
                condition=IfCondition(start_navigation),
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
                condition=IfCondition(start_exploration),
            ),
        ]

    if stack_mode == "incremental_v2":
        common = _require_common(config)
        navigation_share = Path(
            get_package_share_directory("lunar_incremental_navigation_ros")
        )
        platform_config = str(
            navigation_share / "config" / f"{platform_type}.yaml"
        )
        navigation, exploration = _incremental_parameters(
            config, common, platform_type, platform_config, use_sim_time
        )
        return [
            Node(
                package="lunar_incremental_navigation_ros",
                executable="lunar_incremental_navigation_node",
                name="incremental_navigation",
                parameters=[navigation],
                condition=IfCondition(start_navigation),
                output="screen",
            ),
            Node(
                package="lunar_pure_exploration_ros",
                executable="incremental_exploration_node",
                name="incremental_exploration",
                parameters=[exploration],
                condition=IfCondition(start_exploration),
                output="screen",
            ),
        ]

    raise AssertionError("validated stack mode was not dispatched")


def generate_launch_description() -> LaunchDescription:
    exploration_share = get_package_share_directory("lunar_pure_exploration_ros")
    return LaunchDescription(
        [
            DeclareLaunchArgument("stack_mode", default_value=""),
            DeclareLaunchArgument("platform_type", default_value=""),
            DeclareLaunchArgument(
                "config_file",
                default_value=f"{exploration_share}/config/exploration_navigation.yaml",
            ),
            DeclareLaunchArgument("use_sim_time", default_value=""),
            DeclareLaunchArgument("start_navigation", default_value="true"),
            DeclareLaunchArgument("start_exploration", default_value="true"),
            OpaqueFunction(function=_compose),
        ]
    )
