"""Compose the isolated 300 m ROS 2 Jazzy exploration simulation."""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


INTERFACES = {
    "global_map": "/Car/T3/mapping/global_overview",
    "local_map": "/Car/T3/mapping/grid_map",
    "odometry": "/Car/T3/localization/odometry",
    "tf": "/tf",
    "plan_motion": "/Car/T4/plan_motion",
    "planning_diagnostics": "/Car/T4/planning/diagnostics",
    "planning_reference": "/Car/T4/planning/wheeled_reference",
    "task": "/Car/T4/exploration/task",
    "status": "/Car/T4/exploration/status",
    "current_goal": "/Car/T4/exploration/current_goal",
    "frontiers": "/Car/T4/exploration/frontiers",
    "exploration_diagnostics": "/Car/T4/exploration/diagnostics",
    "motion_reference": "/Car/T4/execution/motion_reference",
    "cancel": "/Car/T4/execution/cancel",
    "command": "/Car/T5/Car_Cmd_Vel",
    "fov": "/Car/T4/simulation/sensor_fov",
    "vehicle": "/Car/T4/simulation/vehicle_markers",
    "local_markers": "/Car/T4/simulation/local_map_markers",
    "actual_path": "/Car/T4/simulation/actual_path",
    "planned_path": "/Car/T4/simulation/planned_path",
    "hud": "/Car/T4/simulation/hud",
    "sim_elapsed": "/Car/T4/simulation/sim_elapsed",
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


def _repository_roots() -> list[Path]:
    """Return only concrete Git roots visible to this launch invocation."""
    roots: list[Path] = []
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / ".git").exists():
                if candidate not in roots:
                    roots.append(candidate)
                break
    if not roots:
        roots.append(
            Path(get_package_share_directory("lunar_pure_exploration_sim")).resolve()
        )
    return roots


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_output_dir(output_dir: str, repository_roots: list[Path]) -> Path:
    if not output_dir or not Path(output_dir).is_absolute():
        raise RuntimeError("output_dir must be an absolute external path")
    normalized_output = Path(output_dir).resolve(strict=False)
    if any(_is_within(normalized_output, root.resolve(strict=False))
           for root in repository_roots):
        raise RuntimeError(
            "output_dir must be an absolute external path outside repositories"
        )
    for ancestor in (normalized_output, *normalized_output.parents):
        git_marker = ancestor / ".git"
        if git_marker.is_file() or git_marker.is_dir():
            raise RuntimeError(
                "output_dir must not be inside a Git repository or worktree"
            )
    return normalized_output


def _compose(context, *, rviz_config: str):
    output_dir = LaunchConfiguration("output_dir").perform(context).strip()
    repository_roots = _repository_roots()
    normalized_output = _validate_output_dir(output_dir, repository_roots)

    seed = LaunchConfiguration("seed")
    speed_multiplier = LaunchConfiguration("speed_multiplier")
    start_rviz = LaunchConfiguration("start_rviz")
    planner_share = Path(get_package_share_directory("lunar_pure_planner_ros"))
    explorer_share = Path(get_package_share_directory("lunar_pure_exploration_ros"))

    simulation = Node(
        package="lunar_pure_exploration_sim",
        executable="simulation_node",
        name="lunar_pure_exploration_sim",
        parameters=[{
            "seed": ParameterValue(seed, value_type=int),
            "speed_multiplier": ParameterValue(speed_multiplier, value_type=float),
            "command_topic": "/Car/T5/Car_Cmd_Vel",
            "global_overview_topic": INTERFACES["global_map"],
            "local_grid_map_topic": INTERFACES["local_map"],
            "odometry_topic": INTERFACES["odometry"],
            "tf_topic": INTERFACES["tf"],
            "sensor_fov_topic": INTERFACES["fov"],
            "vehicle_markers_topic": INTERFACES["vehicle"],
            "local_map_markers_topic": INTERFACES["local_markers"],
            "actual_path_topic": INTERFACES["actual_path"],
            "sim_elapsed_topic": INTERFACES["sim_elapsed"],
        }],
        output="screen",
    )
    planner = Node(
        package="lunar_pure_planner_ros",
        executable="lunar_pure_planner_node",
        name="pure_planner",
        parameters=[
            str(planner_share / "config" / "pure_planner.yaml"),
            {
                "platform_type": "wheel",
                "platform_config": str(planner_share / "config" / "wheel.yaml"),
                "wheel_planner_mode": "grid_traversability_v1",
                "rolling_surface_enabled": False,
                "global_map_topic": INTERFACES["global_map"],
                "local_map_topic": INTERFACES["local_map"],
                "odometry_topic": INTERFACES["odometry"],
                "tf_topic": INTERFACES["tf"],
                "action_name": INTERFACES["plan_motion"],
                "diagnostics_topic": INTERFACES["planning_diagnostics"],
                "wheeled_reference_topic": INTERFACES["planning_reference"],
            },
        ],
        output="screen",
    )
    explorer = Node(
        package="lunar_pure_exploration_ros",
        executable="pure_exploration_node",
        name="pure_exploration",
        parameters=[
            str(explorer_share / "config" / "pure_exploration.yaml"),
            {
                "platform_selector": "wheel",
                "platform_config": str(planner_share / "config" / "wheel.yaml"),
                "use_sim_time": False,
                "stop_before_planning": True,
                "global_map_topic": INTERFACES["global_map"],
                "odometry_topic": INTERFACES["odometry"],
                "tf_topic": INTERFACES["tf"],
                "task_topic": INTERFACES["task"],
                "planner_action": INTERFACES["plan_motion"],
                "planner_diagnostics_topic": INTERFACES["planning_diagnostics"],
                "motion_reference_topic": INTERFACES["motion_reference"],
                "execution_cancel_topic": INTERFACES["cancel"],
                **CAPACITIES,
            },
        ],
        output="screen",
    )
    controller = Node(
        package="lunar_pure_wheeled_controller",
        executable="lunar_pure_wheeled_controller_node.py",
        name="lunar_pure_wheeled_controller",
        parameters=[{
            "reference_topic": "/Car/T4/execution/motion_reference",
            "odometry_topic": INTERFACES["odometry"],
            "command_topic": "/Car/T5/Car_Cmd_Vel",
            "execution_cancel_topic": INTERFACES["cancel"],
        }],
        output="screen",
    )
    coordinator = Node(
        package="lunar_pure_exploration_sim",
        executable="run_coordinator",
        name="exploration_run_coordinator",
        parameters=[{
            "seed": ParameterValue(seed, value_type=int),
            "global_overview_topic": INTERFACES["global_map"],
            "local_grid_map_topic": INTERFACES["local_map"],
            "odometry_topic": INTERFACES["odometry"],
            "tf_topic": INTERFACES["tf"],
            "exploration_status_topic": INTERFACES["status"],
            "exploration_task_topic": INTERFACES["task"],
            "planner_action": INTERFACES["plan_motion"],
            "controller_command_topic": "/Car/T5/Car_Cmd_Vel",
        }],
        output="screen",
    )
    recorder = Node(
        package="lunar_pure_exploration_sim",
        executable="run_recorder",
        name="exploration_run_recorder",
        parameters=[{
            "output_dir": str(normalized_output),
            "repository_roots": [str(root) for root in repository_roots],
            "seed": ParameterValue(seed, value_type=int),
            "exploration_status_topic": INTERFACES["status"],
            "odometry_topic": INTERFACES["odometry"],
            "sim_elapsed_topic": INTERFACES["sim_elapsed"],
            "planner_diagnostics_topic": INTERFACES["planning_diagnostics"],
            "exploration_diagnostics_topic": INTERFACES["exploration_diagnostics"],
        }],
        output="screen",
    )
    hud = Node(
        package="lunar_pure_exploration_sim",
        executable="simulation_hud_node",
        name="exploration_simulation_hud",
        parameters=[{
            "exploration_status_topic": INTERFACES["status"],
            "odometry_topic": INTERFACES["odometry"],
            "sim_elapsed_topic": INTERFACES["sim_elapsed"],
            "planner_diagnostics_topic": INTERFACES["planning_diagnostics"],
            "motion_reference_topic": INTERFACES["motion_reference"],
            "hud_topic": INTERFACES["hud"],
            "planned_path_topic": INTERFACES["planned_path"],
        }],
        output="screen",
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="jazzy_300m_exploration_rviz",
        arguments=["-d", rviz_config],
        condition=IfCondition(start_rviz),
        output="screen",
    )
    return [simulation, planner, explorer, controller, coordinator, recorder, hud, rviz]


def generate_launch_description() -> LaunchDescription:
    sim_share = get_package_share_directory("lunar_pure_exploration_sim")
    return LaunchDescription([
        DeclareLaunchArgument("seed", default_value="20260824"),
        DeclareLaunchArgument("speed_multiplier", default_value="20.0"),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        DeclareLaunchArgument("output_dir"),
        OpaqueFunction(
            function=_compose,
            kwargs={"rviz_config": f"{sim_share}/rviz/jazzy_300m_exploration_sim.rviz"},
        ),
    ])
