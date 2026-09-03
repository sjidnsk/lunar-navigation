"""Launch the local-input-only incremental_v2 exploration RViz demo."""

from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _compose(context):
    platform_type = LaunchConfiguration("platform_type").perform(context)
    fine_resolution = float(LaunchConfiguration("fine_resolution_m").perform(context))
    task_size = float(LaunchConfiguration("task_size_m").perform(context))
    start_rviz = LaunchConfiguration("start_rviz").perform(context)
    show_ground_truth = LaunchConfiguration("show_ground_truth").perform(context)
    show_ground_truth_enabled = show_ground_truth.lower() in {"true", "1", "yes"}
    if platform_type not in {"wheel", "legged"}:
        raise RuntimeError("platform_type must be wheel or legged")
    if fine_resolution not in {0.2, 0.1}:
        raise RuntimeError("fine_resolution_m must be 0.2 or 0.1; restart to change it")
    if task_size <= 16.0:
        raise RuntimeError("task_size_m must exceed the 16 m local observation window")

    navigation_share = get_package_share_directory("lunar_incremental_navigation_ros")
    platform_config = f"{navigation_share}/config/{platform_type}.yaml"
    half = task_size / 2.0
    task_message = (
        "{header: {frame_id: map}, task_id: incremental-rviz-demo, command: 1, "
        "boundary: {points: ["
        f"{{x: {-half}, y: {-half}, z: 0.0}}, "
        f"{{x: {half}, y: {-half}, z: 0.0}}, "
        f"{{x: {half}, y: {half}, z: 0.0}}, "
        f"{{x: {-half}, y: {half}, z: 0.0}}"
        "]}}"
    )

    navigator_parameters = {
        "platform_type": platform_type,
        "platform_config": platform_config,
        "coarse_resolution_m": 1.0,
        "local_map_topic": "/planning_demo/grid_map",
        "local_map_qos_reliability": "reliable",
        "local_map_qos_durability": "transient_local",
        "odometry_topic": "/planning_demo/odometry",
        "tf_topic": "/tf",
        "action_name": "/planning_demo/navigation/navigate_to_pose",
        "path_reference_topic": "/planning_demo/planning/path_reference",
        "global_route_topic": "/planning_demo/planning/global_route",
        "diagnostics_topic": "/planning_demo/planning/diagnostics",
        "exploration_map_topic": "/planning_demo/mapping/exploration_map",
        "enable_debug_visualization": True,
        "debug_topic_prefix": "/planning_demo/planning",
        "debug_fine_window_m": 16.0,
    }
    explorer_parameters = {
        "platform_selector": platform_type,
        "platform_config": platform_config,
        "exploration_map_topic": "/planning_demo/mapping/exploration_map",
        "odometry_topic": "/planning_demo/odometry",
        "tf_topic": "/tf",
        "task_topic": "/planning_demo/exploration/task",
        "navigation_action": "/planning_demo/navigation/navigate_to_pose",
        "status_topic": "/planning_demo/exploration/status",
        "task_boundary_topic": "/planning_demo/exploration/task_boundary",
        "task_map_markers_topic": "/planning_demo/exploration/task_map_markers",
        "current_goal_topic": "/planning_demo/exploration/current_goal",
        "frontiers_topic": "/planning_demo/exploration/frontiers",
        "diagnostics_topic": "/planning_demo/exploration/diagnostics",
    }
    return [
        Node(
            package="lunar_incremental_navigation_ros",
            executable="incremental_demo_motion_emulator_node",
            name="incremental_demo_motion_emulator",
            parameters=[{"platform_type": platform_type}],
            output="screen",
        ),
        Node(
            package="lunar_incremental_navigation_ros",
            executable="incremental_demo_scenario_node",
            name="incremental_demo_scenario",
            parameters=[
                {
                    "platform_type": platform_type,
                    "fine_resolution_m": fine_resolution,
                    "task_size_m": task_size,
                    "show_ground_truth": show_ground_truth_enabled,
                }
            ],
            output="screen",
        ),
        Node(
            package="lunar_incremental_navigation_ros",
            executable="lunar_incremental_navigation_node",
            name="incremental_navigation",
            parameters=[navigator_parameters],
            remappings=[
                (
                    "/planning_demo/planning/traversability",
                    "/planning_demo/planning/fine_traversability",
                )
            ],
            output="screen",
        ),
        Node(
            package="lunar_pure_exploration_ros",
            executable="incremental_exploration_node",
            name="incremental_exploration",
            parameters=[explorer_parameters],
            output="screen",
        ),
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "ros2",
                        "topic",
                        "pub",
                        "--once",
                        "--qos-reliability",
                        "reliable",
                        "/planning_demo/exploration/task",
                        "lunar_pure_exploration_msgs/msg/PureExplorationTask",
                        task_message,
                    ],
                    output="screen",
                )
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="incremental_exploration_navigation_rviz",
            arguments=[
                "-d",
                f"{navigation_share}/rviz/incremental_exploration_navigation.rviz",
            ],
            condition=IfCondition(start_rviz),
            output="screen",
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("platform_type", default_value="wheel"),
            DeclareLaunchArgument("fine_resolution_m", default_value="0.2"),
            DeclareLaunchArgument("task_size_m", default_value="300.0"),
            DeclareLaunchArgument("start_rviz", default_value="true"),
            DeclareLaunchArgument("show_ground_truth", default_value="false"),
            OpaqueFunction(function=_compose),
        ]
    )
