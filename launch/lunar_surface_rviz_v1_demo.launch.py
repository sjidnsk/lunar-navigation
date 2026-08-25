"""Launch the global-plus-local grid-traversability V1 RViz demo."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Start the synthetic map, V1 planner, manual goal bridge, and RViz."""
    share = get_package_share_directory("lunar_pure_planner_ros")
    seed = LaunchConfiguration("seed")
    start_rviz = LaunchConfiguration("start_rviz")
    planner_parameters = {
        "platform_type": "wheel",
        "wheel_planner_mode": "grid_traversability_v1",
        "global_map_topic": "/lunar_demo/global_overview",
        "local_map_topic": "/lunar_demo/grid_map",
        "odometry_topic": "/lunar_demo/odometry",
        "tf_topic": "/tf",
        "action_name": "/lunar_demo/plan_motion",
        "diagnostics_topic": "/lunar_demo/diagnostics",
        "wheeled_reference_topic": "/lunar_demo/wheeled_reference",
        "wheeled_path_topic": "/lunar_demo/wheeled_path",
        "wheeled_global_path_topic": "/lunar_demo/global_path",
        "wheeled_timed_path_topic": "/lunar_demo/wheeled_path_timing",
    }
    return LaunchDescription(
        [
            DeclareLaunchArgument("seed", default_value="20260823"),
            DeclareLaunchArgument("start_rviz", default_value="true"),
            Node(
                package="lunar_pure_planner_ros",
                executable="lunar_surface_demo_node",
                name="lunar_surface_demo_v1",
                parameters=[{"seed": seed}],
                remappings=[
                    (
                        "/lunar_demo/rviz_goal",
                        "/lunar_demo/ignored_auto_goal",
                    )
                ],
                output="screen",
            ),
            Node(
                package="lunar_pure_planner_ros",
                executable="lunar_surface_visualizer_node",
                name="lunar_surface_visualizer_v1",
                parameters=[{"seed": seed}],
                output="screen",
            ),
            Node(
                package="lunar_pure_planner_ros",
                executable="lunar_local_traversability_node",
                name="lunar_demo_traversability_v1",
                parameters=[
                    {
                        "platform_type": "wheel",
                        "local_map_topic": "/lunar_demo/grid_map",
                        "traversability_topic": "/lunar_demo/traversability",
                        "input_qos_reliability": "reliable",
                        "input_qos_durability": "volatile",
                    }
                ],
                output="screen",
            ),
            Node(
                package="lunar_pure_planner_ros",
                executable="lunar_pure_planner_node",
                name="pure_planner_v1",
                parameters=[f"{share}/config/pure_planner.yaml", planner_parameters],
                output="screen",
            ),
            Node(
                package="lunar_pure_planner_ros",
                executable="lunar_rviz_goal_bridge",
                name="rviz_goal_bridge_v1",
                parameters=[
                    {
                        "environment_mode": 1,
                        "mission_id": "lunar-demo-v1",
                        "mission_revision": 1,
                        "goal_topic": "/lunar_demo/rviz_goal",
                        "action_name": "/lunar_demo/plan_motion",
                    }
                ],
                output="screen",
            ),
            Node(
                condition=IfCondition(start_rviz),
                package="rviz2",
                executable="rviz2",
                name="lunar_surface_rviz_v1",
                arguments=["-d", f"{share}/rviz/lunar_surface_demo.rviz"],
                output="screen",
            ),
        ]
    )
