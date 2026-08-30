"""Launch an isolated 1 km by 1 km lunar-surface planning demo for RViz."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """Start test-only scenario, planner, RViz bridge, and visualization."""
    share = get_package_share_directory("lunar_pure_planner_ros")
    seed = LaunchConfiguration("seed")
    start_rviz = LaunchConfiguration("start_rviz")
    auto_goal = LaunchConfiguration("auto_goal")
    demo_parameters = {
        "platform_type": "wheel",
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
        "rolling_surface_enabled": True,
        "rolling_horizon_m": 12.0,
    }
    return LaunchDescription(
        [
            DeclareLaunchArgument("seed", default_value="20260823"),
            DeclareLaunchArgument("start_rviz", default_value="true"),
            DeclareLaunchArgument("auto_goal", default_value="false"),
            Node(
                package="lunar_pure_planner_ros",
                executable="lunar_surface_demo_node",
                name="lunar_surface_demo",
                parameters=[{
                    "seed": seed,
                    "auto_goal": ParameterValue(auto_goal, value_type=bool),
                }],
                output="screen",
            ),
            Node(
                package="lunar_pure_planner_ros",
                executable="lunar_surface_visualizer_node",
                name="lunar_surface_visualizer",
                parameters=[{"seed": seed}],
                output="screen",
            ),
            Node(
                package="lunar_pure_planner_ros",
                executable="lunar_local_traversability_node",
                name="lunar_demo_traversability",
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
                executable="lunar_local_traversability_node",
                name="lunar_demo_global_traversability",
                parameters=[
                    {
                        "platform_type": "wheel",
                        "local_map_topic": "/lunar_demo/global_grid_map",
                        "traversability_topic": "/lunar_demo/global_traversability",
                        "input_qos_reliability": "reliable",
                        "input_qos_durability": "transient_local",
                    }
                ],
                output="screen",
            ),
            Node(
                package="lunar_pure_planner_ros",
                executable="lunar_pure_planner_node",
                name="pure_planner",
                parameters=[f"{share}/config/pure_planner.yaml", demo_parameters],
                output="screen",
            ),
            Node(
                package="lunar_pure_planner_ros",
                executable="lunar_rviz_goal_bridge",
                name="rviz_goal_bridge",
                parameters=[
                    {
                        "environment_mode": 1,
                        "mission_id": "lunar-demo",
                        "mission_revision": 1,
                        "goal_topic": "/lunar_demo/rviz_goal",
                        "start_topic": "/lunar_demo/accepted_start",
                        "action_name": "/lunar_demo/plan_motion",
                    }
                ],
                output="screen",
            ),
            Node(
                package="lunar_pure_planner_ros",
                executable="lunar_surface_reporter_node",
                name="lunar_surface_reporter",
                output="screen",
            ),
            Node(
                condition=IfCondition(start_rviz),
                package="rviz2",
                executable="rviz2",
                name="lunar_surface_rviz",
                arguments=["-d", f"{share}/rviz/lunar_surface_demo.rviz"],
                output="screen",
            ),
        ]
    )
