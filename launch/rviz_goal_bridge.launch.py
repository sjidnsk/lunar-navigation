"""Launch the RViz 2D-goal to pure-planner Action bridge."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Create the bridge node without adding map or timestamp admission gates."""
    return LaunchDescription(
        [
            DeclareLaunchArgument("environment_mode", default_value="2"),
            DeclareLaunchArgument("goal_topic", default_value="/Car/T4/rviz_goal"),
            DeclareLaunchArgument("action_name", default_value="/Car/T4/plan_motion"),
            Node(
                package="lunar_pure_planner_ros",
                executable="lunar_rviz_goal_bridge",
                name="rviz_goal_bridge",
                namespace="",
                parameters=[
                    {
                        "environment_mode": LaunchConfiguration("environment_mode"),
                        "goal_topic": LaunchConfiguration("goal_topic"),
                        "action_name": LaunchConfiguration("action_name"),
                    }
                ],
                output="screen",
            ),
        ]
    )
