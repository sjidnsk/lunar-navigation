"""Launch the RViz PoseStamped to incremental NavigateToPose bridge."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("goal_topic", default_value="/Car/T4/rviz_goal"),
            DeclareLaunchArgument(
                "action_name",
                default_value="/Car/T4/navigation/navigate_to_pose",
            ),
            DeclareLaunchArgument("expected_frame", default_value="map"),
            Node(
                package="lunar_incremental_navigation_ros",
                executable="lunar_incremental_rviz_goal_bridge",
                name="incremental_rviz_goal_bridge",
                parameters=[
                    {
                        "goal_topic": LaunchConfiguration("goal_topic"),
                        "action_name": LaunchConfiguration("action_name"),
                        "expected_frame": LaunchConfiguration("expected_frame"),
                    }
                ],
                output="screen",
            ),
        ]
    )
