"""Publish the cached local traversability visualization for RViz."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Run the local-only traversability publisher without input freshness gates."""
    return LaunchDescription(
        [
            DeclareLaunchArgument("platform_type", default_value="wheel"),
            DeclareLaunchArgument(
                "local_map_topic", default_value="/Car/T3/mapping/grid_map"
            ),
            DeclareLaunchArgument(
                "traversability_topic",
                default_value="/Car/T4/planning/local_traversability",
            ),
            DeclareLaunchArgument(
                "input_qos_reliability", default_value="reliable"
            ),
            DeclareLaunchArgument(
                "input_qos_durability", default_value="transient_local"
            ),
            Node(
                package="lunar_pure_planner_ros",
                executable="lunar_local_traversability_node",
                name="local_traversability",
                namespace="",
                parameters=[
                    {
                        "platform_type": LaunchConfiguration("platform_type"),
                        "local_map_topic": LaunchConfiguration("local_map_topic"),
                        "traversability_topic": LaunchConfiguration(
                            "traversability_topic"
                        ),
                        "input_qos_reliability": LaunchConfiguration(
                            "input_qos_reliability"
                        ),
                        "input_qos_durability": LaunchConfiguration(
                            "input_qos_durability"
                        ),
                    }
                ],
                output="screen",
            ),
        ]
    )
