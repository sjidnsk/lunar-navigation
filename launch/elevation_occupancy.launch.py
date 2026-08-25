"""Convert an elevation-only GridMap into the planner's two-layer input map."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "input_topic", default_value="/Car/T3/mapping/elevation_grid_map"
        ),
        DeclareLaunchArgument(
            "output_topic", default_value="/Car/T3/mapping/grid_map"
        ),
        DeclareLaunchArgument("elevation_layer", default_value="elevation"),
        DeclareLaunchArgument("occupancy_layer", default_value="occupancy"),
        DeclareLaunchArgument("overwrite_existing_occupancy", default_value="false"),
        DeclareLaunchArgument("input_qos_reliability", default_value="reliable"),
        DeclareLaunchArgument("input_qos_durability", default_value="transient_local"),
        Node(
            package="lunar_pure_planner_ros",
            executable="lunar_elevation_occupancy_node",
            name="lunar_elevation_occupancy",
            parameters=[{
                "input_topic": LaunchConfiguration("input_topic"),
                "output_topic": LaunchConfiguration("output_topic"),
                "elevation_layer": LaunchConfiguration("elevation_layer"),
                "occupancy_layer": LaunchConfiguration("occupancy_layer"),
                "overwrite_existing_occupancy": LaunchConfiguration(
                    "overwrite_existing_occupancy"
                ),
                "input_qos_reliability": LaunchConfiguration("input_qos_reliability"),
                "input_qos_durability": LaunchConfiguration("input_qos_durability"),
            }],
            output="screen",
        ),
    ])
