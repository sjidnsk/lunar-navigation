from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config_file = DeclareLaunchArgument("config_file")
    adapter = Node(
        package="luna_t3_map_adapter",
        executable="luna_t3_map_adapter_node.py",
        name="luna_t3_map_adapter",
        parameters=[LaunchConfiguration("config_file")],
        output="screen",
    )
    return LaunchDescription([config_file, adapter])
