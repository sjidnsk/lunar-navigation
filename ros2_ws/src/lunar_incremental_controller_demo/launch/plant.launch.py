from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('case', default_value='forward'),
        DeclareLaunchArgument('map_size_m', default_value='32.0'),
        Node(package='lunar_incremental_controller_demo', executable='vehicle_sim',
             output='screen', parameters=[{'case': LaunchConfiguration('case'),
                                          'map_size_m': LaunchConfiguration('map_size_m')}]),
    ])
