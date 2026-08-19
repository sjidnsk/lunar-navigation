from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode


def generate_launch_description() -> LaunchDescription:
    """Launch only the planner; lifecycle ownership remains with ``luna``."""
    params_file = DeclareLaunchArgument("params_file")
    planner = LifecycleNode(
        package="lunar_planner_ros",
        executable="lunar_planner_node",
        name="lunar_planner",
        parameters=[LaunchConfiguration("params_file")],
        output="screen",
    )
    return LaunchDescription([params_file, planner])
