"""Launch the isolated, non-lifecycle pure planner action server."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Create the one-node pure planner launch description."""
    package_share = get_package_share_directory("lunar_pure_planner_ros")
    params_file = LaunchConfiguration("params_file")
    platform_type = LaunchConfiguration("platform_type")
    wheel_planner_mode = LaunchConfiguration("wheel_planner_mode")
    rolling_surface_enabled = LaunchConfiguration("rolling_surface_enabled")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=f"{package_share}/config/pure_planner.yaml",
            ),
            DeclareLaunchArgument("platform_type", default_value="wheel"),
            DeclareLaunchArgument(
                "wheel_planner_mode", default_value="legacy_certified"
            ),
            DeclareLaunchArgument("rolling_surface_enabled", default_value="false"),
            Node(
                package="lunar_pure_planner_ros",
                executable="lunar_pure_planner_node",
                name="pure_planner",
                namespace="",
                parameters=[
                    params_file,
                    {
                        "platform_type": platform_type,
                        "wheel_planner_mode": wheel_planner_mode,
                        "rolling_surface_enabled": rolling_surface_enabled,
                    },
                ],
                output="screen",
            ),
        ]
    )
