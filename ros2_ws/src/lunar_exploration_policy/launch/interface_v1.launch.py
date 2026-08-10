from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("model_dir"),
        DeclareLaunchArgument("platform_profile_file"),
        DeclareLaunchArgument("interface_profile_file"),
        DeclareLaunchArgument("repository_root"),
    ]
    node = LifecycleNode(
        package="lunar_exploration_policy",
        executable="interface_v1_policy",
        name="lunar_interface_v1_policy",
        namespace="",
        output="screen",
        parameters=[
            {
                "model_dir": LaunchConfiguration("model_dir"),
                "platform_profile_file": LaunchConfiguration("platform_profile_file"),
                "interface_profile_file": LaunchConfiguration("interface_profile_file"),
                "repository_root": LaunchConfiguration("repository_root"),
            }
        ],
    )
    return LaunchDescription([*arguments, node])
