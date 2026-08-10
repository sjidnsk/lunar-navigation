from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode

from lunar_external_adapter.profile import load_interface_profile
from lunar_exploration_policy.launch_support import consumer_remappings


def _nodes(context):
    model_dir = LaunchConfiguration("model_dir").perform(context)
    platform_profile = LaunchConfiguration("platform_profile_file").perform(context)
    interface_profile = LaunchConfiguration("interface_profile_file").perform(context)
    remappings = list(consumer_remappings(load_interface_profile(interface_profile)))
    return [
        LifecycleNode(
            package="lunar_external_adapter",
            executable="lunar_external_adapter",
            name="lunar_external_adapter",
            namespace="",
            output="screen",
            parameters=[{"interface_profile_file": interface_profile}],
        ),
        LifecycleNode(
            package="lunar_planner_ros",
            executable="lunar_planner_node",
            name="lunar_planner",
            namespace="",
            output="screen",
            remappings=remappings,
            parameters=[{
                "platform_profile_file": platform_profile,
                "global_map_max_age": 5.0,
                "local_map_max_age": 5.0,
                "odometry_max_age": 5.0,
                "localization_status_max_age": 5.0,
                "tf_max_age": 5.0,
                "max_pairwise_skew": 0.2,
            }],
        ),
        LifecycleNode(
            package="lunar_exploration_policy",
            executable="interface_v1_policy",
            name="lunar_interface_v1_policy",
            namespace="",
            output="screen",
            remappings=remappings,
            parameters=[{
                "model_dir": model_dir,
                "platform_profile_file": platform_profile,
                "interface_profile_file": interface_profile,
            }],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("model_dir"),
        DeclareLaunchArgument(
            "platform_profile_file",
            default_value="/etc/lunar_navigation/platform_profile.yaml",
        ),
        DeclareLaunchArgument(
            "interface_profile_file",
            default_value="/etc/lunar_navigation/interface_profile.yaml",
        ),
        OpaqueFunction(function=_nodes),
    ])
