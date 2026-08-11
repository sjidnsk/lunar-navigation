from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.substitutions import FindPackageShare
from lifecycle_msgs.msg import Transition


def _autostart(node):
    return [
        RegisterEventHandler(
            OnProcessStart(
                target_action=node,
                on_start=[
                    EmitEvent(
                        event=ChangeState(
                            lifecycle_node_matcher=matches_action(node),
                            transition_id=Transition.TRANSITION_CONFIGURE,
                        )
                    )
                ],
            )
        ),
        RegisterEventHandler(
            OnStateTransition(
                target_lifecycle_node=node,
                goal_state="inactive",
                entities=[
                    EmitEvent(
                        event=ChangeState(
                            lifecycle_node_matcher=matches_action(node),
                            transition_id=Transition.TRANSITION_ACTIVATE,
                        )
                    )
                ],
            )
        ),
    ]


def generate_launch_description():
    config_file = PathJoinSubstitution(
        [
            FindPackageShare("lunar_navigation_config"),
            "config",
            "unreal_tcp_wheeled_path_planning.yaml",
        ]
    )
    default_profile = PathJoinSubstitution(
        [
            FindPackageShare("lunar_navigation_config"),
            "config",
            "platform_profiles",
            "wheeled.yaml",
        ]
    )
    server_host = LaunchConfiguration("server_host")
    server_port = LaunchConfiguration("server_port")
    platform_profile = LaunchConfiguration("platform_profile_file")

    bridge = LifecycleNode(
        package="lunar_unreal_tcp_bridge",
        executable="lunar_unreal_tcp_bridge_node",
        name="lunar_unreal_tcp_bridge",
        namespace="",
        output="screen",
        parameters=[
            config_file,
            {
                "use_sim_time": True,
                "server_host": server_host,
                "server_port": server_port,
            },
        ],
    )
    observed_map = LifecycleNode(
        package="lunar_observed_map",
        executable="lunar_observed_map_node",
        name="lunar_observed_map",
        namespace="",
        output="screen",
        parameters=[config_file, {"use_sim_time": True}],
    )
    planner = LifecycleNode(
        package="lunar_planner_ros",
        executable="lunar_planner_node",
        name="lunar_planner",
        namespace="",
        output="screen",
        parameters=[
            config_file,
            {
                "use_sim_time": True,
                "platform_profile_file": platform_profile,
            },
        ],
        remappings=[
            ("/localization/odometry", "/lunar/unreal/wheeled_odometry"),
        ],
    )
    coordinator = LifecycleNode(
        package="lunar_goal_coordinator",
        executable="lunar_goal_coordinator_node",
        name="lunar_goal_coordinator",
        namespace="",
        output="screen",
        parameters=[config_file, {"use_sim_time": True}],
    )
    nodes = [bridge, observed_map, planner, coordinator]

    return LaunchDescription(
        [
            DeclareLaunchArgument("server_host", default_value="192.168.1.20"),
            DeclareLaunchArgument("server_port", default_value="47001"),
            DeclareLaunchArgument(
                "platform_profile_file", default_value=default_profile
            ),
            *nodes,
            *[handler for node in nodes for handler in _autostart(node)],
        ]
    )
