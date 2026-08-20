"""Launch Task3 input adapters and the lifecycle-owned Luna planner."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description() -> LaunchDescription:
    params_file = DeclareLaunchArgument("params_file")
    task3_config_file = DeclareLaunchArgument("task3_config_file")
    controller_enabled = DeclareLaunchArgument("controller_enabled", default_value="false")
    adapter_parameters = [LaunchConfiguration("task3_config_file")]
    return LaunchDescription([
        params_file,
        task3_config_file,
        controller_enabled,
        Node(
            package="luna_t3_map_adapter",
            executable="luna_t3_map_adapter_node.py",
            name="luna_t3_map_adapter",
            parameters=adapter_parameters,
            output="screen",
        ),
        Node(
            package="luna_t3_map_adapter",
            executable="luna_t3_local_map_adapter_node.py",
            name="luna_t3_local_map_adapter",
            parameters=adapter_parameters,
            output="screen",
        ),
        Node(
            package="luna_t3_map_adapter",
            executable="luna_t3_localization_status_adapter_node.py",
            name="luna_t3_localization_status_adapter",
            parameters=adapter_parameters,
            output="screen",
        ),
        LifecycleNode(
            package="lunar_planner_ros",
            executable="lunar_planner_node",
            name="lunar_planner",
            parameters=[LaunchConfiguration("params_file")],
            output="screen",
        ),
        Node(
            package="luna_wheeled_controller",
            executable="luna_task_execution_coordinator_node.py",
            name="luna_task_execution_coordinator",
            parameters=[LaunchConfiguration("params_file")],
            condition=IfCondition(LaunchConfiguration("controller_enabled")),
            output="screen",
        ),
        Node(
            package="luna_wheeled_controller",
            executable="luna_wheeled_controller_node.py",
            name="luna_wheeled_controller",
            parameters=[LaunchConfiguration("params_file")],
            condition=IfCondition(LaunchConfiguration("controller_enabled")),
            output="screen",
        ),
    ])
