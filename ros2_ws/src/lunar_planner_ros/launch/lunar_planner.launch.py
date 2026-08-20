from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description() -> LaunchDescription:
    """Launch only the planner; lifecycle ownership remains with ``luna``."""
    params_file = DeclareLaunchArgument("params_file")
    controller_enabled = DeclareLaunchArgument("controller_enabled", default_value="false")
    planner = LifecycleNode(
        package="lunar_planner_ros",
        executable="lunar_planner_node",
        name="lunar_planner",
        parameters=[LaunchConfiguration("params_file")],
        output="screen",
    )
    return LaunchDescription([
        params_file,
        controller_enabled,
        planner,
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
