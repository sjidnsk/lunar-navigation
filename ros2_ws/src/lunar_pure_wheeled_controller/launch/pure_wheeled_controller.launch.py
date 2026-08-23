"""Launch the isolated pure wheeled controller."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


_PARAMETERS = {
    "path_topic": "/Car/T4/planning/wheeled_path",
    "odometry_topic": "/Car/T3/localization/odometry",
    "command_topic": "/Car/T5/Car_Cmd_Vel",
    "execution_cancel_topic": "/Car/T4/execution/cancel",
    "control_rate_hz": "20.0",
    "lookahead_m": "0.5",
    "max_linear_mps": "0.2",
    "max_angular_radps": "0.5",
    "goal_position_tolerance_m": "0.2",
    "goal_yaw_tolerance_rad": "0.2",
    "max_cross_track_error_m": "1.0",
    "spin_kp": "1.5",
    "translation_epsilon_m": "0.001",
}


def generate_launch_description() -> LaunchDescription:
    """Expose the adapter topics and tracking-policy parameters."""
    return LaunchDescription([
        *(DeclareLaunchArgument(name, default_value=default) for name, default in _PARAMETERS.items()),
        Node(
            package="lunar_pure_wheeled_controller",
            executable="lunar_pure_wheeled_controller_node.py",
            name="lunar_pure_wheeled_controller",
            parameters=[{name: LaunchConfiguration(name) for name in _PARAMETERS}],
        ),
    ])
