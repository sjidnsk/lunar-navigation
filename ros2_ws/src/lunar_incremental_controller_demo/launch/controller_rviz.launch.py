"""Production incremental planning and execution in an isolated simulated vehicle."""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def compose(context):
    prefix = '/lunar_demo/controller'
    scene = LaunchConfiguration('case').perform(context)
    nav_share = get_package_share_directory('lunar_incremental_navigation_ros')
    demo_share = get_package_share_directory('lunar_incremental_controller_demo')
    config = nav_share + '/config/wheel.yaml'
    navigator = {
        'platform_type': 'wheel', 'platform_config': config,
        'local_map_topic': prefix + '/grid_map',
        'odometry_topic': prefix + '/odometry', 'tf_topic': '/tf',
        'action_name': prefix + '/navigate_to_pose',
        'path_reference_topic': prefix + '/path_reference',
        'local_path_topic': prefix + '/local_path',
        'global_route_topic': prefix + '/global_route',
        'diagnostics_topic': prefix + '/diagnostics',
        'exploration_map_topic': prefix + '/exploration_map',
        'enable_tracking_feedback': True,
        'tracking_status_topic': prefix + '/tracking_status',
        'enable_debug_visualization': True, 'debug_topic_prefix': prefix,
        'debug_fine_window_m': 32.0,
    }
    controller = {
        'input_mode': 'incremental_reference', 'platform_config': config,
        'path_reference_topic': prefix + '/path_reference',
        'tracking_status_topic': prefix + '/tracking_status',
        'odometry_topic': prefix + '/odometry', 'tf_topic': '/tf',
        'command_topic': prefix + '/cmd_vel',
        'execution_cancel_topic': prefix + '/execution_cancel',
        'max_linear_mps': 0.2, 'max_angular_radps': 0.5,
    }
    return [
        Node(package='lunar_incremental_controller_demo', executable='vehicle_sim',
             parameters=[{'case': scene, 'observation_radius_m': float(LaunchConfiguration('observation_radius_m').perform(context))}], output='screen'),
        Node(package='lunar_incremental_navigation_ros', executable='lunar_incremental_navigation_node',
             name='controller_demo_navigation', parameters=[navigator], output='screen'),
        Node(package='lunar_pure_wheeled_controller', executable='lunar_pure_wheeled_controller_node.py',
             name='controller_demo_executor', parameters=[controller], output='screen'),
        Node(package='lunar_incremental_navigation_ros', executable='lunar_incremental_rviz_goal_bridge',
             name='controller_demo_goal_bridge', parameters=[{
                 'goal_topic': prefix + '/goal_pose', 'action_name': prefix + '/navigate_to_pose',
                 'expected_frame': 'map'}], output='screen'),
        Node(package='rviz2', executable='rviz2', name='controller_demo_rviz',
             arguments=['-d', demo_share + '/rviz/controller_demo.rviz'],
             condition=IfCondition(LaunchConfiguration('start_rviz')), output='screen'),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('case', default_value='manual'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('observation_radius_m', default_value='10.0'),
        OpaqueFunction(function=compose),
    ])
