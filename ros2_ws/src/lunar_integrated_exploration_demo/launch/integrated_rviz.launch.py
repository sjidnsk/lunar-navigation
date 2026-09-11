"""Rich multiresolution exploration with the formal planner and controller."""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def compose(context):
    def value(name):return LaunchConfiguration(name).perform(context)
    prefix='/lunar_demo/integrated'
    scene_size=float(value('scene_size_m'));task_size=float(value('task_size_m'));scale=float(value('time_scale'))
    if not 20<=task_size<=scene_size or not 1<=scale<=60:
        raise ValueError('Require 20 <= task_size_m <= scene_size_m, time_scale in [1,60]')
    share=get_package_share_directory('lunar_integrated_exploration_demo')
    config=get_package_share_directory('lunar_incremental_navigation_ros')+'/config/wheel.yaml'
    sensor={'sensor_range_m':12.,'sensor_fov_deg':120.}
    sim={'use_sim_time':True}
    navigator=dict(sim,platform_type='wheel',platform_config=config,coarse_resolution_m=1.,local_window_size_m=64.,
        local_map_topic=prefix+'/grid_map',odometry_topic=prefix+'/odometry',tf_topic='/tf',
        action_name=prefix+'/navigate_to_pose',path_reference_topic=prefix+'/path_reference',local_path_topic=prefix+'/local_path',
        global_route_topic=prefix+'/global_route',diagnostics_topic=prefix+'/planning/diagnostics',
        exploration_map_topic=prefix+'/exploration_map',enable_tracking_feedback=True,
        tracking_status_topic=prefix+'/tracking_status',enable_debug_visualization=True,
        debug_topic_prefix=prefix+'/planning',debug_fine_window_m=28.)
    explorer=dict(sim,**sensor,platform_selector='wheel',platform_config=config,coverage_target=float(value('coverage_target')),
        navigation_map_wait_timeout_s=float(value('navigation_map_wait_timeout_s')),
        exploration_map_topic=prefix+'/exploration_map',odometry_topic=prefix+'/exploration/odometry',tf_topic=prefix+'/exploration/tf',
        task_topic=prefix+'/exploration/task',navigation_action=prefix+'/navigate_to_pose',
        status_topic=prefix+'/exploration/status',task_boundary_topic=prefix+'/exploration/task_boundary',
        task_map_markers_topic=prefix+'/exploration/task_map_markers',current_goal_topic=prefix+'/exploration/current_goal',
        frontiers_topic=prefix+'/exploration/frontiers',diagnostics_topic=prefix+'/exploration/diagnostics')
    controller=dict(sim,input_mode='incremental_reference',platform_config=config,path_reference_topic=prefix+'/path_reference',
        tracking_status_topic=prefix+'/tracking_status',odometry_topic=prefix+'/odometry',tf_topic='/tf',
        command_topic=prefix+'/cmd_vel',execution_cancel_topic=prefix+'/execution_cancel',max_linear_mps=.2,max_angular_radps=.5)
    return [
        Node(package='lunar_integrated_exploration_demo',executable='vehicle_sim',name='integrated_vehicle',
             parameters=[dict(sensor,scene_size_m=scene_size,time_scale=scale,fine_resolution_m=.2,local_window_m=28.)],output='screen'),
        Node(package='lunar_incremental_navigation_ros',executable='lunar_incremental_navigation_node',name='integrated_navigation',parameters=[navigator],output='screen'),
        Node(package='lunar_pure_wheeled_controller',executable='lunar_pure_wheeled_controller_node.py',name='integrated_executor',parameters=[controller],output='screen'),
        Node(package='lunar_pure_exploration_ros',executable='incremental_exploration_node',name='integrated_explorer',parameters=[explorer],output='screen'),
        Node(package='lunar_integrated_exploration_demo',executable='visualizer',name='integrated_visualizer',parameters=[dict(sim,**sensor,scene_size_m=scene_size)],output='screen'),
        Node(package='lunar_integrated_exploration_demo',executable='start_task',name='integrated_task_starter',parameters=[dict(sim,task_size_m=task_size,initial_scan=True)],condition=IfCondition(LaunchConfiguration('auto_start')),output='screen'),
        Node(package='lunar_incremental_navigation_ros',executable='lunar_incremental_rviz_goal_bridge',
             name='integrated_goal_bridge',parameters=[dict(sim,goal_topic=prefix+'/goal_pose',
             action_name=prefix+'/navigate_to_pose',expected_frame='map')],output='screen'),
        Node(package='rviz2',executable='rviz2',name='integrated_global_rviz',parameters=[sim],arguments=['-d',share+'/rviz/global.rviz'],condition=IfCondition(LaunchConfiguration('start_rviz')),output='screen'),
        Node(package='rviz2',executable='rviz2',name='integrated_local_rviz',parameters=[sim],arguments=['-d',share+'/rviz/local.rviz'],condition=IfCondition(LaunchConfiguration('start_local_rviz')),output='screen'),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scene_size_m',default_value='300.0'),
        DeclareLaunchArgument('task_size_m',default_value='300.0'),
        DeclareLaunchArgument('time_scale',default_value='30.0'),
        DeclareLaunchArgument('coverage_target',default_value='1.0'),
        DeclareLaunchArgument('navigation_map_wait_timeout_s',default_value='30.0',
                              description='Simulated seconds without map evidence before canceling a waiting frontier goal; 0 disables recovery'),
        DeclareLaunchArgument('auto_start',default_value='true'),
        DeclareLaunchArgument('start_rviz',default_value='true'),
        DeclareLaunchArgument('start_local_rviz',default_value='true'),
        OpaqueFunction(function=compose)])
