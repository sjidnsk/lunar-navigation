"""Optional operator view and PoseStamped goal bridge for the production stack."""
from pathlib import Path
import tempfile
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def compose(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)
    config = yaml.safe_load(Path(value('config_file')).read_text(encoding='utf-8'))
    common = config['common']
    use_sim_time = value('use_sim_time').lower()
    if use_sim_time not in ('', 'true', 'false'):
        raise ValueError('use_sim_time must be true or false')
    sim = common.get('use_sim_time', False) if not use_sim_time else use_sim_time == 'true'
    goal_topic = value('goal_topic')
    actions = [Node(package='lunar_incremental_navigation_ros',
        executable='lunar_incremental_rviz_goal_bridge',
        parameters=[{'expected_frame': common['frames']['map'],
                     'goal_topic': goal_topic, 'action_name': common['navigation_action'],
                     'use_sim_time': sim}], output='screen')]
    if IfCondition(LaunchConfiguration('start_rviz')).evaluate(context):
        share = Path(get_package_share_directory('lunar_incremental_navigation_ros'))
        view = yaml.safe_load((share / 'rviz/navigation.rviz').read_text(encoding='utf-8'))
        manager = view['Visualization Manager']
        manager['Global Options']['Fixed Frame'] = common['frames']['map']
        topics = {'Exploration map': common['exploration_map_topic'],
                  'Global route': config['navigation']['global_route_topic'],
                  'Local path': config['navigation']['local_path_topic'],
                  'Odometry': common['odometry_topic'],
                  'Frontiers': config['exploration']['frontiers_topic'],
                  'Current goal': config['exploration']['current_goal_topic'],
                  'Task boundary': config['exploration']['task_boundary_topic']}
        for display in manager['Displays']:
            if display['Name'] in topics:
                display['Topic']['Value'] = topics[display['Name']]
                if display['Name'] == 'Exploration map':
                    qos = common['exploration_map_qos']
                    display['Topic'].update({'Reliability Policy': 'Reliable' if qos['reliability'] == 'reliable' else 'Best Effort',
                        'Durability Policy': 'Transient Local' if qos['durability'] == 'transient_local' else 'Volatile',
                        'Depth': qos.get('depth', 1)})
        for tool in manager['Tools']:
            if tool['Class'] == 'rviz_default_plugins/SetGoal':
                tool['Topic'] = goal_topic
        temporary = tempfile.TemporaryDirectory(prefix='lunar-rviz-')
        path = Path(temporary.name) / 'navigation.rviz'
        path.write_text(yaml.safe_dump(view, sort_keys=False), encoding='utf-8')
        def cleanup(context):
            temporary.cleanup()
            return []
        actions += [RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=cleanup)])),
                    Node(package='rviz2', executable='rviz2', parameters=[{'use_sim_time': sim}],
                         arguments=['-d', str(path)], output='screen')]
    return actions


def generate_launch_description():
    share = get_package_share_directory('lunar_pure_exploration_ros')
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=share + '/config/exploration_navigation.yaml'),
        DeclareLaunchArgument('use_sim_time', default_value=''),
        DeclareLaunchArgument('goal_topic', default_value='/Car/T4/rviz_goal'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        OpaqueFunction(function=compose)])
