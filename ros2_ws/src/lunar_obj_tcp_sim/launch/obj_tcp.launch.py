"""OBJ-backed TCP simulation wired to the formal navigation stack."""

from pathlib import Path
import json
import tempfile
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from lunar_obj_tcp_sim.configuration import load_simulation_config
from lunar_obj_tcp_sim.observation_state import exploration_bounds


def _as_bool(value):
    return str(value).lower() in {"1", "true", "yes", "on"}


def compose(context):
    value = lambda name: LaunchConfiguration(name).perform(context)
    share = get_package_share_directory("lunar_obj_tcp_sim")
    config_path = value("config") or str(Path(share) / "config" / "simulation.yaml")
    overrides = {}
    for name in ("host", "port", "map_directory", "mode", "prefix"):
        if value(name):
            overrides[name] = value(name)
    for name in ("start_rviz", "start_local_rviz"):
        if value(name):
            overrides[name] = _as_bool(value(name))
    config = load_simulation_config(config_path, overrides)
    if not config.map_directory:
        raise ValueError("map_directory is required")
    platform_config = value("platform_config") or (
        get_package_share_directory("lunar_incremental_navigation_ros") + "/config/wheel.yaml"
    )
    prefix = config.prefix
    rviz_topics = (
        "exploration_map", "exploration/task_boundary", "exploration/frontiers",
        "global_route", "local_path", "trace", "display", "observed_cells",
        "terrain_preview", "planning/fine_state", "planning/traversability",
        "planning/local_goals", "hud_local", "goal_pose",
    )
    rviz_remappings = [
        ("/lunar_sim/" + topic, prefix + "/" + topic) for topic in rviz_topics
    ]
    common = {"use_sim_time": False}
    sensor = dict(
        common, map_directory=config.map_directory, mode=config.mode, prefix=prefix,
        auto_start=config.auto_start, initial_scan=config.initial_scan,
        scan_step_deg=config.scan_step_deg, scan_reanchor_distance_m=config.scan_reanchor_distance_m,
        bootstrap_timeout_s=config.bootstrap_timeout_s,
        exploration_size_m=config.exploration_size_m,
        known_chunk_cells=config.known_chunk_cells,
        sensor_range_m=config.sensor_range_m, sensor_fov_deg=config.sensor_fov_deg,
        near_field_radius_m=config.near_field_radius_m,
        sensor_offset_xyz_m=list(config.sensor_offset_xyz_m),
        observation_window_m=config.observation_window_m,
        observation_rate_hz=config.observation_rate_hz,
        local_window_size_m=config.local_window_size_m,
    )
    navigator = dict(
        common, platform_type="wheel", platform_config=platform_config,
        goal_position_tolerance_m=config.goal_position_tolerance_m,
        goal_yaw_tolerance_rad=config.goal_yaw_tolerance_rad,
        coarse_resolution_m=config.coarse_resolution_m,
        local_window_size_m=config.local_window_size_m,
        local_map_topic=prefix + "/grid_map", odometry_topic=prefix + "/odometry", tf_topic="/tf",
        action_name=prefix + "/navigate_to_pose", path_reference_topic=prefix + "/path_reference",
        local_path_topic=prefix + "/local_path", global_route_topic=prefix + "/global_route",
        diagnostics_topic=prefix + "/planning/diagnostics",
        exploration_map_topic=prefix + "/exploration_map", enable_tracking_feedback=True,
        tracking_status_topic=prefix + "/tracking_status", enable_debug_visualization=True,
        debug_topic_prefix=prefix + "/planning", debug_fine_window_m=config.observation_window_m,
    )
    controller = dict(
        common, input_mode="incremental_reference", platform_config=platform_config,
        path_reference_topic=prefix + "/path_reference", tracking_status_topic=prefix + "/tracking_status",
        odometry_topic=prefix + "/odometry", tf_topic="/tf", command_topic=prefix + "/cmd_vel",
        execution_cancel_topic=prefix + "/execution_cancel",
        goal_position_tolerance_m=config.goal_position_tolerance_m,
        max_linear_mps=config.max_linear_mps,
        goal_yaw_tolerance_rad=config.goal_yaw_tolerance_rad,
        max_angular_radps=config.max_angular_radps,
        max_angular_accel_radps2=config.max_angular_accel_radps2,
        scan_profile_enabled=config.mode=="explore" and config.auto_start and config.initial_scan,
        scan_max_angular_radps=config.scan_max_angular_radps,
        scan_max_angular_accel_radps2=config.scan_max_angular_accel_radps2,
        observation_status_topic=prefix+"/observation_status",
    )
    explorer = dict(
        common, sensor_range_m=config.sensor_range_m, sensor_fov_deg=config.sensor_fov_deg,
        platform_selector="wheel", platform_config=platform_config,
        coverage_target=config.coverage_target,
        navigation_map_wait_timeout_s=config.navigation_map_wait_timeout_s,
        exploration_map_topic=prefix + "/exploration_map", odometry_topic=prefix + "/odometry",
        tf_topic="/tf", task_topic=prefix + "/exploration/task",
        navigation_action=prefix + "/navigate_to_pose", status_topic=prefix + "/exploration/status",
        task_boundary_topic=prefix + "/exploration/task_boundary",
        task_map_markers_topic=prefix + "/exploration/task_map_markers",
        current_goal_topic=prefix + "/exploration/current_goal",
        frontiers_topic=prefix + "/exploration/frontiers",
        diagnostics_topic=prefix + "/exploration/diagnostics",
    )
    nodes = [
        Node(package="lunar_obj_tcp_sim", executable="tcp_vehicle_bridge", name="tcp_vehicle_bridge",
             parameters=[dict(common, host=config.host, port=config.port,
                              platform_config=platform_config, prefix=prefix, vehicle_id=config.vehicle_id,
                              send_rate_hz=config.bridge_send_rate_hz,
                              command_timeout_s=config.command_timeout_s,
                              feedback_timeout_s=config.feedback_timeout_s,
                              connect_timeout_s=config.connect_timeout_s,
                              max_wheel_speed_radps=config.max_wheel_speed_radps,
                              wheel_order=list(config.wheel_order), steering_order=list(config.steering_order),
                              wheel_signs=list(config.wheel_signs), steering_signs=list(config.steering_signs),
                              steering_zero_rad=list(config.steering_zero_rad),
                              max_steering_angle_rad=config.max_steering_angle_rad,
                              steering_tolerance_rad=config.steering_tolerance_rad)], output="screen"),
        Node(package="lunar_obj_tcp_sim", executable="obj_virtual_sensor", name="obj_virtual_sensor",
             parameters=[sensor], output="screen"),
        Node(package="lunar_incremental_navigation_ros", executable="lunar_incremental_navigation_node",
             name="obj_tcp_navigation", parameters=[navigator], output="screen"),
        Node(package="lunar_obj_tcp_sim", executable="obj_scan_controller",
             name="obj_tcp_executor", parameters=[controller], output="screen"),
    ]
    if config.mode == "explore":
        nodes.append(Node(package="lunar_pure_exploration_ros", executable="incremental_exploration_node",
                          name="obj_tcp_explorer", parameters=[explorer], output="screen"))
    else:
        nodes.append(Node(package="lunar_incremental_navigation_ros",
                          executable="lunar_incremental_rviz_goal_bridge", name="obj_tcp_goal_bridge",
                          parameters=[dict(common, goal_topic=prefix + "/goal_pose",
                                           action_name=prefix + "/navigate_to_pose",
                                           expected_frame="map")], output="screen"))
    global_view = str(Path(share) / 'rviz' / 'global.rviz')
    if config.start_rviz:
        metadata = json.loads((Path(config.map_directory)/'metadata.json').read_text())
        xmin,ymin,xmax,ymax = exploration_bounds(metadata['task_bounds_xy'], config.exploration_size_m)
        document = yaml.safe_load(Path(global_view).read_text())
        view = document['Visualization Manager']['Views']['Current']
        view.update({'Value':'TopDownOrtho (rviz_default_plugins)', 'Angle':0., 'Target Frame':'map', 'X':(xmin+xmax)/2, 'Y':(ymin+ymax)/2,
                     'Scale':650./(max(xmax-xmin,ymax-ymin)*1.25)})
        with tempfile.NamedTemporaryFile(mode='w',suffix='.rviz',prefix='obj-task-',delete=False) as stream:
            yaml.safe_dump(document,stream)
            global_view=stream.name
        def remove_view(_context):
            Path(global_view).unlink(missing_ok=True)
            return []
        nodes.append(RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=remove_view)])))
    nodes.extend([
        Node(package="rviz2", executable="rviz2", name="obj_tcp_global_rviz",
             arguments=["-d", global_view],
             remappings=rviz_remappings,
             condition=IfCondition(str(config.start_rviz).lower()), output="screen"),
        Node(package="rviz2", executable="rviz2", name="obj_tcp_local_rviz",
             arguments=["-d", str(Path(share) / "rviz" / "local.rviz")],
             remappings=rviz_remappings,
             condition=IfCondition(str(config.start_local_rviz).lower()), output="screen"),
    ])
    return nodes


def generate_launch_description():
    share = get_package_share_directory("lunar_obj_tcp_sim")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=str(Path(share) / "config" / "simulation.yaml")),
        DeclareLaunchArgument("host", default_value=""), DeclareLaunchArgument("port", default_value=""),
        DeclareLaunchArgument("map_directory", default_value=""), DeclareLaunchArgument("mode", default_value=""),
        DeclareLaunchArgument("prefix", default_value=""), DeclareLaunchArgument("platform_config", default_value=""),
        DeclareLaunchArgument("start_rviz", default_value=""),
        DeclareLaunchArgument("start_local_rviz", default_value=""), OpaqueFunction(function=compose),
    ])
