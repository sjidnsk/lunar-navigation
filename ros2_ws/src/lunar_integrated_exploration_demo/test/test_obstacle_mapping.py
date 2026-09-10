"""Sensor-to-formal-mapper regression; no controller, goal, or velocity output.

Run after sourcing the Jazzy demo install, built with LUNAR_BUILD_DEMO=ON.
The owned navigator and observer use a unique topic prefix on domain 189.
Inputs contain only Terrain.observe returns, including their NaN shadow.
Logs and measured matrices remain in /tmp/lunar-integrated-demo/evidence.
"""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE.parent / 'lunar_incremental_controller_demo'))

pytest.importorskip('rclpy')
from ament_index_python.packages import (
    PackageNotFoundError, get_package_prefix, get_package_share_directory)
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import TransformStamped
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import OccupancyGrid, Odometry
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from tf2_msgs.msg import TFMessage

from lunar_integrated_exploration_demo.terrain import Obstacle, Terrain
from lunar_integrated_exploration_demo.physics import ObservationBatch
from lunar_integrated_exploration_demo.vehicle import observation_grid_map


LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
REVISION_FIELDS = ('received_map_count', 'applied_map_count', 'raw_elevation_revision',
                   'fine_traversability_revision', 'global_guidance_revision')


def _stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _geometry(message):
    return (message.info.origin.position.x, message.info.origin.position.y,
            message.info.width, message.info.height, message.info.resolution)


def _matrix(message):
    return np.asarray(message.data, dtype=np.int8).reshape(
        message.info.height, message.info.width)


def _cell(message, x, y):
    ix = math.floor((x - message.info.origin.position.x) / message.info.resolution)
    iy = math.floor((y - message.info.origin.position.y) / message.info.resolution)
    if not (0 <= ix < message.info.width and 0 <= iy < message.info.height):
        # Sparse mapping need not allocate the all-NaN part of the input window.
        # Keep absence distinct from a literal published UNKNOWN (-1) in evidence.
        return None
    return int(_matrix(message)[iy, ix])


def _observed_cell(observation, x, y):
    ix = math.floor((x - observation['origin_x']) / observation['resolution'])
    iy = math.floor((y - observation['origin_y']) / observation['resolution'])
    return float(observation['values'][iy, ix])


def _probe(observation, fine, coarse, xy):
    states = {'fine': _cell(fine, *xy), 'coarse': _cell(coarse, *xy)}
    return {'xy': xy, 'sensor_finite': math.isfinite(_observed_cell(observation, *xy)),
            **states, 'outside_published_extent': [name for name, value in states.items()
                                                   if value is None]}


def _obstacle_observed_count(terrain, observation, center):
    # Truth labels the assertion region only; it never supplies planner input.
    obstacle = next(o for o in terrain.obstacles if (o.x, o.y) == center)
    xx, yy = np.meshgrid(observation['x'], observation['y'])
    dx, dy = xx - obstacle.x, yy - obstacle.y
    cs, sn = math.cos(obstacle.yaw), math.sin(obstacle.yaw)
    inside = ((np.abs(cs * dx + sn * dy) <= obstacle.length / 2 + 1e-10) &
              (np.abs(-sn * dx + cs * dy) <= obstacle.width / 2 + 1e-10))
    return int(np.count_nonzero(inside & np.isfinite(observation['values'])))


def _front_surface_returns(terrain, observation, center):
    """Measure returned heights in cells intersecting the exposed north face.

    Both fixtures have an axis-aligned solid and a viewpoint to its north.
    A raster surface cell may straddle the continuous boundary, so its centre
    need not lie inside the solid. Truth is used only to label this assertion.
    """
    obstacle = next(o for o in terrain.obstacles if (o.x, o.y) == center)
    assert obstacle.yaw == 0.0
    xx, yy = np.meshgrid(observation['x'], observation['y'])
    half_cell = observation['resolution'] / 2
    face_y = obstacle.y + obstacle.width / 2
    intersects = ((np.abs(yy - face_y) <= half_cell + 1e-10) &
                  (np.abs(xx - obstacle.x) <= obstacle.length / 2 + half_cell))
    observed = intersects & np.isfinite(observation['values'])
    relief = observation['values'][observed] - terrain._ground_elevation(xx[observed], yy[observed])
    return {
        'face_y_m': face_y,
        'intersecting_finite_cells': int(np.count_nonzero(observed)),
        'height_jump_cells': int(np.count_nonzero(relief > 0.75)),
        'maximum_relief_m': float(np.max(relief)) if relief.size else None,
    }


class WideWallTerrain(Terrain):
    """The face at 6.25 m cuts a cell whose centre 6.3 m is outside the wall."""

    def _make_obstacles(self):
        return (Obstacle(10.0, 5.0, 10.0, 2.5, 0.0, 1.3),)


class FormalMapper:
    """An owned formal navigator process plus an observation-only ROS peer."""

    def __init__(self, binary, config, artifact_dir):
        self.artifact_dir = artifact_dir
        suffix = uuid.uuid4().hex[:12]
        self.prefix = '/lunar_demo/obstacle_mapping_' + suffix
        node_name = 'obstacle_mapper_' + suffix
        parameters = {
            'platform_type': 'wheel', 'platform_config': str(config),
            'coarse_resolution_m': 1.0, 'local_window_size_m': 64.0,
            'use_sim_time': False, 'enable_tracking_feedback': False,
            'enable_debug_visualization': True, 'debug_fine_window_m': 28.0,
            'debug_topic_prefix': self.prefix + '/planning',
            **{name: self.prefix + '/' + topic for name, topic in {
                'local_map_topic': 'grid_map', 'odometry_topic': 'odometry',
                'tf_topic': 'tf', 'action_name': 'navigate_to_pose',
                'path_reference_topic': 'path_reference',
                'local_path_topic': 'local_path', 'global_route_topic': 'global_route',
                'diagnostics_topic': 'diagnostics',
                'exploration_map_topic': 'exploration_map',
                'tracking_status_topic': 'tracking_status'}.items()},
        }
        params_path = artifact_dir / 'parameters.json'
        # JSON is valid YAML; the file preserves parameter types and exact names.
        params_path.write_text(json.dumps({node_name: {'ros__parameters': parameters}},
                                         indent=2), encoding='utf-8')
        self.context = Context()
        rclpy.init(args=[], context=self.context, domain_id=189)
        self.node = Node('obstacle_observer_' + suffix, context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.maps = {}
        self.revisions = dict.fromkeys(REVISION_FIELDS, 0)
        self.last_capture = None
        self.node.create_subscription(OccupancyGrid, self.prefix + '/planning/fine_state',
                                      lambda msg: self.maps.update(fine=msg), LATCHED)
        self.node.create_subscription(OccupancyGrid, self.prefix + '/exploration_map',
                                      lambda msg: self.maps.update(coarse=msg), LATCHED)
        self.node.create_subscription(DiagnosticArray, self.prefix + '/diagnostics',
                                      self.on_diagnostics, 10)
        self.map_pub = self.node.create_publisher(GridMap, self.prefix + '/grid_map', LATCHED)
        self.odom_pub = self.node.create_publisher(Odometry, self.prefix + '/odometry',
                                                  qos_profile_sensor_data)
        self.tf_pub = self.node.create_publisher(TFMessage, self.prefix + '/tf',
                                                qos_profile_sensor_data)
        self.pose = None
        self.timer = self.node.create_timer(0.05, self.publish_pose)
        self.log = (artifact_dir / 'navigator.log').open('w', encoding='utf-8')
        environment = dict(os.environ, ROS_DOMAIN_ID='189', ROS_LOCALHOST_ONLY='1')
        self.process = subprocess.Popen(
            [str(binary), '--ros-args', '-r', '__node:=' + node_name,
             '--params-file', str(params_path)], env=environment,
            stdout=self.log, stderr=subprocess.STDOUT)

    def publish_pose(self):
        if self.pose is None:
            return
        x, y, yaw = self.pose
        stamp = self.node.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, 'odom', 'base_link'
        odom.pose.pose.position.x, odom.pose.pose.position.y = x, y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = (
            math.sin(yaw / 2), math.cos(yaw / 2))
        transform = TransformStamped()
        transform.header.stamp, transform.header.frame_id = stamp, 'map'
        transform.child_frame_id = 'odom'
        transform.transform.rotation.w = 1.0
        self.tf_pub.publish(TFMessage(transforms=[transform]))
        self.odom_pub.publish(odom)

    def on_diagnostics(self, message):
        for status in message.status:
            if status.name == 'lunar_incremental_navigation/map':
                for item in status.values:
                    if item.key in self.revisions:
                        # Map workers can publish diagnostics concurrently.
                        self.revisions[item.key] = max(self.revisions[item.key], int(item.value))

    def until(self, predicate, timeout=12.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            assert self.process.poll() is None, self.artifact_dir / 'navigator.log'
            self.executor.spin_once(timeout_sec=0.02)
        assert predicate(), 'Formal mapper timeout; evidence: ' + str(self.artifact_dir)

    def capture(self, observation, pose, expect_geometry_change=False):
        self.pose = tuple(float(v) for v in pose)
        self.until(lambda: self.map_pub.get_subscription_count() > 0 and
                   self.odom_pub.get_subscription_count() > 0 and
                   self.tf_pub.get_subscription_count() > 0 and
                   self.node.count_publishers(self.prefix + '/diagnostics') > 0)
        previous_revisions = self.revisions.copy()
        previous_geometry = {name: _geometry(msg) for name, msg in self.maps.items()}
        self.publish_pose()
        stamp = self.node.get_clock().now().to_msg()
        self.map_pub.publish(observation_grid_map(observation, stamp))
        self.until(lambda: set(self.maps) == {'fine', 'coarse'} and
                   all(self.revisions[key] > previous_revisions[key] for key in REVISION_FIELDS) and
                   all(_stamp_ns(msg.header.stamp) > _stamp_ns(stamp) and
                       np.count_nonzero(_matrix(msg) == 0) > 0 for msg in self.maps.values()) and
                   (not expect_geometry_change or all(
                       _geometry(msg) != previous_geometry.get(name)
                       for name, msg in self.maps.items())))
        self.last_capture = {
            'input_stamp_ns': _stamp_ns(stamp),
            'previous_revisions': previous_revisions, 'revisions': self.revisions.copy(),
            'output_stamps_ns': {name: _stamp_ns(msg.header.stamp) for name, msg in self.maps.items()},
            'geometry_changed': {name: _geometry(msg) != previous_geometry.get(name)
                                 for name, msg in self.maps.items()},
        }
        return self.maps['fine'], self.maps['coarse']

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.send_signal(signal.SIGINT)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=5)
        finally:
            self.log.close()
            self.executor.remove_node(self.node)
            self.node.destroy_node()
            self.executor.shutdown()
            self.context.shutdown()


@pytest.fixture
def formal_mapper(monkeypatch):
    package = 'lunar_incremental_navigation_ros'
    try:
        binary = Path(get_package_prefix(package)) / 'lib' / package / 'lunar_incremental_navigation_node'
        config = Path(get_package_share_directory(package)) / 'config' / 'wheel.yaml'
    except PackageNotFoundError:
        pytest.skip('Source a built Jazzy demo overlay with the formal navigator')
    assert binary.is_file() and config.is_file()
    evidence_root = Path('/tmp/lunar-integrated-demo/evidence')
    evidence_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(tempfile.mkdtemp(prefix='obstacle-mapping-', dir=evidence_root))
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    mapper = FormalMapper(binary, config, artifact_dir)
    try:
        yield mapper
    finally:
        mapper.close()


def _check_obstacle_mapping(formal_mapper, terrain, pose, probes, case,
                            require_allocated_shadow=True, expected_front_coarse=100,
                            observation=None, input_observations=None,
                            expect_geometry_change=False, **sensor_options):
    if observation is None:
        observation = terrain.observe(*pose, **sensor_options)
    fine, coarse = formal_mapper.capture(observation, pose, expect_geometry_change)
    observed_rock_cells = _obstacle_observed_count(terrain, observation, (10.0, 5.0))
    report = {
        'case': case, 'pose': pose, 'sensor_options': sensor_options,
        'input_observations': input_observations or [{'pose': pose, 'options': sensor_options}],
        'capture': formal_mapper.last_capture,
        'prefix': formal_mapper.prefix,
        'observed_cells': int(np.count_nonzero(np.isfinite(observation['values']))),
        # Retained only for diagnosing the old centre-sampled sensor: a correct
        # raster first-return cell is allowed to straddle the physical surface.
        'observed_rock_center_cells': observed_rock_cells,
        'front_surface': _front_surface_returns(terrain, observation, (10.0, 5.0)),
        'probes': {name: _probe(observation, fine, coarse, xy) for name, xy in probes.items()},
        'outputs': {name: {'frame': msg.header.frame_id, 'resolution': msg.info.resolution,
                           'origin': [msg.info.origin.position.x, msg.info.origin.position.y],
                           'width': msg.info.width, 'height': msg.info.height,
                           'free': int(np.count_nonzero(_matrix(msg) == 0)),
                           'blocked': int(np.count_nonzero(_matrix(msg) == 100)),
                           'unknown': int(np.count_nonzero(_matrix(msg) == -1))}
                    for name, msg in [('fine', fine), ('coarse', coarse)]},
    }
    path = formal_mapper.artifact_dir / (case + '.json')
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(formal_mapper.artifact_dir / (case + '-measured-maps.npz'),
                        observation=observation['values'], observation_x=observation['x'],
                        observation_y=observation['y'], fine=_matrix(fine), coarse=_matrix(coarse))
    evidence = json.dumps(report, indent=2) + '\nEvidence: ' + str(path)
    # Positive control proves actual elevation made it through the formal mapper.
    assert report['probes']['visible_ground']['sensor_finite'], evidence
    assert report['probes']['visible_ground']['fine'] == 0, evidence
    assert report['probes']['visible_ground']['coarse'] == 0, evidence
    # Occluded ground must remain UNKNOWN or outside the allocated map extent.
    # None records absence, not a fabricated -1 received from the formal node.
    assert not report['probes']['shadow']['sensor_finite'], evidence
    unknown_values = (-1,) if require_allocated_shadow else (-1, None)
    assert report['probes']['shadow']['fine'] in unknown_values, evidence
    assert report['probes']['shadow']['coarse'] in unknown_values, evidence
    # Require actual elevated returns in a cell intersecting the visible face,
    # then a blocked footprint and the explicit coarse projection for this stage.
    assert report['front_surface']['height_jump_cells'] > 0, evidence
    assert report['probes']['front_face']['fine'] == 100, evidence
    assert report['probes']['front_face']['coarse'] == expected_front_coarse, evidence
    return observation


def test_visible_rock_face_blocks_formal_map_and_shadow_stays_unknown(formal_mapper):
    _check_obstacle_mapping(
        formal_mapper, Terrain(), (9.0, 9.0, math.atan2(-4.0, 1.0)),
        {'front_face': (9.5, 6.5), 'shadow': (10.5, 2.5),
         'visible_ground': (9.1, 8.5)}, 'near_rock_north_face')


def _capture_wide_wall_first_view(formal_mapper):
    terrain = WideWallTerrain()
    pose = (10.0, 9.0, -math.pi / 2)
    probes = {'front_face': (10.5, 6.5), 'shadow': (10.5, 2.5),
              'visible_ground': (10.1, 8.9), 'new_extent_cell': (10.5, 6.1)}
    # One narrow view allocates fine geometry only down to y=6.2. The 1 m
    # coarse cell [6,7] is therefore incomplete even when its fine face blocks.
    first = _check_obstacle_mapping(
        formal_mapper, terrain, pose, probes, 'wide_wall_face_at_6_25',
        require_allocated_shadow=False, expected_front_coarse=-1, fov_deg=30.0)
    return terrain, pose, probes, first


def test_wide_wall_face_crossing_cell_boundary_blocks_formal_map(formal_mapper):
    _capture_wide_wall_first_view(formal_mapper)


def test_map_growth_rederives_existing_obstacle_influence_in_new_extent(formal_mapper):
    terrain, pose, probes, first = _capture_wide_wall_first_view(formal_mapper)

    # Accumulate a real west-facing view. It expands the map's allocated bounds
    # while supplying no wall-face or shadow evidence. Only Terrain.observe
    # samples enter the batch; no unobserved truth is filled into either map.
    west_pose = (0.0, 9.0, math.pi)
    west = terrain.observe(*west_pose)
    assert not math.isfinite(_observed_cell(west, *probes['front_face']))
    assert not math.isfinite(_observed_cell(west, *probes['shadow']))
    batch = ObservationBatch()
    batch.add(first, 1.0)
    batch.add(west, 2.0)
    accumulated = batch.snapshot()
    height, width = accumulated['values'].shape
    resolution = accumulated['resolution']
    accumulated['x'] = (round(accumulated['origin_x'] / resolution) +
                        np.arange(width) + 0.5) * resolution
    accumulated['y'] = (round(accumulated['origin_y'] / resolution) +
                        np.arange(height) + 0.5) * resolution
    # The newly included cell has no raw return. Its footprint is nevertheless
    # blocked by the original wall face, and geometry growth must derive that.
    assert not math.isfinite(_observed_cell(accumulated, *probes['new_extent_cell']))
    _check_obstacle_mapping(
        formal_mapper, terrain, pose, probes, 'wide_wall_after_real_west_observation',
        observation=accumulated, expect_geometry_change=True,
        input_observations=[{'pose': pose, 'options': {'fov_deg': 30.0}},
                            {'pose': west_pose, 'options': {}}])
    assert _cell(formal_mapper.maps['fine'], *probes['new_extent_cell']) == 100
