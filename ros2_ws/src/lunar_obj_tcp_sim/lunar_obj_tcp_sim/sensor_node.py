"""OBJ observation adapter, task bootstrap, and RViz evidence display.

Vehicle dynamics and command generation belong to the remote simulator and the
formal controller respectively. This node never publishes a Twist.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import math
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import Point, PoseStamped, Twist
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from lunar_planning_msgs.action import NavigateToPose
from lunar_planning_msgs.msg import TrackingStatus
from lunar_pure_exploration_msgs.msg import PureExplorationTask, PureExplorationStatus

from .geometry import TerrainMap
from .observation_state import EvidenceCoverage, MapReceipt, exploration_bounds
from .ros_support import observation_grid_map, exploration_task
from .visualization import marker, scene_markers, truth_preview
from .scan_sequence import ScanSequence

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
STATE = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)


class ObjVirtualSensorNode(Node):
    def __init__(self, **kwargs):
        super().__init__('obj_virtual_sensor', **kwargs)
        defaults = {
            'map_directory': '', 'mode': 'explore', 'prefix': '/lunar_sim',
            'auto_start': True, 'initial_scan': True, 'exploration_size_m': 300.,
            'sensor_range_m': 12., 'sensor_fov_deg': 120., 'near_field_radius_m': 2.1,
            'sensor_offset_xyz_m': [0.591, 0., 1.0], 'observation_window_m': 28.,
            'observation_rate_hz': 5., 'odometry_timeout_s': .5,
            'known_chunk_cells': 512, 'bootstrap_timeout_s': 45., 'local_window_size_m': 64.,
            'scan_step_deg': 30., 'scan_reanchor_distance_m': .1,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.settings = {name: self.get_parameter(name).value for name in defaults}
        self.mode = self.settings['mode']
        if self.mode not in ('nav', 'explore'):
            raise ValueError('mode must be nav or explore')
        if self.get_parameter('use_sim_time').value:
            raise ValueError('TCP observation uses local receive time; use_sim_time must be false')
        self.prefix = self.settings['prefix'].rstrip('/')
        self.terrain = TerrainMap(self.settings['map_directory'])
        self.task_bounds = exploration_bounds(self.terrain.metadata['task_bounds_xy'],
                                              self.settings['exploration_size_m'])
        self.coverage = EvidenceCoverage(dict(self.terrain.metadata, task_bounds_xy=self.task_bounds))
        self.sensor_options = {name: self.settings[name] for name in (
            'sensor_range_m', 'sensor_fov_deg', 'near_field_radius_m',
            'sensor_offset_xyz_m', 'observation_window_m')}
        for name in ('observation_rate_hz', 'odometry_timeout_s', 'bootstrap_timeout_s'):
            if not math.isfinite(self.settings[name]) or self.settings[name] <= 0:
                raise ValueError(name + ' must be finite and positive')
        if self.settings['observation_window_m'] < 2 * self.settings['sensor_range_m']:
            raise ValueError('observation_window_m must cover the sensor diameter')
        if int(self.settings['known_chunk_cells']) < 1:
            raise ValueError('known_chunk_cells must be positive')
        self.map_pub = self.create_publisher(GridMap, self.prefix+'/grid_map', LATCHED)
        self.status_pub = self.create_publisher(String, self.prefix+'/observation_status', LATCHED)
        self.task_pub = self.create_publisher(PureExplorationTask, self.prefix+'/exploration/task', 10)
        self.trace_pub = self.create_publisher(Path, self.prefix+'/trace', LATCHED)
        self.display_pub = self.create_publisher(MarkerArray, self.prefix+'/display', LATCHED)
        self.observed_pub = self.create_publisher(MarkerArray, self.prefix+'/observed_cells', LATCHED)
        self.hud_pub = self.create_publisher(MarkerArray, self.prefix+'/hud_local', LATCHED)
        self.truth_pub = self.create_publisher(MarkerArray, self.prefix+'/terrain_preview', LATCHED)
        self.truth_published = False
        self.command = Twist()
        self.command_received = 0.
        self.create_subscription(Twist, self.prefix+'/cmd_vel', self.on_command, 10)
        self.client = ActionClient(self, NavigateToPose, self.prefix+'/navigate_to_pose')
        self.create_subscription(Odometry, self.prefix+'/odometry', self.on_pose, STATE)
        self.create_subscription(OccupancyGrid, self.prefix+'/exploration_map', self.on_map, LATCHED)
        self.create_subscription(DiagnosticArray, self.prefix+'/planning/diagnostics', self.on_diagnostics, 10)
        self.create_subscription(TrackingStatus, self.prefix+'/tracking_status', self.on_tracking, 10)
        self.create_subscription(PureExplorationStatus, self.prefix+'/exploration/status', self.on_exploration, 10)
        self.create_subscription(PureExplorationTask, self.prefix+'/exploration/task', self.on_task_command, 10)
        self.latest = None
        self.latest_received = 0.
        self.last_sample_key = None
        self.samples = 0
        self.last_query_ms = 0.
        self.phase = 'WAITING_FOR_STATE'
        self.map_seen = False
        self.receipt = MapReceipt()
        self.known_chunks = None
        self.known_started = None
        self.known_done = False
        self.known_count = 0
        self.last_known_sent = 0.
        self.pending_chunk = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='obj-observation')
        self.observation_future = None
        self.observation_stamp = None
        self.query_begin = 0.
        self.scan = ScanSequence(self.client, enabled=self.settings['initial_scan'],
                                 step_deg=self.settings['scan_step_deg'],
                                 drift_m=self.settings['scan_reanchor_distance_m'],
                                 timeout_s=self.settings['bootstrap_timeout_s'])
        self.scan_cancellation = None
        self.task_started = False
        self.stopping = False
        self.poses = deque(maxlen=10000)
        self.tracking_text = 'WAITING'
        self.exploration_text = 'IDLE'
        self.exploration_received = False
        clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(1./self.settings['observation_rate_hz'], self.tick, clock=clock)
        self.create_timer(.1, self.finish_observation, clock=clock)
        self.create_timer(.5, self.publish_status, clock=clock)
        self.get_logger().info(
            f'OBJ {self.mode}: {self.terrain.metadata["task_bounds_xy"]}, '
            f'{self.terrain.resolution:g} m; feedback convention is C++ metres / VehicleID; '
            f'alignment={self.terrain.metadata.get("alignment_status", "UNVERIFIED")}')

    def on_pose(self, message):
        self.latest, self.latest_received = message, time.monotonic()
        p = message.pose.pose.position
        if not self.poses or math.hypot(p.x-self.poses[-1].pose.position.x,
                                      p.y-self.poses[-1].pose.position.y) >= .05:
            self.poses.append(PoseStamped(header=message.header, pose=message.pose.pose))

    def on_command(self, message):
        self.command = message
        self.command_received = time.monotonic()

    def on_map(self, message):
        self.map_seen = any(value == 0 for value in message.data)

    def on_tracking(self, message):
        self.tracking_text = f'{message.state}: {message.reason}'

    def on_exploration(self, message):
        self.exploration_received = True
        self.exploration_text = f'{message.reason_code} coarse={message.coverage_ratio:.1%}'

    def on_task_command(self, message):
        if message.command == PureExplorationTask.CANCEL and message.task_id == 'obj-tcp-exploration':
            self.stopping = True
            self.phase = 'SHUTTING_DOWN'
            if self.scan_cancellation is not None:
                self.scan_cancellation.cancel()

    def on_diagnostics(self, message):
        for status in message.status:
            fields = {entry.key: entry.value for entry in status.values}
            if 'received_map_count' in fields:
                self.receipt.update(int(fields['received_map_count']), int(fields['derivation_lag']),
                                    int(fields['fine_traversability_revision']),
                                    int(fields['global_guidance_revision']), int(fields['duplicate_map_count']))
                if int(fields['rejected_map_count']):
                    self.phase = 'MAP_INPUT_REJECTED'
                    self.get_logger().error('Formal mapper rejected an elevation input; inspect planning diagnostics')

    def fresh_pose(self):
        return self.latest is not None and time.monotonic()-self.latest_received <= self.settings['odometry_timeout_s']

    def tick(self):
        if self.stopping or self.phase in ('MAP_INPUT_REJECTED', 'OBSERVATION_ERROR'):
            return
        if not self.fresh_pose() or not self.map_pub.get_subscription_count():
            return
        if self.mode == 'nav':
            self.publish_known_chunk()
        elif self.observation_future is None:
            stamp = self.latest.header.stamp
            key = (stamp.sec, stamp.nanosec)
            if key != self.last_sample_key:
                p, q = self.latest.pose.pose.position, self.latest.pose.pose.orientation
                self.last_sample_key = key
                self.observation_stamp = stamp
                self.query_begin = time.perf_counter()
                self.observation_future = self.pool.submit(
                    self.terrain.observe, (p.x,p.y,p.z), (q.x,q.y,q.z,q.w), **self.sensor_options)
        self.advance_task()

    def finish_observation(self):
        if self.observation_future is None or not self.observation_future.done():
            return
        future, self.observation_future = self.observation_future, None
        try:
            observation = future.result()
            self.last_query_ms = (time.perf_counter()-self.query_begin)*1000
            if not self.stopping:
                self.publish_observation(observation, self.observation_stamp)
        except Exception as error:
            self.phase = 'OBSERVATION_ERROR'
            self.get_logger().error(f'OBJ observation failed: {error}')

    def publish_observation(self, observation, stamp):
        self.map_pub.publish(observation_grid_map(observation, stamp))
        self.samples += 1
        if self.mode == 'explore':
            self.coverage.add(observation)
            if self.phase == 'WAITING_FOR_STATE':
                self.phase = 'OBSERVING'
        if self.mode == 'explore' and self.observed_pub.get_subscription_count():
            indices = np.argwhere(np.isfinite(observation.elevation))
            measured = marker(0, Marker.POINTS, stamp, 'measured')
            measured.scale.x = measured.scale.y = observation.resolution
            measured.color.r, measured.color.g, measured.color.b, measured.color.a = 1., .9, .1, .7
            measured.points = [Point(x=float(observation.origin_xy[0]+(x+.5)*observation.resolution),
                                   y=float(observation.origin_xy[1]+(y+.5)*observation.resolution),
                                   z=float(observation.elevation[y,x])+.03) for y,x in indices]
            self.observed_pub.publish(MarkerArray(markers=[measured]))

    def publish_known_chunk(self):
        if self.known_done:
            return
        if self.known_chunks is None:
            p = self.latest.pose.pose.position
            self.known_chunks = iter(self.terrain.iter_known_chunks(
                int(self.settings['known_chunk_cells']), (p.x,p.y)))
            self.known_started = time.monotonic()
        if not self.receipt.ready:
            self.phase = 'WAITING_FOR_KNOWN_MAP_DERIVATION'
            # Re-send only if no input receipt arrived, never race a slow derivation.
            if (time.monotonic()-self.last_known_sent > 2.
                    and self.receipt.current[0] == self.receipt.waiting[0]):
                self.map_pub.publish(observation_grid_map(self.pending_chunk, self.latest.header.stamp))
                self.last_known_sent = time.monotonic()
            return
        try:
            self.pending_chunk = next(self.known_chunks)
        except StopIteration:
            self.known_done = True
            self.phase = 'KNOWN_MAP_READY'
            self.get_logger().info(f'Known-map loading finished: {self.known_count} chunks accepted and derived in {time.monotonic()-self.known_started:.2f}s')
            return
        self.receipt.sent()
        self.publish_observation(self.pending_chunk, self.latest.header.stamp)
        self.known_count += 1
        self.last_known_sent = time.monotonic()
        self.phase = 'LOADING_KNOWN_MAP'

    def advance_task(self):
        # A failed scan stops task advancement, not the observation pipeline.
        if self.phase == 'BOOTSTRAP_FAILED':
            return
        if self.mode != 'explore' or not self.settings['auto_start'] or self.task_started:
            return
        if not self.map_seen or not self.samples or not self.client.server_is_ready():
            return
        previous_phase=self.phase
        previous_reanchors=self.scan.reanchors
        now=time.monotonic()
        command_stopped=(now-self.command_received<.5 and abs(self.command.linear.x)<1e-3
                         and abs(self.command.angular.z)<1e-3)
        self.phase=self.scan.advance(self.latest,command_stopped,now)
        self.scan_cancellation=self.scan.cancel_token
        if self.scan.reanchors!=previous_reanchors:
            p=self.latest.pose.pose.position
            self.get_logger().info(f'Reanchored initial scan at ({p.x:.3f}, {p.y:.3f}); formal replanning requested')
        if self.phase=='BOOTSTRAP_FAILED' and previous_phase!=self.phase:
            self.get_logger().error('Initial scan failed: '+self.scan.reason)
        if self.phase!='SCAN_COMPLETE':
            return
        # One subscription belongs to this node (shutdown cancellation).
        if self.exploration_received and self.task_pub.get_subscription_count() > 1:
            self.task_pub.publish(exploration_task(self.task_bounds, self.get_clock().now().to_msg()))
            self.task_started = True
            self.phase = 'EXPLORATION_TASK_STARTED'
            self.get_logger().info('Started exploration inside the fixed map task boundary')

    def publish_status(self):
        status = {'task_bounds_xy':self.task_bounds, 'mode':self.mode, 'phase':self.phase, 'fresh_feedback':self.fresh_pose(),
                  'samples':self.samples, 'query_ms':self.last_query_ms,
                  'observed_cells':self.coverage.observed_cells, 'task_cells':self.coverage.total_cells,
                  'observation_coverage':self.coverage.observed_cells/max(1,self.coverage.total_cells),
                  'known_chunks':self.known_count, 'tracking':self.tracking_text,
                  'command_v':self.command.linear.x, 'command_w':self.command.angular.z,
                  'scan_reanchors': self.scan.reanchors, 'scan_reason': self.scan.reason,
                  'scan_targets_remaining': len(self.scan.targets) if self.scan.targets is not None else 0,
                  'fresh_command':time.monotonic()-self.command_received <= .5,
                  'exploration':self.exploration_text, 'trace_retained_poses':len(self.poses)}
        self.status_pub.publish(String(data=json.dumps(status)))
        if self.latest is None:
            return
        stamp = self.latest.header.stamp
        p = self.latest.pose.pose.position
        if self.trace_pub.get_subscription_count():
            trace = Path(); trace.header.frame_id, trace.header.stamp = 'map', stamp
            trace.poses = list(self.poses); self.trace_pub.publish(trace)
        if self.hud_pub.get_subscription_count():
            hud = marker(0, Marker.TEXT_VIEW_FACING, stamp, 'status')
            hud.pose.position = Point(x=p.x+2., y=p.y+2., z=p.z+2.)
            hud.scale.z = .35; hud.color.r = hud.color.g = hud.color.b = hud.color.a = 1.
            v, w = self.latest.twist.twist.linear, self.latest.twist.twist.angular
            hud.text = (f'{self.mode}: {self.phase}\nfeedback vx={v.x:.3f} m/s wz={w.z:.3f} rad/s\n'
                        f'command v={self.command.linear.x:.3f} w={self.command.angular.z:.3f}\n'
                        f'observed={status["observation_coverage"]:.1%} query={self.last_query_ms:.0f} ms\n'
                        f'scan remaining={status["scan_targets_remaining"]} reanchors={self.scan.reanchors}\n'
                        f'{self.tracking_text}\n{self.exploration_text}')
            self.hud_pub.publish(MarkerArray(markers=[hud]))
        if self.display_pub.get_subscription_count():
            self.display_pub.publish(scene_markers(self.latest, self.settings, self.task_bounds))
        if not self.truth_published and self.truth_pub.get_subscription_count():
            self.truth_pub.publish(truth_preview(self.terrain, stamp))
            self.truth_published = True

    def stop_task(self):
        self.stopping = True
        if self.task_started:
            self.task_pub.publish(exploration_task(self.task_bounds,
                                                  self.get_clock().now().to_msg(), PureExplorationTask.CANCEL))
        if self.scan_cancellation is not None:
            return self.scan_cancellation.cancel()
        return None

    def destroy_node(self):
        self.stopping = True
        self.pool.shutdown(wait=True, cancel_futures=True)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = ObjVirtualSensorNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            future = node.stop_task()
            if future is not None and rclpy.ok():
                rclpy.spin_until_future_complete(node, future, timeout_sec=2.)
                pending = node.scan_cancellation.cancel_future
                if pending is not None:
                    rclpy.spin_until_future_complete(node, pending, timeout_sec=2.)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
