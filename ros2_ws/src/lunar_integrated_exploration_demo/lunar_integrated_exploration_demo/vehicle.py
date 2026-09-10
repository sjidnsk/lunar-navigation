"""Wall-paced /clock, constrained command-only vehicle, and local observations."""
from array import array
import json
import math
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, Twist
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.clock import Clock as RclClock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from tf2_ros import TransformBroadcaster

from .clock import PublicationSchedule, finite_number
from .physics import VehicleSimulation

PREFIX = '/lunar_demo/integrated'
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
STATE = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                   durability=DurabilityPolicy.VOLATILE)


def observation_grid_map(observation, stamp):
    """Encode ascending [y,x] cell centres for the existing GridMap adapter.

    GridMap's physical indices run towards negative x/y. In column-major
    storage with zero circular-buffer starts, both axes reverse a flat [y,x]
    matrix. This deliberately preserves NaN cells outside actual observation.
    """
    resolution = finite_number(observation['resolution'], 'resolution', 0.01, 2.0)
    origin_x, origin_y = float(observation['origin_x']), float(observation['origin_y'])
    for origin in (origin_x, origin_y):
        if not math.isfinite(origin) or not math.isclose(
                origin / resolution, round(origin / resolution), abs_tol=1e-7):
            raise ValueError('observation origin must be finite and snapped to resolution')
    values = np.asarray(observation['values'], dtype=np.float32)
    if values.ndim != 2 or not values.size or np.isinf(values).any():
        raise ValueError('observation must be a nonempty [y,x] matrix of finite values or NaN')
    height, width = values.shape
    msg = GridMap()
    msg.header.stamp, msg.header.frame_id = stamp, 'odom'
    msg.info.resolution = resolution
    msg.info.length_x, msg.info.length_y = width * resolution, height * resolution
    msg.info.pose.position.x = origin_x + msg.info.length_x / 2
    msg.info.pose.position.y = origin_y + msg.info.length_y / 2
    msg.info.pose.orientation.w = 1.0
    msg.layers, msg.basic_layers = ['elevation'], ['elevation']
    layer = Float32MultiArray()
    layer.layout.dim = [
        MultiArrayDimension(label='column_index', size=height, stride=width * height),
        MultiArrayDimension(label='row_index', size=width, stride=width)]
    # Populate an array directly: a Python float list makes large map publication
    # unnecessarily expensive at accelerated simulation rates.
    layer.data = array('f', values[::-1, ::-1].ravel())
    msg.data = [layer]
    return msg


class VehicleNode(Node):
    def __init__(self, **kwargs):
        super().__init__('integrated_vehicle', **kwargs)
        scale = self.declare_parameter('time_scale', 30.0, ParameterDescriptor(
            dynamic_typing=True,
            description='Requested simulated seconds per wall second, finite in [1,60].')).value
        readonly = ParameterDescriptor(read_only=True)

        def config(name, default, minimum, maximum):
            return finite_number(self.declare_parameter(name, default, readonly).value,
                                 name, minimum, maximum)

        self.scene_size_m = config('scene_size_m', 300.0, 20.0, 1000.0)
        self.sensor_range_m = config('sensor_range_m', 12.0, 2.0, 30.0)
        self.sensor_fov_deg = config('sensor_fov_deg', 120.0, 30.0, 360.0)
        self.fine_resolution_m = config('fine_resolution_m', 0.2, 0.2, 0.2)
        self.local_window_m = config('local_window_m', 28.0, 8.0, 64.0)
        self.near_field_radius_m = config('near_field_radius_m', 1.4, 0.0, 3.0)
        self.map_sim_rate_hz = config('map_sim_rate_hz', 1.0, 0.05, 20.0)
        self.map_wall_cap_hz = config('map_wall_cap_hz', 5.0, 0.5, 30.0)
        command_timeout = config('command_timeout_s', 0.3, 0.05, 2.0)
        seed = self.declare_parameter('seed', 20260910, readonly).value
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ValueError('seed must be an integer in [0, 2**32-1]')
        if self.local_window_m < 2 * self.sensor_range_m:
            raise ValueError('local_window_m must contain the full sensor diameter')
        if self.near_field_radius_m > self.sensor_range_m:
            raise ValueError('near_field_radius_m must not exceed sensor_range_m')
        cells = self.local_window_m / self.fine_resolution_m
        if not math.isclose(cells, round(cells), abs_tol=1e-7):
            raise ValueError('local_window_m must be an integer multiple of fine_resolution_m')

        from .terrain import Terrain
        self.terrain = Terrain(size_m=self.scene_size_m, seed=seed)
        if self.terrain.collides(0.0, 0.0, 0.0):
            raise ValueError('the initial vehicle footprint is in collision')
        self.sim = VehicleSimulation(time_scale=scale, start_wall_s=time.monotonic(),
                                     collision=self.terrain.collides,
                                     command_timeout_s=command_timeout,
                                     sensor=self.observe, sample_sim_rate_hz=self.map_sim_rate_hz)
        self.map_schedule = PublicationSchedule(self.map_wall_cap_hz)
        self.batch_sample_count = self.published_sample_count = self.batch_observed_cells = 0
        self.batch_first_sample_sim_s = self.batch_last_sample_sim_s = None
        self.map_publication_wall_ms = self.max_map_publication_wall_ms = 0.0
        self.map_length_x_m = self.map_length_y_m = 0.0
        self.last_map_origin = (0.0, 0.0)
        self._last_elevation_xy = None
        self._body_z = 0.0

        self.clock_pub = self.create_publisher(Clock, '/clock', STATE)
        self.odom_pub = self.create_publisher(Odometry, PREFIX + '/odometry', STATE)
        self.map_pub = self.create_publisher(GridMap, PREFIX + '/grid_map', LATCHED)
        self.telemetry_pub = self.create_publisher(String, PREFIX + '/plant_state', 1)
        self.tf = TransformBroadcaster(self, qos=QoSProfile(depth=1))
        self.create_subscription(Twist, PREFIX + '/cmd_vel', self.on_command, 1)

        # Never use the ROS clock for any timer which produces /clock itself.
        self.wall_clock = RclClock(clock_type=ClockType.STEADY_TIME)
        self.clock_timer = self.create_timer(self.sim.clock.wall_period_s, self.tick,
                                             clock=self.wall_clock)
        self.map_timer = self.create_timer(min(0.02, 1.0 / self.map_wall_cap_hz),
                                           self.publish_map, clock=self.wall_clock)
        self.telemetry_timer = self.create_timer(0.2, self.publish_state, clock=self.wall_clock)
        self.add_on_set_parameters_callback(self.on_parameters)
        self.publish_pose_clock()
        self.get_logger().info(
            f'Command-only vehicle: +/-0.2 m/s in simulation time; requested {scale:g}x, '
            f'max clock step 0.02 sim s; observation {self.sensor_range_m:g} m / '
            f'{self.sensor_fov_deg:g} deg + {self.near_field_radius_m:g} m near field; '
            f'sensor samples {self.map_sim_rate_hz:g} Hz sim; measured-cell batches '
            f'publish at most {self.map_wall_cap_hz:g} Hz wall')

    def on_parameters(self, parameters):
        values = [p.value for p in parameters if p.name == 'time_scale']
        if not values:
            return SetParametersResult(successful=True)
        try:
            scale = finite_number(values[-1], 'time_scale', 1.0, 60.0)
        except ValueError as error:
            return SetParametersResult(successful=False, reason=str(error))
        # Account the preceding interval under its preceding scale. The emitted
        # clock and physical state therefore change together at the boundary.
        self.tick()
        self.sim.clock.set_time_scale(scale)
        self.clock_timer.timer_period_ns = int(self.sim.clock.wall_period_s * 1e9)
        self.clock_timer.reset()
        return SetParametersResult(successful=True)

    def on_command(self, message):
        other_finite = all(math.isfinite(value) for value in (
            message.linear.y, message.linear.z, message.angular.x, message.angular.y))
        self.sim.receive_command(message.linear.x, message.angular.z, other_finite)

    def tick(self):
        if self.sim.advance(time.monotonic()) > 0.0:
            self.publish_pose_clock()

    def stamp(self):
        sec, nanosec = self.sim.clock.stamp_parts()
        return Time(sec=sec, nanosec=nanosec)

    def publish_pose_clock(self):
        stamp = self.stamp()
        p = self.sim.plant
        self.clock_pub.publish(Clock(clock=stamp))
        if self._last_elevation_xy != (p.x, p.y):
            self._body_z = float(self.terrain.elevation(p.x, p.y))
            self._last_elevation_xy = (p.x, p.y)
            if not math.isfinite(self._body_z):
                raise RuntimeError('vehicle ground height became nonfinite')
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = stamp, 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x, odom.pose.pose.position.y = p.x, p.y
        odom.pose.pose.position.z = self._body_z
        odom.pose.pose.orientation.z = math.sin(p.yaw / 2)
        odom.pose.pose.orientation.w = math.cos(p.yaw / 2)
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = p.v, p.w
        self.odom_pub.publish(odom)
        identity = TransformStamped()
        identity.header.stamp, identity.header.frame_id = stamp, 'map'
        identity.child_frame_id, identity.transform.rotation.w = 'odom', 1.0
        body = TransformStamped()
        body.header, body.child_frame_id = odom.header, 'base_link'
        body.transform.translation.x, body.transform.translation.y = p.x, p.y
        body.transform.translation.z = self._body_z
        body.transform.rotation = odom.pose.pose.orientation
        self.tf.sendTransform([identity, body])

    def observe(self, x, y, yaw):
        return self.terrain.observe(
            x, y, yaw, resolution=self.fine_resolution_m,
            range_m=self.sensor_range_m, fov_deg=self.sensor_fov_deg,
            window_m=self.local_window_m, near_field_radius_m=self.near_field_radius_m)

    def publish_map(self):
        now_wall = time.monotonic()
        if not self.sim.pending_observation.sample_count or not self.map_schedule.ready(now_wall):
            return
        start = time.perf_counter()
        batch = self.sim.pending_observation.snapshot()
        sec, nanosec = divmod(round(batch['last_sample_sim_s'] * 1e9), 1_000_000_000)
        message = observation_grid_map(batch, Time(sec=sec, nanosec=nanosec))
        self.map_pub.publish(message)
        # Only release evidence after publish accepts the complete batch. No
        # finite sample is cropped to the latest robot-centred sensor window.
        self.sim.pending_observation.clear()
        self.map_schedule.take(batch['last_sample_sim_s'], now_wall)
        self.batch_sample_count = batch['sample_count']
        self.published_sample_count += batch['sample_count']
        self.batch_observed_cells = int(np.isfinite(batch['values']).sum())
        self.batch_first_sample_sim_s = batch['first_sample_sim_s']
        self.batch_last_sample_sim_s = batch['last_sample_sim_s']
        self.last_map_origin = (batch['origin_x'], batch['origin_y'])
        self.map_length_x_m, self.map_length_y_m = message.info.length_x, message.info.length_y
        self.map_publication_wall_ms = (time.perf_counter() - start) * 1000
        self.max_map_publication_wall_ms = max(self.max_map_publication_wall_ms,
                                                self.map_publication_wall_ms)

    def publish_state(self):
        state = self.sim.state()
        state.update(self.map_schedule.snapshot())
        state.update({
            'scene_size_m': self.scene_size_m,
            'sensor_range_m': self.sensor_range_m,
            'sensor_fov_deg': self.sensor_fov_deg,
            'near_field_radius_m': self.near_field_radius_m,
            'fine_resolution_m': self.fine_resolution_m,
            'local_window_m': self.local_window_m,
            'batch_sample_count': self.batch_sample_count,
            'published_sample_count': self.published_sample_count,
            'batch_observed_cells': self.batch_observed_cells,
            'batch_first_sample_sim_s': self.batch_first_sample_sim_s,
            'batch_last_sample_sim_s': self.batch_last_sample_sim_s,
            'map_origin_x': self.last_map_origin[0],
            'map_origin_y': self.last_map_origin[1],
            'map_length_x_m': self.map_length_x_m,
            'map_length_y_m': self.map_length_y_m,
            'map_publication_wall_ms': self.map_publication_wall_ms,
            'max_map_publication_wall_ms': self.max_map_publication_wall_ms,
            'z': self._body_z,
        })
        self.telemetry_pub.publish(String(data=json.dumps(state, allow_nan=False)))


def main(args=None):
    rclpy.init(args=args)
    node = VehicleNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
