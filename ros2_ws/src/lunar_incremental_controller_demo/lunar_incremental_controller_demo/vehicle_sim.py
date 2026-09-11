"""Synthetic elevation and a constrained, command-driven wheeled vehicle."""
import json
import math
import time
from dataclasses import asdict

import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from grid_map_msgs.msg import GridMap
from std_msgs.msg import String, Float32MultiArray, MultiArrayDimension
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from lunar_planning_msgs.msg import TrackingStatus
from .plant import Plant, Scene

PREFIX = '/lunar_demo/controller'
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


class VehicleSim(Node):
    def __init__(self):
        super().__init__('incremental_controller_vehicle')
        self.scene = Scene(self.declare_parameter('case', 'forward').value)
        self.map_size = float(self.declare_parameter('map_size_m', 32.0).value)
        if not math.isfinite(self.map_size) or self.map_size < 12 or self.map_size > 128:
            raise ValueError('map_size_m must be finite and in [12,128]')
        self.observation_radius = float(self.declare_parameter('observation_radius_m', 10.0).value)
        if not math.isfinite(self.observation_radius) or self.observation_radius <= 0:
            raise ValueError('observation_radius_m must be finite and positive')
        self.plant = Plant()
        self.command = (0.0, 0.0)
        self.command_time = None
        self.last_tick = time.monotonic()
        self.start_time = self.last_tick
        self.command_messages = 0
        self.raw_forward = self.raw_reverse = 0.0
        self.raw_violations = self.raw_invalid = 0
        self.odom_pub = self.create_publisher(Odometry, PREFIX + '/odometry', 10)
        self.map_pub = self.create_publisher(GridMap, PREFIX + '/grid_map', LATCHED)
        self.terrain_pub = self.create_publisher(OccupancyGrid, PREFIX + '/terrain', LATCHED)
        self.trace_pub = self.create_publisher(Path, PREFIX + '/trace', LATCHED)
        self.hud_pub = self.create_publisher(MarkerArray, PREFIX + '/hud', LATCHED)
        self.telemetry_pub = self.create_publisher(String, PREFIX + '/plant_state', 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Twist, PREFIX + '/cmd_vel', self.on_command, 10)
        self.tracking_text = 'WAITING'
        self.create_subscription(TrackingStatus, PREFIX + '/tracking_status', self.on_tracking, 10)
        self.trace = Path()
        self.trace.header.frame_id = 'map'
        self.create_timer(0.02, self.tick)
        self.create_timer(0.2, self.visualize)
        self.create_timer(0.5, self.publish_map)
        self.get_logger().info(f'command-only plant case={self.scene.name}, real time, max +/-0.2 m/s')

    def on_tracking(self, msg):
        names = ['WAITING', 'BRAKING', 'ALIGNING', 'TRACKING', 'FINAL_ALIGN', 'COMPLETED', 'FAILED']
        state = names[msg.state] if msg.state < len(names) else str(msg.state)
        self.tracking_text = f'{state} rev={msg.segment_revision} direction={msg.direction} {msg.reason}'

    def on_command(self, msg):
        self.command_messages += 1
        self.command = (msg.linear.x, msg.angular.z)
        self.command_time = time.monotonic()
        if not all(math.isfinite(v) for v in self.command):
            self.raw_invalid += 1
        else:
            self.raw_forward = max(self.raw_forward, msg.linear.x)
            self.raw_reverse = max(self.raw_reverse, -msg.linear.x)
            self.raw_violations += int(abs(msg.linear.x) > 0.2 + 1e-9)

    def tick(self):
        current = time.monotonic()
        dt, self.last_tick = current - self.last_tick, current
        command = self.command if self.command_time is not None and current - self.command_time <= 0.3 else (0., 0.)
        # Bounded integration steps preserve acceleration even if scheduling stalls.
        while dt > 1e-9:
            step = min(dt, 0.02)
            self.plant.step(step, *command, self.scene.collides)
            dt -= step
        stamp = self.get_clock().now().to_msg()
        p = self.plant
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = stamp, 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x, odom.pose.pose.position.y = p.x, p.y
        odom.pose.pose.orientation.z = math.sin(p.yaw / 2)
        odom.pose.pose.orientation.w = math.cos(p.yaw / 2)
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = p.v, p.w
        self.odom_pub.publish(odom)
        identity = TransformStamped()
        identity.header.stamp, identity.header.frame_id, identity.child_frame_id = stamp, 'map', 'odom'
        identity.transform.rotation.w = 1.0
        body = TransformStamped()
        body.header, body.child_frame_id = odom.header, 'base_link'
        body.transform.translation.x, body.transform.translation.y = p.x, p.y
        body.transform.rotation = odom.pose.pose.orientation
        self.tf.sendTransform([identity, body])
        self.latest_pose = PoseStamped(header=odom.header, pose=odom.pose.pose)
        self.latest_pose.header.frame_id = 'map'

    def publish_map(self):
        resolution = 0.2
        n = int(round(self.map_size / resolution))
        size = n * resolution
        values = [self.scene.elevation(-size/2 + (x+.5)*resolution,
                                       -size/2 + (y+.5)*resolution)
                  for y in range(n) for x in range(n)]
        observed = values
        if self.scene.name == 'multiple_exits':
            # Deliberate 360-degree synthetic rolling observation, not a LiDAR model.
            observed = [v if math.hypot(-size/2+(x+.5)*resolution-self.plant.x,
                                       -size/2+(y+.5)*resolution-self.plant.y) <= self.observation_radius
                        else math.nan for y in range(n) for x in range(n)
                        for v in [values[y*n+x]]]
        msg = GridMap()
        msg.header.stamp, msg.header.frame_id = self.get_clock().now().to_msg(), 'odom'
        msg.info.resolution = resolution
        msg.info.length_x = msg.info.length_y = size
        msg.info.pose.orientation.w = 1.0
        msg.layers, msg.basic_layers = ['elevation'], ['elevation']
        layer = Float32MultiArray()
        layer.layout.dim = [MultiArrayDimension(label='column_index', size=n, stride=n*n),
                            MultiArrayDimension(label='row_index', size=n, stride=n)]
        # grid_map uses reversed row/column physical indices, column-major storage.
        layer.data = list(reversed(observed))
        msg.data = [layer]
        self.map_pub.publish(msg)
        terrain = OccupancyGrid()
        terrain.header = msg.header
        terrain.header.frame_id = 'map'
        terrain.info.resolution, terrain.info.width, terrain.info.height = resolution, n, n
        terrain.info.origin.position.x = terrain.info.origin.position.y = -size/2
        terrain.info.origin.orientation.w = 1.0
        terrain.data = [100 if v else 0 for v in values]
        self.terrain_pub.publish(terrain)

    def visualize(self):
        if not hasattr(self, 'latest_pose'):
            return
        p = self.plant
        pose = self.latest_pose
        if not self.trace.poses or math.hypot(p.x-self.trace.poses[-1].pose.position.x,
                                            p.y-self.trace.poses[-1].pose.position.y) >= .03:
            self.trace.poses.append(pose)
        self.trace.header.stamp = pose.header.stamp
        self.trace_pub.publish(self.trace)
        body = Marker()
        body.header, body.ns, body.id, body.type = pose.header, 'vehicle', 0, Marker.CUBE
        body.pose = pose.pose
        body.pose.position.z = .18
        body.scale.x, body.scale.y, body.scale.z = 1.182, .818, .36
        body.color.g, body.color.b, body.color.a = .8, 1., .9
        hud = Marker()
        hud.header, hud.ns, hud.id, hud.type = pose.header, 'actual_speed', 1, Marker.TEXT_VIEW_FACING
        hud.pose.position.x, hud.pose.position.y, hud.pose.position.z = p.x, p.y+1.5, 1.0
        hud.pose.orientation.w, hud.scale.z, hud.color.a = 1., .35, 1.
        hud.color.r = hud.color.g = hud.color.b = 1.
        hud.text = (f'{self.scene.name} | actual v={p.v:+.3f} m/s w={p.w:+.3f} rad/s\n'
                    f'cmd={self.command[0]:+.3f} | cap violations={self.raw_violations} '
                    f'contacts={p.collisions}\n{self.tracking_text}')
        self.hud_pub.publish(MarkerArray(markers=[body, hud]))
        state = asdict(p)
        state.update(case=self.scene.name, observation_radius_m=self.observation_radius, map_size_m=self.map_size, elapsed_wall_s=time.monotonic()-self.start_time,
                     command_messages=self.command_messages, raw_command_forward_max=self.raw_forward,
                     raw_command_reverse_max=self.raw_reverse, raw_command_limit_violations=self.raw_violations,
                     raw_invalid_commands=self.raw_invalid)
        self.telemetry_pub.publish(String(data=json.dumps(state)))


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = VehicleSim()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
