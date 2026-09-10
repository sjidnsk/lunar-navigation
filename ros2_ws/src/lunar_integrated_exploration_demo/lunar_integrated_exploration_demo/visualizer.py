"""Read-only diagnostics and visualization; low-rate exact pose relay for exploration."""
import json
from copy import deepcopy
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Odometry, Path
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import PoseStamped, Point
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from lunar_planning_msgs.msg import TrackingStatus
from lunar_pure_exploration_msgs.msg import PureExplorationStatus
from .terrain import Terrain

PREFIX = '/lunar_demo/integrated'
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
STATE = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


def observed_world_points(message):
    """Decode this simulator's zero-start GridMap without filling unknown cells."""
    if message.outer_start_index or message.inner_start_index:
        raise ValueError('demo observations must have zero circular-buffer starts')
    layer = message.data[message.layers.index('elevation')]
    height, width = (int(d.size) for d in layer.layout.dim)
    values = np.asarray(layer.data, dtype=np.float32).reshape(height, width)[::-1, ::-1]
    iy, ix = np.nonzero(np.isfinite(values))
    info = message.info
    origin_x = info.pose.position.x - info.length_x / 2
    origin_y = info.pose.position.y - info.length_y / 2
    return np.column_stack((origin_x + (ix + .5) * info.resolution,
                            origin_y + (iy + .5) * info.resolution,
                            values[iy, ix]))


class Visualizer(Node):
    def __init__(self):
        super().__init__('integrated_visualizer')
        self.scene_size = float(self.declare_parameter('scene_size_m',300.).value)
        self.sensor_range = float(self.declare_parameter('sensor_range_m',12.).value)
        self.sensor_fov = float(self.declare_parameter('sensor_fov_deg',120.).value)
        self.terrain = Terrain(size_m=self.scene_size)
        self.odom = self.tf = self.status = self.tracking = None
        self.plant = {}; self.bootstrap = 'WAITING'
        self.trace = Path(); self.trace.header.frame_id = 'map'
        self.pose_relay = self.create_publisher(Odometry,PREFIX+'/exploration/odometry',10)
        self.tf_relay = self.create_publisher(TFMessage,PREFIX+'/exploration/tf',10)
        self.trace_pub = self.create_publisher(Path,PREFIX+'/trace',LATCHED)
        self.markers = self.create_publisher(MarkerArray,PREFIX+'/display',LATCHED)
        self.global_hud_pub = self.create_publisher(MarkerArray,PREFIX+'/hud_global',LATCHED)
        self.local_hud_pub = self.create_publisher(MarkerArray,PREFIX+'/hud_local',LATCHED)
        self.observed_pub = self.create_publisher(MarkerArray,PREFIX+'/observed_cells',LATCHED)
        self.terrain_pub = self.create_publisher(MarkerArray,PREFIX+'/terrain_preview',LATCHED)
        self.create_subscription(Odometry,PREFIX+'/odometry',lambda m:setattr(self,'odom',m),STATE)
        self.create_subscription(TFMessage,'/tf',lambda m:setattr(self,'tf',m),STATE)
        self.create_subscription(PureExplorationStatus,PREFIX+'/exploration/status',lambda m:setattr(self,'status',m),10)
        self.create_subscription(TrackingStatus,PREFIX+'/tracking_status',lambda m:setattr(self,'tracking',m),10)
        self.create_subscription(String,PREFIX+'/plant_state',lambda m:setattr(self,'plant',json.loads(m.data)),10)
        self.create_subscription(String,PREFIX+'/bootstrap',lambda m:setattr(self,'bootstrap',m.data),10)
        self.create_subscription(GridMap,PREFIX+'/grid_map',self.on_observation,LATCHED)
        self.create_timer(.2,self.tick,clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.preview_sent = False

    def on_observation(self, message):
        marker = self.marker('actual_observed_cells', Marker.POINTS)
        marker.header = deepcopy(message.header)
        marker.scale.x = marker.scale.y = .08
        marker.color = ColorRGBA(r=1., g=.8, b=.1, a=.55)
        marker.points = [Point(x=float(x), y=float(y), z=.8)
                         for x, y, _ in observed_world_points(message)]
        self.observed_pub.publish(MarkerArray(markers=[marker]))

    def marker(self, name, kind, frame='map'):
        m = Marker(); m.header.frame_id = frame; m.header.stamp=self.get_clock().now().to_msg()
        m.ns=name; m.id=0; m.type=kind; m.action=Marker.ADD; m.pose.orientation.w=1.; m.color.a=1.
        return m

    def preview(self):
        surface=self.terrain.preview(resolution=2.)
        m=self.marker('terrain_truth_display_only',Marker.POINTS)
        m.scale.x=m.scale.y=1.9
        a=np.asarray(surface['values']); low=float(np.min(a)); span=max(float(np.max(a))-low,.1)
        for iy,y in enumerate(surface['y']):
            for ix,x in enumerate(surface['x']):
                h=float(a[iy,ix]);ratio=(h-low)/span
                m.points.append(Point(x=float(x),y=float(y),z=h-.15))
                m.colors.append(ColorRGBA(r=.22+.32*ratio,g=.20+.23*ratio,b=.17+.16*ratio,a=.35))
        rocks=self.marker('obstacle_truth_display_only',Marker.LINE_LIST)
        rocks.scale.x=.13;rocks.color.r=.7;rocks.color.g=.55;rocks.color.b=.4;rocks.color.a=.65
        for o in self.terrain.obstacles:
            cs,sn=math.cos(o.yaw),math.sin(o.yaw)
            pts=[(o.x+cs*x-sn*y,o.y+sn*x+cs*y)for x,y in
                 [(-o.length/2,-o.width/2),(o.length/2,-o.width/2),(o.length/2,o.width/2),(-o.length/2,o.width/2)]]
            for i in range(4):
                for x,y in (pts[i],pts[(i+1)%4]):rocks.points.append(Point(x=x,y=y,z=.2))
        self.terrain_pub.publish(MarkerArray(markers=[m,rocks]))
        self.preview_sent=True

    def tick(self):
        if not self.preview_sent and self.get_clock().now().nanoseconds>0:self.preview()
        if self.odom is None:return
        self.pose_relay.publish(self.odom)
        if self.tf:self.tf_relay.publish(self.tf)
        p=self.odom.pose.pose.position;q=self.odom.pose.pose.orientation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        # map->odom is identity in this simulator; preserve source pose stamps.
        if not self.trace.poses or math.hypot(p.x-self.trace.poses[-1].pose.position.x,p.y-self.trace.poses[-1].pose.position.y)>.08:
            pose=PoseStamped();pose.header=deepcopy(self.odom.header);pose.header.frame_id='map';pose.pose=deepcopy(self.odom.pose.pose)
            self.trace.poses.append(pose)
        self.trace.header.stamp=self.odom.header.stamp;self.trace_pub.publish(self.trace)
        body=self.marker('vehicle',Marker.CUBE);body.pose=deepcopy(self.odom.pose.pose);body.pose.position.z=p.z+.2
        body.scale.x=1.182;body.scale.y=.818;body.scale.z=.4;body.color.g=.9;body.color.b=1.
        sensor=self.marker('geometric_fov_limit_not_observation',Marker.LINE_STRIP);sensor.scale.x=.12;sensor.color.g=.9;sensor.color.b=.9
        sensor.points.append(Point(x=p.x,y=p.y,z=.3))
        for a in np.linspace(yaw-math.radians(self.sensor_fov/2),yaw+math.radians(self.sensor_fov/2),49):
            sensor.points.append(Point(x=p.x+self.sensor_range*math.cos(a),y=p.y+self.sensor_range*math.sin(a),z=.3))
        sensor.points.append(Point(x=p.x,y=p.y,z=.3))
        window=self.marker('planning_window_64m',Marker.LINE_STRIP);window.scale.x=.15;window.color.r=.7;window.color.b=1.
        window.points=[Point(x=p.x+x,y=p.y+y,z=.3)for x,y in [(-32,-32),(32,-32),(32,32),(-32,32),(-32,-32)]]
        s=self.status;t=self.tracking
        stages=['WAITING','BRAKING','ALIGNING','TRACKING','FINAL_ALIGN','COMPLETED','FAILED']
        stage=stages[t.state] if t is not None and t.state<len(stages) else 'WAITING'
        scale=self.plant.get('requested_time_scale',self.plant.get('time_scale',30))
        actual=self.plant.get('actual_time_scale',0)
        hud=(f'MAP -> FRONTIER -> COARSE ROUTE -> FINE SEARCH -> EXECUTOR\n'
             f'{self.scene_size:g}m scene | coarse 1.0m / fine 0.2m | target {scale:.0f}x / actual {actual:.1f}x\n'
             f'classified coarse cells {100*s.coverage_ratio:.2f}% | goals {s.completed_goal_count} | frontiers {s.frontier_cluster_count}\n'
             f'views: generated {s.candidate_count} | eligible {s.reachable_candidate_count}\n' if s else
             f'BOOTSTRAP: {self.bootstrap}\ncoarse 1.0m / fine 0.2m | target {scale:.0f}x / actual {actual:.1f}x\n')
        hud+=f'{s.reason_code if s else self.bootstrap}\nv={self.odom.twist.twist.linear.x:+.3f}m/s  w={self.odom.twist.twist.angular.z:+.3f}rad/s | {stage} rev={t.segment_revision if t else 0}'
        global_hud=self.marker('global_hud',Marker.TEXT_VIEW_FACING);global_hud.pose.position=Point(x=0.,y=self.scene_size*.575,z=10.);global_hud.scale.z=self.scene_size/55;global_hud.color=ColorRGBA(r=.95,g=.95,b=.95,a=1.);global_hud.text=hud
        # The camera follows position in map axes: keep the label above the
        # vehicle, independent of its yaw, with a readable opaque background.
        hud_background=self.marker('local_hud_background',Marker.CUBE)
        hud_background.pose.position=Point(x=p.x,y=p.y+11.5,z=9.9)
        hud_background.scale.x=28.;hud_background.scale.y=3.8;hud_background.scale.z=.02
        hud_background.color=ColorRGBA(r=.05,g=.06,b=.08,a=1.)
        local_hud=self.marker('local_hud',Marker.TEXT_VIEW_FACING);local_hud.pose.position=Point(x=p.x,y=p.y+11.5,z=10.);local_hud.scale.z=.45;local_hud.color=global_hud.color;local_hud.text=hud
        self.markers.publish(MarkerArray(markers=[body,sensor,window]))
        self.global_hud_pub.publish(MarkerArray(markers=[global_hud]))
        self.local_hud_pub.publish(MarkerArray(markers=[hud_background,local_hud]))


def main(args=None):
    rclpy.init(args=args);node=Visualizer()
    try:rclpy.spin(node)
    except (KeyboardInterrupt,ExternalShutdownException):pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
