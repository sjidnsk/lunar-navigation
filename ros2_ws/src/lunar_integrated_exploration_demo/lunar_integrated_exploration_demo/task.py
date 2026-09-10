"""Bootstrap scan through formal navigation, then submit the exploration polygon."""
import math
import json
import time
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Point32
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String
from lunar_planning_msgs.action import NavigateToPose
from lunar_pure_exploration_msgs.msg import PureExplorationTask

PREFIX = '/lunar_demo/integrated'


class TaskStarter(Node):
    def __init__(self):
        super().__init__('integrated_task_starter')
        self.size = float(self.declare_parameter('task_size_m', 300.0).value)
        self.scan = bool(self.declare_parameter('initial_scan', True).value)
        self.client = ActionClient(self, NavigateToPose, PREFIX+'/navigate_to_pose')
        self.publisher = self.create_publisher(PureExplorationTask, PREFIX+'/exploration/task', 10)
        self.progress = self.create_publisher(String, PREFIX+'/bootstrap', 10)
        self.bootstrap_ids = []
        self.sessions_pub = self.create_publisher(String, PREFIX+'/bootstrap_sessions', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.map_seen = False
        self.pose = None
        self.phase = 'WAITING_FOR_MAP_AND_STATE'
        self.yaws = [math.pi/2, math.pi, -math.pi/2, 0.] if self.scan else []
        self.future = None
        self.last_step = time.monotonic()
        self.create_subscription(OccupancyGrid, PREFIX+'/exploration_map', self.on_map,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Odometry, PREFIX+'/exploration/odometry', self.on_pose, 10)
        self.create_timer(.1, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def on_map(self, msg):
        self.map_seen = any(0 <= c < 50 for c in msg.data)

    def on_pose(self, msg):
        self.pose = msg.pose.pose.position

    def tick(self):
        self.progress.publish(String(data=self.phase))
        if self.phase in ('EXPLORATION_TASK_STARTED', 'BOOTSTRAP_FAILED'):
            return
        if not self.map_seen or self.pose is None or not self.client.server_is_ready():
            return
        if self.future:
            if not self.future.done():
                if time.monotonic()-self.last_step > 45:
                    self.phase = 'BOOTSTRAP_FAILED'
                    self.get_logger().error('Bootstrap did not finish within 45 wall seconds; no exploration task sent')
                return
            response = self.future.result()
            self.future = None
            if self.phase == 'WAITING_SCAN_ACCEPTANCE':
                if not response.accepted:
                    self.phase = 'BOOTSTRAP_FAILED'
                    return
                self.bootstrap_ids.append(bytes(response.goal_id.uuid).hex())
                self.future = response.get_result_async()
                self.phase = 'EXECUTING_SCAN'
                return
            if response.result.outcome != 0 or response.result.reason_code != 'GOAL_REACHED':
                self.phase = 'BOOTSTRAP_FAILED'
                self.get_logger().error('Initial scan failed: '+response.result.reason_code)
                return
        if self.yaws:
            yaw = self.yaws.pop(0)
            goal = NavigateToPose.Goal(target_x_m=self.pose.x, target_y_m=self.pose.y,
                                       has_target_yaw=True, target_yaw_rad=yaw)
            self.future = self.client.send_goal_async(goal)
            self.phase = 'WAITING_SCAN_ACCEPTANCE'
            self.last_step = time.monotonic()
            return
        if self.publisher.get_subscription_count() == 0:
            return
        msg = PureExplorationTask()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.task_id, msg.command = 'integrated-complex-terrain', PureExplorationTask.START
        half = self.size/2
        msg.boundary.points = [Point32(x=x,y=y,z=0.) for x,y in
                               [(-half,-half),(half,-half),(half,half),(-half,half)]]
        # Publish one complete receipt only after every scan succeeds. An empty
        # list is a valid receipt only when scanning was explicitly disabled.
        self.sessions_pub.publish(String(data=json.dumps(self.bootstrap_ids)))
        self.publisher.publish(msg)
        self.phase = 'EXPLORATION_TASK_STARTED'
        self.get_logger().info(f'Started {self.size:g}m exploration after formal scan completion')


def main(args=None):
    rclpy.init(args=args); node = TaskStarter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
