"""Measured-input ROS adapters shared by training and deployment.

No Scene, environment, reference calculator, learner or Critic dependency.
"""
from dataclasses import dataclass
from collections import deque
import math
import time
import numpy as np
from .contracts import Pose
from .maps import PolicyMapStore


class InfrastructureError(RuntimeError):
    """Transport/owned process failure; no constructable final transition."""


def stamp_ns(stamp): return int(stamp.sec)*1_000_000_000+int(stamp.nanosec)


def ros_time(nanoseconds):
    from builtin_interfaces.msg import Time
    sec,ns=divmod(int(nanoseconds),1_000_000_000)
    return Time(sec=sec,nanosec=ns)


class MeasurementBuffer:
    """Only real hit centers, retained until producer processed-stamp ACK.

    Latest data per hit cell is sufficient in a static scene; the stamp prevents
    an earlier ACK erasing a newly observed copy. No hidden neighbours are filled.
    """
    def __init__(self, origin, resolution_m):
        self.origin=origin;self.resolution_m=resolution_m
        self._cells={}

    def add(self, measurements, stamp):
        for x,y,h,stats in zip(measurements.cols,measurements.rows,measurements.heights,measurements.stats):
            self._cells[(int(x),int(y))]=(int(stamp),float(h),stats)

    def acknowledge(self, stamp):
        self._cells={k:v for k,v in self._cells.items() if v[0]>stamp}

    def payload(self):
        if not self._cells: return None
        cells=np.asarray(list(self._cells),np.int64)
        lo=cells.min(0);hi=cells.max(0)+1
        planes=np.full((5,int(hi[1]-lo[1]),int(hi[0]-lo[0])),np.nan,np.float32)
        values=list(self._cells.values())
        rows,cols=cells[:,1]-lo[1],cells[:,0]-lo[0]
        planes[0,rows,cols]=[v[1] for v in values]
        planes[1:,rows,cols]=np.asarray([v[2] for v in values],np.float32).T
        return (self.origin[0]+lo[0]*self.resolution_m,
                self.origin[1]+lo[1]*self.resolution_m),planes


def grid_message(planes, origin, resolution, nanoseconds, frame='odom'):
    from grid_map_msgs.msg import GridMap
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension
    msg=GridMap();msg.header.frame_id=frame;msg.header.stamp=ros_time(nanoseconds)
    _,height,width=planes.shape
    msg.info.resolution=float(resolution)
    msg.info.length_x=float(width*resolution);msg.info.length_y=float(height*resolution)
    msg.info.pose.position.x=float(origin[0]+width*resolution/2)
    msg.info.pose.position.y=float(origin[1]+height*resolution/2)
    msg.info.pose.orientation.w=1.
    msg.layers=['elevation','terrain_slope','terrain_relief','terrain_positive_rise','terrain_complete']
    msg.basic_layers=['elevation']
    for plane in planes:
        value=Float32MultiArray()
        value.layout.dim=[MultiArrayDimension(label='column_index',size=height,stride=height*width),
                          MultiArrayDimension(label='row_index',size=width,stride=width)]
        value.data=np.ascontiguousarray(plane[::-1,::-1],dtype=np.float32).ravel()
        msg.data.append(value)
    return msg


def odometry_message(pose,v,w,nanoseconds,height=0.):
    from nav_msgs.msg import Odometry
    msg=Odometry();msg.header.stamp=ros_time(nanoseconds)
    msg.header.frame_id='odom';msg.child_frame_id='base_link'
    msg.pose.pose.position.x=float(pose.x);msg.pose.pose.position.y=float(pose.y)
    msg.pose.pose.position.z=float(height)
    msg.pose.pose.orientation.w=math.cos(pose.yaw/2);msg.pose.pose.orientation.z=math.sin(pose.yaw/2)
    msg.twist.twist.linear.x=float(v);msg.twist.twist.angular.z=float(w)
    return msg


def pose_in_map(odometry, transform=None):
    """Measured base pose with actual direct map->odom TF, including 3D rotation."""
    from scipy.spatial.transform import Rotation
    q=odometry.pose.pose.orientation
    rotation=Rotation.from_quat([q.x,q.y,q.z,q.w])
    p=odometry.pose.pose.position
    position=np.array([p.x,p.y,p.z],dtype=float)
    if transform is not None:
        q=transform.transform.rotation;t=transform.transform.translation
        parent=Rotation.from_quat([q.x,q.y,q.z,q.w])
        position=parent.apply(position)+[t.x,t.y,t.z]
        rotation=parent*rotation
    matrix=rotation.as_matrix()
    return Pose(float(position[0]),float(position[1]),math.atan2(matrix[1,0],matrix[0,0]))


@dataclass(frozen=True)
class NavigationResult:
    outcome: int
    reason_code: str
    last_segment_revision: int


class RosNavigationAdapter:
    """Nonblocking action/map client; caller owns spinning, time and lifecycle."""
    def __init__(self,node,*,action_name,policy_map_service,odometry_topic,
                 diagnostics_topic,path_reference_topic,tf_topic="/tf",map_frame="map",odom_frame="odom"):
        from rclpy.action import ActionClient
        from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
        from lunar_planning_msgs.action import NavigateToPose
        from lunar_planning_msgs.srv import GetPolicyMap
        from lunar_planning_msgs.msg import PathReference
        from nav_msgs.msg import Odometry
        from diagnostic_msgs.msg import DiagnosticArray
        from tf2_msgs.msg import TFMessage
        self.map_frame=map_frame;self.odom_frame=odom_frame;self._map_from_odom=None
        self.node=node;self._service_type=GetPolicyMap;self._action_type=NavigateToPose
        self.action=ActionClient(node,NavigateToPose,action_name)
        self.client=node.create_client(GetPolicyMap,policy_map_service)
        self.store=PolicyMapStore();self.snapshot=None;self.processed_stamp_ns=0
        self.pose=None;self.pose_history=deque(maxlen=4096);self.velocity=(float("nan"),float("nan"));self.pose_received_monotonic=0.
        self.diagnostics=deque(maxlen=64);self.references=deque(maxlen=64);self.planning_evidence=deque(maxlen=64)
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        state_qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
        self.subscriptions=[node.create_subscription(Odometry,odometry_topic,self._pose,state_qos),
            node.create_subscription(DiagnosticArray,diagnostics_topic,self._diagnostic,10),
            node.create_subscription(PathReference,path_reference_topic,self._reference,qos),
            node.create_subscription(TFMessage,tf_topic,self._tf,state_qos)]
        self._map_future=None;self._send_future=None;self._handle=None;self._result_future=None
        self._cancel=False;self._cancel_future=None;self.sent_goal=None
        self._force_full=False;self.map_reason="NO_RESPONSE"
        self._request_serial=0;self._minimum_response_serial=1;self._current_input=False

    def _tf(self,msg):
        for transform in msg.transforms:
            if transform.header.frame_id==self.map_frame and transform.child_frame_id==self.odom_frame:
                self._map_from_odom=transform

    def _pose(self,msg):
        if msg.header.frame_id==self.map_frame:
            self.pose=pose_in_map(msg)
        elif msg.header.frame_id==self.odom_frame and self._map_from_odom is not None:
            self.pose=pose_in_map(msg,self._map_from_odom)
        else:return
        self.velocity=(msg.twist.twist.linear.x,msg.twist.twist.angular.z)
        self.pose_received_monotonic=time.monotonic()
        self.pose_history.append((stamp_ns(msg.header.stamp),self.pose))

    def observation_pose(self,stamp):
        for stamp_value,pose in reversed(self.pose_history):
            if stamp_value<=stamp:return pose
        return None

    def _diagnostic(self,msg):
        for status in msg.status:
            values={v.key:v.value for v in status.values}
            self.diagnostics.append(values)
            if values.get("cycle_result") in ("PLAN_FOUND","GOAL_REACHED","NO_PATH","TIMEOUT"):
                self.planning_evidence.append(values)

    def _reference(self,msg):
        self.references.append({'session_id':bytes(msg.session_id.uuid).hex(),'segment_revision':int(msg.segment_revision),
            'fine_revision':int(msg.traversability_revision),
            'active':msg.state == msg.ACTIVE,'poses':len(msg.path.poses)})

    @property
    def ready(self): return self.client.service_is_ready() and self.action.server_is_ready()

    def require_fresh_snapshot(self):
        """Require a successful request issued after this lifecycle boundary.

        Keep the applied snapshot for delta recovery and cumulative evidence.
        An already-outstanding request cannot certify the new current anchor.
        """
        self._minimum_response_serial=self._request_serial+1
        self._current_input=False

    def poll_snapshot(self,minimum_map_stamp_ns=0):
        service_ready=self.client.service_is_ready()
        if not service_ready:
            self.require_fresh_snapshot()
            self.map_reason="SERVICE_UNAVAILABLE"
            if self._map_future is not None:
                # Only this adapter's request is retired; a dead server need
                # never resolve it, and its eventual reply must be ignored.
                self.client.remove_pending_request(self._map_future)
                self._map_future=None
        if self._map_future is not None and self._map_future.done():
            future=self._map_future;self._map_future=None
            self._current_input=False
            try:response=future.result()
            except Exception as exc:
                self.map_reason="MAP_TRANSPORT_ERROR: "+str(exc)
            else:
                self.map_reason=response.reason_code if response is not None else "NO_RESPONSE"
                if response is not None and response.ready:
                    try:self.snapshot=self.store.apply(response)
                    except (ValueError,TypeError,AttributeError) as exc:
                        self.map_reason=str(exc);self._force_full=True
                    else:
                        self._force_full=False
                        self.processed_stamp_ns=stamp_ns(response.processed_stamp)
                        self._current_input=service_ready and self._request_serial>=self._minimum_response_serial
        if self._map_future is None and service_ready:
            req=self._service_type.Request()
            req.since_revision=0 if self._force_full or self.snapshot is None else self.snapshot.revision
            req.minimum_map_stamp_ns=int(minimum_map_stamp_ns)
            self._map_future=self.client.call_async(req)
            self._request_serial+=1
        if self._current_input and self.snapshot is not None and self.processed_stamp_ns>=minimum_map_stamp_ns:
            return self.snapshot
        return None

    def begin_goal(self,goal):
        if self._send_future is not None: raise RuntimeError('navigation action already in flight')
        if not self.action.server_is_ready(): raise InfrastructureError('navigation action unavailable')
        goal=tuple(map(float,goal))
        if len(goal)!=3 or not all(map(math.isfinite,goal)): raise ValueError('finite x/y/yaw goal required')
        self.sent_goal=goal
        request=self._action_type.Goal(target_x_m=goal[0],target_y_m=goal[1],has_target_yaw=True,target_yaw_rad=goal[2])
        self._cancel=False;self._handle=None;self._result_future=None;self._cancel_future=None
        self._send_future=self.action.send_goal_async(request)

    def cancel_goal(self):
        self._cancel=True
        self._advance_goal()

    def _advance_goal(self):
        if self._send_future is None: return
        if self._handle is None and self._send_future.done():
            self._handle=self._send_future.result()
            if self._handle is None or not self._handle.accepted:
                self._send_future=None
                raise InfrastructureError('navigation action rejected')
            self._result_future=self._handle.get_result_async()
        if self._cancel and self._handle is not None and self._cancel_future is None:
            self._cancel_future=self._handle.cancel_goal_async()

    def poll_goal(self):
        self._advance_goal()
        if self._result_future is None or not self._result_future.done(): return None
        wrapped=self._result_future.result()
        if wrapped is None: raise InfrastructureError('navigation result transport lost')
        result=wrapped.result
        self._send_future=self._handle=self._result_future=None
        return NavigationResult(int(result.outcome),str(result.reason_code),int(result.last_segment_revision))

    @property
    def inflight(self): return self._send_future is not None

    def close(self):
        self.action.destroy()
        self.node.destroy_client(self.client)
        for sub in self.subscriptions:self.node.destroy_subscription(sub)


def velocity_is_stopped(velocity, policy=None):
    """Use the public executor's measured stop contract, with caller overrides."""
    if policy is None:
        from lunar_pure_wheeled_controller.tracking import TrackingPolicy
        policy=TrackingPolicy()
    return (abs(velocity[0])<=policy.stopped_linear_mps and
            abs(velocity[1])<=policy.stopped_angular_radps)
