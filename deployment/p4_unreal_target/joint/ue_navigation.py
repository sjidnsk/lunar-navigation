"""UE requests integrated into the existing P4 feedback adapter (no TCP client)."""
import json
import math
import struct
import time
import uuid
from pathlib import Path
import numpy as np
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import UInt8MultiArray, Empty, Bool
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
from grid_map_msgs.msg import GridMap
from unique_identifier_msgs.msg import UUID
from lunar_planning_msgs.action import NavigateToPose
from lunar_planning_msgs.msg import PathReference
from lunar_pure_exploration_msgs.msg import PureExplorationTask, PureExplorationStatus
from ue_path import decode_request, encode_response, Coordinates, ReferenceGate, height_at

class UENavigation:
    def __init__(self, adapter):
        self.adapter = adapter
        old, self.node = adapter.nodes
        self.reply_pub = old.create_publisher(UInt8MultiArray, '/car/ue_path_response', 10)
        self.authorize_pub = self.node.create_publisher(Empty, '/P4/input/ue_goal_authorize', 10)
        self.task_pub = self.node.create_publisher(PureExplorationTask, '/Car/T4/exploration/task', 10)
        self.client = ActionClient(self.node, NavigateToPose, '/Car/T4/navigation/navigate_to_pose')
        self.current = None
        self.pending = None
        self.handle = None
        self.sending = False
        self.canceling = False
        self.exploration = None
        self.tf = None
        self.tf_at = -math.inf
        self.grid = None
        self.coordinates = None
        self.gate = None
        self.last_status = 0
        self.status = {'state':'WAITING_FOR_UE_CLICK'}
        old.create_subscription(UInt8MultiArray, '/car/ue_path_request', self.request, 10)
        old.create_subscription(Bool, '/car/ue_connected', self.connection,
                                QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.node.create_subscription(TFMessage, '/P4/input/map_to_odom', self.transform, 10)
        self.node.create_subscription(GridMap, '/Car/T3/mapping/grid_map', self.map,
                                      QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))
        self.node.create_subscription(PureExplorationStatus, '/Car/T4/exploration/status', self.task, 10)
        self.node.create_subscription(PathReference, '/Car/T4/planning/path_reference', self.reference,
                                      QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # A later RViz goal or explicit START/RESUME supersedes a UE goal.
        self.node.create_subscription(PoseStamped, '/Car/T4/rviz_goal', lambda _: self.supersede('RVIZ_GOAL'), 10)
        self.node.create_subscription(PureExplorationTask, '/Car/T4/exploration/task', self.task_command, 10)

    def report(self, state, **fields):
        self.status = {'time':time.time(),'state':state,'request_id':self.current,**fields}
        self.node.get_logger().info('UE navigation: '+json.dumps(self.status))

    def reply(self, request, status, points=()):
        self.reply_pub.publish(UInt8MultiArray(data=list(encode_response(request,status,points))))

    def supersede(self, reason):
        if self.current is not None:
            self.reply(self.current,2)
        self.current = None
        self.pending = None
        self.gate = None
        self.cancel()
        self.report(reason)

    def connection(self, msg):
        if not msg.data: self.supersede('TCP_DISCONNECTED')

    def task(self, msg): self.exploration = msg
    def map(self, msg): self.grid = msg
    def task_command(self, msg):
        if msg.command in (msg.START,msg.RESUME): self.supersede('EXPLORATION_TAKEOVER')

    def transform(self, msg):
        for t in msg.transforms:
            if (t.header.frame_id,t.child_frame_id) != ('map','odom'): continue
            p,q = t.transform.translation,t.transform.rotation
            values = (p.x,p.y,p.z,q.x,q.y,q.z,q.w)
            if not all(math.isfinite(v) for v in values): continue
            if self.tf is not None and not np.allclose(values,self.tf,atol=1e-8,rtol=0):
                self.supersede('COORDINATE_FRAME_CHANGED')
            self.tf = values
            self.tf_at = time.monotonic()

    def request(self, msg):
        try: request, point = decode_request(msg.data)
        except ValueError:
            self.supersede('INVALID_REQUEST')
            if len(msg.data)>=4:self.reply(struct.unpack('<I',bytes(msg.data[:4]))[0],2)
            return
        self.supersede('NEW_UE_GOAL')
        self.current = request
        self.pending = (request,point,time.monotonic())
        self.report('WAITING_FOR_NAVIGATION')
        if self.exploration and self.exploration.state in (1,2,3,4,5):
            m=PureExplorationTask();m.task_id=self.exploration.task_id;m.command=m.PAUSE
            m.header.frame_id='map';m.header.stamp=self.node.get_clock().now().to_msg()
            self.task_pub.publish(m)

    def cancel(self):
        if self.handle and not self.canceling:
            self.canceling=True
            self.handle.cancel_goal_async()
        # A late accepted handle is cancelled in accepted().

    def tick(self):
        now=time.monotonic()
        if now-self.last_status>1:
            f=Path(__file__).with_name('unreal-path-status.json');tmp=f.with_suffix('.tmp')
            tmp.write_text(json.dumps(self.status,indent=2));tmp.replace(f);self.last_status=now
        if not self.pending:return
        request,point,started=self.pending
        if now-started>15:
            self.reply(request,2);self.pending=None;self.current=None;self.cancel()
            self.report('GOAL_INPUT_UNAVAILABLE');return
        a=self.adapter
        if (self.handle or self.sending or a.alignment is None or self.tf is None or
            now-self.tf_at>5 or not a.p3 or now-a.p3[-1][0]>5 or
            a.last_received is None or now-a.last_received>.5 or not self.client.server_is_ready()):return
        if self.exploration and self.exploration.state in (1,2,3,4,5):return
        try:
            self.coordinates=Coordinates(a.alignment.theta,a.alignment.offset,self.tf[:3],self.tf[3:])
            target=self.coordinates.to_map(point)
            if not all(math.isfinite(v) for v in target):raise ValueError('Invalid target')
        except ValueError:
            self.reply(request,2);self.pending=None;self.current=None;self.report('INVALID_COORDINATES');return
        sid=uuid.uuid4().bytes
        self.gate=ReferenceGate(sid.hex())
        goal=NavigateToPose.Goal(target_x_m=float(target[0]),target_y_m=float(target[1]),has_target_yaw=False)
        self.target_z=float(target[2])
        self.pending=None;self.sending=True
        self.authorize_pub.publish(Empty())
        future=self.client.send_goal_async(goal,goal_uuid=UUID(uuid=list(sid)))
        future.add_done_callback(lambda f,request=request,sid=sid:self.accepted(f,request,sid))
        self.report('GOAL_SENT',target_map_m=list(target),session=sid.hex(),has_target_yaw=False)

    def accepted(self, future, request, sid):
        self.sending=False
        try:handle=future.result()
        except Exception as e:
            if self.gate and self.gate.session==sid.hex():
                self.reply(request,2);self.report('GOAL_SEND_FAILED',detail=str(e));self.current=None;self.gate=None
            return
        if not handle.accepted:
            if self.gate and self.gate.session==sid.hex():
                self.reply(request,2);self.report('GOAL_REJECTED');self.current=None;self.gate=None
            return
        self.handle=handle;self.canceling=False
        handle.get_result_async().add_done_callback(lambda f,request=request,sid=sid:self.result(f,request,sid))
        if not self.gate or self.gate.session!=sid.hex() or self.current!=request:self.cancel()

    def result(self, future, request, sid):
        self.handle=None;self.canceling=False
        if not self.gate or self.gate.session!=sid.hex() or self.current!=request:return
        try:
            result=future.result().result
            if result.outcome != result.GOAL_REACHED:
                self.reply(request,1 if result.outcome==result.NO_PATH else 2)
            self.report('FINISHED',outcome=result.outcome,reason=result.reason_code)
        except Exception as e:
            self.reply(request,2);self.report('RESULT_ERROR',detail=str(e))
        self.current=None;self.gate=None

    def reference(self, msg):
        if self.current is None or self.gate is None:return
        if not self.gate.accept(bytes(msg.session_id.uuid).hex(),msg.segment_revision,msg.state):return
        if msg.state != msg.ACTIVE or not msg.path.poses:
            self.reply(self.current,2);self.report('PATH_INVALIDATED',revision=msg.segment_revision);return
        if msg.path.header.frame_id!='map':
            self.reply(self.current,2);self.report('INVALID_PATH_FRAME');return
        try:
            poses=[(p.pose.position.x,p.pose.position.y,p.pose.position.z) for p in msg.path.poses]
            if not all(math.isfinite(v) for p in poses for v in p):raise ValueError('Nonfinite path')
            # Wheel reference is planar (z=0). Heights below are for UE rendering only.
            heights=[height_at(self.grid,x,y) if self.grid else None for x,y,z in poses]
            distance=[0.]
            for a,b in zip(poses,poses[1:]):distance.append(distance[-1]+math.hypot(a[0]-b[0],a[1]-b[1]))
            known=[i for i,h in enumerate(heights) if h is not None]
            estimated=sum(h is None for h in heights)
            if known:
                zs=np.interp(distance,[distance[i] for i in known],[heights[i] for i in known])
            else:zs=[self.target_z]*len(poses)
            points=[self.coordinates.to_ue((p[0],p[1],float(z))) for p,z in zip(poses,zs)]
            self.reply(self.current,0,points)
            self.report('PATH_RETURNED',session=self.gate.session,revision=msg.segment_revision,
                        points=len(points),estimated_display_heights=estimated,
                        reaches_final_goal=msg.reaches_final_goal)
        except (ValueError,OverflowError,struct.error) as e:
            self.reply(self.current,2);self.report('PATH_ENCODING_FAILED',detail=str(e))
