"""Permanent P4 ROS-domain forwarding. The old interface alone owns TCP."""
import time,json,signal,math,fcntl,os
from pathlib import Path
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile,DurabilityPolicy,qos_profile_sensor_data
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Twist,PoseStamped
from action_msgs.srv import CancelGoal
from lunar_pure_exploration_msgs.msg import PureExplorationTask,PureExplorationStatus
from std_msgs.msg import Bool,String,Empty
from lunar_planning_msgs.msg import PathReference,TrackingStatus
from command_gate import allowed,StopPolicy

class Relay:
 def __init__(self):
  self.lock=Path(__file__).with_name('command-relay.lock').open('a')
  fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  self.contexts=[];self.nodes=[];self.executors=[];self.running=True
  for domain in (int(os.environ.get('P4_RELAY_DOMAIN',os.environ.get('ROS_DOMAIN_ID','0'))),int(os.environ['P4_LEGACY_DOMAIN'])):
   c=Context();rclpy.init(context=c,domain_id=domain)
   n=rclpy.create_node('p4_persistent_command_'+str(domain),context=c)
   e=SingleThreadedExecutor(context=c);e.add_node(n)
   self.contexts.append(c);self.nodes.append(n);self.executors.append(e)
  p4,old=self.nodes
  self.cmd_pub=old.create_publisher(Twist,'/car/cmd_vel',10)
  self.mode_pub=old.create_publisher(String,'/car/set_mode',10)
  self.lease_pub=old.create_publisher(Bool,'/car/external_control_lease',10)
  self.status_pub=p4.create_publisher(String,'/P4/control/relay_status',10)
  self.path=None;self.tracking=None;self.state=0;self.command=(0.,0.)
  self.command_at=self.tracking_at=self.odom_at=self.vehicle_at=-math.inf
  self.blocked=None;self.armed=False;self.tick_at=0.;self.status_at=0.
  self.started=time.monotonic();self.path_count=0;self.arm_count=0
  self.policy=StopPolicy();self.task_id=None;self.task_state=0
  previous=Path(__file__).with_name('command-relay-status.json')
  if previous.exists():
   try:
    old_status=json.loads(previous.read_text())
    if old_status.get('reason')=='EXECUTION_UNAVAILABLE:P3_MAP_STALE':pass  # Retired policy; never revive its cancelled goal.
    elif old_status.get('stop_state')=='STOPPED':self.policy.latched=old_status.get('reason') or 'EXTERNAL_PARK_REQUIRES_RESUME'
    elif old_status.get('path') and old_status.get('manual_blocked_session')==old_status['path'][0]:self.policy.external_stop()
   except (ValueError,OSError):self.policy.latched='PREVIOUS_STOP_STATE_UNREADABLE'
  self.p3_at=-math.inf;self.p3_stamp=None;self.p3_frames=None;self.map_transform=None
  self.stop_reference=None;self.abort_at=-math.inf;self.cancel_future=None
  self.cancel_result=None;self.stop_state='IDLE';self.stop_reason='STARTING'
  self.task_pub=p4.create_publisher(PureExplorationTask,'/Car/T4/exploration/task',10)
  self.cancel_client=p4.create_client(CancelGoal,'/Car/T4/navigation/navigate_to_pose/_action/cancel_goal')
  p4.create_subscription(PureExplorationStatus,'/Car/T4/exploration/status',self.on_task_status,10)
  p4.create_subscription(PureExplorationTask,'/Car/T4/exploration/task',self.on_task_command,10)
  p4.create_subscription(PoseStamped,'/Car/T4/rviz_goal',lambda m:self.authorize(),10)
  p4.create_subscription(Empty,'/P4/input/ue_goal_authorize',lambda m:self.authorize(),10)
  p4.create_subscription(Odometry,'/Car/T3/localization/odometry',self.on_p3,qos_profile_sensor_data)
  p4.create_subscription(TFMessage,'/P4/input/map_to_odom',self.on_map_transform,10)
  p4.create_subscription(PathReference,'/Car/T4/planning/path_reference',self.on_path,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
  p4.create_subscription(TrackingStatus,'/Car/T4/control/tracking_status',self.on_tracking,10)
  p4.create_subscription(Twist,'/P4/debug/cmd_vel',self.on_command,10)
  p4.create_subscription(Odometry,'/P4/input/odometry',self.on_odom,10)
  old.create_subscription(Odometry,'/car/odom',self.on_vehicle,10)
  old.create_subscription(String,'/car/set_mode',self.on_mode,10)
 @staticmethod
 def identity(m):return (bytes(m.session_id.uuid).hex(),m.segment_revision)
 def on_path(self,m):
  identity=self.identity(m)
  if m.state!=m.ACTIVE or not m.path.poses:
   self.path=None;return
  if identity!=self.path:
   self.path_count+=1;self.command_at=-math.inf;self.tracking_at=-math.inf
  self.path=identity
 def on_tracking(self,m):
  self.tracking=self.identity(m);self.state=m.state;self.tracking_at=time.monotonic()
 def on_command(self,m):
  # Controller owns speed selection; the relay only validates/forwards.
  if any(abs(v)>1e-9 for v in (m.linear.y,m.linear.z,m.angular.x,m.angular.y)):
   self.command=(float('nan'),0.)
  else:self.command=(m.linear.x,m.angular.z)
  self.command_at=time.monotonic()
 def on_odom(self,m):
  p=m.pose.pose.position;q=m.pose.pose.orientation;t=m.twist.twist
  if all(math.isfinite(v) for v in (p.x,p.y,p.z,q.x,q.y,q.z,q.w,t.linear.x,t.linear.y,t.angular.z)):
   self.odom_at=time.monotonic()
  else:self.odom_at=-math.inf
 def on_vehicle(self,m):self.vehicle_at=time.monotonic()
 def localization_changed(self,reason):
  if self.policy.latched is None:
   self.event('LOCALIZATION_CHANGED',reason=reason)
   self.policy.latched=reason
  self.p3_at=-math.inf
 def on_map_transform(self,m):
  for t in m.transforms:
   if (t.header.frame_id,t.child_frame_id)!=('map','odom'):continue
   p=t.transform.translation;q=t.transform.rotation
   values=(p.x,p.y,p.z,q.x,q.y,q.z,q.w)
   if not all(math.isfinite(v) for v in values):
    self.localization_changed('LOCALIZATION_TF_INVALID');return
   norm=math.sqrt(sum(v*v for v in values[3:]))
   if norm<1e-9:self.localization_changed('LOCALIZATION_TF_INVALID');return
   current=(values[:3],tuple(v/norm for v in values[3:]))
   if self.map_transform is not None:
    old_p,old_q=self.map_transform
    if math.dist(old_p,current[0])>1e-6 or 1-abs(sum(a*b for a,b in zip(old_q,current[1])))>1e-10:
     self.localization_changed('LOCALIZATION_FRAME_CHANGED');return
   self.map_transform=current
 def on_p3(self,m):
  frames=(m.header.frame_id,m.child_frame_id)
  if frames!=('odom','base_link'):
   if self.p3_frames is not None:self.localization_changed('LOCALIZATION_FRAME_CHANGED')
   else:self.p3_at=-math.inf
   return
  p=m.pose.pose.position;q=m.pose.pose.orientation
  if not all(math.isfinite(v) for v in (p.x,p.y,p.z,q.x,q.y,q.z,q.w)) or sum(v*v for v in (q.x,q.y,q.z,q.w))<1e-12:
   self.p3_at=-math.inf;return
  stamp=(m.header.stamp.sec,m.header.stamp.nanosec)
  if self.p3_stamp is not None and stamp<self.p3_stamp:
   if time.monotonic()-self.p3_at>5:self.localization_changed('LOCALIZATION_TIME_RESET')
   return
  if stamp!=self.p3_stamp:
   if time.monotonic()-self.p3_at>5:self.command_at=self.tracking_at=-math.inf
   self.p3_stamp=stamp;self.p3_at=time.monotonic();self.p3_frames=frames
 def on_task_status(self,m):self.task_id=m.task_id;self.task_state=m.state
 def event(self,name,**fields):
  record={'time':time.time(),'event':name,**fields}
  with Path(__file__).with_name('command-relay-events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 def on_task_command(self,m):
  self.event('TASK_RECEIVED',command=m.command,task_id=m.task_id,stop_reason=self.policy.latched,publishers=[p.node_name for p in self.nodes[0].get_publishers_info_by_topic('/Car/T4/exploration/task')])
  if m.command in (m.START,m.RESUME):self.authorize()
 def authorize(self):
  self.event('AUTHORIZE',previous_reason=self.policy.latched)
  self.policy.authorize();self.blocked=None;self.stop_reference=None
  self.command_at=self.tracking_at=-math.inf
 def on_mode(self,m):
  # No park is published by this process during normal operation. All received
  # parks are external; the legacy GUI does not encode manual vs watchdog cause.
  if m.data!='park' or not self.running:return
  self.event('EXTERNAL_PARK',publishers=[p.node_name for p in self.nodes[1].get_publishers_info_by_topic('/car/set_mode')])
  self.policy.external_stop();self.blocked=self.path[0] if self.path else None
  self.armed=False
 def unavailable_reason(self,now):
  if now-self.p3_at>5:return 'P3_ODOMETRY_STALE'
  if now-self.odom_at>.5:return 'VEHICLE_FEEDBACK_STALE'
  if now-self.vehicle_at>1:return 'LEGACY_ODOMETRY_STALE'
  if now-self.tracking_at>.5:return 'TRACKING_FEEDBACK_STALE'
  if now-self.command_at>.5:return 'CONTROL_COMMAND_STALE'
  if self.path!=self.tracking:return 'REFERENCE_MISMATCH'
  return 'CONTROL_NOT_EXECUTABLE'
 def abort_execution(self,now):
  if self.stop_reference is None and self.path:self.stop_reference=self.path
  if self.cancel_future is not None:
   if not self.cancel_future.done():return
   try:self.cancel_result=self.cancel_future.result().return_code
   except Exception:self.cancel_result='SERVICE_ERROR'
   self.cancel_future=None
  if now-self.abort_at<1:return
  self.abort_at=now
  if self.task_id and self.task_state in (1,2,3,4,5):
   m=PureExplorationTask();m.header.frame_id='map';m.header.stamp=self.nodes[0].get_clock().now().to_msg()
   m.task_id=self.task_id;m.command=m.PAUSE
   self.event('PAUSE_SENT',task_id=self.task_id,reason=self.stop_reason)
   self.task_pub.publish(m)
  if self.stop_reference and self.cancel_client.service_is_ready():
   req=CancelGoal.Request();req.goal_info.goal_id.uuid=list(bytes.fromhex(self.stop_reference[0]))
   self.cancel_future=self.cancel_client.call_async(req)
 def tick(self):
  now=time.monotonic()
  self.lease_pub.publish(Bool(data=True))
  eligible=allowed(now=now,path=self.path,tracking=self.tracking,state=self.state,command=self.command,
   command_at=self.command_at,tracking_at=self.tracking_at,odom_at=self.odom_at,vehicle_at=self.vehicle_at,blocked=self.blocked)
  eligible=eligible and now-self.started>1.5 and self.lease_pub.get_subscription_count()>0
  eligible=eligible and now-self.p3_at<=5
  self.stop_state,self.stop_reason=self.policy.evaluate(now,self.path is not None,eligible,self.unavailable_reason(now))
  eligible=self.stop_state=='FORWARDING'
  if self.stop_state=='STOPPED':self.abort_execution(now)
  if eligible:
   msg=Twist();msg.linear.x=float(self.command[0]);msg.angular.z=float(self.command[1]);self.cmd_pub.publish(msg)
   if not self.armed:
    self.mode_pub.publish(String(data='auto'));self.armed=True;self.arm_count+=1
    self.nodes[0].get_logger().info('Forwarding current P4 reference '+str(self.path))
  else:
   self.cmd_pub.publish(Twist())
   self.armed=False
  if now-self.status_at>=1:
   status={'state':'FORWARDING' if self.armed else 'PARKED','stop_state':self.stop_state,'reason':self.stop_reason,'exploration_state':self.task_state,'cancel_result':self.cancel_result,'path':self.path,'tracking':self.tracking,'tracking_state':self.state,'manual_blocked_session':self.blocked,'path_count':self.path_count,'arm_count':self.arm_count,'localization_wait':self.policy.localization_wait,'p3_age_s':now-self.p3_at if math.isfinite(self.p3_at) else None,'odom_age_s':now-self.odom_at if math.isfinite(self.odom_at) else None}
   self.status_pub.publish(String(data=json.dumps(status)))
   target=Path(__file__).with_name('command-relay-status.json');tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2));tmp.replace(target)
   self.status_at=now
 def run(self):
  try:
   while self.running:
    for e in self.executors:e.spin_once(timeout_sec=.005)
    if time.monotonic()-self.tick_at>=.05:self.tick();self.tick_at=time.monotonic()
  finally:
   self.armed=False
   for _ in range(10):
    self.lease_pub.publish(Bool(data=True));self.cmd_pub.publish(Twist());self.mode_pub.publish(String(data='park'))
    for e in self.executors:e.spin_once(timeout_sec=.02)
   self.lease_pub.publish(Bool(data=False))
   for e in self.executors:e.spin_once(timeout_sec=.1)
   for e,n,c in zip(self.executors,self.nodes,self.contexts):e.shutdown();n.destroy_node();c.shutdown()
if __name__=='__main__':
 r=Relay();signal.signal(signal.SIGINT,lambda *_:setattr(r,'running',False));signal.signal(signal.SIGTERM,lambda *_:setattr(r,'running',False));r.run()
