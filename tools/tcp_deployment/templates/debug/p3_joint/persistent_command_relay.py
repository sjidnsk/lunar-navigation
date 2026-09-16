"""Permanent P4 ROS-domain forwarding. The old interface alone owns TCP."""
import time,json,signal,math,fcntl
from pathlib import Path
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile,DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool,String
from lunar_planning_msgs.msg import PathReference,TrackingStatus
from command_gate import allowed

class Relay:
 def __init__(self):
  self.lock=Path(__file__).with_name('command-relay.lock').open('a')
  fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  self.contexts=[];self.nodes=[];self.executors=[];self.running=True
  for domain in (57,10):
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
  self.pending_parks=0
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
 def park(self):
  self.armed=False;self.pending_parks+=1;self.mode_pub.publish(String(data='park'))
 def on_mode(self,m):
  if m.data!='park':return
  if self.pending_parks:
   self.pending_parks-=1;return
  if self.path:self.blocked=self.path[0]
  self.armed=False
 def tick(self):
  now=time.monotonic()
  self.lease_pub.publish(Bool(data=True))
  eligible=allowed(now=now,path=self.path,tracking=self.tracking,state=self.state,command=self.command,
   command_at=self.command_at,tracking_at=self.tracking_at,odom_at=self.odom_at,vehicle_at=self.vehicle_at,blocked=self.blocked)
  eligible=eligible and self.pending_parks==0 and now-self.started>1.5 and self.lease_pub.get_subscription_count()>0
  if eligible:
   msg=Twist();msg.linear.x=float(self.command[0]);msg.angular.z=float(self.command[1]);self.cmd_pub.publish(msg)
   if not self.armed:
    self.mode_pub.publish(String(data='auto'));self.armed=True;self.arm_count+=1
    self.nodes[0].get_logger().info('Forwarding current P4 reference '+str(self.path))
  else:
   self.cmd_pub.publish(Twist())
   if self.armed or now-self.status_at>=1:
    self.park()
  if now-self.status_at>=1:
   status={'state':'FORWARDING' if self.armed else 'PARKED','path':self.path,'tracking':self.tracking,'tracking_state':self.state,'manual_blocked_session':self.blocked,'path_count':self.path_count,'arm_count':self.arm_count,'odom_age_s':now-self.odom_at if math.isfinite(self.odom_at) else None}
   self.status_pub.publish(String(data=json.dumps(status)))
   Path(__file__).with_name('command-relay-status.json').write_text(json.dumps(status,indent=2))
   self.status_at=now
 def run(self):
  try:
   while self.running:
    for e in self.executors:e.spin_once(timeout_sec=.005)
    if time.monotonic()-self.tick_at>=.05:self.tick();self.tick_at=time.monotonic()
  finally:
   self.armed=False
   for _ in range(10):
    self.lease_pub.publish(Bool(data=True));self.cmd_pub.publish(Twist());self.park()
    for e in self.executors:e.spin_once(timeout_sec=.02)
   self.lease_pub.publish(Bool(data=False))
   for e in self.executors:e.spin_once(timeout_sec=.1)
   for e,n,c in zip(self.executors,self.nodes,self.contexts):e.shutdown();n.destroy_node();c.shutdown()
if __name__=='__main__':
 r=Relay();signal.signal(signal.SIGINT,lambda *_:setattr(r,'running',False));signal.signal(signal.SIGTERM,lambda *_:setattr(r,'running',False));r.run()
