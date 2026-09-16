"""Read old vehicle ROS telemetry in domain 10; publish P4-only odometry in 57.
No TCP client, velocity publisher, or shared TF publisher is created.
Stationary startup anchor is local commissioning alignment, not landmark calibration.
"""
import json,math,time,signal
from pathlib import Path
from collections import deque
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile,ReliabilityPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from lunar_car_ctrl.msg import VehicleTelemetry
from feedback_geometry import Alignment,converted_pose,PoseRates,yaw

class Adapter:
 def __init__(self):
  self.contexts=[];self.nodes=[];self.executors=[];self.running=True
  for domain in (10,57):
   c=Context();rclpy.init(context=c,domain_id=domain)
   n=rclpy.create_node('p4_legacy_feedback_'+str(domain),context=c)
   e=SingleThreadedExecutor(context=c);e.add_node(n)
   self.contexts.append(c);self.nodes.append(n);self.executors.append(e)
  old,p4=self.nodes
  self.pub=p4.create_publisher(Odometry,'/P4/input/odometry',10)
  self.status_pub=p4.create_publisher(String,'/P4/input/feedback_status',10)
  self.ue=deque();self.p3=deque();self.alignment=None;self.rates=PoseRates()
  self.last_stamp=-1;self.last_received=None;self.count=0;self.last_status=0;self.reason='WAITING_FOR_STATIONARY_ALIGNMENT'
  old.create_subscription(VehicleTelemetry,'/car/telemetry',self.telemetry,QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))
  p4.create_subscription(Odometry,'/Car/T3/localization/odometry',self.p3_pose,QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))
 def p3_pose(self,m):
  if m.header.frame_id!='odom' or m.child_frame_id!='base_link':return
  p=m.pose.pose.position;q=m.pose.pose.orientation
  pp=(p.x,p.y,p.z);qq=(q.x,q.y,q.z,q.w);stamp=m.header.stamp.sec*1000000000+m.header.stamp.nanosec
  if not all(math.isfinite(v) for v in (*pp,*qq)) or sum(v*v for v in qq)<1e-12:return
  if self.p3 and stamp<=self.p3[-1][3]:return
  self.p3.append((time.monotonic(),pp,qq,stamp))
  while self.p3 and self.p3[0][0]<time.monotonic()-8:self.p3.popleft()
 @staticmethod
 def stable(history,distance,angle):
  p,q=history[-1][1:3]
  return all(math.dist(a,p)<=distance and abs(math.atan2(math.sin(yaw(b)-yaw(q)),math.cos(yaw(b)-yaw(q))))<=angle for _,a,b,*rest in history)
 def telemetry(self,m):
  now=time.monotonic();stamp=m.header.stamp.sec*1000000000+m.header.stamp.nanosec
  if stamp<=self.last_stamp:
   self.reason='NONMONOTONIC_TELEMETRY';return
  self.last_stamp=stamp
  try:p,q=converted_pose((m.x_lh,m.y_lh,m.z_lh),(m.qx_lh,m.qy_lh,m.qz_lh,m.qw_lh))
  except ValueError:
   self.rates=PoseRates();self.reason='INVALID_TELEMETRY';return
  self.last_received=now;self.ue.append((now,p,q))
  while self.ue and self.ue[0][0]<now-6:self.ue.popleft()
  rates=self.rates.update(p,q,now)
  if self.alignment is None:
   if not (self.ue and now-self.ue[0][0]>=5.5 and len(self.p3)>=3 and self.p3[-1][0]-self.p3[0][0]>=2 and now-self.p3[-1][0]<3):return
   if not self.stable(self.ue,.01,.01) or not self.stable(self.p3,.03,.02):
    self.reason='WAITING_FOR_STANDSTILL';return
   self.alignment=Alignment(p,q,self.p3[-1][1],self.p3[-1][2])
   record={'type':'STATIONARY_PLANAR_ANCHOR_NOT_LANDMARK_CALIBRATED','theta':self.alignment.theta,'offset':self.alignment.offset,'source_position':p,'source_orientation':q,'target_position':self.p3[-1][1],'target_orientation':self.p3[-1][2],'time':time.time()}
   Path(__file__).with_name('feedback-alignment.json').write_text(json.dumps(record,indent=2))
   self.nodes[1].get_logger().info('Stationary P3 odom alignment ready: '+json.dumps(record))
  pp,qq=self.alignment.apply(p,q)
  out=Odometry();out.header.stamp=m.header.stamp;out.header.frame_id='odom';out.child_frame_id='base_link'
  out.pose.pose.position.x,out.pose.pose.position.y,out.pose.pose.position.z=map(float,pp)
  out.pose.pose.orientation.x,out.pose.pose.orientation.y,out.pose.pose.orientation.z,out.pose.pose.orientation.w=map(float,qq)
  # A derivative gap must never be reported as measured zero speed.
  v,w=rates if rates is not None else ((float('nan'),)*3,(float('nan'),)*3)
  out.twist.twist.linear.x,out.twist.twist.linear.y,out.twist.twist.linear.z=map(float,v)
  out.twist.twist.angular.x,out.twist.twist.angular.y,out.twist.twist.angular.z=map(float,w)
  if rates is None:
   for i in range(6):out.twist.covariance[i*7]=1e6
  self.pub.publish(out);self.count+=1;self.reason='READY' if rates is not None else 'DERIVATIVE_UNAVAILABLE'
 def run(self):
  try:
   while self.running:
    for e in self.executors:e.spin_once(timeout_sec=.005)
    now=time.monotonic()
    if now-self.last_status>1:
     age=None if self.last_received is None else now-self.last_received
     reason='TELEMETRY_STALE' if age is not None and age>.5 else self.reason
     self.status_pub.publish(String(data=json.dumps({'reason':reason,'published':self.count,'telemetry_age_s':age,'aligned':self.alignment is not None})))
     self.last_status=now
  finally:
   for e,n,c in zip(self.executors,self.nodes,self.contexts):e.shutdown();n.destroy_node();c.shutdown()
if __name__=='__main__':
 a=Adapter()
 signal.signal(signal.SIGTERM,lambda *_:setattr(a,'running',False));signal.signal(signal.SIGINT,lambda *_:setattr(a,'running',False))
 a.run()
