"""P4 deployment adapter: reuse bridge coordinates, apply a planar P3 odom anchor."""
import math
from lunar_obj_tcp_sim.coordinates import world_position, orientation, PoseRates

def yaw(q):
 x,y,z,w=q
 return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

def converted_pose(position,quaternion):
 if not all(math.isfinite(v) for v in (*position,*quaternion)) or sum(v*v for v in quaternion)<1e-12:
  raise ValueError('Invalid raw telemetry pose')
 return world_position(position),orientation(quaternion)

class Alignment:
 def __init__(self,source_p,source_q,target_p,target_q):
  if not all(math.isfinite(v) for v in (*source_p,*source_q,*target_p,*target_q)):
   raise ValueError('Nonfinite alignment')
  self.theta=math.atan2(math.sin(yaw(target_q)-yaw(source_q)),math.cos(yaw(target_q)-yaw(source_q)))
  self.c=math.cos(self.theta);self.s=math.sin(self.theta)
  self.offset=(target_p[0]-self.c*source_p[0]+self.s*source_p[1],target_p[1]-self.s*source_p[0]-self.c*source_p[1],target_p[2]-source_p[2])
 def apply(self,p,q):
  x,y,z=p;qx,qy,qz,qw=q;s=math.sin(self.theta/2);c=math.cos(self.theta/2)
  return (self.c*x-self.s*y+self.offset[0],self.s*x+self.c*y+self.offset[1],z+self.offset[2]),(c*qx-s*qy,c*qy+s*qx,c*qz+s*qw,c*qw-s*qz)
