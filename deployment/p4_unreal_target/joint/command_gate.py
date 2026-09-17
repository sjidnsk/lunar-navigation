"""Pure eligibility check for the P4 legacy-interface forwarding boundary."""
import math
def allowed(*,now,path,tracking,state,command,command_at,tracking_at,odom_at,vehicle_at,blocked):
 return bool(path is not None and tracking==path and state in (1,2,3,4)
  and path[0]!=blocked
  and all(0<=now-t<=.5 for t in (command_at,tracking_at,odom_at))
  and 0<=now-vehicle_at<=1.
  and all(math.isfinite(v) for v in command)
  and abs(command[0])<=.20001 and abs(command[1])<=.10001)

class StopPolicy:
 """Localization waits recover without cancellation; external stops remain latched."""
 def __init__(self,timeout=10.):
  self.timeout=timeout;self.since=None;self.latched=None;self.localization_wait=False
 def external_stop(self):self.latched='EXTERNAL_PARK_REQUIRES_RESUME'
 def authorize(self):self.latched=None;self.since=None;self.localization_wait=False
 def evaluate(self,now,active,eligible,reason):
  if self.latched:return 'STOPPED',self.latched
  if not active:
   self.since=None;self.localization_wait=False;return 'IDLE','NO_ACTIVE_REFERENCE'
  if eligible:
   self.since=None;self.localization_wait=False;return 'FORWARDING','READY'
  if reason=='P3_ODOMETRY_STALE':self.localization_wait=True
  if self.localization_wait:
   self.since=None;return 'WAITING',reason
  if self.since is None:self.since=now
  if now-self.since>=self.timeout:
   self.latched='EXECUTION_UNAVAILABLE:'+reason
   return 'STOPPED',self.latched
  return 'WAITING',reason
