"""Pure eligibility check for the P4 legacy-interface forwarding boundary."""
import math
def allowed(*,now,path,tracking,state,command,command_at,tracking_at,odom_at,vehicle_at,blocked):
 return bool(path is not None and tracking==path and state in (1,2,3,4)
  and path[0]!=blocked
  and all(0<=now-t<=.5 for t in (command_at,tracking_at,odom_at))
  and 0<=now-vehicle_at<=1.
  and all(math.isfinite(v) for v in command)
  and abs(command[0])<=.20001 and abs(command[1])<=.03001)
