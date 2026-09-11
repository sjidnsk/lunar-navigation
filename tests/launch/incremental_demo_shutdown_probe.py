"""Run with a sourced demo overlay and isolated ROS_DOMAIN_ID.

Usage: python3 tests/launch/incremental_demo_shutdown_probe.py /tmp/shutdown-evidence
Exercises SIGINT while automatic navigation is active, not after goal completion.
"""
import os,sys,time,signal,subprocess,json
from pathlib import Path
import rclpy
from rclpy.qos import QoSProfile,DurabilityPolicy
from lunar_pure_exploration_msgs.msg import PureExplorationStatus
from std_msgs.msg import String
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
with (out/'auto.log').open('w') as log:
 p=subprocess.Popen(['ros2','launch','lunar_integrated_exploration_demo','integrated_rviz.launch.py','start_rviz:=false','start_local_rviz:=false'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 rclpy.init();n=rclpy.create_node('auto_smoke');seen={}
 sub=n.create_subscription(PureExplorationStatus,'/lunar_demo/integrated/exploration/status',lambda m:seen.update(task_id=m.task_id,state=m.state,reason=m.reason_code,candidates=m.candidate_count,frontiers=m.frontier_cluster_count),QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
 try:
  end=time.monotonic()+75
  while time.monotonic()<end:
   rclpy.spin_once(n,timeout_sec=.1)
   if seen.get('task_id') and seen.get('candidates',0)>0 and seen.get('state')==4:break
  assert seen.get('task_id') and seen.get('candidates',0)>0 and seen.get('state')==4,seen
  print(json.dumps(seen));(out/'auto.json').write_text(json.dumps(seen,indent=2))
 finally:
  p.send_signal(signal.SIGINT)
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=5)
  n.destroy_node();rclpy.shutdown()

assert 'process has died' not in (out/'auto.log').read_text(), (out/'auto.log').read_text()
print('active navigation shutdown PASS')
