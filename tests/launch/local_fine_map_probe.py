"""Verify the production launch publishes fine display data before a goal exists."""
from pathlib import Path
import os,sys,subprocess,signal,time,json,yaml
import rclpy
from rclpy.qos import QoSProfile,DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
repo=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
config=yaml.safe_load((repo/'config/exploration_navigation.yaml').read_text())
config['common'].update(local_map_topic='/lunar_demo/controller/grid_map',odometry_topic='/lunar_demo/controller/odometry',exploration_map_topic='/fine_probe/coarse',navigation_action='/fine_probe/navigate')
config['navigation'].update(publish_local_fine_map=True,local_fine_map_topic='/fine_probe/fine',local_fine_map_window_m=12.)
path=out/'config.yaml';path.write_text(yaml.safe_dump(config));seen={};procs=[];logs=[]
try:
 for name,cmd in [('vehicle',['ros2','run','lunar_incremental_controller_demo','vehicle_sim','--ros-args','-p','case:=detour']),('navigation',['ros2','launch','lunar_pure_exploration_ros','exploration_navigation.launch.py','config_file:='+str(path),'start_exploration:=false','start_rviz:=false'])]:
  log=(out/(name+'.log')).open('w');logs.append(log);procs.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
 rclpy.init();node=rclpy.create_node('fine_map_probe');q=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
 subs=[]
 for label,topic in [('fine','/fine_probe/fine'),('coarse','/fine_probe/coarse')]:
  subs.append(node.create_subscription(OccupancyGrid,topic,lambda m,key=label:seen.update({key:m}),q))
 end=time.monotonic()+20
 while time.monotonic()<end and len(seen)<2:rclpy.spin_once(node,timeout_sec=.1)
 assert len(seen)==2,seen.keys()
 fine=seen['fine'];coarse=seen['coarse']
 assert abs(fine.info.resolution-.2)<1e-6 and abs(coarse.info.resolution-1.)<1e-6
 assert fine.header.frame_id=='map' and fine.info.width<=62 and fine.info.height<=62
 assert 0 in fine.data and 100 in fine.data
 assert not node.get_publishers_info_by_topic('/Car/T5/Car_Cmd_Vel')
 report=dict(fine_resolution=fine.info.resolution,coarse_resolution=coarse.info.resolution,width=fine.info.width,height=fine.info.height,free_cells=sum(v==0 for v in fine.data),blocked_cells=sum(v==100 for v in fine.data))
 print(json.dumps(report));(out/'result.json').write_text(json.dumps(report,indent=2))
 node.destroy_node();rclpy.shutdown()
finally:
 for p in reversed(procs):
  if p.poll() is None:p.send_signal(signal.SIGINT)
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=5)
 for f in logs:f.close()
