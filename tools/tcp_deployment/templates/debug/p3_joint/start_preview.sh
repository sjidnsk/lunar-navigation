#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
source /home/yanfa/Env_X/InterFace/install/setup.bash
source /home/yanfa/P4/artifacts/humble/install/setup.bash
export ROS_DOMAIN_ID=57 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=0 DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority
python3 - <<'PY'
from pathlib import Path
import subprocess,json,os
r=Path('/home/yanfa/P4/debug/p3_joint')
if (r/'pids.json').exists():
 for pid in json.loads((r/'pids.json').read_text()).values():
  stat=Path('/proc',str(pid),'stat')
  if stat.exists() and stat.read_text().split(') ')[1][0]!='Z':raise SystemExit('Previous P4 preview still running; stop it first')
commands={
'command_relay':['python3',str(r/'persistent_command_relay.py')],
'vehicle_feedback':['python3',str(r/'legacy_feedback_input.py')],
'exploration':['ros2','launch','lunar_pure_exploration_ros','exploration_navigation.launch.py','config_file:='+str(r/'exploration-only.yaml'),'start_navigation:=false','start_exploration:=true','start_rviz:=false'],
'static_tf_input':['python3',str(r/'static_tf_input.py')],
'navigation':['ros2','launch','lunar_pure_exploration_ros','exploration_navigation.launch.py','config_file:='+str(r/'navigation.yaml'),'start_exploration:=false','start_rviz:=false'],
'controller':['ros2','run','lunar_pure_wheeled_controller','lunar_pure_wheeled_controller_node.py','--ros-args','--params-file',str(r/'controller.yaml')],
'goal_bridge':['ros2','launch','lunar_incremental_navigation_ros','incremental_rviz_goal_bridge.launch.py'],
'rviz':['rviz2','-d',str(r/'view.rviz'),'--ros-args','-r','__node:=p4_joint_rviz']}
pids={}
for key,cmd in commands.items():
 with (r/(key+'-domain57.log')).open('a') as f:p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 pids[key]=p.pid
(r/'pids.json').write_text(json.dumps(pids,indent=2));print(pids)
PY
