#!/usr/bin/env bash
set -e
# Humble uses Ubuntu Python 3.10, even when launched from Conda base.
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PATH="/usr/bin:/bin:$PATH"
source /opt/ros/humble/setup.bash
source /home/yanfa/Env_X/InterFace/install/setup.bash
source /home/yanfa/P4/artifacts/humble/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=0 DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority
: "${ROS_DOMAIN_ID:?Use scripts/navigation.sh or exploration.sh}"
: "${P4_LEGACY_DOMAIN:?Missing selected legacy domain}"
/usr/bin/python3 /home/yanfa/P4/vehicle_interface/gui/ensure_gui.py
/usr/bin/python3 - <<'PY'
from pathlib import Path
import json
import os
import subprocess

root = Path('/home/yanfa/P4/debug/p3_joint')
commands = {
    'command_relay': ['/usr/bin/python3', str(root / 'persistent_command_relay.py')],
    'vehicle_feedback': ['/usr/bin/python3', str(root / 'legacy_feedback_input.py')],
    'exploration': ['ros2', 'launch', 'lunar_pure_exploration_ros', 'exploration_navigation.launch.py', 'config_file:=' + str(root / 'exploration-only.yaml'), 'start_navigation:=false', 'start_exploration:=true', 'start_rviz:=false'],
    'static_tf_input': ['/usr/bin/python3', str(root / 'static_tf_input.py')],
    'navigation': ['ros2', 'launch', 'lunar_pure_exploration_ros', 'exploration_navigation.launch.py', 'config_file:=' + str(root / 'navigation.yaml'), 'start_exploration:=false', 'start_rviz:=false'],
    'controller': ['ros2', 'run', 'lunar_pure_wheeled_controller', 'lunar_pure_wheeled_controller_node.py', '--ros-args', '--params-file', str(root / 'controller.yaml')],
    'goal_bridge': ['ros2', 'launch', 'lunar_incremental_navigation_ros', 'incremental_rviz_goal_bridge.launch.py'],
    'rviz': ['rviz2', '-d', str(root / 'view.rviz'), '--ros-args', '-r', '__node:=p4_joint_rviz'],
}
persistent = ('command_relay', 'vehicle_feedback', 'static_tf_input')
requested_text = os.environ.get('P4_START_KEYS', '')
requested = tuple(key for key in requested_text.split(',') if key) or tuple(commands)
if len(set(requested)) != len(requested) or any(key not in commands for key in requested):
    raise SystemExit('Invalid P4_START_KEYS: ' + requested_text)


def live(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1][0] != 'Z'
    except (FileNotFoundError, IndexError):
        return False


def verified(key, pid):
    try:
        command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        return commands[key][-1] in command or commands[key][0] in command
    except OSError:
        return False

pid_file = root / 'pids.json'
pids = json.loads(pid_file.read_text()) if pid_file.exists() else {}
if not isinstance(pids, dict):
    raise SystemExit('Invalid P4 pid file')
for key, pid in list(pids.items()):
    if key not in commands:
        raise SystemExit('Unknown process in P4 pid file: ' + str(key))
    if not live(pid):
        pids.pop(key)
        continue
    if key in requested:
        raise SystemExit(f'{key} is still running; stop it before partial start')
    if not verified(key, pid):
        raise SystemExit(f'Unverified retained process: {key} PID {pid}')
if requested != tuple(commands):
    missing = [key for key in persistent if key not in pids]
    if missing:
        raise SystemExit('Soft restart requires retained P4 processes: ' + ', '.join(missing))

for key in requested:
    command = commands[key]
    with (root / f'{key}-domain{os.environ["ROS_DOMAIN_ID"]}.log').open('a') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pids[key] = process.pid
pid_file.write_text(json.dumps(pids, indent=2))
print(pids)
PY
