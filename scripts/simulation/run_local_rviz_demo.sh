#!/usr/bin/env bash
set -e
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
export PYTHONPATH="${repo_root}/ros2_ws/src/lunar_obj_tcp_sim${PYTHONPATH:+:${PYTHONPATH}}"
export LUNAR_ROS_DISTRO="${LUNAR_ROS_DISTRO:-${ROS_DISTRO:-jazzy}}"
exec python3 -m lunar_obj_tcp_sim.local_demo \
  --runner "${script_dir}/run_obj_tcp_sim.sh" \
  --platform-config "${repo_root}/config/wheel.yaml" \
  --map /home/kai/WS/lunar-navigation/simulation-maps/open-scene-1km "$@"
