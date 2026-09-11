#!/usr/bin/env bash
set -eo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
start_rviz="${1:-true}"
time_scale="${2:-30.0}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
source /opt/ros/humble/setup.bash
source "$repository_root/ros2_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${DEMO_ROS_DOMAIN_ID:-73}"
echo "Exploration demo: ROS_DOMAIN_ID=$ROS_DOMAIN_ID; commands=/lunar_demo/integrated/cmd_vel"
exec ros2 launch lunar_integrated_exploration_demo integrated_rviz.launch.py \
  start_rviz:="$start_rviz" start_local_rviz:="$start_rviz" time_scale:="$time_scale" "$@"
