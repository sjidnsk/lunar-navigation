#!/usr/bin/env bash
set -eo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
scene="${1:-manual}"
start_rviz="${2:-true}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
source /opt/ros/humble/setup.bash
source "$repository_root/ros2_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${DEMO_ROS_DOMAIN_ID:-72}"
echo "Controller demo: ROS_DOMAIN_ID=$ROS_DOMAIN_ID; commands=/lunar_demo/controller/cmd_vel"
exec ros2 launch lunar_incremental_controller_demo controller_rviz.launch.py \
  case:="$scene" start_rviz:="$start_rviz" "$@"
