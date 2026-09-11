#!/usr/bin/env bash
set -eo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/humble/setup.bash
source "$repository_root/ros2_ws/install/setup.bash"
set -u
exec ros2 launch lunar_incremental_navigation_ros navigation_rviz.launch.py \
  use_sim_time:="${1:-false}" \
  config_file:="${2:-$repository_root/config/exploration_navigation.yaml}"
