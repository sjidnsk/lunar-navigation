#!/usr/bin/env bash
set -eo pipefail

platform_type="${1:-wheel}"
use_sim_time="${2:-false}"
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /opt/ros/humble/setup.bash
source "$repository_root/ros2_ws/install/setup.bash"
set -u
exec ros2 launch lunar_pure_exploration_ros exploration_navigation.launch.py \
  stack_mode:=incremental_v2 \
  platform_type:="$platform_type" \
  use_sim_time:="$use_sim_time" \
  start_navigation:=true \
  start_exploration:=false
