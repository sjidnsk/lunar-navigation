#!/usr/bin/env bash
set -eo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /opt/ros/humble/setup.bash
source "$repository_root/ros2_ws/install/setup.bash"
set -u
exec ros2 launch lunar_incremental_navigation_ros \
  incremental_rviz_goal_bridge.launch.py
