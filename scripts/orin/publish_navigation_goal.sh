#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "用法：$0 <x-m> <y-m> [yaw-rad]" >&2
  exit 2
fi

target_x="$1"
target_y="$2"
target_yaw="${3:-}"
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /opt/ros/humble/setup.bash
source "$repository_root/ros2_ws/install/setup.bash"
set -u

if [[ -n "$target_yaw" ]]; then
  yaw_fields="has_target_yaw: true, target_yaw_rad: $target_yaw"
else
  yaw_fields="has_target_yaw: false, target_yaw_rad: 0.0"
fi

goal="{target_x_m: $target_x, target_y_m: $target_y, $yaw_fields}"
exec ros2 action send_goal --feedback /Car/T4/navigation/navigate_to_pose \
  lunar_planning_msgs/action/NavigateToPose "$goal"
