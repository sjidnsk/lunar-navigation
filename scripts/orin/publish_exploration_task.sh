#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 5 ]]; then
  echo "用法：$0 <task-id> <min-x-m> <min-y-m> <max-x-m> <max-y-m>" >&2
  exit 2
fi

task_id="$1"
min_x="$2"
min_y="$3"
max_x="$4"
max_y="$5"
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /opt/ros/humble/setup.bash
source "$repository_root/ros2_ws/install/setup.bash"
set -u

task="{header: {frame_id: map}, task_id: $task_id, command: 1, boundary: {points: [\
{x: $min_x, y: $min_y, z: 0.0}, \
{x: $max_x, y: $min_y, z: 0.0}, \
{x: $max_x, y: $max_y, z: 0.0}, \
{x: $min_x, y: $max_y, z: 0.0}]}}"
exec ros2 topic pub --once /Car/T4/exploration/task \
  lunar_pure_exploration_msgs/msg/PureExplorationTask "$task"
