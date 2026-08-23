#!/usr/bin/env bash
# Start the complete wheel planning/control stack from the bundle root.
set -eo pipefail

platform="${1:-wheel}"
environment_mode="${2:-2}"

case "${platform}" in
  wheel) ;;
  *)
    echo "用法：$0 [wheel] [environment_mode]" >&2
    exit 2
    ;;
esac

bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/humble/setup.bash
source "${bundle_root}/ros2_ws/install/setup.bash"

pids=()
cleanup() {
  trap - INT TERM EXIT
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait "${pids[@]}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

ros2 launch lunar_pure_planner_ros pure_planner.launch.py \
  platform_type:="${platform}" &
pids+=("$!")

ros2 launch lunar_pure_planner_ros local_traversability.launch.py \
  platform_type:="${platform}" &
pids+=("$!")

ros2 launch lunar_pure_planner_ros rviz_goal_bridge.launch.py \
  environment_mode:="${environment_mode}" &
pids+=("$!")

ros2 launch lunar_pure_wheeled_controller pure_wheeled_controller.launch.py &
pids+=("$!")

wait -n "${pids[@]}"
