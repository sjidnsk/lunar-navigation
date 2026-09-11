#!/usr/bin/env bash
set -eo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_scope="${1:-runtime}"
case "$build_scope" in
  runtime) packages=(lunar_incremental_navigation_ros lunar_pure_exploration_ros lunar_pure_wheeled_controller) ;;
  demos) packages=(lunar_integrated_exploration_demo) ;;
  *) echo "用法：bash scripts/orin/build.sh [runtime|demos]" >&2; exit 2 ;;
esac
source /opt/ros/humble/setup.bash
set -u
cd "$repository_root/ros2_ws"
colcon build --merge-install --packages-up-to "${packages[@]}" \
  --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
