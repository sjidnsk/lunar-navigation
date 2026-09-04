#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /opt/ros/humble/setup.bash
cd "$repository_root/ros2_ws"
colcon build --merge-install --packages-up-to \
  lunar_incremental_navigation_ros lunar_pure_exploration_ros \
  --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
