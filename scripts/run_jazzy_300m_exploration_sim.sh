#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JAZZY_SETUP="/opt/ros/jazzy/setup.bash"
DEFAULT_OVERLAY="/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/closed_loop/install/setup.bash"
LUNAR_JAZZY_OVERLAY="${LUNAR_JAZZY_OVERLAY:-$DEFAULT_OVERLAY}"

if [[ ! -r "$JAZZY_SETUP" ]]; then
  echo "ROS 2 Jazzy setup not found: $JAZZY_SETUP" >&2
  exit 2
fi
if [[ ! -r "$LUNAR_JAZZY_OVERLAY" ]]; then
  echo "Jazzy exploration overlay not found: $LUNAR_JAZZY_OVERLAY" >&2
  exit 2
fi

# Do not let a previously sourced ROS/workspace overlay select stale packages.
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
unset PYTHONPATH LD_LIBRARY_PATH

set +u
source "$JAZZY_SETUP"
source "$LUNAR_JAZZY_OVERLAY"
set -u
export LUNAR_JAZZY_OVERLAY

exec python3 "$SCRIPT_DIR/jazzy_300m_operator.py" "$@"
