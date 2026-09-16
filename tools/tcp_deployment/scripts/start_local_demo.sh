#!/usr/bin/env bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
export ROS_DISTRO="${ROS_DISTRO:-$LUNAR_ROS_DISTRO}"
exec bash "$bundle_root/scripts/simulation/run_local_rviz_demo.sh" --map "$bundle_root/maps/terrain" --mode nav "$@"
