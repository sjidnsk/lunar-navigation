#!/usr/bin/env bash
set -e
ros_distro="${LUNAR_ROS_DISTRO:-${ROS_DISTRO:-}}"
if [[ -n "${LUNAR_ROS_DISTRO:-}" && -n "${ROS_DISTRO:-}" && "$LUNAR_ROS_DISTRO" != "$ROS_DISTRO" ]]; then
  echo "refusing mixed ROS distributions: ROS_DISTRO=${ROS_DISTRO}, LUNAR_ROS_DISTRO=${LUNAR_ROS_DISTRO}" >&2
  exit 2
fi
if [[ -z "$ros_distro" || ! -r "/opt/ros/${ros_distro}/setup.bash" ]]; then echo "set ROS_DISTRO or LUNAR_ROS_DISTRO to an installed ROS distribution" >&2; exit 2; fi
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH ROS_PACKAGE_PATH
unset PYTHONPATH LD_LIBRARY_PATH
source "/opt/ros/${ros_distro}/setup.bash"
set -u
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
artifact_root="${LUNAR_OBJ_TCP_SIM_BUILD_BASE:-${XDG_CACHE_HOME:-${HOME}/.cache}/lunar_obj_tcp_sim/${ros_distro}}"
mkdir -p "$artifact_root"
cd "${repo_root}/ros2_ws"
exec colcon --log-base "${artifact_root}/log" build --packages-up-to lunar_obj_tcp_sim --build-base "${artifact_root}/build" --install-base "${artifact_root}/install" --cmake-args "-DCMAKE_BUILD_TYPE=${LUNAR_CMAKE_BUILD_TYPE:-Release}" "$@"
