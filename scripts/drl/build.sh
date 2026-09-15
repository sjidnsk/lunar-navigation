#!/usr/bin/env bash
set -e
DRL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec env -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH -u CMAKE_PREFIX_PATH \
  -u ROS_PACKAGE_PATH -u PYTHONPATH -u LD_LIBRARY_PATH -u ROS_DISTRO \
  -u ROS_VERSION -u ROS_PYTHON_VERSION \
  bash --noprofile --norc -c '
    set -e
    DRL_ROOT=$1; shift
    DRL_CACHE=${DRL_CACHE:-$HOME/.cache/lunar-drl-redesign/jazzy}
    source "${DRL_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
    set -u
    if [[ "$ROS_DISTRO" != jazzy ]]; then echo "Use a separate Jazzy build cache" >&2; exit 2; fi
    cd "$DRL_ROOT"
    export CMAKE_BUILD_PARALLEL_LEVEL=${DRL_BUILD_JOBS:-2}
    exec colcon --log-base "$DRL_CACHE/log" build --base-paths ros2_ws/src \
      --packages-up-to lunar_drl_exploration \
      --executor sequential --build-base "$DRL_CACHE/build" \
      --install-base "$DRL_CACHE/install" "$@"
  ' drl-build "$DRL_ROOT" "$@"
