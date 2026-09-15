#!/usr/bin/env bash
# Run in a clean child shell so inherited worktree overlays cannot select old code.
set -e
DRL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec env -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH -u CMAKE_PREFIX_PATH \
  -u ROS_PACKAGE_PATH -u PYTHONPATH -u LD_LIBRARY_PATH -u ROS_DISTRO \
  -u ROS_VERSION -u ROS_PYTHON_VERSION \
  bash --noprofile --norc -c '
    set -e
    DRL_ROOT=$1; shift
    DRL_ROS_SETUP=${DRL_ROS_SETUP:-/opt/ros/jazzy/setup.bash}
    DRL_CACHE=${DRL_CACHE:-$HOME/.cache/lunar-drl-redesign/jazzy}
    DRL_PYTHON=${DRL_PYTHON:-$HOME/.cache/lunar-drl-training/venv/bin/python}
    source "$DRL_ROS_SETUP"
    source "$DRL_CACHE/install/local_setup.bash"
    set -u
    if [[ "$ROS_DISTRO" != jazzy ]]; then
      echo "DRL local launcher requires a separate Jazzy cache" >&2; exit 2
    fi
    export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
    cd "$DRL_ROOT"
    exec "$DRL_PYTHON" -m lunar_drl_exploration.cli "$@"
  ' drl "$DRL_ROOT" "$@"
