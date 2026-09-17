#!/usr/bin/env bash
# Navigation only. Start P3 and start_static_tf_input.sh separately.
set -eo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-59}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source "$repository_root/ros2_ws/install/setup.bash"
set -u
runtime_dir="${P4_RUNTIME_DIR:-$repository_root/deployment-evidence/hardware-navigation}"
mkdir -p "$runtime_dir/log" "$runtime_dir/tmp"
export ROS_LOG_DIR="$runtime_dir/log" TMPDIR="$runtime_dir/tmp" PYTHONDONTWRITEBYTECODE=1
if [[ -z "${CYCLONEDDS_URI:-}" ]]; then
  export CYCLONEDDS_URI="$(sed "s@/tmp/cdds_@$runtime_dir/cdds_@g" /home/yanfa/program/cyclonedds.xml)"
fi
exec ros2 launch lunar_pure_exploration_ros exploration_navigation.launch.py \
  config_file:="$repository_root/config/orin_hardware.yaml" \
  stack_mode:=incremental_v2 start_navigation:=true start_exploration:=false start_rviz:=false
