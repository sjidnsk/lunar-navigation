#!/usr/bin/env bash
set -eo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/humble/setup.bash
set -u
export PYTHONDONTWRITEBYTECODE=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-59}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
runtime_dir="${P4_RUNTIME_DIR:-$repository_root/deployment-evidence/static-tf-runtime}"
mkdir -p "$runtime_dir/log" "$runtime_dir/tmp"
export ROS_LOG_DIR="$runtime_dir/log" TMPDIR="$runtime_dir/tmp"
if [[ -z "${CYCLONEDDS_URI:-}" ]]; then
  export CYCLONEDDS_URI="$(sed "s@/tmp/cdds_@$runtime_dir/cdds_@g" /home/yanfa/program/cyclonedds.xml)"
fi
exec flock -n "$runtime_dir/adapter-domain-$ROS_DOMAIN_ID.lock" \
  /usr/bin/python3 "$repository_root/scripts/orin/static_tf_input.py" "$@"
