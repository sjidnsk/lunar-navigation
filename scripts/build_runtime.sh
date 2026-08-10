#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/humble/setup.bash
set -u

: "${LUNAR_VOLUME1_OUTPUT:?set an absolute output path outside the repository}"

task_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$LUNAR_VOLUME1_OUTPUT" != /* ]]; then
  echo "LUNAR_VOLUME1_OUTPUT must be an absolute path" >&2
  exit 2
fi
task_output_root="$(realpath -m "$LUNAR_VOLUME1_OUTPUT")"
if [[ "$task_output_root" == "$task_repo_root" ||
      "$task_output_root" == "$task_repo_root/"* ]]; then
  echo "LUNAR_VOLUME1_OUTPUT must be outside the repository" >&2
  exit 2
fi

cd "$task_repo_root"
colcon --log-base "$task_output_root/runtime/log" build \
  --merge-install \
  --executor parallel \
  --parallel-workers 2 \
  --base-paths ros2_ws/src model_contract \
  --packages-skip lunar_nav2_adapter \
  --build-base "$task_output_root/runtime/build" \
  --install-base "$task_output_root/runtime/install" \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
