#!/usr/bin/env bash
set -eo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 CHECKPOINT_PATH EXTERNAL_OUTPUT_DIR" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint_path="$1"
output_dir="$2"
venv_python="${LUNAR_TRAINING_PYTHON:-/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python}"

export PYTHONPATH="$repo_root/model_contract:$repo_root/training/lunar_policy_training${PYTHONPATH:+:$PYTHONPATH}"
exec "$venv_python" -m lunar_policy_training.export.interface_v1 \
  --checkpoint "$checkpoint_path" \
  --output "$output_dir" \
  --wheeled-profile "$repo_root/ros2_ws/src/lunar_navigation_config/config/platform_profiles/wheeled.yaml" \
  --legged-profile "$repo_root/ros2_ws/src/lunar_navigation_config/config/platform_profiles/legged.yaml" \
  --hopper-profile "$repo_root/ros2_ws/src/lunar_navigation_config/config/platform_profiles/hopper.yaml"
