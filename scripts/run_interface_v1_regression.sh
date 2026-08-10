#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/interface-v1-model-dir" >&2
  exit 2
fi
case "$1" in
  /*) ;;
  *) echo "model directory must be absolute" >&2; exit 2 ;;
esac

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="${LUNAR_INTERFACE_V1_ARTIFACT_ROOT:-$HOME/CodexDownloads/lunar_navigation/interface_v1}"
run_id="regression-$(date -u +%Y%m%dT%H%M%S%6NZ)"
output_root="$artifact_root/$run_id"
mkdir -p "$output_root"

set +u
source /opt/ros/humble/setup.bash
test "${ROS_DISTRO:-}" = humble
underlay_list="${LUNAR_INTERFACE_V1_UNDERLAYS:-${LUNAR_INTERFACE_V1_UNDERLAY:-}}"
if [[ -n "$underlay_list" ]]; then
  IFS=: read -r -a underlays <<< "$underlay_list"
  for underlay in "${underlays[@]}"; do
    source "$underlay/setup.bash"
  done
fi
set -u

export LUNAR_INTERFACE_V1_MODEL_DIR="$1"
export PYTHONPATH="$repository_root/model_contract:$repository_root/ros2_ws/src/lunar_external_adapter:$repository_root/ros2_ws/src/lunar_exploration_policy:$repository_root/training/lunar_policy_training${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python_bin="${LUNAR_INTERFACE_V1_PYTHON:-python3}"

"$python_bin" -m pytest -q \
  "$repository_root/model_contract/tests/test_interface_manifest.py" \
  "$repository_root/model_contract/tests/test_package_boundary.py" \
  "$repository_root/ros2_ws/src/lunar_external_adapter/test" \
  "$repository_root/ros2_ws/src/lunar_exploration_policy/test" \
  "$repository_root/tests/foundation/test_external_interface_config.py" \
  "$repository_root/tests/foundation/test_platform_profiles.py" \
  2>&1 | tee "$output_root/pytest.log"

python3 "$repository_root/tools/check_repository_boundaries.py" "$repository_root" \
  2>&1 | tee "$output_root/repository-boundaries.log"

printf '%s\n' "$output_root"
