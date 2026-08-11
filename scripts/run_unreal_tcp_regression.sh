#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/external/output-directory" >&2
  exit 2
fi
if [[ "$1" != /* ]]; then
  echo "output directory must be absolute" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
output_root="$(realpath -m -- "$1")"
if [[ "$output_root" == "/" ||
      "$output_root" == "$repository_root" ||
      "$output_root" == "$repository_root/"* ]]; then
  echo "output directory must be a dedicated path outside the repository" >&2
  exit 2
fi
if [[ -d "$output_root" ]] &&
   [[ -n "$(find "$output_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "output directory must be empty: $output_root" >&2
  exit 2
fi

mkdir -p "$output_root"

set +u
source /opt/ros/humble/setup.bash
set -u
if [[ "${ROS_DISTRO:-}" != "humble" ]]; then
  echo "ROS 2 Humble is required" >&2
  exit 2
fi

if ! /usr/bin/python3 - <<'PY'
from importlib.util import find_spec
import sys

required = ("onnxruntime", "rasterio", "shapely")
missing = tuple(name for name in required if find_spec(name) is None)
if missing:
    print("missing ROS Python modules: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
then
  echo "ROS Python dependencies are missing; follow the deployment Python dependency setup" >&2
  exit 2
fi

cd "$repository_root"
colcon --log-base "$output_root/log" build \
  --merge-install \
  --executor parallel \
  --parallel-workers "${LUNAR_UNREAL_TCP_BUILD_WORKERS:-2}" \
  --base-paths ros2_ws/src model_contract \
  --packages-skip lunar_nav2_adapter \
  --build-base "$output_root/build" \
  --install-base "$output_root/install" \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON

set +u
source "$output_root/install/setup.bash"
set -u

colcon --log-base "$output_root/test-log" test \
  --merge-install \
  --packages-skip lunar_nav2_adapter \
  --build-base "$output_root/build" \
  --install-base "$output_root/install" \
  --test-result-base "$output_root/test-results"
colcon test-result \
  --test-result-base "$output_root/test-results" \
  --verbose 2>&1 | tee "$output_root/colcon-test-result.log"

export PYTHONPATH="$repository_root/model_contract:$repository_root/ros2_ws/src/lunar_external_adapter:$repository_root/ros2_ws/src/lunar_exploration_policy:$repository_root/training/lunar_policy_training${PYTHONPATH:+:$PYTHONPATH}"
export ROS_DOMAIN_ID="${LUNAR_UNREAL_TCP_ROS_DOMAIN_ID:-197}"

python3 -m pytest -q \
  tests/integration/unreal_tcp/test_launch_contract.py \
  2>&1 | tee "$output_root/launch-contract.log"
python3 -m pytest -q \
  tests/integration/unreal_tcp/test_unreal_tcp_stack.launch.py \
  2>&1 | tee "$output_root/unreal-stack-integration.log"
python3 -m pytest -q \
  tests/foundation \
  tests/ros \
  ros2_ws/src/lunar_external_adapter/test \
  ros2_ws/src/lunar_exploration_policy/test \
  2>&1 | tee "$output_root/affected-pytest.log"

python3 tools/check_repository_boundaries.py . \
  2>&1 | tee "$output_root/repository-boundaries.log"
python3 -m pytest -q tests/foundation/test_repository_boundaries.py \
  2>&1 | tee "$output_root/repository-boundary-pytest.log"

printf 'ROS-side simulated-ready evidence: %s\n' "$output_root"
