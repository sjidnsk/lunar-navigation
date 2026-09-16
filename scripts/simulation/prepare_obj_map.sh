#!/usr/bin/env bash
set -e
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
package_root="${repo_root}/ros2_ws/src/lunar_obj_tcp_sim"
export PYTHONPATH="${package_root}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m lunar_obj_tcp_sim.prepare "$@"
