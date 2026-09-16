#!/usr/bin/env bash
set -e

host=""; port=""; map_directory=""; mode=""; prefix=""
config=""; platform_config=""; start_rviz=""; start_local_rviz=""
no_rviz="false"; domain_id="${ROS_DOMAIN_ID:-74}"; dry_run="false"
usage() { echo "usage: $0 [--map DIR] [--host HOST] [--port PORT] [--mode explore|nav] [--config FILE] [--platform-config FILE] [--prefix PREFIX] [--rviz|--no-rviz] [--local-rviz] [--domain-id ID] [--dry-run]"; }
while (($#)); do
  case "$1" in
    --host) host="$2"; shift 2;; --port) port="$2"; shift 2;;
    --map) map_directory="$2"; shift 2;; --mode) mode="$2"; shift 2;;
    --prefix) prefix="$2"; shift 2;; --config) config="$2"; shift 2;;
    --platform-config) platform_config="$2"; shift 2;;
    --rviz) start_rviz="true"; shift;; --no-rviz) no_rviz="true"; shift;;
    --local-rviz) start_local_rviz="true"; shift;; --domain-id) domain_id="$2"; shift 2;;
    --dry-run) dry_run="true"; shift;; -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
if [[ -n "$mode" && "$mode" != "explore" && "$mode" != "nav" ]]; then echo "mode must be explore or nav" >&2; exit 2; fi
if [[ "$no_rviz" == "true" ]]; then start_rviz="false"; start_local_rviz="false"; fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
package_root="${repo_root}/ros2_ws/src/lunar_obj_tcp_sim"
effective_config="${config:-${package_root}/config/simulation.yaml}"
resolved="$({ PYTHONPATH="${package_root}${PYTHONPATH:+:${PYTHONPATH}}" python3 - "$effective_config" <<'PY'
import sys
from lunar_obj_tcp_sim.configuration import load_simulation_config
c = load_simulation_config(sys.argv[1])
print(c.host); print(c.port); print(c.map_directory); print(c.mode); print(c.prefix)
PY
} )" || exit 2
mapfile -t configured <<< "$resolved"
display_host="${host:-${configured[0]}}"; display_port="${port:-${configured[1]}}"
display_map="${map_directory:-${configured[2]}}"; display_mode="${mode:-${configured[3]}}"
display_prefix="${prefix:-${configured[4]}}"
if [[ -z "$display_map" ]]; then echo "map_directory is required via --map or --config" >&2; exit 2; fi

launch_args=()
[[ -n "$config" ]] && launch_args+=("config:=${config}")
[[ -n "$host" ]] && launch_args+=("host:=${host}")
[[ -n "$port" ]] && launch_args+=("port:=${port}")
[[ -n "$map_directory" ]] && launch_args+=("map_directory:=${map_directory}")
[[ -n "$mode" ]] && launch_args+=("mode:=${mode}")
[[ -n "$prefix" ]] && launch_args+=("prefix:=${prefix}")
[[ -n "$platform_config" ]] && launch_args+=("platform_config:=${platform_config}")
[[ -n "$start_rviz" ]] && launch_args+=("start_rviz:=${start_rviz}")
[[ -n "$start_local_rviz" ]] && launch_args+=("start_local_rviz:=${start_local_rviz}")
echo "ROS_DOMAIN_ID=${domain_id} host=${display_host} port=${display_port} map=${display_map} mode=${display_mode} command_topic=${display_prefix}/cmd_vel"
operator_args=(python3 -m lunar_obj_tcp_sim.operator --prefix "$display_prefix" --mode "$display_mode" --shutdown-timeout 5 -- ros2 launch lunar_obj_tcp_sim obj_tcp.launch.py)
if [[ "$dry_run" == "true" ]]; then printf '%q ' "${operator_args[@]}"; printf '%q ' "${launch_args[@]}"; printf '\n'; exit 0; fi

ros_distro="${LUNAR_ROS_DISTRO:-${ROS_DISTRO:-}}"
if [[ -n "${LUNAR_ROS_DISTRO:-}" && -n "${ROS_DISTRO:-}" && "$LUNAR_ROS_DISTRO" != "$ROS_DISTRO" ]]; then
  echo "refusing mixed ROS distributions: ROS_DISTRO=${ROS_DISTRO}, LUNAR_ROS_DISTRO=${LUNAR_ROS_DISTRO}" >&2
  exit 2
fi
if [[ -z "$ros_distro" || ! -r "/opt/ros/${ros_distro}/setup.bash" ]]; then echo "set ROS_DISTRO or LUNAR_ROS_DISTRO to an installed ROS distribution" >&2; exit 2; fi
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH ROS_PACKAGE_PATH
unset PYTHONPATH LD_LIBRARY_PATH
source "/opt/ros/${ros_distro}/setup.bash"
overlay="${LUNAR_OBJ_TCP_SIM_OVERLAY:-${XDG_CACHE_HOME:-${HOME}/.cache}/lunar_obj_tcp_sim/${ros_distro}/install}"
if [[ -r "${overlay}/setup.bash" ]]; then source "${overlay}/setup.bash"; fi
set -u
export ROS_DOMAIN_ID="$domain_id"
exec "${operator_args[@]}" "${launch_args[@]}"
