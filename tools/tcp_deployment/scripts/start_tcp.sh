#!/usr/bin/env bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
exec bash "$bundle_root/scripts/simulation/run_obj_tcp_sim.sh" --map "$bundle_root/maps/terrain" --config "$bundle_root/ros2_ws/src/lunar_obj_tcp_sim/config/simulation.yaml" --platform-config "$bundle_root/config/wheel.yaml" --mode nav "$@"
