#!/usr/bin/env bash
# Sourced by the entry scripts. Each extracted bundle has a separate build tree.
bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LUNAR_ROS_DISTRO="${LUNAR_ROS_DISTRO:-humble}"
export LUNAR_OBJ_TCP_SIM_BUILD_BASE="${LUNAR_OBJ_TCP_SIM_BUILD_BASE:-${bundle_root}/artifacts/${LUNAR_ROS_DISTRO}}"
export LUNAR_OBJ_TCP_SIM_OVERLAY="${LUNAR_OBJ_TCP_SIM_BUILD_BASE}/install"
