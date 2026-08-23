#!/usr/bin/env bash
# Interactively send one wheel/lava PlanMotion goal from the bundle root.
set -eo pipefail

bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
overlay="${bundle_root}/ros2_ws/install/setup.bash"
action_name="/Car/T4/plan_motion"

is_number() {
  [[ "$1" =~ ^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]
}

read_number() {
  local prompt="$1"
  local value
  while true; do
    read -r -p "$prompt" value
    if is_number "$value"; then
      printf '%s' "$value"
      return 0
    fi
    echo "请输入有效数字。" >&2
  done
}

if [[ ! -r "$overlay" ]]; then
  echo "未找到构建 overlay：$overlay" >&2
  echo "请先在 ros2_ws 中完成 colcon build。" >&2
  exit 1
fi

source /opt/ros/humble/setup.bash
source "${bundle_root}/ros2_ws/install/setup.bash"

if ! ros2 action list 2>/dev/null | grep -Fxq "$action_name"; then
  echo "Action 服务未启动：$action_name" >&2
  echo "请先执行：${bundle_root}/scripts/start_all.sh" >&2
  exit 1
fi

echo "输入轮式熔岩洞目标（坐标系：odom，单位：m）。"
x="$(read_number "目标 x (odom, m): ")"
y="$(read_number "目标 y (odom, m): ")"

read -r -p "目标 yaw (rad，直接回车表示不约束): " yaw
while [[ -n "$yaw" ]] && ! is_number "$yaw"; do
  echo "请输入有效数字，或直接回车跳过 yaw。" >&2
  read -r -p "目标 yaw (rad，直接回车表示不约束): " yaw
done

read -r -p "本次使用完全信任桥接? [y/N]: " use_bridge
case "${use_bridge,,}" in
  y|yes)
    ros2 param set /pure_planner trusted_bridge_once true
    ;;
  ""|n|no)
    ;;
  *)
    echo "无效选择：请输入 y 或直接回车。" >&2
    exit 2
    ;;
esac

request_id="manual-$(date +%Y%m%d-%H%M%S)"
if [[ -n "$yaw" ]]; then
  yaw_fields="has_yaw_constraint: true, yaw_rad: ${yaw}, yaw_tolerance_rad: 0.2"
else
  yaw_fields="has_yaw_constraint: false, yaw_rad: 0.0, yaw_tolerance_rad: 0.0"
fi

goal="{
  environment_mode: 2,
  request_id: '${request_id}',
  mission_id: 'manual',
  mission_revision: 0,
  goal: {
    header: {frame_id: 'odom'},
    goal_id: '${request_id}',
    goal_type: 1,
    point: {x: ${x}, y: ${y}},
    position_tolerance_m: 0.2,
    ${yaw_fields}
  },
  replace_active_request: true
}"

echo "发送目标：x=${x}, y=${y}${yaw:+, yaw=${yaw}}"
ros2 action send_goal --feedback "$action_name" \
  lunar_planning_msgs/action/PlanMotion "$goal"
