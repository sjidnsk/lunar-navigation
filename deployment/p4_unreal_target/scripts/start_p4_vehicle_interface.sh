#!/usr/bin/env bash
set -euo pipefail

# 可用同名环境变量临时覆盖；正常部署无需设置。
P4_ROOT="${P4_ROOT:-/home/yanfa/P4}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
VEHICLE_SETUP="${VEHICLE_SETUP:-${P4_ROOT}/vehicle_interface/install/setup.bash}"
VEHICLE_NODE="${VEHICLE_NODE:-${P4_ROOT}/vehicle_interface/install/lunar_car_ctrl/lib/lunar_car_ctrl/lunar_car_node}"
RUN_DIR="${RUN_DIR:-${P4_ROOT}/run}"
UNREAL_IP="${UNREAL_IP:-192.168.10.23}"
UNREAL_PORT="${UNREAL_PORT:-6668}"
P4_VEHICLE_ROS_DOMAIN_ID="${P4_VEHICLE_ROS_DOMAIN_ID:-10}"
MAX_LINEAR_SPEED="${MAX_LINEAR_SPEED:-0.2}"
P4_STARTUP_WAIT_SECONDS="${P4_STARTUP_WAIT_SECONDS:-2}"

PID_FILE="${RUN_DIR}/lunar_car_ctrl.pid"
LOG_FILE="${RUN_DIR}/lunar_car_ctrl.log"

if existing_processes="$(pgrep -af '[l]unar_car_node' 2>/dev/null)"; then
    echo "错误：已经存在车辆接口进程，拒绝重复启动：" >&2
    echo "${existing_processes}" >&2
    echo "请确认它属于 Env_X 还是 P4；本脚本不会自动停止已有进程。" >&2
    exit 3
fi

for required_file in "${ROS_SETUP}" "${VEHICLE_SETUP}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "错误：没有找到环境文件：${required_file}" >&2
        exit 2
    fi
done

if [[ ! -x "${VEHICLE_NODE}" ]]; then
    echo "错误：没有找到接口程序或程序不可执行：${VEHICLE_NODE}" >&2
    echo "请先构建 /home/yanfa/P4/vehicle_interface。" >&2
    exit 2
fi

mkdir -p "${RUN_DIR}"

# 避免当前终端中的 Conda/Python 环境污染 ROS 2 Humble。
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV || true
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONNOUSERSITE=1

# ROS/colcon 的 setup.bash 会读取若干可选变量，加载期间不能启用 nounset。
set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
# shellcheck disable=SC1090
source "${VEHICLE_SETUP}"
set -u

export ROS_DOMAIN_ID="${P4_VEHICLE_ROS_DOMAIN_ID}"
export RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
export ROS_LOCALHOST_ONLY=0

echo "启动 P4 车辆接口：Unreal=${UNREAL_IP}:${UNREAL_PORT} ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
nohup "${VEHICLE_NODE}" \
    --ros-args \
    -p "tcp_host:=${UNREAL_IP}" \
    -p "tcp_port:=${UNREAL_PORT}" \
    -p big_endian:=false \
    -p "max_linear_speed:=${MAX_LINEAR_SPEED}" \
    -p control_mode:=park \
    >>"${LOG_FILE}" 2>&1 &
node_pid=$!
printf '%s\n' "${node_pid}" >"${PID_FILE}"

sleep "${P4_STARTUP_WAIT_SECONDS}"
if ! kill -0 "${node_pid}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "错误：接口启动后退出，请检查日志：${LOG_FILE}" >&2
    tail -n 30 "${LOG_FILE}" >&2 || true
    exit 1
fi

echo "P4 车辆接口已启动，PID=${node_pid}"
echo "日志：${LOG_FILE}"
echo "确认连接：tail -f ${LOG_FILE}"
