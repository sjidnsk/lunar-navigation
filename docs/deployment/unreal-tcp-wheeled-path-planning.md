# Unreal TCP 轮式路径规划部署

本文只部署“ROS 主机指定一个目标位姿并滚动规划 WHEELED 轨迹”。它不启动探索策略、覆盖率、
PPO 或训练。计算、障碍分类、已观测地图融合和 PlanMotion 都在 Ubuntu ROS 主机；Windows
Unreal 仅提供传感器局部高程、位姿/速度、执行轨迹并回传反馈。

## 1. 拓扑与前置条件

```text
Windows / Unreal Engine 5.0.1            Ubuntu 22.04 / ROS 2 Humble
TCP server 0.0.0.0:47001  <----------  TCP client
AGX WHEELED executor                     bridge -> observed map -> planner
sensor/state producer                    -> goal coordinator
```

要求：

- 两机在同一可信局域网，地址固定或有可追踪 DHCP 租约；
- Windows 网络配置为 Private；只对 ROS 主机所在子网开放 TCP 47001；
- Unreal 项目能编译和加载 UE 5.0.1 Runtime C++ 插件；
- Ubuntu 已安装 ROS 2 Humble 和仓库正常构建所需依赖；
- 两机时钟无需相同，但 Unreal `simulation_time_ns` 必须单调；
- 首版只启动一个 Unreal 服务、一个 ROS 客户端和一台 WHEELED 机器人。

## 2. Windows Unreal 插件职责

外部 Unreal 项目不进入本仓。Runtime 插件至少分为：

- 游戏线程采样器：生成带仿真时间的 robot root 位姿、线/角速度和传感器已观测局部高程；
- 网络线程：监听、单客户端仲裁、帧编解码、有界队列和心跳；不能直接访问 UObject；
- 游戏线程执行器：消费一条授权轨迹，驱动 AGX 轮式机器人，产生严格有序反馈；
- HOLD 路径：断线、错误或 CONTROL HOLD 时取消活动轨迹并保持位置。

插件必须实现 [lunar-unreal-tcp/v1](../interfaces/lunar-unreal-tcp-v1.md) 并通过黄金向量。
`agx_plugin_version=UNKNOWN` 允许握手和日志传输，但只代表版本未知，不能算真实插件验收通过。

在管理员 PowerShell 中创建最小入站规则，然后用普通 PowerShell 检查监听：

```powershell
New-NetFirewallRule `
  -DisplayName "Lunar Unreal TCP 47001" `
  -Direction Inbound -Action Allow -Protocol TCP -LocalPort 47001 `
  -Profile Private -RemoteAddress 192.168.1.0/24

Get-NetTCPConnection -LocalPort 47001 -State Listen
ipconfig
```

预期：Unreal Editor 或 packaged build 启动后显示 `0.0.0.0:47001` 或 Windows 局域网地址的
LISTEN。若 ROS 主机不在示例子网，必须把 `RemoteAddress` 改成实际子网，不能直接放开公网。

## 3. Ubuntu 构建与仓库回归

所有 artifact 写入仓库外的全新目录：

```bash
cd /mnt/data/WS/.lunar-navigation-worktrees/unreal-tcp-wheeled-path-planning
python3 -m pip install --target \
  "$HOME/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/runtime-python" \
  'onnxruntime==1.23.2' 'rasterio==1.4.4' 'shapely==2.1.2'
export PYTHONPATH="$HOME/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/runtime-python${PYTHONPATH:+:$PYTHONPATH}"
./scripts/run_unreal_tcp_regression.sh \
  "$HOME/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/qualification-001"
```

回归脚本使用 ROS 2 Humble 的 `/usr/bin/python3` 执行包测试，因此必须显式导出上述外部依赖
目录；脚本会在开始构建前快速检查这三个既有 interface-v1 回归依赖。

预期末行：

```text
ROS-side simulated-ready evidence: .../qualification-001
```

这只证明 ROS 侧和 fake server。若只需构建而暂不跑完整回归，可复用仓库既有外置构建入口：

```bash
export LUNAR_VOLUME1_OUTPUT="$HOME/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/runtime"
./scripts/build_runtime.sh
```

## 4. 冻结坐标与标定

启动前在
`ros2_ws/src/lunar_navigation_config/config/unreal_tcp_wheeled_path_planning.yaml` 核对：

- `basis_map_from_unreal`：默认 Unreal `(X,Y,Z)` cm 映射为 ROS `(X,-Y,Z)` m；
- `unreal_world_origin_in_map_m`：Unreal world 原点在 ROS map 中的位置；
- `length_unit_to_m=0.01`；
- robot root、`base_link`、`base_footprint` 两个 4x4 刚体标定；
- `calibration_hash` 与 Unreal HELLO_ACK 完全一致。

Landscape Actor 的 `(-205160,-511559,50)` cm 不能直接当高程图左下角。地图帧必须逐帧提供
`cell_zero_center_world_cm`、u/v 轴和传感器/机器人位姿。首版传感器地图为 320 x 320、0.2 m，
ROS 只累计 valid bit 指示的已观测区域。

修改标定后重新构建/安装配置，并同时修改 Unreal 配置与 hash；运行中禁止热切换。

## 5. 两机连通与启动

在 Ubuntu 把地址替换为 Windows 局域网 IP：

```bash
ping -c 3 192.168.1.20
nc -vz 192.168.1.20 47001
```

然后启动四个 lifecycle 节点。launch 会自动 configure 和 activate：

```bash
source /opt/ros/humble/setup.bash
source "$HOME/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/qualification-001/install/setup.bash"
ros2 launch lunar_navigation_config unreal_tcp_wheeled_path_planning.launch.py \
  server_host:=192.168.1.20 server_port:=47001
```

另开终端检查：

```bash
source /opt/ros/humble/setup.bash
source "$HOME/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/qualification-001/install/setup.bash"
ros2 lifecycle get /lunar_unreal_tcp_bridge
ros2 lifecycle get /lunar_observed_map
ros2 lifecycle get /lunar_planner
ros2 lifecycle get /lunar_goal_coordinator
ros2 topic hz /lunar/unreal/wheeled_odometry
ros2 topic hz /lunar/unreal/observed_elevation
ros2 topic echo /lunar/unreal/status
```

预期四个节点均为 `active`；状态到 `READY`；Odometry 接近 20 Hz；观测高程接近 5 Hz；
诊断中记录新的 session、scene、robot、engine 和 AGX 插件版本。

如 Unreal 初始要求显式 START：

```bash
ros2 service call /lunar/unreal/start std_srvs/srv/Trigger '{}'
```

## 6. 从 ROS 发布单个目标

目标必须在当前已观测、非 obstacle、非 forbidden 且与机器人连通的 map 区域。下面脚本从
`/clock` 取得非零仿真时间并以标准 reliable/volatile QoS 发布一次；目标 Z 会由已观测高程覆盖：

```bash
python3 - <<'PY'
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

rclpy.init()
node = rclpy.create_node('send_lunar_unreal_goal')
node.set_parameters([Parameter('use_sim_time', value=True)])
qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
publisher = node.create_publisher(PoseStamped, '/goal_pose', qos)
deadline = time.monotonic() + 10.0
while node.get_clock().now().nanoseconds <= 0 and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)
if node.get_clock().now().nanoseconds <= 0:
    raise SystemExit('no nonzero /clock within 10 seconds')
goal = PoseStamped()
goal.header.frame_id = 'map'
goal.header.stamp = node.get_clock().now().to_msg()
goal.pose.position.x = 2.0
goal.pose.position.y = 0.0
goal.pose.position.z = 0.0
goal.pose.orientation.w = 1.0
publisher.publish(goal)
rclpy.spin_once(node, timeout_sec=0.5)
node.destroy_node()
rclpy.shutdown()
PY
```

观察闭环：

```bash
ros2 topic echo /lunar/path_planning/status
ros2 topic echo /lunar/motion_reference
ros2 topic echo /execution/motion_feedback
```

预期：目标生成同 session 的 mission，PlanMotion 返回 WHEELED reference；Unreal 依次回传
ACCEPTED、EXECUTING、SEGMENT_COMPLETE；只有匹配反馈并且地图、Odometry 和 TF 都更新后才会
滚动生成下一段。目标容差为 0.5 m，yaw 容差为 15 度。

## 7. 暂停、取消、恢复和重置

```bash
ros2 service call /lunar/path_planning/cancel std_srvs/srv/Trigger '{}'
ros2 service call /lunar/unreal/hold std_srvs/srv/Trigger '{}'
ros2 service call /lunar/unreal/resume std_srvs/srv/Trigger '{}'
ros2 service call /lunar/unreal/reset_session std_srvs/srv/Trigger '{}'
```

- cancel/hold 不允许自动恢复旧 reference；
- resume 只恢复接收新 reference 的能力；
- reset 或 TCP 重连必须得到新 session；
- 断线时 Unreal 本地 watchdog 也必须独立 HOLD，不能依赖 ROS 命令恰好送达。

停止时先取消目标并 HOLD，然后在 launch 终端按 `Ctrl-C`。不要用强制杀进程代替正常停止，
除非进程已经失去响应。

## 8. 资格边界

仓库 fake server 回归通过后只能称为 `ROS-side simulated-ready`。完整资格还需要在真实 Windows
Unreal 项目记录：Editor 和 packaged build 插件加载、实际 AGX 插件版本、平地直达、障碍绕行、
未知/不可达拒绝、拔网 3 秒内 HOLD、重连不重放，以及两机连续 30 分钟稳定运行。记录模板见
[资格清单](../validation/unreal-tcp-wheeled-path-planning-qualification.md)。
