# Unreal 目标与 P4 局部路径（旧车辆接口扩展）

在 P4 下维护 `lunar_car_ctrl` 副本；不修改 `/home/yanfa/Env_X/InterFace/src/lunar_car_ctrl` 或 P3 文件。不启用旧 P4 TCP 桥。一个车辆 TCP 客户端同时处理控制、遥测、目标和路径。

## 数据流

UE 点击 → 控制端口命令 7 → 旧车辆接口副本 → `/car/ue_path_request` → 现有 P4 位姿适配进程中的导航接入逻辑 → `NavigateToPose`（`has_target_yaw=false`）。

当前导航的 `/Car/T4/planning/path_reference` → 同请求 ID 的路径响应 → `/car/ue_path_response` → 接口副本发命令 8 → UE `SetPlannedPath()`。

接口位于旧车辆 ROS 域；P4 适配进程继续使用两个既有 ROS context。没有新增 TCP 连接或常驻 ROS 节点。导航和探索启动入口保持不变。UE 点击会暂停正在运行的探索；旧导航取消完成后再提交新目标。外部停车仍按现有语义暂停/取消，不能通过路径回传绕过。

## 协议

小端。包头 `<4I`：Magic=0x4C494441，Command，PayloadSize，Checksum=三者异或。

- 目标命令 7：`<I3f`，RequestId + UE 世界坐标 XYZ，厘米，负载16字节。
- 响应命令 8：`<IB3xI` + N×`<3f`，RequestId、Status、PointCount、UE 世界坐标 XYZ（厘米）。Status 0 有有效路径；1 导航结果为 NO_PATH；2 失败、取消或路径失效。Status 1/2 均为空点列。
- 路径版本更新继续使用同一个 RequestId。只返回当前 Action UUID 匹配的 ACTIVE PathReference；旧会话、旧版本、同版重复消息丢弃。已失效版本不能重新激活。
- 无目标航向；保留控制器沿路径前进及局部路径末段自身的方向处理。
- `/car/ue_path_request` 和 `/car/ue_path_response` 是仅用于部署适配的 `std_msgs/UInt8MultiArray`，data 为完整协议负载（没有 TCP 包头）。不作为公共规划接口。
- `/car/ue_connected` 为连接状态，断线清除当前 UE 导航关联；`/P4/input/ue_goal_authorize` 仅由有效 UE 目标提交时发布，用于既有转发器授权。

## 坐标和显示高度

目标厘米转米、Y 取反，复用反馈适配进程的当前静止对齐（UE RH→P3 odom），再使用收到的 map→odom TF 转入 map。返回路径严格做逆变换。重启 P3 后应按既有流程重启 P4，建立新会话对齐。

轮式局部路径原始 Z=0，不能直接当作 UE 世界高度。返回保留原始 XY 路径点，显示用 Z 优先读取 P3 `elevation`（支持 GridMap 行列布局和循环缓冲）。缺失高度在已有有效点之间插值、两端延用最近高度；若整条路径都没有可采样高程，则使用点击目标转换后的高度。`estimated_display_heights` 明确记录估算点数。该处理仅用于画线，不修改路径 XY、障碍判断、跟踪或未知格通行性；并非坡面真值保证。

## Orin 使用

部署位置：`/home/yanfa/P4/vehicle_interface`，扩展接口源码在 `src/lunar_car_ctrl`，构建和安装在自身 `build/`、`install/`。接入文件安装到 `/home/yanfa/P4/debug/p3_joint/`。

P3 建图有效且车辆接口运行后：

```bash
bash /home/yanfa/P4/scripts/navigation.sh start
```

等对齐完成，在 Unreal 点击目标即可导航；UE 将收到有效局部路径及后续版本。局部路径终点可先停在已知地图内，不一定立即等于最终点击目标。

检查：

```bash
cat /home/yanfa/P4/debug/p3_joint/unreal-path-status.json
cat /home/yanfa/P4/debug/p3_joint/command-relay-status.json
```

`PATH_RETURNED` 表示路径已交给接口发送，不代表 UE 已渲染，也不代表车已到达。接口日志 `UE path response ...` 表示 TCP 完整写入成功。协议没有 UE 接收确认。

单独启动接口（必须先确认旧接口已经停止，ROS_DOMAIN_ID 使用现场旧车辆域；本次为10）：

```bash
bash /home/yanfa/P4/scripts/start_p4_vehicle_interface.sh
```

脚本默认连接 `192.168.10.23:6668`、使用 ROS 域10、线速度上限0.2 m/s，并以 `park` 模式后台启动。已有 `lunar_car_node` 时脚本拒绝重复启动且不会结束原进程。可用 `UNREAL_IP`、`UNREAL_PORT`、`P4_VEHICLE_ROS_DOMAIN_ID` 或 `MAX_LINEAR_SPEED` 环境变量临时覆盖默认值。

原 P3 采集入口可能在没有接口进程时启动原版本，因此整机重启后应先启动此扩展版本，再启动 P3/P4。不要同时运行两个车辆接口。

## 验证与边界

见 `VALIDATION.md`。提供纯函数测试和 Orin 原生端到端回环测试，后者使用 127.0.0.1 临时端口、ROS 域181/182和假的导航 Action，只能在这两个域空闲时运行。测试不会连接 Unreal。

Unreal 提供的 `TCPComponent.cpp` 仍需接收缓存解决 TCP 半包/粘包，且未显式处理 Status。Orin 完整发送不能保证对端一次 Receive 回调恰好获得一个包；现场显示必须单独确认。本次不修改 UE 源码。

## 回滚

先 `bash /home/yanfa/P4/scripts/navigation.sh stop`。用部署备份记录核对扩展接口 PID/程序路径后停止扩展接口，按备份 `original-runtime.json` 的命令和环境恢复原进程；恢复备份中的两个 P4 适配脚本。原 Env_X 接口安装和源码始终保留。最后按既有流程启动 P4。切勿用宽泛 pkill 结束 P3。

## 开发分支整合说明（2026-09-17）

本目录是旧接口接入的源码覆盖层，不能单独替代完整 Orin 部署包。`joint/` 必须作为同一版本使用，包含 `command_gate.py`；不要搭配 `tools/tcp_deployment/templates` 中较早的转发器依赖。后者是 2026-09-16 部署基线。

`command_gate.py` 从本会话 2026-09-17 现场只读输出恢复，保留当时的 0.2 m/s、0.1 rad/s 与临时定位等待语义；尚未与设备当前文件重新比对。导航/探索入口、域发现、最新配置等仍需设备可达后完整同步。IP 示例采用用户最后指定的 192.168.10.23，实际部署应核对现场端点。
