# Orin Unreal 仿真应用分支

应用分支：`app/orin-unreal-simulation`。工作树：`lunar-runtime/.worktrees/orin-unreal-simulation`。
建立日期：2026-09-17。只整合本地 P4/TCP 源码；没有更新 Orin、重启节点或启动车辆。

## 分支归并

| 来源 | 处理 |
| --- | --- |
| `integration/pure-planner-orin`，`5c23c13c` | 新分支起点，主线自身不变 |
| `feat/obj-tcp-simulation`，`af5feef0` | 快进纳入全部已有提交，包含 OBJ/TCP 仿真、控制器和 P4 导出基线 |
| `fix/exploration-candidate-completion` 的未提交内容 | 实现、测试与文档已在 `af5feef0` 中，不重复导入 |
| `fix/wheel-terrain-basic` 的未提交内容 | 三方应用地形与未知边界补丁，纳入新增测试和文档 |
| `fix/orin-terrain-sync` | 已核对其 tracked 补丁与 wheel-terrain-basic 完全相同；去重，不导入非 TCP 部署专用配置 |
| `fix/unreal-target-path` 的未提交内容 | 纳入 `deployment/p4_unreal_target` 与设计记录 |
| `fix/p4-gui-no-heartbeat` 的未提交内容 | 纳入 `deployment/p4_vehicle_gui` |
| `feat/unreal-tcp-exploration-planning-test`，`b92d8c2e` | 独立旧架构、无共同祖先，保留原分支作参考，不执行跨架构合并 |
| `app/orin-real-vehicle`、历史 Orin 分支和发布分支 | 实车部署线，不整支引入 |

所有原分支、原工作树及其中未提交内容均保留。原 import 快照和 integration 主线未移动。
未推送、未删除分支、未设置 upstream。逐文件来源、来源 HEAD 和 SHA256 见
[tcp-consolidation-sources.json](tcp-consolidation-sources.json)。导航节点采用三方合并，保留 TCP 原有增量修改。

## 代码入口

- `ros2_ws/src/lunar_obj_tcp_sim`：独立 OBJ/TCP 仿真适配，使用正式探索、导航和控制器。
- `ros2_ws/src/lunar_incremental_navigation_core`：5×5 稳健地形判定、未知区域边界保留、起点补偿等算法。
- `ros2_ws/src/lunar_incremental_navigation_ros`：高程接入与导航执行。
- `deployment/p4_unreal_target`：旧 `lunar_car_ctrl` 的 P4 维护副本、Unreal 目标命令 7 与局部路径命令 8；同一 TCP 连接处理控制和反馈。
- `deployment/p4_vehicle_gui`：取消 GUI 控制权心跳超时，保留显式释放与人工停车。
- `tools/tcp_deployment`：2026-09-16 的部署导出基线，**尚未与设备端最新启动脚本和配置完整对齐**。

两种车辆适配方式按场景选择，不能同时连接同一车辆：独立仿真使用 P4 TCP 桥；P3 联调使用旧车辆接口的 P4 副本独占 TCP。
不修改 P1、P2、P3 源码，不新增第二套规划/控制模块。

## 当前设备同步缺口

本轮 SSH 连接 `192.168.100.216` 超时，未获得设备当前文件快照。
`deployment/` 是本地功能源码覆盖层，不能直接宣称为完整可安装的最新 P4 包。
导航/探索入口、域发现脚本、现场配置，以及仅存在 Orin 的后续修复需联网后核对同步。
旧导出工具仍输出原部署模板，不会自动安装这些覆盖层；本轮没有生成新的部署包。

新版 relay 的 `StopPolicy` 依赖从本会话此前实际读取的 Orin `command_gate.py` 恢复，
和 Unreal 接入代码放在同一 `joint/` 目录。它保留当时临时定位等待、恢复后继续、人工停车锁定语义；
非定位持续故障仍按该快照原有 10 秒逻辑报告不可执行，本轮未重新设计此策略。
不要混用新版 relay 与旧模板的 `command_gate.py`。

Unreal IP 示例已采用用户最后指定的 `192.168.10.23`，运行时需核对实际地址。
旧文档中的设备成功记录均为历史记录，不能作为本次整合版本在 Orin 上通过的证据。

## 开发和验证

```bash
cd /home/kai/WS/lunar-navigation/lunar-runtime/.worktrees/orin-unreal-simulation
git status --short
LUNAR_ROS_DISTRO=jazzy LUNAR_OBJ_TCP_SIM_BUILD_BASE=/tmp/tcp-development-jazzy \
  bash scripts/simulation/build.sh --parallel-workers 3
```

构建目录放在仓库外。Humble/Orin 必须使用独立构建目录重新验证，不复用 Jazzy 产物。
本轮验证明细见 [TCP整合验证](TCP整合验证.md)。现场完整部署就绪前，不用本地测试替代闭环验收。
