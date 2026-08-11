# Unreal TCP 轮式路径规划资格边界

本文件按证据来源区分实现完成度，防止把 fake server 结果误写成真实 Unreal/AGX 或两机验收。

状态定义：

- `REPOSITORY_VERIFIED`：本仓代码与自动测试可重复验证；
- `EXTERNAL_UNREAL_PENDING`：需要外部 Windows Unreal 5.0.1 项目与真实 AGX 插件；
- `TWO_MACHINE_PENDING`：需要两台实体计算机在局域网联调；
- `NOT_IN_SCOPE`：本期明确不启动。

## 1. 仓库内证据

权威入口：

```bash
./scripts/run_unreal_tcp_regression.sh \
  "$HOME/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/qualification-001"
```

输出目录必须留存 `colcon-test-result.log`、`launch-contract.log`、
`unreal-stack-integration.log`、`affected-pytest.log` 和 repository boundary 日志。

| 检查项 | 状态 | 仓库证据 |
|---|---|---|
| 48-byte 帧、CRC、封闭 metadata、黄金向量 | REPOSITORY_VERIFIED | bridge protocol/metadata/stream tests |
| sequence、session、心跳、重连失效 | REPOSITORY_VERIFIED | session/TCP client tests |
| Unreal ↔ ROS 位置、姿态、线/角速度转换 | REPOSITORY_VERIFIED | coordinate/ROS conversion tests |
| 20 Hz 状态、5 Hz 观测图与 `/clock` 接线 | REPOSITORY_VERIFIED | fake-server launch test |
| 仅 valid 观测进入稀疏 0.2 m 地图 | REPOSITORY_VERIFIED | sparse map/product/node tests |
| 十层 local/global GridMap 可被真实 planner adapter 接受 | REPOSITORY_VERIFIED | observed map tests + launch test |
| `map→odom→base_link→base_footprint` 规划快照 | REPOSITORY_VERIFIED | planner TF-chain test + launch test |
| `/goal_pose` 到 mission、PlanMotion、WHEELED reference | REPOSITORY_VERIFIED | coordinator tests + launch test |
| 匹配 SEGMENT_COMPLETE 与新快照触发第二段 | REPOSITORY_VERIFIED | launch integration test |
| wrong plan/session、stale、skew、disconnect fail closed | REPOSITORY_VERIFIED | launch fault-injection test |
| 重连不重放旧 reference | REPOSITORY_VERIFIED | launch reconnect test |
| 启动图排除探索、PPO 和训练节点 | REPOSITORY_VERIFIED | launch contract test |
| 冻结 PlanMotion/MotionReference/Topic 未改名 | REPOSITORY_VERIFIED | affected regressions + boundary checks |

上述结果只允许发布结论：`ROS-side simulated-ready`。

## 2. 外部 Windows Unreal 证据

以下项目在外部 checkout 和实际运行证据到位前保持 pending：

| 检查项 | 状态 | 通过条件 |
|---|---|---|
| UE 5.0.1 Editor 加载 Runtime 插件 | EXTERNAL_UNREAL_PENDING | 启动日志、模块名和版本可追踪 |
| UE 5.0.1 packaged build 加载插件 | EXTERNAL_UNREAL_PENDING | 无 Editor 依赖并成功监听 |
| 实际 AGX 插件版本 | EXTERNAL_UNREAL_PENDING | HELLO_ACK 与日志均为真实版本，不是 UNKNOWN |
| 黄金向量互操作 | EXTERNAL_UNREAL_PENDING | Unreal 编解码完整 bytes 与 fixture 一致 |
| 游戏线程/网络线程隔离 | EXTERNAL_UNREAL_PENDING | 代码审查与运行时压力证据 |
| 传感器仅输出已观测单元 | EXTERNAL_UNREAL_PENDING | 与视场遮挡案例核对 valid bit |
| AGX 执行 reference | EXTERNAL_UNREAL_PENDING | ACCEPTED/EXECUTING/SEGMENT_COMPLETE 严格有序 |
| HOLD 和本地断线 watchdog | EXTERNAL_UNREAL_PENDING | ROS 不可达时机器人仍停止并保持 |

`agx_plugin_version=UNKNOWN` 已在仓库协议和 fake server 中验证可传输，但这不关闭“实际版本”
检查项，也不证明 AGX 执行器已接入。

## 3. 两机功能与稳定性证据

| 场景 | 状态 | 通过条件 |
|---|---|---|
| 平坦已观测区域直达 | TWO_MACHINE_PENDING | 到达 0.5 m / 15° 容差并反馈终态 |
| 已观测离散障碍绕行 | TWO_MACHINE_PENDING | 轨迹不穿障碍，最终到达目标 |
| 未知、活动区外或不可达目标 | TWO_MACHINE_PENDING | 不下发运动，保持 HOLD，原因码稳定 |
| 执行中拔断局域网 | TWO_MACHINE_PENDING | 3 秒内本地 HOLD，恢复网络后新 session |
| 重连安全 | TWO_MACHINE_PENDING | 不重放旧目标/reference/反馈 |
| 频率 | TWO_MACHINE_PENDING | 状态不低于 18 Hz，地图不低于 4.5 Hz |
| 连续稳定性 | TWO_MACHINE_PENDING | 两机运行 30 分钟，无死锁、无无界内存增长、无错误运动 |
| Windows 防火墙最小开放 | TWO_MACHINE_PENDING | 仅 Private/目标子网 TCP 47001 可达 |

每次现场运行应在仓库外保存：两机版本和 commit、Unreal/AGX 版本、Windows 监听与防火墙输出、
ROS diagnostics、目标与 reference/feedback 记录、断线时间线、30 分钟资源曲线和最终结论。

## 4. 设计完成条件对账

| 设计完成条件 | 当前分类 |
|---|---|
| ROS 主机主动连接 Windows Unreal | TWO_MACHINE_PENDING |
| 持续接收真实传感器高程、位姿和速度 | TWO_MACHINE_PENDING |
| ROS 只用已观测数据生成合法地图与障碍 | REPOSITORY_VERIFIED；真实输入仍需外部核对 |
| ROS 发布单目标位姿 | REPOSITORY_VERIFIED |
| WHEELED PlanMotion 产生轨迹 | REPOSITORY_VERIFIED |
| Unreal/AGX 执行轨迹 | EXTERNAL_UNREAL_PENDING |
| 匹配反馈驱动两段或到达 | REPOSITORY_VERIFIED；真实执行仍需两机核对 |
| 三功能案例、断线案例和 30 分钟稳定性 | TWO_MACHINE_PENDING |
| 不启动探索、覆盖率、PPO 或训练 | REPOSITORY_VERIFIED / NOT_IN_SCOPE |
| 不改冻结 Action、消息和 Topic | REPOSITORY_VERIFIED |
| 外部 Unreal 源码和 artifact 不进入本仓 | REPOSITORY_VERIFIED |

## 5. 发布结论规则

- 仓库入口全绿但外部和两机项未完成：只能写 `ROS-side simulated-ready`。
- 外部插件项通过但两机项未完成：可以写“插件互操作已验证”，不能写“系统验收通过”。
- 只有外部 Unreal 与全部两机项都有可追踪证据时，才可将完整系统标记为 qualified。
- 任一 fail-closed、重连不重放或 HOLD 时限失败都阻止发布，不允许以成功率平均掩盖。
