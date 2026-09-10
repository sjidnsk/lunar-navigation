# 增量规划控制闭环 demo 验证记录

本次主线导入及 30 秒默认等待超时的验证见[集成记录](集成地图探索规划控制修改.md)。下文历史 demo 数据保留其当时参数。

日期：2026-09-10。工作树：`../incremental-controller-rviz`；分支 `feat/incremental-controller-rviz`；基线 `df469d1`。
没有 merge、push、PR 或修改其他工作树。构建和完整证据位于 `/tmp/lunar-controller-demo`，不提交运行产物。

## 源码与包验证

- 受影响 Jazzy 包成功构建：消息、增量核心、增量 ROS、轮式控制器、legacy planner core/ROS 和新演示包。
- 串行包测试汇总：**693 tests, 0 errors, 0 failures, 0 skipped**。包含控制器 147 项、导航节点 33 项、演示车辆/证据门槛 21 项，以及核心、消息、可视化、配置加载与其他 ROS 测试。汇总包含 CTest 包装计数，不等同于独立业务用例数。
- 根目录相关接口/旧 launch 静态测试：**57 passed, 4 skipped**；跳过的 4 项要求旧 demo 的探索运行环境，本轮未执行这些旧 demo live 用例。
- UTF-8、Python 语法与 `git diff --check` 已检查。
- 日志：`/tmp/lunar-controller-final-test-result.log`、`/tmp/lunar-controller-final-tests.log`、`/tmp/lunar-controller-final-core-tests.log`、`/tmp/lunar-controller-contract-tests.log`。

## 本次实际发现并修复的问题

1. **连续坐标落在栅格角点导致假 TIMEOUT。** 首次 `(0,0)→(3,0)`、0.2 m 分辨率平地场景仅约 6 ms 就失败。旧 supercover DDA 的终端角点舍入越界导致路径后处理拒绝；补齐 `bd39d26` 的保守角点覆盖配套修复，不移动起终点；非超时的后处理失败改为 `PATH_POSTPROCESS_FAILED`。原样红绿测试和角点认证测试通过。首次失败 `/tmp/lunar-controller-demo/evidence/forward.json` 保留。
2. **终点转向时失去位置容差仍可误报完成。** 控制器 FINAL_ALIGN 每周期重新检查位置；偏离时先 BRAKING，实测停稳后 FAILED/GOAL_POSITION_LOST。4 个红绿回归覆盖 final/nonfinal、静止/运动。
3. **地图更新与完成反馈同时到达时跳过地图校验。** 先处理 fine 安全校验，再处理仍有效引用的 TrackingStatus。确定性回归确认阻塞 revision 2 使旧段失效并重规划，不沿用 revision 1 宣布成功。
4. **记录器可能借用旧任务的 PLAN_FOUND 或未停稳 COMPLETED。** 门槛现在严格匹配目标 session 和引用 revision，且要求完成消息实测线/角速度有限并停稳。保留 11 个先失败后通过的反例。

## 场景参数与边界案例

模拟车真实命令驱动，0.2 m/s 双向，0.3 m/s² 加速、0.5 m/s² 制动、0.5 rad/s² 角加速度。后两项与共享能力上限一致。旧首轮使用较快制动/角加速度，最终场景已采用上述值重新验证。

多出口初始目标为 `(8,0)`：

- 4.5 m 观测返回 `LOCAL_NO_PROGRESS`，没有发布有效路径、没有运动；证据 `evidence/repaired-multiple_exits.json`。
- 6.5 m 观测返回 `WAITING_FOR_MAP`，位姿和地图不变时不重复局部搜索，保持停稳，180 s 记录器超时取消；证据 `evidence/final-multiple_exits.json`。不能将其计为成功。
- 固定障碍中央通道宽 1.8 m，小于含 0.2 m 安全间距的车体膨胀直径；外侧绕行需要更多已知空间。短观测的停等是本 demo 明确暴露的边界，不能通过把 UNKNOWN 当 FREE 或放宽平台认证解决。

默认可运行多出口场景采用 10 m 合成观测、目标 `(12,0)`，保证目标起初不在细已知空间内，验证非最终引用及后续切段。合成观测为 360° 圆形，无光线遮挡；这些参数不代表实车传感器。

要复现短观测边界，可启动 `case:=multiple_exits observation_radius_m:=6.5`，再用 RViz 指定 `(8,0)`；或记录器用当前默认 `(12,0)` 检查等待边界。旧 JSON 的目标为 `(8,0)`，不要按更新后的默认目标解释。

## ROS / RViz 证据

固定场景在独立域 74–78 中运行，结果逐项核对有效引用、PLAN_FOUND、同 session/revision 的最终停稳 COMPLETED、Action GOAL_REACHED、实际位置/朝向、速度和碰撞。

| 场景 | 结果 | 路径段数 | 最大命令 / 实际速度 m/s | 位置误差 m | 耗时 s |
|---|---|---:|---|---:|---:|
| forward | PASS | 1 | 0.200 / 0.200 | 0.159 | 18.4 |
| reverse | PASS | 1 | 0.200 / 0.200 | 0.155 | 18.5 |
| final_yaw | PASS | 1 | 0.200 / 0.200 | 0.155 | 22.4 |
| detour | PASS | 1 | 0.200 / 0.200 | 0.155 | 63.7 |
| multiple_exits | PASS | 2 | 0.200 / 0.200 | 0.156 | 99.8 |

前四项证据为 `evidence/final-<case>.json`；多出口为 `evidence/observation10-multiple_exits.json`，其第一段 `reaches_final_goal=false`、第二段为 true。全部无原始命令超限、无碰撞且停稳完成。多出口最终目标为 (12,0)，观测半径 10 m。

RViz 域 72 的绕障运行另存 `evidence/rviz-detour.json`，已通过；旧域 71 demo 保留。可视检查地图、规划路径、实际轨迹、候选/选中点、速度和阶段正常显示。启动时显卡记录过一次 indexed-map shader 链接警告，后续地图正常渲染，见 `evidence/rviz-first.png`。

隔离检查：`/lunar_demo/controller/cmd_vel` 一个发布者 `controller_demo_executor`、一个订阅者 `incremental_controller_vehicle`；域 72 中 `/Car/T5/Car_Cmd_Vel` 不存在。

## 未验证层级

Humble 容器/主机：`NOT_RUN`。Jetson AGX Orin：`NOT_RUN`。真实传感器、DDS 实机链路、实车：`NOT_RUN`。本机 Jazzy 仿真不能证明这些层级就绪。
