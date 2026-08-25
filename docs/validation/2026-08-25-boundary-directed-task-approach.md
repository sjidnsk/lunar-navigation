# Boundary-Directed Task Approach Validation

## 结论与范围

2026-08-25 在隔离特性分支完成“任务区外启动，先安全进入任务区，再自主探索”。实现只扩展纯探索
core/ROS 协调器、300 m 资源参数和测试；规划器生产实现仍为已批准基线
`3f5b0f746a15d48edced0c045387154eb808e59c` 的
`grid_traversability_v1`，未增加 legacy fallback、控制器行为或外部接口。

## 已实现合同

- 完整车体不在任务边界内时进入 `APPROACH_TASK`；完整进入后切换为 `EXPLORE_TASK`。
- UNKNOWN 仅用于边界引导意图。每个 PlanMotion 目标必须是当代地图中的 FREE cell，并通过共享
  `SafePoseValidator` 的车体包络、占用阈值与间距校验。
- 入口候选使用冻结的完整 typed identity；地图变化、滚动 successor、停稳重规划和旧 terminal
  均受 task/request/map generation/goal identity 线性化守卫。
- 活跃入口目标失效时精确取消当前 plan/replan 并基于最新地图重建；真实 authority 缺失保持
  `FINAL_MAP_VALIDATION_ERROR` fail-closed。
- `APPROACH_NO_REACHABLE_TARGET`、`WAITING_FOR_TASK_MAP_COVERAGE` 和
  `APPROACH_STALLED` 均为非完成等待。
- 完成仍只来自 `EXPLORE_TASK` 中 WFD 候选的 Grid V1 穷尽不可达；
  `coverage_ratio > 0.80` 仅是 20 m 合成场景断言，不是停止阈值。
- 300 m launch 保持 `wheel_planner_mode=grid_traversability_v1`、
  `rolling_surface_enabled=false`，并显式设置 guidance
  `1048576 / 8388608 / 4096` 资源上限。

## 本机 Jazzy 测试证据

- Task 6 六个关键地图/重规划交错重复十轮：60/60。
- Task 6 authority/race：10/10；Marker/诊断：4/4。
- 最终组合 CTest：`test_exploration_node` 108/108，
  `test_synthetic_scenarios` 16/16，2/2 targets。
- 正例与三个负例额外重复三轮：12/12。
- Python 场景与 launch：37 passed，1 skipped；`py_compile` 通过。
- 20 m 正例冻结有序 Grid V1 请求/响应，至少两段入口、至少一段 WFD，最终
  `COMPLETED_NO_REACHABLE_FRONTIER`，任务覆盖率 0.875；规范 trace 为
  `fnv1a64:958c9aa1ee6a8991`。
- 三个负例在重复 poll 后保持原 reason、请求数和状态发布数不变，且零
  `COMPLETED`。

冻结基线缺少
`lunar_pure_planner_core/global_goal_feasibility.hpp`，因此受影响 C++ translation units
使用过仅位于 `/tmp` 的 test-only 兼容头重建；该头保留目标 known-free 与 occupied inflation
语义，测试后已删除，不能视为生产 global-feasibility 构建证明。无垫片的完整原生 package build
仍为 **BLOCKED_BY_BASELINE**。

## 权威与接口审计

规划器生产目录相对上述基线零 diff，三个哨兵 SHA256 未变：

- `message_conversion.cpp`:
  `397be394bcf8c460dc2654bc2ef45c7181914461007367e338da2395f862a160`
- `accepted_goal_finalizer.hpp`:
  `bcd4dd1be9e56d6495b63171651b8449994d292d6451f3be960fd0fa72235d63`
- `pure_plan_motion_server.cpp`:
  `ddad903a6838a61e3018ce8a5e98173276de6971baf9a87702fe94970f44dd0a`

外部 schema/config 未变：

- `PureExplorationTask.msg`:
  `7ac65363314c6e61800293820313e01dc250122e8c308d12e2bf5a2d3172bd74`
- `PureExplorationStatus.msg`:
  `c06e8a75e2da249d76a58a8e03a2100a38fa4ccefa36b157ed4d8a5c0eca1a9f`
- `PlanMotion.action`:
  `36e8077bb384deb15ee77a3c0768d1c1ccb3ac52a7db765db90a21e79ffdd55c`
- `config/external_interfaces.yaml`:
  `419e69aa7fbcb09fff499c29617d2c7b0fe0edb3ec1065fa08bcb3a1c0af64d8`

## NOT_RUN

- ROS 2 Humble 原生完整构建；
- Jetson Orin 构建、DDS/QoS、真实 T3 输入与 Action 联调；
- 控制器、rosbag、RViz、车辆执行和实车；
- 真实 300 m 闭环运行与性能。
