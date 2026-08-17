# Ground Observation Throughput V2 Implementation Plan

**Goal:** 将整段地面轨迹作为一次原子观测事务，跨路径延迟 age 写入，增量更新滚动 planner map，并保留现有不可变 checkpoint alias 恢复合同。

**Scope:** 仅训练环境、native visibility bridge、相关测试和 V2 规格。训练源码与运行中的 worker 不被修改；构建、性能和资格产物写到仓库外。

## Task 1 — Freeze equivalence baselines and native batch contract

**Files**

- Modify: `ros2_ws/src/lunar_planner_training_bridge/include/lunar_planner_training_bridge/visibility.hpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/visibility.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/python/lunar_planner_training_bridge/__init__.py`
- Test: `ros2_ws/src/lunar_planner_training_bridge/test/visibility_test.cpp`
- Test: `ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py`

1. 写 RED：缺失 `reveal_from_poses`；断言 batch 的每个 mask 与逐个 `reveal_from_pose` 位相同，非法形状/类型 fail-closed。
2. 实现 C++ batch 核心，复用同一 ray/visibility 内核；单 pose API 不变。
3. pybind 在 C-contiguous NumPy 输入上释放 GIL，返回 bool 三维数组；Python facade 保留单 pose 和增加 batch API。
4. focused C++/bridge GREEN，记录旧单 pose 性能和 batch 性能但不将性能作为语义测试。

## Task 2 — Path window composer and COW trajectory transaction

**Files**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/multires_observation.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/visibility.py`
- Test: `training/lunar_policy_training/tests/test_multires_observation.py`
- Test: `training/lunar_policy_training/tests/test_sensor_closed_loop.py`

1. 写 RED：任意相邻和跨 fixed-tile 边界的滑动 64m window 与 `read_window` 全字段 bit-exact；同一路径非连续重复 pose 的 native reveal 只准备一次但按原事件数增加 observation/evidence。
2. 引入路径 scoped `DetailWindowComposer`，从固定 tile 拼接精确 `ProjectedScene`；不使用近似采样或半分辨率 cache。
3. 引入 `observe_ground_trajectory`，旧 `observe_world_path` 转发。状态 staging 用浅 dict + tile copy-on-write，覆盖/粗图仅触及时复制。
4. 先准备批量 native visibility，再按输入顺序写入 staging；任一失败抛出时状态 bytes 完全不变。
5. controller 只调用新 API 一次，generic doubles 继续兼容路径回退。

## Task 3 — Cross-path lazy age and materialization boundaries

**Files**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/multires_observation.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Test: `training/lunar_policy_training/tests/test_multires_observation.py`
- Test: `training/lunar_policy_training/tests/test_formal_builder.py`
- Test: `training/lunar_policy_training/tests/test_checkpoint.py`

1. 写 RED：两条路径之间的远端 tile 不被写入；随后 planning window/`physical_evidence_sha256`/checkpoint materialization 与 eager reference 逐字节一致。
2. 持久保存 ordered ledger、全局 cursor、per-tile cursor；必要时物化，不在路径末尾扫描全 tile。
3. 增加 rolling-only chain identity；最终 hash 和 checkpoint 前调用完整 materialize，不改 hash 语义。
4. 断言 checkpoint round-trip 后 lazy age ledger/cursor 仍能得到同一最终 state 与 SHA。

## Task 4 — Dirty-tile incremental planner map

**Files**

- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/python/lunar_planner_training_bridge/__init__.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Test: `ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py`
- Test: `training/lunar_policy_training/tests/test_planner_grid_projection.py`
- Test: `training/lunar_policy_training/tests/test_formal_builder.py`

1. 写 RED：bridge `GridLayer`/`GridMap` 缺少限定 flat-index patch；非法 index、重复 index、dtype/shape 错误 fail-closed。
2. 实现 deterministic patch（不依赖 hash 迭代次序），并保留完整 layer construction API。
3. `FormalEpisode` 缓存 planner-global map 和 source-to-target inverse index；滚动边界仅应用 dirty coarse cells。
4. 对冻结滚动局部请求，逐层 map 值、bridge planner output、reason、trajectory、coverage 和 diagnostics 与完整重建基准相同；候选边界强制完整物化。

## Task 5 — Checkpoint alias regression, focused qualification and integration

**Files**

- Modify: `training/lunar_policy_training/tests/test_checkpoint.py`（仅当需要补回归覆盖）
- Modify: `docs/superpowers/specs/2026-08-17-ground-observation-throughput-v2-design.md`
- Modify: `docs/superpowers/plans/2026-08-17-ground-observation-throughput-v2.md`

1. 复跑 hardlink alias inode/payload 回归；不改现有 production implementation，除非测试暴露实际回归。
2. 运行 native bridge、multires、closed-loop、planner projection、checkpoint 的 focused suites，并运行 Python compile / `git diff --check`。
3. 对同一个外部完整 checkpoint 和冻结 WHEELED/LEGGED 代表任务，记录旧/新请求身份、coverage、reward、terminal、planner 输出和宏动作墙钟。只有语义完全一致且外部性能证据达标，才在完整 update 边界切换训练源码。
4. 只暂存本任务 source/test/docs 文件；生成 build/install/log/cache/checkpoint/benchmark 文件保持在仓库外。
