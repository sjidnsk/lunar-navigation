# 0.2 m Detail Candidate Gain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用机器人当前已掌握的 `0.2 m` 稀疏观测状态计算候选预计信息增益，不读取未观测真值，也不改变实际覆盖奖励。

**Architecture:** `MultiresSensorObservationState` 按候选集合拼接最小必要的高分辨率已观测窗口，将候选坐标映射到该窗口后一次批量调用原生可见性核。候选增益以 `0.04 m2` 精细单元面积换算回现有归一化特征单位；原生核先检查射线倒数第二格，避免对不可能成为第一个未知端点的单元遍历整条射线。

**Tech Stack:** Python 3.10、NumPy、pybind11、C++20、pytest、GoogleTest。

## Global Constraints

- 只读取已观测 `valid_mask` 与对应已观测物理障碍；未知障碍值保持零且由 mask 决定不可穿越。
- 传感器保持 `30 m / 360 deg`，候选上限保持 `64`，reward 仍由执行后的真实新增覆盖产生。
- 不改 PPO、平台能力、候选落点或规划器合同。
- 当前正式训练在代码和验证完成前继续运行；切换使用现有受审计 source migration，并精确保留 checkpoint 中尚未执行的一次旧候选边界。

---

### Task 1: 构造精细候选增益输入

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/multires_observation.py`
- Test: `training/lunar_policy_training/tests/test_multires_observation.py`

**Interfaces:**
- Consumes: `estimate_candidate_gains(observed_mask, obstacle_ratio, roi_ratio, priority_weight, candidate_cells)` 的既有候选估算合同。
- Produces: `MultiresSensorObservationState.estimate_candidate_gains(...) -> np.ndarray[N,2]`。

- [ ] 写失败测试，证明增益估算器收到 `0.2 m` 已观测窗口、未知障碍为零且结果按 `0.04 / 16.0` 面积比例换算。
- [ ] 运行定向测试，确认因方法缺失而失败。
- [ ] 实现最小稀疏窗口拼接、坐标映射与一次批量原生调用。
- [ ] 运行定向测试，确认通过。

### Task 2: 正式候选生成切换到精细增益

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Test: `training/lunar_policy_training/tests/test_formal_builder.py`

**Interfaces:**
- Consumes: `MultiresSensorObservationState.estimate_candidate_gains(...)`。
- Produces: `FormalEpisode` 的 `CandidateBuilderV2` 使用精细已观测状态估算增益。

- [ ] 写失败测试，证明正式 episode 的候选估算器分辨率是 `0.2 m`。
- [ ] 运行测试并确认当前 `4.0 m` 路径导致失败。
- [ ] 将正式候选生成器连接到多分辨率观测状态。
- [ ] 运行正式 builder 与恢复测试。

### Task 3: 保持原生核满足性能边界

**Files:**
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/visibility.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/test/visibility_benchmark.cpp`
- Modify: `training/lunar_policy_training/lunar_policy_training/sensor_performance.py`
- Test: `ros2_ws/src/lunar_planner_training_bridge/test/visibility_test.cpp`
- Test: `training/lunar_policy_training/tests/test_sensor_performance.py`

**Interfaces:**
- Consumes: observed-only 第一未知端点语义。
- Produces: 精细候选批量基准与不改变数值结果的快速拒绝路径。

- [ ] 将正式候选基准改为能覆盖候选联合窗口的 `600 x 600 @ 0.2 m`、64 candidates。
- [ ] 运行旧内核基准并记录超限证据。
- [ ] 在遍历射线前检查倒数第二格是否已观测且无障碍。
- [ ] 运行 C++ 等价测试和 Release 基准。

### Task 4: 来源身份与完整验证

**Files:**
- Modify: `docs/superpowers/specs/2026-08-08-sensor-observation-capability-design.md`
- Modify: `docs/superpowers/specs/2026-08-08-formal-polar-training-environment-closure-design.md`

**Interfaces:**
- Consumes: 已验证的新候选增益语义。
- Produces: 新 source commit 和传感器性能报告；旧 checkpoint 只通过已有受审计迁移入口续训。

- [ ] 保持 `30 m / 360 deg`、observed-only、reward 和 observation boundary 语义哈希不变，同步两份冻结设计中的候选增益精度修复。
- [ ] 运行 Python 定向/回归测试、C++ 测试、边界检查和差异检查。
- [ ] 只提交本任务文件；在停止当前训练前报告 checkpoint 切换约束。
