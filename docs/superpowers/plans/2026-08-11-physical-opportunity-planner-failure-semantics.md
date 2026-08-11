# Physical Opportunity and Planner Failure Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将正式训练的覆盖分母、候选和 oracle 恢复为三平台物理机会语义；将地面局部物理地图与固定 `2.0 m` 额外 margin 的搜索域彻底分离；把规划失败改为当前物理快照内、基于稳定候选 ID 的临时抑制，并以新的刷新和终止矩阵替代 `PLANNER_REJECTED_ALL`。

**Architecture:** C++ 继续拥有物理安全投影、地面局部搜索和 Hopper 认证；`LocalPlanningProblem` 同时携带未修改的 observed local map 与独立 `LocalSearchDomain`。Python 以 C++ `ProjectReachability`/`PlatformCandidateReachability` 为物理连通权威，先生成完整有界 `PhysicalCandidateUniverse`，再按当前 `physical_snapshot_id` 的失败集合补位并截断为 64 个策略动作。`FormalEpisode` 负责物理快照、候选 materialization 和失败后重建，`ObservationBoundaryController` 负责零 reveal、零时间增量的新 observation revision，`V3ExplorationEnvironment` 只消费结构化 `CandidateDisposition` 并执行唯一的终止矩阵。

**Tech Stack:** C++20, ROS 2 Humble, GoogleTest, pybind11, Python 3.10, NumPy, PyTorch, pytest, canonical JSON/NPZ formal cache, SHA-256 identities.

## Global Constraints

- 权威设计为 [2026-08-11-physical-opportunity-planner-failure-semantics-design.md](../specs/2026-08-11-physical-opportunity-planner-failure-semantics-design.md)；若计划与设计冲突，以设计为准并先修订计划，不能在实现中自行改变冻结语义。
- 只在 `/mnt/data/WS/.lunar-navigation-worktrees/formal-training-environment-closure` 的 `feature/formal-training-environment-closure` 上工作；每个任务开始前执行 `git status --short --branch`，保留所有无关用户改动。
- 所有编辑使用 `apply_patch`；中文文件保持 UTF-8，完成后显式以 UTF-8 读取。
- build/install/log、cache、checkpoint、closed-loop report、历史回放和 benchmark 均写入 `/home/kai/CodexDownloads/lunar_navigation/physical_opportunity_failure_semantics`，不得写入仓库。
- 执行 ROS 命令前运行：

  ```bash
  set +u
  source /opt/ros/humble/setup.bash
  set -u
  test "$ROS_DISTRO" = humble
  ```

- Python 测试统一使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`；原生最终验证使用 `--merge-install`、Release 和仓库外 build root。
- 每个行为变更遵循 RED -> 观察预期失败 -> 最小 GREEN -> focused regression -> commit；不得把多个任务堆成一个不可审阅提交。
- 中间提交可以暂时只完成一层，但不得合入 `integration`，直到 C++ domain、物理机会、disposition、刷新、终止、cache/checkpoint/gate 全部同时完成。
- 不改变 PPO 七输入名称、`[64,12]` frontier feature、64 动作容量、奖励权重、`1024 m` 场景、`30 m/360°` sensor、`0.2 m` reveal 或 `0.95` 成功阈值。
- 不实现动态扩宽、`0.4 -> 4.0 m` 恢复、永久黑名单、oracle 候选注入、环境自动替策略选下一个候选或旧状态兼容续训。
- 正式训练保持停止；全部门禁通过只表示可重新冻结，不授权启动训练。

### Implementation preflight

- [ ] **Run once before Task 1; stop on repository/OS/ROS mismatch and record hardware limits**

  ```bash
  export PHYSICAL_PREFLIGHT_ROOT=/home/kai/CodexDownloads/lunar_navigation/physical_opportunity_failure_semantics/preflight
  mkdir -p "$PHYSICAL_PREFLIGHT_ROOT"
  {
    date --iso-8601=seconds
    lsb_release -ds
    uname -m
    lscpu | rg '^(Architecture|CPU\(s\)|Model name):'
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 || true
    gcc --version | sed -n '1p'
    python3 --version
  } | tee "$PHYSICAL_PREFLIGHT_ROOT/host-baseline.txt"
  set +u
  source /opt/ros/humble/setup.bash
  set -u
  test "$(lsb_release -rs)" = 22.04
  test "$(uname -m)" = x86_64
  test "$ROS_DISTRO" = humble
  test "$(git branch --show-current)" = feature/formal-training-environment-closure
  git merge-base --is-ancestor b4d2ea42ddfa53aa3beddf725efafa80174bf5f8 HEAD
  git status --short --branch
  ```

  Expected: Ubuntu `22.04`、`x86_64`、ROS 2 Humble；设计 commit 是当前 HEAD 的 ancestor，且进入 Task 1 前工作树 clean。CPU/GPU/driver 结果必须原样记录；非 RTX 4080 SUPER 或 `nvidia-smi` 不可用不阻止 source implementation，但禁止把后续结果写成 RTX 正式训练资格，直到在指定主机重跑相关门禁。

## File Map

### C++ planner core and ROS surface

- `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_config.hpp`：把地面额外走廊 margin 默认值冻结为 `2.0`。
- `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp`：增加 `CandidateDisposition` 和地面 domain/物理目标 diagnostics。
- `ros2_ws/src/lunar_planner_core/src/hierarchical/local_planning_problem.hpp`：定义不可变 `LocalSearchDomain`，并让 local problem 分别携带原始地图、domain、route prefix 和 frontier identity。
- `ros2_ws/src/lunar_planner_core/src/hierarchical/local_frontier.cpp`：以纯 mask 取代 `BuildLocalView()` 的地图改写；固定校验 `2.0`。
- `ros2_ws/src/lunar_planner_core/src/hierarchical/local_frontier.hpp`：暴露 domain diagnostics。
- `ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.{hpp,cpp}`、`wheel_planner.cpp`、`wheel_sweep_validator.{hpp,cpp}`：状态中心/连接器中心受 domain 限制，footprint 仍读原始物理投影。
- `ros2_ws/src/lunar_planner_core/src/legged/legged_lattice.{hpp,cpp}`、`legged_planner.cpp`、`legged_terrain.{hpp,cpp}`：实现与 wheel 相同的 map/domain 分离合同。
- `ros2_ws/src/lunar_planner_core/src/planner.cpp`：聚合 domain diagnostics，并只在完整正常目标级失败后给出 snapshot suppression disposition。
- `ros2_ws/src/lunar_planner_ros/src/message_conversion.cpp`、`plan_motion_server.cpp`：验证并发布新增 diagnostics，不改变外部 motion reference schema。
- C++ tests：`hierarchical_types_test.cpp`、`local_frontier_test.cpp`、`wheel_planner_test.cpp`、`legged_planner_test.cpp`、`hierarchical_planner_test.cpp`、三平台 fault matrix 和 ROS server tests。

### Training bridge

- `ros2_ws/src/lunar_planner_training_bridge/src/conversions.cpp`：bridge hard failure 固定返回 `KEEP`。
- `ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp`：绑定 `CandidateDisposition`、`HierarchicalPlannerMetrics` 和新增字段。
- `ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py`：验证枚举名字、默认值、失败矩阵和 diagnostics。

### Physical opportunity, candidate and oracle

- `training/lunar_policy_training/lunar_policy_training/environment/coverability.py`：把 `PlatformCoverability` 从 primitive graph identity 改成 physical projection identity。
- `training/lunar_policy_training/lunar_policy_training/environment/platform_reachability.py`：返回完整 observed-only 物理安全/连通 mask、algorithm ID 和 canonical hash；候选与 oracle 复用物理定义而不共享排序。
- `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`：定义稳定候选、完整 universe、reserve/top-64 selection 和新 diagnostics。
- `training/lunar_policy_training/lunar_policy_training/environment/frontier_oracle.py`：独立枚举 observed-only 物理机会，不再读 primitive graph。
- `training/lunar_policy_training/lunar_policy_training/environment/multires_observation.py`：维护只随真实 sensor evidence 变化的 generation/hash。
- `training/lunar_policy_training/lunar_policy_training/environment/formal_start_qualification.py`：以物理 candidate universe 判定初始可行动作。
- `training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py`：生成并严格加载 cache v6。

### Environment, replay and gates

- `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`：移除 primitive candidate 主路径，生成 physical snapshot，持有失败集合并实现 failure refresh。
- `training/lunar_policy_training/lunar_policy_training/environment/observation_boundary.py`：增加零 reveal、零时间推进的权威 rebuild。
- `training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py`：消费 disposition、触发 refresh、执行新终止矩阵。
- `training/lunar_policy_training/lunar_policy_training/environment/macro_step.py`：删除 `PLANNER_REJECTED_ALL`，增加 `PLANNER_BLOCKED_WITH_OPPORTUNITY`。
- `training/lunar_policy_training/lunar_policy_training/environment/formal_episode_state.py`：formal state v6，持久化 snapshot/universe/failure/evidence 身份。
- `training/lunar_policy_training/lunar_policy_training/checkpoint.py`：checkpoint v7，拒绝旧 resume。
- `training/lunar_policy_training/lunar_policy_training/training_semantics.py`：training semantics v11。
- `training/lunar_policy_training/lunar_policy_training/closed_loop_gate.py`、`training_metrics.py`、`formal_preflight.py`、`parallel_pool.py`、`cli.py`：迁移报告、指标和入口门禁。
- Python tests：`test_coverability.py`、`test_candidate_builder_v2.py`、`test_frontier_oracle.py`、`test_formal_start_qualification.py`、`test_formal_cache.py`、`test_formal_builder.py`、`test_sensor_closed_loop.py`、`test_v3_environment.py`、`test_formal_resume_state.py`、`test_checkpoint_resume.py`、`test_closed_loop_gate.py`、`test_training_metrics.py`、`test_parallel_pool.py`、`test_formal_preflight.py` 和 `test_cli.py`。

---

### Task 1: Freeze the `2.0 m` contract and separate local map from search domain

**Files:**

- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_config.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/local_planning_problem.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/local_frontier.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hierarchical_types_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/local_frontier_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/wheel_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/legged_planner_test.cpp`

**Contract:**

```cpp
inline constexpr double kFixedAdditionalCorridorMarginM = 2.0;

class LocalSearchDomain final {
 public:
  LocalSearchDomain(
      std::size_t width,
      std::size_t height,
      std::vector<std::uint8_t> allowed);
  [[nodiscard]] bool Contains(shared::GridCell cell) const noexcept;
  [[nodiscard]] std::size_t width() const noexcept;
  [[nodiscard]] std::size_t height() const noexcept;
  [[nodiscard]] std::span<const std::uint8_t> allowed() const noexcept;
  [[nodiscard]] std::size_t allowed_cell_count() const noexcept;
  [[nodiscard]] const std::string& sha256() const noexcept;

 private:
  std::size_t width_{};
  std::size_t height_{};
  std::vector<std::uint8_t> allowed_;
  std::size_t allowed_cell_count_{};
  std::string sha256_;
};

struct LocalPlanningProblem final {
  // existing request/state fields
  GridMap local_map;                         // original observed layers
  LocalSearchDomain search_domain;           // center-search permission only
  std::vector<Vec3> route_prefix_odom;
  std::size_t frontier_attempt_index{};
  double frontier_distance_m{};
  // capability/config/previous_execution/stop_token
};
```

`LocalSearchDomain` 构造时校验 `allowed.size() == width * height` 且每个 byte 只为 `0/1`，随后一次性计算只读字段；`search_domain.sha256()` 固定为 `SHA256(width_be64 || height_be64 || row_major_allowed_bytes)`。`BuildLocalFrontiers()` 不再创建或返回裁剪地图。

- [ ] **Step 1: 写 RED tests**

  将 `MasksEveryCellOutsideTheHorizonCorridorIntersection` 改成 `PreservesEveryLocalMapLayerAndBuildsIndependentSearchDomain`，逐 layer 比较输入 `GridMap` 与每个 problem 的 `local_map`，再逐 cell 验证 domain 等于 route-prefix corridor 与 horizon 的交集。增加：默认 margin 精确为 `2.0`；`0.4`、`0.8`、`4.0`、NaN 和 infinity 均返回 `LOCAL_FRONTIER_CONFIGURATION_INVALID`；wheel/legged corridor half width 等于 support radius + clearance + `2.0`。

- [ ] **Step 2: 运行 RED**

  ```bash
  export PHYSICAL_NATIVE_ROOT=/home/kai/CodexDownloads/lunar_navigation/physical_opportunity_failure_semantics/task1-native
  colcon --log-base "$PHYSICAL_NATIVE_ROOT/log-build" build --merge-install \
    --base-paths ros2_ws/src --packages-select lunar_planner_core \
    --build-base "$PHYSICAL_NATIVE_ROOT/build" \
    --install-base "$PHYSICAL_NATIVE_ROOT/install" \
    --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
  set +u
  source "$PHYSICAL_NATIVE_ROOT/install/setup.bash"
  set -u
  colcon --log-base "$PHYSICAL_NATIVE_ROOT/log-test" test \
    --packages-select lunar_planner_core \
    --build-base "$PHYSICAL_NATIVE_ROOT/build" \
    --test-result-base "$PHYSICAL_NATIVE_ROOT/build" \
    --event-handlers console_direct+ \
    --ctest-args -R 'hierarchical_types|local_frontier'
  ```

  Expected: 默认值仍为 `0.4`，且 local map layer 被改写，新增 domain 字段不存在。

- [ ] **Step 3: 最小实现**

  用 `BuildSearchDomain()` 取代 `BuildLocalView()`；hash 使用现有 `shared/sha256.hpp`。`ValidPlatformGeometry()` 必须用 `Close(config.additional_corridor_margin_m, kFixedAdditionalCorridorMarginM)`，而不是非负判断。临时把 wheel/legged 对 `local_map_view` 的读取机械改为 `local_map` 以保持编译；本任务不宣称搜索域已经被 backend 执行。

- [ ] **Step 4: GREEN、静态扫描和提交**

  ```bash
  colcon --log-base "$PHYSICAL_NATIVE_ROOT/log-green-build" build --merge-install \
    --base-paths ros2_ws/src --packages-select lunar_planner_core \
    --build-base "$PHYSICAL_NATIVE_ROOT/build" \
    --install-base "$PHYSICAL_NATIVE_ROOT/install" \
    --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
  set +u
  source "$PHYSICAL_NATIVE_ROOT/install/setup.bash"
  set -u
  colcon --log-base "$PHYSICAL_NATIVE_ROOT/log-green" test \
    --packages-select lunar_planner_core \
    --build-base "$PHYSICAL_NATIVE_ROOT/build" \
    --test-result-base "$PHYSICAL_NATIVE_ROOT/build" \
    --event-handlers console_direct+ \
    --ctest-args -R 'hierarchical_types|local_frontier'
  colcon test-result --test-result-base "$PHYSICAL_NATIVE_ROOT/build" --verbose
  ! rg -n 'BuildLocalView|local_map_view|additional_corridor_margin_m\{0\.4\}' \
    ros2_ws/src/lunar_planner_core
  git add ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_config.hpp \
    ros2_ws/src/lunar_planner_core/src/hierarchical/local_planning_problem.hpp \
    ros2_ws/src/lunar_planner_core/src/hierarchical/local_frontier.hpp \
    ros2_ws/src/lunar_planner_core/src/hierarchical/local_frontier.cpp \
    ros2_ws/src/lunar_planner_core/src/wheel/wheel_planner.cpp \
    ros2_ws/src/lunar_planner_core/src/legged/legged_planner.cpp \
    ros2_ws/src/lunar_planner_core/test/hierarchical_types_test.cpp \
    ros2_ws/src/lunar_planner_core/test/local_frontier_test.cpp \
    ros2_ws/src/lunar_planner_core/test/wheel_planner_test.cpp \
    ros2_ws/src/lunar_planner_core/test/legged_planner_test.cpp
  git commit -m "refactor(planner): separate local map and search domain"
  ```

  Expected: focused tests pass；`rg` 不再命中旧地图裁剪函数或 `0.4` 默认值。

---

### Task 2: Enforce the search domain in the wheel backend without clipping footprint physics

**Files:**

- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_sweep_validator.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/wheel_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/wheel_fault_matrix_test.cpp`

**Interfaces:**

```cpp
WheelSweepValidator(
    const shared::SafeProjection& physical_projection,
    const WheeledCapability& capability,
    const hierarchical::LocalSearchDomain& search_domain) noexcept;

WheelLatticeSearchResult SearchWheelLatticeRanked(
    const WheeledState& current_state,
    std::span<const GoalRegion> ranked_goals,
    const shared::SafeProjection& physical_projection,
    const hierarchical::LocalSearchDomain& search_domain,
    const WheeledCapability& capability,
    const PlannerConfig& config,
    std::stop_token stop_token);
```

Sweep 的中心采样必须在 domain 内，footprint/support/clearance 采样只查 `physical_projection`。目标 physical feasibility 先在原始 projection 上判断；只有之后的有限 domain 搜索失败才使用 `LOCAL_SEARCH_DOMAIN_EXHAUSTED`。

- [ ] **Step 1: 写四个 RED tests**

  1. 目标物理安全、车体 footprint 越出 domain 但仍落在原始安全地图时成功；
  2. primitive 目标中心越出 domain 时拒绝；
  3. 物理目标真实不安全时在展开前仍为 `WHEEL_GOAL_INFEASIBLE`、expanded=0；
  4. 物理目标安全但 domain 内无路时为 `LOCAL_SEARCH_DOMAIN_EXHAUSTED`，不能伪装为 goal infeasible。

  将历史窗口边界几何压缩成 `WheelPlanner.SearchDomainBoundaryDoesNotBecomePhysicalObstacle`：窄 domain 紧贴车体侧缘，域外地图保持安全；旧实现应在 projection 阶段失败，新实现必须至少展开一个状态并成功。

- [ ] **Step 2: 运行 RED**

  ```bash
  export PHYSICAL_NATIVE_ROOT=/home/kai/CodexDownloads/lunar_navigation/physical_opportunity_failure_semantics/task1-native
  colcon --log-base "$PHYSICAL_NATIVE_ROOT/log-wheel-red-build" build --merge-install \
    --base-paths ros2_ws/src --packages-select lunar_planner_core \
    --build-base "$PHYSICAL_NATIVE_ROOT/build" \
    --install-base "$PHYSICAL_NATIVE_ROOT/install" \
    --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
  set +u
  source "$PHYSICAL_NATIVE_ROOT/install/setup.bash"
  set -u
  colcon --log-base "$PHYSICAL_NATIVE_ROOT/log-wheel-red" test \
    --packages-select lunar_planner_core \
    --build-base "$PHYSICAL_NATIVE_ROOT/build" \
    --test-result-base "$PHYSICAL_NATIVE_ROOT/build" \
    --event-handlers console_direct+ \
    --ctest-args -R 'wheel_planner|wheel_fault_matrix'
  ```

  Expected: wheel lattice 尚未接收 domain，或 footprint/domain 边界仍被混同。

- [ ] **Step 3: 实现 domain-aware center validation**

  在 sweep 的每个连续中心样本上先调用 `search_domain.Contains(map.PositionToCell(center))`；随后照常对完整 footprint 做原始 projection 校验。`ExactGoalConnector()`、lazy primitive expansion、stop/switch/spin 和 final discrete/smoothed validation 使用同一个 validator，不允许 connector 绕过 domain。不要把 domain 纳入 projection cache key 或 map generation。

- [ ] **Step 4: GREEN and commit**

  ```bash
  colcon --log-base "$PHYSICAL_NATIVE_ROOT/log-wheel-green-build" build --merge-install \
    --base-paths ros2_ws/src --packages-select lunar_planner_core \
    --build-base "$PHYSICAL_NATIVE_ROOT/build" \
    --install-base "$PHYSICAL_NATIVE_ROOT/install" \
    --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
  set +u
  source "$PHYSICAL_NATIVE_ROOT/install/setup.bash"
  set -u
  colcon --log-base "$PHYSICAL_NATIVE_ROOT/log-wheel-green" test \
    --packages-select lunar_planner_core \
    --build-base "$PHYSICAL_NATIVE_ROOT/build" \
    --test-result-base "$PHYSICAL_NATIVE_ROOT/build" \
    --event-handlers console_direct+ \
    --ctest-args -R 'wheel_planner|wheel_fault_matrix|local_frontier'
  colcon test-result --test-result-base "$PHYSICAL_NATIVE_ROOT/build" --verbose
  git add ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.hpp \
    ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.cpp \
    ros2_ws/src/lunar_planner_core/src/wheel/wheel_sweep_validator.hpp \
    ros2_ws/src/lunar_planner_core/src/wheel/wheel_sweep_validator.cpp \
    ros2_ws/src/lunar_planner_core/src/wheel/wheel_planner.cpp \
    ros2_ws/src/lunar_planner_core/test/wheel_planner_test.cpp \
    ros2_ws/src/lunar_planner_core/test/wheel_fault_matrix_test.cpp
  git commit -m "fix(planner): constrain wheel centers to local search domain"
  ```

---

### Task 3: Apply the same physical-map/search-domain contract to legged planning and hierarchy diagnostics

**Files:**

- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_lattice.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_terrain.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/legged_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/legged_fault_matrix_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hierarchical_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hierarchical_regression_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/message_conversion.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp`

**Diagnostics contract:**

```cpp
struct HierarchicalPlannerMetrics final {
  // existing global/local timing and expansion fields
  double additional_corridor_margin_m{};
  double corridor_half_width_m{};
  std::size_t search_domain_cell_count{};
  std::string search_domain_sha256;
  std::size_t local_frontier_attempts{};
  std::size_t local_search_runs{};
  std::size_t global_replans{};
  bool physical_goal_feasible{};
};
```

保留旧 `local_attempts`/`corridor_width_m` 一个迁移周期仅用于 ROS 兼容展示时，必须从新字段派生；训练 bridge 和新测试只使用新名字。

- [ ] **Step 1: 写 RED tests**

  为 legged 复制 Task 2 的 footprint-outside-domain positive、center-outside-domain negative、真实 body/foot support goal infeasible、domain exhausted 四类测试。层级测试断言固定 margin、half width、最后一次实际 search domain hash/cell count、frontier attempts、replans 和 physical goal flag；重复请求 hash 必须一致。

- [ ] **Step 2: 运行 RED**

  ```bash
  export PHYSICAL_NATIVE_ROOT=/home/kai/CodexDownloads/lunar_navigation/physical_opportunity_failure_semantics/task1-native
  colcon --log-base "$PHYSICAL_NATIVE_ROOT/log-legged-red-build" build --merge-install \
    --base-paths ros2_ws/src \
    --packages-select lunar_planner_core lunar_planner_ros \
    --build-base "$PHYSICAL_NATIVE_ROOT/build" \
    --install-base "$PHYSICAL_NATIVE_ROOT/install" \
    --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
  set +u
  source "$PHYSICAL_NATIVE_ROOT/install/setup.bash"
  set -u
  colcon --log-base "$PHYSICAL_NATIVE_ROOT/log-legged-red" test \
    --packages-select lunar_planner_core lunar_planner_ros \
    --build-base "$PHYSICAL_NATIVE_ROOT/build" \
    --test-result-base "$PHYSICAL_NATIVE_ROOT/build" \
    --event-handlers console_direct+ \
    --ctest-args -R 'legged_planner|legged_fault_matrix|hierarchical_planner|plan_motion_server'
  ```

  Expected: build 或 focused test 因 legged domain 参数/新 diagnostics 尚不存在而失败；不能接受与新增断言无关的编译器、ROS 环境或 fixture 错误。

- [ ] **Step 3: 实现和聚合**

  `ValidateLeggedBodySweep()` 接收 domain，只限制 body center path；足端、支撑多边形、机身矩形和净空继续使用原始 projection。`BuildLeggedLattice()` 的 state/edge/goal connector 全部传入 domain。`Planner::Plan()` 在每次 local frontier 尝试时记录该 problem 的 domain，最终 diagnostics 取实际成功 problem，或全失败时取最后一次实际搜索 problem；无 local search 时 cell count=0、hash 为空、physical goal flag=false。

- [ ] **Step 4: GREEN, ROS diagnostic validation and commit**

  ```bash
  colcon --log-base "$PHYSICAL_NATIVE_ROOT/log-legged-green-build" build --merge-install \
    --base-paths ros2_ws/src \
    --packages-select lunar_planner_core lunar_planner_ros \
    --build-base "$PHYSICAL_NATIVE_ROOT/build" \
    --install-base "$PHYSICAL_NATIVE_ROOT/install" \
    --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
  set +u
  source "$PHYSICAL_NATIVE_ROOT/install/setup.bash"
  set -u
  colcon --log-base "$PHYSICAL_NATIVE_ROOT/log-legged-green" test \
    --packages-select lunar_planner_core lunar_planner_ros \
    --build-base "$PHYSICAL_NATIVE_ROOT/build" \
    --test-result-base "$PHYSICAL_NATIVE_ROOT/build" \
    --event-handlers console_direct+ \
    --ctest-args -R 'legged|hierarchical|plan_motion_server'
  colcon test-result --test-result-base "$PHYSICAL_NATIVE_ROOT/build" --verbose
  git add ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp \
    ros2_ws/src/lunar_planner_core/src/legged/legged_lattice.hpp \
    ros2_ws/src/lunar_planner_core/src/legged/legged_lattice.cpp \
    ros2_ws/src/lunar_planner_core/src/legged/legged_terrain.hpp \
    ros2_ws/src/lunar_planner_core/src/legged/legged_terrain.cpp \
    ros2_ws/src/lunar_planner_core/src/legged/legged_planner.cpp \
    ros2_ws/src/lunar_planner_core/src/planner.cpp \
    ros2_ws/src/lunar_planner_core/test/legged_planner_test.cpp \
    ros2_ws/src/lunar_planner_core/test/legged_fault_matrix_test.cpp \
    ros2_ws/src/lunar_planner_core/test/hierarchical_planner_test.cpp \
    ros2_ws/src/lunar_planner_core/test/hierarchical_regression_test.cpp \
    ros2_ws/src/lunar_planner_ros/src/message_conversion.cpp \
    ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp \
    ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp
  git commit -m "feat(planner): audit fixed ground search domains"
  ```

  Expected: focused C++/ROS tests pass，domain/physical failure cases 分类准确，新增 diagnostics 通过 finite/range validation。

---

### Task 4: Add structured `CandidateDisposition` through core and Python bridge

**Files:**

- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hopper_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hierarchical_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hopper_fault_matrix_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/conversions.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py`
- Modify: `ros2_ws/src/lunar_planner_ros/src/message_conversion.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp`

**Public contract:**

```cpp
enum class CandidateDisposition : std::uint8_t {
  kKeep,
  kSuppressForCurrentPhysicalSnapshot,
};

struct PlannerOutput final {
  PlanningOutcome outcome{PlanningOutcome::kInvalidRequest};
  ExecutionDirective directive{ExecutionDirective::kNoSafeReference};
  CandidateDisposition candidate_disposition{CandidateDisposition::kKeep};
  // reason/reference/diagnostics/continuation
};
```

Python names must be exactly `CandidateDisposition.KEEP` and `CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT`。

- [ ] **Step 1: 写完整输出矩阵 RED tests**

  Assert `KEEP` for success, canceled, invalid, stale, numerical, resource exhausted, active reference invalidation, bridge exception and continuing committed hop. Assert suppression only for final public-planner `NO_KNOWN_SAFE_ROUTE`/`GOAL_INFEASIBLE` after normal target attempts. Internal wheel/legged local backend failures remain `KEEP`; the hierarchy upgrades only its final target-level result。ROS test 还必须断言 diagnostic key `candidate_disposition` 精确发布 `KEEP` 或 `SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT`，非法枚举值被 conversion validation 拒绝。

- [ ] **Step 2: 运行 RED build/bridge tests**

  ```bash
  export PHYSICAL_BRIDGE_ROOT=/home/kai/CodexDownloads/lunar_navigation/physical_opportunity_failure_semantics/task4-bridge
  colcon --log-base "$PHYSICAL_BRIDGE_ROOT/log-build" build --merge-install \
    --base-paths ros2_ws/src \
    --packages-select lunar_planner_core lunar_planner_training_bridge lunar_planner_ros \
    --build-base "$PHYSICAL_BRIDGE_ROOT/build" \
    --install-base "$PHYSICAL_BRIDGE_ROOT/install" \
    --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
  set +u
  source "$PHYSICAL_BRIDGE_ROOT/install/setup.bash"
  set -u
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py
  colcon --log-base "$PHYSICAL_BRIDGE_ROOT/log-red-test" test \
    --packages-select lunar_planner_core lunar_planner_training_bridge lunar_planner_ros \
    --build-base "$PHYSICAL_BRIDGE_ROOT/build" \
    --test-result-base "$PHYSICAL_BRIDGE_ROOT/build" \
    --event-handlers console_direct+ \
    --ctest-args -R 'hierarchical_planner|hopper_fault_matrix|plan_motion_server'
  ```

  Expected: 至少一个 C++/bridge/ROS 用例因 `CandidateDisposition` 字段、枚举 binding 或 diagnostic key 尚不存在而失败；bridge 异常或 ROS setup 缺失不算有效 RED。

- [ ] **Step 3: 实现显式 disposition，不做环境侧推断**

  所有 failure helper 接收显式 disposition；不得新增 `DispositionFor(outcome, directive)` 供 Python 重复推断。`Planner::Plan()` 只有在正常规划流程穷尽所选目标后设置 suppression。`BridgeFailure()` 和 catch-all 始终 `KEEP`。绑定完整 `HierarchicalPlannerMetrics`，让 Python 能读取 Task 3 的字段。

- [ ] **Step 4: GREEN and commit**

  ```bash
  export PHYSICAL_BRIDGE_ROOT=/home/kai/CodexDownloads/lunar_navigation/physical_opportunity_failure_semantics/task4-bridge
  colcon --log-base "$PHYSICAL_BRIDGE_ROOT/log-green-build" build --merge-install \
    --base-paths ros2_ws/src \
    --packages-select lunar_planner_core lunar_planner_training_bridge lunar_planner_ros \
    --build-base "$PHYSICAL_BRIDGE_ROOT/build" \
    --install-base "$PHYSICAL_BRIDGE_ROOT/install" \
    --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
  set +u
  source "$PHYSICAL_BRIDGE_ROOT/install/setup.bash"
  set -u
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py
  colcon --log-base "$PHYSICAL_BRIDGE_ROOT/log-test" test \
    --packages-select lunar_planner_core lunar_planner_training_bridge lunar_planner_ros \
    --build-base "$PHYSICAL_BRIDGE_ROOT/build" \
    --test-result-base "$PHYSICAL_BRIDGE_ROOT/build" \
    --event-handlers console_direct+
  colcon test-result --test-result-base "$PHYSICAL_BRIDGE_ROOT/build" --verbose
  git add ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp \
    ros2_ws/src/lunar_planner_core/src/planner.cpp \
    ros2_ws/src/lunar_planner_core/src/hopper/hopper_planner.cpp \
    ros2_ws/src/lunar_planner_core/test/hierarchical_planner_test.cpp \
    ros2_ws/src/lunar_planner_core/test/hopper_fault_matrix_test.cpp \
    ros2_ws/src/lunar_planner_training_bridge/src/conversions.cpp \
    ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp \
    ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py \
    ros2_ws/src/lunar_planner_ros/src/message_conversion.cpp \
    ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp \
    ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp
  git commit -m "feat(planner): publish candidate disposition"
  ```

  Expected: core、bridge 与 ROS tests 全部通过；每个输出路径具有显式 disposition，ROS diagnostic 与 Python enum 使用同一值。

---

### Task 5: Replace primitive coverability identity with exact physical projection and cache v6

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/coverability.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/platform_reachability.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py`
- Modify: `training/lunar_policy_training/tests/test_coverability.py`
- Modify: `training/lunar_policy_training/tests/test_formal_cache.py`
- Modify: `ros2_ws/src/lunar_planner_core/test/reachability_projection_test.cpp`

**Data contract:**

```python
FORMAL_CACHE_SCHEMA = "lunar-formal-training-cache/v6"

@dataclass(frozen=True, slots=True)
class PlatformCoverability:
    platform_type: str
    qualified_start_cell: tuple[int, int] | None
    physical_observation_pose_mask: np.ndarray
    physical_projection_schema: str
    physical_reachability_algorithm_id: str
    physical_safe_pose_count: int
    physically_reachable_pose_count: int
    physical_projection_sha256: str
    mission_target_detail_mask_sha256: str
    coverable_detail_shape: tuple[int, int]
    coverable_detail_bits: np.ndarray
    coverable_detail_cell_count: int
    coverable_detail_mask_sha256: str
    sensor_visibility_algorithm_id: str
    capability_content_sha256: str
    start_identity_sha256: str
    # fractions, eligibility and reason remain
```

删除 `primitive_state_count`、`certified_edge_count`、`recoverable_state_count`、`primitive_state_schema`、`primitive_set_sha256`、`world_evidence_sha256` 和 `reachability_graph_sha256` 作为 coverability/cache eligibility 字段。

- [ ] **Step 1: 写 RED contract and primitive-isolation tests**

  用同一 truth/capability/start 构造两组不同 planner motion primitives，断言 ground 和 Hopper 的 physical pose mask、projection hash、coverable detail bits 完全相同。加入障碍、坡度、净空、wheel support、legged foothold/body-height、Hopper landing/tube/connectivity 负例。严格加载测试必须拒绝 v5 manifest 和任何残留 primitive identity 字段；v5 用例名固定为 `test_formal_cache_rejects_v5_manifest`。

- [ ] **Step 2: 运行 RED**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    training/lunar_policy_training/tests/test_coverability.py \
    training/lunar_policy_training/tests/test_formal_cache.py
  export PHYSICAL_BRIDGE_ROOT=/home/kai/CodexDownloads/lunar_navigation/physical_opportunity_failure_semantics/task4-bridge
  colcon --log-base "$PHYSICAL_BRIDGE_ROOT/log-reachability-red-build" build --merge-install \
    --base-paths ros2_ws/src --packages-select lunar_planner_core \
    --build-base "$PHYSICAL_BRIDGE_ROOT/build" \
    --install-base "$PHYSICAL_BRIDGE_ROOT/install" \
    --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
  set +u
  source "$PHYSICAL_BRIDGE_ROOT/install/setup.bash"
  set -u
  colcon --log-base "$PHYSICAL_BRIDGE_ROOT/log-reachability-red" test \
    --packages-select lunar_planner_core \
    --build-base "$PHYSICAL_BRIDGE_ROOT/build" \
    --test-result-base "$PHYSICAL_BRIDGE_ROOT/build" \
    --event-handlers console_direct+ \
    --ctest-args -R reachability_projection
  ```

  Expected: Python contract test 因仍为 cache v5/primitive identity 失败，或 C++ projection test 因新 physical-isolation 断言失败；两组测试不得因 import/build 环境错误失败。

- [ ] **Step 3: 实现 truth physical projection**

  将 `_build_truth_primitive_reachability()` 替换为 `_build_truth_physical_reachability()`：ground 调用 `bridge.project_reachability(request, 30.0)` 并使用 start connected component；Hopper 使用安全 landing evidence 与认证的连续 capability envelope projection。`build_streamed_detail_coverability()` 的 observation positions 改为 `physical_observation_pose_mask` 中的确定性格心/认证落点。projection canonical hash 包含 platform、algorithm、geometry、mask、capability content 和 start identity，但不包含 primitive set。

- [ ] **Step 4: GREEN and commit**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    training/lunar_policy_training/tests/test_coverability.py \
    training/lunar_policy_training/tests/test_formal_cache.py
  export PHYSICAL_BRIDGE_ROOT=/home/kai/CodexDownloads/lunar_navigation/physical_opportunity_failure_semantics/task4-bridge
  colcon --log-base "$PHYSICAL_BRIDGE_ROOT/log-reachability-green-build" build --merge-install \
    --base-paths ros2_ws/src --packages-select lunar_planner_core \
    --build-base "$PHYSICAL_BRIDGE_ROOT/build" \
    --install-base "$PHYSICAL_BRIDGE_ROOT/install" \
    --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
  set +u
  source "$PHYSICAL_BRIDGE_ROOT/install/setup.bash"
  set -u
  colcon --log-base "$PHYSICAL_BRIDGE_ROOT/log-reachability-green" test \
    --packages-select lunar_planner_core \
    --build-base "$PHYSICAL_BRIDGE_ROOT/build" \
    --test-result-base "$PHYSICAL_BRIDGE_ROOT/build" \
    --event-handlers console_direct+ \
    --ctest-args -R reachability_projection
  colcon test-result --test-result-base "$PHYSICAL_BRIDGE_ROOT/build" --verbose
  git add training/lunar_policy_training/lunar_policy_training/environment/coverability.py \
    training/lunar_policy_training/lunar_policy_training/environment/platform_reachability.py \
    training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py \
    training/lunar_policy_training/tests/test_coverability.py \
    training/lunar_policy_training/tests/test_formal_cache.py \
    ros2_ws/src/lunar_planner_core/test/reachability_projection_test.cpp
  git commit -m "feat(training): define physical coverability cache v6"
  ```

  Expected: Python 和 native focused tests pass；v5/primitive-bound fixture 被严格拒绝，primitive-set 变化不改变 physical coverability identities。

---

### Task 6: Build stable physical candidate universes, snapshot identities and reserve refill

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/multires_observation.py`
- Modify: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`
- Modify: `training/lunar_policy_training/tests/test_multires_observation.py`

**Types and canonical identities:**

```python
CANDIDATE_ID_SCHEMA = "lunar-physical-candidate-id/v1"
PHYSICAL_SNAPSHOT_SCHEMA = "lunar-physical-snapshot/v1"

@dataclass(frozen=True, slots=True)
class PhysicalCandidate:
    candidate_id: str
    position_grid_key: tuple[int, int]
    target_position_m: tuple[float, float, float]
    target_yaw_bin: int
    target_yaw_rad: float
    goal_tolerance_mm: int
    feature: np.ndarray
    rank_key: tuple[object, ...]

@dataclass(frozen=True, slots=True)
class PhysicalCandidateUniverse:
    physical_snapshot_id: str
    physical_reachability_algorithm_id: str
    candidates: tuple[PhysicalCandidate, ...]
    universe_sha256: str
    diagnostics: CandidateDiagnostics

@dataclass(frozen=True, slots=True)
class CandidateBuildResult:
    universe: PhysicalCandidateUniverse
    batch: CandidateBatch
```

Candidate ID canonical stream 固定为 schema、platform、mission revision、row/column、z millimetres（Hopper 为 landing key）、64-bin heading、goal tolerance millimetres。frontier/fallback/reserve source 和数组下标不得进入 ID。Universe hash 是按最终稳定 rank 顺序拼接 candidate IDs 的 SHA-256。

`physical_snapshot_id` 由 platform type/id、capability content hash、mission revision、毫米 pose key、微弧度 yaw key、observed evidence generation 和 evidence hash组成。policy observation revision、map materialization revision 和 top-64 不进入该 hash。

- [ ] **Step 1: 写 RED identity/refill tests**

  覆盖：同一目标通过 frontier/fallback 两条生成路径得到同 ID；无 reveal rebuild 保持 snapshot/candidate/universe ID；只重排候选不改变 ID 集；真实移动或 evidence 更新改变 snapshot，并丢弃旧 snapshot failure set、恢复仍物理合格的同目标；primitive set 改变不影响 universe；65+ 候选时失败 top candidate 后第 65 项补位；4097 个合格位置按相同输入重复构建时都稳定压缩为同一 4096 项 universe/hash；padding ID 必须为空；failed IDs 不属于 universe 时 fail closed。

- [ ] **Step 2: 运行 RED**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    training/lunar_policy_training/tests/test_candidate_builder_v2.py \
    training/lunar_policy_training/tests/test_multires_observation.py
  ```

  Expected: stable ID、snapshot invariance、reserve refill 或 stale failed-ID 用例至少一项失败，且 failure 指向旧 index/primitive candidate contract。

- [ ] **Step 3: 实现 bounded universe 和 CandidateBuilder-owned diagnostics**

  CandidateBuilder 先完整资格过滤，再确定性排序，最后应用 `planner_failed_candidate_ids` 并取 64。Universe 上限固定为 `4096`；超过时按既有空间分段代表 + stable farthest fill 降到 4096，并在 hash 前完成。定义 diagnostics：

  ```text
  physical_snapshot_id
  physical_reachability_algorithm_id
  physical_candidate_universe_count
  selected_policy_candidate_count
  available_candidate_count
  untried_reserve_count
  planner_failed_current_snapshot_count
  zero_gain_count
  visited_excluded_count
  physical_unreachable_count
  ```

  `oracle_opportunity_count`、`oracle_opportunity_set_sha256` 和 `remaining_coverable_detail_cell_count` 不由 CandidateBuilder 伪造：前两项在 Task 7 由独立 oracle 附加，最后一项在 Task 9 仅作为 truth-side audit 附加。移除 `planner_rejected_count` 和所有 primitive graph count consistency。`MultiresSensorObservationState.evidence_generation` 只在 `observe_world()` 实际执行权威 sensor update 时递增；增加 `physical_evidence_sha256()`，按排序 tile identity 和 observed layer bytes 计算，不读取 materialization revision。

- [ ] **Step 4: GREEN, stale-field scan and commit**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    training/lunar_policy_training/tests/test_candidate_builder_v2.py \
    training/lunar_policy_training/tests/test_multires_observation.py
  ! rg -n 'planner_rejected_count|primitive_state_ids|primitive_graph_revision' \
    training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py
  git add training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py \
    training/lunar_policy_training/lunar_policy_training/environment/multires_observation.py \
    training/lunar_policy_training/tests/test_candidate_builder_v2.py \
    training/lunar_policy_training/tests/test_multires_observation.py
  git commit -m "feat(training): build stable physical candidate universes"
  ```

  Expected: tests pass；stale-field scan 无命中。

---

### Task 7: Switch formal start, runtime candidates and oracle to the shared physical definition

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_start_qualification.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/frontier_oracle.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/tests/test_formal_start_qualification.py`
- Modify: `training/lunar_policy_training/tests/test_frontier_oracle.py`
- Modify: `training/lunar_policy_training/tests/test_formal_builder.py`

**Runtime flow:**

```text
observed maps + current pose + capability
  -> C++ traversability/reachability projection
  -> PlatformCandidateReachability
  -> CandidateBuilderV2.build_physical_universe(...)
  -> select_available(failed_ids, limit=64)

same observed maps + physical reachability, independent enumeration
  -> FrontierOpportunityOracle.evaluate_physical(...)
```

- [ ] **Step 1: 写 RED tests**

  Monkeypatch `ObservedPrimitiveReachability.update` 和 `CandidateBuilderV2.build_from_primitive_graph` 为抛异常，证明 qualification、runtime 和 oracle 均不调用它们。Oracle test 同时 monkeypatch production `build_physical_universe`，证明 oracle 不调用 production ranking/top-M/reserve。三平台同输入重复运行的 universe/oracle hash 必须一致；改变 primitives 三者不变。

- [ ] **Step 2: 运行 RED**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    training/lunar_policy_training/tests/test_formal_start_qualification.py \
    training/lunar_policy_training/tests/test_frontier_oracle.py \
    training/lunar_policy_training/tests/test_formal_builder.py
  ```

  Expected: monkeypatch guard 证明旧主路径仍调用 primitive graph，或新 physical/oracle identity 断言失败；不得以缺少 capability fixture 代替语义 RED。

- [ ] **Step 3: 切换主路径**

  `qualify_initial_start()` 使用实际 capability、observed local projection、`PlatformCandidateReachability` 和 physical universe count；删除 global-composed primitive capability。`FormalEpisode.build_policy_observation()` 删除 `_observed_primitive_request()`、`_primitive_reachability` 和 `build_from_primitive_graph()`，显式设置 `request.config.local_frontier.additional_corridor_margin_m = 2.0`。Oracle 独立枚举 observed safe/reachable cells并计算正 30m/360 LOS gain；不读 truth coverable mask、failed IDs、reserve 或 primitive graph。Oracle 返回 `oracle_opportunity_count` 和按稳定 opportunity keys 排序计算的 `oracle_opportunity_set_sha256`，由 episode 附加到正式 boundary diagnostics。

- [ ] **Step 4: GREEN and commit**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    training/lunar_policy_training/tests/test_formal_start_qualification.py \
    training/lunar_policy_training/tests/test_frontier_oracle.py \
    training/lunar_policy_training/tests/test_formal_builder.py
  ! rg -n 'build_from_primitive_graph|_observed_primitive_request|_primitive_reachability' \
    training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py \
    training/lunar_policy_training/lunar_policy_training/environment/formal_start_qualification.py \
    training/lunar_policy_training/lunar_policy_training/environment/frontier_oracle.py
  git add training/lunar_policy_training/lunar_policy_training/environment/formal_start_qualification.py \
    training/lunar_policy_training/lunar_policy_training/environment/frontier_oracle.py \
    training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py \
    training/lunar_policy_training/tests/test_formal_start_qualification.py \
    training/lunar_policy_training/tests/test_frontier_oracle.py \
    training/lunar_policy_training/tests/test_formal_builder.py
  git commit -m "refactor(training): use physical opportunities at formal boundaries"
  ```

  Expected: focused tests pass；主路径 stale scan 无命中。

---

### Task 8: Add authoritative zero-evidence refresh for initial and rolling planning failures

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/observation_boundary.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py`
- Modify: `training/lunar_policy_training/tests/test_sensor_closed_loop.py`
- Modify: `training/lunar_policy_training/tests/test_formal_builder.py`
- Modify: `training/lunar_policy_training/tests/test_v3_environment.py`

**Interfaces:**

```python
class ObservationBoundaryController:
    def rebuild_without_sensor_update(
        self,
        *,
        pose_map: Pose2,
        execution_state: str,
    ) -> BoundaryObservationResult: ...

class FormalEpisode:
    def refresh_after_planning_failure(
        self,
        candidate_id: str,
        disposition: CandidateDisposition,
    ) -> BoundaryObservationResult: ...
```

`PreparedPlanRequest` 增加 `candidate_id` 和 `physical_snapshot_id`，continuation request 保留原全局目标 candidate ID；环境不得从 action index 反推失败身份。

- [ ] **Step 1: 写 initial-failure RED test**

  构造第一个 planner 输出 `SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT` 且无 reference。断言：未调用 `observe_world`；state time、pose、evidence generation/hash、coverage、priority coverage 和 reward delta 不变；policy observation revision/candidate set identity 改变；snapshot ID 不变；失败 ID 被排除；reserve 自动补位；仍有候选时 transition 非终止。

- [ ] **Step 2: 写 rolling-failure RED test**

  执行三段 reference，每段产生不同 path evidence；第四次规划返回 suppression。断言前三段的移动、elapsed、coverage 和 reward 各聚合一次；callback 清除 active ground option 和 `_defer_candidate_rebuild`；不重复 reveal；以最新 pose/evidence 生成新 snapshot；全局目标 ID 在新 snapshot 被抑制；下一 observation 是完整 candidate rebuild。另测第三段已经首次跨越 0.95 时，不调用第四次 planner/refresh，直接 success。

- [ ] **Step 3: 运行 RED**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    training/lunar_policy_training/tests/test_sensor_closed_loop.py \
    training/lunar_policy_training/tests/test_formal_builder.py \
    training/lunar_policy_training/tests/test_v3_environment.py \
    -k 'planning_failure or zero_evidence or rolling'
  ```

  Expected: initial/rolling failure 至少一项暴露缺失 refresher、重复 sensor update 或候选未重建；time/coverage/reward 不变量必须显示在 assertion diff 中。

- [ ] **Step 4: 实现 controller/episode ownership**

  将 `_observe()` 中“sensor update”和“build/attach observation identity”拆开。零 evidence rebuild 只增加 observation revision，复用 state time、mission area、pose 和 sensor state。`FormalEpisode.refresh_after_planning_failure()` 校验 disposition；`KEEP` 不改失败集合，suppression 只写当前 snapshot；rolling failure 先 clear option/defer，再 build。`create_v3_environment()` 增加必需的 `planning_failure_refresher`，sensor-closed 模式缺失时构造失败；不得调用 generic `observation_provider`。

- [ ] **Step 5: GREEN and commit**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    training/lunar_policy_training/tests/test_sensor_closed_loop.py \
    training/lunar_policy_training/tests/test_formal_builder.py \
    training/lunar_policy_training/tests/test_v3_environment.py
  git add training/lunar_policy_training/lunar_policy_training/environment/observation_boundary.py \
    training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py \
    training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py \
    training/lunar_policy_training/tests/test_sensor_closed_loop.py \
    training/lunar_policy_training/tests/test_formal_builder.py \
    training/lunar_policy_training/tests/test_v3_environment.py
  git commit -m "fix(training): refresh candidates after planning failure"
  ```

  Expected: 三个 suites pass；initial failure 的物理状态逐位不变，rolling failure 的已执行效果只聚合一次，reserve 可在同一 snapshot 补位。

---

### Task 9: Replace index rejection and `PLANNER_REJECTED_ALL` with the new terminal matrix

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/macro_step.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/training_metrics.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py`
- Modify: `training/lunar_policy_training/tests/test_v3_environment.py`
- Modify: `training/lunar_policy_training/tests/test_training_metrics.py`
- Modify: `training/lunar_policy_training/tests/test_parallel_pool.py`

**Terminal contract:**

```python
class TerminalReason(str, Enum):
    SUCCESS = "SUCCESS"
    NO_RECOVERABLE_OBSERVATION_STATE = "NO_RECOVERABLE_OBSERVATION_STATE"
    VISITED_EXHAUSTED = "VISITED_EXHAUSTED"
    ZERO_GAIN = "ZERO_GAIN"
    NO_TRANSIT_OPPORTUNITY = "NO_TRANSIT_OPPORTUNITY"
    PLANNER_BLOCKED_WITH_OPPORTUNITY = "PLANNER_BLOCKED_WITH_OPPORTUNITY"
    HARD_FAILURE = "HARD_FAILURE"
    CANCELED = "CANCELED"
```

`CANDIDATE_ORACLE_MISMATCH` 是 fail-closed invariant code，不是可训练 terminal reason；遇到它抛 `EnvironmentInvariantError` 并丢弃 rollout。

- [ ] **Step 1: 用参数化 RED test 固化完整矩阵**

  覆盖：reset 的初始真实 sensor boundary 首次从 `<0.95` 跨到 `>=0.95` 时在调用 planner 前成功；available>0 且 oracle>0 继续；reserve>0 且 oracle>0 补位继续；universe=0/oracle=0 选择最后实际耗尽阶段；universe=0/oracle>0 fail closed；universe>0/oracle=0 fail closed；universe>0、尚有未失败 ID 但 available/reserve 都为 0 时 fail closed；all failed/reserve=0/available=0/oracle=146 只能为 `PLANNER_BLOCKED_WITH_OPPORTUNITY`；hard planner failure `KEEP` 不写失败集合并走 `HARD_FAILURE`；任意执行后的首次 crossing 永远优先于随后候选审计，已终止 episode 不得重复发出 success。

- [ ] **Step 2: 运行 RED**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    training/lunar_policy_training/tests/test_v3_environment.py \
    training/lunar_policy_training/tests/test_training_metrics.py \
    training/lunar_policy_training/tests/test_parallel_pool.py
  ```

  Expected: 旧 index blacklist、`PLANNER_REJECTED_ALL` 或 oracle-positive exhaustion 使参数化矩阵至少一项失败；基础设施 hard failure 仍须走既有 hard-failure assertion。

- [ ] **Step 3: 实现纯边界审计函数**

  删除 `_rejected_candidates`、`_mask_rejected_candidate()`、`rejected_candidate_indices` 和基于 `(outcome,directive)` 的 mask 分支。新增 `_audit_candidate_boundary(diagnostics, oracle)`，严格检查：

  ```text
  blocked := universe > 0
             and failed == universe
             and reserve == 0
             and available == 0
             and oracle > 0
  exhausted := universe == 0 and oracle == 0
  ```

  合法 exhaustion 的主原因取 production pipeline 最后非零排除阶段；episode 将 `remaining_coverable_detail_cell_count` 作为 truth-side audit 字段附加，但该值不得改变 observed-only 决策。任何不满足 continue、exhausted 或 blocked 谓词的组合都抛 `EnvironmentInvariantError`。Training metrics 将 planner blocked 单列，不计入 exhaustion 或 oracle contradiction。

- [ ] **Step 4: GREEN, forbidden-symbol scan and commit**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    training/lunar_policy_training/tests/test_v3_environment.py \
    training/lunar_policy_training/tests/test_training_metrics.py \
    training/lunar_policy_training/tests/test_parallel_pool.py
  ! rg -n 'PLANNER_REJECTED_ALL|planner_rejected_count|rejected_candidate_indices|_rejected_candidates' \
    training/lunar_policy_training/lunar_policy_training \
    training/lunar_policy_training/tests
  git add training/lunar_policy_training/lunar_policy_training/environment/macro_step.py \
    training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py \
    training/lunar_policy_training/lunar_policy_training/training_metrics.py \
    training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py \
    training/lunar_policy_training/tests/test_v3_environment.py \
    training/lunar_policy_training/tests/test_training_metrics.py \
    training/lunar_policy_training/tests/test_parallel_pool.py
  git commit -m "refactor(training): rewrite physical opportunity terminals"
  ```

  Expected: tests pass；forbidden-symbol scan 无命中。

---

### Task 10: Migrate formal replay state and checkpoint identity without compatibility resume

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_episode_state.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/checkpoint.py`
- Modify: `training/lunar_policy_training/tests/test_formal_resume_state.py`
- Modify: `training/lunar_policy_training/tests/test_checkpoint_resume.py`

**Schemas:**

```python
FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION = "lunar-formal-environment-state/v6"
CHECKPOINT_SCHEMA_VERSION = "lunar-ppo-checkpoint/v7"
```

Worker state 必须含：`physical_snapshot_id`、`physical_evidence_generation`、`physical_evidence_sha256`、`physical_candidate_universe_sha256`、有序 `planner_failed_candidate_ids`、observation identity、reveal history、`defer_candidate_rebuild`。删除 primitive graph identity 和 rejected indices。

- [ ] **Step 1: 写 RED serialization/replay tests**

  Round trip 一个含 70 个 universe candidates、6 个 failed IDs、64 中 reserve 补位后的 worker；恢复后 candidate IDs/mask、universe hash、oracle count、terminal decision 逐位一致。篡改任一 snapshot/evidence/universe/failure ID 必须拒绝。active ground option 仍不得 snapshot。

- [ ] **Step 2: 写旧身份 fail-closed tests**

  明确拒绝 formal state v5、checkpoint v6、缺新字段 payload、primitive identity payload 和通过默认值伪装的 promoted payload；checkpoint v6 用例名固定为 `test_checkpoint_resume_rejects_v6_checkpoint`。本计划不增加旧 checkpoint 的 resume、promotion 或 weight-import 旁路。

- [ ] **Step 3: 运行 RED**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    training/lunar_policy_training/tests/test_formal_resume_state.py \
    training/lunar_policy_training/tests/test_checkpoint_resume.py
  ```

  Expected: v5/v6 schema、缺失 snapshot/universe/failure identity 或旧 promotion 分支使新 strict tests 失败；现有合法 checkpoint fixtures 不得发生无关损坏。

- [ ] **Step 4: 实现 strict v6/v7 state**

  `FormalEpisode.replay_state()` 重放 reveal history，随后重算 evidence/snapshot/universe 和 batch digest；任何漂移失败。删除 checkpoint promotion 到当前 schema 的旧 resume 分支。新 run 必须使用新 cache 和新身份从 global step 0 开始；本计划不从旧 checkpoint 导入 policy、optimizer、RNG、worker state 或 run identity。

- [ ] **Step 5: GREEN and commit**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    training/lunar_policy_training/tests/test_formal_resume_state.py \
    training/lunar_policy_training/tests/test_checkpoint_resume.py
  git add training/lunar_policy_training/lunar_policy_training/environment/formal_episode_state.py \
    training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py \
    training/lunar_policy_training/lunar_policy_training/checkpoint.py \
    training/lunar_policy_training/tests/test_formal_resume_state.py \
    training/lunar_policy_training/tests/test_checkpoint_resume.py
  git commit -m "feat(training): bind replay to physical snapshots"
  ```

  Expected: replay/checkpoint suites pass；新 v6/v7 state 可逐位恢复，所有旧 formal/checkpoint identity 均 fail closed。

---

### Task 11: Freeze semantics v11 and migrate preflight, metrics and closed-loop gate

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/training_semantics.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/formal_preflight.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/closed_loop_gate.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/training_metrics.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/tests/test_formal_preflight.py`
- Modify: `training/lunar_policy_training/tests/test_closed_loop_gate.py`
- Modify: `training/lunar_policy_training/tests/test_training_metrics.py`
- Modify: `training/lunar_policy_training/tests/test_cli.py`

**Canonical identities:**

```python
FORMAL_TRAINING_SEMANTICS_VERSION = (
    "lunar-training-semantics/"
    "sensor-30m-360-platform-physical-coverable-detail95-observed-physical-"
    "candidates-fixed2m-search-domain-snapshot-planner-failure-option-path-"
    "observation-auditable-failure/v11"
)
CLOSED_LOOP_GATE_SCHEMA = "lunar-physical-opportunity-closed-loop-gate/v4"
TRAINING_UPDATE_METRICS_SCHEMA = "lunar-training-update-metrics/v4"
```

- [ ] **Step 1: 写 RED identity/gate tests**

  Preflight 必须拒绝 cache v5、semantics v10、checkpoint v6、closed-loop v3、primitive-bound report、非 2.0 config 和不完整/代理 capability。semantics 和 closed-loop 旧身份用例名固定为 `test_formal_preflight_rejects_semantics_v10`、`test_closed_loop_gate_rejects_v3_report`。Gate row 只绑定 physical projection/coverability/universe/oracle IDs；不再要求 primitive set/graph。`PLANNER_BLOCKED_WITH_OPPORTUNITY` 单独计数，qualification 上限固定为 0；hard/canceled/mismatch 也为 0。

- [ ] **Step 2: 运行 RED**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    training/lunar_policy_training/tests/test_formal_preflight.py \
    training/lunar_policy_training/tests/test_closed_loop_gate.py \
    training/lunar_policy_training/tests/test_training_metrics.py \
    training/lunar_policy_training/tests/test_cli.py
  ```

  Expected: semantics/cache/checkpoint/gate/metrics 中至少一个仍报告旧版本或接受旧 identity；真实 capability gate 的既有校验不能被放宽来制造 GREEN。

- [ ] **Step 3: 实现原子身份迁移**

  Closed-loop row SHA fields改为 `physical_projection_sha256`、`coverable_mask_sha256`、`physical_candidate_universe_sha256`、`physical_snapshot_id`、`oracle_opportunity_set_sha256`、request/planner sequence SHA。报告必须包含 planner-blocked count、domain reason count 和 margin/domain diagnostics。Preflight 验证三平台真实 capability freeze 后才允许生成 step-0 run manifest；本任务不启动训练。

- [ ] **Step 4: GREEN and commit**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    training/lunar_policy_training/tests/test_formal_preflight.py \
    training/lunar_policy_training/tests/test_closed_loop_gate.py \
    training/lunar_policy_training/tests/test_training_metrics.py \
    training/lunar_policy_training/tests/test_cli.py
  git add training/lunar_policy_training/lunar_policy_training/training_semantics.py \
    training/lunar_policy_training/lunar_policy_training/formal_preflight.py \
    training/lunar_policy_training/lunar_policy_training/closed_loop_gate.py \
    training/lunar_policy_training/lunar_policy_training/training_metrics.py \
    training/lunar_policy_training/lunar_policy_training/cli.py \
    training/lunar_policy_training/tests/test_formal_preflight.py \
    training/lunar_policy_training/tests/test_closed_loop_gate.py \
    training/lunar_policy_training/tests/test_training_metrics.py \
    training/lunar_policy_training/tests/test_cli.py
  git commit -m "feat(training): freeze physical opportunity semantics v11"
  ```

  Expected: preflight/gate/metrics/CLI suites pass；所有 schema/semantic identities 同时升级，旧组合和代理 capability 均不能生成正式 run manifest。

---

### Task 12: Lock the historical boundary failure and oracle-146 behavior as regressions

**Files:**

- Create: `training/lunar_policy_training/tests/test_physical_opportunity_regression.py`
- Modify: `ros2_ws/src/lunar_planner_core/test/hierarchical_regression_test.cpp`
- Modify: `training/lunar_policy_training/tests/test_v3_environment.py`
- Modify: `training/lunar_policy_training/tests/test_formal_builder.py`

**Pinned external evidence:**

- Trace: `/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration/preflight-gate-a2ee1acb4676/b0c2-legged-rejection-trace.jsonl`
- Trace SHA-256: `a2f7b0ece14ae963bcbea77b14aa05b9e3d44850ca1693360eed4185e9975526`
- Worker state: `/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration/preflight-gate-a2ee1acb4676/b0c2-legged-step193-state.json`
- Worker state SHA-256: `d5ba1034f96b58f0e716b94207678ebf98894661eda49f6c6a58644e99c8c5a4`
- Historical terminal evidence: coverage `0.039319027215242386`、oracle `146`、selected target `[17286.0, 97490.0, 374.2354736328125]`、old reason `LOCAL_SEGMENT_INFEASIBLE`。

这些文件只作外部证据，不复制进仓库；source test 使用小型确定性 fixture 表达同一不变量。

- [ ] **Step 1: 写 C++ historical-shape regression**

  在 `hierarchical_regression_test.cpp` 构造原始地图安全、固定 2.0 domain、目标靠近 domain 边界且 footprint 越界的地面案例。断言 map/domain 分离后同一目标成功、expanded>0、margin=2.0、domain hash非空；若 domain 真不足，必须是 `LOCAL_SEARCH_DOMAIN_EXHAUSTED`，绝不能在 expanded=0 时为 `WHEEL_GOAL_INFEASIBLE`/legged physical goal infeasible。

- [ ] **Step 2: 写 Python oracle-146 terminal regression**

  构造 universe count=26、failed count=25、reserve=1、oracle=146：必须 refill/continue；再失败最后一项后 universe=26、failed=26、reserve=0、available=0、oracle=146：必须 terminal `PLANNER_BLOCKED_WITH_OPPORTUNITY`，success=false、frontier-exhausted=false。再断言 hard failure `KEEP` 不推进到该 terminal。

- [ ] **Step 3: 写外部证据重放测试入口**

  `test_physical_opportunity_regression.py` 增加名为 `test_historical_boundary_failure_replays_against_v6_cache` 的 integration test，同时读取 `LUNAR_HISTORICAL_FAILURE_ROOT` 和 `LUNAR_PHYSICAL_CACHE_MANIFEST`。它先校验上述两个 SHA，再把旧 state 的 reveal history 当只读诊断输入，在指定 v6 cache 上通过公开 sensor-boundary API重建 observed evidence；不调用旧 worker resume，也不加载旧 v4 cache。测试重建失败目标 request，断言 fixed margin 为 2.0、原始 map hash不随 domain变化，并记录 planner output/domain diagnostics。任一环境变量缺失时只跳过这个 integration test，source-level regressions仍必跑；Task 13 生成干净 v6 cache 后再执行精确重放。

- [ ] **Step 4: Build and run the historical regression suite**

  ```bash
  export PHYSICAL_BRIDGE_ROOT=/home/kai/CodexDownloads/lunar_navigation/physical_opportunity_failure_semantics/task4-bridge
  colcon --log-base "$PHYSICAL_BRIDGE_ROOT/log-history-build" build --merge-install \
    --base-paths ros2_ws/src --packages-select lunar_planner_core \
    --build-base "$PHYSICAL_BRIDGE_ROOT/build" \
    --install-base "$PHYSICAL_BRIDGE_ROOT/install" \
    --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
  set +u
  source "$PHYSICAL_BRIDGE_ROOT/install/setup.bash"
  set -u
  colcon --log-base "$PHYSICAL_BRIDGE_ROOT/log-history" test \
    --packages-select lunar_planner_core \
    --build-base "$PHYSICAL_BRIDGE_ROOT/build" \
    --test-result-base "$PHYSICAL_BRIDGE_ROOT/build" \
    --event-handlers console_direct+ \
    --ctest-args -R hierarchical_regression
  colcon test-result --test-result-base "$PHYSICAL_BRIDGE_ROOT/build" --verbose
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    training/lunar_policy_training/tests/test_physical_opportunity_regression.py \
    training/lunar_policy_training/tests/test_v3_environment.py \
    training/lunar_policy_training/tests/test_formal_builder.py \
    -k 'historical or oracle_146 or boundary_window'
  ```

  Expected: C++ regression 与 source-level Python regressions 全部通过；仅精确外部重放因缺少 Task 13 的 v6 manifest 而显示一项声明式 skip，不能通过过滤或跳过 source-level tests 获得 GREEN。

  若精确目标仍因 `LOCAL_SEARCH_DOMAIN_EXHAUSTED` 失败，停止本计划并报告固定 2.0 设计门槛；不得把 margin 改为 4.0 或增加隐式恢复。

- [ ] **Step 5: 提交**

  ```bash
  git add ros2_ws/src/lunar_planner_core/test/hierarchical_regression_test.cpp \
    training/lunar_policy_training/tests/test_physical_opportunity_regression.py \
    training/lunar_policy_training/tests/test_v3_environment.py \
    training/lunar_policy_training/tests/test_formal_builder.py
  git commit -m "test(training): lock physical opportunity failure regressions"
  ```

---

### Task 13: Run full three-platform verification, regenerate external identities and document evidence

**Files:**

- Create: `docs/validation/2026-08-11-physical-opportunity-planner-failure-semantics.md`
- Modify: `docs/superpowers/specs/2026-08-11-physical-opportunity-planner-failure-semantics-design.md`
- Modify only if generated identity documentation requires it: `docs/validation/2026-08-08-formal-training-environment-qualification.md`

- [ ] **Step 1: Fresh native build and complete ROS tests outside the repository**

  ```bash
  set +u
  source /opt/ros/humble/setup.bash
  set -u
  test "$ROS_DISTRO" = humble
  export PHYSICAL_FINAL_ROOT=/home/kai/CodexDownloads/lunar_navigation/physical_opportunity_failure_semantics/final
  colcon --log-base "$PHYSICAL_FINAL_ROOT/log-build" build --merge-install \
    --base-paths ros2_ws/src \
    --packages-select lunar_planner_core lunar_planner_training_bridge lunar_planner_ros \
    --build-base "$PHYSICAL_FINAL_ROOT/build" \
    --install-base "$PHYSICAL_FINAL_ROOT/install" \
    --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
  set +u
  source "$PHYSICAL_FINAL_ROOT/install/setup.bash"
  set -u
  colcon --log-base "$PHYSICAL_FINAL_ROOT/log-test" test \
    --packages-select lunar_planner_core lunar_planner_training_bridge lunar_planner_ros \
    --build-base "$PHYSICAL_FINAL_ROOT/build" \
    --test-result-base "$PHYSICAL_FINAL_ROOT/build" \
    --event-handlers console_direct+
  colcon test-result --test-result-base "$PHYSICAL_FINAL_ROOT/build" --verbose
  ```

  Expected: build exit `0`；`colcon test-result` 报告 zero failures/errors，且 build/install/log 只出现在 `$PHYSICAL_FINAL_ROOT`。

- [ ] **Step 2: Full Python regression**

  ```bash
  export PHYSICAL_TRAINING_PY=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
  export PHYSICAL_BRIDGE_PY="$PHYSICAL_FINAL_ROOT/install/local/lib/python3.10/dist-packages"
  set +u
  source "$PHYSICAL_FINAL_ROOT/install/setup.bash"
  set -u
  PYTHONPATH="$PHYSICAL_BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PHYSICAL_TRAINING_PY" -m pytest -q \
    training/lunar_policy_training/tests
  ```

  Expected: training test suite exit `0`，无 failed/error；skips 只能是测试中已声明的外部设备或可选历史 artifact 条件。

- [ ] **Step 3: Generate a new v6 preflight cache and prove old identities fail closed**

  ```bash
  export PHYSICAL_FINAL_ROOT=/home/kai/CodexDownloads/lunar_navigation/physical_opportunity_failure_semantics/final
  export PHYSICAL_TRAINING_PY=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
  export PHYSICAL_BRIDGE_PY="$PHYSICAL_FINAL_ROOT/install/local/lib/python3.10/dist-packages"
  set +u
  source "$PHYSICAL_FINAL_ROOT/install/setup.bash"
  set -u
  PYTHONPATH="$PHYSICAL_BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training" \
    "$PHYSICAL_TRAINING_PY" -m lunar_policy_training.cli prepare-data \
    --source-lock /home/kai/CodexDownloads/lunar_navigation/volume3/data/locks/polar_source_lock_v1.json \
    --split-manifest /home/kai/CodexDownloads/lunar_navigation/volume3/data/splits/polar_split_v2.json \
    --cache-root "$PHYSICAL_FINAL_ROOT/cache-v6" \
    --materialization preflight \
    --preflight-scenario-limit 128
  test -f "$PHYSICAL_FINAL_ROOT/cache-v6/cache-manifest.json"
  "$PHYSICAL_TRAINING_PY" -c \
    'import json,sys; payload=json.load(open(sys.argv[1], encoding="utf-8")); assert payload["schema"] == "lunar-formal-training-cache/v6"' \
    "$PHYSICAL_FINAL_ROOT/cache-v6/cache-manifest.json"
  mkdir -p "$PHYSICAL_FINAL_ROOT/fail-closed"
  set -o pipefail
  PYTHONPATH="$PHYSICAL_BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PHYSICAL_TRAINING_PY" -m pytest -q \
    training/lunar_policy_training/tests/test_formal_cache.py::test_formal_cache_rejects_v5_manifest \
    training/lunar_policy_training/tests/test_formal_preflight.py::test_formal_preflight_rejects_semantics_v10 \
    training/lunar_policy_training/tests/test_checkpoint_resume.py::test_checkpoint_resume_rejects_v6_checkpoint \
    training/lunar_policy_training/tests/test_closed_loop_gate.py::test_closed_loop_gate_rejects_v3_report \
    2>&1 | tee "$PHYSICAL_FINAL_ROOT/fail-closed/strict-identity-tests.log"
  LUNAR_HISTORICAL_FAILURE_ROOT=/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration/preflight-gate-a2ee1acb4676 \
  LUNAR_PHYSICAL_CACHE_MANIFEST="$PHYSICAL_FINAL_ROOT/cache-v6/cache-manifest.json" \
  PYTHONPATH="$PHYSICAL_BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PHYSICAL_TRAINING_PY" -m pytest -q \
    training/lunar_policy_training/tests/test_physical_opportunity_regression.py::test_historical_boundary_failure_replays_against_v6_cache \
    2>&1 | tee "$PHYSICAL_FINAL_ROOT/fail-closed/historical-v6-replay.log"
  ```

  Expected: manifest 文件存在且 `schema` 精确为 `lunar-formal-training-cache/v6`，四个严格拒绝测试和一个历史 v6 重放测试全部通过且无 skip；严格测试通过表示旧 payload 被预期异常拒绝，而不是被 promotion/default 接受。

- [ ] **Step 4: Run deterministic three-platform closed-loop gate**

  ```bash
  export PHYSICAL_FINAL_ROOT=/home/kai/CodexDownloads/lunar_navigation/physical_opportunity_failure_semantics/final
  export PHYSICAL_TRAINING_PY=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
  export PHYSICAL_BRIDGE_PY="$PHYSICAL_FINAL_ROOT/install/local/lib/python3.10/dist-packages"
  PYTHONPATH="$PHYSICAL_BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training" \
    "$PHYSICAL_TRAINING_PY" -m lunar_policy_training.cli closed-loop-gate \
    --cache-manifest "$PHYSICAL_FINAL_ROOT/cache-v6/cache-manifest.json" \
    --artifact-root "$PHYSICAL_FINAL_ROOT/closed-loop-run-1" \
    --minimum-scenes 1 \
    --max-workers 3
  PYTHONPATH="$PHYSICAL_BRIDGE_PY:$PWD/model_contract:$PWD/training/lunar_policy_training" \
    "$PHYSICAL_TRAINING_PY" -m lunar_policy_training.cli closed-loop-gate \
    --cache-manifest "$PHYSICAL_FINAL_ROOT/cache-v6/cache-manifest.json" \
    --artifact-root "$PHYSICAL_FINAL_ROOT/closed-loop-run-2" \
    --minimum-scenes 1 \
    --max-workers 3
  "$PHYSICAL_TRAINING_PY" -c \
    'import json,sys; a=json.load(open(sys.argv[1], encoding="utf-8")); b=json.load(open(sys.argv[2], encoding="utf-8")); assert a["passed"] and b["passed"]; assert a["scene_platform_count"] == 3 == b["scene_platform_count"]; assert a["closed_loop_evidence_sha256"] == b["closed_loop_evidence_sha256"]' \
    "$PHYSICAL_FINAL_ROOT/closed-loop-run-1/closed-loop-gate.json" \
    "$PHYSICAL_FINAL_ROOT/closed-loop-run-2/closed-loop-gate.json"
  ```

  Expected: 两次命令均 exit `0`，每份报告 `passed=true`、`scene_platform_count=3`，且 `closed_loop_evidence_sha256` 完全相同。

  Gate 必须验证：三平台 physical identities deterministic；planner blocked/hard/canceled/mismatch 均为 0；合法失败有 observed-only oracle=0；成功只来自 first crossing；所有 ground diagnostics margin=2.0。正式 24-scene x 3-platform gate 仍是训练解锁前独立后续动作，不在本实现任务内自动启动。

- [ ] **Step 5: Repository and semantic scans**

  ```bash
  ! rg -n 'PLANNER_REJECTED_ALL|planner_rejected_count|rejected_candidate_indices' \
    training/lunar_policy_training/lunar_policy_training \
    training/lunar_policy_training/tests \
    ros2_ws/src/lunar_planner_core \
    ros2_ws/src/lunar_planner_training_bridge
  ! rg -n 'build_from_primitive_graph|primitive_state_schema|reachability_graph_sha256' \
    training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py \
    training/lunar_policy_training/lunar_policy_training/environment/formal_start_qualification.py \
    training/lunar_policy_training/lunar_policy_training/environment/frontier_oracle.py \
    training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py
  python3 tools/check_repository_boundaries.py .
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    tests/foundation/test_repository_boundaries.py
  git diff --check
  ```

  两个语义 scan 都必须无命中；冻结设计和 superseded 历史文档不属于活动源码扫描范围。

- [ ] **Step 6: Write validation evidence and mark the design implemented**

  Validation 文档记录 commit SHA、主机/ROS/GCC、所有命令和 test counts、cache/semantics/checkpoint/gate identities、外部 artifact paths/hashes、历史目标结果、oracle-146 terminal result以及“正式训练未启动”。只有全部验证通过后，把设计状态从“已批准，待实施”改为“已实施，待正式训练重新冻结”。

- [ ] **Step 7: Final commit and review**

  ```bash
  LC_ALL=C.UTF-8 iconv -f UTF-8 -t UTF-8 \
    docs/validation/2026-08-11-physical-opportunity-planner-failure-semantics.md \
    docs/superpowers/specs/2026-08-11-physical-opportunity-planner-failure-semantics-design.md \
    docs/validation/2026-08-08-formal-training-environment-qualification.md \
    >/dev/null
  git diff --check
  git add docs/validation/2026-08-11-physical-opportunity-planner-failure-semantics.md \
    docs/superpowers/specs/2026-08-11-physical-opportunity-planner-failure-semantics-design.md \
    docs/validation/2026-08-08-formal-training-environment-qualification.md
  git commit -m "docs(validation): record physical opportunity semantics evidence"
  git status --short --branch
  git log -13 --oneline
  ```

  最终状态必须 clean；不得 push、merge 或启动训练，等待用户确认。

## Completion Checklist

- [ ] WHEELED/LEGGED 所有正式入口显式使用 `additional_corridor_margin_m=2.0`，C++ 拒绝其他值。
- [ ] Local map layer 逐位保持原始 observed evidence，search domain 是独立、可 hash 的 center-search mask。
- [ ] Wheel 和 legged 的 footprint/support/clearance 可读取 domain 外原始地图，但 state/connector center 不得离开 domain。
- [ ] `WHEEL_GOAL_INFEASIBLE`/legged 对应原因只代表原始物理地图不可行；domain 穷尽使用 `LOCAL_SEARCH_DOMAIN_EXHAUSTED`。
- [ ] Planner bridge 发布结构化 disposition；环境不解析 reason string 或 outcome/directive 组合来淘汰候选。
- [ ] Truth denominator、runtime universe 和 oracle 对 planner primitive changes 不变。
- [ ] Candidate ID、physical snapshot、universe、reserve 和 failure set 均有 canonical deterministic identity。
- [ ] Initial/rolling planning failure 使用 sensor-closed controller 刷新，无重复 reveal/time/coverage/reward。
- [ ] `PLANNER_REJECTED_ALL` 和 index blacklist 从活动代码、schema、指标及测试中删除。
- [ ] Oracle positive 永不解释为探索完成；oracle=146/all failed 只能得到 `PLANNER_BLOCKED_WITH_OPPORTUNITY`。
- [ ] Cache v6、semantics v11、formal state v6、checkpoint v7、gate v4 和 metrics v4 原子迁移，旧 resume fail closed。
- [ ] 历史 boundary regression、三平台 tests、repository boundary 和 external closed-loop evidence 全部通过。
- [ ] 正式训练仍未启动；真实 capability、完整新 cache、worker/micro-batch 和 24x3 gate 仍需在启动前重新冻结。
