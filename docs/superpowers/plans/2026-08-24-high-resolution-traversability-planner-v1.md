# High-Resolution Traversability Planner V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为轮式月面模式增加显式启用的 `grid_traversability_v1`：持续维护课题三地图派生的原分辨率可通行地图，用同一不可变快照完成全局/局部二维规划，并发布基础轨迹、速度、标准 `nav_msgs/Path`、`TimedPath` 和可验收诊断。

**Architecture:** ROS 输入层将全局粗 OccupancyGrid、移动局部 GridMap 和 map→odom 变换融合到 core 的持久稀疏地图；规划请求捕获一个不可变 revision。V1 入口以该 revision 执行稀疏 8 邻域 A*、局部 8 m 截取、supercover 捷径和整段前/倒车时序。旧 wheel/legged/hopper 路径保持不变，V1 失败不回退 legacy。

**Tech Stack:** C++20、ROS 2 Humble/Jazzy、ament/colcon、GoogleTest、`nav_msgs`、`grid_map_msgs`、`lunar_planning_msgs`。

**Spec:** `docs/superpowers/specs/2026-08-24-high-resolution-traversability-planner-v1-design.md`

## Global Constraints

- 仅实现 V1 最小纵向闭环；不实现 D*/LPA*、SE(2) 格点、样条/动力学优化、多次换向、控制器修改或 T5 发布。
- 第一帧有效局部 GridMap 的 `resolution_m` 是 canonical resolution；全链路禁止降采样。容差外变化返回 `MAP_RESOLUTION_MISMATCH`，且不修改地图。
- 旧 planner 默认值保持 `legacy_certified`；仅 wheel + lunar surface + 显式 `grid_traversability_v1` 进入新路径。
- 所有工作在隔离 feature worktree 完成；不覆盖当前 `integration` 工作区已有 README、launch、scenario 和 bag 变化。
- 并行文件所有权：地图任务只改 `traversability_map*`；搜索任务只改 `grid_v1_planner*`；ROS 任务只改 `traversability_input*`、诊断和测试。共享类型、CMake、planner 入口、配置和文档由主线程统一修改。
- 审核边界只有：规格覆盖、编译/测试、失败语义、无关 diff；不扩展功能。

---

## Task 1: Freeze shared V1 contracts (main thread, sequential foundation)

**Files:**

- Create: `ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/traversability_map.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/types/planning_request.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/CMakeLists.txt`
- Test: `ros2_ws/src/lunar_pure_planner_core/test/public_api_test.cpp`

- [ ] **Step 1: Add the failing public-contract test**

```cpp
static_assert(std::is_copy_constructible_v<TraversabilitySnapshot>);
EXPECT_EQ(WheelPlannerModeName(WheelPlannerMode::kGridTraversabilityV1),
          "grid_traversability_v1");
PlanningRequest request;
EXPECT_EQ(request.config.wheel_planner_mode,
          WheelPlannerMode::kLegacyCertified);
EXPECT_FALSE(request.world.traversability_snapshot);
```

- [ ] **Step 2: Run the focused test and confirm the missing symbols fail**

Run: `colcon test --build-base build-plan --install-base install-plan --packages-select lunar_pure_planner_core --ctest-args -R public_api --output-on-failure`

Expected: compile failure naming `TraversabilitySnapshot` / `WheelPlannerMode`.

- [ ] **Step 3: Define the stable API used by all parallel tasks**

The header must provide these exact contracts:

```cpp
enum class TraversabilityState : std::uint8_t { kUnknown, kFree, kBlocked };
enum class WheelPlannerMode : std::uint8_t { kLegacyCertified, kGridTraversabilityV1 };

struct TraversabilityProfile final {
  std::int32_t global_occupancy_threshold{50};
  double local_occupancy_threshold{0.5};
  double maximum_slope_rad{};
  double inflation_radius_m{};
};

struct TraversabilityMetrics final {
  std::uint64_t revision{};
  std::uint64_t profile_hash{};
  std::size_t allocated_tiles{};
  std::size_t estimated_bytes{};
  std::size_t updated_cells{};
  std::size_t dirty_tiles{};
  std::size_t halo_recomputed_cells{};
  std::size_t free_cells{};
  std::size_t blocked_cells{};
  std::size_t unknown_cells{};
  std::size_t prior_conflicts{};
};

class TraversabilitySnapshot final {
 public:
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] double resolution_m() const noexcept;
  [[nodiscard]] Vec3 origin_m() const noexcept;
  [[nodiscard]] std::uint64_t revision() const noexcept;
  [[nodiscard]] std::uint64_t profile_hash() const noexcept;
  [[nodiscard]] TraversabilityState StateAtWorld(double x_m, double y_m) const;
  [[nodiscard]] TraversabilityMetrics metrics() const noexcept;
};

struct TraversabilityUpdateResult final {
  bool accepted{};
  bool changed{};
  std::string reason_code;
  TraversabilityMetrics metrics;
};

class PersistentTraversabilityMap final {
 public:
  explicit PersistentTraversabilityMap(TraversabilityProfile profile);
  [[nodiscard]] TraversabilityUpdateResult UpdateGlobal(const GridMap& global);
  [[nodiscard]] TraversabilityUpdateResult UpdateLocal(
      const GridMap& local, const RigidTransform& map_from_odom,
      std::uint64_t local_sequence);
  [[nodiscard]] std::shared_ptr<const TraversabilitySnapshot> Capture() const;
};
```

Add `wheel_planner_mode`, `grid_v1_local_horizon_m{8.0}` to `AnytimePlannerConfig`, and `std::shared_ptr<const TraversabilitySnapshot> traversability_snapshot` to `MinimalWorldSnapshot`.

- [ ] **Step 4: Build the core public API**

Run: `colcon build --build-base build-plan --install-base install-plan --packages-select lunar_pure_planner_core --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release`

Expected: build succeeds; public API test passes after Task 2 supplies method definitions.

---

## Task 2: Persistent original-resolution traversability map (parallel worker A)

**Files:**

- Create: `ros2_ws/src/lunar_pure_planner_core/src/grid_v1/traversability_map.cpp`
- Create: `ros2_ws/src/lunar_pure_planner_core/test/traversability_map_v1_test.cpp`

- [ ] **Step 1: Write failing map-contract tests**

Cover in named tests: `LocksFirstLocalResolution`, `RejectsResolutionChangeWithoutMutation`, `UsesGlobalFreeAsLazyPrior`, `KnownLocalOverridesPrior`, `UnknownDoesNotEraseHistory`, `IdenticalFrameDoesNotAdvanceRevision`, `RetainsCellsAfterWindowMoves`, `RasterizesRotatedLocalObstacleConservatively`, and `InflatesOnceAcrossTileBoundary`.

Use a 256-cell boundary in the last test and assert both sides query identically; use five translated local windows for persistence.

- [ ] **Step 2: Confirm tests fail before implementation**

Run: `colcon test --build-base build-plan --install-base install-plan --packages-select lunar_pure_planner_core --ctest-args -R traversability_map_v1 --output-on-failure`

Expected: missing implementation or failed assertions.

- [ ] **Step 3: Implement sparse raw fusion and immutable snapshots**

Implementation rules:

- fixed `256 x 256` tiles keyed by signed tile coordinates;
- global map retained as coarse prior and queried lazily at high-resolution cell centers;
- local known raw states overwrite prior/older local; local unknown performs no write;
- occupancy/elevation invalid means unknown, threshold means blocked, slope above capability means blocked;
- inflation tests source cell rectangles against `inflation_radius_m`; no center-only shortcut and no second inflation;
- snapshot owns shared immutable state; changed writes create exactly one new revision;
- invalid geometry/resolution leaves the last snapshot untouched and returns `GLOBAL_MAP_GEOMETRY_INVALID`, `MAP_RESOLUTION_MISMATCH`, or `INVALID_INPUT`.

- [ ] **Step 4: Run focused tests**

Run: `colcon test --build-base build-plan --install-base install-plan --packages-select lunar_pure_planner_core --ctest-args -R traversability_map_v1 --output-on-failure`

Expected: all map tests pass.

---

## Task 3: Sparse A*, local horizon, shortcut and timed trajectory (parallel worker B)

**Files:**

- Create: `ros2_ws/src/lunar_pure_planner_core/src/grid_v1/grid_v1_planner.hpp`
- Create: `ros2_ws/src/lunar_pure_planner_core/src/grid_v1/grid_v1_planner.cpp`
- Create: `ros2_ws/src/lunar_pure_planner_core/test/grid_v1_planner_test.cpp`

- [ ] **Step 1: Write failing planner tests**

Cover: connected FREE success, UNKNOWN/BLOCKED cuts with stage-specific reason, no diagonal corner cutting, all shortcut supercovers FREE, 8 m local limit, raw-grid fallback after injected postprocess failure, deterministic repetition, forward positive velocity, reverse negative velocity, bounded finite velocity, and strictly increasing `time_from_start`.

- [ ] **Step 2: Confirm the focused target fails**

Run: `colcon test --build-base build-plan --install-base install-plan --packages-select lunar_pure_planner_core --ctest-args -R grid_v1_planner --output-on-failure`

Expected: target missing or assertions fail.

- [ ] **Step 3: Implement the single V1 planning entrypoint**

```cpp
namespace lunar::pure_planning::grid_v1 {
[[nodiscard]] PlanningResult Plan(const PlanningRequest& request) noexcept;
[[nodiscard]] bool PathIsFree(const TraversabilitySnapshot& snapshot,
                              const std::vector<Pose3>& path,
                              std::size_t* checked_cells) noexcept;
}
```

Rules: sparse hash-based 8-neighbor A*, Euclidean heuristic, FREE-only expansion, no diagonal corner cutting, cancellation/deadline checks, at most four far-to-near local candidates, supercover farthest-visible shortcut, raw A* fallback, uniform original-resolution-or-finer output, tangent yaw, explicit corner spins, one whole-segment direction choice, speeds capped by wheel capability, start/end zero, strictly increasing time.

- [ ] **Step 4: Run focused tests**

Run: same focused command as Step 2.

Expected: all V1 planner tests pass without calling legacy backends.

---

## Task 4: ROS input maintenance and diagnostic fields (parallel worker C)

**Files:**

- Create: `ros2_ws/src/lunar_pure_planner_ros/include/lunar_pure_planner_ros/traversability_input.hpp`
- Create: `ros2_ws/src/lunar_pure_planner_ros/src/traversability_input.cpp`
- Create: `ros2_ws/src/lunar_pure_planner_ros/test/traversability_input_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/include/lunar_pure_planner_ros/request_diagnostics.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/request_diagnostics.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/request_diagnostics_test.cpp`

- [ ] **Step 1: Write failing ROS tests**

Assert that out-of-order global/local/TF inputs eventually yield one valid snapshot, invalid input preserves prior revision and reason, and final diagnostics contain every V1 key from the spec including `has_reference`, revisions, canonical resolution, origin, tile/memory/update/state counts, conflict count, calls/expanded/open peak, candidate/path/direction/postcheck counts, postprocess mode and phase timings.

- [ ] **Step 2: Implement the adapter-owned maintainer**

`TraversabilityInput` stores the latest valid global/local/TF inputs, adapts them with existing adapters, retries a pending local frame after TF arrival, applies each input sequence at most once, and exposes:

```cpp
struct TraversabilityInputSnapshot final {
  std::shared_ptr<const TraversabilitySnapshot> snapshot;
  std::string reason_code;
};
```

It must not publish or plan; it only maintains core state.

- [ ] **Step 3: Extend diagnostics without changing status mapping**

Use fields stored in `PlanningResult`; retain current `planning_outcome`, timing and severity behavior. `has_reference` is `result.reference.has_value()`.

- [ ] **Step 4: Run focused ROS tests**

Run: `colcon test --build-base build-plan --install-base install-plan --packages-select lunar_pure_planner_ros --ctest-args -R 'traversability_input|request_diagnostics' --output-on-failure`

Expected: all focused ROS tests pass.

---

## Task 5: Main-thread integration of mode dispatch, subscriptions and outputs

**Files:**

- Modify: `ros2_ws/src/lunar_pure_planner_core/src/planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/types/planning_request.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/CMakeLists.txt`
- Modify: `config/pure_planner.yaml`
- Modify: `launch/pure_planner.launch.py`
- Test: `ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp`

- [ ] **Step 1: Add integration tests before wiring**

Add tests proving: default mode still invokes legacy planner; explicit V1 rejects missing snapshot; V1 success yields `planning_outcome == 0`, `has_reference`, non-empty trajectory, equal non-empty `MotionReference.path_preview` / `Path` / `TimedPath.path`; failure emits empty Path/TimedPath.path with nonnegative time and no non-empty executable reference.

- [ ] **Step 2: Wire explicit mode and persistent updates**

Declare `wheel_planner_mode` with accepted values `legacy_certified` and `grid_traversability_v1`; create `TraversabilityInput` only for wheel V1; feed it from existing global/local/TF callbacks; attach its captured immutable snapshot to each request. In `Planner::Plan`, dispatch V1 before legacy global/local backends and never fall back on V1 failure.

- [ ] **Step 3: Preserve output identity**

Keep existing publishers. On success, generate Path/TimedPath from the same converted `MotionReference.path_preview`. On failure, publish empty Path and TimedPath.path with final elapsed time; publish an empty MotionReference only as the existing clearing behavior.

- [ ] **Step 4: Build and run package tests**

Run:

```bash
colcon build --build-base build-plan --install-base install-plan \
  --packages-up-to lunar_pure_planner_ros \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
colcon test --build-base build-plan --install-base install-plan \
  --packages-select lunar_pure_planner_core lunar_pure_planner_ros \
  --event-handlers console_direct+ --return-code-on-test-failure
```

Expected: both packages build and all tests pass.

---

## Task 6: Minimal evidence, documentation and bounded review

**Files:**

- Modify: `README.md`
- Modify: `config/external_interfaces.yaml`
- Modify: `docs/superpowers/specs/2026-08-24-planner-core-incremental-optimization-design.md`
- Create: `docs/validation/2026-08-24-high-resolution-traversability-planner-v1.md`

- [ ] **Step 1: Run deterministic and 30-run Release diagnostics**

Use the fixed synthetic five-window map fixture. Record map/planning correctness, all reason-code scenarios, and 30-run p50/p95/max for fusion/global/local/postprocess/total. These values are observations, not Orin gates.

- [ ] **Step 2: Run boundary checks**

```bash
git diff --check
git status --short
git diff --stat
```

Expected: no whitespace errors; only planned files in the isolated worktree.

- [ ] **Step 3: Update only required docs**

Document T3 inputs, original-resolution persistent traversability, opt-in mode, T4 outputs, formal success criteria, measured host evidence, and explicit `NOT_RUN` for Orin DDS/rosbag/controller/vehicle.

- [ ] **Step 4: Perform the necessary review**

Review exactly four questions: (1) does every implementation diff map to this spec, (2) can V1 ever call legacy after failure, (3) is resolution ever changed/downsampled, (4) do tests and diagnostics prove output identity and failure clearing? Fix only confirmed violations.

- [ ] **Step 5: Commit checkpoints**

```bash
git add ros2_ws/src/lunar_pure_planner_core
git commit -m "feat: add persistent high-resolution traversability planner"
git add ros2_ws/src/lunar_pure_planner_ros config launch
git commit -m "feat: wire traversability planner ROS interfaces"
git add README.md docs config/external_interfaces.yaml
git commit -m "docs: record traversability planner v1 validation"
```

Expected: three scoped commits on the feature branch; no merge, push, or modification of the dirty `integration` worktree.
