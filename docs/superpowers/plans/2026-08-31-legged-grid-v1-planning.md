# 足式 Grid V1 分层路径规划实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让足式月面请求默认使用 Grid V1 持久可通行性快照约束现有全局 ARA*，并继续使用现有 `PlanLegged` 输出 `kLeggedBodyReference`。

**Architecture:** 新增独立的 `LeggedGlobalMode`，不复用或改变轮式模式。足式 Grid V1 将快照的未膨胀三值分类批量投影到现有全局搜索网格，按足式 profile 膨胀一次并缓存，再进入现有 `PlanSurfaceGlobal -> SelectSurfaceLocalGoals -> PlanLegged -> ComposeSurfaceReference` 链路。

**Tech Stack:** C++20、ROS 2、ament/colcon、GoogleTest、pytest、YAML、Python launch

**Spec:** `docs/superpowers/specs/2026-08-31-legged-grid-v1-planning-design.md`

## Global Constraints

- `legged_global_mode` 缺省值必须为 `grid_traversability_v1`。
- `legacy_occupancy` 只允许显式选择；Grid V1 失败不得自动回退 legacy。
- 输入仍仅为全局 occupancy、局部 occupancy/elevation、里程计、TF、目标和既有能力参数。
- 不增加观测时间、高程方差、语义、学习模型或新地图层。
- 不增加足式最新地图重取、发布前 supercover、二次高程认证或 `STALE_PATH_INVALIDATED`。
- 保留现有足式局部 `PlanLegged`、局部地形约束和 `kLeggedBodyReference` 输出。
- 轮式 `wheel_planner_mode`、轮式 Grid V1 路由和轮式发布检查保持不变。
- Hopper 路由保持不变。
- 投影缓存键包含 `traversability_revision`、`profile_hash` 和 capability fingerprint；ARA* 单格查询保持 `O(1)`。
- 当前主机只提供 ROS 2 Jazzy；本计划的本机结果不得表述为 Humble、Orin 或实车验证。

---

### Task 1: 足式模式、未膨胀批量投影与缓存键

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/traversability_map.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/types/planning_request.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/grid_v1/traversability_map.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/active_planner_cache.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/active_planner_cache.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_core/test/public_api_test.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_core/test/traversability_map_v1_test.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_core/test/active_planner_cache_test.cpp`

**Interfaces:**
- Produces: `LeggedGlobalMode::{kLegacyOccupancy,kGridTraversabilityV1}` and `LeggedGlobalModeName()`.
- Produces: `TraversabilitySnapshot::BuildProjectionGrid(SearchControl)` returning an uninflated `GridMap` plus `inflation_radius_m`.
- Produces: `MakeLeggedTraversabilityProjectionCacheKey()` and `MakeLeggedTraversabilityRouteCacheKey()`.

- [ ] **Step 1: Write failing public-mode and projection-grid tests**

Add a default-mode assertion:

```cpp
PlanningRequest request;
EXPECT_EQ(request.config.legged_global_mode,
          LeggedGlobalMode::kGridTraversabilityV1);
EXPECT_EQ(LeggedGlobalModeName(LeggedGlobalMode::kLegacyOccupancy),
          "legacy_occupancy");
```

Add a projection fixture whose global prior is free, whose local elevation creates one blocked slope cell, and whose local NaN cell leaves the prior unchanged. Assert that `BuildProjectionGrid({})` returns global-grid geometry, marks the slope-intersecting global cell occupied, leaves the NaN/prior-free cell free, and reports the profile inflation radius without applying it to neighboring cells.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
source /opt/ros/jazzy/setup.bash
cmake -S ros2_ws/src/lunar_pure_planner_core \
  -B /home/kai/CodexDownloads/lunar_navigation/legged_grid_v1_20260831/working-core \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/home/kai/CodexDownloads/lunar_navigation/legged_grid_v1_20260831/working-install
cmake --build /home/kai/CodexDownloads/lunar_navigation/legged_grid_v1_20260831/working-core \
  --target lunar_pure_planner_core_public_api_test \
           lunar_pure_planner_core_traversability_map_v1_test -j2
```

Expected: compilation fails because `LeggedGlobalMode` and `BuildProjectionGrid` do not exist.

- [ ] **Step 3: Implement the minimal mode and batch projection API**

Add to the core request configuration:

```cpp
enum class LeggedGlobalMode : std::uint8_t {
  kLegacyOccupancy,
  kGridTraversabilityV1,
};

struct AnytimePlannerConfig final {
  // existing fields
  LeggedGlobalMode legged_global_mode{LeggedGlobalMode::kGridTraversabilityV1};
};
```

Add a snapshot result type and method:

```cpp
struct TraversabilityProjectionGrid final {
  GridMap map;
  double inflation_radius_m{};
};

struct TraversabilityProjectionGridBuildResult final {
  std::optional<TraversabilityProjectionGrid> value;
  std::string reason_code;
  [[nodiscard]] bool ok() const noexcept;
};

[[nodiscard]] TraversabilityProjectionGridBuildResult BuildProjectionGrid(
    SearchControl control = {}) const;
```

Implementation rules:

1. Start from the captured global occupancy geometry and normalize it to `-1/0/100`.
2. Iterate the snapshot's raw local classified cells, never call inflated `StateAtWorld()`.
3. Any raw local `BLOCKED` cell conservatively blocks every intersecting global projection cell.
4. A global prior hazard is cleared only when raw local `FREE` cells cover all canonical cell centers inside that global cell; partial coverage or raw `UNKNOWN` does not clear it.
5. Return the profile's inflation radius separately; do not inflate in this method.
6. Check cancellation/deadline during linear scans and return the existing reason string.

- [ ] **Step 4: Add revision/profile-aware cache-key helpers**

Implement:

```cpp
GlobalProjectionCacheKey MakeLeggedTraversabilityProjectionCacheKey(
    std::uint64_t traversability_revision,
    std::uint64_t profile_hash,
    std::uint64_t capability_fingerprint) noexcept;

GlobalRouteCacheKey MakeLeggedTraversabilityRouteCacheKey(
    const PlanningRequest& input,
    std::uint64_t traversability_revision,
    std::uint64_t profile_hash,
    std::uint64_t capability_fingerprint) noexcept;
```

Use the traversability revision as the first source sequence and include a fixed mode discriminator in semantic identities so legacy and legged Grid V1 artifacts cannot alias.

- [ ] **Step 5: Run focused tests and commit**

Run the three focused test binaries and `git diff --check`. Expected: all pass.

```bash
git add \
  ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/traversability_map.hpp \
  ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/types/planning_request.hpp \
  ros2_ws/src/lunar_pure_planner_core/src/grid_v1/traversability_map.cpp \
  ros2_ws/src/lunar_pure_planner_core/src/shared/active_planner_cache.hpp \
  ros2_ws/src/lunar_pure_planner_core/src/shared/active_planner_cache.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/public_api_test.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/traversability_map_v1_test.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/active_planner_cache_test.cpp
git commit -m "feat: add legged traversability projection contract"
```

---

### Task 2: 将足式 Grid V1 投影接入现有全局 ARA* 和局部足式规划

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/global_occupancy_projection.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/global_occupancy_projection.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/hierarchical/global_route_planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/planner.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_core/test/global_occupancy_projection_test.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_core/test/dual_mode_planner_test.cpp`

**Interfaces:**
- Consumes: `TraversabilitySnapshot::BuildProjectionGrid()` and the two legged cache keys from Task 1.
- Produces: `BuildLeggedTraversabilityProjection()` returning the existing `GlobalOccupancyProjection` type.
- Produces: core routing that uses Grid V1 only for legged lunar-surface global planning while preserving `PlanLocalDefault()`.

- [ ] **Step 1: Write failing projection and end-to-end core tests**

Add a projection test that builds a small uninflated grid with one hazard and inflation `0.678 m`, then asserts:

```cpp
ASSERT_TRUE(result.ok()) << result.reason_code;
EXPECT_FALSE(result.projection->View().HardFeasible(hazard));
EXPECT_FALSE(result.projection->View().HardFeasible(neighbor_inside_radius));
EXPECT_TRUE(result.projection->View().HardFeasible(cell_outside_radius));
```

Add core tests covering these observable behaviors:

- default legged lunar-surface request without a traversability snapshot returns `INVALID_INPUT` before backends run;
- explicit `kLegacyOccupancy` still calls the injected legacy global backend;
- default legged Grid V1 routes around a locally classified slope block, then returns a non-empty `TrajectoryReference` with `kLeggedBodyReference`;
- two identical requests make the second request report both global projection and route cache hits;
- a Grid V1 global failure returns the Grid V1 failure and never invokes a legacy fallback backend.

- [ ] **Step 2: Run focused tests and verify RED**

Expected: tests fail because the core currently treats Grid V1 as wheel-only and `PlanSurfaceGlobal()` only builds from global occupancy.

- [ ] **Step 3: Implement one-time projection build**

Implement:

```cpp
GlobalOccupancyProjectionBuildResult BuildLeggedTraversabilityProjection(
    const TraversabilitySnapshot& snapshot,
    SearchControl control = {});
```

The function calls `BuildProjectionGrid()`, creates the existing immutable `MapSnapshot`, then calls `BuildInflatedGlobalOccupancyProjection()` exactly once with the returned radius. It must not call `StateAtWorld()` during ARA* expansion.

- [ ] **Step 4: Select the projection and cache keys in `PlanSurfaceGlobal`**

For `LeggedCapability + kGridTraversabilityV1`:

- require a valid snapshot;
- use `MakeLeggedTraversabilityProjectionCacheKey()`;
- build via `BuildLeggedTraversabilityProjection()`;
- compute start/goal cells from `projection.View().map`;
- use `MakeLeggedTraversabilityRouteCacheKey()`;
- call the existing `SearchSurfaceGlobal()` unchanged.

All other requests continue through the existing occupancy projection code.

- [ ] **Step 5: Route legged Grid V1 through the existing hierarchical pipeline and select a 3 m local target**

In `Planner::Plan()` distinguish:

```cpp
const bool use_wheel_grid_v1 = /* wheeled request + wheel mode */;
const bool use_legged_grid_v1 = /* legged request + legged mode */;
```

Only `use_wheel_grid_v1` calls `grid_v1::Plan()`. `use_legged_grid_v1` continues through the existing hierarchical global/local/reference path, with the snapshot validation added to the common input gate.

In `SelectSurfaceLocalGoals()`, construct `SurfaceRollingSession` with horizon
`3.0 m` for `LeggedCapability` and retain `8.0 m` for the existing wheel/Hopper
paths. Add a test whose route extends beyond both distances and assert that the
selected legged candidate does not exceed the 3 m route horizon.

- [ ] **Step 6: Run focused and core regression tests, then commit**

Run projection, dual-mode, surface-global, legged-local, Grid V1 wheel and Hopper tests. Expected: all pass and wheel/Hopper behavior remains unchanged.

```bash
git add \
  ros2_ws/src/lunar_pure_planner_core/src/shared/global_occupancy_projection.hpp \
  ros2_ws/src/lunar_pure_planner_core/src/shared/global_occupancy_projection.cpp \
  ros2_ws/src/lunar_pure_planner_core/src/hierarchical/global_route_planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/src/planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/global_occupancy_projection_test.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/dual_mode_planner_test.cpp
git commit -m "feat: route legged planning through Grid V1"
```

---

### Task 3: ROS 参数、足式 profile 和请求快照接入

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp`

**Interfaces:**
- Consumes: `LeggedGlobalMode` and existing `TraversabilityInput`.
- Produces: ROS parameter `legged_global_mode` and a foot-specific `TraversabilityProfile`.

- [ ] **Step 1: Write failing server tests**

Add tests proving:

- omitted `legged_global_mode` on a legged server passes `kGridTraversabilityV1` and a valid snapshot into the planner callback;
- explicit `legacy_occupancy` passes no required Grid V1 snapshot and preserves the legacy path;
- an unknown legged mode rejects server construction;
- a legged Grid V1 success is finalized without a latest-map recapture or `STALE_PATH_INVALIDATED` result, while the existing wheel stale-path test remains unchanged.

- [ ] **Step 2: Run the server test and verify RED**

Expected: new tests fail because the parameter and legged traversability input do not exist.

- [ ] **Step 3: Parse the legged mode and construct the correct profile**

Add:

```cpp
LeggedGlobalMode ParseLeggedGlobalMode(std::string_view mode);
```

For legged Grid V1 use:

```cpp
inflation_radius_m =
    std::hypot(body_extent_m.x, body_extent_m.y) / 2.0 +
    minimum_body_clearance_m;
maximum_slope_rad = capability.maximum_slope_rad;
```

Create `TraversabilityInput` when either wheel Grid V1 or legged Grid V1 is active. Keep the current rejection of `wheel_planner_mode=grid_traversability_v1` on non-wheel platforms.

- [ ] **Step 4: Capture and attach the legged snapshot without adding publish checks**

Use a common `uses_traversability_snapshot` boolean for request capture and diagnostics. Keep the existing latest-map `PathIsFree()` block gated only by `use_wheel_grid_v1`; legged requests must skip it. Require the global occupancy input for legged surface planning.

- [ ] **Step 5: Build and run the server test, then commit**

Run the complete `pure_plan_motion_server_test` under the external Jazzy build. Expected: new legged tests and existing wheel stale-path test all pass.

```bash
git add \
  ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp
git commit -m "feat: configure legged Grid V1 requests"
```

---

### Task 4: 默认配置、启动参数和必要回归验证

**Files:**
- Modify: `config/pure_planner.yaml`
- Modify: `launch/pure_planner.launch.py`
- Modify: `tests/test_launch_contract.py`
- Modify: `docs/superpowers/plans/2026-08-31-legged-grid-v1-planning.md`

**Interfaces:**
- Consumes: ROS parameter `legged_global_mode` from Task 3.
- Produces: YAML/launch default `grid_traversability_v1` and explicit rollback value `legacy_occupancy`.

- [ ] **Step 1: Write failing launch/config behavior tests**

Extend the launch contract to load the YAML and launch description, asserting:

```python
assert params["legged_global_mode"] == "grid_traversability_v1"
assert launch_arguments["legged_global_mode"] == "grid_traversability_v1"
```

Also assert the node parameter mapping forwards the launch configuration; do not add source-text assertions for removed safety checks.

- [ ] **Step 2: Run pytest and verify RED**

Expected: failure because the parameter is absent.

- [ ] **Step 3: Add the default parameter**

Add to YAML:

```yaml
legged_global_mode: grid_traversability_v1
```

Declare and forward the same launch argument. Do not rename `wheel_planner_mode`.

- [ ] **Step 4: Run scoped full verification**

Run:

```bash
source /opt/ros/jazzy/setup.bash
cmake --build /home/kai/CodexDownloads/lunar_navigation/legged_grid_v1_20260831/working-core --target install -j2
ctest --test-dir /home/kai/CodexDownloads/lunar_navigation/legged_grid_v1_20260831/working-core \
  --output-on-failure -E 'anytime_wheel_planner_test'
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  tests/test_launch_contract.py tests/test_action_contract.py \
  tests/test_external_interface_contract.py
git diff --check
```

Also run the complete ROS planner package build/test in external `build/install/log` directories. If Humble or Orin is unavailable, record that boundary instead of claiming deployment proof.

- [ ] **Step 5: Mark this plan complete and commit the final scoped files**

Change completed checkboxes in this document to `[x]`, stage only this task's files, inspect `git diff --cached`, then commit:

```bash
git add config/pure_planner.yaml launch/pure_planner.launch.py \
  tests/test_launch_contract.py \
  docs/superpowers/plans/2026-08-31-legged-grid-v1-planning.md
git commit -m "config: default legged planning to Grid V1"
```

Do not push, merge, change wheel defaults, add new safety certification, or add unrelated documentation.
