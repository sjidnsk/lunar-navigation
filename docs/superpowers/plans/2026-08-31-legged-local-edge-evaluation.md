# Legged Local Edge Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated dense legged body sweeps with a cached traversability projection and a conservative fast path while preserving current local-planner behavior and timeout evidence.

**Architecture:** Build one immutable `LeggedTraversalProjection` from the existing local occupancy/elevation projection and legged capability. Every `PlanLegged` edge first queries a conservative whole-edge AABB against an integral image; open terrain uses centerline supercover aggregation, while hazard-containing AABBs fall back to the existing oriented rectangle sweep using precomputed cell fields. Cache the projection by map identity and capability, propagate only the counters needed to diagnose performance, and retain those counters when the outer planner converts a result to `TIMEOUT`.

**Tech Stack:** C++20, ROS 2 Jazzy development build, `ament_cmake`, GoogleTest/CTest, existing `SearchControl` and immutable active-cache infrastructure.

**Spec:** `docs/superpowers/specs/2026-08-31-legged-grid-v1-planning-design.md`

## Global Constraints

- Inputs remain exactly the existing local `occupancy` and `elevation` layers; do not add variance, observation time, semantics, learning, or sensors.
- Preserve the current legged state key, 64/128 yaw bins, six production primitives, ARA* search, costs, 3.0 m local horizon, output contract, cancellation, and deadline behavior.
- Do not add a post-plan full-body certification pass or precomputed `(yaw_bin, primitive)` sweep masks.
- Do not change wheel, Hopper, controller, `/Car/T5/Car_Cmd_Vel`, or `/lunar_demo/*` isolation behavior.
- A fast-path edge is valid only when its conservative whole-edge AABB contains no cell that fails occupancy, finite elevation, slope, or the existing 3 x 3 step rule.
- Jazzy evidence is local development evidence only; Humble, Orin, DDS, rosbag, and vehicle evidence remain `NOT_RUN` unless actually executed.
- Stage and commit only explicit files from this feature; do not merge, push, create a PR, or modify unrelated worktrees.

---

### Task 1: Cached legged traversal projection

**Files:**
- Create: `ros2_ws/src/lunar_pure_planner_core/src/legged/legged_traversal_projection.hpp`
- Create: `ros2_ws/src/lunar_pure_planner_core/src/legged/legged_traversal_projection.cpp`
- Create: `ros2_ws/src/lunar_pure_planner_core/test/legged_traversal_projection_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/active_planner_cache.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/active_planner_cache.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/active_planner_cache_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/CMakeLists.txt`

**Interfaces:**
- Consumes: `shared::LocalTerrainProjection`, `LeggedCapability`, `SearchControl`, and `shared::StableCapabilityFingerprint(const PlatformCapability&)`.
- Produces: `legged::BuildLeggedTraversalProjection(std::shared_ptr<const shared::LocalTerrainProjection>, const LeggedCapability&, const SearchControl&) -> LeggedTraversalProjectionBuildResult`.
- Produces: `LeggedTraversalProjection::AllCellsTraversable(shared::GridCell minimum, shared::GridCell maximum) const noexcept -> bool` and indexed `hard_feasible`, `step_feasible`, `slope_rad`, `roughness_m`, and `clearance_m` arrays.
- Produces: `shared::MakeLeggedLocalProjectionCacheKey(sequence, width, height, resolution, occupancy_threshold, capability_fingerprint)` and `ActivePlannerCache::legged_local_projection()`.

- [ ] **Step 1: Write projection behavior tests**

  Add literal fixtures proving that an open 5 x 5 map is fully traversable, an occupied cell is hard-infeasible, a non-finite elevation is hard-infeasible, a slope over the capability limit is hard-infeasible, and a center cell with a neighboring elevation jump over `maximum_step_height_m` has `step_feasible == 0`. Query rectangles that include and exclude the bad cell so a broken prefix-sum inclusion/exclusion formula fails observably.

  ```cpp
  const auto built = BuildLeggedTraversalProjection(
      fixture.projection, capability, {});
  ASSERT_TRUE(built.ok()) << built.reason_code;
  EXPECT_EQ(built.value->hard_feasible[fixture.Index(2, 2)], 1U);
  EXPECT_EQ(built.value->step_feasible[fixture.Index(2, 2)], 0U);
  EXPECT_FALSE(built.value->AllCellsTraversable({1, 1}, {3, 3}));
  EXPECT_TRUE(built.value->AllCellsTraversable({0, 0}, {0, 0}));
  ```

- [ ] **Step 2: Run the new target and verify RED**

  Configure/build only the affected package using the existing Jazzy overlay, then run:

  ```bash
  source /opt/ros/jazzy/setup.bash
  colcon build --base-paths ros2_ws/src --packages-select lunar_pure_planner_core --build-base build-jazzy-legged-edge --install-base install-jazzy-legged-edge --log-base /tmp/lunar-legged-edge-log --cmake-args -DBUILD_TESTING=ON
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_core -R lunar_pure_planner_core_legged_traversal_projection_test --output-on-failure
  ```

  Expected: build/test fails because `legged_traversal_projection.hpp` and the requested API do not exist.

- [ ] **Step 3: Implement the minimal immutable projection**

  Define the focused data type and builder:

  ```cpp
  struct LeggedTraversalProjection final {
    std::shared_ptr<const shared::LocalTerrainProjection> terrain;
    std::vector<std::uint8_t> hard_feasible;
    std::vector<std::uint8_t> step_feasible;
    std::vector<float> slope_rad;
    std::vector<float> roughness_m;
    std::vector<float> clearance_m;
    std::vector<std::size_t> hard_infeasible_prefix_sum;

    [[nodiscard]] bool AllCellsTraversable(
        shared::GridCell minimum,
        shared::GridCell maximum) const noexcept;
  };

  struct LeggedTraversalProjectionBuildResult final {
    std::shared_ptr<const LeggedTraversalProjection> value;
    std::string reason_code;
    [[nodiscard]] bool ok() const noexcept;
  };
  ```

  Build all arrays in `O(N)` with one 3 x 3 step pass per cell. Set `hard_feasible` from `free_with_height`, finite slope, and the capability slope limit; set `step_feasible` from the current `CellStepFeasible` rule. Count `!(hard_feasible && step_feasible)` in the `(width + 1) x (height + 1)` integral image. Return `REQUEST_CANCELED`, `TIMEOUT`, or `INVALID_INPUT` through the existing `SearchControl` semantics.

- [ ] **Step 4: Write and verify the cache-key RED test**

  Extend `active_planner_cache_test.cpp` so each required identity changes the key independently:

  ```cpp
  const auto original = MakeLeggedLocalProjectionCacheKey(
      7U, 320U, 320U, 0.2, 0.5, 11U);
  EXPECT_NE(original, MakeLeggedLocalProjectionCacheKey(
      8U, 320U, 320U, 0.2, 0.5, 11U));
  EXPECT_NE(original, MakeLeggedLocalProjectionCacheKey(
      7U, 321U, 320U, 0.2, 0.5, 11U));
  EXPECT_NE(original, MakeLeggedLocalProjectionCacheKey(
      7U, 320U, 320U, 0.25, 0.5, 11U));
  EXPECT_NE(original, MakeLeggedLocalProjectionCacheKey(
      7U, 320U, 320U, 0.2, 0.6, 11U));
  EXPECT_NE(original, MakeLeggedLocalProjectionCacheKey(
      7U, 320U, 320U, 0.2, 0.5, 12U));
  ```

  Run the active-cache target and confirm it fails to compile because the key maker is absent.

- [ ] **Step 5: Add the cache domain and production/test sources**

  Add `LeggedLocalProjectionCacheKey = RevisionCacheKey<..., 1U, 5U>`, a typed immutable cache slot for `legged::LeggedTraversalProjection`, and a key maker whose semantic identities are width/height, bitwise resolution, bitwise occupancy threshold, and capability fingerprint. Add the new source and GoogleTest target to `CMakeLists.txt`.

- [ ] **Step 6: Verify GREEN and commit Task 1**

  ```bash
  source /opt/ros/jazzy/setup.bash
  colcon build --base-paths ros2_ws/src --packages-select lunar_pure_planner_core --build-base build-jazzy-legged-edge --install-base install-jazzy-legged-edge --log-base /tmp/lunar-legged-edge-log --cmake-args -DBUILD_TESTING=ON
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_core -R 'legged_traversal_projection|active_planner_cache' --output-on-failure
  git diff --check
  git add ros2_ws/src/lunar_pure_planner_core/CMakeLists.txt ros2_ws/src/lunar_pure_planner_core/src/legged/legged_traversal_projection.hpp ros2_ws/src/lunar_pure_planner_core/src/legged/legged_traversal_projection.cpp ros2_ws/src/lunar_pure_planner_core/src/shared/active_planner_cache.hpp ros2_ws/src/lunar_pure_planner_core/src/shared/active_planner_cache.cpp ros2_ws/src/lunar_pure_planner_core/test/legged_traversal_projection_test.cpp ros2_ws/src/lunar_pure_planner_core/test/active_planner_cache_test.cpp
  git commit -m "feat: cache legged local traversal projection"
  ```

  Expected: both focused targets pass and the worktree contains only the new commit.

### Task 2: Two-stage edge evaluation in the production legged planner

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/legged/anytime_legged_planner.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/legged/anytime_legged_planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/anytime_legged_planner_test.cpp`

**Interfaces:**
- Consumes: `LeggedTraversalProjection::AllCellsTraversable(...)` and its precomputed arrays.
- Produces: shared `LeggedPlanRequest::traversal`, plus `LeggedPlanResult::fast_path_accepts`, `exact_sweep_fallbacks`, and `exact_sweep_cell_checks`.
- Preserves: existing `sweep_cell_checks` as the exact oriented-cell intersection counter for compatibility with current tests.

- [ ] **Step 1: Write fast-path and fallback RED tests**

  Build the projection explicitly in the fixture and pass it to `PlanLegged`. Add one open-map test that succeeds with `fast_path_accepts > 0`, `exact_sweep_fallbacks == 0`, and `exact_sweep_cell_checks == 0`. Add one near-obstacle detour/rotation test that succeeds or returns the same existing no-path result while proving `exact_sweep_fallbacks > 0`; this catches incorrectly accepting an AABB containing hazards.

  ```cpp
  ASSERT_TRUE(open_result.ok()) << open_result.reason_code;
  EXPECT_GT(open_result.fast_path_accepts, 0U);
  EXPECT_EQ(open_result.exact_sweep_fallbacks, 0U);
  EXPECT_EQ(open_result.exact_sweep_cell_checks, 0U);

  EXPECT_GT(hazard_result.exact_sweep_fallbacks, 0U);
  EXPECT_GT(hazard_result.exact_sweep_cell_checks, 0U);
  ```

  In the same RED cycle, add the production-shape fixture with `320 x 320`,
  `0.2 m` resolution, start `(10.1, 10.1, 0.5)`, goal 3.0 m away, yaw
  `1.03242`, body extent `0.68 x 0.33`, clearance `0.3`, slope `30 deg`, step
  `0.5 m`, gap `0.3 m`, and the six 0.2 m production primitives. Use a
  steady-clock deadline of 3 s and assert success, at least one fast accept,
  zero exact fallback on the open map, and under one second elapsed outside
  sanitizer builds. This test fails before implementation because the request
  pointer and counters are absent.

- [ ] **Step 2: Run the legged planner target and verify RED**

  ```bash
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_core -R lunar_pure_planner_core_anytime_legged_planner_test --output-on-failure
  ```

  Expected: compile failure for the missing request pointer and counters.

- [ ] **Step 3: Implement conservative whole-edge AABB and centerline supercover**

  Add the traversal pointer to `LeggedPlanRequest` and reject a missing or mismatched projection as `INVALID_INPUT`. For an edge, calculate `radius = hypot(body_extent.x / 2 + minimum_body_clearance_m, body_extent.y / 2 + minimum_body_clearance_m)` and query the cell rectangle enclosing `[min(source.x,target.x)-radius, max(...)+radius] x [min(source.y,target.y)-radius, max(...)+radius]`. This endpoint segment plus circumscribed radius conservatively contains all interpolated rotations.

  Implement a local integer supercover identical in corner behavior to Grid V1. Walk source-to-target center cells once to check the existing interpolated body-height interval and aggregate maximum slope, maximum roughness, and minimum clearance from the projection arrays.

- [ ] **Step 4: Implement the two branches without a second certification**

  If `AllCellsTraversable` is true, increment `fast_path_accepts`, use the supercover aggregates, and skip all oriented rectangle cell loops. Otherwise increment `exact_sweep_fallbacks`, retain the current `resolution / 4` interpolation and `RectangleIntersectsCell` behavior, replace `CellStepFeasible` calls with indexed projection reads, increment `exact_sweep_cell_checks`, and stop at the first intersecting infeasible cell. Do not add a final trajectory rescan.

- [ ] **Step 5: Keep current edge semantics covered**

  Extend existing translation, lateral, spin, slope, step, boundary, cancellation, and timeout tests to construct/pass the projection. Preserve their status, reason, trajectory, cost, `maximum_sweep_translation_step_m`, and edge-cache assertions. A mutation that changes the prefix query to ignore `step_feasible` must fail the step test.

- [ ] **Step 6: Verify GREEN and commit Task 2**

  ```bash
  source /opt/ros/jazzy/setup.bash
  colcon build --base-paths ros2_ws/src --packages-select lunar_pure_planner_core --build-base build-jazzy-legged-edge --install-base install-jazzy-legged-edge --log-base /tmp/lunar-legged-edge-log --cmake-args -DBUILD_TESTING=ON
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_core -R 'legged_traversal_projection|anytime_legged_planner' --output-on-failure
  git diff --check
  git add ros2_ws/src/lunar_pure_planner_core/src/legged/anytime_legged_planner.hpp ros2_ws/src/lunar_pure_planner_core/src/legged/anytime_legged_planner.cpp ros2_ws/src/lunar_pure_planner_core/test/anytime_legged_planner_test.cpp
  git commit -m "perf: avoid dense legged sweeps on open terrain"
  ```

### Task 3: Planner integration, observable diagnostics, and timeout retention

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/planner.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/types/planning_request.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/dual_mode_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/request_diagnostics.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/request_diagnostics_test.cpp`

**Interfaces:**
- Consumes: active-cache projection slot and `LeggedPlanResult` counters from Tasks 1 and 2.
- Produces: `LeggedLocalDiagnostics { active, traversal_projection_cache_hit, fast_path_accepts, exact_sweep_fallbacks, exact_sweep_cell_checks, edge_validation_cache_hits }` in `LocalStageResult` and `PlanningResult`.
- Produces ROS diagnostic keys with the same snake-case field names prefixed by `legged_`.

- [ ] **Step 1: Write integration/cache RED tests**

  In `dual_mode_planner_test.cpp`, call `Planner::PlanLocal` twice with the same nonzero local-map sequence and legged capability. Assert that the first result has active legged diagnostics and a cold traversal projection, the second is a cache hit, and both solve with fast-path accepts. Change only the capability step-height limit and assert the projection is rebuilt.

- [ ] **Step 2: Write timeout-retention RED test**

  Add a custom backend that returns a timed-out `LocalStageResult` with `expanded_states = 41` and literal legged counters, while the manual clock reaches the hard deadline. Assert the composed `PlanningResult` remains `TIMEOUT` and retains every counter:

  ```cpp
  EXPECT_EQ(output.status, PlanningStatus::kTimedOut);
  EXPECT_EQ(output.expanded_states, 41U);
  EXPECT_EQ(output.legged_local.fast_path_accepts, 17U);
  EXPECT_EQ(output.legged_local.exact_sweep_fallbacks, 3U);
  EXPECT_EQ(output.legged_local.exact_sweep_cell_checks, 29U);
  ```

- [ ] **Step 3: Run core targets and verify RED**

  ```bash
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_core -R 'dual_mode_planner|active_planner_cache' --output-on-failure
  ```

  Expected: compile failures for the missing diagnostics and traversal projection integration.

- [ ] **Step 4: Build/cache the projection only for the legged branch**

  In `PlanLocalDefault`, after the shared local projection succeeds and after validating the legged state/capability, compute the capability fingerprint, build/get `LeggedTraversalProjection`, and pass it to `PlanLegged`. Copy cache-hit and edge counters into `LocalStageResult`. Wheel and Hopper branches must not access this slot.

- [ ] **Step 5: Preserve diagnostics through all outer timeout conversions**

  Add one helper that copies `expanded_states`, selected goal, best cost, cache flags, and `legged_local` from an existing result to a replacement status result. Use it in the hard-deadline conversion inside `finish()` and when `local_finished >= hard_deadline`; do not replace a populated result with a fresh metric-free `Failure(...)`.

- [ ] **Step 6: Add ROS diagnostic output RED/GREEN cycle**

  First add literal expectations for:

  ```text
  legged_traversal_projection_cache_hit
  legged_fast_path_accepts
  legged_exact_sweep_fallbacks
  legged_exact_sweep_cell_checks
  legged_edge_validation_cache_hits
  ```

  Verify the ROS test fails, then emit these keys only when `result.legged_local.active` is true. Run:

  ```bash
  source /opt/ros/jazzy/setup.bash
  source install-jazzy-legged-edge/setup.bash
  colcon build --base-paths ros2_ws/src --packages-select lunar_pure_planner_ros --build-base build-jazzy-legged-edge --install-base install-jazzy-legged-edge --log-base /tmp/lunar-legged-edge-log --cmake-args -DBUILD_TESTING=ON
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_ros -R request_diagnostics --output-on-failure
  ```

- [ ] **Step 7: Verify GREEN and commit Task 3**

  ```bash
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_core -R 'dual_mode_planner|active_planner_cache|anytime_legged_planner' --output-on-failure
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_ros -R request_diagnostics --output-on-failure
  git diff --check
  git add ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/planner.hpp ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/types/planning_request.hpp ros2_ws/src/lunar_pure_planner_core/src/planner.cpp ros2_ws/src/lunar_pure_planner_core/test/dual_mode_planner_test.cpp ros2_ws/src/lunar_pure_planner_ros/src/request_diagnostics.cpp ros2_ws/src/lunar_pure_planner_ros/test/request_diagnostics_test.cpp
  git commit -m "feat: report legged edge evaluation diagnostics"
  ```

### Task 4: Production-shape benchmark evidence and scoped regression

**Files:**
- Modify: `docs/superpowers/specs/2026-08-31-legged-grid-v1-planning-design.md`

**Interfaces:**
- Consumes: production `config/legged.yaml` values represented as literal capability values in a deterministic core test.
- Produces: repeatable 64 m x 64 m, 0.2 m, 3 m goal, six-primitive, nonzero-yaw performance evidence and recorded local Jazzy results.

- [ ] **Step 1: Run the production-shape performance test created in Task 2**

  Run the exact fixture added before the implementation and retain its literal
  assertions:

  ```cpp
  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_GT(result.metrics.expanded_states, 0U);
  EXPECT_GT(result.fast_path_accepts, 0U);
  EXPECT_EQ(result.exact_sweep_fallbacks, 0U);
  #if !defined(__SANITIZE_ADDRESS__)
  EXPECT_LT(elapsed, std::chrono::seconds{1});
  #endif
  ```

  The production change caught is regression to dense oriented sweeps: that
  mutation makes exact counters nonzero and violates the latency target. If it
  fails, fix only the projection/edge implementation; do not relax the map
  size, primitive set, yaw, or thresholds.

- [ ] **Step 2: Run scoped package verification serially**

  ```bash
  source /opt/ros/jazzy/setup.bash
  colcon build --base-paths ros2_ws/src --packages-select lunar_pure_planner_core lunar_pure_planner_ros --build-base build-jazzy-legged-edge --install-base install-jazzy-legged-edge --log-base /tmp/lunar-legged-edge-log --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_core --output-on-failure -j1
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_ros --output-on-failure -j1
  python3 -m pytest tests/test_isolation_contract.py tests/test_action_contract.py
  git diff --check
  ```

  Record exact counts/timings. A pre-existing unrelated failure is reported separately with its test name and does not erase focused passing evidence.

- [ ] **Step 3: Record evidence and remaining target boundaries**

  Append an implementation evidence subsection to the spec containing the commit, build type, exact focused/full test results, benchmark elapsed time and counters. Mark Humble, Orin, DDS, rosbag, RViz rerun, and vehicle as `NOT_RUN` unless this session actually executes them.

- [ ] **Step 4: Commit the evidence**

  ```bash
  git add docs/superpowers/specs/2026-08-31-legged-grid-v1-planning-design.md
  git commit -m "docs: record legged local planning performance"
  ```

  Expected final history: the plan commit plus four narrowly scoped implementation/test commits, with no unrelated files staged or modified.
