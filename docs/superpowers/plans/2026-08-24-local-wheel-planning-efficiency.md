# Local Wheel Planning Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the wheeled local planner finish more certified paths inside the existing three-second deadline by separating measured-obstacle clearance from unknown terrain, enabling an elevation-offset-invariant flat fast path, eliminating only strictly proven redundant clearance scans, and exposing complete wheel work diagnostics.

**Architecture:** Keep the existing request-scoped SE(2) Hybrid A* graph, motion primitives and exact edge certification. Split local projection into hard support, measured occupied clearance and occupancy-frontier refinement fields; propagate wheel metrics through internal results and the existing ROS diagnostic key/value topic; then optimize the exact wheel evaluator without changing its safety or cost results.

**Tech Stack:** C++20, GoogleTest, ROS 2 Jazzy, `diagnostic_msgs/msg/DiagnosticArray`, Python pytest contract tests, colcon/CMake, existing 300 m exploration simulator.

**Spec:** `docs/superpowers/specs/2026-08-24-local-wheel-planning-efficiency-design.md`

## Global Constraints

- Preserve local occupancy `0.0..1.0`, with `NaN` unknown, `0.0` free and `1.0` occupied; the default occupied threshold remains `0.5`.
- Unknown occupancy and non-finite elevation remain non-traversable but do not consume `minimum_clearance_m` around their boundary.
- Only finite occupancy in `[threshold, 1.0]` is a measured occupied clearance source.
- Preserve global 1.0 m and local 0.2 m resolutions; do not downsample either map.
- Preserve the vehicle footprint, wheelbase, track width, 0.2 m minimum clearance, slope, roughness, local relief, underbody, curvature, dynamics, cancellation and three-second deadline checks.
- Keep `PlanMotion.action`, `PlannerDiagnostics.msg`, ROS topics, QoS, TF, odometry, map wire formats and capability YAML unchanged.
- Do not add timestamp, covariance, map-version, freshness or observation-age admission checks.
- Do not modify exploration candidate generation, candidate yaw variants, candidate ordering, retry policy or the number of planner calls.
- Do not reimplement `PlanningLatticeFrame`; commit `32903e8` and its three SE(2) regressions are already present.
- Use external build/install/log and run-artifact directories under `/home/kai/CodexDownloads/lunar_navigation`; never place generated artifacts in the repository.
- Every production edit is test-first and committed separately. Preserve unrelated worktree changes and stage only task-owned files.

## Parallel Execution Order

```text
Task 1 baseline
  ├─ Task 2 local projection semantics ─┐
  └─ Task 3 core wheel metrics ─────────┤
                                       ├─ Task 5 occupied-only wheel evaluator
Task 3 ── Task 4 ROS diagnostics ───────┘   (Task 4 may run beside Task 5)
                                             -> Task 6 elevation fast path
                                             -> Task 7 far-clearance proof
                                             -> Task 8 integration/full run
```

Tasks 2 and 3 use separate external build roots and may execute in parallel. Task 4 may execute in parallel with Task 5 after Task 3 lands. Tasks 5, 6 and 7 are serialized because they modify `anytime_wheel_planner.cpp` and its focused tests.

---

### Task 1: Freeze Source, Test and Runtime Baselines

**Files:**
- Read: `docs/superpowers/specs/2026-08-24-local-wheel-planning-efficiency-design.md`
- Read: `docs/superpowers/specs/2026-08-24-planner-core-incremental-optimization-design.md`
- Read: `docs/superpowers/plans/2026-08-24-request-scoped-se2-lattice.md`
- Evidence only: `/home/kai/CodexDownloads/lunar_navigation/local-wheel-efficiency-evidence/baseline/`

**Interfaces:**
- Consumes: clean branch containing design commit `68b24c7`, this committed implementation plan, the existing Jazzy toolchain and existing smoke planner log.
- Produces: immutable baseline commit, focused test results and the eleven-call outcome/timing table used by Task 8.

- [ ] **Step 1: Verify worktree identity and create the external evidence root**

Run:

```bash
cd /home/kai/CodexDownloads/lunar_navigation/lunar_pure_planner_orin-worktrees/jazzy-300m-exploration
git status --short --branch
git rev-parse HEAD
git merge-base --is-ancestor 68b24c7 HEAD
export LUNAR_WHEEL_EFFICIENCY_ROOT=/home/kai/CodexDownloads/lunar_navigation/local-wheel-efficiency-evidence
mkdir -p "$LUNAR_WHEEL_EFFICIENCY_ROOT/baseline"
git rev-parse HEAD > "$LUNAR_WHEEL_EFFICIENCY_ROOT/baseline/source-sha.txt"
```

Expected: branch `wip/exploration-readiness-20260824`, no source changes, design commit `68b24c7` is an ancestor, and `source-sha.txt` freezes the exact pre-implementation plan commit.

- [ ] **Step 2: Build a fresh Jazzy test baseline outside the repository**

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base "$LUNAR_WHEEL_EFFICIENCY_ROOT/baseline/log" build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_WHEEL_EFFICIENCY_ROOT/baseline/build" \
  --install-base "$LUNAR_WHEEL_EFFICIENCY_ROOT/baseline/install" \
  --packages-up-to lunar_pure_planner_ros \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
```

Expected: `lunar_pure_planner_core` and `lunar_pure_planner_ros` build successfully without repository-local `build/`, `install/` or `log/` directories.

- [ ] **Step 3: Record focused baseline tests**

Run:

```bash
ctest --test-dir "$LUNAR_WHEEL_EFFICIENCY_ROOT/baseline/build/lunar_pure_planner_core" \
  --output-on-failure \
  -R 'lunar_pure_planner_core_(local_terrain_projection|anytime_wheel_planner)_test'
ctest --test-dir "$LUNAR_WHEEL_EFFICIENCY_ROOT/baseline/build/lunar_pure_planner_ros" \
  --output-on-failure \
  -R '(request_diagnostics_test|pure_plan_motion_server_test)'
```

Expected: all currently stable focused tests pass. Record the known full-wheel long-range timeout separately; do not change production behavior to make a baseline-only test green.

- [ ] **Step 4: Freeze the eleven-call smoke evidence**

Run:

```bash
export LUNAR_BASELINE_PLANNER_LOG=/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_runs/live-smoke/run-8f44591d01ea4abd9a2d386598e6a51d/ros-logs/lunar_pure_planner_node_863686_1787544487870.log
test -f "$LUNAR_BASELINE_PLANNER_LOG"
rg 'request_id=jazzy-300m-20260824/candidate/(0|1|2|3|4|5|6|7|8|9|10) ' \
  "$LUNAR_BASELINE_PLANNER_LOG" \
  > "$LUNAR_WHEEL_EFFICIENCY_ROOT/baseline/first-eleven-planner-calls.txt"
test "$(wc -l < "$LUNAR_WHEEL_EFFICIENCY_ROOT/baseline/first-eleven-planner-calls.txt")" -eq 11
rg -c 'reason_code=TIMEOUT' "$LUNAR_WHEEL_EFFICIENCY_ROOT/baseline/first-eleven-planner-calls.txt"
rg -c 'reason_code=PLAN_FOUND' "$LUNAR_WHEEL_EFFICIENCY_ROOT/baseline/first-eleven-planner-calls.txt"
rg -c 'reason_code=NO_PATH' "$LUNAR_WHEEL_EFFICIENCY_ROOT/baseline/first-eleven-planner-calls.txt"
```

Expected: exactly eleven lines, `8` timeout, `2` plan found and `1` no path. Candidates 6 and 7 retain local-search times near 352 ms and 231 ms.

- [ ] **Step 5: Do not commit evidence artifacts**

Run:

```bash
git status --short
```

Expected: no repository change from Task 1. The evidence directory remains outside Git.

---

### Task 2: Separate Local Hard Feasibility, Occupied Clearance and Narrow-Band Refinement

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/local_terrain_projection.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/local_terrain_projection.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_core/test/local_terrain_projection_test.cpp`

**Interfaces:**
- Consumes: `MapSnapshot` float `occupancy` and `elevation` layers and `BuildCellAreaClearance(...)`.
- Produces: `LocalTerrainProjection::occupied`, occupied-only `clearance_m`, and occupancy-only `narrow_band_distance_m`; `BuildLocalTerrainProjection(...)` signature remains unchanged.

- [ ] **Step 1: Add RED tests for unknown, occupied and invalid cells**

Add these focused cases using the existing `Snapshot(...)` helper:

```cpp
TEST(LocalTerrainProjection,
     UnknownOccupancyIsHardInfeasibleWithoutMeasuredClearanceInflation) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const auto result = BuildLocalTerrainProjection(
      Snapshot(3U, {0.0F, nan, 0.0F}, {0.0F, 0.0F, 0.0F}));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.value->free_with_height,
            (std::vector<std::uint8_t>{1U, 0U, 1U}));
  EXPECT_EQ(result.value->occupied,
            (std::vector<std::uint8_t>{0U, 0U, 0U}));
  EXPECT_TRUE(std::ranges::all_of(
      result.value->clearance_m,
      [](const float value) { return std::isinf(value) && value > 0.0F; }));
  EXPECT_FLOAT_EQ(result.value->narrow_band_distance_m[0U], 0.5F);
  EXPECT_FLOAT_EQ(result.value->narrow_band_distance_m[1U], 0.0F);
  EXPECT_FLOAT_EQ(result.value->narrow_band_distance_m[2U], 0.5F);
}

TEST(LocalTerrainProjection,
     ThresholdOccupiedCellDrivesBothOccupiedClearanceAndNarrowBand) {
  const auto result = BuildLocalTerrainProjection(
      Snapshot(3U, {0.0F, 0.5F, 0.0F}, {0.0F, 0.0F, 0.0F}));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.value->occupied,
            (std::vector<std::uint8_t>{0U, 1U, 0U}));
  EXPECT_EQ(result.value->clearance_m,
            result.value->narrow_band_distance_m);
  EXPECT_FLOAT_EQ(result.value->clearance_m[0U], 0.5F);
  EXPECT_FLOAT_EQ(result.value->clearance_m[1U], 0.0F);
  EXPECT_FLOAT_EQ(result.value->clearance_m[2U], 0.5F);
}

TEST(LocalTerrainProjection,
     InvalidOccupancyAndMissingElevationAreNotOccupiedSources) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const auto result = BuildLocalTerrainProjection(
      Snapshot(4U, {0.0F, -0.1F, 1.1F, 0.0F},
               {0.0F, 0.0F, 0.0F, nan}));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.value->free_with_height,
            (std::vector<std::uint8_t>{1U, 0U, 0U, 0U}));
  EXPECT_EQ(result.value->occupied,
            (std::vector<std::uint8_t>{0U, 0U, 0U, 0U}));
  EXPECT_TRUE(std::ranges::all_of(
      result.value->clearance_m,
      [](const float value) { return std::isinf(value) && value > 0.0F; }));
  EXPECT_FLOAT_EQ(result.value->narrow_band_distance_m[1U], 0.0F);
  EXPECT_FLOAT_EQ(result.value->narrow_band_distance_m[2U], 0.0F);
}
```

- [ ] **Step 2: Build and verify RED**

Run:

```bash
export LUNAR_LOCAL_SEMANTICS_ROOT=/home/kai/CodexDownloads/lunar_navigation/local-wheel-efficiency-evidence/local-semantics
source /opt/ros/jazzy/setup.bash
colcon --log-base "$LUNAR_LOCAL_SEMANTICS_ROOT/log" build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_LOCAL_SEMANTICS_ROOT/build" \
  --install-base "$LUNAR_LOCAL_SEMANTICS_ROOT/install" \
  --packages-select lunar_pure_planner_core \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
```

Expected: compile failure because `LocalTerrainProjection` does not yet contain `occupied` or `narrow_band_distance_m`.

- [ ] **Step 3: Add the two projection fields and exact predicates**

Use this public internal structure:

```cpp
struct LocalTerrainProjection final {
  std::shared_ptr<const MapSnapshot> map;
  std::vector<std::uint8_t> free_with_height;
  std::vector<std::uint8_t> occupied;
  std::vector<float> clearance_m;
  std::vector<float> narrow_band_distance_m;
  std::vector<float> slope_rad;
  std::vector<float> roughness_m;
};
```

Add file-local predicates:

```cpp
[[nodiscard]] bool IsOccupiedInflationSource(
    const float value, const float threshold) noexcept {
  return std::isfinite(value) && value >= 0.0F && value <= 1.0F &&
         value >= threshold;
}

[[nodiscard]] bool IsNarrowBandSource(
    const float value, const float threshold) noexcept {
  return !IsFreeOccupancy(value, threshold);
}
```

- [ ] **Step 4: Build both distance transforms with controlled-work propagation**

Replace the single temporary hazard mask with:

```cpp
ControlledFill(&projection.occupied, count, std::uint8_t{0U}, control);
std::vector<std::uint8_t> narrow_band_mask(count, 0U);

for (std::size_t index = 0U; index < count; ++index) {
  if (ControlCheckDue(index)) {
    if (const auto stopped = StopReason(control); stopped.has_value()) {
      return {.reason_code = std::string{*stopped}};
    }
  }
  const bool free = IsFreeOccupancy(occupancy[index], occupancy_threshold);
  projection.free_with_height[index] = static_cast<std::uint8_t>(
      free && std::isfinite(elevation[index]));
  projection.occupied[index] = static_cast<std::uint8_t>(
      IsOccupiedInflationSource(occupancy[index], occupancy_threshold));
  narrow_band_mask[index] = static_cast<std::uint8_t>(
      IsNarrowBandSource(occupancy[index], occupancy_threshold));
}
```

Call `BuildCellAreaClearance(...)` first for `projection.occupied` and then for `narrow_band_mask`. After each call, propagate its exact `TIMEOUT` or `REQUEST_CANCELED` reason before moving its vector into the corresponding output field.

- [ ] **Step 5: Extend controlled-work and allocation regressions**

Update the existing deadline/cancel test with a controlled clock that completes the occupied transform and expires during the narrow-band transform. Change the large flat-map allocation assertion from `kWidth * 16U` to the explicit linear ceiling `kWidth * 24U`; this accounts for the second exact distance-transform workspace without permitting per-cell heap allocation.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
colcon --log-base "$LUNAR_LOCAL_SEMANTICS_ROOT/log" build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_LOCAL_SEMANTICS_ROOT/build" \
  --install-base "$LUNAR_LOCAL_SEMANTICS_ROOT/install" \
  --packages-select lunar_pure_planner_core \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
ctest --test-dir "$LUNAR_LOCAL_SEMANTICS_ROOT/build/lunar_pure_planner_core" \
  --output-on-failure -R lunar_pure_planner_core_local_terrain_projection_test
```

Expected: all local projection tests pass; no runtime artifact appears in Git.

- [ ] **Step 7: Commit the projection contract**

```bash
git add ros2_ws/src/lunar_pure_planner_core/src/shared/local_terrain_projection.hpp \
  ros2_ws/src/lunar_pure_planner_core/src/shared/local_terrain_projection.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/local_terrain_projection_test.cpp
git commit -m "feat: separate local occupied clearance from unknown terrain"
```

---

### Task 3: Propagate a Complete Wheel Metrics Snapshot Through Core Results

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/types/planning_request.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/planner.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/planner.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_core/test/dual_mode_planner_test.cpp`

**Interfaces:**
- Consumes: existing `WheelPlanResult` counters and `LocalStageResult`/`PlanningResult` status mapping.
- Produces: `WheelPlanningMetrics`, `LocalStageResult::wheel_metrics` and `PlanningResult::wheel_metrics`; pre-graph and non-wheel failures remain `nullopt`.

- [ ] **Step 1: Add RED result-propagation tests**

Reuse the existing `FlatTerrain()`, `Capability(...)`, `Primitive(...)`, `RequestTo(...)` and injected-clock fixtures. Add these exact cases:

```cpp
TEST(WheelPlanner, GraphConstructedFailurePreservesCompletedMetrics) {
  constexpr std::size_t kWidth = 120U;
  constexpr std::size_t kHeight = 120U;
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::vector<float>(kWidth * kHeight, 0.0F), 0.05);
  WheeledCapability capability = Capability(3.0, 3.0);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };
  WheelPlanRequest request = RequestTo(
      fixture, capability, 4.0, 3.0, 0.0, Pose(3.0, 3.0));
  const auto deadline = SteadyClock::time_point{} + 1s;
  std::size_t clock_calls = 0U;
  request.control.deadline = deadline;
  request.control.now = [&] {
    ++clock_calls;
    return clock_calls < 5U ? deadline - 1ms : deadline + 1ms;
  };

  const WheelPlanResult result = PlanWheel(request);
  EXPECT_EQ(result.status, LocalPlanStatus::kTimedOut);
  ASSERT_TRUE(result.wheel_metrics.has_value());
  EXPECT_GT(result.wheel_metrics->sweep_cell_checks, 0U);
  EXPECT_EQ(result.wheel_metrics->expanded_states,
            result.metrics.expanded_states);
}

TEST(WheelPlanner, PreGraphInputFailureHasNoWheelMetrics) {
  EXPECT_FALSE(PlanWheel(WheelPlanRequest{}).wheel_metrics.has_value());
}
```

In `dual_mode_planner_test.cpp`, inject a wheel `LocalStageResult` containing `edge_validation_evaluations=7` and verify the final `PlanningResult` retains it for both success and `NO_PATH`.

- [ ] **Step 2: Build and verify RED**

Run:

```bash
export LUNAR_WHEEL_METRICS_ROOT=/home/kai/CodexDownloads/lunar_navigation/local-wheel-efficiency-evidence/wheel-metrics
source /opt/ros/jazzy/setup.bash
colcon --log-base "$LUNAR_WHEEL_METRICS_ROOT/log" build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_WHEEL_METRICS_ROOT/build" \
  --install-base "$LUNAR_WHEEL_METRICS_ROOT/install" \
  --packages-select lunar_pure_planner_core \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
```

Expected: compile failure because the optional metric fields and type do not exist.

- [ ] **Step 3: Define the internal wheel metric transport type**

Add `WheelPlanningMetrics` next to `PlanningResult` with existing counters plus these rejection buckets:

```cpp
struct WheelPlanningMetrics final {
  std::uint64_t expanded_states{};
  std::size_t edge_validation_evaluations{};
  std::size_t edge_validation_cache_hits{};
  std::size_t broad_phase_rejects{};
  std::size_t full_certifications{};
  std::size_t full_invalidations{};
  std::size_t sweep_cell_checks{};
  std::size_t quantization_alias_states{};
  std::size_t quantized_state_reuses{};
  std::size_t quantized_endpoint_aliases{};
  std::size_t quantized_state_count{};
  std::size_t maximum_active_labels_per_key{};
  bool used_narrow_resolution{};
  double finest_xy_key_resolution_m{};
  std::size_t maximum_yaw_bins{};
  std::size_t ara_search_invocations{};
  std::size_t returned_edge_certificate_confirmations{};
  std::size_t mode_switch_edge_count{};
  std::size_t reverse_edge_count{};
  double start_heuristic_lower_bound{};
  bool has_certified_preferred_candidate{};
  std::size_t preferred_candidate_full_primitive_edge_count{};
  std::size_t preferred_candidate_terminal_connector_edge_count{};
  std::size_t preferred_candidate_certified_edge_count{};
  double preferred_candidate_cost{};
  std::size_t preferred_builder_invocations{};
  std::array<double, 5U> cost_components{};
  std::array<double, 5U> cost_scales{};
  std::size_t direct_unknown_or_unsupported_footprint_rejects{};
  std::size_t measured_obstacle_clearance_rejects{};
  std::size_t slope_or_roughness_rejects{};
  std::size_t relief_or_underbody_rejects{};
  std::size_t dynamics_or_primitive_shape_rejects{};
  std::size_t deadline_or_cancellation_interruptions{};
  std::size_t far_clearance_scan_skips{};
  std::size_t occupied_clearance_cell_checks{};
};
```

Add `std::optional<WheelPlanningMetrics> wheel_metrics` to `LocalStageResult`, `PlanningResult` and `WheelPlanResult`. Do not remove the existing direct `WheelPlanResult` fields in this task; populate the snapshot from them in one `decorate()` helper to minimize behavioral churn.

- [ ] **Step 4: Populate graph-created results and preserve pre-graph absence**

After `WheelSearchGraph` construction, make every solved, `NO_PATH`, timeout and canceled return pass through the existing `decorate()` function. Fill the optional snapshot there. Returns before graph construction leave the optional empty. Rejection buckets and far-clearance counters remain zero until Tasks 5 and 7 instrument them.

- [ ] **Step 5: Propagate through `PlanLocalDefault()` and `Planner::Plan()`**

In the wheeled branch set:

```cpp
.wheel_metrics = result.wheel_metrics,
```

Immediately after the local backend call, capture `local.wheel_metrics`. Attach it to all subsequent success, local failure, deadline override, cancellation and composition failure results. Do not attach wheel metrics to global failure, local projection failure, goal-field failure, legged or hopper results.

- [ ] **Step 6: Verify GREEN and run all core result tests**

Run:

```bash
colcon --log-base "$LUNAR_WHEEL_METRICS_ROOT/log" build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_WHEEL_METRICS_ROOT/build" \
  --install-base "$LUNAR_WHEEL_METRICS_ROOT/install" \
  --packages-select lunar_pure_planner_core \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
ctest --test-dir "$LUNAR_WHEEL_METRICS_ROOT/build/lunar_pure_planner_core" \
  --output-on-failure \
  -R 'lunar_pure_planner_core_(anytime_wheel_planner|dual_mode_planner)_test'
```

- [ ] **Step 7: Commit the core metric path**

```bash
git add ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/types/planning_request.hpp \
  ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/planner.hpp \
  ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.hpp \
  ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/src/planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/dual_mode_planner_test.cpp
git commit -m "feat: preserve wheel metrics through planner results"
```

---

### Task 4: Publish Conditional Wheel Metrics on Existing ROS Diagnostics

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/request_diagnostics.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_ros/test/request_diagnostics_test.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp`
- Modify: `tests/exploration/test_exploration_isolation.py`
- Modify: `VERIFICATION.md`

**Interfaces:**
- Consumes: `PlanningResult::wheel_metrics` from Task 3.
- Produces: one common `wheel_metrics_available` diagnostic key and conditional `wheel_*` values; action/message schemas remain unchanged.

- [ ] **Step 1: Add RED formatter tests for available and unavailable metrics**

```cpp
TEST(RequestDiagnostics, EmitsConditionalWheelMetricsWhenAvailable) {
  const PlanningResult result{
      .status = PlanningStatus::kTimedOut,
      .wheel_metrics = WheelPlanningMetrics{
          .expanded_states = 4U,
          .edge_validation_evaluations = 11U,
          .sweep_cell_checks = 29U,
      },
  };
  const auto message = MakeRequestDiagnostics(
      "wheel", PlatformType::kWheeled, EnvironmentMode::kLunarSurface,
      result);
  EXPECT_EQ(FindDiagnosticValue(message, "wheel_metrics_available"), "true");
  EXPECT_EQ(FindDiagnosticValue(
                message, "wheel_edge_validation_evaluations"), "11");
  EXPECT_EQ(FindDiagnosticValue(message, "wheel_sweep_cell_checks"), "29");
}

TEST(RequestDiagnostics, OmitsConditionalWheelMetricsBeforeGraphCreation) {
  const auto message = MakeRequestDiagnostics(
      "invalid", PlatformType::kWheeled, EnvironmentMode::kLunarSurface,
      PlanningResult{.status = PlanningStatus::kInvalidInput});
  EXPECT_EQ(FindDiagnosticValue(message, "wheel_metrics_available"), "false");
  EXPECT_TRUE(FindDiagnosticValue(
      message, "wheel_edge_validation_evaluations").empty());
}
```

- [ ] **Step 2: Add RED rolling propagation coverage**

Extend the rolling server fixture so its local planner returns `wheel_metrics.edge_validation_evaluations=9`. Verify the one published diagnostic has `wheel_metrics_available=true` and `wheel_edge_validation_evaluations=9` on success and on a post-local timeout/cancellation override.

- [ ] **Step 3: Build and verify RED**

Run:

```bash
export LUNAR_ROS_DIAGNOSTICS_ROOT=/home/kai/CodexDownloads/lunar_navigation/local-wheel-efficiency-evidence/ros-diagnostics
source /opt/ros/jazzy/setup.bash
colcon --log-base "$LUNAR_ROS_DIAGNOSTICS_ROOT/log" build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_ROS_DIAGNOSTICS_ROOT/build" \
  --install-base "$LUNAR_ROS_DIAGNOSTICS_ROOT/install" \
  --packages-up-to lunar_pure_planner_ros \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
```

Expected: new diagnostic key assertions fail.

- [ ] **Step 4: Preserve wheel metrics through rolling result reconstruction**

In `ExecuteRollingSurfaceWheel()`, copy `local.wheel_metrics` into the manually constructed `PlanningResult segment` and every result created after that local call. Do not attach it to failures before local `PlanWheel()` execution.

- [ ] **Step 5: Format the common and conditional key sets**

Always append:

```cpp
add_value("wheel_metrics_available",
          result.wheel_metrics.has_value() ? "true" : "false");
```

When available, append all `WheelPlanningMetrics` members with fixed `wheel_*` snake-case names. Serialize booleans as `true|false`, integral counters with `std::to_string`, and finite doubles with the existing `Decimal(...)` helper. Do not append conditional fields when unavailable.

- [ ] **Step 6: Replace exact key-count assertions with contract subsets**

Rename `ExpectTenKeyDiagnostic()` to `ExpectCommonDiagnosticKeys()`. Assert all existing 19 common keys plus `wheel_metrics_available`; separately assert the full conditional wheel set only when the availability value is `true`. Update `tests/exploration/test_exploration_isolation.py` so its ten explorer-consumed timing keys remain a required subset rather than the complete planner message. Update `VERIFICATION.md` from “exactly 10 keys” to “one common set plus conditional wheel metrics,” then verify the Chinese Markdown as UTF-8.

- [ ] **Step 7: Verify GREEN**

Run:

```bash
colcon --log-base "$LUNAR_ROS_DIAGNOSTICS_ROOT/log" build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_ROS_DIAGNOSTICS_ROOT/build" \
  --install-base "$LUNAR_ROS_DIAGNOSTICS_ROOT/install" \
  --packages-up-to lunar_pure_planner_ros \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
ctest --test-dir "$LUNAR_ROS_DIAGNOSTICS_ROOT/build/lunar_pure_planner_ros" \
  --output-on-failure -R '(request_diagnostics_test|pure_plan_motion_server_test)'
python3 -m pytest -q -p no:cacheprovider tests/exploration/test_exploration_isolation.py
iconv -f UTF-8 -t UTF-8 VERIFICATION.md >/dev/null
```

- [ ] **Step 8: Commit the ROS diagnostic contract**

```bash
git add ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp \
  ros2_ws/src/lunar_pure_planner_ros/src/request_diagnostics.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/request_diagnostics_test.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp \
  tests/exploration/test_exploration_isolation.py VERIFICATION.md
git commit -m "feat: publish conditional wheel planning diagnostics"
```

---

### Task 5: Use Measured Occupied Cells for Wheel Clearance and Keep Unknown as Direct Support Failure

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp`

**Interfaces:**
- Consumes: Task 2 `occupied`, occupied-only `clearance_m`, `narrow_band_distance_m`; Task 3 rejection metric buckets.
- Produces: occupied-only broad/exact clearance, direct unknown/support rejection, closed map-extent behavior and preserved narrow quantization.

- [ ] **Step 1: Add RED wheel-safety tests**

Add fixtures for:

```cpp
TEST(WheelPlanner,
     AcceptsKnownFreeFootprintAdjacentToUnknownWithoutClearanceInflation);
TEST(WheelPlanner,
     RejectsUnknownOrNanElevationWhenFootprintOrWheelSupportTouches);
TEST(WheelPlanner,
     ClosedMapExtentAllowsExactTouchAndRejectsToleranceCrossingOnAllSides);
TEST(WheelPlanner,
     PreservesOccupiedCollisionAndExactMinimumClearanceBoundary);
TEST(WheelPlanner,
     UnknownFrontierRetainsNarrowQuantizationWithoutObstacleClearance);
```

The adjacent-unknown fixture must set `minimum_clearance_m=0.2`, keep the entire footprint on known-free cells, place unknown immediately outside the footprint, and require success. The overlap fixture moves the same unknown into a footprint-intersected or bilinear wheel-support cell and requires rejection. The occupied fixture substitutes `0.5F` at the same location and retains the exact clearance boundary behavior.

- [ ] **Step 2: Run the focused tests and verify RED**

Run against the merged Task 2/3 source:

```bash
export LUNAR_WHEEL_EVALUATOR_ROOT=/home/kai/CodexDownloads/lunar_navigation/local-wheel-efficiency-evidence/wheel-evaluator
source /opt/ros/jazzy/setup.bash
colcon --log-base "$LUNAR_WHEEL_EVALUATOR_ROOT/log" build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_WHEEL_EVALUATOR_ROOT/build" \
  --install-base "$LUNAR_WHEEL_EVALUATOR_ROOT/install" \
  --packages-select lunar_pure_planner_core \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
```

Expected: adjacent unknown is rejected by current combined hazard clearance and the map-edge contact fixture is rejected by the clearance-expanded bounds check.

- [ ] **Step 3: Validate the expanded terrain projection contract**

Update `ValidTerrain(...)` to require `occupied.size()` and `narrow_band_distance_m.size()` to equal `map->cell_count()`, while retaining the existing size checks. `IsNarrow()` must read `narrow_band_distance_m`; it must never use occupied-only `clearance_m` for search refinement.

- [ ] **Step 4: Replace combined hazard rows and integral with occupied-only structures**

In `WheelSearchGraph` construction, populate:

```cpp
const bool occupied = terrain_.occupied[index] != 0U;
if (occupied) {
  occupied_by_row_[y].push_back(static_cast<std::int32_t>(x));
}
```

Rename `hazard_integral_`, `hazards_by_row_` and `HasHazardInCells(...)` to occupied-specific names. Keep `complex_terrain_integral_` based on `free_with_height`, finite elevation/slope/roughness and the existing terrain conditions.

- [ ] **Step 5: Separate raw footprint bounds from occupied-clearance windows**

Introduce file-local helpers that:

```cpp
FootprintWithinClosedMapExtent(min_x, min_y, max_x, max_y)
ClipOccupiedWindow(min_x - margin, min_y - margin,
                   max_x + margin, max_y + margin)
```

The first accepts exact contact within `kTolerance` and rejects a crossing greater than tolerance symmetrically on all sides. The second clips to valid raster indices and never treats outside-map space as occupied. Broad inset-disk clearance rejection iterates only `occupied_by_row_`.

- [ ] **Step 6: Split exact occupied clearance from hard support**

In `EvaluateFullExact()`:

- use `occupied_by_row_` only for exact polygon-to-cell collision and `minimum_clearance_m`;
- retain `free_with_height`, finite elevation, slope, roughness, relief, underbody, adjacent-step and wheel-support checks over the actual footprint/support domain;
- increment the Task 3 rejection bucket matching the first rejecting stage without changing validity or reason mapping;
- increment `occupied_clearance_cell_checks` only for measured occupied polygon-distance evaluations.

- [ ] **Step 7: Verify focused and stable wheel tests**

Run:

```bash
colcon --log-base "$LUNAR_WHEEL_EVALUATOR_ROOT/log" build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_WHEEL_EVALUATOR_ROOT/build" \
  --install-base "$LUNAR_WHEEL_EVALUATOR_ROOT/install" \
  --packages-select lunar_pure_planner_core \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
"$LUNAR_WHEEL_EVALUATOR_ROOT/build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test" \
  --gtest_filter=-WheelPlanner.GlobalRouteWithEightMeterRollingHorizonReaches750MeterGoalThroughRandomObstacles
```

Expected: every stable wheel test passes; the known long-range baseline remains separately classified.

- [ ] **Step 8: Commit occupied-only wheel semantics**

```bash
git add ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp
git commit -m "fix: separate wheel occupied clearance from terrain support"
```

---

### Task 6: Make the Flat-Terrain Fast Path Invariant to Absolute Elevation

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp`

**Interfaces:**
- Consumes: Task 5 exact support and occupied-clearance pipeline.
- Produces: strict constant-elevation equivalence; non-flat terrain continues through full exact support evaluation.

- [ ] **Step 1: Add RED constant-offset and support-domain tests**

Add a helper that compares two successful results:

```cpp
void ExpectSamePlanExceptZOffset(const WheelPlanResult& baseline,
                                 const WheelPlanResult& shifted,
                                 const double dz) {
  ASSERT_EQ(baseline.status, shifted.status);
  ASSERT_EQ(baseline.reason_code, shifted.reason_code);
  ASSERT_EQ(baseline.selected_goal_index, shifted.selected_goal_index);
  ASSERT_EQ(baseline.trajectory.size(), shifted.trajectory.size());
  EXPECT_DOUBLE_EQ(baseline.cost, shifted.cost);
  EXPECT_EQ(baseline.cost_components, shifted.cost_components);
  EXPECT_EQ(baseline.cost_scales, shifted.cost_scales);
  for (std::size_t index = 0U; index < baseline.trajectory.size(); ++index) {
    EXPECT_DOUBLE_EQ(baseline.trajectory[index].pose.position_m.x,
                     shifted.trajectory[index].pose.position_m.x);
    EXPECT_DOUBLE_EQ(baseline.trajectory[index].pose.position_m.y,
                     shifted.trajectory[index].pose.position_m.y);
    EXPECT_NEAR(shifted.trajectory[index].pose.position_m.z -
                    baseline.trajectory[index].pose.position_m.z,
                dz, 1.0e-9);
    EXPECT_NEAR(TrajectoryYaw(baseline.trajectory[index]),
                TrajectoryYaw(shifted.trajectory[index]), 1.0e-12);
  }
}
```

Add `ConstantElevationOffsetsPreserveCertifiedPlanAndCost` for `0.0`, `10.0` and `-3.0` m. Add `FlatFastPathProofIncludesWheelBilinearSupportHalo` with a legal capability whose wheel contact/halo extends beyond the footprint AABB and a `NaN` elevation in that support halo; the path must remain rejected or forced through exact validation.

- [ ] **Step 2: Run and verify RED**

Use the Task 5 build root and run:

```bash
"$LUNAR_WHEEL_EVALUATOR_ROOT/build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test" \
  --gtest_filter='WheelPlanner.ConstantElevationOffsetsPreserveCertifiedPlanAndCost:WheelPlanner.FlatFastPathProofIncludesWheelBilinearSupportHalo'
```

Expected: non-zero constant elevation does not use the current fast path or the support-halo fixture exposes the current footprint-only proof domain.

- [ ] **Step 3: Remove only the absolute elevation-zero complexity condition**

Change the complex-cell predicate from:

```cpp
!std::isfinite(elevations[index]) || elevations[index] != 0.0F ||
```

to:

```cpp
!std::isfinite(elevations[index]) ||
```

Keep every `free_with_height`, finite slope/roughness and exact non-zero slope/roughness condition. Do not introduce a “small enough slope” tolerance in this task.

- [ ] **Step 4: Expand the fast-path proof domain**

For each sweep sample, compute the AABB union of the transformed footprint and all four wheel contact points. Expand it by the one-cell bilinear interpolation halo before `HasComplexTerrainInCells(...)`. If any part of the proof domain is outside the map or complex, execute the full existing terrain path.

- [ ] **Step 5: Verify all terrain safety regressions**

Run the two new tests plus existing continuous-slope, step-discontinuity, underbody bump, rotation and arc filters. Then run the complete stable wheel filter from Task 5.

- [ ] **Step 6: Commit the elevation-invariant fast path**

```bash
git add ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp
git commit -m "perf: certify flat wheel terrain independent of elevation offset"
```

---

### Task 7: Elide Far Occupied-Clearance Scans Only After an Exact Equivalence Proof

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp`

**Interfaces:**
- Consumes: occupied-only `clearance_m`, exact scan window and Task 3 observability counters.
- Produces: optional far-clearance skip with identical validity, cost, used clearance and deterministic labels; otherwise exact scanning remains production behavior.

- [ ] **Step 1: Add RED geometric-proof boundary tests**

Test a pure helper over occupied cells positioned at, just inside and just outside the exact scan window. The proof threshold is:

```cpp
const double exact_scan_margin = std::max(
    capability.minimum_clearance_m,
    footprint_radius_m + 2.0 * map_resolution_m);
const double proof_threshold =
    footprint_radius_m + exact_scan_margin +
    std::numbers::sqrt2 * 0.5 * map_resolution_m;
```

For every test pose and yaw, `proof == true` must imply that the exact occupied-cell iterator visits zero cells in the complete polygon scan window. Equality at the threshold is not a proof.

- [ ] **Step 2: Add planner-level equivalence fixtures**

Add:

```cpp
TEST(WheelPlanner, FarOccupiedCellMatchesNoNearbyOccupiedBaseline);
TEST(WheelPlanner, NearOccupiedEdgeNeverUsesFarClearanceSkip);
TEST(WheelPlanner, FarClearanceSkipRetainsUnknownFootprintRejection);
TEST(WheelPlanner, FarClearanceSkipPreservesTwentyRunDeterminism);
```

The first compares a map with no occupied cell in any scan window against a map with an occupied cell beyond every proof window. Require identical status, reason, selected goal, trajectory, cost, five cost components/scales, quantized state count and maximum active labels. Require the optimized case to report `far_clearance_scan_skips > 0` and no increase in `occupied_clearance_cell_checks`.

- [ ] **Step 3: Implement a conservative all-sample proof**

Make `AssessBroadPhase()` set `clearance_proven` only if every interpolated sample's occupied-only `clearance_m` is finite or positive infinity and strictly exceeds the proof threshold. Unknown occupancy, invalid elevation and map bounds do not participate in this proof and remain exact support checks.

- [ ] **Step 4: Integrate the skip without changing the fallback value**

When `clearance_proven` is true, skip only the occupied polygon-distance loop and assign exactly the same `narrow_threshold` used by the current “no occupied cell in scan window” path. Continue every footprint, wheel support, terrain and dynamics stage. Increment `far_clearance_scan_skips`; do not change edge validity or reason mapping.

- [ ] **Step 5: Enforce the hard equivalence gate**

Run the new proof/equivalence tests and the full stable wheel suite. If any validity, cost component, selected label, trajectory or deterministic ordering differs, remove the production skip branch, leave exact scanning active, keep only useful proof tests/counters, and record `FAR_CLEARANCE_SKIP_NOT_ENABLED` in the validation document. A reduction in wall time alone cannot override this gate.

- [ ] **Step 6: Commit only if strict equivalence is green**

If enabled:

```bash
git add ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp
git commit -m "perf: elide proven far wheel clearance scans"
```

If not enabled, remove the production skip branch, retain the equivalence and boundary regressions, stage the same two files, and commit with `git commit -m "test: prove far wheel clearance scan boundaries"`. Task 8 must then record `FAR_CLEARANCE_SKIP_NOT_ENABLED`.

---

### Task 8: Integrate, Verify SE(2), Rebuild Jazzy and Run the 300 m Acceptance

**Files:**
- Modify: `docs/superpowers/plans/2026-08-24-request-scoped-se2-lattice.md` only to mark steps backed by fresh evidence
- Create: `docs/validation/2026-08-24-local-wheel-planning-efficiency.md`
- Modify when required by actual diagnostic output: `README.md`
- No modification: `ros2_ws/src/lunar_pure_exploration_*`, exploration launch/config, controller source or ROS messages

**Interfaces:**
- Consumes: Tasks 2-7 and the fixed seed `20260824` simulation operator.
- Produces: fresh source/test/build/runtime evidence, eleven-call comparison, full exploration result and an explicit Orin readiness boundary.

- [ ] **Step 1: Inspect integration history and diff**

Run:

```bash
git status --short --branch
git log --oneline 68b24c7..HEAD
git diff 68b24c7..HEAD --stat
git diff 68b24c7..HEAD -- \
  ros2_ws/src/lunar_pure_exploration_core \
  ros2_ws/src/lunar_pure_exploration_ros
```

Expected: no exploration candidate or retry behavior changes.

- [ ] **Step 2: Build the complete Jazzy test closure in one clean external overlay**

```bash
export LUNAR_WHEEL_FINAL_ROOT=/home/kai/CodexDownloads/lunar_navigation/local-wheel-efficiency-evidence/final
source /opt/ros/jazzy/setup.bash
colcon --log-base "$LUNAR_WHEEL_FINAL_ROOT/test-log" build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_WHEEL_FINAL_ROOT/test-build" \
  --install-base "$LUNAR_WHEEL_FINAL_ROOT/test-install" \
  --packages-up-to lunar_pure_planner_ros lunar_pure_exploration_sim \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
colcon --log-base "$LUNAR_WHEEL_FINAL_ROOT/ctest-log" test \
  --build-base "$LUNAR_WHEEL_FINAL_ROOT/test-build" \
  --install-base "$LUNAR_WHEEL_FINAL_ROOT/test-install" \
  --packages-select lunar_pure_planner_core lunar_pure_planner_ros \
  --return-code-on-test-failure \
  --event-handlers console_direct+
colcon test-result \
  --test-result-base "$LUNAR_WHEEL_FINAL_ROOT/test-build" --verbose
```

Expected: all selected core/ROS tests pass. Run the full wheel target separately and classify only the already known 750 m baseline if it persists.

- [ ] **Step 3: Reverify the existing request-scoped SE(2) implementation**

```bash
"$LUNAR_WHEEL_FINAL_ROOT/test-build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test" \
  --gtest_filter='WheelPlanner.PlansMultiplePrimitivesFromArbitraryTranslatedStart:WheelPlanner.PlansMultiplePrimitivesFromArbitrarySE2Start:WheelPlanner.RigidTransformPreservesRequestRelativeTrajectory'
```

Expected: all three pass. Mark only corresponding plan steps as completed; do not mark ROS/RViz steps without runtime evidence.

- [ ] **Step 4: Run repository and diagnostic contract checks**

```bash
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q -p no:cacheprovider \
  tests/foundation/test_repository_boundaries.py \
  tests/exploration/test_exploration_isolation.py \
  tests/launch/test_jazzy_300m_exploration_sim.py
git diff --check
```

- [ ] **Step 5: Build the production simulation closure without test binaries**

```bash
colcon --log-base "$LUNAR_WHEEL_FINAL_ROOT/prod-log" build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_WHEEL_FINAL_ROOT/prod-build" \
  --install-base "$LUNAR_WHEEL_FINAL_ROOT/prod-install" \
  --packages-up-to lunar_pure_planner_ros lunar_pure_wheeled_controller lunar_pure_exploration_sim \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF
```

Expected: every operator-required package resolves from this one overlay and reports the same `CMAKE_BUILD_TYPE`.

- [ ] **Step 6: Run a bounded same-seed headless smoke and compare the first eleven calls**

```bash
export LUNAR_JAZZY_OVERLAY="$LUNAR_WHEEL_FINAL_ROOT/prod-install/setup.bash"
./scripts/run_jazzy_300m_exploration_sim.sh \
  --seed 20260824 --no-rviz --max-wall-seconds 180
```

If the bounded smoke ends by wall guard after producing at least eleven planner diagnostics, treat the run as smoke evidence only, not exploration completion. Extract candidates 0-10 from its planner log and require:

- candidates 6 and 7 remain `PLAN_FOUND`;
- no safety oracle fixture becomes newly accepted except the approved adjacent-unknown/boundary cases;
- timeout count is no more than two;
- every graph-created wheel result has `wheel_metrics_available=true` and non-fabricated work counters.

- [ ] **Step 7: Run the full 300 m exploration with RViz**

```bash
export LUNAR_JAZZY_OVERLAY="$LUNAR_WHEEL_FINAL_ROOT/prod-install/setup.bash"
./scripts/run_jazzy_300m_exploration_sim.sh \
  --seed 20260824 --max-wall-seconds 21600
```

Expected acceptance:

- operator reports `outcome=COMPLETED` and validates `summary.json`;
- exploration completion reason is no reachable frontier;
- coverage is recorded but is not tested against a completion threshold;
- global resolution remains 1.0 m, local resolution 0.2 m and speed multiplier 20.0;
- RViz uses valid frame IDs and receives a non-empty certified reference;
- result directory contains non-empty summary/trajectory/timing artifacts;
- wall guard, operator interrupt or launch exit is reported as incomplete, never as completion.

- [ ] **Step 8: Write the validation record and readiness boundary**

Create `docs/validation/2026-08-24-local-wheel-planning-efficiency.md` with:

- source SHA and exact build/test commands;
- baseline and post-change first-eleven table;
- global/local p50/p95 and timeout counts;
- wheel expanded/validation/cache/broad/full/sweep/clearance/rejection counters;
- far-clearance status: enabled with equivalence evidence or exact retained;
- SE(2) test results;
- full 300 m run directory, terminal reason and coverage statistic;
- `Jetson AGX Orin / ROS 2 Humble: NOT_RUN` until separately executed on target.

- [ ] **Step 9: Commit documentation and final evidence references**

```bash
git add docs/superpowers/plans/2026-08-24-request-scoped-se2-lattice.md \
  docs/validation/2026-08-24-local-wheel-planning-efficiency.md README.md
git diff --cached --check
git commit -m "docs: record local wheel planning efficiency evidence"
```

Do not stage `README.md` if no actual operator contract text changed. Do not add runtime result files or external build artifacts.

- [ ] **Step 10: Final scope audit**

```bash
git status --short --branch
git log --oneline 68b24c7..HEAD
git diff 68b24c7..HEAD --name-only
```

Expected: only planner-core, planner-ROS diagnostics, contract tests and documentation files from this plan changed; no exploration candidate behavior, controller, Action/message schema or repository boundary change is present.
