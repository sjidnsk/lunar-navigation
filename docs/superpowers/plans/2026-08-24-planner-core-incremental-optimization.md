# Planner Core Incremental Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the existing pure planner so every request uses one end-to-end 1 s target, 2 s SLA boundary and 3 s hard deadline; wheeled search retains certified continuous primitive endpoints, safely crosses the 750 m regression bottleneck, and meets the agreed correctness, safety and performance gates without replacing the planner stack.

**Architecture:** Keep the request-scoped SE(2) frame and certified scaled-primitive terminal connector. Repair global occupied-cell-area geometry and shared ARA* first, then replace snapped wheeled interior states with bounded-label Hybrid A*. Rolling local goals become one deterministic, bounded multi-goal search. Runtime work becomes sparse and revision-keyed, while exact footprint, terrain and dynamics certification remains mandatory for every returned edge.

**Tech Stack:** C++20, GoogleTest, Python 3/pytest, CMake/ament, ROS 2 Jazzy, colcon, RViz2.

**Spec:** `docs/superpowers/specs/2026-08-24-planner-core-incremental-optimization-design.md`

## Global Constraints

- Keep `ros2_ws/src/lunar_planning_msgs/action/PlanMotion.action` and `PlannerDiagnostics.msg` byte-for-byte unchanged.
- Keep the exact external start pose, request-scoped SE(2) frame, exact certified terminal connector, map/odometry/TF contracts and platform capability YAML semantics.
- Never relax footprint sweep, minimum clearance, wheel support, elevation, slope, roughness, obstacle relief, underbody, curvature, velocity, acceleration, braking or motion-mode checks.
- The time contract is exact: `<1 s` target met, `[1 s,2 s)` slow, `[2 s,3 s)` SLA failed but search continues, and `>=3 s` hard timeout. Only the 3 s boundary is a deadline.
- A reference completed in `[2 s,3 s)` remains `NEW_REFERENCE_AVAILABLE` with `reason_code=PLAN_FOUND_LATE`. A reference not fully certified and committed before 3 s is never published.
- Remove configurable global/local/action planning budgets. A rolling mission may last while the rover moves, but every cold or warm planning cycle has the same fixed 3 s hard deadline and its own latency sample.
- Preserve deterministic ordering. Guidance may break equal anchor keys only; it may not precede `g + epsilon * h_anchor`.
- Do not add cross-request serialized OPEN/CLOSED sessions, background refinement, unconstrained splines, Nav2/OMPL replacement, controller changes, map snapping or test-only safety bypasses.
- Treat `types/planner_io.hpp`, `wheel/wheel_planner.cpp` and `shared/projection_cache.cpp` as the legacy pipeline unless an explicit task below names them. Do not accidentally wire the active planner through legacy types.
- Put every build/log/install tree outside the worktree. Test-enabled evidence and deployment `BUILD_TESTING=OFF` evidence are separate.
- Preserve unrelated changes. Never reset, overwrite, or include them in a task commit.

## Parallel Execution Contract

The coordinator uses `superpowers:using-git-worktrees` before Task 1. The integration worktree is:

```text
/home/kai/CodexDownloads/lunar_navigation/.worktrees/planner-core-opt  branch feat/planner-core-opt
```

Worker worktrees use these fixed sibling paths:

```text
/home/kai/CodexDownloads/lunar_navigation/.worktrees/planner-opt-w1-timing        branch feat/planner-opt-w1-timing
/home/kai/CodexDownloads/lunar_navigation/.worktrees/planner-opt-w1-global-safety branch feat/planner-opt-w1-global-safety
/home/kai/CodexDownloads/lunar_navigation/.worktrees/planner-opt-w1-ara           branch feat/planner-opt-w1-ara
/home/kai/CodexDownloads/lunar_navigation/.worktrees/planner-opt-w2-portals       branch feat/planner-opt-w2-portals
/home/kai/CodexDownloads/lunar_navigation/.worktrees/planner-opt-w2-hybrid        branch feat/planner-opt-w2-hybrid
/home/kai/CodexDownloads/lunar_navigation/.worktrees/planner-opt-w2-projection    branch feat/planner-opt-w2-projection
/home/kai/CodexDownloads/lunar_navigation/.worktrees/planner-opt-w3-certification branch feat/planner-opt-w3-certification
/home/kai/CodexDownloads/lunar_navigation/.worktrees/planner-opt-w3-cache         branch feat/planner-opt-w3-cache
/home/kai/CodexDownloads/lunar_navigation/.worktrees/planner-opt-w3-ros           branch feat/planner-opt-w3-ros
```

At most three workers run concurrently because the coordinator occupies the fourth agent slot. Each worker owns only the files listed in its task, commits on its own branch, and reports the commit SHA plus RED/GREEN evidence. The coordinator reviews and cherry-picks; workers never merge into the integration branch themselves.

| Wave | Concurrent tasks | Required base | Integration order |
| --- | --- | --- | --- |
| 0 | Task 1 only | approved design commit | commit baseline scaffold |
| 1 | Tasks 2, 3, 4 | Task 1 commit | timing, global safety, ARA* |
| Gate 1 | Task 5 only | all Wave 1 commits | full core/ROS regression |
| 2 | Tasks 6, 7, 8 | Task 5 commit | portals, projection, Hybrid A* |
| Gate 2 | Task 9 only | all Wave 2 commits | full core regression |
| 3 | Task 10 only | Task 9 commit | multi-goal integration and 750 m correctness |
| 4 | Tasks 11, 12, 13 | Task 10 commit | certification, cache, ROS |
| Gate 3 | Task 14 only | all Wave 4 commits | cache/certification wiring and performance |
| Final | Task 15 only | Task 14 commit | full verification, RViz and docs |

If a worker discovers that it must edit another worker's owned file, it stops and messages the coordinator. The coordinator either moves that edit to the integration gate or explicitly reassigns file ownership. It does not accept overlapping parallel edits.

---

### Task 1: Freeze the Reproducible Baseline and Performance Recorder

**Owner:** Coordinator, sequential Wave 0.

**Files:**
- Create: `ros2_ws/src/lunar_pure_planner_core/test/wheel_long_range_fixture.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp`
- Create: `tools/measure_planner_performance.py`
- Create: `tests/test_measure_planner_performance.py`
- Create: `docs/validation/planner-core-optimization.md`

**Interfaces:**
- `WheelLongRangeFixture::Run750MeterRollingScenario()` returns one structured run record containing success, failed segment, elapsed times, expanded/generated states, edge evaluations, state labels and sweep-cell checks.
- `measure_planner_performance.py` runs one named GoogleTest repeatedly, parses one JSON metrics line per repetition, computes p50/p95/max, and exits nonzero on missing or malformed metrics.
- The existing GoogleTest name remains `WheelPlanner.GlobalRouteWithEightMeterRollingHorizonReaches750MeterGoalThroughRandomObstacles`.

- [ ] **Step 1: Extract the existing 750 m fixture without changing its map, seed, goal, horizon, primitive set or deadline**

Move the test's map construction and rolling loop into the new header. Emit exactly one line per run:

```cpp
std::cout << "[planner-metrics] {\"scenario\":\"750m\","
          << "\"success\":" << (record.success ? "true" : "false") << ','
          << "\"failed_segment\":" << record.failed_segment << ','
          << "\"elapsed_ms\":" << record.elapsed_ms << ','
          << "\"expanded_states\":" << record.expanded_states << ','
          << "\"edge_evaluations\":" << record.edge_evaluations << ','
          << "\"state_labels\":" << record.state_labels << ','
          << "\"sweep_cell_checks\":" << record.sweep_cell_checks << "}\n";
```

The GoogleTest asserts the same final goal and fewer than 200 rolling segments. It must not shift the target or widen an obstacle gap.

- [ ] **Step 2: Add parser unit tests before writing the recorder**

Cover a valid line, missing key, malformed JSON, nonzero child exit, ten samples, and nearest-rank p95 selection. Run:

```bash
python3 -m pytest -q tests/test_measure_planner_performance.py
```

Expected RED: import or attribute failure because `tools/measure_planner_performance.py` does not yet exist.

- [ ] **Step 3: Implement the recorder and make its unit tests green**

Use `completed = subprocess.run(command, check=False, text=True, capture_output=True)` once per repetition, where `command` is the binary plus the exact `--gtest_filter` argument. Require the exact keys printed above, preserve every raw sample in the JSON report, and compute p95 as `sorted_samples[ceil(0.95*n)-1]`. Do not infer success from process output when the child exit code is nonzero.

Run:

```bash
python3 -m pytest -q tests/test_measure_planner_performance.py
```

Expected GREEN: all parser and percentile tests pass.

- [ ] **Step 4: Build and record the unchanged baseline**

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-opt/baseline/log build \
  --base-paths ros2_ws/src \
  --packages-select lunar_pure_planner_core \
  --build-base /tmp/lunar-planner-opt/baseline/build \
  --install-base /tmp/lunar-planner-opt/baseline/install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
/tmp/lunar-planner-opt/baseline/build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test \
  --gtest_filter=-WheelPlanner.GlobalRouteWithEightMeterRollingHorizonReaches750MeterGoalThroughRandomObstacles
/tmp/lunar-planner-opt/baseline/build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test \
  --gtest_filter=WheelPlanner.GlobalRouteWithEightMeterRollingHorizonReaches750MeterGoalThroughRandomObstacles
```

Expected: every non-750 m wheel test passes; the unchanged 750 m test fails at the known rolling bottleneck and prints a valid metrics line. Record the actual host, compiler, build type, pass count and metrics in `docs/validation/planner-core-optimization.md`. Do not copy historical numbers when the live run differs.

- [ ] **Step 5: Commit the baseline scaffold**

```bash
git add ros2_ws/src/lunar_pure_planner_core/test/wheel_long_range_fixture.hpp \
  ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp \
  tools/measure_planner_performance.py tests/test_measure_planner_performance.py \
  docs/validation/planner-core-optimization.md
git commit -m "test: freeze planner optimization baseline"
```

### Task 2: Implement the Fixed End-to-End Request Time Contract

**Owner:** Wave 1 timing worker.

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/planning_timing.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/planning_timing.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/types/planning_request.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/planner.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/planning_timing_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/dual_mode_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/request_diagnostics.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/message_conversion.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/request_diagnostics_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/message_conversion_test.cpp`
- Modify: `config/pure_planner.yaml`
- Modify: `tests/test_launch_contract.py`

**Interfaces:**
- Add `RequestTimingPolicy`, `RequestLatencyClass`, `PlannerPhase`, `PlannerProgress` and `ProgressFn` to `planning_timing.hpp`.
- Add `std::optional<SteadyClock::time_point> request_started_at` and `ProgressFn progress` to `PlanningRequest`.
- Extend `PlannerCallTiming` with snapshot/projection, global, local-goal, local-search, certification, output and total durations while retaining `global_elapsed`, `local_elapsed` and their call counts for compatibility.
- Remove `AnytimePlannerConfig::global_budget` and `output_reserve`; default `stop_after_first_solution` becomes `true` for first-reference behavior.
- Do not alter ROS message definitions.

- [ ] **Step 1: Add exact-boundary fake-clock tests**

Add table-driven coverage for `999 ms`, `1000 ms`, `1999 ms`, `2000 ms`, `2999 ms` and `3000 ms`. Add backend tests proving that controls received by global, local and composition all carry the same `started + 3 s` deadline. Add ROS finalization tests proving a 2.5 s certified result becomes `PLAN_FOUND_LATE` and a result finalized at exactly 3 s becomes `TIMEOUT` with no reference.

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-opt/w1-timing/log build \
  --base-paths ros2_ws/src \
  --packages-select lunar_planning_msgs lunar_pure_planner_core lunar_pure_planner_ros \
  --build-base /tmp/lunar-planner-opt/w1-timing/build \
  --install-base /tmp/lunar-planner-opt/w1-timing/install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Expected RED: old tests still observe 150 ms/950 ms stage deadlines; 1 s and 2 s boundary tests time out instead of continuing; late success is not classified.

- [ ] **Step 2: Implement one immutable timing window**

Use this public shape and exact comparisons:

```cpp
enum class RequestLatencyClass : std::uint8_t {
  kTargetMet,
  kTargetMissed,
  kSlaMissed,
  kHardTimeout,
};

struct RequestTimingPolicy final {
  static constexpr auto kTarget = std::chrono::seconds{1};
  static constexpr auto kSla = std::chrono::seconds{2};
  static constexpr auto kHard = std::chrono::seconds{3};
  SteadyClock::time_point started_at;
  SteadyClock::time_point target_milestone;
  SteadyClock::time_point sla_milestone;
  SteadyClock::time_point hard_deadline;
};
```

`Planner::Plan()` reads `*request_started_at` when present and calls `now()` only when it is absent, constructs one policy, and passes only `hard_deadline` to every controlled stage. Do not use `optional::value_or(now())`, because `value_or` would eagerly consume a fake-clock tick. Remove the 1 s total, 150 ms global, 50 ms reserve and backend return-reserve termination paths. A clock value equal to a milestone belongs to the later class.

- [ ] **Step 3: Keep output construction inside the hard boundary**

The ROS worker records its start before `InputStore::Capture()`, passes that time into `PlanningRequest`, then performs result conversion and final active-request checks before classification. `MakeOutputs()` must no longer zero a valid core total. At final commit:

```cpp
OutputBundle* committed = &normal;
if (finalized >= policy.hard_deadline) {
  committed = &timeout;
} else if (finalized >= policy.sla_milestone &&
           normal.result.status == PlanningStatus::kSuccess &&
           normal.result.reference.has_value()) {
  normal.result.reason_code = "PLAN_FOUND_LATE";
  normal.action_result->reason_code = "PLAN_FOUND_LATE";
}
```

Cancellation/replacement wins over late success. Populate ROS diagnostic-array values `latency_class`, phase milliseconds, and existing totals. Populate `PlannerDiagnostics.warning_codes` with `TARGET_MISSED` at/after 1 s and `PLANNING_SLA_MISSED` at/after 2 s.

- [ ] **Step 4: Remove configurable planning deadlines**

Delete `rolling_global_budget_ms`, `rolling_local_budget_ms` and `rolling_action_timeout_s` from the parameter struct, declarations, validation, YAML and launch-contract expectations. The rolling loop stops only on final-goal completion, cancellation/replacement, invalidation or planning failure. Each actual planning cycle constructs a fresh `RequestTimingPolicy`; polling and rover travel do not consume a planning cycle's clock.

- [ ] **Step 5: Run focused and full timing/ROS tests**

```bash
/tmp/lunar-planner-opt/w1-timing/build/lunar_pure_planner_core/lunar_pure_planner_core_planning_timing_test
/tmp/lunar-planner-opt/w1-timing/build/lunar_pure_planner_core/lunar_pure_planner_core_dual_mode_planner_test
/tmp/lunar-planner-opt/w1-timing/build/lunar_pure_planner_ros/request_diagnostics_test
/tmp/lunar-planner-opt/w1-timing/build/lunar_pure_planner_ros/message_conversion_test
/tmp/lunar-planner-opt/w1-timing/build/lunar_pure_planner_ros/pure_plan_motion_server_test
python3 -m pytest -q tests/test_launch_contract.py
```

Expected GREEN: exact boundaries pass, all stages share one 3 s deadline, late success retains a reference, hard timeout has none, and old configurable budget parameters are absent.

- [ ] **Step 6: Commit the timing lane**

```bash
git add ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/planning_timing.hpp \
  ros2_ws/src/lunar_pure_planner_core/src/shared/planning_timing.cpp \
  ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/types/planning_request.hpp \
  ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/planner.hpp \
  ros2_ws/src/lunar_pure_planner_core/src/planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/planning_timing_test.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/dual_mode_planner_test.cpp \
  ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp \
  ros2_ws/src/lunar_pure_planner_ros/src/request_diagnostics.cpp \
  ros2_ws/src/lunar_pure_planner_ros/src/message_conversion.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/request_diagnostics_test.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/message_conversion_test.cpp \
  config/pure_planner.yaml tests/test_launch_contract.py
git commit -m "fix: enforce end-to-end planner timing contract"
```

### Task 3: Use Occupied-Cell-Area Distance for Global Safety

**Owner:** Wave 1 global-safety worker.

**Files:**
- Create: `ros2_ws/src/lunar_pure_planner_core/src/shared/cell_area_distance_transform.hpp`
- Create: `ros2_ws/src/lunar_pure_planner_core/src/shared/cell_area_distance_transform.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/global_occupancy_projection.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/global_occupancy_projection_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/surface_global_search_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/CMakeLists.txt`

**Interfaces:**
- `BuildCellAreaClearance(width, height, resolution, hazard_mask, control)` returns exact centre-to-occupied-square clearance for every cell.
- `BuildCellAreaInflationStencil(resolution, inflation_m)` returns stable integer offsets whose exact occupied-square distance is less than `inflation_m`.
- `GlobalOccupancyProjectionView::ClearanceMeters()` keeps its existing signature but changes to the correct cell-area convention.

- [ ] **Step 1: Add geometry regressions**

For a 1 m grid with one occupied cell, require axis-adjacent clearance `0.5`, diagonal-adjacent clearance `sqrt(0.5)`, two-cells-axis clearance `1.5`, and occupied clearance `0`. Require inflation `0.9187 m` to block both immediate axis and diagonal neighbours but not the two-cells-axis centre. Add a global-route fixture whose centre-distance mask accepts a corner but occupied-area inflation must reject it.

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-opt/w1-global-safety/log build \
  --base-paths ros2_ws/src --packages-select lunar_pure_planner_core \
  --build-base /tmp/lunar-planner-opt/w1-global-safety/build \
  --install-base /tmp/lunar-planner-opt/w1-global-safety/install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
/tmp/lunar-planner-opt/w1-global-safety/build/lunar_pure_planner_core/lunar_pure_planner_core_global_occupancy_projection_test
```

Expected RED: adjacent clearances are reported as `1` and `sqrt(2)` under the old centre-distance transform.

- [ ] **Step 2: Implement the exact separable cell-area transform and stencil**

For integer offset `(dx,dy)`, use:

```cpp
const double x = std::max(std::abs(dx) - 0.5, 0.0) * resolution_m;
const double y = std::max(std::abs(dy) - 0.5, 0.0) * resolution_m;
const double distance_m = std::hypot(x, y);
```

The transform minimizes the corresponding separable squared function over all hazard sites in linear map time. The inflation stencil enumerates only offsets within `ceil(inflation/resolution + 0.5)`, sorts by `(dy,dx)`, and blocks when `distance_m < inflation_m`; equality remains feasible. Unknown, out-of-range and threshold-occupied values remain hazards. Check `SearchControl` during both transform passes and stencil application.

- [ ] **Step 3: Make hard feasibility and route cost use the same values**

Replace the centre-distance output in `BuildGlobalOccupancyProjection()`. `BuildInflatedGlobalOccupancyProjection()` uses the deterministic stencil for the hard mask and the same cell-area clearance vector for costs. Keep preview/supercover certification against the inflated mask.

- [ ] **Step 4: Verify global safety tests**

```bash
/tmp/lunar-planner-opt/w1-global-safety/build/lunar_pure_planner_core/lunar_pure_planner_core_global_occupancy_projection_test
/tmp/lunar-planner-opt/w1-global-safety/build/lunar_pure_planner_core/lunar_pure_planner_core_surface_global_search_test
```

Expected GREEN: exact clearance values, strict inflation boundary, cancellation/deadline behavior and global route corner rejection all pass.

- [ ] **Step 5: Commit the safety lane**

```bash
git add ros2_ws/src/lunar_pure_planner_core/src/shared/cell_area_distance_transform.* \
  ros2_ws/src/lunar_pure_planner_core/src/shared/global_occupancy_projection.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/global_occupancy_projection_test.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/surface_global_search_test.cpp \
  ros2_ws/src/lunar_pure_planner_core/CMakeLists.txt
git commit -m "fix: use occupied cell area for global clearance"
```

### Task 4: Repair and Sparsify Shared ARA*

**Owner:** Wave 1 ARA* worker.

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/anytime_ara_star.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/anytime_ara_star.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/ara_star_anytime_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/hierarchical/surface_global_search.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/legged/anytime_legged_planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/hopper/anytime_hopper_planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/anytime_legged_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/anytime_hopper_planner_test.cpp`

**Interfaces:**
- Change `ExpandFn` to receive the settled source path cost: `void(state, source_g, edges)`.
- Add optional `StateExpandableFn` and `RelaxedFn(state,new_g)` callbacks for Hybrid-label retirement and label-cost synchronization.
- Extend `AraStarResult` with generated and reopened counts plus OPEN peak.
- Keep `state_count` as a structural upper bound, but allocate records only for touched sequential state IDs.

- [ ] **Step 1: Add correctness and sparse-allocation tests**

Rename the old guided-first test to `AnchorKeyBeatsGuidanceForUnequalPriority` and require the lower `g + epsilon*h` branch. Keep guidance selection only when anchor and path-cost ties are equal. Add tests for lazy stale entries, reopen counting, inactive label skipping, relax callback order, first-solution return, and a two-node graph with `state_count=max(size_t)` that cannot allocate nominal capacity.

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-opt/w1-ara/log build \
  --base-paths ros2_ws/src --packages-select lunar_pure_planner_core \
  --build-base /tmp/lunar-planner-opt/w1-ara/build \
  --install-base /tmp/lunar-planner-opt/w1-ara/install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
/tmp/lunar-planner-opt/w1-ara/build/lunar_pure_planner_core/lunar_pure_planner_core_ara_star_anytime_test
```

Expected RED: guidance selects the higher anchor branch and the huge nominal graph attempts dense initialization.

- [ ] **Step 2: Make the anchor key primary**

Use a min-heap entry ordered exactly as:

```cpp
return std::tie(lhs.anchor_key, lhs.path_cost, lhs.guidance_cost,
                lhs.state, lhs.sequence) >
       std::tie(rhs.anchor_key, rhs.path_cost, rhs.guidance_cost,
                rhs.state, rhs.sequence);
```

`anchor_key` is `g + epsilon*h_anchor`. Incumbent termination compares against the minimum live anchor entry after stale entries are removed. Non-finite or negative anchor/guidance data remains a typed invalid-problem failure.

- [ ] **Step 3: Replace dense state arrays and tree sets**

Store one `SearchNode` record per touched sequential state. Each record owns g, parent, incoming edge, cached successors, CLOSED/INCONS flags and an OPEN generation counter. `priority_queue` insertion increments the generation; pop discards entries whose generation or g no longer matches. Track OPEN and INCONS state IDs in compact vectors and rekey only those vectors when epsilon changes. Never iterate to nominal `state_count`.

Call `expand(state, g, edges)`, grow records through the largest returned target, then validate `target < state_count`. Call `on_relaxed` immediately after a successful g update. Skip expansion when `state_expandable` returns false, but retain parent records so already certified descendant paths remain reconstructible.

- [ ] **Step 4: Update all active callers and run all-platform tests**

Static global/legged/hopper callers ignore `source_g` and use default callbacks. Wheel temporarily ignores it; Task 7 consumes it. Run:

```bash
/tmp/lunar-planner-opt/w1-ara/build/lunar_pure_planner_core/lunar_pure_planner_core_ara_star_anytime_test
/tmp/lunar-planner-opt/w1-ara/build/lunar_pure_planner_core/lunar_pure_planner_core_surface_global_search_test
/tmp/lunar-planner-opt/w1-ara/build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test \
  --gtest_filter=-WheelPlanner.GlobalRouteWithEightMeterRollingHorizonReaches750MeterGoalThroughRandomObstacles
/tmp/lunar-planner-opt/w1-ara/build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_legged_planner_test
/tmp/lunar-planner-opt/w1-ara/build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_hopper_planner_test
```

Expected GREEN: shared engine and every platform pass; the known 750 m case remains excluded at this stage.

- [ ] **Step 5: Commit the ARA* lane**

```bash
git add ros2_ws/src/lunar_pure_planner_core/src/shared/anytime_ara_star.* \
  ros2_ws/src/lunar_pure_planner_core/test/ara_star_anytime_test.cpp \
  ros2_ws/src/lunar_pure_planner_core/src/hierarchical/surface_global_search.cpp \
  ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/src/legged/anytime_legged_planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/src/hopper/anytime_hopper_planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/anytime_legged_planner_test.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/anytime_hopper_planner_test.cpp
git commit -m "fix: restore sparse anchor-ordered ARA star"
```

### Task 5: Integrate and Review Wave 1

**Owner:** Coordinator, Gate 1.

**Files:**
- Modify only when resolving a reviewed semantic integration issue in files already owned by Tasks 2-4.
- Append evidence: `docs/validation/planner-core-optimization.md`

- [ ] **Step 1: Review each worker commit before cherry-picking**

Check scope, RED/GREEN evidence, exact boundary comparisons, OPEN comparator order, cancellation priority, cell-area equality semantics and absence of ROS message changes. Cherry-pick timing, global safety, then ARA*.

- [ ] **Step 2: Run the integrated suites**

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-opt/gate1/log build \
  --base-paths ros2_ws/src \
  --packages-select lunar_planning_msgs lunar_pure_planner_core lunar_pure_planner_ros \
  --build-base /tmp/lunar-planner-opt/gate1/build \
  --install-base /tmp/lunar-planner-opt/gate1/install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
ctest --test-dir /tmp/lunar-planner-opt/gate1/build/lunar_pure_planner_core \
  --output-on-failure -E lunar_pure_planner_core_anytime_wheel_planner_test
/tmp/lunar-planner-opt/gate1/build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test \
  --gtest_filter=-WheelPlanner.GlobalRouteWithEightMeterRollingHorizonReaches750MeterGoalThroughRandomObstacles
ctest --test-dir /tmp/lunar-planner-opt/gate1/build/lunar_pure_planner_ros \
  --output-on-failure
python3 -m pytest -q tests/test_action_contract.py tests/test_external_interface_contract.py tests/test_launch_contract.py
```

Expected: all shared/global/ROS tests pass. The 750 m test may still fail, but it must no longer fail due a 1 s/150 ms stage deadline or dense ARA initialization.

- [ ] **Step 3: Commit only necessary integration fixes and evidence**

```bash
git add docs/validation/planner-core-optimization.md
git commit -m "test: record planner optimization wave one"
```

### Task 6: Build Deterministic Safe Rolling Portal Sets

**Owner:** Wave 2 portal worker.

**Files:**
- Create: `ros2_ws/src/lunar_pure_planner_core/src/hierarchical/surface_portal_set.hpp`
- Create: `ros2_ws/src/lunar_pure_planner_core/src/hierarchical/surface_portal_set.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/hierarchical/surface_rolling_session.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/hierarchical/surface_rolling_session.cpp`
- Create: `ros2_ws/src/lunar_pure_planner_core/test/surface_portal_set_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/surface_rolling_session_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/CMakeLists.txt`

**Interfaces:**
- `SurfaceRollingDecision` exposes projected route progress, desired horizon progress and final-goal status rather than one interpolated goal.
- `BuildSurfacePortalSet(input, route, decision, max_candidates, control)` returns ordered `SurfacePortalCandidate` objects in odom coordinates.
- Each candidate records route progress, global/local cell, global/local clearance and stable rank; maximum size is 32.

- [ ] **Step 1: Add portal-set tests**

Cover a blocked centreline with a safe lateral candidate, longitudinal backoff, local-map clipping, duplicate-cell removal, deterministic ordering under repeated calls, 32-candidate cap, and exact final-goal-only behavior near mission completion.

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-opt/w2-portals/log build \
  --base-paths ros2_ws/src --packages-select lunar_pure_planner_core \
  --build-base /tmp/lunar-planner-opt/w2-portals/build \
  --install-base /tmp/lunar-planner-opt/w2-portals/install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Expected RED: the current rolling decision exposes only one exact centreline point and cannot provide fallback candidates.

- [ ] **Step 2: Implement route-progress and candidate generation**

Use a stable candidate record:

```cpp
struct SurfacePortalCandidate final {
  GoalRegion goal_odom;
  double route_progress_m{};
  GridCell global_cell;
  GridCell local_cell;
  float global_clearance_m{};
  float local_clearance_m{};
  std::size_t stable_rank{};
};
```

Sample the desired horizon, deterministic backoff distances and lateral global cells inside the inflated global projection. Transform to odom, reject outside/local hazards, and rank by descending progress, descending global clearance, descending local clearance, global `(y,x)`, local `(y,x)`, then creation rank. Intermediate portals have no yaw constraint; exact final goal preserves requested position/yaw tolerances. Return `NO_PATH` only when the bounded safe set is empty.

- [ ] **Step 3: Verify portal tests**

```bash
/tmp/lunar-planner-opt/w2-portals/build/lunar_pure_planner_core/lunar_pure_planner_core_surface_rolling_session_test
/tmp/lunar-planner-opt/w2-portals/build/lunar_pure_planner_core/lunar_pure_planner_core_surface_portal_set_test
```

Expected GREEN: the same inputs always produce the same bounded order and a blocked nominal horizon produces a safe alternative.

- [ ] **Step 4: Commit the portal lane**

```bash
git add ros2_ws/src/lunar_pure_planner_core/src/hierarchical/surface_portal_set.* \
  ros2_ws/src/lunar_pure_planner_core/src/hierarchical/surface_rolling_session.* \
  ros2_ws/src/lunar_pure_planner_core/test/surface_portal_set_test.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/surface_rolling_session_test.cpp \
  ros2_ws/src/lunar_pure_planner_core/CMakeLists.txt
git commit -m "feat: generate deterministic rolling portal sets"
```

### Task 7: Convert the Wheeled Interior Graph to Primitive-Based Hybrid A*

**Owner:** Wave 2 Hybrid worker.

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp`

**Interfaces:**
- Keep `PlanWheel(const WheelPlanRequest&)` and single-goal input for this lane; Task 10 adds the goal set.
- Remove `maximum_search_states` as a production request limit.
- Replace one state ID per `WheelStateKey` with at most four active continuous labels plus archived parent records.
- Consume ARA* `source_g`, `state_expandable` and `on_relaxed` callbacks from Task 4.

- [ ] **Step 1: Replace canonical-state expectations with Hybrid regressions**

Add tests requiring two consecutive project `pi/16` forward arcs before the terminal connector, an S-turn, consecutive reverse arcs, a narrow obstacle-sensitive case with two non-dominated representatives in one key, arbitrary translated/yaw start, stop/switch behavior and deterministic repeat output. Assert the consecutive-arc test has at least two full primitive-length interior edges and cannot pass through one scaled terminal edge.

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-opt/w2-hybrid/log build \
  --base-paths ros2_ws/src --packages-select lunar_pure_planner_core \
  --build-base /tmp/lunar-planner-opt/w2-hybrid/build \
  --install-base /tmp/lunar-planner-opt/w2-hybrid/install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
/tmp/lunar-planner-opt/w2-hybrid/build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test \
  --gtest_filter='WheelPlanner.ConsecutiveProjectArcsRemainPhysicalInteriorEdges:WheelPlanner.PlansCertifiedSTurnWithContinuousPrimitiveEndpoints:WheelPlanner.RetainsObstacleDistinctLabelsInOneKey'
```

Expected RED: canonical snapping rejects the second arc or aliases its physically distinct endpoint.

- [ ] **Step 2: Select yaw bins from primitive increments**

Test bin counts `16,32,64,128,256` in ascending order. Select the first where every nonzero primitive yaw delta divided by bin width is within `1e-9 rad` of an integer; use 256 if none match. Use one selected yaw count for wide and narrow keys; narrow mode refines XY only. The project capability must report 32 bins.

- [ ] **Step 3: Retain exact physical endpoints and bounded labels**

Use this conceptual node state:

```cpp
struct Node final {
  WheelStateKey key;
  Pose3 pose;
  double best_g{std::numeric_limits<double>::infinity()};
  double certified_clearance_m{};
  std::size_t creation_sequence{};
  bool expandable{true};
  bool exact_goal{};
};
```

`ApplyPrimitive()` targets remain unchanged after world elevation support is assigned. Quantization indexes duplicate detection only; delete `CanonicalPose()`, `RepresentativePose()` and endpoint replacement. Merge labels only for matching mode/resolution class within one-quarter XY resolution and one-quarter yaw-bin width when the retained label has no greater g. On a fifth non-dominated label, rank all five by anchor key, g, descending clearance, XY bin residual, yaw residual and creation sequence. Keep four active; an evicted record remains reconstructible but not expandable. Synchronize g through `on_relaxed`.

- [ ] **Step 4: Keep terminal behavior certified and exact**

Do not change `MatchingPrimitiveScale()`, `ScaledPrimitive()`, `AppendGoalTerminal()` or exact sweep evaluation except to source them from continuous labels. Every interior edge is a full supplied primitive; only the existing terminal connector may scale.

- [ ] **Step 5: Run the wheel suite except the cross-module 750 m gate**

```bash
/tmp/lunar-planner-opt/w2-hybrid/build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test \
  --gtest_filter=-WheelPlanner.GlobalRouteWithEightMeterRollingHorizonReaches750MeterGoalThroughRandomObstacles
```

Expected GREEN: every selected wheel test passes, project yaw bins equal 32, no fixed state-capacity failure exists, and repeated trajectories are deterministic.

- [ ] **Step 6: Commit the Hybrid lane**

```bash
git add ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.* \
  ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp
git commit -m "feat: retain continuous labels in wheel hybrid search"
```

### Task 8: Optimize Local Projection and Add Multi-Source Goal Fields

**Owner:** Wave 2 projection worker.

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/local_terrain_projection.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/goal_distance_field.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/goal_distance_field.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/local_terrain_projection_test.cpp`

**Interfaces:**
- Local clearance uses `BuildCellAreaClearance()` introduced by Task 3.
- Add `BuildGoalDistanceField(terrain, std::span<const GridCell> goals, control)` while retaining the one-cell overload as a delegating convenience.
- `GoalDistanceField` records both `distance_m` and `nearest_goal_index` for deterministic portal selection.

- [ ] **Step 1: Add exact-local-clearance, allocation and multi-source tests**

Require the same axis/diagonal occupied-cell-area distances as global projection. Add a large flat-map test whose result is exact and does not use priority-queue propagation. Add multi-source tests where each side selects its nearest seed, equal-distance ties select the lower stable input index, corner blocking matches local traversal, and sealed cells remain infinite with an invalid nearest index.

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-opt/w2-projection/log build \
  --base-paths ros2_ws/src --packages-select lunar_pure_planner_core \
  --build-base /tmp/lunar-planner-opt/w2-projection/build \
  --install-base /tmp/lunar-planner-opt/w2-projection/install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
/tmp/lunar-planner-opt/w2-projection/build/lunar_pure_planner_core/lunar_pure_planner_core_local_terrain_projection_test
```

Expected RED: local adjacent clearance is centre/path distance and the goal-field API accepts only one source.

- [ ] **Step 2: Reuse the exact linear transform and remove per-cell heaps**

Replace local Dijkstra clearance with `BuildCellAreaClearance()`. Replace `std::vector` allocations in `EstimateRoughness()` with `std::array<std::array<double, 3>, 9>` and `std::array<double, 9>` plus a sample count; accumulate the 3x3 normal equations and residuals without allocating per cell. Preserve non-finite elevation/occupancy fail-closed behavior and cancellation checkpoints.

- [ ] **Step 3: Implement one deterministic multi-source field**

Seed every valid unique goal cell in stable input order. Run one 8-connected search with the existing corner-blocking rule. Order queue entries by distance, source index, cell index and insertion sequence. Never map infinity to zero.

- [ ] **Step 4: Verify projection tests and commit**

```bash
/tmp/lunar-planner-opt/w2-projection/build/lunar_pure_planner_core/lunar_pure_planner_core_local_terrain_projection_test
git add ros2_ws/src/lunar_pure_planner_core/src/shared/local_terrain_projection.cpp \
  ros2_ws/src/lunar_pure_planner_core/src/shared/goal_distance_field.* \
  ros2_ws/src/lunar_pure_planner_core/test/local_terrain_projection_test.cpp
git commit -m "perf: linearize local projection and goal fields"
```

### Task 9: Integrate and Review Wave 2

**Owner:** Coordinator, Gate 2.

**Files:**
- Modify only reviewed integration conflicts.
- Append evidence: `docs/validation/planner-core-optimization.md`

- [ ] **Step 1: Review and cherry-pick portal, projection and Hybrid commits**

Verify the portal lane does not call 32 planners, the projection lane preserves corner blocking, and the Hybrid lane never overwrites a physical primitive endpoint with a quantized pose.

- [ ] **Step 2: Run the integrated core suite**

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-opt/gate2/log build \
  --base-paths ros2_ws/src --packages-select lunar_pure_planner_core \
  --build-base /tmp/lunar-planner-opt/gate2/build \
  --install-base /tmp/lunar-planner-opt/gate2/install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
ctest --test-dir /tmp/lunar-planner-opt/gate2/build/lunar_pure_planner_core \
  --output-on-failure -E lunar_pure_planner_core_anytime_wheel_planner_test
/tmp/lunar-planner-opt/gate2/build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test \
  --gtest_filter=-WheelPlanner.GlobalRouteWithEightMeterRollingHorizonReaches750MeterGoalThroughRandomObstacles
```

Expected: all shared, global, portal, wheel (except 750 m), legged and hopper tests pass.

- [ ] **Step 3: Commit gate evidence**

```bash
git add docs/validation/planner-core-optimization.md
git commit -m "test: record planner optimization wave two"
```

### Task 10: Integrate Portals as One Certified Multi-Goal Wheel Search

**Owner:** Coordinator, sequential Wave 3 because this task owns the shared planner/wheel integration seam.

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/types/planning_request.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/planner.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/hierarchical/global_route_planner.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/hierarchical/global_route_planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/dual_mode_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/wheel_long_range_fixture.hpp`

**Interfaces:**
- Add `LocalGoalSet { std::vector<GoalRegion> goals_odom; bool exact_final_goal; }`.
- Change the active local backend seam to consume `const LocalGoalSet&`.
- Add global/local/odometry/TF sequence identities to `MinimalWorldSnapshot`; zero means non-cacheable direct-core input.
- `WheelPlanRequest` consumes one `LocalGoalSet`, local source sequence and capability fingerprint; `WheelPlanResult` reports `selected_goal_index`.

- [ ] **Step 1: Add one-search fallback tests**

Add a wheel fixture where the highest-ranked portal is disconnected and the second is reachable. Assert one ARA invocation, shared expanded states, selected index `1`, and no independent retry. Add exact-final tests requiring only the true mission goal and its yaw. Update backend tests to capture the full set.

Run:

```bash
source /opt/ros/jazzy/setup.bash
cmake --build /tmp/lunar-planner-opt/gate2/build/lunar_pure_planner_core \
  --target lunar_pure_planner_core_anytime_wheel_planner_test \
           lunar_pure_planner_core_dual_mode_planner_test -j2
/tmp/lunar-planner-opt/gate2/build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test \
  --gtest_filter=WheelPlanner.UsesOneSearchForRankedPortalFallback
```

Expected RED: current request and graph accept only one goal.

- [ ] **Step 2: Wire one multi-source heuristic and any-goal terminal connector**

Build one goal field from all portal cells. `Heuristic()` uses the maximum of normalized planar-region lower bound, finite relaxed field lower bound and unavoidable yaw lower bound. `IsGoal()` and scaled connector iterate candidates in stable rank, create an exact terminal node containing its goal index, and return the selected index with the trajectory. Intermediate sets omit yaw; exact-final set contains one untouched requested goal.

- [ ] **Step 3: Replace the custom 750 m target helper with production portal selection**

The long-range fixture must use `SurfaceRollingSession`, `BuildSurfacePortalSet()` and the multi-goal `PlanWheel()` call. Delete its private `route_point_at_horizon` helper. Preserve the seed, obstacles, map sizes, 8 m horizon, start and final goal. Each rolling planning cycle receives a fresh `cycle_started + 3 s` hard deadline; the 1 s and 2 s points are recorded classifications, not direct `PlanWheel` deadlines.

- [ ] **Step 4: Verify functional 750 m success and all-platform compatibility**

```bash
cmake --build /tmp/lunar-planner-opt/gate2/build/lunar_pure_planner_core -j2
/tmp/lunar-planner-opt/gate2/build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test \
  --gtest_filter=WheelPlanner.GlobalRouteWithEightMeterRollingHorizonReaches750MeterGoalThroughRandomObstacles
ctest --test-dir /tmp/lunar-planner-opt/gate2/build/lunar_pure_planner_core \
  --output-on-failure
```

Expected GREEN: the unmodified 750 m world reaches the true goal without shifted targets, and all platform/shared tests pass.

- [ ] **Step 5: Commit multi-goal integration**

```bash
git add ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/types/planning_request.hpp \
  ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/planner.hpp \
  ros2_ws/src/lunar_pure_planner_core/src/planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/src/hierarchical/global_route_planner.hpp \
  ros2_ws/src/lunar_pure_planner_core/src/hierarchical/global_route_planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.hpp \
  ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/dual_mode_planner_test.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/wheel_long_range_fixture.hpp
git commit -m "feat: search rolling portals as one certified goal set"
```

### Task 11: Stage Exact Wheel Edge Certification

**Owner:** Wave 4 certification worker.

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp`

**Interfaces:**
- Extend wheel metrics with broad-phase rejects, full certifications, full invalidations and returned-edge certificate confirmations.
- Keep `EdgeEvaluation` as the exact certificate; a broad-phase result alone never becomes an emitted valid edge.

- [ ] **Step 1: Add staged-certification safety tests**

Cover broad-phase obstacle rejection, safe-clearance fast proof followed by exact terrain/dynamics checks, a false-positive broad phase rejected by exact polygon intersection, cache reuse, identity mismatch and final-path certificate confirmation. Require returned-edge confirmations equal returned path edge count.

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-opt/w3-certification/log build \
  --base-paths ros2_ws/src --packages-select lunar_pure_planner_core \
  --build-base /tmp/lunar-planner-opt/w3-certification/build \
  --install-base /tmp/lunar-planner-opt/w3-certification/install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Expected RED: metrics/certificate stages do not exist and every candidate enters the same expensive polygon-cell loop.

- [ ] **Step 2: Implement conservative stages**

Apply bounds/mode/shape checks first, then hazard integral, clearance field and selected-yaw footprint mask. Competitive edges proceed through existing polygon, support, elevation, slope, roughness, relief, underbody and dynamics evaluation. Store map sequence, capability fingerprint, exact source pose bits and primitive stable identity in every exact certificate. Before trajectory reconstruction, confirm every chosen certificate matches current identities.

- [ ] **Step 3: Verify wheel safety and commit**

```bash
/tmp/lunar-planner-opt/w3-certification/build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test
git add ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.* \
  ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp
git commit -m "perf: stage certified wheel edge evaluation"
```

### Task 12: Add Revision-Keyed Immutable Active-Pipeline Caches

**Owner:** Wave 4 cache worker.

**Files:**
- Create: `ros2_ws/src/lunar_pure_planner_core/src/shared/active_planner_cache.hpp`
- Create: `ros2_ws/src/lunar_pure_planner_core/src/shared/active_planner_cache.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/planner.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/hierarchical/global_route_planner.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/hierarchical/global_route_planner.cpp`
- Create: `ros2_ws/src/lunar_pure_planner_core/test/active_planner_cache_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/CMakeLists.txt`

**Interfaces:**
- `ActivePlannerCache` owns immutable shared global snapshots/projections/routes and local snapshots/projections.
- Cache keys include source sequence plus every semantic threshold/inflation/capability fingerprint.
- Sequence zero bypasses reuse; no cache entry crosses a changed sequence or semantic parameter.

- [ ] **Step 1: Add hit/miss/invalidation tests**

Test cold/warm identity, global sequence change, local sequence change, occupancy threshold change, inflation change, capability change, zero-sequence bypass, cancellation during a miss and concurrent readers receiving one immutable result.

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-opt/w3-cache/log build \
  --base-paths ros2_ws/src --packages-select lunar_pure_planner_core \
  --build-base /tmp/lunar-planner-opt/w3-cache/build \
  --install-base /tmp/lunar-planner-opt/w3-cache/install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Expected RED: the active `Planner` rebuilds maps each request and has no active-pipeline cache API.

- [ ] **Step 2: Implement bounded immutable caches**

Use explicit key structs, equality and stable semantic fingerprints; never key by raw pointer alone. Build outside the mutex, then publish an immutable `shared_ptr<const T>` only if the requested identity is still current. Keep at most the current global and current local identity plus goal fields for the current local identity; replacement releases older entries.

- [ ] **Step 3: Wire global/local projection reuse and diagnostics**

`Planner` owns one cache instance. `PlanSurfaceGlobal()` and `PlanLocalDefault()` request cached immutable projections. Populate cache-hit fields in `PlanningResult` diagnostics. Do not wire wheel edge certificates in this lane; Task 14 performs that integration after Task 11 lands.

- [ ] **Step 4: Verify and commit**

```bash
/tmp/lunar-planner-opt/w3-cache/build/lunar_pure_planner_core/lunar_pure_planner_core_active_planner_cache_test
/tmp/lunar-planner-opt/w3-cache/build/lunar_pure_planner_core/lunar_pure_planner_core_dual_mode_planner_test
git add ros2_ws/src/lunar_pure_planner_core/src/shared/active_planner_cache.* \
  ros2_ws/src/lunar_pure_planner_core/include/lunar_pure_planner_core/planner.hpp \
  ros2_ws/src/lunar_pure_planner_core/src/planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/src/hierarchical/global_route_planner.* \
  ros2_ws/src/lunar_pure_planner_core/test/active_planner_cache_test.cpp \
  ros2_ws/src/lunar_pure_planner_core/CMakeLists.txt
git commit -m "perf: cache active planner projections by revision"
```

### Task 13: Stabilize Rolling ROS Replans and Publish Throttled Feedback

**Owner:** Wave 4 ROS worker.

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/request_diagnostics.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/request_diagnostics_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/message_conversion.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/message_conversion_test.cpp`
- Modify: `config/pure_planner.yaml`

**Interfaces:**
- Pass `InputSnapshot` global/local/odometry/TF sequences into `MinimalWorldSnapshot` identities.
- Publish existing `PlanMotion::Feedback` fields through the core `ProgressFn` no faster than 10 Hz, plus immediate phase changes.
- Emit one diagnostic sample per planning cycle, not one sample for rover travel time.

- [ ] **Step 1: Add rolling-trigger and feedback tests**

Prove no replan occurs for floating-point target drift; replans occur for handoff, a new local sequence after minimum interval, route deviation, global/TF identity invalidation and active-reference invalidation. Prove feedback phase order, 10 Hz throttling, monotonic elapsed/expanded counts, late-result identity recheck, cancellation priority and no reference at 3 s.

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-opt/w3-ros/log build \
  --base-paths ros2_ws/src \
  --packages-select lunar_planning_msgs lunar_pure_planner_core lunar_pure_planner_ros \
  --build-base /tmp/lunar-planner-opt/w3-ros/build \
  --install-base /tmp/lunar-planner-opt/w3-ros/install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Expected RED: `target_changed > 1e-6` causes unnecessary replans and no action feedback is published.

- [ ] **Step 2: Implement semantic triggers and per-cycle timing**

Delete the continuously projected target-change trigger. Keep one global route while global/TF/capability identity remains valid. Start a new 3 s timing policy only when a real planning trigger fires. Recheck active request, global/local sequence, TF sequence and capability before publishing a late reference.

- [ ] **Step 3: Map progress and diagnostics without schema changes**

Throttle with a worker-local last-publish steady time. Map phases to existing feedback constants and fill elapsed, expanded states, `has_best_cost` and `best_cost`. Diagnostic-array values include phase times, generated/reopened states, edge/cache/sweep counters and cache hits. `PlannerDiagnostics.warning_codes` carries latency warnings; it does not carry numeric key/value metrics.

- [ ] **Step 4: Verify and commit**

```bash
/tmp/lunar-planner-opt/w3-ros/build/lunar_pure_planner_ros/pure_plan_motion_server_test
/tmp/lunar-planner-opt/w3-ros/build/lunar_pure_planner_ros/request_diagnostics_test
/tmp/lunar-planner-opt/w3-ros/build/lunar_pure_planner_ros/message_conversion_test
git add ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp \
  ros2_ws/src/lunar_pure_planner_ros/src/request_diagnostics.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/request_diagnostics_test.cpp \
  ros2_ws/src/lunar_pure_planner_ros/src/message_conversion.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/message_conversion_test.cpp \
  config/pure_planner.yaml
git commit -m "fix: stabilize rolling replans and planner feedback"
```

### Task 14: Integrate Caches, Certificates and Performance Gates

**Owner:** Coordinator, Gate 3.

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/active_planner_cache.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/shared/active_planner_cache.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/active_planner_cache_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp`
- Modify: `tools/measure_planner_performance.py`
- Modify: `docs/validation/planner-core-optimization.md`

- [ ] **Step 1: Review and cherry-pick Wave 4 commits**

Cherry-pick certification, cache, then ROS. Resolve only the expected `Planner` metric wiring seam. Run the focused tests from all three lanes before integration changes.

- [ ] **Step 2: Cache multi-source fields and exact certificates by complete identity**

Goal-field key is `(local_sequence, local_threshold, ordered_goal_cells)`. Edge-certificate key is `(local_sequence, capability_fingerprint, exact_source_pose_bits, source_mode, primitive_fingerprint)`. Never use request-local state ID as a cross-request key. Reconfirm identity for every returned edge. Tests must prove every single key component causes a miss when changed.

- [ ] **Step 3: Run ten deterministic 750 m repetitions**

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-opt/gate3/log build \
  --base-paths ros2_ws/src \
  --packages-select lunar_planning_msgs lunar_pure_planner_core lunar_pure_planner_ros \
  --build-base /tmp/lunar-planner-opt/gate3/build \
  --install-base /tmp/lunar-planner-opt/gate3/install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
python3 tools/measure_planner_performance.py \
  --binary /tmp/lunar-planner-opt/gate3/build/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test \
  --gtest-filter WheelPlanner.GlobalRouteWithEightMeterRollingHorizonReaches750MeterGoalThroughRandomObstacles \
  --repetitions 10 \
  --output /tmp/lunar-planner-opt/gate3/750m-report.json
```

Expected: 10/10 success, no fixed capacity exhaustion, every run under the 3 s hard boundary per planning cycle, and segment-21 sweep-cell checks below 5,000,000.

- [ ] **Step 4: Measure target/SLA and simple-scenario regression**

Run at least 30 Release samples for the declared simple wheel scenario and the former segment-21 cycle. Require p95 first-certified-reference `<1 s`; any sample `>=2 s` fails the performance acceptance even if functionally late-successful; no sample reaches 3 s. Compare the simple scenario with the Task 1 baseline on the same host/build class and require p95 regression no greater than 10%. Record raw report paths and summaries.

- [ ] **Step 5: Run full integrated tests and commit**

```bash
ctest --test-dir /tmp/lunar-planner-opt/gate3/build/lunar_pure_planner_core \
  --output-on-failure
ctest --test-dir /tmp/lunar-planner-opt/gate3/build/lunar_pure_planner_ros \
  --output-on-failure
python3 -m pytest -q tests/test_action_contract.py tests/test_external_interface_contract.py tests/test_launch_contract.py tests/test_measure_planner_performance.py
git add ros2_ws/src/lunar_pure_planner_core/src/shared/active_planner_cache.hpp \
  ros2_ws/src/lunar_pure_planner_core/src/shared/active_planner_cache.cpp \
  ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.hpp \
  ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/src/planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/active_planner_cache_test.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp \
  tools/measure_planner_performance.py docs/validation/planner-core-optimization.md
git commit -m "perf: integrate planner caches and performance gates"
```

### Task 15: Complete Jazzy/RViz Verification and Documentation

**Owner:** Coordinator, final sequential gate. Use `superpowers:verification-before-completion` before claiming success.

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-24-planner-core-incremental-optimization-design.md`
- Modify: `docs/superpowers/specs/2026-08-24-request-scoped-se2-lattice-design.md`
- Modify: `docs/superpowers/plans/2026-08-24-planner-core-incremental-optimization.md`
- Modify: `docs/validation/planner-core-optimization.md`
- Modify when diagnostics changed: `rviz/lunar_surface_demo.rviz`

- [ ] **Step 1: Run test-enabled Jazzy verification**

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-opt/final-on/log build \
  --base-paths ros2_ws/src \
  --packages-select lunar_planning_msgs lunar_pure_planner_core lunar_pure_planner_ros \
  --build-base /tmp/lunar-planner-opt/final-on/build \
  --install-base /tmp/lunar-planner-opt/final-on/install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
colcon --log-base /tmp/lunar-planner-opt/final-on/test-log test \
  --build-base /tmp/lunar-planner-opt/final-on/build \
  --install-base /tmp/lunar-planner-opt/final-on/install \
  --packages-select lunar_planning_msgs lunar_pure_planner_core lunar_pure_planner_ros \
  --return-code-on-test-failure \
  --event-handlers console_direct+
colcon test-result --test-result-base /tmp/lunar-planner-opt/final-on/build --verbose
```

Expected: zero failed tests and explicit counts recorded.

- [ ] **Step 2: Run deployment build with tests omitted**

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-planner-opt/final-off/log build \
  --base-paths ros2_ws/src \
  --packages-select lunar_planning_msgs lunar_pure_planner_core lunar_pure_planner_ros \
  --build-base /tmp/lunar-planner-opt/final-off/build \
  --install-base /tmp/lunar-planner-opt/final-off/install \
  --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
```

Expected: production packages build successfully without compiling tests. This does not replace Step 1 evidence.

- [ ] **Step 3: Run the isolated RViz demo and collect runtime evidence**

Terminal 1:

```bash
cd /home/kai/CodexDownloads/lunar_navigation/.worktrees/planner-core-opt
source /opt/ros/jazzy/setup.bash
source /tmp/lunar-planner-opt/final-off/install/setup.bash
ros2 launch lunar_pure_planner_ros lunar_surface_rviz_demo.launch.py
```

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
source /tmp/lunar-planner-opt/final-off/install/setup.bash
ros2 topic echo --once /lunar_demo/path
ros2 topic echo --once /lunar_demo/diagnostics
ros2 action info /lunar_demo/plan_motion
```

Use RViz **2D Goal Pose** in a free region. Require nonempty `frame_id=map` path, exact odometry-derived first pose, certified goal arrival, compatible TF/QoS, and latency classification. Capture the exact command output and screenshot path in the validation document. Do not claim controller execution; the demo remains test-only.

- [ ] **Step 4: Update operator documentation and readiness boundaries**

Document the fixed time classes, `PLAN_FOUND_LATE`, no configurable planning budgets, one-search portal fallback, Hybrid A* labels, 32-bin project capability result, cache identities, metrics and stable rolling triggers. State that output remains a sampled piecewise trajectory of certified lines/arcs/spins/mode switches, not an unconstrained smoothed curve. Keep the earlier request-frame document's exact-start and terminal-connector rules while marking canonical interior endpoints/fixed bins as superseded.

Record Jazzy host evidence separately from Jetson Orin. If Orin was not actually built and exercised, mark Orin performance, DDS, rosbag and field readiness `NOT_RUN`; host RViz evidence cannot promote it.

- [ ] **Step 5: Run documentation/static contract checks**

```bash
python3 -m pytest -q tests/test_action_contract.py tests/test_external_interface_contract.py tests/test_launch_contract.py tests/test_operator_scripts.py
rg -n "rolling_global_budget_ms|rolling_local_budget_ms|rolling_action_timeout_s|150 ms|950 ms|每段局部搜索仍限制为 1 s" README.md config launch ros2_ws/src tests
git diff --check
git status --short
```

Expected: pytest passes; the `rg` command returns no obsolete deadline text; diff check is clean; status contains only intended documentation changes.

- [ ] **Step 6: Commit documentation and final evidence**

```bash
git add README.md \
  docs/superpowers/specs/2026-08-24-planner-core-incremental-optimization-design.md \
  docs/superpowers/specs/2026-08-24-request-scoped-se2-lattice-design.md \
  docs/superpowers/plans/2026-08-24-planner-core-incremental-optimization.md \
  docs/validation/planner-core-optimization.md rviz/lunar_surface_demo.rviz
git commit -m "docs: record optimized planner verification"
```

## Final Acceptance Checklist

- [ ] Exact 1 s/2 s/3 s fake-clock boundaries pass; 1 s and 2 s never reset or terminate search.
- [ ] A `[2 s,3 s)` fully certified reference is `PLAN_FOUND_LATE`; `>=3 s` publishes no new reference.
- [ ] Global and local clearances use occupied-cell-area geometry and unknown data remains hazardous.
- [ ] Shared ARA* anchor ordering, sparse storage, cancellation and all-platform tests pass.
- [ ] Continuous primitive endpoints, maximum-four active labels, derived yaw bins and exact terminal connector are covered.
- [ ] The production 750 m fixture passes ten consecutive runs without target shift or fixed state-capacity exhaustion.
- [ ] Segment-21 sweep checks are below 5,000,000; performance p95 is `<1 s`; simple p95 regression is at most 10%.
- [ ] Every returned edge has a current exact certificate; cache identity changes force misses.
- [ ] Rolling replans use semantic triggers and one planning clock per cycle; feedback is throttled and monotonic.
- [ ] Jazzy test-enabled and deployment builds pass; RViz shows a framed nonempty path and valid diagnostics.
- [ ] `PlanMotion.action` and `PlannerDiagnostics.msg` have no diff.
- [ ] Host and Jetson Orin evidence are reported as separate readiness levels.
