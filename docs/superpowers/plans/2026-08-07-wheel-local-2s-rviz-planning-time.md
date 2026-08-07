# Wheel Local 2-Second Target and RViz Planning Time Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce representative wheeled rolling local planning to Release p95 at or below 2.000 seconds without a runtime search cutoff, and show authoritative total/cumulative timing plus the wheel budget result in RViz.

**Architecture:** Replace repeated far-to-near wheel lattice traversals with a ranked-frontier search that reuses one exhausted search tree, then replace ordered tree containers in the lattice hot path with a deterministic direct state index and lazy binary heap. Accumulate all global/local retries in core diagnostics and forward those values through the external controller to explicit RViz rows.

**Tech Stack:** C++20, ROS 2 Humble, `rclcpp`, `diagnostic_msgs`, Qt5/RViz2 Panel, Python 3.10, pytest, GoogleTest, colcon, CMake Release.

## Global Constraints

- Ubuntu authority is Ubuntu 22.04 amd64 with ROS 2 Humble; source `/opt/ros/humble/setup.bash` and verify `ROS_DISTRO=humble` before every ROS build or test.
- Use Release for performance qualification; do not reuse build/install/log outputs from another ROS distribution.
- Wheel rolling local planning on seed `20260805`, 50 m, 0.2 m and 12% obstacle occupancy must have `p95 <= 2.000 s` after one warmup and at least 20 measurements.
- RViz judges each individual wheel request as `PASS` only when cumulative local planning is `<= 2.000 s`; statistical p95 is only the release gate.
- Do not add a 2-second runtime timeout, expansion cap, Open cap, candidate cap, or incomplete-search infeasibility result.
- Preserve the approved wheel geometry, 0.20 m obstacle clearance, exact terminal connector, continuous sweep certification, deterministic ordering and existing stable failure reasons.
- Do not modify `PlanMotion.action`, `PlannerDiagnostics.msg`, `MotionReference.msg`, platform capability values or legged search behavior. Hopper and rolling-execution changes are implemented only by the companion approved plans.
- Main-repository runtime artifacts stay under `~/CodexDownloads/lunar_navigation/wheel-local-2s-rviz-timing`; no build, install, log, generated map or benchmark output enters either repository.
- Main changes merge only to `integration`; external RViz changes merge only to the external repository `main` after combined-source verification.

---

### Task 1: Ranked wheel lattice search reuses an exhausted tree

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.cpp`
- Test: `ros2_ws/src/lunar_planner_core/test/wheel_planner_test.cpp`

**Interfaces:**
- Consumes: existing `WheeledState`, `GoalRegion`, `SafeProjection`, `WheeledCapability`, `PlannerConfig` and `stop_token`.
- Produces: `SearchWheelLatticeRanked(..., std::span<const GoalRegion>, ...) -> WheelLatticeSearchResult`; successful results set `selected_goal_index` to the chosen ranked goal. Existing `SearchWheelLattice(..., const GoalRegion&, ...)` delegates with a one-element span.

- [ ] **Step 1: Write failing ranked-fallback tests**

Add `#include <span>` and `std::optional<std::size_t> selected_goal_index` to the expected API. Add tests that build one safe projection, provide an unreachable far goal followed by a reachable near goal, and assert the single ranked call selects index 1 and ends at the near point:

```cpp
TEST(WheelPlanner, RankedSearchReusesTheExhaustedTreeForANearerGoal) {
  auto input = test::MakeValidWheelInput();
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok());
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok());
  const std::array goals{
      GoalRegion{.goal_id = "far", .target = PointGoal{
          .position_m = {100.0, 100.0, 0.0}, .tolerance_m = 0.05}},
      input.goal_map,
  };

  const auto result = wheel::SearchWheelLatticeRanked(
      std::get<WheeledState>(input.current_state), goals,
      *projection.projection, std::get<WheeledCapability>(input.capability),
      input.config, {});

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_EQ(result.selected_goal_index, 1U);
  EXPECT_EQ(result.plan->transitions.back().target_pose.position_m,
            std::get<PointGoal>(goals[1].target).position_m);
}
```

Also add tests for: far goal succeeds with index 0; all goals fail with `WHEEL_NO_KNOWN_SAFE_ROUTE`; cancellation remains `REQUEST_CANCELED`; two identical ranked calls produce the same transitions, cost, selected index and expansion count.

- [ ] **Step 2: Run the wheel test and verify RED**

Run from a fresh external build root after sourcing Humble:

```bash
colcon --log-base "$ARTIFACT_ROOT/log/red-ranked" build --merge-install \
  --build-base "$ARTIFACT_ROOT/build/red-ranked" \
  --install-base "$ARTIFACT_ROOT/install/red-ranked" \
  --base-paths "$MAIN_WORKTREE/ros2_ws/src" \
  --packages-select lunar_planning_msgs lunar_planner_core \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Expected: compilation fails because `SearchWheelLatticeRanked` and `selected_goal_index` do not exist.

- [ ] **Step 3: Implement ranked search with the existing tree**

In `wheel_lattice.hpp` add:

```cpp
struct WheelLatticeSearchResult final {
  WheelLatticeStatus status{WheelLatticeStatus::kInvalidRequest};
  std::optional<WheelDiscretePlan> plan;
  std::optional<std::size_t> selected_goal_index;
  std::string reason_code;
  // existing ok()
};

WheelLatticeSearchResult SearchWheelLatticeRanked(
    const WheeledState&, std::span<const GoalRegion>,
    const shared::SafeProjection&, const WheeledCapability&,
    const PlannerConfig&, std::stop_token);
```

Refactor the current function so goal 0 drives the existing admissible heuristic and exact connector. If goal 0 is not found and Open is exhausted, iterate goals 1..N-1. For each goal scan all finite discovered nodes, accept `GoalContainsPose()` or an `ExactGoalConnector()`, and retain the lowest `node.path_cost + EdgeCost()` candidate. Stop at the first ranked goal with any candidate, reconstruct from the retained parent/terminal edge and set that goal index. Do not enter fallback scanning after cancellation, invalid input, allocation failure or `start_has_valid_edge == false`.

The one-goal public function becomes:

```cpp
const std::array goals{goal};
return SearchWheelLatticeRanked(
    current_state, goals, projection, capability, config, stop_token);
```

- [ ] **Step 4: Build and verify GREEN**

Re-run the Task 1 colcon build and:

```bash
source "$ARTIFACT_ROOT/install/red-ranked/setup.bash"
"$ARTIFACT_ROOT/build/red-ranked/lunar_planner_core/lunar_planner_core_wheel_planner_test"
```

Expected: all wheel planner tests pass, including the four new ranked cases.

- [ ] **Step 5: Commit Task 1**

```bash
git add ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.hpp \
  ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.cpp \
  ros2_ws/src/lunar_planner_core/test/wheel_planner_test.cpp
git commit -m "perf: reuse wheel lattice tree across frontiers"
```

---

### Task 2: Integrate ranked frontiers into the hierarchical wheel planner

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_planner.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/planner.cpp`
- Test: `ros2_ws/src/lunar_planner_core/test/hierarchical_planner_test.cpp`
- Test: `ros2_ws/src/lunar_planner_core/test/wheel_planner_test.cpp`

**Interfaces:**
- Consumes: Task 1 `SearchWheelLatticeRanked` and `LocalFrontierResult.problems`, already ordered far-to-near.
- Produces: `RankedWheelPlannerOutput WheelPlanner::PlanRanked(const LocalPlanningProblem&, std::span<const GoalRegion>) const`, containing `PlannerOutput output` and optional `selected_goal_index`. Existing `Plan(problem)` remains source compatible.

- [ ] **Step 1: Write failing planner-integration tests**

Add this result to `wheel_planner.hpp`:

```cpp
struct RankedWheelPlannerOutput final {
  PlannerOutput output;
  std::optional<std::size_t> selected_goal_index;
};
```

Extend the existing exact-frontier and fallback hierarchical tests so that a successful fallback asserts:

```cpp
ASSERT_TRUE(output.diagnostics.hierarchical.has_value());
EXPECT_GT(output.diagnostics.hierarchical->local_attempts, 1U);
EXPECT_EQ(output.diagnostics.hierarchical->local_search_runs, 1U);
EXPECT_NE(std::ranges::find(output.diagnostics.warning_codes,
                            "LOCAL_FRONTIER_BACKOFF"),
          output.diagnostics.warning_codes.end());
```

Add a deterministic blocked-farthest fixture where at least three logical frontiers are checked but only one wheel tree is executed.

- [ ] **Step 2: Build and verify RED**

Run the Task 1 build command. Expected: compilation fails because `RankedWheelPlannerOutput`, `PlanRanked` and `local_search_runs` are absent.

- [ ] **Step 3: Implement `WheelPlanner::PlanRanked`**

Refactor `WheelPlanner::Plan()` into an internal ranked implementation. Build `MapSnapshot` and `SafeProjection` once from the common farthest `LocalPlanningProblem.local_map_view`; require at least one ranked goal and require at least one position-feasible ranked goal. Call Task 1 once, then use the selected goal for endpoint error, trajectory diagnostics and final reference. Preserve every current smoothing, sweep revalidation, timing and failure branch.

`Plan(problem)` delegates with `std::array{problem.goal_odom}` and returns `.output`.

- [ ] **Step 4: Replace the wheel per-frontier loop in `Planner::Plan`**

For wheel, construct a `std::vector<GoalRegion>` from every `frontiers.problems[i].goal_odom`, call `PlanRanked(frontiers.problems.front(), goals)` once, and treat `selected_goal_index + 1` as logical attempts. Use the selected index for `frontier_distances_m`, `LOCAL_FRONTIER_BACKOFF` and conditional-corridor retry. Keep the existing per-problem loop unchanged for legged.

Set the new metric `local_search_runs` to 1 for each wheel ranked call and to the actual calls for legged. A successful wheel output without a selected index is an internal numerical failure.

- [ ] **Step 5: Run targeted core tests**

```bash
source "$ARTIFACT_ROOT/install/red-ranked/setup.bash"
"$ARTIFACT_ROOT/build/red-ranked/lunar_planner_core/lunar_planner_core_wheel_planner_test"
"$ARTIFACT_ROOT/build/red-ranked/lunar_planner_core/lunar_planner_core_local_frontier_test"
"$ARTIFACT_ROOT/build/red-ranked/lunar_planner_core/lunar_planner_core_hierarchical_planner_test"
```

Expected: all tests pass; existing outcome, endpoint and warning tests remain unchanged except the explicit one-search assertion.

- [ ] **Step 6: Commit Task 2**

```bash
git add ros2_ws/src/lunar_planner_core/src/wheel/wheel_planner.hpp \
  ros2_ws/src/lunar_planner_core/src/wheel/wheel_planner.cpp \
  ros2_ws/src/lunar_planner_core/src/planner.cpp \
  ros2_ws/src/lunar_planner_core/test/hierarchical_planner_test.cpp \
  ros2_ws/src/lunar_planner_core/test/wheel_planner_test.cpp
git commit -m "perf: plan ranked wheel frontiers in one search"
```

---

### Task 3: Replace wheel lattice ordered trees with deterministic indexed storage

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.cpp`
- Test: `ros2_ws/src/lunar_planner_core/test/wheel_planner_test.cpp`
- Modify: `tests/performance/hierarchical_planner_benchmark.cpp`
- Modify: `tests/performance/test_hierarchical_planner_benchmark.py`

**Interfaces:**
- Consumes: Task 1 ranked search behavior.
- Produces: identical ranked references and diagnostics with a direct state index and lazy `std::priority_queue`; adds benchmark case `wheel_local_frontier_stress` with a 2.0-second Release p95 gate.

- [ ] **Step 1: Add failing determinism and stress-contract tests**

Add a test helper that invokes ranked search repeatedly and serializes selected goal index, transition primitive IDs/poses, cost and expansion count. Assert all 20 signatures are identical. Add `wheel_local_frontier_stress` to `CASES` in the Python benchmark contract with platform `WHEELED`, L0 0.2 m local map, expected `WHEEL_PLAN_AVAILABLE` and threshold `2.0`.

Expected RED: the benchmark document lacks the case; the implementation still uses `std::map`/`std::set`, which is checked by a focused source assertion in `tests/foundation/test_planner_search_semantics.py`:

```python
wheel = Path("ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.cpp").read_text()
assert "std::map<WheelLatticeState" not in wheel
assert "std::set<OpenEntry" not in wheel
assert "std::priority_queue<OpenEntry" in wheel
```

- [ ] **Step 2: Run tests and verify RED**

```bash
python3 -m pytest -q tests/foundation/test_planner_search_semantics.py
```

Expected: fail on the ordered-container assertions and missing benchmark case.

- [ ] **Step 3: Implement direct state indexing**

Compute a checked capacity `width * height * yaw_bin_count * 3`. Add:

```cpp
std::optional<std::size_t> StateIndex(
    const WheelLatticeState& key, std::size_t width, std::size_t height,
    std::size_t yaw_bins) noexcept;
```

Use `std::vector<std::size_t> node_by_state(capacity, kNoParent)` in place of the map. Reject an out-of-range generated key as `WHEEL_LATTICE_EDGE_STATE_INVALID`; catch capacity overflow/allocation as the existing allocation failure.

Use `std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenGreater>`. Do not erase stale entries on decrease-key. On pop, discard entries whose node is closed, whose `path_cost` differs from the node, or whose `sequence` differs from `node.open_sequence`. Preserve the current comparison tuple exactly by reversing it in `OpenGreater`.

- [ ] **Step 4: Add the representative stress benchmark fixture**

In the C++ benchmark create a deterministic 56x56, 0.2 m local map with a blocked far corridor and reachable nearer frontier, then run the full hierarchical wheel planner. Record complete, global and cumulative local timings. Add one warmup and 20 measured runs through the existing benchmark machinery; do not include the external `.npz` in the main repository.

- [ ] **Step 5: Build and run GREEN tests and benchmark**

```bash
python3 -m pytest -q tests/foundation/test_planner_search_semantics.py
colcon --log-base "$ARTIFACT_ROOT/log/indexed" build --merge-install \
  --build-base "$ARTIFACT_ROOT/build/indexed" \
  --install-base "$ARTIFACT_ROOT/install/indexed" \
  --base-paths "$MAIN_WORKTREE/ros2_ws/src" \
  --packages-select lunar_planning_msgs lunar_planner_core \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source "$ARTIFACT_ROOT/install/indexed/setup.bash"
"$ARTIFACT_ROOT/build/indexed/lunar_planner_core/lunar_planner_core_wheel_planner_test"
"$ARTIFACT_ROOT/build/indexed/lunar_planner_core/lunar_planner_core_hierarchical_planner_test"
LUNAR_HIERARCHICAL_PLANNER_BENCHMARK="$ARTIFACT_ROOT/build/indexed/lunar_planner_core/lunar_hierarchical_planner_benchmark" \
  python3 -m pytest -q tests/performance/test_hierarchical_planner_benchmark.py
```

Expected: all tests pass and `wheel_local_frontier_stress.p95_s <= 2.0`.

- [ ] **Step 6: Commit Task 3**

```bash
git add ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.cpp \
  ros2_ws/src/lunar_planner_core/test/wheel_planner_test.cpp \
  tests/performance/hierarchical_planner_benchmark.cpp \
  tests/performance/test_hierarchical_planner_benchmark.py \
  tests/foundation/test_planner_search_semantics.py
git commit -m "perf: index wheel lattice states directly"
```

---

### Task 4: Accumulate planner timing and retry diagnostics

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Test: `ros2_ws/src/lunar_planner_core/test/hierarchical_planner_test.cpp`
- Test: `ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp`

**Interfaces:**
- Consumes: Task 2 logical selected-frontier index and actual search-run count.
- Produces: cumulative `global_elapsed`, cumulative `local_elapsed`, `local_search_runs`, `global_replans`; ROS diagnostic keys `planner_total_elapsed_s`, `hierarchical_local_search_runs`, `hierarchical_global_replans`.

- [ ] **Step 1: Write failing cumulative metric tests**

Extend `HierarchicalPlannerMetrics` expectations in the conditional-corridor test:

```cpp
EXPECT_GE(metrics.global_replans, 1U);
EXPECT_GE(metrics.local_search_runs, 2U);
EXPECT_GE(metrics.global_elapsed, final_global_elapsed);
EXPECT_GE(metrics.local_elapsed, final_local_elapsed);
```

Extend `PlanMotionServerTest.PublishesStableHierarchicalDiagnosticMetrics` to require exact keys and values for total elapsed, local search runs and global replans.

- [ ] **Step 2: Build and verify RED**

Run the core hierarchical and ROS server tests. Expected: compilation fails on missing metric members and diagnostic keys.

- [ ] **Step 3: Accumulate metrics across retries**

In `Planner::Plan`, maintain nanosecond accumulators and counters outside the global retry loop. Add every `PlanGroundGlobalRoute().elapsed`; time every wheel ranked or legged attempt block and add it to local cumulative time. Increment `global_replans` only for a conditional-corridor global recomputation and `local_search_runs` for each actual wheel tree or legged backend call. Pass these accumulators into every success/failure `GroundMetrics()` return.

Keep `PlannerDiagnostics.elapsed` as wall time from entry to return.

- [ ] **Step 4: Publish stable ROS diagnostic keys**

Whenever `diagnostics != nullptr`, append:

```cpp
append("planner_total_elapsed_s",
       DiagnosticDouble(std::chrono::duration<double>(diagnostics->elapsed).count()));
```

When hierarchical metrics exist, append decimal values for
`hierarchical_local_search_runs` and `hierarchical_global_replans`. Keep the old aliases but make `global_search_elapsed_s` and `local_planning_elapsed_s` use the cumulative values.

- [ ] **Step 5: Run targeted and boundary tests**

```bash
source "$ARTIFACT_ROOT/install/indexed/setup.bash"
"$ARTIFACT_ROOT/build/indexed/lunar_planner_core/lunar_planner_core_hierarchical_planner_test"
"$ARTIFACT_ROOT/build/indexed/lunar_planner_ros/lunar_planner_ros_plan_motion_server_test"
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
```

Expected: all pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp \
  ros2_ws/src/lunar_planner_core/src/planner.cpp \
  ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp \
  ros2_ws/src/lunar_planner_core/test/hierarchical_planner_test.cpp \
  ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp
git commit -m "feat: report cumulative planner timing"
```

---

### Task 5: Forward timing evidence and render the 2-second budget in RViz

**Files (external repository):**
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py`
- Modify: `ros2_ws/src/lunar_isaac_rviz_plugins/include/lunar_isaac_rviz_plugins/lunar_planner_panel.hpp`
- Modify: `ros2_ws/src/lunar_isaac_rviz_plugins/src/lunar_planner_panel.cpp`
- Modify: `ros2_ws/src/lunar_isaac_rviz_plugins/test/lunar_planner_panel_test.cpp`

**Interfaces:**
- Consumes: Task 4 `/diagnostics` keys.
- Produces: exact external status keys and RViz labels `planner_total_time_label`, `global_cumulative_time_label`, `wheel_local_budget_label`, `frontier_search_count_label`, `global_replan_count_label`.

- [ ] **Step 1: Write failing Python bridge tests**

Extend the approved planner key sets and terminal diagnostic fixture. Assert `_extract_hierarchical_diagnostics()` accepts and `_build_interactive_diagnostic()` emits:

```python
{
    "planner_total_elapsed_s": "1.842",
    "global_search_elapsed_s": "0.102",
    "local_planning_elapsed_s": "1.706",
    "hierarchical_local_attempts": "7",
    "hierarchical_local_search_runs": "1",
    "hierarchical_global_replans": "0",
}
```

Add malformed negative/`nan` and reason-mismatch tests that must publish `unknown` or `-`, never prior-request values.

- [ ] **Step 2: Run Python tests and verify RED**

```bash
python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py
```

Expected: fail because the new keys are not forwarded.

- [ ] **Step 3: Write failing Qt Panel tests**

Extend the panel diagnostic helper with the new exact keys. Add tests:

```cpp
EXPECT_EQ(label("planner_total_time_label")->text(), "1.842 s");
EXPECT_EQ(label("global_cumulative_time_label")->text(), "0.102 s");
EXPECT_EQ(label("wheel_local_budget_label")->text(),
          "1.706 / 2.000 s  PASS");
EXPECT_TRUE(label("wheel_local_budget_label")->styleSheet().contains("#4caf50"));
```

Change local elapsed to `2.001` and assert red `FAIL`; change platform to `legged` and assert `1.706 s` without PASS/FAIL.

- [ ] **Step 4: Implement bridge and panel**

Add the three Task 4 keys plus `hierarchical_local_attempts` to the external controller status diagnostic. In the panel, replace the duplicate ambiguous search-time presentation with explicit rows. Parse every numeric field as finite and nonnegative. Format the budget with exactly three decimals and constant `kWheelLocalBudgetSeconds = 2.0`.

Use green `#4caf50` bold for PASS and red `#ff5252` bold for FAIL. Clear labels to `unknown` for invalid diagnostics and remove budget styling for non-wheel platforms.

- [ ] **Step 5: Build and run external unit tests**

Use a combined Release build with both source trees:

```bash
colcon --log-base "$ARTIFACT_ROOT/log/combined" build --merge-install \
  --build-base "$ARTIFACT_ROOT/build/combined" \
  --install-base "$ARTIFACT_ROOT/install/combined" \
  --base-paths "$MAIN_WORKTREE/ros2_ws/src" "$EXTERNAL_WORKTREE/ros2_ws/src" \
  --packages-up-to lunar_planner_ros lunar_isaac_validation lunar_isaac_rviz_plugins \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source "$ARTIFACT_ROOT/install/combined/setup.bash"
"$ARTIFACT_ROOT/build/combined/lunar_isaac_rviz_plugins/lunar_planner_panel_test"
python3 -m pytest -q "$EXTERNAL_WORKTREE/ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py"
```

Expected: all pass.

- [ ] **Step 6: Commit external Task 5**

```bash
git add ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_node.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py \
  ros2_ws/src/lunar_isaac_rviz_plugins/include/lunar_isaac_rviz_plugins/lunar_planner_panel.hpp \
  ros2_ws/src/lunar_isaac_rviz_plugins/src/lunar_planner_panel.cpp \
  ros2_ws/src/lunar_isaac_rviz_plugins/test/lunar_planner_panel_test.cpp
git commit -m "feat: show planner timing budget in rviz"
```

---

### Task 6: Qualify the exact synthetic rolling workflow and integrate both branches

**Files:**
- Modify: `tests/performance/test_hierarchical_planner_benchmark.py`
- Modify (external): `ros2_ws/src/lunar_isaac_validation/test/test_synthetic_interactive_integration.py`
- Create: `docs/validation/2026-08-07-wheel-local-2s-rviz-qualification.md`

**Interfaces:**
- Consumes: Tasks 1-5 combined Release install.
- Produces: exact seed-20260805 process evidence, p95 report outside repositories, and qualification documentation with immutable commands and commit IDs.

- [ ] **Step 1: Add failing exact-process assertions**

Extend the external `Status` parser with total time, logical frontiers, search runs and global replans. In the wheel rolling process test assert:

```python
assert terminal.local_elapsed_s <= 2.0
assert terminal.local_search_runs == 1
assert terminal.local_attempts >= 1
assert terminal.planner_total_elapsed_s >= terminal.local_elapsed_s
```

Collect at least 20 isolated rolling measurements in the qualification runner and write JSON containing p50, p95, maximum and every raw sample under the artifact root. The repository test reads the report and requires p95 <= 2.0; the report itself is not committed.

- [ ] **Step 2: Run the process test and verify RED before using the new install**

Run against the historical install. Expected: failure because the old result takes about 11 seconds and lacks the new fields.

- [ ] **Step 3: Run complete combined Release verification**

Build the final combined install, then run:

```bash
LUNAR_INTERACTIVE_INTEGRATION=1 \
LUNAR_SYNTHETIC_MAP_MANIFEST=/home/kai/CodexDownloads/lunar_navigation/ros_random_maps/synthetic-ad1d8d34d49c1056/map_manifest.json \
python3 -m pytest -q \
  "$EXTERNAL_WORKTREE/ros2_ws/src/lunar_isaac_validation/test/test_synthetic_interactive_integration.py"
```

Run all core and ROS package tests via `colcon test`, inspect `colcon test-result --verbose`, then run repository boundary checks and `git diff --check` in both repositories.

- [ ] **Step 4: Write qualification documentation**

Record host baseline, CPU governor, main/external commit IDs, Release CMake evidence, map ID/hash, exact target, raw/p50/p95/max timing, selected frontier/search counts, test totals and artifact paths. State clearly that the current running RViz still uses the historical install until restarted with the new qualified install.

- [ ] **Step 5: Commit qualification docs and tests**

Commit the main performance contract and qualification doc to the main branch, and the external process test to the external branch with explicit file lists.

- [ ] **Step 6: Fast-forward local integration branches and reverify**

After both feature branches are clean and all tests pass:

```bash
git -C /mnt/data/WS/lunar-navigation merge --ff-only feature/wheel-local-2s-rviz-timing
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression \
  merge --ff-only feature/wheel-local-2s-rviz-timing
```

Re-run the combined Release smoke from the merged `integration` and external `main`. Do not push or publish remotely.
