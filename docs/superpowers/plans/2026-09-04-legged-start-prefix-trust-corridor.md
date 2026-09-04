# Legged Start Prefix Trust Corridor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow a legged planner to leave its bounded request-local start blind-zone patch and enter observed free terrain without rechecking raw unknown elevation within that one start-prefix corridor.

**Architecture:** `RequestLocalStartPatchBuilder` remains the sole authority that creates `StartAssumedFree`: it only overrides raw `UNKNOWN`, preserves `BLOCKED`, bounds overrides to the patch disk, and requires an observed-free exit. The legged directed-edge and terminal certificates will treat `StartAssumedFree` as a trusted terrain corridor only while the search state is `kStartPrefix`; they will skip raw elevation support/step/gap checks only for an edge containing an assumed cell. `AdvancePhase()` already changes to `kNormal` on `EvidenceFree` and rejects re-entry, so no map or ROS interface changes are required.

**Tech Stack:** C++20, ament_cmake, GoogleTest, ROS 2 Humble.

**Spec:** User-confirmed runtime policy in this conversation: a patch is for the start blind zone; patch obstacles remain blocked; `UNKNOWN` is usable only when it became `StartAssumedFree` inside the bounded patch; the prefix must enter real observed `EvidenceFree`; after that, full legged certification resumes and the path cannot return to the patch.

## Global Constraints

- Do not modify wheel planning, patch radius calculation, persistent traversability maps, ROS topics, launch files, or platform configuration.
- Do not change `RequestLocalStartPatchBuilder`: plane fitting and its existing ability to reject an unbuildable patch remain intact.
- `BLOCKED`, raw `UNKNOWN`, and an assumed cell reached after `kNormal` remain impassable.
- The exception is only raw elevation support/step/gap certification for a directed edge that contains `StartAssumedFree` while its source state is `kStartPrefix`.
- Motion primitive length/kind validation, local-window bounds, terminal target `EvidenceFree` requirement, and all normal-phase legged certification remain unchanged.

---

### Task 1: Capture the intended multi-cell blind-zone regression

**Files:**
- Modify: `ros2_ws/src/lunar_incremental_navigation_core/test/legged_local_planner_v2_test.cpp:374-421`

**Interfaces:**
- Consumes: `LeggedLocalPlanner::Plan(const RequestLocalPlanningView&, Pose2, LocalTarget, SearchDeadline, StopToken)`.
- Produces: a regression stating that a two-cell `StartAssumedFree` prefix with `NaN` raw elevation can reach an `EvidenceFree` exit.

- [ ] **Step 1: Replace the obsolete direct-exit expectation with a failing multi-cell prefix test.**

  Build a one-cell-wide corridor at `y=4`: `{2,4}` and `{3,4}` are raw `kUnknown` and overrides with `kStartAssumedFree`; `{4,4}` and `{5,4}` are raw `kFree`; every other cell is `kBlocked`. Set raw elevation at both assumed cells to `NaN`, `maximum_gap_width_m = 0.0`, and use only the 1 m forward primitive. Plan from `PoseAt({2,4})` to `PoseAt({5,4})`.

  Assert:

  ```cpp
  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_GE(result.raw_path.size(), 4U);
  EXPECT_EQ(result.raw_path[0].phase, StartPhase::kStartPrefix);
  EXPECT_EQ(result.raw_path[1].phase, StartPhase::kStartPrefix);
  EXPECT_EQ(result.raw_path[2].phase, StartPhase::kNormal);
  EXPECT_EQ(result.raw_path.back().phase, StartPhase::kNormal);
  ```

- [ ] **Step 2: Run the focused test and confirm the current code fails.**

  Run:

  ```bash
  cd /home/kai/WS/lunar-navigation/lunar-runtime/.worktrees/pure-planner-orin/ros2_ws
  colcon test --packages-select lunar_incremental_navigation_core \
    --ctest-args -R '^lunar_incremental_navigation_core_legged_local_planner_v2_test$'
  colcon test-result --verbose
  ```

  Expected before the repair: the new test reports `kNoPath`, because the directed edge rejects a non-anchor assumed target and raw `NaN` elevation fails the gap/support check.

### Task 2: Permit the bounded trusted prefix in legged edge and terminal certification

**Files:**
- Modify: `ros2_ws/src/lunar_incremental_navigation_core/src/legged/legged_local_planner.cpp:480-548`
- Modify: `ros2_ws/src/lunar_incremental_navigation_core/src/legged/legged_local_planner.cpp:657-670`

**Interfaces:**
- Consumes: `SearchPhase(source_state)`, `RequestLocalPlanningView::Source()`, `RequestLocalPlanningView::AdvancePhase()`, and `ElevationStepAndGapFeasible()`.
- Produces: a feasible certificate for a `kStartPrefix` edge containing only `kEvidenceFree` and `kStartAssumedFree` cells; `uses_assumed_support=true` records the exception for the existing phase checks.

- [ ] **Step 1: Define prefix eligibility locally in `LeggedDirectedEdgeCache::Certify`.**

  After computing `source_cell`, define:

  ```cpp
  const bool source_is_start_prefix =
      SearchPhase(source_state) == StartPhase::kStartPrefix;
  const bool source_is_assumed =
      source_kind == LocalCellSource::kStartAssumedFree;
  ```

  Accept an assumed source or target only when `source_is_start_prefix` is true. Keep `kUnknown` and `kBlocked` rejected. This replaces the current special case that permits only the initial anchor.

- [ ] **Step 2: Treat an edge containing an assumed cell as the sole raw-elevation exception.**

  While checking `SegmentCells`, allow `kStartAssumedFree` only for `source_is_start_prefix` and set `uses_assumed_support = true`; keep the existing early return for raw `kUnknown`/`kBlocked`.

  Call `ElevationStepAndGapFeasible(...)` only when `uses_assumed_support` is false:

  ```cpp
  if (!uses_assumed_support &&
      !ElevationStepAndGapFeasible(*view.base()->elevation(), cells,
                                   capability.maximum_step_height_m,
                                   capability.maximum_gap_width_m, false)) {
    entries_.emplace(key, certificate);
    return {.certificate = certificate};
  }
  ```

  Do not bypass `SupportsTranslation()`: the available legged motion primitive still must cover the commanded translation.

- [ ] **Step 3: Make `CertifyTerminal` use the same prefix rule.**

  Its current `CanCertifyLeggedSupport(source_cell, source_at_anchor)` rejects a second assumed cell before `LeggedDirectedEdgeCache::Certify` runs. Replace that precondition with the same source rule used by the directed edge: evidence-free is accepted; assumed is accepted only in `kStartPrefix`; all other sources are rejected. Keep `view.CanBeEndpoint(target_cell)` unchanged, so terminal targets remain raw observed free.

- [ ] **Step 4: Run the focused regression.**

  Run the Task 1 commands again. Expected: the multi-cell prefix is `kPlanFound`; the path has two `kStartPrefix` points, then `kNormal` at the observed-free exit.

### Task 3: Lock down non-goals and normal certification

**Files:**
- Modify: `ros2_ws/src/lunar_incremental_navigation_core/test/legged_local_planner_v2_test.cpp`

**Interfaces:**
- Consumes: `RequestLocalPlanningView::AdvancePhase()` and the same fixture helper.
- Produces: regressions proving the exception cannot escape into normal planning.

- [ ] **Step 1: Add a legged no-reentry test.**

  Create a corridor with assumed `{2,4}`, observed free `{3,4}`, assumed `{4,4}`, and observed free target `{5,4}`. Start at `{2,4}` and target `{5,4}`. Assert `kNoPath`: after `{3,4}`, `AdvancePhase(kNormal, {4,4})` must reject the re-entry.

- [ ] **Step 2: Preserve observed-terrain gap rejection.**

  Run the existing `EnforcesGapWidthOnACachedDirectedEdge` test unchanged. It contains no assumed cells, so it must still return `kNoPath` when the observed corridor has an over-limit/unknown gap.

- [ ] **Step 3: Preserve blocked-cell rejection.**

  Add an assumed-prefix fixture whose only exit is `kBlocked`; do not add an override for that blocked cell. Assert `kNoPath`. This proves the exception cannot convert an obstacle to traversable terrain.

- [ ] **Step 4: Run the focused core test suite.**

  Run:

  ```bash
  cd /home/kai/WS/lunar-navigation/lunar-runtime/.worktrees/pure-planner-orin/ros2_ws
  colcon test --packages-select lunar_incremental_navigation_core \
    --ctest-args -R '^(lunar_incremental_navigation_core_legged_local_planner_v2_test|lunar_incremental_navigation_core_request_local_start_patch_test|lunar_incremental_navigation_core_wheel_local_planner_test)$'
  colcon test-result --verbose
  ```

  Expected: all three test executables pass; wheel behavior is unchanged.

### Task 4: Verify integration and Orin replay behavior

**Files:**
- No source-interface changes.

**Interfaces:**
- Consumes: existing `scripts/orin/build.sh`, `scripts/orin/start_navigation.sh`, diagnostics, and RViz goal bridge.
- Produces: host evidence plus a bounded target-side replay observation.

- [ ] **Step 1: Build the affected production closure on the host.**

  Run:

  ```bash
  cd /home/kai/WS/lunar-navigation/lunar-runtime/.worktrees/pure-planner-orin/ros2_ws
  colcon build --packages-up-to lunar_incremental_navigation_ros \
    --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
  ```

- [ ] **Step 2: Deploy the complete source tree, excluding generated artifacts.**

  Run from the host:

  ```bash
  SRC=/home/kai/WS/lunar-navigation/lunar-runtime/.worktrees/pure-planner-orin
  rsync -a --exclude .git --exclude ros2_ws/build --exclude ros2_ws/install \
    --exclude ros2_ws/log "$SRC/" \
    yanfa@rovercomputer:~/P4/orin-humble-c386746/
  ```

- [ ] **Step 3: Build and replay on Orin.**

  In terminal A, start the patched runtime and replay the legged bag. In terminal B, send exactly one RViz goal or action goal after the map and odometry have arrived; do not issue later clicks that cancel the active bridge action.

  ```bash
  cd ~/P4/orin-humble-c386746
  bash ./scripts/orin/build.sh
  ./scripts/orin/start_navigation.sh legged false
  ```

- [ ] **Step 4: Check the result boundary.**

  Expected for this regression: diagnostics show nonzero `start_patch_assumed_cells`, `PLAN_FOUND`, `local_expanded_states > 1`, and `path_reference` begins with `StartPrefix` then transitions to normal. This proves the planner can leave the blind zone; static bag odometry still means the Action need not reach its final target.

## Self-Review

- Spec coverage: Tasks 1-2 implement the one bounded trusted prefix; Task 3 proves blocked/unknown/re-entry and normal terrain certification remain intact; Task 4 covers host and Orin evidence.
- Non-goals: no new inputs, no permanent map mutation, no patch-radius expansion, no wheel semantic change, and no relaxation after observed-free exit.
- Type consistency: all changes use existing `StartPhase`, `LocalCellSource`, `RequestLocalPlanningView`, and `DirectedEdgeCertificate` interfaces.
