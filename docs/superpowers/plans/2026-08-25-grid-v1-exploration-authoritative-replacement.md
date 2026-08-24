# Grid Traversability V1 Exploration Authoritative Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the exploration branch planner with frozen Grid Traversability V1 commit `bf9172f`, then add only the adapters required for correct frontier cost, failure handling, diagnostics, configuration, and validation.

**Architecture:** Planner packages and planner-owned interfaces are imported path-for-path from the frozen source commit. Exploration remains the stop-plan-execute orchestrator and consumes the V1 bounded local reference, full-route cost, detailed reason codes, and diagnostic superset through existing ROS messages. Each semantic adapter is test-first and committed separately from the mechanical import.

**Tech Stack:** C++20, ROS 2 Jazzy/Humble, `rclcpp`, ROS 2 Actions, `nav_msgs`, `grid_map_msgs`, GoogleTest, pytest, colcon, Docker `osrf/ros:humble-desktop-full-jammy`.

**Spec:** `docs/superpowers/specs/2026-08-25-grid-v1-exploration-authoritative-replacement-design.md`

## Global Constraints

- Freeze the planner authority at `bf9172f09dba9f9bcf5bcf6477ef1b10f29ddec6`; do not follow later source-branch commits during this implementation.
- Preserve `/Car/T3/...` inputs, `/Car/T4/...` planning/exploration outputs, and `/Car/T5/Car_Cmd_Vel`.
- Do not change frontier detection, information gain, candidate count, candidate ordering rules, Grid V1 search algorithms, controller algorithm, vehicle footprint, clearance, map resolution, or obstacle thresholds.
- `UNKNOWN` remains non-searchable but is not an inflation source; only occupied or terrain-blocked cells inflate.
- Exploration completes only with `COMPLETED_NO_REACHABLE_FRONTIER`; coverage is reported without a threshold.
- Do not add timestamp, map-version, covariance, map-freshness, or observation-time admission checks.
- Do not use ordinary branch merge or whole-tree cherry-pick; import only the planner-owned paths listed below.
- Keep runtime artifacts outside Git under `~/CodexDownloads/lunar_navigation/`.
- Record Jazzy, Humble, Orin, DDS, rosbag, and vehicle evidence separately; unexecuted levels remain `NOT_RUN`.

## File Structure and Ownership

**Planner-authoritative paths imported from `bf9172f`:**

- `ros2_ws/src/lunar_pure_planner_core/**`
- `ros2_ws/src/lunar_pure_planner_ros/**`
- `ros2_ws/src/lunar_planning_msgs/**`
- `config/pure_planner.yaml`
- `config/external_interfaces.yaml`
- `launch/local_traversability.launch.py`
- `launch/lunar_surface_rviz_demo.launch.py`
- `launch/lunar_surface_rviz_v1_demo.launch.py`
- `launch/pure_planner.launch.py`
- `tools/create_car_orin_bundle.py`
- `tests/test_car_orin_bundle.py`
- `docs/superpowers/specs/2026-08-24-high-resolution-traversability-planner-v1-design.md`
- `docs/validation/2026-08-24-high-resolution-traversability-planner-v1.md`

**Exploration-owned paths preserved from the exploration branch:**

- `ros2_ws/src/lunar_pure_exploration_core/**`
- `ros2_ws/src/lunar_pure_exploration_ros/**`
- `ros2_ws/src/lunar_pure_exploration_sim/**`
- `ros2_ws/src/lunar_pure_wheeled_controller/**`
- `config/pure_exploration.yaml`
- `launch/jazzy_300m_exploration_sim.launch.py`
- `launch/pure_exploration.launch.py`
- `rviz/jazzy_300m_exploration_sim.rviz`
- `scripts/run_jazzy_300m_exploration_sim.sh`
- exploration, controller, simulation, and operator tests

---

### Task 1: Import the Frozen Planner Authority

**Files:**
- Replace from authority: all planner-authoritative paths in the File Structure section
- Preserve: all exploration-owned paths in the File Structure section

**Interfaces:**
- Consumes: Git object `bf9172f09dba9f9bcf5bcf6477ef1b10f29ddec6`
- Produces: one buildable tree containing the exact V1 planner plus unchanged exploration/controller/simulation packages

- [ ] **Step 1: Verify source and target identities before copying**

Run:

```bash
git status --short --branch
git rev-parse HEAD
git cat-file -t bf9172f09dba9f9bcf5bcf6477ef1b10f29ddec6
```

Expected: clean adaptation branch based on the design/plan commit; the frozen object type is `commit`.

- [ ] **Step 2: Import only planner-owned paths from the frozen Git tree**

Generate the exact manifest with:

```bash
git diff --name-status --no-renames HEAD..bf9172f -- \
  ros2_ws/src/lunar_pure_planner_core \
  ros2_ws/src/lunar_pure_planner_ros \
  ros2_ws/src/lunar_planning_msgs \
  config/pure_planner.yaml config/external_interfaces.yaml \
  launch/local_traversability.launch.py \
  launch/lunar_surface_rviz_demo.launch.py \
  launch/lunar_surface_rviz_v1_demo.launch.py \
  launch/pure_planner.launch.py \
  tools/create_car_orin_bundle.py tests/test_car_orin_bundle.py \
  docs/superpowers/specs/2026-08-24-high-resolution-traversability-planner-v1-design.md \
  docs/validation/2026-08-24-high-resolution-traversability-planner-v1.md
```

For every `A` or `M` entry, read the authoritative bytes with `git show bf9172f:<path>` and replace the target through `apply_patch`. For every `D` entry, delete only that explicit file through `apply_patch`. Do not import the deletion of `launch/jazzy_300m_exploration_sim.launch.py` visible in the unrestricted branch diff.

- [ ] **Step 3: Verify authoritative equality before adapters**

Run one `git diff --exit-code bf9172f -- <path>` for every planner-authoritative path. For `launch/`, compare only the four named planner launch files. Also run:

```bash
test -f launch/jazzy_300m_exploration_sim.launch.py
test -d ros2_ws/src/lunar_pure_exploration_ros
test -d ros2_ws/src/lunar_pure_exploration_sim
test -d ros2_ws/src/lunar_pure_wheeled_controller
git diff --check
```

Expected: every authority comparison is empty and every preserved path exists.

- [ ] **Step 4: Run the imported planner's fast non-ROS checks**

Run:

```bash
python3 -m pytest -q tests/test_car_orin_bundle.py tests/test_action_contract.py tests/test_external_interface_contract.py tests/test_launch_contract.py
```

Expected: all selected pytest cases pass.

- [ ] **Step 5: Commit the mechanical import**

```bash
git add config/pure_planner.yaml config/external_interfaces.yaml \
  launch/local_traversability.launch.py \
  launch/lunar_surface_rviz_demo.launch.py \
  launch/lunar_surface_rviz_v1_demo.launch.py \
  launch/pure_planner.launch.py \
  ros2_ws/src/lunar_pure_planner_core \
  ros2_ws/src/lunar_pure_planner_ros \
  ros2_ws/src/lunar_planning_msgs \
  tools/create_car_orin_bundle.py tests/test_car_orin_bundle.py \
  docs/superpowers/specs/2026-08-24-high-resolution-traversability-planner-v1-design.md \
  docs/validation/2026-08-24-high-resolution-traversability-planner-v1.md
git commit -m "feat: import grid traversability v1 planner"
```

### Task 2: Export Full Global Route Cost Without Changing the Action ABI

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/grid_v1/grid_v1_planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/grid_v1_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/message_conversion_test.cpp`

**Interfaces:**
- Consumes: `GlobalRoutePreview`, `PlanningResult.best_cost`, `PlannerDiagnostics.has_best_cost/best_cost`
- Produces: finite non-negative complete global grid-route length in `PlanningResult.best_cost` and unchanged bounded local `MotionReference.preview`

- [ ] **Step 1: Add a failing core test for a route longer than the local horizon**

Add a Grid V1 test that creates a straight free map with a goal more than 20 m from the start and asserts:

```cpp
ASSERT_EQ(result.status, PlanningStatus::kSuccess);
ASSERT_TRUE(result.best_cost.has_value());
EXPECT_GT(*result.best_cost, 20.0);
ASSERT_TRUE(result.reference.has_value());
EXPECT_LT(LocalPreviewLength(*result.reference), 9.0);
EXPECT_GT(GlobalPreviewLength(result.global_route_preview), 20.0);
EXPECT_NEAR(*result.best_cost,
            GlobalPreviewLength(result.global_route_preview), 1.0e-9);
```

- [ ] **Step 2: Run the focused core test and confirm the missing cost**

Run from a sourced Jazzy workspace build directory:

```bash
colcon test --packages-select lunar_pure_planner_core \
  --ctest-args -R lunar_pure_planner_core_grid_v1_planner_test --output-on-failure
colcon test-result --verbose
```

Expected: failure because `result.best_cost` is empty.

- [ ] **Step 3: Implement complete route length in Grid V1**

Add a focused helper beside the existing grid geometry helpers:

```cpp
double RouteLengthMeters(const TraversabilitySnapshot& snapshot,
                         const std::vector<Cell>& path) {
  long double total = 0.0L;
  for (std::size_t i = 1U; i < path.size(); ++i) {
    total += static_cast<long double>(Heuristic(path[i - 1U], path[i])) *
             static_cast<long double>(snapshot.resolution_m());
  }
  return static_cast<double>(total);
}
```

Compute it immediately after successful global search and set it on the success `PlanningResult`:

```cpp
const double global_route_cost_m = RouteLengthMeters(snapshot, global.path);
// ... preserve local planning and reference construction ...
.best_cost = global_route_cost_m,
```

Reject non-finite or negative computed values as `PLANNER_ERROR`; do not substitute straight-line distance.

- [ ] **Step 4: Add and run ROS conversion coverage**

Extend `message_conversion_test.cpp` to assert an input `PlanningResult` with `best_cost=23.5` produces:

```cpp
EXPECT_TRUE(converted.diagnostics.has_best_cost);
EXPECT_DOUBLE_EQ(converted.diagnostics.best_cost, 23.5);
```

Also retain the existing empty-cost case with `has_best_cost == false`.

Run:

```bash
colcon test --packages-select lunar_pure_planner_core lunar_pure_planner_ros \
  --ctest-args -R 'grid_v1_planner_test|message_conversion_test' --output-on-failure
colcon test-result --verbose
```

Expected: focused tests pass.

- [ ] **Step 5: Commit the route-cost contract**

```bash
git add ros2_ws/src/lunar_pure_planner_core/src/grid_v1/grid_v1_planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/grid_v1_planner_test.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/message_conversion_test.cpp
git commit -m "feat: expose grid v1 global route cost"
```

### Task 3: Preserve Detailed V1 Failure Semantics

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/message_conversion.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/message_conversion_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp`

**Interfaces:**
- Consumes: `PlanningResult.status` and detailed `PlanningResult.reason_code`
- Produces: Action outcome/reason pairs that retain the V1 reason and distinguish candidate exhaustion, stale invalidation, input failure, and planner failure

- [ ] **Step 1: Write a failing conversion matrix test**

Add parameterized cases with these exact expectations:

```text
kNoPath       GOAL_NOT_FREE          -> GOAL_INFEASIBLE / GOAL_NOT_FREE
kNoPath       GLOBAL_NO_PATH         -> GOAL_INFEASIBLE / GLOBAL_NO_PATH
kNoPath       LOCAL_NO_CANDIDATE     -> GOAL_INFEASIBLE / LOCAL_NO_CANDIDATE
kNoPath       LOCAL_NO_PATH          -> GOAL_INFEASIBLE / LOCAL_NO_PATH
kNoPath       STALE_PATH_INVALIDATED -> ACTIVE_REFERENCE_INVALIDATED / STALE_PATH_INVALIDATED
kNoPath       START_NOT_FREE         -> INVALID_REQUEST / START_NOT_FREE
kInvalidInput MAP_RESOLUTION_MISMATCH -> INVALID_REQUEST / MAP_RESOLUTION_MISMATCH
kPlannerError POSTCHECK_FAILED       -> NUMERICAL_FAILURE / POSTCHECK_FAILED
kPlannerError PLANNER_ERROR          -> NUMERICAL_FAILURE / PLANNER_ERROR
```

Every failure must have `NO_SAFE_REFERENCE`, `has_reference == false`, and an empty reference.

- [ ] **Step 2: Run the conversion test and confirm reason folding**

```bash
colcon test --packages-select lunar_pure_planner_ros \
  --ctest-args -R message_conversion_test --output-on-failure
colcon test-result --verbose
```

Expected: detailed cases fail because the current conversion replaces them with coarse reasons.

- [ ] **Step 3: Implement reason-aware failure conversion**

Change the internal helper to accept the source reason:

```cpp
void SetFailure(PlanningStatus status, std::string_view source_reason,
                Action::Result& result);
```

Map the exact matrix from Step 1. Preserve existing `TIMEOUT`, `REQUEST_CANCELED`, `GOAL_OUTSIDE_LOCAL_MAP`, and empty-reason fallbacks. Call it from `ConvertResult(source.status, source.reason_code, result)` without changing `PlanMotion.action`.

- [ ] **Step 4: Add server regression for publish-check invalidation**

Extend `pure_plan_motion_server_test.cpp` so a path invalidated by the latest traversability snapshot returns `ACTIVE_REFERENCE_INVALIDATED / STALE_PATH_INVALIDATED` and publishes no executable reference.

Run:

```bash
colcon test --packages-select lunar_pure_planner_ros \
  --ctest-args -R 'message_conversion_test|pure_plan_motion_server_test' --output-on-failure
colcon test-result --verbose
```

Expected: both targets pass.

- [ ] **Step 5: Commit detailed failure propagation**

```bash
git add ros2_ws/src/lunar_pure_planner_ros/src/message_conversion.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/message_conversion_test.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp
git commit -m "fix: preserve grid v1 planning reasons"
```

### Task 4: Adapt Exploration Cost, Failure Classification, and Diagnostic Supersets

**Files:**
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/src/planner_client.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_planner_client.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/src/planner_timing_accumulator.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_planner_timing_accumulator.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_synthetic_scenarios.cpp`

**Interfaces:**
- Consumes: Action result diagnostics `has_best_cost/best_cost`, detailed Action outcome/reason, and diagnostic key/value arrays
- Produces: `PlannerEvaluation.path_length_m` from full route cost; correct exhaustive/retryable/contract-error classification; timing ingestion with required-base-field semantics

- [ ] **Step 1: Write failing PlannerClient cost tests**

Add cases proving:

```cpp
result.diagnostics.has_best_cost = true;
result.diagnostics.best_cost = 31.25;
// path_preview is a valid 8 m local path
EXPECT_DOUBLE_EQ(evaluation.path_length_m.value(), 31.25);
```

Also prove `NaN`, infinity, or negative `best_cost` is a contract error, and `has_best_cost == false` retains the legacy `path_preview` fallback.

- [ ] **Step 2: Write failing failure-classification tests**

Verify the four candidate-exhaustion reasons return `kExhaustiveNoPath`; `STALE_PATH_INVALIDATED` and `TIMEOUT` return `kRetryable`; locally requested `REQUEST_CANCELED` remains `kCanceled`; `START_NOT_FREE`, `INVALID_INPUT`, `MAP_RESOLUTION_MISMATCH`, `POSTCHECK_FAILED`, and `PLANNER_ERROR` return `kContractError` and cannot enter candidate failure memory.

- [ ] **Step 3: Implement result-cost and failure classification**

Add a helper with this priority:

```cpp
std::optional<double> ResultPathLength(const Action::Result& result,
                                       std::size_t preview_limit) {
  if (result.diagnostics.has_best_cost) {
    return std::isfinite(result.diagnostics.best_cost) &&
                   result.diagnostics.best_cost >= 0.0
               ? std::optional<double>{result.diagnostics.best_cost}
               : std::nullopt;
  }
  return PathLength(result.reference, preview_limit);
}
```

Use explicit reason sets with the Action outcomes from Task 3. Do not make every `INVALID_REQUEST` exhaustive.

- [ ] **Step 4: Write failing diagnostic-superset tests**

Replace the strict unknown-field rejection case with:

```cpp
auto extended = TimingMessage("grid-v1");
extended.status.front().values.push_back(
    KeyValue{}.set__key("grid_v1_active").set__value("true"));
extended.status.front().values.push_back(
    KeyValue{}.set__key("traversability_revision").set__value("42"));
EXPECT_EQ(accumulator.Ingest(extended), TimingIngestResult::kAccepted);
```

Keep rejection tests for missing required fields, duplicate keys, multiple statuses, invalid numbers, empty IDs, and invalid base enums.

- [ ] **Step 5: Implement required-base-field timing parsing**

Remove the exact value-count check and the rejection of unknown keys. Insert every key/value into the map and reject duplicate keys; then require and parse all ten entries in `kTimingKeys`. Extra unique fields remain ignored by this accumulator.

- [ ] **Step 6: Run focused exploration tests**

```bash
colcon test --packages-select lunar_pure_exploration_ros \
  --ctest-args -R 'test_planner_client|test_planner_timing_accumulator|test_synthetic_scenarios' --output-on-failure
colcon test-result --verbose
```

Expected: all focused tests pass, including a synthetic stale-path retry that does not complete exploration.

- [ ] **Step 7: Commit the exploration adapter**

```bash
git add ros2_ws/src/lunar_pure_exploration_ros/src/planner_client.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_planner_client.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/src/planner_timing_accumulator.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_planner_timing_accumulator.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_synthetic_scenarios.cpp
git commit -m "fix: adapt exploration to grid v1 results"
```

### Task 5: Freeze V1 Launch and Unknown-Inflation Contracts

**Files:**
- Modify: `launch/jazzy_300m_exploration_sim.launch.py`
- Modify: `tests/launch/test_jazzy_300m_exploration_sim.py`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/traversability_map_v1_test.cpp`
- Modify: `docs/superpowers/specs/2026-08-24-high-resolution-traversability-planner-v1-design.md`
- Modify: `docs/superpowers/specs/2026-08-24-stop-gated-local-segment-exploration-design.md`
- Modify: `config/external_interfaces.yaml`

**Interfaces:**
- Consumes: planner node parameters and V1 traversability states
- Produces: explicit V1 activation, disabled legacy rolling session, regression evidence that UNKNOWN is not an inflation source, and consistent interface documentation

- [ ] **Step 1: Add failing launch-contract assertions**

Parse the launch source and assert the planner node parameter dictionary contains:

```python
assert planner_parameters["wheel_planner_mode"] == "grid_traversability_v1"
assert planner_parameters["rolling_surface_enabled"] is False
```

Run:

```bash
python3 -m pytest -q tests/launch/test_jazzy_300m_exploration_sim.py
```

Expected: failure because the V1 mode is not yet explicit.

- [ ] **Step 2: Set the explicit planner mode**

Add `"wheel_planner_mode": "grid_traversability_v1"` beside the existing `rolling_surface_enabled: False` planner parameter. Do not change map sizes, resolutions, sensor field of view, range, candidate limits, footprint, or simulation speed.

- [ ] **Step 3: Add the UNKNOWN non-inflation regression**

Construct a map row with `FREE`, adjacent `UNKNOWN`, and no blocked cells. After traversability generation, assert the FREE cell stays FREE even when its center lies within the inflation radius of the UNKNOWN cell. Add the mirror case with a BLOCKED cell and assert the same FREE position becomes BLOCKED. Keep UNKNOWN itself UNKNOWN and non-searchable.

Run:

```bash
colcon test --packages-select lunar_pure_planner_core \
  --ctest-args -R traversability_map_v1_test --output-on-failure
colcon test-result --verbose
```

Expected: both UNKNOWN and BLOCKED cases pass against production behavior.

- [ ] **Step 4: Correct superseded documentation and interface keys**

Change the old V1 sentence from “obstacles and unknown inflate” to “occupied/terrain-blocked cells inflate; UNKNOWN is non-searchable but does not inflate.” Change the old stop-gated design so Grid V1 `path_preview` is the bounded local path and `/Car/T4/planning/wheeled_global_path` is the complete global path. Preserve exploration-owned interfaces when adopting the V1 `external_interfaces.yaml` diagnostic/output additions.

- [ ] **Step 5: Run contract checks**

```bash
python3 -m pytest -q \
  tests/launch/test_jazzy_300m_exploration_sim.py \
  tests/test_external_interface_contract.py \
  tests/test_launch_contract.py \
  tests/test_car_orin_bundle.py
```

Expected: all selected checks pass.

- [ ] **Step 6: Commit configuration and contract updates**

```bash
git add launch/jazzy_300m_exploration_sim.launch.py \
  tests/launch/test_jazzy_300m_exploration_sim.py \
  ros2_ws/src/lunar_pure_planner_core/test/traversability_map_v1_test.cpp \
  docs/superpowers/specs/2026-08-24-high-resolution-traversability-planner-v1-design.md \
  docs/superpowers/specs/2026-08-24-stop-gated-local-segment-exploration-design.md \
  config/external_interfaces.yaml
git commit -m "test: freeze grid v1 exploration contracts"
```

### Task 6: Build, Integrate, and Run the Exploration Acceptance Sequence

**Files:**
- Create: `docs/validation/2026-08-25-grid-v1-exploration-adaptation.md`
- Modify only on observed contract mismatch: affected test or adapter file from Tasks 2-5
- Runtime output outside repository: `~/CodexDownloads/lunar_navigation/exploration_jazzy_runs/<run_id>/`

**Interfaces:**
- Consumes: the fully adapted planner/explorer/controller/simulation tree
- Produces: reproducible static, Jazzy, Humble, smoke, RViz, and complete-run evidence with explicit platform limits

- [ ] **Step 1: Run repository and Python contract suites**

```bash
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
python3 -m pytest -q tests/test_action_contract.py tests/test_external_interface_contract.py \
  tests/test_isolation_contract.py tests/test_launch_contract.py \
  tests/test_car_orin_bundle.py tests/launch/test_jazzy_300m_exploration_sim.py \
  tests/launch/test_jazzy_300m_operator.py
```

Expected: zero failures.

- [ ] **Step 2: Perform a clean local Jazzy build and package tests**

Use fresh external build, install, and log bases so no repository directory is deleted or reused. Run:

```bash
source /opt/ros/jazzy/setup.bash
adapt_jazzy_root="$HOME/CodexDownloads/lunar_navigation/grid_v1_adaptation_jazzy"
mkdir -p "$adapt_jazzy_root"
colcon --log-base "$adapt_jazzy_root/log-build" build \
  --base-paths ros2_ws/src \
  --build-base "$adapt_jazzy_root/build" \
  --install-base "$adapt_jazzy_root/install" \
  --packages-up-to lunar_pure_planner_ros lunar_pure_exploration_ros \
    lunar_pure_exploration_sim lunar_pure_wheeled_controller \
  --cmake-args -DCMAKE_BUILD_TYPE=Release --event-handlers console_direct+
source "$adapt_jazzy_root/install/setup.bash"
colcon --log-base "$adapt_jazzy_root/log-test" test \
  --build-base "$adapt_jazzy_root/build" \
  --install-base "$adapt_jazzy_root/install" \
  --packages-select lunar_pure_planner_core lunar_pure_planner_ros \
  lunar_pure_exploration_core lunar_pure_exploration_ros \
  lunar_pure_exploration_sim lunar_pure_wheeled_controller \
  --event-handlers console_direct+
colcon test-result --test-result-base "$adapt_jazzy_root/build" --verbose
```

Expected: build succeeds and all selected tests pass.

- [ ] **Step 3: Build and test the source bundle in Humble**

Generate the production bundle into a new external directory and compile that bundle in `osrf/ros:humble-desktop-full-jammy`. In a separate container invocation, mount the full adaptation worktree read-only with external writeable build/install/log bases and run the planner/explorer/controller tests. Install declared ROS dependencies in the container, use C++20, and record the exact image ID, commands, package results, and dependency installation in the validation document.

Expected: clean Humble build and selected tests pass; Orin hardware remains `NOT_RUN`.

- [ ] **Step 4: Run the 120 s smoke acceptance**

Use a unique `ROS_DOMAIN_ID`, disable automatic RViz when collecting machine evidence, and bound wall time. Start through `scripts/run_jazzy_300m_exploration_sim.sh` with the fixed seed and record:

```text
grid_v1_active=true
candidate requests admitted while stationary
at least one full-route best_cost > 8 m
non-empty bounded local trajectory
vehicle pose change and coverage increase
global/local/total planner timing
no planner contract mismatch
no false COMPLETED state after input/system error
```

Expected: all smoke assertions pass before the complete run starts.

- [ ] **Step 5: Run RViz and complete 300 m acceptance**

Run the same seed with RViz enabled and retain screenshots or a rosbag showing both `/Car/T4/planning/wheeled_global_path` and `/Car/T4/planning/wheeled_path`. Continue until the task reaches a terminal state or the explicitly documented external timeout. Accept algorithm completion only when:

```text
terminal_reason == COMPLETED_NO_REACHABLE_FRONTIER
completed_local_segments > 0
coverage statistics are present
result files are non-empty and internally consistent
input/system errors were not counted as candidate exhaustion
```

- [ ] **Step 6: Write validation evidence**

Create `docs/validation/2026-08-25-grid-v1-exploration-adaptation.md` with commit IDs, environment versions, commands, test totals, smoke metrics, complete-run result, RViz evidence paths, planner timing, reason-code distribution, coverage, and explicit `NOT_RUN` entries for Orin/DDS/vehicle evidence not executed.

- [ ] **Step 7: Run final checks and commit evidence**

```bash
git diff --check
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
git status --short
git add docs/validation/2026-08-25-grid-v1-exploration-adaptation.md
git commit -m "docs: validate grid v1 exploration adaptation"
```

Expected: only intended files are committed and the worktree is clean.
