# Boundary-Directed Task Approach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let pure exploration start outside `PureExplorationTask.boundary`, advance through map-backed known-safe goals toward that boundary, and switch unchanged WFD exploration on only after the complete inflated footprint is safely inside the task.

**Architecture:** Add a ROS-independent `SafePoseValidator` and `BoundaryGuidance` pipeline beside the existing WFD pipeline. The ROS coordinator owns an explicit `APPROACH_TASK`/`EXPLORE_TASK` phase and a separate frozen approach cycle, but both kinds of executable goals use the same `PlannerClient -> /Car/T4/plan_motion` path and typed `ActiveGoal` ownership. UNKNOWN cells guide intent only; every sent target is current-map FREE and full-footprint safe.

**Tech Stack:** C++20, ROS 2 Jazzy/Humble, `rclcpp`, ROS 2 Actions, `nav_msgs/OccupancyGrid`, `diagnostic_msgs`, `visualization_msgs`, GoogleTest, pytest, colcon.

**Spec:** `docs/superpowers/specs/2026-08-25-boundary-directed-task-approach-design.md`

## Global Constraints

- Build the feature from exactly `feat/grid-v1-exploration-adaptation@3f5b0f746a15d48edced0c045387154eb808e59c` in an isolated worktree and feature branch.
- Use `wheel_planner_mode=grid_traversability_v1`; do not silently fall back to `legacy_certified`.
- Do not modify Grid V1 search, vehicle envelope, minimum clearance, obstacle threshold, final path checks, controller behavior, map resolution, or UNKNOWN non-traversability.
- Every actual approach target sent to `PlanMotion` must be map-backed FREE and pass the exact closed full-footprint plus clearance check.
- UNKNOWN may participate only in boundary guidance cost and visibility; it may never become an executable target or path-feasibility override.
- Keep `PureExplorationTask.msg`, `PureExplorationStatus.msg`, `/Car/T3/...`, `/Car/T4/...`, and `/Car/T5/...` contracts unchanged.
- Keep WFD detection, WFD gain/ranking, failure memory, and completion semantics unchanged for an initially fully-inside pose.
- `APPROACH_TASK` no-route/no-target/stalled results are nonterminal; only `EXPLORE_TASK` may call `CompleteNoReachableFrontier()`.
- Coverage is calculated only from `TaskRaster`; `coverage_ratio > 0.80` is a synthetic acceptance assertion, never a core termination threshold.
- Preserve unrelated dirty files and runtime artifacts; do not merge, reset, clean, push, or start chassis control.
- Record local Jazzy proof separately from Humble/Orin/DDS/vehicle proof; unrun target-host checks remain `NOT_RUN`.

## File Structure

**New core safety unit**

- `lunar_pure_exploration_core/platform_geometry.hpp`: shared platform geometry value, moved without semantic changes from `candidate_generator.hpp` to avoid include cycles.
- `lunar_pure_exploration_core/safe_pose_validator.hpp/.cpp`: the single footprint, clearance, closed-cell-contact implementation for both WFD and approach.
- `test_safe_pose_validator.cpp`: direct map-free versus task-free safety and exact resource-boundary tests.

**New core approach unit**

- `lunar_pure_exploration_core/goal_identity.hpp`: typed task-frontier and boundary-approach identity values.
- `lunar_pure_exploration_core/boundary_guidance.hpp/.cpp`: boundary intents, lexicographic guidance, safe truncation, visibility value, deterministic ranking, and wait reasons.
- `test_boundary_guidance.cpp`: all approach algorithm fixtures.

**ROS coordination**

- `exploration_node.hpp/.cpp`: phase selection, frozen approach-cycle planning, completion guard, wake signatures, map invalidation, diagnostics, and action orchestration.
- `marker_builder.hpp/.cpp`: separate approach intent/route/safe-target namespaces.
- `planner_client.hpp/.cpp`: generic safe pose input plus Grid V1 full-route cost and detailed reason handling.

**Configuration, scenarios, and evidence**

- `ExplorationNodeParameters` and parameter-loading tests: guidance limits derived from existing task/candidate limits unless explicitly overridden.
- `launch/jazzy_300m_exploration_sim.launch.py`, `scripts/start_all.sh`: explicit Grid V1 planner selection where the exploration stack starts a planner, with explicit 300 m guidance limits.
- `tests/scenarios/fixtures/outside_to_inside.yaml`: deterministic 20 m outside-start scenario.
- Existing core, ROS, launch, operator, and scenario test files: regression and acceptance coverage.
- `README.md` and `docs/validation/2026-08-25-boundary-directed-task-approach.md`: operator semantics and evidence, without changing unrelated demo prose.

---

### Task 1: Bind Exploration to the Frozen Latest Grid V1 Planner

**Files:**
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/src/planner_client.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_planner_client.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/src/planner_timing_accumulator.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_planner_timing_accumulator.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/CMakeLists.txt`
- Modify: `launch/jazzy_300m_exploration_sim.launch.py`
- Modify: `scripts/start_all.sh`
- Modify: `tests/launch/test_jazzy_300m_exploration_sim.py`
- Modify: `tests/test_operator_scripts.py`

**Interfaces:**
- Consumes: Grid V1 `PlanMotion::Result.diagnostics.has_best_cost/best_cost`, detailed `reason_code`, and planner diagnostic key/value supersets.
- Produces: `PlannerEvaluation.path_length_m` from complete global route cost; exhaustive/retryable/contract classifications; launch-time `grid_traversability_v1` activation.

- [ ] **Step 1: Add failing Grid V1 result-contract tests**

Add PlannerClient cases whose local `path_preview` length is 8.0 m but whose diagnostics contain:

```cpp
result.diagnostics.has_best_cost = true;
result.diagnostics.best_cost = 31.25;
EXPECT_EQ(evaluation.kind, PlannerEvaluationKind::kReachable);
ASSERT_TRUE(evaluation.path_length_m.has_value());
EXPECT_DOUBLE_EQ(*evaluation.path_length_m, 31.25);
```

Add a table asserting `GOAL_NOT_FREE`, `GLOBAL_NO_PATH`, `LOCAL_NO_CANDIDATE`, and `LOCAL_NO_PATH` are `kExhaustiveNoPath`; `STALE_PATH_INVALIDATED` and `TIMEOUT` are `kRetryable`; locally canceled `REQUEST_CANCELED` is `kCanceled`; `START_NOT_FREE`, `INVALID_INPUT`, `MAP_RESOLUTION_MISMATCH`, `POSTCHECK_FAILED`, and `PLANNER_ERROR` are `kContractError`. Add NaN, infinity, and negative `best_cost` cases that return `BEST_COST_INVALID`, plus `has_best_cost=false` compatibility coverage that still uses the validated preview length.

- [ ] **Step 2: Add failing diagnostic-superset and launch-mode tests**

Extend `test_planner_timing_accumulator.cpp` with the ten required base keys plus `grid_v1_active=true`, `global_input_sequence=9`, and `fused_tile_count=4`; expect `kAccepted`. Retain rejection for a duplicate required key, missing required key, or malformed required value. In the Python tests assert both runtime entries contain the exact mode. The live 300 m launch test must also observe `grid_v1_active=true` in the real planner diagnostic before accepting any exploration request:

```python
assert '"wheel_planner_mode": "grid_traversability_v1"' in launch_source
assert 'wheel_planner_mode:="grid_traversability_v1"' in start_all_source
```

- [ ] **Step 3: Run the focused tests and record the expected failures**

Run:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install --packages-up-to lunar_pure_exploration_ros
colcon test --packages-select lunar_pure_exploration_ros --ctest-args -R 'test_planner_client|test_planner_timing_accumulator' --output-on-failure
cd ..
python3 -m pytest -q tests/launch/test_jazzy_300m_exploration_sim.py tests/test_operator_scripts.py
```

Expected: the new cases fail because preview cost is used, detailed Grid V1 reasons are rejected, extra diagnostics are rejected, and runtime mode is not explicit.

- [ ] **Step 4: Implement the minimum Grid V1 binding**

In success classification, require finite nonnegative `best_cost` when `has_best_cost`, otherwise use the existing bounded preview calculation. Implement exact reason sets rather than prefix matching. In timing parsing, reject duplicate required keys but ignore unique unknown extension keys. Set the 300 m planner node parameter and bundle startup argument explicitly:

```python
"wheel_planner_mode": "grid_traversability_v1",
"rolling_surface_enabled": False,
```

```bash
wheel_planner_mode:="grid_traversability_v1"
```

Update the planner compatibility digest gate to the frozen `3f5b0f7` bytes:

```cmake
397be394bcf8c460dc2654bc2ef45c7181914461007367e338da2395f862a160
bcd4dd1be9e56d6495b63171651b8449994d292d6451f3be960fd0fa72235d63
ddad903a6838a61e3018ce8a5e98173276de6971baf9a87702fe94970f44dd0a
```

- [ ] **Step 5: Rebuild, rerun focused tests, and commit**

Run the Step 3 commands and `git diff --check`. Expected: all selected tests pass and configuration reports no planner-source digest error.

Commit only the listed files:

```bash
git add ros2_ws/src/lunar_pure_exploration_ros/src/planner_client.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_planner_client.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/src/planner_timing_accumulator.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_planner_timing_accumulator.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/CMakeLists.txt \
  launch/jazzy_300m_exploration_sim.launch.py scripts/start_all.sh \
  tests/launch/test_jazzy_300m_exploration_sim.py tests/test_operator_scripts.py
git commit -m "fix: bind exploration to grid v1 planner"
```

---

### Task 2: Extract the Exact Shared Safe Pose Validator

**Files:**
- Create: `ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/platform_geometry.hpp`
- Create: `ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/safe_pose_validator.hpp`
- Create: `ros2_ws/src/lunar_pure_exploration_core/src/safe_pose_validator.cpp`
- Create: `ros2_ws/src/lunar_pure_exploration_core/test/test_safe_pose_validator.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/candidate_generator.hpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_core/src/candidate_generator.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_core/test/test_candidate_generator.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_core/CMakeLists.txt`

**Interfaces:**
- Consumes: `PlatformGeometry`, `Pose2`, `OccupancyGridView`, `TaskRaster`, and a cumulative work counter.
- Produces: `SafePoseValidator::IsMapFree(...)`, `IsTaskFree(...)`, platform dimensions, standoff, identical closed collision math, and the existing pose quantizer exposed as `MakeCandidateKey(Pose2)`.

- [ ] **Step 1: Add the public validator contract and failing tests**

Define:

```cpp
class SafePoseValidator final {
 public:
  SafePoseValidator(PlatformGeometry platform,
                    std::size_t maximum_collision_work_units);
  bool IsMapFree(const OccupancyGridView& map, Pose2 pose,
                 std::size_t& consumed_work) const;
  bool IsTaskFree(const TaskRaster& raster, Pose2 pose,
                  std::size_t& consumed_work) const;
  double platform_length_m() const;
  double platform_width_m() const;
  double footprint_circumscribed_radius_m() const;
  double minimum_standoff_m() const;
};
```

Move `PlatformGeometry` byte-for-byte into `platform_geometry.hpp`; include that file from both validator and candidate generator. Declare `CandidateKey MakeCandidateKey(Pose2 pose);` next to `CandidateKey` so boundary candidates use the same millimetre/tenth-degree identity as WFD candidates.

Tests must prove an outside-task but map-FREE pose passes `IsMapFree` and fails `IsTaskFree`; UNKNOWN/OCCUPIED/OUTSIDE_MAP fail both; boundary tangency fails task-free; rotated footprint plus clearance matches the existing approved fixtures; exactly-at-limit work succeeds and one-less fails with `std::length_error`.

- [ ] **Step 2: Run the new target and verify it fails to link**

Run:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install --packages-select lunar_pure_exploration_core
```

Expected: compile/link failure because `SafePoseValidator` has no implementation.

- [ ] **Step 3: Move, do not rewrite, the collision implementation**

Move footprint normalization, transforms, polygon-cell closed distance, clearance comparison, work checks, and dimension derivation from `candidate_generator.cpp` into `safe_pose_validator.cpp`. Implement one internal traversal accepting a classifier lambda; `IsMapFree` classifies through `OccupancyGridView::Classify`, while `IsTaskFree` additionally requires `TaskRaster::IsMapBacked` and `TaskRaster::Classify == kFree`. Keep the existing 64-epsilon closed tolerance exactly. Rename the existing internal `BuildKey` implementation to the public `MakeCandidateKey`; do not change its normalization or quantization.

- [ ] **Step 4: Make CandidateGenerator delegate and prove equivalence**

Replace its private platform/collision ownership with `SafePoseValidator validator_`; preserve public metric methods by delegation. In `Generate`, replace `CollisionFree(...)` with:

```cpp
if (!validator_.IsTaskFree(raster, pose, collision_work)) {
  continue;
}
```

Add a test comparing the complete ordered `CandidateKey` vector for the existing rotated-map, tangency, and concave-boundary fixtures against the frozen expected vectors already in `test_candidate_generator.cpp`.

- [ ] **Step 5: Run all core tests and commit**

Run:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install --packages-select lunar_pure_exploration_core
colcon test --packages-select lunar_pure_exploration_core --event-handlers console_direct+ --return-code-on-test-failure
colcon test-result --verbose
```

Expected: every pre-existing core test and `test_safe_pose_validator` passes.

Commit:

```bash
git add ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/platform_geometry.hpp \
  ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/safe_pose_validator.hpp \
  ros2_ws/src/lunar_pure_exploration_core/src/safe_pose_validator.cpp \
  ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/candidate_generator.hpp \
  ros2_ws/src/lunar_pure_exploration_core/src/candidate_generator.cpp \
  ros2_ws/src/lunar_pure_exploration_core/test/test_safe_pose_validator.cpp \
  ros2_ws/src/lunar_pure_exploration_core/test/test_candidate_generator.cpp \
  ros2_ws/src/lunar_pure_exploration_core/CMakeLists.txt
git commit -m "refactor: share exact exploration pose safety"
```

---

### Task 3: Add Typed Task-Frontier and Boundary-Approach Goal Ownership

**Files:**
- Create: `ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/goal_identity.hpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/exploration_state_machine.hpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_core/src/exploration_state_machine.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_core/test/test_exploration_state_machine.cpp`

**Interfaces:**
- Consumes: existing `CandidateKey`, frontier canonical keys, intent `GridIndex`, and request ID.
- Produces: `GoalKind`, `ApproachCandidateKind`, `TaskFrontierGoalIdentity`, `BoundaryApproachGoalIdentity`, and typed `ActiveGoal` factories.

- [ ] **Step 1: Define identities and failing state-machine tests**

Add:

```cpp
enum class GoalKind : std::uint8_t { kBoundaryApproach, kTaskFrontier };
enum class ApproachCandidateKind : std::uint8_t { kTranslation, kRotation };
struct TaskFrontierGoalIdentity {
  std::vector<std::int64_t> frontier_canonical_key;
  CandidateKey candidate_key;
};
struct BoundaryApproachGoalIdentity {
  GridIndex intent_cell;
  CandidateKey candidate_key;
  ApproachCandidateKind candidate_kind;
  auto operator<=>(const BoundaryApproachGoalIdentity&) const = default;
};
```

Test that a task goal still matches only the full canonical frontier key; an approach goal exposes `kind()==kBoundaryApproach`, has no frontier key, owns its intent/key/kind after source storage is destroyed, and `MatchesAnyFrontier` returns false.

- [ ] **Step 2: Run the focused test and verify compile failure**

Run `colcon test --packages-select lunar_pure_exploration_core --ctest-args -R test_exploration_state_machine --output-on-failure` after building. Expected: missing typed APIs.

- [ ] **Step 3: Implement variant-backed ActiveGoal without changing lifecycle transitions**

Store `std::variant<TaskFrontierGoalIdentity, BoundaryApproachGoalIdentity>`. Preserve `MakeActiveGoal(const CandidateView&, ...)` as the task-frontier factory for source compatibility, and add:

```cpp
ActiveGoal MakeBoundaryApproachActiveGoal(
    std::uint64_t display_id, BoundaryApproachGoalIdentity identity,
    Pose2 target, std::string request_id);
```

Add `kind()`, `candidate_key()`, `target()`, `request_id()`, and `boundary_approach_identity()` accessors. Do not change START/PAUSE/RESUME/CANCEL/replan counters or release reasons.

- [ ] **Step 4: Run the complete state-machine and core regression, then commit**

Run the Task 2 Step 5 core commands. Expected: all tests pass.

Commit:

```bash
git add ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/goal_identity.hpp \
  ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/exploration_state_machine.hpp \
  ros2_ws/src/lunar_pure_exploration_core/src/exploration_state_machine.cpp \
  ros2_ws/src/lunar_pure_exploration_core/test/test_exploration_state_machine.cpp
git commit -m "feat: type exploration goal ownership"
```

---

### Task 4: Implement Deterministic Boundary Guidance and Approach Ranking

**Files:**
- Create: `ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/boundary_guidance.hpp`
- Create: `ros2_ws/src/lunar_pure_exploration_core/src/boundary_guidance.cpp`
- Create: `ros2_ws/src/lunar_pure_exploration_core/test/test_boundary_guidance.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_core/CMakeLists.txt`

**Interfaces:**
- Consumes: `OccupancyGridView`, `TaskRaster`, `Pose2`, `PlatformGeometry`, `SensorModel`, `SafePoseValidator`, and resource limits.
- Produces: owned `BoundaryGuidanceResult`, stable approach candidates, coarse/final index order, phase predicate, and nonterminal wait reason.

- [ ] **Step 1: Add the complete public data contract**

Define exact owning types:

```cpp
enum class NavigationPhase : std::uint8_t { kApproachTask, kExploreTask };
enum class ApproachWaitReason : std::uint8_t {
  kNone, kWaitingForSafeStart, kWaitingForTaskMapCoverage,
  kNoGuidanceRoute, kNoSafeCandidate
};
struct GuidanceCost { std::uint32_t unknown_cell_count; double path_length_m; };
struct ApproachIntent { GridIndex cell; GuidanceCost total_cost;
                        std::vector<GridIndex> route; };
struct ApproachCandidate {
  std::uint64_t id;
  BoundaryApproachGoalIdentity identity;
  Pose2 pose;
  GuidanceCost remaining_cost;
  double task_unknown_area_m2;
  std::uint32_t guidance_unknown_cell_count;
  bool fully_inside_task;
  std::vector<Vec2> guidance_route;
};
struct BoundaryGuidanceResult {
  NavigationPhase phase;
  ApproachWaitReason wait_reason;
  bool fully_inside_task;
  std::vector<ApproachIntent> intents;
  std::vector<ApproachCandidate> candidates;
  std::size_t consumed_work_units;
};
```

`BoundaryGuidance::Limits` contains `maximum_guidance_grid_cells`, `maximum_guidance_work_units`, and `maximum_approach_candidates`. Its constructor receives `PlatformGeometry`, `SensorModel`, goal yaw tolerance, and the limits. Add `Build(map, raster, robot_pose)`, `CoarseOrder(result, robot_pose)`, and `FinalOrder(result, planned, robot_pose)`.

- [ ] **Step 2: Write failing algorithm tests before implementation**

Cover: outside start yields a FREE executable target toward a task UNKNOWN intent; closest blocked entrance selects another boundary; known detour beats UNKNOWN shortcut by `(unknown_count,path_length)`; map rotation/negative logical origin/concave polygon/vertex order remain deterministic; safe-prefix truncation stops before UNKNOWN; same-XY rotation is allowed only when yaw changes beyond tolerance and sees route UNKNOWN; zero-observation candidates are rejected; fully-inside footprint returns `kExploreTask`; OUTSIDE_MAP boundary returns map-coverage wait; unsafe start returns safe-start wait; exact limits pass and one-less throws `std::length_error`.

Use assertions that inspect every candidate:

```cpp
std::size_t work = 0U;
EXPECT_TRUE(validator.IsMapFree(map, candidate.pose, work));
EXPECT_NE(map.Classify(*map.WorldToCell({candidate.pose.x, candidate.pose.y})),
          CellState::kUnknown);
```

- [ ] **Step 3: Run `test_boundary_guidance` and verify failure**

Build the core package and run `colcon test --packages-select lunar_pure_exploration_core --ctest-args -R test_boundary_guidance --output-on-failure`. Expected: missing implementation.

- [ ] **Step 4: Implement multi-target guidance and safe truncation**

Enumerate task cells adjacent by four-neighborhood to `kOutsideTask`; accept only global FREE/UNKNOWN intents. Use one four-neighbor Dijkstra whose comparison is exact lexicographic `(unknown_count, path_length_m, predecessor world x, predecessor world y, logical index)`. Reconstruct each route, scan it with `SafePoseValidator::IsMapFree`, stop at the first unsafe pose, and create only strict-progress translation or visibility-producing rotation candidates. Deduplicate by full `BoundaryApproachGoalIdentity`; use checked arithmetic before allocations and per-work increment.

- [ ] **Step 5: Implement stable approach ranks**

Coarse order is `fully_inside_task` descending, remaining unknown count ascending, remaining metres ascending, task UNKNOWN area descending, Euclidean distance ascending, full identity ascending. Final order replaces Euclidean distance with actual `PlannedCandidate.path_length_m`, then heading change, then identity. Reject nonfinite metrics instead of changing ordering.

- [ ] **Step 6: Run all core tests and commit**

Run the Task 2 Step 5 commands and `git diff --check`. Expected: every core target passes.

Commit:

```bash
git add ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/boundary_guidance.hpp \
  ros2_ws/src/lunar_pure_exploration_core/src/boundary_guidance.cpp \
  ros2_ws/src/lunar_pure_exploration_core/test/test_boundary_guidance.cpp \
  ros2_ws/src/lunar_pure_exploration_core/CMakeLists.txt
git commit -m "feat: add boundary-directed approach guidance"
```

---

### Task 5: Integrate a Separate Frozen Approach Cycle into the ROS Coordinator

**Files:**
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/planner_client.hpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/src/planner_client.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/exploration_node.hpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/src/exploration_node.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_planner_client.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_node.cpp`

**Interfaces:**
- Consumes: `BoundaryGuidanceResult`, typed `ActiveGoal`, `PlannerClient`, existing stationary gate and generation/epoch guards.
- Produces: separate `FrozenApproachCycle`, sequential Grid V1 validation of up to 16 approach candidates, approach commitment/execution, and safe phase switching.

- [ ] **Step 1: Generalize only the PlannerClient target input**

Replace the `CandidateView` argument with a neutral value:

```cpp
struct PlannerTarget final {
  std::uint64_t display_id;
  lunar::pure_exploration::Pose2 pose;
};
void Evaluate(std::string task_id, std::string request_id,
              PlannerTarget target, double position_tolerance_m,
              double yaw_tolerance_rad, Completion completion);
```

Update existing WFD calls to pass `{candidate.id, candidate.pose}` and add a PlannerClient test showing an approach target produces the exact POINT/yaw goal. Do not add UNKNOWN intent coordinates to this type.

- [ ] **Step 2: Add failing node tests for outside-start progression**

Add fixtures that start at `x=-2` with a task boundary beginning at `x=0`. Assert the first action target is map FREE, lies on the safe prefix rather than the UNKNOWN intent, has a boundary-approach identity, and uses the same `/Car/T4/plan_motion` fake server. After publishing a revealed map and arrival odometry, assert the next target has smaller remaining guidance; after the complete footprint is inside, assert the next batch is the existing WFD path. Run PAUSE/RESUME, CANCEL, and replacement START while each phase has an active request and prove the existing exact cancel/epoch rules remain identical.

- [ ] **Step 3: Add a separate owning FrozenApproachCycle**

Keep `FrozenPlanningCycle` unchanged for WFD regression. Add a private cycle holding frozen map content, the complete `BoundaryGuidanceResult`, coarse cursor, request-to-index map, reachable `{PlannedCandidate, MotionReference}` rows, and a final-order call. It must expose only owning spans and exact request association, mirroring the existing batch lifetime checks without manufacturing a `FrontierCluster`.

- [ ] **Step 4: Branch QueueBuild by the fully-inside predicate**

Always build `TaskRaster` and coverage first. Call `BoundaryGuidance::Build(map,raster,pose)`; when its phase is `kExploreTask`, execute the existing WFD block byte-for-byte. When it is `kApproachTask`, freeze its candidates and wait reason. Never call `FrontierDetector::Detect` to decide approach completion.

- [ ] **Step 5: Add approach pumping, final rank, and commitment**

Use the same stationary admission, request sequence, PlannerClient retry bound, planner timing provenance, and resource-result handling. On reachable rows, call `BoundaryGuidance::FinalOrder`; commit with `MakeBoundaryApproachActiveGoal`. Publish only the selected candidate pose/reference. Do not write approach no-path results to WFD `FailureMemory`.

- [ ] **Step 6: Add the completion guard and nonterminal waits**

Replace both existing direct completion sites with one helper that requires `navigation_phase==kExploreTask`, `fully_inside_task`, latest-map equality, exhausted WFD order, and zero reachable plans. Map approach waits exactly: `kWaitingForSafeStart -> WAITING_FOR_SAFE_APPROACH_START`, `kWaitingForTaskMapCoverage -> WAITING_FOR_TASK_MAP_COVERAGE`, `kNoGuidanceRoute -> APPROACH_NO_GUIDANCE_ROUTE`, and planner exhaustion -> `APPROACH_NO_REACHABLE_TARGET`. Replan exhaustion with unchanged map/pose/task sets `APPROACH_STALLED`. Store the map data/geometry, pose generation, and task generation signature so unchanged snapshots do not resubmit action requests; only new map, new pose, RESUME, or new START clears it. Map checked guidance resource failures to stable `RESOURCE_GUIDANCE_*` errors and never to an empty-wait result.

- [ ] **Step 7: Run node and PlannerClient tests, then commit**

Run:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install --packages-up-to lunar_pure_exploration_ros
colcon test --packages-select lunar_pure_exploration_ros --ctest-args -R 'test_planner_client|test_exploration_node' --output-on-failure
colcon test-result --verbose
```

Expected: existing initially-inside traces remain unchanged and new approach traces pass.

Commit:

```bash
git add ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/planner_client.hpp \
  ros2_ws/src/lunar_pure_exploration_ros/src/planner_client.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/exploration_node.hpp \
  ros2_ws/src/lunar_pure_exploration_ros/src/exploration_node.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_planner_client.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_node.cpp
git commit -m "feat: coordinate safe task-boundary approach"
```

---

### Task 6: Validate Active Approach Goals and Publish Distinct Observability

**Files:**
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/marker_builder.hpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/src/marker_builder.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/src/exploration_node.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_outputs.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_node.cpp`

**Interfaces:**
- Consumes: current `GoalKind`, latest map/task/pose, `BoundaryGuidanceResult`, and current execution plan ID.
- Produces: exact approach invalidation/cancel behavior; phase diagnostics; separate intent/route/safe-target markers; truthful `current_goal`.

- [ ] **Step 1: Write failing invalidation and observability tests**

Tests must show: a newly OCCUPIED actual approach target issues exactly one execution cancel containing the current plan ID and rebuilds; a better candidate alone does not preempt; an intent becoming OCCUPIED invalidates; an unchanged map does not duplicate cancel/build/action calls. Assert diagnostics contain exactly these keys and values:

```text
navigation_phase=APPROACH_TASK|EXPLORE_TASK
fully_inside_task=true|false
approach_intent_count=<uint>
approach_candidate_count=<uint>
remaining_guidance_m=<finite or unavailable>
guidance_unknown_cell_count=<uint or unavailable>
approach_wait_reason=<stable code or unavailable>
```

Assert marker namespaces are `approach_intents`, `approach_guidance`, and `approach_candidates`; `current_goal` equals only the real safe candidate pose.

- [ ] **Step 2: Extend MarkerBuilder with typed approach input**

Add an optional `ApproachMarkers` value containing intent points, selected guidance polyline, candidate poses, and selected identity. Emit DELETE markers for stale counts in all three new namespaces. Do not pass approach candidates through the existing `frontiers` or `candidates` namespaces.

- [ ] **Step 3: Branch active-map validation by GoalKind**

For `kTaskFrontier`, retain canonical frontier regeneration and positive information gain. For `kBoundaryApproach`, require: actual pose remains `SafePoseValidator::IsMapFree`, intent remains a FREE/UNKNOWN boundary intent, and the candidate still either becomes fully inside or sees a closer route UNKNOWN. Visibility must trace the complete global grid and stop at OCCUPIED cells; UNKNOWN may be counted but not traversability-promoted. Use the existing serialized validation queue and exact current plan ID cancel path.

- [ ] **Step 4: Publish stable reasons and diagnostics without message changes**

Map internal phase activity to `SELECTING_APPROACH_TARGET`, `PLANNING_APPROACH`, and `EXECUTING_APPROACH` reason strings while preserving lifecycle enum values. Populate diagnostics and markers from frozen owning values under the existing mutex-copy-publish pattern.

- [ ] **Step 5: Run all ROS package tests and commit**

Run:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install --packages-up-to lunar_pure_exploration_ros
colcon test --packages-select lunar_pure_exploration_ros --event-handlers console_direct+ --return-code-on-test-failure
colcon test-result --verbose
```

Expected: all ROS exploration tests pass.

Commit:

```bash
git add ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/marker_builder.hpp \
  ros2_ws/src/lunar_pure_exploration_ros/src/marker_builder.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/src/exploration_node.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_outputs.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_node.cpp
git commit -m "feat: expose approach safety and phase diagnostics"
```

---

### Task 7: Add Resource Configuration and the 20 m Outside-to-Inside Scenario

**Files:**
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/src/exploration_node.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_node.cpp`
- Modify: `launch/jazzy_300m_exploration_sim.launch.py`
- Modify: `tests/launch/test_jazzy_300m_exploration_sim.py`
- Create: `tests/scenarios/fixtures/outside_to_inside.yaml`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_synthetic_scenarios.cpp`
- Modify: `tests/scenarios/test_synthetic_exploration.py`

**Interfaces:**
- Consumes: derived/overridden resource parameters, scripted map/pose events, fake PlanMotion responses, status/diagnostic output.
- Produces: explicit 300 m launch limits and deterministic proof of approach -> WFD -> completion with task-only coverage above 0.80.

- [ ] **Step 1: Add failing parameter and fixture-registration tests**

Require the parameter loader to derive defaults exactly as approved:

```cpp
EXPECT_EQ(parameters.boundary_guidance_limits.maximum_guidance_grid_cells,
          parameters.task_raster_limits.maximum_raster_cell_count);
EXPECT_EQ(parameters.boundary_guidance_limits.maximum_guidance_work_units,
          8U * parameters.task_raster_limits.maximum_raster_cell_count);
EXPECT_EQ(parameters.boundary_guidance_limits.maximum_approach_candidates,
          parameters.candidate_limits.maximum_candidate_views);
```

Use checked multiplication and accept positive explicit overrides. The 300 m launch must pass `1048576`, `8388608`, and `4096`, matching its existing task-raster and candidate limits. Add `outside_to_inside` to the committed scenario list and require deterministic double-load equality.

- [ ] **Step 2: Create the deterministic 20 m fixture**

Use a 0.5 m, 48 x 40 map whose task polygon is `[0,0]-[20,20]`, initial robot pose is `[-2,10,0]`, the nearest boundary cells around `[0,10]` are OCCUPIED, and a FREE southern approach corridor leads to a different entrance. Events reveal successive UNKNOWN bands only after the prior safe target is reached. Freeze ordered request IDs and response kinds for at least two approach segments, at least one WFD segment, and final exhaustive WFD evidence.

The expected block must include:

```yaml
navigation_phases: [APPROACH_TASK, EXPLORE_TASK]
completed: true
terminal_reason: COMPLETED_NO_REACHABLE_FRONTIER
minimum_coverage_ratio: 0.80
approach_goal_count_minimum: 2
wfd_goal_count_minimum: 1
all_action_targets_known_free: true
nearest_blocked_entry_selected: false
```

- [ ] **Step 3: Extend the C++ scenario driver**

Teach only the test fixture loader/driver about map-reveal events and expected navigation diagnostics. Before each fake Action response, classify the requested goal cell in the contemporaneous map and fail unless it is FREE. Record the first status with `fully_inside_task=true` and assert every later candidate is a task-frontier goal.

- [ ] **Step 4: Prove approach failures never complete**

Add variants where every approach action returns `GLOBAL_NO_PATH`, where the task boundary is outside map coverage, and where no input changes after stall. Assert zero `COMPLETED` statuses and no repeated requests for the unchanged snapshot.

- [ ] **Step 5: Run launch and scenario tests, then commit**

Run:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install --packages-up-to lunar_pure_exploration_ros
colcon test --packages-select lunar_pure_exploration_ros --ctest-args -R test_synthetic_scenarios --output-on-failure
cd ..
python3 -m pytest -q tests/scenarios/test_synthetic_exploration.py tests/launch/test_jazzy_300m_exploration_sim.py
```

Expected: all selected tests pass; the fixture terminates only after WFD exhaustion and reports coverage greater than 0.80.

Commit:

```bash
git add ros2_ws/src/lunar_pure_exploration_ros/src/exploration_node.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_node.cpp \
  launch/jazzy_300m_exploration_sim.launch.py \
  tests/launch/test_jazzy_300m_exploration_sim.py \
  tests/scenarios/fixtures/outside_to_inside.yaml \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_synthetic_scenarios.cpp \
  tests/scenarios/test_synthetic_exploration.py
git commit -m "test: prove outside-to-inside autonomous exploration"
```

---

### Task 8: Document, Audit, and Verify the Integrated Feature

**Files:**
- Modify: `README.md`
- Create: `docs/validation/2026-08-25-boundary-directed-task-approach.md`
- Modify only if required by an actual changed contract: `tests/exploration/test_interface_contract.py`

**Interfaces:**
- Consumes: final implementation, commit list, test output, and local environment facts.
- Produces: operator-facing semantics, evidence matrix, and a clean reviewable diff against `3f5b0f7`.

- [ ] **Step 1: Update only relevant project documentation**

Document: boundary-derived automatic entry; UNKNOWN-intent/FREE-execution rule; `APPROACH_TASK` and `EXPLORE_TASK`; wait/reason codes; diagnostics; exact Grid V1 commit/mode; resource limits; completion guard; and the fact that coverage is not a core threshold. Keep existing unrelated README edits out of the feature commit by applying the change in the isolated worktree only.

- [ ] **Step 2: Run static and contract checks**

Run:

```bash
git diff --check 3f5b0f746a15d48edced0c045387154eb808e59c..HEAD
python3 -m pytest -q tests/exploration tests/launch/test_pure_exploration_launch.py tests/launch/test_jazzy_300m_exploration_sim.py tests/test_action_contract.py tests/test_external_interface_contract.py tests/test_operator_scripts.py
```

Expected: no whitespace errors; no message/topic/action drift; runtime mode assertions pass.

- [ ] **Step 3: Run the complete relevant Jazzy build and tests**

Run:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install --packages-up-to lunar_pure_planner_ros lunar_pure_exploration_ros lunar_pure_exploration_sim lunar_pure_wheeled_controller
colcon test --packages-select lunar_pure_planner_core lunar_pure_planner_ros lunar_pure_exploration_core lunar_pure_exploration_ros lunar_pure_exploration_sim lunar_pure_wheeled_controller --event-handlers console_direct+ --return-code-on-test-failure
colcon test-result --verbose
```

Expected: build succeeds and zero selected tests fail.

- [ ] **Step 4: Audit safety and latest-planner authority**

Run targeted searches and record their output in the validation document:

```bash
git diff --name-status 3f5b0f746a15d48edced0c045387154eb808e59c..HEAD
git diff --exit-code 3f5b0f746a15d48edced0c045387154eb808e59c..HEAD -- ros2_ws/src/lunar_pure_planner_core ros2_ws/src/lunar_pure_planner_ros/src ros2_ws/src/lunar_pure_planner_ros/include
rg -n "grid_traversability_v1|grid_v1_active|CompleteNoReachableFrontier|APPROACH_" launch scripts ros2_ws/src/lunar_pure_exploration_ros docs
```

Expected: planner production code is byte-identical to `3f5b0f7`; only exploration/client/config/tests/docs changed; every completion call is guarded from approach phase.

- [ ] **Step 5: Record evidence boundaries and commit docs**

The validation matrix must mark local Jazzy static/build/core/ROS/scenario evidence with exact commands and counts. Mark Humble build, Jetson AGX Orin, real T3 maps/odometry/TF, DDS, controller motion, rosbag, and RViz closed loop as `NOT_RUN` unless executed during this task; do not infer them from local success.

Commit:

```bash
git add README.md docs/validation/2026-08-25-boundary-directed-task-approach.md \
  tests/exploration/test_interface_contract.py
git commit -m "docs: validate boundary-directed task approach"
```

- [ ] **Step 6: Request final code review and address only in-scope findings**

Review against the spec and this plan, prioritizing: UNKNOWN never executable, full-footprint phase gate, approach cannot complete, latest Grid V1 path used, stale async callbacks cannot commit, and initially-inside WFD regression. Apply only verified in-scope corrections, rerun the affected focused test plus Task 8 Steps 2-4, and record the final commit IDs.
