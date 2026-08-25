# Stop-Gated Local-Segment Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the explorer evaluate frontiers and plan only while stationary, execute one complete bounded local trajectory without stopping between primitives, then stop and rebuild frontiers for the next local segment.

**Architecture:** Keep `rolling_surface_enabled: false` and reuse the existing normal `PlanMotion` result: `path_preview` is the global route used for candidate cost, while `trajectory` is the approximately 8 m executable local segment. Add a small ROS-independent stationary gate, integrate it into exploration orchestration, and separate local-segment completion from persistent candidate failure.

**Tech Stack:** C++20, ROS 2 Humble/Jazzy, rclcpp, ROS 2 Actions, GoogleTest, colcon, Python launch/config, RViz2.

**Spec:** `docs/superpowers/specs/2026-08-24-stop-gated-local-segment-exploration-design.md`

## Global Constraints

- Do not change the global or local planning algorithms, motion primitives, footprint, obstacle inflation, unknown-cell handling, or footprint sweep validation.
- Keep all candidate planning calls; do not reduce `candidates_per_planning_batch`.
- Keep `PlanMotion.action`, `MotionReference.msg`, Task 3 input topics, and Task 4 output topic names unchanged.
- Keep planner-internal `rolling_surface_enabled: false` for the 300 m exploration simulation.
- Treat a frontier as exploration guidance, not a waypoint that must be reached.
- Execute all primitives in one local `trajectory` continuously; stop only at the local endpoint or an execution invalidation.
- Use odometry twist only for stationary admission; do not add timestamp, covariance, map-version, or freshness checks.
- End only at `COMPLETED_NO_REACHABLE_FRONTIER`; coverage is reported and never used as a completion threshold.
- Preserve global, local, certification, and total planner timings; record stop-wait timing separately with the local steady clock.
- Build authoritative ROS packages in Ubuntu 22.04 + ROS 2 Humble. Run the requested 300 m simulation in the existing native Jazzy environment and label the two evidence sets separately.

---

### Task 1: Add a Deterministic Stationary Planning Gate

**Files:**
- Create: `ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/stationary_planning_gate.hpp`
- Create: `ros2_ws/src/lunar_pure_exploration_ros/src/stationary_planning_gate.cpp`
- Create: `ros2_ws/src/lunar_pure_exploration_ros/test/test_stationary_planning_gate.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/CMakeLists.txt`

**Interfaces:**
- Consumes: planar linear and angular speed magnitudes plus `std::chrono::steady_clock::time_point`.
- Produces: `StationaryPlanningGate`, `StationaryGateParameters`, and `StationaryGateUpdate` for the exploration runtime.

- [ ] **Step 1: Write failing tests for consecutive stationary samples**

Add tests with this public contract:

```cpp
using Clock = std::chrono::steady_clock;

StationaryPlanningGate gate({
    .maximum_linear_speed_mps = 0.01,
    .maximum_angular_speed_radps = 0.02,
    .confirmation_samples = 3U,
    .diagnostic_period = std::chrono::seconds{5},
});
const auto start = Clock::time_point{};
gate.Begin(start, {.linear_speed_mps = 0.2, .angular_speed_radps = 0.1});

EXPECT_FALSE(gate.Observe(start + 100ms, {0.0, 0.0}).confirmed);
EXPECT_FALSE(gate.Observe(start + 200ms, {0.0, 0.0}).confirmed);
EXPECT_TRUE(gate.Observe(start + 300ms, {0.0, 0.0}).confirmed);
EXPECT_TRUE(gate.waiting() == false);
```

Add separate tests proving:

- a moving sample resets the consecutive count;
- exactly-equal threshold values are stationary;
- non-finite observations reset the count and never confirm;
- `diagnostic_due` first becomes true at 5 seconds and advances by one period;
- `Cancel()` clears waiting state and counters;
- invalid negative thresholds, zero samples, or nonpositive diagnostic periods throw `std::invalid_argument`.

- [ ] **Step 2: Run the new target and verify it fails before production code exists**

Run in the sourced Humble build environment:

```bash
colcon build --packages-select lunar_pure_exploration_ros --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select lunar_pure_exploration_ros --ctest-args -R test_stationary_planning_gate --output-on-failure
```

Expected: compile failure because `stationary_planning_gate.hpp` and its types do not exist.

- [ ] **Step 3: Implement the minimal ROS-independent gate**

Define the exact interface:

```cpp
struct SpeedObservation final {
  double linear_speed_mps{};
  double angular_speed_radps{};
};

struct StationaryGateParameters final {
  double maximum_linear_speed_mps{0.01};
  double maximum_angular_speed_radps{0.02};
  std::uint32_t confirmation_samples{3U};
  std::chrono::steady_clock::duration diagnostic_period{
      std::chrono::seconds{5}};
};

struct StationaryGateUpdate final {
  bool confirmed{false};
  bool diagnostic_due{false};
  std::uint32_t consecutive_samples{};
  SpeedObservation observation{};
  std::chrono::steady_clock::duration elapsed{};
};

class StationaryPlanningGate final {
 public:
  explicit StationaryPlanningGate(StationaryGateParameters parameters);
  void Begin(std::chrono::steady_clock::time_point now,
             SpeedObservation entry_speed);
  StationaryGateUpdate Observe(std::chrono::steady_clock::time_point now,
                               SpeedObservation observation);
  void Cancel() noexcept;
  [[nodiscard]] bool waiting() const noexcept;
  [[nodiscard]] SpeedObservation entry_speed() const noexcept;
};
```

`Observe()` must compare speed magnitudes with inclusive thresholds, reset the
counter on a moving or non-finite observation, and close the gate exactly once
when the required sample count is reached. It must not read ROS time.

- [ ] **Step 4: Register the source and test in CMake and run focused tests**

Add `src/stationary_planning_gate.cpp` to `${PROJECT_NAME}` and add
`ament_add_gtest(test_stationary_planning_gate ...)` linked to the package
library.

Run:

```bash
colcon build --packages-select lunar_pure_exploration_ros --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select lunar_pure_exploration_ros --ctest-args -R test_stationary_planning_gate --output-on-failure
colcon test-result --verbose
```

Expected: the new focused test passes with zero failures.

- [ ] **Step 5: Commit the helper**

```bash
git add ros2_ws/src/lunar_pure_exploration_ros/CMakeLists.txt \
  ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/stationary_planning_gate.hpp \
  ros2_ws/src/lunar_pure_exploration_ros/src/stationary_planning_gate.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_stationary_planning_gate.cpp
git commit -m "feat: add stationary exploration planning gate"
```

### Task 2: Represent Local-Segment Completion Without Candidate Failure

**Files:**
- Modify: `ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/exploration_state_machine.hpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_core/src/exploration_state_machine.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_core/test/test_exploration_state_machine.cpp`

**Interfaces:**
- Consumes: the existing `ReleaseGoal(GoalReleaseReason)` transition.
- Produces: `GoalReleaseReason::kLocalSegmentCompleted` with reason code `LOCAL_SEGMENT_COMPLETED` and no persistent failure side effect.

- [ ] **Step 1: Add a failing state-machine test**

Construct and commit an active goal, call:

```cpp
machine.ReleaseGoal(GoalReleaseReason::kLocalSegmentCompleted);
EXPECT_EQ(machine.state(), ExplorationState::kSelectingFrontier);
EXPECT_EQ(machine.reason_code(), "LOCAL_SEGMENT_COMPLETED");
EXPECT_FALSE(machine.active_goal().has_value());
```

Also retain the existing assertion that `BeginReplanning(kRollingSegment)` does
not increment the stuck-replan counter.

- [ ] **Step 2: Run the core state-machine target and verify the enum is missing**

```bash
colcon build --packages-select lunar_pure_exploration_core --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select lunar_pure_exploration_core --ctest-args -R test_exploration_state_machine --output-on-failure
```

Expected: compile failure naming `kLocalSegmentCompleted`.

- [ ] **Step 3: Add the release reason and literal mapping**

Add `kLocalSegmentCompleted` to `GoalReleaseReason` and map it in the private
release-code switch to exactly `LOCAL_SEGMENT_COMPLETED`. Do not increment
`completed_goal_count` and do not call `FailureMemory::RecordPersistentFailure`.

- [ ] **Step 4: Run the full exploration-core tests**

```bash
colcon build --packages-select lunar_pure_exploration_core --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select lunar_pure_exploration_core --event-handlers console_direct+
colcon test-result --verbose
```

Expected: all exploration-core tests pass.

- [ ] **Step 5: Commit the transition**

```bash
git add ros2_ws/src/lunar_pure_exploration_core/include/lunar_pure_exploration_core/exploration_state_machine.hpp \
  ros2_ws/src/lunar_pure_exploration_core/src/exploration_state_machine.cpp \
  ros2_ws/src/lunar_pure_exploration_core/test/test_exploration_state_machine.cpp
git commit -m "feat: distinguish local exploration segment completion"
```

### Task 3: Load Stop-Gate Parameters and Admit Planning Only While Stationary

**Files:**
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/exploration_node.hpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/src/exploration_node.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_node.cpp`
- Modify: `config/pure_exploration.yaml`
- Modify: `launch/jazzy_300m_exploration_sim.launch.py`

**Interfaces:**
- Consumes: `StationaryPlanningGate` from Task 1 and existing odometry, cancel, planner Action, and steady-clock seams.
- Produces: `ExplorationNodeParameters::stationary_gate`, a cached `SpeedObservation`, and a private `PendingStationaryAction` gate before `QueueBuild`, `Pump`, or `StartExecutionReplan` can submit an Action request.

- [ ] **Step 1: Add parameter-load and node-level failing tests**

Extend the test parameter fixture with:

```cpp
.stationary_gate = {
    .maximum_linear_speed_mps = 0.01,
    .maximum_angular_speed_radps = 0.02,
    .confirmation_samples = 3U,
    .diagnostic_period = 5s,
},
```

Add node tests that publish a valid map/task/TF and odometry with
`linear.x = 0.2`. Assert no planner goal arrives. Publish two zero-twist
odometry samples and assert no goal; publish the third and assert the first
candidate goal arrives. Add a second test in which a moving sample between the
second and third zero samples resets admission.

- [ ] **Step 2: Run the focused exploration-node tests and verify they fail**

```bash
colcon build --packages-select lunar_pure_exploration_ros --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select lunar_pure_exploration_ros --ctest-args -R test_exploration_node --output-on-failure
```

Expected: the planner goal arrives before the required stationary samples.

- [ ] **Step 3: Add parameters and validate them at node construction**

Add the `StationaryGateParameters stationary_gate` member with defaults to
`ExplorationNodeParameters`. Declare and load exactly these ROS parameters:

```text
stop_before_planning
stationary_linear_speed_mps
stationary_angular_speed_radps
stationary_confirmation_samples
stop_wait_diagnostic_s
```

When `stop_before_planning` is false, preserve the old immediate-admission path
for isolated regression diagnosis only. Production YAML and the 300 m launch
must set it to true. Reject negative thresholds, zero confirmation samples, and
nonpositive diagnostic periods at startup.

- [ ] **Step 4: Cache odometry speed without changing pose resolution**

Create a dedicated odometry handler that still calls
`PoseResolver::UpdateOdometry()` and computes magnitudes as:

```cpp
const auto& linear = message.twist.twist.linear;
const auto& angular = message.twist.twist.angular;
SpeedObservation speed{
    .linear_speed_mps = std::hypot(linear.x, linear.y, linear.z),
    .angular_speed_radps = std::hypot(angular.x, angular.y, angular.z),
};
```

TF handling remains pose-only. Do not inspect covariance or header stamps.

- [ ] **Step 5: Centralize planning admission**

Add a private runtime enum:

```cpp
enum class PendingStationaryAction : std::uint8_t {
  kNone,
  kBuildFreshBatch,
  kReplanActiveGoal,
};
```

Add helpers with these responsibilities:

```cpp
RequestStationaryActionLocked(runtime, action)
  -> optional active plan_id to cancel;

ConsumeConfirmedStationaryActionLocked(runtime)
  -> PendingStationaryAction;

PlanningIsAdmittedLocked(runtime)
  -> true only when no stop action is pending;

StatusReasonLocked(runtime)
  -> `WAITING_FOR_STOP` while the gate is waiting, otherwise the state-machine
     reason code;
```

Task start/resume and every fresh `QueueBuild` request must request
`kBuildFreshBatch`. While the gate is waiting, `QueueBuild`, `Pump`, candidate
retry, and `StartExecutionReplan` must not submit `PlanMotion`. The odometry
handler consumes the pending action only after the gate confirms, increments
the planning epoch, and dispatches the requested build or replan outside the
runtime mutex.

- [ ] **Step 6: Prove cancel identity is retained until stop confirmation**

Add an execution fixture containing an active `MotionReference` with
`plan_id = "wheel-active"`. Trigger a fresh planning request while publishing
moving odometry. Assert `/Car/T4/execution/cancel` receives exactly
`wheel-active`, `SnapshotActiveReferenceForTest()` remains populated during the
stop wait, and no Action request is sent. After the third stationary sample,
assert the reference is cleared and a new candidate batch starts.

- [ ] **Step 7: Run focused tests and commit**

```bash
colcon build --packages-select lunar_pure_exploration_ros --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select lunar_pure_exploration_ros --ctest-args -R 'test_stationary_planning_gate|test_exploration_node' --output-on-failure
colcon test-result --verbose
```

```bash
git add config/pure_exploration.yaml launch/jazzy_300m_exploration_sim.launch.py \
  ros2_ws/src/lunar_pure_exploration_ros/include/lunar_pure_exploration_ros/exploration_node.hpp \
  ros2_ws/src/lunar_pure_exploration_ros/src/exploration_node.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_node.cpp
git commit -m "feat: gate exploration planning on stationary odometry"
```

### Task 4: Rebuild Frontiers After Each Complete Local Trajectory

**Files:**
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/src/exploration_node.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_node.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_synthetic_scenarios.cpp`

**Interfaces:**
- Consumes: `PendingStationaryAction` from Task 3 and `GoalReleaseReason::kLocalSegmentCompleted` from Task 2.
- Produces: endpoint-stop-refresh behavior; stuck recovery remains same-candidate replanning but is admitted only after stationary confirmation.

- [ ] **Step 1: Add failing local-endpoint tests**

Extend the existing executable-endpoint fixture. Move odometry to the local
trajectory endpoint while the frontier target remains farther away. Assert:

```cpp
EXPECT_EQ(status.state, Status::REPLANNING);
EXPECT_EQ(status.reason_code, "WAITING_FOR_STOP");
EXPECT_EQ(cancel_message.data, active_plan_id);
EXPECT_EQ(planner_goal_count, previous_goal_count);
```

Then publish three stationary samples and assert:

```cpp
EXPECT_EQ(status.reason_code, "LOCAL_SEGMENT_COMPLETED");
EXPECT_FALSE(ExplorationNodeTestPeer::ActiveReference(node).has_value());
EXPECT_GT(new_candidate_batch_generation, old_candidate_batch_generation);
```

Do not require the old frontier to remain selected.

- [ ] **Step 2: Add failing stale-result and failure-memory tests**

Start a planner request, force a stop/rebuild epoch change, and inject the old
request's `NO_PATH`. Assert it is ignored, no persistent failure entry is added,
and the task cannot transition to `COMPLETED_NO_REACHABLE_FRONTIER` from that
result.

Add a contrasting test in which a current-epoch request starts after stationary
confirmation and returns exhaustive `NO_PATH`; assert existing failure-memory
handling remains active.

- [ ] **Step 3: Change endpoint handling from same-goal rolling replan to fresh batch**

In `PollExecution()`, when `ReachedSegmentEndpoint()` is true and the endpoint
is not the frontier target:

1. retain the active reference long enough to publish its cancel identity;
2. call `BeginReplanning(kRollingSegment)`;
3. request `PendingStationaryAction::kBuildFreshBatch`;
4. do not call `StartExecutionReplan()`;
5. after stop confirmation, call
   `ReleaseGoal(kLocalSegmentCompleted)`, clear execution payloads, increment
   epoch/build generation, and call `QueueBuild()`.

Do not append the local endpoint to `completed_goal_positions`; that vector is
reserved for actual final-goal arrivals.

- [ ] **Step 4: Gate stuck and map-invalidation recovery**

For stuck recovery, preserve the existing `maximum_replans_per_candidate` and
persistent failure behavior, but replace immediate `StartExecutionReplan()`
with `PendingStationaryAction::kReplanActiveGoal`.

For candidate disappearance or active-path invalidation caused by a map update,
cancel execution, wait for stationary confirmation, release the old goal with
the existing nonfailure reason, and build a fresh batch. A map update alone
must not start background candidate planning while execution remains valid.

- [ ] **Step 5: Update the synthetic scenario to prove a frontier is guidance**

Create a scenario with a frontier beyond the 8 m local endpoint. The scripted
planner returns a `path_preview` to the frontier and a shorter executable
trajectory. Verify:

- the controller-facing reference contains the whole local trajectory;
- no cancel occurs between its primitives;
- endpoint completion does not increment completed-goal count;
- after stop confirmation, a new map causes the old frontier to disappear;
- no failure-memory entry is created; and
- the next candidate batch uses the refreshed map.

- [ ] **Step 6: Run node and scenario tests and commit**

```bash
colcon build --packages-select lunar_pure_exploration_core lunar_pure_exploration_ros --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select lunar_pure_exploration_core lunar_pure_exploration_ros \
  --ctest-args -R 'test_exploration_state_machine|test_exploration_node|test_synthetic_scenarios' --output-on-failure
colcon test-result --verbose
```

```bash
git add ros2_ws/src/lunar_pure_exploration_ros/src/exploration_node.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_node.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_synthetic_scenarios.cpp
git commit -m "fix: refresh frontiers after local exploration segments"
```

### Task 5: Publish Stop-Wait Diagnostics Without Polluting Planner Timing

**Files:**
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/src/exploration_node.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_outputs.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_ros/test/test_planner_timing_accumulator.cpp`

**Interfaces:**
- Consumes: gate entry/final observations and elapsed steady-clock duration.
- Produces: exploration diagnostic key-values `stop_wait_elapsed_ms`, `stop_entry_linear_mps`, `stop_entry_angular_radps`, `stop_confirmed_linear_mps`, `stop_confirmed_angular_radps`, `stop_confirmation_samples`, and `planning_epoch`.

- [ ] **Step 1: Write failing diagnostics tests**

Drive a gate from entry speed `(0.2, 0.1)` to three stationary observations and
assert the exploration diagnostic contains all seven exact keys. Assert
`stop_wait_elapsed_ms` is derived from the injected steady clock.

Feed the same planner diagnostics before and after a stop cycle and assert the
existing global/local/total elapsed aggregates are byte-for-byte unchanged.

- [ ] **Step 2: Run output and timing tests and verify missing keys**

```bash
colcon test --packages-select lunar_pure_exploration_ros --ctest-args \
  -R 'test_exploration_outputs|test_planner_timing_accumulator' --output-on-failure
```

Expected: diagnostic key assertions fail; existing planner timing assertions
remain green.

- [ ] **Step 3: Add bounded stop-wait diagnostic storage**

Store only the current and most recently completed stop-wait record in runtime.
On `diagnostic_due`, publish the current elapsed duration without failing the
task. On confirmation, store the final speeds/sample count and advance
`planning_epoch`. Do not add stop time to `PlannerTimingSummary` or
`PlannerTimingAccumulator`.

- [ ] **Step 4: Run focused tests and commit**

```bash
colcon build --packages-select lunar_pure_exploration_ros --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select lunar_pure_exploration_ros --ctest-args \
  -R 'test_exploration_outputs|test_planner_timing_accumulator|test_exploration_node' --output-on-failure
colcon test-result --verbose
```

```bash
git add ros2_ws/src/lunar_pure_exploration_ros/src/exploration_node.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_exploration_outputs.cpp \
  ros2_ws/src/lunar_pure_exploration_ros/test/test_planner_timing_accumulator.cpp
git commit -m "feat: report exploration stop-wait timing"
```

### Task 6: Full Humble Regression and Planner-Contract Verification

**Files:**
- Modify only if evidence requires: `docs/validation/2026-08-24-local-wheel-planning-efficiency.md`

**Interfaces:**
- Consumes: completed Tasks 1-5.
- Produces: authoritative Humble build/test evidence and proof that planner algorithms and interfaces did not change.

- [ ] **Step 1: Verify the source diff stays inside the approved boundary**

```bash
git diff a6c63c6 --stat
git diff a6c63c6 -- ros2_ws/src/lunar_pure_planner_core \
  ros2_ws/src/lunar_pure_planner_ros \
  ros2_ws/src/lunar_planning_msgs
```

Expected: no production diff in planner core, planner ROS, or planning messages.

- [ ] **Step 2: Run repository-boundary checks**

```bash
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
```

Expected: both commands pass.

- [ ] **Step 3: Run the full affected Humble package suites**

Inside `osrf/ros:humble-desktop-full-jammy`, with the workspace mounted and
`/opt/ros/humble/setup.bash` sourced:

```bash
colcon build --packages-up-to lunar_pure_exploration_ros lunar_pure_wheeled_controller lunar_pure_planner_ros \
  --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select lunar_pure_exploration_core lunar_pure_exploration_ros \
  lunar_pure_wheeled_controller lunar_pure_planner_core lunar_pure_planner_ros \
  --event-handlers console_direct+
colcon test-result --verbose
```

Expected: zero failed tests. Record package/test counts and elapsed time.

- [ ] **Step 4: Verify the ROS contracts and default configuration**

```bash
rg -n "rolling_surface_enabled" config/pure_planner.yaml launch/jazzy_300m_exploration_sim.launch.py
rg -n "/Car/T3/|/Car/T4/|/Car/T5/" config/pure_exploration.yaml launch/jazzy_300m_exploration_sim.launch.py
git diff a6c63c6 -- ros2_ws/src/lunar_planning_msgs
```

Expected: rolling remains false in the 300 m launch, topics retain their
approved namespaces, and planning messages are unchanged.

- [ ] **Step 5: Update validation evidence and commit**

Append the exact commands, pass/fail counts, commit IDs, and the boundary result
to `docs/validation/2026-08-24-local-wheel-planning-efficiency.md`. State that
this is Humble container evidence, not native Jazzy simulation or Orin DDS
evidence.

```bash
git add docs/validation/2026-08-24-local-wheel-planning-efficiency.md
git commit -m "docs: record stop-gated exploration regression"
```

### Task 7: Reproduce the Moving-Start Smoke and Complete the 300 m Jazzy Run

**Files:**
- Modify: `docs/validation/2026-08-24-local-wheel-planning-efficiency.md`
- Runtime artifacts: write only under `~/CodexDownloads/lunar_navigation/jazzy_300m_exploration_runs/<run_id>`; do not add them to Git.

**Interfaces:**
- Consumes: native Jazzy overlay, 300 m launch, RViz configuration, and all completed implementation commits.
- Produces: runtime evidence for stationary candidate batches, uninterrupted local execution, repeated map/frontier refresh, planner timing, coverage, and terminal reason.

- [ ] **Step 1: Build a clean native Jazzy overlay**

Use an external artifact-specific build directory so no existing Humble or
Jazzy build tree is reused. Source Jazzy, then run:

```bash
source /opt/ros/jazzy/setup.bash
mkdir -p /home/kai/CodexDownloads/lunar_navigation/jazzy_300m_exploration_runs/stop_gated_acceptance/build
mkdir -p /home/kai/CodexDownloads/lunar_navigation/jazzy_300m_exploration_runs/stop_gated_acceptance/install
mkdir -p /home/kai/CodexDownloads/lunar_navigation/jazzy_300m_exploration_runs/stop_gated_acceptance/log
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/jazzy_300m_exploration_runs/stop_gated_acceptance/log \
  build \
  --build-base /home/kai/CodexDownloads/lunar_navigation/jazzy_300m_exploration_runs/stop_gated_acceptance/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/jazzy_300m_exploration_runs/stop_gated_acceptance/install \
  --packages-up-to lunar_pure_exploration_ros lunar_pure_wheeled_controller lunar_pure_planner_ros \
  --cmake-args -DBUILD_TESTING=OFF
source /home/kai/CodexDownloads/lunar_navigation/jazzy_300m_exploration_runs/stop_gated_acceptance/install/setup.bash
```

Expected: all three production packages build and the sourced overlay reports
`ROS_DISTRO=jazzy`.

- [ ] **Step 2: Re-run the bounded moving-start smoke**

Launch the 300 m simulation with RViz disabled for the bounded smoke and record:

```text
/Car/T3/localization/odometry
/Car/T4/exploration/status
/Car/T4/planning/diagnostics
/Car/T4/execution/motion_reference
/Car/T4/execution/cancel
/Car/T5/Car_Cmd_Vel
```

Expected acceptance:

- every candidate batch begins after three stationary odometry samples;
- the previous candidates 6 and 7 are not rejected because of a moving start;
- `planning_outcome == 0` and `has_reference == true` for at least one selected
  candidate;
- global and local planning times are present; and
- stop-wait time is reported separately.

- [ ] **Step 3: Verify continuous execution of one local trajectory in RViz2**

Launch `jazzy_300m_exploration_sim.launch.py` with RViz enabled. Capture one
selected `MotionReference` containing multiple trajectory points/primitives.
Verify `/Car/T5/Car_Cmd_Vel` does not return to zero at internal primitive
boundaries and does return to zero at the local trajectory endpoint.

- [ ] **Step 4: Run the full exploration without an artificial coverage stop**

Run until the explorer publishes:

```text
state: COMPLETED
reason_code: COMPLETED_NO_REACHABLE_FRONTIER
```

Record:

- final coverage ratio and known/unknown areas;
- number of local segments completed;
- candidate count and reachable/failed counts per cycle;
- global/local/total planning time distributions;
- stop-wait time distribution;
- total simulated and wall-clock duration; and
- any remaining failure reason frequencies.

Do not report completion if zero local segments execute or if completion is
caused by stale/moving-start `NO_PATH` results.

- [ ] **Step 5: Record evidence and commit only the validation document**

Add a concise results table and artifact paths to the validation document.
Label screenshots, bags and CSV/JSON artifacts as external files. Then run:

```bash
git add docs/validation/2026-08-24-local-wheel-planning-efficiency.md
git commit -m "docs: record 300m stop-gated exploration acceptance"
git status --short --branch
```

Expected: validation evidence is committed, runtime artifacts remain outside
the repository, and the worktree contains no unrelated changes.

### Task 8: Independent Final Review

**Files:**
- Review only: all files changed after commit `a6c63c6`.

**Interfaces:**
- Consumes: implementation commits and Humble/Jazzy evidence.
- Produces: an independent findings-first review with explicit readiness boundaries.

- [ ] **Step 1: Review semantic requirements**

The reviewer must verify:

- no planner call can start while the stationary gate is closed;
- one local trajectory executes continuously;
- local endpoint completion is not goal arrival or candidate failure;
- stale results cannot exhaust frontiers;
- no-reachable-frontier completion requires a current stationary batch;
- planner timing and stop-wait timing remain separate; and
- planner algorithms, messages, inflation and sweep behavior are unchanged.

- [ ] **Step 2: Review concurrency and lifecycle safety**

Inspect mutex boundaries, callbacks, epoch/generation comparisons, pending
task controls, planner cancellation, execution cancellation, teardown, and
exception paths. Confirm no Action call or ROS publication occurs while holding
the runtime mutex unless already established safe practice requires it.

- [ ] **Step 3: Re-run the smallest decisive tests**

```bash
colcon test --packages-select lunar_pure_exploration_core lunar_pure_exploration_ros \
  --ctest-args -R 'test_stationary_planning_gate|test_exploration_state_machine|test_exploration_node|test_synthetic_scenarios' --output-on-failure
colcon test-result --verbose
```

- [ ] **Step 4: Report readiness**

Classify separately:

- source/build readiness;
- Humble automated-test readiness;
- native Jazzy 300 m simulation readiness;
- RViz/result-file acceptance; and
- Orin/Task 3 DDS readiness.

Do not infer Orin readiness from container or native simulation evidence.
