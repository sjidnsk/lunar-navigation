# Closed-loop Task 4 implementation report — bounded live smoke

## Scope and isolation

- Start HEAD: `685d484`.
- Added an opt-in live pytest to
  `tests/launch/test_jazzy_300m_exploration_sim.py`; normal CI continues to run
  the nine static tests and skips the live case unless
  `LUNAR_RUN_LIVE_JAZZY_SMOKE=1` is set.
- Every live attempt chose and locked a candidate domain in the range 20–199,
  set `ROS_LOCALHOST_ONLY=1` and `ROS2CLI_NO_DAEMON=1`, and rejected a nonempty
  bounded `ros2 node list --no-daemon` preflight.
- The installed launch is started without a shell, with
  `Popen(start_new_session=True)`, `start_rviz:=false`, and unique absolute log
  and output paths below
  `/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_runs/live-smoke`.
- Teardown targets only the launch process group: SIGINT, at most 10 seconds of
  waiting, then SIGTERM to that same group when needed. No recursive removal,
  broad process match, broad kill, or SIGKILL is used. The probe destroys its
  Action client and node before the final bounded empty-graph check.

This domain selection minimizes local collision but cannot prove that no other
DDS participant exists globally; this remains the plan's explicit isolation
ruling.

## Integration rulings found by real RED attempts

The first live test was written before changing runtime behavior. Seven bounded
iterations were retained outside Git and informed only minimum simulation and
ROS publication-boundary corrections; the frontier selector and global/local
planning algorithms were not changed.

1. `run-701ebadf...`: task publication raced initial readiness and ended as
   `INVALID_INPUT`; no planner call occurred. The coordinator now requires five
   consecutive fully-ready 100 ms polls and resets the count after any loss.
2. `run-341c90d...`: real global/local calls occurred, but all returned
   `NO_PATH` and exploration reached no reachable frontier without motion.
3. `run-8c6ac...`: a local-map diagnostic experiment reproduced `NO_PATH`; the
   experiment was reverted completely.
4. `run-577915...`: candidate evidence plus a separate diagnostic run
   (`run-4857996...`) localized the rejection to `WHEEL_START_INFEASIBLE`.
   Temporary planner instrumentation was reverted; no planner source remains in
   the Task 4 diff.
5. `run-f366a733...`: a current contact observation made the start locally
   finite, but a full-disk startup prior exposed rear candidates outside the
   strict current FOV, causing 119 bounded retries whose individual planner
   calls were about 952 ms and ended in timeout.
6. `run-1feb71...`: the startup prior was limited to a forward 90-degree wedge;
   real motion, references, planner calls, and coverage growth appeared, but the
   transient `PLANNING` state was not observable, so the test correctly timed
   out rather than declaring success.
7. `run-de58c243...`: after publishing one genuine `PLANNING` snapshot before
   commit, all live conditions passed and the selected domain was empty after
   teardown.

The frozen simulation observation contract is now explicit:

- A persistent free global cell requires its conservative visibility envelope
  to lie inside the strict current 10 m/90-degree FOV and LOS. The derived
  margin is
  `hypot(0.8175/2, 0.67/2) + 0.2 + sqrt(2)/2 = 1.435... m`.
- A directly sensed occupied return may be marked known at the hit itself; it is
  not treated as traversable and does not expose truth behind the hit.
- The current local contact envelope is derived from the 0.591 m by 0.409 m
  footprint half-extents, 0.2 m clearance, and the 0.2 m cell half diagonal.
  It is recomputed every frame, is not persisted, and does not alter global
  knowledge away from the separately defined startup prior. Tests prove the
  start footprint is finite, a point 5 m behind remains NaN, and the old patch
  disappears after moving.
- The guaranteed-free 4 m startup prior is applied once at the origin. Apart
  from the vehicle's required start-contact anchor, complete cells must lie in
  the forward 90-degree wedge; rear cells such as the tested 2.5 m point remain
  unknown. It does not read additional scene truth or change later persistent
  observation.
- `BeginPlanning()` now produces a true `PLANNING` status snapshot before
  `CommitGoal()` and reference publication. Publication occurs outside the
  runtime mutex, after which cycle/epoch/state are revalidated, so cancellation
  and failure transitions retain their atomic boundary.

## Fresh live evidence

Final verification run:

`/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_runs/live-smoke/run-ea44a6ccafaa485dbb6d3e792ee4a38a`

- Pytest: `1 passed, 9 deselected in 3.52s`.
- Domain: `133`; exact task ID `jazzy-300m-20260824`, delivered once.
- Streams: global 15, local 22, odometry 23, TF 22; TF edges are
  `map->odom` and `odom->base_link`; Action server ready.
- State values observed: `[0, 1, 2, 3, 4]`, including genuine
  `SELECTING_FRONTIER`, `PLANNING`, and `EXECUTING` snapshots.
- Selected goal: `(4.649634894, 0.850365106, yaw 1.299744822)`; reference:
  385 points; nonzero T5 command observed.
- Displacement: `0.200003788 m`; coverage increased from `0.0` to
  `0.0004756242568370987`.
- Odometry cadence: 23 samples, mean arrival gap `0.0499038 s`, below the
  0.1 s systematic-miss guard.
- First plan: global 72.967 ms / one call, local 336.445 ms / one call,
  total 413.091 ms, `PLAN_FOUND`.
- After cleanup, bounded `ros2 node list --no-daemon` in domain 133 returned no
  nodes and every exact child PID reported by launch was absent.

The smoke stops after demonstrating bounded progress. It does not claim full
exploration completion; timeout, ERROR, or a terminal state before motion is a
test failure.

## Build and regression evidence

- Fresh external Jazzy build of `lunar_pure_exploration_ros` and
  `lunar_pure_exploration_sim`: 2 packages finished, exit 0.
- `lunar_pure_exploration_sim`: 8/8 CTest entries passed.
- Static launch tests: 9 passed; live test deselected.
- Python scenario tests: 21 passed.
- Synthetic ROS scenarios: 11/11 passed.
- Four impacted node concurrency/publication tests passed under a 60-second
  SIGHUP-bounded wrapper.
- `git diff --check`: passed before report creation and is repeated before
  commit.

Build, install, log, probe, and ROS log artifacts remain outside the repository
under `/home/kai/CodexDownloads/lunar_navigation/`.

## Known limitations and unrelated observations

- Jazzy warns that `ROS_LOCALHOST_ONLY` is deprecated, while also confirming it
  is honored. The frozen Task 4 contract requires it, so it remains enabled.
- On SIGINT, the existing Python controller attempts a final publish after its
  rclpy context is invalid and prints an `RCLError`. The bounded launch group
  still exits and the selected DDS graph is empty. Task 4 does not change the
  controller shutdown implementation.
- A prior complete node-test invocation exposed existing timing-sensitive tests:
  `FailureMemoryLimitStillCancelsThirdStuckPlanBeforeStableError`,
  `ExecutionReplanTimeoutPreservesCommittedGoal`, and once
  `AcceptedActionCancelReachesPausedOnlyAfterCanceledTerminal`. Targeted runs of
  the first two can pass independently, but the full unordered/timing behavior
  is not claimed fixed by this task. One repeat process ignored SIGINT/SIGTERM;
  it was terminated by SIGHUP to its exact PID and confirmed absent.
- A clean all-package `BUILD_TESTING=ON` build is blocked in unchanged planner
  test code by a missing include for `std::ranges::any_of`; the runtime closure
  builds cleanly with tests disabled, and the two modified packages build and
  pass their focused tests with testing enabled. No planner behavior or source
  is modified here.
- This is local Ubuntu/Jazzy functional evidence only, not Humble or Jetson AGX
  Orin certification and not the long Task 5 terminal exploration run.

## Independent-review teardown fix

The independent review found that the original helper escalated to SIGTERM only
while the launch leader itself was alive. A leader could therefore exit after
SIGINT while a same-process-group non-ROS child survived, and an empty ROS graph
would not detect that child.

The fixed helper captures the exact launch identity as
`(pid, /proc start_time, pgid, session_id)` immediately after
`Popen(start_new_session=True)`. It continuously accumulates only identities in
that exact session and process group. After SIGINT it waits for all recorded
identities, not merely the leader; any residual member triggers SIGTERM to that
same PGID. PID liveness also requires the captured `/proc` start time, preventing
PID reuse from being mistaken for the original child. The bounded ROS graph
check remains an independent final condition.

A synthetic regression forks a leader and child in a new session. The leader
exits on SIGINT while the child ignores SIGINT. The test proves that teardown
detects the residual child, sends same-PGID SIGTERM, and leaves neither identity
alive. Static plus synthetic launch tests pass `10/10`.

A fresh external runtime overlay was rebuilt from the current worktree under
`/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/task4_teardown_fix_20260824`;
all eight runtime-closure packages built successfully. The installed launch and
source launch have identical SHA-256
`6681f84273b195ea48a7dad1ee80672f1351e017a876df273718f0a662e00724`,
and `ros2 pkg prefix lunar_pure_exploration_sim` resolves to that fixed overlay.

Fresh fixed-overlay live run:
`/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_runs/live-smoke/run-ecb5597b562e48b3a3a16ec78f919d54`.

- Pytest: `1 passed, 10 deselected in 3.69s`.
- Domain 91; exact task published once; states `[0,1,2,3,4]`.
- Reference 385 points; nonzero T5 observed; displacement `0.400019496 m`.
- Coverage changed from `0.0` to `0.0004756242568370987`; this is initial-map
  accounting evidence, not a claim that motion caused the increase.
- First successful plan: global `69.85396 ms`, local `335.70915 ms`, total
  `408.868357 ms`; odometry mean gap `0.050064979 s`.
- Teardown recorded the leader plus all seven launch child identities. All eight
  exact `(pid,start_time)` identities are absent, the process group drained on
  SIGINT without requiring SIGTERM in this normal run, and the bounded domain-91
  `ros2 node list --no-daemon` result is empty.
