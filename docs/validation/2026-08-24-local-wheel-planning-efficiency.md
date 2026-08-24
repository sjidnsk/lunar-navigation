# Local Wheel Planning Efficiency Validation — 2026-08-24

## Verdict

`INCOMPLETE` from the runtime acceptance captured at source
`e47c8f54e3addc939297e57ff539f9edb55d4b5c`. The later metrics-retention
fix-wave source is `639ce5134e1eda140a15d8df87ea922f198f18cc`, but no smoke or
field acceptance was rerun from that source.

At the original Task 8 `e47c8f5`/documentation-commit `f60a964` stage, the
planner-core performance change had fresh Jazzy unit evidence and the
production eight-package overlay built successfully. The fixed-seed
closed-loop smoke is not accepted: candidates 6 and 7 are `NO_PATH` from a
moving start, no goal completes, and the operator rejects the result. The full
300 m RViz run was therefore not started.

Readiness boundaries:

- full 300 m + RViz: `NOT_RUN / INCOMPLETE`;
- Jetson AGX Orin / ROS 2 Humble: `NOT_RUN`;
- Task 7 far-clearance result: `FAR_CLEARANCE_SKIP_NOT_ENABLED`;
- exploration candidate/yaw/order/retry/call-count behavior: unchanged;
- controller, Action/messages and admission contracts: unchanged.

## Source and scope

```bash
git status --short --branch
git log --oneline 68b24c7..HEAD
git diff 68b24c7..HEAD --stat
git diff 68b24c7..HEAD -- \
  ros2_ws/src/lunar_pure_exploration_core \
  ros2_ws/src/lunar_pure_exploration_ros
```

The original Task 8 integration worktree was clean at `e47c8f5`. There is no
diff in either exploration package over the inspected range.

All generated build, log and run data is outside the repository. The principal
evidence root is:

```text
/home/kai/CodexDownloads/lunar_navigation/local-wheel-efficiency-evidence
```

## Fresh Jazzy build and tests

### Test-enabled closure

The brief's clean test build used:

```bash
export LUNAR_WHEEL_FINAL_ROOT=/home/kai/CodexDownloads/lunar_navigation/\
local-wheel-efficiency-evidence/final
source /opt/ros/jazzy/setup.bash
colcon --log-base "$LUNAR_WHEEL_FINAL_ROOT/test-log" build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_WHEEL_FINAL_ROOT/test-build" \
  --install-base "$LUNAR_WHEEL_FINAL_ROOT/test-install" \
  --packages-up-to lunar_pure_planner_ros lunar_pure_exploration_sim \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
```

Core and planner ROS built. The full closure stopped in the
`lunar_pure_exploration_ros` test-only compatibility gate because the expected
`pure_plan_motion_server.cpp` digest is
`e9c2e67e1522d9d509a7222240a253b23a01547df3583efe1347a311aeb4885e`,
while the integrated source digest is
`a39cdc3be7d652c9593072b676d726432a2d44d610133cff7c099099fe958caa`.
The other two gated files still match. Updating this digest requires an
explicit Task 10 compatibility re-review; it was not changed here.

Original Task 8 fresh results at planner/runtime source `e47c8f5`, first
recorded by documentation commit `f60a964`:

| Verification | Result |
| --- | --- |
| Full wheel GoogleTest target | 97/97 passed, including 750 m |
| Focused request-scoped SE(2) | 3/3 passed |
| Launch pytest | 9/9 passed |
| Core CTest | 19/19 targets passed |
| Planner ROS CTest | 11/12 targets passed |

The result table above comes from the following actually executed test and
result commands:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base "$LUNAR_WHEEL_FINAL_ROOT/ctest-log" test \
  --build-base "$LUNAR_WHEEL_FINAL_ROOT/test-build" \
  --install-base "$LUNAR_WHEEL_FINAL_ROOT/test-install" \
  --packages-select lunar_pure_planner_core lunar_pure_planner_ros \
  --return-code-on-test-failure \
  --event-handlers console_direct+
colcon test-result \
  --test-result-base "$LUNAR_WHEEL_FINAL_ROOT/test-build" --verbose
```

Colcon delegated to these two CTest processes. The working directories and
commands below are copied from the package `command.log` files:

```bash
(
  cd /home/kai/CodexDownloads/lunar_navigation/\
local-wheel-efficiency-evidence/final/test-build/lunar_pure_planner_core
  /usr/bin/ctest -C RelWithDebInfo -D ExperimentalTest \
    --no-compress-output -V --force-new-ctest-process
)
(
  cd /home/kai/CodexDownloads/lunar_navigation/\
local-wheel-efficiency-evidence/final/test-build/lunar_pure_planner_ros
  /usr/bin/ctest -C RelWithDebInfo -D ExperimentalTest \
    --no-compress-output -V --force-new-ctest-process
)
```

The exact test working directory is `final/test-build`, not `final/build`.
The latter is an external operator-preflight symlink to `prod-build` and is
not test evidence. The core invocation returned 0 and the ROS invocation
returned 8 for its one failing target. Their command and output evidence is:

- `final/ctest-log/test_2026-08-24_17-01-15/lunar_pure_planner_core/command.log`;
- `final/ctest-log/test_2026-08-24_17-01-15/lunar_pure_planner_core/stdout.log`;
- `final/ctest-log/test_2026-08-24_17-01-15/lunar_pure_planner_ros/command.log`;
- `final/ctest-log/test_2026-08-24_17-01-15/lunar_pure_planner_ros/stdout.log`.

The `colcon test` evidence is retained at
`final/ctest-log/test_2026-08-24_17-01-15/logger_all.log`, with package output
under the sibling `lunar_pure_planner_core/stdout.log` and
`lunar_pure_planner_ros/stdout.log`. At the original `e47c8f5`/`f60a964`
stage, `colcon test-result --verbose` was observed as 431 tests and two failure
records, but no standalone stdout capture of that summary was saved. It is a
historical observation and cannot be reconstructed from today's shared XML.

The durable historical result anchors are the timestamped, unmodified package
logs:

- `final/ctest-log/test_2026-08-24_17-01-15/lunar_pure_planner_core/stdout.log`
  records 97/97 wheel tests and 19/19 core CTest targets;
- `final/ctest-log/test_2026-08-24_17-01-15/lunar_pure_planner_ros/stdout.log`
  records the launch pytest 9/9 and ROS CTest 11/12, including the original
  `RollingFirstLocalRequestKeepsTheColdGlobalTimingWindow` failure.

The shared `final/test-build` result XML was overwritten by the later
metrics-retention fix wave. At this documentation update, rebuilding a summary
from that current XML yields 435 tests and four failure records: the current
wheel XML is 98 tests/one failure, the current ROS server XML is 46 tests/one
failure, and the core/ROS CTest aggregates each contain one failed target.
Those XML files describe the fix-wave state audited in the addendum below;
they are not evidence for the historical 431/two, 97/97 or 19/19 results. No
duplicate-failure interpretation from the current XML is applied to the
original stage.

The 97/97 wheel result was the complete wheel target executed by that core
CTest run, not a second inferred standalone result. Its logged subprocess was:

```bash
"$LUNAR_WHEEL_FINAL_ROOT/test-build/lunar_pure_planner_core/\
lunar_pure_planner_core_anytime_wheel_planner_test" \
  --gtest_output=xml:"$LUNAR_WHEEL_FINAL_ROOT/test-build/\
lunar_pure_planner_core/test_results/lunar_pure_planner_core/\
lunar_pure_planner_core_anytime_wheel_planner_test.gtest.xml"
```

The historical core stdout log records `Running 97 tests` and `97 tests`
passed. The `gtest_output` destination shown above is command provenance only;
the XML now present at that shared path belongs to the later fix wave. The
exact launch-only pytest subprocess executed by the original ROS CTest target
was:

```bash
/usr/bin/python3 -u -m pytest \
  /home/kai/CodexDownloads/lunar_navigation/\
lunar_pure_planner_orin-worktrees/jazzy-300m-exploration/\
tests/launch/test_pure_plan_motion_server.py \
  -o cache_dir="$LUNAR_WHEEL_FINAL_ROOT/test-build/lunar_pure_planner_ros/\
ament_cmake_pytest/pure_plan_motion_server_launch_test/.cache" \
  --junit-xml="$LUNAR_WHEEL_FINAL_ROOT/test-build/lunar_pure_planner_ros/\
test_results/lunar_pure_planner_ros/pure_plan_motion_server_launch_test.xunit.xml" \
  --junit-prefix=lunar_pure_planner_ros
```

The historical ROS stdout log records 9/9 passed. The xUnit destination is in
the same shared `test-build` tree and is not used as the original-stage anchor.

The one ROS failure is
`PurePlanMotionServer.RollingFirstLocalRequestKeepsTheColdGlobalTimingWindow`.
Focused repeats were fail/pass/fail, consistent with a cancel/goal-completion
race. The separately known
`RollingFirstCycleFinalizedAtTwoPointFiveSecondsIsPlanFoundLate` baseline passed
this fresh run. No unrelated production code was changed.

The 300 m unit timing assertion measured approximately 1.04/1.06 s under the
host powersave environment. Both baseline and HEAD exceeded the unchanged
`<1 s` threshold, so this is not reported as a new functional regression or a
passing timing gate.

### SE(2) command

```bash
"$LUNAR_WHEEL_FINAL_ROOT/test-build/lunar_pure_planner_core/\
lunar_pure_planner_core_anytime_wheel_planner_test" \
  --gtest_filter='WheelPlanner.PlansMultiplePrimitivesFromArbitraryTranslatedStart:WheelPlanner.PlansMultiplePrimitivesFromArbitrarySE2Start:WheelPlanner.RigidTransformPreservesRequestRelativeTrajectory'
```

All three tests passed. Only the corresponding fresh verification checkboxes
are marked in the SE(2) plan; ROS Action and RViz steps remain unchecked.

### Production closure

The first production attempt used the brief's original `prod-log` command:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base "$LUNAR_WHEEL_FINAL_ROOT/prod-log" build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_WHEEL_FINAL_ROOT/prod-build" \
  --install-base "$LUNAR_WHEEL_FINAL_ROOT/prod-install" \
  --packages-up-to lunar_pure_planner_ros \
    lunar_pure_wheeled_controller lunar_pure_exploration_sim \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF
```

It failed while configuring `lunar_pure_exploration_sim` because
`nlohmann_jsonConfig.cmake` was unavailable. That failure is retained at
`final/prod-log/build_2026-08-24_17-06-02/logger_all.log`. The package-level
file that contains the quoted `CMake Error` and missing-config text is
`final/prod-log/build_2026-08-24_17-06-02/lunar_pure_exploration_sim/stdout_stderr.log`;
`prod-log` is not success evidence.

The successful resume used an external deb-extracted prefix and the separate
`prod-log-resume` log root, without adding a dependency or artifact to the
repository:

```bash
export CMAKE_PREFIX_PATH=/home/kai/CodexDownloads/lunar_navigation/\
exploration_jazzy_build/deps/nlohmann-json3-dev/usr${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}
source /opt/ros/jazzy/setup.bash
colcon --log-base "$LUNAR_WHEEL_FINAL_ROOT/prod-log-resume" build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_WHEEL_FINAL_ROOT/prod-build" \
  --install-base "$LUNAR_WHEEL_FINAL_ROOT/prod-install" \
  --packages-up-to lunar_pure_planner_ros \
    lunar_pure_wheeled_controller lunar_pure_exploration_sim \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF
```

Result: 8/8 packages built. Every operator-required executable resolves from
the same `prod-install` overlay and every package reports `RelWithDebInfo`.
The successful log is
`final/prod-log-resume/build_2026-08-24_17-08-32/logger_all.log`.

## Post-review metrics-retention fix wave

This is a source-and-test audit after the original Task 8 runtime capture; it
does not replace the `e47c8f5` test table above or promote runtime acceptance.
The scoped code review approved both fix-wave commits:

- `326bc3c` adds core post-graph exception metric retention and the ROS
  `ReplaceWithFailure()` result-replacement helper;
- `639ce51` has no planner-core diff and extends the ROS post-plan catch,
  finalization fallback and trusted-bridge failure coverage while preserving
  result timing and optional wheel metrics.

Focused verification at the fix-wave source:

| Check | Result | External evidence |
| --- | --- | --- |
| Core post-graph metric regression | 1/1 passed | `final/final-fix-round1-core-metrics-green.log` |
| Three new ROS replacement regressions | 3/3 passed | `final/final-fix-round1-focused-ros-green.log` |
| Request diagnostics GoogleTest | 5/5 passed | `final/final-fix-round1-request-diagnostics-gtest.log` |
| Focused request-diagnostics contract pytest | 1/1 selected test passed; 5 deselected | `final/final-fix-round1-request-diagnostics-pytest.log` |

The intentional core regression increases the complete wheel target from the
original Task 8 count of 97 to 98 tests. Two distinct full-result groups must
remain visible:

| Evidence run | Full wheel target | Core CTest | External evidence |
| --- | --- | --- | --- |
| Earlier fix-wave run of the `326bc3c` core implementation, unchanged at `639ce51` | 98/98 passed | 19/19 passed | `final/final-fix-full-wheel-verified.log`; `final/final-fix-core-ctest.log` |
| Current-source rerun at `639ce51` | 96/98 passed | 18/19 passed | `final/final-fix-round1-current-source-full-wheel.log`; `final/final-fix-round1-current-source-core-ctest.log` |

The current-source standalone wheel rerun observed
`ProjectWheelCapabilityDetoursForMoreThan300Meters` at 1067 ms against the
unchanged 1 s assertion and the 750 m regression failing at segment 107. The
current-source core CTest observed the 750 m segment-107 failure, yielding
18/19 targets. Although `639ce51` has no planner-core diff and the earlier run
used the same core implementation, the reason for the differing outcomes is
undetermined. This document does not attribute them to environment, algorithm,
randomness or pre-existing behavior.

The current-source ROS CTest remains 11/12. Its one failure is the timing
fixture
`PurePlanMotionServer.RollingFirstCycleFinalizedAtTwoPointFiveSecondsIsPlanFoundLate`;
evidence is `final/final-fix-round1-ros-ctest.log`. This is distinct from the
one ROS target failure recorded in the original `e47c8f5` table.

No fixed-seed smoke was rerun after `326bc3c`/`639ce51`. The only smoke rows
below remain the `e47c8f5` runtime evidence at capture, and their `HEAD` labels
refer to that captured runtime source rather than the repository's later
documentation HEAD. The existing smoke therefore remains `INCOMPLETE`.
Full 300 m/RViz and Jetson AGX Orin/ROS 2 Humble remain:

```text
run directory: NONE
terminal: NOT_RUN
coverage: NOT_MEASURED
```

The approved metric-retention code scope does not establish field readiness.

## Repository and contract checks

```bash
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q -p no:cacheprovider \
  tests/foundation/test_repository_boundaries.py \
  tests/exploration/test_exploration_isolation.py \
  tests/launch/test_jazzy_300m_exploration_sim.py
git diff --check
python3 tools/check_pure_planner_external_interfaces.py
```

- The first script and the named foundation test do not exist in this checkout
  (exit 2 and pytest collection exit 4 respectively).
- The available exploration/launch set produced 16 passed, 1 failed and
  1 skipped. The failure reports pre-existing forbidden dependencies in
  exploration CMake/package/source; this branch has no exploration diff.
- The external-interface checker and `git diff --check` passed.

## Fixed-seed first-eleven comparison

Baseline evidence:

```text
/home/kai/CodexDownloads/lunar_navigation/local-wheel-efficiency-evidence/
baseline/first-eleven-planner-calls.txt
```

Current HEAD runs:

```text
/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_runs/
run-20260824-171018-seed-20260824-c4a7fcc1
/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_runs/
run-20260824-171521-seed-20260824-53116e77
```

Both current runs have the same first-eleven outcome distribution. The timing
and counter tables below are specifically parsed from the stable rerun
`run-20260824-171521-seed-20260824-53116e77`, not combined across both runs.

| Evidence | TIMEOUT | PLAN_FOUND | PLAN_FOUND_LATE | NO_PATH | candidate 6 | candidate 7 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| baseline `348c5ca...` | 8 | 2 | 0 | 1 | PLAN_FOUND | PLAN_FOUND |
| HEAD `e47c8f5` | 0 | 4 | 2 | 5 | NO_PATH | NO_PATH |

First-eleven wall timings:

| Evidence | global p50 ms | global p95 ms | local p50 ms | local p95 ms | total p50 ms | total p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline `348c5ca...` | 4.985851 | 6.8119645 | 2995.152062 | 3000.491233 | 3001.583536 | 3006.779240 |
| HEAD rerun `...53116e77` | 0.408986 | 3.378396 | 17.009285 | 2263.837283 | 19.567615 | 2265.312886 |

These are not direct same-input algorithm timings: the faster planner changes
the closed-loop vehicle/simulation phase reached by later candidate IDs. They
are retained as same-seed runtime evidence, not treated as an open-loop
performance A/B.

HEAD first-eleven wheel counters:

| Counter | Total |
| --- | ---: |
| expanded states | 24932 |
| edge validation evaluations | 172163 |
| cache hits | 90 |
| broad-phase rejects | 0 |
| full certifications | 172543 |
| full invalidations | 8709 |
| sweep cell checks | 53843656 |
| direct unknown/unsupported rejects | 8669 |
| measured-clearance rejects | 0 |
| slope/roughness rejects | 0 |
| relief/underbody rejects | 0 |
| dynamics/primitive-shape rejects | 40 |
| deadline/cancellation interruptions | 0 |
| far-clearance skips | 0 |
| occupied-clearance checks | 0 |

Every graph-created diagnostic in the stable rerun reports real wheel metrics.
Candidates 6 and 7 each have one expanded state, nine edge evaluations, eight
full invalidations and eight dynamics/primitive-shape rejects.

No dedicated baseline-versus-HEAD safety-oracle fixture acceptance-delta check
was executed. The original `e47c8f5`-stage 97/97 suite result shows that all
wheel safety cases registered at that stage passed, including the approved
adjacent-unknown and exact-boundary cases, but that historical passing suite
alone does not prove that no previously rejected fixture became newly
accepted. Consequently that smoke hard condition remains an explicit evidence
gap rather than a claimed pass.

## Historical same-seed phase diagnosis

Historical source was exported with `git archive` and built into independent
external production overlays using the same Jazzy/nlohmann configuration.

| Source | Run directory | first11 | candidate 6/7 | Final result |
| --- | --- | --- | --- | --- |
| `742abe3` | `local-wheel-efficiency-evidence/stage-bisect/742abe3/runs/run-20260824-172435-seed-20260824-1d7059b6` | 6 timeout / 3 found / 2 no-path | PLAN_FOUND / PLAN_FOUND | ERROR, result-contract mismatch, 0 goals, 0.4000 m |
| `6175c88` (`bc8^`) | `local-wheel-efficiency-evidence/stage-bisect/6175c88/runs/run-20260824-174819-seed-20260824-da456c8f` | 6 timeout / 3 found / 2 no-path | PLAN_FOUND / PLAN_FOUND | ERROR, result-contract mismatch, 0 goals, 0.4001 m |
| `bc8b8f2` | `local-wheel-efficiency-evidence/stage-bisect/bc8b8f2/runs/run-20260824-172756-seed-20260824-20f5425e` | 0 timeout / 4 found / 2 late / 5 no-path | NO_PATH / NO_PATH | operator rejected 0 completed goals, 0.3998 m |
| `e47c8f5`, first | current run `...c4a7fcc1` above | same as bc8 | NO_PATH / NO_PATH | ERROR, result-contract mismatch, 0 goals, 0.3999 m |
| `e47c8f5`, rerun | current run `...53116e77` above | same as bc8 | NO_PATH / NO_PATH | summary says no reachable frontier, but operator rejects 0 goals, 0.4000 m |

`bc8b8f2` is the smallest commit that changes the observed closed-loop phase,
but it does not modify initial-velocity continuity. Its preceding commits are
diagnostics/metric retention only. The speedup lets the first candidate batch
finish and execution begin before candidates 6 and 7; at 742/6175 those IDs
are still evaluated in the earlier stationary phase. Therefore candidate-ID
outcomes across these builds are not equal-input planner-core comparisons.

## Moving-start root cause and frozen boundary

External diagnostic-only build/run:

```text
/home/kai/CodexDownloads/lunar_navigation/local-wheel-efficiency-evidence/
stage-bisect/bc8-shape-debug/runs/
run-20260824-173743-seed-20260824-feee95b8
```

All eight candidate 6/7 first-edge rejections occur in
`ExecutionDuration()`. The request begins with:

```text
linear=(0.2, 0, 0) m/s
angular=(0, 0, -0.032688374903295229) rad/s
curvature=-0.16344187451647615 1/m
```

The frozen wheel capability supplies translation curvatures `0` and `±1 1/m`
plus spins. The straight primitives require zero yaw rate, the arcs require
`±0.2 rad/s` at 0.2 m/s, and spins require zero linear velocity. No supplied
primitive can be the dynamically continuous first edge, so the fail-closed
`NO_PATH` result is consistent with the frozen primitive contract.

The closed-loop controller computes a pure-pursuit command with an arbitrary
curvature rather than retaining one of the supplied primitive curvatures. A
separate interface risk also remains: `nav_msgs/Odometry` defines twist in
`child_frame_id`, while the current ROS adapter copies the body-frame twist and
the core projects linear velocity using the source pose yaw. Fixing only this
frame risk would not resolve the unsupported `-0.16344 1/m` curvature.

The approved SE(2) and incremental-planner designs prohibit a physical or
synthetic start connector, require every interior edge to be a full supplied
platform primitive, and prohibit changing the vehicle motion primitives in
this task. No tolerance was relaxed, input twist was not snapped to zero, and
no unconfigured-curvature reference was emitted.

Resolution requires an upstream stop-before-replan contract or a separately
approved capability/initial-connector design. It is not authorized as a local
wheel-efficiency change.

## Terrain safety and full-run boundary

`KnownFreeBilinearSupportNeighborhood()` ignores zero-weight cells and validates
only cells with positive bilinear weight. Actual positive-weight wheel support
cells remain fail-closed even if they lie outside the footprint AABB. Candidate
6/7 have zero direct-support rejects, so support validation is not their cause.

Task 7 remains `FAR_CLEARANCE_SKIP_NOT_ENABLED`: exact occupied polygon-distance
certification is retained and no skip was re-enabled.

Because the smoke fails the required candidate 6/7 condition and completes no
goal, the six-hour full 300 m RViz command was not run. No wall guard, launch
exit, operator rejection or partial summary is recorded as completion.

```text
run directory: NONE
terminal: NOT_RUN
coverage: NOT_MEASURED
```

## Stop-gated exploration Humble regression — 2026-08-25

This regression used exact source SHA
`ac6c16a57037c33e4c25a06885afaa1c6b289c76` in one disposable amd64
`osrf/ros:humble-desktop-full-jammy` container. The seven-package
`BUILD_TESTING=ON` build passed in `71.097 s`.

The five selected package suites did **not** finish green:

| Package | CTest result | Leaf result |
| --- | ---: | ---: |
| `lunar_pure_exploration_core` | 11/11 passed | 192 tests, 0 errors/failures |
| `lunar_pure_exploration_ros` | 9/9 passed | 157 tests, 0 errors/failures |
| `lunar_pure_wheeled_controller` | 3/3 passed | 83 tests, 0 errors/failures |
| `lunar_pure_planner_core` | 18/19 passed | 205 records, 1 missing-result error |
| `lunar_pure_planner_ros` | 12/12 passed | 102 tests, 0 errors/failures |

The full retained XML summary is `793 tests, 1 error, 1 failure, 0 skipped`.
The error and failure are the leaf missing-result and outer CTest records for
the same planner-core wheel target terminated by its 60 s CTest timeout; they
are not two independent product failures. The complete planner-server target
passed 46/46 in this Humble run, including the four timing/concurrency cases
that had failed in the separate native-Jazzy baseline. The planner ROS launch
pytest also passed 9/9.

One explicitly bounded direct-binary recheck reused the same Humble build and
did not rerun the package suites. It completed with 0/2 passed rather than
timing out:

- `ProjectWheelCapabilityDetoursForMoreThan300Meters` failed the unchanged
  `<1 s` assertion after `12.023 s`;
- `GlobalRouteWithEightMeterRollingHorizonReaches750MeterGoalThroughRandomObstacles`
  failed after `126.032 s` with `success=false` and `failed_segment=107`.

The production diff from `a6c63c609589468fddef056f90b3ff913701b6e3` is
empty for `lunar_pure_planner_core`, `lunar_pure_planner_ros` and
`lunar_planning_msgs`. Static checks retain
`rolling_surface_enabled: false` in the default planner YAML and `False` in
the 300 m launch, preserve the approved `/Car/T3/`, `/Car/T4/` and `/Car/T5/`
topic namespaces, and show no planning-message diff.

The two mandated repository-boundary entrypoints remain absent:

```text
tools/check_repository_boundaries.py: NOT_RUN / MISSING
tests/foundation/test_repository_boundaries.py: NOT_RUN / MISSING
```

No substitute was presented as either required gate. Full stdout, XML,
timings, commands and the focused recheck are retained under:

```text
/home/kai/CodexDownloads/lunar_navigation/sdd_builds/stop_gated_task6_humble/
ac6c16a57037-20260824T170006Z
```

Verdict for this evidence set: the Humble build and both exploration suites
pass, but the full five-package regression is **FAIL / NOT GREEN** because the
planner-core suite is nonzero and the two mandatory boundary gates are
`NOT_RUN / MISSING`. This container evidence is not a native Jazzy 300 m/RViz
run, Orin execution, DDS verification, or field-readiness acceptance.
