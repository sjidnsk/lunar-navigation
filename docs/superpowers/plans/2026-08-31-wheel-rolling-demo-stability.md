# Wheel Rolling Demo Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate avoidable rolling-demo `NO_PATH`, stale-map churn, stale-path motion, misleading empty-path RViz warnings, and stale reporter starts without weakening planner identity or timeout safety contracts.

**Architecture:** Keep production rolling-plan identity and the 3 s cycle deadline authoritative. Broaden only portal sampling, add a bounded post-success recovery loop that clears outputs before retrying, and move demo-only motion/map cadence into a small testable state object. Preserve the isolated `/lunar_demo/*` interface and keep all production failure outcomes unchanged once recovery is inapplicable or exhausted.

**Tech Stack:** C++20, ROS 2 Jazzy local validation, ament/colcon, GoogleTest, pytest, YAML, Markdown.

**Spec:** `docs/superpowers/specs/2026-08-31-wheel-rolling-demo-stability-design.md`

## Global Constraints

- Work only on `fix/wheel-rolling-demo-stability`; do not merge, push, create a PR, or touch the read-only import branch.
- Use test-first changes: state the production mutation each test catches, observe the focused test fail, add minimal production code, then observe it pass.
- Do not weaken `SamePlanningIdentity`, increase the 3 s hard deadline, introduce `/tf_static`, or publish vehicle-control topics.
- Keep formal success gated by `planning_outcome: 0`, `reason_code: PLAN_FOUND` or the existing accepted late-success code, and `has_reference: true`.
- Use `build-jazzy-fix` and `install-jazzy-fix`; put colcon logs under `/tmp/lunar-wheel-rolling-demo-fix-log` so generated artifacts are not committed.
- Stage explicit files only and run `git diff --check` before every commit.

---

### Task 1: Expand rolling portal coverage and expose portal-stage failure

**Files:**

- Modify: `ros2_ws/src/lunar_pure_planner_core/test/surface_portal_set_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/hierarchical/surface_portal_set.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp`

- [ ] Add `SurfacePortalSet.SearchesEveryLongitudinalCellInsideShortHorizon` with blocked columns arranged so only a 1–4-cell backoff is valid; this catches restoring the fixed `{0..7}` list that skips near-project progress for a 12 m horizon.
- [ ] Build `lunar_pure_planner_core` and run `ctest --test-dir build-jazzy-fix/lunar_pure_planner_core -R surface_portal_set_test --output-on-failure`; confirm the new test fails because no portal is returned.
- [ ] Replace `kBackoffCells` with deterministic longitudinal progress generation from `desired_horizon_progress_m` down to one global cell beyond `projected_progress_m`, capped at 32 longitudinal samples; retain nine lateral offsets, existing global/local safety checks, stable sorting, deduplication, exact final-goal behavior, and the 32-candidate output cap.
- [ ] Rebuild and rerun the focused core test; confirm all portal-set cases pass.
- [ ] Add a ROS server assertion that a failed portal set emits `rolling_failure_stage=PORTAL_SET`; this catches silently collapsing portal construction and local search into the same public `NO_PATH` diagnostic.
- [ ] Run the focused server test and confirm RED, then add the diagnostic field without changing public `planning_outcome` or `reason_code`, rebuild, and confirm GREEN.
- [ ] Run `git diff --check`, stage the four explicit files, and commit `fix: cover rolling portal horizon consistently`.

### Task 2: Add the bounded transient-recovery configuration contract

**Files:**

- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp`
- Modify: `tests/test_launch_contract.py`
- Modify: `config/pure_planner.yaml`

- [ ] Add parameter tests proving `rolling_transient_retry_limit` defaults to `2`, accepts `0..10`, and rejects values outside that range; these catch omission or unsafe unbounded retry configuration.
- [ ] Add the parameter to `EXPECTED_PARAMETERS` and the YAML contract test with value `2`; run `/usr/bin/python3 -m pytest -q tests/test_launch_contract.py` against the existing overlay and confirm RED.
- [ ] Declare `rolling_transient_retry_limit` in `RollingSurfaceParameters`, validate it during server configuration, and add `rolling_transient_retry_limit: 2` to `config/pure_planner.yaml`.
- [ ] Rebuild `lunar_pure_planner_ros`, rerun the focused C++ parameter tests and `tests/test_launch_contract.py`, and confirm GREEN.
- [ ] Run `git diff --check`, stage the four explicit files, and commit `feat: configure bounded rolling recovery`.

### Task 3: Recover only post-success transient local failures and clear stale outputs

**Files:**

- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp`

- [ ] Add integration tests for: first-cycle local `NO_PATH` remains terminal; post-success local `NO_PATH` retries; post-success hard `TIMEOUT` retries; each retry publishes empty local/global/timed outputs with `odom`/`map`/`odom` frames; retry diagnostics include `rolling_recovery=true` and a one-based `rolling_recovery_attempt`; a success resets the counter; exhaustion returns the original terminal failure.
- [ ] Include assertions that cancellation, invalid input, planner error, and global-route failure never enter recovery; these catch over-broad retry classification.
- [ ] Rebuild and run `ctest --test-dir build-jazzy-fix/lunar_pure_planner_ros -R pure_plan_motion_server_test --output-on-failure`; confirm the new cases fail for missing retry/clear behavior.
- [ ] Add a per-goal recovery state with `has_committed_segment` and consecutive transient retry count. Route only local portal/local planner `NO_PATH` and local-cycle `TIMEOUT` through recovery after a committed success and below the configured limit.
- [ ] On recovery, publish the original diagnostic plus `rolling_recovery=true` and `rolling_recovery_attempt=N`, publish empty global/local/timed outputs with valid frames, keep the action active, wait one existing poll interval, and begin a fresh snapshot/deadline cycle. Never republish the last segment.
- [ ] Preserve current `STALE_INPUT` retry behavior outside the new counter and reset the transient counter after every committed successful segment.
- [ ] Rebuild and rerun the complete server test executable; confirm GREEN and verify existing hard-deadline and late-result tests remain unchanged.
- [ ] Run `git diff --check`, stage the two explicit files, and commit `fix: recover bounded rolling local failures`.

### Task 4: Decouple demo local-map cadence from odometry and stop on cleared paths

**Files:**

- Create: `ros2_ws/src/lunar_pure_planner_ros/include/lunar_pure_planner_ros/lunar_surface_demo_state.hpp`
- Create: `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_demo_state.cpp`
- Create: `ros2_ws/src/lunar_pure_planner_ros/test/lunar_surface_demo_state_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_demo_node.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/CMakeLists.txt`

- [ ] Add state tests proving: the first local map is due; motion below 4 m does not make another map due; reaching 4 m does; resetting the start forces the next map; a non-empty path starts motion; an empty path clears it and prevents further motion. These catch both the 2 Hz local-map sequence churn and stale-path continuation after a failure clear.
- [ ] Add the test target to CMake, rebuild it, and confirm RED because the state API is not implemented.
- [ ] Implement `LunarSurfaceDemoState` with rover pose, active-path progress, optional last local-map center, `Reset`, `AcceptPath`, `Advance`, `LocalMapDue(4.0)`, and `MarkLocalMapPublished`.
- [ ] Refactor the node to delegate start reset, path acceptance (including empty paths), and 0.5 m motion to the state object. Keep the 500 ms timer, odometry, TF delivery, global visualization, and goal behavior unchanged.
- [ ] Gate both `/lunar_demo/grid_map` and `/lunar_demo/local_map_viz` on first publish or at least 4 m displacement, and mark the center only after both are published.
- [ ] Rebuild and run `lunar_surface_demo_state_test`, `lunar_surface_scenario_test`, and the server test; confirm GREEN.
- [ ] Run `git diff --check`, stage the five explicit files, and commit `fix: stabilize rolling demo input cadence`.

### Task 5: Report each rolling cycle from current odometry

**Files:**

- Modify: `ros2_ws/src/lunar_pure_planner_ros/include/lunar_pure_planner_ros/lunar_surface_reporter.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_reporter.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/lunar_surface_reporter_test.cpp`

- [ ] Add a reporter-state test that begins at one pose, updates the start before a later result, and expects the emitted summary to use the updated pose; this catches reporting the original clicked start for every rolling cycle.
- [ ] Rebuild and run `lunar_surface_reporter_test`; confirm RED because no start-update API exists.
- [ ] Add `PlanningReportState::SetStart(double, double)` and call it from `HandleDiagnostics` using the latest odometry immediately before `SetResult`; retain the clicked goal and accumulated path lengths.
- [ ] Rebuild and rerun `lunar_surface_reporter_test`; confirm GREEN.
- [ ] Run `git diff --check`, stage the three explicit files, and commit `fix: report current rolling start pose`.

### Task 6: Update operator documentation and verification evidence

**Files:**

- Modify: `README.md`
- Modify: `docs/操作指令.md`
- Modify: `VERIFICATION.md`

- [ ] Document the unchanged wheel-demo launch command, the 4 m local-map refresh rule, bounded two-attempt transient recovery, empty-path stop behavior, and diagnostic keys. State explicitly that the 3 s deadline and identity checks were not relaxed.
- [ ] Record current evidence by layer: source/static, local Jazzy, and `NOT_RUN` for Humble, Orin, DDS, rosbag, controller, and vehicle.
- [ ] Search the three documents for stale claims that local maps publish every 500 ms or that any action `SUCCEEDED` alone proves planning success; correct only affected passages.
- [ ] Run documentation/contract tests and `git diff --check`, stage the three explicit files, and commit `docs: explain rolling demo recovery contract`.

### Task 7: Full validation and runtime acceptance

**Files:**

- Verify only; update `VERIFICATION.md` if measured evidence differs from Task 6 wording.

- [ ] Run `/usr/bin/python3 -m pytest -q --ignore=build-jazzy-baseline --ignore=install-jazzy-baseline --ignore=build-jazzy-fix --ignore=install-jazzy-fix` from the repository root with Jazzy and the fix overlay sourced.
- [ ] Run package CTest serially: `ctest --test-dir build-jazzy-fix/lunar_pure_planner_core --output-on-failure -j1` and `ctest --test-dir build-jazzy-fix/lunar_pure_planner_ros --output-on-failure -j1`.
- [ ] Launch the isolated wheel demo on a fresh `ROS_DOMAIN_ID`, start subscribers before the one-shot/static inputs, and exercise the previously failing goal long enough to cross multiple local-map refresh boundaries.
- [ ] Confirm each accepted success has `planning_outcome: 0`, `reason_code: PLAN_FOUND` or the existing accepted late-success code, and `has_reference: true`; confirm no rover motion follows a failure clear, no empty-path frame warning occurs, recovery is at most two attempts, and reporter starts track odometry.
- [ ] Stop all demo processes, record runtime observations in `VERIFICATION.md`, and leave Humble/Orin/DDS/vehicle layers `NOT_RUN`.
- [ ] Run `git status --short`, `git diff --check`, and inspect `git diff integration/pure-planner-orin...HEAD --stat` plus the full diff for unrelated files or generated artifacts.
- [ ] If evidence wording changed, stage only `VERIFICATION.md` and commit `docs: record rolling demo validation`.

### Task 8: Branch completion review

- [ ] Read and apply `superpowers:verification-before-completion`, then run fresh final verification commands and quote their exit status/counts in the handoff.
- [ ] Read and apply `superpowers:finishing-a-development-branch`; present integration choices without merging, pushing, creating a PR, or deleting the worktree unless the user explicitly authorizes one.
- [ ] Report commits, changed behavior, exact launch command, validation boundaries, and remaining `NOT_RUN` target evidence.
