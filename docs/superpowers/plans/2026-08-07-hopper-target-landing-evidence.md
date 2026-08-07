# Hopper Target Landing Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the hopper L0 local map certify the selected landing target instead of the launch-centered neighborhood, while preserving certified overflight of finite-height obstacles.

**Architecture:** The core planner retains its existing split: `local_map` certifies the exact landing support region and `global_map` certifies the complete 3-D ballistic tube. The external RViz harness atomically moves only the hopper local window through the bridge node's ROS parameter service, then uses the existing `ready` burst to publish a same-generation snapshot before sending `PlanMotion`.

**Tech Stack:** C++20, ROS 2 Humble, rclcpp/rclpy, GridMap, ROS parameter services, GoogleTest, pytest, colcon Release.

## Global Constraints

- Work in `/mnt/data/WS/.lunar-navigation-worktrees/hopper-target-landing-evidence` on branch `hopper-target-landing-evidence`; preserve `/mnt/data/WS/lunar-navigation/.vscode/`.
- Manage `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression` as an independent Git root; create an external feature worktree and merge only into that repository's `main`.
- Source `/opt/ros/humble/setup.bash` and verify `ROS_DISTRO=humble` before every ROS build or test.
- Build C++ with `-DCMAKE_BUILD_TYPE=Release`; all build/install/log/test artifacts stay under `/home/kai/CodexDownloads/lunar_navigation/hopper_target_landing_evidence/`.
- Do not change platform capability values, `PlanMotion`, `GridMap`, or the established environment map Topic names.
- Keep wheel/legged windows platform-centered. Keep the hopper target window fixed until the next accepted target; hopper odometry must not recenter it.
- Treat ordinary `obstacle + obstacle_height` as finite-height geometry; keep `forbidden` fail-closed as an absolute exclusion layer.
- Follow test-first red/green cycles for every production behavior change and use UTF-8 for Chinese documentation.

---

### Task 1: Freeze the formal map and overflight contract in the main repository

**Files:**
- Modify: `docs/interfaces/external-input-baseline.md`
- Modify: `ros2_ws/src/lunar_planner_core/test/hopper_planner_test.cpp`
- Modify: `docs/validation/hierarchical-global-planning.md`

**Interfaces:**
- Consumes: existing `WorldSnapshot.global_map`, `WorldSnapshot.local_map`, `obstacle`, `obstacle_height`, and `forbidden` layers.
- Produces: a documented platform-dependent local-map centering contract and named regression evidence for finite-height overflight.

- [ ] **Step 1: Run the existing positive overflight test as characterization evidence**

  Run:

  ```bash
  source /opt/ros/humble/setup.bash
  test "$ROS_DISTRO" = humble
  ctest --test-dir /home/kai/CodexDownloads/lunar_navigation/hopper_target_landing_evidence/baseline/build/lunar_planner_core \
    -R lunar_planner_core_hopper_planner_test --output-on-failure
  ```

  Expected: PASS, including `SearchesAnAlternateFeasibleTimeWhenMinimumArcIsBlocked`, proving a 20 m request can select a higher arc over a 6 m obstacle.

- [ ] **Step 2: Add the finite-height negative regression**

  Add a test named `RejectsObstacleThatIntersectsEveryFuelFeasibleFlightTube`. Use `LongHopInput(20.0)`, place a sufficiently tall obstacle at the midpoint in `global_map`, and assert no reference is returned with the stable flight-tube collision/no-safe-route reason produced by the current certifier. This is a contract-locking characterization test; no core production change is expected.

- [ ] **Step 3: Run the hopper test target**

  Run the same `ctest -R lunar_planner_core_hopper_planner_test` command. Expected: PASS with both the higher-arc positive and all-arcs-blocked negative cases.

- [ ] **Step 4: Document the map split**

  Add these exact rules to `external-input-baseline.md`:

  ```text
  wheel/legged local_map: platform-centered L0 execution evidence
  hopper local_map: target landing-region L0 evidence
  hopper global_map: complete ballistic flight-tube evidence
  ordinary obstacle: finite top at elevation + max(obstacle_height, resolution)
  forbidden: absolute exclusion until a separate no-landing/no-fly schema is frozen
  ```

  Record the new design, test name, and Release verification command in `hierarchical-global-planning.md` without rewriting historical baselines.

- [ ] **Step 5: Commit Task 1**

  ```bash
  git add docs/interfaces/external-input-baseline.md \
    docs/validation/hierarchical-global-planning.md \
    ros2_ws/src/lunar_planner_core/test/hopper_planner_test.cpp
  git commit -m "test: lock hopper target evidence semantics"
  ```

### Task 2: Add an atomic hopper target-window control to the external bridge

**Files:**
- Modify: `ros2_ws/src/lunar_isaac_validation/package.xml`
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_bridge_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py`

**Interfaces:**
- Consumes: ROS parameters `landing_evidence_center_x_m` and `landing_evidence_center_y_m` set atomically.
- Produces: `InteractiveSnapshotBridge._prepare_landing_evidence_parameters(parameters) -> SetParametersResult` and a target-centered `_local_map_message`/`_local_hazard_message` for hopper only.

- [ ] **Step 1: Create the external feature worktree and run focused baseline tests**

  ```bash
  git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression worktree add \
    /home/kai/CodexDownloads/lunar_navigation/hopper_target_landing_evidence/rviz-worktree \
    -b hopper-target-landing-evidence-rviz main
  source /opt/ros/humble/setup.bash
  test "$ROS_DISTRO" = humble
  python3 -m pytest -q \
    ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py \
    --ignore=build --ignore=install --ignore=log
  ```

  Expected: the current focused suite passes before edits.

- [ ] **Step 2: Write failing bridge tests**

  Add tests that instantiate a synthetic hopper bridge and call `set_parameters_atomically()` with two `rclpy.parameter.Parameter` values. Assert:

  ```python
  result.successful is True
  bridge._local_map_message.info.pose.position.x == pytest.approx(0.0)
  bridge._local_map_message.info.pose.position.y == pytest.approx(12.0)
  ```

  The target `(0.0, 12.0)` is about 18.44 m from the synthetic hopper start `(-18.0, 16.0)`, beyond the old 12 m half extent and far enough from map boundaries for an exact-centered window. Add rejection cases for one-coordinate-only updates, non-finite values, wheel sessions, non-L0 source data, and targets outside the source map. Add a hopper odometry case proving the prepared target center remains unchanged while the same odometry still recenters wheel/legged windows.

- [ ] **Step 3: Run focused tests and verify RED**

  Expected failure: the parameters are undeclared or no callback moves the local window.

- [ ] **Step 4: Implement the minimal bridge behavior**

  Add `rcl_interfaces` as an execution dependency. In `InteractiveSnapshotBridge`:

  ```python
  _LANDING_CENTER_X = "landing_evidence_center_x_m"
  _LANDING_CENTER_Y = "landing_evidence_center_y_m"
  ```

  Declare both parameters at the platform's initial `x/y`, register an on-set callback, require both names in the same atomic request, validate hopper/L0/finite/source coverage, call `_recenter_local_window((x, y))`, and return `SetParametersResult(successful=True)`. Return stable reasons `LANDING_EVIDENCE_TARGET_INVALID` or `LANDING_EVIDENCE_SOURCE_NOT_L0` without mutating the previous map on failure. Change `_accept_simulated_odometry()` so only wheel/legged call `_recenter_local_window()`.

- [ ] **Step 5: Run focused tests and verify GREEN**

  Expected: all bridge tests pass.

- [ ] **Step 6: Commit Task 2 in the external worktree**

  ```bash
  git add ros2_ws/src/lunar_isaac_validation/package.xml \
    ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_bridge_node.py \
    ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py
  git commit -m "feat: prepare hopper target landing window"
  ```

### Task 3: Synchronize target-window preparation before the ready snapshot

**Files:**
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_session.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_session.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py`

**Interfaces:**
- Consumes: `_Transport.prepare_landing_evidence(target_xy, timeout_s)`.
- Produces: `SessionSupervisor.refresh_generation(landing_target_xy: tuple[float, float] | None = None)` that orders `prepare -> ready` for hopper and preserves `ready` only for ground platforms.

- [ ] **Step 1: Write failing supervisor ordering tests**

  Extend the fake transport with:

  ```python
  def prepare_landing_evidence(self, target_xy, timeout_s):
      self.events.append(f"prepare:{target_xy[0]}:{target_xy[1]}:{timeout_s}")
  ```

  Assert a hopper refresh produces `prepare:0.0:12.0:9.0` immediately before `ready:9.0`; a wheel refresh with a landing target raises `LANDING_EVIDENCE_TARGET_INVALID`; and an injected preparation failure does not call ready.

- [ ] **Step 2: Run session tests and verify RED**

  Expected failure: `refresh_generation()` does not accept `landing_target_xy`.

- [ ] **Step 3: Implement supervisor ordering**

  Extend `_Transport`, validate the optional pair, require an active hopper when it is present, call `prepare_landing_evidence()` first, then call the existing `bridge_ready()` and retain the returned generation stamp.

- [ ] **Step 4: Implement the ROS transport using Humble's standard service**

  Create a client for:

  ```text
  /lunar_isaac_interactive_bridge/set_parameters_atomically
  rcl_interfaces/srv/SetParametersAtomically
  ```

  Build both parameter messages with `rclpy.parameter.Parameter(...).to_parameter_msg()`, wait using the existing worker-thread event loop, and raise `SessionError("LANDING_EVIDENCE_PREPARATION_FAILED:<reason>")` when the response is missing or unsuccessful. Include this endpoint in pre/post absence checks.

- [ ] **Step 5: Write failing controller propagation tests**

  Update the fake supervisor signature, capture each `landing_target_xy`, and assert:

  ```python
  hopper goal (0.0, 12.0) -> refresh_generation((0.0, 12.0)) -> Action send
  wheel/legged goal         -> refresh_generation(None)       -> Action send
  preparation failure      -> no Action, status LANDING_EVIDENCE_PREPARATION_FAILED
  committed hopper reset   -> switch hopper -> prepare target -> ready -> Action send
  ```

- [ ] **Step 6: Run controller tests and verify RED**

  Expected failure: the controller refreshes all inputs without passing the hopper target.

- [ ] **Step 7: Implement controller propagation and verify GREEN**

  In `_refresh_inputs_and_send()` and `_reset_hopper_and_send()`, pass `(pose.pose.position.x, pose.pose.position.y)` only for hopper. Preserve the existing request token, cancellation, mission revision, and Action ordering rules. Convert preparation failures to stable controller reason `LANDING_EVIDENCE_PREPARATION_FAILED` while retaining the detailed child/service reason in logs.

- [ ] **Step 8: Commit Task 3 in the external worktree**

  ```bash
  git add ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_session.py \
    ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_node.py \
    ros2_ws/src/lunar_isaac_validation/test/test_interactive_session.py \
    ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py
  git commit -m "feat: synchronize hopper landing evidence"
  ```

### Task 4: Qualify a target beyond the former local-window radius

**Files:**
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_synthetic_interactive_integration.py`
- Modify: `README.md`
- Modify: `docs/validation/2026-08-07-v2-synthetic-rviz-qualification.md`

**Interfaces:**
- Consumes: the synthetic 50 m L0 map, RViz `/goal_pose`, prepared `/environment/map_local`, and `/plan_motion`.
- Produces: an end-to-end proof that the local map covers a target farther than 12 m from launch and the returned hop clears finite-height obstacles in the 3-D tube.

- [ ] **Step 1: Write the failing integration test**

  Use deterministic synthetic seed `20260805`, select hopper, and use fixed safe target `(0.0, 12.0)`, which is about 18.44 m from `(-18.0, 16.0)`. Assert before Action completion:

  ```python
  math.dist(start_xy, target_xy) > 12.0
  local_map_contains(target_xy, support_radius_m=0.65)
  result.reason_code == "HOPPER_SINGLE_HOP_AVAILABLE"
  len(result.motion_reference.hops) == 1
  ```

  Also assert the RViz evidence contains `hopper_flight_tube` and a closed `hopper_landing_region` polygon.

- [ ] **Step 2: Run the integration test and verify RED**

  Expected pre-fix failure: `LANDING_EVIDENCE_INSUFFICIENT` or the old local map does not contain the selected target.

- [ ] **Step 3: Bind the fixed target to the frozen map fixture**

  Assert the full 0.65 m support disk around `(0.0, 12.0)` is valid, obstacle-free and non-forbidden in seed `20260805`; fail the fixture rather than scanning for a replacement. Do not change platform capabilities or obstacle occupancy.

- [ ] **Step 4: Run integration and focused external suites**

  ```bash
  source /opt/ros/humble/setup.bash
  test "$ROS_DISTRO" = humble
  python3 -m pytest -q \
    ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py \
    ros2_ws/src/lunar_isaac_validation/test/test_interactive_session.py \
    ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py \
    ros2_ws/src/lunar_isaac_validation/test/test_synthetic_interactive_integration.py \
    --ignore=build --ignore=install --ignore=log
  ```

  Expected: PASS.

- [ ] **Step 5: Update the external operator guide and commit**

  Document that the translucent tube may pass over red finite-height obstacle cells, while the landing polygon itself must remain obstacle-free. Explain that `forbidden` remains non-traversable and that map evidence missing around the target produces a preparation/landing-evidence failure.

  ```bash
  git add README.md \
    docs/validation/2026-08-07-v2-synthetic-rviz-qualification.md \
    ros2_ws/src/lunar_isaac_validation/test/test_synthetic_interactive_integration.py
  git commit -m "test: qualify distant hopper landing target"
  ```

### Task 5: Run full qualification and integrate both repositories

**Files:**
- Modify: `docs/validation/hierarchical-global-planning.md` with append-only final evidence.

**Interfaces:**
- Consumes: main feature branch and external RViz feature branch.
- Produces: Release build/test evidence, repository-boundary evidence, and fast-forward integration into the correct branch of each Git root.

- [ ] **Step 1: Run main Release qualification**

  Use a fresh artifact root `.../qualification` and run:

  ```bash
  source /opt/ros/humble/setup.bash
  test "$ROS_DISTRO" = humble
  colcon --log-base "$QUAL/log" build --base-paths ros2_ws/src \
    --build-base "$QUAL/build" --install-base "$QUAL/install" \
    --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
  source "$QUAL/install/setup.bash"
  colcon --log-base "$QUAL/log-test" test --base-paths ros2_ws/src \
    --build-base "$QUAL/build" --install-base "$QUAL/install" \
    --test-result-base "$QUAL/test-results"
  colcon test-result --test-result-base "$QUAL/test-results" --verbose
  python3 tools/check_repository_boundaries.py .
  python3 -m pytest -q tests/foundation/test_repository_boundaries.py
  git diff --check
  ```

  Set `QUAL=/home/kai/CodexDownloads/lunar_navigation/hopper_target_landing_evidence/qualification` before the commands.

- [ ] **Step 2: Run full external qualification**

  From the external worktree, run source pytest with `--ignore=build --ignore=install --ignore=log`, then Release colcon build/test against the updated main worktree's `ros2_ws/src` plus external `ros2_ws/src`. Save all artifacts outside both repositories.

- [ ] **Step 3: Append exact evidence and commit**

  Record commit hashes, test counts, artifact roots, the selected distant target distance, local-map dimensions/origin, returned reason code, and finite-height overflight result. Do not edit historical qualification numbers.

- [ ] **Step 4: Merge the main feature branch**

  Confirm `/mnt/data/WS/lunar-navigation` still has only the user's `.vscode/` change, then:

  ```bash
  git -C /mnt/data/WS/lunar-navigation merge --ff-only hopper-target-landing-evidence
  ```

  Re-run repository boundary checks from `integration`.

- [ ] **Step 5: Merge the external feature branch**

  Confirm the external `main` checkout is clean, then:

  ```bash
  git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression \
    merge --ff-only hopper-target-landing-evidence-rviz
  ```

  Re-run the focused external tests from `main`. Do not push or publish remotely unless separately requested.
