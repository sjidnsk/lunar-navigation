# Jazzy Exploration Closed Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compose simulation, existing explorer/planner, extended controller, RViz, task startup, and result recording into a reproducible 300 m Jazzy run that ends only with no reachable frontier.

**Architecture:** Add coordinator, recorder, and HUD executables to the test-only simulation package, then compose all packages in one launch with exact Car/T3/T4/T5 interfaces. Separate a bounded headless smoke test from the long full run while using the same launch and parameters.

**Tech Stack:** ROS 2 Jazzy launch, rclcpp, rclcpp_action, RViz2, MarkerArray, diagnostic_msgs, JSON/CSV, pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-300m-jazzy-exploration-simulation-design.md`

**Depends on:** `2026-08-24-wheel-controller-motion-modes.md` and `2026-08-24-300m-simulation-domain.md`.

## Global Constraints

- Use exact `/Car/T3/...`, `/Car/T4/...`, and sole command `/Car/T5/Car_Cmd_Vel` interfaces.
- Set controller `reference_topic=/Car/T4/execution/motion_reference`; do not change its standalone default.
- Success requires `COMPLETED` plus `COMPLETED_NO_REACHABLE_FRONTIER`; coverage is never a threshold.
- Build/log/install and run artifacts stay under `/home/kai/CodexDownloads/lunar_navigation/`, outside Git.
- Local Jazzy evidence is not Humble/Orin certification.

---

## File Structure

- Create coordinator, recorder, and HUD sources in `lunar_pure_exploration_sim`.
- Create `launch/jazzy_300m_exploration_sim.launch.py` and `rviz/jazzy_300m_exploration_sim.rviz`.
- Create `tests/launch/test_jazzy_300m_exploration_sim.py`.
- Create `scripts/run_jazzy_300m_exploration_sim.sh` and update `README.md`.

### Task 1: Readiness-gated task coordinator

**Files:**
- Create: `ros2_ws/src/lunar_pure_exploration_sim/include/lunar_pure_exploration_sim/run_coordinator.hpp`
- Create: `ros2_ws/src/lunar_pure_exploration_sim/src/run_coordinator.cpp`
- Create: `ros2_ws/src/lunar_pure_exploration_sim/src/run_coordinator_main.cpp`
- Test: `ros2_ws/src/lunar_pure_exploration_sim/test/run_coordinator_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_sim/{CMakeLists.txt,package.xml}`

**Interfaces:**
- Observes: global/local maps, odometry, TF, exploration status, `/Car/T4/plan_motion` readiness.
- Produces once: `/Car/T4/exploration/task` START with task ID `jazzy-300m-<seed>` and corners `(-145,-145),(145,-145),(145,145),(-145,145)`.

- [ ] **Step 1: Add a failing readiness test**

Expose `CoordinatorReadiness` with six booleans and `ShouldStart()`. Assert false until every input and Action readiness is true, true exactly once, and false after `MarkStarted()`.

- [ ] **Step 2: Implement state machine and ROS node**

Subscribe to five Topics, create a `PlanMotion` Action client, check `action_server_is_ready()` on a 100 ms timer, and publish a reliable/transient-local task only after readiness. Set `header.frame_id="map"`, command START, and the four approved points in order.

- [ ] **Step 3: Build, test, and commit**

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/closed_loop/log \
  build --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/closed_loop/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/closed_loop/install \
  --packages-up-to lunar_pure_exploration_sim
source /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/closed_loop/install/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/closed_loop/log \
  test \
  --build-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/closed_loop/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/closed_loop/install \
  --packages-select lunar_pure_exploration_sim --ctest-args -R run_coordinator_test
git add ros2_ws/src/lunar_pure_exploration_sim
git commit -m "feat: start simulation exploration after readiness"
```

Expected: coordinator test passes and executable installs.

### Task 2: Run recorder and RViz HUD

**Files:**
- Create: `ros2_ws/src/lunar_pure_exploration_sim/include/lunar_pure_exploration_sim/run_recorder.hpp`
- Create: `ros2_ws/src/lunar_pure_exploration_sim/src/run_recorder.cpp`
- Create: `ros2_ws/src/lunar_pure_exploration_sim/src/run_recorder_main.cpp`
- Create: `ros2_ws/src/lunar_pure_exploration_sim/src/simulation_hud_node.cpp`
- Test: `ros2_ws/src/lunar_pure_exploration_sim/test/run_recorder_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_sim/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_pure_exploration_sim/package.xml`

**Interfaces:**
- Consumes: exploration status/diagnostics, planner diagnostics, odometry, and `/Car/T4/simulation/sim_elapsed`.
- Writes: `summary.json`, `coverage.csv`, `trajectory.csv` under absolute `output_dir`.
- Publishes: `/Car/T4/simulation/hud` MarkerArray and `/Car/T4/simulation/planned_path` (`nav_msgs/msg/Path`) converted from the active MotionReference preview.

- [ ] **Step 1: Add failing recorder tests**

Feed two status samples and poses `(0,0)->(1,0)->(1,1)`. Assert distance `2.0 m`, CSV headers/rows, timing keys, wall/sim elapsed, final coverage equality, and success only when:

```cpp
status.state == PureExplorationStatus::COMPLETED &&
status.reason_code == "COMPLETED_NO_REACHABLE_FRONTIER"
```

Assert ERROR, timeout, and shutdown remain unsuccessful.

- [ ] **Step 2: Implement external recording**

Require an absolute `output_dir` outside the repository path supplied by launch. Flush each CSV row and write `summary.json.tmp` followed by same-directory rename at terminal state. Use `nlohmann_json` for JSON encoding and declare it in CMake/package.xml. Aggregate `candidate_`, `rolling_`, and `stuck_` global/local timing keys. Use steady clock for wall elapsed and plant state for simulated elapsed.

- [ ] **Step 3: Implement persistent HUD**

Publish transient-local text containing state/reason, coverage/areas, frontier/candidate counts, completed goals, planning calls/times, distance, and wall/sim elapsed. Never delete the terminal marker.

- [ ] **Step 4: Run test and commit**

Use Task 1 external build paths with `--ctest-args -R run_recorder_test`, require PASS, then commit the simulation package with message `feat: record and display exploration results`.

### Task 3: Full launch and RViz composition

**Files:**
- Create: `launch/jazzy_300m_exploration_sim.launch.py`
- Create: `rviz/jazzy_300m_exploration_sim.rviz`
- Modify: `ros2_ws/src/lunar_pure_exploration_sim/CMakeLists.txt`
- Test: `tests/launch/test_jazzy_300m_exploration_sim.py`

**Interfaces:**
- Arguments: `seed=20260824`, `speed_multiplier=20.0`, `start_rviz=true`, required absolute `output_dir`.
- Starts: simulation, planner, explorer, controller, coordinator, recorder, HUD, optional RViz.

- [ ] **Step 1: Add failing static launch tests**

Assert every required executable appears, controller reference is `/Car/T4/execution/motion_reference`, command is `/Car/T5/Car_Cmd_Vel`, map inputs are `/Car/T3/...`, speed is `20.0`, RViz is conditional, and no `/lunar_demo/` endpoint appears.

- [ ] **Step 2: Implement launch composition**

Start the planner in wheel mode and exploration with these test-only capacities:

```text
maximum_position_probes=8192
maximum_candidate_views=4096
maximum_collision_work_units=4194304
maximum_visibility_work_units=4096
maximum_path_preview_poses=4096
maximum_executable_path_points=4096
maximum_failure_entries=2048
maximum_failure_patch_cells_per_entry=512
maximum_failure_total_patch_cells=262144
```

Pass exact Topic names, override only the controller reference Topic, and start simulation-side nodes. These capacities are functional-test inputs, not an Orin capability claim.

- [ ] **Step 3: Add RViz displays**

Set fixed frame `map`; add global Map, actual/planned Paths, exploration and vehicle MarkerArrays, FOV, current goal, and HUD. Show local window/elevation through markers, without a nonstandard GridMap plugin.

- [ ] **Step 4: Run static tests and commit**

```bash
python3 -m pytest -q -p no:cacheprovider tests/launch/test_jazzy_300m_exploration_sim.py -k static
git add launch/jazzy_300m_exploration_sim.launch.py
git add rviz/jazzy_300m_exploration_sim.rviz
git add ros2_ws/src/lunar_pure_exploration_sim/CMakeLists.txt
git add tests/launch/test_jazzy_300m_exploration_sim.py
git commit -m "feat: compose 300m Jazzy exploration demo"
```

Expected: static contract passes.

### Task 4: Bounded live smoke test

**Files:**
- Modify: `tests/launch/test_jazzy_300m_exploration_sim.py`

**Interfaces:**
- Proves: graph readiness, START, planning/execution, motion, coverage increase, and planner timing.

- [ ] **Step 1: Add a failing live test**

Launch with `start_rviz:=false`, unique `ROS_DOMAIN_ID`, and `ROS_LOCALHOST_ONLY=1`. Within 120 wall seconds require Task 3 inputs, Action server, task ID, selecting/planning/executing states, odometry displacement >=0.2 m, coverage increase, nonempty reference, nonzero T5 command, and diagnostics with global/local timing records.

- [ ] **Step 2: Implement bounded lifecycle**

Use `Popen(start_new_session=True)`, log externally, and in `finally` send SIGINT to the process group, wait 10 seconds, then SIGTERM only if still alive. Require `ros2 node list --no-daemon` to contain no simulation nodes. Never classify smoke timeout as completion.

- [ ] **Step 3: Build and run smoke test**

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/closed_loop/log \
  build --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/closed_loop/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/closed_loop/install \
  --packages-up-to lunar_pure_exploration_sim lunar_pure_exploration_ros \
    lunar_pure_planner_ros lunar_pure_wheeled_controller
source /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/closed_loop/install/setup.bash
python3 -m pytest -q -p no:cacheprovider tests/launch/test_jazzy_300m_exploration_sim.py -k live
```

Expected: smoke test passes and cleans its graph.

- [ ] **Step 4: Commit**

Commit the updated launch test with message `test: verify Jazzy exploration closed loop`.

### Task 5: Operator entry point and full-run acceptance

**Files:**
- Create: `scripts/run_jazzy_300m_exploration_sim.sh`
- Modify: `README.md`

**Interfaces:**
- Creates a unique run directory under `/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_runs`.
- Returns success only after a valid terminal `summary.json` exists.

- [ ] **Step 1: Write the operator script**

Use `set -euo pipefail`, source Jazzy/external overlay, choose an unused `ROS_DOMAIN_ID`, set `ROS_LOCALHOST_ONLY=1`, create `run-YYYYmmdd-HHMMSS-seed-20260824`, launch RViz by default, and trap signals for bounded cleanup. After exit validate:

```python
summary["terminal_state"] == "COMPLETED"
summary["reason_code"] == "COMPLETED_NO_REACHABLE_FRONTIER"
summary["coverage_ratio"] == summary["status_coverage_ratio"]
```

- [ ] **Step 2: Update README**

Document the external Jazzy build, script command, RViz signals, result paths, movement primitives, 20× time semantics, and the Jazzy-versus-Humble/Orin readiness boundary.

- [ ] **Step 3: Run full verification and exploration**

Run package tests, static/live tests, `git diff --check`, and UTF-8 reads. Then run the operator script until terminal status. An operator stop or wall guard is incomplete evidence, not success.

- [ ] **Step 4: Verify artifacts**

Require nonempty JSON and two CSV files; equal final coverage across status/CSV/summary; positive distance; at least one completed goal; at least one global and local planner call; and terminal `COMPLETED_NO_REACHABLE_FRONTIER`. Record run directory and wall duration.

- [ ] **Step 5: Commit docs and script**

```bash
git add scripts/run_jazzy_300m_exploration_sim.sh README.md
git commit -m "docs: add 300m Jazzy exploration workflow"
```
