# Rolling Reliability, Cache, and Route Layers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Execute test-first and preserve the two-repository ownership boundary.

**Goal:** Make hopper goals repeatable with monotonic simulated fuel, make wheel/legged rolling execution continuous at 1x, reuse safe projections without changing search completeness, and show provisional global versus certified local routes plus authoritative timings in RViz.

**Architecture:** Keep core planning and ROS schemas unchanged at their existing entrypoints. Fix terminal execution semantics and timed interpolation in the external simulator, add a one-entry immutable projection cache and an optional non-authoritative planner observer in the core, then publish the observer output on a separate transient-local marker topic. Combine these changes with the wheel ranked-search 2-second plan in one Release qualification install.

**Tech Stack:** C++20, ROS 2 Humble, `rclcpp`, Python 3.10, Qt5/RViz2, GoogleTest, pytest, colcon Release.

## Global constraints

- Source `/opt/ros/humble/setup.bash` and verify `ROS_DISTRO=humble` before ROS builds/tests.
- Use `/home/kai/CodexDownloads/lunar_navigation/wheel-local-2s-rviz-timing` for build/install/log/evidence; do not write artifacts into either repository.
- Preserve `Planner::Plan(const PlannerInput&)`, all ROS message/action schemas, platform capability values, complete-search semantics and cancellation.
- A cache hit must be output-equivalent to a cold call. No timeout, expansion cap, candidate cap or Open cap may be added.
- Main changes merge only to `integration`; external changes merge only to its `main`, after one combined-source qualification.

---

### Task 1: Make terminal hopper cleanup silent and map generations independent

**Files:**
- Modify: external `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/rolling_execution.py`
- Modify: external `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_node.py`
- Test: external `ros2_ws/src/lunar_isaac_validation/test/test_rolling_execution.py`
- Test: external `ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py`

- [ ] Add RED tests proving `COMPLETED.cancel()` is silent/idempotent and an active hopper cancel still emits exactly one `CANCELED`.
- [ ] Add a RED controller test: finish hopper settle, submit a second Goal, assert no old-plan CANCELED is published and a second Action is sent from the landed pose.
- [ ] Change `cancel()` to mutate/publish only for active states; make `_clear_rolling_locked()` rely on that contract.
- [ ] Validate global and local hopper generations independently as positive integers; accept `(7, 8)` and reject zero/bool/non-integer.
- [ ] Run the two test modules and commit the external changes.

---

### Task 2: Commit hopper fuel once after canonical stable landing

**Files:**
- Modify: external `.../lunar_isaac_validation/rolling_execution.py`
- Modify: external `.../lunar_isaac_validation/interactive_bridge_node.py`
- Modify: external `.../lunar_isaac_validation/interactive_session.py`
- Modify: external `.../lunar_isaac_validation/interactive_node.py`
- Test: external `.../test/test_rolling_execution.py`
- Test: external `.../test/test_interactive_bridge_node.py`
- Test: external `.../test/test_interactive_session.py`
- Test: external `.../test/test_interactive_node.py`

- [ ] Add RED tests for a one-shot `hopper_fuel_commit_kg` update at the transition to COMPLETED, no duplicate on later ticks, and no commit before canonical landing/settle.
- [ ] Add RED bridge parameter tests for `0.2 -> 0.189`, derived total mass `19.989`, idempotent equal value, and rejection of increases/negative/non-finite/wrong platform/mixed atomic updates.
- [ ] Extend the transport and supervisor with `commit_hopper_fuel(remaining_kg, timeout_s)` using the existing atomic parameter service.
- [ ] On a successful one-shot commit, clear the rolling hopper commitment and keep the active processes; on failure publish `HOPPER_FUEL_COMMIT_FAILED` and keep the commitment.
- [ ] Verify the next reference is planned against the decremented propellant message and commit external changes.

---

### Task 3: Preserve trajectory time and use shortest-arc SLERP

**Files:**
- Modify: external `.../lunar_isaac_validation/rolling_execution.py`
- Test: external `.../test/test_rolling_execution.py`

- [ ] Add RED tests with uneven times `(0, 0.1, 2.0)`, a `179 deg -> -179 deg` yaw pair, zero-time fallback and invalid duplicate/non-monotonic timestamps.
- [ ] Add `sample_times_s` to `GroundReference`; retain ROS `time_from_start` or derive it from cumulative arc length only for all-zero input.
- [ ] Interpolate by elapsed time via binary search; linearly interpolate position/velocity and use normalized shortest-arc SLERP for orientation.
- [ ] Compute exact remaining arc from the interpolated pose; run all rolling execution tests and commit.

---

### Task 4: Add prefetch and a validated ground-reference double buffer

**Files:**
- Modify: external `.../lunar_isaac_validation/rolling_execution.py`
- Modify: external `.../lunar_isaac_validation/interactive_node.py`
- Modify: external `scripts/run_interactive_rviz.sh`
- Test: external `.../test/test_rolling_execution.py`
- Test: external `.../test/test_interactive_node.py`
- Test: external launcher/CLI tests in `.../test/test_interactive_node.py`

- [ ] Add RED tests for 3-second lead-time prefetch, `NEXT_READY`, no pose reset when a result arrives, exact position/velocity/yaw continuity at a valid splice, and safe replan when no splice exists.
- [ ] Store the immutable request start pose; convert a result using that pose rather than the moving acceptance pose.
- [ ] Keep active and pending references separately. Trim only an already-certified prefix whose forward intersection is within `0.05 m`, `5 deg`, and `0.10 m/s`; otherwise discard and replan at the endpoint.
- [ ] Continue active execution if prefetch planning is infeasible; do not emit execution FAILED for a still-valid active segment.
- [ ] Default visual execution to scale `1.0` and lead time `3.0 s`; retain fast-forward as explicit opt-in. Run wheel and legged controller tests and commit.

---

### Task 5: Add local-window hysteresis and encoded-message reuse

**Files:**
- Modify: external `.../lunar_isaac_validation/interactive_bridge_node.py`
- Test: external `.../test/test_interactive_bridge_node.py`

- [ ] Add RED tests showing sub-threshold 0.2 m motion does not recenter/re-encode, crossing `max(2*resolution, 0.25*half_extent)` recenters exactly once, and the platform remains inside the new window.
- [ ] Replace one-cell recentering with a center keep zone. Reuse encoded GridMap/hazard content until the center changes; update only headers on publication.
- [ ] Preserve content generation identity and run bridge tests; commit external changes.

---

### Task 6: Cache immutable global and wheel-local safe projections

**Files:**
- Modify/Add: main `ros2_ws/src/lunar_planner_core/src/shared/projection_cache.{hpp,cpp}`
- Modify: main `ros2_ws/src/lunar_planner_core/src/hierarchical/global_route_planner.{hpp,cpp}`
- Modify: main `ros2_ws/src/lunar_planner_core/src/wheel/wheel_planner.{hpp,cpp}`
- Modify: main `ros2_ws/src/lunar_planner_core/src/planner.cpp`
- Modify: main `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp`
- Modify: main `ros2_ws/src/lunar_planner_core/CMakeLists.txt`
- Test: main `ros2_ws/src/lunar_planner_core/test/hierarchical_planner_test.cpp`
- Test: main `ros2_ws/src/lunar_planner_core/test/wheel_planner_test.cpp`

- [ ] Add RED tests: same complete content key yields a hit and identical output; map generation, capability, geometry or safety config change yields a miss; canceled/failed builds never populate cache.
- [ ] Factor global planning context construction from route search. Store a one-entry immutable context in `Planner::Impl` and pass it to every conditional-corridor retry.
- [ ] Reuse one local projection for all ranked wheel goals and only reuse across calls when the complete local key matches.
- [ ] Add cumulative cache-hit, local-search-run and global-replan metrics; do not change any message schema.
- [ ] Run core tests and commit main changes.

---

### Task 7: Publish provisional global routes before certified local results

**Files:**
- Modify: main `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/planner.hpp`
- Modify: main `ros2_ws/src/lunar_planner_core/src/planner.cpp`
- Add/Modify: main `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/provisional_route_marker_publisher.hpp`
- Add/Modify: main `ros2_ws/src/lunar_planner_ros/src/provisional_route_marker_publisher.cpp`
- Modify: main `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Modify: main `ros2_ws/src/lunar_planner_ros/CMakeLists.txt`
- Test: main core and ROS server/publisher tests
- Modify: external `.../config/rviz/lunar_interactive_planning.rviz`
- Modify: external `.../lunar_isaac_validation/constants.py`
- Test: external `.../test/test_rviz_config.py`
- Test: external `.../test/test_synthetic_interactive_integration.py`

- [ ] Add RED core test that the observer receives the simplified global preview before local planner completion and observer exceptions do not alter the result.
- [ ] Keep `Plan(input)` and add an observer overload/delegating implementation. Invoke it only after a new/reused ground global route is available.
- [ ] Publish owned cyan thin `provisional_global_route` markers on `/planning/provisional_route_markers`; clear them per request/lifecycle independently of certified markers.
- [ ] Add the separate RViz display and integration assertions that provisional ADD precedes certified ADD/result and styles/namespaces differ.
- [ ] Run main ROS and external RViz tests; commit both repositories.

---

### Task 8: Forward cumulative timing and expose it in the RViz panel

**Files:**
- Modify: main `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Test: main `ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp`
- Modify: external `.../lunar_isaac_validation/interactive_node.py`
- Modify: external `ros2_ws/src/lunar_isaac_rviz_plugins/include/.../lunar_planner_panel.hpp`
- Modify: external `ros2_ws/src/lunar_isaac_rviz_plugins/src/lunar_planner_panel.cpp`
- Test: external Python diagnostics and Qt panel tests

- [ ] Follow the wheel 2-second plan to expose planner total, cumulative global/local, frontiers/searches, retries, cache hits and wheel `PASS/FAIL`.
- [ ] Fail closed on missing/nonfinite/cross-generation values; show non-wheel timing without applying the wheel budget.
- [ ] Run panel and diagnostic tests; commit.

---

### Task 9: Combined Release and process regressions

- [ ] Build main + external sources into one new merge install with `-DCMAKE_BUILD_TYPE=Release`; do not reuse the active RViz install.
- [ ] Run repository boundary checks, all targeted C++/Python/Qt tests, and the wheel 1-warmup + 20-run p95 gate.
- [ ] In isolated ROS domains run: two consecutive hopper Goals with fuel decrement; wheel rolling to a distant goal; legged rolling to a distant goal; provisional-before-certified topic ordering; cancel and replacement negatives.
- [ ] Save logs, timing JSON, marker ordering and propellant evidence under the external artifact root.
- [ ] Self-review diffs, verify UTF-8 and `git diff --check`, then fast-forward main feature into `integration` and external feature into `main` only if every qualification passes. Re-run smoke tests on both merged tips.
