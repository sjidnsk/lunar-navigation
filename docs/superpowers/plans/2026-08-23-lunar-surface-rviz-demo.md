# Lunar Surface RViz Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, isolated ROS 2 Jazzy RViz demonstration that plans a wheel route across a synthetic 1 km by 1 km lunar surface.

**Architecture:** A test-only C++ scenario publisher emits the existing planner's global map, local GridMap, odometry, and identity TF under `/lunar_demo`. A second test-only C++ visualizer turns the planner's existing `MotionReference.path_preview` into standard RViz `Path` and height-coloured markers. A launch file composes both with the planner and existing RViz goal bridge, while an RViz config exposes `/lunar_demo/rviz_goal` as `2D Goal Pose`.

**Tech Stack:** ROS 2 Jazzy, rclcpp, nav_msgs, grid_map_msgs, tf2_msgs, visualization_msgs, lunar_planning_msgs, C++20, ament_cmake_gtest, pytest launch checks, RViz2.

**Spec:** `docs/superpowers/specs/2026-08-23-lunar-surface-rviz-demo-design.md`

## Global Constraints

- Keep all demo endpoints under `/lunar_demo/*`; never publish or subscribe to `/Car/T4/plan_motion`.
- Use `wheel` and the existing planner/action implementation without modifying planning algorithms, feasibility checks, or production interfaces.
- Generate a deterministic physical 1 km by 1 km scene with 1000 by 1000 cells at 1 m resolution, origin `(-500, -500)`, finite elevation, nonzero obstacle density, safe start, and a reachable default goal.
- Build the ROS executable without test targets on Jazzy using `-DBUILD_TESTING=OFF`; run new focused tests in a separate test-enabled build once the pre-existing Hopper include failure is repaired.
- This is an RViz/ROS input-and-path demonstration, not a dynamics, controller, Orin, or production-readiness certification.
- The workspace has no Git root; record verification output in the final handoff instead of making commits.

---

## File Structure

- Create `ros2_ws/src/lunar_pure_planner_ros/include/lunar_pure_planner_ros/lunar_surface_scenario.hpp`: pure, ROS-independent scenario geometry and deterministic generation API.
- Create `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_scenario.cpp`: moon terrain, obstacles, free-cell BFS, start/default-goal selection.
- Create `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_demo_node.cpp`: periodic scenario input publishers and initial default-goal publisher.
- Create `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_visualizer_node.cpp`: `MotionReference` to `Path` conversion plus terrain marker publication.
- Create `ros2_ws/src/lunar_pure_planner_ros/test/lunar_surface_scenario_test.cpp`: deterministic geometry/connectivity tests.
- Modify `ros2_ws/src/lunar_pure_planner_ros/CMakeLists.txt`: compile/install the two nodes and register the scenario GTest.
- Modify `ros2_ws/src/lunar_pure_planner_ros/package.xml`: add `visualization_msgs` dependency.
- Create `launch/lunar_surface_rviz_demo.launch.py`: isolated demo composition and planner parameter overrides.
- Create `rviz/lunar_surface_demo.rviz`: fixed-frame RViz displays and `2D Goal Pose` publisher.
- Create `tests/launch/test_lunar_surface_demo_contract.py`: source/launch contract tests that do not require an interactive RViz window.
- Modify `README.md`: Jazzy build, launch, RViz interaction, success signals, and the explicit test-only boundary.

### Task 1: Deterministic lunar surface domain model

**Files:**
- Create: `ros2_ws/src/lunar_pure_planner_ros/include/lunar_pure_planner_ros/lunar_surface_scenario.hpp`
- Create: `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_scenario.cpp`
- Create: `ros2_ws/src/lunar_pure_planner_ros/test/lunar_surface_scenario_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/CMakeLists.txt`

**Interfaces:**
- Produces `lunar::pure_planner_ros::LunarSurfaceScenario` with `width`, `height`, `resolution_m`, `origin_x_m`, `origin_y_m`, `occupancy`, `elevation_m`, `start_cell`, and `default_goal_cell`.
- Produces `BuildLunarSurfaceScenario(std::uint32_t seed) -> LunarSurfaceScenario` and `CellsConnected(const LunarSurfaceScenario&, GridCell, GridCell) -> bool`.
- Consumes only standard C++ types; ROS message construction remains outside this task.

- [ ] **Step 1: Write the failing GTest for deterministic geometry and safety**

```cpp
TEST(LunarSurfaceScenario, HasFixedGeometryAndReachableDefaultGoal) {
  const auto scenario = BuildLunarSurfaceScenario(20260823U);
  EXPECT_EQ(scenario.width, 100U);
  EXPECT_EQ(scenario.height, 100U);
  EXPECT_DOUBLE_EQ(scenario.resolution_m, 1.0);
  EXPECT_DOUBLE_EQ(scenario.origin_x_m, -50.0);
  EXPECT_DOUBLE_EQ(scenario.origin_y_m, -50.0);
  EXPECT_FALSE(scenario.Occupied(scenario.start_cell));
  EXPECT_FALSE(scenario.Occupied(scenario.default_goal_cell));
  EXPECT_TRUE(CellsConnected(scenario, scenario.start_cell,
                             scenario.default_goal_cell));
}
```

- [ ] **Step 2: Run the new test and verify the expected compile failure**

Run: `source /opt/ros/jazzy/setup.bash && colcon test --packages-select lunar_pure_planner_ros --ctest-args -R lunar_surface_scenario_test`

Expected: FAIL because `lunar_surface_scenario.hpp` and its symbols do not yet exist.

- [ ] **Step 3: Add the minimal scenario API and generator**

```cpp
struct GridCell { std::size_t x; std::size_t y; };
struct LunarSurfaceScenario {
  std::size_t width{100U}, height{100U};
  double resolution_m{1.0}, origin_x_m{-50.0}, origin_y_m{-50.0};
  std::vector<std::int8_t> occupancy;
  std::vector<float> elevation_m;
  GridCell start_cell{10U, 10U};
  GridCell default_goal_cell{};
  [[nodiscard]] bool Occupied(GridCell cell) const;
};
LunarSurfaceScenario BuildLunarSurfaceScenario(std::uint32_t seed);
bool CellsConnected(const LunarSurfaceScenario&, GridCell, GridCell);
```

Use `std::mt19937(seed)` for rocks/crater placement, calculate each elevation as a low gradient plus sine ripples and radial crater depressions, clear a 3 m start disk, and select the farthest free cell reachable by four-connected BFS as `default_goal_cell`.

- [ ] **Step 4: Register and run the GTest**

Add `lunar_surface_scenario.cpp` to the existing ROS library and:

```cmake
ament_add_gtest(lunar_surface_scenario_test test/lunar_surface_scenario_test.cpp)
target_link_libraries(lunar_surface_scenario_test ${PROJECT_NAME})
```

Run: `source /opt/ros/jazzy/setup.bash && colcon test --packages-select lunar_pure_planner_ros --ctest-args -R lunar_surface_scenario_test && colcon test-result --all --verbose`

Expected: the new GTest passes; report unrelated pre-existing test failures separately.

- [ ] **Step 5: Checkpoint**

Run: `git rev-parse --show-toplevel || true`

Expected: no Git root; preserve the working-tree changes and record the tested command/output in the handoff.

### Task 2: Scenario publisher and RViz-compatible visualizer

**Files:**
- Create: `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_demo_node.cpp`
- Create: `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_visualizer_node.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/package.xml`

**Interfaces:**
- Consumes `BuildLunarSurfaceScenario(seed)` from Task 1.
- Produces `/lunar_demo/global_overview` (`nav_msgs/msg/OccupancyGrid`), `/lunar_demo/grid_map` (`grid_map_msgs/msg/GridMap`), `/lunar_demo/odometry`, `/tf`, `/lunar_demo/default_goal`, `/lunar_demo/terrain_markers`, and `/lunar_demo/path`.
- Consumes `/lunar_demo/wheeled_reference` (`lunar_planning_msgs/msg/MotionReference`) and uses its `path_preview` unchanged.

- [ ] **Step 1: Write a failing node-level GTest for message conversion**

```cpp
TEST(LunarSurfaceDemoNode, ScenarioMapsCarryMatchingGeometryAndLayers) {
  const auto scenario = BuildLunarSurfaceScenario(20260823U);
  const auto global = MakeGlobalOverview(scenario, rclcpp::Time{1, 0});
  const auto local = MakeLocalGridMap(scenario, rclcpp::Time{1, 0});
  EXPECT_EQ(global.info.width, 100U);
  EXPECT_EQ(global.info.height, 100U);
  EXPECT_EQ(global.data.size(), 10000U);
  EXPECT_EQ(local.layers, (std::vector<std::string>{"occupancy", "elevation"}));
  EXPECT_EQ(local.data.size(), 2U);
}
```

- [ ] **Step 2: Run it and verify failure because conversion helpers are absent**

Run: `source /opt/ros/jazzy/setup.bash && colcon test --packages-select lunar_pure_planner_ros --ctest-args -R lunar_surface_demo_node_test`

Expected: FAIL due to missing `MakeGlobalOverview` and `MakeLocalGridMap`.

- [ ] **Step 3: Implement message helpers and publishers**

Implement helpers in a small `lunar_surface_demo_node.cpp` internal namespace. Publish identical, current-stamped scenario inputs at 2 Hz with best-effort QoS; populate `GridMap.info.resolution`, `length_x`, `length_y`, `pose.position`, `layers`, `basic_layers`, and two `Float32MultiArray` payloads. Publish a stable `map -> odom` identity TF and an `odom -> base_link` pose at the scenario start. Publish the default goal once after subscribers have had 1 second to connect.

- [ ] **Step 4: Implement standard RViz outputs**

Subscribe to `/lunar_demo/wheeled_reference`; publish its `path_preview` directly as `/lunar_demo/path`. Create one `visualization_msgs::msg::Marker::CUBE_LIST` named `lunar_surface` with sampled cell centres and RGB values normalized from the finite elevation range, and a second marker for obstacle cells. Publish both markers latched with transient-local QoS.

- [ ] **Step 5: Wire build and dependency declarations**

Add `find_package(visualization_msgs REQUIRED)`, `visualization_msgs` to `ament_target_dependencies`, the two executable targets, and:

```xml
<depend>visualization_msgs</depend>
```

Install both targets to `lib/${PROJECT_NAME}`.

- [ ] **Step 6: Build without test targets on Jazzy**

Run: `source /opt/ros/jazzy/setup.bash && colcon build --packages-select lunar_pure_planner_ros --cmake-clean-cache --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF`

Expected: `lunar_surface_demo_node` and `lunar_surface_visualizer_node` install successfully alongside `lunar_pure_planner_node`.

### Task 3: Isolated launch composition and RViz display

**Files:**
- Create: `launch/lunar_surface_rviz_demo.launch.py`
- Create: `rviz/lunar_surface_demo.rviz`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/CMakeLists.txt`

**Interfaces:**
- Starts the executables from Task 2 plus existing `lunar_pure_planner_node` and `lunar_rviz_goal_bridge`.
- Planner endpoint parameters are `/lunar_demo/global_overview`, `/lunar_demo/grid_map`, `/lunar_demo/odometry`, `/tf`, `/lunar_demo/plan_motion`, `/lunar_demo/diagnostics`, and `/lunar_demo/wheeled_reference`.
- Bridge endpoint parameters are `goal_topic=/lunar_demo/rviz_goal` and `action_name=/lunar_demo/plan_motion`.

- [ ] **Step 1: Write a failing Python contract test for isolated endpoints**

```python
def test_demo_launch_never_mentions_production_plan_motion() -> None:
    launch_text = LAUNCH_FILE.read_text(encoding="utf-8")
    assert "/lunar_demo/plan_motion" in launch_text
    assert '"/Car/T4/plan_motion"' not in launch_text
```

- [ ] **Step 2: Run it and verify failure because the launch file is absent**

Run: `python3 -m pytest -q -p no:cacheprovider tests/launch/test_lunar_surface_demo_contract.py`

Expected: FAIL with missing launch file.

- [ ] **Step 3: Write the composed launch file**

Use `DeclareLaunchArgument("seed", default_value="20260823")` and `DeclareLaunchArgument("start_rviz", default_value="true")`. Start the scenario and visualizer nodes, then include `pure_planner.launch.py` with its endpoint parameter dictionary overridden to the exact `/lunar_demo/*` names. Start the existing RViz bridge with its demo goal and Action names. Start `rviz2` only when `start_rviz` is true, with `-d` set to the installed RViz config.

- [ ] **Step 4: Add the RViz config and install launch/assets**

Set fixed frame `map`. Add displays: `Map` on `/lunar_demo/global_overview`, `MarkerArray` on `/lunar_demo/terrain_markers`, `Path` on `/lunar_demo/path`, `Odometry` on `/lunar_demo/odometry`, and `Pose` on `/lunar_demo/default_goal`. Configure the `2D Goal Pose` tool to publish `/lunar_demo/rviz_goal`. Install the root `launch/` and new `rviz/` directories through the ROS package CMake install rule.

- [ ] **Step 5: Run source contract tests**

Run: `python3 -m pytest -q -p no:cacheprovider tests/launch/test_lunar_surface_demo_contract.py`

Expected: PASS, proving the launch uses only demo Action endpoints and exposes the required RViz topics.

### Task 4: Live ROS action smoke test and operator documentation

**Files:**
- Create: `tests/launch/test_lunar_surface_demo_contract.py`
- Modify: `README.md`

**Interfaces:**
- Consumes the installed launch file from Task 3 and `/lunar_demo/plan_motion` from the launched planner.
- Validates that map inputs arrive before an Action request and that a success result includes `has_reference=true` and nonempty `reference.path_preview.poses`.

- [ ] **Step 1: Extend the failing launch test with the required success criterion**

```python
assert result["planning_outcome"] == 0
assert result["has_reference"] is True
assert len(result["reference"]["path_preview"]["poses"]) > 1
```

The test must wait for `/lunar_demo/global_overview`, `/lunar_demo/grid_map`, `/lunar_demo/odometry`, and `/lunar_demo/plan_motion` before invoking `ros2 action send_goal --feedback` with a point goal at the published default-goal coordinates.

- [ ] **Step 2: Run the smoke test and verify it fails before its fixture/launcher exists**

Run: `source /opt/ros/jazzy/setup.bash && source install/setup.bash && python3 -m pytest -q -p no:cacheprovider tests/launch/test_lunar_surface_demo_contract.py -k action`

Expected: FAIL until the test launches the demo and obtains the Action result.

- [ ] **Step 3: Implement bounded ROS-process setup and cleanup**

Launch with `start_rviz:=false` and an isolated `ROS_DOMAIN_ID`; wait at most 20 seconds for the four input topics and Action server. Always terminate the launch process in `finally`, then use bounded `ros2 node list --no-daemon` checks to ensure no demo nodes remain. Decode the Action response and assert the exact business result fields above; do not treat an accepted Action alone as success.

- [ ] **Step 4: Add README operator instructions**

Document this exact sequence:

```bash
cd /home/kai/CodexDownloads/lunar_navigation/lunar_pure_planner_orin
source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to lunar_pure_planner_ros \\
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF
source install/setup.bash
ros2 launch lunar_pure_planner_ros lunar_surface_rviz_demo.launch.py
```

State that the RViz success signal is a new `/lunar_demo/path` after `2D Goal Pose`; a rejected/no-reference response for a blocked location is expected safe behavior. State explicitly that this demo must not be used to drive a vehicle.

- [ ] **Step 5: Run final verification**

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to lunar_pure_planner_ros \\
  --cmake-clean-cache \\
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF
source install/setup.bash
python3 -m pytest -q -p no:cacheprovider tests/launch/test_lunar_surface_demo_contract.py
timeout 30 ros2 launch lunar_pure_planner_ros lunar_surface_rviz_demo.launch.py start_rviz:=false
```

Expected: production executables and both demo executables build; source contract tests pass; bounded launch reaches published demo inputs and Action availability. If the timeout ends the launch, label the visual check as not run rather than a failure or success.

- [ ] **Step 6: Checkpoint**

Run: `git status --short 2>&1 || true`

Expected: report the non-Git status and every changed file; do not create a commit or publish anything.
