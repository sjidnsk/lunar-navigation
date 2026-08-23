# 300 m Simulation Domain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated test package that generates a deterministic 300 m lunar truth scene, reveals it through a 90°/10 m sensor, publishes Task 3 map interfaces, and moves a coordinated-steering vehicle at 20× simulated speed.

**Architecture:** Keep truth, visibility, map conversion, and vehicle kinematics in ROS-independent C++20 units. A single ROS node owns their shared state, consumes Twist, integrates the plant, updates observations, and publishes maps, odometry, TF, and visualization support without exposing truth to planner or explorer.

**Tech Stack:** ROS 2 Jazzy, rclcpp, nav_msgs, grid_map_msgs, tf2_msgs, geometry_msgs, visualization_msgs, C++20, ament_cmake_gtest.

**Spec:** `docs/superpowers/specs/2026-08-24-300m-jazzy-exploration-simulation-design.md`

## Global Constraints

- New package: `ros2_ws/src/lunar_pure_exploration_sim`; production packages must not depend on it.
- Global map: exactly 300×300 at 1.0 m with native `-1/0/100` values.
- Local map: 64×64 m at 0.2 m with `occupancy`, `semantic_id`, `elevation`, `roughness`; unknown is NaN.
- Sensor: exactly 10 m range and 90° FOV with obstacle occlusion.
- Plant: only `linear.x` and `angular.z`, no lateral primitive, `sim_dt=wall_dt*20`.
- Runtime artifacts remain outside the repository.

---

## File Structure

- Create `ros2_ws/src/lunar_pure_exploration_sim/{CMakeLists.txt,package.xml}`.
- Create `include/lunar_pure_exploration_sim/lunar_scene.hpp` and `src/lunar_scene.cpp` for deterministic truth.
- Create `include/lunar_pure_exploration_sim/visibility.hpp` and `src/visibility.cpp` for ray-cast observation.
- Create `include/lunar_pure_exploration_sim/coordinated_steering_plant.hpp` and matching source for the plant.
- Create `include/lunar_pure_exploration_sim/map_messages.hpp` and matching source for ROS conversion.
- Create `src/simulation_node.cpp` and `src/simulation_main.cpp` for the shared-state ROS adapter.
- Create four focused GTest files under `test/`.

### Task 1: Package and deterministic 300 m truth scene

**Files:**
- Create: `ros2_ws/src/lunar_pure_exploration_sim/CMakeLists.txt`
- Create: `ros2_ws/src/lunar_pure_exploration_sim/package.xml`
- Create: `ros2_ws/src/lunar_pure_exploration_sim/include/lunar_pure_exploration_sim/lunar_scene.hpp`
- Create: `ros2_ws/src/lunar_pure_exploration_sim/src/lunar_scene.cpp`
- Test: `ros2_ws/src/lunar_pure_exploration_sim/test/lunar_scene_test.cpp`

**Interfaces:**
- Produces: `LunarScene BuildLunarScene(std::uint32_t seed)`.
- Produces: `TruthSample LunarScene::Sample(double world_x_m, double world_y_m) const`.
- `TruthSample`: `occupied`, `semantic_id`, `elevation_m`, `roughness`.

- [ ] **Step 1: Add the package skeleton and failing scene test**

Use an `ament_cmake` C++20 library. The test asserts:

```cpp
const auto scene = BuildLunarScene(20260824U);
EXPECT_DOUBLE_EQ(scene.min_x_m(), -150.0);
EXPECT_DOUBLE_EQ(scene.max_x_m(), 150.0);
EXPECT_EQ(scene.global_width(), 300U);
EXPECT_EQ(scene.global_height(), 300U);
EXPECT_DOUBLE_EQ(scene.global_resolution_m(), 1.0);
EXPECT_EQ(scene.GlobalOccupancy().size(), 90'000U);
EXPECT_GT(std::ranges::count(scene.GlobalOccupancy(), std::int8_t{100}), 0);
EXPECT_EQ(scene.GlobalOccupancy(), BuildLunarScene(20260824U).GlobalOccupancy());
EXPECT_FALSE(scene.Sample(0.0, 0.0).occupied);
```

Declare build/runtime dependencies for `geometry_msgs`, `grid_map_msgs`, `nav_msgs`, `rclcpp`, `std_msgs`,
`tf2_msgs`, and `visualization_msgs`, plus `ament_cmake_gtest` for tests.

- [ ] **Step 2: Run the test and verify the expected compile failure**

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/log \
  build --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/install \
  --packages-select lunar_pure_exploration_sim
```

Expected: FAIL because the scene API is absent.

- [ ] **Step 3: Implement deterministic scene generation**

Store 90,000 global cells and analytic elevation/obstacle primitives so `Sample` can populate 0.2 m local cells. Generate seeded rocks, clusters, and crater rings with `std::mt19937`; clear a 4 m disk around `(0,0)`; reserve a deterministic connected cross-and-loop free backbone; calculate roughness as the maximum elevation difference at `+/-0.2 m` in X/Y. Use semantic ID `0` for terrain, `1` for rock, `2` for crater rim.

- [ ] **Step 4: Build, test, and commit**

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/log \
  build --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/install \
  --packages-select lunar_pure_exploration_sim
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/log \
  test --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/install \
  --packages-select lunar_pure_exploration_sim --ctest-args -R lunar_scene_test
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/build --all --verbose
git add ros2_ws/src/lunar_pure_exploration_sim
git commit -m "feat: add deterministic 300m lunar scene"
```

Expected: `lunar_scene_test` passes before commit.

### Task 2: Persistent FOV observation and dual-resolution maps

**Files:**
- Create: `ros2_ws/src/lunar_pure_exploration_sim/include/lunar_pure_exploration_sim/visibility.hpp`
- Create: `ros2_ws/src/lunar_pure_exploration_sim/src/visibility.cpp`
- Create: `ros2_ws/src/lunar_pure_exploration_sim/include/lunar_pure_exploration_sim/map_messages.hpp`
- Create: `ros2_ws/src/lunar_pure_exploration_sim/src/map_messages.cpp`
- Test: `ros2_ws/src/lunar_pure_exploration_sim/test/visibility_test.cpp`
- Test: `ros2_ws/src/lunar_pure_exploration_sim/test/map_messages_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_sim/CMakeLists.txt`

**Interfaces:**
- Produces: `ObservationState::Observe(const LunarScene&, Pose2, SensorModel)`.
- Produces: `nav_msgs::msg::OccupancyGrid MakeGlobalOverview(const LunarScene&, const ObservationState&, const rclcpp::Time&)`.
- Produces: `grid_map_msgs::msg::GridMap MakeLocalGridMap(const LunarScene&, const ObservationState&, Pose2, const rclcpp::Time&)`.

- [ ] **Step 1: Add failing visibility and map tests**

Use a hand-built scene with one occupied cell. Assert a cell at yaw offset `44.9°` and range `9.9 m` is observed, offsets `45.1°` or range `10.1 m` are not, the blocking occupied cell is observed, a collinear cell behind it is not, and prior observations remain known. Assert global geometry and `-1/0/100`, plus a 320×320 local window with four ordered layers, NaN outside visibility, and exact `0.0/1.0` occupancy for visible free/occupied samples.

- [ ] **Step 2: Build and confirm missing-API failures**

Run the Task 1 build command. Expected: FAIL while compiling the two new tests.

- [ ] **Step 3: Implement observation and map conversion**

Use angular step no larger than `atan(0.1/10.0)` and radial step `0.1 m`. Map samples to unique global cells, mark each traversed cell known, and stop after the first occupied sample. Keep a 90,000-cell persistent known mask. `MakeGlobalOverview` reveals truth only under that mask. `MakeLocalGridMap` centers a 320×320 window on the rover and fills four `Float32MultiArray` layers using the repository's reversed GridMap physical indexing; every layer is `quiet_NaN()` outside the current visible sector.

- [ ] **Step 4: Run focused tests and commit**

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/log \
  build --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/install \
  --packages-select lunar_pure_exploration_sim
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/log \
  test --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/install \
  --packages-select lunar_pure_exploration_sim \
  --ctest-args -R 'visibility_test|map_messages_test'
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/build --all --verbose
git add ros2_ws/src/lunar_pure_exploration_sim
git commit -m "feat: reveal dual resolution exploration maps"
```

Expected: both focused tests pass.

### Task 3: Coordinated-steering plant

**Files:**
- Create: `ros2_ws/src/lunar_pure_exploration_sim/include/lunar_pure_exploration_sim/coordinated_steering_plant.hpp`
- Create: `ros2_ws/src/lunar_pure_exploration_sim/src/coordinated_steering_plant.cpp`
- Test: `ros2_ws/src/lunar_pure_exploration_sim/test/coordinated_steering_plant_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_sim/CMakeLists.txt`

**Interfaces:**
- Produces: `PlantState StepPlant(PlantState, BodyCommand, double wall_dt_s, PlantParameters)`.
- Parameters: wheelbase `0.8175`, track width `0.67`, speed multiplier `20.0`.
- State: map pose, longitudinal/yaw velocity, front/rear display angles, simulated elapsed.

- [ ] **Step 1: Add failing kinematic tests**

Assert forward/reverse signs, left/right arc yaw signs, zero XY movement during spin, exact `2.0 s` simulated elapsed after `0.1 s` wall time, and symmetric `front=-rear` steering for an arc.

- [ ] **Step 2: Implement exact constant-twist integration**

For nonzero omega integrate the SE(2) arc analytically over `dt=wall_dt*multiplier`; otherwise integrate straight motion. Set `kappa=omega/v`, `front=atan(wheelbase*kappa/2)`, `rear=-front`. During spin preserve XY, integrate yaw, and compute four wheel marker angles tangential to circles about body center. Reject non-finite commands and clamp to `|v|<=1.5`, `|omega|<=1.0`.

- [ ] **Step 3: Build, test, and commit**

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/log \
  build --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/install \
  --packages-select lunar_pure_exploration_sim
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/log \
  test --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/install \
  --packages-select lunar_pure_exploration_sim \
  --ctest-args -R coordinated_steering_plant_test
git add ros2_ws/src/lunar_pure_exploration_sim
git commit -m "feat: add coordinated steering simulation plant"
```

Expected: plant test passes.

### Task 4: ROS simulation node

**Files:**
- Create: `ros2_ws/src/lunar_pure_exploration_sim/include/lunar_pure_exploration_sim/simulation_node.hpp`
- Create: `ros2_ws/src/lunar_pure_exploration_sim/src/simulation_node.cpp`
- Create: `ros2_ws/src/lunar_pure_exploration_sim/src/simulation_main.cpp`
- Test: `ros2_ws/src/lunar_pure_exploration_sim/test/simulation_node_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_exploration_sim/CMakeLists.txt`

**Interfaces:**
- Consumes: `/Car/T5/Car_Cmd_Vel`.
- Produces: `/Car/T3/mapping/global_overview`, `/Car/T3/mapping/grid_map`, `/Car/T3/localization/odometry`, `/tf`, `/Car/T4/simulation/sensor_fov`, `/Car/T4/simulation/vehicle_markers`, `/Car/T4/simulation/actual_path`, `/Car/T4/simulation/sim_elapsed` (`std_msgs/msg/Float64`).
- Parameters: `seed=20260824`, `speed_multiplier=20.0`, `sensor_range_m=10.0`, `sensor_fov_deg=90.0`, `update_rate_hz=20.0`.

- [ ] **Step 1: Add a failing node test**

Instantiate with test Topics, feed a forward Twist, call a testable `Tick(0.05)`, and assert moved odometry, identity `map->odom`, dynamic `odom->base_link`, newly known global cells, four local layers, `sim_elapsed=1.0`, and zero odometry `twist.linear.y`.

- [ ] **Step 2: Implement the ROS adapter**

Use reliable QoS for planner/explorer inputs and transient-local QoS for visualization. The timer measures wall time with `std::chrono::steady_clock`, caps one wall step at `0.1 s`, calls `StepPlant`, observes, and publishes current-stamped messages. Publish both required transforms in one `TFMessage`; do not publish `/clock`.

- [ ] **Step 3: Run full package verification and commit**

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/log \
  build --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/install \
  --packages-up-to lunar_pure_exploration_sim
source /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/install/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/log \
  test --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/install \
  --packages-select lunar_pure_exploration_sim --event-handlers console_direct+
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/sim/build --all --verbose
git add ros2_ws/src/lunar_pure_exploration_sim
git commit -m "feat: publish Task3 simulation interfaces"
```

Expected: all simulation package tests pass and the node installs under `lib/lunar_pure_exploration_sim`.
