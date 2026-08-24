# Request-Scoped SE(2) Wheel Lattice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the wheeled local planner start a bounded canonical search from any physically feasible continuous translation and yaw without coupling state-lattice phase to the map raster.

**Architecture:** First anchor translation quantization to the exact request start and prove the existing half-cell failure is removed. Then promote that origin into an immutable request-scoped `PlanningLatticeFrame` that rigidly translates and rotates state keys while leaving world-space terrain and motion validation unchanged. The exact start remains the first trajectory pose and the existing certified scaled-primitive goal connector remains the only terminal connector.

**Tech Stack:** C++20, GoogleTest, CMake/ament, ROS 2 Jazzy, colcon.

**Spec:** `docs/superpowers/specs/2026-08-24-request-scoped-se2-lattice-design.md`

## Global Constraints

- Keep `PlanMotion.action`, `WheelPlanRequest`, `WheelPlanResult`, ROS topics, TF, map and odometry contracts unchanged.
- Do not add a map phase, cell-centre or ROS-specific behavior flag.
- Do not relax `kPhysicalMatchTolerance`, footprint, obstacle, slope, clearance, curvature, dynamics, deadline or state-capacity checks.
- Do not add a synthetic start segment, full-route continuous-state search, path smoothing or controller behavior.
- Build production Jazzy packages with `-DBUILD_TESTING=OFF`; use the separate `build-wheel-regression` tree for focused GoogleTests.
- Treat the fresh baseline `GlobalRouteWithEightMeterRollingHorizonReaches750MeterGoalThroughRandomObstacles` segment-21 timeout separately: the unmodified baseline is 51/52 passing.

---

### Task 1: Anchor Translation Phase to the Exact Start

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp`

**Interfaces:**
- Consumes: `request_.start.pose.position_m`, existing wide/narrow `xy_resolution`, existing `MapSnapshot` world-space sampling.
- Produces: request-relative integer `WheelStateKey::x/y` and world-space canonical positions translated by the exact start.

- [ ] **Step 1: Add the translated-start regression test before production changes**

Add after `ReachesOffGridPoseOnFlatFreeMap`:

```cpp
TEST(WheelPlanner, PlansMultiplePrimitivesFromArbitraryTranslatedStart) {
  const TerrainFixture fixture = FlatTerrain(80U, 80U);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };
  const Pose3 start = Pose(1.13, 1.07);

  const WheelPlanResult result = PlanWheel(
      RequestTo(fixture, capability, 2.13, 1.07, 0.0, start));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_GT(result.trajectory.size(), 2U);
  EXPECT_NEAR(result.trajectory.front().pose.position_m.x, 1.13, 1.0e-12);
  EXPECT_NEAR(result.trajectory.front().pose.position_m.y, 1.07, 1.0e-12);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x, 2.13, 1.0e-9);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.y, 1.07, 1.0e-9);
}
```

This test catches any implementation that still derives canonical translation phase from `map_.origin_m()` or inserts a snapped first trajectory pose.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
source /opt/ros/jazzy/setup.bash
cmake --build build-wheel-regression/lunar_pure_planner_core \
  --target lunar_pure_planner_core_anytime_wheel_planner_test -j2
./build-wheel-regression/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test \
  --gtest_filter=WheelPlanner.PlansMultiplePrimitivesFromArbitraryTranslatedStart
```

Expected: the test compiles and fails at `ASSERT_TRUE(result.ok())` with `NO_PATH`.

- [ ] **Step 3: Implement the minimal translation-relative lattice phase**

In `WheelSearchGraph`, initialize two immutable values from the exact request start:

```cpp
const double lattice_origin_x_m_;
const double lattice_origin_y_m_;
```

Initialize them in the constructor and replace only the translation expressions:

```cpp
.x = static_cast<std::int64_t>(std::llround(
    (pose.position_m.x - lattice_origin_x_m_) / xy_resolution)),
.y = static_cast<std::int64_t>(std::llround(
    (pose.position_m.y - lattice_origin_y_m_) / xy_resolution)),
```

```cpp
const double x = lattice_origin_x_m_ +
                 static_cast<double>(key.x) * xy_resolution;
const double y = lattice_origin_y_m_ +
                 static_cast<double>(key.y) * xy_resolution;
```

Keep map lookup, yaw quantization, safety evaluation and goal connection unchanged.

Update the old `RegistersTheSpecialStartKeyWithoutCreatingASecondNode` regression to `KeepsRequestStartAsOnlyStateForSubResolutionReturn` and require `quantized_state_count == 1U`. The exact start is now canonical, so allocating a second fixed-map representative for the same bucket would violate the new design.

In the 300 m detour test, retain the behavioral requirement that the path leaves the wall's direct `y=[70,130] m` band. Use a `>30.0 m` deviation from `y=100 m`; do not require the historical `>31.0 m` route shape because every accepted edge already undergoes the full oriented-footprint sweep validation.

- [x] **Step 4: Verify GREEN and run the stable wheel regression subset**

  Fresh Task 8 evidence at `e47c8f5`: the translated-start regression passed
  in the focused 3/3 SE(2) run, and the complete wheel target passed 97/97.
  See `docs/validation/2026-08-24-local-wheel-planning-efficiency.md` for the
  separate closed-loop acceptance boundary.

Run the new test, then run all wheel tests except the known baseline timeout:

```bash
./build-wheel-regression/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test \
  --gtest_filter=-WheelPlanner.GlobalRouteWithEightMeterRollingHorizonReaches750MeterGoalThroughRandomObstacles
```

Expected: new test passes and every selected test passes.

- [ ] **Step 5: Commit the translation behavior**

```bash
git add ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp
git commit -m "fix: anchor wheel lattice translation at request start"
```

### Task 2: Rotate the Full SE(2) Lattice With Start Yaw

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp`

**Interfaces:**
- Consumes: the exact validated start yaw, world poses and existing wide/narrow yaw-bin counts.
- Produces: private `PlanningLatticeFrame::ToLocalPosition()`, `ToWorldPosition()`, `ToLocalYaw()` and `ToWorldYaw()` transformations used by `Quantize()` and `CanonicalPose()`.

- [ ] **Step 1: Add arbitrary-yaw and rigid-transform tests before yaw production changes**

Add these two tests next to the translated-start test:

```cpp
TEST(WheelPlanner, PlansMultiplePrimitivesFromArbitrarySE2Start) {
  const TerrainFixture fixture = FlatTerrain(80U, 80U);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };
  constexpr double kStartYaw = 0.37;
  const Pose3 start = Pose(2.13, 2.17, kStartYaw);
  const double goal_x = 2.13 + std::cos(kStartYaw);
  const double goal_y = 2.17 + std::sin(kStartYaw);

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, goal_x, goal_y, kStartYaw, start));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_GT(result.trajectory.size(), 2U);
  EXPECT_NEAR(result.trajectory.front().pose.position_m.x, 2.13, 1.0e-12);
  EXPECT_NEAR(result.trajectory.front().pose.position_m.y, 2.17, 1.0e-12);
  EXPECT_NEAR(TrajectoryYaw(result.trajectory.front()), kStartYaw, 1.0e-12);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x, goal_x, 1.0e-9);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.y, goal_y, 1.0e-9);
  EXPECT_NEAR(TrajectoryYaw(result.trajectory.back()), kStartYaw, 1.0e-9);
}

TEST(WheelPlanner, RigidTransformPreservesRequestRelativeTrajectory) {
  const TerrainFixture fixture = FlatTerrain(100U, 100U);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };
  const WheelPlanResult baseline = PlanWheel(RequestTo(
      fixture, capability, 3.0, 2.0, 0.0, Pose(2.0, 2.0)));
  constexpr double kYaw = 0.63;
  const double transformed_goal_x = 4.17 + std::cos(kYaw);
  const double transformed_goal_y = 3.11 + std::sin(kYaw);
  const WheelPlanResult transformed = PlanWheel(RequestTo(
      fixture, capability, transformed_goal_x, transformed_goal_y, kYaw,
      Pose(4.17, 3.11, kYaw)));

  ASSERT_TRUE(baseline.ok()) << baseline.reason_code;
  ASSERT_TRUE(transformed.ok()) << transformed.reason_code;
  ASSERT_EQ(transformed.trajectory.size(), baseline.trajectory.size());
  EXPECT_NEAR(transformed.cost, baseline.cost, 1.0e-9);
  for (std::size_t index = 0U; index < baseline.trajectory.size(); ++index) {
    const double world_dx =
        transformed.trajectory[index].pose.position_m.x - 4.17;
    const double world_dy =
        transformed.trajectory[index].pose.position_m.y - 3.11;
    const double local_x = std::cos(kYaw) * world_dx +
                           std::sin(kYaw) * world_dy;
    const double local_y = -std::sin(kYaw) * world_dx +
                           std::cos(kYaw) * world_dy;
    EXPECT_NEAR(local_x,
                baseline.trajectory[index].pose.position_m.x - 2.0,
                1.0e-8);
    EXPECT_NEAR(local_y,
                baseline.trajectory[index].pose.position_m.y - 2.0,
                1.0e-8);
    EXPECT_NEAR(ShortestYawDelta(
                    kYaw, TrajectoryYaw(transformed.trajectory[index])),
                TrajectoryYaw(baseline.trajectory[index]), 1.0e-8);
  }
}
```

The first catches a lattice that translates but does not rotate. The second catches inconsistent forward/inverse transforms without computing expected values using production helpers.

- [ ] **Step 2: Run both new tests and verify RED**

Run:

```bash
cmake --build build-wheel-regression/lunar_pure_planner_core \
  --target lunar_pure_planner_core_anytime_wheel_planner_test -j2
./build-wheel-regression/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test \
  --gtest_filter='WheelPlanner.PlansMultiplePrimitivesFromArbitrarySE2Start:WheelPlanner.RigidTransformPreservesRequestRelativeTrajectory'
```

Expected: both tests fail with `NO_PATH` for the rotated request while the baseline request remains solved.

- [ ] **Step 3: Replace raw translation fields with `PlanningLatticeFrame`**

Add a private file-local value type before `WheelSearchGraph`:

```cpp
struct PlanningLatticeFrame final {
  Vec2 origin_world_m;
  double yaw_world_rad{};
  double cosine{1.0};
  double sine{};

  explicit PlanningLatticeFrame(const Pose3& start)
      : origin_world_m{.x = start.position_m.x, .y = start.position_m.y},
        yaw_world_rad(
            YawFromQuaternion(start.orientation).value_or(0.0)),
        cosine(std::cos(yaw_world_rad)),
        sine(std::sin(yaw_world_rad)) {}

  [[nodiscard]] Vec2 ToLocalPosition(const Vec2 world) const noexcept {
    const double dx = world.x - origin_world_m.x;
    const double dy = world.y - origin_world_m.y;
    return {.x = cosine * dx + sine * dy,
            .y = -sine * dx + cosine * dy};
  }

  [[nodiscard]] Vec2 ToWorldPosition(const Vec2 local) const noexcept {
    return {.x = origin_world_m.x + cosine * local.x - sine * local.y,
            .y = origin_world_m.y + sine * local.x + cosine * local.y};
  }

  [[nodiscard]] double ToLocalYaw(const double world_yaw) const noexcept {
    return NormalizeYaw(world_yaw - yaw_world_rad);
  }

  [[nodiscard]] double ToWorldYaw(const double local_yaw) const noexcept {
    return NormalizeYaw(yaw_world_rad + local_yaw);
  }
};
```

Construct `lattice_frame_` from `request.start.pose`. In `Quantize()`, transform world position and yaw before rounding. In `CanonicalPose()`, construct local position and yaw from integer keys, transform them to world, then sample elevation and run `IsNarrow()` in world coordinates.

Do not change primitive application, `PrimitiveReachesTarget()`, edge evaluation or goal connectors.

- [x] **Step 4: Verify GREEN and run all stable wheel tests**

  Fresh Task 8 evidence at `e47c8f5`: the arbitrary-SE(2), rigid-transform
  and translated-start regressions passed 3/3; the complete wheel target
  passed 97/97.

Run both new tests, the translated-start test and then every wheel test except the known baseline timeout. After adding the three regressions this stable subset contains 54 tests.

Expected: all selected tests pass with no warnings or errors.

- [x] **Step 5: Run the full wheel target and classify any difference from baseline**

  The fresh Jazzy full wheel target passed 97/97, including the current 750 m
  regression. This unit result does not promote the failed same-seed smoke or
  the unrun full RViz/300 m acceptance.

Run the complete 55-test target. Expected baseline boundary: only the same 750m segment-21 timeout may remain. Any other failure blocks progress and must be investigated before continuing.

- [ ] **Step 6: Commit the complete SE(2) frame**

```bash
git add ros2_ws/src/lunar_pure_planner_core/src/wheel/anytime_wheel_planner.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/anytime_wheel_planner_test.cpp
git commit -m "fix: rotate wheel lattice with request start pose"
```

### Task 3: Document and Verify the ROS Demo Contract

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: installed Jazzy packages and the approved `PlanningLatticeFrame` behavior.
- Produces: documented arbitrary-start contract and runtime evidence for the default demo request.

- [ ] **Step 1: Update documentation**

Document these exact points:

```text
- External odometry may provide any finite, in-bounds, physically feasible x/y/yaw start.
- The wheel state lattice is fixed for one local planning call at that exact start pose.
- Map origin remains a terrain-sampling property and is not rewritten.
- No start snap or synthetic path segment is emitted.
- Exact goal arrival still uses certified scaled motion primitives.
```

- [ ] **Step 2: Run source-level demo contract tests**

Run the existing focused source contract and require all three tests to pass:

```bash
python3 -m pytest -q -p no:cacheprovider \
  tests/launch/test_lunar_surface_demo_contract.py
```

- [ ] **Step 3: Build the Jazzy production closure without tests**

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to lunar_pure_planner_ros \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF
```

Expected: every selected package finishes successfully.

- [ ] **Step 4: Verify the default ROS Action after a clean demo restart**

Launch the rebuilt demo, send the documented default goal, and require:

```text
planning_outcome: 0
has_reference: true
reason_code: SUCCESS or the project's solved-equivalent reason
path_preview.header.frame_id: map
path_preview.poses: non-empty
```

Do not stop an unrelated user process. Use a separate `ROS_DOMAIN_ID` for automated runtime verification if the user's launch is still active.

- [ ] **Step 5: Commit documentation and evidence-ready commands**

```bash
git add README.md
git commit -m "docs: describe request-scoped wheel lattice"
```

### Task 4: Final Verification and Scope Audit

**Files:**
- Verify only: all files changed since `23e55a3`.

**Interfaces:**
- Consumes: Tasks 1-3 commits.
- Produces: final evidence and an explicit remaining-readiness boundary.

- [ ] **Step 1: Inspect the complete diff**

Run:

```bash
git diff --check 23e55a3..HEAD
git diff --stat 23e55a3..HEAD
git status --short
```

Require no whitespace errors and no unrelated files.

- [ ] **Step 2: Re-run fresh verification**

Re-run the stable wheel test subset, full wheel target, source contract tests, Jazzy production build and ROS Action verification. Do not rely on prior task output for completion claims.

- [ ] **Step 3: Report exact readiness boundary**

Report separately:

```text
- arbitrary-start unit behavior;
- stable wheel regression count;
- known 750m baseline timeout status;
- Jazzy build status;
- ROS default Action business result;
- RViz visualization status;
- untested field/Orin boundaries.
```
