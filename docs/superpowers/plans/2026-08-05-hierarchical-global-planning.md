# Hierarchical Global Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved stateless C++ hierarchical planner that returns a complete capability-aware `map` route preview plus one certified `odom` execution segment for wheeled, legged, and hopper platforms.

**Architecture:** `SnapshotBuilder` freezes a level-validated global map, an L0 local window, a map-frame goal, and one `map_from_odom` transform. `GlobalRoutePlanner` performs bounded deterministic 2-D or landing-graph search; `LocalFrontierSelector` extracts a bounded local problem; the existing platform backends certify only that local segment; `ReferenceComposer` joins the map preview and odom execution data without changing ROS message schemas. The external RViz harness produces the approved conservative pyramid and visualizes the two frames separately.

**Tech Stack:** C++20, ROS 2 Humble, `grid_map_msgs`, `lunar_planning_msgs`, ament/colcon, GoogleTest, Python 3.10, NumPy, pytest, RViz2.

## Global Constraints

- Preserve `Planner::Plan(const PlannerInput&) noexcept`, `PlanMotion.action`, and `MotionReference.msg`; this is a source migration inside the core, not a ROS schema migration.
- Rename the core input field to `PlannerInput::goal_map`; do not retain an ambiguous `goal` alias.
- Use `r_l = r0 * 2^l`, `l in {0,1,2,3,4}`, with default `r0 = 0.2 m`; local maps must remain L0.
- Select the smallest admissible level with at most `1,048,576` cells and at most `4,096` cells on either axis; reject an impossible L4 map with `GLOBAL_MAP_SCALE_UNSUPPORTED`.
- Use conservative child aggregation exactly as approved: valid AND, obstacle/forbidden OR, height/age/obstacle variance max, quality/count min, elevation mean only when all children are valid, and elevation variance as maximum child variance plus population variance.
- Bound global expanded states, open states, and cells at `1,048,576`, work memory at `256 MiB`, and preview output at `4,096` points.
- Keep `Planner::Plan` stateless. Only a complete global topology plus a certified local trajectory or first hop may produce a reference.
- `MotionReference.header` and `path_preview` use `map`; trajectory and authorized hops use `odom`.
- Hopper success may preview a complete nominal landing chain, but `maximum_authorized_hops` is clamped to one and only the first hop is certified.
- No Python A*, Nav2 planner, or other backend may be used as fallback when the C++ global layer fails. `lunar_nav2_adapter` remains only a consumer of the returned preview.
- Source `/opt/ros/humble/setup.bash` before ROS builds and confirm `ROS_DISTRO=humble`; do not reuse build artifacts from another ROS distribution.
- Preserve the user's untracked `.vscode/` directory and all unrelated worktree changes. Do not write generated maps, logs, benchmarks, or RViz evidence into this repository.
- Ubuntu measurements are development evidence only; AGX performance claims require an AGX run.

## File and Interface Map

### Main repository: public and internal C++

- Modify `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp`: replace `goal` with `goal_map`; add hierarchical diagnostic fields.
- Modify `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/motion_reference.hpp`: add `GlobalRoutePreview` and store it independently of execution data.
- Modify `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_config.hpp`: add frozen map-level, global-search, and local-frontier limits.
- Create `ros2_ws/src/lunar_planner_core/src/hierarchical/frame_transform.{hpp,cpp}`: map/odom point, pose, and goal conversion.
- Create `ros2_ws/src/lunar_planner_core/src/hierarchical/map_level.{hpp,cpp}`: expected-level calculation and received-map validation.
- Create `ros2_ws/src/lunar_planner_core/src/hierarchical/global_route.{hpp,cpp}`: immutable route/result types and preview-point simplification.
- Create `ros2_ws/src/lunar_planner_core/src/hierarchical/grid_search.{hpp,cpp}`: deterministic bounded implicit 8-neighbor search.
- Create `ros2_ws/src/lunar_planner_core/src/hierarchical/global_route_planner.{hpp,cpp}`: wheel/leg platform projection and goal-cell dispatch.
- Create `ros2_ws/src/lunar_planner_core/src/hierarchical/local_planning_problem.hpp`: the only input accepted by local backends.
- Create `ros2_ws/src/lunar_planner_core/src/hierarchical/local_frontier.{hpp,cpp}`: bounded local view, goal conversion, and deterministic backoff candidates.
- Create `ros2_ws/src/lunar_planner_core/src/hierarchical/hopper_route_planner.{hpp,cpp}`: landing-node sampling, bounded edge generation, and route search.
- Create `ros2_ws/src/lunar_planner_core/src/hierarchical/reference_composer.{hpp,cpp}`: complete-preview plus certified-segment invariant.
- Modify platform planners and hopper certifier under `src/wheel`, `src/legged`, and `src/hopper`: consume `LocalPlanningProblem`; never inspect `global_map`.
- Modify `ros2_ws/src/lunar_planner_core/src/planner.cpp`: run the four-stage hierarchy and stable failure mapping.

### Main repository: ROS, contracts, and consumers

- Modify `ros2_ws/src/lunar_planner_ros/src/snapshot_builder.cpp` and its header/tests: normalize goals into map, validate pyramid level and L0 local resolution.
- Modify `ros2_ws/src/lunar_planner_ros/src/message_conversion.cpp` and its header/tests: split preview and execution frames.
- Modify `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp` and tests: freeze new parameters and publish hierarchical diagnostic values.
- Modify `ros2_ws/src/lunar_nav2_adapter/src/lunar_global_planner.cpp` and tests: consume only non-empty map-frame `path_preview`.
- Modify `docs/interfaces/external-input-baseline.md`, `ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml`, `tools/check_external_interfaces.py`, and `tests/foundation/test_external_interface_config.py`: version and enforce the map pyramid contract.

### External RViz validation repository

- Create `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/map_pyramid.py`: level selection and exact conservative aggregation.
- Create `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/local_window.py`: platform-centered L0 extraction in odom.
- Modify `synthetic_map.py`, `planning_map.py`, `bridge_node.py`, `interactive_bridge_node.py`, `interactive_goal.py`, `rviz_evidence.py`, `interactive_node.py`, and the RViz config: publish and show distinct global/local products.
- Add pytest coverage beside the existing external package tests; generated artifacts continue to live under `~/CodexDownloads`.

---

### Task 1: Version and enforce the external multiresolution map contract

**Files:**
- Modify: `docs/interfaces/external-input-baseline.md`
- Modify: `ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml`
- Modify: `tools/check_external_interfaces.py`
- Modify: `tests/foundation/test_external_interface_config.py`

**Interfaces:**
- Consumes: approved dyadic level and aggregation rules in the design specification.
- Produces: schema `lunar-external-interfaces/v3` with `map_pyramid` and exact global/local topic semantics used by Tasks 2, 8, and 11.

- [ ] **Step 1: Write the failing v3 contract tests**

Add an exact `map_pyramid` object to `EXPECTED_DOCUMENT` and cases that reject a non-dyadic resolution, a level above four, a global level other than the smallest admissible level, and a local resolution other than `r0`:

```python
"map_pyramid": {
    "base_resolution_m": 0.2,
    "maximum_level": 4,
    "maximum_cells": 1048576,
    "maximum_axis_cells": 4096,
    "global_selection": "smallest_admissible_level",
    "local_level": 0,
    "aggregation_version": "lunar-conservative-grid-aggregation/v1",
}
```

- [ ] **Step 2: Verify the old v2 checker fails the new tests**

Run: `python3 -m pytest -q tests/foundation/test_external_interface_config.py`

Expected: FAIL because `_SCHEMA_VERSION` is v2 and `map_pyramid` is unexpected.

- [ ] **Step 3: Implement the exact v3 checker and documentation**

Set `_SCHEMA_VERSION = "lunar-external-interfaces/v3"`, add an immutable `_MAP_PYRAMID` mapping, include it in `required_top_level`, and validate it with `_validate_fixed_mapping`. Add `level_semantics: selected_dyadic_global` to `map_global` and `level_semantics: l0_platform_window` to `map_local`. Document all ten layer aggregation rules and the failure codes `GLOBAL_MAP_LEVEL_INVALID` and `GLOBAL_MAP_SCALE_UNSUPPORTED`.

- [ ] **Step 4: Run interface and repository-boundary tests**

Run:

```bash
python3 -m pytest -q tests/foundation/test_external_interface_config.py
python3 -c 'import pathlib,yaml; from tools.check_external_interfaces import validate_external_config; p=pathlib.Path("ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml"); assert validate_external_config(yaml.safe_load(p.read_text(encoding="utf-8"))) == []'
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
```

Expected: all tests pass; both checkers print no contract errors.

- [ ] **Step 5: Commit the contract atomically**

```bash
git add docs/interfaces/external-input-baseline.md ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml tools/check_external_interfaces.py tests/foundation/test_external_interface_config.py
git commit -m "feat: define multiresolution map contract"
```

### Task 2: Add core hierarchical types, frame transforms, and map-level validation

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/motion_reference.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_config.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/frame_transform.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/frame_transform.cpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/map_level.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/map_level.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/hierarchical_types_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/test_fixtures.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`

**Interfaces:**
- Consumes: `GridMap`, `RigidTransform`, `GoalRegion`, and `Pose3`.
- Produces: `PlannerInput::goal_map`; `GlobalRoutePreview{std::vector<Pose3> poses_map}`; `ExpectedGlobalMapLevel(double,double,const GlobalMapConfig&)`; `ValidateMapLevels(const WorldSnapshot&,const GlobalMapConfig&)`; `TransformGoal(const GoalRegion&,const RigidTransform&,TransformDirection)`.

- [ ] **Step 1: Add compile-time and numerical failing tests**

Cover L0-L4 boundary selection, a 4097-cell axis, L4 overflow, non-unit translation/rotation, point and polygon goal transforms, and these public defaults:

```cpp
EXPECT_DOUBLE_EQ(config.global_map.base_resolution_m, 0.2);
EXPECT_EQ(config.global_map.maximum_level, 4U);
EXPECT_EQ(config.global_map.maximum_cells, 1'048'576U);
EXPECT_EQ(config.global_map.maximum_axis_cells, 4'096U);
EXPECT_EQ(config.global_search.resources.maximum_expanded_states, 1'048'576U);
EXPECT_EQ(config.global_search.maximum_preview_points, 4'096U);
EXPECT_DOUBLE_EQ(config.local_frontier.wheel_horizon_m, 4.0);
EXPECT_DOUBLE_EQ(config.local_frontier.legged_horizon_m, 3.0);
EXPECT_EQ(config.local_frontier.maximum_attempts, 3U);
```

- [ ] **Step 2: Verify the new test target does not compile**

Run:

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log build --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --cmake-args -DBUILD_TESTING=ON
```

Expected: build fails because hierarchical types and functions are absent.

- [ ] **Step 3: Implement exact public types and pure helpers**

Use these declarations:

```cpp
struct GlobalMapConfig final {
  double base_resolution_m{0.2};
  std::size_t maximum_level{4U};
  std::size_t maximum_cells{1'048'576U};
  std::size_t maximum_axis_cells{4'096U};
};

struct GlobalSearchConfig final {
  SearchResourceLimits resources{1'048'576U, 1'048'576U, 1'048'576U,
                                 1'048'576U, 256U * 1024U * 1024U};
  std::size_t maximum_preview_points{4'096U};
  double slope_weight{1.0};
  double roughness_weight{1.0};
  double clearance_weight{1.0};
};

struct LocalFrontierConfig final {
  double wheel_horizon_m{4.0};
  double legged_horizon_m{3.0};
  std::size_t maximum_attempts{3U};
  double additional_corridor_margin_m{0.4};
};
```

Use overflow-safe `ceil(size/resolution)` arithmetic and relative tolerance `1e-9` for dyadic-level matching. Normalize quaternions before applying `map_from_odom`; reject non-finite or zero-norm transforms.

- [ ] **Step 4: Migrate all fixture and production references from `.goal` to `.goal_map`**

Run `rg -n '\.goal\b|input\.goal\b' ros2_ws/src/lunar_planner_core tests` and change only `PlannerInput` accesses; ROS action fields named `goal` remain unchanged.

- [ ] **Step 5: Build and run the focused test**

Run:

```bash
source /opt/ros/humble/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log build --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --cmake-args -DBUILD_TESTING=ON
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R hierarchical_types
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --verbose
```

Expected: build succeeds and `hierarchical_types` passes.

- [ ] **Step 6: Commit the type migration**

```bash
git add ros2_ws/src/lunar_planner_core
git commit -m "refactor: add hierarchical planner core types"
```

### Task 3: Implement bounded deterministic global grid search and route simplification

**Files:**
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/global_route.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/global_route.cpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/grid_search.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/grid_search.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/global_grid_search_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`

**Interfaces:**
- Consumes: `shared::SafeProjection`, a start cell, a stable goal mask, `GlobalSearchConfig`, and `std::stop_token`.
- Produces: `GlobalGridSearchResult SearchGlobalGrid(const GlobalGridSearchProblem&)`; `std::vector<shared::GridCell> SimplifyRouteSupercover(const shared::SafeProjection&, std::span<const shared::GridCell>, std::size_t maximum_points)`; `GlobalRoute` with raw/simplified cells, map poses, cost, and search metrics.

- [ ] **Step 1: Write failing search behavior tests**

Create fixtures for a diagonal corner trap, symmetric tie, goal tolerance mask, disconnected wall, immediate cancel, expanded/open exhaustion, and a simplification line that crosses one unsafe supercover cell. Assert stable status and codes:

```cpp
EXPECT_EQ(result.status, GlobalSearchStatus::kResourceExhausted);
EXPECT_EQ(result.reason_code, "GLOBAL_SEARCH_RESOURCE_LIMIT");
EXPECT_EQ(first.path_cells, second.path_cells);
EXPECT_EQ(first.cost, second.cost);
```

- [ ] **Step 2: Run the focused target and confirm failure**

Run: `colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R global_grid_search`

Expected: test target is unavailable or fails to compile.

- [ ] **Step 3: Implement implicit A* with stable ordering**

Use row-major state IDs, neighbor order `N, NE, E, SE, S, SW, W, NW`, prohibit a diagonal unless both adjacent orthogonal cells are hard feasible, and order the heap by `(f, h, g, state_id, insertion_serial)`. Allocate fixed-size `g`, `parent`, and state arrays after checking the configured memory budget. Check `stop_token` in every pop and neighbor loop.

The edge cost is:

```cpp
const double distance = resolution * (diagonal ? std::numbers::sqrt2 : 1.0);
const double risk = config.slope_weight * Square(projection.SlopeRadians(next)) +
                    config.roughness_weight * projection.RoughnessMeters(next) +
                    config.clearance_weight /
                        std::max<double>(projection.ClearanceMeters(next), resolution);
return distance + projection.TraversalCost(next) + distance * risk;
```

- [ ] **Step 4: Implement deterministic supercover simplification**

Walk candidate endpoints from farthest to nearest. Accept a shortcut only if every supercover cell is hard feasible and every diagonal transition satisfies the same no-corner-cut rule. If simplification exceeds `maximum_preview_points`, retain the first and last points plus evenly spaced route-index points using integer arithmetic; never omit a turn needed for hard safety.

- [ ] **Step 5: Run search, determinism, and sanitizer tests**

Run:

```bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R 'global_grid_search|shared_core_determinism'
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/asan/log build --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/asan/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/asan/install --cmake-args -DBUILD_TESTING=ON -DLUNAR_ENABLE_SANITIZERS=ON
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/asan/log test --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/asan/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/asan/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/asan/test-results --ctest-args -R global_grid_search
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/asan/test-results --verbose
```

Expected: all selected tests pass with no ASan/UBSan report.

- [ ] **Step 6: Commit the reusable global search**

```bash
git add ros2_ws/src/lunar_planner_core
git commit -m "feat: add bounded global grid search"
```

### Task 4: Build wheel and leg capability-aware global routes

**Files:**
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/global_route_planner.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/global_route_planner.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/global_route_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`

**Interfaces:**
- Consumes: `PlannerInput`, map-level validation from Task 2, and grid search from Task 3.
- Produces: `GlobalRoutePlanResult PlanGroundGlobalRoute(const PlannerInput&)` with `PlanningOutcome`, stable reason, full route, and metrics.

- [ ] **Step 1: Add failing wheel/leg route tests**

Use literal maps and start/goal coordinates to cover wheel/leg route differences, a target with no safe tolerance cell, a disconnected start/goal, a translated and rotated `map_from_odom`, and repeatability. Assert:

```cpp
EXPECT_EQ(no_goal.reason_code, "GLOBAL_GOAL_INFEASIBLE");
EXPECT_EQ(disconnected.reason_code, "GLOBAL_NO_KNOWN_SAFE_ROUTE");
ASSERT_TRUE(success.route.has_value());
EXPECT_EQ(success.route->poses_map.front().position_m, expected_start_map);
EXPECT_LE(DistanceXY(success.route->poses_map.back(), goal), tolerance);
```

- [ ] **Step 2: Confirm tests fail before the planner exists**

Run: `colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R global_route_planner`

Expected: missing target or missing symbols.

- [ ] **Step 3: Implement goal masks and platform-specific projection**

Build one `MapSnapshot` and one `SafeProjection` per request using the active platform capability. Convert current odom pose to map. For point goals, mark hard-feasible cells whose centers are within tolerance; for planar regions, use boundary-inclusive point-in-polygon plus `normal_tolerance_m`. Require start and at least one goal cell in the same connected component before calling search. Preserve user yaw only at the final pose; assign route-tangent yaw to intermediate poses.

- [ ] **Step 4: Verify both platforms and stable failures**

Run:

```bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R 'global_route_planner|wheel_fault|legged_fault'
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --verbose
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the ground global planner**

```bash
git add ros2_ws/src/lunar_planner_core
git commit -m "feat: plan capability-aware ground routes"
```

### Task 5: Select bounded local frontiers and migrate wheel/leg local backends

**Files:**
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/local_planning_problem.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/local_frontier.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/local_frontier.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/local_frontier_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_planner.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_planner.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_planner.cpp`
- Modify: wheel/leg tests and `test/test_fixtures.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`

**Interfaces:**
- Consumes: a complete `GlobalRoute`, current state, L0 local map, capability, and config.
- Produces: `LocalFrontierResult BuildLocalFrontiers(const PlannerInput&,const GlobalRoute&)`; `LocalPlanningProblem`; `WheelPlanner::Plan(const LocalPlanningProblem&)`; `LeggedPlanner::Plan(const LocalPlanningProblem&)`.

- [ ] **Step 1: Write failing local coverage and view tests**

Cover edge margin, 4 m wheel horizon, 3 m leg horizon, transformed route, no forward-covered point, corridor clipping, and exactly three farthest-to-nearest candidates. Assert that every created local-map cell lies in the route corridor and that the local backends cannot access `WorldSnapshot::global_map` at compile time.

- [ ] **Step 2: Confirm the focused tests fail**

Run: `colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R local_frontier`

Expected: new types and target are missing.

- [ ] **Step 3: Implement the bounded problem type and selector**

Define:

```cpp
struct LocalPlanningProblem final {
  std::string request_id;
  TimePoint state_time;
  PlatformState current_state;
  GoalRegion goal_odom;
  GridMap local_map_view;
  PlatformCapability capability;
  PlannerConfig config;
  std::optional<ExecutionContext> previous_execution;
  std::stop_token stop_token;
};
```

Compute margin as footprint/body support radius plus platform minimum clearance plus `0.4 m`. Transform global route points into odom, require the full prefix to remain inside valid L0 coverage, choose the farthest point within the platform horizon, then expose at most three monotonically nearer candidates. Cells outside the horizon/corridor intersection are marked invalid and forbidden in the copied request-local view.

- [ ] **Step 4: Migrate wheel and leg planners without compatibility overloads**

Replace every `input.goal_map`/`input.world.local_map` access inside the local backends with `problem.goal_odom`/`problem.local_map_view`. Preserve existing failure codes inside the local backend; the hierarchy maps exhaustion of all frontier attempts to `LOCAL_SEGMENT_INFEASIBLE`.

- [ ] **Step 5: Run local and existing platform regression tests**

Run:

```bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R 'local_frontier|wheel_planner|wheel_fault|legged_planner|legged_fault'
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --verbose
```

Expected: all selected tests pass and fixture maps remain small L0 maps.

- [ ] **Step 6: Commit the bounded local backend migration**

```bash
git add ros2_ws/src/lunar_planner_core
git commit -m "refactor: bound ground local planning problems"
```

### Task 6: Add hopper landing-graph global routing and first-hop handoff

**Files:**
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/hopper_route_planner.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/hopper_route_planner.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/hopper_route_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hopper_planner.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hopper_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hop_certifier.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hop_certifier.cpp`
- Modify: hopper tests and `ros2_ws/src/lunar_planner_core/CMakeLists.txt`

**Interfaces:**
- Consumes: map-frame global goal, hopper capability, global projection, local L0 problem, and existing landing/ballistic certifiers.
- Produces: `HopperRoutePlanResult PlanHopperGlobalRoute(const PlannerInput&)`; a complete nominal landing `GlobalRoute`; `CertifyFirstHop(const LocalPlanningProblem&,const LandingRegion&,const LandingRegion&)`.

- [ ] **Step 1: Write failing hopper topology tests**

Create literal fixtures for a three-hop route, an interrupted landing chain, resolution greater than `R_hop_max/2`, node exhaustion, out-degree exhaustion, cancel, and one route whose first local flight tube is unavailable. Assert resource and physical failures remain distinct and only one segment is authorized.

- [ ] **Step 2: Confirm tests fail before implementation**

Run: `colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R hopper_route_planner`

Expected: missing symbols or test target.

- [ ] **Step 3: Implement the conservative hop reach and graph**

Derive horizontal reach by bounded sampling over flight time and launch velocity under `gravity_mps2`, `maximum_launch_speed_mps`, impulse, and landing-speed constraints; choose the largest finite certified horizontal displacement. Reject `resolution > reach/2` as `HOPPER_GLOBAL_RESOLUTION_INSUFFICIENT`.

Insert current hold and all target-region candidates first, then sample remaining safe landing candidates in stable row-major order up to `maximum_graph_nodes`. Generate feasible ballistic-envelope edges on demand, sort each adjacency list by `(cost,to_node)`, retain `maximum_graph_out_degree`, and run bounded Dijkstra ordered by `(cost,hops,node_id)`. If truncation prevents a completeness claim, return `HOPPER_GLOBAL_ROUTE_RESOURCE_LIMIT`; return `GLOBAL_NO_KNOWN_SAFE_ROUTE` only after an untruncated search.

- [ ] **Step 4: Hand only the first nominal edge to the existing certifier**

Build a `LocalPlanningProblem` whose goal is the next landing region in odom. Change `CertifyFirstHop` to accept the explicit local problem and L0 regions. Always set one `HopSegment`; add `HOPPER_REMAINING_HOPS_PREVIEW_ONLY` when the route contains later nodes and `HOPPER_AUTHORIZATION_CLAMPED_TO_ONE` if configured above one.

- [ ] **Step 5: Run hopper and commitment regression tests**

Run:

```bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R 'hopper_route|hopper_planner|hopper_fault|hopper_commitment'
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --verbose
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit hopper hierarchy**

```bash
git add ros2_ws/src/lunar_planner_core
git commit -m "feat: route hopper through landing graph"
```

### Task 7: Compose references and integrate the stateless planner facade

**Files:**
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/reference_composer.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hierarchical/reference_composer.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/hierarchical_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/public_api_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/facade_summary.cpp`
- Modify: differential fixtures and `ros2_ws/src/lunar_planner_core/CMakeLists.txt`

**Interfaces:**
- Consumes: full global route result and one successful local backend result.
- Produces: a public `MotionReference` whose `preview.poses_map` is complete and whose `data` contains one certified local reference; facade diagnostics named `cpp_v3_hierarchical`.

- [ ] **Step 1: Add failing facade invariant tests**

Cover complete-preview success, global-only rejection, local-only rejection, frontier backoff, summed expanded states, best global cost, warning ordering, all stable error mappings, cancel during each stage, and repeated calls proving no cross-request state.

- [ ] **Step 2: Confirm old direct dispatch fails the tests**

Run: `colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R hierarchical_planner`

Expected: old facade returns local-only references and wrong planner name.

- [ ] **Step 3: Implement `ReferenceComposer` invariants**

Reject an empty global preview, a local output without reference data, a platform mismatch, a non-finite preview pose, or more than one authorized hop. Copy the complete map poses before moving local execution data. Set `plan_id`, platform, and input time from the same request.

- [ ] **Step 4: Replace direct backend dispatch with the four stages**

Validate required layers and map levels first; dispatch platform-specific global search; build local candidates; invoke the local backend in order until success or three failures; compose the reference. Use the design's exact `PlanningOutcome` and reason-code table. Set `best_cost` to global route cost and sum global plus local expanded states.

- [ ] **Step 5: Run the complete core suite**

Run:

```bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --verbose
```

Expected: zero failed tests; public-header and no-legacy checks pass.

- [ ] **Step 6: Commit the integrated core**

```bash
git add ros2_ws/src/lunar_planner_core tests/differential
git commit -m "feat: integrate hierarchical planner facade"
```

### Task 8: Normalize ROS goals to map and validate received map levels

**Files:**
- Modify: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/snapshot_builder.hpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/snapshot_builder.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/snapshot_builder_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp`

**Interfaces:**
- Consumes: Task 2 map validation and transform helpers.
- Produces: `PlannerInput::goal_map`; configure-time parameters `base_resolution_m`, `maximum_global_level`, `maximum_global_cells`, and `maximum_global_axis_cells` frozen in `PlannerConfig`.

- [ ] **Step 1: Replace the old odom-normalization test with failing map-normalization tests**

Test map goals unchanged, odom point/polygon goals transformed to map, non-unit rotation, local map not L0, non-dyadic global resolution, over-fine resource violation, over-coarse level, axis limit, and L4 scale exhaustion. Assert the exact snapshot reason codes.

- [ ] **Step 2: Run snapshot tests and confirm semantic failure**

Run: `colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_ros --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R snapshot_builder`

Expected: old builder transforms map goals into odom and fails new assertions.

- [ ] **Step 3: Implement map normalization and level checks before input construction**

For `frame_id == "map"`, validate and retain the request goal. For `frame_id == "odom"`, apply `map_from_odom`. Reject all other frames. Validate local resolution equals `r0` and global resolution equals the smallest admissible dyadic level for its represented physical extent. Return snapshot `kInvalidGlobalMap`/`kInvalidLocalMap` with the approved stable reasons.

- [ ] **Step 4: Declare, read, and freeze exact lifecycle parameters**

Declare defaults `0.2`, `4`, `1048576`, and `4096`; require finite positive base resolution and integer bounds matching the supported maximum level. Pass the configured `PlannerConfig` into `SnapshotBuilder` instead of relying on constructor defaults.

- [ ] **Step 5: Build and test the ROS package**

Run:

```bash
source /opt/ros/humble/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log build --base-paths ros2_ws/src --packages-up-to lunar_planner_ros --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --cmake-args -DBUILD_TESTING=ON
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_ros --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R 'snapshot_builder|plan_motion_server'
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --verbose
```

Expected: selected tests pass.

- [ ] **Step 6: Commit ROS input semantics**

```bash
git add ros2_ws/src/lunar_planner_ros
git commit -m "feat: freeze map-frame hierarchical snapshots"
```

### Task 9: Convert split-frame results and publish hierarchical diagnostics

**Files:**
- Modify: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/message_conversion.hpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/message_conversion.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/message_conversion_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp`

**Interfaces:**
- Consumes: `MotionReference::preview`, local execution data, and hierarchical core metrics.
- Produces: `PlannerResultContext{preview_frame="map", execution_frame="odom"}` and an unchanged ROS message with independently framed headers.

- [ ] **Step 1: Add failing split-frame conversion tests**

Assert top-level/header and preview are map, trajectory/hops are odom, preview poses come from `GlobalRoutePreview` rather than regenerated execution data, only one hop is serialized, and invalid frame combinations fail closed.

- [ ] **Step 2: Verify old single-frame conversion fails**

Run: `colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_ros --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R message_conversion`

Expected: old conversion emits all fields in odom.

- [ ] **Step 3: Implement exact split-frame conversion**

Define:

```cpp
struct PlannerResultContext final {
  TimePoint global_map_stamp;
  TimePoint local_map_stamp;
  TimePoint state_stamp;
  std::uint64_t mission_revision{};
  std::string preview_frame{"map"};
  std::string execution_frame{"odom"};
};
```

Set `reference.header` and `reference.path_preview.header` to preview frame. Pass only execution frame to trajectory/hop conversion. Populate every preview `PoseStamped` directly from `poses_map`.

- [ ] **Step 4: Publish diagnostic key/value metrics**

Extend the server diagnostic call to include level, global resolution/cells, per-stage elapsed and expanded states, open peak, estimated memory, raw/simplified points, local frontier distance/attempts/corridor width, and hopper node/edge/hop/certification counts. Stable key names are lowercase snake case prefixed with `hierarchical_`.

- [ ] **Step 5: Run ROS conversion and server tests**

Run:

```bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_ros --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R 'message_conversion|plan_motion_server'
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --verbose
```

Expected: all selected tests pass and the ROS interfaces are not regenerated.

- [ ] **Step 6: Commit result semantics**

```bash
git add ros2_ws/src/lunar_planner_ros
git commit -m "feat: expose global previews and local references"
```

### Task 10: Keep the Nav2 adapter as a strict map-preview consumer

**Files:**
- Modify: `ros2_ws/src/lunar_nav2_adapter/src/lunar_global_planner.cpp`
- Modify: `ros2_ws/src/lunar_nav2_adapter/test/lunar_global_planner_test.cpp`

**Interfaces:**
- Consumes: successful `PlanMotion` result with a map-frame `path_preview`.
- Produces: the same `nav_msgs::msg::Path`; never calls a fallback planner.

- [ ] **Step 1: Add failing consumer-boundary tests**

Add cases for an odom preview, empty preview, missing reference, and a valid far map preview. Assert invalid responses throw the adapter's existing planner exception and no alternative route is returned.

- [ ] **Step 2: Run the adapter test**

Run: `colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_nav2_adapter --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R lunar_global_planner`

Expected: odom/empty cases currently escape validation and fail.

- [ ] **Step 3: Enforce the map-preview contract**

Immediately before returning, require `has_reference`, `path_preview.header.frame_id == global_frame_`, at least two poses, and every pose header equal to the path header. Keep the existing action timeout/cancel logic unchanged.

- [ ] **Step 4: Run adapter and boundary tests**

Run:

```bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_nav2_adapter --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --verbose
```

Expected: all checks pass.

- [ ] **Step 5: Commit the strict consumer**

```bash
git add ros2_ws/src/lunar_nav2_adapter
git commit -m "fix: validate hierarchical path previews"
```

### Task 11: Produce exact conservative pyramid maps and L0 local windows externally

**Files:**
- Create: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/map_pyramid.py`
- Create: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/local_window.py`
- Create: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/ros2_ws/src/lunar_isaac_validation/test/test_map_pyramid.py`
- Create: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/ros2_ws/src/lunar_isaac_validation/test/test_local_window.py`
- Modify: external `synthetic_map.py`, `planning_map.py`, and their tests.

**Interfaces:**
- Consumes: an L0 `dict[str,np.ndarray]`, base `GridDescriptor`, platform pose, and the v3 contract.
- Produces: `select_global_level(width: int, height: int, r0: float) -> int`; `aggregate_level(layers: dict[str, np.ndarray]) -> dict[str, np.ndarray]`; `extract_local_window(layers: dict[str, np.ndarray], descriptor: GridDescriptor, center_map_xy: tuple[float, float], map_from_odom_xyyaw: tuple[float, float, float], half_extent_m: float) -> tuple[dict[str, np.ndarray], GridDescriptor]`; bundle metadata carrying `global_level` and `base_resolution_m`.

- [ ] **Step 1: Write exact aggregation and level-selection tests**

Use a 2x2 literal layer fixture and assert every output scalar, including:

```python
assert parent["valid_mask"][0, 0] == 0
assert parent["obstacle"][0, 0] == 1
assert parent["observation_quality"][0, 0] == 0.6
assert parent["observation_count"][0, 0] == 2
assert math.isnan(parent["elevation"][0, 0])
expected_variance = max(child_variances) + np.var(child_elevations)
assert parent["elevation_variance"][0, 0] == pytest.approx(expected_variance)
```

Cover odd dimensions by padding missing children as invalid/forbidden, L0-L4 transitions, axis limits, and L4 failure.

- [ ] **Step 2: Run tests and confirm modules are absent**

Run: `python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test/test_map_pyramid.py ros2_ws/src/lunar_isaac_validation/test/test_local_window.py`

Working directory: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression`

Expected: import failure.

- [ ] **Step 3: Implement level selection and aggregation without resampling shortcuts**

Use pure NumPy reductions over explicit 2x2 child groups. Preserve dtypes required by `grid_map_codec`; set invalid parent elevation to `NaN` and invalid parent `valid_mask` to zero. Return `GLOBAL_MAP_SCALE_UNSUPPORTED` after L4 rather than inventing L5.

- [ ] **Step 4: Implement platform-centered local windows**

Extract an L0 rectangle large enough for the configured horizon plus platform margin, clip it to the source map, set the descriptor origin to the first included cell center convention used by `encode_grid_map`, and transform it into odom coordinates using the frozen map/odom transform. Never copy the selected global array as local data.

- [ ] **Step 5: Update synthetic bundles and tests**

Generate L0 arrays once, select and aggregate only `global__*`, then generate distinct `wheel__*`, `legged__*`, and `hopper__*` local arrays from L0. Store aggregation version, base resolution, and selected level in the manifest identity so map hashes change when pyramid semantics change.

- [ ] **Step 6: Run the complete external Python suite**

Run: `python3 -m pytest -q`

Working directory: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression`

Expected: all tests pass.

- [ ] **Step 7: Commit in the external repository**

```bash
git add ros2_ws/src/lunar_isaac_validation
git commit -m "feat: generate hierarchical planning maps"
```

### Task 12: Move interactive validation to the global map and expose layered RViz evidence

**Files:**
- Modify: external `lunar_isaac_validation/interactive_goal.py`
- Modify: external `lunar_isaac_validation/bridge_node.py`
- Modify: external `lunar_isaac_validation/interactive_bridge_node.py`
- Modify: external `lunar_isaac_validation/rviz_evidence.py`
- Modify: external `lunar_isaac_validation/interactive_node.py`
- Modify: external `config/rviz/lunar_interactive_planning.rviz`
- Modify: external tests `test_interactive_goal.py`, `test_bridge_node.py`, `test_interactive_bridge_node.py`, `test_rviz_evidence.py`, and `test_interactive_node.py`

**Interfaces:**
- Consumes: Task 11 distinct global/local bundle and Task 9 map preview/odom execution result.
- Produces: RViz markers for full global route, current execution segment, local frontier, hopper nominal landings, certified first hop, and stable failure target; left panel diagnostics for level, resolution, cells, route length, frontier distance, stage timing, and hop counts.

- [ ] **Step 1: Add failing goal and visualization tests**

Verify a goal outside local but inside global is accepted; a global obstacle is rejected; global and local published maps have different descriptors; wheel/leg preview path requires map; execution trajectory/hops require odom; hopper preview landings are pale while the sole certified hop is saturated; an infeasible result draws no path.

- [ ] **Step 2: Run the focused tests and observe old assumptions fail**

Run:

```bash
python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_goal.py \
  ros2_ws/src/lunar_isaac_validation/test/test_bridge_node.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py \
  ros2_ws/src/lunar_isaac_validation/test/test_rviz_evidence.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py
```

Working directory: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression`

Expected: failures mention local-map goal bounds and odom-only preview validation.

- [ ] **Step 3: Validate RViz goals against `bundle.grids["global"]`**

Read `global__valid_mask/elevation/obstacle/forbidden`, retain `goal.header.frame_id = "map"`, and change the outside-map reason to `GOAL_OUTSIDE_GLOBAL_MAP`. Platform choice continues to affect tolerance and Action target capability, not target bounds.

- [ ] **Step 4: Publish and render separate layers**

Keep `/environment/map_global` in map and `/environment/map_local` in odom. Publish visualization topics with stable names:

```text
/lunar_isaac_validation/global_route
/lunar_isaac_validation/local_execution
/lunar_isaac_validation/local_frontier
/lunar_isaac_validation/hopper_route
/lunar_isaac_validation/certified_hop
/lunar_isaac_validation/goal_status
```

Set wheel colors to pale/strong blue, legged pale/strong green, and hopper pale/strong orange. Preserve the left-panel platform filter so only the selected platform is visible.

- [ ] **Step 5: Add hierarchical diagnostic rows to the existing RViz panel**

Read the exact `hierarchical_*` values published in Task 9 and show `map level`, `global resolution`, `global cells`, `route length`, `local frontier`, `global/local time`, and `hop count`. Missing optional hopper metrics render as `-`, not zero.

- [ ] **Step 6: Run external Python and RViz plugin tests**

Run:

```bash
python3 -m pytest -q
source /opt/ros/humble/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/external/log build --base-paths ros2_ws/src --packages-select lunar_isaac_validation lunar_isaac_rviz_plugins --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/external/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/external/install --cmake-args -DBUILD_TESTING=ON
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/external/log test --base-paths ros2_ws/src --packages-select lunar_isaac_validation lunar_isaac_rviz_plugins --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/external/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/external/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/external/test-results
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/external/test-results --verbose
```

Working directory: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression`

Expected: all tests pass.

- [ ] **Step 7: Commit external interactive behavior**

```bash
git add ros2_ws/src/lunar_isaac_validation ros2_ws/src/lunar_isaac_rviz_plugins
git commit -m "feat: visualize hierarchical planning evidence"
```

### Task 13: Add global-scale regression, performance tiers, and compatibility gates

**Files:**
- Create: `ros2_ws/src/lunar_planner_core/test/hierarchical_regression_test.cpp`
- Create: `tests/performance/hierarchical_planner_benchmark.cpp`
- Create: `tests/performance/test_hierarchical_planner_benchmark.py`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`
- Modify: `tests/differential/test_v3_semantic_equivalence.py`
- Modify: PPO bridge/test fixtures found by `rg -l 'PlannerInput|cpp_v3' ros2_ws/src tests | sort`
- Modify: `docs/superpowers/specs/2026-08-05-hierarchical-global-planning-design.md`
- Create: `docs/validation/hierarchical-global-planning.md`

**Interfaces:**
- Consumes: the complete main and external implementations.
- Produces: literal three-platform positive/negative fixtures, benchmark JSON `lunar-hierarchical-benchmark/v1`, and an explicit PPO behavior-compatibility verdict.

- [ ] **Step 1: Add literal end-to-end fixtures**

Add fixed-coordinate cases for distant wheel success/wall failure, distant leg success and a wheel-fail/leg-pass terrain, hopper multi-hop success/landing-chain failure/resolution failure/resource failure, global-success/local-failure, map transform, cancel, and repeated determinism. Do not scan or resample at test runtime to find successful endpoints.

- [ ] **Step 2: Add a benchmark contract test before the executable**

Require JSON keys `schema_version`, `platform`, `fixture`, `cells`, `resolution_m`, `runs`, `p50_s`, `p95_s`, `maximum_s`, `expanded_states`, `open_peak`, `peak_memory_bytes`, and `route_hash`. Require 30 measured runs after warm-up across cell tiers `65536`, `262144`, and `1048576` for open, fixed-obstacle, narrow-channel, and no-route fixtures.

- [ ] **Step 3: Run new tests to verify failure**

Run:

```bash
python3 -m pytest -q tests/performance/test_hierarchical_planner_benchmark.py
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/log test --base-paths ros2_ws/src --packages-select lunar_planner_core --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/main/test-results --ctest-args -R hierarchical_regression
```

Expected: benchmark executable/fixtures are absent.

- [ ] **Step 4: Implement benchmark and compatibility checks**

Record search metrics from the planner rather than process RSS estimates. Enforce Ubuntu p95 thresholds `0.5/1.0/2.0 s` for global search and `2.0/3.0/4.0 s` for the complete core call; emit AGX thresholds `1.0/2.0/4.0 s` and `3.0/4.0/6.0 s` as a device-only profile that is not evaluated on Ubuntu. Compare PPO macro-step observations/actions against existing golden fixtures; record `compatible` only if exact expected behavior remains, otherwise record `retraining_required` and do not relabel old checkpoints.

- [ ] **Step 5: Run the full authoritative Ubuntu verification**

Run:

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/log build --base-paths ros2_ws/src --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/install --cmake-args -DBUILD_TESTING=ON
source /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/install/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/log test --base-paths ros2_ws/src --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/test-results
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/test-results --verbose
python3 -m pytest -q tests/foundation tests/differential tests/performance
python3 tools/check_external_interfaces.py --config ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml --expected-lunar-navigation-prefix /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/full/install
python3 tools/check_repository_boundaries.py .
git diff --check
```

Expected: every command exits zero. Benchmark output stays under `~/CodexDownloads/lunar_navigation/hierarchical_global_planning`, not in Git.

- [ ] **Step 6: Run the external automated regression without Isaac Sim**

Run:

```bash
source /opt/ros/humble/setup.bash
python3 -m pytest -q
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/external/log test --base-paths ros2_ws/src --packages-select lunar_isaac_validation lunar_isaac_rviz_plugins --build-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/external/build --install-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/external/install --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/external/test-results
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/hierarchical_global_planning/external/test-results --verbose
```

Working directory: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression`

Expected: all tests pass. This proves ROS/RViz logic; it does not claim Isaac Sim or AGX acceptance.

- [ ] **Step 7: Record verified evidence and close the approved design status**

In `docs/validation/hierarchical-global-planning.md`, record commit IDs, host/ROS baseline, exact commands, pass counts, measured tier values, PPO compatibility verdict, and the explicit unverified rows `AGX performance` and `live Isaac Sim/RViz operator run`. Change the design status only to `implemented; Ubuntu automated validation complete` when all automated commands above pass.

- [ ] **Step 8: Commit the final main-repository evidence**

```bash
git add ros2_ws/src tests docs/superpowers/specs/2026-08-05-hierarchical-global-planning-design.md docs/validation/hierarchical-global-planning.md
git commit -m "test: qualify hierarchical global planning"
```

## Final Review Checklist

- [ ] The plan contains no deferred implementation markers, generic error-handling instructions, or cross-task shorthand.
- [ ] Every design section 5-18 maps to at least one task above.
- [ ] All signatures use `goal_map`, `preview_frame`, `execution_frame`, `GlobalRoutePreview::poses_map`, and `LocalPlanningProblem::goal_odom` consistently.
- [ ] `rg -n 'PlannerInput[^;]*goal|\.goal\b' ros2_ws/src/lunar_planner_core` shows only `goal_map`, `goal_odom`, or unrelated goal-region locals.
- [ ] `git status --short` shows `.vscode/` untouched and no generated artifacts.
- [ ] Main and external repositories each contain only their own scoped commits.
