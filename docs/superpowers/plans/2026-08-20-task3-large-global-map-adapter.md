# 课题三大范围地图适配与有界全局规划图 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a revision-consistent Task3 map adapter that turns a large read-only SQLite map and explicit task ROI into the bounded conservative global `GridMap` required by the planner, while retaining an independent 60 m L0 local-map path.

**Architecture:** Create one ROS 2 package, `luna_t3_map_adapter`, with C++ reference components for ROI/level selection and conservative aggregation, plus a Python runtime adapter for the Task3 SQLite/NumPy boundary, revisioned cache, and canonical publication. A thin Python ROS node consumes Task3 revision/task/local-map inputs, owns atomic map snapshots, and publishes canonical planner inputs. The planner core remains unchanged.

**Tech Stack:** C++20, Python 3.10, ROS 2 Humble, rclcpp, `grid_map_msgs`, `lunar_navigation_msgs`, SQLite3 read-only API, `numpy`, yaml-cpp, ament_cmake, GoogleTest, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-task3-large-global-map-adapter-design.md`

## Global Constraints

- Build for Ubuntu 22.04 + ROS 2 Humble on amd64 and Jetson AGX Orin R36 aarch64; do not commit build/install/log artifacts.
- Do not change `lunar_planner_core` or its `GridMap` snapshot interface.
- The only global input store is Task3 `global_grid_map.sqlite3`, opened read-only; never copy, modify, or publish the entire database.
- Freeze `ExplorationTask` ROI in `map`; only a new task revision creates a new ROI working set.
- Use the exact existing map pyramid: base 0.2 m, factors `[1,2,4,8,16,20]`, target axis 256, L5 maximum 4.0 m.
- Implement `lunar-conservative-grid-aggregation/v1` exactly; unknown/incomplete source data must not become free space.
- Publish only complete atomic `/environment/map_global` snapshots in `map`; retain local maps at 0.2 m in `odom`.
- No source tile read or global-map reconstruction when the source revision is unchanged.
- Keep the existing planner's stale-data and geometry safety gates; deployment convenience must not weaken safety behavior.

---

## File structure

Create package `ros2_ws/src/luna_t3_map_adapter/`:

| File | Responsibility |
| --- | --- |
| `CMakeLists.txt`, `package.xml` | C++20 package, ROS and SQLite dependencies, tests and executable installation |
| `include/luna_t3_map_adapter/types.hpp` | Value types: `TaskRoi`, `MapRevision`, `TileKey`, `FineCell`, `FineTile`, `GlobalMapSnapshot`, diagnostic counters |
| `include/luna_t3_map_adapter/roi_level.hpp`, `src/roi_level.cpp` | ROI validation, tile intersection, exact selection of planner-compatible global level |
| `include/luna_t3_map_adapter/conservative_aggregation.hpp`, `src/conservative_aggregation.cpp` | Pure 0.2 m to selected-level conservative aggregation |
| `python/luna_t3_map_adapter/t3_sqlite_reader.py` | Read-only SQLite URI connection, zlib + NumPy payload decode, schema validation, revisioned ROI tile read transaction |
| `python/luna_t3_map_adapter/global_map_cache.py` | Bounded raw-tile LRU and immutable global snapshot cache, atomic replacement semantics |
| `python/luna_t3_map_adapter/grid_map_conversion.py` | Global snapshot to canonical `grid_map_msgs.msg.GridMap` conversion |
| `python/luna_t3_map_adapter/local_map_adapter.py` | Task3 60 m map layer conversion into canonical 0.2 m local map |
| `python/luna_t3_map_adapter/t3_map_adapter_node.py` | Subscriptions, task/revision state machine, publication and diagnostics |
| `config/task3_map_adapter.yaml` | All source topic, SQLite path, semantic mapping, freshness, cache, and conservative-bound configuration |
| `launch/task3_map_adapter.launch.py` | One-node launch entry point |
| `test/*.cpp` | Isolated GTest coverage for every non-ROS component and node integration |

Modify:

| File | Change |
| --- | --- |
| `ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml` | Add Task3 source topics, `std_msgs/msg/UInt64` revision type, adapter ownership and canonical output binding without changing canonical planner topic names |
| `docs/interfaces/external-input-baseline.md` | Add Task3 adapter as the approved producer of canonical map topics and document SQLite read-only/revision semantics |
| `scripts/luna` and deployment documentation only if existing runtime package discovery requires registering the new ROS package | Include the adapter in build/start/status packaging paths; do not add unrelated CLI features |

## Task 1: Package skeleton and pure ROI/level contract

**Files:**
- Create: `ros2_ws/src/luna_t3_map_adapter/CMakeLists.txt`
- Create: `ros2_ws/src/luna_t3_map_adapter/package.xml`
- Create: `ros2_ws/src/luna_t3_map_adapter/include/luna_t3_map_adapter/types.hpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/include/luna_t3_map_adapter/roi_level.hpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/src/roi_level.cpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/test/roi_level_test.cpp`

**Interfaces:**
- Consumes: scalar ROI bounds and the fixed `GlobalMapConfig` values from `lunar_planner_core/types/planner_config.hpp`.
- Produces the C++20-compatible result contract used by every adapter component:
  ```cpp
  template <typename T>
  struct Result {
    std::optional<T> value;
    std::string reason_code;
    bool ok() const noexcept { return value.has_value() && reason_code.empty(); }
  };
  struct TaskRoi { double min_x_m, min_y_m, max_x_m, max_y_m; };
  struct SelectedGlobalLevel {
    std::size_t level, width, height;
    double resolution_m;
  };
  Result<SelectedGlobalLevel>
  SelectGlobalLevel(const TaskRoi&, const lunar::planning::GlobalMapConfig&);
  ```

- [ ] **Step 1: Write failing ROI-level tests**

  Add parameterized cases proving exact output:
  ```cpp
  EXPECT_EQ(SelectGlobalLevel({0, 0, 100, 100}, config).value->level, 1U);
  EXPECT_EQ(SelectGlobalLevel({0, 0, 300, 300}, config).value->level, 3U);
  EXPECT_EQ(SelectGlobalLevel({0, 0, 500, 500}, config).value->level, 4U);
  EXPECT_EQ(SelectGlobalLevel({0, 0, 1024.1, 10}, config).reason_code,
            "GLOBAL_MAP_SCALE_UNSUPPORTED");
  ```

- [ ] **Step 2: Run the test to verify failure**

  Run:
  ```bash
  cd ros2_ws && colcon test --packages-select luna_t3_map_adapter \
    --ctest-args -R luna_t3_map_adapter_roi_level_test
  ```
  Expected: package/target does not yet exist.

- [ ] **Step 3: Create the minimal package and implementation**

  Define `TaskRoi::Valid()` as finite, strictly positive width/height. Implement `SelectGlobalLevel` by delegating to the same ceiling and scale rules as `hierarchical::ExpectedGlobalMapLevel`; return `GLOBAL_MAP_SCALE_UNSUPPORTED` for valid ROIs exceeding L5 and `TASK_ROI_INVALID` for invalid bounds. Add CMake dependencies on `lunar_planner_core` and register one GTest target.

- [ ] **Step 4: Run focused test and package build**

  Run:
  ```bash
  cd ros2_ws
  colcon build --packages-select luna_t3_map_adapter
  colcon test --packages-select luna_t3_map_adapter --ctest-args -R roi_level
  colcon test-result --verbose
  ```
  Expected: all ROI-level cases pass.

- [ ] **Step 5: Commit**

  ```bash
  git add ros2_ws/src/luna_t3_map_adapter
  git commit -m "feat(task3-map): add bounded ROI level selection"
  ```

## Task 2: Conservative aggregation at the pure data boundary

**Files:**
- Modify: `ros2_ws/src/luna_t3_map_adapter/include/luna_t3_map_adapter/types.hpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/include/luna_t3_map_adapter/conservative_aggregation.hpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/src/conservative_aggregation.cpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/test/conservative_aggregation_test.cpp`
- Modify: `ros2_ws/src/luna_t3_map_adapter/CMakeLists.txt`

**Interfaces:**
- Consumes: `FineTile` of L0 `FineCell` values and a `SelectedGlobalLevel`.
- Produces:
  ```cpp
  struct AggregatedGrid { SelectedGlobalLevel level; std::vector<FineCell> cells; };
  Result<AggregatedGrid>
  AggregateConservatively(const FineTile&, const TaskRoi&, const SelectedGlobalLevel&);
  ```

- [ ] **Step 1: Write failing aggregation tests**

  Use a 2x2 L0 fixture and assert all required behavior:
  ```cpp
  EXPECT_TRUE(parent.obstacle);             // one child obstacle
  EXPECT_TRUE(parent.forbidden);            // one child forbidden
  EXPECT_FALSE(parent.valid_mask);          // one invalid child
  EXPECT_EQ(parent.obstacle_height, 2.0);   // maximum
  EXPECT_EQ(parent.observation_age_s, 8.0); // maximum
  EXPECT_EQ(parent.observation_quality, 0.2); // minimum
  EXPECT_EQ(parent.observation_count, 3U);  // minimum
  EXPECT_FALSE(parent.valid_mask);          // incomplete boundary block
  ```
  Add one valid-elevation case asserting mean elevation and the specified variance formula.

- [ ] **Step 2: Run the test to verify failure**

  Run:
  ```bash
  cd ros2_ws && colcon test --packages-select luna_t3_map_adapter \
    --ctest-args -R conservative_aggregation
  ```
  Expected: missing aggregation symbols.

- [ ] **Step 3: Implement exact conservative aggregation**

  Implement only the v1 rules from the approved spec. Reject source resolution other than 0.2 m, non-rectangular tile data, nonfinite scalar fields, and target shapes that disagree with `SelectedGlobalLevel`. Treat a missing odd-boundary child as invalid and forbidden.

- [ ] **Step 4: Run focused and regression tests**

  Run:
  ```bash
  cd ros2_ws
  colcon build --packages-select luna_t3_map_adapter
  colcon test --packages-select luna_t3_map_adapter --ctest-args -R 'roi_level|conservative_aggregation'
  colcon test-result --verbose
  ```
  Expected: all tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add ros2_ws/src/luna_t3_map_adapter
  git commit -m "feat(task3-map): aggregate global tiles conservatively"
  ```

## Task 3: Read-only Task3 SQLite tile reader

**Files:**
- Create: `ros2_ws/src/luna_t3_map_adapter/python/luna_t3_map_adapter/t3_sqlite_reader.py`
- Create: `ros2_ws/src/luna_t3_map_adapter/test/test_t3_sqlite_reader.py`
- Modify: `ros2_ws/src/luna_t3_map_adapter/CMakeLists.txt`
- Modify: `ros2_ws/src/luna_t3_map_adapter/package.xml`

**Interfaces:**
- Consumes: absolute SQLite path, `std::uint64_t` published revision, `TaskRoi`.
- Produces:
  ```cpp
  read_roi_at_revision(database_path, roi, required_revision) -> Task3MapRead
  ```

- [ ] **Step 1: Write failing provider tests against a temporary SQLite fixture**

  Create schema/data in a test-owned temporary database, then verify:
  ```cpp
  EXPECT_EQ(provider.ReadRoiAtRevision(roi, 7U).value->size(), 4U);
  EXPECT_EQ(provider.ReadRoiAtRevision(roi, 8U).reason_code, "TASK3_MAP_REVISION_MISMATCH");
  EXPECT_EQ(provider.ReadRoiAtRevision(roi, 7U).reason_code, "TASK3_TILE_MISSING"); // missing fixture tile
  ```
  Add an OS-permission fixture that confirms no SQL write statement is accepted.

- [ ] **Step 2: Run the test to verify failure**

  Run:
  ```bash
  cd ros2_ws && colcon test --packages-select luna_t3_map_adapter \
    --ctest-args -R t3_sqlite_reader
  ```
  Expected: provider does not exist.

- [ ] **Step 3: Implement transaction and schema checks**

  Open the database as `file:<path>?mode=ro`, execute `PRAGMA query_only=ON` and `BEGIN`, validate the metadata revision equals `required_revision`, read only all tiles intersecting the ROI, outer-decompress with `zlib`, and decode the documented NumPy `.npz` arrays with `numpy.load(..., allow_pickle=False)`. Return exact reason codes `TASK3_SQLITE_OPEN_FAILED`, `TASK3_MAP_REVISION_MISMATCH`, `TASK3_TILE_MISSING`, `TASK3_TILE_SCHEMA_INVALID`, or `TASK3_TILE_PAYLOAD_INVALID`. Roll back and return an error for every incomplete transaction; never retry by reading a newer revision under the old request.

- [ ] **Step 4: Run focused tests**

  Run:
  ```bash
  cd ros2_ws
  colcon build --packages-select luna_t3_map_adapter
  colcon test --packages-select luna_t3_map_adapter --ctest-args -R t3_sqlite_reader
  colcon test-result --verbose
  ```
  Expected: temporary database tests pass and no fixture file is altered.

- [ ] **Step 5: Commit**

  ```bash
  git add ros2_ws/src/luna_t3_map_adapter
  git commit -m "feat(task3-map): add revisioned read-only tile reader"
  ```

## Task 4: Bounded revisioned cache and immutable global snapshot assembly

**Files:**
- Create: `ros2_ws/src/luna_t3_map_adapter/include/luna_t3_map_adapter/global_map_cache.hpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/src/global_map_cache.cpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/test/global_map_cache_test.cpp`
- Modify: `ros2_ws/src/luna_t3_map_adapter/CMakeLists.txt`

**Interfaces:**
- Consumes: Python Task3 SQLite reader, `TaskRoi`, task identity, map revision, configured maximum cached tiles.
- Produces:
  ```cpp
  struct GlobalMapSnapshot {
    std::string mission_id;
    std::uint64_t mission_revision;
    std::uint64_t map_revision;
    AggregatedGrid grid;
  };
  class GlobalMapCache {
   public:
    Result<std::shared_ptr<const GlobalMapSnapshot>>
    Refresh(std::string_view mission_id, std::uint64_t mission_revision,
            const TaskRoi&, std::uint64_t map_revision);
  };
  ```

- [ ] **Step 1: Write failing cache tests**

  Use a fake provider with counted reads:
  ```cpp
  EXPECT_EQ(fake.read_count(), 4U);          // first ROI assembly
  EXPECT_EQ(cache.Refresh("m", 1U, roi, 7U)->map_revision, 7U);
  EXPECT_EQ(fake.read_count(), 4U);          // unchanged revision, no read
  EXPECT_NE(cache.Refresh("m", 1U, roi, 8U).value().get(), old.get());
  EXPECT_EQ(old->map_revision, 7U);          // previous request remains immutable
  ```
  Add LRU eviction with a small cache budget and task-revision replacement clearing old ROI tiles.

- [ ] **Step 2: Run the test to verify failure**

  Run:
  ```bash
  cd ros2_ws && colcon test --packages-select luna_t3_map_adapter \
    --ctest-args -R global_map_cache
  ```
  Expected: missing cache symbols.

- [ ] **Step 3: Implement cache semantics**

  Cache raw tiles by `(TileKey, map_revision)`. Key immutable snapshots by `(mission_id, mission_revision, map_revision, aggregation_version)`. Return the identical immutable shared pointer for an unchanged key. Build a new snapshot privately and replace the current pointer only after every ROI tile is read and aggregation succeeds. Keep at most `max_cached_tiles` raw tiles using deterministic LRU tie-breaking by tile key.

- [ ] **Step 4: Run cache and aggregation regression tests**

  Run:
  ```bash
  cd ros2_ws
  colcon build --packages-select luna_t3_map_adapter
  colcon test --packages-select luna_t3_map_adapter \
    --ctest-args -R 'global_map_cache|conservative_aggregation|sqlite_tile_provider'
  colcon test-result --verbose
  ```
  Expected: no stale pointer mutation, no unchanged-revision source read, all tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add ros2_ws/src/luna_t3_map_adapter
  git commit -m "feat(task3-map): cache immutable global map snapshots"
  ```

## Task 5: Canonical GridMap conversion and local-map layer adaptation

**Files:**
- Create: `ros2_ws/src/luna_t3_map_adapter/include/luna_t3_map_adapter/grid_map_conversion.hpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/src/grid_map_conversion.cpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/include/luna_t3_map_adapter/local_map_adapter.hpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/src/local_map_adapter.cpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/test/grid_map_conversion_test.cpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/test/local_map_adapter_test.cpp`

**Interfaces:**
- Consumes: `GlobalMapSnapshot`, Task3 `grid_map_msgs::msg::GridMap`, configured occupancy/semantic class sets and conservative scalar bounds.
- Produces:
  ```cpp
  Result<grid_map_msgs::msg::GridMap>
  ToCanonicalGlobalGridMap(const GlobalMapSnapshot&, const rclcpp::Time&);
  Result<grid_map_msgs::msg::GridMap>
  AdaptTask3LocalMap(const grid_map_msgs::msg::GridMap&, const LocalMapAdapterConfig&,
                      const geometry_msgs::msg::TransformStamped& odom_from_map,
                      const rclcpp::Time&);
  ```

- [ ] **Step 1: Write failing conversion tests**

  Assert global conversion carries `frame_id == "map"`, selected resolution/dimensions, all ten required layer names in exact canonical order, valid `data` dimensions, and the source map revision in diagnostics. Assert local adaptation produces `frame_id == "odom"`, `resolution == 0.2`, and rejects a 0.1 m Task3 message unless configured explicit conservative resampling is enabled.

  Add missing-field tests:
  ```cpp
  EXPECT_EQ(AdaptTask3LocalMap(no_elevation, config, tf, now).reason_code,
            "TASK3_LOCAL_ELEVATION_MISSING");
  EXPECT_TRUE(cell.forbidden);  // invalid input cell never becomes free
  EXPECT_GT(cell.elevation_variance, 0.0F);
  ```

- [ ] **Step 2: Run the tests to verify failure**

  Run:
  ```bash
  cd ros2_ws && colcon test --packages-select luna_t3_map_adapter \
    --ctest-args -R 'grid_map_conversion|local_map_adapter'
  ```
  Expected: conversion and local adapter symbols are absent.

- [ ] **Step 3: Implement canonical layer conversion**

  Convert only complete `FineCell` data into all ten canonical layers. Map occupancy and configured semantic classes to `obstacle`; merge configured task forbidden geometry and invalid input into `forbidden`. Use configured conservative variances and obstacle-height policy, rejecting the message rather than producing a traversable cell if policy preconditions are absent. Apply the explicit `odom_from_map` transform to local-map pose and preserve source stamp; do not derive global content from the rolling local map.

- [ ] **Step 4: Run focused tests and planner input compatibility test**

  Run:
  ```bash
  cd ros2_ws
  colcon build --packages-select luna_t3_map_adapter lunar_planner_ros
  colcon test --packages-select luna_t3_map_adapter lunar_planner_ros \
    --ctest-args -R 'grid_map_conversion|local_map_adapter|grid_map_adapter'
  colcon test-result --verbose
  ```
  Expected: canonical maps pass the existing `lunar_planner_ros` grid-map adapter tests.

- [ ] **Step 5: Commit**

  ```bash
  git add ros2_ws/src/luna_t3_map_adapter
  git commit -m "feat(task3-map): adapt Task3 maps to canonical layers"
  ```

## Task 6: ROS node, task/revision state machine, and diagnostics

**Files:**
- Create: `ros2_ws/src/luna_t3_map_adapter/include/luna_t3_map_adapter/t3_map_adapter_node.hpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/src/t3_map_adapter_node.cpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/src/main.cpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/test/t3_map_adapter_node_test.cpp`
- Modify: `ros2_ws/src/luna_t3_map_adapter/CMakeLists.txt`
- Modify: `ros2_ws/src/luna_t3_map_adapter/package.xml`

**Interfaces:**
- Consumes: `/mission/exploration_task` (`lunar_navigation_msgs/msg/ExplorationTask`), `/Car/T3/mapping/global_map_revision` (`std_msgs/msg/UInt64`), `/Car/T3/mapping/grid_map` (`grid_map_msgs/msg/GridMap`), and `map -> odom` TF.
- Produces: `/environment/map_global`, `/environment/map_local`, and `/diagnostics` status entries.

- [ ] **Step 1: Write failing node integration tests**

  Use a single-process rclcpp executor and temporary SQLite fixture. Assert this sequence:
  ```cpp
  PublishActiveTask("mission-a", 1U, roi_300m);
  PublishRevision(7U);
  EXPECT_THAT(WaitForGlobalMap(), HasGridMap("map", 1.6, 188U, 188U));

  PublishRevision(7U);
  EXPECT_EQ(fake_provider->read_count(), first_read_count); // no rebuild

  PublishRevision(8U);
  EXPECT_THAT(WaitForGlobalMap(), HasDiagnosticRevision(8U));
  ```
  Add `PAUSED`, invalid ROI, over-limit ROI, SQLite failure, and incomplete-local-map cases asserting no new map publication and an exact diagnostic reason.

- [ ] **Step 2: Run node test to verify failure**

  Run:
  ```bash
  cd ros2_ws && colcon test --packages-select luna_t3_map_adapter \
    --ctest-args -R t3_map_adapter_node
  ```
  Expected: node target is absent.

- [ ] **Step 3: Implement the node as a thin coordinator**

  Subscribe to task and revision with `reliable`, `transient_local`, depth 1; subscribe to Task3 local map with its documented `reliable`, `transient_local`, depth 1. On an `ACTIVE` task with a new revision, freeze ROI and invalidate old mission snapshots. On a strictly newer `UInt64` map revision, call `GlobalMapCache::Refresh`; publish only its complete canonical conversion. Ignore duplicate revisions. Publish local map only after a fresh map-to-odom transform and successful `AdaptTask3LocalMap`. Publish diagnostic counters and last error with `diagnostic_msgs`.

- [ ] **Step 4: Build and run all package tests**

  Run:
  ```bash
  cd ros2_ws
  colcon build --packages-select luna_t3_map_adapter lunar_planner_ros
  colcon test --packages-select luna_t3_map_adapter lunar_planner_ros
  colcon test-result --verbose
  ```
  Expected: all adapter tests and planner ROS compatibility tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add ros2_ws/src/luna_t3_map_adapter
  git commit -m "feat(task3-map): publish revisioned planner map inputs"
  ```

## Task 7: Configuration, launch, external-interface contract, and deployment registration

**Files:**
- Create: `ros2_ws/src/luna_t3_map_adapter/config/task3_map_adapter.yaml`
- Create: `ros2_ws/src/luna_t3_map_adapter/launch/task3_map_adapter.launch.py`
- Modify: `ros2_ws/src/luna_navigation_config/config/external_interfaces.yaml`
- Modify: `docs/interfaces/external-input-baseline.md`
- Modify: `scripts/luna`
- Modify: deployment command documentation under `docs/deployment/`
- Create: `ros2_ws/src/luna_t3_map_adapter/test/launch_config_test.cpp`

**Interfaces:**
- Consumes: user-supplied `sqlite_path`, Task3 topic names, semantic/occupancy mapping, conservative bounds, cache count, stale thresholds.
- Produces: one launchable `luna_t3_map_adapter_node` and documented `luna start` profile inclusion.

- [ ] **Step 1: Write failing configuration tests**

  Verify valid minimal YAML loads every required field and invalid configurations reject exactly:
  ```cpp
  EXPECT_EQ(LoadConfig(missing_sqlite).reason_code, "TASK3_SQLITE_PATH_MISSING");
  EXPECT_EQ(LoadConfig(zero_variance).reason_code, "CONSERVATIVE_VARIANCE_INVALID");
  EXPECT_EQ(LoadConfig(zero_cache).reason_code, "TASK3_TILE_CACHE_INVALID");
  ```
  Add a text-level test asserting `external_interfaces.yaml` declares Task3 revision as `std_msgs/msg/UInt64` and preserves canonical `/environment/map_global` ownership as the adapter.

- [ ] **Step 2: Run tests to verify failure**

  Run:
  ```bash
  cd ros2_ws && colcon test --packages-select luna_t3_map_adapter \
    --ctest-args -R launch_config
  ```
  Expected: config/launch files and loader are absent.

- [ ] **Step 3: Add one explicit runtime configuration and launch entry**

  Define all source inputs in `task3_map_adapter.yaml`; include no default real SQLite path, no test fixture path, and no zero uncertainty. Add `task3_map_adapter.launch.py` accepting `config_file` only. Register the package in `scripts/luna` build and start profiles so `luna build` and `luna start --profile task3` invoke the installed package, not source-tree Python or an ad-hoc command.

- [ ] **Step 4: Validate package, interface documentation, and deployment boundary**

  Run:
  ```bash
  cd ros2_ws
  colcon build --packages-select luna_t3_map_adapter lunar_planner_ros
  colcon test --packages-select luna_t3_map_adapter lunar_planner_ros
  colcon test-result --verbose
  cd ..
  ./scripts/luna doctor --profile task3 --config ros2_ws/src/luna_t3_map_adapter/config/task3_map_adapter.yaml
  git diff --check
  ```
  Expected: all tests pass; `doctor` reports missing real Task3 input files as explicit deployment prerequisites rather than treating them as a successful live connection.

- [ ] **Step 5: Commit**

  ```bash
  git add ros2_ws/src/luna_t3_map_adapter \
    ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml \
    docs/interfaces/external-input-baseline.md scripts/luna docs/deployment
  git commit -m "feat(task3-map): configure deployable Task3 map adapter"
  ```

## Task 8: Recorded-input compatibility and final verification

**Files:**
- Create: `ros2_ws/src/luna_t3_map_adapter/test/recorded_input_compatibility_test.cpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/test/fixtures/README.md`
- Modify: `docs/interfaces/external-input-baseline.md`

**Interfaces:**
- Consumes: a sanitized, small recorded Task3 SQLite fixture plus task/revision/local-map message fixtures.
- Produces: repeatable compatibility evidence, not a new runtime feature.

- [ ] **Step 1: Write failing end-to-end compatibility test**

  Define a fixture with a 300 m ROI and two revisions. Assert the node publishes L3 global map, 0.2 m `odom` local map, ten canonical layers, no map before `ACTIVE` task plus revision, cache reuse on duplicate revision, and a newly immutable global snapshot after revision increment.

- [ ] **Step 2: Run the test to verify failure**

  Run:
  ```bash
  cd ros2_ws && colcon test --packages-select luna_t3_map_adapter \
    --ctest-args -R recorded_input_compatibility
  ```
  Expected: missing fixture or compatibility target.

- [ ] **Step 3: Add the minimal sanitized fixture and test**

  Keep the fixture below repository size limits and include no production map data. Document its source schema, expected map revisions, and SHA-256 in the fixture README. The test must drive the public ROS topics and inspect only canonical outputs/diagnostics.

- [ ] **Step 4: Run final focused verification**

  Run:
  ```bash
  cd ros2_ws
  colcon build --packages-select luna_t3_map_adapter lunar_planner_ros
  colcon test --packages-select luna_t3_map_adapter lunar_planner_ros
  colcon test-result --verbose
  cd ..
  git diff --check
  ```
  Expected: all selected tests pass and the worktree contains only intentional source/config/docs changes.

- [ ] **Step 5: Commit**

  ```bash
  git add ros2_ws/src/luna_t3_map_adapter docs/interfaces/external-input-baseline.md
  git commit -m "test(task3-map): verify recorded map adapter compatibility"
  ```

## Plan self-review

- Spec coverage: Tasks 1–2 implement ROI, level selection, and v1 conservative aggregation; Tasks 3–4 implement SQLite read-only/revision/cache behavior; Task 5 implements canonical global/local map layer adaptation; Task 6 implements ROS atomic publication and diagnostics; Task 7 implements deployable configuration and interface documentation; Task 8 verifies recorded compatibility.
- Safety: every source-data failure becomes an explicit rejection or invalid/forbidden cell; no task uses synthetic free space, a coarser-than-L5 map, or an in-place snapshot mutation.
- Type consistency: `TaskRoi`, `SelectedGlobalLevel`, `FineTile`, `AggregatedGrid`, and `GlobalMapSnapshot` are defined in Tasks 1–4 before later tasks consume them; canonical output topic names match the existing planner input contract.
- Placeholder scan: no deferred implementation step is required for the described integration; actual Task3 database path and semantic classes remain deployment configuration values, not source defaults.
