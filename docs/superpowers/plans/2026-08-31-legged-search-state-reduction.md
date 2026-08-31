# 足式局部搜索状态缩减实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用机身中心必要可行图和距离场主启发函数减少足式局部搜索展开，并将足式状态格与旋转原语统一为 32 航向。

**Architecture:** `LeggedTraversalProjection` 在原逐格地形结论之后构建机身中心必要可行掩码；规划器用该掩码构建已有多目标距离场。`LeggedSearchGraph` 将距离场换算为主启发代价，同时保留完整动作边认证。状态键统一为 32 航向格，生产旋转原语同步为 `±pi/16`。

**Tech Stack:** C++20、ROS 2 Jazzy、GoogleTest、CTest、现有 ARA*、Grid V1 局部投影。

**Spec:** `docs/superpowers/specs/2026-08-31-legged-grid-v1-planning-design.md`

## Global Constraints

- 只使用现有 occupancy、elevation、capability 和目标距离场，不增加地图输入层。
- 不增加 preferred path、多标签、路线走廊、状态数量参数或内存预算参数。
- 轮式与 Hopper 的外部行为保持不变。
- 每条成功足式边继续由现有 `SweepBody` 认证。
- 本机验证只形成 Jazzy 证据；Humble、Orin、DDS、rosbag 和实车保持 `NOT_RUN`。

---

### Task 1: 机身中心必要可行掩码

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/legged/legged_traversal_projection.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/legged/legged_traversal_projection.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_core/test/legged_traversal_projection_test.cpp`

**Interfaces:**
- Produces: `LeggedTraversalProjection::body_center_feasible`, length equal to `map->cell_count()`.
- Consumes: `hard_feasible`, `step_feasible`, body XY extent and `minimum_body_clearance_m`.

- [x] **Step 1: Write failing projection tests**

  Add one flat-map test with a single infeasible cell. Assert the hazard cell and centers closer than the expanded-body inscribed radius are false, while a sufficiently distant interior cell is true. Add a boundary assertion proving centers closer than the same radius to the map edge are false.

- [x] **Step 2: Run the projection test and verify RED**

  ```bash
  source /opt/ros/jazzy/setup.bash
  colcon --log-base /tmp/lunar-legged-state-reduction-log build \
    --base-paths ros2_ws/src --packages-select lunar_pure_planner_core \
    --build-base build-jazzy-legged-edge --install-base install-jazzy-legged-edge \
    --cmake-args -DBUILD_TESTING=ON
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_core \
    -R legged_traversal_projection --output-on-failure
  ```

  Expected: compilation fails because `body_center_feasible` does not exist.

- [x] **Step 3: Implement the minimal projection**

  Build a hazard mask from `hard_feasible == 0 || step_feasible == 0`, run existing `shared::BuildCellAreaClearance`, and mark a center false when it is certainly inside the expanded rectangle's inscribed disk or is too close to the physical map boundary. Preserve cancellation/deadline propagation.

- [x] **Step 4: Run the projection test and verify GREEN**

  Run the command from Step 2 and require zero failures.

### Task 2: Make the body-center distance field the legged anchor heuristic

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/legged/anytime_legged_planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/anytime_legged_planner_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/dual_mode_planner_test.cpp`

**Interfaces:**
- Consumes: `LeggedTraversalProjection::body_center_feasible` and `GoalDistanceField`.
- Produces: ARA* `heuristic(state)` whose obstacle-aware component is in the same normalized cost units as legged edge cost.

- [x] **Step 1: Write failing planner tests**

  Change test request builders to construct their distance fields from `body_center_feasible`. Add a body-impossible passage case that returns `LEGGED_NO_PATH` before local state expansion. Add a detour case with a literal expansion upper bound that the current Euclidean-only anchor exceeds.

- [x] **Step 2: Run tests and verify RED**

  ```bash
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_core \
    -R 'anytime_legged_planner|dual_mode_planner' --output-on-failure -j1
  ```

  Expected: new connectivity or expansion assertion fails against the old mask/heuristic.

- [x] **Step 3: Implement the minimal mask and heuristic wiring**

  In `planner.cpp`, pass `body_center_feasible` to `BuildGoalDistanceField`. In `LeggedSearchGraph::Heuristic`, read the state's distance-field value and nearest goal, subtract the goal tolerance and cell-center allowance, normalize distance and minimum execution time with existing `cost_scales`, and take the maximum with the normalized Euclidean target-region lower bound. Keep `Guidance` unchanged.

- [x] **Step 4: Run tests and verify GREEN**

  Run the command from Step 2 and require zero failures.

### Task 3: Unify the legged lattice at 32 yaw bins

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/legged/anytime_legged_planner.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/anytime_legged_planner_test.cpp`
- Modify: `config/legged.yaml`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/platform_config.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_ros/test/platform_config_test.cpp`

**Interfaces:**
- Produces: `maximum_yaw_bin_count == 32` in normal and narrow areas.
- Produces: production spin primitives `yaw_change_rad == ±pi/16` and updated capability version.

- [x] **Step 1: Write failing yaw behavior tests**

  Update the narrow-corridor expectation to 32 and add a two-step spin reachability case using literal `pi/8` final yaw with two `pi/16` primitives. Update the platform-config test to assert the loaded legged spin increments are `±pi/16`.

- [x] **Step 2: Run tests and verify RED**

  ```bash
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_core \
    -R anytime_legged_planner --output-on-failure
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_ros \
    -R platform_config_test --output-on-failure
  ```

  Expected: old 64/128-bin diagnostics and old `±pi/32` config fail.

- [x] **Step 3: Implement 32-bin quantization and config amendment**

  Use 32 bins in `KeyFor` and all result diagnostics. Change only the two spin primitive yaw increments in `config/legged.yaml` to `±0.19634954084936207`, update the capability version to `quad48-approved-baseline-v2`, and record the local amendment without falsifying the original source provenance.

- [x] **Step 4: Run tests and verify GREEN**

  Rebuild core and ROS packages, then run both commands from Step 2.

### Task 4: Necessary regression and evidence

**Files:**
- Modify: `docs/superpowers/specs/2026-08-31-legged-grid-v1-planning-design.md`
- Modify: `docs/superpowers/plans/2026-08-31-legged-search-state-reduction.md`

- [x] **Step 1: Run core package tests serially**

  ```bash
  source /opt/ros/jazzy/setup.bash
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_core \
    --output-on-failure -j1
  ```

- [x] **Step 2: Run affected ROS and contract tests**

  ```bash
  source /opt/ros/jazzy/setup.bash
  source install-jazzy-legged-edge/setup.bash
  ctest --test-dir build-jazzy-legged-edge/lunar_pure_planner_ros \
    --output-on-failure -j1
  python3 -m pytest tests/test_isolation_contract.py tests/test_launch_contract.py -q
  ```

- [x] **Step 3: Check the exact diff**

  ```bash
  git diff --check
  git status --short
  git diff --stat
  ```

  Confirm no wheel/Hopper source, controller, launch interface, generated artifact, or unrelated file changed.

- [x] **Step 4: Record evidence**

  Append the actual build/test counts and remaining `NOT_RUN` boundaries to the design document. Do not claim RViz, Humble, Orin, DDS, rosbag or vehicle validation unless run.
