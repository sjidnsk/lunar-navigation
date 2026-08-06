# Three-Platform Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已批准的轮式、Quad48 足式和飞跃式设计落实为同一版本化能力合同、可执行 C++/ROS 规划语义、可审计 RViz 证据和正式 capability freeze，并保持完整搜索与单段滚动授权语义。

**Architecture:** 先升级外部接口与能力 schema，再由纯 C++ 核心消费不可变能力和同请求快照。轮式、足式共用二维完整全局 A*，在 L0 局部图分别执行连续运动学与旋转足迹认证；飞跃式绕过地面全局路线，使用实时推进剂状态认证精确目标、单条双脉冲抛物线、连续飞行管和强保证凸落区。ROS 层只负责外部消息、时间一致性、能力加载、Action 转换和 Marker 发布，不复制规划判定。

**Tech Stack:** C++20、ROS 2 Humble、ament/colcon、yaml-cpp、GoogleTest、pytest、grid_map、RViz Marker、CMake Release。

## Global Constraints

- 所有生产离散搜索只以有限地图/状态空间自然穷尽、合作取消、输入版本失效、真实分配失败或数值不确定结束；不得用展开数、候选数、Open 数或固定 wall-clock 阈值冒充不可行。
- 平滑器可以有迭代上限，但只能回退到已完整碰撞认证的离散轨迹，并显式输出 `DISCRETE_FALLBACK`。
- 规划终点保持请求中的连续坐标；`0.05 m` 与 `5°` 仅用于执行完成判定。
- 静态平台能力和推进剂 Topic 仍由外部单位拥有；仓内 provisional schema 与 capability freeze 是消费合同和已批准工程基线，不接管外部数据生产。
- 所有 ROS 命令先 source `/opt/ros/humble/setup.bash` 并验证 `ROS_DISTRO=humble`；构建、日志和运行 artifact 写入 `/home/kai/CodexDownloads/lunar_navigation/three_platform_capability`。
- 每项行为变更严格执行 RED → GREEN → REFACTOR；每个任务完成后只提交该任务相关文件。
- 当前比较基线：Release 构建成功，ROS/C++ 共 `228 tests, 0 failures`；静态仓库/接口测试 `55 passed`。

---

### Task 1: Upgrade the externally owned hopper propellant contract

**Files:**

- Create: `ros2_ws/src/lunar_navigation_msgs/msg/HopperPropellantState.msg`
- Modify: `ros2_ws/src/lunar_navigation_msgs/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml`
- Modify: `docs/interfaces/external-input-baseline.md`
- Modify: `tools/check_external_interfaces.py`
- Modify: `tests/foundation/test_external_interface_config.py`
- Modify: `tests/foundation/test_navigation_message_package.py`

- [x] **Step 1: Write failing contract tests**

  Add exact message declarations and require external topic `/platform/hopper_propellant_state`, reliable/volatile/depth 10, frame `platform_base_frame`, max age `0.5 s`, and fields `header/platform_id/capability_version/total_mass_kg/remaining_usable_fuel_mass_kg`. Bump the external interface document to `lunar-external-interfaces/v4`.

- [x] **Step 2: Run the focused tests and verify RED**

  Run: `python3 -m pytest -q tests/foundation/test_external_interface_config.py tests/foundation/test_navigation_message_package.py`

  Expected: failures for the absent message and topic contract.

- [x] **Step 3: Implement the provisional message and contract**

  Message body:

  ```text
  std_msgs/Header header
  string platform_id
  string capability_version
  float64 total_mass_kg
  float64 remaining_usable_fuel_mass_kg
  ```

  Register it in `rosidl_generate_interfaces`; update checker declarations and documentation while retaining `owner: external` and `replacement_policy: atomic`.

- [x] **Step 4: Verify GREEN and commit**

  Run the focused tests plus `python3 tools/check_repository_boundaries.py .`.

  Commit: `feat: define hopper propellant input contract`

### Task 2: Freeze the v2 three-platform capability schema and provenance

**Files:**

- Create: `ros2_ws/src/lunar_navigation_config/config/platform_capability_schema_v2.yaml`
- Create: `ros2_ws/src/lunar_navigation_config/config/three_platform_capability_freeze_v1.yaml`
- Create: `tools/check_platform_capability_freeze.py`
- Create: `tests/foundation/test_platform_capability_freeze.py`
- Modify: `ros2_ws/src/lunar_navigation_config/CMakeLists.txt`
- Modify: `docs/interfaces/external-input-baseline.md`

- [x] **Step 1: Write failing freeze validation tests**

  Require exactly `WHEELED`, `LEGGED`, `HOPPER`, distinct increasing capability versions, per-field source type, explicit wheel unknown fields, no `0.16 m` legged climb, no historical wheel proxy values, hopper reference mass only under `reference_conditions`, and a deterministic SHA-256 over canonicalized platform payloads.

- [x] **Step 2: Verify RED**

  Run: `python3 -m pytest -q tests/foundation/test_platform_capability_freeze.py`

- [x] **Step 3: Implement the consumer schema and approved freeze**

  Freeze these approved values:

  - wheel: `1.182×0.818×1.29996 m`, diameter `0.319`, width `0.148`, wheelbase `0.8175`, track `0.670`, underbody `0.210 approximate`, clearance/relief `0.20`, speed `±1.5`, yaw `1.0`, slope `20°`;
  - legged: `0.68×0.33×0.35 m`, mass `15.89`, body height `[0.28,0.38]`, clearance `0.30`, slope `30°`, step `0.50`, gap `0.30`, speeds `1.5/0.8/1.0`;
  - hopper: `Isp=301 s`, landing radius `0.45`, collision radius `0.55`, slope `10°`, plane residual `0.05`, margins `0.20`, delta-v margin `0.10`, reference `20 kg/0.20 kg/100 m`.

  The file explicitly declares external ownership and project engineering provenance; it is not an external publisher implementation.

- [x] **Step 4: Validate digest, UTF-8, and commit**

  Run checker directly and its pytest; read both YAML files with UTF-8.

  Commit: `feat: freeze approved three-platform capabilities`

### Task 3: Replace runtime capability types and loader with schema v2

**Files:**

- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/platform_capability.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_config.hpp`
- Modify: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/capability_loader.hpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/capability_loader.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/capability_loader_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/test_fixtures.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/test_fixtures.hpp`
- Modify: `tests/fixtures/capabilities/test-only/{wheeled,legged,hopper}.yaml`

- [x] **Step 1: Add failing type/loader tests**

  Require the following public shapes:

  ```cpp
  struct WheeledCapability {
    std::vector<Vec2> footprint_xy_m;
    Vec3 body_extent_m;
    double wheel_diameter_m, wheel_width_m, wheelbase_m, track_width_m;
    double minimum_underbody_clearance_m, maximum_local_obstacle_relief_m;
    bool allow_unsupported_gap;
    // approved speed, acceleration, curvature, slope, clearance fields
    std::vector<WheelMotionPrimitive> motion_primitives; // geometry only
  };

  struct LeggedCapability {
    Vec3 body_extent_m;
    double platform_mass_kg, maximum_payload_kg;
    double maximum_slope_rad, maximum_step_height_m, maximum_gap_width_m;
    double minimum_body_clearance_m, step_vertical_rate_mps;
    Interval body_height_m, forward_speed_mps, lateral_speed_mps, yaw_rate_radps;
    double maximum_linear_acceleration_mps2, maximum_yaw_acceleration_radps2;
  };

  struct HopperCapability {
    double specific_impulse_s, landing_support_radius_m;
    double flight_collision_radius_m, maximum_landing_slope_rad;
    double maximum_landing_plane_residual_m;
    double landing_lateral_margin_m, flight_map_margin_m;
    double reachability_delta_v_margin_ratio, standard_gravity_mps2;
  };
  ```

  Remove physical roughness/confidence fields from legged capability, remove wheel fixed primitive duration, and reject every retired hopper v1 field with `CAPABILITY_SCHEMA_VERSION_INCOMPATIBLE`.

- [x] **Step 2: Verify compile/test RED**

  Build only `lunar_planner_core`/`lunar_planner_ros` tests and confirm new assertions fail before implementation.

- [x] **Step 3: Implement v2 parsing and validation**

  Parse `platform-control-capability-source/v2`, preserve source metadata/digest in `LoadedCapabilities`, accept `base_footprint` for wheel and `base_link` for legged/hopper, validate derived wheel geometry, and reject unknown/retired keys rather than ignoring them.

- [x] **Step 4: Migrate test builders and commit**

  Keep fixtures explicitly `test-only`; do not rename them as formal material.

  分阶段迁移说明：v2 加载器已经拒绝所有旧 schema/飞跃式 v1 字段；为保持每次提交均可构建，旧后端仍依赖的内部兼容成员在本步只做隔离赋值，并在 Task 6、7、9 替换对应算法时随同删除。它们不再是 v2 外部输入合同。

  Commit: `refactor: adopt platform capability schema v2`

### Task 4: Remove fixed discrete-search resource ceilings

**Files:**

- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_config.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/shared/ara_star.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/grid_search.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_lattice.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_ros/config/planner_ros.schema.json`
- Modify: `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/{shared_core_determinism_test,global_grid_search_test,wheel_planner_test,legged_planner_test}.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp`

- [x] **Step 1: Add old-limit regression tests**

  Construct routes that exceed prior expanded/generated/open/terminal/subdivision values and assert success or true Open exhaustion; separately assert stop-token cancellation and `std::bad_alloc` mapping remain “search incomplete”.

- [x] **Step 2: Verify RED against current guards**

- [x] **Step 3: Delete production count ceilings**

  Remove `SearchResourceLimits`, terminal candidate caps and fixed continuous subdivision caps from production config. Generate neighbors lazily and derive collision samples as:

  ```cpp
  sections = ceil((translation_m + corner_radius_m * abs(delta_yaw)) /
                  (local_resolution_m / 4.0));
  ```

  Retain cancellation checks at expansion/section boundaries. Catch actual allocation failures at the planner facade and return `SEARCH_RESOURCE_FAILURE`/platform-specific allocation failure.

- [x] **Step 4: Verify and commit**

  Commit: `fix: exhaust finite searches without count ceilings`

### Task 5: Implement the shared three-band ground clearance projection

**Files:**

- Modify: `ros2_ws/src/lunar_planner_core/src/shared/safe_projection.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/global_route_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/local_frontier.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/{shared_core_test,global_route_planner_test,local_frontier_test}.cpp`

- [x] **Step 1: Add exact boundary tests**

  Wheel bands are `<0.609`, `[0.609,0.918722)`, `>=0.918722`; legged bands are `<0.465`, `[0.465,0.678)`, `>=0.678`. Obstacle cells participate as complete `0.20 m` squares and no extra `sqrt(2)*resolution` inflation is permitted.

- [x] **Step 2: Verify RED**

- [x] **Step 3: Implement tri-state projection**

  Add `ClearanceClass { kRejected, kConditional, kUnconditional }`; global A* may traverse conditional cells but marks their corridors for L0 oriented-footprint certification. A failed conditional corridor disables only that edge/corridor and resumes global alternatives.

- [x] **Step 4: Verify and commit**

  Commit: `feat: add conditional ground-clearance projection`

### Task 6: Implement wheel geometry, terrain, lazy primitives, and timing

**Files:**

- Modify: `ros2_ws/src/lunar_planner_core/src/shared/map_snapshot.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/shared/terrain_checks.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_types.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_sweep_validator.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_timing.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/{wheel_planner_test,wheel_fault_matrix_test}.cpp`

- [x] **Step 1: Add terrain and primitive RED tests**

  Cover four-wheel support-plane fitting; arbitrary-direction `20°`; continuous `0.20 m` positive relief; `>0.20 m` and vertical discontinuity rejection; no unsupported gap; `0.210 m` underbody clearance; roughness affecting cost/speed but not hard feasibility; uncertainty remaining a hard map policy; forward/reverse arcs, spin, stop-and-switch, continuous primitive endpoints and exact start/end anchors.

- [x] **Step 2: Implement independent physical roughness**

  Derive roughness from L0 elevation residuals relative to the support plane. Do not reuse `elevation_variance`. Use:

  ```cpp
  scale = 1.0 / (1.0 + square(roughness_m / 0.1595));
  speed = kinematic_speed * cos(surface_slope) * scale;
  ```

- [x] **Step 3: Implement lazy continuous wheel search and timing**

  Preserve continuous endpoint pose in every node while using `(cell,yaw_bin,mode)` only as search key. Generate the approved 0.20 m straight, 1 m radius arc, spin and switch primitives on expansion. Derive trapezoidal/triangular segment time from current velocity, slope, roughness, curvature, acceleration and braking; no fixed nominal duration.

- [x] **Step 4: Run focused tests and commit**

  Commit: `feat: implement approved wheeled capability semantics`

### Task 7: Implement Quad48 ordinary-edge terrain and oriented-body planning

**Files:**

- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_types.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_terrain.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_lattice.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_timing.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_spline_optimizer.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/{legged_planner_test,legged_fault_matrix_test}.cpp`

- [x] **Step 1: Add Quad48 RED tests**

  Cover `30°`, `0.50 m` step, directional unsupported spans `0.30 m`/`0.40 m`, diagonal metric gap, roughness invariance, unknown hard rejection, lateral travel, task yaw, exact start/end, rotated rectangle cases where AABB/circumcircle would misreject, and smooth/fallback evidence.

- [x] **Step 2: Implement ordinary-edge terrain semantics**

  Treat step/gap as normal feasible edges. Compute execution lower bound:

  ```cpp
  max(d_xy / v_xy,
      abs(delta_h) / 0.10,
      abs(delta_yaw) / 1.0,
      linear_accel_time,
      yaw_accel_time)
  ```

  Roughness is diagnostic only and cannot alter feasibility, cost or speed.

- [x] **Step 3: Implement oriented rectangle sweep**

  Replace rotated-AABB occupancy with rectangle-vs-full-cell SAT intersection and spatial sampling `<=0.05 m`. Retain yaw only in the L0 local state; global state remains `(x,y)`.

- [x] **Step 4: Verify and commit**

  Commit: `feat: implement approved Quad48 planning semantics`

### Task 8: Implement hopper propellant snapshots and exact-goal validation

**Files:**

- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp`
- Modify: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/snapshot_store.hpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/snapshot_store.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/snapshot_builder.hpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/snapshot_builder.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/{snapshot_builder_test,plan_motion_server_test}.cpp`

- [x] **Step 1: Add snapshot RED tests**

  Test missing, zero/future/older-than-0.5s, frame/platform/version mismatch, invalid mass/fuel and >0.25s skew. Verify wheel/legged do not require propellant state. Require hopper `POINT`, tolerance exactly zero, no yaw.

- [x] **Step 2: Add immutable core state**

  ```cpp
  struct HopperPropellantState {
    TimePoint stamp;
    std::string platform_id;
    std::string capability_version;
    double total_mass_kg;
    double remaining_usable_fuel_mass_kg;
  };
  ```

  Add `std::optional<HopperPropellantState> hopper_propellant` to `PlannerInput`; bind it to the same frozen request.

- [x] **Step 3: Wire ROS subscription/store/freeze**

  Use its own mutually-exclusive callback group and reliable/volatile/depth-10 QoS. Map failures to `HOPPER_PROPELLANT_STATE_INVALID` or `HOPPER_PROPELLANT_STATE_STALE` before calling C++.

- [x] **Step 4: Verify and commit**

  Commit: `feat: bind hopper propellant state to planning snapshots`

### Task 9: Replace legacy hopper routing with one fuel-certified ballistic hop

**Files:**

- Create: `ros2_ws/src/lunar_planner_core/src/hopper/propellant_model.{hpp,cpp}`
- Create: `ros2_ws/src/lunar_planner_core/test/propellant_model_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/{hopper_types.hpp,ballistic_kinematics.cpp,flight_tube_certifier.cpp,hop_certifier.cpp,landing_region.cpp,hopper_planner.cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/hopper_route_planner.{hpp,cpp}`
- Modify: `ros2_ws/src/lunar_planner_core/test/{ballistic_envelope_test,hopper_planner_test,hopper_fault_matrix_test,hopper_route_planner_test}.cpp`

- [x] **Step 1: Add mathematical RED tests**

  Freeze the 100 m reference: launch speed `12.727922`, time `11.111111 s`, apex `25 m`, ideal delta-v `25.455844`, certified delta-v `28.001429`, certified fuel about `0.188827 kg`. Add monotonic mass/fuel, elevation, boundary fuel, numerical-domain and deterministic tests.

- [x] **Step 2: Implement rocket equation and full feasible-time search**

  ```cpp
  available_dv = isp * g0 * log(m0 / (m0 - fuel));
  v0(T) = displacement / T - 0.5 * gravity * T;
  vf(T) = v0(T) + gravity * T;
  certified_dv(T) = 1.10 * (norm(v0) + norm(vf));
  ```

  Search the complete fuel-feasible time interval with deterministic adaptive interval branch-and-bound. Fixed sample arrays or candidate/section caps are prohibited. Unresolved intervals return `NUMERICAL_FAILURE`, not infeasible.

- [x] **Step 3: Certify exact target, continuous tube, and strong convex region**

  Use exact requested `x/y`, terrain-derived `z`, full `0.65 m` L0 landing support disk, slope `10°`, plane residual `0.05 m`, and continuous `0.75 m` flight tube. Grow a positive-area convex region only where every interior landing center retains terrain, fuel and tube proof; never snap or substitute the target.

- [x] **Step 4: Remove multi-hop semantics**

  Bypass landing-node graph, promotion, route cursor and automatic continuation. A successful hopper request emits exactly one certified segment; preserve commitment protection only.

- [x] **Step 5: Verify and commit**

  Commit: `feat: certify exact single hopper flight from fuel state`

### Task 10: Extend hop wire output and enforce one-segment invariants

**Files:**

- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/motion_reference.hpp`
- Modify: `ros2_ws/src/lunar_planning_msgs/msg/HopSegment.msg`
- Modify: `ros2_ws/src/lunar_planner_ros/src/message_conversion.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/reference_guard.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/{message_conversion_test,reference_guard_test}.cpp`
- Modify: `tests/foundation/test_planning_message_package.py`

- [ ] **Step 1: Add output RED tests**

  Require nominal landing point, ideal/certified/remaining fuel, required/available delta-v, capability version and both map generations. Reject hopper output with zero or more than one segment at the conversion boundary.

- [ ] **Step 2: Implement lossless conversion**

  Ensure finite nonnegative fuel/delta-v, `certified >= ideal`, expected remaining consistency, exact map generations, and a reconstruction-consistent launch velocity/flight time.

- [ ] **Step 3: Verify guard behavior and commit**

  Commit: `feat: publish certified single-hop evidence`

### Task 11: Publish platform-specific RViz evidence without duplicating planning

**Files:**

- Modify: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/route_marker_publisher.hpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/route_marker_publisher.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/route_marker_publisher_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`

- [ ] **Step 1: Add Marker RED tests**

  Wheel: correct body/four-wheel scale, raw footprint, `0.20 m` margin, start/goal/global/current segment and direction. Legged: `0.68×0.33 m` body, start/goal/global/current segment. Hopper: platform, exact target, `0.65 m` disk, filled convex region, nominal point, parabola, `1.50 m` tube diameter and fuel/delta-v/version text. All namespaces must be deterministic and clear stale markers on platform switch/deactivate.

- [ ] **Step 2: Implement evidence-only markers**

  Consume core reference and diagnostics; Marker code must not re-run collision or change outcomes. Keep physical roughness diagnostic colors distinct from obstacle/unknown colors.

- [ ] **Step 3: Verify and commit**

  Commit: `feat: visualize approved platform planning evidence`

### Task 12: Qualify Release performance and capability-freeze integration

**Files:**

- Modify: `tests/performance/hierarchical_planner_benchmark.cpp`
- Modify: `tests/performance/test_hierarchical_planner_benchmark.py`
- Create: `docs/validation/three-platform-capability-qualification.md`
- Modify: `docs/migration/volume-2-completion.md`
- Modify: `docs/superpowers/plans/2026-08-07-three-platform-capability-implementation.md`

- [ ] **Step 1: Replace legacy benchmark fixtures**

  Use the approved v2 values and emit build type, map extent/resolution/level, expansions, p50/p95/max, peak memory, mode and reason. Ground benchmarks include the existing `50×50 m @ 0.20 m` plus a larger dyadic-map case; hopper includes 100 m direct, alternate-time-safe, and complete blocked cases.

- [ ] **Step 2: Enforce Release-only regression thresholds**

  Assert wheel `50×50 m p95 <= 2.0 s`; hopper direct `<=1.0 s`, alternate-time `<=2.0 s`, fully blocked `<=5.0 s`. Record, but do not use thresholds as production stop conditions. Freeze legged/large-map thresholds from the first verified Release baseline and document them before final qualification.

- [ ] **Step 3: Run the full qualification suite**

  Build to `/home/kai/CodexDownloads/lunar_navigation/three_platform_capability/final` with `-DCMAKE_BUILD_TYPE=Release`, then run:

  ```bash
  colcon test --packages-select lunar_navigation_msgs lunar_planning_msgs lunar_planner_core lunar_planner_ros
  colcon test-result --verbose
  python3 -m pytest -q tests/foundation tests/ros tests/differential tests/performance
  python3 tools/check_external_interfaces.py --config ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml
  python3 tools/check_platform_capability_freeze.py ros2_ws/src/lunar_navigation_config/config/three_platform_capability_freeze_v1.yaml
  python3 tools/check_repository_boundaries.py .
  python3 -m pytest -q tests/foundation/test_repository_boundaries.py
  git diff --check
  ```

- [ ] **Step 4: Record evidence and commit**

  The validation document must distinguish: approved engineering baseline, current Ubuntu simulation qualification, not Isaac dynamics acceptance, not formal PPO training completion, and not AGX acceptance.

  Commit: `docs: qualify three-platform capability implementation`

### Task 13: Integrate locally into `integration`

**Files:** Git history only.

- [ ] **Step 1: Re-run branch verification from a clean feature worktree**

  Confirm no untracked implementation artifacts and all Task 12 evidence is fresh.

- [ ] **Step 2: Fast-forward local integration**

  From `/mnt/data/WS/lunar-navigation`, preserve the user-owned `.vscode/` and run `git merge --ff-only three-platform-capability-implementation`.

- [ ] **Step 3: Verify merged HEAD**

  Re-run repository boundaries and focused capability/interface tests on `integration`; do not push any remote branch.
