# 足式局部搜索契约对齐实施计划

> **执行要求：** 按任务顺序使用 `superpowers:executing-plans` 执行；每项功能修改遵循测试先行。只实现本文列出的范围。

**目标：** 让现有足式 Grid V1 局部规划一次搜索完整门户集合，并采用请求级状态格、足式多目标距离场和动态状态编号，消除首门户失败与固定状态槽耗尽，同时保持轮式外部行为不变。

**架构：** 门户层先把足式中间目标吸附到对应局部安全格中心。共享距离场增加“地图 + 可行掩码”入口，足式以 `hard_feasible && step_feasible` 构建多源场。`PlanLegged` 接收完整 `LocalGoalSet`，用请求起点/起始航向量化状态，单次 ARA* 为多个目标建立动态终端节点，并返回实际目标索引。沿用现有动作原语、边认证、deadline、取消和代价模型。

**技术栈：** C++20、ROS 2 Jazzy、ament/colcon、GoogleTest/CTest、现有 ARA* 与 active planner cache。

**设计依据：** `docs/superpowers/specs/2026-08-31-legged-grid-v1-planning-design.md`

## 范围约束

- 保留现有足式 64/128 航向格、动作原语、单键单状态、两级边认证和输出语义。
- 不增加路线走廊、多标签状态、preferred path、新内存配置、发布前复核或二次轨迹认证。
- 不修改轮式/Hopper 搜索行为、控制器或 `/lunar_demo/*` 接口。
- 轮式继续调用现有 `BuildGoalDistanceField(LocalTerrainProjection, ...)`；新增入口不得改变其结果。
- 本机结果只作为 Jazzy 证据；Humble、Orin、DDS、rosbag 和实车未运行时标记 `NOT_RUN`。

---

## Task 1：稳定足式门户并吸附中间目标

**文件：**

- 修改：`ros2_ws/src/lunar_pure_planner_core/src/hierarchical/surface_portal_set.hpp`
- 修改：`ros2_ws/src/lunar_pure_planner_core/src/hierarchical/surface_portal_set.cpp`
- 修改：`ros2_ws/src/lunar_pure_planner_core/src/hierarchical/global_route_planner.hpp`
- 修改：`ros2_ws/src/lunar_pure_planner_core/src/hierarchical/global_route_planner.cpp`
- 修改：`ros2_ws/src/lunar_pure_planner_core/test/surface_portal_set_test.cpp`

- [x] **1.1 先补转换失败测试**

  在已注册的 `surface_portal_set_test.cpp` 构造全局格中心落在局部格边界的足式中间门户，断言转换后的 `PointGoal.position_m` 等于局部格中心，容差为 `0.5 * local_resolution - epsilon` 且大于零；再断言精确最终目标的位置、容差和朝向原样保留。

- [x] **1.2 运行门户相关测试并确认 RED**

  ```bash
  source /opt/ros/jazzy/setup.bash
  colcon build --base-paths ros2_ws/src --packages-select lunar_pure_planner_core \
    --build-base build-jazzy-legged-search --install-base install-jazzy-legged-search \
    --log-base /tmp/lunar-legged-search-log --cmake-args -DBUILD_TESTING=ON
  ctest --test-dir build-jazzy-legged-search/lunar_pure_planner_core \
    -R 'surface_portal_set|global_route_planner' --output-on-failure
  ```

  预期：新的局部格中心/非零容差断言失败；现有足式同进度中心线排序测试保留。

- [x] **1.3 实现最小转换修正**

  仅对足式且 `exact_final_goal == false` 的门户使用其 `local_cell` 中心生成目标，容差设为半格宽减小量；精确最终目标分支不变。保留当前足式“同路线进度时横向偏移绝对值优先”的排序，轮式排序和中间目标转换均保持原行为。

- [x] **1.4 验证并提交 Task 1**

  运行上述两个测试目标和 `git diff --check`，显式暂存五个文件，提交：

  ```text
  fix: stabilize legged local portal targets
  ```

---

## Task 2：增加足式必要可行掩码距离场

**文件：**

- 修改：`ros2_ws/src/lunar_pure_planner_core/src/shared/goal_distance_field.hpp`
- 修改：`ros2_ws/src/lunar_pure_planner_core/src/shared/goal_distance_field.cpp`
- 修改：`ros2_ws/src/lunar_pure_planner_core/test/local_terrain_projection_test.cpp`

- [x] **2.1 先补掩码距离场测试**

  增加覆盖以下行为的测试：多目标源记录稳定 `nearest_goal_index`；`free_with_height` 可通但传入掩码阻断的区域不可达；对角移动仍禁止穿过两个侧向阻挡格；取消/deadline 返回空结果。保留现有轮式入口测试作为兼容性证据。

- [x] **2.2 运行目标并确认 RED**

  ```bash
  ctest --test-dir build-jazzy-legged-search/lunar_pure_planner_core \
    -R local_terrain_projection --output-on-failure
  ```

  预期：缺少接收 `MapSnapshot + span<uint8_t> feasibility_mask` 的构建入口而编译失败。

- [x] **2.3 实现共享的最小重载**

  新增按 `MapSnapshot`、与 `cell_count` 等长的只读可行掩码和有序目标格构建距离场的重载。传播规则、稳定源索引、对角穿角规则和 `SearchControl` 与现有实现一致；旧 `LocalTerrainProjection` 重载只委托给新入口及 `free_with_height`，不改变轮式调用方。

- [x] **2.4 验证并提交 Task 2**

  运行 `local_terrain_projection` 与 `anytime_wheel_planner` 测试、`git diff --check`，显式暂存三个文件，提交：

  ```text
  feat: support masked multi-goal distance fields
  ```

---

## Task 3：对齐足式多目标搜索、状态格和资源模型

**文件：**

- 修改：`ros2_ws/src/lunar_pure_planner_core/src/legged/anytime_legged_planner.hpp`
- 修改：`ros2_ws/src/lunar_pure_planner_core/src/legged/anytime_legged_planner.cpp`
- 修改：`ros2_ws/src/lunar_pure_planner_core/test/anytime_legged_planner_test.cpp`

- [ ] **3.1 先补足式搜索契约测试**

  增加以下失败测试：

  1. 第一个门户被精确边认证拒绝、第二个门户可达时，一次 `PlanLegged` 成功并返回 `selected_goal_index == 1`；
  2. 起点在足式掩码距离场上与全部目标断开时返回 `LEGGED_NO_PATH`，不展开状态；
  3. 同一局部几何连同起终点平移，及围绕起点整体旋转后，得到相同状态数量、路径动作数和近似相等代价；
  4. 人工构造超过旧 `131072` 普通状态编号的图增长场景不再返回 `LEGGED_SEARCH_CAPACITY_EXHAUSTED`；
  5. 模拟/捕获真实 `std::bad_alloc` 时返回 `LEGGED_RESOURCE_EXHAUSTED`，而普通 OPEN 耗尽仍返回 `LEGGED_NO_PATH`。

- [ ] **3.2 运行足式测试并确认 RED**

  ```bash
  ctest --test-dir build-jazzy-legged-search/lunar_pure_planner_core \
    -R anytime_legged_planner --output-on-failure
  ```

  预期：`LeggedPlanRequest` 尚不接受完整目标集合/距离场，结果也没有目标索引。

- [ ] **3.3 修改请求与结果契约**

  将 `goal_odom` 改为 `goals_odom: LocalGoalSet`，增加只读 `goal_distance_field`，并在 `LeggedPlanResult` 增加 `std::optional<std::size_t> selected_goal_index`。拒绝空集合、超过现有 32 个中间目标、非 PointGoal、距离场尺寸/源索引不匹配等非法输入；精确最终目标仍限定为一个。

- [ ] **3.4 使用请求级状态格**

  在搜索图构造时冻结起点 `x/y/yaw`。`KeyFor` 按 `(pose.xy - request_start.xy)` 和相对起始 yaw 生成现有 `x/y/yaw/mode/narrow` 键；地图原点只用于 `PositionToCell` 查询。保持实际状态中的世界位姿和所有边几何不变。

- [ ] **3.5 实现单次多目标 ARA***

  主启发为到全部目标区域的欧氏下界最小值；距离场按当前状态所在栅格提供非约束 `guidance`。每个目标拥有带 `goal_index` 的动态终端节点，状态展开仅尝试一个动作原语范围内的现有缩放连接，并沿用完整 `SweepBody` 认证。重建路径时从终端节点取出实际目标索引。

- [ ] **3.6 删除固定状态槽**

  删除 `kMaximumSearchStates`、预留的固定 `goal_state_` 和容量耗尽分支。普通状态及终端状态按首次创建顺序连续编号，并让 ARA* 使用动态增长的逻辑状态范围。捕获 `std::bad_alloc` 返回 `LEGGED_RESOURCE_EXHAUSTED`；不得新增状态数/内存预算参数。

- [ ] **3.7 验证并提交 Task 3**

  运行 `anytime_legged_planner`、`legged_fault_matrix`、`legged_planner` 目标和 `git diff --check`，显式暂存三个文件，提交：

  ```text
  feat: align legged local search contracts
  ```

---

## Task 4：接入核心规划器并保留失败诊断

**文件：**

- 修改：`ros2_ws/src/lunar_pure_planner_core/src/planner.cpp`
- 修改：`ros2_ws/src/lunar_pure_planner_core/test/dual_mode_planner_test.cpp`
- 修改：`ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp`
- 修改：`ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp`

- [ ] **4.1 先补核心集成测试**

  在 `dual_mode_planner_test.cpp` 构造两个足式局部目标，断言规划器把完整集合交给足式搜索、返回实际 `selected_goal_index`，相同地图/capability/有序目标集合第二次请求命中现有 goal-field cache；改变目标顺序或 capability 后重建。断言轮式相同场景的结果不变。

- [ ] **4.2 先补 ROS 失败诊断测试**

  在 `pure_plan_motion_server_test.cpp` 让局部后端返回失败且携带非零 `expanded_states`、`best_cost`、`selected_goal_index` 与 `legged_local` 统计，断言 server 转换后的 segment 保留这些字段，而 status/reason 仍是原失败结果。

- [ ] **4.3 运行集成目标并确认 RED**

  ```bash
  ctest --test-dir build-jazzy-legged-search/lunar_pure_planner_core \
    -R dual_mode_planner --output-on-failure
  ```

  ROS 包构建后运行：

  ```bash
  ctest --test-dir build-jazzy-legged-search/lunar_pure_planner_ros \
    -R pure_plan_motion_server_test --output-on-failure
  ```

- [ ] **4.4 接入足式距离场和完整目标集合**

  在 `planner.cpp` 从全部足式目标提取目标格，以 `hard_feasible && step_feasible` 生成掩码，复用现有 ordered-goal cache key 和 cache slot。若起点距离为无穷，返回局部 `NO_PATH`。调用 `PlanLegged` 时传完整 `LocalGoalSet` 和距离场，并原样传播 `selected_goal_index`、缓存命中与现有边认证诊断。轮式分支不改。

- [ ] **4.5 修复 ROS 外层失败证据丢失**

  `LocalFailure(...)` 转换后只回填现有结果字段：`expanded_states`、`best_cost`、`selected_goal_index`、`legged_local`。不增加新的安全检查或诊断类别。

- [ ] **4.6 验证并提交 Task 4**

  分别构建 core 与 ROS 包，运行上述两个目标、`git diff --check`，显式暂存四个文件，提交：

  ```text
  feat: integrate legged multi-goal local planning
  ```

---

## Task 5：必要回归与 RViz 演示复核

- [ ] **5.1 运行 core 包串行测试**

  ```bash
  source /opt/ros/jazzy/setup.bash
  colcon build --base-paths ros2_ws/src --packages-select lunar_pure_planner_core \
    --build-base build-jazzy-legged-search --install-base install-jazzy-legged-search \
    --log-base /tmp/lunar-legged-search-log --cmake-args -DBUILD_TESTING=ON
  ctest --test-dir build-jazzy-legged-search/lunar_pure_planner_core \
    --output-on-failure -j1
  ```

- [ ] **5.2 运行 ROS 受影响包测试**

  ```bash
  source /opt/ros/jazzy/setup.bash
  source install-jazzy-legged-search/setup.bash
  colcon build --base-paths ros2_ws/src --packages-select lunar_pure_planner_ros \
    --build-base build-jazzy-legged-search --install-base install-jazzy-legged-search \
    --log-base /tmp/lunar-legged-search-log --cmake-args -DBUILD_TESTING=ON
  ctest --test-dir build-jazzy-legged-search/lunar_pure_planner_ros \
    --output-on-failure -j1
  python3 -m pytest tests/contract -q
  ```

  若仓库不存在 `tests/contract`，记录为不适用并运行仓库现有与 pure planner 对应的契约测试路径，不扩展到无关历史失败。

- [ ] **5.3 运行已有 RViz demo**

  使用仓库现有足式月面 demo 启动方式，不新增 launch 或控制接口。记录至少两轮滚动规划，正式成功条件为 `planning_outcome: 0`、`reason_code: PLAN_FOUND`、`has_reference: true`；同时确认局部路径不再因首门户排序形成稳定平行偏移，后续请求不再出现 `LEGGED_SEARCH_CAPACITY_EXHAUSTED`。

- [ ] **5.4 最终审查与记录边界**

  仅检查本计划文件清单、`git diff --check`、目标测试与 demo 日志。更新本计划复选框和设计文档中的实际 Jazzy 证据；Humble、Orin、DDS、rosbag、实车保持 `NOT_RUN`。不合并、不推送、不创建 PR。
