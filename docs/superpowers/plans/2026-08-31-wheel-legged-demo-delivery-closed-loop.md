# Wheel And Legged Demo Delivery Closed Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在隔离 `/lunar_demo/*` 中实现轮式与足式共用的地图交付确认、路径身份绑定、12 m 局部前视和 4 m 停车刷新闭环，同时保持生产默认行为不变。

**Architecture:** `pure_plan_motion_server` 继续持有唯一 Action/滚动会话；新增 Demo 专用 ACK 与可执行段消息。`InputStore` 在协议模式下形成同 token 的局部图/odometry 快照，足式还要求同序列 traversability 投影就绪；Demo 状态只执行匹配 `request_id + map_token + segment_index` 的段，并在 4 m 或段结束时停车交付新地图。

**Tech Stack:** C++20、ROS 2 Jazzy、rclcpp/rclcpp_action、rosidl、GoogleTest、pytest/launch contracts、colcon/CTest。

**Spec:** `docs/superpowers/specs/2026-08-31-wheel-legged-rolling-demo-closed-loop-design.md`

## Global Constraints

- 新协议只在 `/lunar_demo/*` 使用，参数 `demo_delivery_protocol_enabled` 默认 `false`。
- 输入仍只有全局占据图、局部占据/高程图、odometry、TF 和目标；token 仅作身份，不作时间新鲜度判断。
- 轮式和足式名义局部目标均为 12 m，最多 32 个门户；Demo 每移动 4 m 停车刷新。
- 不增加发布前密集机身扫掠、二次轨迹认证、variance、观测时间窗或新传感器层。
- 不修改 HOPPER、Task3 生产接口或 `/Car/T5/Car_Cmd_Vel`。
- 750 m 随机障碍测试保留但标记 `SKIPPED_BY_USER`，不作为本计划门禁。
- 所有行为修改遵循 RED → GREEN → REFACTOR；每个提交只显式暂存列出的文件。

---

### Task 1: Demo 协议消息与执行状态机

**Files:**
- Create: `ros2_ws/src/lunar_planning_msgs/msg/DemoMapAck.msg`
- Create: `ros2_ws/src/lunar_planning_msgs/msg/DemoPlanSegment.msg`
- Modify: `ros2_ws/src/lunar_planning_msgs/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/include/lunar_pure_planner_ros/lunar_surface_demo_state.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_demo_state.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_ros/test/lunar_surface_demo_state_test.cpp`

**Interfaces:**
- Consumes: `nav_msgs/Path` 和已有 Demo 位置状态。
- Produces: `DemoMapAck`、`DemoPlanSegment`，以及 `LunarSurfaceDemoState::EnableDeliveryProtocol`、`HandleSegment`、`BeginMapDelivery`、`HandleMapAck`、`ShouldBeginMapDelivery`、`map_delivery_pending`。

- [ ] **Step 1: 写请求/地图/段身份的失败测试**

测试必须证明以下真实故障会被捕获：STOP 未清空旧路径、ACK 前提前移动、旧 request/token 的 EXECUTE 被接受、先到的有效 EXECUTE 在 ACK 后不能恢复。

```cpp
state.EnableDeliveryProtocol(DemoPlanSegment::WHEELED);
EXPECT_TRUE(state.HandleSegment(StopSegment("rviz-2")));
EXPECT_FALSE(state.can_advance());
state.BeginMapDelivery(Stamp(10));
EXPECT_TRUE(state.HandleSegment(ExecuteSegment("rviz-2", Stamp(10), 1U,
                                               StraightPath(12.0))));
state.Advance(0.5);
EXPECT_DOUBLE_EQ(state.x_m(), 0.0);
EXPECT_TRUE(state.HandleMapAck(Ack("rviz-2", Stamp(10))));
state.Advance(0.5);
EXPECT_DOUBLE_EQ(state.x_m(), 0.5);
EXPECT_FALSE(state.HandleSegment(ExecuteSegment("rviz-1", Stamp(9), 2U,
                                                StraightPath(12.0))));
```

- [ ] **Step 2: 构建目标测试并确认 RED**

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths ros2_ws/src --build-base build-jazzy-demo-delivery \
  --install-base install-jazzy-demo-delivery \
  --packages-select lunar_planning_msgs lunar_pure_planner_ros \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Expected: FAIL，缺少 `DemoMapAck`/`DemoPlanSegment` 或新状态 API。

- [ ] **Step 3: 增加最小消息定义**

`DemoMapAck.msg`：

```text
uint8 WHEELED=1
uint8 LEGGED=2
string request_id
builtin_interfaces/Time map_token
uint64 local_input_sequence
uint64 projection_revision
uint8 platform_type
```

`DemoPlanSegment.msg`：

```text
uint8 WHEELED=1
uint8 LEGGED=2
uint8 STOP=1
uint8 EXECUTE=2
string request_id
builtin_interfaces/Time map_token
uint32 segment_index
uint8 platform_type
uint8 command
nav_msgs/Path executable_path
```

- [ ] **Step 4: 实现最小 Demo 状态机**

状态为 `kLegacy/kIdle/kWaitingForMap/kWaitingForAck/kWaitingForSegment/kExecuting`。STOP 采用新 request 并清路径；EXECUTE 必须匹配 platform/request/token 且 segment index 递增。ACK 前到达的有效 EXECUTE 只缓存不执行，匹配 ACK 到达后再激活。`ShouldBeginMapDelivery(4.0)` 在累计 4 m 或当前段耗尽时返回 true。

- [ ] **Step 5: 运行 GREEN 测试并提交**

Run:

```bash
source /opt/ros/jazzy/setup.bash
source install-jazzy-demo-delivery/setup.bash
build-jazzy-demo-delivery/lunar_pure_planner_ros/lunar_surface_demo_state_test
```

Expected: all tests PASS。

Commit:

```bash
git add ros2_ws/src/lunar_planning_msgs/msg/DemoMapAck.msg \
  ros2_ws/src/lunar_planning_msgs/msg/DemoPlanSegment.msg \
  ros2_ws/src/lunar_planning_msgs/CMakeLists.txt \
  ros2_ws/src/lunar_pure_planner_ros/include/lunar_pure_planner_ros/lunar_surface_demo_state.hpp \
  ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_demo_state.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/lunar_surface_demo_state_test.cpp
git commit -m "feat: add demo delivery state protocol"
```

### Task 2: 同 token 输入快照与足式投影幂等

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_ros/include/lunar_pure_planner_ros/input_store.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/input_store.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/include/lunar_pure_planner_ros/traversability_input.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/traversability_input.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_ros/test/input_store_test.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_ros/test/traversability_input_test.cpp`

**Interfaces:**
- Consumes: 局部 GridMap 与 odometry 的 header stamp。
- Produces: `InputStore(bool token_idempotent)`、`CaptureSynchronized()`、`local_arrival_sequence`；`TraversabilityInputSnapshot.local_sequence/local_map_stamp`。

- [ ] **Step 1: 写同版快照和重复 token 失败测试**

```cpp
InputStore store{true};
store.UpdateLocal(LocalMap(10));
store.UpdateOdometry(StampedOdometry(9));
EXPECT_FALSE(store.CaptureSynchronized().has_value());
store.UpdateOdometry(StampedOdometry(10));
ASSERT_TRUE(store.CaptureSynchronized().has_value());
const auto first = *store.CaptureSynchronized();
store.UpdateLocal(LocalMap(10));
const auto duplicate = *store.CaptureSynchronized();
EXPECT_EQ(duplicate.local_sequence, first.local_sequence);
EXPECT_GT(duplicate.local_arrival_sequence, first.local_arrival_sequence);
```

足式测试连续两次提交相同 token，断言 `local_sequence`、投影 revision 和 persistent-map 更新次数不变；提交新 token 后三者前进。

- [ ] **Step 2: 运行测试确认 RED**

Run:

```bash
cmake --build build-jazzy-demo-delivery/lunar_pure_planner_ros \
  --target input_store_test traversability_input_test -j2
```

Expected: FAIL，缺少同步捕获、arrival sequence 和投影身份。

- [ ] **Step 3: 实现可选幂等模式**

默认构造保持现有“每次到达序列递增”。只有 `token_idempotent=true` 时，相同 stamp 的局部图不增加 `local_sequence`，但增加 `local_arrival_sequence`；`CaptureSynchronized()` 仅在局部图与 odometry stamp 完全一致时返回快照。`TraversabilityInput` 采用同样的唯一 token 规则，并把已应用 local sequence/stamp 返回给调用者。

- [ ] **Step 4: 运行 GREEN 与既有兼容测试并提交**

Run:

```bash
build-jazzy-demo-delivery/lunar_pure_planner_ros/input_store_test
build-jazzy-demo-delivery/lunar_pure_planner_ros/traversability_input_test
```

Expected: both PASS，包括既有 `LastArrivalWinsRegardlessOfStamp` 默认模式测试。

Commit:

```bash
git add ros2_ws/src/lunar_pure_planner_ros/include/lunar_pure_planner_ros/input_store.hpp \
  ros2_ws/src/lunar_pure_planner_ros/src/input_store.cpp \
  ros2_ws/src/lunar_pure_planner_ros/include/lunar_pure_planner_ros/traversability_input.hpp \
  ros2_ws/src/lunar_pure_planner_ros/src/traversability_input.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/input_store_test.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/traversability_input_test.cpp
git commit -m "feat: capture synchronized demo map inputs"
```

### Task 3: 规划器 ACK、执行段与恢复闭环

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp`

**Interfaces:**
- Consumes: Task 2 的同步快照与足式投影 sequence。
- Produces: `/lunar_demo/map_ack` 和 `/lunar_demo/plan_segment`；参数 `demo_delivery_protocol_enabled`、`demo_map_ack_topic`、`demo_plan_segment_topic`、`demo_map_ack_timeout_ms`、`rolling_replan_distance_m`。

- [ ] **Step 1: 扩展 RunningSystem 并写协议失败测试**

测试观察真实 ROS topic 和真实 planner 调用，覆盖：

```cpp
// 新目标先收到 STOP；旧输入不能触发 local planner。
auto handle = system.SendGoal(system.Goal("delivery-1"));
ASSERT_TRUE(WaitFor([&] { return system.PlanSegments().size() == 1U; }));
EXPECT_EQ(system.PlanSegments().front().command, DemoPlanSegment::STOP);
EXPECT_EQ(local_calls.load(), 0U);

// 只有相同 token 的 local+odom 和足式投影就绪后才 ACK/规划。
system.PublishStampedLocalAndOdometry(10, 0.0);
ASSERT_TRUE(WaitFor([&] { return system.MapAcks().size() == 1U; }));
EXPECT_EQ(system.MapAcks().front().request_id, "delivery-1");
ASSERT_TRUE(WaitFor([&] { return system.ExecuteSegments().size() == 1U; }));
```

另写三个行为测试：不同 stamp 不 ACK；相同 token 重发只重发 ACK、不重复规划；替换目标后旧请求不再发布 EXECUTE。

- [ ] **Step 2: 运行目标测试确认 RED**

Run:

```bash
cmake --build build-jazzy-demo-delivery/lunar_pure_planner_ros \
  --target pure_plan_motion_server_test -j2
build-jazzy-demo-delivery/lunar_pure_planner_ros/pure_plan_motion_server_test \
  --gtest_filter='PurePlanMotionServer.DemoDelivery*'
```

Expected: FAIL，协议 publisher/参数/等待逻辑不存在。

- [ ] **Step 3: 增加默认关闭的协议参数和 publisher**

仅当 `demo_delivery_protocol_enabled=true` 时创建两个可靠 publisher，并让 `InputStore`、`TraversabilityInput` 启用 token 幂等。Action 开始立即发布 STOP，记录开始前 local sequence，要求下一版同步 token 才能 ACK。

- [ ] **Step 4: 在滚动循环中实现 ACK 门控**

协议模式下先用最新 odometry 判断最终目标，再等待 `CaptureSynchronized()` 返回新 local sequence。足式还要求 `traversability.local_sequence == snapshot.local_sequence`。就绪后发布 ACK、冻结该快照并规划；成功后发布绑定同一 token 的递增 EXECUTE 段。

- [ ] **Step 5: 实现 ACK 超时和局部失败恢复**

ACK 等待使用 `demo_map_ack_timeout_ms`，超时 reason 为 `MAP_ACK_TIMEOUT`。ACK 超时、`NO_PATH`、`TIMED_OUT` 和旧窗口输入在协议模式均执行：清旧路径、发 STOP、等待新 token，最多复用 `rolling_transient_retry_limit=2`；全局失败、配置错误和取消保持终止。

- [ ] **Step 6: 增加诊断身份并运行 GREEN**

诊断增加 `demo_delivery_state`、`demo_map_token`、`demo_segment_index`、`local_input_sequence` 和恢复 stage，不改现有成功三元组。

Run:

```bash
build-jazzy-demo-delivery/lunar_pure_planner_ros/pure_plan_motion_server_test \
  --gtest_filter='PurePlanMotionServer.DemoDelivery*'
```

Expected: protocol tests PASS。

- [ ] **Step 7: 运行完整 server 测试并提交**

Run:

```bash
build-jazzy-demo-delivery/lunar_pure_planner_ros/pure_plan_motion_server_test
```

Expected: all existing and new tests PASS。

Commit:

```bash
git add ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp
git commit -m "feat: close demo map delivery loop"
```

### Task 4: Demo 节点接入协议并在 4 m 停车

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_demo_node.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/CMakeLists.txt`
- Modify: `launch/lunar_surface_rviz_demo.launch.py`
- Test: `ros2_ws/src/lunar_pure_planner_ros/test/lunar_surface_demo_state_test.cpp`
- Test: `tests/test_launch_contract.py`

**Interfaces:**
- Consumes: Task 1 消息和状态 API。
- Produces: 固定 token 的局部图/odometry 重发，ACK 后执行，4 m 或段结束停车刷新。

- [ ] **Step 1: 写 4 m 停车和 launch 隔离失败测试**

状态测试验证执行到 3.5 m 可继续，执行到 4.0 m 后 `ShouldBeginMapDelivery(4.0)` 为 true，`BeginMapDelivery` 立即清路径且后续 `Advance` 不移动。launch contract 验证只有 `lunar_surface_rviz_demo.launch.py` 将协议设为 true，`pure_planner.launch.py` 和 `config/pure_planner.yaml` 默认 false。

- [ ] **Step 2: 运行测试确认 RED**

Run:

```bash
build-jazzy-demo-delivery/lunar_pure_planner_ros/lunar_surface_demo_state_test
python3 -m pytest tests/test_launch_contract.py -q
```

Expected: FAIL，Demo launch 未声明协议参数。

- [ ] **Step 3: 修改 Demo 发布周期**

协议关闭时保持现有路径订阅与运动。协议开启时只让 `DemoPlanSegment` 解除停止：STOP 或 4 m/段结束进入地图交付，首次生成 token 后用同一 stamp 重发局部图和停止 odometry；匹配 ACK 后停止重发；匹配 EXECUTE 后才恢复 0.5 m/tick 运动。普通 wheeled/legged path 只用于 RViz，不再授权移动。

- [ ] **Step 4: 启用 Demo 参数并运行 GREEN**

在 Demo launch 同时给 planner 与 demo node 设置：

```python
"demo_delivery_protocol_enabled": True,
"demo_map_ack_topic": "/lunar_demo/map_ack",
"demo_plan_segment_topic": "/lunar_demo/plan_segment",
"rolling_horizon_m": 12.0,
"rolling_replan_distance_m": 4.0,
```

Run:

```bash
colcon build --base-paths ros2_ws/src --build-base build-jazzy-demo-delivery \
  --install-base install-jazzy-demo-delivery \
  --packages-select lunar_planning_msgs lunar_pure_planner_ros \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
source install-jazzy-demo-delivery/setup.bash
build-jazzy-demo-delivery/lunar_pure_planner_ros/lunar_surface_demo_state_test
python3 -m pytest tests/test_launch_contract.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_demo_node.cpp \
  ros2_ws/src/lunar_pure_planner_ros/CMakeLists.txt \
  ros2_ws/src/lunar_pure_planner_ros/test/lunar_surface_demo_state_test.cpp \
  launch/lunar_surface_rviz_demo.launch.py tests/test_launch_contract.py
git commit -m "feat: gate demo motion on delivered segments"
```

### Task 5: 统一 12 m 门户与 4 m 重规划距离

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_core/src/hierarchical/surface_portal_set.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_core/test/surface_portal_set_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp`
- Modify: `config/pure_planner.yaml`

**Interfaces:**
- Consumes: `rolling_horizon_m=12.0`、`rolling_replan_distance_m=4.0`。
- Produces: 两平台相同的 12→1 m 门户范围与稳定排序；平台专用过滤/边认证保持不变。

- [ ] **Step 1: 写统一门户失败测试**

core 测试构造相同路线与地图，分别用轮式/足式 capability，断言候选均按“进度降序、绝对横向偏移升序、净空降序、stable rank”排序，最远进度为 12 m、最短不小于 1 m、数量不超过 32。ROS 测试捕获两平台首个 `LocalGoalSet`，断言最大路线前视为 12 m。

- [ ] **Step 2: 运行测试确认 RED**

Run:

```bash
cmake --build build-jazzy-demo-delivery/lunar_pure_planner_core \
  --target lunar_pure_planner_core_surface_portal_set_test -j2
build-jazzy-demo-delivery/lunar_pure_planner_core/lunar_pure_planner_core_surface_portal_set_test
```

Expected: wheel 的同进度排序仍优先净空，足式 server 仍使用硬编码 4 m。

- [ ] **Step 3: 实现统一排序和范围**

`CandidateLess` 对两平台均先比较绝对横向偏移；纵向回退的下界为当前进度前方 1 m，仍最多采样 32 个。删除 `kLeggedRollingHorizonM` 和 `kLeggedReplanStrideM`，两平台均使用参数 12 m/4 m；非协议滚动也保持参数化行为。

- [ ] **Step 4: 运行 GREEN 与核心回归并提交**

Run:

```bash
build-jazzy-demo-delivery/lunar_pure_planner_core/lunar_pure_planner_core_surface_portal_set_test
build-jazzy-demo-delivery/lunar_pure_planner_ros/pure_plan_motion_server_test
```

Expected: PASS。

Commit:

```bash
git add ros2_ws/src/lunar_pure_planner_core/src/hierarchical/surface_portal_set.cpp \
  ros2_ws/src/lunar_pure_planner_core/test/surface_portal_set_test.cpp \
  ros2_ws/src/lunar_pure_planner_ros/src/pure_plan_motion_server.cpp \
  ros2_ws/src/lunar_pure_planner_ros/test/pure_plan_motion_server_test.cpp \
  config/pure_planner.yaml
git commit -m "feat: unify rolling portals at twelve metres"
```

### Task 6: 端到端非简单场景、文档与最终验证

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_ros/test/lunar_surface_scenario_test.cpp`
- Modify: `docs/操作指令.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-31-wheel-legged-rolling-demo-closed-loop-design.md`
- Modify: `docs/superpowers/plans/2026-08-31-wheel-legged-demo-delivery-closed-loop.md`

**Interfaces:**
- Consumes: 完整协议与 deterministic seed `20260823`。
- Produces: 可复制的 wheel/legged headless/RViz 命令、困难目标选择证据和完整验证记录。

- [ ] **Step 1: 增加困难目标选择行为测试**

在固定场景中选取三个已知 FREE、与起点连通且距离超过 100 m 的目标；测试用离散 supercover 证明起终点直线至少穿过一个 occupied cell，排除开阔短直线。若某个文字候选不满足约束，测试本身失败，必须改为另一个经同一判据验证的固定 cell，不能放宽判据。

- [ ] **Step 2: 构建并运行完整自动测试**

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths ros2_ws/src --build-base build-jazzy-demo-delivery \
  --install-base install-jazzy-demo-delivery \
  --packages-select lunar_planning_msgs lunar_pure_planner_core lunar_pure_planner_ros \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
build-jazzy-demo-delivery/lunar_pure_planner_core/lunar_pure_planner_core_anytime_wheel_planner_test \
  --gtest_filter='-WheelPlanner.GlobalRouteWithEightMeterRollingHorizonReaches750MeterGoalThroughRandomObstacles'
ctest --test-dir build-jazzy-demo-delivery/lunar_pure_planner_core \
  -E '^lunar_pure_planner_core_anytime_wheel_planner_test$' --output-on-failure -j1
source install-jazzy-demo-delivery/setup.bash
ctest --test-dir build-jazzy-demo-delivery/lunar_pure_planner_ros \
  --output-on-failure -j1
python3 -m pytest tests/test_isolation_contract.py tests/test_launch_contract.py -q
```

Expected: all selected tests PASS；750 m 明确 `SKIPPED_BY_USER`。

- [ ] **Step 3: 运行 wheel 与 legged headless Demo**

分别启动一次且不重启进程，依次发布 Task 6 Step 1 固定的三个困难目标。每个平台记录至少一个任务满足路线 ≥100 m 或刷新 ≥10 次，并检查每段 request/token/index 单调、ACK 前无移动、正式成功三元组成立。命令以 `timeout` 限界，日志写到仓库外 `/tmp`。

- [ ] **Step 4: 更新操作文档和设计实施证据**

只记录实际运行结果；Jazzy、RViz/headless、Humble、Orin、DDS、rosbag、实车分别标记。未运行层级写 `NOT_RUN`，不以本机构建替代。

- [ ] **Step 5: 静态检查并提交**

Run:

```bash
git diff --check
rg -n '^(<<<<<<<|=======|>>>>>>>)' README.md docs config launch ros2_ws/src tests
git status --short
```

Commit:

```bash
git add ros2_ws/src/lunar_pure_planner_ros/test/lunar_surface_scenario_test.cpp \
  README.md docs/操作指令.md \
  docs/superpowers/specs/2026-08-31-wheel-legged-rolling-demo-closed-loop-design.md \
  docs/superpowers/plans/2026-08-31-wheel-legged-demo-delivery-closed-loop.md
git commit -m "docs: record demo delivery closed loop verification"
```

## Plan Self-Review

- Spec coverage: Tasks 1–5 cover protocol, token identity, ACK, segment gating, recovery, preemption, 12 m/4 m and diagnostics；Task 6 covers non-simple tests and evidence boundaries。
- Placeholder scan: no placeholder marker or unspecified implementation step remains；困难目标必须通过 executable test 判据选择。
- Type consistency: message constants and fields are introduced in Task 1 and consumed unchanged by Tasks 3–4；Task 2 sequences are consumed by Task 3；12 m/4 m parameter names are shared by Tasks 3–5。
- Non-goals: no HOPPER、Task3 production topic、vehicle controller、new terrain data or dense post-plan certification changes。
