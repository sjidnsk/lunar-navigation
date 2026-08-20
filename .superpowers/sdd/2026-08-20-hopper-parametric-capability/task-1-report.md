# Task 1 报告：hopper-v1 参数化能力

## 结果

已实现受追踪的 `hopper/hopper-v1` 参数化能力候选；未改动 canonical freeze YAML 或其 digest，未安装任何 `/opt` 活动文件，也未推送或合并。

## 修改文件与设计映射

- `deployment/config/hopper.yaml`：批准的无 URDF/mesh 数值文件；`reference_remaining_usable_fuel_mass_kg` 仅映射到固定、不可递减的单跳参考推进剂量。
- `ros2_ws/src/lunar_planner_ros/src/capability_loader.cpp` 与其测试：`hopper/hopper-v1` 参数模式对身份、根/HOPPER keys、来源 keys/值、月面重力、参考量和 `runtime_fallback_allowed: false` fail-closed；legacy URDF HOPPER 路径保持旧 key 兼容。
- `platform_capability.hpp`、`projection_cache.cpp`、HOPPER planner/graph/reachability projection：完整 typed fields、投影 hash/primitive bytes，并由 configured gravity 驱动弹道认证。
- `python_bindings.cpp`、`capability_freeze.py`、`project_capability.py`：pybind 与正式 freeze adapter 传递完整 HOPPER projection；legacy external-bundle HOPPER 的旧字段集合保留 lunar/default single-hop 兼容映射。
- `deployment/docs/README.runtime.md`、`deployment/docs/COMMANDS.runtime.md`：wheel、legged、hopper 三选一活动选择说明；仅文档命令，未执行安装。

## TDD RED -> GREEN

1. RED：`python3 -m pytest -q tests/deployment/test_hopper_capability.py`
   - `1 failed`；原因是预期的 `deployment/config/hopper.yaml` 尚不存在。
2. RED：在 `ros2-humble` clean copy 中先以不完整 `--packages-select` 构建，失败于缺失消息包；改用 `colcon build --packages-up-to lunar_planner_ros` 后，失败于 `HopperCapability` 缺少 `gravity_mps2`、reference range/elevation 与 fallback typed fields，属于所需功能尚未实现。
3. RED：`PYTHONPATH=training/lunar_policy_training python3 -m pytest -q training/lunar_policy_training/tests/test_project_capability.py::test_project_formal_capability_uses_approved_freeze`
   - `1 failed`，`FrozenHopperCapability.gravity_mps2` 不存在。
4. GREEN：容器 `colcon build --packages-up-to lunar_planner_ros lunar_planner_training_bridge --cmake-args -DBUILD_TESTING=ON`：`5 packages finished`。
5. GREEN：loader 过滤器：`21 tests, 0 errors, 0 failures`。
6. GREEN：HopperPlanner、loader、PlanMotionServer 选择回归：`57 tests, 0 errors, 0 failures`。
7. GREEN：直接运行新增 projection 行为测试：`1 test ... PASSED`；新增 pybind projection 测试：`1 passed, 40 deselected`。
8. GREEN：带 Humble overlay 的 formal adapter/project-adapter tests：`50 passed`。
9. GREEN：`python3 -m pytest -q tests/foundation/test_platform_capability_freeze.py tests/deployment/test_hopper_capability.py`：`15 passed`；project adapter：`3 passed`。

## Freeze 不变证据

```
python3 tools/check_platform_capability_freeze.py \
  --schema ros2_ws/src/lunar_navigation_config/config/platform_capability_schema_v2.yaml \
  --freeze ros2_ws/src/lunar_navigation_config/config/three_platform_capability_freeze_v1.yaml
capability freeze: OK sha256=60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95
```

## 已知失败与归因

- clean-copy reachability projection 完整二进制存在既有失败：`GroundMaskIsTheCppStartConnectedComponent` 仍期待 `cpp-ground-start-connected-component/v1`，实际为 `cpp-ground-global-cost-tree/v1`。新增 `HopperProjectionUsesConfiguredGravity` 同次通过；失败不涉及 HOPPER 或本次改动。
- 主机直接跑 external formal-bundle test 时缺少 `lunar_planner_training_bridge`；在 Humble overlay 中补齐 `AMENT_PREFIX_PATH`、`PYTHONPATH`、`LD_LIBRARY_PATH` 后 50 项通过，故为主机测试环境缺失而非产品失败。
- `clang-format` 在宿主不存在；已通过 `git diff --check` 与容器 C++ 编译验证格式/编译完整性。

## 自审

- 参数 HOPPER 路径 exact-key/source fail-closed；wrong identity/version/type/base-frame、extra root/HOPPER/source、missing source、unsupported geometry、URDF 混入、无效燃料/重力及 fallback=true 均有真实 loader 行为断言。
- 旧 URDF HOPPER parser 使用旧 `reference_propellant_mass_kg` 路径，未被参数模式覆盖；wheel/legged 选择回归通过。
- `gravity_mps2` 已进入 HopperPlanner、HopperReachabilityGraph、reachability projection、cache fingerprint、graph primitive bytes 和 pybind/formal projection；两个新行为测试可捕捉重力只存储而未消费的缺陷。
- 未引入实时燃料状态、订阅、跨跳递减或 runtime fallback。

## Fix round 1/5：YAML 重复键与亚容差冻结漂移

### TDD RED

在 `ros2-humble` clean-copy 中，仅补测试后运行：

```bash
.../lunar_planner_ros_capability_loader_test \
  --gtest_filter='CapabilityLoader.RejectsDuplicateKeysInParametricHopperDocuments:CapabilityLoader.RejectsSubToleranceParametricHopperFrozenValueDrift'
```

结果 `2 FAILED TESTS`：重复 root/platform/geometry/hopper key 的四个 mutation 都被 loader 错误接受；`specific_impulse_s: 301.0000000000005` 也被错误接受。这直接复现 yaml-cpp 首值读取与 `Near(..., 1e-12)` 漂移漏洞，不是 fixture 或依赖失败。

### 最小修复与 GREEN

- `RejectUnexpectedKeys` 现在维护已见 key 集合，并对重复 key 返回 `CAPABILITY_SCHEMA_INVALID`；该路径统一保护所有已调用的 root、platform、geometry 和 typed section exact-key 验证。
- parametric `hopper-v1` 的批准数值改为精确 IEEE double 比较；不再将非 canonical 的亚容差值写入 typed capability/hash。

同一聚焦过滤器 GREEN：`2 tests, 0 failures`。

回归命令与计数：

```bash
.../lunar_planner_ros_capability_loader_test                 # 22 passed
.../lunar_planner_core_hopper_planner_test --gtest_filter='HopperPlanner.*' # 8 passed
.../lunar_planner_ros_capability_loader_test \
  --gtest_filter='CapabilityLoader.LoadsApprovedParametricHopperWithoutUrdfOrMesh:CapabilityLoader.AdaptsLeggedAndHopperSourcesToTypedCapabilities' # 2 passed
```

最后运行 freeze checker 仍为 `60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95`。

### 自审

- 重复检测在 YAML map 迭代层执行，先于 `node[key]` 读取，故“批准首值 + 冲突后值”无法穿过任何已调用 exact-key gate。
- HOPPER frozen value 采用精确比较；无 `Near` 旁路。legacy URDF HOPPER 不走该 parametric frozen gate，已由 legacy typed-adaptation 正例回归覆盖。

## Fix round 2/5：signed-zero canonical gap

### TDD RED

新增 `CapabilityLoader.RejectsSignedZeroInParametricHopperFrozenFields` 后，在 `ros2-humble` clean-copy 运行：

```bash
.../lunar_planner_ros_capability_loader_test \
  --gtest_filter='CapabilityLoader.RejectsSignedZeroInParametricHopperFrozenFields'
```

结果为 `1 FAILED TEST`，其中 `gravity_mps2: [-0.0, 0.0, -1.62]` 与 `reference_elevation_delta_m: -0.0` 都被错误接受。这证明普通 `==` 不能闭合 raw-byte fingerprint 上的 signed-zero 分歧。

### 最小修复与 GREEN

- 增加 `ExactFrozenDouble`，以 `std::bit_cast<uint64_t>` 比较 parametric `hopper-v1` 全部批准 double 值的 IEEE raw bytes。
- 该 fail-closed 检查拒绝非 canonical `-0.0`，因此任何被接受的 frozen zero 在 typed capability 和 projection cache hash 中均为批准的 `+0.0` bytes。

聚焦 GREEN：`1 test, 0 failures`。

回归：完整 `lunar_planner_ros_capability_loader_test` 为 `23 passed`；tracked parametric HOPPER 与 legacy HOPPER typed-adaptation 过滤器为 `2 passed`；freeze checker 仍输出 `60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95`。

### 自审

- 覆盖了批准为零的 HOPPER frozen scalar/vector components：gravity x/y 与 reference elevation delta；其余批准值均非零。
- bitwise gate 只用于 strict parametric `hopper-v1`，不会重写或拒绝 legacy URDF/external-bundle HOPPER 的兼容路径。

## Final fix wave：configured gravity 贯穿 ROS 输出与承诺保护链

### 接管与 TDD 说明

接管时上一实现代理的测试和生产 diff 已同时遗留在工作树中，因此本 wave 不把这些现成改动伪称为亲自完成的“测试先行”。先只读审计新增测试，随后在独立容器目录 `/tmp/hopper-finalfix-red/runtime` 中以 `29dccd7` archive 为基线，仅保留新增测试/API 声明，并恢复四个消费者的旧硬编码行为，执行 mutation/revert 式 RED。首次 RED build 因旧 `ReferenceGuard::Commit` 定义与新声明签名不一致而编译失败；这不是行为 RED，未计入验收。将 RED harness 调整为“新签名、旧 `-1.62` 行为”后，四个测试均可运行并因目标缺陷失败。

### RED 证据

RED build：

```bash
cd /tmp/hopper-finalfix-red/runtime/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-up-to lunar_planner_ros --cmake-args -DBUILD_TESTING=ON
```

在保留新测试但恢复旧硬编码消费者后运行：

```bash
build/lunar_planner_ros/lunar_planner_ros_message_conversion_test \
  --gtest_filter=MessageConversion.AcceptsHopperBallisticsUsingConfiguredExecutionFrameGravity
build/lunar_planner_ros/lunar_planner_ros_reference_guard_test \
  --gtest_filter=ReferenceGuard.CommitsHopperUsingConfiguredExecutionFrameGravity
build/lunar_planner_ros/lunar_planner_ros_route_marker_publisher_test \
  --gtest_filter=RouteMarkerPublisher.DrawsHopperArcUsingConfiguredMapFrameGravity
build/lunar_planner_ros/lunar_planner_ros_plan_motion_server_test \
  --gtest_filter=PlanMotionServerTest.ReturnsAndCommitsHopperReferenceUsingConfiguredGravityAcrossLayers
```

结果为 `RED_EXIT_CODES=1,1,1,1`：

- message conversion：`1 FAILED TEST`，reason 为 `REFERENCE_HOP_BALLISTIC_INCONSISTENT`；
- ReferenceGuard：`1 FAILED TEST`，configured `-0.81` 下 `Commit` 错误返回 false；
- route marker：`1 FAILED TEST`，弧线末点 `z=0.76`，期望 `z=2.38`，差 `1.62`；
- PlanMotionServer：`1 FAILED TEST`，action 返回 ABORTED 而非 SUCCEEDED。

跨层 fixture 初版的 landing polygon 未包含旋转后 nominal y，GREEN 时 guard 正确报 `HOP_REFERENCE_COMMIT_FAILED`。将 polygon 修正为覆盖 nominal 点后，重新验证同一测试：旧实现仍以 `REFERENCE_HOP_BALLISTIC_INCONSISTENT` RED（`RED_RC=1`），最终实现 GREEN（`GREEN_RC=0`）。因此最终跨层测试失败原因是旧 gravity threading，而不是无效 fixture。

### 最小 API threading 与 frame 表达

- 新增最小 public `lunar_planner_core/frame_transform.hpp`，只公开 `TransformDirection` 和 `TransformVector`；内部 point/pose/goal transforms 仍保持内部声明。`TransformVector` 正规化并旋转向量，明确不应用 `map_from_odom.translation_m`。
- `HopperPlanner` 复用 `TransformVector` 将 map-frame launch velocity 表达到 odom，移除“清零平移后把向量当点”的本地适配。
- `PlannerResultContext` 新增 `execution_gravity_mps2`；`ConvertPlannerOutput` 用该三轴执行帧向量重建 landing，并拒绝非有限 gravity。
- `PlanMotionServer` 从同一个 typed `HopperCapability::gravity_mps2` 取 map-frame gravity，通过 `map_from_odom` 的 parent-to-child 纯旋转得到 odom-frame gravity，同时传给 converter 和 `ReferenceGuard::Commit`。非 HOPPER 结果使用零向量占位，不复制 lunar capability 常量。
- `ReferenceGuard::Commit` 显式要求 execution-frame gravity，三轴重建 hop landing；生产 API 不保留 `-1.62` 默认值。
- `RouteMarkerPublisher::BallisticArc` 直接使用 `input.capability.gravity_mps2`，因为 certified preview 与 marker 均在 map frame。
- PlanMotionServer 行为测试使用 90° `map_from_odom` 旋转和 fixture 中既有非零平移：map gravity `{0,0,-0.81}` 必须成为 odom gravity `{0,-0.81,0}`，同时证明向量变换不会错误叠加平移。

### GREEN 证据

最终 clean-copy affected build：

```bash
cd /tmp/hopper-finalfix-green/runtime/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-up-to lunar_planner_ros lunar_planner_training_bridge \
  --cmake-args -DBUILD_TESTING=ON
```

结果：`5 packages finished`，0 failed。

最终 C++ 回归：

```bash
build/lunar_planner_ros/lunar_planner_ros_message_conversion_test
build/lunar_planner_ros/lunar_planner_ros_reference_guard_test
build/lunar_planner_ros/lunar_planner_ros_route_marker_publisher_test
build/lunar_planner_ros/lunar_planner_ros_plan_motion_server_test
build/lunar_planner_core/lunar_planner_core_hopper_planner_test
build/lunar_planner_core/lunar_planner_core_reachability_projection_test \
  --gtest_filter=ReachabilityProjection.HopperProjectionUsesConfiguredGravity
build/lunar_planner_ros/lunar_planner_ros_capability_loader_test
```

结果依次为 `9 passed`、`5 passed`、`5 passed`、`27 passed`、`8 passed`、`1 passed`、`23 passed`，合计 `78 passed, 0 failed`。其中包含四个新增 altered-gravity 行为测试、现有 core HopperPlanner configured-gravity、ReachabilityProjection configured-gravity 和完整 loader 23。

最终 freeze/deployment 回归：

```bash
python3 tools/check_platform_capability_freeze.py \
  --schema ros2_ws/src/lunar_navigation_config/config/platform_capability_schema_v2.yaml \
  --freeze ros2_ws/src/lunar_navigation_config/config/three_platform_capability_freeze_v1.yaml
python3 -m pytest -q \
  tests/foundation/test_platform_capability_freeze.py \
  tests/deployment/test_hopper_capability.py
git diff --check
```

结果：freeze checker 仍为 `capability freeze: OK sha256=60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95`；Python `15 passed`；`git diff --check` 干净。

### Final fix 自审与 concerns

- mutation check：把 converter、guard 或 marker 任一处恢复为 `-1.62`，对应 altered-gravity 单测失败；PlanMotionServer 测试还会捕捉未传 gravity、错误旋转方向或向量错误叠加 TF 平移。
- 三个原问题生产消费者不再包含 lunar gravity 常量；strict tracked `hopper-v1` loader 仍只接受批准的 `-1.62` raw bits，未放宽 loader，也未删除 typed configured-gravity 功能。
- canonical freeze、`wheel.yaml`、`legged.yaml` 均未改；未安装 `/opt`、未推送、未合并，未新增 fuel state/fallback/primitive。
- 无遗留 correctness concern。宿主与容器均无 `clang-format`，沿用 `git diff --check`、C++20 clean-copy 编译和行为测试作为格式/编译证据。
