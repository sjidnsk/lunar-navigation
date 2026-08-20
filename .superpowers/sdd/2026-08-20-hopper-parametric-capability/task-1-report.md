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
