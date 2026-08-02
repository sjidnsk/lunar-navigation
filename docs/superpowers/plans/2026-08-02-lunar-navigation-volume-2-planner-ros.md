# Lunar Navigation 卷二：C++ v3 与 ROS Action 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把旧 C++ v3 迁成 ROS 无关的简化规划库，并用独立 Lifecycle `PlanMotion` Action 节点适配外部地图、定位与 TF。

**Architecture:** 先复制固定提交中的 v3 源码和有效 fixture 到新仓私有迁移区，用简化 `PlannerInput`/`PlannerOutput` facade 建立差分基线，再按共享核心、轮式、足式、飞跃式顺序消除旧合同。ROS 包只负责外部消息转换、一致快照、并发/取消和消息输出；可选 Nav2 插件只能调用 Action 并返回轮式 Path。

**Tech Stack:** C++20、GCC 11、CMake 3.22、ament_cmake、GoogleTest、ROS 2 Humble、rclcpp、rclcpp_lifecycle、rclcpp_action、tf2_ros、grid_map_msgs、launch_testing、nav2_core（可选）。

**Codex Estimate:** 24–42 agent-hours。

## Global Constraints

- `lunar_planner_core` 不得依赖 ROS、Python、TensorRT、JSON Schema、文件系统相邻仓库或运行时注册表。
- 长期公共入口必须保持 `PlannerOutput Plan(const PlannerInput& input)`。
- C++ v3 必须保留轮式、足式、飞跃式规划、安全检查、时间参数化、飞跃承诺、确定性排序和必要诊断。
- 平台有效最大坡度必须取外部能力 `maximum_slope_rad` 与项目 `30°` 硬上限中的较小值。
- 缺失地图层、非法值、TF 不可用、定位无效或时间不一致时不得调用 v3。
- v3 失败不得回退 Python A*；Python A* 只允许用于迁移差分测试。
- 同一规划节点只允许一个正在执行的规划请求；替换必须显式设置 `replace_active_request=true`。
- Action、地图、定位和 TF 必须使用分离 callback group；规划必须在独立 worker 中运行。
- committed 或 in-flight 的飞跃参考不得被普通取消或替换。
- `lunar_nav2_adapter` 默认不构建，只接受轮式参考。
- AGX 固定规划基准必须满足 `P95 < 1 s`。
- 本卷不得连接 executor 或发布真实运动控制命令。

---

## File Structure

```text
migration/v3_file_map.yaml
tools/import_v3_snapshot.py
tests/differential/fixtures/                       # 小型固定输入与结果摘要
ros2_ws/src/lunar_planner_core/
├── CMakeLists.txt
├── package.xml
├── include/lunar_planner_core/
│   ├── planner.hpp
│   ├── types/geometry.hpp
│   ├── types/goal.hpp
│   ├── types/world_snapshot.hpp
│   ├── types/platform_capability.hpp
│   ├── types/planner_config.hpp
│   ├── types/execution_context.hpp
│   ├── types/motion_reference.hpp
│   └── types/planner_io.hpp
├── src/
│   ├── planner.cpp
│   ├── shared/                                    # map/search/corridor/cost/optimization
│   ├── wheel/
│   ├── legged/
│   ├── hopper/
│   └── migration/legacy_v3/                       # 逐平台迁完后从构建移除
└── test/
ros2_ws/src/lunar_planner_ros/
├── CMakeLists.txt
├── package.xml
├── include/lunar_planner_ros/
│   ├── capability_loader.hpp
│   ├── grid_map_adapter.hpp
│   ├── snapshot_store.hpp
│   ├── snapshot_builder.hpp
│   ├── reference_guard.hpp
│   ├── message_conversion.hpp
│   └── plan_motion_server.hpp
├── src/grid_map_adapter.cpp
├── src/capability_loader.cpp
├── src/snapshot_store.cpp
├── src/snapshot_builder.cpp
├── src/reference_guard.cpp
├── src/message_conversion.cpp
├── src/plan_motion_server.cpp
├── src/main.cpp
├── config/planner_ros.schema.json
└── test/
ros2_ws/src/lunar_nav2_adapter/                    # 仅显式选择时构建
tests/integration/planner_ros/
tests/differential/test_v3_against_legacy_summary.py
tests/performance/planner_core_benchmark.cpp
```

### Task 1: 导入固定 v3 源码快照并建立差分摘要

**Execution environment:** Windows 生成迁移提交；Ubuntu 编译验证。

**Estimated Codex time:** 1.5–3 小时。

**Files:**
- Create: `migration/v3_file_map.yaml`
- Create: `tools/import_v3_snapshot.py`
- Create: `tests/differential/fixtures/wheel_cases.json`
- Create: `tests/differential/fixtures/legged_cases.json`
- Create: `tests/differential/fixtures/hopper_cases.json`
- Create: `tests/differential/legacy_expected.json`
- Create: `tests/differential/test_v3_snapshot_manifest.py`
- Create: `ros2_ws/src/lunar_planner_core/src/migration/legacy_v3/` files listed by the generated import manifest
- Read only: frozen `path-planner/cpp` commit from `migration/source_inventory.yaml`

**Interfaces:**
- Consumes: 卷一 `foundation-v1` 和固定旧 `path-planner` commit。
- Produces: 新仓内自包含的私有 legacy v3 静态库、三平台差分输入和只含稳定语义的 expected 摘要。

- [ ] **Step 1: 写导入边界测试**

```python
def test_v3_file_map_excludes_governance_surfaces():
    manifest = yaml.safe_load(Path("migration/v3_file_map.yaml").read_text())
    excluded = set(manifest["exclude_roots"])
    assert "schemas" in excluded
    assert "benchmarks" in excluded
    assert "python_bindings" in excluded
    assert manifest["source_commit"] == SOURCE_INVENTORY["repositories"]["path_planner"]["commit"]
```

- [ ] **Step 2: 固定迁移文件映射**

`v3_file_map.yaml` 必须按 `shared`、`wheel`、`legged`、`hopper`、`temporary_contract_adapter`、`tests` 六组列出源相对路径和目标相对路径。允许导入 `path-planner/cpp/include`、`src` 与选定测试；明确排除 `schemas/`、`benchmarks/`、Python bindings、JSON fixture、vcpkg 清单和 benchmark report。

- [ ] **Step 3: 实现受控复制器**

复制器必须验证源 Git commit、逐文件 SHA-256、目标位于新仓 `lunar_planner_core`、文本 UTF-8/LF，并拒绝符号链接、`.git`、object、binary 和清单外文件。输出 `migration/v3_import_result.json`，其中记录每个源/目标相对路径和 hash。

- [ ] **Step 4: 从旧 v3 生成稳定差分摘要**

每个平台至少覆盖：安全可达、无安全路线、目标不可行、数值失败和确定性重复。`legacy_expected.json` 只记录：

```json
{
  "case_id": "wheel-safe-corridor",
  "reachable": true,
  "planning_outcome": "NEW_REFERENCE_AVAILABLE",
  "collision_free": true,
  "goal_reached": true,
  "platform_constraints_satisfied": true,
  "cost": 12.5
}
```

不得记录 ContentRef、bundle hash、registry handle 或要求几何点逐字节相同。

- [ ] **Step 5: 运行清单测试和私有 legacy build**

```bash
python3 -m pytest -q tests/differential/test_v3_snapshot_manifest.py
colcon build --base-paths ros2_ws/src --packages-select lunar_planner_core --cmake-args -DLUNAR_BUILD_LEGACY_V3=ON
```

Expected: 导入清单 hash 全部匹配，legacy 私有目标可编译但没有安装公共 legacy header。

- [ ] **Step 6: 提交导入快照**

```bash
git add migration/v3_file_map.yaml migration/v3_import_result.json tools/import_v3_snapshot.py tests/differential ros2_ws/src/lunar_planner_core
git commit -m "refactor: import frozen planner v3 snapshot"
```

### Task 2: 定义简化纯 C++ 类型与唯一公共 API

**Execution environment:** Ubuntu 22.04 amd64。

**Estimated Codex time:** 1.5–3 小时。

**Files:**
- Create: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/geometry.hpp`
- Create: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/goal.hpp`
- Create: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/world_snapshot.hpp`
- Create: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/platform_capability.hpp`
- Create: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_config.hpp`
- Create: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/execution_context.hpp`
- Create: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/motion_reference.hpp`
- Create: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp`
- Create: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/planner.hpp`
- Create: `ros2_ws/src/lunar_planner_core/test/test_fixtures.hpp`
- Create: `ros2_ws/src/lunar_planner_core/test/public_api_test.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/public_header_boundary_test.py`

**Interfaces:**
- Consumes: C++20 standard library。
- Produces: 后续所有核心和 ROS 代码使用的稳定命名、枚举与类型。

- [ ] **Step 1: 写公共 API 测试**

```cpp
TEST(PublicApi, RejectsMissingRequiredMapLayersWithoutCallingBackend) {
  Planner planner;
  auto input = MakeValidWheelInput();
  input.world.local_map.layers.erase("forbidden");
  const auto output = planner.Plan(input);
  EXPECT_EQ(output.outcome, PlanningOutcome::kInvalidRequest);
  EXPECT_EQ(output.reason_code, "MISSING_MAP_LAYER_FORBIDDEN");
  EXPECT_FALSE(output.reference.has_value());
}
```

- [ ] **Step 2: 定义稳定枚举**

```cpp
enum class PlatformType : std::uint8_t { kWheeled = 1, kLegged = 2, kHopper = 3 };
enum class PlanningOutcome : std::uint8_t {
  kNewReferenceAvailable = 0,
  kSafeFrontierReferenceAvailable = 1,
  kNoKnownSafeRoute = 2,
  kGoalInfeasible = 3,
  kInvalidRequest = 4,
  kStaleInput = 5,
  kNumericalFailure = 6,
  kResourceExhausted = 7,
  kActiveReferenceInvalidated = 8,
  kCanceled = 9,
};
enum class ExecutionDirective : std::uint8_t {
  kActivateNewReference = 0,
  kContinueActiveReference = 1,
  kHoldPosition = 2,
  kContinueCommittedHop = 3,
  kNoSafeReference = 4,
};
```

`geometry.hpp` 和 `goal.hpp` 必须把 ROS 表示消解为普通值类型：

```cpp
struct TimePoint final { std::int64_t nanoseconds_since_epoch{}; };
struct Vec3 final { double x{}; double y{}; double z{}; };
struct PointGoal final { Vec3 position_m; double tolerance_m{}; };
struct PlanarRegionGoal final {
  std::vector<Vec3> boundary_m;
  double normal_tolerance_m{};
};
using GoalTarget = std::variant<PointGoal, PlanarRegionGoal>;
struct GoalRegion final {
  std::string goal_id;
  GoalTarget target;
  std::optional<double> yaw_rad;
  double yaw_tolerance_rad{};
};
```

ROS `GoalRegion.POINT=1` 与 `PLANAR_REGION=2` 只在卷二消息转换层选择对应 variant；数值枚举不得进入 core 业务分支。

- [ ] **Step 3: 定义输入输出结构**

`planner_io.hpp` 的完整公共表面必须为：

```cpp
struct PlannerInput final {
  std::string request_id;
  TimePoint state_time;
  PlatformState current_state;
  GoalRegion goal;
  WorldSnapshot world;
  PlatformCapability capability;
  PlannerConfig config;
  std::optional<ExecutionContext> previous_execution;
  std::stop_token stop_token;
};

struct PlannerDiagnostics final {
  std::string planner_name{"cpp_v3"};
  std::chrono::nanoseconds elapsed{};
  std::uint64_t expanded_states{};
  std::optional<double> best_cost;
  std::vector<std::string> warning_codes;
};

struct PlannerOutput final {
  PlanningOutcome outcome{PlanningOutcome::kInvalidRequest};
  ExecutionDirective directive{ExecutionDirective::kNoSafeReference};
  std::string reason_code;
  std::optional<MotionReference> reference;
  PlannerDiagnostics diagnostics;
};
```

`WorldSnapshot` 持有不可变、按行连续的 typed 栅格层；`PlatformCapability` 是三平台 `std::variant`；`MotionReference` 是 wheel/legged trajectory 与有界 hop segments 的 `std::variant`，不得带 JSON 或 hash graph。

- [ ] **Step 4: 定义唯一 planner 入口**

```cpp
class Planner final {
 public:
  Planner();
  ~Planner();
  Planner(Planner&&) noexcept;
  Planner& operator=(Planner&&) noexcept;
  Planner(const Planner&) = delete;
  Planner& operator=(const Planner&) = delete;

  [[nodiscard]] PlannerOutput Plan(const PlannerInput& input) noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
```

- [ ] **Step 5: 审计公共 header 依赖**

`public_header_boundary_test.py` 必须扫描 installed headers，拒绝 `rclcpp`、`geometry_msgs`、`nlohmann`、`ContentRef`、`Registry`、`schema`、`TensorRT` 和 Python include。

- [ ] **Step 6: 构建测试并提交**

```bash
colcon build --base-paths ros2_ws/src --packages-select lunar_planner_core
colcon test --base-paths ros2_ws/src --packages-select lunar_planner_core
colcon test-result --verbose
git add ros2_ws/src/lunar_planner_core
git commit -m "feat: define simplified planner core API"
```

### Task 3: 建立 legacy facade 并锁定简化语义差分

**Execution environment:** Ubuntu 22.04 amd64。

**Estimated Codex time:** 1.5–3 小时。

**Files:**
- Create: `ros2_ws/src/lunar_planner_core/src/migration/legacy_v3_adapter.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/migration/legacy_v3_adapter.cpp`
- Create: `ros2_ws/src/lunar_planner_core/src/planner.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/legacy_adapter_test.cpp`
- Create: `tests/differential/test_v3_semantic_equivalence.py`

**Interfaces:**
- Consumes: Task 1 私有 legacy v3 与 Task 2 公共类型。
- Produces: `LegacyV3Adapter` 私有实现；由 `Planner::Impl` 调用，只作为逐模块迁移的临时对照，不安装其 header。

- [ ] **Step 1: 写 outcome/directive 映射测试**

```cpp
TEST(LegacyAdapter, MapsNoPathWithoutFallback) {
  auto input = MakeBlockedWheelInput();
  LegacyV3Adapter adapter(MakeLegacyBackend());
  const auto output = adapter.Plan(input);
  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_FALSE(output.reference.has_value());
  EXPECT_EQ(adapter.fallback_invocations(), 0U);
}
```

- [ ] **Step 2: 实现单向转换**

adapter 只允许 `PlannerInput -> private legacy request` 与 `private legacy response -> PlannerOutput`。转换时在内存中构造临时旧对象，不读取 schema/JSON 文件，不暴露 ContentRef；hash 或 handle 仅可使用进程内固定测试值满足旧实现，且不得进入输出。

- [ ] **Step 3: 增加协作取消**

adapter 在每个搜索扩展、优化迭代和平台候选认证边界检查 `input.stop_token.stop_requested()`；取消输出必须是 `kCanceled`、`kHoldPosition`、`reason_code="REQUEST_CANCELED"` 且无新参考。

- [ ] **Step 4: 运行三平台差分**

```bash
colcon test --base-paths ros2_ws/src --packages-select lunar_planner_core --ctest-args -R legacy_adapter
python3 -m pytest -q tests/differential/test_v3_semantic_equivalence.py
```

Expected: 可达性、安全、目标到达、平台约束、outcome/directive 和成本合理性匹配；路径几何不要求逐点相同。

- [ ] **Step 5: 提交 facade**

```bash
git add ros2_ws/src/lunar_planner_core tests/differential
git commit -m "refactor: isolate legacy v3 behind planner facade"
```

### Task 4: 迁移共享地图、搜索、走廊与优化核心

**Execution environment:** Ubuntu 22.04 amd64。

**Estimated Codex time:** 3–5 小时。

**Files:**
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/shared/map_snapshot.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/shared/safe_projection.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/shared/ara_star.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/shared/candidate_ranker.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/shared/convex_corridor.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/shared/bounded_qp_solver.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/shared/terrain_checks.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/shared_core_test.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/shared_core_determinism_test.cpp`

**Interfaces:**
- Consumes: Task 2 typed maps/config/capability，不消费 legacy contract。
- Produces: 三平台可复用的 immutable snapshot、safe projection、ARA*、corridor、QP 和 terrain predicates。

- [ ] **Step 1: 写缺层、NaN、禁入和坡度测试**

测试必须覆盖循环缓冲区展开后的 row-major 层、未知单元、`forbidden=1`、NaN、高程方差、障碍阈值、净空，以及 `min(capability_slope, 30°)`。

- [ ] **Step 2: 迁移 shared 算法并替换旧身份类型**

把 cache key 改为局部值对象，删除 ContentRef/registry lookup；把 benchmark profile 从普通 Plan 调用剥离；保持候选排序 `(cost, stable_index)` 确定性。

- [ ] **Step 3: 运行 shared 测试和 sanitizer**

```bash
colcon build --base-paths ros2_ws/src --packages-select lunar_planner_core --cmake-args -DCMAKE_BUILD_TYPE=Debug -DLUNAR_ENABLE_SANITIZERS=ON
colcon test --base-paths ros2_ws/src --packages-select lunar_planner_core --ctest-args -R 'shared|map|search|corridor|optimization'
colcon test-result --verbose
```

Expected: 全部通过，无 ASan/UBSan 报告。

- [ ] **Step 4: 审计 shared 依赖方向并提交**

```bash
rg -n 'legacy_v3|ContentRef|Registry|lunar_planner_core/(wheel|legged|hopper)' ros2_ws/src/lunar_planner_core/src/shared && exit 1 || true
git add ros2_ws/src/lunar_planner_core
git commit -m "refactor: migrate shared planner core"
```

### Task 5: 迁移轮式规划器

**Execution environment:** Ubuntu 22.04 amd64。

**Estimated Codex time:** 2–3.5 小时。

**Files:**
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_planner.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_sweep_validator.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_spline_optimizer.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/wheel/wheel_timing.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/wheel_planner_test.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/wheel_fault_matrix_test.cpp`

**Interfaces:**
- Consumes: Task 4 shared API 与 `WheeledCapability`。
- Produces: wheel `TrajectoryReference`，满足曲率、扫掠碰撞、速度、加速度和制动约束。

- [ ] **Step 1: 写轮式完成与故障矩阵测试**

至少覆盖前进、倒车、原地旋转、曲率上限、障碍扫掠、无通路、取消和确定性。

- [ ] **Step 2: 迁移算法并直接消费简化类型**

轮式 planner 不得再构造 `PlanningRequest`、`ReferenceBundle` 或 registry binding。输出直接构造 `MotionReference{PlatformType::kWheeled, TrajectoryReference}`。

- [ ] **Step 3: 运行轮式差分与单元测试**

```bash
colcon test --base-paths ros2_ws/src --packages-select lunar_planner_core --ctest-args -R wheel
python3 -m pytest -q tests/differential/test_v3_semantic_equivalence.py -k wheel
```

Expected: 全部通过。

- [ ] **Step 4: 从 wheel 构建移除 legacy adapter 路径并提交**

```bash
rg -n 'legacy_v3|ContentRef|Registry|ReferenceBundle' ros2_ws/src/lunar_planner_core/src/wheel && exit 1 || true
git add ros2_ws/src/lunar_planner_core tests/differential
git commit -m "refactor: migrate wheel planner to simplified core"
```

### Task 6: 迁移足式规划器

**Execution environment:** Ubuntu 22.04 amd64。

**Estimated Codex time:** 2–3.5 小时。

**Files:**
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_planner.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_lattice.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_terrain.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_spline_optimizer.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/legged/legged_timing.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/legged_planner_test.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/legged_fault_matrix_test.cpp`

**Interfaces:**
- Consumes: Task 4 shared API 与 `LeggedCapability`。
- Produces: 机体参考点 trajectory；不宣称足端轨迹或完整落足可行性。

- [ ] **Step 1: 写足式安全语义测试**

覆盖坡度、粗糙度、台阶、机体净空、速度/加速度、未知地形拒绝、取消和确定性；断言输出只表示机体参考轨迹。

- [ ] **Step 2: 迁移算法并删除 legacy 类型依赖**

保留 body primitive、height interval、corridor、search、spline 和 timing；把所有 capability lookup 改为 typed `LeggedCapability` 直接字段访问。

- [ ] **Step 3: 运行足式差分和测试**

```bash
colcon test --base-paths ros2_ws/src --packages-select lunar_planner_core --ctest-args -R legged
python3 -m pytest -q tests/differential/test_v3_semantic_equivalence.py -k legged
```

Expected: 全部通过。

- [ ] **Step 4: 审计并提交**

```bash
rg -n 'legacy_v3|ContentRef|Registry|ReferenceBundle|footstep_feasibility_guaranteed' ros2_ws/src/lunar_planner_core/src/legged && exit 1 || true
git add ros2_ws/src/lunar_planner_core tests/differential
git commit -m "refactor: migrate legged planner to simplified core"
```

### Task 7: 迁移飞跃式规划器和简化承诺上下文

**Execution environment:** Ubuntu 22.04 amd64。

**Estimated Codex time:** 3–5 小时。

**Files:**
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hopper_planner.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/hopper/ballistic_kinematics.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/hopper/landing_region.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/hopper/flight_tube_certifier.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hop_certifier.cpp`
- Create/Modify: `ros2_ws/src/lunar_planner_core/src/hopper/commitment_state_machine.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/hopper_planner_test.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/hopper_commitment_test.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/hopper_fault_matrix_test.cpp`

**Interfaces:**
- Consumes: Task 4 shared API、`HopperCapability` 与简化 `ExecutionContext`。
- Produces: 只授权第一跳的 `HopReference`、`kContinueCommittedHop` 和安全拒绝语义。

- [ ] **Step 1: 写飞跃认证和承诺测试**

测试必须覆盖着陆坡度/粗糙度/残差、顶部与侧向净空、着陆面积、发射速度/冲量、飞行时间、着陆速度、姿态、未知或数值不确定拒绝，以及 committed/in-flight 不发布新跳。

- [ ] **Step 2: 定义简化执行上下文**

```cpp
enum class HopperExecutionState : std::uint8_t {
  kGroundHold,
  kJumpReady,
  kJumpCommitted,
  kInFlight,
  kLandedHold,
  kEmergencyDelegated,
};

struct HopperExecutionContext final {
  HopperExecutionState state{HopperExecutionState::kGroundHold};
  std::optional<std::string> active_plan_id;
  std::optional<std::string> active_segment_id;
};
```

它不得包含 bundle ref、registry handle、地图 hash 或控制器 JSON。

- [ ] **Step 3: 迁移物理认证与状态机**

保持“未知即拒绝”“只授权第一跳”“committed/in-flight 只能继续当前边界”。`PlannerOutput` 在保护期内返回 `kContinueActiveReference` 或 `kActiveReferenceInvalidated`，directive 必须是 `kContinueCommittedHop` 或 `kNoSafeReference`。

- [ ] **Step 4: 运行飞跃差分和测试**

```bash
colcon test --base-paths ros2_ws/src --packages-select lunar_planner_core --ctest-args -R hopper
python3 -m pytest -q tests/differential/test_v3_semantic_equivalence.py -k hopper
```

Expected: 全部通过；任何认证数据缺失都不产生 hop reference。

- [ ] **Step 5: 审计并提交**

```bash
rg -n 'legacy_v3|ContentRef|Registry|ReferenceBundle' ros2_ws/src/lunar_planner_core/src/hopper && exit 1 || true
git add ros2_ws/src/lunar_planner_core tests/differential
git commit -m "refactor: migrate hopper planner and commitment state"
```

### Task 8: 删除核心中的旧合同支配面

**Execution environment:** Ubuntu 22.04 amd64。

**Estimated Codex time:** 1.5–3 小时。

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_planner_core/src/planner.cpp`
- Remove one explicit file at a time: files under `ros2_ws/src/lunar_planner_core/src/migration/legacy_v3/`
- Remove: `ros2_ws/src/lunar_planner_core/src/migration/legacy_v3_adapter.hpp`
- Remove: `ros2_ws/src/lunar_planner_core/src/migration/legacy_v3_adapter.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/no_legacy_contract_test.py`

**Interfaces:**
- Consumes: Tasks 4–7 的原生简化实现。
- Produces: 不再编译或安装 legacy contract 的原生 `Planner::Impl`。

- [ ] **Step 1: 让 no-legacy 测试先失败**

测试递归扫描 `lunar_planner_core`，发现 `ContentRef`、`ContractObjectRegistry`、`ReferenceBundle`、`json_codec`、`schema_codec`、`registry_handle` 或 `legacy_v3_adapter` 即失败。

- [ ] **Step 2: 切换 `Planner::Impl` 到简化编排器**

`planner.cpp` 必须只验证 `PlannerInput`、按 `PlatformCapability` variant 调用 wheel/legged/hopper，并统一构造 `PlannerOutput`。不得创建内部 JSON 文档。由于公共 `Plan` 是 `noexcept`，实现必须在边界捕获 `std::bad_alloc` 并返回 `kResourceExhausted/RESOURCE_EXHAUSTED`，捕获其他未预期异常并返回 `kNumericalFailure/INTERNAL_PLANNER_EXCEPTION`；异常不得越过 API。

- [ ] **Step 3: 从 CMake 移除 legacy targets**

确认所有迁移测试使用稳定摘要而非链接 legacy 后，一次只移除一个明确文件并立即运行核心测试；不得执行递归删除命令。

- [ ] **Step 4: 运行全核心门槛**

```bash
colcon build --base-paths ros2_ws/src --packages-select lunar_planner_core --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLUNAR_BUILD_LEGACY_V3=OFF
colcon test --base-paths ros2_ws/src --packages-select lunar_planner_core
colcon test-result --verbose
python3 ros2_ws/src/lunar_planner_core/test/no_legacy_contract_test.py
```

Expected: 全部通过，安装树只有简化 public headers。

- [ ] **Step 5: 提交合同清理**

```bash
git add -A ros2_ws/src/lunar_planner_core
git commit -m "refactor: remove legacy planner contracts from runtime"
```

### Task 9: 实现外部输入适配和一致快照

**Execution environment:** Ubuntu 22.04 amd64。

**Estimated Codex time:** 3–5 小时。

**Files:**
- Create: `ros2_ws/src/lunar_planner_ros/package.xml`
- Create: `ros2_ws/src/lunar_planner_ros/CMakeLists.txt`
- Create: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/grid_map_adapter.hpp`
- Create: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/capability_loader.hpp`
- Create: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/snapshot_store.hpp`
- Create: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/snapshot_builder.hpp`
- Create: `ros2_ws/src/lunar_planner_ros/src/grid_map_adapter.cpp`
- Create: `ros2_ws/src/lunar_planner_ros/src/capability_loader.cpp`
- Create: `ros2_ws/src/lunar_planner_ros/src/snapshot_store.cpp`
- Create: `ros2_ws/src/lunar_planner_ros/src/snapshot_builder.cpp`
- Create: `ros2_ws/src/lunar_planner_ros/config/planner_ros.schema.json`
- Create: `ros2_ws/src/lunar_planner_ros/test/grid_map_adapter_test.cpp`
- Create: `ros2_ws/src/lunar_planner_ros/test/snapshot_builder_test.cpp`
- Create: `ros2_ws/src/lunar_planner_ros/test/capability_loader_test.cpp`

**Interfaces:**
- Consumes: 外部 GridMap/Odometry/LocalizationStatus/TF 与平台/观测能力资料。
- Produces: `SnapshotBuildResult Freeze(const GoalRequest&, rclcpp::Time now)`；`SnapshotBuildResult` 只允许“有效 `PlannerInput`”或“一个 `SnapshotError`”二选一。

- [ ] **Step 1: 写 GridMap 和时间一致性失败测试**

覆盖循环缓冲区、重复层、缺层、NaN、非法范围、wrong frame、过期、map/odom/TF skew 和 covariance 上限。

- [ ] **Step 2: 实现 GridMap 展开与派生层**

按 `outer_start_index`/`inner_start_index` 展开为 row-major typed layers；计算坡度、粗糙度、障碍、置信度和 v3 需要的安全派生层。缺前提必须返回具体 `SnapshotErrorCode`，不得填零或 synthetic 数据。

- [ ] **Step 3: 实现能力加载**

在 Lifecycle configure 期间通过 ament package share 解析 YAML/JSON、URDF 与 mesh；验证有限值、范围和资源存在；适配为 typed capability 并冻结。坡度应用 `min(external_limit, std::numbers::pi / 6.0)`。

- [ ] **Step 4: 实现快照选择策略**

配置必须显式提供 `global_map_max_age`、`local_map_max_age`、`odometry_max_age`、`localization_status_max_age`、`tf_max_age` 和 `max_pairwise_skew`；缺任何字段时 configure 失败。`Freeze` 只在定位 VALID 或 covariance 合法的 DEGRADED 状态返回输入。

`SnapshotBuildResult` 的 C++20 定义固定为：

```cpp
struct SnapshotBuildResult final {
  std::optional<lunar::planning::PlannerInput> input;
  std::optional<SnapshotError> error;

  [[nodiscard]] bool ok() const noexcept {
    return input.has_value() && !error.has_value();
  }
};
```

接受全局目标时必须在选定 `state_time` 获取 `map -> odom -> base_link`，把目标转换到局部规划使用的 `odom`；TF 缺失、外推或与 state/local map 超过 `max_pairwise_skew` 时返回 `STALE_TF`，不调用 core。

- [ ] **Step 5: 运行适配测试并提交**

```bash
colcon build --base-paths ros2_ws/src --packages-up-to lunar_planner_ros
colcon test --base-paths ros2_ws/src --packages-select lunar_planner_ros --ctest-args -R 'grid_map|snapshot|capability'
colcon test-result --verbose
git add ros2_ws/src/lunar_planner_ros
git commit -m "feat: adapt external inputs into planning snapshots"
```

### Task 10: 实现 Lifecycle Action、取消、替换和飞跃保护

**Execution environment:** Ubuntu 22.04 amd64。

**Estimated Codex time:** 3–5 小时。

**Files:**
- Create: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/reference_guard.hpp`
- Create: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/message_conversion.hpp`
- Create: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/plan_motion_server.hpp`
- Create: `ros2_ws/src/lunar_planner_ros/src/reference_guard.cpp`
- Create: `ros2_ws/src/lunar_planner_ros/src/message_conversion.cpp`
- Create: `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Create: `ros2_ws/src/lunar_planner_ros/src/main.cpp`
- Create: `ros2_ws/src/lunar_planner_ros/test/reference_guard_test.cpp`
- Create: `ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp`
- Create: `tests/integration/planner_ros/test_lifecycle_action.launch.py`
- Create: `tests/integration/planner_ros/test_action_concurrency.launch.py`

**Interfaces:**
- Consumes: Task 9 snapshot builder、Task 8 core planner、卷一 `PlanMotion.action`。
- Produces: `/plan_motion` Action server 和标准 `/diagnostics`。

- [ ] **Step 1: 写 Lifecycle 和并发状态测试**

覆盖 unconfigured/inactive 拒绝、configure 失败、activate、单请求、默认拒绝第二请求、显式替换、取消、旧 mission revision、PAUSED/CANCELED、节点 error 和恢复。

- [ ] **Step 2: 实现 callback group 与 worker 所有权**

地图、定位、TF、Action 各用 MutuallyExclusive callback group；Action 执行创建一个 `std::jthread` 和对应 `std::stop_source`。ROS callback 只冻结输入和更新状态，不在线程中直接运行 v3。

- [ ] **Step 3: 实现替换顺序**

新 goal 验证顺序固定为：Lifecycle active → request/mission 校验 → `ReferenceGuard::MayReplace()` → 是否已有 worker → `replace_active_request`。允许替换时先请求旧 worker stop，等待其协作结束并终结旧 Action，再启动新 worker；任一时刻不得有两个 v3 调用。

- [ ] **Step 4: 实现保守飞跃保护**

在没有 executor 状态输入的当前边界下，发布 `ACTIVATE_NEW_REFERENCE` 的 hopper reference 即视为锁定该首跳。`ReferenceGuard` 使用 `HopSegment.header.stamp`、`flight_time`、平台 `minimum_settle_guard_s` 和 Odometry 判断 `kJumpCommitted -> kInFlight -> kLandedHold -> kGroundHold`；计划窗口内或未确认稳定着陆时拒绝替换。若计划时间已过但位置/速度不满足着陆条件，保持锁定并发布 `HOP_EXECUTION_STATE_UNRESOLVED`，只能通过明确 Lifecycle deactivate/reconfigure 恢复，不自动解锁。

- [ ] **Step 5: 实现 Action Result 不变量**

`has_reference=false` 时输出默认空 `MotionReference`；`has_reference=true` 时 plan_id、frame、input_time 和平台 variant 必须完整。取消使用 Action canceled 终态和 outcome `CANCELED`；无路线是正常 succeeded Action，outcome `NO_KNOWN_SAFE_ROUTE`。

- [ ] **Step 6: 运行 Action 集成测试**

```bash
colcon build --base-paths ros2_ws/src --packages-up-to lunar_planner_ros
colcon test --base-paths ros2_ws/src --packages-select lunar_planner_ros
python3 -m pytest -q tests/integration/planner_ros
colcon test-result --verbose
```

Expected: Lifecycle、取消、替换、callback group、多线程和飞跃保护全部通过。

- [ ] **Step 7: 提交 Action 节点**

```bash
git add ros2_ws/src/lunar_planner_ros tests/integration/planner_ros
git commit -m "feat: add lifecycle PlanMotion action server"
```

### Task 11: 添加默认不构建的 Nav2 轮式薄适配器

**Execution environment:** Ubuntu 22.04 amd64，只有显式 Nav2 测试 job 构建。

**Estimated Codex time:** 1–1.5 小时。

**Files:**
- Create: `ros2_ws/src/lunar_nav2_adapter/package.xml`
- Create: `ros2_ws/src/lunar_nav2_adapter/CMakeLists.txt`
- Create: `ros2_ws/src/lunar_nav2_adapter/lunar_nav2_plugin.xml`
- Create: `ros2_ws/src/lunar_nav2_adapter/include/lunar_nav2_adapter/lunar_global_planner.hpp`
- Create: `ros2_ws/src/lunar_nav2_adapter/src/lunar_global_planner.cpp`
- Create: `ros2_ws/src/lunar_nav2_adapter/test/lunar_global_planner_test.cpp`
- Modify: `scripts/build_runtime.sh`

**Interfaces:**
- Consumes: Nav2 `nav2_core::GlobalPlanner` request 和 `/plan_motion` Action。
- Produces: 仅轮式 `nav_msgs::msg::Path`；足式/飞跃式明确失败。

- [ ] **Step 1: 写类型拒绝测试**

```cpp
TEST(LunarGlobalPlanner, RejectsHopperReference) {
  auto result = MakeActionResult(lunar_planning_msgs::msg::MotionReference::HOPPER);
  EXPECT_THROW(adapter.ConvertResult(result), nav2_core::PlannerException);
}
```

- [ ] **Step 2: 实现薄 plugin**

`createPlan(start, goal)` 必须验证 start 与最新 odometry 在配置容差内，将 goal 转为 `GoalRegion::POINT`，同步等待 Action 结果，并只在 `has_reference=true` 且 platform_type=WHEELED 时返回 `path_preview`。不得调用 core、订阅地图或把足式/飞跃式降维。

- [ ] **Step 3: 保持默认构建排除**

`scripts/build_runtime.sh` 必须继续 `--packages-skip lunar_nav2_adapter`。单独的显式命令为：

```bash
colcon build --base-paths ros2_ws/src --packages-select lunar_nav2_adapter
```

- [ ] **Step 4: 测试并提交**

```bash
colcon test --base-paths ros2_ws/src --packages-select lunar_nav2_adapter
colcon test-result --verbose
git add ros2_ws/src/lunar_nav2_adapter scripts/build_runtime.sh
git commit -m "feat: add optional wheeled Nav2 adapter"
```

### Task 12: 卷二全量验收、性能基线和回退点

**Execution environment:** Ubuntu amd64；ARM64 编译通道；AGX 性能结果在卷四成为发布门槛。

**Estimated Codex time:** 1–1.5 小时。

**Files:**
- Create: `tests/performance/planner_core_benchmark.cpp`
- Create: `tests/differential/test_python_astar_reference.py`
- Create: `docs/migration/volume-2-completion.md`
- Verify: `ros2_ws/src/lunar_planner_core/`
- Verify: `ros2_ws/src/lunar_planner_ros/`
- Verify: `ros2_ws/src/lunar_nav2_adapter/`

**Interfaces:**
- Consumes: Tasks 1–11。
- Produces: `planner-action-v1` tag、固定 AGX benchmark fixture 与 Ubuntu 报告。

- [ ] **Step 1: 添加固定性能 fixture**

benchmark 必须固定地图、能力、目标、随机种子、warmup 次数和 measured 次数；输出 JSON 含 count、median、p95、max、设备指纹 hash 和 Git commit。Ubuntu 数值只做回归预警，AGX `P95 < 1 s` 在卷四阻断发布。

- [ ] **Step 2: 添加迁移期 Python A* 差分**

测试通过离线 fixture 调用旧 Python A*，只比较可达性、安全、目标到达和成本合理性。测试文件必须位于 `tests/differential`，且部署包清单明确排除整个目录和旧 Python 包。

- [ ] **Step 3: 运行全量门槛**

```bash
colcon build --base-paths ros2_ws/src --packages-up-to lunar_planner_ros --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
colcon test --base-paths ros2_ws/src --packages-select lunar_planner_core lunar_planner_ros
python3 -m pytest -q tests/differential tests/integration/planner_ros
colcon test-result --verbose
```

Expected: 全部通过。

- [ ] **Step 4: 审计生产边界**

```bash
rg -n 'ContentRef|ContractObjectRegistry|ReferenceBundle|json_codec|schema_codec|path_planner\.search|AStarPlanner' ros2_ws/src && exit 1 || true
rg -n '#include.*(rclcpp|geometry_msgs|nav_msgs|tf2)' ros2_ws/src/lunar_planner_core && exit 1 || true
```

Expected: 无匹配；Nav2 包不在默认 build/install。

- [ ] **Step 5: 记录完成报告并打标签**

```bash
git add tests/performance tests/differential docs/migration/volume-2-completion.md
git commit -m "test: qualify planner core and action boundary"
git tag -a planner-action-v1 -m "C++ v3 and PlanMotion action v1"
```

**Rollback:** 卷二未通过时保留 `foundation-v1`；新代码不得回写旧 v3 仓。若某平台迁移失败，只恢复新仓上一个通过提交并继续使用私有 facade 对照，不改变旧部署。
