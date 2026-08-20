# Parametric Capability Training Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在最新地面训练基线中接入三平台参数化能力，把训练与部署的观测语义统一为 10 m、90°，以 update 457 的受限策略权重启动一个 12 WHEELED + 12 LEGGED 的 fresh-episode 新运行。

**Architecture:** 从 `972eb49` 建立隔离实现分支，先合入 `d068faf` 的参数化能力链；`deployment/config/{wheel,legged,hopper,observation}.yaml` 成为单一能力权威。方向性可见性由同一 C++ kernel 接受位置和 yaw，Python 观测事务、候选标准收益及任务分母共享这一实现。任务分母只允许用经等价测试证明的 10 m 全方位并集优化，rollout 观测与 Reward V4 始终使用实际 90° 视场。旧 30 m/360° 缓存和精确恢复 fail closed；update 457 只加载允许的共享策略参数。

**Tech Stack:** C++20, pybind11, Python 3.10, NumPy, PyTorch, pytest, GoogleTest/CTest, ROS 2 Humble, colcon, YAML, SHA-256 identity manifests

**Spec:** `docs/superpowers/specs/2026-08-20-parametric-capability-training-alignment-design.md`

## Global Constraints

- 实现基线固定为 `feat/global-ground-anchor-budget` @ `972eb49ba222676c95936a502c9f2070be6775c3`；能力来源固定为 `github/feat/wheel-parametric-capability` @ `d068faf7c79071416e658bcd32fa0a8cf19ed2f1`。
- 训练与部署观测能力固定为 `sensor_range_m: 10.0`、`sensor_fov_deg: 90.0`。
- 10 m 只表示传感器观测距离。地面候选与全局路线没有固定 30 m 总距离上限；global cost tree 必须继续覆盖当前全局安全地图的完整连通区域。兼容接口中的 `maximum_edge_distance_m=30.0` 不得被解释为 ground 搜索截断。滚动局部前视仍为 WHEELED 4 m、LEGGED 3 m，并通过连续滚动执行长路线；本计划不调整规划算法或这些前视配置。
- PPO 动作仍为 `(candidate_index, theta_rad)`；候选张量仍为 64 个策略槽位和 32 个 reserve；不增加网络输入、输出或绝对收益特征。
- 候选 `canonical_yaw` 只服务于资格、标准收益和排序；正式地面请求必须继续使用 PPO 的 `theta_rad`。
- Reward V4、80% 成功阈值、95% 物理可覆盖资格门、96 个全局锚点、32 个精细窗口均保持不变。
- 本轮只运行 12 WHEELED + 12 LEGGED；HOPPER 能力必须可加载和桥接，但不得创建训练、预检或评估 worker。
- 不运行全场景闭环门、不做三种子评估、不启用长时间阻塞评估。资格证据只包含聚焦测试以及 WHEELED/LEGGED 各一个完整宏动作。
- 旧 30 m/360° 的可见性、候选收益、初始观测、任务分母和资格缓存不得迁移。原始地形源、任务区域定义和与传感器无关的基础栅格可以复用。
- 所有 build/install/log/cache/run 产物放在仓库外；不得提交生成物、checkpoint 或用户已有脏改动。
- 每个生产改动先观察有效 RED，再做最小 GREEN；每个任务单独提交，并在提交前运行 `git diff --check`。

## File Structure

### 新建

- `deployment/config/observation.yaml`：训练与部署共同的 10 m、90° 观测能力权威。
- `tests/deployment/test_observation_capability.py`：部署观测配置的严格 schema 与数值测试。
- `training/lunar_policy_training/tests/test_directional_visibility.py`：Python/C++ 方向性边界、批量 yaw 和全向并集等价测试。
- `training/lunar_policy_training/tests/test_parametric_training_alignment.py`：能力源、run manifest、旧身份拒绝和 ground-only 分配的跨层契约测试。
- 外部产物根：`/home/kai/CodexDownloads/lunar_navigation/parametric-capability-training-alignment/`。

### 修改

- `training/lunar_policy_training/lunar_policy_training/capability_freeze.py`：冻结观测能力和参数化平台能力的类型/摘要证据。
- `training/lunar_policy_training/lunar_policy_training/project_capability.py`：直接读取部署 YAML，不再从旧三平台 freeze 重建数值。
- `training/lunar_policy_training/lunar_policy_training/training_semantics.py`：10 m/90° 语义与缓存身份。
- `training/lunar_policy_training/lunar_policy_training/cli.py`：能力源冻结、run manifest、ground-only 预检和新 run 启动边界。
- `training/lunar_policy_training/lunar_policy_training/formal_preflight.py`：只执行 WHEELED/LEGGED 的短宏动作预检。
- `training/lunar_policy_training/lunar_policy_training/environment/visibility.py`：yaw-aware Python visibility API。
- `training/lunar_policy_training/lunar_policy_training/environment/sensor_observation.py`：单格观测提交显式携带 yaw。
- `training/lunar_policy_training/lunar_policy_training/environment/multires_observation.py`：按 `(0.2 m cell, yaw)` 批量方向性观测和精确位置收益。
- `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`：ROI 前沿法向、标准 yaw 和方向性标准收益。
- `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`：法向见证接线、初始 yaw 和 PPO yaw 执行隔离。
- `training/lunar_policy_training/lunar_policy_training/environment/coverability.py`：可达朝向可见并集接口。
- `training/lunar_policy_training/lunar_policy_training/polar_data/task_coverability.py`：10 m 分母、90° 初始观测和新算法身份。
- `training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py`：新能力/观测身份和旧缓存拒绝。
- `training/lunar_policy_training/lunar_policy_training/polar_data/task_cache.py`：传感器派生任务键和平台作用域身份。
- `training/lunar_policy_training/lunar_policy_training/checkpoint.py`：update 457 的 theta/value/training-state 重置合同。
- `training/lunar_policy_training/lunar_policy_training/sensor_performance.py`：10 m/90° source-bound 性能证据。
- `training/lunar_policy_training/tests/test_{capability_freeze,project_capability,visibility,visibility_equivalence,sensor_observation,sensor_closed_loop,multires_observation,candidate_builder_v2,reachable_constrained_ground_candidates,coverability,formal_cache,formal_builder,checkpoint,checkpoint_resume,cli,formal_preflight,training_smoke,v3_environment}.py`：相应聚焦回归。
- `ros2_ws/src/lunar_planner_training_bridge/include/lunar_planner_training_bridge/visibility.hpp`：方向性 native API。
- `ros2_ws/src/lunar_planner_training_bridge/src/visibility.cpp`：90° 射线筛选。
- `ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp`：连续 yaw 与批量 yaw pybind 边界。
- `ros2_ws/src/lunar_planner_training_bridge/test/visibility_test.cpp`：native 方向性和确定性测试。
- `ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py`：pybind dtype/shape/yaw 测试。
- `training/tools/qualify_task_cache_training_entry.py`：新的 ground-only 聚焦资格 profile。
- `training/configs/rtx4080_super_v4_joint.yaml`：保留 Reward V4 ground allocation 12+12，并明确当前 stage 不创建 HOPPER。

---

## Task 1: Create the isolated implementation branch and integrate the parametric capability source

**Files:**

- Merge: `d068faf7c79071416e658bcd32fa0a8cf19ed2f1`
- Resolve if needed: `training/lunar_policy_training/tests/test_formal_cache.py`

- [ ] **Step 1: Verify both immutable inputs and create the worktree**

```bash
BASE=/home/kai/CodexDownloads/lunar_navigation/worktrees/global-ground-anchor-budget
TARGET=/home/kai/CodexDownloads/lunar_navigation/worktrees/parametric-capability-training-alignment

test "$(git -C "$BASE" rev-parse HEAD)" = 972eb49ba222676c95936a502c9f2070be6775c3
test -z "$(git -C "$BASE" status --porcelain)"
git -C "$BASE" cat-file -e d068faf7c79071416e658bcd32fa0a8cf19ed2f1^{commit}
git -C "$BASE" worktree add -b feat/parametric-capability-training-alignment \
  "$TARGET" 972eb49ba222676c95936a502c9f2070be6775c3
```

Expected: 新 worktree 位于精确基线，原训练 worktree 未被修改。

- [ ] **Step 2: Merge without committing and inspect the complete boundary**

```bash
cd "$TARGET"
git merge --no-ff --no-commit d068faf7c79071416e658bcd32fa0a8cf19ed2f1 || true
git diff --name-only --diff-filter=U
git status --short
```

Expected: 唯一预期语义重叠是 `test_formal_cache.py`；禁止使用整文件 `ours` 或 `theirs`。

- [ ] **Step 3: Resolve the overlap semantically**

在 `test_formal_cache.py` 中同时保留：

1. `972eb49` 的 frozen task cache / runtime-only repair 回归；
2. `d068faf` 的 WHEELED、LEGGED、HOPPER 参数化能力字段回归。

然后检查未引入后来 Task3 适配器：

```bash
test -z "$(git diff --name-only --diff-filter=U)"
test ! -e ros2_ws/src/luna_t3_map_adapter
git diff --check
```

- [ ] **Step 4: Run the capability merge qualification**

```bash
PY=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
export PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q \
  tests/deployment/test_wheel_capability.py \
  tests/deployment/test_legged_capability.py \
  tests/deployment/test_hopper_capability.py \
  training/lunar_policy_training/tests/test_capability_freeze.py \
  training/lunar_policy_training/tests/test_project_capability.py \
  training/lunar_policy_training/tests/test_formal_cache.py
```

Expected: all selected tests PASS；不要求执行 Task3 或全仓测试。

- [ ] **Step 5: Commit the merge**

```bash
git add -- \
  deployment ros2_ws/src/lunar_navigation_config \
  ros2_ws/src/lunar_planner_core ros2_ws/src/lunar_planner_ros \
  ros2_ws/src/lunar_planner_training_bridge \
  training/lunar_policy_training tests tools scripts model_contract
git commit -m "merge: integrate parametric platform capabilities"
git merge-base --is-ancestor d068faf7c79071416e658bcd32fa0a8cf19ed2f1 HEAD
```

Expected: merge commit 同时包含最新训练基线和固定能力来源，且没有未暂存冲突。

---

## Task 2: Make deployment YAML the single capability authority

**Files:**

- Create: `deployment/config/observation.yaml`
- Create: `tests/deployment/test_observation_capability.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/project_capability.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/capability_freeze.py`
- Modify: `training/lunar_policy_training/tests/test_project_capability.py`
- Modify: `training/lunar_policy_training/tests/test_capability_freeze.py`
- Modify: `tests/foundation/test_platform_capability_freeze.py`

**Interfaces:**

```python
def project_capability_paths(repository_root: str | Path) -> tuple[Path, Path, Path, Path]:
    """Return wheel, legged, hopper and observation authority paths."""

def load_project_formal_capability(repository_root: str | Path) -> FrozenCapabilityBundle:
    """Parse deployment YAMLs once and return immutable typed capabilities."""

def project_capability_source_evidence(
    repository_root: str | Path,
    *,
    active_platforms: tuple[str, ...],
) -> dict[str, object]:
    """Return canonical source paths, file hashes and content hashes."""
```

- [ ] **Step 1: Write RED tests for the missing shared observation authority**

`test_observation_capability.py` must require the exact document:

```yaml
sensor_range_m: 10.0
sensor_fov_deg: 90.0
```

`test_project_capability.py` must assert:

```python
bundle = load_project_formal_capability(REPOSITORY_ROOT)
assert {p.platform_type for p in bundle.platforms} == {
    "WHEELED", "LEGGED", "HOPPER"
}
assert all(p.observation_capability.sensor_range_m == 10.0 for p in bundle.platforms)
assert all(
    p.observation_capability.sensor_fov_rad == pytest.approx(math.pi / 2.0)
    for p in bundle.platforms
)
```

还要逐字段证明 WHEELED、LEGGED、HOPPER 的 typed capability 来自对应 `deployment/config/*.yaml`，而不是旧 freeze 常量。

```bash
PY=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q \
  tests/deployment/test_observation_capability.py \
  training/lunar_policy_training/tests/test_project_capability.py
```

Expected RED: `observation.yaml` 不存在，且当前训练仍产生 30 m/360° observation。

- [ ] **Step 2: Add the strict observation document and direct deployment adapters**

`project_capability.py` 必须：

- 严格读取 `wheel.yaml` 的 `platform` + `wheeled`；
- 严格读取 `legged.yaml` 的 `platform` + `legged`；
- 严格读取 `hopper.yaml` 的 `platform` + `hopper`；
- 严格读取仅含两个字段的 `observation.yaml`；
- 复用 `_parse_wheeled`、`_parse_legged`、`_parse_hopper` 生成 typed objects；
- 对部署 YAML 中不属于 planner typed payload 的来源说明字段进行显式适配，不把它们悄悄塞入 C++ 请求；
- 平台 `content_sha256` 只绑定对应平台的 canonical 物理内容；
- bundle 摘要绑定三平台内容摘要和共同 observation 摘要。

禁止继续使用 `APPROVED_PROJECT_CAPABILITY_SHA256` 作为硬编码值；新摘要必须从四个当前 authority 文件确定性计算。

- [ ] **Step 3: Add mutation and platform-scope tests**

使用临时仓库副本验证：

- 改 `wheel.yaml` 只改变 WHEELED content digest 和 bundle digest；
- 改 `observation.yaml` 改变共同 observation digest 和 bundle digest；
- 仅改 HOPPER 文件不改变 `active_platforms=("WHEELED", "LEGGED")` 的活动平台组合摘要，但完整 bundle 摘要仍改变；
- 非有限值、未知字段、错误平台类型、FOV > 360° 均 fail closed。

- [ ] **Step 4: Run GREEN tests and commit**

```bash
PY=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
export PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q \
  tests/deployment/test_observation_capability.py \
  tests/deployment/test_wheel_capability.py \
  tests/deployment/test_legged_capability.py \
  tests/deployment/test_hopper_capability.py \
  tests/foundation/test_platform_capability_freeze.py \
  training/lunar_policy_training/tests/test_capability_freeze.py \
  training/lunar_policy_training/tests/test_project_capability.py
git diff --check
git add -- deployment/config/observation.yaml \
  tests/deployment/test_observation_capability.py \
  tests/foundation/test_platform_capability_freeze.py \
  training/lunar_policy_training/lunar_policy_training/capability_freeze.py \
  training/lunar_policy_training/lunar_policy_training/project_capability.py \
  training/lunar_policy_training/tests/test_capability_freeze.py \
  training/lunar_policy_training/tests/test_project_capability.py
git commit -m "feat(training): freeze deployment capability authority"
```

---

## Task 3: Freeze 10 m/90° semantics and source evidence into every run identity

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/training_semantics.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/task_cache.py`
- Create: `training/lunar_policy_training/tests/test_parametric_training_alignment.py`
- Modify: `training/lunar_policy_training/tests/test_cli.py`
- Modify: `training/lunar_policy_training/tests/test_formal_cache.py`

**Run manifest addition:**

```json
{
  "capability_sources": {
    "active_platforms": ["WHEELED", "LEGGED"],
    "platforms": {
      "WHEELED": {"path": ".../wheel.yaml", "file_sha256": "...", "content_sha256": "..."},
      "LEGGED": {"path": ".../legged.yaml", "file_sha256": "...", "content_sha256": "..."}
    },
    "observation": {
      "path": ".../observation.yaml",
      "file_sha256": "...",
      "content_sha256": "...",
      "sensor_range_m": 10.0,
      "sensor_fov_deg": 90.0
    },
    "inactive_platforms": {"HOPPER": {"worker_count": 0, "load_verified": true}}
  }
}
```

- [ ] **Step 1: Write RED identity tests**

Tests must prove that current source incorrectly accepts the old identity, and then require:

```python
assert FORMAL_SENSOR_RANGE_M == 10.0
assert FORMAL_SENSOR_FOV_RAD == math.pi / 2.0
assert "sensor-10m-90" in FORMAL_TRAINING_SEMANTICS_VERSION
```

Also construct an old manifest with 30 m/360° hashes and require `FormalCacheError`/`PreflightError` before any task artifact is read.

- [ ] **Step 2: Advance the semantic identities atomically**

Use new, explicit version strings:

```python
FORMAL_TASK_CACHE_SEMANTICS_VERSION = (
    "lunar-training-task-semantics/sensor-10m-90deg-"
    "reachable-yaw-union-initial-yaw-platform-task-v18"
)
FORMAL_TRAINING_SEMANTICS_VERSION = (
    "lunar-training-semantics/sensor-10m-90deg-"
    "ground-anchor96-window32-policy64-reserve32-reward-v4/v20"
)
```

不要修改 `FORMAL_MINIMUM_MISSION_COVERABLE_RATIO=0.95` 或 `FORMAL_SUCCESS_COVERAGE_RATIO=0.80`。

- [ ] **Step 3: Freeze source files before worker construction**

在 calibrate 阶段把四个 authority 文件复制到：

```text
<artifact-root>/capabilities/wheel.yaml
<artifact-root>/capabilities/legged.yaml
<artifact-root>/capabilities/hopper.yaml
<artifact-root>/capabilities/observation.yaml
```

使用临时文件 + `fsync` + `os.replace` 原子提交；manifest 写入源路径、冻结路径、file SHA、content SHA 和活动平台组合摘要。worker 只接收冻结对象，不在宏动作中重新打开 YAML。

- [ ] **Step 4: Prove drift rejection and planning-range separation**

Tests must assert:

- calibrate 后修改任一活动平台或 observation authority，新 worker 加入旧 run 时 fail closed；
- 仅修改 HOPPER 时 ground-only run 不失效，但 HOPPER 不能加入；
- sensor range 是 10 m 时，连接安全地图中直线距离和最小路径成本均超过 30 m 的 ground endpoint 仍可由同一 global cost tree 判为 reachable；
- 兼容接口即使继续报告 `maximum_edge_distance_m=30.0`，也不得把它用于裁剪 ground global cost tree；
- planner configs and request search-domain fields remain byte-for-byte unchanged，且 local frontier 仍为 WHEELED 4 m、LEGGED 3 m。

- [ ] **Step 5: Run focused GREEN tests and commit**

```bash
PY=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
export PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q \
  training/lunar_policy_training/tests/test_parametric_training_alignment.py \
  training/lunar_policy_training/tests/test_cli.py \
  training/lunar_policy_training/tests/test_formal_cache.py
git diff --check
git add -- \
  training/lunar_policy_training/lunar_policy_training/training_semantics.py \
  training/lunar_policy_training/lunar_policy_training/cli.py \
  training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py \
  training/lunar_policy_training/lunar_policy_training/polar_data/task_cache.py \
  training/lunar_policy_training/tests/test_parametric_training_alignment.py \
  training/lunar_policy_training/tests/test_cli.py \
  training/lunar_policy_training/tests/test_formal_cache.py
git commit -m "feat(training): bind runs to directional observation semantics"
```

---

## Task 4: Extend the native visibility kernel with deterministic yaw-aware FOV

**Files:**

- Modify: `ros2_ws/src/lunar_planner_training_bridge/include/lunar_planner_training_bridge/visibility.hpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/visibility.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/test/visibility_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py`

**Native interface:**

```cpp
VisibilityKernel(double resolution_m, double range_m, double fov_rad);

std::vector<CandidateGain> EstimateCandidateGains(
    GridShape shape,
    std::span<const std::uint8_t> observed,
    std::span<const float> obstacle_ratio,
    std::span<const float> roi_ratio,
    std::span<const float> priority_weight,
    std::span<const GridCell> candidates,
    std::span<const double> candidate_yaws_rad) const;

std::vector<std::uint8_t> RevealFromPose(
    GridShape shape, GridCell pose, double yaw_rad,
    std::span<const float> truth_obstacle_ratio) const;

std::vector<std::uint8_t> RevealFromPoses(
    GridShape shape,
    std::span<const GridCell> poses,
    std::span<const double> yaws_rad,
    std::span<const float> truth_obstacle_ratios) const;
```

- [ ] **Step 1: Add native RED tests**

Tests must establish map/yaw convention:

- yaw `0` sees increasing column (east/right) but not west;
- yaw `pi/2` sees decreasing row (north/up) but not south;
- a 90° boundary ray at ±45° is included deterministically;
- an obstacle blocks only rays in the selected sector;
- per-candidate yaws produce different gains at the same cell;
- batch reveal equals ordered single-pose reveal;
- non-finite yaw, wrong yaw count, FOV <= 0 or FOV > 2π are rejected;
- a 2π kernel remains bit-exact with the pre-change full-circle fixture.

```bash
cmake -S ros2_ws/src/lunar_planner_training_bridge \
  -B /home/kai/CodexDownloads/lunar_navigation/parametric-capability-training-alignment/native-red \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build /home/kai/CodexDownloads/lunar_navigation/parametric-capability-training-alignment/native-red -j2
ctest --test-dir /home/kai/CodexDownloads/lunar_navigation/parametric-capability-training-alignment/native-red \
  --output-on-failure -R visibility
```

Expected RED: constructor/API lacks FOV/yaw.

- [ ] **Step 2: Implement the minimum sector filter**

保留当前圆盘 endpoint/ray 预计算。对每次请求按：

```cpp
bearing = std::atan2(-offset.row, offset.column);
delta = std::remainder(bearing - yaw_rad, 2.0 * std::numbers::pi);
inside = full_circle || std::abs(delta) <= fov_rad / 2.0 + 1.0e-12;
```

筛选 endpoint/ray。中心 pose 始终可见。不得改变 Bresenham、遮挡、ROI 权重或浮点累加顺序；全向路径必须继续遍历原有 endpoint 顺序。

- [ ] **Step 3: Bind exact NumPy contracts**

pybind 必须要求：

- scalar yaw 为有限 Python float；
- `candidate_yaws_rad` / `yaws_rad` 为 C-contiguous `float64 [N]`；
- 数量严格等于 candidate/pose 数；
- native 计算期间释放 GIL；
- 结果仍为 C-contiguous `float32 [N,2]` 或 `bool [N,H,W]`。

- [ ] **Step 4: Run native and bridge GREEN tests**

```bash
NATIVE=/home/kai/CodexDownloads/lunar_navigation/parametric-capability-training-alignment/native-task4
rm -rf "$NATIVE/build" "$NATIVE/install" "$NATIVE/log"
colcon build \
  --base-paths ros2_ws/src \
  --build-base "$NATIVE/build" --install-base "$NATIVE/install" --log-base "$NATIVE/log" \
  --packages-up-to lunar_planner_training_bridge \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
ctest --test-dir "$NATIVE/build/lunar_planner_training_bridge" \
  --output-on-failure -R visibility
source "$NATIVE/install/setup.bash"
PY=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
PYTHONPATH="$NATIVE/install/local/lib/python3.10/dist-packages:$PWD/training/lunar_policy_training" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q \
  ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py
```

- [ ] **Step 5: Commit**

```bash
git diff --check
git add -- \
  ros2_ws/src/lunar_planner_training_bridge/include/lunar_planner_training_bridge/visibility.hpp \
  ros2_ws/src/lunar_planner_training_bridge/src/visibility.cpp \
  ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp \
  ros2_ws/src/lunar_planner_training_bridge/test/visibility_test.cpp \
  ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py
git commit -m "feat(training): add directional native visibility"
```

---

## Task 5: Carry yaw through Python visibility and atomic observation transactions

**Files:**

- Create: `training/lunar_policy_training/tests/test_directional_visibility.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/visibility.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/sensor_observation.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/multires_observation.py`
- Modify: `training/lunar_policy_training/tests/test_visibility.py`
- Modify: `training/lunar_policy_training/tests/test_visibility_equivalence.py`
- Modify: `training/lunar_policy_training/tests/test_sensor_observation.py`
- Modify: `training/lunar_policy_training/tests/test_sensor_closed_loop.py`
- Modify: `training/lunar_policy_training/tests/test_multires_observation.py`

**Python interface:**

```python
estimate_candidate_gains(..., candidate_cells: np.ndarray,
                         candidate_yaws_rad: np.ndarray) -> np.ndarray
reveal_from_pose(truth_obstacle_ratio: np.ndarray,
                 pose_cell: tuple[int, int], yaw_rad: float) -> np.ndarray
reveal_from_poses(truth_obstacle_ratios: np.ndarray,
                  pose_cells: np.ndarray,
                  yaws_rad: np.ndarray) -> np.ndarray
```

- [ ] **Step 1: Add RED tests for directional rollout observation**

At the same obstacle-free pose, yaw 0 and yaw π must produce different 90° masks; four cardinal yaws must cover the full 10 m disk union. A trajectory containing the same 0.2 m cell at two different yaws must execute two visibility events, not collapse to one.

- [ ] **Step 2: Remove the full-circle guard and validate yaws**

Replace `_require_full_circle` with strict finite-yaw helpers. `SlowVisibilityReference` must apply the same sector convention as C++ so it remains a test oracle for both 90° and 360°.

- [ ] **Step 3: Make sensor state APIs explicit**

Change cell-level methods to require keyword-only yaw:

```python
observe(pose_cell, *, yaw_rad: float, elapsed_s: float)
observe_repeated(pose_cell, *, yaw_rad: float,
                 elapsed_steps_s: tuple[float, ...])
```

`observe_world*` derives yaw from `Pose2.yaw_rad`. No default yaw is allowed on production APIs, preventing accidental full-circle behavior.

- [ ] **Step 4: Preserve trajectory batching without collapsing yaw**

Replace `first_pose_by_cell` with a stable first-occurrence map keyed by:

```text
(detail_row, detail_column, canonical_float64_yaw_bits)
```

Canonical yaw is normalized to `[-pi, pi)` before its IEEE-754 bits are used. Do not round to coarse angle bins. Batch at most 32 `(truth window, pose cell, yaw)` items per native call, preserve event order, and include canonical yaw in `_PreparedDetailObservation` plus physical evidence hashing.

- [ ] **Step 5: Pass yaw into exact-position candidate gain**

Add `candidate_yaws_rad: np.ndarray` to:

- `estimate_candidate_gains_at_positions`;
- `_estimate_detail_candidate_gains`;
- every coarse fallback gain call.

Require exact `float64 [N]` shape and preserve the current detail-area scaling.

- [ ] **Step 6: Run GREEN tests and commit**

```bash
NATIVE=/home/kai/CodexDownloads/lunar_navigation/parametric-capability-training-alignment/native-task4
source "$NATIVE/install/setup.bash"
PY=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
export PYTHONPATH="$NATIVE/install/local/lib/python3.10/dist-packages:$PWD/model_contract:$PWD/training/lunar_policy_training"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q \
  training/lunar_policy_training/tests/test_directional_visibility.py \
  training/lunar_policy_training/tests/test_visibility.py \
  training/lunar_policy_training/tests/test_visibility_equivalence.py \
  training/lunar_policy_training/tests/test_sensor_observation.py \
  training/lunar_policy_training/tests/test_sensor_closed_loop.py \
  training/lunar_policy_training/tests/test_multires_observation.py
git diff --check
git add -- \
  training/lunar_policy_training/lunar_policy_training/environment/visibility.py \
  training/lunar_policy_training/lunar_policy_training/environment/sensor_observation.py \
  training/lunar_policy_training/lunar_policy_training/environment/multires_observation.py \
  training/lunar_policy_training/tests/test_directional_visibility.py \
  training/lunar_policy_training/tests/test_visibility.py \
  training/lunar_policy_training/tests/test_visibility_equivalence.py \
  training/lunar_policy_training/tests/test_sensor_observation.py \
  training/lunar_policy_training/tests/test_sensor_closed_loop.py \
  training/lunar_policy_training/tests/test_multires_observation.py
git commit -m "feat(training): observe ground motion with directional FOV"
```

---

## Task 6: Compute candidate standard gain from the ROI frontier normal

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`
- Modify: `training/lunar_policy_training/tests/test_reachable_constrained_ground_candidates.py`
- Modify: `training/lunar_policy_training/tests/test_formal_builder.py`
- Modify: `training/lunar_policy_training/tests/test_v3_environment.py`

**Internal witness:**

```python
@dataclass(frozen=True, slots=True)
class _NarrowFrontierStripWitness:
    pose_cell: tuple[int, int]
    canonical_yaw_rad: float
```

- [ ] **Step 1: Write RED tests for the yaw authority boundary**

Tests must prove:

- only `mission_roi AND NOT observed` neighbors influence the normal;
- map exterior and ROI-exterior unknown cells do not influence yaw;
- in grid convention, east unknown gives yaw 0 and north unknown gives yaw π/2;
- symmetric/empty/non-finite unknown direction rejects that anchor;
- rotating `canonical_yaw` changes standard expected gain in a 90° fixture;
- changing PPO `theta_rad` changes the formal request yaw and is never overwritten by canonical yaw;
- candidate ID for ground remains position-based with `heading_authority=policy-action`.

- [ ] **Step 2: Return yaw-bearing strip witnesses**

For each fixed 1/4, 1/2, 3/4 anchor, compute the task-ROI unknown-side centroid. For every safe 0.2 m strip pose:

```python
delta_column = unknown_centroid_column - pose_column
delta_row = unknown_centroid_row - pose_row
canonical_yaw = math.atan2(-delta_row, delta_column)
```

Reject zero-length or non-finite vectors. Preserve the current bounded strip (maximum 4 m, lateral half-width one 0.2 m cell), 96-anchor and 32-window limits. Do not add complete-frontier or residual-region fallback.

- [ ] **Step 3: Qualify all exact witnesses with their own yaw**

Add `canonical_yaw_rad` to `_RawFrontierCandidate`. Pass ordered `float64 [N]` yaws through both coarse-cell and exact-position gain APIs. The resulting `PhysicalCandidate.target_yaw_rad` may carry canonical yaw for diagnostics/features, but `_physical_candidate` must keep the ground identity independent of yaw.

- [ ] **Step 4: Preserve PPO execution authority**

In `FormalEpisode.build_request` retain:

```python
target_yaw = (
    candidate.target_yaw_rad if platform_type == "HOPPER"
    else action.theta_rad
)
```

Add a request-level test with one fixed candidate and two theta actions that produces distinct goal yaws while candidate IDs and positions stay identical.

- [ ] **Step 5: Run GREEN tests and commit**

```bash
NATIVE=/home/kai/CodexDownloads/lunar_navigation/parametric-capability-training-alignment/native-task4
source "$NATIVE/install/setup.bash"
PY=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
export PYTHONPATH="$NATIVE/install/local/lib/python3.10/dist-packages:$PWD/model_contract:$PWD/training/lunar_policy_training"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py \
  training/lunar_policy_training/tests/test_reachable_constrained_ground_candidates.py \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_v3_environment.py
git diff --check
git add -- \
  training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py \
  training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py \
  training/lunar_policy_training/tests/test_reachable_constrained_ground_candidates.py \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_v3_environment.py
git commit -m "feat(training): score ground candidates along frontier normals"
```

---

## Task 7: Rebuild a policy-independent directional coverability denominator

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/coverability.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/task_coverability.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/task_cache.py`
- Modify: `training/lunar_policy_training/tests/test_coverability.py`
- Modify: `training/lunar_policy_training/tests/test_formal_cache.py`
- Modify: `training/lunar_policy_training/tests/test_formal_builder.py`
- Modify: `training/lunar_policy_training/tests/test_training_smoke.py`

- [ ] **Step 1: Write RED denominator tests**

For obstacle-free and obstacle-containing fixtures, prove:

```python
union_90 = reveal(yaw=0) | reveal(yaw=pi/2) | reveal(yaw=pi) | reveal(yaw=-pi/2)
assert np.array_equal(union_90, reveal_with_10m_full_circle_kernel())
```

Also prove initial observation at yaw 0 is exactly one 90° sector and is a strict subset of the denominator whenever the task surrounds the start.

- [ ] **Step 2: Separate denominator visibility from rollout visibility**

`build_coverable_detail_mask` receives either explicit attainable yaws or a tested union callback. For current WHEELED/LEGGED capabilities, use a 10 m full-circle kernel only as an optimized representation of the four cardinal 90° union. Name the helper `build_ground_attainable_yaw_visibility_union` so it cannot be reused as rollout observation.

- [ ] **Step 3: Update the task builder**

`build_ground_task_coverability` must:

- accept 10 m/90° authority;
- use the full-azimuth union only for `coverable` denominator;
- use initial `Pose2.yaw_rad` (currently deterministic 0) with the 90° estimator for `initial_observed`;
- use directional standard yaw for initial candidate count;
- persist new sensor/visibility algorithm IDs in `PlatformTaskKey`;
- retain the 95% physical qualification calculation and 80% rollout success threshold.

HOPPER task-cache requests under the new observation identity must return an explicit unsupported/deferred error in this ground-only release; they must not silently reuse old 30 m/360° payloads.

- [ ] **Step 4: Prove old cache rejection and raw-source reuse**

Tests must show:

- an old platform task key with 30 m/360° semantics is rejected;
- a new WHEELED or LEGGED platform task includes observation digest and platform content digest;
- common terrain/source arrays are regenerated from the existing immutable source lock without rewriting source data;
- WHEELED physical derived tasks always rebuild;
- LEGGED is reused only if physical typed fields and algorithm identities are bit-exact, but its sensor-derived denominator still rebuilds;
- no HOPPER platform task is requested for `GROUND_R1`.

- [ ] **Step 5: Run GREEN tests and commit**

```bash
NATIVE=/home/kai/CodexDownloads/lunar_navigation/parametric-capability-training-alignment/native-task4
source "$NATIVE/install/setup.bash"
PY=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
export PYTHONPATH="$NATIVE/install/local/lib/python3.10/dist-packages:$PWD/model_contract:$PWD/training/lunar_policy_training"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q \
  training/lunar_policy_training/tests/test_coverability.py \
  training/lunar_policy_training/tests/test_formal_cache.py \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_training_smoke.py
git diff --check
git add -- \
  training/lunar_policy_training/lunar_policy_training/environment/coverability.py \
  training/lunar_policy_training/lunar_policy_training/polar_data/task_coverability.py \
  training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py \
  training/lunar_policy_training/lunar_policy_training/polar_data/task_cache.py \
  training/lunar_policy_training/tests/test_coverability.py \
  training/lunar_policy_training/tests/test_formal_cache.py \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_training_smoke.py
git commit -m "feat(training): freeze directional ground coverability"
```

---

## Task 8: Restrict update 457 warm start to sensor-compatible policy tensors

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/checkpoint.py`
- Modify: `training/lunar_policy_training/tests/test_checkpoint.py`
- Modify: `training/lunar_policy_training/tests/test_checkpoint_resume.py`
- Modify: `training/lunar_policy_training/tests/test_cli.py`

**Allowed prefixes:**

```python
POLICY_WARM_START_PREFIXES = (
    "global_encoder",
    "local_encoder",
    "pose_encoder",
    "platform_encoder",
    "cross_attention_blocks",
    "frontier_logit_head",
)

POLICY_WARM_START_RESET_PREFIXES = (
    "theta_sin_head",
    "theta_cos_head",
    "theta_kappa_head",
    "frontier_encoder",
    "frontier_position_encoder",
    "action_output_mlp",
    "value_mlp",
)
```

- [ ] **Step 1: Write RED warm-start tests**

Load a fixture with distinct parent values in every module. Require exact equality only for allowlisted prefixes and require deterministic new values for every reset prefix. Assert optimizer, scheduler, normalization, GAE, RNG, worker states, suppression and metrics journal are absent from the new run.

- [ ] **Step 2: Move theta heads into the reset set**

Retain exact-name/exact-shape checks and restricted `torch.load`. Any missing or shape-mismatched allowlisted tensor triggers the existing audited random fallback; no partial silent load is allowed.

- [ ] **Step 3: Add cross-semantics provenance**

Warm-start manifest must record:

```json
{
  "mode": "policy-partial",
  "source_observation": {"range_m": 30.0, "fov_deg": 360.0},
  "target_observation": {"range_m": 10.0, "fov_deg": 90.0},
  "equivalent_resume": false,
  "fresh_episode": true
}
```

Do not call this `resume` and do not inherit parent global step; the new run begins at step 0.

- [ ] **Step 4: Verify against the real update 457 file**

```bash
PY=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
PARENT=/home/kai/CodexDownloads/lunar_navigation/emergency-live-training-recovery/5897fd406a66c27025cf3dd8091e06ade1f1a270/run/checkpoints/update-00000457.pt
test -f "$PARENT"
export PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q \
  training/lunar_policy_training/tests/test_checkpoint.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py \
  training/lunar_policy_training/tests/test_cli.py
```

Expected: no unsafe deserialization; parameter evidence contains only `loaded` and `reset`, with theta heads classified as reset.

- [ ] **Step 5: Commit**

```bash
git diff --check
git add -- \
  training/lunar_policy_training/lunar_policy_training/checkpoint.py \
  training/lunar_policy_training/tests/test_checkpoint.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py \
  training/lunar_policy_training/tests/test_cli.py
git commit -m "fix(training): reset directional heads on sensor warm start"
```

---

## Task 9: Make preflight and qualification ground-only, bounded and non-blocking

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/formal_preflight.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/sensor_performance.py`
- Modify: `training/tools/qualify_task_cache_training_entry.py`
- Modify: `training/configs/rtx4080_super_v4_joint.yaml`
- Modify: `training/lunar_policy_training/tests/test_formal_preflight.py`
- Modify: `training/lunar_policy_training/tests/test_qualification_entry_targets.py`
- Modify: `training/lunar_policy_training/tests/test_sensor_performance.py`
- Modify: `training/lunar_policy_training/tests/test_config.py`

- [ ] **Step 1: Write RED tests for active-platform preflight**

Use a factory spy and require:

```python
assert allocation == {"WHEELED": 12, "LEGGED": 12}
assert created_platforms == {"WHEELED", "LEGGED"}
assert "HOPPER" not in created_platforms
```

One action per active platform must finish or legally terminate; no evaluation process may be launched inline.

- [ ] **Step 2: Replace three-platform naming and schema**

Advance the preflight schema once and replace:

- `three_platform_worker_construction` -> `active_ground_worker_construction`;
- `three_platform_macro_step` -> `active_ground_macro_step`;
- `three_platform_step_sha256` -> `active_ground_step_sha256`.

`_direct_environment_checks`, `_resume_check` and macro probe iterate the calibrated allocation, not the global platform enum. Remove the HOPPER fuel/delta-v probe from this ground-only preflight; keep HOPPER loader/bridge tests in Task 2/Task 1.

- [ ] **Step 3: Add a short qualification profile**

Add `--profile parametric-ground-v1` containing only current source nodes for:

1. deployment capability loading;
2. directional visibility;
3. ROI-normal candidate gain;
4. denominator/initial-observation separation;
5. warm-start reset contract;
6. WHEELED one-macro smoke;
7. LEGGED one-macro smoke.

The report remains `lunar-reward-v4-focused-qualification/v1`, task range 1–11, with exactly two `short_smokes` platform entries. Do not run HOPPER, full suites, latency repeats or three seeds.

- [ ] **Step 4: Keep current training allocation and light evaluation**

The config continues to express future joint allocation for curriculum compatibility, but `GROUND_R1` and `GROUND_R2` use the existing `ground_worker_counts` of 12+12. Tests must assert automatic heavy evaluation remains disabled for Reward V4 and no HOPPER worker is constructed in the current stage.

- [ ] **Step 5: Run focused GREEN tests and commit**

```bash
PY=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
export PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q \
  training/lunar_policy_training/tests/test_formal_preflight.py \
  training/lunar_policy_training/tests/test_qualification_entry_targets.py \
  training/lunar_policy_training/tests/test_sensor_performance.py \
  training/lunar_policy_training/tests/test_config.py \
  training/lunar_policy_training/tests/test_cli.py
git diff --check
git add -- \
  training/lunar_policy_training/lunar_policy_training/formal_preflight.py \
  training/lunar_policy_training/lunar_policy_training/cli.py \
  training/lunar_policy_training/lunar_policy_training/sensor_performance.py \
  training/tools/qualify_task_cache_training_entry.py \
  training/configs/rtx4080_super_v4_joint.yaml \
  training/lunar_policy_training/tests/test_formal_preflight.py \
  training/lunar_policy_training/tests/test_qualification_entry_targets.py \
  training/lunar_policy_training/tests/test_sensor_performance.py \
  training/lunar_policy_training/tests/test_config.py \
  training/lunar_policy_training/tests/test_cli.py
git commit -m "fix(training): bound qualification to active ground platforms"
```

---

## Task 10: Build Release evidence, rebuild only invalidated caches, and start training

**Files:**

- External only: `/home/kai/CodexDownloads/lunar_navigation/parametric-capability-training-alignment/`
- Source must be clean before evidence generation.

- [ ] **Step 1: Run final source verification**

```bash
ROOT=/home/kai/CodexDownloads/lunar_navigation/parametric-capability-training-alignment
NATIVE="$ROOT/native"
PY=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
rm -rf "$NATIVE/build" "$NATIVE/install" "$NATIVE/log"
mkdir -p "$ROOT" "$NATIVE"

colcon build \
  --base-paths ros2_ws/src \
  --build-base "$NATIVE/build" --install-base "$NATIVE/install" --log-base "$NATIVE/log" \
  --packages-up-to lunar_planner_training_bridge lunar_planner_ros \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source "$NATIVE/install/setup.bash"
export PYTHONPATH="$NATIVE/install/local/lib/python3.10/dist-packages:$PWD/model_contract:$PWD/training/lunar_policy_training"

ctest --test-dir "$NATIVE/build/lunar_planner_training_bridge" --output-on-failure
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q \
  tests/deployment/test_observation_capability.py \
  training/lunar_policy_training/tests/test_parametric_training_alignment.py \
  training/lunar_policy_training/tests/test_directional_visibility.py \
  training/lunar_policy_training/tests/test_reachable_constrained_ground_candidates.py \
  training/lunar_policy_training/tests/test_coverability.py \
  training/lunar_policy_training/tests/test_checkpoint.py \
  training/lunar_policy_training/tests/test_formal_preflight.py \
  training/lunar_policy_training/tests/test_qualification_entry_targets.py
"$PY" -m compileall -q \
  training/lunar_policy_training/lunar_policy_training \
  deployment/luna_runtime
git diff --check
test -z "$(git status --porcelain)"
```

Expected: all selected checks PASS in Release; no repository-generated artifacts.

- [ ] **Step 2: Generate current 10 m/90° performance evidence**

```bash
RUN="$ROOT/run"
mkdir -p "$RUN"
"$PY" training/tools/benchmark_sensor_observation.py \
  --native-benchmark \
    "$NATIVE/install/lib/lunar_planner_training_bridge/lunar_training_visibility_benchmark" \
  --output "$RUN/sensor-performance.json" \
  --workers 24
```

Expected: `passed=true`, current source/capability/semantics hashes, finite throughput. This is one source-bound measurement, not a three-seed evaluation.

- [ ] **Step 3: Rebuild the bounded manifest and only ground derived task artifacts**

```bash
CACHE="$ROOT/cache"
SOURCE_LOCK=/home/kai/CodexDownloads/lunar_navigation/volume3/data/locks/polar_source_lock_v1.json
SPLIT=/home/kai/CodexDownloads/lunar_navigation/volume3/data/splits/polar_split_v2.json

"$PY" -m lunar_policy_training.cli prepare-data \
  --source-lock "$SOURCE_LOCK" \
  --split-manifest "$SPLIT" \
  --cache-root "$CACHE" \
  --materialization bounded \
  --preflight-scenario-limit 128

"$PY" -m lunar_policy_training.cli prepare-tasks \
  --config training/configs/rtx4080_super_v4_joint.yaml \
  --cache-manifest "$CACHE/cache-manifest.json" \
  --stage GROUND_R1 \
  --prefetch-depth 24 \
  --builder-workers 8 \
  --report "$ROOT/task-prefetch-ground-r1.json"
```

Expected: source datasets are not recopied; new manifest binds 10 m/90°, WHEELED and LEGGED task artifacts exist, and no HOPPER platform task is built.

- [ ] **Step 4: Calibrate the fresh run and write focused qualification**

```bash
"$PY" -m lunar_policy_training.cli calibrate \
  --config training/configs/rtx4080_super_v4_joint.yaml \
  --artifact-root "$RUN" \
  --cache-manifest "$CACHE/cache-manifest.json" \
  --sensor-performance-report "$RUN/sensor-performance.json"

"$PY" training/tools/qualify_task_cache_training_entry.py \
  --profile parametric-ground-v1 \
  --cache-manifest "$CACHE/cache-manifest.json" \
  --artifact-root "$RUN/qualification" \
  --output "$RUN/qualification/ground-short-rollout.json"
```

Expected: qualification contains WHEELED and LEGGED only, zero failure/error/skip, and current source SHA.

- [ ] **Step 5: Execute the one-macro-per-ground-platform formal preflight**

```bash
PREFLIGHT="$RUN/formal-preflight"
mkdir -p "$PREFLIGHT"
"$PY" -m lunar_policy_training.cli formal-preflight \
  --config training/configs/rtx4080_super_v4_joint.yaml \
  --cache-manifest "$CACHE/cache-manifest.json" \
  --artifact-root "$PREFLIGHT" \
  --calibration-root "$RUN" \
  --sensor-performance-report "$RUN/sensor-performance.json"
```

Expected: WHEELED 与 LEGGED 各消费一个完整宏动作；无 HOPPER、hard error、非有限值、身份漂移或 inline evaluation。若失败，停在此处，不启动训练。

- [ ] **Step 6: Start the 24-worker fresh run from update 457 policy weights**

```bash
CONTROLLER="$ROOT/controller"
PARENT=/home/kai/CodexDownloads/lunar_navigation/emergency-live-training-recovery/5897fd406a66c27025cf3dd8091e06ade1f1a270/run/checkpoints/update-00000457.pt
mkdir -p "$CONTROLLER" "$RUN/checkpoints" "$RUN/journal" "$RUN/metrics"
test -f "$PARENT"

nohup "$PY" -m lunar_policy_training.cli train \
  --config training/configs/rtx4080_super_v4_joint.yaml \
  --artifact-root "$RUN" \
  --sensor-performance-report "$RUN/sensor-performance.json" \
  --formal-preflight-report "$PREFLIGHT/formal-preflight.json" \
  --qualification-report "$RUN/qualification/ground-short-rollout.json" \
  --policy-warm-start "$PARENT" \
  > "$CONTROLLER/train.log" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$CONTROLLER/train.pid"
```

Expected: manifest says fresh step 0, source observation 30/360, target observation 10/90, theta/value/training state reset, allocation `WHEELED=12, LEGGED=12`, no HOPPER.

- [ ] **Step 7: Report milestones without overstating health**

Verify separately:

```bash
PID=$(cat "$CONTROLLER/train.pid")
ps -p "$PID" -o pid=,etime=,%cpu=,%mem=,stat=,cmd=
python3 - "$RUN/run-manifest.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
print({
    "global_step": d["global_step"],
    "platform_allocation": d["platform_allocation"],
    "initialization": d["initialization"],
    "capability_sources": d["capability_sources"],
})
PY
```

Report four states independently:

1. training process launched;
2. first immutable checkpoint committed;
3. first `train.jsonl` update committed with finite PPO/reward/coverage values;
4. trend/convergence remains unknown until multiple committed updates exist.

Do not label process liveness, first checkpoint, or one update as convergence.

---

## Final Requirement Trace

- 10 m/90° authority: Tasks 2–5.
- Candidate standard yaw from ROI frontier normal: Task 6.
- PPO theta remains execution authority: Task 6.
- Policy-independent denominator and actual directional reward: Task 7.
- Old cache/exact resume rejected; raw sources preserved: Tasks 3 and 7.
- update 457 partial warm start with theta/value/training reset: Task 8.
- 12 WHEELED + 12 LEGGED, no HOPPER training: Tasks 9–10.
- No planner algorithm/range change: Tasks 3, 6 and final focused regression.
- Minimal qualification and fast restart: Tasks 9–10.

## Plan Completion Review

Before execution begins, the implementer must confirm all of the following from this file:

- there is no task that changes global or local path-planning algorithms;
- no task introduces a fixed 30 m ground candidate or global-route limit, and no task replaces local 4 m/3 m frontiers with the 10 m sensor range;
- no task adds model fields, candidate slots, reserve slots or Reward V4 terms;
- rollout observation always has an explicit yaw;
- full-circle visibility occurs only in the tested denominator union helper;
- no HOPPER worker appears in preflight, qualification or training launch;
- every production task has a preceding RED test and a bounded GREEN command;
- external artifacts remain outside Git.
