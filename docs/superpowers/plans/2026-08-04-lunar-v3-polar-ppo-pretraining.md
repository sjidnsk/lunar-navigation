# Lunar Navigation 卷三：月球极区 PPO 正式训练前置实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有三平台 proxy PPO 骨架迁移到冻结的月球极区 V2 合同，取得并锁定真实极区数据，完成能力无关训练组件和 capability-freeze 硬门，最终停在正式 seed 4080 rollout 之前。

**Architecture:** 保留现有共享 PPO、C++ v3 bridge、宏步环境、并行池和 checkpoint 骨架；原子替换动态 7/8/8/22 输入为固定 4/3/4/12 合同。真实 DEM 与程序危险物由独立数据层构建，平台可通行性直接复用 C++ v3 safe projection；正式 CLI 必须消费外置三平台 capability bundle，并把其身份绑定到 run manifest、checkpoint 和评估。

**Tech Stack:** Ubuntu 22.04 amd64、ROS 2 Humble、C++20、pybind11、Python 3.10、PyTorch 2.13 FP32、NumPy 2.2、Rasterio 1.4.4、Shapely 2.1.2、PyYAML、pytest、RTX 4080 SUPER。

## Global Constraints

- 批准设计以 [`2026-08-04-lunar-v3-polar-ppo-complete-design.md`](../specs/2026-08-04-lunar-v3-polar-ppo-complete-design.md) 为准；冲突时旧设计和旧卷三计划让位。
- 权威实施 worktree 为 `/mnt/data/WS/.lunar-navigation-worktrees/volume-3-policy-pipeline`；不得在 `volume-1-foundation` 重建卷三代码。
- 当前平台运动能力未确定；所有能力 fixture 必须标记 `development-smoke/test_only/proxy`，禁止启动正式 PPO rollout。
- 七输入固定为 `prior_channels [B,4,256,256]`、`coverage_summary [B,3,256,256]`、`local_crop [B,4,32,32]`、`frontier_features [B,64,12]`、`pose_features [B,6]`、`candidate_mask [B,64]`、`platform_context [B,3]`。
- 单网络、单权重和单组 head；不新增 capability 网络输入或平台专用 head。
- 全局画布固定 `1024 m × 1024 m @ 4 m/cell`；局部画布固定 `8 m × 8 m @ 0.25 m/cell`，二者均按 `map` 轴对齐。
- 训练热路径必须调用真实 C++ `lunar_planner_core` v3；Python A*、ROS Action 和缓存结果不得代替。
- 轮式和足式默认 `yaw_bin_count=64`；目标 yaw tolerance 固定 `π/24`。
- 奖励固定为覆盖 20、优先覆盖 5、首次成功 50、未成功终止 10、硬安全终止 50；不再奖励 goal progress。
- 正式验收为三平台分别至少 `61/64` 成功且硬违规为零，但本计划不运行正式评估。
- 原始数据和派生 artifact 只写 `/home/kai/CodexDownloads/lunar_navigation/volume3/` 下的明确目录；不得提交 DEM、DTM、GeoTIFF、checkpoint、日志或场景缓存。
- Python 测试使用 `/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python`，禁用 bytecode 和 pytest cache；ROS 命令先 source `/opt/ros/humble/setup.bash` 并确认 `ROS_DISTRO=humble`。
- 每个任务只提交任务相关文件；不得触碰 foundation worktree 的 `.vscode/` 或其它用户改动。

---

## File Structure

- `model_contract/lunar_model_contract/{observation,action}.py`：V2 七输入与四输出的唯一合同。
- `training/.../policy/`：固定 token geometry、共享 policy 和动作分布。
- `training/.../polar_data/`：官方 source lock、GeoTIFF、split 和程序危险物。
- `training/.../environment/{observation_builder,candidate_builder}.py`：固定观测和 64 候选。
- `lunar_planner_core/traversability_projection.*`：v3 safe projection 的窄公共只读结果。
- `training/.../capability_freeze.py`：外置 capability bundle closure 和正式启动门。
- `training/.../{reward,cli,checkpoint}.py`：新奖励及完整冻结身份。
- `training/tools/{fetch_polar_data,watch_training}.py`：仓库外下载和只读监督。

---

### Task 1: 冻结 ObservationContractV2 与 ActionContractV2

**Files:**
- Modify: `model_contract/lunar_model_contract/observation.py`
- Create: `model_contract/lunar_model_contract/action.py`
- Modify: `model_contract/lunar_model_contract/__init__.py`
- Modify: `model_contract/tests/test_observation_contract.py`
- Create: `model_contract/tests/test_action_contract.py`

**Interfaces:**
- Consumes: 已批准的固定七输入和四输出设计。
- Produces: `ObservationContractV2`、`validate_observation_inputs()`、`ActionContractV2`。

- [ ] **Step 1: 写 V2 输入失败测试**

```python
def test_v2_contract_freezes_exact_shapes_channels_and_fields():
    assert ObservationContractV2.shapes == {
        "prior_channels": (None, 4, 256, 256),
        "coverage_summary": (None, 3, 256, 256),
        "local_crop": (None, 4, 32, 32),
        "frontier_features": (None, 64, 12),
        "pose_features": (None, 6),
        "candidate_mask": (None, 64),
        "platform_context": (None, 3),
    }
    assert ObservationContractV2.frontier_fields[7:10] == (
        "normal_sin", "normal_cos", "normal_confidence"
    )
```

同时覆盖错误 channel、空间尺寸、M/F、NaN/Inf、错误 bool dtype 和非法 one-hot。合同层允许全 false mask，以便 builder 明确 bypass。

- [ ] **Step 2: 运行测试并确认 V1 无法满足**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -p no:cacheprovider -q model_contract/tests
```

Expected: FAIL，报告 `ObservationContractV2` 和 action contract 尚未定义。

- [ ] **Step 3: 实现单一 V2 常量与校验**

```python
class ObservationContractV2:
    version = "lunar-observation-contract/v2"
    input_names = (
        "prior_channels", "coverage_summary", "local_crop",
        "frontier_features", "pose_features", "candidate_mask",
        "platform_context",
    )
    prior_channels = (
        "observed_relative_elevation", "mission_priority",
        "observed_physical_obstacle_ratio", "active_platform_traversable_ratio",
    )
```

`validate_observation_inputs(mapping)` 先验证名称集合和 batch，再验证精确 dtype/shape/finite/one-hot。通道、12 个 candidate 字段及 6 个 pose 字段全部定义在 model contract。

- [ ] **Step 4: 实现动作合同**

```python
class ActionContractV2:
    version = "lunar-action-contract/v2"
    output_names = ("frontier_logits", "theta_mu", "theta_kappa", "value")
    candidate_count = 64
    yaw_tolerance_rad = math.pi / 24.0
    theta_kappa_min = 0.05
    theta_kappa_max = 64.0
```

测试三项 candidate 输出 `[B,64]`、value `[B]`、有限值及 kappa 范围。

- [ ] **Step 5: 运行合同测试并提交**

运行 Step 2；预期全 PASS，且 `rg -n 'ObservationContractV1|\[B,M,22\]' model_contract` 无生产命中。

```bash
git add model_contract
git commit -m "feat: freeze polar PPO v2 contracts"
```

---

### Task 2: 迁移共享 PPO、rollout 与 proxy smoke 到固定 V2

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/policy/observation_core.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/policy/observation.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/policy/backbone_core.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/policy/cross_attention.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/ppo/rollout.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/ppo/collector.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/proxy_scenario.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/eval/baselines.py`
- Modify: `training/lunar_policy_training/tests/test_policy_forward.py`
- Modify: `training/lunar_policy_training/tests/test_ppo_training.py`
- Modify: `training/lunar_policy_training/tests/test_collector.py`
- Modify: `training/lunar_policy_training/tests/test_parallel_pool.py`
- Modify: `training/lunar_policy_training/tests/test_curriculum.py`

**Interfaces:**
- Consumes: Task 1 两个 V2 合同。
- Produces: 固定 64 候选的共享 policy、PPO rollout 和 development-smoke padding。

- [ ] **Step 1: 把训练 fixture 改成精确 V2 并确认失败**

```python
def make_v2_batch(batch_size=2):
    mask = torch.zeros((batch_size, 64), dtype=torch.bool)
    mask[:, :3] = True
    return PolicyBatch(
        prior_channels=torch.zeros((batch_size, 4, 256, 256)),
        coverage_summary=torch.zeros((batch_size, 3, 256, 256)),
        local_crop=torch.zeros((batch_size, 4, 32, 32)),
        frontier_features=torch.zeros((batch_size, 64, 12)),
        pose_features=torch.zeros((batch_size, 6)),
        candidate_mask=mask,
        platform_context=torch.tensor([[1, 0, 0]], dtype=torch.float32).repeat(batch_size, 1),
    )
```

运行 policy/ppo/collector 三个测试文件；预期旧 15/8 通道与 22 维 candidate 导致 FAIL。

- [ ] **Step 2: 让训练包直接消费 model contract**

删除训练侧重复字段元组。`validate_policy_batch()` 精确检查 V2；全 false row 在调用 policy 前 bypass，动作分布仍拒绝全 false。

- [ ] **Step 3: 固定编码器 token geometry**

```python
self.global_encoder = MapEncoder(input_channels=7, output_grid=(32, 32))
self.local_encoder = MapEncoder(input_channels=4, output_grid=(16, 16))
self.frontier_encoder = nn.Linear(12, TOKEN_DIM)
```

`MapEncoder` 接收显式 `output_grid`；`TOKEN_DIM=128`、4 heads、2 blocks 不变。测试断言全局 1024、局部 256 个 tokens。

- [ ] **Step 4: 删除 recommended theta fallback 并固定 Von Mises**

```python
theta_mu = normalize_theta(torch.atan2(raw_sin, raw_cos))
theta_kappa = torch.clamp(F.softplus(raw_kappa), min=0.05, max=64.0)
```

sin/cos 初始 `(0,1)`，kappa 初始 1；测试 deterministic mean、采样、log-prob 重算和 mask。

- [ ] **Step 5: 将 proxy 改为 64 槽 development fixture**

保留 3 个实际 proxy 候选，填 V2 字段 0–11，剩余 61 个全零且 mask=false。rollout、共享内存和 baseline 不再接受动态 M 或 22 维字段。

- [ ] **Step 6: 运行回归并提交**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -p no:cacheprovider -q \
  model_contract/tests \
  training/lunar_policy_training/tests/test_policy_forward.py \
  training/lunar_policy_training/tests/test_ppo_training.py \
  training/lunar_policy_training/tests/test_collector.py \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_curriculum.py
git add training/lunar_policy_training
git commit -m "refactor: migrate shared PPO to polar v2 inputs"
```

---

### Task 3: 建立官方极区数据 registry、下载锁和 split

**Files:**
- Modify: `training/lunar_policy_training/pyproject.toml`
- Modify: `training/tools/lock_training_stack.py`
- Modify: `training/constraints/ubuntu22.04-rtx4080_super.txt`
- Create: `training/data_sources/polar_source_registry_v1.json`
- Create: `training/lunar_policy_training/lunar_policy_training/polar_data/__init__.py`
- Create: `training/lunar_policy_training/lunar_policy_training/polar_data/source_lock.py`
- Create: `training/lunar_policy_training/lunar_policy_training/polar_data/split.py`
- Create: `training/tools/fetch_polar_data.py`
- Create: `training/lunar_policy_training/tests/test_polar_source_lock.py`
- Create: `training/lunar_policy_training/tests/test_polar_split.py`
- Modify: `tools/check_repository_boundaries.py`
- Modify: `tests/foundation/test_repository_boundaries.py`

**Interfaces:**
- Consumes: NASA PGDA 5 m DEM/count 和 Zenodo record `17153447`。
- Produces: `PolarSourceLock`、`verify_source_lock()`、`build_split_catalog()` 和仓库外下载 CLI。

- [ ] **Step 1: 写 source lock 与泄漏失败测试**

```python
def test_jaxa_sites_are_holdout_only():
    catalog = build_split_catalog(nasa_windows(), jaxa_sites(), seed=4080)
    assert {row.split for row in catalog if row.source == "JAXA_LUPEX"} == {"holdout"}
    assert sum(row.split == "train" for row in catalog) == 192
    assert sum(row.split == "validation" for row in catalog) == 48
    assert sum(row.split == "test" for row in catalog) == 48
```

source lock 测试覆盖仓库内 data root、symlink、缺文件、错误 size/hash、CRS/transform/NoData 缺失和 JAXA 泄漏。

- [ ] **Step 2: 锁定并安装地理依赖**

把 `rasterio==1.4.4`、`shapely==2.1.2` 加入 lock script、constraints 和 pyproject，然后：

```bash
/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pip install "rasterio==1.4.4" "shapely==2.1.2"
```

运行 `pip check`；已有 ROS `typeguard` 缺口只记录，不修改系统 Python。

- [ ] **Step 3: 实现 registry 和原子下载器**

```json
{
  "schema": "lunar-polar-source-registry/v1",
  "sources": [
    {"id": "NASA_LOLA_87S_DEM", "url": "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldem_87s_5mpp.tif", "filename": "ldem_87s_5mpp.tif"},
    {"id": "NASA_LOLA_87S_COUNT", "url": "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldec_87s_5mpp.tif", "filename": "ldec_87s_5mpp.tif"},
    {"id": "JAXA_LUPEX_DATA_S1", "record_api": "https://zenodo.org/api/records/17153447", "filename": "DataS1.zip"}
  ]
}
```

下载器要求绝对、仓库外且非 symlink 的 data root，写同目录 `.part`、flush/fsync 后原子 rename；已有文件只在 lock 匹配时复用。

- [ ] **Step 4: 实现确定性 split**

NASA 窗口固定 1024 m、不重叠、跨 split 中心至少 2000 m、seed 4080，稳定选择 192/48/48；JAXA 六站点只为 holdout。输出不含用户名或绝对路径并提供 `split_sha256`。

- [ ] **Step 5: 扩展仓库边界**

拒绝 tracked `.tif/.tiff/.dem/.dtm/.img/.vrt` 和数据集 zip。测试在 pytest 临时目录中即时生成小型 GeoTIFF，不为真实或合成 raster 设置 Git 例外。新增 `terrain.tif` 与 `datasets/raw.zip` 失败测试。

- [ ] **Step 6: 运行测试并提交**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -p no:cacheprovider -q \
  training/lunar_policy_training/tests/test_polar_source_lock.py \
  training/lunar_policy_training/tests/test_polar_split.py \
  tests/foundation/test_repository_boundaries.py
python3 tools/check_repository_boundaries.py .
git add training/data_sources training/tools training/constraints \
  training/lunar_policy_training tools/check_repository_boundaries.py \
  tests/foundation/test_repository_boundaries.py
git commit -m "feat: add locked lunar polar data sources"
```

---

### Task 4: 构建固定地图、程序危险物与 64 个前沿候选

**Files:**
- Create: `training/lunar_policy_training/lunar_policy_training/polar_data/raster.py`
- Create: `training/lunar_policy_training/lunar_policy_training/polar_data/hazards.py`
- Create: `training/lunar_policy_training/lunar_policy_training/environment/observation_builder.py`
- Create: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Create: `training/lunar_policy_training/tests/test_polar_raster.py`
- Create: `training/lunar_policy_training/tests/test_procedural_hazards.py`
- Create: `training/lunar_policy_training/tests/test_observation_builder_v2.py`
- Create: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`

**Interfaces:**
- Consumes: V2 合同、source/split、mission/sensor 字段及注入的能力 projection。
- Produces: `PolarWindow`、`HazardScene`、`ObservationBuilderV2.build()`、`CandidateBuilderV2.build()`。

- [ ] **Step 1: 写固定几何和坐标失败测试**

```python
assert GLOBAL_GEOMETRY == GridGeometry(size_m=1024.0, resolution_m=4.0, cells=256)
assert LOCAL_GEOMETRY == GridGeometry(size_m=8.0, resolution_m=0.25, cells=32)
assert observation["prior_channels"].shape == (1, 4, 256, 256)
assert observation["local_crop"].shape == (1, 4, 32, 32)
```

覆盖 ROI 超画布、map/odom TF 缺失、NoData 不能变为已观测、比例层面积平均和高程双线性读取。

- [ ] **Step 2: 实现 raster 读取与语义重采样**

高程用 bilinear，mask/label 用 nearest，面积比例用 average。完整 DEM 真值只放 `WorldTruth`，网络 builder 只读已观测区域。

- [ ] **Step 3: 实现确定性月岩/月坑**

```python
scene_seed = sha256(f"{window_sha256}:{scenario_seed}:{GENERATOR_VERSION}")
```

独立 RNG streams 生成月岩 footprint、坑高程增量和禁入多边形。Shapely 求 union，polygon-cell intersection 得到物理障碍面积比例；坑坡不自动写入 physical obstacle。

- [ ] **Step 4: 实现固定 observation builder**

```python
class ObservationBuilderV2:
    def build(
        self, world: ObservedWorld, mission: MissionRaster, pose_map: Pose2,
        projection: PlatformProjection, candidates: CandidateBatch,
        platform_type: str,
    ) -> dict[str, np.ndarray]: ...
```

填充 4/3/4 通道、6 pose 和 one-hot；traversability/clearance 只来自注入 projection。

- [ ] **Step 5: 实现 observed-only 候选**

管线固定为 contour segments → anchors/standoff → observed-only LOS → 面积收益 → 每段代表点 → deterministic farthest fill → stable sort → 64 槽 padding。只输出 12 字段；无候选返回全 false，不制造当前位置。

- [ ] **Step 6: 运行测试并提交**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -p no:cacheprovider -q \
  training/lunar_policy_training/tests/test_polar_raster.py \
  training/lunar_policy_training/tests/test_procedural_hazards.py \
  training/lunar_policy_training/tests/test_observation_builder_v2.py \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py
git add training/lunar_policy_training
git commit -m "feat: build deterministic polar PPO observations"
```

---

### Task 5: 复用 v3 safe projection 并固定 64 yaw bins

**Files:**
- Create: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/traversability_projection.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/shared/traversability_projection.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_config.hpp`
- Create: `ros2_ws/src/lunar_planner_core/test/traversability_projection_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/public_api_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/include/lunar_planner_training_bridge/request.hpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/proxy_scenario.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py`
- Modify: `training/lunar_policy_training/tests/test_curriculum.py`
- Modify: `training/lunar_policy_training/tests/test_v3_environment.py`

**Interfaces:**
- Consumes: `WorldSnapshot`、`PlatformCapability`、`MapSafetyConfig` 和 internal `BuildSafeProjection()`。
- Produces: 公共 `ProjectTraversability()` 结果及 bridge `project_traversability()`；Python 不复制 v3 terrain limits。

- [ ] **Step 1: 写投影和 64-bin 失败测试**

```cpp
TEST(PlannerDefaults, GroundPlatformsUseSixtyFourYawBins) {
  EXPECT_EQ(WheelPlannerConfig{}.yaw_bin_count, 64U);
  EXPECT_EQ(LeggedPlannerConfig{}.yaw_bin_count, 64U);
}

TEST(TraversabilityProjection, SameTerrainDiffersByCapability) {
  const auto wheel = ProjectTraversability(world, strict_wheel, config, {});
  const auto legged = ProjectTraversability(world, permissive_legged, config, {});
  EXPECT_NE(wheel.projection->hard_feasible, legged.projection->hard_feasible);
}
```

- [ ] **Step 2: 实现窄公共结果，不暴露 internal class**

```cpp
struct TraversabilityProjection final {
  PlatformType platform_type{};
  std::size_t width{};
  std::size_t height{};
  std::vector<std::uint8_t> known;
  std::vector<std::uint8_t> hard_feasible;
  std::vector<float> clearance_m;
  std::vector<float> slope_rad;
  std::vector<float> roughness_m;
  std::vector<float> traversal_cost;
  std::vector<std::int32_t> connected_component;
};
```

内部调用已有 `BuildSafeProjection()` 并复制不可变结果；错误返回稳定 reason code。bridge/Python 不重算坡度、净空或平台 limits。

- [ ] **Step 3: 绑定 projection 并释放 GIL**

`PlannerBridge.project_traversability(request)` 接收同一 world/capability/config，返回数组字段；测试三平台、shape、dtype、错误地图和 GIL release。

- [ ] **Step 4: 固定 yaw 和 theta 目标语义**

wheel/leg 默认 bins 改为 64；development proxy 显式设置 64，`GoalRegion.yaw_tolerance_rad = π/24`。测试 sampled theta 实际进入 request，不能继续使用 `π` 容差。

- [ ] **Step 5: 固定 v3 拒绝后的重新决策语义**

正常不可行结果消耗一个 decision budget，在同一 observation identity 内只屏蔽被拒候选并重新决策；地图、任务 revision 或状态变化后清除临时拒绝集。候选全 false 时 bypass policy，不伪造当前位置。非法 outcome/directive/reference 组合继续使 rollout 作废。

- [ ] **Step 6: 构建并运行 C++/bridge 回归**

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/volume3/pretraining/log build \
  --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/volume3/pretraining/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/volume3/pretraining/install \
  --packages-select lunar_planner_core lunar_planner_training_bridge
source /home/kai/CodexDownloads/lunar_navigation/volume3/pretraining/install/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/volume3/pretraining/log test \
  --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/volume3/pretraining/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/volume3/pretraining/install \
  --packages-select lunar_planner_core lunar_planner_training_bridge
```

Expected: build/test 和 bridge Python tests 全部 PASS。

- [ ] **Step 7: 提交 v3 投影**

```bash
git add ros2_ws/src/lunar_planner_core ros2_ws/src/lunar_planner_training_bridge \
  training/lunar_policy_training
git commit -m "feat: expose v3 platform traversability projection"
```

---

### Task 6: 替换奖励、PPO 基线和终局状态

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/macro_step.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/reward.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/ppo/trainer_core.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/ppo/trainer.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/budget.py`
- Modify: `training/configs/rtx4080_super_smoke.yaml`
- Modify: `training/configs/rtx4080_super_v3_joint.yaml`
- Modify: `training/lunar_policy_training/tests/test_reward.py`
- Modify: `training/lunar_policy_training/tests/test_ppo_training.py`
- Modify: `training/lunar_policy_training/tests/test_training_budget.py`

**Interfaces:**
- Consumes: 宏步覆盖、优先覆盖、执行代价、耗时、终局和安全事件。
- Produces: `RewardInputsV2`、固定 `RewardWeightsV2` 和设计指定 PPO 配置。

- [ ] **Step 1: 写成功主导奖励失败测试**

```python
def test_success_dominates_all_dense_positive_shaping():
    assert reward(coverage_delta=1.0, priority_delta=1.0) == 25.0
    assert reward(success_first_crossing=True) == 50.0

def test_unsuccessful_and_hard_safety_terminals_are_distinct():
    assert terminal_adjustment(success=False, hard_safety=False) == -10.0
    assert terminal_adjustment(success=False, hard_safety=True) == -60.0
```

再覆盖成功只发一次、无新增覆盖 `-0.20`、三个 planner outcome 和非法 transition fail closed。

- [ ] **Step 2: 扩展 PlannerTransition 并移除 goal progress**

```python
@dataclass(frozen=True)
class PlannerTransition:
    mission_observed_delta: float
    priority_observed_delta: float
    normalized_plan_or_execution_cost: float
    normalized_macro_step_time: float
    executed_without_new_coverage: bool
    success_first_crossing: bool
    episode_ended_without_success: bool
    hard_safety_violation: bool
```

保留已有 outcome/directive/reference 字段；所有构造点显式赋值，不能从旧 `goal_progress` 猜测。

- [ ] **Step 3: 实现固定奖励**

权重精确为 `20/5/0.10/0.05/0.20/50/10/50`。覆盖增量只接受 `[0,1]`；硬安全仍写 gate event，不能只依靠数值扣分。

- [ ] **Step 4: 固定 PPO 配置**

YAML 写入 gamma `.995`、lambda `.95`、policy/value clip `.2`、AdamW `3e-4/1e-4/1e-5`、4 epochs、KL `.03`、value `.5`、frontier/theta entropy `.01/.001`、grad norm `.5`、horizon 32、FP32。loader 对缺失或漂移 fail closed。

- [ ] **Step 5: 实现显式 6 小时预算扩展**

初始累计上限仍为 86400 秒；只有用户命令显式提供正整数个 6 小时 block 才能增加上限。扩展写入 run manifest 和后续 checkpoint，不重置 `consumed_gpu_seconds`，watcher 不得自行调用。

- [ ] **Step 6: 运行回归并提交**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -p no:cacheprovider -q \
  training/lunar_policy_training/tests/test_reward.py \
  training/lunar_policy_training/tests/test_ppo_training.py \
  training/lunar_policy_training/tests/test_training_budget.py \
  training/lunar_policy_training/tests/test_evaluation_determinism.py
git add training
git commit -m "feat: prioritize PPO mission success"
```

---

### Task 7: 实现 capability-freeze 正式启动硬门

**Files:**
- Create: `training/lunar_policy_training/lunar_policy_training/capability_freeze.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/config.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/checkpoint.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/proxy_scenario.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/curriculum.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/evaluation/report.py`
- Modify: `training/configs/rtx4080_super_smoke.yaml`
- Modify: `training/configs/rtx4080_super_v3_joint.yaml`
- Create: `training/lunar_policy_training/tests/test_capability_freeze.py`
- Modify: `training/lunar_policy_training/tests/test_cli.py`
- Modify: `training/lunar_policy_training/tests/test_checkpoint_resume.py`
- Modify: `training/lunar_policy_training/tests/test_curriculum.py`
- Modify: `training/lunar_policy_training/tests/test_evaluation_determinism.py`
- Modify: `training/lunar_policy_training/tests/test_training_smoke.py`

**Interfaces:**
- Consumes: 外置 `lunar-training-capability-freeze/v1` lock 和三平台资源 closure。
- Produces: `FrozenCapabilityBundle`、`bundle_sha256`、显式 `run_kind`，并将同一能力实际注入 v3 worker。

- [ ] **Step 1: 写正式启动前失败测试**

```python
def test_formal_without_capability_lock_fails_before_artifact_or_cuda(tmp_path, monkeypatch):
    touched = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: touched.append("cuda"))
    with pytest.raises(PreflightError, match="formal capability bundle"):
        main(["train", "--config", FORMAL_CONFIG, "--artifact-root", str(tmp_path)])
    assert touched == []
    assert list(tmp_path.iterdir()) == []
```

覆盖缺/重复平台、错误 type/version、文档/URDF/mesh hash 漂移、字段顺序不影响 bundle hash、proxy lock 被 formal 拒绝。

- [ ] **Step 2: 实现 closure hash 和规范化 bundle**

```python
@dataclass(frozen=True)
class FrozenCapabilityBundle:
    schema: str
    platforms: tuple[FrozenPlatformCapability, ...]
    bundle_sha256: str
    formal_eligible: bool
```

哈希只含规范化类型、version、内容 hash 和按相对路径排序的资源 hash；绝对路径不进入身份。验证后一次解析为可 pickle typed capability，worker 不再读取源文件。

- [ ] **Step 3: 显式区分 run kind**

formal YAML 固定 `run_kind: formal`，smoke YAML 固定 `run_kind: development-smoke`。formal `train/resume/evaluate` 必须提供 `--capability-lock`；development smoke 只能用 proxy factory，不能生成正式 candidate。formal 不公开 `--max-updates`，smoke helper 保留有界更新。

- [ ] **Step 4: 确保 formal worker 实际消费 bundle**

```python
EnvironmentFactory = Callable[
    [int, str, FrozenPlatformCapability, ScenarioIdentity], V3Environment
]
```

formal 禁止调用硬编码 `_wheel/_legged/_hopper_capability()`；curriculum request 的 capability version/hash 必须匹配 bundle。

- [ ] **Step 5: 升级 checkpoint 和评估身份**

checkpoint 升级为 `lunar-ppo-checkpoint/v3`，新增 `run_kind`、data/split/generator/capability/reward/v3 哈希。resume 重算并逐项比较；旧 v2 只能由 development smoke 显式读取，formal 拒绝。evaluation report 写同一身份。

- [ ] **Step 6: 运行门禁回归并提交**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -p no:cacheprovider -q \
  training/lunar_policy_training/tests/test_capability_freeze.py \
  training/lunar_policy_training/tests/test_cli.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py \
  training/lunar_policy_training/tests/test_curriculum.py \
  training/lunar_policy_training/tests/test_evaluation_determinism.py \
  training/lunar_policy_training/tests/test_training_smoke.py
git add training
git commit -m "feat: gate formal training on capability freeze"
```

---

### Task 8: 下载并锁定 NASA/JAXA 数据，不启动训练

**Files:**
- Create: `docs/migration/volume-3-polar-data-source-lock.md`
- No tracked raster/data files.

**Interfaces:**
- Consumes: Task 3 下载器和 registry。
- Produces: 仓库外 verified source lock、NASA split manifest、JAXA holdout inventory 和非敏感摘要。

- [ ] **Step 1: 核对空间并创建明确目录**

```bash
df -h /home/kai /mnt/data
mkdir -p /home/kai/CodexDownloads/lunar_navigation/volume3/data/raw
mkdir -p /home/kai/CodexDownloads/lunar_navigation/volume3/data/locks
mkdir -p /home/kai/CodexDownloads/lunar_navigation/volume3/data/splits
```

Expected: `/home/kai` 至少 10 GB 可用；目录均在仓库外。

- [ ] **Step 2: 下载并验证官方源**

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  training/tools/fetch_polar_data.py \
  --registry training/data_sources/polar_source_registry_v1.json \
  --data-root /home/kai/CodexDownloads/lunar_navigation/volume3/data/raw \
  --lock-output /home/kai/CodexDownloads/lunar_navigation/volume3/data/locks/polar_source_lock_v1.json
```

连接中断时保留 `.part` 并用 HTTP range 恢复，不改用旧 DEM。再次运行只能校验复用。

- [ ] **Step 3: 生成 NASA split 与 JAXA inventory**

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m lunar_policy_training.polar_data.split \
  --source-lock /home/kai/CodexDownloads/lunar_navigation/volume3/data/locks/polar_source_lock_v1.json \
  --output /home/kai/CodexDownloads/lunar_navigation/volume3/data/splits/polar_split_v1.json \
  --seed 4080
```

Expected: NASA 为 192/48/48，JAXA 六站点只为 holdout，输出含 split hash。

- [ ] **Step 4: 记录非敏感锁摘要**

摘要只记录 source id、最终 URL、字节数、SHA-256、CRS、窗口数量、split hash、JAXA 六站点和生成命令；不记录用户名、绝对 data root 或 raster 内容。

- [ ] **Step 5: 验证并提交摘要**

```bash
python3 tools/check_repository_boundaries.py .
git status --short
git add docs/migration/volume-3-polar-data-source-lock.md
git commit -m "docs: record lunar polar source lock"
```

Expected: 没有 tracked `.tif/.zip/.npy/.npz`。

---

### Task 9: 端到端前置 smoke、只读监督与就绪交接

**Files:**
- Create: `training/tools/watch_training.py`
- Create: `training/lunar_policy_training/tests/test_watch_training.py`
- Modify: `training/lunar_policy_training/tests/test_training_smoke.py`
- Create: `docs/migration/volume-3-pretraining-readiness.md`
- Modify: `.superpowers/sdd/2026-08-02-lunar-navigation-volume-3-policy-pipeline/progress.md`

**Interfaces:**
- Consumes: Tasks 1–8 和 development-only capability fixture。
- Produces: 最多两个 PPO updates 的 development smoke、checkpoint/resume、formal 拒绝和待定能力清单。

- [ ] **Step 1: 写 watcher 与 formal-stop 测试**

watcher 只读 PID、run manifest、latest checkpoint mtime、GPU、磁盘和 finite metrics；每 10 分钟输出 JSON。pause marker 时为 `paused`，缺能力时为 `blocked-capability`，不自动改配置、预算或激活状态。

- [ ] **Step 2: 跑 CPU 端到端 development smoke**

用合成极区 GeoTIFF、程序危险物和 test-only 三平台 bundle 串联 source lock → split → observation/candidates → policy → C++ projection/plan → 一个 PPO update → checkpoint → resume；identity 必须完整且 artifact 只在 pytest tmp。

- [ ] **Step 3: 跑 RTX 4080 SUPER 有界 CUDA smoke**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -p no:cacheprovider -q -m cuda \
  training/lunar_policy_training/tests/test_policy_forward.py \
  training/lunar_policy_training/tests/test_training_smoke.py
```

Expected: FP32 forward、一个 update、信号边界保存和 resume PASS；不运行正式 seed。

- [ ] **Step 4: 证明 formal 停在能力门**

formal preflight 无 lock 时必须在 artifact/CUDA/worker 前失败；test-only bundle 也必须因 `formal_eligible=false` 拒绝。

- [ ] **Step 5: 运行全量前置验证**

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -p no:cacheprovider -q -m "not cuda" \
  model_contract/tests training/lunar_policy_training/tests
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
git diff --check
```

随后复用 Task 5 的外置 colcon 目录运行 core/bridge 全量测试。

- [ ] **Step 6: 写就绪交接并提交**

`volume-3-pretraining-readiness.md` 记录提交、数据/split hash、V2 合同、CPU/CUDA/ROS 结果、formal 拒绝证据和待定三平台能力 closure。状态只能写 `pretraining-ready / blocked-on-capability`。

```bash
git add training/tools/watch_training.py \
  training/lunar_policy_training/tests/test_watch_training.py \
  training/lunar_policy_training/tests/test_training_smoke.py \
  docs/migration/volume-3-pretraining-readiness.md \
  .superpowers/sdd/2026-08-02-lunar-navigation-volume-3-policy-pipeline/progress.md
git commit -m "test: qualify lunar PPO pretraining readiness"
```

---

## Final Stop Condition

计划完成后必须停止在 `pretraining-ready / blocked-on-capability`：不执行 seed 4080 正式 rollout，不消耗正式 24 小时 GPU 预算，不从旧 proxy checkpoint 恢复 formal，不生成正式 ONNX/四文件候选，也不推进 AGX 标签。

下一阶段唯一解锁条件是用户确定轮式、足式、飞跃式三份运动能力资料，形成 `lunar-training-capability-freeze/v1` 正式 bundle 并通过 schema、资源 closure 和 SHA-256 校验。随后重新生成平台化 traversability/candidate cache、冻结 worker/micro-batch，再启动正式训练。
