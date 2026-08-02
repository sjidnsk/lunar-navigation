# Lunar Navigation 卷三：PPO 训练与 ONNX/TensorRT 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把旧 Stage6 PPO 的有效模型、环境、训练和评估核心迁入新仓，并建立从 RTX 4080 checkpoint 到 AGX TensorRT 推理的唯一发布链。

**Architecture:** 训练代码位于 `training/`，只在 Ubuntu amd64/RTX 4080 安装；公共模型合同位于 `model_contract/`，训练、导出、探索输入和设备运行时共同消费。PyTorch 模型通过固定 inference wrapper 导出 ONNX，设备包只含四个文件；AGX 使用 C++ TensorRT runtime 和进程内 Python binding，不经 DDS 传高分辨率张量，也不允许后端静默回退。

**Tech Stack:** Python 3.10、PyTorch、NumPy、pytest、ONNX、ONNX Runtime、JSON Schema、C++20、pybind11、CUDA、TensorRT、ament_cmake、ament_python、rclpy、rclpy.lifecycle。

**Codex Estimate:** 22–41 agent-hours；完整 PPO 训练至收敛和外部等待不计入。

## Global Constraints

- PPO 训练、checkpoint 读取、评估和 ONNX 导出必须在 Ubuntu 22.04 amd64、RTX 4080 上进行。
- AGX Orin 只执行推理，不得安装 PyTorch、训练包、checkpoint、优化器、训练数据或训练日志。
- 设备模型包必须且只能包含 `policy.onnx`、`manifest.json`、`golden_inputs.npz`、`golden_outputs.npz`。
- 模型 manifest schema ID 必须是 `lunar-policy-manifest/v1`。
- TensorRT engine 必须在 AGX 本机从 ONNX 生成，缓存 key 必须包含 ONNX SHA-256 与设备/运行时指纹。
- FP32 必须先通过等价；FP16 只有单独数值与动作一致性通过后才允许启用。
- TensorRT 初始化、解析、构建、反序列化、shape/dtype 或黄金样例失败时节点不得进入 Active。
- TensorRT 失败不得静默回退到 PyTorch、ONNX Runtime 或 CPU。
- 探索管理、观测构建和 TensorRT 调用必须处于同一 Python 进程；高分辨率张量不得通过 DDS 传输。
- 学习模型不得修改硬安全边界或替代 C++ v3 的解析可行性判断。
- 训练 artifact root 必须由 CLI 显式传入绝对路径；缺失时拒绝启动。
- Windows 上的模型、缓存和大型临时产物必须写入 `D:/CodexDownloads`；生产训练和发布必须在 Ubuntu 执行。
- 旧 Stage authority/repair、ContentRef、多级 artifact hash graph、10k 行 runner 和历史 job state 不得迁移。

---

## File Structure

```text
migration/ppo_file_map.yaml
tools/import_ppo_core.py
model_contract/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/lunar_model_contract
├── schema/manifest.schema.json
├── lunar_model_contract/
│   ├── manifest.py
│   ├── hashing.py
│   ├── observation.py
│   └── package.py
└── tests/
training/
├── lunar_policy_training/
│   ├── pyproject.toml
│   ├── lunar_policy_training/
│   │   ├── policy/cross_attention.py
│   │   ├── environment/
│   │   ├── ppo/
│   │   ├── evaluation/
│   │   └── cli.py
│   └── tests/
├── model_export/
│   ├── inference_module.py
│   ├── export_onnx.py
│   ├── parity.py
│   ├── publish.py
│   └── tests/
└── configs/
    ├── rtx4080_smoke.yaml
    └── release_gate_v1.yaml
ros2_ws/src/lunar_policy_runtime/
├── include/lunar_policy_runtime/manifest.hpp
├── include/lunar_policy_runtime/engine_cache.hpp
├── include/lunar_policy_runtime/runner.hpp
├── src/manifest.cpp
├── src/engine_cache.cpp
├── src/tensorrt_runner.cpp
├── src/engine_builder.cpp
├── src/python_bindings.cpp
├── python/lunar_policy_runtime/__init__.py
├── python/lunar_policy_runtime/runtime.py
└── test/
ros2_ws/src/lunar_exploration/
├── lunar_exploration/input_adapter.py
├── lunar_exploration/observation_builder.py
├── lunar_exploration/mission_state.py
├── lunar_exploration/goal_selector.py
├── lunar_exploration/planner_client.py
├── lunar_exploration/node.py
└── test/
tests/device/policy_runtime/
```

### Task 1: 迁移 PPO 核心而不迁移 Stage 编排合同

**Execution environment:** Windows 清点并提交；Ubuntu RTX 4080 验证。

**Estimated Codex time:** 2–4 小时。

**Files:**
- Create: `migration/ppo_file_map.yaml`
- Create: `tools/import_ppo_core.py`
- Create: `training/lunar_policy_training/pyproject.toml`
- Create/Modify: `training/lunar_policy_training/lunar_policy_training/policy/cross_attention.py`
- Create/Modify: `training/lunar_policy_training/lunar_policy_training/policy/observation.py`
- Create/Modify: `training/lunar_policy_training/lunar_policy_training/environment/` selected files
- Create/Modify: `training/lunar_policy_training/lunar_policy_training/ppo/` selected files
- Create/Modify: `training/lunar_policy_training/lunar_policy_training/evaluation/` selected files
- Create: `training/lunar_policy_training/tests/test_import_boundary.py`
- Create: `training/lunar_policy_training/tests/test_policy_forward.py`

**Interfaces:**
- Consumes: 卷一固定旧 PPO commit 与选定源文件 hash。
- Produces: 可独立安装的 `lunar-policy-training` Python 包；不依赖旧仓路径和 Stage workflow。

- [ ] **Step 1: 写导入边界失败测试**

```python
def test_training_package_has_no_legacy_workflow_imports():
    violations = scan_imports(Path("training/lunar_policy_training"), (
        "lunar_exploration_ppo.workflows",
        "xunce_artifact_io",
        "stage6_input_pinning",
        "ContentRef",
    ))
    assert violations == []
```

- [ ] **Step 2: 固定允许迁移的模块**

`ppo_file_map.yaml` 必须只选择：policy/observation、cross-attention 模型、环境状态/覆盖/frontier/sensor、collector/rollout/trainer/checkpoint/resume、evaluation metrics/evaluator/baselines。明确排除 `workflows/`、所有 `scripts/run_*stage*`、authority/repair、durable job state 和 artifact registry。

- [ ] **Step 3: 实现受控导入和包内 import 改写**

`import_ppo_core.py` 验证源 commit 与 SHA-256，再复制为新包相对 import。若选定文件引用排除模块，脚本必须失败并列出 import，不允许自动复制依赖扩张范围。

- [ ] **Step 4: 保留模型前向数值基线**

固定种子和小型合成输入，断言以下输出名称、shape、dtype 与 finite：`frontier_logits`、`theta_mu`、`theta_kappa`、`value`。候选 mask 无效位置必须不会被 argmax 选择。

- [ ] **Step 5: 运行 CPU 与 RTX 4080 冒烟**

```bash
python3 -m pytest -q training/lunar_policy_training/tests/test_import_boundary.py training/lunar_policy_training/tests/test_policy_forward.py
CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q training/lunar_policy_training/tests/test_policy_forward.py -m cuda
```

Expected: CPU 和 CUDA 冒烟通过；CUDA 设备名称记录为 RTX 4080。

- [ ] **Step 6: 提交 PPO 核心**

```bash
git add migration/ppo_file_map.yaml tools/import_ppo_core.py training/lunar_policy_training
git commit -m "refactor: migrate maintainable PPO training core"
```

### Task 2: 建立显式训练配置、checkpoint 和单一 CLI

**Execution environment:** Ubuntu 22.04 amd64 + RTX 4080。

**Estimated Codex time:** 2–4 小时。

**Files:**
- Create: `training/lunar_policy_training/lunar_policy_training/config.py`
- Create: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Create: `training/lunar_policy_training/lunar_policy_training/checkpoint.py`
- Create: `training/configs/rtx4080_smoke.yaml`
- Create: `training/constraints/ubuntu22.04-rtx4080.txt`
- Create: `training/tools/lock_training_stack.py`
- Create: `training/lunar_policy_training/tests/test_cli.py`
- Create: `training/lunar_policy_training/tests/test_training_stack_lock.py`
- Create: `training/lunar_policy_training/tests/test_checkpoint_roundtrip.py`
- Create: `training/lunar_policy_training/tests/test_training_smoke.py`

**Interfaces:**
- Consumes: Task 1 policy/environment/trainer。
- Produces: `python -m lunar_policy_training.cli train|resume|evaluate` 和版本化 checkpoint。

- [ ] **Step 1: 写缺少 artifact root 的失败测试**

```python
def test_train_rejects_missing_artifact_root(cli_runner):
    result = cli_runner.invoke(app, ["train", "--config", SMOKE_CONFIG])
    assert result.exit_code != 0
    assert "--artifact-root must be an absolute path" in result.output
```

- [ ] **Step 2: 探测并锁定实际训练栈**

`lock_training_stack.py` 读取卷一 Ubuntu 指纹，并从当前环境验证 NVIDIA driver、`torch.version.cuda`、`torch.cuda.get_device_name(0)`、cuDNN、NumPy、ONNX 和 ONNX Runtime GPU provider；GPU 必须是 RTX 4080。它只把训练直接依赖的精确版本写入 `training/constraints/ubuntu22.04-rtx4080.txt`，不得把 Windows/Conda 路径或完整 `pip freeze` 写入仓库。测试使用固定 probe fixture 验证版本不匹配时退出非零。

- [ ] **Step 3: 定义训练配置**

`rtx4080_smoke.yaml` 固定随机种子、1 个短 rollout、2 个优化 epoch、小型场景集、FP32、`cuda:0` 和 `max_traversable_slope_deg: 30.0`；明确标记 synthetic terrain 为 `proxy`。

- [ ] **Step 4: 实现单一 CLI**

CLI 对所有命令要求绝对 `--artifact-root`；训练输出固定为：

```text
<artifact-root>/
├── checkpoints/latest.pt
├── metrics/train.jsonl
├── evaluation/report.json
└── run-manifest.json
```

`run-manifest.json` 只记录配置、commit、环境指纹和输出 hash，不构造跨阶段 authority/repair graph。

- [ ] **Step 5: 实现 checkpoint v1**

checkpoint 必须包含 schema version、model state、optimizer state、global step、config、normalization、随机数状态和 source commit；加载时严格验证网络/观测格式版本。checkpoint 只用于训练端，发布器不得复制它。

- [ ] **Step 6: 运行 roundtrip 和 RTX 4080 smoke**

```bash
python3 -m pytest -q training/lunar_policy_training/tests/test_cli.py training/lunar_policy_training/tests/test_training_stack_lock.py training/lunar_policy_training/tests/test_checkpoint_roundtrip.py
python3 training/tools/lock_training_stack.py --fingerprint "$LUNAR_UBUNTU_FINGERPRINT" --output training/constraints/ubuntu22.04-rtx4080.txt
python3 -m lunar_policy_training.cli train --config training/configs/rtx4080_smoke.yaml --artifact-root "$LUNAR_TRAIN_ARTIFACT_ROOT"
```

Expected: 训练完成、checkpoint 可恢复、输出目录外无文件写入。

- [ ] **Step 7: 提交训练入口**

```bash
git add training/lunar_policy_training training/configs/rtx4080_smoke.yaml training/constraints training/tools/lock_training_stack.py
git commit -m "feat: add explicit PPO training and checkpoint CLI"
```

### Task 3: 合并 G1/G2/G3 为单一模型发布评估

**Execution environment:** Ubuntu RTX 4080。

**Estimated Codex time:** 2–3 小时。

**Files:**
- Create: `training/configs/release_gate_v1.yaml`
- Create: `training/lunar_policy_training/lunar_policy_training/evaluation/release_gate.py`
- Create: `training/lunar_policy_training/lunar_policy_training/evaluation/report.py`
- Create: `training/lunar_policy_training/tests/test_release_gate.py`
- Create: `training/lunar_policy_training/tests/test_evaluation_determinism.py`

**Interfaces:**
- Consumes: Task 2 checkpoint 与固定 evaluation scenario schedule。
- Produces: `lunar-policy-release-evaluation/v1` 报告和单一 passed/failed 结论。

- [ ] **Step 1: 写门槛聚合测试**

```python
def test_release_gate_fails_any_safety_violation():
    metrics = valid_metrics() | {"safety_violation_count": 1}
    result = evaluate_release_gate(metrics, RELEASE_RULES)
    assert result.passed is False
    assert result.failed_rules == ("safety_violation_count_max",)
```

- [ ] **Step 2: 固定 release gate v1**

```yaml
schema_version: lunar-policy-release-gate/v1
rules:
  success_coverage_rate_min: 0.99
  safety_violation_count_max: 0
  invalid_action_count_max: 0
  selected_action_observed_safe_rate_min: 1.0
  deterministic_repeat_match_rate_min: 1.0
  output_finite_rate_min: 1.0
required_methods:
  - ppo_policy
  - nearest_frontier
  - gain_over_cost_frontier
```

这些规则吸收真正影响模型质量的 Stage6/G1/G2/G3 判断；不生成阶段 authority、repair 或多级批准 artifact。

- [ ] **Step 3: 实现一次性评估命令**

`evaluate` 对同一 scenario schedule 运行 PPO 与两个 baseline，记录 scenario seeds、coverage、安全、无效动作、规划失败和确定性重复；报告写入 `<artifact-root>/evaluation/report.json`。

- [ ] **Step 4: 运行确定性和门槛测试**

```bash
python3 -m pytest -q training/lunar_policy_training/tests/test_release_gate.py training/lunar_policy_training/tests/test_evaluation_determinism.py
python3 -m lunar_policy_training.cli evaluate --checkpoint "$LUNAR_TRAIN_ARTIFACT_ROOT/checkpoints/latest.pt" --gate training/configs/release_gate_v1.yaml --artifact-root "$LUNAR_TRAIN_ARTIFACT_ROOT"
```

Expected: 报告 schema 合法；同 seed 重复 hash 一致；不满足任一硬规则时退出非零。

- [ ] **Step 5: 提交统一评估**

```bash
git add training/configs/release_gate_v1.yaml training/lunar_policy_training
git commit -m "feat: consolidate PPO model release evaluation"
```

### Task 4: 定义共享模型合同与四文件包

**Execution environment:** Ubuntu amd64；纯 Python 测试也可在 Windows 运行但不作为发布依据。

**Estimated Codex time:** 1–2 小时。

**Files:**
- Create: `model_contract/package.xml`
- Create: `model_contract/setup.py`
- Create: `model_contract/setup.cfg`
- Create: `model_contract/resource/lunar_model_contract`
- Create: `model_contract/schema/manifest.schema.json`
- Create: `model_contract/lunar_model_contract/manifest.py`
- Create: `model_contract/lunar_model_contract/hashing.py`
- Create: `model_contract/lunar_model_contract/observation.py`
- Create: `model_contract/lunar_model_contract/package.py`
- Create: `model_contract/tests/test_manifest.py`
- Create: `model_contract/tests/test_package_boundary.py`
- Create: `model_contract/tests/test_observation_contract.py`
- Modify: `scripts/bootstrap_ubuntu.sh`
- Modify: `scripts/build_runtime.sh`

**Interfaces:**
- Consumes: Task 3 evaluation report 与 Task 1 policy tensor surface。
- Produces: `ModelManifest`、`ObservationContract`、`validate_model_package(path)`。

- [ ] **Step 1: 写只允许四文件的失败测试**

```python
def test_model_package_rejects_checkpoint(tmp_path):
    write_valid_model_package(tmp_path)
    (tmp_path / "checkpoint.pt").write_bytes(b"forbidden")
    with pytest.raises(ModelPackageError, match="unexpected file: checkpoint.pt"):
        validate_model_package(tmp_path)
```

- [ ] **Step 2: 定义 manifest schema**

required 字段固定为：`schema_version`、`model_id`、`version`、`source_commit`、`onnx_opset`、`observation_contract`、`action_contract`、`inputs`、`outputs`、`normalization`、`files`、`tolerances`、`release_evaluation`。`schema_version` const 为 `lunar-policy-manifest/v1`；每个 input/output 声明 name、shape、dtype。

- [ ] **Step 3: 固定部署 tensor 表面**

inputs：

```text
prior_channels      float32 [B,Cp,Hg,Wg]
coverage_summary    float32 [B,Cc,Hg,Wg]
local_crop          float32 [B,Cl,Hl,Wl]
frontier_features   float32 [B,M,22]
pose_features       float32 [B,P]
candidate_mask      bool    [B,M]
```

outputs：

```text
frontier_logits     float32 [B,M]
theta_mu            float32 [B,M]
theta_kappa         float32 [B,M]
value               float32 [B]
```

部署模型使用 manifest 固定的 H/W/M/P，不允许 runtime 猜测 shape。候选不足 M 时 padding 并以 mask 排除。

- [ ] **Step 4: 实现 package 校验**

校验精确文件集合、SHA-256、JSON Schema、NPZ key/shape/dtype、finite、normalization 长度、source commit 40 位 hash、评估 passed 与 report hash。禁止 NPZ object dtype 和 pickle。

- [ ] **Step 5: 测试并提交**

把 bootstrap 的 rosdep source paths 和默认 runtime 的 colcon base paths 都改为 `ros2_ws/src model_contract`；该包依赖仅限 Python 标准库、NumPy、PyYAML 和 jsonschema，不依赖 PyTorch/ONNX Runtime/TensorRT。

```bash
colcon build --base-paths model_contract --packages-select lunar_model_contract
source install/setup.bash
python3 -m pytest -q model_contract/tests
git add model_contract scripts/bootstrap_ubuntu.sh scripts/build_runtime.sh
git commit -m "feat: define deployable policy model contract"
```

### Task 5: 实现 ONNX 导出、黄金样例和等价验证

**Execution environment:** Ubuntu amd64 + RTX 4080。

**Estimated Codex time:** 4–7 小时。

**Files:**
- Create: `training/model_export/__init__.py`
- Create: `training/model_export/inference_module.py`
- Create: `training/model_export/export_onnx.py`
- Create: `training/model_export/parity.py`
- Create: `training/model_export/publish.py`
- Create: `training/model_export/verify_package.py`
- Create: `training/model_export/tests/test_inference_module.py`
- Create: `training/model_export/tests/test_export_onnx.py`
- Create: `training/model_export/tests/test_parity.py`
- Create: `training/model_export/tests/test_publish_boundary.py`

**Interfaces:**
- Consumes: 通过 Task 3 gate 的 checkpoint、Task 4 model contract。
- Produces: 四文件模型包和 PyTorch/ONNX 等价报告。

- [ ] **Step 1: 写 inference wrapper 输出测试**

```python
def test_inference_module_returns_only_deploy_outputs(policy, batch):
    outputs = InferenceModule(policy)(
        batch.prior_channels,
        batch.coverage_summary,
        batch.local_crop,
        batch.frontier_features,
        batch.pose_features,
        batch.candidate_mask,
    )
    assert len(outputs) == 4
    assert outputs[0].shape == batch.candidate_mask.shape
    assert outputs[3].shape == (batch.candidate_mask.shape[0],)
```

- [ ] **Step 2: 实现纯 tensor inference wrapper**

wrapper 内部构造 `PolicyBatch`，调用 policy，并只返回 `frontier_logits, theta_mu, theta_kappa, value`。不得导出 sampling、VonMises、训练 loss、optimizer 或中间 token。

- [ ] **Step 3: 实现固定 ONNX 导出**

使用 opset 17、固定部署 shape、constant folding 和显式 input/output names；导出前强制 `model.eval()`、FP32、固定 seed，导出后运行 `onnx.checker.check_model`。不设置动态 H/W/M axes。

- [ ] **Step 4: 生成黄金样例**

从固定 release evaluation scenario 中选择正常、稀疏候选、边界 mask 和高不确定性四类 observation；保存为无 pickle NPZ。黄金 outputs 来源必须是同 checkpoint 的 PyTorch inference wrapper。

- [ ] **Step 5: 实现 ONNX Runtime 等价**

FP32 初始容差固定 `atol=1e-4`、`rtol=1e-4`；要求所有输出 finite、shape/dtype 完全匹配、masked argmax candidate index 完全一致、选中 `theta_mu` 角度误差不超过 `1e-4 rad`。

- [ ] **Step 6: 实现原子发布**

发布器先在同一 artifact root 的临时目录生成四文件、验证、fsync，再原子 rename 到 `<artifact-root>/published/<model-id>/<version>`。目标已存在且 hash 不同则拒绝覆盖。

- [ ] **Step 7: 在 RTX 4080 发布 smoke 模型**

```bash
python3 -m pytest -q training/model_export/tests
python3 -m training.model_export.publish \
  --checkpoint "$LUNAR_TRAIN_ARTIFACT_ROOT/checkpoints/latest.pt" \
  --evaluation "$LUNAR_TRAIN_ARTIFACT_ROOT/evaluation/report.json" \
  --output-root "$LUNAR_TRAIN_ARTIFACT_ROOT/published"
python3 -m training.model_export.verify_package "$LUNAR_TRAIN_ARTIFACT_ROOT/published/smoke-policy/1"
```

Expected: 包精确包含四个文件，ONNX parity 通过。

- [ ] **Step 8: 提交导出链**

```bash
git add training/model_export
git commit -m "feat: publish verified ONNX policy packages"
```

### Task 6: 实现 TensorRT runtime、engine cache 与 Python binding

**Execution environment:** Ubuntu CPU 构建接口；AGX R36.0.0 构建 TensorRT 实现。

**Estimated Codex time:** 4–8 小时。

**Files:**
- Create: `ros2_ws/src/lunar_policy_runtime/package.xml`
- Create: `ros2_ws/src/lunar_policy_runtime/CMakeLists.txt`
- Create: `ros2_ws/src/lunar_policy_runtime/include/lunar_policy_runtime/manifest.hpp`
- Create: `ros2_ws/src/lunar_policy_runtime/include/lunar_policy_runtime/engine_cache.hpp`
- Create: `ros2_ws/src/lunar_policy_runtime/include/lunar_policy_runtime/runner.hpp`
- Create: `ros2_ws/src/lunar_policy_runtime/src/manifest.cpp`
- Create: `ros2_ws/src/lunar_policy_runtime/src/engine_cache.cpp`
- Create: `ros2_ws/src/lunar_policy_runtime/src/tensorrt_runner.cpp`
- Create: `ros2_ws/src/lunar_policy_runtime/src/engine_builder.cpp`
- Create: `ros2_ws/src/lunar_policy_runtime/src/python_bindings.cpp`
- Create: `ros2_ws/src/lunar_policy_runtime/src/build_engine_main.cpp`
- Create: `ros2_ws/src/lunar_policy_runtime/src/verify_engine_main.cpp`
- Create: `ros2_ws/src/lunar_policy_runtime/python/lunar_policy_runtime/__init__.py`
- Create: `ros2_ws/src/lunar_policy_runtime/python/lunar_policy_runtime/runtime.py`
- Create: `ros2_ws/src/lunar_policy_runtime/test/engine_cache_test.cpp`
- Create: `ros2_ws/src/lunar_policy_runtime/test/manifest_test.cpp`
- Create: `ros2_ws/src/lunar_policy_runtime/test/fake_runner_test.py`
- Create: `tests/device/policy_runtime/test_tensorrt_parity.py`

**Interfaces:**
- Consumes: Task 5 模型包和卷一设备指纹。
- Produces: `_lunar_policy_runtime.TensorRtRunner.infer(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]`。

- [ ] **Step 1: 写 cache key 测试**

```cpp
TEST(EngineCache, ChangesForAnyRuntimeFingerprintField) {
  auto base = MakeFingerprint();
  const auto key = MakeEngineCacheKey(kOnnxSha, base, Precision::kFp32);
  base.tensorrt_version = "different";
  EXPECT_NE(key, MakeEngineCacheKey(kOnnxSha, base, Precision::kFp32));
}
```

- [ ] **Step 2: 定义 engine cache key 和路径**

fingerprint 至少包含 GPU compute capability、aarch64、L4T、CUDA、TensorRT、precision、workspace bytes、builder flags 和 ONNX parser version。路径固定：

```text
/var/cache/lunar_navigation/tensorrt/<onnx-sha256>/<fingerprint-sha256>.engine
```

cache metadata 同目录保存 `<fingerprint-sha256>.json`，engine 与 metadata 写临时文件后原子 rename；hash 或字段不符即忽略旧 cache 并重建。
同一 key 的构建必须用 Linux `flock` 持有 `<fingerprint-sha256>.lock` 文件描述符；第二进程等待第一个完成后重新验证 cache，不得并发写同一 engine。

- [ ] **Step 3: 实现 TensorRT runner 接口**

```cpp
enum class TensorDType : std::uint8_t { kFloat32, kBool };

struct TensorView final {
  const void* data{};
  std::vector<std::int64_t> shape;
  TensorDType dtype{TensorDType::kFloat32};
};

struct OwnedTensor final {
  std::vector<std::byte> bytes;
  std::vector<std::int64_t> shape;
  TensorDType dtype{TensorDType::kFloat32};
};

using InputMap = std::unordered_map<std::string, TensorView>;
using OutputMap = std::unordered_map<std::string, OwnedTensor>;

class PolicyRunner {
 public:
  virtual ~PolicyRunner() = default;
  [[nodiscard]] virtual OutputMap Infer(const InputMap& inputs) = 0;
};
```

实现必须按 manifest 名称、shape、dtype 绑定 tensor，复用 CUDA stream/device buffers，检查 CUDA/TensorRT 每个返回值，输出 finite。TensorRT 8 与 10 的 enqueue/binding 差异集中在一个 `tensorrt_compat.hpp` 编译期分支，不扩散到 runner。

- [ ] **Step 4: 实现 engine builder CLI**

`build_engine` 加载并验证四文件包，用显式 batch 解析 ONNX，FP32 默认；只有传 `--precision fp16` 且设备支持时才启用 FP16。engine 构建失败必须退出非零，不创建 cache 完成文件。

- [ ] **Step 5: 实现 pybind11 进程内绑定**

Python binding 只接受 C-contiguous NumPy arrays；自动复制、dtype coercion 或 shape broadcast 必须拒绝。`infer` 释放 GIL 等待 GPU，返回拥有自身内存的 NumPy arrays。CPU build 提供可注入 `FakePolicyRunner` 测试接口，但生产工厂 `backend=tensorrt` 不可用时直接抛出 `RuntimeUnavailableError`。

- [ ] **Step 6: 运行 CPU 接口测试**

```bash
colcon build --base-paths ros2_ws/src model_contract --packages-select lunar_model_contract lunar_policy_runtime --cmake-args -DLUNAR_ENABLE_TENSORRT=OFF
colcon test --base-paths ros2_ws/src model_contract --packages-select lunar_policy_runtime
colcon test-result --verbose
```

Expected: manifest/cache/fake runner 通过，真实 TensorRT factory 明确报告 unavailable，不回退。

- [ ] **Step 7: 在 AGX 构建并通过 FP32 黄金等价**

```bash
colcon build --base-paths ros2_ws/src model_contract --packages-select lunar_model_contract lunar_policy_runtime --cmake-args -DLUNAR_ENABLE_TENSORRT=ON -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 run lunar_policy_runtime build_engine --model-dir /var/lib/lunar_navigation/models/current --cache-root /var/cache/lunar_navigation/tensorrt --precision fp32
ros2 run lunar_policy_runtime verify_engine --model-dir /var/lib/lunar_navigation/models/current --cache-root /var/cache/lunar_navigation/tensorrt
python3 -m pytest -q tests/device/policy_runtime/test_tensorrt_parity.py
```

Expected: 所有黄金输出在 manifest 容差内，candidate index 完全一致。

- [ ] **Step 8: 提交 runtime**

```bash
git add ros2_ws/src/lunar_policy_runtime tests/device/policy_runtime
git commit -m "feat: add strict TensorRT policy runtime"
```

### Task 7: 实现探索 Lifecycle 节点和同进程观测推理

**Execution environment:** Ubuntu CPU 使用 fake runner；AGX 使用 TensorRT runner。

**Estimated Codex time:** 4–7 小时。

**Files:**
- Create: `ros2_ws/src/lunar_exploration/package.xml`
- Create: `ros2_ws/src/lunar_exploration/setup.py`
- Create: `ros2_ws/src/lunar_exploration/setup.cfg`
- Create: `ros2_ws/src/lunar_exploration/resource/lunar_exploration`
- Create: `ros2_ws/src/lunar_exploration/lunar_exploration/input_adapter.py`
- Create: `ros2_ws/src/lunar_exploration/lunar_exploration/observation_builder.py`
- Create: `ros2_ws/src/lunar_exploration/lunar_exploration/mission_state.py`
- Create: `ros2_ws/src/lunar_exploration/lunar_exploration/goal_selector.py`
- Create: `ros2_ws/src/lunar_exploration/lunar_exploration/planner_client.py`
- Create: `ros2_ws/src/lunar_exploration/lunar_exploration/node.py`
- Create: `ros2_ws/src/lunar_exploration/config/exploration.schema.json`
- Create: `ros2_ws/src/lunar_exploration/test/test_input_adapter.py`
- Create: `ros2_ws/src/lunar_exploration/test/test_observation_builder.py`
- Create: `ros2_ws/src/lunar_exploration/test/test_goal_selector.py`
- Create: `ros2_ws/src/lunar_exploration/test/test_mission_state.py`
- Create: `ros2_ws/src/lunar_exploration/test/test_lifecycle.py`

**Interfaces:**
- Consumes: 外部 map/task/localization/TF、Task 6 runtime、卷一 Action、卷二 Action server。
- Produces: 与当前 mission revision 绑定的 `PlanMotion.Goal`；不发布运动控制命令。

- [ ] **Step 1: 写任务状态与观测一致性测试**

覆盖 revision 乱序、ACTIVE/PAUSED/CANCELED、定位 UNKNOWN/INVALID/RELOCALIZING/DEGRADED、地图过期、缺层、candidate padding/mask、normalization 和同输入确定性。

- [ ] **Step 2: 实现输入适配**

`input_adapter.py` 直接订阅外部 Topic，生成探索所需 typed snapshot；不得重发布大地图。字段、frame、时间和范围不合法时返回明确错误，不调用 observation builder。

- [ ] **Step 3: 复用共享 ObservationContract**

`observation_builder.py` 必须从 `lunar_model_contract.observation` 导入 shape、channel order、normalization 和 padding 规则；训练端也使用同一模块。输出六个 C-contiguous arrays，name/shape/dtype 与 manifest 完全一致。

- [ ] **Step 4: 实现确定性目标选择**

运行时只用 `argmax(masked frontier_logits)` 和该 candidate 的 `theta_mu`；任何 NaN、Inf、越界、全 false mask 或 shape 不符都丢弃本轮目标并发布诊断。目标必须带当前 mission_id/revision 和 map frame。

- [ ] **Step 5: 实现 Lifecycle 配置门槛**

configure 必须通过 ament package share 加载并冻结外部观测能力 YAML/JSON，验证模型四文件、manifest、TensorRT runner、黄金样例、`sensor_range_m`、`sensor_fov_deg` 和 observation channels；任一失败返回 FAILURE。activate 后才订阅/定时推理；PAUSED 停止推理并取消未提交 Action，CANCELED 清任务上下文。

- [ ] **Step 6: 实现 Action client 规则**

一次只保留一个未完成规划 goal；新模型目标默认不替换，只有任务 revision 增加或原 goal 明确失效时设置 `replace_active_request=true`。result mission_revision 不等于当前 revision 时丢弃。

- [ ] **Step 7: 运行 CPU fake 与 ROS 测试**

```bash
colcon build --base-paths ros2_ws/src model_contract --packages-up-to lunar_exploration --cmake-args -DLUNAR_ENABLE_TENSORRT=OFF
colcon test --base-paths ros2_ws/src model_contract --packages-select lunar_exploration
colcon test-result --verbose
```

Expected: 单进程观测→fake inference→目标→Action client 测试通过；没有高分辨率 tensor Topic。

- [ ] **Step 8: 提交探索节点**

```bash
git add ros2_ws/src/lunar_exploration
git commit -m "feat: add lifecycle PPO exploration node"
```

### Task 8: 验证 AGX 推理性能、缓存和失败不降级

**Execution environment:** AGX Orin 64GB、R36.0.0。

**Estimated Codex time:** 2–4 小时，engine 首次构建耗时包含在内。

**Files:**
- Create: `tests/device/policy_runtime/test_engine_cache_reuse.py`
- Create: `tests/device/policy_runtime/test_runtime_failure_modes.py`
- Create: `tests/device/policy_runtime/benchmark_inference.py`
- Create: `docs/migration/volume-3-agx-report-template.md`

**Interfaces:**
- Consumes: Tasks 5–7 同一模型包。
- Produces: TensorRT FP32 parity、cache reuse、P95 和故障报告。

- [ ] **Step 1: 验证首次构建与二次复用**

第一次删除仅针对一个明确测试 key 的临时 cache 文件，构建 engine；第二次加载必须复用且不改变 engine hash/mtime。修改 ONNX hash 或 fingerprint fixture 后必须产生不同 key。

- [ ] **Step 2: 验证失败不降级**

逐个使用损坏 ONNX、错误 manifest hash、缺黄金 key、错误 shape、NaN output、损坏 engine、fingerprint mismatch。每个场景必须导致 runtime 创建失败或节点保持 Inactive；进程日志不得出现 PyTorch/ORT/CPU fallback。

- [ ] **Step 3: 测量推理 P95**

```bash
python3 tests/device/policy_runtime/benchmark_inference.py \
  --model-dir /var/lib/lunar_navigation/models/current \
  --warmup 100 \
  --iterations 1000 \
  --output /var/log/lunar_navigation/policy-benchmark.json
```

Expected: 报告包含设备指纹、模型 hash、决策周期、median/p95/max；`p95 <= 0.8 * decision_period`。

本卷发布精度固定为 FP32。FP16 不随本卷自动启用；未来启用时必须新增独立 manifest tolerance、黄金等价、candidate index 和设备性能验收。

- [ ] **Step 4: 提交设备测试**

```bash
git add tests/device/policy_runtime docs/migration/volume-3-agx-report-template.md
git commit -m "test: qualify AGX policy inference runtime"
```

### Task 9: 卷三全量验收与模型发布回退点

**Execution environment:** Ubuntu RTX 4080 与 AGX R36.0.0。

**Estimated Codex time:** 1–2 小时。

**Files:**
- Create: `docs/migration/volume-3-completion.md`
- Verify: `training/`
- Verify: `model_contract/`
- Verify: `ros2_ws/src/lunar_policy_runtime/`
- Verify: `ros2_ws/src/lunar_exploration/`

**Interfaces:**
- Consumes: Tasks 1–8。
- Produces: `policy-pipeline-v1` tag 与通过等价的四文件模型包。

- [ ] **Step 1: 运行 Ubuntu 全量门槛**

```bash
python3 -m pytest -q model_contract/tests training/lunar_policy_training/tests training/model_export/tests
colcon build --base-paths ros2_ws/src model_contract --packages-up-to lunar_exploration --cmake-args -DLUNAR_ENABLE_TENSORRT=OFF
colcon test --base-paths ros2_ws/src model_contract --packages-select lunar_policy_runtime lunar_exploration
colcon test-result --verbose
```

Expected: 全部通过。

- [ ] **Step 2: 审计部署边界**

```bash
find "$LUNAR_TRAIN_ARTIFACT_ROOT/published/smoke-policy/1" -maxdepth 1 -type f -printf '%f\n' | sort
rg -n 'import torch|onnxruntime|fallback' ros2_ws/src/lunar_policy_runtime ros2_ws/src/lunar_exploration
```

Expected: 第一条只输出四个允许文件；运行包不 import torch/onnxruntime，`fallback` 只可出现在明确禁止或测试断言文本。

- [ ] **Step 3: 运行 AGX completion gate**

```bash
ros2 run lunar_policy_runtime verify_engine --model-dir /var/lib/lunar_navigation/models/current --cache-root /var/cache/lunar_navigation/tensorrt
python3 -m pytest -q tests/device/policy_runtime
```

Expected: FP32 等价、cache、故障和性能全部通过。

- [ ] **Step 4: 记录并标记 release point**

`volume-3-completion.md` 记录 source commit、checkpoint hash（仅记录，不复制）、ONNX hash、manifest hash、evaluation report hash、Ubuntu/AGX 指纹 hash、TensorRT precision 和测试结果。

```bash
git add docs/migration/volume-3-completion.md
git commit -m "docs: record policy pipeline completion gate"
git tag -a policy-pipeline-v1 -m "PPO ONNX TensorRT pipeline v1"
```

**Rollback:** 模型发布失败时保持上一个完整四文件模型包和 engine cache；不得覆盖同 model_id/version。训练 checkpoint 留在 Ubuntu artifact root；AGX `current` 模型链接不切换，旧部署保持运行。
