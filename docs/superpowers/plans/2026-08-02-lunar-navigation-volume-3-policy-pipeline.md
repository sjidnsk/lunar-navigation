# Lunar Navigation 卷三：v3 三平台 PPO 与 ONNX/TensorRT 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 RTX 4080 SUPER 上用轮式、足式和飞跃式 C++ v3 规划闭环重新训练一个共享 PPO 网络，并建立从可恢复 checkpoint 到 AGX TensorRT 推理的唯一发布链。

**Architecture:** 训练代码位于 `training/`，通过训练专用 `pybind11` 包进程内调用 ROS 无关的 `lunar_planner_core`；一个共享策略通过 `platform_context float32 [B,3]` 区分三类平台。公共七输入模型合同位于 `model_contract/`，训练、ONNX 导出、探索节点和 TensorRT runtime 共同消费；ROS Action 只用于部署集成，不进入训练热循环。

**Tech Stack:** Ubuntu 22.04 amd64、ROS 2 Humble、Python 3.10、PyTorch、NumPy、pytest、C++20、pybind11、ament_cmake、ONNX opset 17、ONNX Runtime、JSON Schema、CUDA、TensorRT、rclpy、rclpy.lifecycle。

**Codex Estimate:** 36–62 agent-hours，加最多 24 小时累计 GPU 工作时间；AGX 不可用等待和物理操作不计入。

## Global Constraints

- 批准设计以 [`2026-08-03-lunar-v3-three-platform-ppo-retraining-design.md`](../specs/2026-08-03-lunar-v3-three-platform-ppo-retraining-design.md) 为准。
- U80/A* 模型只作历史保底参考，不导入 checkpoint、不继续训练，也不作为 v3 模型发布基线。
- 只训练并发布一个 PPO 网络、一套权重、一个 checkpoint 候选和一个 ONNX；禁止平台专用网络或平台专用 head。
- 七项模型输入固定为 `prior_channels`、`coverage_summary`、`local_crop`、`frontier_features`、`pose_features`、`candidate_mask`、`platform_context`。
- `platform_context` 固定为 `float32 [B,3]` one-hot：轮式 `[1,0,0]`、足式 `[0,1,0]`、飞跃式 `[0,0,1]`；不包含 capability 数值。
- 部署侧平台类型来源和所有权以 [`docs/interfaces/external-input-baseline.md`](../../interfaces/external-input-baseline.md) 的 `platform-control-capability-source/v1` 为准；本卷只校验并编码，不复制静态能力资料。
- 训练每个规划执行宏步必须调用本仓 C++ `lunar_planner_core` v3；Python A*、ROS Action、DDS 和规划结果缓存不得替代训练热路径。
- v3 独占硬可行性和安全裁决；PPO 不得执行 v3 拒绝或未认证的参考，不得覆盖已承诺飞跃。
- 训练、checkpoint、评估和 ONNX 导出必须在 Ubuntu 22.04 amd64、RTX 4080 SUPER 上执行；AGX 只执行推理。
- 正式训练只有一个固定随机种子；奖励校准允许三个短种子，但必须包含在 24 小时总 GPU 预算内。
- 总 GPU 工作时间最多 24 小时，暂停期间不计时，恢复后累计计时不得清零。
- 自动 checkpoint 周期固定 30 分钟；正常信号在完整 PPO update 边界立即保存。
- 轮式、足式和飞跃式的 `success_coverage_rate` 必须分别不低于 `0.95`；任何平台失败都令整体失败。
- 三平台安全违规和非法动作必须分别为零，输出有限率必须为 `1.0`，飞跃承诺违规必须为零。
- 当前训练地形和能力数据必须标记为 `proxy`，不得写成实机能力结论。
- 训练 artifact root 必须由 CLI 显式传入仓库外的绝对路径；checkpoint、日志、场景缓存和 optimizer 不得提交 Git。
- Python 训练虚拟环境必须位于仓库外的绝对路径 `LUNAR_TRAIN_VENV`；不得在仓库内创建 `.venv` 或复制系统 Python 环境。
- 设备模型包必须且只能包含 `policy.onnx`、`manifest.json`、`golden_inputs.npz`、`golden_outputs.npz`。
- 模型 manifest schema ID 固定为 `lunar-policy-manifest/v1`；TensorRT engine 必须在 AGX 本机由 ONNX 生成。
- FP32 必须先通过等价；TensorRT 失败不得静默回退到 PyTorch、ONNX Runtime 或 CPU。
- 高分辨率 PPO tensor 只能在探索节点进程内传递，不得经 DDS 发布。
- 不迁移旧 Stage authority/repair、ContentRef、多级 artifact graph、job registry 或历史 durable state。

## AGX Git 安装协调边界

本卷继续遵循 [`2026-08-03-agx-git-one-command-deployment-design.md`](../specs/2026-08-03-agx-git-one-command-deployment-design.md)。卷三只向卷四交付一个仓库外绝对路径的四文件候选目录；卷四用 `deploy_agx_from_git.sh --candidate <absolute-path>` 安装。Git 只传递源码和发布身份，不提交模型包、checkpoint 或 TensorRT engine。

状态顺序保持：

1. `model-package-ready`：三平台 95% 正式发布评估、ONNX 等价、四文件包校验、Ubuntu runtime 接口和探索 fake-runner 测试全部通过。
2. `device-verified`：AGX 本机构建 TensorRT engine，设备等价、缓存、失败和性能测试全部通过。
3. `policy-pipeline-v1`：只能在 `device-verified` 后创建。

AGX 不可用时只能记录 `model-package-ready` 和 `AGX native verification: pending`。AGX 验证必须使用绝对版本目录 `/var/lib/lunar_navigation/models/<model-id>/<version>`；qualified manifest 和人工激活前不得读取或修改 `current` 链接。

---

## File Structure

```text
migration/ppo_file_map.yaml
tools/import_ppo_core.py
model_contract/
├── package.xml
├── setup.py
├── setup.cfg
├── schema/manifest.schema.json
├── lunar_model_contract/
│   ├── observation.py
│   ├── manifest.py
│   ├── hashing.py
│   └── package.py
└── tests/
training/
├── lunar_policy_training/
│   ├── pyproject.toml
│   ├── lunar_policy_training/
│   │   ├── policy/
│   │   ├── environment/
│   │   │   ├── v3_environment.py
│   │   │   ├── macro_step.py
│   │   │   └── parallel_pool.py
│   │   ├── ppo/
│   │   ├── evaluation/
│   │   ├── reward.py
│   │   ├── curriculum.py
│   │   ├── budget.py
│   │   ├── checkpoint.py
│   │   └── cli.py
│   └── tests/
├── model_export/
└── configs/
    ├── rtx4080_super_smoke.yaml
    ├── rtx4080_super_v3_joint.yaml
    ├── candidate_gate_v1.yaml
    └── release_gate_v1.yaml
ros2_ws/src/lunar_planner_training_bridge/
├── package.xml
├── CMakeLists.txt
├── include/lunar_planner_training_bridge/request.hpp
├── src/conversions.cpp
├── src/python_bindings.cpp
├── python/lunar_planner_training_bridge/__init__.py
└── test/
ros2_ws/src/lunar_policy_runtime/
ros2_ws/src/lunar_exploration/
tests/device/policy_runtime/
docs/migration/
├── volume-3-training-candidate.md
├── volume-3-ubuntu-readiness.md
└── volume-3-completion.md
```

### Task 1: 迁移共享 PPO 骨干并固定七输入合同

**Execution environment:** Ubuntu 22.04 amd64；RTX 4080 SUPER 只用于 CUDA forward 冒烟。

**Estimated Codex time:** 4–6 小时。

**Files:**
- Create: `migration/ppo_file_map.yaml`
- Create: `tools/import_ppo_core.py`
- Create: `model_contract/package.xml`
- Create: `model_contract/setup.py`
- Create: `model_contract/setup.cfg`
- Create: `model_contract/resource/lunar_model_contract`
- Create: `model_contract/lunar_model_contract/__init__.py`
- Create: `model_contract/lunar_model_contract/observation.py`
- Create: `model_contract/tests/test_observation_contract.py`
- Create: `training/lunar_policy_training/pyproject.toml`
- Create/Modify: `training/lunar_policy_training/lunar_policy_training/policy/observation.py`
- Create/Modify: `training/lunar_policy_training/lunar_policy_training/policy/cross_attention.py`
- Create/Modify: `training/lunar_policy_training/lunar_policy_training/ppo/` selected files
- Create: `training/lunar_policy_training/tests/test_import_boundary.py`
- Create: `training/lunar_policy_training/tests/test_policy_forward.py`

**Interfaces:**
- Consumes: `migration/source_inventory.yaml` 中固定旧 PPO commit/hash，仅迁移可维护模型和 PPO 数学核心。
- Produces: `ObservationContractV1`；`PolicyBatch` 七 tensor；`CrossAttentionPolicy.forward(batch: PolicyBatch) -> PolicyOutput`。

- [ ] **Step 1: 准备仓库外训练虚拟环境**

```bash
test -n "$LUNAR_TRAIN_VENV"
python3 -c 'import os, pathlib, subprocess; v=pathlib.Path(os.environ["LUNAR_TRAIN_VENV"]).resolve(); r=pathlib.Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()).resolve(); assert v != r and r not in v.parents'
python3 -m venv "$LUNAR_TRAIN_VENV"
source "$LUNAR_TRAIN_VENV/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install torch numpy PyYAML pytest onnx onnxruntime-gpu jsonschema
```

Expected: `python3 -c 'import torch, numpy, onnx, onnxruntime, jsonschema'` 成功。Task 3 将当前解析结果探测并锁定为精确 constraints；不使用系统 Python 代替该 venv。

- [ ] **Step 2: 写导入边界和七输入失败测试**

```python
EXPECTED_INPUTS = (
    "prior_channels", "coverage_summary", "local_crop",
    "frontier_features", "pose_features", "candidate_mask",
    "platform_context",
)

def test_observation_contract_has_exact_seven_inputs():
    assert ObservationContractV1.input_names == EXPECTED_INPUTS

def test_policy_rejects_non_one_hot_platform_context(policy, batch):
    batch.platform_context[0] = torch.tensor([1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="platform_context must be one-hot"):
        policy(batch)
```

- [ ] **Step 3: 运行测试并确认缺少新合同**

```bash
python3 -m pytest -q model_contract/tests/test_observation_contract.py training/lunar_policy_training/tests/test_policy_forward.py
```

Expected: FAIL，报告 `ObservationContractV1`、`platform_context` 或训练包尚未定义。

- [ ] **Step 4: 实现受控 PPO 导入**

`ppo_file_map.yaml` 只选择 observation、cross-attention、collector、rollout、trainer、checkpoint 数学核心和 evaluation metrics。`import_ppo_core.py` 必须核对 repository、commit、SHA-256，改写为包内相对 import；发现 `workflows`、Stage runner、ContentRef、authority/repair、artifact registry 或未列入依赖时退出非零并列出路径。

```bash
python3 tools/import_ppo_core.py --inventory migration/source_inventory.yaml --map migration/ppo_file_map.yaml
```

Expected: 只生成 map 明确允许的训练文件。

- [ ] **Step 5: 实现共享观测合同和 one-hot 校验**

```python
PLATFORM_CONTEXT_WIDTH = 3
PLATFORM_CONTEXTS = {
    "WHEELED": (1.0, 0.0, 0.0),
    "LEGGED": (0.0, 1.0, 0.0),
    "HOPPER": (0.0, 0.0, 1.0),
}

def validate_platform_context(array: np.ndarray) -> None:
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 3:
        raise ObservationContractError("platform_context must be float32 [B,3]")
    if not np.isfinite(array).all() or not np.isin(array, (0.0, 1.0)).all():
        raise ObservationContractError("platform_context must be finite one-hot")
    if not np.equal(array.sum(axis=1), 1.0).all():
        raise ObservationContractError("platform_context must be one-hot")
```

训练 `PolicyBatch` 使用相同名称、顺序、shape 和 dtype，不复制另一份常量。

- [ ] **Step 6: 把平台编码融合进共享网络**

在 `cross_attention.py` 增加 `nn.Linear(3, d_model)`，将结果与 `pose_features` 的全局 embedding 相加后进入现有 attention；不增加平台 head。输出仍精确为 `frontier_logits`、`theta_mu`、`theta_kappa`、`value`，无效 candidate mask 不能被 argmax 选中。

```python
platform_embedding = self.platform_encoder(batch.platform_context)
global_embedding = self.pose_encoder(batch.pose_features) + platform_embedding
```

- [ ] **Step 7: 构建合同包并运行 CPU、CUDA 和边界测试**

```bash
colcon build --base-paths model_contract --packages-select lunar_model_contract
source install/setup.bash
python3 -m pip install --no-deps -e training/lunar_policy_training
python3 -m pytest -q model_contract/tests/test_observation_contract.py training/lunar_policy_training/tests/test_import_boundary.py training/lunar_policy_training/tests/test_policy_forward.py
CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q training/lunar_policy_training/tests/test_policy_forward.py -m cuda
```

Expected: 三类 one-hot 均通过；非法 one-hot 拒绝；CPU/CUDA 输出 shape、dtype、finite 和 mask 行为一致；GPU 名称为 RTX 4080 SUPER。

- [ ] **Step 8: 提交七输入 PPO 骨干**

```bash
git add migration/ppo_file_map.yaml tools/import_ppo_core.py model_contract training/lunar_policy_training
git commit -m "refactor: migrate seven-input shared PPO core"
```

### Task 2: 建立训练专用 C++ v3 bridge 和三平台宏步环境

**Execution environment:** Ubuntu 22.04 amd64 + ROS 2 Humble；不需要 GPU。

**Estimated Codex time:** 5–8 小时。

**Files:**
- Create: `ros2_ws/src/lunar_planner_training_bridge/package.xml`
- Create: `ros2_ws/src/lunar_planner_training_bridge/CMakeLists.txt`
- Create: `ros2_ws/src/lunar_planner_training_bridge/include/lunar_planner_training_bridge/request.hpp`
- Create: `ros2_ws/src/lunar_planner_training_bridge/src/conversions.cpp`
- Create: `ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp`
- Create: `ros2_ws/src/lunar_planner_training_bridge/python/lunar_planner_training_bridge/__init__.py`
- Create: `ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py`
- Create: `training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py`
- Create: `training/lunar_policy_training/lunar_policy_training/environment/macro_step.py`
- Create: `training/lunar_policy_training/tests/test_v3_environment.py`
- Create: `training/lunar_policy_training/tests/test_hopper_macro_step.py`

**Interfaces:**
- Consumes: `lunar::planning::Planner` 和所有公开 `PlannerInput`/`PlannerOutput` 类型；Task 1 `ObservationContractV1`。
- Produces: `_lunar_planner_training_bridge.PlannerBridge.plan(request: TrainingPlanRequest) -> PlannerOutput`；`V3ExplorationEnvironment.step(action: PolicyAction) -> PlannerTransition`。

- [ ] **Step 1: 写三平台 bridge 失败测试**

```python
@pytest.mark.parametrize("platform_type", ["WHEELED", "LEGGED", "HOPPER"])
def test_bridge_uses_matching_v3_planner(bridge, easy_request, platform_type):
    request = easy_request(platform_type)
    output = bridge.plan(request)
    if output.reference is not None:
        assert output.reference.platform_type == platform_type
    assert output.diagnostics.planner_name == "cpp_v3"
```

- [ ] **Step 2: 写飞跃承诺宏步失败测试**

```python
def test_hopper_does_not_request_policy_while_committed(hopper_env, policy_spy):
    hopper_env.begin_committed_hop()
    result = hopper_env.advance_until_decision_boundary(policy_spy)
    assert policy_spy.call_count == 0
    assert result.execution_state == "LANDED_HOLD"
```

- [ ] **Step 3: 运行测试并确认 bridge 尚不存在**

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
python3 -m pytest -q training/lunar_policy_training/tests/test_v3_environment.py training/lunar_policy_training/tests/test_hopper_macro_step.py
```

Expected: FAIL，报告 bridge module 或三平台宏步类型未定义。

- [ ] **Step 4: 实现有界 bridge 请求类型**

`request.hpp` 定义不含 ROS 类型、仅缺省 stop token 的训练请求：

```cpp
struct TrainingPlanRequest final {
  std::string request_id;
  lunar::planning::TimePoint state_time;
  lunar::planning::PlatformState current_state;
  lunar::planning::GoalRegion goal;
  lunar::planning::WorldSnapshot world;
  lunar::planning::PlatformCapability capability;
  lunar::planning::PlannerConfig config;
  std::optional<lunar::planning::ExecutionContext> previous_execution;
};

class PlannerBridge final {
 public:
  [[nodiscard]] lunar::planning::PlannerOutput Plan(
      const TrainingPlanRequest& request) noexcept;
 private:
  lunar::planning::Planner planner_;
};
```

`package.xml` 声明 `ament_cmake`、`ament_cmake_python`、`lunar_planner_core`、`pybind11_vendor` 和 Python/NumPy 运行依赖；CMake 使用 `pybind11_add_module` 构建 `_lunar_planner_training_bridge`，链接已导出的 `lunar_planner_core` target，并把 Python 包安装到当前 colcon install prefix。

绑定公开 geometry、grid、goal、state、capability、execution 和 output 类型；Python enum 名称固定为 `NEW_REFERENCE_AVAILABLE`、`SAFE_FRONTIER_REFERENCE_AVAILABLE`、`NO_KNOWN_SAFE_ROUTE`、`GOAL_INFEASIBLE`、`INVALID_REQUEST`、`STALE_INPUT`、`NUMERICAL_FAILURE`、`RESOURCE_EXHAUSTED`、`ACTIVE_REFERENCE_INVALIDATED`、`CANCELED`，以及对应 execution directive 大写名称。NumPy layer 只接受 C-contiguous 且 dtype 精确匹配的数组。`plan()` 调用时释放 GIL，不复制或重写 planner 算法。

- [ ] **Step 5: 实现三平台规划执行宏步**

在 `macro_step.py` 精确定义：

```python
@dataclass(frozen=True)
class PolicyAction:
    frontier_index: int
    theta_rad: float

@dataclass(frozen=True)
class PlannerTransition:
    next_observation: PolicyBatch
    coverage_delta: float
    goal_progress: float
    normalized_plan_cost: float
    normalized_elapsed_time: float
    repeated_visit: bool
    planning_outcome: PlanningOutcome
    execution_directive: ExecutionDirective
    reason_code: str
    terminated: bool
```

轮式和足式使用返回轨迹推进到下一探索决策边界；飞跃式在 `kJumpCommitted`/`kInFlight` 时禁止新动作，只有落入 `kLandedHold` 后恢复决策。没有 reference 时保持状态并把 `PlanningOutcome` 交给奖励层，绝不伪造轨迹。

执行 reference 前必须验证 `MotionReference.platform_type` 与当前回合平台一致，并确认 directive 允许执行该 reference；不一致或拒绝态携带 reference 时抛 `EnvironmentInvariantError`、废弃未完成 rollout 并停止训练，不能把它编码成奖励。

```python
if output.directive == ExecutionDirective.CONTINUE_COMMITTED_HOP:
    return self._advance_committed_hop_without_policy(output)
if output.reference is None:
    return self._hold_transition(
        outcome=output.outcome,
        directive=output.directive,
        reason_code=output.reason_code,
        planner_elapsed=output.diagnostics.elapsed,
    )
return self._execute_reference_until_decision_boundary(output.reference)
```

`_hold_transition()` 复制当前观测，coverage/goal/plan-cost 增量置零，保留归一化 planner elapsed、outcome、directive 和 reason code；它不得把无 reference 伪装成成功轨迹。

- [ ] **Step 6: 构建 bridge 并运行三平台测试**

```bash
source /opt/ros/humble/setup.bash
colcon build --base-paths ros2_ws/src --packages-select lunar_planner_core lunar_planner_training_bridge --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
colcon test --base-paths ros2_ws/src --packages-select lunar_planner_training_bridge
colcon test-result --verbose
python3 -m pytest -q training/lunar_policy_training/tests/test_v3_environment.py training/lunar_policy_training/tests/test_hopper_macro_step.py
```

Expected: 三类平台调用 `cpp_v3`；无 reference 保持；已承诺飞跃期间 policy call count 为零。

- [ ] **Step 7: 提交 v3 训练闭环**

```bash
git add ros2_ws/src/lunar_planner_training_bridge training/lunar_policy_training
git commit -m "feat: add in-process v3 PPO training bridge"
```

### Task 3: 实现激进并行、30 分钟 checkpoint 和累计预算恢复

**Execution environment:** Ubuntu 22.04 amd64 + RTX 4080 SUPER。

**Estimated Codex time:** 5–8 小时。

**Files:**
- Create: `training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py`
- Create: `training/lunar_policy_training/lunar_policy_training/budget.py`
- Create: `training/lunar_policy_training/lunar_policy_training/config.py`
- Create: `training/lunar_policy_training/lunar_policy_training/checkpoint.py`
- Create: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Create: `training/configs/rtx4080_super_smoke.yaml`
- Create: `training/configs/rtx4080_super_v3_joint.yaml`
- Create: `training/constraints/ubuntu22.04-rtx4080_super.txt`
- Create: `training/tools/lock_training_stack.py`
- Create: `training/lunar_policy_training/tests/test_parallel_pool.py`
- Create: `training/lunar_policy_training/tests/test_checkpoint_resume.py`
- Create: `training/lunar_policy_training/tests/test_training_budget.py`
- Create: `training/lunar_policy_training/tests/test_cli.py`
- Create: `training/lunar_policy_training/tests/test_training_smoke.py`

**Interfaces:**
- Consumes: Task 1 shared policy；Task 2 `V3ExplorationEnvironment`。
- Produces: `ParallelEnvPool`；`TrainingBudget(total_gpu_seconds=86400)`；可注入 `reward_fn(PlannerTransition) -> float` 的 trainer；`train|resume` CLI 和内部 runtime calibration；checkpoint schema `lunar-ppo-checkpoint/v2`。Task 4 再向同一 CLI 注册 `calibrate|evaluate`。

- [ ] **Step 1: 写并行配比和 checkpoint 周期失败测试**

```python
def test_joint_pool_has_eight_workers_per_platform():
    allocation = joint_worker_allocation(total_workers=24)
    assert allocation == {"WHEELED": 8, "LEGGED": 8, "HOPPER": 8}

def test_checkpoint_interval_is_thirty_minutes(resolved_config):
    assert resolved_config.checkpoint_interval_seconds == 1800

def test_joint_candidate_interval_is_one_hour(resolved_config):
    assert resolved_config.candidate_checkpoint_interval_seconds == 3600
```

- [ ] **Step 2: 写恢复累计预算失败测试**

```python
def test_resume_preserves_consumed_gpu_budget(tmp_path):
    checkpoint = make_checkpoint(consumed_gpu_seconds=7200.0)
    save_checkpoint_atomic(tmp_path / "latest.pt", checkpoint)
    resumed = load_checkpoint(tmp_path / "latest.pt")
    assert resumed.consumed_gpu_seconds == 7200.0
    budget = TrainingBudget.from_checkpoint(resumed)
    assert budget.remaining_gpu_seconds == 86400.0 - 7200.0
```

- [ ] **Step 3: 运行测试并确认并行/恢复模块缺失**

```bash
python3 -m pytest -q training/lunar_policy_training/tests/test_parallel_pool.py training/lunar_policy_training/tests/test_checkpoint_resume.py training/lunar_policy_training/tests/test_training_budget.py training/lunar_policy_training/tests/test_cli.py
```

Expected: FAIL，报告 `ParallelEnvPool`、checkpoint v2 或累计预算未定义。

- [ ] **Step 4: 固定激进并行配置**

```yaml
parallel:
  worker_candidates: [18, 24]
  preferred_workers: 24
  joint_workers:
    WHEELED: 8
    LEGGED: 8
    HOPPER: 8
  nested_compute_threads: 1
  gpu_memory_fraction_max: 0.90
checkpoint_interval_seconds: 1800
candidate_checkpoint_interval_seconds: 3600
total_gpu_budget_seconds: 86400
formal_training_seeds: [4080]
proxy: true
```

`calibrate` 在相同小型 workload 上测 18/24 workers；24 workers 只有在无 OOM、无 planner 超时且总吞吐不低于 18 时入选。micro-batch 逐级增大到显存峰值不超过 90%，解析结果写入 run manifest 后冻结。

该探测必须明确比较 18 与 24 workers；不得因 24 是首选值而跳过 18-worker 对照。

- [ ] **Step 5: 实现同步 on-policy 共享内存池**

每个 worker 独占 environment/bridge；主进程预分配共享 observation/action buffers，使用页锁定 staging 和双缓冲。一次 rollout 记录固定 `policy_version`，混入其他版本立即拒绝。为每个 worker 设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。

- [ ] **Step 6: 实现 checkpoint v2 和信号暂停**

checkpoint 精确包含 schema、model、optimizer、scheduler、global step、curriculum phase、normalization、Python/NumPy/Torch CPU/CUDA RNG、冻结配置、source commit、累计 GPU 秒。写入同目录临时文件，flush/fsync 后 `os.replace()` 为 `latest.pt`。

`SIGINT`/`SIGTERM` 只设置 stop flag；trainer 完成当前 PPO update、丢弃未完成 rollout、立即保存后退出。恢复只允许 schema、七输入合同、config hash 和 source commit 完全匹配。

`latest.pt` 每 30 分钟更新；进入联合阶段后，另按每小时边界保存不可变候选 checkpoint，供三小时评估选择，不用候选写入替代 `latest.pt` 的恢复职责。

runtime calibration、三个短奖励 seed、正式训练和周期评估都通过同一个 `TrainingBudget` 累计 GPU 秒；Task 7 的 ONNX 导出不计入。任何 CLI 子命令不得创建新的预算对象来绕过已有 run manifest。

- [ ] **Step 7: 锁定训练栈并运行短 CUDA 中断恢复**

```bash
python3 training/tools/lock_training_stack.py --fingerprint "$LUNAR_UBUNTU_FINGERPRINT" --output training/constraints/ubuntu22.04-rtx4080_super.txt
python3 -m pytest -q training/lunar_policy_training/tests/test_parallel_pool.py training/lunar_policy_training/tests/test_checkpoint_resume.py training/lunar_policy_training/tests/test_training_budget.py training/lunar_policy_training/tests/test_cli.py
CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q training/lunar_policy_training/tests/test_training_smoke.py -m cuda
```

Expected: 中断前后累计预算、global step 和 platform allocation 正确；暂停墙钟不计入 GPU 秒；artifact root 外无写入。

- [ ] **Step 8: 提交可恢复并行训练器**

```bash
git add training/lunar_policy_training training/configs training/constraints training/tools/lock_training_stack.py
git commit -m "feat: add resumable multi-platform PPO trainer"
```

### Task 4: 实现 v3 奖励、课程和三平台独立 95% 评估

**Execution environment:** Ubuntu CPU 单测；RTX 4080 SUPER 短校准冒烟。

**Estimated Codex time:** 4–7 小时。

**Files:**
- Create: `training/lunar_policy_training/lunar_policy_training/reward.py`
- Create: `training/lunar_policy_training/lunar_policy_training/curriculum.py`
- Create: `training/lunar_policy_training/lunar_policy_training/evaluation/release_gate.py`
- Create: `training/lunar_policy_training/lunar_policy_training/evaluation/report.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Create: `training/configs/candidate_gate_v1.yaml`
- Create: `training/configs/release_gate_v1.yaml`
- Create: `training/lunar_policy_training/tests/test_reward.py`
- Create: `training/lunar_policy_training/tests/test_curriculum.py`
- Create: `training/lunar_policy_training/tests/test_release_gate.py`
- Create: `training/lunar_policy_training/tests/test_evaluation_determinism.py`

**Interfaces:**
- Consumes: v3 `PlanningOutcome`/`ExecutionDirective`、Task 3 trainer/checkpoint。
- Produces: `compute_reward(inputs: RewardInputs) -> float`；`CurriculumSchedule`；`lunar-policy-release-evaluation/v1`。

- [ ] **Step 1: 写结果分类和三平台聚合失败测试**

```python
def test_infrastructure_failure_is_not_a_reward_sample():
    with pytest.raises(InvalidTransition, match="numerical failure"):
        compute_reward(reward_inputs(outcome="NUMERICAL_FAILURE"))

def test_gate_fails_when_only_hopper_is_below_95_percent():
    report = report_for_platforms(wheeled=0.97, legged=0.95, hopper=0.94)
    result = evaluate_release_gate(report, release_rules())
    assert result.passed is False
    assert result.failed_rules == ("HOPPER.success_coverage_rate_min",)
```

- [ ] **Step 2: 运行测试并确认 reward/gate 尚不存在**

```bash
python3 -m pytest -q training/lunar_policy_training/tests/test_reward.py training/lunar_policy_training/tests/test_curriculum.py training/lunar_policy_training/tests/test_release_gate.py
```

Expected: FAIL，报告 reward 分类或按平台 gate 未定义。

- [ ] **Step 3: 实现共享无量纲奖励**

正向项只包含新增有效覆盖和目标/前沿推进；负向项包含归一化规划代价、模拟耗时、重复访问、`kGoalInfeasible`、`kNoKnownSafeRoute` 和 `kResourceExhausted`。`kInvalidRequest`、`kStaleInput`、`kNumericalFailure`、非预期 `kCanceled`、C++ 异常和非有限 tensor 抛出 `InvalidTransition`，不得写入 rollout。

奖励配置对三平台共用同一权重；校准候选按三平台最低得分选择，正式种子开始后 config hash 冻结。

- [ ] **Step 4: 实现 2h/6h/16h 课程预算**

```python
schedule = CurriculumSchedule(
    calibration_limit_s=2 * 60 * 60,
    warmup_limit_s=6 * 60 * 60,
    joint_minimum_s=16 * 60 * 60,
    platform_warmup_limit_s=2 * 60 * 60,
    evaluation_interval_s=3 * 60 * 60,
)
assert schedule.total_gpu_limit_s == 24 * 60 * 60
```

预热顺序固定 WHEELED、LEGGED、HOPPER；联合训练固定 8+8+8。前一阶段节余只转入联合训练，不能减少联合阶段 16 小时最低预算。

- [ ] **Step 5: 固定候选与正式发布 gate**

```yaml
schema_version: lunar-policy-release-gate/v1
per_platform_rules:
  success_coverage_rate_min: 0.95
  safety_violation_count_max: 0
  invalid_action_count_max: 0
  output_finite_rate_min: 1.0
  platform_reference_mismatch_count_max: 0
  hopper_commitment_violation_count_max: 0
required_methods:
  - ppo_policy
  - nearest_frontier
  - gain_over_cost_frontier
```

候选 gate 使用上述六项。正式发布配置在相同 `per_platform_rules` 下再固定：

```yaml
  selected_action_observed_safe_rate_min: 1.0
  deterministic_repeat_match_rate_min: 1.0
```

两个 baseline 使用同一 v3 和 scenario schedule，只记录诊断，不要求 PPO 本轮胜出。CLI 的 `evaluate` 同时写按平台报告和单一 passed/failed；`calibrate` 串联 Task 3 runtime calibration 与三个短奖励 seed。

- [ ] **Step 6: 实现确定性评估和最佳 checkpoint 选择**

报告必须按平台分别保存 scenario seeds、coverage、安全、无效动作、规划结果率、finite 和承诺违规。最佳 checkpoint 按“最低平台 coverage 最大、规划失败率最小、完成时间最短”排序；不使用三平台平均值覆盖单平台失败。

- [ ] **Step 7: 运行单测和 CUDA 评估冒烟**

```bash
python3 -m pytest -q training/lunar_policy_training/tests/test_reward.py training/lunar_policy_training/tests/test_curriculum.py training/lunar_policy_training/tests/test_release_gate.py training/lunar_policy_training/tests/test_evaluation_determinism.py
CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q training/lunar_policy_training/tests/test_evaluation_determinism.py -m cuda
```

Expected: 报告按平台独立且相同 seed 重复 hash 一致；不启动正式训练或正式发布评估。

- [ ] **Step 8: 提交奖励、课程和评估**

```bash
git add training/lunar_policy_training training/configs/candidate_gate_v1.yaml training/configs/release_gate_v1.yaml
git commit -m "feat: add v3 PPO curriculum and platform gates"
```

### Task 5: 执行可暂停的 24 小时正式训练并冻结候选

**Execution environment:** Ubuntu 22.04 amd64、RTX 4080 SUPER、ROS 2 Humble。

**Estimated Codex time:** 1–2 agent-hours 负责预检、监控和报告；GPU 工作累计最多 24 小时。

**Files:**
- Create: `docs/migration/volume-3-training-candidate.md`
- External only: `$LUNAR_TRAIN_ARTIFACT_ROOT/checkpoints/`
- External only: `$LUNAR_TRAIN_ARTIFACT_ROOT/metrics/`
- External only: `$LUNAR_TRAIN_ARTIFACT_ROOT/evaluation/`
- External only: `$LUNAR_TRAIN_ARTIFACT_ROOT/run-manifest.json`

**Interfaces:**
- Consumes: Tasks 1–4；单一正式 seed `4080`；仓库外绝对 artifact root。
- Produces: `checkpoints/best.pt`、candidate/release reports 和不含主机用户名/绝对 artifact 路径的训练候选记录。

- [ ] **Step 1: 运行训练前门槛**

```bash
source /opt/ros/humble/setup.bash
source "$LUNAR_TRAIN_VENV/bin/activate"
test "$ROS_DISTRO" = humble
test "$(uname -m)" = x86_64
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
test -n "$LUNAR_TRAIN_ARTIFACT_ROOT"
python3 -c 'import os, pathlib, subprocess; p=pathlib.Path(os.environ["LUNAR_TRAIN_ARTIFACT_ROOT"]).resolve(); r=pathlib.Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()).resolve(); assert p.is_absolute() and p != r and r not in p.parents'
python3 -m pytest -q model_contract/tests training/lunar_policy_training/tests
```

Expected: GPU 是 RTX 4080 SUPER；测试通过；artifact root 是仓库外绝对路径。

- [ ] **Step 2: 解析并冻结并行与 micro-batch**

```bash
python3 -m lunar_policy_training.cli calibrate \
  --config training/configs/rtx4080_super_v3_joint.yaml \
  --artifact-root "$LUNAR_TRAIN_ARTIFACT_ROOT"
```

Expected: run manifest 记录选中的 18 或 24 workers、micro-batch、三个短校准 seed、奖励 hash 和 `total_gpu_budget_seconds=86400`。

- [ ] **Step 3: 启动或恢复唯一正式训练**

首次运行：

```bash
python3 -m lunar_policy_training.cli train \
  --config training/configs/rtx4080_super_v3_joint.yaml \
  --artifact-root "$LUNAR_TRAIN_ARTIFACT_ROOT"
```

中断后只使用：

```bash
python3 -m lunar_policy_training.cli resume \
  --checkpoint "$LUNAR_TRAIN_ARTIFACT_ROOT/checkpoints/latest.pt" \
  --artifact-root "$LUNAR_TRAIN_ARTIFACT_ROOT"
```

Expected: 每 30 分钟更新 `latest.pt`，联合阶段每小时保存候选 checkpoint；正常中断立即保存；恢复后累计 GPU 秒继续增长而非清零。

- [ ] **Step 4: 在每个三小时评估点检查按平台指标**

```bash
python3 -m lunar_policy_training.cli evaluate \
  --checkpoint "$LUNAR_TRAIN_ARTIFACT_ROOT/checkpoints/best.pt" \
  --gate training/configs/candidate_gate_v1.yaml \
  --artifact-root "$LUNAR_TRAIN_ARTIFACT_ROOT"
```

任一 checkpoint 的三平台一次完整评估全部通过即可提前停止。没有通过时继续到累计 86400 GPU 秒；暂停等待不计时。

- [ ] **Step 5: 运行正式发布评估**

```bash
python3 -m lunar_policy_training.cli evaluate \
  --checkpoint "$LUNAR_TRAIN_ARTIFACT_ROOT/checkpoints/best.pt" \
  --gate training/configs/release_gate_v1.yaml \
  --artifact-root "$LUNAR_TRAIN_ARTIFACT_ROOT"
```

Expected: 轮式、足式、飞跃式分别达到 0.95；安全/非法/承诺违规为零；finite、安全选择和确定性为 1.0。

- [ ] **Step 6: 记录成功或明确停止**

`volume-3-training-candidate.md` 记录 source commit、正式 seed、冻结 config hash、累计 GPU 秒、worker 数、checkpoint SHA-256 和三平台各自指标，不记录 checkpoint 本体、用户名或绝对 artifact 路径。

如果候选或正式发布 gate 未通过，写入 `status: not-converged` 和失败规则，提交报告后停止卷三，不执行 Task 6–11。通过时写入 `status: release-gate-passed`。

- [ ] **Step 7: 只提交训练报告**

```bash
git add docs/migration/volume-3-training-candidate.md
git commit -m "docs: record v3 PPO training candidate"
```

### Task 6: 完成共享模型 manifest 与四文件包合同

**Execution environment:** Ubuntu amd64；纯 Python 合同测试可在 Windows 运行但不是发布依据。

**Estimated Codex time:** 2–3 小时。

**Files:**
- Create: `model_contract/schema/manifest.schema.json`
- Create: `model_contract/lunar_model_contract/manifest.py`
- Create: `model_contract/lunar_model_contract/hashing.py`
- Create: `model_contract/lunar_model_contract/package.py`
- Create: `model_contract/tests/test_manifest.py`
- Create: `model_contract/tests/test_package_boundary.py`
- Modify: `model_contract/lunar_model_contract/observation.py`
- Create: `scripts/bootstrap_ubuntu.sh`
- Modify: `scripts/build_runtime.sh`

**Interfaces:**
- Consumes: Task 1 七输入 `ObservationContractV1`；Task 5 passed release report。
- Produces: `ModelManifest`；`validate_model_package(path: Path) -> ModelManifest`；卷四 `--candidate` 精确四文件目录合同。

- [ ] **Step 1: 写七输入 manifest 和四文件失败测试**

```python
def test_manifest_requires_platform_context(valid_manifest):
    valid_manifest["inputs"] = [
        item for item in valid_manifest["inputs"]
        if item["name"] != "platform_context"
    ]
    with pytest.raises(ModelManifestError, match="platform_context"):
        ModelManifest.from_dict(valid_manifest)

def test_model_package_rejects_checkpoint(tmp_path):
    write_valid_model_package(tmp_path)
    (tmp_path / "checkpoint.pt").write_bytes(b"forbidden")
    with pytest.raises(ModelPackageError, match="unexpected file: checkpoint.pt"):
        validate_model_package(tmp_path)
```

- [ ] **Step 2: 运行测试并确认 manifest/package 尚不完整**

```bash
python3 -m pytest -q model_contract/tests/test_manifest.py model_contract/tests/test_package_boundary.py
```

Expected: FAIL，报告 manifest schema 或 package validator 未定义。

- [ ] **Step 3: 固定 manifest v1**

required 字段为 `schema_version`、`model_id`、`version`、`source_commit`、`onnx_opset`、`observation_contract`、`action_contract`、`inputs`、`outputs`、`normalization`、`files`、`tolerances`、`release_evaluation`。输入必须与 Task 1 七项名称/顺序完全相同，`platform_context` shape 为固定 `[B,3]`、dtype 为 `float32`。

- [ ] **Step 4: 实现精确包校验**

只允许四个文件；校验 SHA-256、JSON Schema、NPZ key/shape/dtype、finite、normalization 长度、40 位 source commit、release evaluation passed 和 report hash。NPZ 禁止 object dtype 和 pickle。

- [ ] **Step 5: 构建并运行合同测试**

```bash
colcon build --base-paths model_contract --packages-select lunar_model_contract
source install/setup.bash
python3 -m pytest -q model_contract/tests
```

Expected: 七输入合同和精确四文件边界通过；缺失或多余任一文件失败。

- [ ] **Step 6: 提交模型合同**

```bash
git add model_contract scripts/bootstrap_ubuntu.sh scripts/build_runtime.sh
git commit -m "feat: define seven-input policy package contract"
```

### Task 7: 导出七输入 ONNX、黄金样例和四文件候选

**Execution environment:** Ubuntu amd64 + RTX 4080 SUPER。

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
- Consumes: Task 5 `best.pt`/passed release report；Task 6 model contract。
- Produces: `<artifact-root>/published/<model-id>/<version>/` 四文件模型包和 PyTorch/ONNX FP32 等价报告。

- [ ] **Step 1: 写七输入 wrapper 失败测试**

```python
def test_inference_module_requires_platform_context(policy, batch):
    outputs = InferenceModule(policy)(
        batch.prior_channels,
        batch.coverage_summary,
        batch.local_crop,
        batch.frontier_features,
        batch.pose_features,
        batch.candidate_mask,
        batch.platform_context,
    )
    assert len(outputs) == 4
    assert outputs[0].shape == batch.candidate_mask.shape
    assert outputs[3].shape == (batch.candidate_mask.shape[0],)
```

- [ ] **Step 2: 运行测试并确认 wrapper 未定义**

```bash
python3 -m pytest -q training/model_export/tests/test_inference_module.py
```

Expected: FAIL，报告 `InferenceModule` 或第七输入未实现。

- [ ] **Step 3: 实现纯 tensor inference wrapper 和 ONNX 导出**

wrapper 只构造 `PolicyBatch` 并返回四项输出，不导出 sampling、VonMises、loss、optimizer 或中间 token。导出使用 opset 17、FP32、固定 H/W/M/P/3、显式七个 input names、四个 output names，不设置动态 H/W/M axes。

- [ ] **Step 4: 生成覆盖三平台的黄金样例**

固定 release scenario 中至少包含每个平台一个正常样例，并增加稀疏候选、边界 mask、高不确定性样例。每条 `platform_context` 必须与对应 v3 平台一致。NPZ 无 pickle；outputs 来源必须是同一 `best.pt` 的 PyTorch inference wrapper。

- [ ] **Step 5: 实现 FP32 ONNX Runtime 等价**

容差固定 `atol=1e-4`、`rtol=1e-4`；所有输出 finite，shape/dtype 完全相同，masked argmax index 完全一致，选中 `theta_mu` 误差不超过 `1e-4 rad`。

- [ ] **Step 6: 实现原子四文件发布**

发布器在同一 artifact root 临时目录生成、验证和 fsync 四文件，再原子 rename 到版本目录；已存在且 hash 不同的 model ID/version 拒绝覆盖。训练 checkpoint 和报告本体不复制进设备包，manifest 只记录 release report hash。

- [ ] **Step 7: 运行导出测试并发布正式候选**

```bash
python3 -m pytest -q training/model_export/tests
python3 -m training.model_export.publish \
  --checkpoint "$LUNAR_TRAIN_ARTIFACT_ROOT/checkpoints/best.pt" \
  --evaluation "$LUNAR_TRAIN_ARTIFACT_ROOT/evaluation/report.json" \
  --output-root "$LUNAR_TRAIN_ARTIFACT_ROOT/published"
python3 -m training.model_export.verify_package "$LUNAR_TRAIN_ARTIFACT_ROOT/published/$LUNAR_MODEL_ID/$LUNAR_MODEL_VERSION"
```

Expected: 目录精确含四文件；三平台黄金样例和 ONNX parity 全部通过。

- [ ] **Step 8: 提交导出链**

```bash
git add training/model_export
git commit -m "feat: publish seven-input PPO ONNX package"
```

### Task 8: 实现严格 TensorRT runtime、engine cache 与 Python binding

**Execution environment:** Ubuntu CPU 构建接口；AGX 实现在 Task 10 验证。

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

**Interfaces:**
- Consumes: Task 7 四文件包；卷一设备指纹。
- Produces: `_lunar_policy_runtime.TensorRtRunner.infer(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]`，精确接受七项输入。

- [ ] **Step 1: 写 cache key 和七输入 manifest 失败测试**

```cpp
TEST(Manifest, RequiresPlatformContextBinding) {
  auto manifest = ValidManifest();
  manifest.inputs.erase("platform_context");
  EXPECT_THROW(ValidateManifest(manifest), ManifestError);
}

TEST(EngineCache, ChangesForAnyRuntimeFingerprintField) {
  auto base = MakeFingerprint();
  const auto key = MakeEngineCacheKey(kOnnxSha, base, Precision::kFp32);
  base.tensorrt_version = "different";
  EXPECT_NE(key, MakeEngineCacheKey(kOnnxSha, base, Precision::kFp32));
}
```

- [ ] **Step 2: 实现 manifest 和 engine cache**

cache key 包含 ONNX SHA-256、compute capability、aarch64、L4T、CUDA、TensorRT、precision、workspace bytes、builder flags 和 parser version。路径固定：

```text
/var/cache/lunar_navigation/tensorrt/<onnx-sha256>/<fingerprint-sha256>.engine
```

engine/metadata 临时写后原子 rename；同 key 使用 `flock`；字段或 hash 不符时忽略并重建。

- [ ] **Step 3: 实现严格七输入 runner**

```cpp
enum class TensorDType : std::uint8_t {
  kFloat32,
  kBool,
};

struct TensorView {
  const void* data{};
  std::vector<std::int64_t> shape;
  TensorDType dtype{TensorDType::kFloat32};
};

struct OwnedTensor {
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

runner 按 manifest 精确绑定七个名称、shape、dtype；拒绝缺失、多余、非 contiguous、自动 dtype coercion 和 broadcast。复用 CUDA stream/device buffers，检查所有 CUDA/TensorRT 返回值和输出 finite。TensorRT 8/10 差异只存在 `tensorrt_compat.hpp`。

- [ ] **Step 4: 实现 builder CLI 和 pybind11**

builder 默认 FP32；未通过单独验收时不启用 FP16。构建失败退出非零且不产生完成 cache。Python `infer()` 释放 GIL、返回自有内存 NumPy；CPU build 可注入 `FakePolicyRunner`，生产 TensorRT factory 不可用时抛 `RuntimeUnavailableError`，不回退。

- [ ] **Step 5: 运行 Ubuntu CPU 接口测试**

```bash
colcon build --base-paths ros2_ws/src model_contract --packages-select lunar_model_contract lunar_policy_runtime --cmake-args -DLUNAR_ENABLE_TENSORRT=OFF
colcon test --base-paths ros2_ws/src model_contract --packages-select lunar_policy_runtime
colcon test-result --verbose
```

Expected: manifest/cache/fake runner 通过；缺少 `platform_context` 失败；真实 TensorRT factory 明确 unavailable，不回退。

- [ ] **Step 6: 提交 runtime**

```bash
git add ros2_ws/src/lunar_policy_runtime
git commit -m "feat: add strict seven-input TensorRT runtime"
```

### Task 9: 实现探索 Lifecycle 节点和平台 one-hot 输入

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
- Consumes: 外部 map/task/localization/TF、`platform-control-capability-source/v1` 平台类型、Task 8 runtime、卷二 `PlanMotion` server。
- Produces: 七个 C-contiguous inference arrays；与 mission revision 绑定的 `PlanMotion.Goal`；不发布控制命令。

- [ ] **Step 1: 写 one-hot 和观测一致性失败测试**

```python
@pytest.mark.parametrize("platform_type,expected", [
    ("WHEELED", np.array([[1, 0, 0]], dtype=np.float32)),
    ("LEGGED", np.array([[0, 1, 0]], dtype=np.float32)),
    ("HOPPER", np.array([[0, 0, 1]], dtype=np.float32)),
])
def test_observation_builder_encodes_platform_type(builder, snapshot, platform_type, expected):
    batch = builder.build(snapshot, platform_type)
    np.testing.assert_array_equal(batch["platform_context"], expected)
```

- [ ] **Step 2: 实现输入适配和共享 ObservationContract**

`input_adapter.py` 直接订阅外部 Topic，不重发布大地图；验证字段、frame、时间和范围。`observation_builder.py` 从 `lunar_model_contract.observation` 导入七项名称、shape、channel order、normalization 和 padding，输出精确七个 C-contiguous arrays。

平台类型来自外部能力配置，不从地图或模型推断；未知类型、类型变化或与 planner result 不一致时拒绝推理并报告诊断。

- [ ] **Step 3: 实现确定性目标选择和 Action 规则**

运行时使用 `argmax(masked frontier_logits)` 与该 candidate 的 `theta_mu`。NaN/Inf、越界、全 false mask 或合同不符时丢弃本轮目标。一次只保留一个规划 goal；只有 mission revision 增加或原 goal 明确失效时允许 `replace_active_request=true`。

- [ ] **Step 4: 实现 Lifecycle 门槛**

configure 验证外部观测能力、平台类型、模型四文件、manifest、TensorRT runner、黄金样例和 observation channels；任一失败返回 FAILURE。activate 后才订阅/推理；PAUSED 停止推理并取消未提交 Action；CANCELED 清除任务上下文。

- [ ] **Step 5: 运行 CPU fake 与 ROS 测试**

```bash
source /opt/ros/humble/setup.bash
colcon build --base-paths ros2_ws/src model_contract --packages-up-to lunar_exploration --cmake-args -DLUNAR_ENABLE_TENSORRT=OFF
colcon test --base-paths ros2_ws/src model_contract --packages-select lunar_exploration
colcon test-result --verbose
```

Expected: 单进程观测→七输入 fake inference→目标→Action client 通过；三平台 one-hot 正确；不存在高分辨率 tensor Topic。

- [ ] **Step 6: 提交探索节点**

```bash
git add ros2_ws/src/lunar_exploration
git commit -m "feat: add platform-conditioned PPO exploration node"
```

### Task 10: 在 AGX 验证 FP32 TensorRT、缓存和失败不降级

**Execution environment:** Jetson AGX Orin 64GB、R36.0.0。

**Estimated Codex time:** 2–4 小时，首次 engine 构建包含在内。

**Files:**
- Create: `tests/device/policy_runtime/test_tensorrt_parity.py`
- Create: `tests/device/policy_runtime/test_engine_cache_reuse.py`
- Create: `tests/device/policy_runtime/test_runtime_failure_modes.py`
- Create: `tests/device/policy_runtime/benchmark_inference.py`
- Create: `docs/migration/volume-3-agx-report-template.md`

**Interfaces:**
- Consumes: Tasks 7–9 同一四文件模型包；显式 `LUNAR_POLICY_MODEL_DIR` 版本目录。
- Produces: 三平台 FP32 parity、cache reuse、P95 和故障报告。

- [ ] **Step 1: 构建 runtime 和本机 FP32 engine**

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
test -n "$LUNAR_POLICY_MODEL_DIR"
python3 -c 'import os, pathlib; p=pathlib.Path(os.environ["LUNAR_POLICY_MODEL_DIR"]); assert p.is_absolute(); root=pathlib.Path("/var/lib/lunar_navigation/models"); rel=p.relative_to(root); assert len(rel.parts) == 2 and rel.parts[-1] != "current"'
colcon build --base-paths ros2_ws/src model_contract --packages-select lunar_model_contract lunar_policy_runtime --cmake-args -DLUNAR_ENABLE_TENSORRT=ON -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 run lunar_policy_runtime build_engine --model-dir "$LUNAR_POLICY_MODEL_DIR" --cache-root /var/cache/lunar_navigation/tensorrt --precision fp32
```

Expected: engine 只在 AGX cache 生成；不读取 `current`；训练 checkpoint 不在设备目录。

- [ ] **Step 2: 验证三平台黄金等价和 cache 复用**

```bash
ros2 run lunar_policy_runtime verify_engine --model-dir "$LUNAR_POLICY_MODEL_DIR" --cache-root /var/cache/lunar_navigation/tensorrt
python3 -m pytest -q tests/device/policy_runtime/test_tensorrt_parity.py tests/device/policy_runtime/test_engine_cache_reuse.py
```

Expected: 七输入三平台黄金输出在 manifest 容差内，candidate index 一致；第二次复用 cache 且 engine hash/mtime 不变。

- [ ] **Step 3: 验证失败不降级**

逐个测试损坏 ONNX、错误 manifest hash、缺 `platform_context`、错误 one-hot shape、NaN output、损坏 engine 和 fingerprint mismatch。每项必须导致 runtime 创建失败或节点保持 Inactive，日志不得出现 PyTorch/ORT/CPU fallback。

```bash
python3 -m pytest -q tests/device/policy_runtime/test_runtime_failure_modes.py
```

- [ ] **Step 4: 测量推理 P95**

```bash
python3 tests/device/policy_runtime/benchmark_inference.py \
  --model-dir "$LUNAR_POLICY_MODEL_DIR" \
  --warmup 100 \
  --iterations 1000 \
  --output /var/log/lunar_navigation/policy-benchmark.json
```

Expected: 报告含设备指纹、模型 hash、median/p95/max；`p95 <= 0.8 * decision_period`。本卷不自动启用 FP16。

- [ ] **Step 5: 提交设备测试和报告模板**

```bash
git add tests/device/policy_runtime docs/migration/volume-3-agx-report-template.md
git commit -m "test: qualify AGX seven-input policy runtime"
```

### Task 11: 卷三全量验收、状态记录与回退点

**Execution environment:** Ubuntu RTX 4080 SUPER；AGX completion 部分在真实设备执行。

**Estimated Codex time:** 1–2 小时。

**Files:**
- Create: `docs/migration/volume-3-ubuntu-readiness.md`
- Create: `docs/migration/volume-3-completion.md`
- Verify: `training/`
- Verify: `model_contract/`
- Verify: `ros2_ws/src/lunar_planner_training_bridge/`
- Verify: `ros2_ws/src/lunar_policy_runtime/`
- Verify: `ros2_ws/src/lunar_exploration/`

**Interfaces:**
- Consumes: Tasks 1–9 生成 `model-package-ready`；Task 10 生成 `device-verified`。
- Produces: Ubuntu readiness 报告；真实 AGX 完成后的 `policy-pipeline-v1` tag。

- [ ] **Step 1: 运行 Ubuntu 全量门槛**

```bash
source /opt/ros/humble/setup.bash
python3 -m pytest -q model_contract/tests training/lunar_policy_training/tests training/model_export/tests
colcon build --base-paths ros2_ws/src model_contract --packages-up-to lunar_exploration --cmake-args -DLUNAR_ENABLE_TENSORRT=OFF
colcon test --base-paths ros2_ws/src model_contract --packages-select lunar_planner_training_bridge lunar_policy_runtime lunar_exploration
colcon test-result --verbose
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
```

Expected: 全部通过；训练候选和正式 release report 的三平台 coverage 均不低于 0.95。

- [ ] **Step 2: 审计发布目录和运行时依赖**

```bash
find "$LUNAR_TRAIN_ARTIFACT_ROOT/published/$LUNAR_MODEL_ID/$LUNAR_MODEL_VERSION" -maxdepth 1 -type f -printf '%f\n' | sort
rg -n 'import torch|onnxruntime|fallback' ros2_ws/src/lunar_policy_runtime ros2_ws/src/lunar_exploration
```

Expected: 第一条只列四个允许文件；运行包不 import torch/onnxruntime；`fallback` 只在禁止语义或测试断言中出现。

- [ ] **Step 3: 记录 Ubuntu readiness**

`volume-3-ubuntu-readiness.md` 记录 model ID/version、四文件 hash、source commit、训练候选报告 hash、三平台指标、Ubuntu 指纹和测试结果，不记录用户名或持久化主机绝对路径。AGX 未完成时逐字记录：

```text
model-package-ready
AGX native verification: pending
```

```bash
git add docs/migration/volume-3-ubuntu-readiness.md
git commit -m "docs: record v3 policy Ubuntu readiness"
```

- [ ] **Step 4: 运行 AGX completion gate**

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
test -n "$LUNAR_POLICY_MODEL_DIR"
ros2 run lunar_policy_runtime verify_engine --model-dir "$LUNAR_POLICY_MODEL_DIR" --cache-root /var/cache/lunar_navigation/tensorrt
python3 -m pytest -q tests/device/policy_runtime
```

Expected: 三平台 FP32 等价、cache、失败和性能门槛全部通过，状态为 `device-verified`。

- [ ] **Step 5: 记录 completion 并创建 tag**

只有 Step 4 在真实 AGX 通过后才执行。`volume-3-completion.md` 记录 source commit、checkpoint hash（只记录）、ONNX/manifest/evaluation hash、Ubuntu/AGX 指纹 hash、TensorRT precision 和测试结果。

```bash
git add docs/migration/volume-3-completion.md
git commit -m "docs: record v3 policy pipeline completion"
git tag -a policy-pipeline-v1 -m "PPO v3 ONNX TensorRT pipeline v1"
```

**Rollback:** 模型训练或发布失败时保留上一个完整四文件包和独立 engine cache，不覆盖既有 model ID/version。训练 checkpoint 只留在 Ubuntu artifact root；安装默认不启动，AGX `current` 不切换，旧部署保持运行。
