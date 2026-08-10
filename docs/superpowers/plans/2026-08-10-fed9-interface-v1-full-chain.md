# fed9 interface-v1 完整链路实施计划

> **执行要求：** 使用 `superpowers:executing-plans` 逐任务执行；新增行为严格测试先行。构建、模型和证据全部写入 `/home/kai/CodexDownloads/lunar_navigation/interface_v1`，不得写入仓库。

**目标：** 从 fed9 源码和 step 251 旧检查点生成一套真实标记为 `integration_only` 的四文件 ONNX 模型包，并在 ROS 2 Humble 中贯通外部输入、旧 Observation V3/候选算法、Action V2、PlanMotion、MotionReference 和执行反馈，完成三平台连续决策接口回归。

**架构：** 保留 fed9 的策略、候选、覆盖和规划语义。移植已经独立验证的固定路径平台 YAML 与外部接口 adapter；新增集成专用 manifest、确定性 ONNX Runtime、运行时中立的决策协调器及薄 ROS 生命周期节点。训练 checkpoint 只在离线导出工具中出现，运行时只加载严格四文件包。

**技术栈：** Ubuntu 22.04 amd64、ROS 2 Humble、Python 3.10、PyTorch、ONNX opset 17、ONNX Runtime、NumPy、C++20、rclpy/rclcpp、pytest、GoogleTest、colcon Release。

## 全局约束

- 源码基线固定为 `fed9ea92c8ae55eb8423a008598935f515e801a3`，不得合入 v7 语义。
- checkpoint 固定为 step 251，文件 SHA-256 `5c19113953bc91854abac88835a833e7bebe25ed6865cf0a09f36e828b359727`。
- interface-v1 只可标记 `integration_only`；不得通过构造字段伪装正式发布。
- 每项代码行为先增加会因目标缺陷而失败的测试，观察 RED 后才写最小实现。
- ROS 命令先 source `/opt/ros/humble/setup.bash`，完成后确认 `ROS_DISTRO=humble`；不要在 source 前启用 `set -u`。
- 构建使用 `--merge-install`、`-DCMAKE_BUILD_TYPE=Release`，最多两个并行作业。
- 不修改或停止 formal-training/v7 工作树与进程。

---

### Task 1：移植固定路径平台能力与外部接口基础设施

**来源提交：**

- `6a806bf feat(config): add replaceable platform profiles`
- `304a302 feat(planner): load one complete platform profile`
- `c9e5b6b feat(ros): add replaceable external interface adapter`

**主要文件：**

- Add: `ros2_ws/src/lunar_navigation_config/config/platform_profiles/{wheeled,legged,hopper}.yaml`
- Add: `ros2_ws/src/lunar_navigation_config/config/platform_profile.schema.json`
- Add: `ros2_ws/src/lunar_navigation_config/config/interface_profiles/default.yaml`
- Add: `ros2_ws/src/lunar_navigation_config/config/interface_profile.schema.json`
- Add: `ros2_ws/src/lunar_external_adapter/**`
- Modify: `ros2_ws/src/lunar_planner_ros/src/capability_loader.cpp`
- Test: `tests/foundation/test_platform_profiles.py`
- Test: `ros2_ws/src/lunar_external_adapter/test/{test_profile,test_conversions}.py`
- Test: `ros2_ws/src/lunar_planner_ros/test/capability_loader_test.cpp`

- [ ] 核对三个来源提交相对 fed9 的 diff，仅允许配置加载、adapter 和相应文档/测试。
- [ ] 按原提交顺序移植，解决冲突时保留 fed9 规划行为；记录每个移植提交来源。
- [ ] 运行平台 profile、接口 profile、adapter 测试和 Release capability loader 测试。
- [ ] 用临时目录模拟 `/etc/lunar_navigation/platform_profile.yaml` 的逐文件替换，证明一次只读取一个完整平台能力且重启式重载生效。
- [ ] 提交：`feat(interface-v1): add replaceable deployment profiles`。

---

### Task 2：定义诚实的 integration-only 四文件合同

**文件：**

- Add: `model_contract/lunar_model_contract/hashing.py`
- Add: `model_contract/lunar_model_contract/interface_manifest.py`
- Add: `model_contract/lunar_model_contract/package.py`
- Modify: `model_contract/lunar_model_contract/__init__.py`
- Add: `model_contract/schema/interface-manifest.schema.json`
- Test: `model_contract/tests/test_interface_manifest.py`
- Test: `model_contract/tests/test_package_boundary.py`

- [ ] RED：合法 `integration_only` manifest 可加载；缺少任一旧冻结身份或出现 `release_evaluation` 被拒绝。
- [ ] RED：目录只能有四个普通非空文件；附带 `.pt`、额外文件、符号链接或任一 hash 变化均拒绝。
- [ ] RED：golden NPZ 必须精确匹配 Observation V3 七输入和 Action V2 四输出，且三行 platform context 分别为 WHEELED/LEGGED/HOPPER。
- [ ] 实现 `InterfaceModelManifest` 与 `validate_interface_model_package()`；独立于正式发布 manifest，禁止 schema 混用。
- [ ] 对 schema 运行 JSON Schema 正反例，并提交：`feat(interface-v1): define integration model package`。

manifest 的核心分流必须是：

```json
{
  "schema_version": "lunar-policy-interface-manifest/v1",
  "qualification": "integration_only",
  "source_commit": "fed9ea92c8ae55eb8423a008598935f515e801a3",
  "checkpoint_step": 251,
  "observation_contract": "lunar-observation-contract/v3",
  "action_contract": "lunar-action-contract/v2"
}
```

---

### Task 3：从 step 251 导出确定性 ONNX 四文件包

**文件：**

- Add: `training/lunar_policy_training/lunar_policy_training/export/interface_v1.py`
- Add: `training/lunar_policy_training/tests/test_interface_v1_export.py`
- Add: `scripts/export_interface_v1_model.sh`

- [ ] RED：错误 checkpoint 文件/hash/schema/step/source commit/model state key 会在导出前失败。
- [ ] RED：导出 wrapper 的输入输出顺序必须分别等于 Observation V3 和 Action V2，dynamic axis 只允许 batch。
- [ ] RED：三平台手工构造的合法 golden batch 在 PyTorch 与 ONNX Runtime 间逐输出满足冻结容差。
- [ ] 实现只读 checkpoint loader，加载完整 `model_state`，调用 `eval()`，禁止读取 optimizer 或恢复训练状态。
- [ ] 使用 opset 17 导出；生成三平台 golden NPZ、计算文件 hash，最后原子写 `manifest.json`。
- [ ] 脚本要求显式 checkpoint 与外部输出目录，不得默认写仓库。
- [ ] 在 `/home/kai/CodexDownloads/lunar_navigation/interface_v1/model-step251` 生成实物并运行严格包校验。
- [ ] 提交：`feat(interface-v1): export frozen step251 policy`。

---

### Task 4：实现运行时 ONNX 推理和确定性动作解释

**文件：**

- Add: `ros2_ws/src/lunar_exploration_policy/lunar_exploration_policy/inference.py`
- Add: `ros2_ws/src/lunar_exploration_policy/lunar_exploration_policy/action_selection.py`
- Add: `ros2_ws/src/lunar_exploration_policy/test/test_inference.py`
- Add: `ros2_ws/src/lunar_exploration_policy/test/test_action_selection.py`

- [ ] RED：加载时先验证四文件与 ONNX session 的精确输入/输出名称、dtype 和 shape；不一致 fail closed。
- [ ] RED：含 invalid 大 logit 的 masked 候选不能被选中；all-false mask 返回 `NO_CANDIDATE` 且不调用 session。
- [ ] RED：有效候选按最大 logit 确定性选择；同输入重复运行结果逐位一致。
- [ ] RED：WHEELED/LEGGED 使用选中项 `theta_mu`；HOPPER 使用 fed9 `theta_action_mask` 的输出语义。
- [ ] 实现 ONNX Runtime CPU execution provider；运行时禁止导入 torch、checkpoint、reward、collector 或 PPO optimizer。
- [ ] 提交：`feat(interface-v1): add deterministic onnx runtime`。

---

### Task 5：抽取 fed9 旧观测与候选运行时边界

**文件：**

- Add: `ros2_ws/src/lunar_exploration_policy/lunar_exploration_policy/observation_runtime.py`
- Add: `ros2_ws/src/lunar_exploration_policy/lunar_exploration_policy/identity.py`
- Test: `ros2_ws/src/lunar_exploration_policy/test/test_observation_runtime.py`
- Test fixture: `ros2_ws/src/lunar_exploration_policy/test/fixtures/fed9_observation_fixture.npz`

- [ ] 从 fed9 的 `ObservationBuilderV2`、`CandidateBuilderV2` 和 `VisibilityEstimator` 提取或薄封装运行时所需代码；训练端与 ROS 端必须调用同一实现，不复制两套公式。
- [ ] RED：受控 1024 m/4 m 全局画布和 0.2 m 局部图的七输入 shape/dtype、手算 pose 归一化、覆盖率和候选目标坐标一致。
- [ ] RED：30 m/360°、最多 64 候选、候选 mask、三平台投影和 HOPPER 落点过滤保持 fed9 行为。
- [ ] RED：identity 中任一 mission revision、map snapshot、robot state、timestamp 或 candidate set 不匹配均不能构造请求。
- [ ] 保留 C++ v3 `project_traversability()` 为平台可达性权威入口；测试 proxy 只能显式标记 `test_only/proxy`。
- [ ] 提交：`refactor(interface-v1): expose fed9 observation runtime`。

---

### Task 6：实现平台无关的闭环决策协调器

**文件：**

- Add: `ros2_ws/src/lunar_exploration_policy/lunar_exploration_policy/coordinator.py`
- Add: `ros2_ws/src/lunar_exploration_policy/lunar_exploration_policy/planner_request.py`
- Test: `ros2_ws/src/lunar_exploration_policy/test/test_coordinator.py`

- [ ] RED：`WAITING_INPUTS -> READY -> INFERENCING -> PLANNING -> EXECUTING` 正常转换，并输出包含同一 identity 的 PlanMotion goal。
- [ ] RED：候选坐标按画布反归一化，地面 tolerance 为 `0.2 m`、HOPPER 为 `0.0 m`，goal theta 遵循 fed9。
- [ ] RED：旧 revision、旧地图、无 TF、全 false mask、invalid model output 和 planner 不可行不会发出错误动作。
- [ ] RED：foreign/late feedback 不清除当前有效 context；只有匹配且递增的反馈提交新决策边界。
- [ ] RED：轮式、足式执行反馈后第二次请求使用更新 pose/map，不能复用第一次候选 identity。
- [ ] RED：HOPPER 已提交时进入 `LANDED_HOLD`；匹配 landing 后清空旧 episode 状态，第二次选点能再次发起规划。
- [ ] 实现纯 Python 协调器；ROS、ONNX 和 planner 通过小型协议边界注入，测试真实状态转换而非断言 mock 存在。
- [ ] 提交：`feat(interface-v1): add closed-loop policy coordinator`。

---

### Task 7：增加 ROS 2 Humble 生命周期节点

**文件：**

- Add: `ros2_ws/src/lunar_exploration_policy/package.xml`
- Add: `ros2_ws/src/lunar_exploration_policy/setup.py`
- Add: `ros2_ws/src/lunar_exploration_policy/setup.cfg`
- Add: `ros2_ws/src/lunar_exploration_policy/resource/lunar_exploration_policy`
- Add: `ros2_ws/src/lunar_exploration_policy/lunar_exploration_policy/node.py`
- Add: `ros2_ws/src/lunar_exploration_policy/lunar_exploration_policy/diagnostics.py`
- Add: `ros2_ws/src/lunar_exploration_policy/test/test_node.py`
- Add: `ros2_ws/src/lunar_exploration_policy/test/test_interface_launch.py`

- [ ] RED：节点在 configure 阶段读取固定模型目录、平台 YAML 与接口 YAML；任一身份错误则 configure 失败。
- [ ] RED：订阅完整外部输入、锁定同一决策快照、调用 `/plan_motion`，并发布状态/原因/耗时诊断。
- [ ] RED：inactive 不发请求；deactivate/cancel 清理 owned goal/context；重新 activate 不复用旧平台 episode。
- [ ] RED：真实 ROS action fake server 收到完整 Goal；成功/不可行/取消/陈旧结果映射到协调器正确状态。
- [ ] 使用 mutually exclusive callback groups 保护状态；推理与 action 等待不得阻塞反馈订阅。
- [ ] 在主要状态转换、消息转换、固定配置入口添加中文注释，避免逐行翻译式注释。
- [ ] 提交：`feat(interface-v1): add ros exploration policy node`。

---

### Task 8：建立三平台完整链路回归

**文件：**

- Add: `ros2_ws/src/lunar_exploration_policy/test/test_three_platform_chain.py`
- Add: `ros2_ws/src/lunar_exploration_policy/test/fixtures/three_platform_chain.yaml`
- Add: `scripts/run_interface_v1_regression.sh`

- [ ] WHEELED：外部消息 -> adapter -> observation -> 实物 ONNX -> PlanMotion -> MotionReference -> feedback -> 第二次请求。
- [ ] LEGGED：执行相同双决策链，并核对足式状态/净空字段未退化为轮式语义。
- [ ] HOPPER：第一次 goal -> hop reference -> matching LANDED -> 新快照 -> 第二次 goal；第二次 plan_id/candidate set/pose 均不同且可执行。
- [ ] 反例：模型 hash 改变、缺地图层、陈旧 revision、foreign feedback、planner infeasible、all-false mask 均输出明确 HOLD reason 且无误动作。
- [ ] 回归脚本使用独立 `ROS_DOMAIN_ID`、真实四文件包和 Release install；证据写到外部时间戳目录。
- [ ] 提交：`test(interface-v1): cover three-platform full chain`。

---

### Task 9：构建、文档和冻结候选

**文件：**

- Modify: `scripts/build_runtime.sh`
- Add: `scripts/install_interface_v1_config.sh`
- Add: `docs/deployment/interface-v1-quickstart.md`
- Add: `docs/deployment/interface-v1-identity.md`
- Modify: `docs/interfaces/external-input-baseline.md`

- [ ] 将 `lunar_exploration_policy`、`lunar_external_adapter` 和 `lunar_model_contract` 加入 production-like `--packages-select`，保持输出目录外置。
- [ ] 安装脚本只接受一个明确平台 YAML 和一个接口 YAML，复制到固定路径；默认安装但不启动服务。
- [ ] quickstart 给出构建、替换 YAML、启动 planner/adapter/policy、检查 Topic/action/诊断和停止命令。
- [ ] identity 文档记录源码、checkpoint、合同、能力、模型四文件 hash 与 `integration_only` 限制。
- [ ] 运行 UTF-8 回读、`git diff --check`、仓库边界、全部目标 pytest、ROS Release build/test 和三平台进程回归。
- [ ] 自审不存在 v7 语义、训练输出、checkpoint 或构建 artifact 进入 Git。
- [ ] 提交：`docs(interface-v1): add reproducible deployment handoff`。

---

### Task 10：分支收尾

- [ ] 运行 `superpowers:requesting-code-review` 做最终需求符合性和代码质量复审。
- [ ] 修复所有阻断项并重跑对应验证。
- [ ] 使用 `superpowers:verification-before-completion` 汇总新鲜证据。
- [ ] 使用 `superpowers:finishing-a-development-branch` 给出集成选项；未经用户另外授权，不推送远端、不创建正式发布、不宣称 AGX/TensorRT 资格。
- [ ] 只有全部 interface-v1 验证通过后才创建候选 tag；tag 名和合入 `integration` 属于分支收尾动作，不提前执行。
