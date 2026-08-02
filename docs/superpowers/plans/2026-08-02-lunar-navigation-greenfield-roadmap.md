# Lunar Navigation 绿地迁移总路线实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从当前父仓库与两个 gitlink 子仓库迁出一个以 C++ v3 为唯一生产规划内核、可在 ROS 2 Humble 与 Jetson AGX Orin 上部署的单 Git 根新项目。

**Architecture:** 迁移分成四个可独立验收的卷册：先建立新仓与外部依赖基线，再迁移 C++ v3 和独立 `PlanMotion` Action，然后重构 PPO 训练与 ONNX/TensorRT 发布链，最后完成 Ubuntu 集成、AGX 实机验收和旧仓切换。Windows 只管理源码，Ubuntu amd64/RTX 4080 是权威开发训练环境，AGX aarch64/R36.0.0 是唯一设备发布门槛。

**Tech Stack:** Git、Ubuntu 22.04、ROS 2 Humble、GCC 11、CMake 3.22、C++20、Python 3.10、PyTorch、ONNX、TensorRT、ament、colcon、rosdep、pytest、GoogleTest、launch_testing。

## Global Constraints

- 新项目必须是单一 Git 根，不包含 gitlink、嵌套 Git 仓库或相邻仓库导入。
- 目标运行系统必须是 Ubuntu 22.04 LTS 与 ROS 2 Humble。
- C++ 工具链必须以 GCC 11、CMake 3.22、C++20 为基线。
- Python 工具链必须以 Python 3.10 为基线。
- 最终部署设备必须是 Jetson AGX Orin 64GB、aarch64、L4T R36.0.0；新 L4T 未通过验收前不得删除 R36.0.0 基线。
- C++ v3 必须是唯一生产规划内核；Python A* 只能存在于迁移差分测试，且不得进入生产部署包。
- ROS 主入口必须是独立 `PlanMotion` Action；Nav2 只能是默认不构建的轮式薄适配器。
- PPO 训练必须在 Ubuntu 22.04 amd64、RTX 4080 上进行；AGX 只执行推理。
- 训练端必须使用 PyTorch，跨机器模型交换格式必须是 ONNX，TensorRT engine 必须在 AGX 本机生成。
- 外部 ROS 消息必须由外部项目定义和发布；本项目只声明依赖、订阅、校验和适配，不得复制定义。
- 平台有效最大坡度必须取外部能力上限与项目 `30°` 硬上限中的较小值。
- synthetic terrain 只能标为 proxy，不得声明为真实物理障碍。
- v3 或 TensorRT 失败时不得静默回退到 Python A*、PyTorch、ONNX Runtime 或 CPU 推理。
- 规划固定基准必须在 AGX 满足 `P95 < 1 s`；PPO 推理 P95 必须至少保留 20% 决策周期余量。
- 发布前必须完成一次 4 小时 AGX 稳定性测试。
- Windows 不得成为 ROS 2、Linux wheel、C++ 发布包或 TensorRT 的权威构建环境。
- 大 rosbag、checkpoint、训练数据、优化器、日志和构建产物不得提交 Git。
- 所有下载、模型、缓存和大型运行 artifact 在 Windows 上必须写入 `D:/CodexDownloads` 或明确的 D 盘目录。

---

## 1. 执行环境责任矩阵

| 环境/责任方 | 执行任务 | 权威产物 | 不得承担 |
|---|---|---|---|
| Windows 开发机 | 旧仓审计、迁移清单、源码编辑、Git 提交与审查、触发远程任务 | Git commit/tag、迁移清单 | ROS/Linux 发布构建、训练、TensorRT engine、AGX 验收 |
| Ubuntu 22.04 amd64 + RTX 4080 | rosdep、colcon、C++/ROS 测试、PPO 训练、checkpoint 读取、ONNX 导出与等价、rosbag 回放 | 测试报告、checkpoint、模型包候选、发布候选清单 | AGX TensorRT engine、aarch64 最终发布结论 |
| ARM64 编译通道 | 编译和非 GPU 架构预警 | ARM64 编译报告 | 代替 AGX R36.0.0 实机门槛 |
| AGX Orin 64GB/R36.0.0 | aarch64 原生构建、TensorRT engine、Action/异常/性能/功耗/稳定性、安装验证 | engine cache、设备报告、发布准入结论 | PPO 训练、接收 amd64 二进制或 RTX 4080 engine |
| 外部项目 | 定义并发布地图、定位、TF、任务、科学区域和能力资料 | ROS/Debian 包、固定版本、共同 rosbag | 依赖本项目内部 `lunar_planning_msgs` |

## 2. 跨系统制品流

```text
Windows
  Git commit/tag + migration/source_inventory.yaml
            |
            v
Ubuntu amd64 / RTX 4080
  source tag + test reports + policy.onnx + manifest.json
  + golden_inputs.npz + golden_outputs.npz
            |
            v
AGX Orin / R36.0.0
  native install + locally-built TensorRT engine + device report
```

只允许向 AGX 传递：

- 已固定的 Git 提交或标签；
- 外部消息包版本；
- `policy.onnx`；
- `manifest.json`；
- `golden_inputs.npz` 与 `golden_outputs.npz`；
- 平台、算法和运行参数；
- Ubuntu 测试与发布候选报告。

严禁向 AGX 传递：

- RTX 4080 生成的 TensorRT engine；
- PyTorch checkpoint、优化器状态、训练数据和训练日志；
- amd64 二进制、Windows 构建目录或 Python 虚拟环境。

## 3. 四卷计划及依赖

| 顺序 | 计划卷 | 工作软件门槛 | Codex agent-hours |
|---:|---|---|---:|
| 1 | [`2026-08-02-lunar-navigation-volume-1-foundation.md`](2026-08-02-lunar-navigation-volume-1-foundation.md) | 新仓能在 Ubuntu 干净 checkout 上解析外部依赖、生成内部 ROS 接口并通过 amd64/ARM64 基础构建 | 7–12 |
| 2 | [`2026-08-02-lunar-navigation-volume-2-planner-ros.md`](2026-08-02-lunar-navigation-volume-2-planner-ros.md) | 简化 C++ API、三平台 v3、Lifecycle Action、输入快照和可选 Nav2 通过测试 | 24–42 |
| 3 | [`2026-08-02-lunar-navigation-volume-3-policy-pipeline.md`](2026-08-02-lunar-navigation-volume-3-policy-pipeline.md) | RTX 4080 训练冒烟、ONNX 模型包、AGX TensorRT 等价和探索节点通过 | 22–41 |
| 4 | [`2026-08-02-lunar-navigation-volume-4-integration-cutover.md`](2026-08-02-lunar-navigation-volume-4-integration-cutover.md) | 完整 rosbag 链、AGX 性能与 4 小时稳定性、安装、回退和旧仓归档全部通过 | 22–41 |
| 横向 | 四卷接口复核、失败重试和发布回归 | 四卷产物来自同一 release candidate，接口/配置/模型 hash 一致 | 10–19 |

总量：**85–155 Codex agent-hours**。

墙钟目标：

- Ubuntu 首个可运行纵向版本：24–40 小时；
- Jetson 首个端到端版本：42–70 小时；
- 完整迁移、合同清理和一次实机稳定性验收：65–110 小时；
- 包含 R36.0.0、ARM64 和 TensorRT 重试的 P80：80–140 小时。

不计入：完整 PPO 重新训练至收敛、等待外部项目交付、凭据等待、接线、刷机和人工重启。R36.0.0 若出现重大兼容或刷机问题，另加 8–20 Codex 调试小时；人工等待不计入 Codex 时间。

## 4. 稳定跨卷接口

### 4.1 C++ 规划接口

卷 2 必须发布以下唯一长期入口，卷 3 和卷 4 只能消费它，不得绕过：

```cpp
namespace lunar::planning {

class Planner final {
 public:
  Planner();
  ~Planner();
  Planner(Planner&&) noexcept;
  Planner& operator=(Planner&&) noexcept;
  Planner(const Planner&) = delete;
  Planner& operator=(const Planner&) = delete;

  [[nodiscard]] PlannerOutput Plan(
      const PlannerInput& input) noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::planning
```

`PlannerInput` 只包含当前状态、目标、不可变地图、平台能力、算法配置、简化执行上下文和协作取消令牌；不得包含 ROS 消息、JSON Schema、ContentRef、registry handle 或文件路径。

### 4.2 ROS 规划接口

卷 1 固定 `lunar_planning_msgs`，卷 2 实现其行为：

```text
PlanMotion.action
GoalRegion.msg
MotionReference.msg
HopSegment.msg
PlannerDiagnostics.msg
```

Action Goal 不携带地图、Odometry、TF、平台能力、算法配置或模型；规划节点在接收请求时冻结自身订阅数据的一致快照。

### 4.3 模型包接口

卷 3 必须只向卷 4 发布以下目录内容：

```text
<model-id>/<version>/
├── policy.onnx
├── manifest.json
├── golden_inputs.npz
└── golden_outputs.npz
```

`manifest.json` 的 schema ID 固定为 `lunar-policy-manifest/v1`。卷 4 不得直接读取 checkpoint。

### 4.4 发布候选接口

卷 4 生成的 `release-manifest.json` 必须绑定：

- 源 Git commit/tag；
- 外部 ROS 包名与版本；
- `policy.onnx` SHA-256；
- 平台、算法和运行配置 SHA-256；
- Ubuntu 测试报告 SHA-256；
- AGX 设备指纹和设备报告 SHA-256。

## 5. 总执行顺序

### Task 1: 完成卷 1 并冻结新仓基线

**Files:**
- Execute: `docs/superpowers/plans/2026-08-02-lunar-navigation-volume-1-foundation.md`

**Interfaces:**
- Consumes: 已批准架构规范、设计提交 `47716eb` 和执行 Task 1 时冻结的旧仓源码提交。
- Produces: 新仓 Git 根、内部 ROS 接口、外部依赖清单、环境探测报告和基础 CI。

- [ ] **Step 1: 执行卷 1 的全部任务并逐任务提交**

每个任务只提交新仓中的相关文件；旧仓保持可回放，不修改两个 gitlink 内容。

- [ ] **Step 2: 在 Ubuntu 执行卷 1 completion gate**

Run:

```bash
python3 tools/check_repository_boundaries.py .
python3 tools/check_external_interfaces.py --config ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml
colcon build --base-paths ros2_ws/src --packages-select lunar_planning_msgs lunar_navigation_config
colcon test --base-paths ros2_ws/src --packages-select lunar_planning_msgs lunar_navigation_config --event-handlers console_direct+
colcon test-result --verbose
```

Expected: 全部命令退出码为 0，且没有复制的 `lunar_navigation_msgs/msg` 文件。

- [ ] **Step 3: 标记卷 1 release point**

```bash
git tag -a foundation-v1 -m "lunar navigation foundation v1"
```

### Task 2: 完成卷 2 并固定规划接口

**Files:**
- Execute: `docs/superpowers/plans/2026-08-02-lunar-navigation-volume-2-planner-ros.md`

**Interfaces:**
- Consumes: `foundation-v1`、外部消息包与已冻结 v3 源码/fixture 清单。
- Produces: `lunar_planner_core`、`lunar_planner_ros`、可选 `lunar_nav2_adapter` 和 Action 集成报告。

- [ ] **Step 1: 执行纯 C++ 迁移任务**

每个平台迁移后必须先通过 core 单元与差分测试，再删除对应旧合同适配；不得一次性替换全部平台。

- [ ] **Step 2: 执行 ROS Action 与输入适配任务**

Action、地图、定位和 TF 必须使用分离 callback group；规划调用必须在独立 `std::jthread` 中运行。

- [ ] **Step 3: 在 Ubuntu 执行卷 2 completion gate**

```bash
colcon build --base-paths ros2_ws/src --packages-up-to lunar_planner_ros --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
colcon test --base-paths ros2_ws/src --packages-select lunar_planner_core lunar_planner_ros --event-handlers console_direct+
colcon test-result --verbose
```

Expected: core、三平台、Action、取消、替换、陈旧输入和 Lifecycle 测试全部通过。

- [ ] **Step 4: 标记卷 2 release point**

```bash
git tag -a planner-action-v1 -m "C++ v3 and PlanMotion action v1"
```

### Task 3: 完成卷 3 并发布模型包

**Files:**
- Execute: `docs/superpowers/plans/2026-08-02-lunar-navigation-volume-3-policy-pipeline.md`

**Interfaces:**
- Consumes: `planner-action-v1`、经清点的 PPO 核心源码和选定 checkpoint。
- Produces: 训练/评估命令、ONNX 模型包、`lunar_policy_runtime`、`lunar_exploration` 和等价报告。

- [ ] **Step 1: 在 Ubuntu RTX 4080 执行训练冒烟、评估和 ONNX 发布**

```bash
python3 -m pytest -q training/lunar_policy_training/tests training/model_export/tests
python3 -m lunar_policy_training.cli train-smoke --config training/configs/rtx4080_smoke.yaml --artifact-root "$LUNAR_TRAIN_ARTIFACT_ROOT"
python3 -m model_export.publish --checkpoint "$LUNAR_TRAIN_ARTIFACT_ROOT/checkpoints/smoke.pt" --output "$LUNAR_TRAIN_ARTIFACT_ROOT/model-package"
python3 -m model_export.verify_package "$LUNAR_TRAIN_ARTIFACT_ROOT/model-package"
```

Expected: 训练冒烟通过，模型包只有四个允许文件且哈希和 ONNX Runtime 等价通过。

- [ ] **Step 2: 在 AGX 生成 TensorRT engine 并通过黄金等价**

```bash
ros2 run lunar_policy_runtime build_engine --model-dir /var/lib/lunar_navigation/models/current --cache-root /var/cache/lunar_navigation/tensorrt --precision fp32
ros2 run lunar_policy_runtime verify_engine --model-dir /var/lib/lunar_navigation/models/current --cache-root /var/cache/lunar_navigation/tensorrt
```

Expected: engine 路径包含 ONNX hash 与设备运行时指纹；黄金结果通过 manifest 声明的 FP32 容差。

- [ ] **Step 3: 标记卷 3 release point**

```bash
git tag -a policy-pipeline-v1 -m "PPO ONNX TensorRT pipeline v1"
```

### Task 4: 完成卷 4 并切换唯一维护主线

**Files:**
- Execute: `docs/superpowers/plans/2026-08-02-lunar-navigation-volume-4-integration-cutover.md`

**Interfaces:**
- Consumes: `planner-action-v1`、`policy-pipeline-v1`、共同确认的外部 rosbag 和 AGX 访问。
- Produces: 完整发布候选、AGX 设备报告、安装包、回退点和旧仓归档说明。

- [ ] **Step 1: 在 Ubuntu 通过完整 rosbag 与故障矩阵**

```bash
pytest -q tests/integration
scripts/run_rosbag_integration.sh --bag "$LUNAR_INTEGRATION_BAG" --output "$LUNAR_INTEGRATION_OUTPUT"
```

Expected: PPO 目标到 Action 到 v3 的纵向链路和所有降级场景通过。

- [ ] **Step 2: 在 AGX 通过发布门槛**

```bash
scripts/run_agx_release_gate.sh --release-manifest release/release-manifest.json --output /var/log/lunar_navigation/release-gate
```

Expected: 原生构建、TensorRT 等价、规划 `P95 < 1 s`、推理余量、故障矩阵和 4 小时稳定性全部通过。

- [ ] **Step 3: 安装新版本并保留回退点**

只更新 `/opt/lunar_navigation/current` 符号链接；保留上一个已验收 release 目录和旧系统，直到新系统完成最终观察期。

- [ ] **Step 4: 归档旧仓**

旧仓打只读标签并记录恢复方法；不得删除旧仓、批量清理文件或让新运行时导入旧仓源码。

- [ ] **Step 5: 标记新主线发布**

```bash
git tag -a lunar-navigation-v1.0.0 -m "first AGX-qualified lunar navigation release"
```

## 6. 全局完成门槛

- [ ] 新仓无 gitlink、嵌套 Git 和相邻目录导入。
- [ ] Ubuntu amd64 与 AGX aarch64 均从同一干净 checkout 构建。
- [ ] 外部消息只通过发布依赖消费，没有复制定义。
- [ ] C++ v3 是唯一生产规划器。
- [ ] Python A* 不在生产安装和部署包中。
- [ ] Action 完成、取消、替换、陈旧输入和定位异常全部通过。
- [ ] checkpoint 可训练、评估并导出 ONNX。
- [ ] AGX TensorRT 与黄金结果等价。
- [ ] 发布候选清单绑定源码、外部依赖、模型、配置和报告。
- [ ] AGX 规划性能、推理余量和 4 小时稳定性通过。
- [ ] 部署包不含训练代码、checkpoint、优化器和大测试数据。
- [ ] 旧 Stage 合同、ContentRef 图和父子仓依赖不再影响新代码维护。

## 7. 回退原则

每卷通过后建立带注释 Git tag。卷 2、卷 3 或卷 4 失败时，回到最近通过的卷册标签继续修复，不回写旧仓生产代码。AGX 切换使用版本目录加 `current` 符号链接；失败时只把链接恢复到上一个已验收版本，不删除失败版本，保留日志用于诊断。
