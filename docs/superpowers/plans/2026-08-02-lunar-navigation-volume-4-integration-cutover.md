# Lunar Navigation 卷四：Ubuntu/AGX 集成与切换实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Ubuntu amd64 完成外部 rosbag 驱动的全链路集成，并在 AGX Orin R36.0.0 上完成原生构建、TensorRT、性能、4 小时稳定性、版本化安装和旧仓切换。

**Architecture:** `lunar_navigation_bringup` 依序管理规划与探索 Lifecycle 节点；真实外部输入通过固定 hash 的 rosbag 验证，不建立万能网关。Ubuntu 生成只含源码/依赖/模型/配置/报告身份的 release candidate，AGX 从同一 Git tag 原生构建并在版本目录安装；`current` 符号链接提供原子切换和可恢复回退。

**Tech Stack:** ROS 2 Humble、launch_ros、rclpy、rosbag2_py、pytest、launch_testing、colcon、rosdep、C++20、TensorRT、systemd、Bash、JSON Schema、SHA-256、tegrastats。

**Codex Estimate:** 22–41 agent-hours；其中包含一次 4 小时自动稳定性运行，不包含物理操作等待。

## Global Constraints

- Ubuntu 22.04 amd64/RTX 4080 是完整集成与发布候选生成环境；AGX aarch64/R36.0.0 是最终发布环境。
- Ubuntu 与 AGX 必须使用同一 Git tag、外部消息版本、ONNX、配置和 release manifest。
- 普通 ARM64 或交叉编译只做预警，不得替代 AGX 实机验收。
- AGX 必须原生构建并本机生成 TensorRT engine；不得接收 amd64 二进制或 RTX 4080 engine。
- 外部 rosbag 不得提交 Git；仓库只提交 asset ID、SHA-256、Topic/类型和回放预期。
- 规划节点与探索节点任一 configure 失败都不得进入 Active。
- v3 必须满足 AGX 固定 benchmark `P95 < 1 s`。
- PPO 推理 P95 必须至少保留 20% 决策周期余量。
- 发布前必须完成一次连续 4 小时 AGX 稳定性测试；预热后 CPU/GPU 内存不得持续单调增长。
- Action 取消、替换、任务暂停/取消、定位失效、TF/地图陈旧和 engine 失败必须按设计降级，不得回退旧 A* 或其他推理后端。
- 部署包不得包含 `training/`、checkpoint、优化器、训练数据、大 rosbag、Python A* 或旧 Stage runner。
- 初始安装根必须是 `/opt/lunar_navigation`；模型根 `/var/lib/lunar_navigation/models`；engine cache `/var/cache/lunar_navigation/tensorrt`；日志根 `/var/log/lunar_navigation`。
- 切换前必须保留上一已验收版本和旧系统；不得批量删除旧仓或旧部署。
- R36.0.0 基线在任何新 L4T 通过完整验收前不得移除。

---

## File Structure

```text
ros2_ws/src/lunar_navigation_bringup/
├── package.xml
├── setup.py
├── lunar_navigation_bringup/lifecycle_manager.py
├── launch/navigation.launch.py
├── launch/integration_replay.launch.py
├── config/runtime.schema.json
└── test/
platform/deploy_agx_orin_r36/
├── runtime.yaml
├── power-profile-policy.yaml
└── expected-install-manifest.txt
test_assets/external_rosbags.yaml
tools/derive_freshness_profile.py
tools/verify_rosbag_asset.py
tools/build_release_manifest.py
release/release-manifest.schema.json
scripts/run_rosbag_integration.sh
scripts/build_release_candidate.sh
scripts/install_agx.sh
scripts/verify_agx_install.sh
scripts/run_agx_release_gate.sh
scripts/rollback_agx.sh
deployment/systemd/lunar-navigation.service
deployment/systemd/lunar-navigation.env.example
tests/integration/full_chain/
tests/integration/faults/
tests/device/agx/
docs/runbooks/ubuntu-integration.md
docs/runbooks/agx-install.md
docs/runbooks/agx-rollback.md
docs/migration/cutover-report.md
```

### Task 1: 从真实 rosbag 派生运行时新鲜度配置并建立 Bringup

**Execution environment:** Ubuntu 22.04 amd64。

**Estimated Codex time:** 1–3 小时；等待 rosbag 交付不计时。

**Files:**
- Create: `test_assets/external_rosbags.yaml`
- Create: `tools/verify_rosbag_asset.py`
- Create: `tools/derive_freshness_profile.py`
- Create: `ros2_ws/src/lunar_navigation_bringup/package.xml`
- Create: `ros2_ws/src/lunar_navigation_bringup/setup.py`
- Create: `ros2_ws/src/lunar_navigation_bringup/lunar_navigation_bringup/lifecycle_manager.py`
- Create: `ros2_ws/src/lunar_navigation_bringup/launch/navigation.launch.py`
- Create: `ros2_ws/src/lunar_navigation_bringup/config/runtime.schema.json`
- Create: `platform/deploy_agx_orin_r36/runtime.yaml`
- Create: `ros2_ws/src/lunar_navigation_bringup/test/test_lifecycle_manager.py`
- Create: `tests/integration/test_freshness_profile.py`

**Interfaces:**
- Consumes: 共同确认的外部 rosbag、卷二/卷三 Lifecycle 节点。
- Produces: 带 source bag hash 的运行时 freshness/skew 配置和确定性节点启动顺序。

- [ ] **Step 1: 写 asset hash 和配置推导测试**

```python
def test_profile_rejects_rate_slower_than_safety_ceiling():
    stats = valid_topic_stats()
    stats["/localization/odometry"]["period_p99_s"] = 0.30
    with pytest.raises(ProfileError, match="odometry_max_age exceeds 0.5 s ceiling"):
        derive_profile(stats)
```

- [ ] **Step 2: 登记真实 rosbag 而不提交数据**

`external_rosbags.yaml` 每项必须记录 `asset_id`、SHA-256、storage format、duration、Topic/type 列表、外部接口版本和用途。`verify_rosbag_asset.py --bag <absolute-path> --asset-id <id>` 校验 hash 和 `ros2 bag info`；路径只来自 CLI，不写入 Git。

- [ ] **Step 3: 实现新鲜度推导规则**

对每个 Topic 用 bag receive time 与 header stamp 计算 period p99、delay p99 和跨消息 skew p99。候选 max age 为 `max(3 * period_p99, delay_p99 + 2 * period_p99)`；候选 pairwise skew 为 `max(0.05, skew_p99 + 0.02)`。硬 ceiling 固定：global map 10.0 s、local map 1.0 s、odometry 0.5 s、localization status 1.0 s、TF 0.5 s、pairwise skew 0.25 s；超过即失败并要求外部输入频率/时间戳修复。

- [ ] **Step 4: 生成生产配置**

```bash
python3 tools/verify_rosbag_asset.py --bag "$LUNAR_INTEGRATION_BAG" --asset-id primary-integration-v1
python3 tools/derive_freshness_profile.py \
  --bag "$LUNAR_INTEGRATION_BAG" \
  --output platform/deploy_agx_orin_r36/runtime.yaml
```

Expected: `runtime.yaml` 含 source asset ID/hash、六个显式时限、Topic 名、frame、模型目录、engine cache、Action 名和 decision period。

- [ ] **Step 5: 实现 Lifecycle manager**

启动顺序固定为：planner configure → planner activate → exploration configure → exploration activate。任一步失败时逆序 deactivate/cleanup 已成功节点并退出非零；运行中任一节点进入 Error 时停止探索并将规划节点置为 Inactive。

- [ ] **Step 6: 运行测试并提交**

```bash
python3 -m pytest -q tests/integration/test_freshness_profile.py
colcon build --base-paths ros2_ws/src model_contract --packages-up-to lunar_navigation_bringup
colcon test --base-paths ros2_ws/src model_contract --packages-select lunar_navigation_bringup
colcon test-result --verbose
git add test_assets tools/derive_freshness_profile.py tools/verify_rosbag_asset.py ros2_ws/src/lunar_navigation_bringup platform/deploy_agx_orin_r36/runtime.yaml
git commit -m "feat: derive runtime timing and orchestrate lifecycle"
```

### Task 2: 建立完整 rosbag 纵向集成测试

**Execution environment:** Ubuntu 22.04 amd64；TensorRT 用已验证 fake runner，规划使用真实 C++ v3。

**Estimated Codex time:** 3–5 小时。

**Files:**
- Create: `ros2_ws/src/lunar_navigation_bringup/launch/integration_replay.launch.py`
- Create: `tests/integration/full_chain/test_goal_to_reference.launch.py`
- Create: `tests/integration/full_chain/test_mission_revision.launch.py`
- Create: `tests/integration/full_chain/test_output_contract.py`
- Create: `scripts/run_rosbag_integration.sh`
- Create: `docs/runbooks/ubuntu-integration.md`

**Interfaces:**
- Consumes: Task 1 runtime config、卷二规划 Action、卷三探索节点、外部 rosbag。
- Produces: 外部输入 → observation → policy goal → Action → v3 → MotionReference 的报告。

- [ ] **Step 1: 写端到端测试断言**

测试至少断言：Lifecycle 全 Active、mission ID/revision 保持、每个 Action Result 的实际 map/local/state stamps 来自回放、`has_reference` 不变量、轮式/足式/飞跃式输出类型正确、没有控制器 Topic 发布。

- [ ] **Step 2: 实现隔离回放 launch**

launch 使用测试 namespace、`use_sim_time=true`、只 remap 配置中列出的外部 Topic，并把 fake policy runner 输出固定到一个安全 candidate。不得启动 Nav2、executor 或旧 A*。

- [ ] **Step 3: 实现集成脚本**

脚本参数固定为 `--bag`、`--output`、`--runtime-config`；先验证 bag hash，清理只允许由本次命令创建的一个明确临时目录，运行 launch test，再写 `summary.json` 和 JUnit。output 必须是显式绝对路径。

- [ ] **Step 4: 运行完整链**

```bash
python3 -m pytest -q tests/integration/full_chain
scripts/run_rosbag_integration.sh \
  --bag "$LUNAR_INTEGRATION_BAG" \
  --runtime-config platform/deploy_agx_orin_r36/runtime.yaml \
  --output "$LUNAR_INTEGRATION_OUTPUT"
```

Expected: 全链路通过，输出报告绑定 bag/model/config/core commit hash。

- [ ] **Step 5: 提交纵向测试**

```bash
git add ros2_ws/src/lunar_navigation_bringup tests/integration/full_chain scripts/run_rosbag_integration.sh docs/runbooks/ubuntu-integration.md
git update-index --chmod=+x scripts/run_rosbag_integration.sh
git commit -m "test: add external rosbag end-to-end integration"
```

### Task 3: 建立跨节点异常与恢复矩阵

**Execution environment:** Ubuntu 22.04 amd64；关键子集在 AGX 重跑。

**Estimated Codex time:** 2–4 小时。

**Files:**
- Create: `tests/integration/faults/fault_cases.yaml`
- Create: `tests/integration/faults/test_localization_faults.launch.py`
- Create: `tests/integration/faults/test_input_time_faults.launch.py`
- Create: `tests/integration/faults/test_mission_faults.launch.py`
- Create: `tests/integration/faults/test_runtime_faults.launch.py`
- Create: `tests/integration/faults/test_process_restart.launch.py`
- Create: `tests/integration/faults/fault_injector.py`

**Interfaces:**
- Consumes: Task 2 integration launch。
- Produces: 每个设计降级条件的机器可读预期和测试证据。

- [ ] **Step 1: 固定故障表**

`fault_cases.yaml` 必须逐项声明 stimulus、expected Lifecycle、Action status/outcome/directive、是否允许新目标和 diagnostics code。覆盖：定位 UNKNOWN/INVALID/RELOCALIZING/DEGRADED、局部图缺失/过期、全局图过期、TF 缺失/外推/skew、mission PAUSED/CANCELED/revision 乱序、TensorRT NaN/shape/error、v3 无路线/数值/资源失败、规划进程重启、探索进程重启和 engine cache 损坏。

- [ ] **Step 2: 实现 namespace 内故障注入器**

注入器只能发布测试 namespace Topic 或控制 fake backend；不得改真实设备 Topic、系统时间或模型目录。每个测试 setup/teardown 使用唯一临时目录并验证目标位于测试 output root。

- [ ] **Step 3: 运行故障矩阵**

```bash
python3 -m pytest -q tests/integration/faults
```

Expected: 所有 case 与 YAML 预期一致；日志中不存在 A*、ORT、PyTorch 或 CPU fallback。

- [ ] **Step 4: 提交故障测试**

```bash
git add tests/integration/faults
git commit -m "test: cover navigation degradation and recovery matrix"
```

### Task 4: 生成不可混用二进制的发布候选清单

**Execution environment:** Ubuntu 22.04 amd64 + RTX 4080。

**Estimated Codex time:** 2–3 小时。

**Files:**
- Create: `release/release-manifest.schema.json`
- Create: `tools/build_release_manifest.py`
- Create: `scripts/build_release_candidate.sh`
- Create: `tests/release/test_release_manifest.py`
- Create: `tests/release/test_deploy_boundary.py`

**Interfaces:**
- Consumes: 卷二 tag、卷三模型包、Task 1 配置和 Ubuntu 测试报告。
- Produces: `lunar-navigation-release/v1` manifest；AGX 安装和 release gate 的唯一输入。

- [ ] **Step 1: 写禁止 amd64 binary 的失败测试**

```python
def test_release_candidate_rejects_host_binary(tmp_path):
    candidate = make_candidate(tmp_path)
    (candidate / "planner_amd64").write_bytes(b"ELF")
    with pytest.raises(ReleaseError, match="unexpected release file"):
        validate_candidate(candidate)
```

- [ ] **Step 2: 定义 release manifest**

manifest 必须记录：schema/version/release ID、Git tag+commit、外部包版本、model ID/version/四文件 hash、配置 hash、Ubuntu 指纹 hash、Ubuntu test report hash、AGX baseline ID，以及明确的 `agx_device_report: null` 预发布状态。AGX gate 完成后生成新的 qualified manifest，不原地覆盖预发布文件。

- [ ] **Step 3: 实现候选生成器**

候选目录只包含：`release-manifest.json`、四文件模型包、平台/算法/runtime 配置副本和 Ubuntu 报告。源码通过 Git tag 获取，不打包 build/install/log；脚本发现 ELF、wheel、venv、checkpoint、engine、training 数据或 rosbag 即失败。

- [ ] **Step 4: 用测试 fixture 验证候选生成器**

```bash
python3 -m pytest -q tests/release
scripts/build_release_candidate.sh --self-test
```

Expected: fixture candidate 边界测试通过，未包含 amd64 或训练 artifact；真实 release candidate 必须等 Tasks 5–7 的安装、验收和激活工具全部提交后在 Task 8 生成。

- [ ] **Step 5: 提交发布工具**

```bash
git add release tools/build_release_manifest.py scripts/build_release_candidate.sh tests/release
git update-index --chmod=+x scripts/build_release_candidate.sh
git commit -m "build: create source-only AGX release candidates"
```

### Task 5: 实现 AGX 原生构建与版本化安装工具

**Execution environment:** Ubuntu 进行路径/脚本测试；Task 8 在 AGX R36.0.0 执行。

**Estimated Codex time:** 2–4 小时。

**Files:**
- Create: `scripts/install_agx.sh`
- Create: `scripts/verify_agx_install.sh`
- Create: `platform/deploy_agx_orin_r36/expected-install-manifest.txt`
- Create: `tests/device/agx/test_native_install.py`
- Create: `docs/runbooks/agx-install.md`

**Interfaces:**
- Consumes: Task 4 manifest schema 与 fixture candidate。
- Produces: `/opt/lunar_navigation/releases/<release-id>` 原生 install、模型版本目录和本机 engine cache。

- [ ] **Step 1: 写安装路径与边界测试**

测试必须验证 release ID 只含 `[A-Za-z0-9._-]`、resolved 目标位于 `/opt/lunar_navigation/releases`、模型目标位于 `/var/lib/lunar_navigation/models`、不接受 amd64 ELF、候选 hash 全匹配。

- [ ] **Step 2: 实现 AGX 预检**

安装脚本先运行环境指纹并严格要求 aarch64、Ubuntu 22.04、ROS Humble、AGX Orin 64GB、R36.0.0；验证 Git HEAD、外部包版本、candidate 和模型 hash。任一不符时在写 `/opt` 前退出。

- [ ] **Step 3: 实现原生构建**

脚本以普通用户在显式临时 build root 执行 rosdep 和：

```bash
colcon build \
  --base-paths ros2_ws/src model_contract \
  --packages-skip lunar_nav2_adapter \
  --merge-install \
  --install-base "/opt/lunar_navigation/releases/${release_id}" \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DLUNAR_ENABLE_TENSORRT=ON
```

脚本只为单个明确 release 目录设置写权限，不以 root 运行编译器。构建失败不更新任何 `current` 链接。

- [ ] **Step 4: 实现模型安装与 engine 生成步骤**

四文件包复制到 `/var/lib/lunar_navigation/models/<model-id>/<version>`，验证后本机执行 `build_engine` 与 `verify_engine`。不得复制 Ubuntu/RTX engine；测试用命令注入器验证调用参数，真实 engine 在 Task 8 生成。

- [ ] **Step 5: 运行脚本和安装清单测试**

```bash
bash -n scripts/install_agx.sh scripts/verify_agx_install.sh
python3 -m pytest -q tests/device/agx/test_native_install.py
```

Expected: 临时前缀中的 installed package、shared library、Python module、launch/config/model 和模拟 engine 均被清单验证；`training/`、checkpoint、A*、Nav2 adapter 被拒绝。

- [ ] **Step 6: 提交安装工具**

```bash
git add scripts/install_agx.sh scripts/verify_agx_install.sh platform/deploy_agx_orin_r36/expected-install-manifest.txt tests/device/agx/test_native_install.py docs/runbooks/agx-install.md
git update-index --chmod=+x scripts/install_agx.sh scripts/verify_agx_install.sh
git commit -m "deploy: add native AGX versioned installation"
```

### Task 6: 实现 AGX 性能、功耗、异常和稳定性门槛工具

**Execution environment:** Ubuntu 用 fixture 验证分析器；Task 8 在 AGX R36.0.0 执行。

**Estimated Codex time:** 2–4 小时。

**Files:**
- Create: `platform/deploy_agx_orin_r36/power-profile-policy.yaml`
- Create: `tests/device/agx/benchmark_planner.py`
- Create: `tests/device/agx/benchmark_policy.py`
- Create: `tests/device/agx/test_fault_subset.py`
- Create: `tests/device/agx/monotonic_rosbag_replayer.py`
- Create: `tests/device/agx/run_soak.py`
- Create: `tests/device/agx/analyze_soak.py`
- Create: `tests/device/agx/test_benchmark_analysis.py`
- Create: `tests/device/agx/test_soak_analysis.py`
- Create: `tests/device/agx/test_release_gate_report.py`
- Create: `scripts/run_agx_release_gate.sh`

**Interfaces:**
- Consumes: Task 5 安装目录合同、真实 rosbag 元数据和 candidate manifest schema。
- Produces: `agx-device-report.json`、benchmark、fault、tegrastats 和 soak 报告。

- [ ] **Step 1: 固定但不自动修改设备功耗状态**

`power-profile-policy.yaml` 要求 gate 前后记录 `nvpmodel -q`、`jetson_clocks --show`、温度和后台负载。脚本不得自动切换功耗模式或锁频；每次比较必须使用同一记录状态，不一致即失败重跑。

- [ ] **Step 2: 实现规划和推理 benchmark**

规划 benchmark：100 warmup、1000 measured、固定 fixture，要求 `p95 < 1.0 s`。策略 benchmark：100 warmup、1000 measured，要求 `p95 <= 0.8 * decision_period`。两者记录 median/p95/p99/max、CPU/GPU memory、温度和 throttling。

- [ ] **Step 3: 实现 AGX 关键故障子集**

至少覆盖 Action cancel/replace、PAUSED、CANCELED、定位 INVALID/RELOCALIZING、local map stale、TF skew、TensorRT output NaN、engine cache 损坏和进程重启。任何 fallback 或节点错误恢复不符即失败。

- [ ] **Step 4: 定义 4 小时 soak 命令和分析规则**

```bash
python3 tests/device/agx/run_soak.py \
  --duration-seconds 14400 \
  --bag "$LUNAR_INTEGRATION_BAG" \
  --sample-period-seconds 5 \
  --output /var/log/lunar_navigation/soak
python3 tests/device/agx/analyze_soak.py --input /var/log/lunar_navigation/soak
```

`monotonic_rosbag_replayer.py` 必须逐轮给 `/clock`、GridMap、Odometry、LocalizationStatus、TF 和 ExplorationTask 的 header stamp 增加 `loop_index * (bag_duration + 1 s)`，并给任务 revision 增加 `loop_index * (max_revision + 1)`；地图数组和任务语义不变。不得用 `ros2 bag play --loop` 产生向后时间跳变。分析规则：前 15 分钟为 warmup；之后无未处理异常、无 Lifecycle Error、Action 成功/正常无路线语义合法、RSS 与 GPU memory 不持续单调增长、线性斜率均不超过 1 MiB/hour、末 30 分钟均值不超过首个稳态 30 分钟均值的 5%。真实 14400 秒执行在 Task 8；本 Task 使用固定时序 fixture 测试通过与内存泄漏拒绝两条路径。

- [ ] **Step 5: 测试设备报告生成器**

```bash
python3 -m pytest -q tests/device/agx/test_benchmark_analysis.py tests/device/agx/test_soak_analysis.py tests/device/agx/test_release_gate_report.py
bash -n scripts/run_agx_release_gate.sh
```

Expected: 合格 fixture 生成 passed 报告；任一 P95、余量、故障、内存或指纹不合格 fixture 生成 failed 报告。

- [ ] **Step 6: 实现 qualified manifest 并提交测试工具**

qualified manifest 在新文件中写入 AGX report hash 和 `qualified: true`，不得改写原 candidate manifest。

```bash
git add platform/deploy_agx_orin_r36/power-profile-policy.yaml tests/device/agx scripts/run_agx_release_gate.sh
git update-index --chmod=+x scripts/run_agx_release_gate.sh
git commit -m "test: add AGX release qualification gate"
```

### Task 7: 实现 systemd、原子切换和回退工具

**Execution environment:** Ubuntu 使用临时前缀测试；Task 8 在 AGX R36.0.0 执行。

**Estimated Codex time:** 2–3 小时。

**Files:**
- Create: `deployment/systemd/lunar-navigation.service`
- Create: `deployment/systemd/lunar-navigation.env.example`
- Create: `deployment/start_lunar_navigation.sh`
- Create: `scripts/activate_agx_release.sh`
- Create: `scripts/rollback_agx.sh`
- Create: `tests/device/agx/test_activation_rollback.py`
- Create: `docs/runbooks/agx-rollback.md`
- Modify: `ros2_ws/src/lunar_navigation_bringup/setup.py`

**Interfaces:**
- Consumes: Task 6 qualified manifest fixture 和 Task 5 安装目录合同。
- Produces: `/opt/lunar_navigation/current`、模型 `current` 和可恢复的 systemd 服务。

- [ ] **Step 1: 写未验收版本拒绝测试**

```python
def test_activation_rejects_unqualified_manifest(tmp_path):
    manifest = valid_manifest() | {"qualified": False}
    write_manifest(tmp_path, manifest)
    result = run_activate(tmp_path)
    assert result.returncode != 0
    assert current_release() == "previous-qualified"
```

- [ ] **Step 2: 创建最小权限启动脚本和服务**

`deployment/start_lunar_navigation.sh` 必须 `source /opt/ros/humble/setup.bash`、`source /opt/lunar_navigation/current/setup.bash`，要求 EnvironmentFile 提供绝对路径 `LUNAR_RUNTIME_CONFIG`，再执行 `exec ros2 launch lunar_navigation_bringup navigation.launch.py runtime_config:="${LUNAR_RUNTIME_CONFIG}"`；通过 bringup `setup.py` 安装到 `lib/lunar_navigation_bringup/`。unit 使用专用 `lunar-navigation` 用户、`Restart=on-failure`、明确 EnvironmentFile 和 `ExecStart=/usr/bin/bash /opt/lunar_navigation/current/lib/lunar_navigation_bringup/start_lunar_navigation.sh`；不以 root 运行节点，不启动 Nav2 或 executor。

- [ ] **Step 3: 实现原子激活**

激活脚本验证 qualified manifest、installed files、model 和 engine，再创建同目录临时 symlink 并用单次 rename 更新 `current`。更新前记录 previous release/model；systemd restart 和健康检查成功后才写 `activation.json`。

- [ ] **Step 4: 实现单版本回退**

回退脚本只接受明确 `--release-id` 和 `--model-version`，验证目标已 qualified，再原子更新两个链接并重启服务。不得删除失败或当前目录。

- [ ] **Step 5: 测试并提交**

```bash
python3 -m pytest -q tests/device/agx/test_activation_rollback.py
systemd-analyze verify deployment/systemd/lunar-navigation.service
git add deployment scripts/activate_agx_release.sh scripts/rollback_agx.sh tests/device/agx/test_activation_rollback.py docs/runbooks/agx-rollback.md ros2_ws/src/lunar_navigation_bringup/setup.py
git update-index --chmod=+x deployment/start_lunar_navigation.sh scripts/activate_agx_release.sh scripts/rollback_agx.sh
git commit -m "deploy: add atomic AGX activation and rollback"
```

### Task 8: 生成 RC、执行 AGX 实机验收并切换新主线

**Execution environment:** Ubuntu 生成候选；AGX R36.0.0 安装验收；Windows/Git 审阅与旧仓归档。

**Estimated Codex time:** 7–13 小时，其中包含一次 4 小时稳定性运行；登录、传输和人工操作等待不计入。

**Files:**
- Create: `docs/migration/cutover-report.md`
- Create: `docs/migration/legacy-recovery.md`
- Modify: `README.md`
- Generate outside Git: `$LUNAR_RELEASE_ROOT/candidate/`
- Generate outside Git: `/var/log/lunar_navigation/release-gate/`
- Verify: 旧仓和两个旧 gitlink，只读

**Interfaces:**
- Consumes: Tasks 1–7 的干净提交、卷三四文件模型包和真实外部 rosbag。
- Produces: `lunar-navigation-v1.0.0-rc.1`、qualified manifest、已激活 AGX release、新仓唯一维护主线声明和旧仓恢复点。

- [ ] **Step 1: 冻结完整 RC 源码并生成真实候选**

```bash
git status --short
git tag -a lunar-navigation-v1.0.0-rc.1 -m "first complete AGX release candidate"
scripts/build_release_candidate.sh \
  --git-tag lunar-navigation-v1.0.0-rc.1 \
  --model-dir "$LUNAR_RELEASE_MODEL_DIR" \
  --ubuntu-report "$LUNAR_INTEGRATION_OUTPUT/summary.json" \
  --output "$LUNAR_RELEASE_ROOT/candidate"
```

`LUNAR_RELEASE_MODEL_DIR` 必须是绝对路径，并指向卷三统一 release gate 已通过的四文件模型包；完整训练至收敛的等待不计入 Codex 时间。Expected: 打标签前工作区为空；candidate 只包含 manifest、四文件模型包、配置和 Ubuntu 报告，manifest 的 Git commit 等于 RC tag。

- [ ] **Step 2: 在 AGX checkout 同一 tag 并原生安装**

通过已批准传输通道把 candidate 复制到 AGX 明确目录并验证整目录 hash；在 AGX checkout `lunar-navigation-v1.0.0-rc.1`，随后执行：

```bash
scripts/install_agx.sh \
  --release-id lunar-navigation-v1.0.0-rc.1 \
  --candidate "$LUNAR_RELEASE_CANDIDATE"
scripts/verify_agx_install.sh \
  --release-id lunar-navigation-v1.0.0-rc.1 \
  --candidate "$LUNAR_RELEASE_CANDIDATE"
```

Expected: aarch64 原生安装、四文件模型和本机 FP32 TensorRT engine 通过；未更新 `current`。

- [ ] **Step 3: 执行完整 AGX release gate**

```bash
scripts/run_agx_release_gate.sh \
  --release-manifest "$LUNAR_RELEASE_CANDIDATE/release-manifest.json" \
  --bag "$LUNAR_INTEGRATION_BAG" \
  --output /var/log/lunar_navigation/release-gate
```

Expected: 规划 `p95 < 1.0 s`、推理 `p95 <= 0.8 * decision_period`、关键故障矩阵、R36.0.0 指纹和 14400 秒 soak 全部通过；生成新 qualified manifest，不覆盖 candidate manifest。

- [ ] **Step 4: 激活并观察新服务**

```bash
scripts/activate_agx_release.sh \
  --release-id lunar-navigation-v1.0.0-rc.1 \
  --qualified-manifest /var/log/lunar_navigation/release-gate/qualified-release-manifest.json
```

至少完成一个正常任务周期、一次 Action cancel、一次 PAUSED/ACTIVE 恢复和一次节点重启健康检查。任一失败立即使用 `scripts/rollback_agx.sh` 回到记录的 previous qualified release，并保留失败日志。

- [ ] **Step 5: 确认生产安装无旧路线**

```bash
rg -n 'AStarPlanner|path_planner\.search|ContentRef|stage6|xunce_mid|repair_authority' /opt/lunar_navigation/current && exit 1 || true
find /opt/lunar_navigation/current -type d -name training -print -quit | grep . && exit 1 || true
```

Expected: 无旧 A*、合同、Stage runner 或训练目录。旧命令未被证明有活跃消费者，因此本计划不创建兼容 wrapper；未来若出现明确消费者，单独设计一个仅调用新命令的过渡计划。

- [ ] **Step 6: 建立旧仓恢复点并提交切换报告**

记录旧根仓与两个 gitlink commit，创建带注释 archive tag；远程只读权限由仓库管理员执行。不得删除本地旧仓、gitlink 或历史 artifact。README 明确新仓是唯一维护主线；`legacy-recovery.md` 只说明 checkout/archive replay，不允许新运行时 import 旧仓。

```bash
git add README.md docs/migration/cutover-report.md docs/migration/legacy-recovery.md
git commit -m "docs: cut over to qualified lunar navigation mainline"
```

### Task 9: 最终发布验收和 `v1.0.0` 标签

**Execution environment:** Ubuntu amd64、AGX R36.0.0、Windows Git 审阅。

**Estimated Codex time:** 1–2 小时。

**Files:**
- Create: `docs/migration/final-acceptance.md`
- Verify: 四卷 completion reports、qualified manifest、AGX device report

**Interfaces:**
- Consumes: Tasks 1–8 与前三卷 release points。
- Produces: `lunar-navigation-v1.0.0`。

- [ ] **Step 1: 对照 12 项架构验收标准**

逐项把证据映射到 Git file/test/report hash：单 Git 根、双架构干净构建、外部消息所有权、v3 唯一生产规划器、A* 仅差分、Action 异常、训练/ONNX、TensorRT parity、release manifest、AGX 性能/soak、部署边界、旧合同退出维护路径。

- [ ] **Step 2: 复验 release manifest 闭包**

```bash
python3 tools/build_release_manifest.py verify --manifest "$LUNAR_QUALIFIED_MANIFEST"
python3 tools/check_repository_boundaries.py .
git status --short
```

Expected: manifest 闭包和仓库边界通过，Git 工作区干净。

- [ ] **Step 3: 写最终验收报告**

`final-acceptance.md` 记录 qualified release ID、Ubuntu/AGX 指纹、所有 report hash、仍保留的旧仓 archive tag、回退版本以及排除项（训练收敛等待、外部项目延期、物理操作）。

- [ ] **Step 4: 提交并打发布标签**

```bash
git add docs/migration/final-acceptance.md
git commit -m "docs: record lunar navigation v1 acceptance"
git tag -a lunar-navigation-v1.0.0 -m "first AGX-qualified lunar navigation release"
```

**Rollback:** 激活后出现问题时按 `docs/runbooks/agx-rollback.md` 回到上一个 qualified release/model，保留失败版本、engine、日志和报告。旧系统在约定观察期结束前保持可启动；任何删除或权限归档由用户单独确认。
