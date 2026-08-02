# Lunar Navigation 卷一：新仓与平台基线实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立没有父子仓关系的新 `lunar_navigation` Git 仓库，并在 Ubuntu 22.04/ROS 2 Humble 上固定内部消息、外部依赖、平台探测和基础 CI。

**Architecture:** Windows 只生成旧仓迁移清单并提交源码；所有 ROS 接口生成和发布构建都在 Ubuntu 完成。新仓依赖外部项目发布的 `lunar_navigation_msgs`，本卷只创建项目自有的 `lunar_planning_msgs` 和配置包，并用自动检查阻止复制外部消息或引入嵌套 Git。

**Tech Stack:** Git、Python 3.10、YAML、Ubuntu 22.04、ROS 2 Humble、ament_cmake、colcon、rosdep、pytest、GitHub Actions 或等价 CI。

**Codex Estimate:** 7–12 agent-hours；等待外部消息包或设备登录不计入。

## Global Constraints

- 新仓必须是单一 Git 根，不包含 gitlink、嵌套 Git 仓库或相邻目录导入。
- Windows 只承担审计、编辑、提交和远程任务触发，不得产出权威 Linux/ROS/AGX 构建物。
- Ubuntu 22.04 amd64、ROS 2 Humble、GCC 11、CMake 3.22、C++20 和 Python 3.10 是权威开发基线。
- 外部消息由外部项目定义发布；本项目不得复制 `lunar_navigation_msgs` 源码或建立重复 schema。
- 大 rosbag、模型、checkpoint、训练数据和构建目录不得提交 Git。
- Windows 下载和大型临时资源必须写入 `D:/CodexDownloads`。
- 文件必须使用 LF；脚本必须保留 executable bit；源码不得硬编码盘符、用户目录或相邻仓库路径。
- 本卷不迁移算法、PPO 代码、训练 runner、ContentRef、旧 Stage authority/repair 合同或历史 artifact。

---

## File Structure

本卷在新仓根创建以下文件；路径均相对新仓 `lunar_navigation/`：

```text
.gitattributes                         # 跨 Windows/Linux 文本与可执行位规则
.gitignore                             # 构建、训练、模型、rosbag 和设备产物排除
README.md                              # 三机职责和最小构建入口
dependencies.repos                    # 仅在外部依赖没有 Debian/ROS 发布时固定源码
migration/source_inventory.yaml       # 旧仓固定提交和允许迁移的源码清单
migration/fixture_inventory.yaml      # 允许迁移的小型 fixture 及 hash
docs/architecture/system-ownership.md # Windows/Ubuntu/AGX/外部项目职责
docs/migration/source-baseline.md     # 旧仓冻结与恢复说明
tools/check_repository_boundaries.py  # 嵌套 Git、gitlink、路径和产物审计
tools/check_external_interfaces.py    # 外部包、消息类型和字段可见性检查
tools/capture_environment.py          # Ubuntu/AGX 机器可复现环境指纹
tools/create_source_inventory.py      # 从旧仓固定提交生成迁移清单
tests/foundation/test_repository_boundaries.py
tests/foundation/test_external_interface_config.py
tests/foundation/test_source_inventory.py
tests/foundation/test_environment_fingerprint.py
ros2_ws/src/lunar_planning_msgs/      # 本项目自有 Action 与消息
ros2_ws/src/lunar_navigation_config/  # Topic、frame 和平台基线配置
platform/train_amd64_rtx4080/baseline.yaml
platform/deploy_agx_orin_r36/baseline.yaml
scripts/bootstrap_ubuntu.sh           # Ubuntu 依赖初始化
scripts/build_runtime.sh              # 默认不构建 Nav2 和训练目录
.github/workflows/ci-amd64.yaml       # amd64 CPU 发布前基础门槛
.github/workflows/ci-arm64.yaml       # self-hosted ARM64 编译预警
```

### Task 1: 冻结旧仓来源并生成受控迁移清单

**Execution environment:** Windows 开发机。

**Estimated Codex time:** 3–5 小时，包含旧仓、两个 gitlink 和 fixture hash 审计。

**Files:**
- Create: `tools/create_source_inventory.py`
- Create: `migration/source_inventory.yaml`
- Create: `migration/fixture_inventory.yaml`
- Create: `docs/migration/source-baseline.md`
- Create: `tests/foundation/test_source_inventory.py`
- Read only: 旧仓根、旧 `path-planner` gitlink、旧 `dev-platform-constraints` gitlink

**Interfaces:**
- Consumes: 显式绝对路径 `LUNAR_NEW_REPO_ROOT`、三个仓库各自的 `git rev-parse HEAD`、`git remote get-url origin` 和被选 fixture 字节。
- Produces: `lunar-migration-source-inventory/v1` 与 `lunar-migration-fixture-inventory/v1`；后续卷只能从清单列出的提交和文件迁移。

- [ ] **Step 1: 初始化空的新仓 Git 根**

`LUNAR_NEW_REPO_ROOT` 必须指向不存在或为空的绝对目录，resolved path 不得位于旧仓内部。创建该单一目录后运行 `git init --initial-branch=main "$LUNAR_NEW_REPO_ROOT"`，并从此步骤开始把它作为所有相对路径的工作目录；不得复制旧仓 `.git` 或两个 gitlink。

- [ ] **Step 2: 写清单生成器的失败测试**

```python
def test_inventory_records_each_repository_and_sha256(tmp_path):
    repository = tmp_path / "root"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.invalid/legacy.git"],
        cwd=repository,
        check=True,
    )
    (repository / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-m", "fixture",
        ],
        cwd=repository,
        check=True,
    )
    output = create_inventory(
        repositories={"legacy_root": repository},
        selected_files=("README.md",),
    )
    assert output["schema_version"] == "lunar-migration-source-inventory/v1"
    assert len(output["repositories"]["legacy_root"]["commit"]) == 40
    assert len(output["files"][0]["sha256"]) == 64
    assert "absolute_path" not in output["files"][0]
```

- [ ] **Step 3: 运行测试确认失败**

Run:

```powershell
python -m pytest -q tests/foundation/test_source_inventory.py
```

Expected: FAIL，原因是 `tools.create_source_inventory` 尚不存在。

- [ ] **Step 4: 实现确定性清单生成器**

`tools/create_source_inventory.py` 必须提供签名 `create_inventory(*, repositories: dict[str, pathlib.Path], selected_files: Sequence[str]) -> dict[str, object]` 和 `write_inventory(document: dict[str, object], output: pathlib.Path) -> None`。

实现规则：repository key 排序、文件 path 使用 POSIX 相对路径、hash 使用文件原始字节 SHA-256、commit 必须是 40 位小写 hex、输出 UTF-8/LF；任何 dirty 文件若在选择清单中则拒绝生成。

- [ ] **Step 5: 运行测试确认通过**

```powershell
python -m pytest -q tests/foundation/test_source_inventory.py
```

Expected: PASS。

- [ ] **Step 6: 生成实际清单**

从新仓执行，显式传入旧仓根路径，不把该绝对路径写入输出：

```powershell
python tools/create_source_inventory.py `
  --legacy-root "C:/Users/77634/.codex/worktrees/ca49/lunar-path-planning" `
  --output migration/source_inventory.yaml `
  --fixture-output migration/fixture_inventory.yaml
```

Expected: 清单记录旧根仓、`path-planner`、`dev-platform-constraints` 的提交；只选 C++ v3、PPO 核心、平台能力和小型测试 fixture，不选择历史 runner、checkpoint、job state 或 `.obj`。

- [ ] **Step 7: 写冻结与恢复说明**

`docs/migration/source-baseline.md` 必须记录三个 commit、旧仓只读标签、备份位置、fixture hash 校验命令和“新运行时不得导入旧仓”的规则。

- [ ] **Step 8: 提交清单**

```powershell
git add tools/create_source_inventory.py migration/source_inventory.yaml migration/fixture_inventory.yaml docs/migration/source-baseline.md tests/foundation/test_source_inventory.py
git commit -m "chore: freeze legacy migration sources"
```

### Task 2: 创建单 Git 根与仓库边界检查

**Execution environment:** Windows 编辑；Ubuntu 验证。

**Estimated Codex time:** 0.5–1 小时。

**Files:**
- Create: `.gitattributes`
- Create: `.gitignore`
- Create: `README.md`
- Create: `docs/architecture/system-ownership.md`
- Create: `docs/interfaces/external-input-baseline.md`
- Create: `docs/superpowers/specs/2026-08-02-lunar-navigation-greenfield-ros2-jetson-design.md`
- Create: `docs/superpowers/plans/2026-08-02-lunar-navigation-greenfield-roadmap.md`
- Create: `docs/superpowers/plans/2026-08-02-lunar-navigation-volume-1-foundation.md`
- Create: `docs/superpowers/plans/2026-08-02-lunar-navigation-volume-2-planner-ros.md`
- Create: `docs/superpowers/plans/2026-08-02-lunar-navigation-volume-3-policy-pipeline.md`
- Create: `docs/superpowers/plans/2026-08-02-lunar-navigation-volume-4-integration-cutover.md`
- Create: `tools/check_repository_boundaries.py`
- Create: `tests/foundation/test_repository_boundaries.py`

**Interfaces:**
- Consumes: Task 1 清单。
- Produces: 可由 CI 调用的 `check_repository(root: Path) -> list[str]`，空列表代表仓库边界合法。

- [ ] **Step 1: 迁入批准文档**

从设计提交 `47716eb` 逐个迁入已批准设计、总路线图和四卷计划，保持 UTF-8/LF，并在迁移提交说明中记录旧仓来源 commit；不得复制旧仓 `.git` 或两个 gitlink。把用户提供的外部输入文档 hash、Topic/type 和外部所有权整理到 `docs/interfaces/external-input-baseline.md`，将迁入设计文档的旧相对链接改指该文件；文档头必须声明“这是接收基线，不是本项目消息定义源”，且不得包含 `.msg` 源码副本。

- [ ] **Step 2: 写边界检查失败测试**

```python
def test_rejects_nested_git_and_windows_absolute_paths(tmp_path):
    (tmp_path / "nested" / ".git").mkdir(parents=True)
    source = tmp_path / "bad.py"
    source.write_text('MODEL = "D:/models/policy.onnx"\n', encoding="utf-8")
    errors = check_repository(tmp_path)
    assert any("nested Git" in error for error in errors)
    assert any("Windows absolute path" in error for error in errors)
```

- [ ] **Step 3: 运行测试确认失败**

```bash
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
```

Expected: FAIL，模块尚不存在。

- [ ] **Step 4: 实现仓库检查器**

检查器必须：

```python
IGNORED_DIRS = {".git", "build", "install", "log", ".venv", "__pycache__"}
FORBIDDEN_SUFFIXES = {".obj", ".engine", ".pt", ".pth", ".bag", ".db3"}
```

模块必须导出签名 `check_repository(root: pathlib.Path) -> list[str]`。

它必须使用 `git ls-files --stage` 拒绝 mode `160000`，要求所有受跟踪 `.sh` 使用 mode `100755`，扫描根目录之外的 `.git`，拒绝受跟踪大产物和源码中的 Windows 盘符/用户目录，并允许文档中作为明确反例出现的路径。

- [ ] **Step 5: 固定文本规则和忽略规则**

`.gitattributes` 至少包含：

```gitattributes
* text=auto eol=lf
*.sh text eol=lf
*.py text eol=lf
*.cpp text eol=lf
*.hpp text eol=lf
*.md text eol=lf
*.yaml text eol=lf
*.json text eol=lf
```

`.gitignore` 必须排除 `build/`、`install/`、`log/`、`.venv/`、`*.engine`、`*.pt`、`*.pth`、`*.db3`、`*.mcap`、`training-output/` 和 `device-output/`。

- [ ] **Step 6: 运行测试和真实仓审计**

```bash
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
python3 tools/check_repository_boundaries.py .
```

Expected: PASS，真实仓输出 `repository boundaries: OK`。

- [ ] **Step 7: 提交仓库基线**

```bash
git add .gitattributes .gitignore README.md docs/architecture docs/interfaces docs/superpowers tools/check_repository_boundaries.py tests/foundation/test_repository_boundaries.py
git commit -m "build: establish single-repository boundaries"
```

### Task 3: 固定外部接口依赖而不复制消息

**Execution environment:** Ubuntu 22.04 amd64。

**Estimated Codex time:** 0.5–1 小时；等待外部包交付不计时。

**Files:**
- Create: `dependencies.repos`
- Create: `ros2_ws/src/lunar_navigation_config/package.xml`
- Create: `ros2_ws/src/lunar_navigation_config/CMakeLists.txt`
- Create: `ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml`
- Create: `tools/check_external_interfaces.py`
- Create: `tests/foundation/test_external_interface_config.py`

**Interfaces:**
- Consumes: ROS 包 `grid_map_msgs`、`nav_msgs`、`tf2_msgs`、外部 `lunar_navigation_msgs`。
- Produces: Topic/type/frame/字段依赖清单和 `check_interfaces(config: Path) -> list[str]`。

- [ ] **Step 1: 写配置结构测试**

```python
def test_external_interfaces_declare_owner_and_required_fields():
    config = yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))
    localization = config["topics"]["localization_status"]
    assert localization["owner"] == "external"
    assert localization["type"] == "lunar_navigation_msgs/msg/LocalizationStatus"
    assert localization["required_fields"] == ["header", "status"]
```

- [ ] **Step 2: 创建权威外部接口配置**

`external_interfaces.yaml` 必须包含：

```yaml
schema_version: lunar-external-interfaces/v1
topics:
  map_global:
    name: /environment/map_global
    type: grid_map_msgs/msg/GridMap
    owner: external
    frame: map
    required_fields: [header, info, layers, basic_layers, data, outer_start_index, inner_start_index]
  map_local:
    name: /environment/map_local
    type: grid_map_msgs/msg/GridMap
    owner: external
    frame: odom
    required_fields: [header, info, layers, basic_layers, data, outer_start_index, inner_start_index]
  odometry:
    name: /localization/odometry
    type: nav_msgs/msg/Odometry
    owner: external
    frame: odom
    child_frame: base_link
    required_fields: [header, child_frame_id, pose, twist]
  localization_status:
    name: /localization/status
    type: lunar_navigation_msgs/msg/LocalizationStatus
    owner: external
    frame: odom
    required_fields: [header, status]
  exploration_task:
    name: /mission/exploration_task
    type: lunar_navigation_msgs/msg/ExplorationTask
    owner: external
    frame: map
    required_fields: [header, mission_id, revision, desired_state, science_regions]
tf:
  topic: /tf
  type: tf2_msgs/msg/TFMessage
  chain: [map, odom, base_link]
required_grid_layers:
  - elevation
  - valid_mask
  - obstacle
  - obstacle_height
  - observation_age_s
  - observation_quality
  - elevation_variance
  - obstacle_variance
  - observation_count
  - forbidden
static_inputs:
  observation_capability:
    owner: external
    formats: [yaml, json]
    required_fields: [sensor_range_m, sensor_fov_deg]
  platform_capability:
    owner: external
    schema: platform-control-capability-source/v1
    formats: [yaml, json, urdf]
    required_fields: [platform, geometry_source]
```

- [ ] **Step 3: 声明包依赖**

`lunar_navigation_config/package.xml` 必须声明 `grid_map_msgs`、`nav_msgs`、`tf2_msgs`、`lunar_navigation_msgs` 为 `<exec_depend>`。`dependencies.repos` 初始内容固定为：

```yaml
repositories: {}
```

先运行 `rosdep resolve lunar_navigation_msgs` 和 `ros2 pkg prefix lunar_navigation_msgs`。若两者都找不到，停止本卷并向外部项目索取唯一上游仓库 URL 与固定 tag/commit，再只在 `dependencies.repos` 增加该上游；不得复制消息源码。

- [ ] **Step 4: 实现 ROS 接口检查器**

`check_external_interfaces.py` 逐项运行 `ros2 interface show <type>`，验证 required field 的顶层字段存在，并运行 `ros2 pkg prefix` 记录包位置；退出码非零或字段缺失必须返回失败。

- [ ] **Step 5: 运行配置与现场接口测试**

```bash
python3 -m pytest -q tests/foundation/test_external_interface_config.py
python3 tools/check_external_interfaces.py --config ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml
```

Expected: 配置测试通过，所有外部类型可见；输出不包含新仓内复制的 `lunar_navigation_msgs/msg` 路径。

- [ ] **Step 6: 提交外部依赖基线**

```bash
git add dependencies.repos ros2_ws/src/lunar_navigation_config tools/check_external_interfaces.py tests/foundation/test_external_interface_config.py
git commit -m "build: pin external ROS interface dependencies"
```

### Task 4: 创建本项目内部 `lunar_planning_msgs`

**Execution environment:** Ubuntu 22.04 amd64。

**Estimated Codex time:** 1–1.5 小时。

**Files:**
- Create: `ros2_ws/src/lunar_planning_msgs/package.xml`
- Create: `ros2_ws/src/lunar_planning_msgs/CMakeLists.txt`
- Create: `ros2_ws/src/lunar_planning_msgs/msg/GoalRegion.msg`
- Create: `ros2_ws/src/lunar_planning_msgs/msg/HopSegment.msg`
- Create: `ros2_ws/src/lunar_planning_msgs/msg/MotionReference.msg`
- Create: `ros2_ws/src/lunar_planning_msgs/msg/PlannerDiagnostics.msg`
- Create: `ros2_ws/src/lunar_planning_msgs/action/PlanMotion.action`
- Create: `tests/foundation/test_planning_interfaces.py`

**Interfaces:**
- Consumes: ROS 标准消息包。
- Produces: 卷 2、卷 3 和卷 4 唯一允许依赖的内部 ROS 规划接口。

- [ ] **Step 1: 写接口常量和字段测试**

```python
def test_plan_motion_constants_and_goal_surface():
    assert PlanMotion.Goal.get_fields_and_field_types() == {
        "request_id": "string",
        "mission_id": "string",
        "mission_revision": "uint64",
        "goal": "lunar_planning_msgs/GoalRegion",
        "replace_active_request": "boolean",
    }
    assert PlanMotion.Result.STALE_INPUT == 5
    assert PlanMotion.Result.CONTINUE_COMMITTED_HOP == 3
```

- [ ] **Step 2: 定义目标和参考消息**

`GoalRegion.msg`：

```text
uint8 POINT=1
uint8 PLANAR_REGION=2
std_msgs/Header header
string goal_id
uint8 goal_type
geometry_msgs/Point point
geometry_msgs/Polygon planar_region
float64 position_tolerance_m
bool has_yaw_constraint
float64 yaw_rad
float64 yaw_tolerance_rad
```

`HopSegment.msg`：

```text
std_msgs/Header header
string segment_id
geometry_msgs/Pose launch_pose
geometry_msgs/Polygon landing_region
builtin_interfaces/Duration flight_time
geometry_msgs/Vector3 launch_velocity
float64 flight_tube_radius_m
```

`MotionReference.msg`：

```text
uint8 WHEELED=1
uint8 LEGGED=2
uint8 HOPPER=3
std_msgs/Header header
string plan_id
uint8 platform_type
builtin_interfaces/Time input_time
nav_msgs/Path path_preview
trajectory_msgs/MultiDOFJointTrajectory trajectory
lunar_planning_msgs/HopSegment[] hops
```

`PlannerDiagnostics.msg`：

```text
string planner_name
float64 elapsed_s
uint64 expanded_states
bool has_best_cost
float64 best_cost
string[] warning_codes
```

- [ ] **Step 3: 定义 `PlanMotion.action`**

```text
string request_id
string mission_id
uint64 mission_revision
lunar_planning_msgs/GoalRegion goal
bool replace_active_request
---
uint8 NEW_REFERENCE_AVAILABLE=0
uint8 SAFE_FRONTIER_REFERENCE_AVAILABLE=1
uint8 NO_KNOWN_SAFE_ROUTE=2
uint8 GOAL_INFEASIBLE=3
uint8 INVALID_REQUEST=4
uint8 STALE_INPUT=5
uint8 NUMERICAL_FAILURE=6
uint8 RESOURCE_EXHAUSTED=7
uint8 ACTIVE_REFERENCE_INVALIDATED=8
uint8 CANCELED=9
uint8 ACTIVATE_NEW_REFERENCE=0
uint8 CONTINUE_ACTIVE_REFERENCE=1
uint8 HOLD_POSITION=2
uint8 CONTINUE_COMMITTED_HOP=3
uint8 NO_SAFE_REFERENCE=4
uint8 planning_outcome
uint8 execution_directive
string reason_code
builtin_interfaces/Time global_map_stamp
builtin_interfaces/Time local_map_stamp
builtin_interfaces/Time state_stamp
uint64 mission_revision
bool has_reference
lunar_planning_msgs/MotionReference reference
lunar_planning_msgs/PlannerDiagnostics diagnostics
---
uint8 VALIDATING_INPUT=0
uint8 BUILDING_SNAPSHOT=1
uint8 SEARCHING=2
uint8 OPTIMIZING=3
uint8 CERTIFYING=4
uint8 phase
float64 elapsed_s
uint64 expanded_states
bool has_best_cost
float64 best_cost
```

- [ ] **Step 4: 配置 rosidl 生成**

`CMakeLists.txt` 必须用 `rosidl_generate_interfaces` 枚举上述五个文件，并声明 `builtin_interfaces`、`geometry_msgs`、`nav_msgs`、`std_msgs`、`trajectory_msgs` 依赖。`package.xml` 必须声明 `rosidl_default_generators` 和 `rosidl_default_runtime`。

- [ ] **Step 5: 构建并运行接口测试**

```bash
colcon build --base-paths ros2_ws/src --packages-select lunar_planning_msgs
source install/setup.bash
python3 -m pytest -q tests/foundation/test_planning_interfaces.py
ros2 interface show lunar_planning_msgs/action/PlanMotion
```

Expected: PASS；Action Goal 不包含地图、Odometry、TF、能力、算法配置或模型字段。

- [ ] **Step 6: 提交内部接口**

```bash
git add ros2_ws/src/lunar_planning_msgs tests/foundation/test_planning_interfaces.py
git commit -m "feat: define internal planning action interfaces"
```

### Task 5: 建立平台基线与环境指纹

**Execution environment:** Ubuntu amd64 和 AGX 各执行一次；Windows 只审阅输出。

**Estimated Codex time:** 0.5–1 小时，不含设备登录等待。

**Files:**
- Create: `platform/train_amd64_rtx4080/baseline.yaml`
- Create: `platform/deploy_agx_orin_r36/baseline.yaml`
- Create: `tools/capture_environment.py`
- Create: `tests/foundation/test_environment_fingerprint.py`

**Interfaces:**
- Consumes: `/etc/os-release`、`uname`、ROS、编译器和 NVIDIA 命令输出。
- Produces: `lunar-platform-fingerprint/v1` JSON；卷 3 engine cache 和卷 4发布清单依赖其字段。

- [ ] **Step 1: 写指纹解析测试**

```python
def test_agx_fingerprint_requires_r36_and_aarch64():
    document = parse_probe(FIXTURE_AGX)
    validate_fingerprint(document, expected_profile="deploy_agx_orin_r36")
    assert document["architecture"] == "aarch64"
    assert document["l4t"] == "R36.0.0"
    assert document["device_model"] == "Jetson AGX Orin 64GB"
```

- [ ] **Step 2: 固定两个基线文件**

训练基线：

```yaml
schema_version: lunar-platform-baseline/v1
profile: train_amd64_rtx4080
os: Ubuntu 22.04 LTS
architecture: amd64
ros_distro: humble
python: "3.10"
gpu_model: NVIDIA GeForce RTX 4080
responsibilities: [ros_integration, ppo_training, onnx_export, rosbag_replay]
```

部署基线：

```yaml
schema_version: lunar-platform-baseline/v1
profile: deploy_agx_orin_r36
os: Ubuntu 22.04 LTS
architecture: aarch64
ros_distro: humble
device_model: Jetson AGX Orin 64GB
l4t: R36.0.0
jetpack: "6.0"
responsibilities: [native_build, tensorrt_engine, inference, device_release_gate]
```

- [ ] **Step 3: 实现环境探测**

`capture_environment.py` 必须记录：OS、arch、kernel、ROS distro、Python、GCC、CMake、GPU 名称、驱动、CUDA、TensorRT、L4T、JetPack、功耗模式和时钟状态。命令不可用时字段写入 `available: false` 和命令错误，不得伪造版本。

- [ ] **Step 4: 在两台 Linux 主机采集指纹**

```bash
python3 tools/capture_environment.py --profile train_amd64_rtx4080 --output /tmp/ubuntu-fingerprint.json
python3 tools/capture_environment.py --profile deploy_agx_orin_r36 --output /tmp/agx-fingerprint.json
```

Expected: Ubuntu 指纹匹配 amd64/RTX 4080；AGX 指纹匹配 aarch64/R36.0.0。发现版本差异时停止发布流程并保留实际输出，不修改基线掩盖差异。

- [ ] **Step 5: 运行测试并提交**

```bash
python3 -m pytest -q tests/foundation/test_environment_fingerprint.py
git add platform tools/capture_environment.py tests/foundation/test_environment_fingerprint.py
git commit -m "build: define Ubuntu and AGX platform baselines"
```

### Task 6: 创建 Ubuntu 权威构建入口

**Execution environment:** Ubuntu 22.04 amd64。

**Estimated Codex time:** 0.5–1 小时。

**Files:**
- Create: `scripts/bootstrap_ubuntu.sh`
- Create: `scripts/build_runtime.sh`
- Create: `tests/foundation/test_build_scripts.py`

**Interfaces:**
- Consumes: apt/rosdep 和新仓源码。
- Produces: 可重复的依赖安装命令与默认运行时构建；默认跳过训练目录和 `lunar_nav2_adapter`。

- [ ] **Step 1: 写脚本静态测试**

测试必须断言脚本包含 `set -euo pipefail`、检查 Ubuntu 22.04/Humble、运行 `rosdep install --from-paths ros2_ws/src --ignore-src`，并确认默认 `colcon build` 参数含 `--packages-skip lunar_nav2_adapter`。

- [ ] **Step 2: 实现 bootstrap**

`bootstrap_ubuntu.sh` 安装或验证 `build-essential`、`gcc-11`、`g++-11`、`cmake`、`python3.10`、`python3-pip`、`python3-venv`、`python3-colcon-common-extensions`、`python3-rosdep` 和 ROS Humble。脚本若检测到非 Ubuntu 22.04 或非 Humble 必须退出非零。

- [ ] **Step 3: 实现默认运行时构建**

```bash
#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/humble/setup.bash
colcon build \
  --base-paths ros2_ws/src \
  --packages-skip lunar_nav2_adapter \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

该脚本不得安装 `training/`，不得读取 Windows 构建目录。

- [ ] **Step 4: 验证并提交**

```bash
python3 -m pytest -q tests/foundation/test_build_scripts.py
bash -n scripts/bootstrap_ubuntu.sh scripts/build_runtime.sh
git add scripts tests/foundation/test_build_scripts.py
git update-index --chmod=+x scripts/bootstrap_ubuntu.sh scripts/build_runtime.sh
git commit -m "build: add authoritative Ubuntu bootstrap"
```

### Task 7: 建立 amd64 基础 CI 与 ARM64 编译预警

**Execution environment:** CI；本地 Ubuntu 可复现。

**Estimated Codex time:** 0.5–1 小时。

**Files:**
- Create: `.github/workflows/ci-amd64.yaml`
- Create: `.github/workflows/ci-arm64.yaml`
- Create: `docs/architecture/ci-release-responsibilities.md`

**Interfaces:**
- Consumes: Tasks 2–6 的脚本与包。
- Produces: PR 基础门槛和非发布性质的 ARM64 编译报告。

- [ ] **Step 1: 创建 amd64 job**

job 必须在 Ubuntu 22.04 + ROS 2 Humble 环境依次运行：repository boundary、外部接口配置静态测试、rosdep、`scripts/build_runtime.sh`、`colcon test` 和 `colcon test-result --verbose`。

- [ ] **Step 2: 创建 ARM64 self-hosted job**

runner labels 固定为 `[self-hosted, linux, arm64, lunar-arm64]`。job 只运行依赖解析、构建和非 GPU 测试，并在摘要中打印：`ARM64 compile check is not an AGX release gate`。

- [ ] **Step 3: 本地复现 CI**

```bash
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation
scripts/build_runtime.sh
colcon test --base-paths ros2_ws/src --packages-select lunar_planning_msgs lunar_navigation_config
colcon test-result --verbose
```

Expected: 全部通过。

- [ ] **Step 4: 提交 CI**

```bash
git add .github/workflows docs/architecture/ci-release-responsibilities.md
git commit -m "ci: add amd64 and arm64 foundation gates"
```

### Task 8: 卷一完成门槛与回退点

**Execution environment:** Ubuntu 22.04 amd64；结果由 Windows 审阅提交。

**Estimated Codex time:** 0.5 小时。

**Files:**
- Verify: 本卷全部文件
- Create: `docs/migration/volume-1-completion.md`

**Interfaces:**
- Consumes: Tasks 1–7 全部提交。
- Produces: 卷 2 可依赖的 `foundation-v1` tag。

- [ ] **Step 1: 从干净 checkout 验证**

```bash
git status --short
python3 tools/check_repository_boundaries.py .
python3 tools/check_external_interfaces.py --config ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml
python3 -m pytest -q tests/foundation
scripts/build_runtime.sh
colcon test --base-paths ros2_ws/src --packages-select lunar_planning_msgs lunar_navigation_config
colcon test-result --verbose
```

Expected: `git status` 为空，其余命令全部通过。

- [ ] **Step 2: 审计禁止内容**

```bash
git ls-files | grep -E '(^|/)(build|install|log)/|\.(pt|pth|engine|db3|mcap|obj)$' && exit 1 || true
git ls-files | grep -E '(^|/)lunar_navigation_msgs/(msg|action|srv)/' && exit 1 || true
git ls-files --stage | awk '$1 == "160000" { exit 1 }'
```

Expected: 不存在受跟踪构建/模型/rosbag/object、外部消息源码或 gitlink。

- [ ] **Step 3: 记录完成报告**

`docs/migration/volume-1-completion.md` 记录 Git commit、Ubuntu 指纹 hash、AGX 指纹 hash、外部包版本、构建和测试结果；不嵌入绝对用户目录。

- [ ] **Step 4: 提交并打标签**

```bash
git add docs/migration/volume-1-completion.md
git commit -m "docs: record foundation completion gate"
git tag -a foundation-v1 -m "lunar navigation foundation v1"
```

**Rollback:** 本卷尚未接入运行设备。任何失败都在新仓修复；旧仓和旧部署保持不变，不删除或回退用户文件。
