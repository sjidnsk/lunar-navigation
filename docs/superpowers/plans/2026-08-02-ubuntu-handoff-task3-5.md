# Lunar Navigation Ubuntu 交接 Task 3–5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Ubuntu 22.04 amd64 权威主机上落地暂定 `lunar_navigation_msgs`、完成内部 `lunar_planning_msgs`、建立 RTX 4080 SUPER 平台基线与可审计环境指纹，并恢复卷一 Task 3–5 的连续可执行性。

**Architecture:** Topic 数据仍由外部系统生产，仓库只临时提供同名消息 schema，并通过配置、仓库边界和现场 ament 来源检查保证同名包唯一。ROS 构建产物和机器指纹全部写入仓库外的显式输出根；平台基线只保存期望合同，现场指纹保存实际探测值并独立给出 readiness 结果。

**Tech Stack:** Ubuntu 22.04 LTS、ROS 2 Humble、Python 3.10、pytest、YAML、ament_cmake、rosidl、colcon、NVIDIA 595.84、CUDA Toolkit 13.2。

## Global Constraints

- `docs/interfaces/external-input-baseline.md` 是三个暂定消息字段与语义的唯一权威来源；`.msg`、配置和检查器不得与其漂移。
- `/localization/status` 与 `/mission/exploration_task` 的 Topic 数据所有者保持 `external`；schema 提供方暂记为 `in_repository_provisional`。
- 工作区中任何时刻只能存在一个 `lunar_navigation_msgs`；不得同时引入同名 Debian、源码仓或第二个 overlay。
- 暂定包只能包含 `LocalizationStatus.msg`、`ScienceTargetRegion.msg` 和 `ExplorationTask.msg`，不得新增 `.srv`、`.action` 或第四个 `.msg`。
- `ExplorationTask.desired_state=0` 非法；`science_regions` 必须使用上限为 64 的 bounded sequence。
- 权威训练基线固定为 Ubuntu 22.04 LTS、amd64、ROS 2 Humble、Python 3.10、NVIDIA GeForce RTX 4080 SUPER、PCI ID `10de:2702`，profile 为 `train_amd64_rtx4080_super`。
- 当前已验证 NVIDIA 595.84、DRM KMS 和 CUDA 13.2 正常；指纹工具仍必须保留所有探测失败的原始错误，不能用这些计划值回填现场结果。
- 驱动安装前没有由正式工具生成的 JSON 指纹，因此不得反向合成历史 artifact；只采集实现时的当前就绪指纹，并在实施记录中说明时间顺序。
- 所有 colcon `build/`、`install/`、`log/`、测试报告和指纹必须位于显式 `LUNAR_VOLUME1_OUTPUT`，不得写入仓库。
- 执行 ROS 命令前必须 `source /opt/ros/humble/setup.bash` 并确认 `ROS_DISTRO=humble`。
- 保留现有 `AGENTS.md` 和 `.vscode/` 用户改动；提交只纳入本计划相关文件，`AGENTS.md` 只允许暂存设备型号修正和已批准的暂定 schema 例外，不能带入用户其他修改。
- 本计划不实现外部 Topic 发布节点、运行时任务语义校验器、训练流程、TensorRT engine 或 AGX 实机验收。

---

## File Structure

```text
AGENTS.md
README.md
docs/architecture/system-ownership.md
docs/interfaces/external-input-baseline.md
docs/superpowers/specs/2026-08-02-lunar-navigation-greenfield-ros2-jetson-design.md
docs/superpowers/specs/2026-08-02-ubuntu-handoff-baseline-amendment-design.md
docs/superpowers/plans/2026-08-02-lunar-navigation-greenfield-roadmap.md
docs/superpowers/plans/2026-08-02-lunar-navigation-volume-1-foundation.md
docs/superpowers/plans/2026-08-02-lunar-navigation-volume-3-policy-pipeline.md
docs/superpowers/plans/2026-08-02-lunar-navigation-volume-4-integration-cutover.md
tests/foundation/test_documentation_baseline.py

ros2_ws/src/lunar_navigation_msgs/package.xml
ros2_ws/src/lunar_navigation_msgs/CMakeLists.txt
ros2_ws/src/lunar_navigation_msgs/msg/LocalizationStatus.msg
ros2_ws/src/lunar_navigation_msgs/msg/ScienceTargetRegion.msg
ros2_ws/src/lunar_navigation_msgs/msg/ExplorationTask.msg
tests/foundation/test_navigation_message_package.py
tests/ros/test_generated_navigation_interfaces.py

ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml
tools/check_external_interfaces.py
tests/foundation/test_external_interface_config.py
tools/check_repository_boundaries.py
tests/foundation/test_repository_boundaries.py

ros2_ws/src/lunar_planning_msgs/package.xml
ros2_ws/src/lunar_planning_msgs/CMakeLists.txt
ros2_ws/src/lunar_planning_msgs/msg/GoalRegion.msg
ros2_ws/src/lunar_planning_msgs/msg/HopSegment.msg
ros2_ws/src/lunar_planning_msgs/msg/MotionReference.msg
ros2_ws/src/lunar_planning_msgs/msg/PlannerDiagnostics.msg
ros2_ws/src/lunar_planning_msgs/action/PlanMotion.action
tests/foundation/test_planning_message_package.py
tests/ros/test_generated_planning_interfaces.py

platform/train_amd64_rtx4080_super/baseline.yaml
platform/deploy_agx_orin_r36/baseline.yaml
tools/capture_environment.py
tests/foundation/test_environment_fingerprint.py
```

## Execution Setup

每个需要 ROS 或仓库外 artifact 的 Task 都先执行以下设置；不得依赖前一个 shell 留下的变量：

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
export LUNAR_VOLUME1_OUTPUT="$HOME/CodexDownloads/lunar_navigation/volume1-foundation"
mkdir -p "$LUNAR_VOLUME1_OUTPUT"
```

### Task 1: 统一权威文档和 RTX 4080 SUPER 命名

**Files:**
- Create: `tests/foundation/test_documentation_baseline.py`
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/architecture/system-ownership.md`
- Modify: `docs/interfaces/external-input-baseline.md`
- Modify: `docs/superpowers/specs/2026-08-02-lunar-navigation-greenfield-ros2-jetson-design.md`
- Modify: `docs/superpowers/plans/2026-08-02-lunar-navigation-greenfield-roadmap.md`
- Modify: `docs/superpowers/plans/2026-08-02-lunar-navigation-volume-1-foundation.md`
- Modify: `docs/superpowers/plans/2026-08-02-lunar-navigation-volume-3-policy-pipeline.md`
- Modify: `docs/superpowers/plans/2026-08-02-lunar-navigation-volume-4-integration-cutover.md`

**Interfaces:**
- Consumes: 已批准的 Ubuntu 交接修订设计和当前工作区用户改动。
- Produces: 唯一训练设备/profile 命名，以及明确区分 Topic 数据所有权和暂定 schema 所有权的文档合同。

- [ ] **Step 1: 写文档一致性的失败测试**

```python
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_GPU_DOCS = (
    "AGENTS.md",
    "README.md",
    "docs/architecture/system-ownership.md",
    "docs/superpowers/specs/2026-08-02-lunar-navigation-greenfield-ros2-jetson-design.md",
    "docs/superpowers/plans/2026-08-02-lunar-navigation-greenfield-roadmap.md",
    "docs/superpowers/plans/2026-08-02-lunar-navigation-volume-1-foundation.md",
    "docs/superpowers/plans/2026-08-02-lunar-navigation-volume-3-policy-pipeline.md",
    "docs/superpowers/plans/2026-08-02-lunar-navigation-volume-4-integration-cutover.md",
)


def test_active_documents_use_rtx4080_super_names():
    stale = []
    for relative in ACTIVE_GPU_DOCS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        if re.search(r"RTX 4080(?! SUPER)", text) or re.search(r"rtx4080(?!_super)", text):
            stale.append(relative)
    assert stale == []


def test_external_baseline_declares_provisional_schema_authority():
    text = (ROOT / "docs/interfaces/external-input-baseline.md").read_text(encoding="utf-8")
    assert "暂定消息字段与语义的唯一权威基线" in text
    assert "Topic 数据生产者仍由外部项目拥有" in text
    assert "不得据此复制" not in text


def test_agent_rules_record_the_approved_provisional_exception():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "暂定提供同名 schema" in text
    assert "不得与上游同名包共存" in text
```

- [ ] **Step 2: 运行测试确认旧命名和旧所有权声明导致失败**

Run:

```bash
python3 -m pytest -q tests/foundation/test_documentation_baseline.py
```

Expected: FAIL；报告仍含 `RTX 4080`/`rtx4080` 的文件，并指出外部输入基线仍声明“不是消息定义源”。

- [ ] **Step 3: 修订外部输入基线的权威性和所有权段落**

文档开头必须改为：

```markdown
> **这是本项目暂定消息字段与语义的唯一权威基线。** `lunar_navigation_msgs` 上游尚未定义期间，本仓 `.msg`、配置和检查器必须与本文一致；Topic 数据生产者仍由外部项目拥有。未来切换上游必须执行固定版本、schema 对比和原子替换，不能叠加同名包。
```

“来源与所有权”必须明确：`grid_map_msgs`、`nav_msgs` 和 `tf2_msgs` 仍来自 ROS/外部系统；本仓暂定提供三个 `lunar_navigation_msgs` schema；定位与任务系统仍负责消息数据的发布和演进协商。

紧随 Topic 表增加“暂定消息 schema”章节，逐字包含批准设计中的 `LocalizationStatus.msg`、`ScienceTargetRegion.msg` 和 `ExplorationTask.msg` 三个 `text` 代码块；常量、字段顺序、类型和 `[<=64]` 不得省略。后续静态测试直接把这些代码块与实际 `.msg` 比较。

- [ ] **Step 4: 修订所有仍生效的设备、profile 和文件名引用**

逐文件应用以下精确映射：

```text
RTX 4080                         -> RTX 4080 SUPER
train_amd64_rtx4080             -> train_amd64_rtx4080_super
rtx4080_smoke.yaml              -> rtx4080_super_smoke.yaml
ubuntu22.04-rtx4080.txt         -> ubuntu22.04-rtx4080_super.txt
```

同时把卷一 Task 3 改为“本仓暂定 schema、外部 Topic 生产者”，把 Task 5 的训练基线补入 `gpu_pci_device_id: "10de:2702"`。批准修订设计中用于解释历史错误的原始型号文字保留，不参加旧命名扫描。

把 `AGENTS.md` 的长期边界改为：外部 Topic 数据和静态能力资料仍由外部项目拥有；上游未定义期间本仓按批准设计暂定提供同名 schema；不得与上游同名包共存，未来只能原子切换。工作副本中用户新增的 Ubuntu artifact 路径和三机说明原样保留。

- [ ] **Step 5: 以 UTF-8 读取并运行文档测试**

```bash
python3 - <<'PY'
from pathlib import Path

for path in Path("docs").rglob("*.md"):
    path.read_text(encoding="utf-8")
Path("AGENTS.md").read_text(encoding="utf-8")
Path("README.md").read_text(encoding="utf-8")
print("UTF-8 documentation: OK")
PY
python3 -m pytest -q tests/foundation/test_documentation_baseline.py
```

Expected: `UTF-8 documentation: OK` 且 pytest PASS。

- [ ] **Step 6: 只暂存 `AGENTS.md` 的两项批准修正并提交**

先暂存其他文档和测试。对 `AGENTS.md` 使用 index-only 补丁，只把 HEAD 中的 `RTX 4080` 改为 `RTX 4080 SUPER`，并写入暂定 schema 例外；不得暂存用户新增的 Ubuntu 下载目录和三机说明。提交前必须确认：

```bash
git diff --cached -- AGENTS.md
git diff -- AGENTS.md
git diff --cached --name-only
```

Expected: cached diff 对 `AGENTS.md` 只有设备型号和暂定 schema 例外两项变化；working-tree diff 仍保留用户其他修改；`.vscode/` 不在暂存区。

```bash
git commit -m "docs: align Ubuntu handoff baselines"
```

### Task 2: 以 TDD 创建暂定 `lunar_navigation_msgs` 并加固仓库边界

**Files:**
- Create: `ros2_ws/src/lunar_navigation_msgs/package.xml`
- Create: `ros2_ws/src/lunar_navigation_msgs/CMakeLists.txt`
- Create: `ros2_ws/src/lunar_navigation_msgs/msg/LocalizationStatus.msg`
- Create: `ros2_ws/src/lunar_navigation_msgs/msg/ScienceTargetRegion.msg`
- Create: `ros2_ws/src/lunar_navigation_msgs/msg/ExplorationTask.msg`
- Create: `tests/foundation/test_navigation_message_package.py`
- Modify: `tools/check_repository_boundaries.py`
- Modify: `tests/foundation/test_repository_boundaries.py`

**Interfaces:**
- Consumes: `std_msgs/Header`、`geometry_msgs/Polygon` 和外部输入基线。
- Produces: `lunar_navigation_msgs/msg/LocalizationStatus`、`ScienceTargetRegion`、`ExplorationTask`，包版本 `0.1.0`。

- [ ] **Step 1: 写三个消息逐字合同的失败测试**

```python
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ros2_ws/src/lunar_navigation_msgs"
EXPECTED = {
    "LocalizationStatus.msg": """uint8 UNKNOWN=0
uint8 VALID=1
uint8 DEGRADED=2
uint8 INVALID=3
uint8 RELOCALIZING=4
std_msgs/Header header
uint8 status
""",
    "ScienceTargetRegion.msg": """string region_id
string objective_id
geometry_msgs/Polygon boundary
float64 priority
""",
    "ExplorationTask.msg": """uint8 ACTIVE=1
uint8 PAUSED=2
uint8 CANCELED=3
std_msgs/Header header
string mission_id
uint64 revision
uint8 desired_state
float64 roi_min_x_m
float64 roi_min_y_m
float64 roi_max_x_m
float64 roi_max_y_m
lunar_navigation_msgs/ScienceTargetRegion[<=64] science_regions
""",
}


def test_provisional_message_sources_are_exact():
    actual = {
        path.name: path.read_text(encoding="utf-8")
        for path in (PACKAGE / "msg").glob("*.msg")
    }
    assert actual == EXPECTED


def test_package_declares_only_required_rosidl_dependencies():
    root = ET.parse(PACKAGE / "package.xml").getroot()
    names = [element.text for element in root]
    assert root.findtext("name") == "lunar_navigation_msgs"
    assert root.findtext("version") == "0.1.0"
    assert "rosidl_default_generators" in names
    assert "rosidl_default_runtime" in names
    assert "geometry_msgs" in names
    assert "std_msgs" in names
    assert root.findtext("member_of_group") == "rosidl_interface_packages"


def test_message_sources_match_the_authoritative_baseline():
    baseline = (ROOT / "docs/interfaces/external-input-baseline.md").read_text(encoding="utf-8")
    for source in EXPECTED.values():
        assert f"```text\n{source}```" in baseline


def test_no_same_name_upstream_is_pinned():
    assert (ROOT / "dependencies.repos").read_text(encoding="utf-8") == "repositories: {}\n"
```

- [ ] **Step 2: 写同名包和额外接口的边界失败测试**

```python
def test_rejects_second_lunar_navigation_msgs_package(tmp_path):
    canonical = tmp_path / "ros2_ws/src/lunar_navigation_msgs"
    duplicate = tmp_path / "vendor/lunar_navigation_msgs"
    canonical.mkdir(parents=True)
    duplicate.mkdir(parents=True)
    package_xml = "<package format='3'><name>lunar_navigation_msgs</name></package>"
    (canonical / "package.xml").write_text(package_xml, encoding="utf-8")
    (duplicate / "package.xml").write_text(package_xml, encoding="utf-8")
    assert any("duplicate lunar_navigation_msgs" in error for error in check_repository(tmp_path))


def test_rejects_unapproved_provisional_interface(tmp_path):
    package = tmp_path / "ros2_ws/src/lunar_navigation_msgs"
    (package / "action").mkdir(parents=True)
    (package / "action/Unexpected.action").write_text("string input\n---\nbool ok\n", encoding="utf-8")
    assert any("unapproved lunar_navigation_msgs interface" in error for error in check_repository(tmp_path))
```

- [ ] **Step 3: 运行测试确认包缺失且边界检查尚未实现**

```bash
python3 -m pytest -q \
  tests/foundation/test_navigation_message_package.py \
  tests/foundation/test_repository_boundaries.py
```

Expected: FAIL；消息目录不存在，且重复包/额外接口 fixture 未被拒绝。

- [ ] **Step 4: 创建最小 rosidl 接口包**

`CMakeLists.txt` 必须使用：

```cmake
cmake_minimum_required(VERSION 3.22)
project(lunar_navigation_msgs)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LocalizationStatus.msg"
  "msg/ScienceTargetRegion.msg"
  "msg/ExplorationTask.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
```

`package.xml` 只声明 `ament_cmake`、`rosidl_default_generators`、`rosidl_default_runtime`、`geometry_msgs`、`std_msgs` 和 `rosidl_interface_packages` 成员关系；三个 `.msg` 内容必须与 Step 1 完全一致。

- [ ] **Step 5: 扩展仓库边界检查器**

在 `tools/check_repository_boundaries.py` 中增加：

```python
PROVISIONAL_PACKAGE_PATH = Path("ros2_ws/src/lunar_navigation_msgs")
ALLOWED_PROVISIONAL_INTERFACES = {
    Path("msg/LocalizationStatus.msg"),
    Path("msg/ScienceTargetRegion.msg"),
    Path("msg/ExplorationTask.msg"),
}
```

扫描所有非忽略目录中的 `package.xml`，解析 `<name>`；第二个 `lunar_navigation_msgs` 或 canonical path 之外的唯一实现均报错。canonical 包下发现未批准的 `.msg`、`.srv`、`.action` 文件时，返回包含相对路径的 `unapproved lunar_navigation_msgs interface` 错误。缺失 canonical 包由静态消息测试报告，避免破坏检查器的独立临时仓 fixture。

- [ ] **Step 6: 运行静态合同和真实仓边界检查**

```bash
python3 -m pytest -q \
  tests/foundation/test_navigation_message_package.py \
  tests/foundation/test_repository_boundaries.py
python3 tools/check_repository_boundaries.py .
```

Expected: pytest PASS；真实仓输出 `repository boundaries: OK`。

- [ ] **Step 7: 提交暂定消息源码**

```bash
git add \
  ros2_ws/src/lunar_navigation_msgs \
  tools/check_repository_boundaries.py \
  tests/foundation/test_navigation_message_package.py \
  tests/foundation/test_repository_boundaries.py
git commit -m "feat: define provisional navigation interfaces"
```

### Task 3: 升级外部接口配置并验证现场消息来源

**Files:**
- Modify: `ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml`
- Modify: `tools/check_external_interfaces.py`
- Modify: `tests/foundation/test_external_interface_config.py`
- Create: `tests/ros/test_generated_navigation_interfaces.py`

**Interfaces:**
- Consumes: Task 2 的暂定消息包、`/opt/ros/humble` 标准包和本次仓库外 merge-install 前缀。
- Produces: `lunar-external-interfaces/v2` 配置；`check_interfaces(config, *, expected_lunar_navigation_prefix, run, ament_prefix_path, package_locations) -> list[str]`。

- [ ] **Step 1: 把配置期望改成 v2 并写来源失败测试**

配置期望新增：

```python
"schema_version": "lunar-external-interfaces/v2",
"interface_packages": {
    "lunar_navigation_msgs": {
        "schema_provider": "in_repository_provisional",
        "source_path": "ros2_ws/src/lunar_navigation_msgs",
        "upstream_status": "undefined",
        "replacement_policy": "atomic",
    }
},
```

`exploration_task.required_fields` 必须是：

```python
[
    "header", "mission_id", "revision", "desired_state",
    "roi_min_x_m", "roi_min_y_m", "roi_max_x_m", "roi_max_y_m",
    "science_regions",
]
```

保留现有 `write_config()` 和 `complete_valid_config()` helper，并新增下列可调用 fake；所有测试引用的名称都由同一测试文件定义：

```python
from dataclasses import dataclass
from pathlib import Path
import subprocess


VALID_LOCALIZATION_STATUS = """uint8 UNKNOWN=0
uint8 VALID=1
uint8 DEGRADED=2
uint8 INVALID=3
uint8 RELOCALIZING=4
std_msgs/Header header
uint8 status
"""
VALID_SCIENCE_TARGET_REGION = """string region_id
string objective_id
geometry_msgs/Polygon boundary
float64 priority
"""
VALID_EXPLORATION_TASK = """uint8 ACTIVE=1
uint8 PAUSED=2
uint8 CANCELED=3
std_msgs/Header header
string mission_id
uint64 revision
uint8 desired_state
float64 roi_min_x_m
float64 roi_min_y_m
float64 roi_max_x_m
float64 roi_max_y_m
lunar_navigation_msgs/ScienceTargetRegion[<=64] science_regions
"""


def field_definition(*names: str) -> str:
    return "".join(f"string {name}\n" for name in names)


@dataclass
class SourceAwareRosRunner:
    lunar_prefix: Path
    interface_outputs: dict[str, str]

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["ros2", "pkg", "prefix"]:
            package = command[3]
            prefix = self.lunar_prefix if package == "lunar_navigation_msgs" else Path("/opt/ros/humble")
            return subprocess.CompletedProcess(command, 0, stdout=f"{prefix}\n", stderr="")
        if command[:3] == ["ros2", "interface", "show"]:
            output = self.interface_outputs[command[3]]
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
        raise AssertionError(f"unexpected command: {command!r}")


def source_aware_ros_runner(lunar_prefix: Path) -> SourceAwareRosRunner:
    document = complete_valid_config()
    outputs = {
        topic["type"]: field_definition(*topic["required_fields"])
        for topic in document["topics"].values()
    }
    outputs["tf2_msgs/msg/TFMessage"] = field_definition("transforms")
    outputs.update({
        "lunar_navigation_msgs/msg/LocalizationStatus": VALID_LOCALIZATION_STATUS,
        "lunar_navigation_msgs/msg/ScienceTargetRegion": VALID_SCIENCE_TARGET_REGION,
        "lunar_navigation_msgs/msg/ExplorationTask": VALID_EXPLORATION_TASK,
    })
    return SourceAwareRosRunner(lunar_prefix.resolve(), outputs)


def make_package_provider(prefix: Path, package: str) -> Path:
    resolved = prefix.resolve()
    index = resolved / "share/ament_index/resource_index/packages"
    index.mkdir(parents=True)
    (index / package).write_text("", encoding="utf-8")
    return resolved
```

所有既有 `check_interfaces()` 调用都补入 `expected_lunar_navigation_prefix=tmp_path / "expected"`。只验证配置错误且断言 ROS 未调用的测试继续使用现有 `successful_ros_runner({}, calls)`；进入 ROS 查询的成功/失败测试改用 `make_package_provider()` 和 `source_aware_ros_runner()`，并显式传入对应 `ament_prefix_path`。这样配置短路、ROS 命令错误和新来源门槛分别保留独立断言。

新增三个独立行为测试：

```python
def test_rejects_provisional_package_from_wrong_prefix(tmp_path):
    other = make_package_provider(tmp_path / "other", "lunar_navigation_msgs")
    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", complete_valid_config()),
        expected_lunar_navigation_prefix=tmp_path / "expected",
        run=source_aware_ros_runner(other),
        ament_prefix_path=str(other),
    )
    assert any("unexpected package prefix" in error for error in errors)


def test_rejects_duplicate_ament_provider(tmp_path):
    expected = make_package_provider(tmp_path / "expected", "lunar_navigation_msgs")
    duplicate = make_package_provider(tmp_path / "duplicate", "lunar_navigation_msgs")
    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", complete_valid_config()),
        expected_lunar_navigation_prefix=expected,
        run=source_aware_ros_runner(expected),
        ament_prefix_path=f"{expected}:{duplicate}:/opt/ros/humble",
    )
    assert any("multiple ament providers" in error for error in errors)


def test_rejects_provisional_declaration_drift(tmp_path):
    expected = make_package_provider(tmp_path / "expected", "lunar_navigation_msgs")
    run = source_aware_ros_runner(lunar_prefix=expected)
    run.interface_outputs["lunar_navigation_msgs/msg/ExplorationTask"] = (
        VALID_EXPLORATION_TASK.replace("[<=64]", "[]")
    )
    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", complete_valid_config()),
        expected_lunar_navigation_prefix=expected,
        run=run,
        ament_prefix_path=f"{expected}:/opt/ros/humble",
    )
    assert any("declaration mismatch" in error for error in errors)
```

- [ ] **Step 2: 运行配置测试确认 v1 检查器失败**

```bash
python3 -m pytest -q tests/foundation/test_external_interface_config.py
```

Expected: FAIL；旧配置缺 `interface_packages`，旧函数不接受期望前缀，也不会拒绝重复 provider 或 bounded-sequence 漂移。

- [ ] **Step 3: 写入 v2 配置**

```yaml
schema_version: lunar-external-interfaces/v2
interface_packages:
  lunar_navigation_msgs:
    schema_provider: in_repository_provisional
    source_path: ros2_ws/src/lunar_navigation_msgs
    upstream_status: undefined
    replacement_policy: atomic
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
    required_fields: [header, mission_id, revision, desired_state, roi_min_x_m, roi_min_y_m, roi_max_x_m, roi_max_y_m, science_regions]
tf:
  topic: /tf
  type: tf2_msgs/msg/TFMessage
  chain: [map, odom, base_link]
required_grid_layers: [elevation, valid_mask, obstacle, obstacle_height, observation_age_s, observation_quality, elevation_variance, obstacle_variance, observation_count, forbidden]
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

- [ ] **Step 4: 实现精确声明和来源检查**

检查器必须定义三个 top-level 声明 tuple，与 Task 2 的文本逐行一致。`parse_top_level_declarations()` 只收集没有缩进、去除注释后的常量和字段行；三个暂定接口逐 tuple 比较，标准 ROS 接口继续只验证必需顶层字段。

来源规则：

1. `grid_map_msgs`、`nav_msgs`、`tf2_msgs` 的 `ros2 pkg prefix` 必须 resolve 为 `/opt/ros/humble`。
2. `lunar_navigation_msgs` 必须 resolve 为调用方传入的 `expected_lunar_navigation_prefix`。
3. 按 `AMENT_PREFIX_PATH` 顺序检查 `<prefix>/share/ament_index/resource_index/packages/lunar_navigation_msgs`；去重后的 provider 必须恰好一个并等于期望前缀。
4. CLI 新增必填参数 `--expected-lunar-navigation-prefix`；所有路径在比较前使用 `resolve()`。
5. 任何 ROS CLI 失败、字段缺失、声明漂移、错误来源或重复 provider 均产生确定性错误并返回非零。

- [ ] **Step 5: 运行配置与检查器单元测试**

```bash
python3 -m pytest -q tests/foundation/test_external_interface_config.py
```

Expected: PASS，包括错误前缀、重复 provider 和 `[<=64]` 漂移用例。

- [ ] **Step 6: 构建暂定消息和配置包到仓库外**

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
export LUNAR_VOLUME1_OUTPUT="$HOME/CodexDownloads/lunar_navigation/volume1-foundation"
mkdir -p \
  "$LUNAR_VOLUME1_OUTPUT/build" \
  "$LUNAR_VOLUME1_OUTPUT/install" \
  "$LUNAR_VOLUME1_OUTPUT/log"
colcon --log-base "$LUNAR_VOLUME1_OUTPUT/log" build \
  --merge-install \
  --base-paths ros2_ws/src \
  --packages-select lunar_navigation_msgs lunar_navigation_config \
  --build-base "$LUNAR_VOLUME1_OUTPUT/build" \
  --install-base "$LUNAR_VOLUME1_OUTPUT/install"
source "$LUNAR_VOLUME1_OUTPUT/install/setup.bash"
```

Expected: 两个 package 均 `Finished`，仓库根没有新增 `build/`、`install/` 或 `log/`。

- [ ] **Step 7: 写生成 Python 类型测试并运行**

```python
import pytest
from lunar_navigation_msgs.msg import ExplorationTask, LocalizationStatus, ScienceTargetRegion


def test_localization_constants_and_fields():
    assert LocalizationStatus.UNKNOWN == 0
    assert LocalizationStatus.VALID == 1
    assert LocalizationStatus.DEGRADED == 2
    assert LocalizationStatus.INVALID == 3
    assert LocalizationStatus.RELOCALIZING == 4
    assert LocalizationStatus.get_fields_and_field_types() == {
        "header": "std_msgs/Header",
        "status": "uint8",
    }


def test_exploration_state_zero_is_not_defined_and_regions_are_bounded():
    assert (ExplorationTask.ACTIVE, ExplorationTask.PAUSED, ExplorationTask.CANCELED) == (1, 2, 3)
    assert ExplorationTask.get_fields_and_field_types()["science_regions"] == (
        "sequence<lunar_navigation_msgs/ScienceTargetRegion, 64>"
    )
    with pytest.raises(AssertionError):
        ExplorationTask(science_regions=[ScienceTargetRegion() for _ in range(65)])
```

Run:

```bash
python3 -m pytest -q tests/ros/test_generated_navigation_interfaces.py
python3 tools/check_external_interfaces.py \
  --config ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml \
  --expected-lunar-navigation-prefix "$LUNAR_VOLUME1_OUTPUT/install"
ros2 interface show lunar_navigation_msgs/msg/LocalizationStatus
ros2 interface show lunar_navigation_msgs/msg/ScienceTargetRegion
ros2 interface show lunar_navigation_msgs/msg/ExplorationTask
```

Expected: pytest PASS；检查器输出四个包来源和 `external interfaces: OK`；暂定包来源为仓库外 install，三个系统包来源为 `/opt/ros/humble`。

- [ ] **Step 8: 提交 v2 配置和现场检查**

```bash
git add \
  ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml \
  tools/check_external_interfaces.py \
  tests/foundation/test_external_interface_config.py \
  tests/ros/test_generated_navigation_interfaces.py
git commit -m "fix: validate provisional interface source"
```

### Task 4: 以 TDD 创建内部 `lunar_planning_msgs`

**Files:**
- Create: `ros2_ws/src/lunar_planning_msgs/package.xml`
- Create: `ros2_ws/src/lunar_planning_msgs/CMakeLists.txt`
- Create: `ros2_ws/src/lunar_planning_msgs/msg/GoalRegion.msg`
- Create: `ros2_ws/src/lunar_planning_msgs/msg/HopSegment.msg`
- Create: `ros2_ws/src/lunar_planning_msgs/msg/MotionReference.msg`
- Create: `ros2_ws/src/lunar_planning_msgs/msg/PlannerDiagnostics.msg`
- Create: `ros2_ws/src/lunar_planning_msgs/action/PlanMotion.action`
- Create: `tests/foundation/test_planning_message_package.py`
- Create: `tests/ros/test_generated_planning_interfaces.py`

**Interfaces:**
- Consumes: `builtin_interfaces`、`geometry_msgs`、`nav_msgs`、`std_msgs`、`trajectory_msgs`。
- Produces: 卷二至卷四唯一允许依赖的内部规划接口和 `PlanMotion` Action。

- [ ] **Step 1: 写五个源接口逐字测试**

测试以字典逐字比较以下文件内容：

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

`PlanMotion.action`：

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

- [ ] **Step 2: 运行静态测试确认内部包缺失**

```bash
python3 -m pytest -q tests/foundation/test_planning_message_package.py
```

Expected: FAIL，`ros2_ws/src/lunar_planning_msgs` 尚不存在。

- [ ] **Step 3: 创建内部 rosidl 包**

`CMakeLists.txt` 必须精确枚举四个 `.msg` 和一个 `.action`：

```cmake
cmake_minimum_required(VERSION 3.22)
project(lunar_planning_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GoalRegion.msg"
  "msg/HopSegment.msg"
  "msg/MotionReference.msg"
  "msg/PlannerDiagnostics.msg"
  "action/PlanMotion.action"
  DEPENDENCIES builtin_interfaces geometry_msgs nav_msgs std_msgs trajectory_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
```

`package.xml` 使用版本 `0.1.0`，声明同一组依赖、`rosidl_default_generators`、`rosidl_default_runtime` 和 `rosidl_interface_packages` 成员关系。

- [ ] **Step 4: 运行静态测试并构建内部包**

```bash
python3 -m pytest -q tests/foundation/test_planning_message_package.py
source /opt/ros/humble/setup.bash
source "$LUNAR_VOLUME1_OUTPUT/install/setup.bash"
colcon --log-base "$LUNAR_VOLUME1_OUTPUT/log" build \
  --merge-install \
  --base-paths ros2_ws/src \
  --packages-select lunar_planning_msgs \
  --build-base "$LUNAR_VOLUME1_OUTPUT/build" \
  --install-base "$LUNAR_VOLUME1_OUTPUT/install"
source "$LUNAR_VOLUME1_OUTPUT/install/setup.bash"
```

Expected: 静态测试 PASS；`lunar_planning_msgs` 构建完成。

- [ ] **Step 5: 写并运行生成 Action 测试**

```python
from lunar_planning_msgs.action import PlanMotion
from lunar_planning_msgs.msg import GoalRegion, MotionReference


def test_plan_motion_goal_surface_and_result_constants():
    assert PlanMotion.Goal.get_fields_and_field_types() == {
        "request_id": "string",
        "mission_id": "string",
        "mission_revision": "uint64",
        "goal": "lunar_planning_msgs/GoalRegion",
        "replace_active_request": "boolean",
    }
    assert PlanMotion.Result.STALE_INPUT == 5
    assert PlanMotion.Result.CONTINUE_COMMITTED_HOP == 3


def test_goal_and_motion_reference_constants():
    assert (GoalRegion.POINT, GoalRegion.PLANAR_REGION) == (1, 2)
    assert (MotionReference.WHEELED, MotionReference.LEGGED, MotionReference.HOPPER) == (1, 2, 3)
```

Run:

```bash
python3 -m pytest -q tests/ros/test_generated_planning_interfaces.py
ros2 interface show lunar_planning_msgs/action/PlanMotion
```

Expected: PASS；Goal 中不含地图、Odometry、TF、能力、算法配置或模型字段。

- [ ] **Step 6: 提交内部接口**

```bash
git add \
  ros2_ws/src/lunar_planning_msgs \
  tests/foundation/test_planning_message_package.py \
  tests/ros/test_generated_planning_interfaces.py
git commit -m "feat: define internal planning action interfaces"
```

### Task 5: 建立平台基线和可审计环境指纹

**Files:**
- Create: `platform/train_amd64_rtx4080_super/baseline.yaml`
- Create: `platform/deploy_agx_orin_r36/baseline.yaml`
- Create: `tools/capture_environment.py`
- Create: `tests/foundation/test_environment_fingerprint.py`

**Interfaces:**
- Consumes: `/etc/os-release`、`uname`、`lscpu`、`free`、ROS 环境、编译器、`lspci`、`nvidia-smi`、`nvcc`、TensorRT/Jetson 命令输出。
- Produces: `lunar-platform-fingerprint/v1` JSON 和 `readiness.ready/errors`；CLI 无论匹配与否均先保存实际探测，再以退出码表达 readiness。

- [ ] **Step 1: 写解析、失败保真和 readiness 测试**

```python
def valid_training_fingerprint() -> dict[str, object]:
    unavailable = {
        "available": False,
        "command": "not-applicable-on-training-host",
        "returncode": 127,
        "error": "not available",
    }
    return {
        "schema_version": "lunar-platform-fingerprint/v1",
        "profile": "train_amd64_rtx4080_super",
        "captured_at_utc": "2026-08-02T12:00:00Z",
        "os": {
            "name": "Ubuntu",
            "version_id": "22.04",
            "pretty_name": "Ubuntu 22.04.5 LTS",
        },
        "architecture": "amd64",
        "kernel": "6.8.0-124-generic",
        "cpu": {
            "available": True,
            "model": "Intel(R) Core(TM) i7-14700KF",
            "logical_cpus": 28,
        },
        "memory": {"available": True, "total_bytes": 67223306240},
        "ros": {"available": True, "distro": "humble"},
        "python": {"available": True, "version": "3.10.12"},
        "gcc": {"available": True, "version": "11.4.0"},
        "cmake": {"available": True, "version": "3.22.1"},
        "gpu": {
            "available": True,
            "model": "NVIDIA GeForce RTX 4080 SUPER",
            "pci_bus_id": "00000000:01:00.0",
            "pci_device_id": "10de:2702",
            "memory_total_mib": 16376,
            "compute_capability": "8.9",
            "driver_version": "595.84",
        },
        "cuda": {"available": True, "release": "13.2", "compiler_version": "13.2.78"},
        "tensorrt": unavailable.copy(),
        "device_model": unavailable.copy(),
        "l4t": unavailable.copy(),
        "jetpack": unavailable.copy(),
        "power_mode": unavailable.copy(),
        "clocks": unavailable.copy(),
        "readiness": {"ready": True, "errors": []},
    }


def test_parse_nvidia_probe_normalizes_rtx4080_super():
    result = parse_nvidia_smi(
        "NVIDIA GeForce RTX 4080 SUPER, 00000000:01:00.0, "
        "0x270210DE, 16376, 8.9, 595.84\n"
    )
    assert result == {
        "available": True,
        "model": "NVIDIA GeForce RTX 4080 SUPER",
        "pci_bus_id": "00000000:01:00.0",
        "pci_device_id": "10de:2702",
        "memory_total_mib": 16376,
        "compute_capability": "8.9",
        "driver_version": "595.84",
    }


def test_failed_probe_preserves_command_error():
    result = unavailable_probe("nvidia-smi", returncode=9, stdout="", stderr="driver unavailable")
    assert result == {
        "available": False,
        "command": "nvidia-smi",
        "returncode": 9,
        "error": "driver unavailable",
    }


def test_training_readiness_requires_exact_gpu_and_cuda():
    baseline = load_baseline("train_amd64_rtx4080_super")
    fingerprint = valid_training_fingerprint()
    assert validate_fingerprint(fingerprint, baseline) == []
    fingerprint["gpu"]["model"] = "NVIDIA GeForce RTX 4080"
    assert validate_fingerprint(fingerprint, baseline) == [
        "gpu.model: expected 'NVIDIA GeForce RTX 4080 SUPER', got 'NVIDIA GeForce RTX 4080'"
    ]


def test_training_profile_allows_tensorrt_to_be_unavailable():
    fingerprint = valid_training_fingerprint()
    fingerprint["tensorrt"] = unavailable_probe("dpkg-query libnvinfer", 1, "", "not installed")
    assert validate_fingerprint(fingerprint, load_baseline("train_amd64_rtx4080_super")) == []
```

- [ ] **Step 2: 运行测试确认平台模块和基线缺失**

```bash
python3 -m pytest -q tests/foundation/test_environment_fingerprint.py
```

Expected: FAIL；`tools.capture_environment` 和两个 baseline 尚不存在。

- [ ] **Step 3: 写入两个基线**

训练基线：

```yaml
schema_version: lunar-platform-baseline/v1
profile: train_amd64_rtx4080_super
os: Ubuntu 22.04 LTS
architecture: amd64
ros_distro: humble
python: "3.10"
gpu_model: NVIDIA GeForce RTX 4080 SUPER
gpu_pci_device_id: "10de:2702"
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

- [ ] **Step 4: 实现纯解析函数和固定 JSON 结构**

`tools/capture_environment.py` 必须导出以下精确签名：

```text
parse_os_release(contents: str) -> dict[str, str]
parse_nvidia_smi(output: str) -> dict[str, object]
parse_nvcc(output: str) -> dict[str, object]
unavailable_probe(command: str, returncode: int, stdout: str, stderr: str) -> dict[str, object]
load_baseline(profile: str, baseline_root: Path = DEFAULT_BASELINE_ROOT) -> dict[str, object]
capture_environment(profile: str, *, run: CommandRunner = run_command) -> dict[str, object]
validate_fingerprint(document: Mapping[str, object], baseline: Mapping[str, object]) -> list[str]
write_fingerprint(document: Mapping[str, object], output: Path) -> None
```

`CommandRunner` 固定为接收命令序列并返回 `subprocess.CompletedProcess[str]` 的 callable；`DEFAULT_BASELINE_ROOT` 固定指向仓库根的 `platform/`。

JSON 顶层键固定为：

```text
schema_version, profile, captured_at_utc, os, architecture, kernel,
cpu, memory, ros, python, gcc, cmake, gpu, cuda, tensorrt,
device_model, l4t, jetpack, power_mode, clocks, readiness
```

主要值的类型固定如下，验证器按这些字段映射 baseline，不把展示字符串和结构化值混用：

```python
{
    "os": {
        "name": "Ubuntu",
        "version_id": "22.04",
        "pretty_name": "Ubuntu 22.04.5 LTS",
    },
    "architecture": "amd64",
    "kernel": "6.8.0-124-generic",
    "cpu": {"available": True, "model": "Intel(R) Core(TM) i7-14700KF", "logical_cpus": 28},
    "memory": {"available": True, "total_bytes": 67223306240},
    "ros": {"available": True, "distro": "humble"},
    "python": {"available": True, "version": "3.10.12"},
    "gcc": {"available": True, "version": "11.4.0"},
    "cmake": {"available": True, "version": "3.22.1"},
    "gpu": {
        "available": True,
        "model": "NVIDIA GeForce RTX 4080 SUPER",
        "pci_bus_id": "00000000:01:00.0",
        "pci_device_id": "10de:2702",
        "memory_total_mib": 16376,
        "compute_capability": "8.9",
        "driver_version": "595.84",
    },
    "cuda": {"available": True, "release": "13.2", "compiler_version": "13.2.78"},
    "readiness": {"ready": True, "errors": []},
}
```

`device_model`、`l4t` 和 `jetpack` 可用时使用 `{"available": true, "value": "解析后的版本或型号字符串"}`；`power_mode` 和 `clocks` 可用时把完整命令 stdout 放入同形 `value`。不可用时统一使用含 command、returncode 和 error 的结构。训练 OS 匹配使用 `os.name == "Ubuntu"` 和 `os.version_id == "22.04"`，因此补丁版本 `PRETTY_NAME` 不会误报。ROS 和 Python 分别读取 `ros.distro` 与 `python.version`；Python 只比较 major.minor。

所有子进程显式使用 `LC_ALL=C` 和 `LANG=C`，避免本地化输出破坏解析。`gpu` 使用 `nvidia-smi --query-gpu=name,pci.bus_id,pci.device_id,memory.total,compute_cap,driver_version --format=csv,noheader,nounits`；把 `0x270210DE` 规范化为 `10de:2702`。CUDA 使用 `/usr/local/cuda/bin/nvcc --version`。TensorRT 通过 `dpkg-query` 探测；Jetson 字段通过 `/etc/nv_tegra_release`、设备树 model、`nvpmodel -q` 和 `jetson_clocks --show` 探测。命令或文件不可用时使用统一 unavailable 结构并保留 stderr/stdout/退出码。

训练 readiness 必须精确匹配 baseline 的 OS、architecture、ROS、Python major.minor、GPU model 和 PCI ID，并要求 `gpu.available=true`、非空驱动版本和 `cuda.available=true`；TensorRT、L4T、JetPack、Jetson 功耗模式在训练 profile 可为 unavailable。AGX profile 必须精确匹配 architecture、device model、L4T、JetPack，并要求 CUDA 和 TensorRT 可用。

- [ ] **Step 5: 实现先写实际值、再返回 readiness 的 CLI**

CLI 参数：

```text
--profile {train_amd64_rtx4080_super,deploy_agx_orin_r36}
--output PATH
--baseline-root PATH   # 默认仓库 platform/
```

流程固定为：capture → validate → 填入 `readiness` → 以 UTF-8/LF、排序键和末尾换行写 JSON → 若 readiness false 返回 1。若输出 resolve 后位于 Git 根内，拒绝写入并返回 2。

- [ ] **Step 6: 运行单元测试和当前主机采集**

```bash
python3 -m pytest -q tests/foundation/test_environment_fingerprint.py
source /opt/ros/humble/setup.bash
mkdir -p "$LUNAR_VOLUME1_OUTPUT/fingerprints"
python3 tools/capture_environment.py \
  --profile train_amd64_rtx4080_super \
  --output "$LUNAR_VOLUME1_OUTPUT/fingerprints/train_amd64_rtx4080_super.json"
python3 -m json.tool \
  "$LUNAR_VOLUME1_OUTPUT/fingerprints/train_amd64_rtx4080_super.json" >/dev/null
```

Expected: pytest PASS；CLI exit 0；JSON 记录当前内核、CPU、内存、NVIDIA 595.84、RTX 4080 SUPER、PCI ID `10de:2702` 和 CUDA 13.2；`readiness.ready=true`。TensorRT 未安装时如实为 unavailable，但不阻塞训练 profile 基线。

- [ ] **Step 7: 提交基线和指纹工具，不提交现场 JSON**

```bash
git add \
  platform/train_amd64_rtx4080_super/baseline.yaml \
  platform/deploy_agx_orin_r36/baseline.yaml \
  tools/capture_environment.py \
  tests/foundation/test_environment_fingerprint.py
git commit -m "build: define Ubuntu and AGX platform baselines"
```

### Task 6: 运行卷一 Task 3–5 总门槛并记录实施状态

**Files:**
- Modify: `docs/superpowers/specs/2026-08-02-ubuntu-handoff-baseline-amendment-design.md`

**Interfaces:**
- Consumes: Tasks 1–5 的所有源码、仓库外 ROS overlay 和当前主机指纹。
- Produces: 可复验的 Ubuntu Task 3–5 完成证据；不扩大到运行时语义适配器或训练 smoke。

- [ ] **Step 1: 运行全部静态基础测试和仓库边界检查**

```bash
source /opt/ros/humble/setup.bash
source "$LUNAR_VOLUME1_OUTPUT/install/setup.bash"
python3 -m pytest -q tests/foundation
python3 tools/check_repository_boundaries.py .
```

Expected: foundation 测试零失败；`repository boundaries: OK`。

- [ ] **Step 2: 从干净的仓库外构建根重建三个 ROS 包**

为避免删除未知 artifact，使用 `mktemp` 创建新的明确子目录，不清理旧目录：

```bash
export LUNAR_VOLUME1_VERIFY
LUNAR_VOLUME1_VERIFY="$(mktemp -d "$LUNAR_VOLUME1_OUTPUT/verify-task3-5.XXXXXX")"
mkdir -p \
  "$LUNAR_VOLUME1_VERIFY/build" \
  "$LUNAR_VOLUME1_VERIFY/install" \
  "$LUNAR_VOLUME1_VERIFY/log"
source /opt/ros/humble/setup.bash
colcon --log-base "$LUNAR_VOLUME1_VERIFY/log" build \
  --merge-install \
  --base-paths ros2_ws/src \
  --packages-select lunar_navigation_msgs lunar_navigation_config lunar_planning_msgs \
  --build-base "$LUNAR_VOLUME1_VERIFY/build" \
  --install-base "$LUNAR_VOLUME1_VERIFY/install"
source "$LUNAR_VOLUME1_VERIFY/install/setup.bash"
```

Expected: 三个 package 均完成，退出码 0。

- [ ] **Step 3: 运行生成类型、现场来源和 colcon 测试**

```bash
python3 -m pytest -q tests/ros/test_generated_navigation_interfaces.py
python3 -m pytest -q tests/ros/test_generated_planning_interfaces.py
python3 tools/check_external_interfaces.py \
  --config ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml \
  --expected-lunar-navigation-prefix "$LUNAR_VOLUME1_VERIFY/install"
colcon --log-base "$LUNAR_VOLUME1_VERIFY/log" test \
  --merge-install \
  --base-paths ros2_ws/src \
  --packages-select lunar_navigation_msgs lunar_navigation_config lunar_planning_msgs \
  --build-base "$LUNAR_VOLUME1_VERIFY/build" \
  --install-base "$LUNAR_VOLUME1_VERIFY/install" \
  --test-result-base "$LUNAR_VOLUME1_VERIFY/test-results" \
  --return-code-on-test-failure
colcon test-result \
  --test-result-base "$LUNAR_VOLUME1_VERIFY/test-results" \
  --verbose
```

Expected: 两组生成类型测试 PASS；来源检查 `OK`；`colcon test-result` 无失败。

- [ ] **Step 4: 重新采集最终训练主机指纹**

```bash
python3 tools/capture_environment.py \
  --profile train_amd64_rtx4080_super \
  --output "$LUNAR_VOLUME1_VERIFY/train_amd64_rtx4080_super.json"
```

Expected: exit 0 且 `readiness.ready=true`；现场 JSON 留在仓库外。

- [ ] **Step 5: 更新设计实施状态并说明历史指纹边界**

把设计文档状态改为“已实施”，新增实施结果段，明确：三个暂定接口和 v2 来源门槛已落地；RTX 4080 SUPER 驱动维护在指纹工具实现前完成；没有伪造安装前 JSON；最终指纹位于调用方指定的仓库外输出根。

- [ ] **Step 6: 复验中文 UTF-8、完整 diff 和用户改动保护**

```bash
python3 - <<'PY'
from pathlib import Path

for path in [Path("AGENTS.md"), *Path("docs").rglob("*.md")]:
    path.read_text(encoding="utf-8")
print("UTF-8 documentation: OK")
PY
git status --short
git diff --check
git diff -- AGENTS.md
```

Expected: UTF-8 读取成功，`git diff --check` 无输出；`AGENTS.md` 仍只呈现用户未提交内容相对最新 HEAD 的差异；`.vscode/` 保持未跟踪且未暂存。

- [ ] **Step 7: 提交实施状态记录**

```bash
git add docs/superpowers/specs/2026-08-02-ubuntu-handoff-baseline-amendment-design.md
git commit -m "docs: record Ubuntu handoff verification"
```

最终再运行一次：

```bash
python3 -m pytest -q tests/foundation
python3 tools/check_repository_boundaries.py .
git status --short --branch
```

Expected: 全部通过；分支只保留执行前已存在的用户工作区改动和仓库外 artifact。
