# Dual-Target `luna` Runtime Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从当前唯一有效算法基线构建两份可原生编译的源码运行时包，并提供安全、短命令的 `luna` 操作入口。

**Architecture:** 开发仓新增一个纯 Python 的 `deployment/luna_runtime` 管理层，负责 profile、配置、可复现 bundle、构建目录和受管 planner 进程；它不把产物写回源码树。`lunar_planner_ros` 保持唯一规划服务，但将固定 Topic/Action 名称参数化并由 `luna` 生成的 ROS 参数文件驱动。模型包在首版仅做不可变 artifact 的校验、存储和原子指针切换；因为当前 ROS 服务没有 V4 候选/观测到模型的调用链，非 `fallback` 策略模式必须 fail-closed，而不能假装模型已参与规划。

**Tech Stack:** Python 3.10 标准库与 PyYAML、pytest、ROS 2 Humble、C++20、ament/colcon、ONNX Runtime（amd64 模型包加载冒烟）、TensorRT `trtexec`（Orin 本机 engine 冒烟）。

**Spec:** `docs/superpowers/specs/2026-08-20-dual-target-runtime-deployment-design.md`

## Global Constraints

- 当前发布基线是 manifest 固定的单一 `source_commit`；bundle 使用白名单，绝不扫描或携带旧分支、实验、迁移、训练、测试、Git 历史或 artifact。
- 默认 profile 仅支持 Ubuntu 22.04 + ROS 2 Humble 的 amd64，以及 Ubuntu 22.04 + ROS 2 Humble 的 Jetson AGX Orin R36 aarch64；两者必须在目标机原生构建。
- `build/`、`install/`、`log/`、PID、模型和 TensorRT engine 只能写入 `LUNA_HOME`，默认位于用户 XDG 目录，绝不写入源码包。
- 不提供关闭安全投影、足迹/净空、精确端点、路径认证、输入新鲜度或输入一致性校验的配置。
- 默认 `policy.mode: fallback`；训练 checkpoint、优化器、缓存和 `latest.pt` 绝不进入部署包或被轮询。
- 当前 Planner ROS 没有模型候选选择入口；模型 artifact 可以安装和激活为“已验证待接入”，但 `luna start` 必须拒绝非 `fallback` 模式并返回 `POLICY_RUNTIME_UNBOUND`。
- 全局地图处理和路径跟踪只建立扩展注册边界；首版不得猜测执行器低级控制 Topic，也不得修改 `PlanMotion`、`MotionReference` 或外部 v5 消息。
- `lunar_nav2_adapter` 是 WHEELED 可选包，默认不构建；`lunar_planner_training_bridge` 与 `training/` 永远不属于运行时包。
- 测试、设计文档和运行 artifact 可以存在开发仓，但 bundle 根目录仅交付 `README.md` 与 `COMMANDS.md` 两份操作者文档。

## File Structure

```text
deployment/
  __init__.py
  runtime_source_allowlist.yaml          # 唯一可打包源码路径与排除规则
  config/
    runtime.schema.json                  # runtime.yaml 的 JSON Schema
    runtime.default.yaml                 # fallback 默认模板
  docs/
    README.runtime.md                    # bundle 根 README 模板
    COMMANDS.runtime.md                  # bundle 根命令参考模板
  profiles/
    ubuntu22-humble-amd64.yaml
    jetson-orin-r36.yaml
  luna_runtime/
    __init__.py
    cli.py                               # luna 子命令解析与稳定退出码
    config.py                            # profile/config/schema/参数渲染
    host.py                              # 可注入的目标机身份检测
    state.py                             # XDG/LUNA_HOME 目录、JSON 状态和原子写入
    commands.py                          # init/doctor/config/extension 编排
    build.py                             # package-select colcon 计划与执行
    process.py                           # planner launch、lifecycle、PID/status/log
    bundle.py                            # allowlist 的确定性源码包构建
    model_store.py                       # 模型包安装、激活、回退和本机 probe 编排
scripts/luna                             # 开发仓及 bundle 根使用的薄入口
ros2_ws/src/lunar_policy_runtime/
  package.xml
  setup.py
  setup.cfg
  resource/lunar_policy_runtime
  lunar_policy_runtime/
    __init__.py
    manifest.py                          # model-manifest 解析与契约/hash 校验
    probe.py                             # ORT 或 trtexec 的可注入加载 probe
  test/                                  # 开发仓测试；bundle 不复制
ros2_ws/src/lunar_planner_ros/
  launch/lunar_planner.launch.py
  src/plan_motion_server.cpp
  test/plan_motion_server_test.cpp
  CMakeLists.txt
  package.xml
tests/deployment/
  test_config.py
  test_cli.py
  test_build.py
  test_process.py
  test_bundle.py
  test_model_store.py
  test_runtime_docs.py
```

`model_contract/` 继续是 ObservationContractV4 与 ActionContractV2 的唯一 Python 权威；bundle 仅复制其运行时模块、`package.xml`、`setup.py`、`setup.cfg` 与 resource，不复制其测试目录。

---

### Task 1: 冻结 profile、运行时配置与 bundle 白名单

**Files:**
- Create: `deployment/__init__.py`
- Create: `deployment/luna_runtime/__init__.py`
- Create: `deployment/luna_runtime/config.py`
- Create: `deployment/luna_runtime/host.py`
- Create: `deployment/config/runtime.schema.json`
- Create: `deployment/config/runtime.default.yaml`
- Create: `deployment/profiles/ubuntu22-humble-amd64.yaml`
- Create: `deployment/profiles/jetson-orin-r36.yaml`
- Create: `deployment/runtime_source_allowlist.yaml`
- Create: `tests/deployment/test_config.py`

**Interfaces:**
- Produces `TargetProfile`, `HostFacts`, `RuntimeConfig`, `load_profile(profile_id)`, `load_runtime_config(path)`, `validate_host(profile, facts)`, `render_planner_params(config, output_path)` and `load_allowlist(path)`.
- Consumes the existing `lunar-external-interfaces/v5` names in `ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml`; it does not redefine message fields.
- `RuntimeConfig.interfaces` contains exactly `map_global`, `map_local`, `odometry`, `localization_status`, `tf`, `exploration_task`, `motion_feedback`, `plan_motion`, `diagnostics`, `certified_route_markers`, and `provisional_route_markers`.

- [ ] **Step 1: Write the failing profile/config tests**

```python
from deployment.luna_runtime.config import ConfigError, load_runtime_config, load_profile, validate_host
from deployment.luna_runtime.host import HostFacts


def test_orin_profile_rejects_amd64_host(tmp_path):
    profile = load_profile("jetson-orin-r36")
    facts = HostFacts(os_id="ubuntu", os_version="22.04", architecture="x86_64", ros_distro="humble", l4t="R36.0.0")
    assert validate_host(profile, facts) == ("ARCHITECTURE_MISMATCH",)


def test_runtime_config_rejects_missing_required_interface(tmp_path):
    config = tmp_path / "runtime.yaml"
    config.write_text("""
profile: ubuntu22-humble-amd64
interfaces:
  map_local: /environment/map_local
  odometry: /localization/odometry
  localization_status: /localization/status
  tf: /tf
  exploration_task: /mission/exploration_task
  motion_feedback: /execution/motion_feedback
  plan_motion: /plan_motion
  diagnostics: /diagnostics
  certified_route_markers: /planning/certified_route_markers
  provisional_route_markers: /planning/provisional_route_markers
capabilities: {platform_file: /tmp/platform.yaml, observation_file: /tmp/observation.yaml}
planner: {snapshot_policy: {global_map_max_age: 1.0, local_map_max_age: 1.0, odometry_max_age: 1.0, localization_status_max_age: 1.0, tf_max_age: 1.0, max_pairwise_skew: 1.0}}
policy: {mode: fallback, model_id: null}
extensions: {map_pipeline: false, path_tracking: false}
runtime: {log_level: INFO}
""", encoding="utf-8")
    with pytest.raises(ConfigError, match="interfaces.map_global"):
        load_runtime_config(config)
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest -q tests/deployment/test_config.py`

Expected: FAIL because `deployment.luna_runtime` and its config/profile interfaces do not exist.

- [ ] **Step 3: Implement immutable profile and config parsing**

```python
@dataclass(frozen=True)
class HostFacts:
    os_id: str
    os_version: str
    architecture: str
    ros_distro: str
    l4t: str | None = None


@dataclass(frozen=True)
class TargetProfile:
    id: str
    os_id: str
    os_version: str
    architecture: str
    ros_distro: str
    l4t_prefix: str | None
    policy_probe: Literal["onnxruntime", "trtexec"]


def render_planner_params(config: RuntimeConfig, output_path: Path) -> None:
    payload = {"/lunar_planner": {"ros__parameters": config.planner_ros_parameters()}}
    atomic_write_yaml(output_path, payload)
```

Use `yaml.safe_load`; require all six top-level blocks from the spec; require absolute topic/action names; reject duplicate interface names, retired global/hopper truncation keys, false safety switches, profile mismatch, unknown keys and `policy.mode` values other than `fallback`, `onnx`, `tensorrt`. The default template contains current v5 topics, required positive snapshot durations, capability file paths as explicit empty strings, `planner.enable_nav2_adapter: false`, both extensions disabled, and `policy.mode: fallback`. `onnx` and `tensorrt` are syntactically valid stored-model modes; Task 5 is the only place that refuses them at process start until a ROS policy adapter exists.

The allowlist uses explicit roots, not broad repository matching:

```yaml
schema_version: lunar-runtime-allowlist/v1
runtime_files: [scripts/luna, deployment]
runtime_roots:
  - model_contract
  - ros2_ws/src/lunar_navigation_msgs
  - ros2_ws/src/lunar_planning_msgs
  - ros2_ws/src/lunar_navigation_config
  - ros2_ws/src/lunar_planner_core
  - ros2_ws/src/lunar_planner_ros
  - ros2_ws/src/lunar_policy_runtime
optional_roots:
  nav2_adapter: ros2_ws/src/lunar_nav2_adapter
excluded_path_components: [test, tests, docs, build, install, log, cache, __pycache__]
```

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest -q tests/deployment/test_config.py`

Expected: PASS; cover amd64/Orin identity checks, missing/duplicate topic rejection, nonpositive timing rejection, false safety-toggle rejection, valid fallback config, planner ROS YAML rendering and exact allowlist roots.

- [ ] **Step 5: Commit**

```bash
git add deployment tests/deployment/test_config.py
git commit -m "feat(deploy): define luna runtime profiles and config"
```

### Task 2: 实现 `luna init`、`doctor` 与配置检查的安全状态层

**Files:**
- Create: `deployment/luna_runtime/state.py`
- Create: `deployment/luna_runtime/commands.py`
- Create: `deployment/luna_runtime/cli.py`
- Create: `scripts/luna`
- Create: `tests/deployment/test_cli.py`

**Interfaces:**
- Consumes `RuntimeConfig`, `TargetProfile`, `HostFacts` and the default config from Task 1.
- Produces `RuntimePaths`, `RuntimeState`, `atomic_write_json(path, value)`, `init_runtime(...)`, `doctor_runtime(...)`, `check_runtime_config(...)`, and CLI exit codes `0` (success), `2` (operator/config error), `3` (host/dependency mismatch), `4` (safe runtime refusal).
- The first positional CLI token is one of `init`, `doctor`, `config`, `build`, `start`, `stop`, `status`, `logs`, `model`, `extension`, or `bundle`.

- [ ] **Step 1: Write the failing command/state tests**

```python
def test_init_writes_only_xdg_paths_and_never_overwrites_existing_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    result = run_cli(["init", "--profile", "ubuntu22-humble-amd64"], facts=AMD64_FACTS)
    assert result.exit_code == 0
    assert (tmp_path / "home/.config/luna/runtime.yaml").exists()
    assert not (repo_root / "build").exists()
    assert run_cli(["init", "--profile", "ubuntu22-humble-amd64"], facts=AMD64_FACTS).exit_code == 2


def test_doctor_reports_host_mismatch_without_running_build(tmp_path):
    result = run_cli(["doctor", "--config", str(orin_config)], facts=AMD64_FACTS)
    assert result.exit_code == 3
    assert result.json["reasons"] == ["ARCHITECTURE_MISMATCH"]
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest -q tests/deployment/test_cli.py`

Expected: FAIL because the CLI/state APIs do not exist.

- [ ] **Step 3: Implement paths, atomic state and thin command entry**

```python
def resolve_runtime_paths(release_id: str, environ: Mapping[str, str]) -> RuntimePaths:
    override = environ.get("LUNA_HOME")
    if override:
        root = Path(override).expanduser().resolve()
        return RuntimePaths(config=root / "config/runtime.yaml", data=root / "state" / release_id)
    return RuntimePaths(
        config=Path.home() / ".config/luna/runtime.yaml",
        data=Path.home() / ".local/share/luna" / release_id,
    )


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
```

`init` creates a fallback-only config and `state.json`, then optionally creates `~/.local/bin/luna` only when `--install-command` is given and `shutil.which("luna")` is empty or resolves to the same package-local executable. It never overwrites a different command. `doctor` emits a stable JSON object with `profile`, `host`, `config`, `dependencies`, `reasons` and no build side effect. `config check` parses and validates only; it does not source ROS, run a gate or launch a process. The executable `scripts/luna` contains only:

```python
#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
module_root = root if (root / "luna_runtime").is_dir() else root.parent / "deployment"
sys.path.insert(0, str(module_root))
from luna_runtime.cli import main

raise SystemExit(main())
```

and has executable mode set with `chmod 755 scripts/luna`.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest -q tests/deployment/test_cli.py tests/deployment/test_config.py`

Expected: PASS; test `LUNA_HOME` override, malformed state recovery, collision refusal, JSON diagnostics and no source-tree writes.

- [ ] **Step 5: Commit**

```bash
git add deployment scripts/luna tests/deployment/test_cli.py
git commit -m "feat(deploy): add luna initialization and diagnostics"
```

### Task 3: 用白名单构建两个确定性源码运行时包

**Files:**
- Create: `deployment/luna_runtime/bundle.py`
- Create: `deployment/docs/README.runtime.md`
- Create: `deployment/docs/COMMANDS.runtime.md`
- Create: `tests/deployment/test_bundle.py`
- Create: `tests/deployment/test_runtime_docs.py`
- Modify: `deployment/luna_runtime/cli.py`

**Interfaces:**
- Consumes Task 1 allowlist/profile and a clean Git `HEAD` file tree.
- Produces `BundleRequest`, `BundleResult`, `collect_bundle_files(repo_root, revision, target)`, `build_bundle(request)`, `<bundle>/release-manifest.json`, bundle-root `luna` file and `luna_runtime/` directory.
- `luna bundle --target {ubuntu22-humble-amd64,jetson-orin-r36} --output <absolute-dir>` creates exactly one `.tar.gz` without changing the Git worktree.

- [ ] **Step 1: Write the failing bundle and documentation tests**

```python
def test_bundle_has_only_allowlisted_current_runtime_source(tmp_path):
    result = build_bundle(BundleRequest(repo_root=repo_root, revision="HEAD", target="ubuntu22-humble-amd64", output_dir=tmp_path))
    names = archive_names(result.archive)
    assert "lunar-runtime-ubuntu22-humble-amd64-src/luna" in names
    assert "lunar-runtime-ubuntu22-humble-amd64-src/luna_runtime/cli.py" in names
    assert all("/training/" not in name and "/test/" not in name and "/tests/" not in name for name in names)
    assert "lunar-runtime-ubuntu22-humble-amd64-src/ros2_ws/src/lunar_planner_training_bridge/" not in "\n".join(names)


def test_manifests_share_source_commit_but_not_target_profile(tmp_path):
    amd = read_manifest(build_bundle(BundleRequest(repo_root, "HEAD", "ubuntu22-humble-amd64", tmp_path)).archive)
    orin = read_manifest(build_bundle(BundleRequest(repo_root, "HEAD", "jetson-orin-r36", tmp_path)).archive)
    assert amd["source_commit"] == orin["source_commit"]
    assert amd["target_profile"] != orin["target_profile"]
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest -q tests/deployment/test_bundle.py tests/deployment/test_runtime_docs.py`

Expected: FAIL because no deterministic bundle builder or deployment-document templates exist.

- [ ] **Step 3: Implement commit-addressed allowlist packaging**

```python
def collect_bundle_files(repo_root: Path, revision: str, target: TargetProfile) -> tuple[BundleFile, ...]:
    tracked = git_lines(repo_root, ["ls-tree", "-r", "--name-only", revision])
    allowed = tuple(path for path in tracked if allowlisted(path, target))
    reject_dirty_allowlisted_paths(repo_root, allowed)
    return tuple(BundleFile(source_path=path, content=git_show_bytes(repo_root, revision, path)) for path in allowed)


def build_bundle(request: BundleRequest) -> BundleResult:
    files = collect_bundle_files(request.repo_root, request.revision, load_profile(request.target))
    manifest = build_release_manifest(request, files)
    return write_deterministic_tar_gz(files + bundle_generated_files(manifest), request.output_dir)
```

Read source only with `git ls-tree`/`git show <revision>:<path>` so untracked files, local build trees and stale checkout paths cannot enter. Reject dirty changes touching an allowlisted path, but allow unrelated user edits to remain untouched. Filter every path component listed in the allowlist, exclude all test directories, and add only the two rendered operator docs at bundle root. Copy `scripts/luna` as root `luna`, and copy `deployment/luna_runtime` as root `luna_runtime`; this preserves the invariant that `./luna` is executable without colliding with a directory of the same name.

Use fixed tar metadata (`uid=0`, `gid=0`, empty owner names, `mtime=0`, sorted paths) and a SHA-256 over sorted `path + NUL + sha256(content)` entries. The manifest contains the exact Git commit, source-list hash, runtime schema version, profile identity, interface contract version, observation/action contract versions, optional extension list and model mode `fallback`.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest -q tests/deployment/test_bundle.py tests/deployment/test_runtime_docs.py`

Expected: PASS; assert deterministic archive hash on two builds, correct two-profile manifest identity, no historical branch/test/training/bridge/artifact paths, executable root command, and documentation contains every public `luna` command plus v5 input/output summary.

- [ ] **Step 5: Commit**

```bash
git add deployment tests/deployment/test_bundle.py tests/deployment/test_runtime_docs.py
git commit -m "feat(deploy): bundle allowlisted luna runtime sources"
```

### Task 4: 让 Planner ROS 的所有外部 endpoint 可由运行时配置参数化

**Files:**
- Modify: `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp`
- Create: `ros2_ws/src/lunar_planner_ros/launch/lunar_planner.launch.py`
- Modify: `ros2_ws/src/lunar_planner_ros/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_planner_ros/package.xml`

**Interfaces:**
- Produces eleven read-once ROS parameters: `interfaces.map_global`, `interfaces.map_local`, `interfaces.odometry`, `interfaces.localization_status`, `interfaces.tf`, `interfaces.exploration_task`, `interfaces.motion_feedback`, `interfaces.plan_motion`, `interfaces.diagnostics`, `interfaces.certified_route_markers`, `interfaces.provisional_route_markers`.
- Their defaults exactly match `lunar-external-interfaces/v5` and the existing marker names.
- `lunar_planner.launch.py` accepts the required launch argument `params_file` and launches only `lunar_planner_node`; lifecycle configure/activate remains owned by `luna start`.

- [ ] **Step 1: Write the failing ROS endpoint override test**

```cpp
TEST_F(PlanMotionServerTest, UsesConfiguredExternalInterfaceNames) {
  auto options = ValidOptions();
  options.append_parameter_override("interfaces.map_global", "/demo/global");
  options.append_parameter_override("interfaces.plan_motion", "/demo/plan_motion");
  options.append_parameter_override("interfaces.diagnostics", "/demo/diagnostics");
  RunningSystem system{DefaultDependencies(), true, std::move(options)};
  system.PublishInputsOn("/demo/global", "/demo/local", "/demo/odometry", "/demo/status", "/demo/tf", "/demo/task", "/demo/feedback");
  EXPECT_TRUE(system.WaitForActionServer("/demo/plan_motion"));
  EXPECT_TRUE(system.SawDiagnosticOn("/demo/diagnostics"));
}
```

Also add cases rejecting an empty name, a relative name and duplicate `interfaces.map_global == interfaces.map_local` during configure.

- [ ] **Step 2: Verify RED**

Run: `source /opt/ros/humble/setup.bash && colcon test --base-paths ros2_ws/src --packages-select lunar_planner_ros --ctest-args -R lunar_planner_ros_plan_motion_server_test --output-on-failure`

Expected: FAIL because the current server hard-codes `/environment/map_global`, `/plan_motion` and all other endpoints.

- [ ] **Step 3: Implement one immutable interface-name object**

```cpp
struct InterfaceNames final {
  std::string map_global;
  std::string map_local;
  std::string odometry;
  std::string localization_status;
  std::string tf;
  std::string exploration_task;
  std::string motion_feedback;
  std::string plan_motion;
  std::string diagnostics;
  std::string certified_route_markers;
  std::string provisional_route_markers;
};

[[nodiscard]] InterfaceNames ReadInterfaceNames() const;
```

Call `DeclareParameters()` before creating any publisher or action server; then read and validate the object once. Require every name to be nonempty, absolute and globally unique. Use only that object in publisher, action-server and subscription construction. Do not permit dynamic parameter mutation to reroute an active safety service. Install the launch directory and add `launch_ros` as an execution dependency.

- [ ] **Step 4: Verify GREEN**

Run: `source /opt/ros/humble/setup.bash && colcon test --base-paths ros2_ws/src --packages-select lunar_planner_ros --ctest-args -R lunar_planner_ros_plan_motion_server_test --output-on-failure`

Expected: PASS; existing default-endpoint tests remain green, the override test uses only renamed endpoints, invalid configurations fail before subscriptions are created, and lifecycle behavior is unchanged.

- [ ] **Step 5: Commit**

```bash
git add ros2_ws/src/lunar_planner_ros
git commit -m "feat(ros): configure luna runtime interface endpoints"
```

### Task 5: 在源码树外构建并受管启动生命周期规划服务

**Files:**
- Create: `deployment/luna_runtime/build.py`
- Create: `deployment/luna_runtime/process.py`
- Modify: `deployment/luna_runtime/commands.py`
- Modify: `deployment/luna_runtime/cli.py`
- Create: `tests/deployment/test_build.py`
- Create: `tests/deployment/test_process.py`

**Interfaces:**
- Consumes `RuntimeConfig`, `RuntimePaths`, profile and generated `planner-params.yaml`.
- Produces `BuildPlan`, `build_runtime(plan, runner)`, `start_runtime(config, paths, runner)`, `stop_runtime(paths)`, `read_runtime_status(paths)`, and `tail_log(paths, follow)`.
- Builds the exact required packages `lunar_navigation_msgs`, `lunar_planning_msgs`, `lunar_navigation_config`, `lunar_model_contract`, `lunar_planner_core`, `lunar_planner_ros`, `lunar_policy_runtime`; adds `lunar_nav2_adapter` only when enabled.

- [ ] **Step 1: Write failing build/process tests with a fake command runner**

```python
def test_default_build_selects_current_runtime_packages_and_never_training_bridge(tmp_path):
    plan = make_build_plan(valid_config, runtime_paths(tmp_path))
    assert "lunar_planner_training_bridge" not in plan.packages
    assert plan.packages[-1] == "lunar_policy_runtime"
    assert plan.build_base == runtime_paths(tmp_path).data / "build"


def test_start_configures_then_activates_lifecycle_and_records_verified_pid(tmp_path):
    runner = FakeRunner.lifecycle_sequence(["unconfigured", "inactive", "active"])
    state = start_runtime(valid_config, runtime_paths(tmp_path), runner)
    assert runner.commands[-2:] == [("ros2", "lifecycle", "set", "/lunar_planner", "configure"), ("ros2", "lifecycle", "set", "/lunar_planner", "activate")]
    assert state["lifecycle_state"] == "active"
```

Also test failed configure kills only the process started by this runtime, stale PID fingerprints do not signal an unrelated process, `stop` is idempotent, and a nonfallback policy config exits with `POLICY_RUNTIME_UNBOUND` before launch.

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest -q tests/deployment/test_build.py tests/deployment/test_process.py`

Expected: FAIL because no build/process interfaces exist.

- [ ] **Step 3: Implement package-select build and lifecycle orchestration**

```python
def make_build_plan(config: RuntimeConfig, paths: RuntimePaths) -> BuildPlan:
    packages = REQUIRED_PACKAGES + (("lunar_nav2_adapter",) if config.planner.enable_nav2_adapter else ())
    return BuildPlan(
        base_paths=(config.source_root / "ros2_ws/src", config.source_root / "model_contract"),
        packages=packages,
        build_base=paths.data / "build",
        install_base=paths.data / "install",
        log_base=paths.data / "log",
    )


def start_runtime(config: RuntimeConfig, paths: RuntimePaths, runner: CommandRunner) -> RuntimeState:
    if config.policy.mode != "fallback":
        raise RuntimeRefusal("POLICY_RUNTIME_UNBOUND")
    render_planner_params(config, paths.data / "generated/planner-params.yaml")
    child = runner.popen(launch_command(paths))
    wait_for_lifecycle(runner, "/lunar_planner", "unconfigured")
    runner.run(("ros2", "lifecycle", "set", "/lunar_planner", "configure"), check=True)
    runner.run(("ros2", "lifecycle", "set", "/lunar_planner", "activate"), check=True)
    return persist_running_state(child, paths)
```

`build_runtime` invokes `colcon build` with two base paths, selected packages, `--build-base`, `--install-base`, `--log-base` under `LUNA_HOME`, and `-DCMAKE_BUILD_TYPE=RelWithDebInfo`. It never invokes formal preflight, replay, qualification, coverage evaluation or a release gate. `start_runtime` launches `ros2 launch lunar_planner_ros lunar_planner.launch.py params_file:=...` after sourcing only `/opt/ros/humble/setup.bash` and the generated install setup. It waits for the node, then performs configure and activate; any failure records stable JSON, terminates its own child and leaves a previous install untouched.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest -q tests/deployment/test_build.py tests/deployment/test_process.py tests/deployment/test_cli.py`

Expected: PASS; no command contains training/qualification tokens, no output base is inside source, lifecycle ordering is exact, and all failure paths remain fail-closed.

- [ ] **Step 5: Commit**

```bash
git add deployment tests/deployment/test_build.py tests/deployment/test_process.py
git commit -m "feat(deploy): build and manage luna planner runtime"
```

### Task 6: 实现模型 artifact 交接、原子模型指针和扩展注册边界

**Files:**
- Create: `ros2_ws/src/lunar_policy_runtime/package.xml`
- Create: `ros2_ws/src/lunar_policy_runtime/setup.py`
- Create: `ros2_ws/src/lunar_policy_runtime/setup.cfg`
- Create: `ros2_ws/src/lunar_policy_runtime/resource/lunar_policy_runtime`
- Create: `ros2_ws/src/lunar_policy_runtime/lunar_policy_runtime/__init__.py`
- Create: `ros2_ws/src/lunar_policy_runtime/lunar_policy_runtime/manifest.py`
- Create: `ros2_ws/src/lunar_policy_runtime/lunar_policy_runtime/probe.py`
- Create: `ros2_ws/src/lunar_policy_runtime/test/test_manifest.py`
- Create: `deployment/luna_runtime/model_store.py`
- Modify: `deployment/luna_runtime/commands.py`
- Modify: `deployment/luna_runtime/cli.py`
- Create: `tests/deployment/test_model_store.py`

**Interfaces:**
- Produces `ModelManifest`, `ValidatedModelPackage`, `validate_model_package(root, contract)`, `probe_onnx(package, profile, runner)`, `ModelStore.install(source)`, `ModelStore.activate(model_id)`, `ModelStore.rollback()`, `ModelStore.status()`.
- Accepts a directory or `.tar.gz` containing exactly `model-manifest.json`, `policy.onnx`, `normalization.npz`; no `.pt`, optimizer state or unknown executable files.
- State uses `<LUNA_HOME>/models/<model-id>/<model-sha256>/`, `active-model.json` and `previous-model.json`; pointer replacement is atomic.

- [ ] **Step 1: Write failing manifest/store tests**

```python
def test_install_rejects_contract_or_hash_mismatch_without_changing_active_model(tmp_path):
    store = ModelStore(tmp_path)
    store.install(valid_model_package(tmp_path / "good"))
    store.activate("demo-v4")
    with pytest.raises(ModelInstallError, match="NORMALIZATION_SHA256_MISMATCH"):
        store.install(corrupt_normalization_package(tmp_path / "bad"))
    assert store.status().active_model_id == "demo-v4"


def test_start_refuses_policy_mode_until_ros_policy_adapter_exists(tmp_path):
    config = valid_config(policy={"mode": "onnx", "model_id": "demo-v4"})
    assert run_cli(["start", "--config", str(write_config(tmp_path, config))]).json["reason"] == "POLICY_RUNTIME_UNBOUND"
```

The manifest test must reject an unsupported observation/action contract version, changed ONNX hash, wrong input/output names, wrong shapes/dtypes, nonfinite normalization values and a source-commit incompatibility.

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest -q tests/deployment/test_model_store.py`

Expected: FAIL because the runtime package and model store do not exist.

- [ ] **Step 3: Implement contract validator and profile-specific loading probe**

```python
@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    package_sha256: str
    source_commit: str
    observation_contract: str
    action_contract: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    normalization_sha256: str


def validate_model_package(root: Path) -> ValidatedModelPackage:
    manifest = ModelManifest.from_json(root / "model-manifest.json")
    require_sha256(root / "policy.onnx", manifest.onnx_sha256, "ONNX_SHA256_MISMATCH")
    require_sha256(root / "normalization.npz", manifest.normalization_sha256, "NORMALIZATION_SHA256_MISMATCH")
    require_contract(manifest, ObservationContractV4, ActionContractV2)
    return ValidatedModelPackage(root=root, manifest=manifest)
```

`lunar_policy_runtime` is an ament Python package used for validation/probing only. On amd64, its probe constructs an ONNX Runtime CPU session and validates named tensor metadata before returning success. On Orin, `luna model install` invokes the target-local `trtexec` with the immutable ONNX path and records ONNX SHA-256, TensorRT version, device fingerprint and precision in engine metadata; mismatch removes only that generated engine and rebuilds it. Tests inject a fake probe runner, so they never require ONNX Runtime or TensorRT on the development machine.

`luna model activate` updates only `active-model.json` after a successful probe. `rollback` atomically restores `previous-model.json`, or changes to `fallback` when none exists. The command output and `luna status` must say `model_binding: staged_not_connected` until a separately approved candidate-policy ROS adapter exists. This is a deliberate safety guarantee, not an incomplete silent fallback.

Add extension registry entries `map_pipeline` and `path_tracking` with fields `{enabled: bool, package: str, status: "not_installed" | "disabled" | "enabled"}`. `luna extension enable` may only set configuration when the declared package is installed in this runtime; it cannot synthesize a map publisher or controller Topic.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest -q tests/deployment/test_model_store.py ros2_ws/src/lunar_policy_runtime/test/test_manifest.py`

Expected: PASS; artifact installation is immutable, failed install/activation preserves active state, architecture-specific probes are selected without device execution, rollback is atomic, unsupported policy start fails closed, and extensions cannot be enabled without their package.

- [ ] **Step 5: Commit**

```bash
git add ros2_ws/src/lunar_policy_runtime deployment tests/deployment/test_model_store.py
git commit -m "feat(deploy): stage validated runtime model artifacts"
```

### Task 7: 完成操作者文档、bundle 自检与双目标验证矩阵

**Files:**
- Modify: `deployment/docs/README.runtime.md`
- Modify: `deployment/docs/COMMANDS.runtime.md`
- Modify: `tests/deployment/test_runtime_docs.py`
- Modify: `tests/deployment/test_bundle.py`
- Modify: `README.md` only if it needs a one-line pointer to deployment bundle generation; otherwise leave it unchanged.

**Interfaces:**
- Consumes all public `luna` subcommands from Tasks 2–6 and the external interface baseline.
- Produces a bundle-root `README.md`, `COMMANDS.md`, `release-manifest.json` and a target-verification record outside Git.
- No command in either document requires users to manually source ROS, call colcon, construct lifecycle requests or type a long Python module path.

- [ ] **Step 1: Write failing documentation/bundle acceptance tests**

```python
def test_rendered_commands_document_every_public_subcommand_and_safe_policy_boundary():
    commands = rendered_commands()
    for token in ("init", "doctor", "config check", "build", "start", "stop", "status", "logs", "model install", "model activate", "model rollback", "extension", "bundle"):
        assert f"luna {token}" in commands
    assert "POLICY_RUNTIME_UNBOUND" in commands
    assert "formal preflight" not in commands.lower()


def test_bundle_root_contains_exactly_two_markdown_documents(tmp_path):
    names = archive_names(make_amd64_bundle(tmp_path))
    markdown = [name for name in names if name.endswith(".md")]
    assert markdown == ["lunar-runtime-ubuntu22-humble-amd64-src/COMMANDS.md", "lunar-runtime-ubuntu22-humble-amd64-src/README.md"]
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest -q tests/deployment/test_runtime_docs.py tests/deployment/test_bundle.py`

Expected: FAIL until templates document the implemented public behavior and bundle filtering is exact.

- [ ] **Step 3: Write operator instructions and run the target matrix**

`README.runtime.md` must contain: project purpose; the fallback/model-staging distinction; a compact component diagram; both profile support matrices; all external v5 input/output names, owners and frame expectations; the five-minute `./luna init`, `./luna doctor`, `./luna build`, `./luna start` workflow; and non-disableable safety boundaries.

`COMMANDS.runtime.md` must contain one copyable command per operation, expected success/error JSON fields, a complete `runtime.yaml` example, model package layout, `POLICY_RUNTIME_UNBOUND` explanation, model rollback, extension state semantics and recovery commands. It must also state that target amd64 and Orin native builds are separate evidence, not cross-compilation claims.

Run the following verification matrix after implementation:

```bash
python3 -m pytest -q tests/deployment
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
git diff --check

LUNA_HOME="$(mktemp -d)" ./scripts/luna bundle --target ubuntu22-humble-amd64 --output "$HOME/CodexDownloads/lunar_navigation/runtime-bundles"
tar -tzf "$HOME/CodexDownloads/lunar_navigation/runtime-bundles/lunar-runtime-ubuntu22-humble-amd64-src.tar.gz"
```

On the independent amd64 target, run `./luna init`, `doctor`, `config check`, `build`, then `start` with real external capability-file paths and confirm the planner becomes lifecycle `active`. On Jetson Orin R36, run the same sequence from its own source bundle and capture the native build/engine probe result. Label the latter as device validation only after it actually runs; do not infer it from amd64 tests.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest -q tests/deployment && python3 tools/check_repository_boundaries.py . && python3 -m pytest -q tests/foundation/test_repository_boundaries.py && git diff --check`

Expected: all deployment tests pass, repository boundaries pass, the bundle contains only current allowlisted source plus the two root docs, and the working diff has no whitespace errors.

- [ ] **Step 5: Commit**

```bash
git add deployment tests/deployment README.md
git commit -m "docs(deploy): document luna runtime operation"
```

## Self-Review

### Spec coverage

- Two profile-specific native source bundles, their manifests and strict active-source allowlist are covered by Tasks 1 and 3.
- `luna` initialization, configuration, host diagnosis, native build, lifecycle start/stop, status/logs, model operations, extension registration and bundle generation are covered by Tasks 2, 5 and 6.
- Existing v5 inputs/outputs become configurable without message drift in Task 4; the rendered parameter file is produced by Tasks 1 and 5.
- Source-tree/output separation, no training bridge, no old algorithms and no bundle artifacts are enforced in Tasks 1, 3 and 5.
- README/COMMANDS and independent amd64/Orin evidence are covered in Task 7.
- The specification's intended future learned-policy behavior is deliberately not claimed as present: Tasks 5 and 6 enforce `POLICY_RUNTIME_UNBOUND` until an independently designed ROS candidate/Observation V4 adapter is approved. This preserves the stated fallback deployment behavior while avoiding an inactive model that appears to control planning.

### Placeholder scan

The plan contains no unbounded file collection, implicit build output location, automatic checkpoint pickup, unverified architecture claim, or undefined safety fallback. Every created module, public interface, test command and runtime error code is named above.

### Type consistency

- Tasks 2, 3 and 5 all consume the same `RuntimeConfig`, `TargetProfile`, `RuntimePaths` and `RuntimeState` types from Tasks 1–2.
- Task 5 generates the exact `interfaces.*` ROS parameters read by Task 4.
- Task 6's `ModelManifest` validates `ObservationContractV4` and `ActionContractV2`, then stores only `ValidatedModelPackage`; it does not pass a raw checkpoint to Task 5.
- Task 7 verifies the same public subcommands, model-state labels and bundle-root layout produced in Tasks 2–6.
