# RViz2 Single-Platform Interactive Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在仓库外验证工程中提供 RViz 左侧平台选择、`2D Goal Pose` 交互规划，以及当前平台路线、弹道、飞行管、落区和不可行原因的实时显示。

**Architecture:** 保持主仓规划接口不变，在外部 `lunar_isaac_validation` 包中增加纯目标契约、交互快照桥、会话监管、Action 控制器和 RViz 消息构造；另建小型 `ament_cmake` RViz Panel 插件包。启动脚本租用独立 ROS domain，同时启动控制器和 RViz；控制器一次只拥有一个 bridge/planner 子进程对和一个 Action goal。

**Tech Stack:** Ubuntu 22.04 amd64、ROS 2 Humble、Python 3.10、rclpy、rclcpp、Qt5、rviz_common/pluginlib、grid_map_rviz_plugin、pytest、ament_cmake_gtest、colcon。

## Global Constraints

- 所有实现路径均相对外部仓 `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression`；主仓只新增本计划，不修改 `PlanMotion.action`、规划核心或外部 Topic 基线。
- 执行 ROS 命令前必须 `source /opt/ros/humble/setup.bash` 并确认 `ROS_DISTRO=humble`；Python 必须为 3.10。
- 不修改或保存 Isaac Sim USD，不控制平台运动，不重新采集或改写快照，不修改能力文件和正式 `scenario_lock.json`。
- 正式 `run_action_regression.sh` 默认行为、JSON/JUnit schema 和六案例判定必须保持不变。
- RViz 任何时刻只显示当前平台和当前请求；平台切换或新请求必须清除旧 Path/Marker。
- 交互结果属于探索证据，不写入正式回归 artifact；build/install/log/session artifact 继续保存在外部仓未跟踪目录。
- 禁止 `rm -rf`、`pkill`、按名称批量终止进程或覆盖既有 artifact；只清理由当前入口记录的 PID/PGID。
- 严格 TDD：每项行为先写失败测试、观察预期失败，再写最小实现并运行回归。
- 所有中文 Markdown、Python、JSON 和配置文件使用 UTF-8；保留主仓未跟踪 `.vscode/`。

---

### Task 1: Interactive Goal Contract

**Files:**
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_goal.py`
- Create: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_goal.py`

**Interfaces:**
- Consumes: `SnapshotBundle`, `geometry_msgs.msg.PoseStamped`, `lunar_planning_msgs.action.PlanMotion`。
- Produces:

```python
@dataclass(frozen=True)
class BuiltInteractiveGoal:
    message: PlanMotion.Goal
    cell_xy: tuple[int, int]
    target_xyz_m: tuple[float, float, float]
    yaw_rad: float

class InteractiveGoalError(ValueError):
    code: str

def build_interactive_goal(
    pose: PoseStamped,
    *,
    platform_key: str,
    bundle: SnapshotBundle,
    mission_id: str,
    mission_revision: int,
    request_id: str,
) -> BuiltInteractiveGoal: ...
```

- Stable local errors: `GOAL_FRAME_INVALID`, `GOAL_NONFINITE`, `GOAL_OUTSIDE_LOCAL_MAP`, `GOAL_ELEVATION_INVALID`, `GOAL_OBSTACLE`, `GOAL_FORBIDDEN`。
- Fixed tolerances: wheel/legged `0.5 m`, hopper `0.75 m`, yaw `math.pi / 12.0`。

- [ ] **Step 1: Write failing happy-path tests**

```python
def test_build_goal_uses_grid_elevation_and_pose_yaw(bundle):
    pose = _pose(x=1.25, y=2.25, yaw=math.pi / 2.0)
    built = build_interactive_goal(
        pose,
        platform_key="wheel",
        bundle=bundle,
        mission_id="interactive-wheel-0001",
        mission_revision=1,
        request_id="interactive-wheel-0001-goal-0001",
    )
    assert built.cell_xy == (1, 2)
    assert built.message.goal.point.z == pytest.approx(
        bundle.arrays["wheel__elevation"][2, 1]
    )
    assert built.message.goal.has_yaw_constraint is True
    assert built.message.goal.yaw_rad == pytest.approx(math.pi / 2.0)
    assert built.message.goal.position_tolerance_m == pytest.approx(0.5)
    assert built.message.goal.yaw_tolerance_rad == pytest.approx(math.pi / 12.0)
    assert built.message.replace_active_request is False
```

- [ ] **Step 2: Run the focused tests and confirm the missing-module failure**

Run:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_goal.py
```

Expected: FAIL during import because `interactive_goal` does not exist.

- [ ] **Step 3: Implement coordinate, elevation and yaw mapping**

```python
_POSITION_TOLERANCE_M = {"wheel": 0.5, "legged": 0.5, "hopper": 0.75}
_YAW_TOLERANCE_RAD = math.pi / 12.0

def _cell(descriptor: GridDescriptor, x: float, y: float) -> tuple[int, int]:
    cell_x = math.floor((x - descriptor.origin_xy_m[0]) / descriptor.resolution_m)
    cell_y = math.floor((y - descriptor.origin_xy_m[1]) / descriptor.resolution_m)
    if not 0 <= cell_x < descriptor.width or not 0 <= cell_y < descriptor.height:
        raise InteractiveGoalError("GOAL_OUTSIDE_LOCAL_MAP")
    return cell_x, cell_y
```

Construct a `POINT` goal in frame `map`; normalize the quaternion before computing yaw; require non-empty mission/request ids and positive revision.

- [ ] **Step 4: Add fail-closed table tests**

```python
@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda p, b: setattr(p.header, "frame_id", "odom"), "GOAL_FRAME_INVALID"),
        (lambda p, b: setattr(p.pose.position, "x", math.nan), "GOAL_NONFINITE"),
        (lambda p, b: setattr(p.pose.position, "x", 1.0e6), "GOAL_OUTSIDE_LOCAL_MAP"),
        (lambda p, b: b.arrays["wheel__valid_mask"].__setitem__((2, 1), 0), "GOAL_ELEVATION_INVALID"),
        (lambda p, b: b.arrays["wheel__obstacle"].__setitem__((2, 1), 1), "GOAL_OBSTACLE"),
        (lambda p, b: b.arrays["wheel__forbidden"].__setitem__((2, 1), 1), "GOAL_FORBIDDEN"),
    ],
)
def test_build_goal_rejects_invalid_targets(bundle, mutate, code):
    pose = _pose(x=1.25, y=2.25, yaw=0.0)
    mutate(pose, bundle)
    with pytest.raises(InteractiveGoalError, match=code):
        _build(pose, bundle)
```

- [ ] **Step 5: Run goal-contract tests and the existing snapshot tests**

Run:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_goal.py \
  ros2_ws/src/lunar_isaac_validation/test/test_snapshot_contract.py
```

Expected: PASS.

- [ ] **Step 6: Commit the goal contract**

```bash
git add ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_goal.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_goal.py
git commit -m "feat: validate interactive planning goals"
```

---

### Task 2: Interactive Snapshot Bridge

**Files:**
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/bridge_node.py`
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_bridge_node.py`
- Create: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/setup.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class SnapshotSession:
    platform_key: str
    start_position_m: tuple[float, float, float]
    mission_id: str
    mission_revision: int

class InteractiveSnapshotBridge(_SnapshotPublisherBase):
    def __init__(
        self,
        manifest_path: Path,
        platform_key: str,
        session_id: str,
        mission_revision: int,
        *,
        context: Context | None = None,
    ) -> None: ...
```

- Preserves: `SnapshotBridge(manifest_path, lock_path, scenario_id, *, context=None)` and its exact six published inputs plus `/lunar_isaac_validation/ready` service.
- New console entry: `lunar_isaac_interactive_bridge = lunar_isaac_validation.interactive_bridge_node:main`。

- [ ] **Step 1: Write regression-preservation and interactive-session tests**

```python
def test_interactive_bridge_uses_manifest_platform_pose(snapshot_files):
    bridge = InteractiveSnapshotBridge(
        snapshot_files.manifest,
        "hopper",
        "interactive-hopper-0001",
        7,
        context=snapshot_files.context,
    )
    odometry = bridge._odometry(Time(sec=1))
    mission = bridge._mission(Time(sec=1))
    assert _xyz(odometry.pose.pose.position) == bridge.bundle.platforms["hopper"].planning_position_m
    assert mission.mission_id == "interactive-hopper-0001"
    assert mission.revision == 7
```

Also assert invalid platform, unsafe session id and zero revision fail before any publishers are created.

- [ ] **Step 2: Run bridge tests and observe the missing class failure**

Run:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py \
  ros2_ws/src/lunar_isaac_validation/test/test_bridge_node.py
```

Expected: new tests FAIL on import; existing bridge tests PASS.

- [ ] **Step 3: Extract the shared publisher base without changing the locked bridge API**

Move map, odometry, TF, localization, mission, ready and timer publication into
`_SnapshotPublisherBase(bundle, session, node_name, context)`; leave `SnapshotBridge` as a thin adapter that loads the lock case and sets `self.case` for existing tests.

```python
session = SnapshotSession(
    platform_key=case.platform_key,
    start_position_m=case.start_position_m,
    mission_id=f"{case.case_id}-mission",
    mission_revision=1,
)
```

- [ ] **Step 4: Implement the interactive adapter and strict CLI**

Parser actions must be exactly `help`, `manifest`, `platform`, `session_id`, and
`mission_revision`; platform choices are `wheel`, `legged`, `hopper`.

- [ ] **Step 5: Run bridge tests and console-entry contract tests**

Run:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py \
  ros2_ws/src/lunar_isaac_validation/test/test_bridge_node.py \
  test/test_cli_contract.py
```

Expected: PASS.

- [ ] **Step 6: Commit the interactive bridge**

```bash
git add ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/bridge_node.py \
  ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_bridge_node.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py \
  ros2_ws/src/lunar_isaac_validation/setup.py
git commit -m "feat: publish interactive snapshot sessions"
```

---

### Task 3: RViz Evidence Builder

**Files:**
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/rviz_evidence.py`
- Create: `ros2_ws/src/lunar_isaac_validation/test/test_rviz_evidence.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/package.xml`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class VisualizationBatch:
    path: nav_msgs.msg.Path
    markers: visualization_msgs.msg.MarkerArray
    terminal_state: str
    reason_code: str

class RvizEvidenceError(ValueError):
    code: str

def clear_visualization(stamp: Time) -> VisualizationBatch: ...

def local_rejection_visualization(
    *, platform_key: str, start_xyz_m: tuple[float, float, float],
    start_orientation_wxyz: tuple[float, float, float, float],
    target_xyz_m: tuple[float, float, float], tolerance_m: float,
    reason_code: str, stamp: Time,
) -> VisualizationBatch: ...

def action_result_visualization(
    *, platform_key: str, result: PlanMotion.Result,
    capability_document: Mapping[str, object],
    start_xyz_m: tuple[float, float, float],
    start_orientation_wxyz: tuple[float, float, float, float],
    target_xyz_m: tuple[float, float, float],
    tolerance_m: float, stamp: Time,
) -> VisualizationBatch: ...
```

- Marker namespace is always `lunar_interactive_current`; stable ids cover platform, start, goal, tolerance, route arrows, ballistic centerline, flight-tube samples, landing boundary/fill, landing point and text.

- [ ] **Step 1: Write failing tests for clear, wheel, legged, hopper and infeasible batches**

```python
def test_hopper_batch_contains_33_point_ballistic_and_landing_region(result, capability):
    batch = action_result_visualization(
        platform_key="hopper",
        result=result,
        capability_document=capability,
        target_xyz_m=(2.0, 3.0, 0.5),
        tolerance_m=0.75,
        stamp=Time(sec=5),
    )
    centerline = _marker(batch, "ballistic_centerline")
    tube = _marker(batch, "flight_tube")
    landing = _marker(batch, "landing_boundary")
    assert len(centerline.points) == 33
    assert len(tube.points) == 33
    assert len(landing.points) >= 5
    assert landing.points[0] == landing.points[-1]
```

For wheel/legged assert the outgoing Path copies the validated `reference.path_preview` geometry into frame `map`, and a `route_line` Marker renders wheel blue or legged green. For no-reference outcomes assert empty Path, red X and exact Action reason. Every non-clear batch includes the current platform/start marker so a preceding `DELETEALL` cannot leave the view without its selected platform.

- [ ] **Step 2: Run focused tests and confirm missing-module failure**

Run:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test/test_rviz_evidence.py
```

Expected: FAIL on import.

- [ ] **Step 3: Implement shared marker primitives and DELETEALL clearing**

`clear_visualization()` must return an empty `map` Path and a MarkerArray whose first Marker uses `Marker.DELETEALL`. Every non-clear marker has lifetime zero and a normalized identity quaternion.

- [ ] **Step 4: Implement result validation and platform-specific rendering**

For hopper, sample

```python
p(t) = launch_position + launch_velocity * t + 0.5 * gravity * t * t
```

at 33 inclusive times from zero through `flight_time`; use hop `flight_tube_radius_m` as the diameter basis for translucent `SPHERE_LIST`, and close the actual `landing_region` polygon. Reject non-finite geometry, wrong platform type, wrong frame, empty required containers, invalid duration/radius or malformed polygon with stable `RVIZ_RESULT_*` codes.

- [ ] **Step 5: Declare runtime dependencies and run focused plus existing geometry tests**

Add `diagnostic_msgs` and `visualization_msgs` as `exec_depend`; retain all existing dependencies.

Run:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_rviz_evidence.py \
  ros2_ws/src/lunar_isaac_validation/test/test_action_assertions.py \
  ros2_ws/src/lunar_isaac_validation/test/test_hop_checks.py \
  ros2_ws/src/lunar_isaac_validation/test/test_trajectory_checks.py
```

Expected: PASS.

- [ ] **Step 6: Commit the RViz evidence builder**

```bash
git add ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/rviz_evidence.py \
  ros2_ws/src/lunar_isaac_validation/test/test_rviz_evidence.py \
  ros2_ws/src/lunar_isaac_validation/package.xml
git commit -m "feat: build live rviz planning evidence"
```

---

### Task 4: Single-Platform Session Supervisor

**Files:**
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_session.py`
- Create: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_session.py`

**Interfaces:**
- Produces:

```python
class InteractiveState(str, Enum):
    NO_PLATFORM = "NO_PLATFORM"
    SWITCHING = "SWITCHING"
    READY = "READY"
    PLANNING = "PLANNING"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    CANCELED = "CANCELED"
    ERROR = "ERROR"

@dataclass(frozen=True)
class ActiveSession:
    platform_key: str
    session_id: str
    mission_revision: int
    generation_stamp: Time

class SessionSupervisor:
    def switch_platform(self, platform_key: str) -> ActiveSession: ...
    def stop(self) -> None: ...
    @property
    def active_session(self) -> ActiveSession | None: ...
```

- Consumes injected `process_factory(argv: list[str], log_path: Path) -> ManagedProcess` and a transport protocol with `wait_for_endpoints_absent`, `change_state`, `get_state`, and `bridge_ready`.

- [ ] **Step 1: Write failing state/lifecycle tests using fakes**

```python
def test_switch_stops_old_session_before_starting_new(supervisor, events):
    first = supervisor.switch_platform("wheel")
    second = supervisor.switch_platform("legged")
    assert first.platform_key == "wheel"
    assert second.platform_key == "legged"
    assert events.index("stop:wheel:planner") < events.index("start:legged:bridge")
    assert events.index("stop:wheel:bridge") < events.index("start:legged:bridge")
```

Cover configure/activate sequence, generation stamp parsing, unexpected child exit, cleanup aggregation, invalid platform, retry after failure and idempotent `stop()`.

- [ ] **Step 2: Run focused tests and confirm missing-module failure**

Run:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test/test_interactive_session.py
```

Expected: FAIL on import.

- [ ] **Step 3: Implement exact child commands and session identity**

Bridge command:

```python
[
    "ros2", "run", "lunar_isaac_validation", "lunar_isaac_interactive_bridge",
    "--manifest", str(manifest_path), "--platform", platform_key,
    "--session-id", session_id, "--mission-revision", str(revision),
]
```

Planner command uses the selected params file. Session ids are monotonic within the controller process: `interactive-{platform}-{counter:04d}`; revisions increment with each successful switch.

- [ ] **Step 4: Implement lifecycle startup and cleanup**

Require endpoints absent before start; start bridge then planner; configure to state 2; activate to state 3; require bridge readiness; on failure attempt all safe cleanup and leave `active_session is None`. Stop order is lifecycle deactivate/cleanup/shutdown, planner process, bridge process, then endpoint-absence verification.

- [ ] **Step 5: Run supervisor and existing runner/process tests**

Run:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_session.py \
  ros2_ws/src/lunar_isaac_validation/test/test_process_manager.py \
  ros2_ws/src/lunar_isaac_validation/test/test_regression_runner.py
```

Expected: PASS.

- [ ] **Step 6: Commit the session supervisor**

```bash
git add ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_session.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_session.py
git commit -m "feat: supervise interactive planner sessions"
```

---

### Task 5: Interactive ROS Controller

**Files:**
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_node.py`
- Create: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/setup.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/package.xml`

**Interfaces:**
- Node name: `lunar_isaac_interactive`。
- Services: `/lunar_isaac_validation/select_wheel`, `select_legged`, `select_hopper`, `cancel_plan`, all `std_srvs/Trigger`。
- Subscription: `/goal_pose` (`geometry_msgs/PoseStamped`)。
- Publishers: `/lunar_isaac_validation/interactive_status` (`DiagnosticArray`), `current_path` (`Path`), `current_markers` (`MarkerArray`) with reliable + transient-local QoS。
- Action client: `/plan_motion`。
- New console entry: `lunar_isaac_interactive = lunar_isaac_validation.interactive_node:main`。

- [ ] **Step 1: Write failing callback/state tests with injected fakes**

```python
def test_goal_is_sent_only_after_ready(node, supervisor, action_client):
    node.receive_goal(_pose())
    assert action_client.goals == []
    assert node.last_status.reason_code == "NO_PLATFORM"
    node.request_platform("wheel")
    supervisor.complete_switch("wheel")
    node.receive_goal(_pose())
    assert len(action_client.goals) == 1
    assert node.last_status.state == InteractiveState.PLANNING
```

Add tests for asynchronous accepted switch response, `BUSY`, cancellation, feedback phase, feasible/infeasible terminal state, rejected/aborted/timeout, malformed result, switch failure, and shutdown cleanup.

- [ ] **Step 2: Run focused tests and confirm missing-module failure**

Run:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py
```

Expected: FAIL on import.

- [ ] **Step 3: Implement diagnostics and visualization publication**

Use one diagnostic status named `lunar_isaac_validation/interactive_planner`; put the state enum in `message` and publish keys `session`, `active_platform`, `action_phase`, `reason_code`. Clear messages are published before a switch and before processing every basically valid new click.

- [ ] **Step 4: Implement non-blocking selection and Action handling**

Service callbacks validate state under a lock, publish `SWITCHING`, start exactly one owned worker thread, and return `success=True` with `message=f"SWITCH_ACCEPTED:{platform_key}"` without waiting for `READY`. Run rclpy with a `MultiThreadedExecutor`; Action response/result/feedback callbacks update state without touching Qt.

Use `build_interactive_goal()` and `action_result_visualization()`. A goal received during `PLANNING` publishes stable `BUSY` without `replace_active_request`; cancel uses the active goal handle and reaches `CANCELED` only after the server confirms cancellation.

- [ ] **Step 5: Add strict CLI and shutdown behavior**

Required CLI paths: `--manifest`, `--artifact-dir`, three `--*-params`, and three `--*-capability`. Refuse a non-empty artifact directory. On SIGINT/SIGTERM, stop accepting callbacks, cancel or settle the active goal, join the owned switch thread, call supervisor cleanup, destroy the node and shut down rclpy.

- [ ] **Step 6: Run focused and package-level Python tests**

Run:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_goal.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_session.py \
  ros2_ws/src/lunar_isaac_validation/test/test_rviz_evidence.py
```

Expected: PASS.

- [ ] **Step 7: Commit the controller**

```bash
git add ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_node.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py \
  ros2_ws/src/lunar_isaac_validation/setup.py \
  ros2_ws/src/lunar_isaac_validation/package.xml
git commit -m "feat: control interactive plan motion sessions"
```

---

### Task 6: Lunar Planner RViz Panel Plugin

**Files:**
- Create: `ros2_ws/src/lunar_isaac_rviz_plugins/CMakeLists.txt`
- Create: `ros2_ws/src/lunar_isaac_rviz_plugins/package.xml`
- Create: `ros2_ws/src/lunar_isaac_rviz_plugins/plugin_description.xml`
- Create: `ros2_ws/src/lunar_isaac_rviz_plugins/include/lunar_isaac_rviz_plugins/lunar_planner_panel.hpp`
- Create: `ros2_ws/src/lunar_isaac_rviz_plugins/src/lunar_planner_panel.cpp`
- Create: `ros2_ws/src/lunar_isaac_rviz_plugins/test/lunar_planner_panel_test.cpp`
- Modify: `scripts/build_external.sh`

**Interfaces:**
- Plugin class id: `lunar_isaac_rviz_plugins/LunarPlannerPanel`。
- Produces Qt object names used by tests: `wheel_button`, `legged_button`, `hopper_button`, `cancel_button`, `platform_label`, `state_label`, `phase_label`, `reason_label`。
- Produces a test seam `applyStatusForTest(const std::string&, const std::string&, const std::string&, const std::string&)` that delegates directly to the same private UI-state method used by queued ROS callbacks; it exists only to test rendering without a live ROS graph.
- Consumes only the standard services and `interactive_status` Topic defined in Task 5.

- [ ] **Step 1: Write the plugin package and failing Qt/GTest contract first**

```cpp
TEST(LunarPlannerPanelTest, PlanningLocksPlatformButtonsAndEnablesCancel) {
  LunarPlannerPanel panel;
  panel.applyStatusForTest("wheel", "PLANNING", "SEARCHING", "");
  EXPECT_FALSE(panel.findChild<QPushButton*>("wheel_button")->isEnabled());
  EXPECT_FALSE(panel.findChild<QPushButton*>("legged_button")->isEnabled());
  EXPECT_TRUE(panel.findChild<QPushButton*>("cancel_button")->isEnabled());
  EXPECT_EQ(panel.findChild<QLabel*>("state_label")->text(), "PLANNING");
}
```

Tests also cover mutually exclusive checked buttons, `READY`, `ERROR`, missing/malformed diagnostic keys and service-button mapping.

- [ ] **Step 2: Configure and build only the new package to observe failure**

Run:

```bash
source /opt/ros/humble/setup.bash
LUNAR_PANEL_FAIL_TMP="$(mktemp -d)"
colcon build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_PANEL_FAIL_TMP/build" \
  --install-base "$LUNAR_PANEL_FAIL_TMP/install" \
  --packages-select lunar_isaac_rviz_plugins \
  --cmake-args -DBUILD_TESTING=ON
```

Expected: FAIL because the panel implementation/plugin export is absent or incomplete.

- [ ] **Step 3: Implement the Panel UI and ROS initialization**

Create a `QButtonGroup` with exclusive wheel/legged/hopper buttons. In `onInitialize()`, obtain the RViz raw node from `getDisplayContext()->getRosNodeAbstraction()`, create four async Trigger clients and the transient-local diagnostic subscription.

ROS callbacks must marshal UI changes through a queued Qt invocation:

```cpp
QMetaObject::invokeMethod(
  this,
  [this, platform, state, phase, reason]() {
    applyStatus(platform, state, phase, reason);
  },
  Qt::QueuedConnection);
```

- [ ] **Step 4: Implement async service behavior and plugin export**

Button clicks send only their exact platform service. Disable all platform buttons in `SWITCHING`/`PLANNING`; enable cancel only in `PLANNING`; surface unavailable/rejected service responses in `reason_label`. Export with `PLUGINLIB_EXPORT_CLASS` and `pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)`.

- [ ] **Step 5: Build and run panel tests offscreen**

Run:

```bash
source /opt/ros/humble/setup.bash
LUNAR_PANEL_PASS_TMP="$(mktemp -d)"
colcon build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_PANEL_PASS_TMP/build" \
  --install-base "$LUNAR_PANEL_PASS_TMP/install" \
  --packages-select lunar_isaac_rviz_plugins \
  --cmake-args -DBUILD_TESTING=ON
QT_QPA_PLATFORM=offscreen colcon test \
  --build-base "$LUNAR_PANEL_PASS_TMP/build" \
  --install-base "$LUNAR_PANEL_PASS_TMP/install" \
  --packages-select lunar_isaac_rviz_plugins \
  --event-handlers console_direct+ \
  --ctest-args --output-on-failure
colcon test-result \
  --test-result-base "$LUNAR_PANEL_PASS_TMP/build" --verbose
```

Expected: PASS.

- [ ] **Step 6: Include both external packages in the normal build entry**

Change `scripts/build_external.sh` to build `--packages-up-to lunar_isaac_validation lunar_isaac_rviz_plugins` while retaining merge-install, unique paths and all existing preflight behavior.

- [ ] **Step 7: Commit the RViz panel**

```bash
git add ros2_ws/src/lunar_isaac_rviz_plugins scripts/build_external.sh
git commit -m "feat: add lunar planner rviz panel"
```

---

### Task 7: RViz Configuration and Safe Interactive Launcher

**Files:**
- Create: `ros2_ws/src/lunar_isaac_validation/config/rviz/lunar_interactive_planning.rviz`
- Create: `scripts/run_interactive_rviz.sh`
- Modify: `ros2_ws/src/lunar_isaac_validation/setup.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/package.xml`
- Modify: `scripts/preflight.py`
- Modify: `test/test_cli_contract.py`
- Modify: `README.md`

**Interfaces:**
- Launcher CLI: required `--manifest`, `--install`; optional validated `--session-id` using the existing UTC format.
- RViz fixed frame: `map`。
- Displays: `grid_map_rviz_plugin/GridMap` for `/environment/map_global` and `/environment/map_local`, `rviz_default_plugins/Path` for `current_path`, and `rviz_default_plugins/MarkerArray` for `current_markers`。
- Panel: `lunar_isaac_rviz_plugins/LunarPlannerPanel`；Goal Tool Topic: `/goal_pose`。

- [ ] **Step 1: Write failing launcher/package/config contract tests**

```python
def test_interactive_launcher_is_owned_and_non_destructive():
    text = (SCRIPTS / "run_interactive_rviz.sh").read_text(encoding="utf-8")
    assert "source /opt/ros/humble/setup.bash" in text
    assert "ROS_DOMAIN_ID" in text
    assert "pkill" not in text
    assert "rm -rf" not in text
    assert "lunar_isaac_interactive" in text
    assert "rviz2 -d" in text
```

Parse the RViz YAML and assert the fixed frame, exact panel class, one GoalTool topic, two GridMap displays, one current Path and one current MarkerArray.

- [ ] **Step 2: Run CLI contract tests and observe missing-file failure**

Run:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test/test_cli_contract.py
```

Expected: FAIL because launcher/config do not exist.

- [ ] **Step 3: Add the installed RViz configuration**

Install `config/rviz/*.rviz` through `setup.py`; declare `rviz2`, `grid_map_rviz_plugin` and `lunar_isaac_rviz_plugins` as runtime dependencies. Configure top-down view centered over the selected snapshot region, elevation display plus translucent obstacle/forbidden displays, a neutral current Path backed by the platform-colored `route_line` Marker, and no stale multi-platform topics.

- [ ] **Step 4: Add an interactive preflight phase and launcher**

`preflight.py --phase interactive` performs Ubuntu/ROS/Python/main-commit/external-root checks but does not require the formal scenario lock or a live Isaac server.

The launcher must:

1. canonicalize manifest/install and require both under the external root;
2. require `${install}/setup.bash` and installed RViz config/plugin;
3. source Humble and the overlay;
4. lease one domain from 131–230, excluding ambient;
5. create one new `artifacts/${SESSION_ID}/interactive/` directory;
6. start the controller with explicit manifest, artifact, params and capabilities;
7. start `rviz2 -d "${SHARE}/config/rviz/lunar_interactive_planning.rviz"`;
8. trap EXIT/INT/TERM and terminate/wait only the recorded RViz and controller PIDs.

- [ ] **Step 5: Document the exact operator workflow**

README commands:

```bash
LUNAR_BUILD_RUN_ID="$(date -u +%Y%m%dT%H%M%S%6NZ)"
bash scripts/build_external.sh --run-id "$LUNAR_BUILD_RUN_ID"
bash scripts/run_interactive_rviz.sh \
  --manifest snapshots/20260804T025645914927Z/snapshot_manifest.json \
  --install "install/$LUNAR_BUILD_RUN_ID"
```

Document: select one platform, wait for `READY`, choose `2D Goal Pose`, read feasible/infeasible status, cancel before switching during planning, and understand that this uses a frozen snapshot rather than live Isaac motion.

- [ ] **Step 6: Run shell syntax and CLI/package tests**

Run:

```bash
bash -n scripts/run_interactive_rviz.sh
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test/test_cli_contract.py
```

Expected: PASS.

- [ ] **Step 7: Commit launcher and configuration**

```bash
git add ros2_ws/src/lunar_isaac_validation/config/rviz/lunar_interactive_planning.rviz \
  scripts/run_interactive_rviz.sh \
  ros2_ws/src/lunar_isaac_validation/setup.py \
  ros2_ws/src/lunar_isaac_validation/package.xml \
  scripts/preflight.py test/test_cli_contract.py README.md
git commit -m "feat: launch interactive rviz planning"
```

---

### Task 8: Hash-Bound Interactive Targets and ROS Integration

**Files:**
- Create: `scripts/qualify_interactive_targets.py`
- Create: `ros2_ws/src/lunar_isaac_validation/config/interactive_targets.json`
- Create: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_integration.py`
- Create: `test/test_interactive_target_contract.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/setup.py`

**Interfaces:**
- Fixture schema: `lunar-interactive-targets/v1`。
- Fixture binds `arrays_sha256` and stores, for each platform, exactly `feasible`, `action_infeasible`, and `local_rejection` targets with XY, yaw and expected class/reason.
- Qualifier output goes to an explicit new artifact path and never overwrites the tracked fixture or formal lock.

- [ ] **Step 1: Write failing schema and integration-harness tests**

```python
def test_interactive_targets_are_complete_and_snapshot_bound(document, manifest):
    assert document["schema_version"] == "lunar-interactive-targets/v1"
    assert document["arrays_sha256"] == manifest["arrays_sha256"]
    assert set(document["platforms"]) == {"wheel", "legged", "hopper"}
    for cases in document["platforms"].values():
        assert set(cases) == {"feasible", "action_infeasible", "local_rejection"}
```

The ROS integration test is opt-in via `LUNAR_INTERACTIVE_INTEGRATION=1`; without it, only fixture contract tests run.

- [ ] **Step 2: Implement deterministic target qualification**

The qualifier loads the hash-bound snapshot, uses the existing positive lock target as the first feasible candidate, chooses a local rejection from cells with obstacle/forbidden set, and enumerates remaining valid non-obstacle cells by descending distance then `(cell_y, cell_x)`. Through one ready session per platform, it sends candidates until it obtains a succeeded Action result with `has_reference == false`; it records the first such target and exact planner reason. It fails if any platform lacks all three classes and publishes only a report under the requested new output directory.

- [ ] **Step 3: Build a fresh candidate overlay for target qualification**

Run:

```bash
LUNAR_INTERACTIVE_BUILD_RUN_ID="$(date -u +%Y%m%dT%H%M%S%6NZ)"
bash scripts/build_external.sh --run-id "$LUNAR_INTERACTIVE_BUILD_RUN_ID"
```

Expected: the new overlay contains `lunar_isaac_interactive`, the interactive bridge and the RViz plugin.

- [ ] **Step 4: Run the qualifier against the candidate overlay and review its report**

Run:

```bash
source /opt/ros/humble/setup.bash
LUNAR_INTERACTIVE_QUALIFY_RUN_ID="$(date -u +%Y%m%dT%H%M%S%6NZ)"
source "install/$LUNAR_INTERACTIVE_BUILD_RUN_ID/setup.bash"
python3 scripts/qualify_interactive_targets.py \
  --manifest snapshots/20260804T025645914927Z/snapshot_manifest.json \
  --lock ros2_ws/src/lunar_isaac_validation/scenarios/scenario_lock.json \
  --output "artifacts/$LUNAR_INTERACTIVE_QUALIFY_RUN_ID/interactive-targets.json"
```

Expected: one hash-bound report with three target classes for all platforms; formal lock remains byte-identical.

- [ ] **Step 5: Add the reviewed exact fixture and contract validation**

Copy only the reviewed target coordinates, yaw, class and reason plus snapshot hash into `config/interactive_targets.json` using a patch. The test rejects unknown keys, booleans as numbers, non-finite values, wrong hash, duplicate targets and local-rejection reasons outside the stable list.

- [ ] **Step 6: Implement isolated ROS integration tests**

For each platform, call its select service, wait for `READY`, publish each fixture PoseStamped, and assert:

- feasible: correct current Path or hopper markers and `FEASIBLE`;
- action-infeasible: empty Path, red marker, exact planner reason and `INFEASIBLE`;
- local-rejection: no Action goal observed, red marker and exact local reason;
- switch: previous namespace is deleted before new platform `READY`;
- busy/cancel: second goal is rejected as `BUSY`, cancel reaches `CANCELED`.

- [ ] **Step 7: Run contract tests, then opt-in integration tests**

Run:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test/test_interactive_target_contract.py

source "install/$LUNAR_INTERACTIVE_BUILD_RUN_ID/setup.bash"
LUNAR_INTERACTIVE_INTEGRATION=1 python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_integration.py
```

Expected: PASS.

- [ ] **Step 8: Commit fixtures and integration tests**

```bash
git add scripts/qualify_interactive_targets.py \
  ros2_ws/src/lunar_isaac_validation/config/interactive_targets.json \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_integration.py \
  test/test_interactive_target_contract.py \
  ros2_ws/src/lunar_isaac_validation/setup.py
git commit -m "test: qualify interactive planning targets"
```

---

### Task 9: Full Regression and Visual Acceptance

**Files:**
- Modify only if verification uncovers a scoped defect: files from Tasks 1–8 plus their direct tests.
- Do not commit: `artifacts/`, `build/`, `install/`, `log/`, screenshots or session logs.

**Interfaces:**
- Completion evidence: source-first tests, merged overlay build/test, plugin discovery, interactive integration, formal six-case run, RViz visual inspection and safe shutdown.

- [ ] **Step 1: Re-probe the authoritative Ubuntu baseline**

Run:

```bash
uname -m
source /opt/ros/humble/setup.bash
printf '%s\n' "$ROS_DISTRO"
python3 --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
ros2 pkg prefix rviz2
ros2 pkg prefix grid_map_rviz_plugin
```

Expected: amd64/x86_64, Humble, Python 3.10, RTX 4080 SUPER available, RViz packages resolve.

- [ ] **Step 2: Run all source-first Python tests**

Run:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test ros2_ws/src/lunar_isaac_validation/test
```

Expected: all tests PASS.

- [ ] **Step 3: Build a fresh immutable overlay**

Run:

```bash
LUNAR_BUILD_RUN_ID="$(date -u +%Y%m%dT%H%M%S%6NZ)"
bash scripts/build_external.sh --run-id "$LUNAR_BUILD_RUN_ID"
```

Expected: a new `install/$LUNAR_BUILD_RUN_ID` containing both packages and the installed RViz config/plugin.

- [ ] **Step 4: Run colcon tests and plugin discovery**

Run:

```bash
source /opt/ros/humble/setup.bash
source "install/$LUNAR_BUILD_RUN_ID/setup.bash"
colcon --log-base "log/${LUNAR_BUILD_RUN_ID}-test" test \
  --merge-install \
  --base-paths /mnt/data/WS/lunar-navigation/ros2_ws/src ros2_ws/src \
  --build-base "build/$LUNAR_BUILD_RUN_ID" \
  --install-base "install/$LUNAR_BUILD_RUN_ID" \
  --packages-select lunar_planner_core lunar_planner_ros \
    lunar_isaac_validation lunar_isaac_rviz_plugins \
  --ctest-args -E '^lunar_planner_core_public_header_boundary$'
colcon test-result --test-result-base "build/$LUNAR_BUILD_RUN_ID" --verbose
ros2 pkg prefix lunar_isaac_rviz_plugins
```

Expected: all selected tests PASS and plugin package resolves from the new overlay.

- [ ] **Step 5: Run interactive integration against the new overlay**

Run:

```bash
source "install/$LUNAR_BUILD_RUN_ID/setup.bash"
LUNAR_INTERACTIVE_INTEGRATION=1 python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_integration.py
```

Expected: all three platforms pass feasible, Action-infeasible, local-rejection, busy/cancel and switch-clearing checks.

- [ ] **Step 6: Protect the formal six-case regression**

Run:

```bash
LUNAR_FORMAL_RUN_ID="$(date -u +%Y%m%dT%H%M%S%6NZ)"
bash scripts/run_action_regression.sh \
  --manifest snapshots/20260804T025645914927Z/snapshot_manifest.json \
  --lock ros2_ws/src/lunar_isaac_validation/scenarios/scenario_lock.json \
  --install "install/$LUNAR_BUILD_RUN_ID" \
  --run-id "$LUNAR_FORMAL_RUN_ID"
```

Expected: exit 0 and summary exactly `total=6, passed=6, failed=0, exit_code=0`; existing report schema and normalized semantics remain valid.

- [ ] **Step 7: Launch RViz and perform the approved visual checks**

Run:

```bash
bash scripts/run_interactive_rviz.sh \
  --manifest snapshots/20260804T025645914927Z/snapshot_manifest.json \
  --install "install/$LUNAR_BUILD_RUN_ID"
```

For wheel, legged and hopper: select platform, wait for `READY`, use the fixture feasible and infeasible targets, verify only the current platform remains, and visually confirm terrain/path or ballistic/tube/landing region plus readable state/reason labels.

- [ ] **Step 8: Verify safe shutdown and repository cleanliness**

After closing the owned RViz window, verify the launcher exits, its controller/child PIDs are gone, unrelated Isaac Sim/ROS/RViz processes remain untouched, and Git shows only intentional source commits:

```bash
git status --short
git log --oneline --decorate -12
```

Expected: tracked worktree clean; no runtime directories staged.

- [ ] **Step 9: If a scoped verification defect was fixed, rerun its failing test first and then Steps 2–8 before the final completion claim**

Do not weaken tests, alter the formal lock, or convert a missing visual/manual check into a claimed pass.
