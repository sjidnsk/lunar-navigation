# WHEELED 参考轨迹执行控制器实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增可选启用的 WHEELED 任务执行协调器与 Pure Pursuit 控制器，将已认证的 `MotionReference` 安全执行为 `/Car/T5/Car_Cmd_Vel` 的 `Twist`，并回传可审计的执行反馈。

**Architecture:** `luna_task_execution_coordinator` 是唯一 `PlanMotion` Action 客户端：它接收外部 `GoalRegion` 和活跃 `ExplorationTask`，转交唯一有效的 WHEELED 参考。`luna_wheeled_controller` 只跟踪已转交的参考并向底盘发 `Twist`；二者不规划、不产生候选、不发布 TF。启动、构建与配置由 `luna` 统一管理，默认不启动自动控制。

**Tech Stack:** ROS 2 Humble、Python 3.10、rclpy、`geometry_msgs/msg/Twist`、`nav_msgs/msg/Odometry`、`lunar_planning_msgs`、`lunar_navigation_msgs`、pytest、ament_cmake_python。

**Spec:** `docs/superpowers/specs/2026-08-20-wheeled-controller-execution-design.md`

## Global Constraints

- 只支持 `MotionReference.WHEELED`；LEGGED/HOPPER 绝不发布 `/Car/T5/Car_Cmd_Vel`。
- 命令 Topic 固定默认 `/Car/T5/Car_Cmd_Vel`，消息类型固定 `geometry_msgs/msg/Twist`；不得同时运行键盘控制器与自动控制器。
- 控制器不更改 `PlanMotion`、`MotionReference`、`MotionExecutionFeedback` 或任何 TF 消息定义。
- 地面反馈必须使用 `segment_id = plan_id`，并在每次新参考后从 `sequence=1` 开始递增。
- 任一参考、里程计、TF、路径或偏离检查失败时，同周期先发布零 `Twist`，再发布失败/取消反馈。
- `controller.wheeled.enabled` 默认 `false`；关闭时不允许任何自动 Twist 输出。
- 运行时所有构建、日志和 PID 仍写入 `LUNA_HOME`，不写回源码树；实机底盘验证不作为本计划内验证。

---

### Task 1: 建立独立执行包与确定性 Pure Pursuit 核心

**Files:**
- Create: `ros2_ws/src/luna_wheeled_controller/package.xml`
- Create: `ros2_ws/src/luna_wheeled_controller/CMakeLists.txt`
- Create: `ros2_ws/src/luna_wheeled_controller/python/luna_wheeled_controller/__init__.py`
- Create: `ros2_ws/src/luna_wheeled_controller/python/luna_wheeled_controller/tracking.py`
- Create: `ros2_ws/src/luna_wheeled_controller/test/test_tracking.py`

**Interfaces:**
- Produces immutable `TrackingPolicy`, `TrackingState`, `TrackingCommand`, `TrackingFailure`, `validate_tracking_policy(policy)`, and `track_path(path_xy_yaw, state, policy) -> TrackingCommand`.
- `TrackingCommand` contains `linear_x_mps`, `angular_z_radps`, `complete`, and `failure_reason: str | None`.
- `track_path` is ROS-message-free and returns `INVALID_REFERENCE`, `STALE_INPUT`, or `PATH_DEVIATION` instead of raising for invalid runtime input.

- [ ] **Step 1: Write the failing tracking tests**

```python
from luna_wheeled_controller.tracking import (
    TrackingPolicy, TrackingState, track_path, validate_tracking_policy,
)


def policy() -> TrackingPolicy:
    return TrackingPolicy(
        lookahead_m=1.0, max_linear_mps=0.2, max_angular_radps=0.5,
        max_cross_track_error_m=1.0, goal_position_tolerance_m=0.25,
        goal_yaw_tolerance_rad=0.35,
    )


def test_straight_path_commands_bounded_forward_motion():
    result = track_path(((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)), TrackingState(0.0, 0.0, 0.0), policy())
    assert result.failure_reason is None
    assert 0.0 < result.linear_x_mps <= 0.2
    assert result.angular_z_radps == 0.0
    assert not result.complete


def test_goal_within_pose_tolerances_completes_without_motion():
    result = track_path(((0.0, 0.0, 0.0),), TrackingState(0.1, 0.1, 0.1), policy())
    assert result.complete
    assert (result.linear_x_mps, result.angular_z_radps) == (0.0, 0.0)


def test_excessive_cross_track_error_fails_closed():
    result = track_path(((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)), TrackingState(0.0, 1.01, 0.0), policy())
    assert result.failure_reason == "PATH_DEVIATION"
    assert (result.linear_x_mps, result.angular_z_radps) == (0.0, 0.0)


def test_invalid_policy_is_rejected_before_control():
    assert validate_tracking_policy(policy().__class__(**{**policy().__dict__, "lookahead_m": 0.0})) == "LOOKAHEAD_INVALID"
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest -q ros2_ws/src/luna_wheeled_controller/test/test_tracking.py`

Expected: fail during import because `luna_wheeled_controller.tracking` does not exist.

- [ ] **Step 3: Implement the minimal pure functions**

```python
@dataclass(frozen=True)
class TrackingCommand:
    linear_x_mps: float
    angular_z_radps: float
    complete: bool
    failure_reason: str | None = None


def track_path(path_xy_yaw, state, policy) -> TrackingCommand:
    if not _finite_path(path_xy_yaw) or not _finite_state(state):
        return TrackingCommand(0.0, 0.0, False, "INVALID_REFERENCE")
    nearest, cross_track = _nearest_path_point(path_xy_yaw, state)
    if cross_track > policy.max_cross_track_error_m:
        return TrackingCommand(0.0, 0.0, False, "PATH_DEVIATION")
    if _at_goal(path_xy_yaw[-1], state, policy):
        return TrackingCommand(0.0, 0.0, True)
    target = _lookahead_point(path_xy_yaw, nearest, state, policy.lookahead_m)
    return _bounded_pure_pursuit(target, path_xy_yaw[-1], state, policy)
```

- [ ] **Step 4: Verify GREEN and package installability**

Run: `python3 -m pytest -q ros2_ws/src/luna_wheeled_controller/test/test_tracking.py`

Expected: pass with deterministic bounded output and fail-closed invalid cases.

- [ ] **Step 5: Commit**

```bash
git add ros2_ws/src/luna_wheeled_controller
git commit -m "feat(execution): add wheeled pure pursuit core"
```

### Task 2: 实现任务执行协调器与受限参考转交

**Files:**
- Create: `ros2_ws/src/luna_wheeled_controller/python/luna_wheeled_controller/coordinator.py`
- Create: `ros2_ws/src/luna_wheeled_controller/python/luna_wheeled_controller/task_execution_coordinator_node.py`
- Create: `ros2_ws/src/luna_wheeled_controller/scripts/luna_task_execution_coordinator_node.py`
- Create: `ros2_ws/src/luna_wheeled_controller/test/test_coordinator.py`

**Interfaces:**
- Consumes `/mission/exploration_task: lunar_navigation_msgs/msg/ExplorationTask` and `/mission/execution_goal: lunar_planning_msgs/msg/GoalRegion`.
- Produces a single `PlanMotion.Goal` with a generated request ID and the active task's `mission_id`/`revision`; publishes `MotionReference` to `/execution/wheeled_reference` only after a WHEELED `ACTIVATE_NEW_REFERENCE` result.
- Produces `CoordinatorDecision(reference: MotionReference | None, cancel_reason: str | None)` through the ROS-free `decide_plan_result(result)` helper.

- [ ] **Step 1: Write failing coordinator tests**

```python
from luna_wheeled_controller.coordinator import ActiveMission, make_plan_goal, decide_plan_result


def test_active_mission_and_goal_form_one_plan_motion_request():
    goal = make_plan_goal(ActiveMission("mission-a", 7), goal_region=object(), request_id="wheel-1")
    assert (goal.request_id, goal.mission_id, goal.mission_revision) == ("wheel-1", "mission-a", 7)


def test_only_wheeled_activation_is_forwarded():
    decision = decide_plan_result(plan_result(has_reference=True, directive="ACTIVATE_NEW_REFERENCE", platform="WHEELED"))
    assert decision.reference is not None
    assert decision.cancel_reason is None


def test_hold_or_non_wheeled_result_cancels_without_reusing_old_reference():
    decision = decide_plan_result(plan_result(has_reference=False, directive="HOLD_POSITION", platform="WHEELED"))
    assert decision.reference is None
    assert decision.cancel_reason == "REFERENCE_WITHDRAWN"
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest -q ros2_ws/src/luna_wheeled_controller/test/test_coordinator.py`

Expected: fail because coordinator module is absent.

- [ ] **Step 3: Implement pure result classification and ROS node**

```python
def decide_plan_result(result: PlanMotion.Result) -> CoordinatorDecision:
    reference = result.reference
    if (result.has_reference and
        result.execution_directive == PlanMotion.Result.ACTIVATE_NEW_REFERENCE and
        reference.platform_type == MotionReference.WHEELED and
        reference.plan_id):
        return CoordinatorDecision(reference=reference, cancel_reason=None)
    return CoordinatorDecision(reference=None, cancel_reason="REFERENCE_WITHDRAWN")
```

The node stores only the latest active task. It rejects execution goals while the task is absent, paused, canceled, or revision-mismatched. It sends one Action goal at a time; a newer external execution goal cancels the outstanding Action before replacing it. A cancellation decision publishes an empty `MotionReference` with `plan_id=""` to the private controller handoff topic; the controller interprets that exact encoding as cancel and never follows a previous reference.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest -q ros2_ws/src/luna_wheeled_controller/test/test_coordinator.py`

Expected: pass; only valid WHEELED activation becomes a reference and every other terminal planner result is an explicit cancellation.

- [ ] **Step 5: Commit**

```bash
git add ros2_ws/src/luna_wheeled_controller
git commit -m "feat(execution): coordinate wheeled plan references"
```

### Task 3: 实现 ROS 轨迹控制器、反馈与零速度故障路径

**Files:**
- Create: `ros2_ws/src/luna_wheeled_controller/python/luna_wheeled_controller/wheeled_controller_node.py`
- Create: `ros2_ws/src/luna_wheeled_controller/scripts/luna_wheeled_controller_node.py`
- Create: `ros2_ws/src/luna_wheeled_controller/config/wheeled_controller.yaml`
- Create: `ros2_ws/src/luna_wheeled_controller/test/test_wheeled_controller_node.py`
- Modify: `ros2_ws/src/luna_wheeled_controller/CMakeLists.txt`

**Interfaces:**
- Consumes `/execution/wheeled_reference: lunar_planning_msgs/msg/MotionReference` and `/Car/T3/semantic/current_pose: nav_msgs/msg/Odometry`.
- Produces `/Car/T5/Car_Cmd_Vel: geometry_msgs/msg/Twist` and `/execution/motion_feedback: lunar_navigation_msgs/msg/MotionExecutionFeedback`.
- Parameters are exactly `reference_topic`, `odometry_topic`, `command_topic`, `feedback_topic`, `control_rate_hz`, `lookahead_m`, `max_linear_mps`, `max_angular_radps`, `max_cross_track_error_m`, `goal_position_tolerance_m`, `goal_yaw_tolerance_rad`, `reference_max_age_s`, and `odometry_max_age_s`.

- [ ] **Step 1: Write failing ROS node tests**

```python
def test_valid_wheeled_reference_publishes_accept_execute_and_complete():
    controller = WheeledControllerNode(parameters=_policy_parameters())
    controller._on_odometry(fresh_odometry(x=0.0, y=0.0, yaw=0.0))
    controller._on_reference(wheeled_reference(plan_id="wheel-1", path=[(0.0, 0.0), (1.0, 0.0)]))
    controller._tick()
    assert feedback_states(controller)[:2] == [Feedback.ACCEPTED, Feedback.EXECUTING]
    assert latest_twist(controller).linear.x > 0.0
    controller._on_odometry(fresh_odometry(x=1.0, y=0.0, yaw=0.0))
    controller._tick()
    assert feedback_states(controller)[-1] == Feedback.SEGMENT_COMPLETE
    assert latest_twist(controller).linear.x == 0.0


def test_stale_odometry_stops_before_failed_feedback():
    controller = WheeledControllerNode(parameters=_policy_parameters())
    controller._on_reference(wheeled_reference(plan_id="wheel-1", path=[(0.0, 0.0), (1.0, 0.0)]))
    controller._on_odometry(stale_odometry())
    controller._tick()
    assert published_events(controller)[-2:] == [("twist", 0.0, 0.0), ("feedback", Feedback.FAILED, "STALE_INPUT")]


def test_legged_reference_never_produces_a_nonzero_twist():
    controller = WheeledControllerNode(parameters=_policy_parameters())
    controller._on_reference(legged_reference(plan_id="leg-1"))
    assert latest_twist(controller).linear.x == 0.0
    assert latest_feedback(controller).reason_code == "INVALID_REFERENCE"
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest -q ros2_ws/src/luna_wheeled_controller/test/test_wheeled_controller_node.py`

Expected: fail because `WheeledControllerNode` is not defined.

- [ ] **Step 3: Implement node state machine and feedback identity**

```python
def _fail_closed(self, state: int, reason: str) -> None:
    self._publish_zero_twist()
    self._publish_feedback(state=state, reason_code=reason)
    self._active_reference = None

def _publish_feedback(self, state: int, reason_code: str = "") -> None:
    msg = MotionExecutionFeedback()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = "base_link"
    msg.sequence = self._sequence + 1
    msg.platform_type = MotionExecutionFeedback.WHEELED
    msg.plan_id = self._active_reference.plan_id
    msg.segment_id = self._active_reference.plan_id
    msg.state = state
    msg.reason_code = reason_code
    self._feedback.publish(msg)
    self._sequence = msg.sequence
```

The ROS callback validates the reference before storing it, publishes `ACCEPTED` only after validation, resets sequence to zero for a new plan, and emits `CANCELED/REFERENCE_WITHDRAWN` for the exact empty-reference cancellation encoding from Task 2. The timer checks both source timestamps before calling `track_path`; its exception handler calls `_fail_closed(FAILED, "CONTROLLER_EXCEPTION")`. `destroy_node` and lifecycle shutdown call `_publish_zero_twist` exactly once when a nonzero command was active.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest -q ros2_ws/src/luna_wheeled_controller/test/test_tracking.py ros2_ws/src/luna_wheeled_controller/test/test_wheeled_controller_node.py`

Expected: pass; every unsafe path produces a zero command before the matching failure feedback.

- [ ] **Step 5: Commit**

```bash
git add ros2_ws/src/luna_wheeled_controller
git commit -m "feat(execution): track certified wheeled references safely"
```

### Task 4: 将可选控制器接入统一运行时配置、构建和启动

**Files:**
- Modify: `deployment/luna_runtime/config.py`
- Modify: `deployment/config/runtime.schema.json`
- Modify: `deployment/config/runtime.default.yaml`
- Modify: `deployment/config/task3-adapted.runtime.yaml`
- Modify: `deployment/luna_runtime/build.py`
- Modify: `deployment/luna_runtime/process.py`
- Modify: `ros2_ws/src/luna_t3_map_adapter/launch/task3_live_integration.launch.py`
- Modify: `ros2_ws/src/lunar_planner_ros/launch/lunar_planner.launch.py`
- Modify: `deployment/runtime_source_allowlist.yaml`
- Modify: `tests/deployment/test_config.py`
- Modify: `tests/deployment/test_build.py`
- Modify: `tests/deployment/test_process.py`

**Interfaces:**
- Adds `RuntimeConfig.controller: Mapping[str, Mapping[str, object]]` and validates exactly one `wheeled` mapping.
- Adds controller parameter YAML beneath `/luna_task_execution_coordinator` and `/luna_wheeled_controller`; `render_planner_params` is renamed `render_runtime_params` and emits all three node parameter mappings.
- `make_build_plan` includes `luna_wheeled_controller` only when `controller.wheeled.enabled` is true.
- `start_runtime` uses the unified launch in both `external_canonical` and `task3_adapted` modes when enabled; disabled mode keeps the current exact launch command and does not build/start controller nodes.

- [ ] **Step 1: Write failing deployment tests**

```python
def test_runtime_config_defaults_to_disabled_wheeled_controller(tmp_path):
    config = load_runtime_config(write_valid_config(tmp_path, controller={"wheeled": {"enabled": False}}))
    assert config.controller["wheeled"]["enabled"] is False


def test_enabled_controller_requires_nonempty_twist_and_feedback_topics(tmp_path):
    with pytest.raises(ConfigError, match="controller.wheeled.command_topic"):
        load_runtime_config(write_valid_config(tmp_path, controller={"wheeled": {"enabled": True, "command_topic": ""}}))


def test_task3_enabled_controller_builds_controller_package(tmp_path):
    plan = make_build_plan(enabled_task3_config(), paths(tmp_path), REPO_ROOT)
    assert "luna_wheeled_controller" in plan.packages


def test_disabled_controller_keeps_existing_task3_launch_command(tmp_path):
    start_runtime(disabled_task3_config(), paths(tmp_path), FakeRunner())
    assert FakeRunner.last_popen[0:3] == ("ros2", "launch", "luna_t3_map_adapter")
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest -q tests/deployment/test_config.py tests/deployment/test_build.py tests/deployment/test_process.py`

Expected: fail because `RuntimeConfig` has no controller section and build/launch do not select the controller package.

- [ ] **Step 3: Implement strict config and launch selection**

```python
controller = _require_mapping(data["controller"], "controller")
wheeled = _require_mapping(controller.get("wheeled"), "controller.wheeled")
if set(wheeled) != _WHEELED_CONTROLLER_KEYS:
    raise ConfigError("controller.wheeled keys mismatch")
if not isinstance(wheeled["enabled"], bool):
    raise ConfigError("controller.wheeled.enabled must be boolean")
for key in _WHEELED_POSITIVE_KEYS:
    if not _positive_number(wheeled[key]):
        raise ConfigError(f"controller.wheeled.{key} must be positive")
```

Use launch arguments `controller_enabled:=true|false` and `runtime_params_file:=...`. The combined Task3 launch starts coordinator and controller Nodes only when `controller_enabled` is true. The generic planner launch gains the same conditional Nodes. `luna` passes only generated params; it never reads a training checkpoint or enables a controller through an extension toggle.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest -q tests/deployment/test_config.py tests/deployment/test_build.py tests/deployment/test_process.py`

Expected: pass; disabled profiles have no controller build/process side effect and enabled profiles use only validated endpoints/limits.

- [ ] **Step 5: Commit**

```bash
git add deployment ros2_ws/src/luna_t3_map_adapter/launch ros2_ws/src/lunar_planner_ros/launch tests/deployment
git commit -m "feat(runtime): launch optional wheeled controller"
```

### Task 5: 更新运行时文档并完成最小集成验证

**Files:**
- Modify: `deployment/docs/README.runtime.md`
- Modify: `deployment/docs/COMMANDS.runtime.md`
- Modify: `tests/deployment/test_runtime_docs.py`
- Modify: `ros2_ws/src/luna_wheeled_controller/CMakeLists.txt`

**Interfaces:**
- Documents `controller.wheeled.enabled`, mutual exclusion with `keyboard_teleop.py`, `/mission/execution_goal`, `/execution/wheeled_reference`, `/Car/T5/Car_Cmd_Vel`, and the feedback state/reason contract.
- The bundle allowlist includes `luna_wheeled_controller`; deployment source packages continue to exclude tests and artifacts.

- [ ] **Step 1: Write failing documentation acceptance tests**

```python
def test_runtime_commands_document_optional_wheeled_execution_and_stop_boundary():
    commands = rendered_commands()
    for token in ("controller.wheeled.enabled", "/Car/T5/Car_Cmd_Vel", "/mission/execution_goal", "keyboard_teleop.py", "STALE_INPUT"):
        assert token in commands
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest -q tests/deployment/test_runtime_docs.py`

Expected: fail because the controller operation boundary is not documented.

- [ ] **Step 3: Document activation and execute focused verification**

Document that automatic execution requires a valid task publisher, goal publisher, Task3 odometry and a single controller owner. Include the exact configuration fragment with `enabled: false`, and a separate explicit `enabled: true` fragment. State that source-only/ROS fake-node tests do not authorize real vehicle motion.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python3 -m pytest -q tests/deployment/test_runtime_docs.py
python3 -m pytest -q ros2_ws/src/luna_wheeled_controller/test
python3 -m pytest -q tests/deployment
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
git diff --check
```

Expected: all focused controller and deployment tests pass; repository boundaries and diff checks pass. No real ROS graph or `/Car/T5/Car_Cmd_Vel` publisher is contacted.

- [ ] **Step 5: Commit**

```bash
git add deployment/docs tests/deployment ros2_ws/src/luna_wheeled_controller
git commit -m "docs(runtime): describe optional wheeled execution"
```

## Self-Review

### Spec coverage

- Optional WHEELED-only tracking, bounded Pure Pursuit, coordinate-independent topic ownership and terminal feedback are covered by Tasks 1 and 3.
- `GoalRegion → PlanMotion → MotionReference` ownership and no stale-reference reuse are covered by Task 2.
- Zero Twist before each fail/cancel feedback, stale inputs, path deviation and shutdown behavior are covered in Task 3.
- Default-disabled configuration, source bundle inclusion and unified Task3/generic start behavior are covered in Task 4.
- Operator safety boundary and test-only versus real-motion distinction are covered in Task 5.

### Placeholder scan

The plan names every package, Topic, configuration key, helper interface, failure code and verification command required for implementation. It contains no deferred safety behavior or unspecified low-level command interface.

### Type consistency

- Task 2 publishes `MotionReference`; Task 3 consumes that same type.
- Ground feedback always uses `segment_id = plan_id`, matching `ExecutionFeedbackTracker`.
- Task 4 renders parameters for the node names created by Tasks 2–3 and selects their package only when the shared `enabled` flag is true.
