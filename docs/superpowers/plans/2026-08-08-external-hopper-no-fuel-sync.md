# External Hopper No-Fuel Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the independent Isaac/ROS/RViz validation repository so repeated hopper goals never publish, commit, or decrement simulated fuel and consume the main repository's delta-v-only hop contract.

**Architecture:** The external bridge remains a visualization and execution-feedback client. It converts `HopSegment` geometry and delta-v evidence, executes one parabola, settles at the landing pose, then accepts the next Goal without a fuel transaction. All changes stay in the external repository and are cross-built against the main feature branch's ROS install.

**Tech Stack:** Ubuntu 22.04 amd64, ROS 2 Humble, Python 3.10, rclpy, pytest, colcon.

## Global Constraints

- External Git root: `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression`.
- Implement in isolated branch/worktree `feature/hopper-no-fuel-budget-sync`; do not edit or commit directly on external `main`.
- Main repository source and history must not be copied into the external repository.
- Hopper remains one certified parabola to one safe convex landing region per Goal.
- No `simulated_remaining_usable_fuel_mass_kg`, propellant publisher, fuel commit, remaining-fuel handoff, or fuel-commit failure state may remain active.
- Repeated-goal correctness still requires completed-session cleanup to remain silent and preserve landed state.
- Build artifacts stay under `/home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/external/`.

---

### Task 1: Remove fuel transactions from rolling execution

**Files:**
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/rolling_execution.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_rolling_execution.py`

**Interfaces:**
- Consumes: `HopSegment` geometry, `required_delta_v_mps`, and `available_delta_v_mps`.
- Produces: landing/settle feedback with no `hopper_fuel_commit_kg` field.

- [ ] **Step 1: Write failing no-commit tests**

```python
def test_hopper_settle_completes_without_fuel_commit() -> None:
    session = hopper_session(delta_v_only_reference())
    settled = drive_to_canonical_settle(session)
    assert settled.state is RollingExecutionState.COMPLETED
    assert not hasattr(settled, "hopper_fuel_commit_kg")


def test_second_hopper_goal_does_not_depend_on_first_hop_budget() -> None:
    first = execute_hopper_goal("goal-a")
    second = execute_hopper_goal("goal-b", start=first.landing_pose)
    assert first.state is RollingExecutionState.COMPLETED
    assert second.state is RollingExecutionState.COMPLETED
```

Use the module's existing `_hopper_reference`, `RollingExecutionSession`, canonical odometry, and settle helpers
for these tests; `delta_v_only_reference`, `hopper_session`, `drive_to_canonical_settle`, and
`execute_hopper_goal` in the sketch denote thin test-local wrappers around those existing fixtures and must not
enter production code.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation" \
python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_rolling_execution.py
```

Expected: old update still exposes `hopper_fuel_commit_kg` and old reference parsing expects three fuel fields.

- [ ] **Step 3: Remove fuel evidence and commit emission**

Delete fuel members from the internal hopper reference dataclass, conversion tuple, validation, and `RollingExecutionUpdate`. Keep required/available delta-v validation and all trajectory, map-generation, version, commitment, settle, cancellation, and collision evidence checks unchanged.

- [ ] **Step 4: Run rolling tests and commit**

```bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation" \
python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_rolling_execution.py
git add \
  ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/rolling_execution.py \
  ros2_ws/src/lunar_isaac_validation/test/test_rolling_execution.py
git commit -m "refactor: remove hopper fuel transaction from rolling execution"
```

### Task 2: Remove bridge publisher and supervisor fuel API

**Files:**
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_bridge_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_session.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_session.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py`
- Modify: launch/config files that declare the simulated fuel parameter.

**Interfaces:**
- Consumes: platform selection, maps, odometry, planner Action results, and execution feedback.
- Produces: no `/platform/hopper_propellant_state` publisher and no parameter-service fuel update.

- [ ] **Step 1: Write failing bridge-boundary tests**

Add assertions that a hopper bridge has no simulated-fuel parameter or publisher, `InteractiveSessionSupervisor` has no `commit_hopper_fuel` method, and a completed first Goal immediately permits a second Goal without any parameter transaction.

- [ ] **Step 2: Run focused tests and verify RED**

Run the three interactive test modules. Expected: old APIs, parameter callbacks, and fuel-commit invocation still exist.

- [ ] **Step 3: Remove the obsolete APIs**

Delete `_SIMULATED_REMAINING_FUEL`, dry-mass derivation, propellant message/timer/publisher, atomic fuel parameter validation, transport/supervisor `commit_hopper_fuel`, node-side commit exception handling, and `HOPPER_FUEL_COMMIT_FAILED`. Do not change goal-session cleanup, landed pose handoff, or platform selection.

- [ ] **Step 4: Run external Python closure and commit**

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation" \
python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test
git add ros2_ws/src/lunar_isaac_validation
git commit -m "refactor: stop publishing and committing simulated hopper fuel"
```

### Task 3: Cross-repository build and repeated-goal regression

**Files:**
- Modify: external validation documentation and launch instructions that mention fuel decrement.
- External artifacts only: `/home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/external/`.

**Interfaces:**
- Consumes: main repository Release install containing the updated messages and planner.
- Produces: an external branch proven source-compatible with the main feature branch.

- [ ] **Step 1: Build against the main feature install**

```bash
source /opt/ros/humble/setup.bash
source /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/main-install/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/external/log \
  build --merge-install \
  --base-paths "$PWD/ros2_ws/src" \
  --build-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/external/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/external/install
```

Expected: external packages compile against the delta-v-only `HopSegment`.

- [ ] **Step 2: Run all external tests**

```bash
colcon test \
  --build-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/external/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/external/install \
  --test-result-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/external/results
colcon test-result \
  --test-result-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/external/results \
  --verbose
```

Expected: zero errors and zero failures.

- [ ] **Step 3: Run the two-goal process regression**

In an isolated ROS domain, execute “select HOPPER -> Goal A -> parabola -> landed settle -> Goal B -> parabola -> landed settle”. Require two distinct plan/segment IDs, two movements, no stale `CANCELED`, no propellant Topic, and no fuel-related reason code.

- [ ] **Step 4: Audit and document**

```bash
if rg -n \
  "simulated_remaining_usable_fuel|hopper_fuel_commit|HOPPER_FUEL_COMMIT_FAILED" \
  ros2_ws/src/lunar_isaac_validation; then
  exit 1
fi
git diff --check
git status --short --branch
```

Update docs to state that delta-v is per-hop reachability evidence and repeated Goals do not share a resource budget.

- [ ] **Step 5: Commit documentation and leave branch ready**

```bash
git add docs ros2_ws/src/lunar_isaac_validation
git commit -m "docs: align RViz hopper workflow with no-fuel planning"
```

Do not merge or push automatically. Report the external branch commit and cross-repository test evidence to the main-repository qualification document.
