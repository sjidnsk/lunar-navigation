# Wheel Controller Motion Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing isolated Pure Pursuit controller to execute signed forward/reverse trajectories, in-place spin segments, and matching-plan cancellation without lateral motion.

**Architecture:** Parse the executable `MultiDOFJointTrajectory` into immutable planar samples while retaining the existing path-only compatibility mode. Add a cursor-based trajectory tracker that delegates translational segments to reverse-aware Pure Pursuit and zero-translation segments to bounded yaw control, then connect matching `plan_id` cancellation at the ROS boundary.

**Tech Stack:** ROS 2 Jazzy, rclpy, `lunar_planning_msgs/MotionReference`, `trajectory_msgs`, `std_msgs/String`, pytest, ament_cmake_pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-300m-jazzy-exploration-simulation-design.md`

## Global Constraints

- Preserve package/node names and sole command output `/Car/T5/Car_Cmd_Vel`.
- Preserve the current path-only forward Pure Pursuit behavior when `trajectory.points` is empty.
- Do not read timestamps, map versions, covariance, or freshness.
- Never publish nonzero `Twist.linear.y`.
- Cancellation only clears the active reference when the received nonempty plan ID matches it.
- Do not change planner or exploration algorithms.

---

## File Structure

- Modify `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/reference.py`: parse planar trajectory samples and signed longitudinal velocity.
- Modify `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/tracking.py`: cursor-based translation/spin tracking.
- Modify `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/node.py`: select trajectory/path mode and consume cancel messages.
- Modify `ros2_ws/src/lunar_pure_wheeled_controller/launch/pure_wheeled_controller.launch.py`: expose cancel and spin-control parameters.
- Modify `ros2_ws/src/lunar_pure_wheeled_controller/{CMakeLists.txt,package.xml}`: declare `std_msgs`.
- Modify the three existing controller test files; do not create a second controller package.

### Task 1: Executable trajectory parsing

**Files:**
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/reference.py`
- Test: `ros2_ws/src/lunar_pure_wheeled_controller/test/test_reference.py`

**Interfaces:**
- Produces: `TrajectorySample(x_m, y_m, yaw_rad, signed_speed_mps, yaw_rate_radps)`.
- Produces: `ParsedReference(plan_id, path_xy_yaw, trajectory_samples, reason)`.
- Consumes: the first transform and first velocity of each trajectory point.

- [ ] **Step 1: Add failing parser tests**

Add helpers that append trajectory points and assert:

```python
parsed = parse_reference(make_trajectory_reference([
    ((0.0, 0.0, 0.0), (0.2, 0.0), 0.0),
    ((-0.2, 0.0, 0.0), (-0.2, 0.0), 0.0),
]))
assert parsed.trajectory_samples[0].signed_speed_mps == 0.2
assert parsed.trajectory_samples[1].signed_speed_mps == -0.2
```

Also assert a map-frame velocity at yaw `pi/2` projects onto body forward, a spin point retains signed yaw rate, non-finite fields fail with `INVALID_REFERENCE`, a point with zero or multiple transforms/velocities fails, a nonempty malformed trajectory never falls back to `path_preview`, and the existing empty-trajectory path test remains valid.

- [ ] **Step 2: Run the focused test and confirm the missing-field failure**

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/controller/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/controller/install \
  --log-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/controller/log \
  --packages-up-to lunar_planning_msgs
source /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/controller/install/setup.bash
PYTHONPATH=ros2_ws/src/lunar_pure_wheeled_controller/python:$PYTHONPATH \
python3 -m pytest -q -p no:cacheprovider \
  ros2_ws/src/lunar_pure_wheeled_controller/test/test_reference.py
```

Expected: FAIL because `TrajectorySample` and `trajectory_samples` do not exist.

- [ ] **Step 3: Implement the immutable sample model and parser**

Use this public shape:

```python
@dataclass(frozen=True)
class TrajectorySample:
    x_m: float
    y_m: float
    yaw_rad: float
    signed_speed_mps: float
    yaw_rate_radps: float

@dataclass(frozen=True)
class ParsedReference:
    plan_id: str
    path_xy_yaw: tuple[tuple[float, float, float], ...]
    trajectory_samples: tuple[TrajectorySample, ...]
    reason: str | None
```

Normalize each quaternion exactly as the path parser does. For every trajectory point require exactly one transform and exactly one velocity, reject non-finite planar values, and calculate:

```python
signed_speed = velocity.linear.x * math.cos(yaw) + velocity.linear.y * math.sin(yaw)
```

Ignore stamp and `time_from_start`. If `trajectory.points` is nonempty, it is authoritative and every point must parse successfully; any failure returns `INVALID_REFERENCE` even when `path_preview` is valid. Only when `trajectory.points` is empty may a nonempty valid path enable forward-only compatibility mode.

- [ ] **Step 4: Run parser and legacy tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit the parser unit**

```bash
git add ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/reference.py \
  ros2_ws/src/lunar_pure_wheeled_controller/test/test_reference.py
git commit -m "feat: parse executable wheel trajectories"
```

### Task 2: Reverse-aware translation and in-place spin

**Files:**
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/tracking.py`
- Test: `ros2_ws/src/lunar_pure_wheeled_controller/test/test_tracking.py`

**Interfaces:**
- Consumes: `tuple[TrajectorySample, ...]`, `TrackingState`, `TrackingPolicy`, and current cursor.
- Produces: `TrajectoryTrackingResult(command: TrackingCommand, next_cursor: int)`.
- Preserves: `track_path(path, state, policy)` for forward-only compatibility.

- [ ] **Step 1: Add failing motion-mode tests**

Cover these exact outcomes:

```python
assert track_trajectory(reverse_samples, state(0, 0, 0), policy(), 0).command.linear_x_mps < 0.0
assert track_trajectory(reverse_left_arc, state(0, 0, 0), policy(), 0).command.angular_z_radps < 0.0
spin = track_trajectory(spin_left, state(0, 0, 0), policy(), 0).command
assert spin.linear_x_mps == 0.0 and spin.angular_z_radps > 0.0
```

Add right-spin, cursor advance, final pose-and-yaw completion, bound checks, path deviation stop, and a test asserting no command type contains lateral velocity. Add one combined trajectory test with `forward -> stop/spin -> reverse -> zero-speed terminal point` and assert the reverse segment remains negative.

- [ ] **Step 2: Run the tracker test and confirm the missing API failure**

Run:

```bash
source /opt/ros/jazzy/setup.bash
source /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/controller/install/setup.bash
PYTHONPATH=ros2_ws/src/lunar_pure_wheeled_controller/python:$PYTHONPATH \
python3 -m pytest -q -p no:cacheprovider \
  ros2_ws/src/lunar_pure_wheeled_controller/test/test_tracking.py
```

Expected: FAIL because `track_trajectory` is absent.

- [ ] **Step 3: Add cursor tracking and yaw control**

Extend `TrackingPolicy` with `spin_kp: float = 1.5` and
`translation_epsilon_m: float = 1.0e-3`.

Add immutable `TrajectoryTrackingResult(command: TrackingCommand, next_cursor: int)` and the function
`track_trajectory(samples: tuple[TrajectorySample, ...], state: TrackingState, policy: TrackingPolicy, cursor: int) -> TrajectoryTrackingResult`.

Advance past reached intermediate samples in trajectory order. Treat a target as spin when XY distance from its predecessor is at most `translation_epsilon_m` and yaw is outside tolerance. Spin with
`clip(spin_kp * normalized_yaw_error, +/-max_angular_radps)` and zero linear speed. For translation, choose the trajectory target after the cursor, use its signed-speed sign, clamp magnitude by
`max_linear_mps` and remaining distance, then retain `angular = linear * 2*local_y/L^2`. Resolve a zero-speed translation point within its current contiguous translation segment: prefer its own nonzero speed, otherwise scan forward to the first nonzero translation speed, then scan backward. Stop the scan at a spin segment or explicit direction switch; never inherit a forward sign across `forward -> spin -> reverse`.

- [ ] **Step 4: Run all pure Python controller tests**

Run:

```bash
source /opt/ros/jazzy/setup.bash
source /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/controller/install/setup.bash
PYTHONPATH=ros2_ws/src/lunar_pure_wheeled_controller/python:$PYTHONPATH \
python3 -m pytest -q -p no:cacheprovider \
  ros2_ws/src/lunar_pure_wheeled_controller/test/test_reference.py \
  ros2_ws/src/lunar_pure_wheeled_controller/test/test_tracking.py
```

Expected: PASS with existing forward tests unchanged.

- [ ] **Step 5: Commit the tracking unit**

```bash
git add ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/tracking.py \
  ros2_ws/src/lunar_pure_wheeled_controller/test/test_tracking.py
git commit -m "feat: execute reverse and spin wheel segments"
```

### Task 3: ROS mode selection and cancellation

**Files:**
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/node.py`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/launch/pure_wheeled_controller.launch.py`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/package.xml`
- Test: `ros2_ws/src/lunar_pure_wheeled_controller/test/test_node.py`

**Interfaces:**
- Consumes: `/Car/T4/execution/cancel` (`std_msgs/msg/String`).
- Publishes: only `/Car/T5/Car_Cmd_Vel` (`geometry_msgs/msg/Twist`).
- Parameters: `execution_cancel_topic`, `spin_kp`, `translation_epsilon_m`.

- [ ] **Step 1: Add failing node tests**

Create a reverse trajectory reference and assert negative `linear.x`; create a spin trajectory and assert zero `linear.x` plus signed `angular.z`; call `_on_cancel(String(data="wheel-plan"))` and assert immediate zero plus cleared active state; assert a different plan ID does not clear it; assert every published Twist has all components except `linear.x` and `angular.z` equal to zero.

- [ ] **Step 2: Run node tests and verify failure**

Run:

```bash
source /opt/ros/jazzy/setup.bash
source /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/controller/install/setup.bash
PYTHONPATH=ros2_ws/src/lunar_pure_wheeled_controller/python:$PYTHONPATH \
python3 -m pytest -q -p no:cacheprovider \
  ros2_ws/src/lunar_pure_wheeled_controller/test/test_node.py
```

Expected: FAIL because cancel subscription and trajectory mode are absent.

- [ ] **Step 3: Wire the node and dependencies**

Initialize `_trajectory_cursor = 0` on every accepted reference. In `_tick`, call `track_trajectory` when samples exist and otherwise call `track_path`. Add:

```python
def _on_cancel(self, message: String) -> None:
    if self._active is not None and message.data == self._active.plan_id:
        self._active = None
        self._trajectory_cursor = 0
        self._publish_twist()
```

Declare absolute `execution_cancel_topic` with default `/Car/T4/execution/cancel`; add the subscription, launch arguments, `find_package(std_msgs REQUIRED)`, `<depend>std_msgs</depend>`, and ament dependency.

- [ ] **Step 4: Build and run all package tests**

Run:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/controller/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/controller/install \
  --log-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/controller/log \
  --packages-up-to lunar_pure_wheeled_controller
source /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/controller/install/setup.bash
colcon test --base-paths ros2_ws/src \
  --build-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/controller/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/controller/install \
  --log-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/controller/log \
  --packages-select lunar_pure_wheeled_controller --event-handlers console_direct+
colcon test-result \
  --test-result-base /home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/controller/build \
  --all --verbose
```

Expected: controller package builds and all controller pytest suites pass.

- [ ] **Step 5: Commit the ROS integration**

```bash
git add ros2_ws/src/lunar_pure_wheeled_controller
git commit -m "feat: connect wheel execution cancellation"
```
