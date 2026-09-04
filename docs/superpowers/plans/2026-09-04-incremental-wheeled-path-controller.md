# Incremental Wheeled Path Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the isolated wheeled controller so an explicitly started node can track the incremental navigator's local `nav_msgs/msg/Path` and publish bounded `/Car/T5/Car_Cmd_Vel` commands.

**Architecture:** Keep the existing controller package and make its two inputs mutually exclusive: legacy `MotionReference` remains the default, while `incremental_path` consumes only `/Car/T4/planning/local_path`, odometry, and the direct `map <- odom` transform from `/tf`. ROS message/frame conversion stays in small adapter modules; the Pure Pursuit core continues to consume only same-frame numeric path/state data.

**Tech Stack:** ROS 2 Jazzy and Humble, rclpy, `nav_msgs/msg/Path`, `tf2_msgs/msg/TFMessage`, `geometry_msgs/msg/Twist`, pytest, ament_cmake_pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-incremental-wheeled-path-controller-design.md`

## Global Constraints

- Work only on `feat/incremental-wheeled-path-controller` in its isolated worktree; do not touch the dirty integration worktree or the read-only import branch.
- Preserve `motion_reference` as the default legacy mode and preserve all existing legacy controller tests.
- In `incremental_path` mode, subscribe only to `/Car/T4/planning/local_path`, `/Car/T3/localization/odometry`, and `/tf`; do not add planner, map, Action, exploration, revision, timestamp, covariance, or freshness inputs.
- Keep `/Car/T5/Car_Cmd_Vel` as the sole output and leave `linear.y`, `linear.z`, `angular.x`, and `angular.y` at zero.
- Do not add the controller to `exploration_navigation.launch.py`; vehicle control remains an explicit, separate launch operation.
- Use `map` as the only nonempty incremental local-path frame and use only a direct `map <- odom` transform; do not add TF graph traversal, timestamp synchronization, or historical-transform behavior.
- Empty or malformed paths, path deviation, completion, matching legacy cancellation, and shutdown clear the active target and publish zero. Temporarily unavailable odometry or direct TF publish zero while retaining the latest valid path.
- Use `python3 -m pytest` from the repository root for Python tests. Keep Jazzy/Humble build, install, and log directories outside the repository.
- Treat Jazzy, Humble, Orin, DDS, real TF, chassis, emergency stop, and vehicle closed-loop evidence as separate layers; do not claim an unrun layer.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/geometry.py` | Finite quaternion-to-yaw conversion shared by reference, Path, and TF adapters. |
| `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/incremental_path.py` | Validate the standard local `Path` and represent either a trackable map-frame path or an explicit clear. |
| `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/frame.py` | Extract a direct planar `map <- odom` transform and convert odometry into a map-frame `TrackingState`. |
| `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/reference.py` | Retain legacy `MotionReference` parsing while using the shared yaw conversion. |
| `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/tracking.py` | Add final-yaw spin convergence to path-only Pure Pursuit. |
| `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/node.py` | Select one input mode, own ROS subscriptions/state, and publish commands. |
| `ros2_ws/src/lunar_pure_wheeled_controller/launch/pure_wheeled_controller.launch.py` | Expose explicit mode/path/TF launch arguments without changing the default mode. |
| `ros2_ws/src/lunar_pure_wheeled_controller/{package.xml,CMakeLists.txt}` | Declare `tf2_msgs` and register new Python tests. |
| `ros2_ws/src/lunar_pure_wheeled_controller/test/test_incremental_path.py` | Path parser and direct-frame adapter regression coverage. |
| `ros2_ws/src/lunar_pure_wheeled_controller/test/test_node.py` | Incremental mode, stop/recovery, QoS, and legacy-mode boundary coverage. |
| `config/{exploration_navigation.yaml,incremental_navigation_interfaces.yaml}` | Make `local_path` ownership, frame, QoS, and navigation parameter explicit. |
| `tests/{test_incremental_navigation_interface_contract.py,launch/test_exploration_navigation_launch.py}` | Guard config/publisher/launch consistency and no automatic controller startup. |
| `README.md`, `docs/操作指令.md` | Document explicit controller launch, TF prerequisite, command exclusivity, and evidence boundary. |

### Task 1: Parse incremental local paths through a focused adapter

**Files:**
- Create: `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/geometry.py`
- Create: `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/incremental_path.py`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/reference.py`
- Create: `ros2_ws/src/lunar_pure_wheeled_controller/test/test_incremental_path.py`

**Interfaces:**
- Consumes: `nav_msgs.msg.Path` with `header.frame_id == "map"` for a nonempty path.
- Produces: `ParsedIncrementalPath(path_xy_yaw: tuple[tuple[float, float, float], ...], clear: bool, reason: str | None)` from `parse_incremental_path(path: Path)`.
- Produces: `yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float | None` for all adapter modules.

- [ ] **Step 1: Write failing parser tests**

```python
def make_path(*, frame_id: str, points: list[tuple[float, float, float]]) -> Path:
    path = Path()
    path.header.frame_id = frame_id
    for x, y, yaw in points:
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        path.poses.append(pose)
    return path


def test_nonempty_map_path_becomes_finite_xy_yaw_samples() -> None:
    path = make_path(frame_id="map", points=[(0.0, 0.0, 0.0), (1.0, 0.0, math.pi / 2.0)])
    parsed = parse_incremental_path(path)
    assert parsed.reason is None
    assert not parsed.clear
    assert parsed.path_xy_yaw[0] == pytest.approx((0.0, 0.0, 0.0))
    assert parsed.path_xy_yaw[1] == pytest.approx((1.0, 0.0, math.pi / 2.0))


def test_empty_path_means_clear_even_without_a_header() -> None:
    parsed = parse_incremental_path(Path())
    assert parsed.clear
    assert parsed.reason is None
    assert parsed.path_xy_yaw == ()


def test_nonempty_path_with_wrong_frame_or_bad_quaternion_is_rejected() -> None:
    assert parse_incremental_path(make_path(frame_id="odom", points=[(0.0, 0.0, 0.0)])).reason == "INVALID_PATH"
    malformed = make_path(frame_id="map", points=[(0.0, 0.0, 0.0)])
    malformed.poses[0].pose.orientation.w = 0.0
    assert parse_incremental_path(malformed).reason == "INVALID_PATH"
```

- [ ] **Step 2: Run the parser tests and verify the expected red failure**

Run:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/kai/WS/lunar-navigation/lunar-runtime/.worktrees/incremental-wheeled-path-controller/ros2_ws
LUNAR_PATH_BUILD=$(mktemp -d /tmp/lunar-incremental-path-XXXXXX)
colcon --log-base "$LUNAR_PATH_BUILD/log" build --base-paths src --packages-select lunar_planning_msgs --build-base "$LUNAR_PATH_BUILD/build" --install-base "$LUNAR_PATH_BUILD/install"
source "$LUNAR_PATH_BUILD/install/setup.bash"
cd ..
PYTHONPATH=ros2_ws/src/lunar_pure_wheeled_controller/python:$PYTHONPATH python3 -m pytest -q -p no:cacheprovider ros2_ws/src/lunar_pure_wheeled_controller/test/test_incremental_path.py
```

Expected: FAIL because `lunar_pure_wheeled_controller.incremental_path` does not exist.

- [ ] **Step 3: Implement the minimal geometry and Path adapters**

```python
# geometry.py
def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float | None:
    values = (float(qx), float(qy), float(qz), float(qw))
    if not all(math.isfinite(value) for value in values):
        return None
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0.0:
        return None
    qx, qy, qz, qw = (value / norm for value in values)
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return yaw if math.isfinite(yaw) else None


# incremental_path.py
@dataclass(frozen=True)
class ParsedIncrementalPath:
    path_xy_yaw: tuple[tuple[float, float, float], ...]
    clear: bool
    reason: str | None


def parse_incremental_path(path: Path) -> ParsedIncrementalPath:
    if not path.poses:
        return ParsedIncrementalPath((), True, None)
    if path.header.frame_id != "map":
        return ParsedIncrementalPath((), False, "INVALID_PATH")
    samples: list[tuple[float, float, float]] = []
    for stamped_pose in path.poses:
        pose = stamped_pose.pose
        try:
            x, y = float(pose.position.x), float(pose.position.y)
            yaw = yaw_from_quaternion(
                float(pose.orientation.x), float(pose.orientation.y),
                float(pose.orientation.z), float(pose.orientation.w),
            )
        except (TypeError, ValueError):
            return ParsedIncrementalPath((), False, "INVALID_PATH")
        if not math.isfinite(x) or not math.isfinite(y) or yaw is None:
            return ParsedIncrementalPath((), False, "INVALID_PATH")
        samples.append((x, y, yaw))
    return ParsedIncrementalPath(tuple(samples), False, None)
```

Replace the private quaternion math in `reference.py` with the shared helper without changing legacy return values or trajectory precedence.

- [ ] **Step 4: Run new and legacy parser tests to verify green**

Run:

```bash
PYTHONPATH=ros2_ws/src/lunar_pure_wheeled_controller/python:$PYTHONPATH python3 -m pytest -q -p no:cacheprovider ros2_ws/src/lunar_pure_wheeled_controller/test/test_incremental_path.py ros2_ws/src/lunar_pure_wheeled_controller/test/test_reference.py
```

Expected: PASS; an empty local path is a clear, malformed nonempty paths are rejected, and all legacy reference tests remain green.

- [ ] **Step 5: Commit the adapter unit**

```bash
git add ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/geometry.py \
  ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/incremental_path.py \
  ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/reference.py \
  ros2_ws/src/lunar_pure_wheeled_controller/test/test_incremental_path.py
git commit -m "feat: parse incremental wheel paths"
```

### Task 2: Make path-only terminal yaw converge

**Files:**
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/tracking.py:212-244`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/test/test_tracking.py:107-112`

**Interfaces:**
- Consumes: `track_path(path: PathXYYaw, state: TrackingState, policy: TrackingPolicy)`.
- Produces: a non-complete `TrackingCommand(0.0, signed_angular, False, None)` when only final yaw remains, and a complete zero command only when both terminal tolerances hold.

- [ ] **Step 1: Add the failing final-yaw test**

```python
def test_path_at_terminal_position_spins_until_terminal_yaw_is_aligned() -> None:
    command = track_path(((1.0, 0.0, math.pi / 2.0),), state(1.0, 0.0, 0.0), policy())
    assert not command.complete
    assert command.failure_reason is None
    assert command.linear_x_mps == 0.0
    assert 0.0 < command.angular_z_radps <= policy().max_angular_radps
```

- [ ] **Step 2: Run the focused test and verify the expected red failure**

Run:

```bash
PYTHONPATH=ros2_ws/src/lunar_pure_wheeled_controller/python:$PYTHONPATH python3 -m pytest -q -p no:cacheprovider ros2_ws/src/lunar_pure_wheeled_controller/test/test_tracking.py::test_path_at_terminal_position_spins_until_terminal_yaw_is_aligned
```

Expected: FAIL because the current path-only tracker returns zero angular velocity before reporting completion.

- [ ] **Step 3: Add the minimal final-yaw branch**

```python
if goal_distance <= policy.goal_position_tolerance_m:
    yaw_error = _normalized_yaw_error(goal_yaw, yaw)
    if abs(yaw_error) <= policy.goal_yaw_tolerance_rad:
        return TrackingCommand(0.0, 0.0, True, None)
    angular = _clip(policy.spin_kp * yaw_error, policy.max_angular_radps)
    return TrackingCommand(0.0, angular, False, None)
```

Keep the existing nearest-point, deviation, lookahead, curvature, and speed-clamp behavior unchanged for paths whose terminal position has not been reached.

- [ ] **Step 4: Run the complete tracking suite and verify green**

Run:

```bash
PYTHONPATH=ros2_ws/src/lunar_pure_wheeled_controller/python:$PYTHONPATH python3 -m pytest -q -p no:cacheprovider ros2_ws/src/lunar_pure_wheeled_controller/test/test_tracking.py
```

Expected: PASS; forward, reverse, spin-trajectory, path-deviation, and new path-only terminal-yaw cases all remain bounded.

- [ ] **Step 5: Commit the terminal-yaw behavior**

```bash
git add ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/tracking.py \
  ros2_ws/src/lunar_pure_wheeled_controller/test/test_tracking.py
git commit -m "fix: align terminal yaw for path tracking"
```

### Task 3: Adapt direct map-to-odom TF into controller state

**Files:**
- Create: `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/frame.py`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/test/test_incremental_path.py`

**Interfaces:**
- Consumes: `tf2_msgs.msg.TFMessage` containing one direct `TransformStamped` with `header.frame_id == "map"` and `child_frame_id == "odom"`.
- Produces: `MapFromOdom(x_m: float, y_m: float, yaw_rad: float)` and `MapFromOdomUpdate(found: bool, transform: MapFromOdom | None)` from `parse_map_from_odom(message: TFMessage) -> MapFromOdomUpdate`.
- Produces: `map_tracking_state(odometry: Odometry, transform: MapFromOdom) -> TrackingState | None`.

- [ ] **Step 1: Add failing direct-frame tests**

```python
def make_odometry(*, x: float, y: float, yaw: float) -> Odometry:
    odometry = Odometry()
    odometry.pose.pose.position.x = x
    odometry.pose.pose.position.y = y
    odometry.pose.pose.orientation.z = math.sin(yaw / 2.0)
    odometry.pose.pose.orientation.w = math.cos(yaw / 2.0)
    return odometry


def make_tf_message(*, parent: str, child: str, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> TFMessage:
    message = TFMessage()
    transform = TransformStamped()
    transform.header.frame_id = parent
    transform.child_frame_id = child
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.rotation.z = math.sin(yaw / 2.0)
    transform.transform.rotation.w = math.cos(yaw / 2.0)
    message.transforms.append(transform)
    return message


def test_direct_map_from_odom_transforms_position_and_yaw() -> None:
    update = parse_map_from_odom(make_tf_message(parent="map", child="odom", x=10.0, y=20.0, yaw=math.pi / 2.0))
    assert update.found and update.transform is not None
    state = map_tracking_state(make_odometry(x=2.0, y=0.0, yaw=0.0), update.transform)
    assert state is not None
    assert state.x_m == pytest.approx(10.0)
    assert state.y_m == pytest.approx(22.0)
    assert state.yaw_rad == pytest.approx(math.pi / 2.0)


def test_unrelated_or_invalid_tf_cannot_create_a_map_transform() -> None:
    assert parse_map_from_odom(TFMessage()) == MapFromOdomUpdate(False, None)
    invalid = make_tf_message(parent="map", child="odom")
    invalid.transforms[0].transform.rotation.w = 0.0
    assert parse_map_from_odom(invalid) == MapFromOdomUpdate(True, None)
```

- [ ] **Step 2: Run the frame tests and verify the expected red failure**

Run:

```bash
PYTHONPATH=ros2_ws/src/lunar_pure_wheeled_controller/python:$PYTHONPATH python3 -m pytest -q -p no:cacheprovider ros2_ws/src/lunar_pure_wheeled_controller/test/test_incremental_path.py::test_direct_map_from_odom_transforms_position_and_yaw ros2_ws/src/lunar_pure_wheeled_controller/test/test_incremental_path.py::test_unrelated_or_invalid_tf_cannot_create_a_map_transform
```

Expected: FAIL because `frame.py` and its public functions do not exist.

- [ ] **Step 3: Implement the direct planar transform adapter**

```python
@dataclass(frozen=True)
class MapFromOdom:
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class MapFromOdomUpdate:
    found: bool
    transform: MapFromOdom | None


def parse_map_from_odom(message: TFMessage) -> MapFromOdomUpdate:
    matching = [
        item for item in message.transforms
        if item.header.frame_id == "map" and item.child_frame_id == "odom"
    ]
    if not matching:
        return MapFromOdomUpdate(False, None)
    item = matching[-1]
    translation = item.transform.translation
    rotation = item.transform.rotation
    try:
        x_m, y_m = float(translation.x), float(translation.y)
        yaw_rad = yaw_from_quaternion(
            float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)
        )
    except (TypeError, ValueError):
        return MapFromOdomUpdate(True, None)
    if not math.isfinite(x_m) or not math.isfinite(y_m) or yaw_rad is None:
        return MapFromOdomUpdate(True, None)
    return MapFromOdomUpdate(True, MapFromOdom(x_m, y_m, yaw_rad))


def map_tracking_state(odometry: Odometry, transform: MapFromOdom) -> TrackingState | None:
    position = odometry.pose.pose.position
    orientation = odometry.pose.pose.orientation
    try:
        odometry_x, odometry_y = float(position.x), float(position.y)
        yaw = yaw_from_quaternion(
            float(orientation.x), float(orientation.y),
            float(orientation.z), float(orientation.w),
        )
    except (TypeError, ValueError):
        return None
    if not math.isfinite(odometry_x) or not math.isfinite(odometry_y) or yaw is None:
        return None
    cos_yaw = math.cos(transform.yaw_rad)
    sin_yaw = math.sin(transform.yaw_rad)
    return TrackingState(
        transform.x_m + cos_yaw * odometry_x - sin_yaw * odometry_y,
        transform.y_m + sin_yaw * odometry_x + cos_yaw * odometry_y,
        math.atan2(math.sin(transform.yaw_rad + yaw), math.cos(transform.yaw_rad + yaw)),
    )
```

`parse_map_from_odom` scans only the current `TFMessage` for the direct map/odom edge. A message without that edge leaves the node's last valid transform unchanged; a matching malformed edge returns `found=True, transform=None` so the node clears the unusable transform and stops until a valid direct edge arrives.

- [ ] **Step 4: Run all adapter tests and verify green**

Run:

```bash
PYTHONPATH=ros2_ws/src/lunar_pure_wheeled_controller/python:$PYTHONPATH python3 -m pytest -q -p no:cacheprovider ros2_ws/src/lunar_pure_wheeled_controller/test/test_incremental_path.py
```

Expected: PASS; the conversion is planar, directionally correct, and has no multi-hop or timestamp behavior.

- [ ] **Step 5: Commit the frame adapter**

```bash
git add ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/frame.py \
  ros2_ws/src/lunar_pure_wheeled_controller/test/test_incremental_path.py
git commit -m "feat: adapt map odometry for path tracking"
```

### Task 4: Add the explicit incremental controller mode

**Files:**
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/node.py`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/launch/pure_wheeled_controller.launch.py`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/{package.xml,CMakeLists.txt}`
- Modify: `ros2_ws/src/lunar_pure_wheeled_controller/test/test_node.py`

**Interfaces:**
- Consumes in `motion_reference`: existing `MotionReference`, odometry, and matching legacy cancel topics.
- Consumes in `incremental_path`: `Path` on `path_topic`, `Odometry` on `odometry_topic`, and `TFMessage` on `tf_topic`.
- Produces: `incremental_path_qos() -> QoSProfile` with Reliable, Transient Local, KeepLast depth 1.
- Produces: zero or bounded `Twist` through the existing command publisher, never both input modes at once.

- [ ] **Step 1: Add failing incremental-node tests**

```python
def make_path(*, frame_id: str, points: list[tuple[float, float, float]]) -> Path:
    path = Path()
    path.header.frame_id = frame_id
    for x, y, yaw in points:
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        path.poses.append(pose)
    return path


def make_direct_map_from_odom(*, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> TFMessage:
    message = TFMessage()
    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.child_frame_id = "odom"
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.rotation.z = math.sin(yaw / 2.0)
    transform.transform.rotation.w = math.cos(yaw / 2.0)
    message.transforms.append(transform)
    return message


def test_incremental_path_mode_tracks_map_path_after_direct_tf_arrives(
    incremental_controller_with_observer,
) -> None:
    controller, observer, received = incremental_controller_with_observer
    controller._on_path(make_path(frame_id="map", points=[(10.0, 0.0, 0.0), (11.0, 0.0, 0.0)]))
    controller._on_odometry(make_odometry(x=0.0))
    controller._on_tf(make_direct_map_from_odom(x=10.0))
    controller._tick()
    wait_for_twists(controller, observer, received)
    assert received[-1].linear.x > 0.0


def test_incremental_empty_path_stops_and_missing_tf_preserves_latest_path(
    incremental_controller_with_observer,
) -> None:
    controller, observer, received = incremental_controller_with_observer
    controller._on_path(make_path(frame_id="map", points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]))
    controller._on_odometry(make_odometry(x=0.0))
    controller._tick()
    wait_for_twists(controller, observer, received)
    assert controller._active is not None
    assert received[-1].linear.x == 0.0
    controller._on_path(Path())
    wait_for_twists(controller, observer, received, count=2)
    assert controller._active is None
    assert received[-1].linear.x == 0.0


def test_incremental_path_qos_is_reliable_transient_local_keep_last_one() -> None:
    qos = incremental_path_qos()
    assert qos.reliability == ReliabilityPolicy.RELIABLE
    assert qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert qos.depth == 1
```

Define `incremental_controller_with_observer` by reusing the existing controller/observer cleanup pattern but initializing rclpy with `-p input_mode:=incremental_path`; it yields `(controller, observer, received)` and always destroys the controller before `rclpy.shutdown()`. Add a parameterized constructor test that rejects an unknown `input_mode`, a relative `path_topic`, and a relative `tf_topic`. Retain the existing legacy default-topic integration test unchanged.

- [ ] **Step 2: Run the focused node tests and verify the expected red failure**

Run:

```bash
PYTHONPATH=ros2_ws/src/lunar_pure_wheeled_controller/python:$PYTHONPATH python3 -m pytest -q -p no:cacheprovider ros2_ws/src/lunar_pure_wheeled_controller/test/test_node.py -k 'incremental_path or input_mode'
```

Expected: FAIL because `input_mode`, `path_topic`, `tf_topic`, callbacks, and incremental QoS are absent.

- [ ] **Step 3: Implement mutually exclusive subscriptions and stop/recovery behavior**

```python
input_mode = self.declare_parameter("input_mode", "motion_reference").value
if input_mode not in {"motion_reference", "incremental_path"}:
    raise ValueError("input_mode must be motion_reference or incremental_path")

if input_mode == "motion_reference":
    self.create_subscription(MotionReference, reference_topic, self._on_reference, 10)
    self.create_subscription(String, cancel_topic, self._on_cancel, 10)
else:
    self.create_subscription(Path, path_topic, self._on_path, incremental_path_qos())
    self.create_subscription(TFMessage, tf_topic, self._on_tf, 10)
```

In incremental mode, `_on_path` atomically replaces a valid nonempty parsed path; an empty or malformed path clears it and immediately publishes zero. `_tick` builds a map-frame state only when both odometry and a direct transform are available. If either is absent, publish zero without clearing the active path. Path deviation or completion clears the active path after publishing the final command. Keep legacy `ParsedReference` trajectory selection and matching cancellation behavior unchanged.

Add `tf2_msgs` to `package.xml`, `find_package(tf2_msgs REQUIRED)` to `CMakeLists.txt`, and register `test_incremental_path.py`. Add `input_mode`, `path_topic`, and `tf_topic` to the existing launch argument map with default values `motion_reference`, `/Car/T4/planning/local_path`, and `/tf`.

- [ ] **Step 4: Run all controller Python tests and build the package**

Run:

```bash
PYTHONPATH=ros2_ws/src/lunar_pure_wheeled_controller/python:$PYTHONPATH python3 -m pytest -q -p no:cacheprovider ros2_ws/src/lunar_pure_wheeled_controller/test
source /opt/ros/jazzy/setup.bash
cd ros2_ws
LUNAR_CTRL_BUILD=$(mktemp -d /tmp/lunar-incremental-controller-XXXXXX)
colcon --log-base "$LUNAR_CTRL_BUILD/log" build --base-paths src --packages-up-to lunar_pure_wheeled_controller --build-base "$LUNAR_CTRL_BUILD/build" --install-base "$LUNAR_CTRL_BUILD/install" --event-handlers console_direct+
source "$LUNAR_CTRL_BUILD/install/setup.bash"
colcon --log-base "$LUNAR_CTRL_BUILD/log-test" test --base-paths src --packages-select lunar_pure_wheeled_controller --build-base "$LUNAR_CTRL_BUILD/build" --install-base "$LUNAR_CTRL_BUILD/install" --event-handlers console_direct+ --return-code-on-test-failure
colcon test-result --test-result-base "$LUNAR_CTRL_BUILD/build" --verbose
```

Expected: PASS with controller parser, tracking, frame, node, launch dependencies, and legacy behavior all green.

- [ ] **Step 5: Commit the mode integration**

```bash
git add ros2_ws/src/lunar_pure_wheeled_controller/python/lunar_pure_wheeled_controller/node.py \
  ros2_ws/src/lunar_pure_wheeled_controller/launch/pure_wheeled_controller.launch.py \
  ros2_ws/src/lunar_pure_wheeled_controller/package.xml \
  ros2_ws/src/lunar_pure_wheeled_controller/CMakeLists.txt \
  ros2_ws/src/lunar_pure_wheeled_controller/test/test_node.py
git commit -m "feat: track incremental local wheel paths"
```

### Task 5: Close configuration, operator, and static interface contracts

**Files:**
- Modify: `config/exploration_navigation.yaml`
- Modify: `config/incremental_navigation_interfaces.yaml`
- Modify: `tests/test_incremental_navigation_interface_contract.py`
- Modify: `tests/launch/test_exploration_navigation_launch.py`
- Modify: `README.md`
- Modify: `docs/操作指令.md`

**Interfaces:**
- Produces: `navigation.local_path_topic: /Car/T4/planning/local_path` passed by the existing `**dict(config["navigation"])` mechanism.
- Produces: `outputs.local_path` with `nav_msgs/msg/Path`, owner `lunar_incremental_navigation_ros`, frame `map`, and Reliable + Transient Local + KeepLast(1).
- Preserves: exactly two nodes in incremental `exploration_navigation.launch.py` and no `/Car/T5/Car_Cmd_Vel` publication there.

- [ ] **Step 1: Extend static tests before editing contracts and docs**

```python
def test_incremental_interfaces_publish_a_durable_controller_local_path() -> None:
    interfaces = _yaml(INCREMENTAL_INTERFACES)
    assert interfaces["outputs"]["local_path"] == {
        "name": "/Car/T4/planning/local_path",
        "type": "nav_msgs/msg/Path",
        "owner": "lunar_incremental_navigation_ros",
        "frame": "map",
        "qos": {"reliability": "reliable", "durability": "transient_local", "history": "keep_last", "depth": 1},
    }


def test_incremental_launch_passes_local_path_but_starts_no_controller(monkeypatch) -> None:
    module, Node, _ = _launch_module(monkeypatch)
    actions = _compose(module)
    navigation = actions[0].kwargs["parameters"][0]
    assert navigation["local_path_topic"] == "/Car/T4/planning/local_path"
    assert [action.kwargs["package"] for action in actions] == [
        "lunar_incremental_navigation_ros", "lunar_pure_exploration_ros"
    ]
```

- [ ] **Step 2: Run static tests and verify the expected red failure**

Run:

```bash
python3 -m pytest -q -p no:cacheprovider tests/test_incremental_navigation_interface_contract.py tests/launch/test_exploration_navigation_launch.py
```

Expected: FAIL because `local_path` is not yet represented in both YAML contracts or injected from the top-level configuration.

- [ ] **Step 3: Apply the minimal contract and documentation updates**

```yaml
# config/exploration_navigation.yaml
navigation:
  local_path_topic: /Car/T4/planning/local_path

# config/incremental_navigation_interfaces.yaml
  local_path:
    name: /Car/T4/planning/local_path
    type: nav_msgs/msg/Path
    owner: lunar_incremental_navigation_ros
    frame: map
    qos:
      reliability: reliable
      durability: transient_local
      history: keep_last
      depth: 1
```

Update the expected output set and exact YAML dictionaries in the static contract test. Update `README.md` and `docs/操作指令.md` with one explicit standalone launch command:

```bash
ros2 launch lunar_pure_wheeled_controller pure_wheeled_controller.launch.py \
  input_mode:=incremental_path \
  path_topic:=/Car/T4/planning/local_path \
  odometry_topic:=/Car/T3/localization/odometry \
  tf_topic:=/tf
```

State beside the command that it starts the sole `/Car/T5/Car_Cmd_Vel` publisher, requires the authorized vehicle-control/stop procedure, must not run alongside another command publisher, and is not started by the incremental navigation launch. Document that the controller follows only map-frame local paths, uses the direct map/odom TF, and stops on a cleared path.

- [ ] **Step 4: Run static and affected launch contracts to verify green**

Run:

```bash
python3 -m pytest -q -p no:cacheprovider tests/test_incremental_navigation_interface_contract.py tests/launch/test_exploration_navigation_launch.py tests/test_launch_contract.py
git diff --check
```

Expected: PASS; all local-path metadata is consistent and the normal incremental launch still creates only navigation and exploration nodes.

- [ ] **Step 5: Commit contracts and operator documentation**

```bash
git add config/exploration_navigation.yaml \
  config/incremental_navigation_interfaces.yaml \
  tests/test_incremental_navigation_interface_contract.py \
  tests/launch/test_exploration_navigation_launch.py \
  README.md docs/操作指令.md
git commit -m "docs: document incremental wheel controller operation"
```

### Task 6: Run scoped cross-layer verification and record evidence boundaries

**Files:**
- Modify: `docs/validation/2026-09-04-incremental-wheeled-path-controller.md`

**Interfaces:**
- Consumes: completed controller package, incremental interface config, and test results.
- Produces: a concise evidence matrix separating static, Jazzy, Humble, Orin/DDS, and vehicle evidence.

- [ ] **Step 1: Run the complete repository-level affected Python suite**

Run:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m pytest -q -p no:cacheprovider \
  ros2_ws/src/lunar_pure_wheeled_controller/test \
  tests/test_incremental_navigation_interface_contract.py \
  tests/launch/test_exploration_navigation_launch.py \
  tests/test_launch_contract.py
```

Expected: PASS with no test collection errors.

- [ ] **Step 2: Build and test affected packages in a fresh Jazzy prefix**

Run:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
LUNAR_VERIFY_BUILD=$(mktemp -d /tmp/lunar-incremental-controller-verify-XXXXXX)
colcon --log-base "$LUNAR_VERIFY_BUILD/log" build --base-paths src --packages-up-to lunar_incremental_navigation_ros lunar_pure_wheeled_controller --build-base "$LUNAR_VERIFY_BUILD/build" --install-base "$LUNAR_VERIFY_BUILD/install" --event-handlers console_direct+
source "$LUNAR_VERIFY_BUILD/install/setup.bash"
colcon --log-base "$LUNAR_VERIFY_BUILD/log-test" test --base-paths src --packages-select lunar_pure_wheeled_controller lunar_incremental_navigation_ros --build-base "$LUNAR_VERIFY_BUILD/build" --install-base "$LUNAR_VERIFY_BUILD/install" --event-handlers console_direct+ --return-code-on-test-failure
colcon test-result --test-result-base "$LUNAR_VERIFY_BUILD/build" --verbose
```

Expected: PASS or an exact environment/dependency failure recorded without mislabeling it as target validation.

- [ ] **Step 3: Attempt the matching Humble package build/test in an external container**

Run:

```bash
docker run --rm -v "$PWD/..:/workspace:ro" -w /workspace/ros2_ws osrf/ros:humble-desktop-full-jammy bash -lc '
  apt-get update && apt-get install -y python3-colcon-common-extensions ros-humble-grid-map-msgs &&
  source /opt/ros/humble/setup.bash &&
  colcon build --base-paths src --packages-up-to lunar_incremental_navigation_ros lunar_pure_wheeled_controller --build-base /tmp/lunar-build --install-base /tmp/lunar-install --event-handlers console_direct+ &&
  source /tmp/lunar-install/setup.bash &&
  colcon test --base-paths src --packages-select lunar_pure_wheeled_controller lunar_incremental_navigation_ros --build-base /tmp/lunar-build --install-base /tmp/lunar-install --event-handlers console_direct+ --return-code-on-test-failure &&
  colcon test-result --test-result-base /tmp/lunar-build --verbose'
```

Expected: record either a clean Humble package result or the exact image/dependency blocker. Do not use Jazzy results as Humble proof.

- [ ] **Step 4: Write the evidence note and verify documentation formatting**

Record exact commands, pass/fail counts, build environment, and the remaining `NOT_RUN` boundaries: Jetson AGX Orin, DDS, real planner/TF topics, chassis command arbitration, emergency stop, and real vehicle path tracking. Do not claim formal planning success or vehicle readiness from controller tests.

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and only this feature's files modified.

- [ ] **Step 5: Commit validation evidence**

```bash
git add docs/validation/2026-09-04-incremental-wheeled-path-controller.md
git commit -m "test: record incremental controller validation"
```
