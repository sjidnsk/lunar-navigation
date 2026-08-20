# Task3 Live Localization and Execution Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 直接消费课题三实时 0.2 m `odom` 局部图、车体 Odometry 与 `map -> odom -> base_link` TF，生成规划器可安全消费的十层局部图和定位状态，并提供可启动的 Task3 规划联调入口。

**Architecture:** 保留现有 `luna_t3_map_adapter` 的只读、ROI 有界全局 SQLite 适配职责，在同一包中新增局部证据转换、局部 ROS 节点和定位状态校验节点。课题三现在直接提供 `map -> odom -> base_link` 和 `odom -> base_link` 车体 Odometry，本项目不再转换相机位姿、复制 Odometry 或发布 TF。部署运行时通过明确的 `input_mode: external_canonical | task3_adapted` 选择是否启动适配器；`PlanMotion` 仍是唯一的高层输出，控制器契约不明时不实现或猜测底层运动命令。

**Tech Stack:** ROS 2 Humble、Python 3、rclpy、grid_map_msgs、nav_msgs、NumPy、SQLite、ament_cmake_python、pytest、ament_cmake_pytest。

**Spec:** `docs/superpowers/specs/2026-08-20-task3-live-localization-execution-integration-design.md`

## Global Constraints

- 不修改课题三 `/Car/T3/*` topic、SQLite、WAL/SHM 或数据所有权。
- 课题三局部输入和输出均严格为 `0.2 m`、`frame_id=odom`、单位朝向；不得插值、重采样、改 frame 或改变格数。
- 规划器输入保持 `lunar-external-interfaces/v5` 的十层、`map -> odom -> base_link` 和新鲜度合同。
- 缺失高度范围、方差、有效性或坐标证据的格必须 fail-closed，不能以 0、空数组或虚构质量值替代。
- 首轮只运行 `policy.mode: fallback`；非 fallback 仍保持 `POLICY_RUNTIME_UNBOUND`。
- 外部控制器拥有底层命令和 `/execution/motion_feedback` 的生产权；本计划不猜测控制 topic。
- 运行时 bundle 必须显式 allowlist 新包，且不得包含 tests、docs、build、install、log、cache 或训练产物。
- 所有 ROS 构建/回放验证在 Ubuntu 22.04 + ROS 2 Humble 进行；Windows 不作为该项权威验证环境。

---

## File Structure

| 路径 | 职责 |
| --- | --- |
| `ros2_ws/src/luna_t3_map_adapter/python/luna_t3_map_adapter/grid_map_codec.py` | GridMap 与 canonical NumPy 层之间的确定性编解码，复用现有 ring-buffer 约定。 |
| `ros2_ws/src/luna_t3_map_adapter/python/luna_t3_map_adapter/local_grid_conversion.py` | 纯局部证据模型、十层派生和 observation ledger，不依赖 ROS Node。 |
| `ros2_ws/src/luna_t3_map_adapter/python/luna_t3_map_adapter/local_map_adapter_node.py` | Task3 `odom` 局部图订阅、只读 `map -> odom` evidence 索引、tile evidence 读取、几何合同校验和 canonical 局部图发布。 |
| `ros2_ws/src/luna_t3_map_adapter/config/task3_local_map_adapter.yaml` | 局部地图 topic、frame、安全映射、缓存和新鲜度配置。 |
| `ros2_ws/src/luna_t3_map_adapter/launch/task3_live_integration.launch.py` | 以 Task3 适配模式共同启动全局、局部、定位状态校验与 planner。 |
| `ros2_ws/src/luna_t3_map_adapter/python/luna_t3_map_adapter/localization_status_adapter.py` | 直接消费课题三车体 Odometry、校验 frame/新鲜度/协方差并发布定位状态；不转换 pose。 |
| `deployment/luna_runtime/{config.py,process.py,build.py}` | 输入模式校验、启动命令选择及新包构建选择。 |
| `deployment/config/{runtime.default.yaml,task3-adapted.runtime.yaml,task3-adapters.default.yaml}` | generic 与 Task3 输入模式配置；Task3 SQLite、topic、frame 与安全映射均可由操作者覆盖。 |
| `deployment/runtime_source_allowlist.yaml` | 将两个活动适配包加入源码运行时 bundle。 |
| `deployment/docs/{README.runtime.md,COMMANDS.runtime.md}` | Task3 联调启动、探测和执行反馈接口说明。 |

## Task 1: 提取并验证 GridMap 编解码边界

**Files:**
- Create: `ros2_ws/src/luna_t3_map_adapter/python/luna_t3_map_adapter/grid_map_codec.py`
- Modify: `ros2_ws/src/luna_t3_map_adapter/python/luna_t3_map_adapter/global_grid_conversion.py`
- Test: `ros2_ws/src/luna_t3_map_adapter/test/test_grid_map_codec.py`

**Interfaces:**
- Consumes: `grid_map_msgs.msg.GridMap`，其 `layers` 与 `data` 顺序一致。
- Produces: `decode_layers(message: GridMap, required_layers: tuple[str, ...]) -> dict[str, np.ndarray]` 和 `encode_grid_map(frame_id: str, stamp: Time, resolution_m: float, origin_x_m: float, origin_y_m: float, layers: Mapping[str, np.ndarray], basic_layers: tuple[str, ...]) -> GridMap`。

- [ ] **Step 1: Write the failing codec tests**

```python
def test_round_trip_preserves_02m_shape_values_and_ring_indices():
    original = {"elevation": np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)}
    message = encode_grid_map("odom", stamp(), 0.2, -1.0, 3.0, original, ("elevation",))
    assert message.info.resolution == 0.2
    assert decode_layers(message, ("elevation",))["elevation"].tolist() == original["elevation"].tolist()

def test_decode_rejects_duplicate_missing_or_wrong_shaped_layers():
    with pytest.raises(GridMapCodecError, match="LOCAL_MAP_LAYER_INVALID"):
        decode_layers(malformed_grid_map(), ("elevation", "roughness"))
```

- [ ] **Step 2: Run the test to verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q ros2_ws/src/luna_t3_map_adapter/test/test_grid_map_codec.py`

Expected: FAIL because `grid_map_codec` does not exist.

- [ ] **Step 3: Implement the smallest shared codec**

```python
class GridMapCodecError(ValueError):
    pass

def decode_layers(message: GridMap, required_layers: tuple[str, ...]) -> dict[str, np.ndarray]:
    if message.layers.count("elevation") != 1 or message.info.resolution <= 0.0:
        raise GridMapCodecError("LOCAL_MAP_LAYER_INVALID")
    # Verify every required layer exactly once, decode the established inverse ring order,
    # and return immutable-shape float32 [y, x] arrays.

def encode_grid_map(...):
    # Validate all arrays have one common non-zero [height, width] shape, then encode
    # the inverse order expected by lunar_planner_ros and set both ring indices to zero.
```

Move `_encode_layer` in `global_grid_conversion.py` to this codec and keep the global output byte-equivalent.

- [ ] **Step 4: Run focused regression tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q ros2_ws/src/luna_t3_map_adapter/test/test_grid_map_codec.py ros2_ws/src/luna_t3_map_adapter/test/test_global_grid_conversion.py`

Expected: PASS; existing global conversion output remains unchanged.

- [ ] **Step 5: Commit**

```bash
git add ros2_ws/src/luna_t3_map_adapter/python/luna_t3_map_adapter/grid_map_codec.py \
  ros2_ws/src/luna_t3_map_adapter/python/luna_t3_map_adapter/global_grid_conversion.py \
  ros2_ws/src/luna_t3_map_adapter/test/test_grid_map_codec.py
git commit -m "refactor(task3): share canonical GridMap codec"
```

## Task 2: 实现无 ROS 依赖的局部十层证据转换

**Files:**
- Create: `ros2_ws/src/luna_t3_map_adapter/python/luna_t3_map_adapter/local_grid_conversion.py`
- Test: `ros2_ws/src/luna_t3_map_adapter/test/test_local_grid_conversion.py`

**Interfaces:**
- Consumes: `LocalSourceGrid`（0.2 m occupancy、semantic_id、elevation、roughness）、`TileEvidence`（height_range、elevation_variance、roughness、observation_count、semantic_confidence）和 `LocalMapPolicy`。
- Produces: `local_source_from_grid(message: GridMap) -> LocalSourceGrid`、`CanonicalLocalGrid`（origin、resolution、十个 float32 layers）以及 `ObservationLedger.observe(stamp_ns: int, valid_cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]`，返回 count 和 age。

- [ ] **Step 1: Write failing local conversion tests**

```python
def test_local_conversion_preserves_02m_grid_shape_without_resampling():
    output = convert_local_grid(source_grid_300_by_300(), complete_tile_evidence(), policy())
    assert output.resolution_m == 0.2
    assert output.layers["elevation"].shape == (300, 300)
    assert output.layers["elevation"][17, 23] == source.elevation[17, 23]

def test_missing_height_or_variance_is_invalid_and_forbidden_not_zero_filled():
    output = convert_local_grid(source_grid(), evidence_missing("height_range"), policy())
    assert output.layers["valid_mask"][8, 9] == 0.0
    assert output.layers["forbidden"][8, 9] == 1.0

def test_same_stamp_does_not_increment_observation_count_twice():
    ledger = ObservationLedger(width=2, height=2)
    first_count, _ = ledger.observe(100, np.array([[True, False], [False, True]]))
    second_count, _ = ledger.observe(100, np.array([[True, False], [False, True]]))
    assert second_count.tolist() == first_count.tolist()
```

- [ ] **Step 2: Run the test to verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q ros2_ws/src/luna_t3_map_adapter/test/test_local_grid_conversion.py`

Expected: FAIL because `local_grid_conversion` does not exist.

- [ ] **Step 3: Implement the deterministic converter**

```python
@dataclass(frozen=True)
class LocalMapPolicy:
    occupancy_obstacle_threshold: int
    semantic_obstacle_ids: frozenset[int]
    semantic_forbidden_ids: frozenset[int]
    max_age_s: float

def convert_local_grid(source: LocalSourceGrid, evidence: TileEvidence,
                       policy: LocalMapPolicy, ledger: ObservationLedger) -> CanonicalLocalGrid:
    # Reject non-0.2 m data. valid_mask requires every source/evidence field needed
    # by the platform-safety contract. obstacle is occupancy OR semantic. forbidden
    # is invalid OR policy-forbidden. Never synthesize missing height/variance.
```

Use `np.isfinite` and documented Task3 sentinels; preserve source elevation exactly; record only touched valid cells in the ledger.

- [ ] **Step 4: Run focused tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q ros2_ws/src/luna_t3_map_adapter/test/test_local_grid_conversion.py`

Expected: PASS, including no-resampling, obstacle mapping, missing-evidence rejection, ledger idempotence and stale-age cases.

- [ ] **Step 5: Commit**

```bash
git add ros2_ws/src/luna_t3_map_adapter/python/luna_t3_map_adapter/local_grid_conversion.py \
  ros2_ws/src/luna_t3_map_adapter/test/test_local_grid_conversion.py
git commit -m "feat(task3): derive canonical local planning evidence"
```

## Task 3: 接线 Task3 局部图 ROS 节点

**Files:**
- Create: `ros2_ws/src/luna_t3_map_adapter/python/luna_t3_map_adapter/local_map_adapter_node.py`
- Create: `ros2_ws/src/luna_t3_map_adapter/scripts/luna_t3_local_map_adapter_node.py`
- Create: `ros2_ws/src/luna_t3_map_adapter/config/task3_local_map_adapter.yaml`
- Modify: `ros2_ws/src/luna_t3_map_adapter/CMakeLists.txt`
- Modify: `ros2_ws/src/luna_t3_map_adapter/package.xml`
- Test: `ros2_ws/src/luna_t3_map_adapter/test/test_local_map_adapter_node.py`

**Interfaces:**
- Consumes: `/Car/T3/mapping/grid_map`（`frame_id="odom"`、单位朝向、0.2 m）, a time-matched read-only `map -> odom` transform used only to index global evidence, and a read-only `TileEvidenceProvider`.
- Produces: `/environment/map_local` with the ten required layers and unchanged source geometry, `frame_id="odom"`, source timestamp, reliable/transient-local depth-1 QoS; diagnostics reasons `LOCAL_MAP_RESOLUTION_INVALID`, `LOCAL_MAP_FRAME_INVALID`, `LOCAL_MAP_GEOMETRY_INVALID`, `LOCAL_MAP_EVIDENCE_INCOMPLETE`.

- [ ] **Step 1: Write failing node tests with fakes**

```python
def test_node_publishes_unchanged_02m_dimensions_in_odom_frame():
    node = LocalMapAdapterNode(evidence_provider=complete_provider())
    node.on_source_grid(task3_grid_map(frame_id="odom", resolution=0.2, width=300, height=300))
    published = node.publisher.messages[-1]
    assert published.header.frame_id == "odom"
    assert published.info.resolution == 0.2
    assert set(published.layers) == REQUIRED_PLANNER_LAYERS

def test_node_does_not_publish_when_source_geometry_or_tile_evidence_is_unavailable():
    node = LocalMapAdapterNode(evidence_provider=missing_provider())
    node.on_source_grid(task3_grid_map(frame_id="map", resolution=0.2))
    assert node.publisher.messages == []
    assert node.last_reason in {"LOCAL_MAP_FRAME_INVALID", "LOCAL_MAP_EVIDENCE_INCOMPLETE"}
```

- [ ] **Step 2: Run the test to verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q ros2_ws/src/luna_t3_map_adapter/test/test_local_map_adapter_node.py`

Expected: FAIL because `LocalMapAdapterNode` does not exist.

- [ ] **Step 3: Implement the node and package wiring**

```python
class LocalMapAdapterNode(Node):
    def _on_source_grid(self, message: GridMap) -> None:
        source = local_source_from_grid(message)
        odom_from_map = self._transformer.odom_from_map(message.header.stamp)
        evidence = self._evidence_provider.read_window(source.map_bounds(), odom_from_map)
        canonical = convert_local_grid(source, evidence, self._policy, self._ledger)
        self._publisher.publish(encode_grid_map("odom", message.header.stamp, ...))
```

Reject instead of transforming a source map whose frame is not `odom`, whose orientation is non-identity, or whose geometry is malformed: `lunar_planner_ros` applies the same restriction. `map -> odom` may only index global L0 evidence at local-cell centers; it must not alter GridMap origin, orientation, resolution or dimensions. Add `tf2_ros`, `diagnostic_msgs` and required ROS message dependencies to `package.xml`; install the second script in CMake. The YAML must declare source/output topic names, `odom`, `0.2`, occupancy threshold, semantic IDs, cache size and `max_age_s`.

- [ ] **Step 4: Run package tests and build**

Run: `source /opt/ros/humble/setup.bash && colcon build --packages-select luna_t3_map_adapter --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo && source install/setup.bash && colcon test --packages-select luna_t3_map_adapter --event-handlers console_direct+`

Expected: build succeeds and all existing plus new adapter tests pass.

- [ ] **Step 5: Commit**

```bash
git add ros2_ws/src/luna_t3_map_adapter
git commit -m "feat(task3): publish canonical local map evidence"
```

## Task 4: 直接消费车体 Odometry 并发布定位状态

**Files:**
- Create: `ros2_ws/src/luna_t3_map_adapter/python/luna_t3_map_adapter/localization_status_adapter.py`
- Create: `ros2_ws/src/luna_t3_map_adapter/scripts/luna_t3_localization_status_adapter_node.py`
- Create: `ros2_ws/src/luna_t3_map_adapter/config/task3_localization_status_adapter.yaml`
- Modify: `ros2_ws/src/luna_t3_map_adapter/CMakeLists.txt`
- Modify: `ros2_ws/src/luna_t3_map_adapter/package.xml`
- Test: `ros2_ws/src/luna_t3_map_adapter/test/test_localization_status_adapter.py`

**Interfaces:**
- Consumes: `/Car/T3/semantic/current_pose` already expressing `odom -> base_link`, and `/tf` already containing `map -> odom -> base_link`.
- Produces: `/localization/status`; the planner `interfaces.odometry` points directly at `/Car/T3/semantic/current_pose`; the adapter must never publish `/localization/odometry` or `/tf`. Missing `map -> odom -> base_link` is rejected by the planner's existing TF input contract, not hidden by this adapter.

- [ ] **Step 1: Write failing status tests**

```python
def test_valid_task3_body_odometry_produces_valid_status_without_pose_rewrite():
    status = classify_task3_odometry(body_odometry(frame="odom", child="base_link"), now_ns=2_000_000_000)
    assert status == LocalizationStatus.VALID

def test_camera_frame_stale_stamp_or_invalid_covariance_produces_invalid_status():
    assert classify_task3_odometry(camera_odometry(), now_ns=2_000_000_000) == LocalizationStatus.INVALID
    assert classify_task3_odometry(stale_body_odometry(), now_ns=2_000_000_000) == LocalizationStatus.INVALID
```

- [ ] **Step 2: Run the tests to verify RED**

Run: `source /opt/ros/humble/setup.bash && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="ros2_ws/src/luna_t3_map_adapter/python:${PYTHONPATH}" python3 -m pytest -q ros2_ws/src/luna_t3_map_adapter/test/test_localization_status_adapter.py`

Expected: FAIL because `localization_status_adapter` does not exist.

- [ ] **Step 3: Implement validation-only status publication**

```python
def classify_task3_odometry(source: Odometry, now_ns: int, policy: LocalizationStatusPolicy) -> int:
    # Require odom/base_link, finite pose/twist/covariance, a unit quaternion and a
    # fresh stamp. Return VALID, DEGRADED or INVALID; never alter the source pose.

class LocalizationStatusAdapterNode(Node):
    # Subscribe to Task3 body Odometry and publish only LocalizationStatus with
    # header.frame_id="odom" and the same source stamp.
```

Install the node script and add only `nav_msgs`, `lunar_navigation_msgs` and `rclpy` dependencies required by the status adapter.

- [ ] **Step 4: Run package tests and build**

Run: `source /opt/ros/humble/setup.bash && colcon build --packages-select luna_t3_map_adapter --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo && source install/setup.bash && colcon test --packages-select luna_t3_map_adapter --event-handlers console_direct+`

Expected: status tests and all existing adapter tests pass; no duplicate TF or Odometry publisher exists.

- [ ] **Step 5: Commit**

```bash
git add ros2_ws/src/luna_t3_map_adapter
git commit -m "feat(task3): validate direct vehicle localization"
```

## Task 5: 将 Task3 适配模式接入部署运行时

**Files:**
- Create: `ros2_ws/src/luna_t3_map_adapter/launch/task3_live_integration.launch.py`
- Modify: `deployment/luna_runtime/config.py`
- Modify: `deployment/luna_runtime/process.py`
- Modify: `deployment/luna_runtime/build.py`
- Modify: `deployment/runtime_source_allowlist.yaml`
- Modify: `deployment/config/runtime.default.yaml`
- Create: `deployment/config/task3-adapted.runtime.yaml`
- Create: `deployment/config/task3-adapters.default.yaml`
- Test: `tests/deployment/test_runtime_config.py`
- Test: `tests/deployment/test_runtime_process.py`
- Test: `tests/deployment/test_runtime_bundle.py`

**Interfaces:**
- Consumes: existing runtime configuration plus an optional `input_adapters` mapping with exact keys `mode` and `task3_config_file`.
- Produces: `external_canonical` launches only planner; `task3_adapted` launches global/local/localization-status adapters plus planner and passes the selected adapter YAML paths.

- [ ] **Step 1: Write failing deployment tests**

```python
def test_task3_adapted_config_requires_absolute_adapter_config_path():
    with pytest.raises(ConfigError, match="input_adapters.task3_config_file"):
        load_runtime_config(write_config(input_adapters={"mode": "task3_adapted", "task3_config_file": "relative.yaml"}))

def test_task3_mode_launches_single_composed_launch_and_builds_both_adapters():
    plan = make_build_plan(task3_config(), paths(), repo_root())
    assert "luna_t3_map_adapter" in plan.packages
    assert "luna_t3_map_adapter" in plan.packages
    assert "task3_live_integration.launch.py" in start_runtime_command(task3_config(), paths())
```

- [ ] **Step 2: Run tests to verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/deployment/test_runtime_config.py tests/deployment/test_runtime_process.py tests/deployment/test_runtime_bundle.py`

Expected: FAIL because `input_adapters` is not a supported runtime configuration section.

- [ ] **Step 3: Implement explicit input mode selection**

```yaml
input_adapters:
  mode: task3_adapted
  task3_config_file: /etc/luna/task3-adapters.yaml
```

Extend `RuntimeConfig`, `_TOP_LEVEL_KEYS` validation and generated launch command without weakening safety-key rejection. Both `runtime.default.yaml` and `task3-adapted.runtime.yaml` must explicitly contain `input_adapters`; the former chooses `external_canonical` with `task3_config_file: null`, and the latter chooses `task3_adapted` with an absolute path. In Task3 mode point `interfaces.odometry` directly to `/Car/T3/semantic/current_pose`; add `luna_t3_map_adapter` to the build plan and runtime allowlist. The composed launch includes global/local/localization-status adapters plus planner but no execution bridge.

- [ ] **Step 4: Run deployment verification**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/deployment/test_runtime_config.py tests/deployment/test_runtime_process.py tests/deployment/test_runtime_bundle.py && python3 tools/check_repository_boundaries.py . && python3 -m pytest -q tests/foundation/test_repository_boundaries.py`

Expected: PASS; generic bundles remain Task3-free unless task3-adapted configuration explicitly selects the adapters.

- [ ] **Step 5: Commit**

```bash
git add deployment/luna_runtime deployment/config/runtime.default.yaml \
  deployment/config/task3-adapted.runtime.yaml deployment/config/task3-adapters.default.yaml \
  deployment/runtime_source_allowlist.yaml ros2_ws/src/luna_t3_map_adapter/launch/task3_live_integration.launch.py \
  tests/deployment
git commit -m "feat(deploy): add Task3 adapted planner input mode"
```

## Task 6: 完成规划—反馈联调证据和操作者文档

**Files:**
- Modify: `ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp`
- Create: `ros2_ws/src/luna_t3_map_adapter/scripts/luna_plan_smoke_client.py`
- Modify: `ros2_ws/src/luna_t3_map_adapter/CMakeLists.txt`
- Modify: `deployment/docs/README.runtime.md`
- Modify: `deployment/docs/COMMANDS.runtime.md`
- Test: `tests/deployment/test_runtime_docs.py`

**Interfaces:**
- Consumes: active task、canonical Task3-adapted maps、direct Task3 body Odometry、`PlanMotion` POINT goal、`MotionExecutionFeedback`。
- Produces: identity-matched `MotionReference` result; accepted feedback progression; rejected stale, duplicate, wrong-platform or wrong-plan feedback; an installed `luna_plan_smoke_client.py` that accepts explicit mission/goal CLI arguments and never publishes controller commands.

- [ ] **Step 1: Add failing integration tests and documentation assertions**

```cpp
TEST_F(PlanMotionServerTest, ContinuesOnlyAfterTask3AdaptedGroundFeedbackMatchesReference) {
  PublishTask3AdaptedSnapshot();
  const auto result = SendPointGoal("task3-short-safe-goal");
  ASSERT_TRUE(result.has_reference);
  PublishFeedback(result.reference.plan_id, result.reference.plan_id,
                  MotionExecutionFeedback::ACCEPTED, 1U);
  PublishFeedback(result.reference.plan_id, result.reference.plan_id,
                  MotionExecutionFeedback::SEGMENT_COMPLETE, 2U);
  EXPECT_TRUE(SendChangedPointGoal("task3-next-safe-goal").accepted);
}
```

Add doc tests requiring the exact Task3 source topics, `fallback` limitation, the `PlanMotion` goal/result contract, and the rule that no generic controller command topic is provided. Add a smoke-client argument parser test for `--mission-id`, `--mission-revision`, `--goal-id`, `--x`, `--y` and `--tolerance`.

- [ ] **Step 2: Run tests to verify RED**

Run: `source /opt/ros/humble/setup.bash && colcon test --packages-select lunar_planner_ros --ctest-args -R PlanMotionServer && PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/deployment/test_runtime_docs.py`

Expected: FAIL until the new fixture/helper and documentation are present.

- [ ] **Step 3: Implement only fixture and operator-facing contract evidence**

Add a Task3-adapted fixture that publishes canonical maps and direct Task3 `odom -> base_link`; do not add a controller command publisher. Implement the client surface:

```bash
ros2 run luna_t3_map_adapter luna_plan_smoke_client.py \
  --mission-id "$MISSION_ID" --mission-revision "$MISSION_REVISION" \
  --goal-id short-safe-point --x "$SAFE_X" --y "$SAFE_Y" --tolerance 0.2
```

Document that the controller must consume `MotionReference` from the Action result and publish matching feedback; its actual command API is outside this repository.

- [ ] **Step 4: Run complete focused verification**

Run: `source /opt/ros/humble/setup.bash && colcon build --packages-select luna_t3_map_adapter lunar_planner_ros && source install/setup.bash && colcon test --packages-select luna_t3_map_adapter lunar_planner_ros --event-handlers console_direct+ && PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/deployment/test_runtime_config.py tests/deployment/test_runtime_process.py tests/deployment/test_runtime_bundle.py tests/deployment/test_runtime_docs.py && git diff --check`

Expected: PASS. A real target-host rosbag/live run remains required for final external integration acceptance.

- [ ] **Step 5: Commit**

```bash
git add ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp \
  ros2_ws/src/luna_t3_map_adapter/scripts/luna_plan_smoke_client.py \
  ros2_ws/src/luna_t3_map_adapter/CMakeLists.txt \
  deployment/docs/README.runtime.md deployment/docs/COMMANDS.runtime.md \
  tests/deployment/test_runtime_docs.py
git commit -m "test(integration): document Task3 planning feedback loop"
```

## Task 7: 目标机联调验收（不写入仓库）

**Files:**
- Create outside repository: `$LUNA_HOME/evidence/task3-live-integration-<timestamp>.json`

**Interfaces:**
- Consumes: target-host Task3 publishers、真实 SQLite 路径、实际标定文件、一个平台能力文件和外部 Action client/controller。
- Produces: 可复现的输入、Action、反馈及拒绝原因记录；不提交到 Git。

- [ ] **Step 1: Preflight the live ROS graph**

```bash
luna config check --config /etc/luna/task3-adapted.runtime.yaml
luna doctor --live
ros2 topic info /Car/T3/mapping/grid_map -v
ros2 topic info /Car/T3/semantic/current_pose -v
ros2 topic echo --once /Car/T3/mapping/global_map_revision
```

Expected: source topics have one Task3 publisher, local map reports `0.2`, and all configured files are readable.

- [ ] **Step 2: Verify input adaptation without motion**

```bash
luna start --config /etc/luna/task3-adapted.runtime.yaml
ros2 topic echo --once /environment/map_local
ros2 run tf2_ros tf2_echo map base_link
ros2 lifecycle get /lunar_planner
```

Expected: ten local layers, `frame_id=odom`, valid TF chain, and active lifecycle state.

- [ ] **Step 3: Issue one short safe reference request**

```bash
ros2 run luna_t3_map_adapter luna_plan_smoke_client.py \
  --mission-id "$MISSION_ID" --mission-revision "$MISSION_REVISION" \
  --goal-id short-safe-point --x "$SAFE_X" --y "$SAFE_Y" --tolerance 0.2
```

Expected: an explicit result with a diagnostic reason; a reference, if present, is handed to the external controller only after its `plan_id` is recorded.

- [ ] **Step 4: Validate feedback identity and safe failure**

Use the external controller to publish matching `ACCEPTED`, `EXECUTING`, and `SEGMENT_COMPLETE` feedback while fresh odometry continues. Repeat once with a wrong `plan_id`; expect the planner to reject the feedback and retain safe hold behavior.

- [ ] **Step 5: Archive external evidence and record outcome**

Record config SHA-256, capability IDs, source topic types/QoS, map revision, calibration SHA-256, Action result, feedback sequence and diagnostic codes in `$LUNA_HOME/evidence/`. Do not copy SQLite, rosbag, controller logs or generated artifacts into Git.

## Plan Self-Review

- Spec coverage: Tasks 1–3 implement unresampled ten-layer local evidence; Task 4 directly validates Task3 vehicle localization without pose conversion; Task 5 implements runtime selection and package delivery; Task 6 fixes the Action/feedback integration boundary and documentation; Task 7 defines target-host proof. No spec requirement is omitted.
- Placeholder scan: all task interfaces, file paths, tests, commands and failure reasons are explicit. Task 7 accepts mission ID, revision and approved safe point through named shell variables supplied by the active external task; the repository never fabricates a mission or destination.
- Type consistency: all local map tasks use `CanonicalLocalGrid`, `LocalMapPolicy`, `ObservationLedger`, `GridMap`; Task 4 uses direct Task3 `Odometry` and `LocalizationStatus`; deployment mode is consistently `external_canonical | task3_adapted`.
