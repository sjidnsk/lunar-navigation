# DRL Exploration Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkboxes for tracking.

**Goal:** 在独立分支实现可构建、可闭环采集、可学习/恢复/部署推理的相邻图节点与八方向探索策略。

**Architecture:** 地图模块独占测量和地形分类；纯探索核心负责任务机会、稀疏图和冻结联合动作；Actor 只看测量。训练私有真值参考、双 Critic、回放和生命周期独立于部署。现有导航规划及公共控制器接轻量运动学平台。

**Tech Stack:** C++20 原生地图与 ROS 2 Jazzy 开发构建，Python 3、NumPy、SciPy、PyTorch 2.8、ament/colcon、pytest/CTest。Humble/Orin 单独验证。

**Spec:** `docs/superpowers/specs/2026-09-12-drl-exploration-redesign.md`（含 2026-09-15 实施前复审修正）。

## 执行与复审状态（2026-09-16）

Tasks1–9 的实现、测试执行、文档和各任务独立复审已完成；下列勾选表示对应工作已执行并记录结果，
不表示每个历史测试均通过或学习质量达标。Task9 独立 Spec/Quality 复审 Approved。
全分支 `5c23c13..ff60e7c` 首轮复审为 With fixes；I1/I2/M1 已修正，限定范围复审待完成。
[实施验证](../../validation/2026-09-15-drl-exploration-redesign-implementation.md)保留实际测试、历史 NO_PATH 基线失败、
源码版本与性能范围；正式学习质量、Humble/Orin 和现场证据仍为 NOT_RUN。未合并、推送或启动无界训练。

## Global Constraints

- 仅写 `feat/drl-exploration-redesign` 工作树；不合并、不推送、不创建 PR、不删除其他工作树、旧训练产物或他人修改。
- 任务多边形只约束覆盖/奖励；目标和导航路径允许在任务外，任务外运动计代价。
- 平台最高速度 0.2 m/s，允许前后退转弯和原地旋转，不能横向移动。保留公共控制器和轻量运动学平台，不引入 TCP、Unreal、Isaac 或轮胎动力学。
- 首轮 10 m、90°；8 环境，30 倍目标，0.05 仿真秒积分、2 Hz 仿真观测。实际倍率单独测量，不能跳步维持倍率。
- 地图生产者独占分类；B 为固有遮挡，M 为足迹派生导航状态。失联自由岛不重标为障碍，地图存储边缘不是墙。
- 有效中心测量近似只发布命中格的中心高程和原生 3×3 地形测量；不发布隐藏邻居。height-only 实物输入仍可用。
- Actor 输入为 19 维图节点、任务角点、平台/传感器上下文；没有高程整图、真值、可覆盖分母或 Critic 输入。当前位置为实际位姿锚点。
- 至多 20 个相邻位置含当前位置、每位置 8 个世界朝向，至多 160 个联合动作，单一联合 softmax；推理取联合最大概率动作。
- 图注意力 6 层、8 头、宽 128，沿边稀疏实现；一次当前位置全图读取。无 GRU、无 NxN 密集注意力、无按硬上限截断任务相关图。
- 奖励为 ΔA/100 − 0.02ΔL/10 − 0.005ΔΘ/π − 0.001；实际全程面积/路程/绝对转角；无完成或里程碑奖金。
- γ=1；只有真实耗尽 terminated 不自举，课程预算 truncated 使用重置前后继状态自举。80%/99% 仅评估。
- 目标熵 0.01log(N_valid)，α 初值 0.00005、上限 0.0001；学习率 0.00001、Polyak 系数 0.005。
- 有效 batch=64，microbatch=16，梯度累计4次；Critic→Actor→温度→目标软更新各一次，梯度隔离。预热1024条无额度债务；之后每新增4条有效转移更新一次，Actor每16次更新发布。
- 8 环境异步，CPU 集中推理就绪批1～8、1 GPU Learner；数值线程 worker1/collector2/learner4，DataLoader workers=0；有界更新额度和背压，不丢完成转移或更新额度。
- 小/中/大课程预算512/2048/8192；预算是截断，零增益不提前换图；基础课程40～80/80～150/100～300m，300～1000m先独立验证。
- 产物项目内 `training-output/drl-exploration-redesign/`；1800秒单调墙钟保存，resume.pt一份、best.pt可选一份、metrics.jsonl上限8MiB、run.json；无默认视频/bag/地图图片/历史模型。
- 回放内存8GiB、保存回放2GiB、总产物20GiB含原子写入临时文件。引用计数释放场景。恢复模型/优化器/RNG/课程/计数/额度/有限回放，环境从新回合开始。
- 新 schema 不加载旧 GRU 模型或经验；不要求前沿对照、99%通过率或额外地图准入流程才能开训。
- Python 测试在仓库根用 `python3 -m pytest`（需要 torch 时使用依赖 venv 的 Python）；ROS 变更运行受影响构建和测试。Jazzy 证据不代替 Humble/Orin/DDS/实车。

## Implementation decisions and shared boundaries

1. 核心包保持名 `lunar_drl_exploration`，新 schema `task_graph_v1`；源码从本分支新增，不搬运旧学习器或整套旧导航修改。
2. 测量近似暴露原生 `LocalTerrainMeasurements`（center_known、neighborhood_complete、slope_rad、relief_m、positive_rise_m）。有效测量与中心高程同次融合；已保存的有效测量不因部分邻域重算退回 UNKNOWN。同中心高程改变而本次未附带统计量时撤销该中心旧统计量，恢复 height-only 推导；不使用过期证据。
3. `ElevationRangeView` 可选返回中心测量，默认无；原生测量函数优先读取有效中心测量，否则测量累计3×3高程。阈值仍只在原生平台 evaluator。ROS 可选层为 `terrain_slope`、`terrain_relief`、`terrain_positive_rise`、`terrain_complete`，缺整组时为 height-only，部分组属于明确消息格式错误。中心无效时不采信统计量。
4. 只读 GetPolicyMap 导出同一导航地图，不另起地图节点；字段含当前固有分类、有效分类标记、导航状态、代价/中心高程、epoch/revision和起点连接。探索对有效分类作任务内累计首次确认，不把膨胀格视为观测。
5. 纯 Python 入口位于 `ros2_ws/src/lunar_drl_exploration/lunar_drl_exploration/`。`contracts.py` 定义不可变数组快照；`maps.py` 是导出快照客户端缓存，不能自己重新分类地形。
6. 图构建先保持已知连通、回路与观测见证，再压缩无用途的中间节点；边保留已知连接长度。几何连通不等于正式导航成功，发送相邻端点由导航独立处理。
7. 新构建写 `/home/kai/.cache/lunar-drl-redesign/jazzy/{build,install,log}`。已有 `/home/kai/.cache/lunar-drl-training/venv/bin/python` 可复用 PyTorch 依赖，但不能混用旧 native 扩展或 ROS overlay。
8. 课程按有效采集数推进：0～19999小图；20000～59999以0.25小/0.75中混合；60000以后0.15小/0.25中/0.60大混合。每槽新回合再换课程，槽号奇偶保持月表/洞穴4+4。这些是可配置工程初值，不是收敛门槛。
9. 参考计算可用紧凑布尔/位集合和分块原生循环；不为每条经验存全图。1km验证不对2500万格建立Python逐格对象或密集注意力。
10. 仿真、参考、utility 共用站位格中心观测原点（floor世界坐标/分辨率，格边界归正侧格）及格角扫掠规则，yaw与控制实际位姿不量化。参考与plant合法移动共用真值M格/扫掠连通。首版虚拟传感器平移外参为零，非零值明确报不支持，不能静默丢弃；安装偏航仍支持。

## File ownership

| 单元 | 文件 |
|---|---|
| 原生中心测量 | core 的 elevation_map.hpp、persistent_elevation_map.cpp、platform_elevation_physics.cpp；ROS map_adapters |
| 只读地图接口 | planning_msgs 的 PolicyMapTile/GetPolicyMap；ROS policy_map_exporter 与导航节点小型接入 |
| 公共数据契约 | DRL contracts.py、config.py、maps.py |
| 环境几何与参考 | DRL scene.py、sensor.py、reference.py、native/{terrain,visibility}.cpp |
| 纯探索内核 | DRL task_analysis.py、graph.py、decision.py、history.py |
| 学习 | DRL model.py、sac.py、batch.py |
| 状态与恢复 | DRL replay.py、checkpoint.py、schedule.py、metrics.py |
| 执行入口 | DRL plant.py、ros_env.py、processes.py、worker.py、collector.py、training.py、runtime.py、cli.py |
| 操作 | config/drl_exploration.yaml、scripts/drl/*、docs/操作指令.md、实施验证文档 |

### Task 1: 原生有效中心地形测量

**Files:** Modify `ros2_ws/src/lunar_incremental_navigation_core/include/lunar_incremental_navigation_core/elevation_map.hpp`, `src/map/persistent_elevation_map.cpp`, `src/map/platform_elevation_physics.cpp`, `include/lunar_incremental_navigation_core/fine_traversability_builder.hpp`; Modify ROS `include/lunar_incremental_navigation_ros/map_adapters.hpp`, `src/map_adapters.cpp`; Test core `test/fine_traversability_builder_test.cpp`, `test/persistent_elevation_map_test.cpp`, ROS `test/map_adapters_test.cpp`.

**Interfaces:** Publish `LocalTerrainMeasurements` with booleans center_known/neighborhood_complete and doubles slope_rad/relief_m/positive_rise_m; `MeasureLocalTerrain(const ElevationRangeView&, GridIndex)` returns this struct. `ElevationEvidence` has default-empty span of center measurements; `ElevationRangeView::TerrainMeasurementsAt(GridIndex)` defaults nullopt; ElevationSnapshot overrides. Normal elevation-only callers remain source compatible.

- [x] Write regression tests: 3×3 wall with first BLOCKED on lower floor; visible centers plus measured stats must retain wall closure after native derive, no hidden neighbor elevation. Also height-only full/partial neighborhood, changed-only stats revisions, duplicate evidence, COW snapshot immutability and rotated/wrapped ROS GridMap layer ordering.
- [x] Run the new tests and record expected pre-implementation failure. Example behavior:
```cpp
EXPECT_EQ(raw_only->ElevationRangeAt(hidden_wall), std::nullopt);
EXPECT_EQ(evaluator->Evaluate(*with_center_evidence, hit_floor).state,
          IntrinsicCellState::kBlocked);
EXPECT_EQ(evaluator->Evaluate(*raw_only, hit_floor).state,
          IntrinsicCellState::kUnknown);
```
- [x] Implement atomically stored optional measurements and the same native threshold path. No simulator flags in planner search, no globally FREE overrides; reject unsupported nonhorizontal transforms for optional scalar terrain measurements rather than rotate slopes incorrectly. Height-only path retains its current transform support.
- [x] Build affected core and ROS packages in the new Jazzy cache with `colcon build --packages-up-to lunar_incremental_navigation_ros --executor sequential`, source `/opt/ros/jazzy/setup.bash` first and use explicit external build/install/log bases. Run their tests serially and inspect `colcon test-result --verbose`.
- [x] UTF-8 readback, `git diff --check`, explicitly stage changed paths and commit `feat: preserve effective terrain measurements in the native map`.

### Task 2: 导出地图与不可变策略数据契约

**Files:** Create `ros2_ws/src/lunar_planning_msgs/msg/PolicyMapTile.msg`, `srv/GetPolicyMap.srv`; modify its CMakeLists. Create ROS `include/lunar_incremental_navigation_ros/policy_map_exporter.hpp`, `src/policy_map_exporter.cpp`, `test/policy_map_exporter_test.cpp`; modify ROS CMakeLists and `src/incremental_navigation_node.cpp`. Create DRL package `package.xml`, `setup.py`, `setup.cfg`, `resource/lunar_drl_exploration`, Python `__init__.py`, `contracts.py`, `config.py`, `maps.py`; create `test/test_contracts_maps.py`.

**Interfaces:** GetPolicyMap request has since_revision/minimum_map_stamp_ns; response ready/reason_code, epoch, processed stamp, raw/fine revisions, full_snapshot, geometry/profile, native start connections, local window, goal_position_tolerance_m/goal_yaw_tolerance_rad and tile deltas. Tile arrays: states(uint8), intrinsic_states(uint8), observed(uint8 effective classification), costs(float32), elevation_m(float32). Unknown=0, FREE=1, BLOCKED=2. Exported observed requires finite center and known intrinsic state, never only M state. Export实际导航容差，训练controller及访问历史读取该值对齐，不混用TCP分支或公共节点的不同默认值。

Python frozen types:
```python
Pose(x: float, y: float, yaw: float)
TaskSpec(task_id: str, frame_id: str, polygon: np.ndarray)  # [P,2]
SensorSpec(range_m: float = 10., fov_deg: float = 90.,
           offset_x_m: float = 0., offset_y_m: float = 0., offset_yaw_rad: float = 0.)
MapSnapshot(epoch: str, revision: int, resolution_m: float, origin: tuple,
            tiles: Mapping, pose: Pose, start_connections: np.ndarray,
            local_bounds: tuple, profile_hash: str)
TaskReport(known_area_m2: float, new_area_m2: float, frontier_cells: np.ndarray,
           witnesses: np.ndarray, exhausted: bool, revision: int)
DecisionObservation(node_ids: np.ndarray, positions: np.ndarray, features: np.ndarray,
    edges: np.ndarray, edge_lengths: np.ndarray, current_index: int,
    polygon: np.ndarray, context: np.ndarray, action_nodes: np.ndarray,
    action_yaws: np.ndarray, goals: np.ndarray, epoch: str, revision: int,
    schema: str = 'task_graph_v1')
PrivilegedState(scene_id: str, observed: np.ndarray)  # packed effective-center bits
RewardParts(new_area_m2: float, distance_m: float, turn_rad: float)
Transition(observation: DecisionObservation, action: int, reward: float,
    next_observation: DecisionObservation, privileged: PrivilegedState,
    next_privileged: PrivilegedState, parts: RewardParts, terminated: bool,
    truncated: bool, episode_id: str, actor_version: int)
```
Array buffers are immutable by construction, not merely a writable owner with flags disabled. `PolicyMapStore.apply(response)` returns a MapSnapshot; `snapshot.cell_at(ix,iy)` and bounded `snapshot.raster(bounds)` provide all measured fields. Actor observation context has 8 scalars: cos(yaw), sin(yaw), v/0.2, ω/configured_max_ω, range/10, FOV/π, footprint_radius/1m, max_v/0.2.

- [x] Add tests for delta/full equivalence, epoch replacement, missed revision requiring full snapshot, effective classification versus inflated unknown, and frozen action-index/world-goal identity. Red test example:
```python
before = store.apply(full_response)
after = store.apply(delta_response)
assert before.cell_at(1, 1).state == UNKNOWN
assert after.cell_at(1, 1).state == FREE
assert not before.tiles[(0, 0)].states.flags.writeable
```
- [x] Implement exporter against target APIs. Old exporter in `observability-coverage` is read-only reference; adapt API names and observed semantics, do not copy native/core wholesale. Snapshot pointer lock boundaries keep service serialization outside navigation critical sections.
- [x] Implement the Python package/contracts/client cache and config defaults matching Global Constraints. Config reads native platform footprint/limits from this branch, not duplicated constants. Register Python tests through package and root-import setup.
- [x] Build messages/ROS exporter, test new C++ exporter and Python contracts. Verify no service enables a new velocity publisher. UTF-8, diff check and explicit-path commit `feat: export measured policy maps and freeze graph action contracts`.

### Task 3: 统一传感器、场景与固定可覆盖参考

**Files:** Create DRL `scene.py`, `sensor.py`, `reference.py`, `native/terrain.cpp`, `native/visibility.cpp`, `native/grid_buffer.hpp`; modify `setup.py`; tests `test/test_sensor_reference.py`, `test/test_scene.py`.

**Interfaces:** `Scene(seed, family, extent_m, resolution_m, platform)` supplies task_polygon, initial_pose and bounded height tiles from deterministic geometry; `scene_id` includes generator version, seed and configuration. `TerrainGrid` compact intrinsic/navigation arrays derives only via native MeasureLocalTerrain/PlatformElevationEvaluator/FineCellEvaluator. `SensorModel.observe(terrain, pose, sensor) -> VisibleMeasurements` returns center indices/heights/stats. `CoverageReference.build(terrain, start, task, sensor)` returns fixed packed mask, cell area and area_m2; `reference.covered_area(observed_bits)` computes exact intersection. Reference and step share `sensor.OBSERVATION_MODEL_VERSION = "finite_center_tip_prefix_v1"`: every integer center-tip beam in nominal range emits its whole prefix and simultaneous first-hit group; side contacts precede diagonal entry. Beam angle defines FOV; source→target queries are directed. Scene generator v5 and descriptor bind this version and complete analytic feature bounds plus 12 m context, retaining the old lattice and RNG streams.

- [x] Write independent continuous closed-cell-square beam/event oracles with first-hit simultaneous corner groups, finite FOV/mount yaw and fixed-pose self-closure (including the proven 7×7 case): union of all reachable-cell viewpoints and all headings equals initialized reference; boundary obstacle counts, behind obstacle does not, visible narrow opening with nonstandable cells counts; ring-only approximation fails a fixture; task-external stance sees inside; sensor change rebuilds reference; AE=0 not100%.
```python
union = np.zeros(terrain.shape, bool)
for pose in all_reachable_grid_poses:
    hit = sensor_model.observe(terrain, pose, full_circle).mask
    union |= hit & (terrain.intrinsic != UNKNOWN)  # effectively classified centers
assert np.array_equal(reference.mask(), union & task_mask)
```
- [x] Implement native terrain/visibility loops releasing GIL and bounded buffers. Target native API has Evaluate, not old EvaluateStates. Native module name `lunar_drl_terrain_native` avoids accidental old extension loading. No Python second slope/footprint classifier.
- [x] Use the shared floor-to-cell center observation origin in sensor and reference; yaw remains actual and footprint/nativeM swept motion uses the same reachable-cell relation. Test fractional/negative coordinates and exact grid corners, comparing reference membership with every simulated observation. Reject nonzero virtual-sensor translation until its reference mapping is supported; mounting yaw does not change this rule.
- [x] Reference first computes native truth reachability including outside task; union visibility never shrinks with policy failures. Use exact reverse prefix lookups with target-major all-R nearest-source early-out, packed masks and bounded thread-safe geometry cache; conservative optical-component prefilters cannot replace exact finite beam visibility. Release temporary heights. Supply enumeration oracle on small maps to verify any optimized union algorithm rather than asserting equivalent by construction.
- [x] Adapt deterministic old scene geometry and plant-independent start terrain as needed, maintaining moon/cave, narrow passages/loops/disconnected chambers/external connections. Seeded task polygon and main-component native-free start sampling; preserve the independently sampled yaw without visibility filtering. Starts from legitimate main terrain, no map override. No asset download required. Retain source attribution for copied helpers.
- [x] Measure current generator/model 100/300/1000m initialization timing/peak memory in both families, a 10m/90° sensor call and directional graph utility in sequential bounded-thread processes; report measured scalability, no ten-minute precondition or reference baseline admission. Commit `feat: unify effective observations and fixed exploration reference` with tests and native build evidence.

### Task 4: 任务机会、稀疏图和联合位姿动作

**Files:** Create DRL `task_analysis.py`, `graph.py`, `decision.py`, `history.py`; tests `test/test_task_analysis.py`, `test/test_graph_decision.py`.

**Interfaces:** `TaskAnalyzer(task, sensor).update(snapshot) -> TaskReport`; `DirectionHistory.record(pose, observed_cells, sensor)` records actual observation directions at world positions; `GraphBuilder(config).build(snapshot, report, history, task, sensor, velocity) -> DecisionObservation`; `DecisionCore.observe(snapshot, velocity) -> (DecisionObservation, TaskReport)`. Pure NumPy/SciPy/native geometry, no ROS/PyTorch/truth dependency. `GraphBuilder.build_truth(terrain, reference)` produces compact `PrivilegedScene` static graph and index-to-reference mapping for Task5, with owned packed reference and generator descriptor.

- [x] Test open world, sealed chamber, outside-only transit, outside irrelevant branch, visible but unstandable gap, longer-than-range impassable corridor, unknown storage edge, 1m corridor missed by 2m regular sampling, loop and only-viewpoint preservation. Current action exact XY must equal actual Pose, all8 headings available. Freeze map and change hidden truth: actor observation unchanged.
```python
obs, report = core.observe(snapshot, velocity=(0., 0.))
assert obs.features.shape[1] == 19
assert len(obs.goals) <= 160
assert np.all(obs.goals[obs.action_nodes == obs.current_index, :2] == actual_xy)
assert not report.exhausted  # sole outside witness retained
```
- [x] Implement relevance from observation witness relations, not task cells as motion nodes. R=known native reachable; potential movement admits UNKNOWN but not proven M BLOCKED. Unknown task demands may include missing classification support in height-only mode. Find potential range/LOS viewpoints with B blockers; relevant components of potential movement minus R connect those viewpoints to entry boundaries. External UNKNOWN component represents unbounded storage exterior. After unifying exterior labels, restrict potential source queries to whole components that actually adjoin R; disconnected non-entry components cannot create executable interfaces. Keep direct R optical opportunity independent. Use exact first-UNKNOWN interface witnesses over the whole workspace (including UNKNOWN R sources) and their conservative2R support only as a necessary demand prefilter; every surviving demand still requires an exact forward query. Never spread through R into unrelated external branches.
- [x] Bind every real frontier interface to an actual finite-beam R witness; first pending hit uses that successful forward demand beam. Preserve direct observations across M BLOCKED/B FREE gaps independently of potential component membership. Verify complete native-cave measured producer roundtrip has identical R, area E and exhausted/no residual interfaces. Build stable world graph, native-known local connections, actual anchor and witness insertions; preserve corridor/branch/loop topology when compressing. Known empty interior may use longer compressed edges; never enforce max-node cropping. Each action is an adjacent graph endpoint, at most19 neighbors plus actual anchor; if neighbor degree exceeds19, topology-preserving local relay decomposition keeps connectivity across decisions rather than silently excluding unique branches.
- [x] Features: relative XY/10m, inside-task indicator, eight directional visible-frontier counts from all supporting selected beam angles with per-target deduplication (never target-center bearings), normalized by fixed sensor-independent frontier unit100 cells, eight actual-direction visit bits. Unknown-area optimistic visibility is not a feature. Direction histories survive resampling; mark observations from executed poses, not requested goal indices. Polygon corners relative XY/10; context as Task2.
- [x] Test utility around known blocker goes to zero when no remaining frontier, while zero-utility transit remains valid. Run graph growth/large-open-map node/edge/memory timing; fix structural scaling failures rather than remove mandatory witnesses. Commit `feat: build task-related sparse graphs and adjacent pose actions`.

### Task 5: 非循环图 SAC 与正确累计更新

**Files:** Create DRL `model.py`, `batch.py`, `sac.py`; tests `test/test_model.py`, `test/test_sac.py`.

**Interfaces:** `Actor(config).forward(list[DecisionObservation])` returns per-observation joint logits/probabilities; no PrivilegedState argument. `Critic(config).forward(observations, states, scene_registry)` returns Q for identical actor action lists. `SACLearner.update(transitions: list[Transition], scenes) -> dict` accepts64 transitions, increments updates once; `actor_state()` publishes CPU state_dict+version. `PrivilegedScene` static topology is scene-shared; dynamic effective-observation bits are pooled onto its nodes each Q call. Critic conditions on both measured graph/action and privileged graph/current coverage, with separate learned parameters from actor and other Q.

- [x] Test permutation-consistent graph outputs, varying graph/action sizes, no dense NxN attention, actor truth isolation, joint softmax normalization/argmax, exact enumerated SAC TD/actor losses and scalar temperature direction. Batch64 equals deterministic micro16×4 update within floating-point tolerance; no critic gradient pollution; truncated target bootstraps, terminated does not.
```python
expected = rewards + (1 - terminated) * sum_a(
    next_probs * (torch.minimum(q1_target, q2_target) - alpha * next_log_probs))
assert torch.allclose(actual_target, expected)
assert updates_after - updates_before == 1
```
- [x] Implement edge-index sparse multihead attention with packed disconnected graphs and scatter/segment softmax, six residual layers, 128 wide/eight heads. One current query over each full graph produces context; combine polygon/platform/sensor encoding and position+heading embeddings in shared160action head. No candidate-specific full-graph queries.
- [x] Implement detached TD target, critic microbatch weighted accumulation→one step each; actor detached-Q exact action expectation across microbatches→one step; α entropy detached→one step with cap; target Polyak once. Keep GPU tensors until one final metrics transfer; no per-node/per-transition `.item()` synchronization. Sampling loss expectations use untruncated valid actions.
- [x] Measure CPU batched inference1/8 and GPU one64update on representative graphs, save only small numerical/timing summaries. Commit `feat: train sparse graph actors with asymmetric discrete SAC`.

### Task 6: 有界回放、额度、指标与恢复

**Files:** Create DRL `replay.py`, `checkpoint.py`, `schedule.py`, `metrics.py`; tests `test/test_replay_checkpoint.py`, `test/test_schedule.py`.

**Interfaces:** `ReplayBuffer(max_bytes).add(transition, scenes)`, `sample(64,rng)`, `snapshot(max_bytes)`; memory counts shared arrays/scene references once, evictions free unreferenced scenes. `UpdateSchedule(warmup=1024, ratio=.25, max_credit=32)` controls credit and `can_dispatch(inflight_count)`, reserve enough for inflight completions. `CheckpointManager(output_dir, interval_s=1800).save(state)` atomic one resume.pt and `load()` schema/semantic checks. `TrainingState` includes learner/targets/optimizers/α/counters/course/RNG/credit/replay/scenes.

- [x] Test no warmup debt, no credit loss under backpressure and inflight completions, exact update counts, bytes/refcount eviction including aliased graph arrays and scenes, constrained save+restore with no dangling refs, RNG equality, config-compatible resume and schema mismatch reason, interrupted atomic replacement retains last checkpoint, metrics bounded while readable.
```python
for _ in range(1024): schedule.collected()
assert schedule.credit == 0
for _ in range(4): schedule.collected()
assert schedule.credit == 1
schedule.updated()
assert schedule.credit == 0
```
- [x] Implement memory/resource accounting based on owned process PSS plus system reserve, avoiding sum-RSS shared-memory false positives. Resource exception contains observed budget/reason, saves orderly; no new hardware admission prerequisite. Check available CPU/GPU/memory/disk live and put selected operational settings in run.json.
- [x] Checkpoint snapshot includes only finished transitions, does not wait for episodes; temporary file and old file counted in20GiB. Keep one resume, optional best only evaluated, one bounded metrics file and run.json. SIGINT requests consistent save after complete optimizer step; new env episodes on restore, no unfinished action stitching. Finish default config and commit `feat: bound exploration replay and support consistent training resume`.

### Task 7: ROS 导航控制环境及部署运行器

**Files:** Create DRL `plant.py`, `ros_env.py`, `processes.py`, `runtime.py`, `ros_messages.py`; tests `test/test_plant.py`, `test/test_runtime.py`, `test/test_ros_env.py`.

**Interfaces:** `RosExplorationEnv(config, env_id).reset(seed, family, extent) -> (obs, priv)`; `.step(action_index) -> Transition/result` executes frozen goal via NavigateToPose and returns final snapshot before any reset. `.close()` ordered owned-process cleanup. `InferenceRuntime` shares DecisionCore and Actor, owns START/PAUSE/RESUME/CANCEL exploration state only. InferenceRuntime loads Actor only and real readonly map/pose/TF/task input; shared runtime and observation adapters are callable by Task8 CLI.

- [x] Test motion0.2cap, forward/reverse curved motion, in-place rotation, swept footprint collisions, actual distance/absolute angle; controller never duplicated. Stub only transport in unit tests, not in later ROS validation. Test one slow environment does not block other completed transitions; pause/backpressure retains inflight output; infrastructure restart excludes only unfinished action and does not silently stall a slot.
```python
transition = env.step(index)
assert transition.observation.goals[index].tolist() == sent_goal
assert transition.parts.turn_rad >= abs(wrapped_end_yaw_minus_start)
assert not (transition.truncated and transition.terminated)
```
- [x] Adapt old read-only ROS message encoding/process ownership/plant code to new contracts. Launch target overlay navigation and public controller per isolated `/lunar_training/env_N` and separate DDS domains; no `/Car/T5` publisher. Use sim clock for plant/control/observation, monotonic wall time for pacing/watchdogs/save. Observation publishes only visible centers and optional measured stats to navigation map producer, then waits for processed map revision before next decision.
- [x] Finish valid navigation failures as transitions with measured motion and final snapshot; collision has explicit non-success result, never fake completion. Implement inference task START/PAUSE/RESUME/CANCEL, cancel-and-stop before pause, preserve measured map/history, resume fresh decision. Status explicitly omits unavailable AE rather than misuse traditional coverage_ratio.
- [x] Run transport/plant/runtime tests and one isolated navigation goal against the new overlay. Commit `feat: execute graph exploration goals through native navigation and control`.

### Task 8: 异步采集、训练调度及可运行命令

**Files:** Create DRL `worker.py`, `collector.py`, `training.py`, `cli.py`, `evaluation.py`; configuration `config/drl_exploration.yaml`; scripts `scripts/drl/{build.sh,_run.sh,train.sh,evaluate.sh,infer.sh}`; modify `setup.py` entry points; tests `test/test_collector.py`, `test/test_training_cli.py`.

**Interfaces:** Consume RosExplorationEnv/InferenceRuntime, Actor/SACLearner, ReplayBuffer/UpdateSchedule/CheckpointManager. `python -m lunar_drl_exploration.cli {train,evaluate,infer,export}` and script wrappers. Train has --resume, --max-transitions and --probe; evaluate has seed/family/extent/budget/frozen-actor parameters; export emits one actor schema/config/weights file. Learner process owns replay, admitted-transition count and update credit; collector owns ROS workers and acknowledges admission before treating a finished message as collected.

- [x] Write tests for independent slow/fast workers, bounded inflight transitions under backpressure, accepted-transition/update-credit equality across save barrier, no warmup debt, actor version publication and resumed episode separation. Verify CLI main config remains8env/30x/1024warmup/64batch, probe overrides are explicit.
```python
collector.receive(fast_worker_done)
assert collector.ready_to_dispatch(fast_worker_id)
assert not collector.waiting_for_all_environments
saved = learner.snapshot_at_barrier()
assert saved.counters.transitions == saved.schedule.admitted_transitions
```
- [x] Implement8async pipes, CPU ready-batch actor inference, separate GPU learner and bounded messages. Finished messages are retained until learner admission acknowledgment. Learner owns credit and replay consistently; collector pauses dispatch at backpressure while inflight actions finish. Every16complete updates publish CPU Actor weights for the next decisions. Bounded lifecycle retries publicly identify env/reason; exclude only unfinished infrastructure-damaged actions, no silent permanent paused or zero-gain reset.
- [x] Implement curriculum/size budgets, fixed rewardparts, progressbar+perenv scenes/reasons/actualRTF and sample/update rates/credit/actor/save countdown. Periodic snapshot captures admitted transitions at a learner barrier after a complete optimizer update, does not wait for long episodes; graceful exit drains already finished transitions and cancels unfinished goals before final save.
- [x] Implement frozen evaluation with no train replay, metrics80/99/exhaustion/latepath and actor export. Inference uses Task7 runtime and Task2 measured contracts, no scene/critic/reference import requirement. Scripts resolve root/ROS cache/venv without old overlays; sourceROS before strict unset checks, build affected dependencies including controller/exploration messages.
- [x] Run CLI help, collector/recovery tests and a short real ROS sampling command; commit `feat: run asynchronous graph exploration training and frozen evaluation`.

### Task 9: 闭环验证、修复和操作文档

**Files:** Create `docs/validation/2026-09-15-drl-exploration-redesign-implementation.md`, new appropriate integration tests under DRL test; update `README.md`, `docs/操作指令.md`, spec implementation status and review doc. Only amend implementation files when evidence reveals a defect within this design.

**Interfaces:** scripts and CLI from Task8 must be runnable from this target worktree. Short probe executes real native nav/public controller/plant, categorical Actor, replay, at least one64sample GPU update, save, resume and actor-only load. It does not claim trained policy success or convergence.

- [x] Run `python3 -m pytest` for affected core Python/controller/contracts tests and torch-venv package tests. Rebuild affected ROS packages and CTests sequentially including existing contracts; note exact commands/results and skips. Run `git diff --check` and explicit UTF-8 doc reads.
- [x] Execute isolated one-env translational goal plus in-place yaw: require PLAN_FOUND/has_reference and GOAL_REACHED, verify one cmd publisher, speed<=0.2, measured L/Θ and along-path observations. Verify static obstacle footprint/nav/collision agreement; no shortcuts via direct endpoint teleport.
- [x] Execute bounded8env smoke with small training maps and probe-only warmup override sufficient for one genuine64batch update; retain main1024warmup default. Save/stop/restart using resume, check network/optimizer counters/finite replay with new episodes. Check scope-limited infrastructure recovery and no paused slot without reason. Do not overwrite or stop another training run.
- [x] Inspect initialization/sample/update/RTF/memory/disk evidence. If30× cannot be met, report actual and bottleneck; do not change physicalspeed,step,FOV,validactions to improve measured throughput. 1km numerical/reference/graph validation is distinct from a completed1kmROS exploration mission.
- [x] Write copyable build/train/resume/evaluate/export/infer procedures with exact project paths, output files, progress semantics and known limitations. Mark Humble/Orin, actualTask3sensor, liveDDS/vehicle, trained convergence and sensor/generalization successes NOT_RUN unless directly evidenced. State oldmodel incompatibility and30minsave. Final review addresses spec+implementation; no merge/push. Commit `docs: record graph exploration implementation and validation procedures`.
