# Task 2: 导出地图与不可变策略数据契约

## 完成内容

- 新增 `PolicyMapTile` 和只读 `GetPolicyMap` 服务；请求以 `since_revision` 和
  `minimum_map_stamp_ns` 取得连续 delta，缺失连续 revision 时服务回退 full snapshot。
- `PolicyMapExporter` 在捕获后的不可变 fine snapshot 上序列化，不持有导航临界区锁。
  `states` 是导航（含足迹）状态；`intrinsic_states` 由同一原生
  `PlatformElevationEvaluator` 重算；只有中心高程有限且 intrinsic 已知时 `observed`
  才输出 FREE/BLOCKED。没有把 M/足迹派生状态当成观测。
- 节点新增 `/Car/T4/mapping/get_policy_map`（参数 `policy_map_service`）且不创建任何
  velocity publisher。`minimum_map_stamp_ns` 与最近本地 GridMap header stamp 比较，
  不使用服务时钟伪造新鲜度。epoch 每个节点实例唯一。
- 服务输出当前本地窗口、原生 start-prefix 到首个 `EvidenceFree` 的连接、连接状态、
  profile hash，以及本轮 native wheel 到达容差 `0.30 m` / `0.2617993877991494 rad`。
  无 anchor/无法建立连接的状态保留为 unavailable，不表示任务耗尽。
- 新增 `lunar_drl_exploration`：bytes-backed 不可写 ndarray、冻结 mapping、
  `PolicyMapStore.apply()` 的 full/delta/epoch 规则，`cell_at`/`raster`，以及从
  canonical `config/wheel.yaml` 读取平台速度、转速和足迹半径的 8 标量 context。
  `TaskReport.available=True`、`reason_code='READY'` 与 MapSnapshot 的连接状态/实际容差
  向下兼容保留。
- 为复用 native start patch，只扩展了 `BuildStartConnections` 和共享
  `AdvanceGridStep`；wheel A* 使用同一对角不切角规则。`PatchedElevationView` 仅转发
  已测中心的 `TerrainMeasurementsAt`，不为拟合支撑制造测量。

## TDD 证据

RED（Python，生产模块不存在）：

```text
$ python3 -m pytest ros2_ws/src/lunar_drl_exploration/test/test_contracts_maps.py -q
ModuleNotFoundError: No module named 'lunar_drl_exploration.config'
```

GREEN：

```text
$ python3 -m pytest ros2_ws/src/lunar_drl_exploration/test/test_contracts_maps.py -q
..... [100%]
5 passed

$ colcon ... build --packages-select lunar_incremental_navigation_core lunar_planning_msgs lunar_incremental_navigation_ros ...
Summary: 3 packages finished

$ colcon ... test --packages-select lunar_incremental_navigation_ros --ctest-args -R policy_map_exporter_test ...
Summary: 1 package finished

$ colcon ... test-result --all --verbose
request_local_start_patch_test: 13 tests, 0 failures
wheel_local_planner_test: 15 tests, 0 failures
policy_map_exporter_test: 1 test, 0 failures
```

最终重复：Jazzy cache `/home/kai/.cache/lunar-drl-redesign/jazzy` 中 messages/ROS 构建成功；
Python 5/5 通过；`git diff --check` 通过。

## 边界与关注点

- 这是源码/本机 Jazzy 构建和测试证据；Humble、Orin、DDS、rosbag、控制器闭环和实车均为
  `NOT_RUN`。
- 导出器为每个请求 tile 使用 native intrinsic evaluator；这是服务线程上的只读序列化工作，
  不进入 map/导航临界区。大面积 full snapshot 的延迟仍需在目标硬件测量。
- 未改动既有 OccupancyGrid publisher 计数测试的已知基线问题，也未引入 TCP、Unreal、Isaac、
  新图生产者、全局认证或任何 `/Car/T5/Car_Cmd_Vel` publisher。

## Round 1 修正

- `PolicyMapStore` 现在直接消费生成的 ROS `GetPolicyMap.Response`：Point origin、Pose
  quaternion 和 split start-connection arrays；不存在 pose/origin 不再静默替换为零值。
- action node 索引按 graph node 数量验证，并要求每个 action 恰有一行 yaw/world goal。
- `ElevationPipeline` 将每个 accepted raw revision 的输入地图 stamp 与 atomically published
  fine snapshot 一起保存；服务不再把异步晚到的输入地图 stamp 绑定到旧 fine。
- exporter 无完整历史时对 bootstrap 和任意 revision change 发送 true full snapshot；只有相同
  revision 才是无 tile 的 pose refresh，供 yaw-only action 使用。
- `load_platform_config()` 优先从 `lunar_incremental_navigation_ros` 已安装 share/config/wheel.yaml
  读取，源码测试才回退仓库 canonical 文件；未复制能力阈值。

Round 1 RED：生成 wire response 当前 client 抛 `TypeError: 'Point' object is not iterable`；
200-node/one-goal action 当前错误拒绝；pipeline 新 stamp API 编译失败；direct Export bootstrap
regression 失败（`full_snapshot=false`）。

Round 1 GREEN：source Jazzy overlay 下 `python3 -m pytest ...test_contracts_maps.py -q` 为 `7 passed`；
`ctest -R 'policy_map_exporter_test|elevation_pipeline_test'` 为 `2/2 passed`。

## Round 2 修正

- processed stamp 改为 bounded latest-accepted revision/stamp 生命周期；duplicate 在已发布 fine
  上原子刷新 stamp，pending/interleaved raw 不会泄漏给旧 fine。
- pipeline 保留 128 条 tile metadata journal；capture 从请求 base revision 聚合 tile union，
  bootstrap/history gap 保持 full，same revision 返回空 tile pose refresh。服务增加 `base_revision`。
- Python cache 使用 base revision 接受 journal 覆盖的连续 delta，保留 same revision refresh；wire
  start status 转换为 READY/START_BLOCKED/INPUT_UNAVAILABLE，零 quaternion anchor 明确拒绝。

Round 2 GREEN：Jazzy messages+ROS affected build 成功；`ctest -R
'policy_map_exporter_test|elevation_pipeline_test'` 2/2 通过；source overlay Python contracts 7/7；
`git diff --check` 通过。

补充回归：`test_real_wire_accepts_same_revision_refresh_and_journal_based_delta`、
`test_raster_handles_negative_partial_tiles_and_keeps_prior_snapshot_immutable` 已随 source overlay
运行；`JournalDeltaCoversCoalescedAcceptedRawTiles` 已在 `elevation_pipeline_test` 运行。

## Round 2 最终补齐（2026-09-15，起点 e82329b）

状态：DONE。最终修复范围在 `beffd51`、`65daa42`、`e82329b` 之上继续完成本节；
上文 Round 1 的“revision change 一律 full”是历史阶段行为，已由 Round 2 的 128 条
metadata journal、`base_revision` 连续区间和缺失历史 full fallback 取代。

### 本轮发现及修复

1. 受控阻塞 fine deriver 后，同 raw revision 的 duplicate 到达，旧实现仍发布启动派生时的
   stamp。现在在与 raw 输入共用的短锁内确认最新 accepted raw revision，只有它仍匹配
   待发布 raw 时才更新该 stamp；已收到更新 raw 时保持原捕获 stamp。journal 与 fine/stamp
   一起发布，capture 在同一锁下取得对应历史。duplicate 更新 immutable wrapper 使用 CAS，
   保留并发 guidance 发布。没有新增 raw revision→stamp 历史或完整地图副本。
2. 原生 fine 目录有意省略 M 全 UNKNOWN 的 tile，旧 Export 因只遍历 fine 目录漏掉其中
   已测中心高程。full 现在遍历 fine/raw tile 并集；delta 仍只读取 journal 指定 tile。
   没有 fine tile 时 M 为 UNKNOWN、cost 为 0，B/observed/height 仍取同一个 native raw。
   保守 halo 中既无 fine tile 也无有限测量的存储空隙不序列化，确保 delta/full tile 集合一致。
3. 修正旧 action 身份断言：action row 0 对应 graph index 1、node id 9，而 world goal
   从 goals row 0 读取。继续保留 200 个 graph node / 唯一 action 指向 node 199 的回归。
4. 将 Python same-revision / skipped-revision 测试全部改为实际生成的 ROS response，验证
   新 pose、stamp 输入的接受、tile 共享、旧 snapshot 不变、epoch delta 拒绝；新增四种 native
   start status 的语义转换、零 quaternion 拒绝后 cache 保持、raster 五字段及不可写 buffer 回归。

### 本轮 RED 证据

在修改生产源码前，扩展 `elevation_pipeline_test` 后运行：

```bash
source /opt/ros/jazzy/setup.bash
source /home/kai/.cache/lunar-drl-redesign/jazzy/install/setup.bash
cmake --build /home/kai/.cache/lunar-drl-redesign/jazzy/build/lunar_incremental_navigation_ros \
  --target elevation_pipeline_test -j 2
/home/kai/.cache/lunar-drl-redesign/jazzy/build/lunar_incremental_navigation_ros/elevation_pipeline_test \
  --gtest_filter='PolicyMapPipelineExport.*:PendingDuplicate/*:ElevationPipelineTest.DuplicatePendingBeforeDerivationAdvancesOnlyMatchingFine'
```

8 个用例中 6 passed、2 failed：

- `PolicyMapPipelineExport.MeasuredUnknownTileIsExportedWithoutAllocatedNavigationTile`：
  bootstrap 实际 `tiles.size() == 0`，预期 1。
- `PendingDuplicate/DuplicateDuringDerivation.PublishesExactRawStampPairWithoutNewerRawLeak/0`：
  实际 stamp 202，预期同 raw duplicate 的 303。

其余 journal/export 回归直接验证既有 Round 2 修复，未声称它们在 e82329b 上失败。
补充将 coalesced fixture 改为三个间隔 tile（0、2、4）后，发现首次 raw-only 修复将 halo
空隙也输出为 tile，delta 重建 5 个 tile 而 full 为 3；最终只保留 fine 或有实测中心的 tile。

注意：直接运行 build tree binary 时若已 source 旧 install overlay，`LD_LIBRARY_PATH` 可优先
加载旧 installed `.so`。中间验证曾明确把 build library 路径放在其前确认新库；下列最终证据
全部在 colcon 完整安装当前源码后运行，不以旧 overlay 结果代替当前源码验证。

### 回归名称与实际覆盖

新增 C++ 用例位于 `ros2_ws/src/lunar_incremental_navigation_ros/test/elevation_pipeline_test.cpp`，
其中 `PolicyMapPipelineExport` 直接执行真实 `CapturePolicyMapSnapshot` → `Export`，将 wire delta
逐 tile 应用于先前 full，然后逐一比较 `states`、`intrinsic_states`、`observed`、`costs`、
`elevation_m`（NaN-aware）与当前独立 full 导出：

- `HeightOnlyDeltaMatchesFullWithUnchangedNavigationFields`：height 改变，M/cost 数组和 native
  changed_tiles 均不变，delta 仍包含新 height。
- `ObservedOnlyDeltaMatchesFullWithUnchangedNavigationFields`：固有 BLOCKED→FREE、observed
  同步变化，邻近障碍保持 M BLOCKED/cost 不变，delta 与 full 完全一致。
- `MeasuredUnknownTileIsExportedWithoutAllocatedNavigationTile`：fine 目录为空时 bootstrap
  仍有实测 height；后续同 tile height 变化及远端新增 raw-only tile 的 delta/full 一致。
- `CoalescedRawUpdatesAndSkippedFineRevisionsMatchFull`：不同 sparse tile 的两次 raw 更新合并
  派生，再跳过一个 fine revision 取历史并集；比较全字段，并确认其他客户端请求不会消耗历史。
- `JournalRetains128TransitionsThenFallsBackToFull`：revision 1→129 的 128 条转换仍可 delta；
  revision 130 时同一 base 必须 full；近期 base 仍可 delta。两种响应均与 full 比较。
- `ElevationPipelineTest.DuplicatePendingBeforeDerivationAdvancesOnlyMatchingFine`：派生前
  pending duplicate 不刷新旧 fine，随后发布匹配 raw 的最新 stamp。
- `PendingDuplicate/DuplicateDuringDerivation.PublishesExactRawStampPairWithoutNewerRawLeak/0`
  和 `/1`：使用既有 deriver 依赖注入及 promise gate；覆盖派生中 duplicate、并发新 raw 与它的
  duplicate、rejected adapter / raw 输入不改 stamp、旧 capture 不变。没有依赖时间 sleep 猜测交错。

已有并重跑：`DuplicateRefreshesPublishedFineStampWithoutChangingMap`（发布后 duplicate）、
`FineSnapshotRetainsItsExactAcceptedMapStamp`、`JournalDeltaCoversCoalescedAcceptedRawTiles`、
并发 guidance/fine capture、失败批次恢复及 snapshot immutability 等 pipeline 用例；
`policy_map_exporter_test` 的 effective observed 与 direct bootstrap/same-revision refresh 用例。

Python 新增 / 强化名称：

- `test_real_wire_accepts_same_revision_refresh_and_journal_based_delta`
- `test_real_wire_native_start_status_is_semantic`（0/1→READY，2→START_BLOCKED，3→INPUT_UNAVAILABLE，4 cases）
- `test_real_wire_missing_anchor_rejected_without_replacing_cached_snapshot`
- `test_raster_handles_negative_partial_tiles_and_keeps_prior_snapshot_immutable`
- `test_decision_observation_freezes_action_indices_and_world_goals_at_construction`

### 最终验证命令及计数

从 worktree 根执行（先 source ROS，再 source 已安装 overlay）：

```bash
source /opt/ros/jazzy/setup.bash
source /home/kai/.cache/lunar-drl-redesign/jazzy/install/setup.bash
test "$ROS_DISTRO" = jazzy
colcon --log-base /home/kai/.cache/lunar-drl-redesign/jazzy/log build \
  --base-paths ros2_ws/src \
  --build-base /home/kai/.cache/lunar-drl-redesign/jazzy/build \
  --install-base /home/kai/.cache/lunar-drl-redesign/jazzy/install \
  --packages-select lunar_planning_msgs lunar_incremental_navigation_ros \
  --executor sequential --cmake-args -DBUILD_TESTING=ON
source /home/kai/.cache/lunar-drl-redesign/jazzy/install/setup.bash
ctest --test-dir /home/kai/.cache/lunar-drl-redesign/jazzy/build/lunar_incremental_navigation_ros \
  -R '^(policy_map_exporter_test|elevation_pipeline_test)$' --output-on-failure -j 1
python3 -m pytest ros2_ws/src/lunar_drl_exploration/test/test_contracts_maps.py -q
git diff --check
```

实际结果：affected build **2 packages finished**；CTest **2/2 targets passed**，XML 明细
`elevation_pipeline_test` **21 tests, 0 failures, 0 errors**，`policy_map_exporter_test`
**2 tests, 0 failures, 0 errors**；Python **14 passed**；UTF-8 显式读取与 `git diff --check` 通过。

构建、安装、日志与测试 XML 均在仓库外 `/home/kai/.cache/lunar-drl-redesign/jazzy/`；
本轮提交仅源码、回归测试及本报告。未修改 native controller，也未 merge/push/创建 PR。
本轮属于源码和本机 Jazzy evidence；Humble、Orin、DDS、rosbag、控制器闭环和实车 **NOT_RUN**。
