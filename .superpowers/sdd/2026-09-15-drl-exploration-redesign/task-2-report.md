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
