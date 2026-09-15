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
