# High-Resolution Traversability Planner V1 Validation

## 结论与边界

2026-08-25 在 x86_64 主机、ROS 2 Jazzy、Release 构建下完成 V1 的基础设计与诊断验收。
`grid_traversability_v1` 仍为显式启用模式，生产默认保持 `legacy_certified`。本记录证明主机代码与
ROS 接口行为，不代表 Jetson Orin、课题三现场 DDS、控制器跟踪或实车就绪。

## 已验证合同

- local GridMap、odometry 和 `map -> odom` TF 可独立建立高分辨率可通行快照；global overview 是可选粗先验，无新增外部可通行图输入。
- 首个有效 local GridMap 锁定 canonical resolution；地图维护、全局/局部搜索、捷径和发布前复检均不降采样。
- 地图采用稀疏 `256 x 256` tile、全局先验和局部覆盖，并以不可变 revision 快照贯穿一次规划。
- V1 只用于 `wheel + LUNAR_SURFACE`，不调用 legacy 后端，也不在失败时回退 legacy。
- 全局稀疏二维 A* 后生成最长 8 m 的局部段；整段选择前进或倒车并生成有界速度与严格递增时间。
- 成功的 `MotionReference.path_preview`、`nav_msgs/msg/Path` 和 `TimedPath.path` 位姿序列一致；
  `TimedPath.planning_time` 与 diagnostics `total_elapsed_ms` 一致。
- 发布前在最新 revision 上复检；路径被新观测阻断时返回 `STALE_PATH_INVALIDATED`，失败输出为空。
- 正式成功条件为 `planning_outcome=0`、`has_reference=true` 和非空 trajectory。

## 构建与测试证据

构建命令使用 `ros2_ws/build-v1` 和 `ros2_ws/install-v1`，构建类型为 Release：

```bash
source /opt/ros/jazzy/setup.bash
colcon build --build-base build-v1 --install-base install-v1 \
  --packages-up-to lunar_pure_planner_ros \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
```

结果：`lunar_planning_msgs`、`lunar_pure_planner_core`、`lunar_pure_planner_ros` 三包构建成功。

最终全量包测试：core `21/21` CTest targets、ROS `15/15` CTest targets；
`colcon test-result --verbose` 汇总 `456 tests, 0 errors, 0 failures, 0 skipped`。其中包括：

- 持久高分辨率地图 `18/18`；
- V1 全局/局部规划、复检、方向与时序 `11/11`；
- 双模式入口 `34/34`，证明 V1 不调用注入的 legacy backends；
- ROS V1 输入顺序、正式成功、三路径一致、规划时间和 stale 清空路径；
- ROS launch `9/9`。

仓库静态合同测试：

```bash
python3 -m pytest -q \
  tests/test_external_interface_contract.py \
  tests/exploration/test_exploration_isolation.py \
  tests/test_launch_contract.py
```

结果：`50 passed`。

## 30 次固定场景观测

同一 Release ROS 正式成功用例重复 30 次，30/30 均为 `planning_outcome=0`、
`has_reference=true`、canonical resolution `0.2 m`；全局和局部各展开 2 个状态，分配 1 个 tile，
估算地图存储 `65536 bytes`。

| diagnostics 字段 | p50 (ms) | p95 (ms) | max (ms) |
|---|---:|---:|---:|
| map fusion | 0 | 0 | 0 |
| traversability recompute | 0 | 0 | 0 |
| global | 0.011962 | 0.022743 | 0.026820 |
| local | 0.007824 | 0.016310 | 0.019236 |
| postprocess | 0 | 0 | 0 |
| total | 0.277000 | 0.532178 | 0.557436 |

固定用例在 Action 前完成异步地图更新，当前首版没有把该异步更新和后处理内部耗时回填到请求摘要，
因此表中三个 `0` 只能解释为“该字段在此夹具中未计量”，不能解释为零计算成本。首版硬验收使用
可验证的 global/local/total 计时和 TimedPath 总规划时间；进程峰值 RSS 与目标机时延门槛未测。

## 指定 rosbag 的 local-only 验收

2026-08-25 在同一 Jazzy 主机回放
`/home/kai/CodexDownloads/lunar_navigation/lunar_pure_planner_orin/t3_20260824_163252`；
命令与结果摘录保存在
[2026-08-25-t3-local-only-rosbag.md](evidence/2026-08-25-t3-local-only-rosbag.md)：

- bag 共 626 条、409.469749193 s；`global_overview=0`、`grid_map=1`、`odometry=312`、`tf=312`；
- 以隔离 DDS 域、`--clock --rate 50 --loop` 启动显式 V1，没有生成 synthetic global；
- 月表请求诊断为 `global_input_sequence=0`、`canonical_resolution_m=0.2`、`traversability_revision>0`，证明缺 global 不再导致 `INVALID_INPUT`；
- 目标 `(2.0,-0.2)` 的首次及覆盖至少一个完整循环的 16 次请求均到达核心地图门，阶段原因稳定为 `START_NOT_FREE`；失败输出为空 Path/TimedPath，TimedPath 保留规划耗时；
- 因 bag 仅含一帧局部图，且现有膨胀语义把回放里程计起点判为不可通行，本 bag 未达到 `planning_outcome=0 + has_reference=true`。没有为追求成功而注入假里程计、伪造全局图或放宽安全语义。

## 限定审核

1. 实现 diff 均对应 V1 地图、搜索、ROS 接线、输出、诊断或合同测试；未加入控制器和新算法族。
2. V1 dispatch 位于 legacy 入口之前，缺快照或规划失败直接返回，不会调用 legacy。
3. canonical resolution 来自首个有效 local GridMap，代码和测试均未出现降采样路径。
4. ROS 测试证明成功输出三路径同源，失败和 stale 发布空 Path/TimedPath 且不留下新执行参考。

两轮限定复审的剩余 Critical/Important 均为零。异常注入式 postprocess fallback 动态覆盖未增加生产
测试 seam；真实捷径拒绝与 raw-path 安全复检已覆盖，这是保留的证据限制。

## NOT_RUN

- ROS 2 Humble 原生构建；
- Jetson Orin 构建与性能；
- 课题三真实发布者 DDS/QoS 联调和多帧局部地图连续滚动；
- 控制器、`/Car/T5/Car_Cmd_Vel`、仿真闭环和实车。
