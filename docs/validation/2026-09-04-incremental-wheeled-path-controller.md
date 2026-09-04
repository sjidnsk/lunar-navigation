# 增量轮式路径控制器验证记录

- 日期：2026-09-04
- 分支：`feat/incremental-wheeled-path-controller`
- 验证实现基线：`1abb6c2 docs: document incremental wheel controller operation`
- 范围：`lunar_pure_wheeled_controller` 的显式 `incremental_path` 输入模式，以及增量导航器已发布的
  `/Car/T4/planning/local_path` 配置/文档合同。

## 已验证的源码边界

- 默认 `input_mode=motion_reference` 不变；`incremental_path` 只订阅局部 `Path`、odometry 与 `/tf`，
  不启动 planner、explorer 或控制器以外的节点。
- 非空路径必须为 `map` frame；控制器只采用当前 `/tf` 消息中的直接 `map <- odom` 边，不做图遍历、
  时间同步或历史变换。
- 空/非法路径、轨迹偏离和完成会清空目标并发布零速；暂缺 odometry 或直接 TF 时发布零速但保留最近有效路径。
- 顶层 `exploration_navigation.launch.py` 仍只构造增量导航与探索两个节点；轮式控制器保持显式独立启动。

这些是源码和本机/容器测试结论，不是实际 ROS 图、控制授权或车辆运动结论。

## Python 与静态合同

在仓库根目录、Jazzy 环境下运行：

```bash
source /opt/ros/jazzy/setup.bash
source /tmp/lunar-incremental-controller-tdd-pCG7Z5/install-interfaces/setup.bash
PYTHONPATH=ros2_ws/src/lunar_pure_wheeled_controller/python:$PYTHONPATH \
  python3 -m pytest -q -p no:cacheprovider \
  ros2_ws/src/lunar_pure_wheeled_controller/test \
  tests/test_incremental_navigation_interface_contract.py \
  tests/launch/test_exploration_navigation_launch.py \
  tests/test_launch_contract.py
```

结果：`152 passed in 0.76s`。覆盖路径解析、直接平面变换、终点 yaw 原地对齐、控制器零速/恢复语义、
旧 `MotionReference` 回归、QoS、YAML 以及“顶层 launch 不启动控制器”合同。

## Fresh Jazzy 构建与包测试

所有 Jazzy build/install/log 都位于仓库外的
`/tmp/lunar-incremental-controller-verify-3wYFr8`：

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon --log-base /tmp/lunar-incremental-controller-verify-3wYFr8/log build \
  --base-paths src \
  --packages-up-to lunar_incremental_navigation_ros lunar_pure_wheeled_controller \
  --build-base /tmp/lunar-incremental-controller-verify-3wYFr8/build \
  --install-base /tmp/lunar-incremental-controller-verify-3wYFr8/install
source /tmp/lunar-incremental-controller-verify-3wYFr8/install/setup.bash
colcon --log-base /tmp/lunar-incremental-controller-verify-3wYFr8/log-test test \
  --base-paths src \
  --packages-select lunar_pure_wheeled_controller lunar_incremental_navigation_ros \
  --build-base /tmp/lunar-incremental-controller-verify-3wYFr8/build \
  --install-base /tmp/lunar-incremental-controller-verify-3wYFr8/install \
  --parallel-workers 1 --return-code-on-test-failure
```

结果：依赖闭包中的 4 个包构建完成；包测试串行通过，`184 tests, 0 errors, 0 failures, 0 skipped`。

## Humble 容器构建与复核

使用 `osrf/ros:humble-desktop-full-jammy`，源码以只读方式挂载到 `/workspace`，所有构建、安装和日志
均在容器 `/tmp`。安装 `python3-colcon-common-extensions` 与 `ros-humble-grid-map-msgs` 后，
`lunar_incremental_navigation_core`、`lunar_planning_msgs`、`lunar_incremental_navigation_ros` 和
`lunar_pure_wheeled_controller` 均构建成功。

第一次容器运行遗漏 `--log-base /tmp/...`，`colcon` 尝试向只读 `/workspace/ros2_ws/log` 写入并以
`OSError: [Errno 30] Read-only file system` 停止；修正日志路径后重新构建，未修改任何源码。

修正后的首次包测试中，控制器 4 个 CTest 单元（96 个 Python 测试）全部通过。增量导航器的 12 个
CTest 单元中有 1 个首次失败：
`IncrementalNavigationNode.NewGoalAlwaysPreemptsAndEveryTerminalInvalidatesTheActiveSegmentFirst` 在
`incremental_navigation_node_test.cpp:1211` 观察到事件数 `4`，低于断言下限 `6`。本分支未修改
`lunar_incremental_navigation_ros` 源码；该测试直接构造节点，不读取本次增加的顶层 YAML 参数。

为区分偶发时序与稳定回归，在相同已构建 Humble 镜像、相同只读源码挂载下执行：

```bash
/tmp/lunar-build/lunar_incremental_navigation_ros/incremental_navigation_node_test \
  --gtest_filter=IncrementalNavigationNode.NewGoalAlwaysPreemptsAndEveryTerminalInvalidatesTheActiveSegmentFirst \
  --gtest_repeat=10
```

结果：连续 `10/10` 通过；随后对整个 `lunar_incremental_navigation_ros` 构建目录执行
`ctest --output-on-failure`，结果 `12/12` 通过。该首轮事件时序观察已保留在本记录中，但未以此修改
导航器或扩大控制器范围。

## 未运行边界

以下均为 `NOT_RUN`：Jetson AGX Orin 原生构建与性能、DDS 跨机通信、真实
`/Car/T4/planning/local_path`/`/tf`/odometry 样本、真实底盘命令仲裁、急停流程和实车闭环路径跟踪。
本记录不产生 `planning_outcome: 0`、`reason_code: PLAN_FOUND`、`has_reference: true` 的现场规划成功证据，
也不宣称车辆或控制器已获运行授权。
