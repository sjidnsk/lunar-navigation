# Incremental Exploration-Navigation 集成验证记录

- 日期：2026-09-03
- 集成分支：`feat/incremental-exploration-navigation`
- 实现基线：`77102c5 docs: record incremental exploration navigation validation`（最终审查修复前）
- 最终审查修复：`fix/incremental-exploration-navigation-review`（接口契约修复与后续 demo 回归稳定化提交）
- 环境：本机 Ubuntu / ROS 2 Jazzy
- 最终审查临时构建根：`/tmp/lunar-final-verify2-ZXzCb8`

## 范围与结论边界

本记录验证新增 `incremental_v2` 的接口隔离、唯一增量地图、探索决策—导航会话分层、wheel/legged
既定 8 邻域规划回归、双栈 launch 合同和无 RViz GUI 的本机闭环。它不修改新版规划算法、legacy
规划/探索行为、控制器、Hopper、TF 发布器或课题三数据生产端。

legacy 路径仍为原 `PlanMotion.action`、`pure_exploration_node`、`lunar_pure_planner_*` 与
`/Car/T3/mapping/global_overview: nav_msgs/msg/OccupancyGrid`；新版通过
`NavigateToPose` 与 `PathReference` 独立运行。两条栈由启动时的 `stack_mode` 互斥选择，未合并到
`integration/pure-planner-orin`，未推送、未创建 PR。

## 输入、地图和运行时职责

| 项目 | 类型/语义 | 责任 |
| --- | --- | --- |
| `/Car/T3/mapping/grid_map` | `grid_map_msgs/msg/GridMap`，只读取 `elevation` layer | 仅 `incremental_navigation` 订阅，累计为唯一 `PersistentElevationMap`。 |
| `/Car/T3/localization/odometry` | `nav_msgs/msg/Odometry`；`odom -> base_link` | 导航器与探索器的状态输入。 |
| `/tf` | `tf2_msgs/msg/TFMessage` | 直接 `map -> odom` 变换。 |
| `/Car/T4/mapping/exploration_map` | `nav_msgs/msg/OccupancyGrid`；`-1/0/100` | 导航器从同一 fine snapshot 发布；探索器只读。QoS：Reliable + Transient Local + KeepLast(1)。 |
| `/Car/T4/navigation/navigate_to_pose` | `lunar_planning_msgs/action/NavigateToPose` | 探索器唯一 client，导航器 server；探索器一次只有一个 current goal。 |
| `/Car/T4/planning/path_reference` | `lunar_planning_msgs/msg/PathReference` | 导航器发布会话路径；探索器不订阅它。 |

新版不订阅、等待、转换或检查 `/Car/T3/mapping/global_overview`、`/Car/T3/semantic/current_pose`。第一个
有效局部 `GridMap` 冻结 fine resolution；`0.2 m` 与 `0.1 m` 都直接保持输入分辨率，改变分辨率需要
重启，不存在运行时热切换或 `0.1 -> 0.2` 静默重采样。`coarse_resolution_m` 默认 `1.0 m`。

规划周期的正式成功判据是 `cycle_result=PLAN_FOUND`、非空 `PathReference.state=ACTIVE`，并且
segment/traversability revision 一致。目标导航完成的正式判据是 Action Result `GOAL_REACHED`；
Action accepted、单独 RViz 线条或仅有 `PathReference` 都不充当完成证据。

## 最终审查修复的契约闭合

本次只修复四项最终审查契约，不改轮式/足式的 8 邻域 A* 展开、代价、足式有向边认证、legacy
explorer 或控制器：

- `navigation.local_window_size_m` 默认 `64.0 m`，局部窗口按 fine resolution 换算：`0.2 m -> 320`
  格/轴，`0.1 m -> 640` 格/轴。固定局部工作区容量同步为 `640` 格/轴；超过容量时返回
  `LOCAL_WINDOW_CAPACITY_EXCEEDED`，不再把物理窗口静默裁成旧的 320 格。
- 显式 `stack_mode:=legacy` 在读取 YAML 或查找增量导航包之前分支，使用 legacy 自己的默认值；因此
  在 `lunar_incremental_navigation_ros` 未安装且 `config_file` 缺失/不可读时仍可启动。默认 YAML 和顶层
  launch 由 `lunar_pure_exploration_ros` 安装；`incremental_v2` 仍会读取 YAML 并要求增量导航包。
- 新 explorer 入口只接受 `/Car/T4/mapping/exploration_map` 的精确 `{-1, 0, 100}` 值。`49`、`50`、`101`
  均产生 `INVALID_EXPLORATION_MAP`，且不会提交 goal；共享 legacy `OccupancyGridView` 未修改。
- 对外状态契约统一为 `odom -> base_link`，同时更新接口 YAML、顶层 YAML、launch、导航 state adapter
  和 explorer 的严格 PoseResolver。轮式 `base_frame_id: base_footprint` 保留为 footprint 几何参数，不是
  odometry child frame。

RViz scenario 标注的 `16 m` 局部观测仍是输入地图范围，不是 `local_window_size_m`。它可验证 demo
闭环，不可当作 64 m 规划窗口的容量或性能证据。

## Fresh Jazzy 构建与回归

先按 Task 7 原始 five-package 选择命令在全新 `/tmp` install 中构建：

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base "$INCREMENTAL_BUILD_DIR/log" build --base-paths ros2_ws/src \
  --packages-select lunar_planning_msgs lunar_incremental_navigation_core \
    lunar_incremental_navigation_ros lunar_pure_exploration_core \
    lunar_pure_exploration_ros \
  --build-base "$INCREMENTAL_BUILD_DIR/build" \
  --install-base "$INCREMENTAL_BUILD_DIR/install"
```

该命令在配置 `lunar_pure_exploration_ros` 前停止，错误为缺少以下既有 legacy 同包 target 的
`package.sh`：`lunar_pure_exploration_msgs`、`lunar_pure_planner_core`、
`lunar_pure_planner_ros`。这是 fresh install 未包含 legacy explorer 既有编译依赖的前提问题，
不是 incremental 接线、算法或测试失败；未为此修改源码。

随后在**同一个** fresh 临时根用依赖闭包重建：

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base "$INCREMENTAL_BUILD_DIR/log-closure" build --base-paths ros2_ws/src \
  --packages-up-to lunar_incremental_navigation_ros \
    lunar_pure_exploration_ros lunar_pure_planner_ros \
  --build-base "$INCREMENTAL_BUILD_DIR/build-closure" \
  --install-base "$INCREMENTAL_BUILD_DIR/install-closure"
source "$INCREMENTAL_BUILD_DIR/install-closure/setup.bash"
```

结果：8 个包构建完成（含 legacy 编译依赖），所有 build/install/log 均在 `/tmp`，仓库工作树没有构建产物。
随后按 `-j1` 串行运行计划指定的四个包：

| 包 | CTest 结果 |
| --- | --- |
| `lunar_incremental_navigation_core` | `13/13` passed |
| `lunar_incremental_navigation_ros` | `9/9` passed |
| `lunar_pure_exploration_core` | `13/13` passed |
| `lunar_pure_exploration_ros` | `12/12` passed |

接口/launch/RViz 合同命令：

```bash
python3 -m pytest -q \
  tests/test_incremental_navigation_message_contract.py \
  tests/test_incremental_navigation_interface_contract.py \
  tests/launch/test_exploration_navigation_launch.py \
  tests/launch/test_incremental_exploration_navigation_rviz_contract.py
```

原始结果为 `31 passed in 21.93s`。最终审查修复在此基础上新增物理窗口、显式 legacy 回退、三态图和
`odom -> base_link` 的回归；本次重跑结果与命令见下一节，不能把历史结果代替修复后的验证。

### 最终审查修复重跑

最终使用新的 `/tmp/lunar-final-verify2-ZXzCb8` Jazzy 依赖闭包重建 8 个包；唯一 stderr 是
`lunar_pure_planner_ros` 既有的 C++ explicit-constructor warning，构建退出码为 0。以
`ROS_LOCALHOST_ONLY=1` 和隔离 `ROS_DOMAIN_ID` 分包重跑 CTest，结果为：

| 包 | CTest 结果 |
| --- | --- |
| `lunar_incremental_navigation_core` | `13/13` passed |
| `lunar_incremental_navigation_ros` | `9/9` passed |
| `lunar_pure_exploration_core` | `13/13` passed |
| `lunar_pure_exploration_ros` | `12/12` passed |

新增 core 回归实际构造 `local_window_size_m=64.0`：`0.2 m -> 320 × 320`、
`0.1 m -> 640 × 640`，并验证超容量返回 `LOCAL_WINDOW_CAPACITY_EXCEEDED`；wheel/legged 的 640-cell
工作区回归也已覆盖。这是容量/正确性证据，不是整栈实时性或 Orin 性能结论。

首轮 aggregate CTest 曾各出现一次既有 legacy 并发测试和新三态测试的状态时序现象。legacy
`FineMapValidationOverBudgetFailsClosedWithoutLeavingFinalRankBusy` 未改源码，隔离环境连续 3 次通过；
新三态回归改为记录状态历史而不是读取可能被后续回调覆盖的最后状态，连续 3 次通过，并在最终包 CTest
中通过。这里不把偶发时序现象隐去，也不将其归因于轮式/足式 A*。

同一 fresh install 的归属检查确认 `exploration_navigation.yaml` 和顶层
`exploration_navigation.launch.py` 仅存在于 `lunar_pure_exploration_ros`，在
`lunar_incremental_navigation_ros` 与 `lunar_pure_planner_ros` 均不存在。已 source 此闭包后，接口、launch
和 RViz 合同实跑 `37 passed in 24.46s`：覆盖缺少增量导航包/缺失配置时的显式 legacy、仍依赖该包的
incremental_v2、默认 YAML 归属、`49/50/101` 三态图拒绝、`base_footprint` odometry 拒绝，以及四个
headless demo 场景。

NO_PATH 场景使用已有的 `LUNAR_DEMO_REQUIRE_NO_PATH_RECOVERY=1` 测试开关：场景节点在首个 ACTIVE path
后仅 15 个 200 ms 周期叠加局部障碍带，然后恢复常规局部图。因此首候选得到 `NO_PATH`，explorer 可提交
不同候选；探针不再创建与 scenario 竞争的第二个 `GridMap` publisher。该场景连续 3 次得到
`no_path_observed=true`、`candidate_replaced=true` 和 `clean_shutdown=true`。它只用于隔离 demo/自动回归，
不是生产输入、地图源或 A* 参数。Humble/Orin 未运行。

## Jazzy 无 RViz 闭环与运行时图

以下为最终 Jazzy 闭包的 headless probe 记录：

| 重启场景 | 实测结果 |
| --- | --- |
| wheel + `0.2 m` | 收到 `GridMap`、探索图、frontier、唯一 goal 与非空 ACTIVE path；诊断记录 `PLAN_FOUND`；resolution 断言为真。 |
| wheel + `0.1 m` | 同上，`fine_resolution_m=0.1` 且 resolution 断言为真；没有重采样参数。该 demo 的输入观测仅 16 m，不能证明 64 m 局部规划窗口性能。 |
| legged + `0.2 m` | 同上，`platform=legged`，未用 wheel 配置替代。 |
| wheel + `0.2 m`，首个 ACTIVE path 后短时局部障碍带 | `no_path_observed=true`、`candidate_replaced=true`、`unique_goals=2`；关闭前导航 Action 已无 active goal。 |

四次 probe 均报告 `global_overview_publishers=0`、`ground_truth_algorithm_subscribers=0`、
`control_topic_present=false` 和 `clean_shutdown=true`。它们在关闭时先发布任务 `CANCEL`，再确认 explorer
取消与 Action 无活跃状态；不以 RViz 线条代替任何规划或导航成功判据。

另在 wheel + `0.2 m` 的 live launch 中执行：

```bash
ros2 topic info -v /planning_demo/grid_map
ros2 topic info -v /planning_demo/mapping/exploration_map
ros2 action info /planning_demo/navigation/navigate_to_pose
ros2 node info /incremental_navigation
ros2 node info /incremental_exploration
```

实际图为：

- `/planning_demo/grid_map` 类型 `grid_map_msgs/msg/GridMap`，publisher 仅
  `incremental_demo_scenario`，subscriber 仅 `incremental_navigation`，QoS 为 Reliable + Transient Local；
- `/planning_demo/mapping/exploration_map` 类型 `nav_msgs/msg/OccupancyGrid`，publisher 仅
  `incremental_navigation`，subscriber 仅 `incremental_exploration`，QoS 为 Reliable + Transient Local；
- `/planning_demo/navigation/navigate_to_pose` 有 1 个 Action client
  (`incremental_exploration`) 和 1 个 Action server (`incremental_navigation`)；
- navigator 的订阅只有 demo GridMap、odometry 与 `/tf`；explorer 的订阅只有 task、正式探索图、odometry
  与 `/tf`，没有 GridMap；
- `/Car/T3/mapping/global_overview`、`/Car/T3/semantic/current_pose` 和 `/Car/T5/Car_Cmd_Vel` 不在该 live graph。

首次刚启动后立即调用 ROS CLI 曾遇到 discovery 未稳定：node list 已显示 navigator，但一次
`topic info`/`action info` 尚报 0 publisher/server。进程树与 launch 日志确认节点没有退出；短暂等待后
复查得到上表完整图。故只使用第二次稳定图作为证据，不把首次缓存/发现时序当作产品失败或通过。

probe 在活跃周期记录了 `PLAN_FOUND` 与 ACTIVE 非空路径。手工点读较晚的 latched
`PathReference` 时，任务已进入覆盖终态，读到 `state=INVALIDATED` 且空 path；这符合终态失效生命周期，
不被作为 ACTIVE 成功证据。NO_PATH probe 的收尾中
`navigator_canceled_after_cancel=false`、且可选 `path_invalidated_after_cancel` 在该场景为 false；关闭判据
使用公开 Action status 的无活跃状态与 clean shutdown，不将上述非必需诊断观察写成成功条件。

## RViz 演示设计与证据级别

演示 launch 为
`lunar_incremental_navigation_ros/incremental_exploration_navigation_rviz.launch.py`；全算法 Topic 位于
`/planning_demo/*`，仅 `/tf` 例外。RViz Fixed Frame 为 `map`，布局应含任务边界、三态任务图、16 m
局部观测窗口、frontier、唯一洋红 current goal、fine traversability、细蓝 global route、粗橙 active path、
robot/trace 与 HUD；可选 ground truth 只供显示，不是算法输入。配置刻意不显示 A* OPEN/CLOSED。

以上四个场景是 `start_rviz:=false` 的自动 Jazzy 闭环证据。RViz GUI 的人工视觉检查为 `NOT_RUN`；应在
实际启动 `start_rviz:=true` 后逐项核对上述布局和必演场景，不能将 headless 结果表述为 GUI 验收。

## 已知历史观察与未运行边界

Task 6 的历史验证曾在 legacy `test_exploration_node` 与 `test_synthetic_scenarios` 各观察到一次非确定性
时序失败。未改 legacy 源码或测试后，Task 6 的独立复跑与整包复跑最终全绿；本次 fresh final CTest 的
`lunar_pure_exploration_ros` 也为 `12/12` passed。这是最终事实，不据此宣称 legacy 已获得稳定性证明。

本记录只证明源码静态合同与本机 ROS 2 Jazzy 构建/测试/无 GUI 演示。以下均为 `NOT_RUN`：

- ROS 2 Humble 主机或容器构建与运行；
- Jetson AGX Orin 构建、性能、功耗和稳定性；
- DDS 跨机通信、真实 rosbag、真实 mapper/TF；
- RViz GUI 人工视觉验收；
- 控制器、`/Car/T5/Car_Cmd_Vel`、实车导航与实车探索。

因此不得由本记录外推生产部署、控制授权或实车就绪。
