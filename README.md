# Pure Planner 操作手册

`lunar_pure_planner_ros` 是独立的普通 ROS 2 节点，不使用 Lifecycle manager，也不启动旧规划器。

## 纯前沿探索整合

目标目录已加入三个物理隔离的探索包：

- `lunar_pure_exploration_core`：任务区域栅格化、可达前沿检测、候选生成与评分、覆盖率统计；
- `lunar_pure_exploration_msgs`：任务区域与探索状态消息；
- `lunar_pure_exploration_ros`：订阅课题三地图/位姿/TF，调用 `/Car/T4/plan_motion`，发布执行参考和探索状态。

整合来源为 `feat/pure-exploration-planner-integration` 的
`66d86363e62e784f0341c72bd8e47b5cc8336d3a`；冻结设计与执行记录位于
`docs/superpowers/specs/2026-08-22-pure-frontier-exploration-design.md` 和
`docs/superpowers/plans/2026-08-22-pure-frontier-exploration.md`。

探索结束条件只有“当前任务区域内已不存在可达前沿”；覆盖率只统计并发布，不作为结束阈值。输入
stamp、地图版本、协方差、地图新鲜度和观测时间不参与探索准入。全局探索使用
`/Car/T3/mapping/global_overview` 的原生 `nav_msgs/msg/OccupancyGrid` 语义，局部高度与可通行性仍由
规划器处理。传感器模型为 10 m、90°，所有项目 Topic 均位于 `/Car/T4/...`。

START 时若完整车体尚未进入任务多边形，协调器先进入 `APPROACH_TASK`：`boundary` 上的
UNKNOWN 只生成探索意图，实际下发给规划器的目标始终是当前地图中经完整车体包络和间距校验的
FREE 位姿；完整车体进入后切换为 `EXPLORE_TASK` 并继续原有 WFD。规划 Action 显式绑定冻结的
`grid_traversability_v1`，不复制规划算法、不回退 legacy。入口不可达、地图尚未覆盖任务边界或
入口停滞分别稳定报告 `APPROACH_NO_REACHABLE_TARGET`、
`WAITING_FOR_TASK_MAP_COVERAGE`、`APPROACH_STALLED`，都不能证明探索完成。诊断同时发布
`navigation_phase`、`fully_inside_task`、入口意图/候选数量、剩余引导距离、引导 UNKNOWN
数量和等待原因；`current_goal` 仍只表示实际安全执行目标。

入口资源默认继承任务栅格与候选上限，搜索工作量为经溢出检查的
`8 × maximum_guidance_grid_cells`；300 m 测试 launch 显式使用
`1048576 / 1048576 / 8388608 / 4096`。覆盖率仍只用于观测和验收断言，唯一完成条件仍是进入
`EXPLORE_TASK` 后 Grid V1 对当前 WFD 候选给出穷尽不可达证据。

本机 Humble 验证 overlay 位于 `install-humble-exploration-66d8636`，与原有 `install` 分开。在
Ubuntu 22.04 + ROS 2 Humble 中启动前执行：

```bash
cd /home/kai/CodexDownloads/lunar_navigation/lunar_pure_planner_orin
source /opt/ros/humble/setup.bash
source install-humble-exploration-66d8636/setup.bash
ros2 launch lunar_pure_exploration_ros pure_exploration.launch.py \
  platform_selector:=wheel \
  maximum_position_probes:=64 \
  maximum_candidate_views:=64 \
  maximum_collision_work_units:=4096 \
  maximum_visibility_work_units:=4096 \
  maximum_path_preview_poses:=64 \
  maximum_executable_path_points:=64 \
  maximum_failure_entries:=8 \
  maximum_failure_patch_cells_per_entry:=256 \
  maximum_failure_total_patch_cells:=1024
```

这些容量值只用于本机功能联调，不代表 Jetson AGX Orin 性能验收结果。启动规划器、课题三输入和
控制器后，通过区域角点下发任务：

```bash
ros2 topic pub --once /Car/T4/exploration/task \
  lunar_pure_exploration_msgs/msg/PureExplorationTask "{
    header: {frame_id: map},
    task_id: area-001,
    command: 1,
    boundary: {points: [
      {x: 0.0, y: 0.0, z: 0.0},
      {x: 20.0, y: 0.0, z: 0.0},
      {x: 20.0, y: 20.0, z: 0.0},
      {x: 0.0, y: 20.0, z: 0.0}
    ]}
  }"
```

状态与覆盖率读取：

```bash
ros2 topic echo /Car/T4/exploration/status
```

当前 Humble 整合验证：三个探索包构建通过，core 201 项和 ROS 134 项测试均为 0 失败；安装态
假 `PlanMotion` server 的 Goal、Result、MotionReference 握手通过。真实课题三在线输入、真实控制器、
车辆执行和 Jetson AGX Orin 性能/稳定性尚未运行，不能由本机验证替代。

## Jazzy 300 m 探索闭环（测试专用）

这套入口在本机 ROS 2 Jazzy 中组合固定种子的 `300 m × 300 m` 月表、纯探索器、纯规划器、
前后轮协调转向控制器、结果记录器和 RViz2。它使用 `/Car/T3/...` 输入、`/Car/T4/...` 探索与
规划接口以及唯一的 `/Car/T5/Car_Cmd_Vel` 控制输出；不会接入实车。

### 依赖与仓库外构建

系统需安装 ROS 2 Jazzy desktop、`python3-colcon-common-extensions`、`python3-rosdep` 和
`nlohmann-json3-dev`。`nlohmann_json` 也可以由已 source 的外部前缀提供。推荐先让 rosdep 核对
工作区，再把 build/install/log 全部放到仓库外：

```bash
cd /home/kai/CodexDownloads/lunar_navigation/lunar_pure_planner_orin-worktrees/jazzy-300m-exploration
source /opt/ros/jazzy/setup.bash
sudo apt-get install nlohmann-json3-dev
rosdep install --from-paths ros2_ws/src --ignore-src -r -y --rosdistro jazzy

export LUNAR_JAZZY_BUILD=/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/closed_loop
mkdir -p "$LUNAR_JAZZY_BUILD"
colcon --log-base "$LUNAR_JAZZY_BUILD/log" build \
  --base-paths ros2_ws/src \
  --build-base "$LUNAR_JAZZY_BUILD/build" \
  --install-base "$LUNAR_JAZZY_BUILD/install" \
  --packages-up-to lunar_pure_exploration_sim lunar_pure_exploration_ros \
    lunar_pure_planner_ros lunar_pure_wheeled_controller \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

完整场景必须使用上述优化构建；规划器以 `<1 s` 为目标、`3 s` 为单次规划硬截止。探索器先用
`1 s` 等待 Action Goal 接受，再从接受时刻起用 `3.5 s` 等待结果；因此 `2–3 s` 的
`PLAN_FOUND_LATE` 仍是有效路径，不能被客户端过早取消。超过这两个 transport watchdog 的请求才
按 `CLIENT_RESULT_TIMEOUT` 重试，不能作为“无可达前沿”证据。

### 完整运行与 RViz

入口默认启动 RViz，并在启动前检查安装态包/可执行文件、选择并锁定一个候选
`ROS_DOMAIN_ID`，以 `ROS_LOCALHOST_ONLY=1` 和 `ros2 node list --no-daemon` 拒绝非空图。该过程
用于降低本机 DDS 冲突，不是对所有外部 DDS participant 的全局证明。

```bash
./scripts/run_jazzy_300m_exploration_sim.sh
```

无图形环境时显式关闭 RViz；不要把 headless 结果写成 RViz 人工验收证据：

```bash
./scripts/run_jazzy_300m_exploration_sim.sh --no-rviz
```

默认 overlay 是
`/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_build/closed_loop/install/setup.bash`。
可显式覆盖，但必须指向当前工作树构建出的 Jazzy 安装态：

```bash
LUNAR_JAZZY_OVERLAY=/absolute/path/to/install/setup.bash \
  ./scripts/run_jazzy_300m_exploration_sim.sh
```

RViz fixed frame 为 `map`。运行中应看到逐步显露的全局地图、10 m/90° FOV、当前局部窗口、
前沿与候选、当前目标、规划路径、车辆实际轨迹、轮组姿态以及覆盖率/规划计时 HUD。终止后保留
最后一帧。车辆原语只有前进/后退、前进/后退圆弧和原地旋转，不提供横向平移。

仿真的 plant 使用 `sim_dt = wall_dt × 20`；控制器命令仍是正常物理单位，算法耗时仍由本机
单调 wall clock 统计。该 launch 不发布 `/clock`，各算法也不启用 `use_sim_time`。20 倍表示车辆
状态相对 wall time 加速推进，不表示规划算法变快 20 倍。

### 完成条件与结果

唯一成功条件是任务区内不存在可达前沿：终态必须为 `COMPLETED`，reason 必须为
`COMPLETED_NO_REACHABLE_FRONTIER`。覆盖率只计算和记录，绝不作为停止阈值；人工中断、wall
guard、ERROR 或 launch 提前退出均返回失败。完整运行的实测耗时应从本次结果读取；在形成完整
终态实测前不承诺预计完成时间。

每次运行创建独立目录：

```text
/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_runs/run-<时间>-seed-<种子>-<随机后缀>/
├── launch.log
├── operator_evidence.json
├── ros-logs/
└── results/
    ├── summary.json
    ├── coverage.csv
    └── trajectory.csv
```

`summary.json` 包含地图/传感器参数、终态与 reason、覆盖率和面积、完成目标数、重规划数、路程、
wall/sim elapsed、全局/局部规划调用次数及最近/累计耗时。入口仅在三个文件非空、CSV 最后一条
coverage 与 summary/status 完全一致、路程和完成目标数为正、全局/局部规划均至少调用一次时返回
成功。`operator_evidence.json` 记录候选 domain、RViz 请求、wall guard、精确进程身份和清理后的
空 ROS 图。

这是 Ubuntu 本机 ROS 2 Jazzy 的功能仿真证据，不是 ROS 2 Humble、Jetson AGX Orin、实车控制、
实时性、功耗或稳定性验收。部署前仍需在指定 Humble/Orin 环境重新构建并执行对应验收。

## Jazzy 1 km 月表滚动规划 RViz 演示（测试专用）

该演示生成固定种子的 `1 km × 1 km` 全局月表，分辨率为 `1 m/cell`（`1000 × 1000` 栅格）；规划器同时接收以车辆为中心的 `64 m × 64 m` 局部 GridMap，分辨率为 `0.2 m/cell`（`320 × 320` 栅格）。这保留全图路线语境，同时以适合轮式安全检查的局部精度滚动规划。所有接口位于 `/lunar_demo/*`，Action 为 `/lunar_demo/plan_motion`，不会连接
生产 `/Car/T4/plan_motion`。

外部里程计可提供地图范围内、有限且物理可行的任意 `x/y/yaw` 起点，不要求与地图原点、
单元边界或单元中心对齐。每次局部规划以该真实起点建立一个在本次调用期间固定的内部
SE(2) 搜索坐标系；地图原点仍只用于地形采样，不会被重写。输出轨迹的首点保持真实起点，
不会插入吸附或几何直连段；末端仍只允许通过经过完整安全校验的缩放运动原语到达目标。

```bash
cd /home/kai/CodexDownloads/lunar_navigation/lunar_pure_planner_orin
source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to lunar_pure_planner_ros \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF
source install/setup.bash
ros2 launch lunar_pure_planner_ros lunar_surface_rviz_demo.launch.py
```

Demo 的 timer、里程计和车辆显示保持 `2 Hz`；局部 GridMap 及其 OccupancyGrid 显示首次发布后，
仅在车辆相对上一局部图中心移动至少 `4 m` 时成对更新。规划失败或恢复重试会发布带合法 frame 的
空路径，Demo 收到空轮式/足式路径后立即停止沿旧路径移动。

RViz 固定坐标系为 `map`，初始顶视范围约为整张 1 km 地图。为规避部分 Mesa/RViz 组合中
`Map` 与 GridMap 插件同时启用的 GLSL sampler 冲突，默认关闭原始占据图；**Wheel traversability**
仍以同一张地图的障碍与安全判定显示可规划区域。等待启动后约 6 秒，再使用 **2D Goal Pose** 在空白可达区域选择目标，成功信号是
`/lunar_demo/path` 更新；障碍物或不可达位置没有有效路径是预期的安全行为。此演示不提供
控制器，绝不可用于驱动车辆。

RViz 默认启用 **Wheel traversability** 图层。该图层由
`lunar_local_traversability_node` 根据 `/lunar_demo/grid_map` 的障碍和高程实时计算，并发布到
`/lunar_demo/traversability`：绿色表示轮式车包络可通行，红色表示障碍、坡度或净空不满足，
无色区域表示未知。它与全局障碍、高程、车辆和规划路径同时显示。
它只使用下列固定 ROS 接口：

| 方向 | 接口 |
| --- | --- |
| 订阅 | `/Car/T3/mapping/global_overview`、`/Car/T3/mapping/grid_map`、`/Car/T3/localization/odometry`、`/tf` |
| Action server | `/Car/T4/plan_motion` |
| 诊断发布 | `/Car/T4/planning/diagnostics` |
| 可执行轮式参考发布 | `/Car/T4/planning/wheeled_reference`（`lunar_planning_msgs/msg/MotionReference`） |
| 路径发布 | `/Car/T4/planning/wheeled_path`（`nav_msgs/msg/Path`） |
| 带计时路径发布 | `/Car/T4/planning/wheeled_path_timing`（`lunar_planning_msgs/msg/TimedPath`） |

## 局部可通行性地图

`local_traversability.launch.py` 将 `/Car/T3/mapping/grid_map` 的 `occupancy` 与
`elevation` 转为 `/Car/T4/planning/local_traversability`。输出使用 `odom` 坐标系，
`traversability` 同时是唯一的 basic layer：`1.0` 为可通行、`0.0` 为不可通行、`NaN`
为未知。轮式车对障碍做圆形膨胀，半径为车体角点外接半径加 `minimum_clearance_m`，当前
`wheel.yaml` 配置约为 `0.919 m`；这张图不替代规划器的姿态、支撑平面和底盘净空校验。

输入 QoS 默认是 `reliable + transient_local`，以支持节点晚于缓存型地图发布者启动。若
T3 发布者实际提供 `best_effort + volatile`，启动时显式覆盖为：

```bash
ros2 launch lunar_pure_planner_ros local_traversability.launch.py \
  input_qos_reliability:=best_effort input_qos_durability:=volatile
```

`scripts/start_all.sh` 会先等待 `/local_traversability` 创建，再继续启动其余本包节点；外部
地图发布者或 rosbag 仍应在该节点就绪后再启动或从头回放。

## 构建

以下命令必须在 Ubuntu 22.04 的 ROS 2 Humble 环境执行，且把所有构建物放到仓库外。

```bash
cd /path/to/lunar-navigation
source /opt/ros/humble/setup.bash
export PURE_ARTIFACT_ROOT=/home/kai/CodexDownloads/lunar_navigation/pure_planner
mkdir -p "$PURE_ARTIFACT_ROOT"
colcon --log-base "$PURE_ARTIFACT_ROOT/log" build \
  --base-paths pure_planner/ros2_ws/src ros2_ws/src/lunar_planning_msgs \
  --build-base "$PURE_ARTIFACT_ROOT/build" \
  --install-base "$PURE_ARTIFACT_ROOT/install" \
  --packages-select lunar_planning_msgs lunar_pure_planner_core lunar_pure_planner_ros
source "$PURE_ARTIFACT_ROOT/install/setup.bash"
```

容器构建同样使用 Humble；设置可用的 Humble 镜像后可直接复制执行：

```bash
cd /path/to/lunar-navigation
export PURE_ARTIFACT_ROOT=/home/kai/CodexDownloads/lunar_navigation/pure_planner
export HUMBLE_IMAGE=your-ros2-humble-image
mkdir -p "$PURE_ARTIFACT_ROOT"
docker run --rm -it \
  -v "$PWD:/workspace/lunar-navigation:ro" \
  -v "$PURE_ARTIFACT_ROOT:/artifacts" \
  -w /workspace/lunar-navigation "$HUMBLE_IMAGE" bash -lc '
    source /opt/ros/humble/setup.bash
    colcon --log-base /artifacts/log build \
      --base-paths pure_planner/ros2_ws/src ros2_ws/src/lunar_planning_msgs \
      --build-base /artifacts/build --install-base /artifacts/install \
      --packages-select lunar_planning_msgs lunar_pure_planner_core lunar_pure_planner_ros'
```

## 启动

每个终端都先 source Humble 和上述 install overlay。三个平台一次只能启动其中一个实例：

```bash
ros2 launch lunar_pure_planner_ros pure_planner.launch.py platform_type:=wheel
ros2 launch lunar_pure_planner_ros pure_planner.launch.py platform_type:=legged
ros2 launch lunar_pure_planner_ros pure_planner.launch.py platform_type:=hopper
```

轮式月表的全局-局部滚动调度默认关闭。确认全局图、持续局部图与独立轮式控制器均已运行后，才显式启用：

```bash
ros2 launch lunar_pure_planner_ros pure_planner.launch.py \
  platform_type:=wheel rolling_surface_enabled:=true
```

启用后，一条月表 Action 先计算一次按轮式包络膨胀的全局路线；随后以 8 m 路线前瞻重复产生严格局部路径，直到 odometry 进入最终目标容差。每个真正触发的冷启动或滚动规划周期共用一个时钟：`<1 s` 达到目标，`[1 s,2 s)` 记录变慢，`[2 s,3 s)` 记录 SLA 失败但继续搜索，只有 `>=3 s` 才硬停止且不发布新路径；车辆行驶和轮询时间不计入规划周期，也没有 300 s Action 业务截止时间。取消或替换优先于迟到结果，失败会同时发布空 `MotionReference`、空 `Path` 和空 `TimedPath`。控制器只订阅保留正反向速度及角速度的 `MotionReference`；两个 Path Topic 仅用于 RViz、rosbag 和外部观测。该模式不替代外部全局图生产者或独立轮式控制器。

`rolling_transient_retry_limit` 默认是 `2`，合法范围为 `0..10`。它只适用于已经发布过至少一个
有效滚动段之后的局部 `NO_PATH` 或单周期 `TIMEOUT`：每次恢复先发布空输出，并在诊断中增加
`rolling_recovery=true`、`rolling_recovery_attempt=N`；任一后续成功会把连续计数清零。首次局部失败、
全局失败、`INVALID_INPUT`、`PLANNER_ERROR`、取消和超过上限仍立即终止。`STALE_INPUT` 继续表示
旧输入身份的结果未发布并重新计算，不占用恢复次数。该机制没有放宽地图身份校验，也没有延长
单周期 `3 s` 硬截止。

默认参数在 `config/pure_planner.yaml`；其中全局占据阈值为 `50` percent，局部占据阈值为 `0.5`。

### 一次性完全信任桥接（轮式、熔岩洞）

当机器人起点周围为未知、但局部可通行性图中已存在四邻域连续的可通行区域时，可在发送目标前执行：

```bash
ros2 param set /pure_planner trusted_bridge_once true
```

下一条被接受的 `environment_mode: 2` 轮式 Action 会无条件拼接一条“当前位姿到最近四邻域可通行区域”的直线，再从该落点运行原有严格规划。该参数会立刻自动复位为 `false`；桥接线段不校验地图、高程、障碍或坡度，后半段仍按车辆包络与地图约束规划。

## 发送 Action Goal

目标是平面 `x/y`，可选 yaw；不要求 goal `z`。若外部消息中 `z` 为 `NaN` 也会忽略，输出轨迹的 `z` 从局部 elevation 派生。下面的示例刻意不提供 `z`。

月表模式 (`environment_mode: 1`) 同时需要全局总览和局部图：

```bash
ros2 action send_goal /Car/T4/plan_motion lunar_planning_msgs/action/PlanMotion "{
  environment_mode: 1,
  request_id: surface-001,
  mission_id: demo,
  mission_revision: 0,
  goal: {
    header: {frame_id: map},
    goal_id: surface-point,
    goal_type: 1,
    point: {x: 4.0, y: -2.0},
    position_tolerance_m: 0.25,
    has_yaw_constraint: true,
    yaw_rad: 0.0,
    yaw_tolerance_rad: 0.2
  },
  replace_active_request: false
}"
```

熔岩洞模式 (`environment_mode: 2`) 仅使用当前高精度局部图：

```bash
ros2 action send_goal /Car/T4/plan_motion lunar_planning_msgs/action/PlanMotion "{
  environment_mode: 2,
  request_id: tube-001,
  mission_id: demo,
  mission_revision: 0,
  goal: {
    header: {frame_id: odom},
    goal_id: tube-point,
    goal_type: 1,
    point: {x: 2.0, y: 1.0},
    position_tolerance_m: 0.20,
    has_yaw_constraint: false,
    yaw_rad: 0.0,
    yaw_tolerance_rad: 0.0
  },
  replace_active_request: false
}"
```

读取单次诊断：

```bash
ros2 topic echo --once /Car/T4/planning/diagnostics
```

取消正在执行的请求使用 Humble 的前台 Action CLI 流程：保持上述
`ros2 action send_goal ...` 在前台运行，看到 `Goal accepted with ID:` 后，在**同一个进程所在终端**按
`Ctrl+C`。`send_goal` 收到 `SIGINT` 后会对这个已接受 goal 发出 `CancelGoal` 请求；不要另开终端构造
一个独立的 cancel 子命令。服务端接受取消后，终态的失败原因是 `REQUEST_CANCELED`。

替换请求时重新发送 goal，并把 `replace_active_request` 设为 `true`；新请求会取代当前请求，而不是等待旧请求自然结束。

## 结果与执行语义

Action Result 和 diagnostics Topic 的生产成功原因码为 `PLAN_FOUND`。失败时 `reason_code` 才限定为以下六类：

- `INVALID_INPUT`
- `GOAL_OUTSIDE_LOCAL_MAP`
- `NO_PATH`
- `TIMEOUT`
- `REQUEST_CANCELED`
- `PLANNER_ERROR`

正式规划成功还必须同时满足 `planning_outcome: 0` 与 `has_reference: true`；滚动周期在 SLA 后但硬
截止前完成时原因码为 `PLAN_FOUND_LATE`。Action 状态或可视化路径单独出现都不足以证明成功。

计时和调用次数的实际输出位置与单位如下；所有耗时均由 `steady_clock` wall duration 产生：

| 输出 | 字段 | 单位与含义 |
| --- | --- | --- |
| Action Result | `diagnostics.elapsed_s` | seconds；从 worker 读取请求快照到终态输出收尾的总耗时 |
| diagnostics Topic | `global_elapsed_ms`、`local_elapsed_ms`、`total_elapsed_ms` | milliseconds；分别为全局、局部和请求总耗时 |
| diagnostics Topic | `global_call_count`、`local_call_count` | 次数；进入对应搜索函数的实际调用数 |
| TimedPath Topic | `planning_time` | seconds；`builtin_interfaces/Duration` 以 sec + nanosec 编码，表示同消息 `path` 的本次请求总规划耗时 |

diagnostics Topic 使用 `DiagnosticStatus.values`，所以上述毫秒值和计数在线上表现为字符串。
Action Result 不提供分阶段耗时或调用次数；不能把其中的 `elapsed_s` 当成毫秒，也不能从它推导
`global_call_count` 或 `local_call_count`。

下列兼容字段不是准入条件：输入消息的 stamp 不用于排序、新鲜度或时间差判断，只在 Result 中把本次
快照的 `global_map_stamp`、`local_map_stamp` 和 `state_stamp` 原样回填；Goal 的
`mission_revision` 原样回传，`mission_id` 不被规划算法读取；Odometry 的 covariance 字段不读取、
不拒绝请求，也不回传。运行时地图 freshness/revision/version、定位 status、任务、execution feedback
没有对应 pure planner 订阅或参数，完全不在请求准入面。这里的 map version 不包括启动时加载并核对的
静态平台 `capability_version`；后者是固定平台配置合同，不是运行时地图质量门禁。

月表成功会给出完整全局粗路线和当前已验证的局部执行段。月表执行方必须以新的 Action 请求反复推进局部执行段，不能把一次返回当作整段路径已执行完毕。熔岩洞模式返回由已验证运动边组成的完整点到点路径。

## Task 16 验证摘要

当前 Ubuntu 22.04 amd64 + ROS 2 Humble 的构建、测试、性能分布、矩阵证据和 readiness 分级见
[`VERIFICATION.md`](VERIFICATION.md)。生产源码/前置测试基线为
`e93a4685ed4cebbe06bc2983b6ea3e03159554c5`；core+ROS、pure tests 和完整 foundation 均通过。
真实 bag、月表全局输入、生产 ROS domain 原子切换与 Jetson AGX Orin 仍受下表边界约束。

## 当前证据边界

| key | status | 边界 |
| --- | --- | --- |
| `bag_metadata` | `SEARCHED_ZERO_CANDIDATES` | 已搜索 `/home/kai`（maxdepth 8）下的 `metadata.yaml`，候选数为 0。 |
| `real_bag_replay` | `NOT_RUN_REAL_BAG_NOT_LOCATED` | metadata 搜索已完成且候选数为 0；没有候选可执行 `ros2 bag info`，也未回放任何真实 bag。 |
| `lava_tube_replay` | `NOT_RUN_REAL_BAG_NOT_LOCATED` | 未定位真实 bag，未执行 `ros2 bag info` 或回放，不能声明 `LAVA_TUBE` 已验证。 |
| `lunar_surface_e2e` | `NOT_RUN_MISSING_GLOBAL_INPUT` | 缺少真实 `/Car/T3/mapping/global_overview` producer 或包含该 Topic 的 bag，不能证明 `LUNAR_SURFACE` 全局加局部端到端成功。 |
| `global_producer` | `NOT_RUN` | 未完成真实 producer 集成。 |
| `jetson_agx` | `NOT_RUN` | 未执行 Jetson AGX Orin 原生构建、性能、功耗或稳定性验收。 |
| `historical_missing_global_bag` | `HISTORICAL_SUBSET_ONLY` | 历史记录中的 bag 缺少 `/Car/T3/mapping/global_overview`，仅可作为 `LAVA_TUBE` 局部输入子集；本轮未找到其 metadata，未实际回放。 |
| `synthetic_global` | `FORBIDDEN` | 禁止生成或使用假全局图；不得用合成、零填充或局部图替代真实 `/Car/T3/mapping/global_overview`；月表验收仍需实机在线全局总览，或重新录制并核验包含该 Topic 的 rosbag。 |
