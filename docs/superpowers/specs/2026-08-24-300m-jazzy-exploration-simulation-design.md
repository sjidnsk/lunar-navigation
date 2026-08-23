# 300 m Jazzy 探索闭环仿真设计

日期：2026-08-24

状态：已批准（2026-08-24，控制器采用方案 1）

## 1. 目标

在本机 ROS 2 Jazzy 环境中，为现有纯前沿探索器、纯规划器和纯轮式控制器建立一套可重复的
完整闭环仿真。场景是 300 m × 300 m 的月表任务区，全局地图分辨率 1.0 m，局部地图分辨率
0.2 m，传感器观测半径 10 m、水平视场角 90°。车辆从局部已知区域出发，地图随运动逐步显露，
探索器持续选择可达前沿并调用现有规划算法，直到任务区域内不存在可达前沿。

覆盖率只统计，不作为终止阈值。仿真以 20 倍执行速度推进，并在 RViz2 中展示地图显露、前沿、
候选、当前目标、规划路径、车辆运动、覆盖率和最终状态。

## 2. 边界

- 不改变 `lunar_pure_exploration_core` 的前沿、信息增益、排序或终止算法。
- 不改变 `lunar_pure_planner_core` 的全局或局部规划算法。
- 保留 `lunar_pure_wheeled_controller` 包和 Pure Pursuit 主体，只补齐其已声明但尚不能执行的轮式
  运动模式：后退、后退圆弧、原地旋转和取消停车。
- 仿真只允许纵向速度和偏航角速度，不生成横向平移命令。
- 测试车辆是前后轮协调转向的运动学模型，不按差速车辆建模。
- 使用课题三输入名 `/Car/T3/...`、课题四输出名 `/Car/T4/...` 和控制命令
  `/Car/T5/Car_Cmd_Vel`，但整个 launch 是本机测试专用，必须使用隔离的 `ROS_DOMAIN_ID`，不得与
  实车节点或另一控制命令发布者同时运行。
- Jazzy 结果只证明本机算法闭环和可视化，不替代 Ubuntu 22.04 + ROS 2 Humble 或 Jetson AGX
  Orin 的部署验收。

## 3. 总体数据流

```text
确定性月表真值
  -> 90°/10 m 遮挡观测
  -> /Car/T3/mapping/global_overview (1.0 m，逐步显露)
  -> /Car/T3/mapping/grid_map       (0.2 m，rolling local map)
  -> 纯探索器 -> /Car/T4/plan_motion -> 纯规划器
  -> /Car/T4/execution/motion_reference
  -> 纯轮式控制器 -> /Car/T5/Car_Cmd_Vel
  -> 前后轮协调转向测试车辆
  -> /Car/T3/localization/odometry + /tf
  -> 新一轮观测、探索和规划
```

新增测试组件放在独立包 `lunar_pure_exploration_sim` 中。它拥有真值场景、观测模型、车辆 plant、
任务自动启动、运行统计和 RViz 辅助显示；不得让探索器或规划器直接读取真值地图。

## 4. 场景和地图

### 4.1 真值场景

- 固定物理范围：`[-150, 150) m × [-150, 150) m`。
- 默认种子：`20260824`；相同种子必须生成相同地图。
- 障碍由确定性伪随机岩石、成簇岩块和环形陨石坑边缘组成；高度由缓坡、波纹、坑底和岩石叠加。
- 初始车辆周围保留满足平台足迹和净空的自由区。
- 生成后检查自由区连通性，并保留从起点向主要区域延伸的连通自由骨架，避免随机场景在起点处
  立即退化为无可达前沿。障碍仍可形成局部不可达区域。
- 任务多边形默认是地图边缘内缩 5 m 的矩形，即四角
  `(-145,-145)、(145,-145)、(145,145)、(-145,145)`，按 `map` 表达。

### 4.2 全局地图

发布 `/Car/T3/mapping/global_overview`，类型 `nav_msgs/msg/OccupancyGrid`：

- `300 × 300` cells，`resolution=1.0` m，origin `(-150,-150)`；
- 未观测为 `-1`，自由为 `0`，障碍为 `100`；
- 直接使用 `OccupancyGrid` 的原生 `-1/0..100` 语义，不转换成局部地图编码；
- 只有传感器实际看到的 cell 才从 `-1` 变为真值，已观测内容不回退为未知。

### 4.3 局部地图

发布 `/Car/T3/mapping/grid_map`，类型 `grid_map_msgs/msg/GridMap`：

- `resolution=0.2` m，默认 rolling window 为 `64 × 64` m；
- frame 为 `odom`，窗口中心随车辆移动，姿态保持单位四元数；
- layers 固定为 `occupancy、semantic_id、elevation、roughness`；
- `occupancy` 使用课题三定义：`0.0` 自由、`1.0` 占据、`NaN` 未知，中间值保留为占据概率；
- 其余层只在已观测 cell 中给出有限值，未观测 cell 为 `NaN`；
- 局部障碍高度和 roughness 从确定性高程邻域派生，供现有局部规划器使用；探索器本身仍不订阅
  该局部图。

### 4.4 观测模型

- 以车辆当前 yaw 为视轴，在 90° 扇区内投射离散射线，最大距离 10 m；
- 障碍 cell 被看见后终止该射线，障碍后的 cell 保持未知；
- 初始位姿立即产生第一帧观测，以形成首批前沿；
- 真值、可见性和已知掩码彼此分离，规划/探索 Topic 中不得泄漏扇区外真值；
- 单元测试覆盖视场边界、量程边界、障碍遮挡和全局/局部分辨率对齐。

## 5. 控制器方案 1：最小能力补齐

### 5.1 保留内容

- 包名、节点名及唯一输出 `/Car/T5/Car_Cmd_Vel` 不变；
- 平移段继续使用现有 Pure Pursuit 的前视点和横向误差反馈；
- 继续执行速度、角速度、路径偏差和目标容差限制；
- `Twist.linear.y` 始终为零。

### 5.2 执行权威

`MotionReference.trajectory` 是新闭环的执行权威。每个
`MultiDOFJointTrajectoryPoint` 的首个 transform 提供 `map` 位姿，首个 velocity 提供 `map`
速度和偏航角速度。控制器把平面线速度投影到该点的车体前向轴：

```text
signed_speed = vx_map * cos(yaw) + vy_map * sin(yaw)
```

其符号决定前进或后退。只做执行必需的结构和有限值检查，不读取或校验消息时间戳、地图版本、
协方差或新鲜度。若 trajectory 为空，则保留当前 `path_preview` 前进跟踪作为兼容模式；兼容模式
不推断倒车或原地旋转。

### 5.3 平移和旋转模式

- 前进、后退及相应圆弧按轨迹顺序维护执行 cursor；不能每周期仅按全路径最近欧氏点重新猜测
  运动方向。
- 平移段使用带符号 `linear.x`。Pure Pursuit 曲率仍由局部目标点计算，`angular.z` 随带符号线速度
  产生，因此后退圆弧方向与车辆运动学一致。
- 连续轨迹点 XY 位移在位置容差内而 yaw 尚未达到容差时进入原地旋转模式：`linear.x=0`，
  `angular.z` 由规范化 yaw 误差的有界比例控制得到。
- 轨迹给出的非零偏航速度用于确认旋转方向和限幅；轨迹末点的零速度不应丢失前一段的运动模式。
- 达到末点位置和 yaw 后发布一次零速度并清除 active reference。

### 5.4 取消

控制器订阅 `/Car/T4/execution/cancel` (`std_msgs/msg/String`)。内容与当前 `plan_id` 相同时立即清除
参考并发布零速度；不匹配的旧 plan ID 不影响当前参考。节点销毁、无有效 odometry、无有效参考或
路径偏差失败时继续发布零速度。

## 6. 前后轮协调转向测试车辆

plant 订阅 `/Car/T5/Car_Cmd_Vel`，只接受 `linear.x=v` 和 `angular.z=omega`：

- `|v|>epsilon` 时按平面刚体运动学积分；曲率 `kappa=omega/v`；
- RViz 轮角采用对称前后轮协调转向表示：
  `delta_front=atan(wheelbase*kappa/2)`，`delta_rear=-delta_front`；
- `|v|<=epsilon` 且 `|omega|>epsilon` 时按车辆中心原地旋转，并把四轮显示为绕中心圆周的切向；
- 不实现横向平移、轮胎侧偏、悬架、驱动扭矩或地形动力学；
- 发布 `/Car/T3/localization/odometry` (`odom -> base_link`) 和 `/tf` 中的恒等
  `map -> odom` 及动态 `odom -> base_link`。
- 发布 `/Car/T4/simulation/sim_elapsed` (`std_msgs/msg/Float64`)，供测试记录器明确区分仿真时间与
  本机 wall time；该 Topic 不进入探索或规划算法。

20 倍加速由 plant 使用 `sim_dt = wall_dt * 20` 实现。控制器仍发布正常物理单位的速度，不把
速度上限乘以 20；算法调用耗时继续用本机单调时钟统计。结果同时记录 wall elapsed 和 simulated
elapsed，避免把算法耗时误报为仿真时间。

## 7. 探索闭环与终止

- 启动器在地图、odometry、TF、规划 Action 和控制器均就绪后发布一次
  `/Car/T4/exploration/task` START。
- 探索器只把最终选中的 `MotionReference` 发布到
  `/Car/T4/execution/motion_reference`；控制器 launch 将其 `reference_topic` 指向该 Topic。
- 地图随车辆运动更新后，探索器按现有规则重建前沿、候选和路径。
- 正常终止条件保持为任务区内无可达前沿，对应 `PureExplorationStatus.COMPLETED`；不得增加覆盖率
  阈值或固定运行时长作为成功条件。
- `ERROR`、进程退出、超时或人工停止都不能记作探索完成。

## 8. RViz2 和运行结果

RViz fixed frame 为 `map`，至少显示：

- 逐步显露的全局 OccupancyGrid；
- 当前 90°/10 m 观测扇区和 rolling local-map 边界；
- 前沿簇、候选、选中候选和当前目标；
- 当前 `MotionReference` 路径、车辆实际轨迹和平台足迹；
- 前后轮转角/原地旋转姿态；
- 文本 HUD：探索状态、reason、覆盖率、已知/未知面积、前沿/候选数、完成目标数、规划次数、
  全局与局部规划累计耗时、wall/sim elapsed。

终止后 HUD 和最后一帧 Marker 保持可见。运行记录写到仓库外：

```text
~/CodexDownloads/lunar_navigation/exploration_jazzy_runs/<run_id>/summary.json
~/CodexDownloads/lunar_navigation/exploration_jazzy_runs/<run_id>/coverage.csv
~/CodexDownloads/lunar_navigation/exploration_jazzy_runs/<run_id>/trajectory.csv
```

`summary.json` 至少包含 seed、地图和传感器参数、终态及 reason、最终覆盖率、各面积、完成目标数、
重规划数、行驶距离、wall/sim elapsed，以及全局/局部规划调用次数、最近值和累计耗时。

## 9. 验收

### 9.1 自动验证

1. 场景测试证明物理范围、双分辨率、种子确定性、安全起点和非零障碍密度。
2. 观测测试证明 90°、10 m、遮挡和未知保持语义。
3. 控制器测试分别证明前进、前进圆弧、后退、后退圆弧、原地左右旋转、取消停车、零横向速度和
   旧 path-only 兼容模式。
4. plant 测试证明前/后退积分、圆弧方向、原地旋转、无横移和 20 倍时间缩放。
5. launch smoke test 证明输入和 Action 就绪后自动 START，状态至少经历
   `SELECTING_FRONTIER/PLANNING/EXECUTING`，车辆位置变化、覆盖率上升，并收到全局和局部规划
   计时诊断。
6. 完整 300 m 运行必须以 `COMPLETED` 和“无可达前沿”原因结束，生成三个非空结果文件；最终
   coverage 必须与状态 Topic 一致，但不要求达到固定百分比。

### 9.2 人工可视验证

在 RViz2 中确认未知区域只随扇区观测显露、障碍产生遮挡、目标和路径随探索更新、车辆没有横向
平移、倒车与原地旋转方向正确，以及终止后最终状态和覆盖率仍可查看。

## 10. 非目标

- 不模拟月尘、轮土作用、打滑、侧倾、碰撞冲击或执行器动态。
- 不以本机 Jazzy 的吞吐量推断 Orin 实时性能。
- 不把测试真值或自动任务发布器纳入实车 launch。
- 不为完成仿真而降低足迹、障碍、坡度或可达性判定。
- 不修改“无可达前沿完成、覆盖率仅统计”的探索合同。
