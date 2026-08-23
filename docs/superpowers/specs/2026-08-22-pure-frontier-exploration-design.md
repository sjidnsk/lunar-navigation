# 课题三输入驱动的纯前沿探索算法设计

日期：2026-08-22

状态：已批准（2026-08-22）

配套设计：[`2026-08-22-dual-mode-anytime-pure-planner-design.md`](2026-08-22-dual-mode-anytime-pure-planner-design.md)

## 1. 目标

在本仓库内新增一套与旧正式训练、PPO 和旧规划运行时物理隔离的纯前沿探索算法。给定课题三发布的全局占据地图、位姿与 TF，以及课题四发布的任务区域多边形，算法应持续选择可观测未知区域的已知自由位姿，调用纯路径规划器，并把最终选中的运动参考交给控制器，直到任务区域内不存在可达前沿。

首版面向当前 `Car` 轮式平台验收。核心数据结构使用通用二维足迹表达，但本文不声明足式或飞跃式平台探索能力。

本设计明确不使用：

- PPO、ONNX、TensorRT、训练 checkpoint 或模型合同；
- 地图 revision、地图版本、地图新鲜度或时间戳准入；
- Odometry 协方差、LocalizationStatus 或观测证据层；
- 固定覆盖率终止阈值；
- 全局 TSP、Orienteering 或学习式候选排序；
- 旧 `candidate_builder.py` 的正式训练身份、哈希、cache、恢复和发布门。

覆盖率只用于统计。唯一正常完成条件是：对当前任务区域完整生成并验证全部前沿候选后，没有任何候选能获得有效规划路径。

## 2. 算法来源与取舍

首版采用“WFD 前沿检测 + 候选观测位姿 + 信息增益/路径代价评分”的确定性规则算法：

- 使用 Wavefront Frontier Detection 的自由空间搜索思想提取前沿；
- 借鉴 FUEL 的前沿结构、全局粗筛和局部候选精算思想；
- 借鉴 TARE 的远处粗粒度决策、近处详细规划分工；
- 保留当前仓库规则基线中“信息增益与路径代价共同决定目标”的思想；
- 不复制无人机三维轨迹、全局巡回优化或策略网络。

参考资料：

- Keidar and Kaminka, *Fast Frontier Detection for Robot Exploration: Theory and Experiments*: <https://u.cs.biu.ac.il/~kaminkg/publications/b2hd-aamas12matan.html>
- Zhou et al., *FUEL: Fast UAV Exploration using Incremental Frontier Structure and Hierarchical Planning*: <https://arxiv.org/abs/2010.11561>
- Cao et al., *TARE: A Hierarchical Framework for Efficiently Exploring Complex 3D Environments*: <https://publications.ri.cmu.edu/tare-a-hierarchical-framework-for-efficiently-exploring-complex-3d-environments>

## 3. 物理隔离与包边界

新增独立包，不链接旧训练环境、旧探索策略或旧规划节点：

```text
pure_planner/ros2_ws/src/lunar_pure_exploration_msgs/
pure_planner/ros2_ws/src/lunar_pure_exploration_core/
pure_planner/ros2_ws/src/lunar_pure_exploration_ros/
```

- `lunar_pure_exploration_msgs`：精简任务和状态消息；
- `lunar_pure_exploration_core`：ROS 无关的任务栅格、前沿、候选、增益、评分和统计算法；
- `lunar_pure_exploration_ros`：订阅、TF、状态机、规划 Action 客户端、参考发布和计时；

纯算法包不得依赖：

- `training/lunar_policy_training`；
- `model_contract`；
- `lunar_policy_runtime`；
- 旧 `lunar_planner_ros`、`luna_t3_map_adapter` 或旧 deployment 配置。

纯规划器拥有 `pure_planner/config/{wheel,legged,hopper}.yaml`，并由
`lunar_pure_planner_ros` 安装到自身 package share。探索器只读取同一份已安装配置，不移动文件、
不复制平台尺寸，也不链接规划器算法库。ROS/launch 选择器使用 `platform_selector: wheel`；读取
规划器配置后映射成核心 `PlatformGeometry.platform_type: WHEELED`。两者大小写和职责不同，不得
用同一字段混写。足式和跳跃式配置存在不代表本版声明对应探索能力。

## 4. ROS 接口

所有新增课题四接口统一使用 `/Car/T4/...`。

### 4.1 输入

| 接口 | 类型 | 所有者 | 用途 |
| --- | --- | --- | --- |
| `/Car/T3/mapping/global_overview` | `nav_msgs/msg/OccupancyGrid` | 课题三 | 前沿、覆盖统计、候选增益 |
| `/Car/T3/localization/odometry` | `nav_msgs/msg/Odometry` | 课题三 | 当前位姿、路径进度和到达判断 |
| `/tf` | `tf2_msgs/msg/TFMessage` | 课题三 | 最新直接 `map -> odom`；`odom -> base` 只取 Odometry |
| `/Car/T4/exploration/task` | `lunar_pure_exploration_msgs/msg/PureExplorationTask` | 任务发布者 | 多边形任务和控制命令 |
| `/Car/T4/plan_motion` | `lunar_planning_msgs/action/PlanMotion` | 纯规划器 | 候选可达性、路径和运动参考 |
| `/Car/T4/planning/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | 纯规划器 | 按 `request_id` 关联全局/局部计时 |

探索节点不订阅 `/Car/T3/mapping/grid_map`。局部 `occupancy/elevation` 由纯规划器直接处理，探索算法不复制局部坡度、高差、粗糙度或障碍高度计算。

### 4.2 输出

| 接口 | 类型 | 用途 |
| --- | --- | --- |
| `/Car/T4/exploration/status` | `lunar_pure_exploration_msgs/msg/PureExplorationStatus` | 任务状态、覆盖和计时统计 |
| `/Car/T4/exploration/current_goal` | `geometry_msgs/msg/PoseStamped` | 当前承诺目标和 RViz 显示 |
| `/Car/T4/exploration/frontiers` | `visualization_msgs/msg/MarkerArray` | 前沿簇、候选和选择结果 |
| `/Car/T4/exploration/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | 前沿、增益、评分和重规划诊断 |
| `/Car/T4/execution/motion_reference` | `lunar_planning_msgs/msg/MotionReference` | 只发布最终选中的运动参考 |
| `/Car/T4/execution/cancel` | `std_msgs/msg/String` | 取消当前 `plan_id` |

候选评估产生但未选中的 `MotionReference` 不得发布给控制器。

## 5. 精简任务消息

`PureExplorationTask.msg`：

```text
uint8 START=1
uint8 PAUSE=2
uint8 RESUME=3
uint8 CANCEL=4

std_msgs/Header header
string task_id
uint8 command
geometry_msgs/Polygon boundary
```

语义：

- `header.frame_id` 固定为 `map`；
- `header.stamp` 不参与任何判断；
- `START` 立即取消旧参考、替换当前任务并清空旧任务统计；
- `PAUSE` 取消当前参考，保留任务、覆盖和失败记录；
- `RESUME` 使用当前地图和位姿重新构建候选；
- `CANCEL` 取消参考、清除任务并回到 `IDLE`；
- `START` 的 `boundary` 是按边界顺序排列的简单多边形，允许凸多边形和凹多边形；
- `PAUSE/RESUME/CANCEL` 不读取 `boundary`。

只做算法必需的结构检查：`task_id` 非空、命令合法、`START` 至少三个有限且不同的二维角点、边界不自交且面积大于零。不检查时间戳、revision 或消息新鲜度。

## 6. 地图与任务区域模型

### 6.1 全局图原生编码

探索算法直接使用 `/Car/T3/mapping/global_overview` 的 `OccupancyGrid`：

```text
-1      -> UNKNOWN
0..49   -> FREE
50..100 -> OCCUPIED
其他值   -> UNKNOWN
```

占据阈值参数 `global_occupied_threshold` 默认 `50`。全局图不得转换成局部 `GridMap` 的 `NaN/0.0..1.0` 表示。

局部 `/Car/T3/mapping/grid_map` 继续由纯规划器按其原生编码处理：`NaN` 为未知、有限 `0.0..1.0` 为占据概率、默认 `0.5` 为障碍阈值，`elevation` 用于局部地形。探索算法不读取该消息。

### 6.2 任务逻辑栅格

任务多边形按当前全局图的 origin 和 resolution 建立逻辑栅格。栅格中心位于多边形内部或边界上时属于任务区域。

- 位于 `OccupancyGrid` 数组内的任务栅格读取原生占据值；
- 位于任务多边形内但超出当前消息数组的栅格标为 `OUTSIDE_MAP`；
- `OUTSIDE_MAP` 按未知面积统计，但不生成候选；
- 全局图扩大后，这些位置自动进入正常分类；
- 多边形精确面积使用鞋带公式统计，覆盖率使用确定性的任务逻辑栅格面积作为分母。

地图 origin、尺寸或 resolution 改变时，以新消息重新建立逻辑栅格。这是算法坐标变换，不是地图版本或新鲜度校验。

`OccupancyGridView` 是后续几何算法的唯一坐标数学实现。它提供世界点与连续逻辑 grid 坐标之间
的双向变换、世界点到有符号 logical cell 的转换，以及 logical cell 在 grid frame 中的四个闭
方块角点；唯一数学写在 `OccupancyGridView` 的 geometry-taking static overload 中，其实例方法和
`TaskRaster` 同名方法都只用各自保存的 `GridGeometry` 委托这些 overload，不复制地图 data，也不
重复 origin yaw、逆旋转、`floor` 或 cell 边界数学。`OccupancyGridView::CellCenter(index)` 与
`TaskRaster::CellCenter(index)` 都必须构造局部点 `{x+0.5,y+0.5}` 并调用同一个 static
`GridToWorld(geometry, point)` primitive；两个 CellCenter 内不得另写 `sin/cos`、origin 平移或
resolution 乘加公式。Task 5/6 只调用 TaskRaster 的共享接口。连续逻辑 grid 的整数坐标位于
cell 边界，`GridIndex{x,y}` 对应闭方块 `[x,x+1] x [y,y+1]`，中心为
`[x+0.5,y+0.5]`。非有限输入返回空值；有限但无法用接口类型表示的坐标按具体接口抛
`std::overflow_error` 或返回空值，有限的 map 外坐标保留有符号索引。

为避免车辆上由极大但数值合法的任务多边形触发 OOM 或无界循环，构建前按 polygon 的逻辑栅格
包围盒做资源上限检查。默认 `maximum_task_raster_cells=1048576`，限制的是待扫描 bbox cell 数而
不是已知覆盖率；在当前 1 m 全局图上约对应 1 km² 正方形任务，核心状态和索引约为十几 MB。
超限属于结构/资源 `ERROR`，不能解释为无可达前沿。该值可显式调大，但必须先在 Orin 验证内存和
决策时延；不能按地图时间戳或版本动态改变。

## 7. 平台与分辨率派生量

规划器拥有的轮式平台配置至少提供：

```text
platform_id
platform_type
base_frame_id
capability.footprint_xy_m
capability.minimum_clearance_m
```

当前规划基线轮式足迹为 `1.182 m x 0.818 m`，四个顶点
`[[0.591, 0.409], [0.591, -0.409], [-0.591, -0.409], [-0.591, 0.409]]` 相对
`base_footprint` 表达，最小附加净空为 `0.2 m`。探索器从足迹计算：

- `platform_length_m`：足迹 X 方向最大跨度；
- `platform_width_m`：足迹 Y 方向最大跨度；
- `footprint_circumscribed_radius_m`：足迹顶点最大欧氏半径。

所有离散参数先用米制公式计算，再由当前地图 resolution 换算：

| 算法量 | 公式 |
| --- | --- |
| 最小前沿长度 | `max(platform_width_m, 2 * resolution_m)` |
| 候选最小间距 | `max(platform_length_m, 2 * resolution_m)` |
| 最小观测退距 | `footprint_circumscribed_radius_m + minimum_clearance_m` |
| 重访抑制半径 | `max(2 * platform_length_m, 2 * resolution_m)` |
| 失败解除半径 | `max(platform_length_m, 2 * resolution_m)` |
| 规划及到达位置容差 | `max(0.5 * resolution_m, 0.25 * platform_width_m)` |
| 观测半径栅格数 | `ceil(10.0 / resolution_m)` |

信息增益以平方米表达，路径代价以米表达，不能使用裸栅格数量跨分辨率比较。

## 8. 前沿检测与聚类

每次进入 `SELECTING_FRONTIER` 时冻结一份进程内地图、位姿和任务多边形快照。该快照只保证一次计算内部一致，不检查消息时间关系。

1. 将机器人 `map` 位姿投影到全局图；
2. 若机器人所在格不是 FREE，在相邻 FREE 格中选择世界距离最近、坐标字典序最小的搜索起点；
3. 从该起点按四邻域在 FREE 栅格上进行 WFD 波前搜索，禁止通过对角接触穿越障碍角；若机器人
   格非 FREE，只检查其四邻 FREE 格，并按格中心世界距离、世界 `x/y` 字典序选择；
4. FREE 格的四邻域存在任务区域内 UNKNOWN 时，该 FREE 格为前沿格；
5. 使用八邻域把前沿格组成前沿簇；
6. 每个 FREE 前沿格到四邻 UNKNOWN 格的唯一有向接口边贡献 `resolution_m` 长度；
   `OUTSIDE_MAP` 不属于 UNKNOWN 前沿。簇长度是所有唯一接口边长度之和，小于最小前沿长度的簇
   作为噪声删除；
7. 每簇保存按 FREE 格世界坐标、方向、UNKNOWN 格世界坐标排序的 canonical 接口边序列，包含
   FREE/UNKNOWN 索引、方向和世界中点，供候选分位位置和未知侧法向使用。方向采用逻辑栅格中
   `+x=0`、`+y=1`、`-x=2`、`-y=3` 的固定编码；
8. 接口边世界中点按毫米、half-away-from-zero 量化；完整 canonical key 是逐边展开的
   `[midpoint_x_mm, midpoint_y_mm, direction]` 有符号 64 位三元组序列。显示 ID 对每边的两个
   `int64` 二补码小端字节和一个方向字节依次执行 FNV-1a-64；跨周期相等和承诺匹配必须比较完整
   key，不能只相信哈希且不读取地图 revision；
9. 没有可用 FREE 起点时返回 `NO_REACHABLE_FREE_START`，它表示当前内容没有搜索起点，不是结构
   ERROR，也不能单独证明任务完成。

WFD 的自由区连通只用于候选搜索，不替代平台足迹和规划器可达性裁决。

## 9. 平台安全候选生成

Task 4 保存的是 canonical 接口边序列，不是几何边界链。对每个前沿簇的 `N` 条等长
`interface_edges`，按 `q={0.25,0.50,0.75}` 依次选择索引 `ceil(q*N)-1`，再按所选完整接口边
去重；因此每簇最多三个代表点。代表点是所选接口边的世界中点。局部 UNKNOWN 质心是所选
`free_cell` 的全部四邻 `kUnknown` cell 中心的算术均值，不含 `kOutsideMap`。未知侧法向是
`local_unknown_centroid - free_cell_center` 的单位向量；若其范数为零，改用所选接口边的
`unknown_cell_center - free_cell_center` 并单位化。位置沿此法向的反方向向已知侧搜索。标准 yaw
始终从最终候选中心指向该局部 UNKNOWN 质心；若该向量本身为零，同样使用上述未知侧法向。
上述质心、法向、fallback、退距搜索和标准 yaw 必须先在消除世界平移的 logical-grid frame 中
计算；所选接口边的 logical midpoint 由其 free/unknown cell center 取中点，最终中心只通过共享
`GridToWorld` 转换一次，世界 yaw 由 local-grid yaw 加 map origin yaw 后规范化。该点仍是同一完整
接口边的世界中点，不改变代表点语义。禁止用两个大世界坐标相减恢复局部法向。0/30/90 度 origin
yaw 和仍能表达分辨率的大平移必须得到相同 local-grid normal、相对候选位置、相对 yaw 和
collision cell 集。

对每个代表位置：

1. 使用 `d(k)=minimum_standoff_m+k*resolution_m`，只枚举满足
   `k>=0` 且 `k*resolution_m <= 2*platform_length_m` 的整数 `k`；用整数边界计算，不用累计
   浮点加法决定末项。`maximum_search_step` 的 authority 固定为先以 double 计算
   `extra=2.0*platform_length_m` 和 `ratio=extra/resolution_m`，检查 finite/size_t 可表示性后以
   `floor(ratio)` 得到初值，再按 double `step*resolution_m <= extra` 的原始闭不等式向下/向上校正
   到最大合法整数；不提升输入后重算、不加 epsilon。ratio 恰为整数 n 时返回 n，
   `nextafter(n,0)` 返回 n-1，`nextafter(n,+inf)` 仍返回 n。`0.2 m` 与 `1.182 m` 基线只检查
   `k=0..11`，共 12 个位置；
2. 候选中心必须位于任务多边形内的 FREE、map-backed cell；
3. 最小间距只作用于位置组的 XY，不作用于同一 XY 的 yaw 变体。新位置与任一已接受位置组距离
   `< minimum_spacing_m` 时拒绝，恰好等于时接受；用确定性空间桶查询，禁止全局 O(N^2) 扫描；
4. 对通过中心和间距检查的位置分别检查配置中升序的五个 yaw。至少一个 yaw 安全时接受该位置
   组、保留全部安全 yaw，并停止该代表点的更远搜索；五个 yaw 均不安全时继续检查下一个 `k`；
5. 平台足迹必须是有限、非退化、无重复顶点的简单凸多边形，顺/逆时针输入均可并在构造时
   规范化；`platform_type` 必须为 `WHEELED`，platform/base frame ID 非空，clearance 有限且非负；
6. clearance 膨胀严格定义为旋转后原足迹与半径 `minimum_clearance_m` 的闭圆盘 Minkowski 和。
   将候选足迹通过 `TaskRaster` 共享变换放到逻辑 grid frame；一个 logical cell 的闭方块与原足迹
   的最小 grid 距离 `<= minimum_clearance_m/resolution_m` 即被膨胀足迹接触，相切也计接触；这与
   米制最小距离 `<=minimum_clearance_m` 等价，不用径向放大顶点近似圆角；
7. 所有被接触 cell 必须同时满足 `Classify(index)==kFree` 和 `IsMapBacked(index)`；因此
   UNKNOWN、OUTSIDE_MAP、OCCUPIED、OUTSIDE_TASK 均拒绝。AABB 枚举采用有符号索引及受检乘加；
8. 无合法观测位置的前沿簇保留为不可用诊断，不补造位于未知区的目标。

生成器不相信调用方的簇顺序：先按完整 `canonical_key` 排序簇，再按 25/50/75% 代表顺序、
距离序列和配置 yaw 顺序输出。重复完整 canonical key 是结构错误。`frontier_distance_m` 固定为
最终候选中心到所选接口边中点的欧氏距离，不表示到簇质心或机器人距离。

候选完整身份由完整 frontier canonical key 与下列 pose key 共同组成；两个显示哈希都不参与
相等判断：

```text
CandidateKey = [x_mm, y_mm, yaw_tenth_deg]
```

XY 按毫米、yaw 按 0.1 度使用 half-away-from-zero 量化。yaw 先规范化到 `[-pi,pi)`，其中
`+pi -> -pi`。量化输入非有限时抛 `std::invalid_argument`；缩放后的有限值若超出 int64 可表示
范围，在调用 `llround` 前抛 `std::overflow_error`。显示 candidate ID 对完整 frontier canonical
key 后接 CandidateKey 的有符号
64 位二补码小端字节执行 FNV-1a-64。生成 display candidate ID 时直接读取输入 cluster 的完整
canonical key。为使后续 factory 能无碰撞地验证候选确实属于给定冻结 span，`CandidateView` 除两个
display ID、CandidateKey 和 `frontier_index` 外，还保存
`shared_ptr<const vector<int64_t>> frontier_canonical_key`。CandidateGenerator 对每个实际产出候选的
frontier 只深拷贝一次完整 key，该 frontier 的所有 CandidateView 共享同一 const owner；禁止为每个
view 重复复制可变长 key。该 index 始终是调用方所给冻结 `frontiers` span 的原始下标，即使生成器
内部按 canonical key 排序处理也不得重编号；不增加 snapshot ID、revision、版本或时间戳。
Task 7 或诊断组合完整身份时，bounds-check `frontier_index` 并要求 shared key 与该 cluster 的完整
key 精确相等后，再与 CandidateKey 组合；失败记忆按批准的物理位置/yaw 语义只保存 CandidateKey。

ROS 协调层用 owning `FrozenCandidateBatch` 保存同一计算周期的 `vector<FrontierCluster>`、
`vector<CandidateView>`、冻结的全局地图 geometry/data 内容以及 `request_id -> candidate_index`
关联。frontiers/candidates/map content 构造后不可变，只有受节点状态 mutex 保护的 request 关联表可
在发请求前追加。ROS 内部 `FrozenPlanningCycle` 再共同拥有该 batch、完整唯一 CandidateGain 表、
冻结 robot pose/resolution、完整 coarse order/单调 cursor 和 reachable index/path/reference rows；它不
扩大 core 稳定 API。PlannerClient 的 transport callback 只捕获 `weak_ptr<CallbackState>`；CallbackState
中 resident active Completion 强持有同一 cycle，从而间接拥有 batch 及全部 FinalRank authority。节点
释放活动引用后，只要该 request 仍 active，Completion 就必须维持 cycle/batch；terminal Completion
返回或 client teardown 清除 active Completion 后二者才可析构。transport callback 不得直接捕获
cycle。CandidateView
的 `frontier_index` 只索引其 cycle owning batch 的 `frontiers`。地图/odometry callback 可替换 latest
cache，但不得改变 in-flight cycle 的 gain、pose、resolution、order、cursor 或 reachable rows；在
两个串行 planner result 之间更新 latest pose/map，也必须用原 cycle authority 完成 FinalRank。

`PlannerEvaluation.candidate_id` 仅供显示；算法回关联只允许以 `request_id` 查 owning batch 的 map，
再取得 candidate index。禁止按 candidate/frontier hash、显示 ID、数组位置猜测或当前 latest map
重新绑定。准备宣布候选穷尽或正常完成前，若最新全局地图 geometry/data 与 cycle batch 冻结内容
已实际不同，丢弃该 cycle 的结论并从最新内容新建 cycle；此判断只比较数据内容，不增加或读取
stamp、revision、版本或新鲜度字段。

提交目标必须调用 `MakeActiveGoal(candidate, frozen_frontiers, request_id)`（或等价 factory）。factory
先检查 index bounds、候选/request 的有限性和结构，要求 candidate shared key 非空，并同时检查
`candidate.frontier_id == cluster.id` 与 `*candidate.frontier_canonical_key == cluster.canonical_key`。
因此候选 A 与同 display hash 但不同完整 key 的 span B 交叉配对必须拒绝；不靠显示哈希、snapshot
ID、revision、版本或时间戳。全部检查通过后才从 shared key 深拷贝一次到 ActiveGoal。人工同
hash/不同 key 时，分别传入各自合法 span 必须得到保存不同完整 key 的 ActiveGoal，后续跨周期匹配
只比较 ActiveGoal 的完整 key。

`CandidateGenerator::Limits` 的 `maximum_position_probes`、`maximum_candidate_views` 和
`maximum_collision_work_units` 都由调用方显式传入正数，核心不提供生产默认值。设 footprint 顶点
数为 V；通过基本平台字段和 `V>=3` 后、任何逐顶点或二重验证循环前，构造器必须用 checked
size_t 算术计算静态几何验证的确定最坏上界：

```text
static_footprint_validation_work =
    V*(V-1)/2                      # duplicate-point pairs
  + V*(V-3)/2                      # non-adjacent edge pairs
  + V                              # convexity and signed-area scan
  = V*(V-1)
```

任一乘加无法表示或该上界大于 `maximum_collision_work_units`，都必须在上述循环前抛
`std::length_error`。该比较按最坏上界准入，和输入 winding 或提前发现某个非法顶点无关；limit
恰等于上界成功进入验证，少一失败。有限性、规范化和 extents/radius 的线性扫描也只能在该准入后
执行；通过准入后仍按既定规则以 `invalid_argument` 拒绝非有限、重复、退化、自交或非凸 footprint。

同一数值上限还独立约束每次 `Generate` 的累计 collision work；构造器的静态准入不从某次
`Generate` 额度扣除，每次调用从 `used=0` 开始。每次 `CollisionFree` 在 reserve 或变换前先用
`remaining=limit-used` 检查 `V<=remaining`；随后每变换一个 footprint vertex 前消费一个 collision
unit，再每从膨胀足迹 AABB 取出一个 logical cell 时，在分类或几何接触测试前消费一个 unit。
所有消费统一为访问前若 `used==limit` 则抛，否则 `++used`；因此已知需要 N 单位的 fixture 在
limit=N 成功、N-1 抛 `std::length_error`。每个被检查的代表点/距离中心消耗一个 position probe，
每个准备加入输出的安全 yaw 消耗一个 candidate-view 单位。任一资源超限都在对应 reserve、最终
大向量分配或追加前抛 `std::length_error`，ROS 映射为 `ERROR`，绝不能返回空候选并支持任务完成。

`CandidateParameters` 的五个 offset 必须有限、严格按批准顺序递增且无重复，部署值精确为
`[-45,-22.5,0,22.5,45] deg`。输入 pose、接口边 midpoint、局部质心、簇长度和 resolution
出现 NaN/Inf 时是结构错误，不能交给排序比较产生未定义顺序。
有限输入若在推导 footprint 长宽/半径、minimum standoff、maximum extra search、spacing、search
ratio、候选中心或变换坐标时产生非有限或不可表示结果，统一抛 `std::overflow_error`；不得把派生
溢出误报为非法输入，也不得截断为最后一个合法 search step。

候选预筛只减少明显无效的规划调用；纯规划器仍拥有足迹、运动学、全局路线和局部高程的最终裁决权。

## 10. 10 m / 90 度信息增益

Task 5 以前沿局部 UNKNOWN 质心方向为标准 yaw，并生成：

```text
standard_yaw + {-45 deg, -22.5 deg, 0 deg, +22.5 deg, +45 deg}
```

Task 6 对每个已经带有最终 pose/yaw 的 `CandidateView` 独立评估：

- 观测半径固定 `10.0 m`；
- 水平视场固定 `90.0 deg`；
- 传感器原点固定为候选 `pose.x/y`，首版没有额外传感器外参；
- 只枚举 cell center 满足欧氏距离 `<=10.0 m` 且与候选 yaw 的最小环绕 bearing 差
  `<=45 deg` 的目标；闭边界比较先转入去除大世界平移的局部 grid frame。数值 authority 固定为：
  先以 double 计算 `range_cells=range_m/resolution_m`，再提升为 long double 做距离平方/AABB；
  bearing、`remainder` 最小环差和 `half_fov=field_of_view_rad/2.0` 也先按 double 计算再提升比较。
  不使用 epsilon 或正容差扩张，因此批准的 10/0.2=50、10/0.1=100 和 ±45° 边界包含，而
  `nextafter` 到边界外侧排除；
- UNKNOWN 和 OUTSIDE_MAP 计为潜在信息增益；
- 在逻辑 grid frame 中从候选连续位置到目标 cell center 做流式 `O(K)` closed-grid DDA；禁止按
  每条射线二维 bbox 扫描。supercover 是与闭射线段相交的全部闭 cell 方块，按首次接触参数 `t`
  递增分组且每 ray/index 去重；target cell 不作为 pre-target blocker；
- FREE 栅格不增加信息增益且不遮挡；
- UNKNOWN 和 OUTSIDE_MAP 不遮挡；首次接触任何 OUTSIDE_TASK 即终止该射线，禁止在凹任务区域
  中离开后重新进入并隔空计增益；
- 同一未知栅格在一个候选中只计一次；
- `information_gain_m2 = visible_unknown_cell_count * resolution_m^2`。

`InformationGainEvaluator` 只评估一个输入 CandidateView 并原样返回可见数量和面积，不排序、
不删除 zero-gain 候选。其 `maximum_visibility_work_units` 由调用方显式传入正数，核心不提供生产
默认值。候选 grid 坐标为 `(sx,sy)`、`r=range_cells` 时，目标中心最紧闭 AABB 固定为
`xmin=ceil(sx-r-0.5)`、`xmax=floor(sx+r-0.5)`，y 同理；用受检 long double/int64 与 int32 域
求交，按 y 升序、x 升序枚举。理论 AABB cell 数超过 remaining budget 时，在分配/循环前抛
`std::length_error`。每取出一个 AABB cell，必须在 range/FOV/Classify 前消费一个 work unit；只有
通过 range/FOV 且状态为 UNKNOWN/OUTSIDE_MAP 的 target 才发射射线，其他状态不产生 ray work。

每条 DDA 的 `t=0` 闭 cell 组：普通点 1 格、单格线 2 格、格角 4 格。非沿格线 crossing 每次加入
新进入格；`tMaxX==tMaxY` 的角点组加入 x-side、y-side、diagonal 三个新格。整条射线若沿整数
格线，显式枚举两侧，crossing 时两侧新格属于同一 t 组，禁止用 epsilon 偏移模拟。每个去重 cell
在分类前消费一个 work unit；整组全部消费/分类后，先看任一非 target OUTSIDE_TASK，再看任一非
target OCCUPIED，优先级 `OUTSIDE_TASK > OCCUPIED > pass`。阻挡后不再生成或消费后续组。
计数器规则固定为访问前若 `used==limit` 则抛，否则 `++used`，不得先 reserve 理论射线/bbox 大小。
因此恰好 N 次访问在 limit=N 成功，limit=N-1 抛 `length_error`。

为使上述几何、资源和溢出规则能够被直接验证，Task 6 在 `src/detail/visibility_traversal.hpp` 暴露
并 inline 实现仅供核心实现和白盒测试链接的内部 seam；该头文件不安装、不导出，也不属于稳定
C++/ROS API，禁止把遍历本体藏到另一个未受检 cpp/helper。
`detail::TraceClosedSegment` 接受任意有限连续 grid 起终点和共享 work-budget，以流式 callback 暴露
按 t 递增的规范 cell groups，并返回本 ray 的唯一 visited-cell count；group 内按 `(x,y)` 排序，
消费发生在 callback 分类前，callback 可在完整同 t group 分类后停止后续生成。这样可以直接测试
整段沿整数格线、t=0 普通点/格线/角点、角点三新格、同 t 分组、每 ray 去重和 exact work count，
而不依赖某个公开 `Evaluate` fixture 恰好产生可区分 gain。

同一 detail seam 还提供 target-aware pre-target contact 分类和
`CheckedVisibleIncrement(std::uint32_t&)`：前者对 `visited==target` 必须返回 pass，否则区分
OUTSIDE_TASK、OCCUPIED 与 pass；后者在当前值为 UINT32_MAX 时抛 `std::overflow_error`。生产
`InformationGainEvaluator::Evaluate` 必须调用共享 `ConsumeVisibilityWork` 以及这些 seam，不能另写
一份预算消费、遍历、target 排除或计数逻辑。
源码合同测试必须同时抽取 `information_gain.cpp` 的 Evaluate 函数体和该 detail header 中真实的
`TraceClosedSegment` 函数体：前者检查调用链，后者拒绝 ray-bbox/slab 二重扫描，并要求每条 cell
traversal 路径在分类/callback 前调用 `ConsumeVisibilityWork`；即使没有双循环，也必须拒绝按理论
二维 ray-bbox 面积执行 `vector(width*height)`、`reserve(width*height)`、`resize(width*height)` 或
等价 allocation。mutation fixtures 分别删除 Evaluate seam 调用、把真实 TraceClosedSegment body
替换为 bbox 双循环、删除其中的消费调用、或仅插入上述二维面积 allocation，四类都必须失败。
`Evaluate` 仍是唯一稳定公共 API；detail 类型不得进入 ROS 接口、安装头文件或 Task 7/8 消费者。

`SensorModel` 要求 range 和 FOV 有限，range `>0`，FOV 位于 `(0,2*pi]`；部署值精确为
`10.0 m` 和 `pi/2`。`yaw_grid=wrap(candidate.pose.yaw-origin_yaw)`；最小环差采用 double
`remainder` 的绝对值，+pi/-pi 等价。target center 与传感器原点重合时视为任意 FOV 内；
FOV=2*pi 显式全向。range/FOV 是无正容差扩张的闭边界。

AABB/DDA 步进必须检查 int32/int64/size_t 算术；沿 int32 边界扩出的闭 side cell 视作虚拟
OUTSIDE_TASK，不允许 wrap 或伪造成 OUTSIDE_MAP。只有唯一枚举的 qualifying target 自身能贡献
gain，中间 UNKNOWN/OUTSIDE_MAP 只是不遮挡，不能顺路贡献。`visible_unknown_cells` 增加前检查
UINT32_MAX。面积先以 long double 预检可表示性，最终权威表达式固定为
`double(count) * (resolution * resolution)` 并要求 finite；溢出抛 `std::overflow_error`。

跨分辨率 oracle 固定两类：对齐 UNKNOWN 矩形 `x in [2,3), y in [-0.4,0.4)` 在 0.2 m 为
20 cells、0.1 m 为 80 cells，两者按冻结 double 表达式均为同一 0.8 m²；无遮挡普通 10 m/90°
扇区在 0.2 m 为 1996 cells/79.84 m²，在 0.1 m 为 7924 cells/79.24 m²。黄金计数由独立整数参考
`i²+j²<=R², i>=0, |j|<=i` 或锁定常量验证，不能调用生产 evaluator 自证，也不强行令普通扇区
跨分辨率面积相等。所有情况满足冻结顺序 `area=double(count)*(resolution*resolution)`。

## 11. 两级候选排序与规划验证

### 11.1 廉价粗排

```text
coarse_utility =
    information_gain_m2 /
    (euclidean_distance_m + platform_length_m)
```

Task 7 的输入记录只携带 owning frozen candidate span 的 `candidate_index` 和 Task 6 产生的 gain；
Ranker 同时借用同一 batch 的 frozen candidate/frontier spans 和一个冻结机器人 pose，不保存这些
span、CandidateView 副本或引用。每个 index 必须唯一且覆盖每个候选，先对 candidate index 和
`CandidateView.frontier_index` 两级 bounds-check，再验证 CandidateView/robot pose 结构有限；不能用
candidate/frontier display hash 回绑或补齐缺失记录。

欧氏距离和 heading change 由 Ranker 从该冻结 pose 与 CandidateView.pose 内部派生，调用方不得注入。
heading 固定为 `abs(remainder(normalized_target_yaw-robot_yaw, 2*pi))`，位于 `[0,pi]`，`+pi/-pi`
等价，负零规范为正零。Task 7 先精确删除 `information_gain_m2 == 0.0` 的候选，不用 epsilon；gain
必须有限且非负。`platform_length_m` 必须有限且大于零；coarse denominator 必须有限且大于零，
utility 非有限时抛 `std::overflow_error`，NaN 绝不能进入 comparator。

CoarseRank 返回的 `RankedCandidate.path_length_m` 和 `revisit_penalty` 固定初始化为有限正零
`0.0`；粗排没有规划路径或 completed-goal 输入，禁止在这两个字段中放未初始化值、占位 NaN 或
调用方旁路数据。

Ranker 返回**全部**正增益候选的 owning `candidate_index` 完整有序结果，而不是只返回第一批或复制
CandidateView。Task 11 维护单调 cursor，按该结果依次切成最多 16 个的 `[cursor,
min(cursor+16,count))` 批次；因此 17 个及以上候选不会重复首批或遗漏尾批。粗分从高到低；同分
依次使用：信息增益更大、欧氏距离更小、heading change 更小、候选世界 x/y 字典序更小、规范化
yaw 更小。上述可见字段仍完全相同时，以该 index 解析的完整
`(frontier.canonical_key, CandidateKey)` 作最后总序；重复完整身份是结构错误。实现可使用
`std::stable_sort`，但交换输入记录排列不得改变输出；display ID 不参与 equality/order。
`InformationGainEvaluator` 不承担这些职责。

### 11.2 分批规划

- 默认每批最多 `16` 个候选；
- 按粗排顺序串行调用 `/Car/T4/plan_motion`；
- 每个 Goal 固定 `environment_mode=LUNAR_SURFACE`、`replace_active_request=false`；探索器不发送洞内
  模式，也不并发占用规划器 single worker；
- `request_id` 对每次调用唯一；
- `mission_id=task_id`、`mission_revision=0`；两个兼容字段不参与纯规划器准入或结果关联；
- `PlannerEvaluation.candidate_id` 只显示，唯一回关联键是客户端保存的单 in-flight `request_id`；
- 一批全部获得结论性穷尽结果时继续下一批；
- 必须穷尽全部候选才能得出“无可达前沿”。

`GoalRegion` 完整冻结为：`header.frame_id="map"`、确定非空
`goal_id=request_id + "/goal"`、`goal_type=POINT`、`point.x/y` 精确取候选 XY、`point.z=+0.0`、
空 `planar_region`、有限非负位置容差、`has_yaw_constraint=true`、`yaw_rad` 精确取候选 yaw、有限
非负 yaw 容差。Explorer canonical outgoing `point.z` 必须精确为正零，Task 10 client 测试不得改变 z。
Goal header stamp 与所有 Result/MotionReference/path pose stamp 不参与准入、关联、分类或超时；stamp
测试可任意设置为零、旧值或乱序值而不改变结论。只有绑定最终 server source SHA 的独立 adapter
合同可用任意 finite incoming z 证明 server 忽略 z；这不放宽 explorer client 输出合同。

规划结果不用三个可矛盾 bool，而使用唯一互斥 typed kind：

```cpp
enum class PlannerEvaluationKind : std::uint8_t {
  kReachable,
  kExhaustiveNoPath,
  kRetryable,
  kCanceled,
  kContractError,
};
struct PlannerEvaluation {
  std::string request_id;
  std::uint64_t candidate_id;  // display only
  PlannerEvaluationKind kind;
  std::string reason_code;
  std::optional<double> path_length_m;
  std::optional<lunar_planning_msgs::msg::MotionReference> reference;
};
class PlannerClient {
 public:
  ~PlannerClient() noexcept;
  PlannerClient(const PlannerClient&) = delete;
  PlannerClient& operator=(const PlannerClient&) = delete;
  PlannerClient(PlannerClient&&) = delete;
  PlannerClient& operator=(PlannerClient&&) = delete;
  // Evaluate, CancelActive and PollTimeout are defined by the plan interface.
};
```

非空 typed Result 必须先按完整 payload 分类，再检查 ROS Action wrapper 是否与 payload 相容；
不得先把 `ABORTED` 整体降级为 transport retry。冻结矩阵如下：

- `SUCCEEDED` 加
  `NEW_REFERENCE_AVAILABLE/ACTIVATE_NEW_REFERENCE/has_reference=true` 是 `kReachable`；
- `ABORTED` 加 `GOAL_INFEASIBLE/NO_SAFE_REFERENCE/has_reference=false`，且 reason 精确为
  `NO_PATH` 或 `GOAL_OUTSIDE_LOCAL_MAP`，是 `kExhaustiveNoPath`；二者是且仅是任务穷尽证据；
- `ABORTED` 加 `RESOURCE_EXHAUSTED/NO_SAFE_REFERENCE/has_reference=false/reason_code=TIMEOUT`
  是 `kRetryable`；这是 `bf1bb79` final server 唯一允许的 typed resource reason，任何其他
  `RESOURCE_EXHAUSTED` reason 都是 `kContractError`；server unavailable、Goal reject、任何没有 typed
  Result 的 transport 终态和客户端
  steady result deadline 也都是 `kRetryable`；
- 只有本客户端已经请求取消，且收到 `CANCELED` wrapper 加
  `CANCELED/NO_SAFE_REFERENCE/has_reference=false/REQUEST_CANCELED`，才是 `kCanceled`；
- `INVALID_REQUEST`、`NUMERICAL_FAILURE`、`STALE_INPUT`、`ACTIVE_REFERENCE_INVALIDATED`、
  `SAFE_FRONTIER_REFERENCE_AVAILABLE`、`NO_KNOWN_SAFE_ROUTE`、未知 outcome/directive、outcome 不支持的
  reason、非本客户端
  请求的 typed `CANCELED`，以及 wrapper/outcome/directive/has-reference/reference/path 任一矛盾组合，
  都是 `kContractError`。typed success 被 `ABORTED` 包裹、typed failure 被 `SUCCEEDED` 包裹、空 reason
  或其他 `GOAL_INFEASIBLE` reason 同样是合同错误。

`PlannerEvaluation.reference` 和 `path_length_m` 使用联合不变量：只有 `kReachable` 同时携带完整
reference 与 finite nonnegative path length；其他四个 kind 两项都必须为 `nullopt`，不能用失败态
`0.0` 冒充已验证路径。成功 reference 逐字段原样保存，包括 header、input time、path pose、
trajectory 和 hop stamps，不得因校验而重写。

`maximum_path_preview_poses` 是部署必填正数，在遍历或复制前检查；over-limit 是合同/资源错误。
preview 为空是合同错误；一个有限 XY pose 合法且路径长精确为 `+0.0`。只使用 XY：先把每个有限
double 坐标提升为 long double，再做差和 `hypot`；每段和累计值都须有限，累计结果必须可表示为
finite double，最终负零规范为正零。z、orientation 和全部 stamp 被忽略。完整 reference 仍原样保存。

最终 planner 内部的一秒 absolute deadline 是协作式：已发布的 deadline-aware planner 算法只在内部安全
检查点观察 `control.deadline` 和 stop token，不实现 detached worker 或 hard Action preemption；终态收尾
可以跨过该一秒预算。客户端另设精确默认 `2.0 s` 的
`std::chrono::steady_clock` transport watchdog。`Evaluate` 在发送前用可注入 `SteadyNow` 固定闭 deadline，
生产 steady timer 与测试都调用 `PollTimeout()`；它覆盖 Goal response 迟到、accepted-without-result、
server crash 和 DDS loss，不读取 ROS time，也不是 freshness 检查。deadline 获胜返回
`kRetryable/CLIENT_RESULT_TIMEOUT`：先在 mutex 内 claim terminal、移出 Completion 和 optional
already-accepted Goal handle，并清除 active request/Goal/cancel intent/deadline。解锁后，若该 handle
存在，只对该旧 Goal best-effort `async_cancel_goal(old_handle)`，然后调用唯一 Completion；若不存在，
不发送任何 cancel，直接调用唯一 Completion。Production MUST NOT call `async_cancel_all_goals`，因为其
晚处理可能取消由 Completion 立即启动的新 generation。旧 scoped-cancel response 与全部旧 callback 都
绑定已终止 generation，不能触及新 Goal。晚到 Goal response、Result、cancel response、feedback 或
timer callback 在 terminal claim 后全部忽略；Feedback 明确不消费，不影响结果、计时、取消、watchdog
或 freshness。

Goal response、Result、CancelActive、watchdog 与析构共享 single-winner terminal claim。所有传输
callback 只捕获 shared `CallbackState` 的 `weak_ptr`，禁止跨对象寿命捕获裸 `this` 或直接捕获
FrozenPlanningCycle。CallbackState 中 resident active Completion 强持有 Task 11 cycle。非析构 winner
在 mutex 内把 Completion、request 和 owning captures 移到局部并清空 active state，解锁后才调用
Completion；因此 Completion 可立即 reentrant `Evaluate` 下一串行 Goal。Completion 抛异常必须在
callback 边界被吞并，不能回滚清理或产生第二次完成。接口显式声明 `~PlannerClient() noexcept` 并删除
copy/move 构造和赋值；析构锁住 state mutex，claim/记录 teardown，清空 active Completion、Goal、
cancel intent 和 deadline 且不调用用户 Completion。callback 即使已 weak-lock state，只要在 teardown
后才取得 mutex，也必须看见 teardown 并 no-op；已持 mutex 的 callback 与析构按该锁串行。

取消矩阵固定为：Goal response 前取消只记录 intent；只有 accepted handle 到达时该 evaluation 仍
active/non-terminal，才仅补发一次 per-handle cancel；重复
CancelActive 幂等；Goal reject 若赢得竞态始终产生一次 `kRetryable` transport terminal，即使 cancel
intent 已记录；cancel response 无论接受或
拒绝都不是 Action terminal，仍等待 typed Result 或客户端 watchdog；cancel 接受但 Result 迟到/不来
同理；Result 先于 cancel response 时 Result 获胜，晚响应忽略；只有实际 locally-requested typed
canceled Result 产生 `kCanceled`，未经请求的 canceled payload 是合同错误。候选评估始终等待这个
single-winner 终态后才发送下一 Goal。若 PAUSE/CANCEL/replacement START 已进入 cancel-and-wait，
旧请求的任一 winner（包括 Goal reject 或 watchdog 的 `kRetryable`）只关闭 transport gate并完成控制
过渡，不重试旧候选，也不能影响随后建立的新任务。

精确竞态 `CancelActive -> PollTimeout(2s deadline) -> late accepted Goal response` 中，watchdog winner
claim 时尚无 handle，因此清除 cancel intent、完全不发送 cancel、调用唯一 retryable Completion 并允许
新 `Evaluate`；晚到 Goal response 不得再次 completion、改变 kind 或补发 per-handle cancel。generation
安全 RED 必须覆盖：(a) handle-before-timeout 只 scoped cancel 被取走的旧 handle；(b) timeout-before-
handle 不 cancel、不 cancel-all，late response no-op；(c) Completion 启动新 Goal 后，旧 scoped-cancel
response 和其他旧 callback 都不能取消或修改新 generation。旧 remote Goal 可能让 final server 的
single worker 短暂 busy，导致新 Goal reject；该 reject 仍为 `kRetryable`。`PlanMotion` 只计算
reference、不发布或执行 reference，因此没有 controller side effect。若 accepted handle 先于 watchdog
claim 且 cancel intent 仍 active，则正常控制路径恰发送一次 per-handle cancel；watchdog 后续清理仍只
能 scoped 到同一旧 handle。

当前探索分支尚未包含最终 planner integration；已审计基线是
`feat/pure-planner-isolated@bf1bb79`。只有该提交或合同等价后继合入当前分支并重新验证真实 server 的
`ABORTED+typed payload`、Action 生命周期和 diagnostics-before-terminal 后，才能实现/完成 Task 10。
fake server 只能提供确定性 RED/GREEN，不能解除 gate 或证明最终接口兼容。

规划调用仅计算参考，不自动下发控制器，因此可以用于候选比较。

### 11.3 最终评分

在当前可达候选集合内计算：

```text
score =
    0.60 * normalized_information_gain
  - 0.30 * normalized_global_path_length
  - 0.05 * normalized_heading_change
  - 0.05 * revisit_penalty
```

- `normalized_information_gain`：除以当前可达集合最大增益；
- `normalized_global_path_length`：除以当前可达集合最大路径长度；
- `normalized_heading_change`：当前 yaw 与目标 yaw 的最小绝对差除以 pi；
- `revisit_penalty`：候选距任一已完成目标不大于重访半径时为 `1`，否则为 `0`；
- 分母为零时对应归一化值定义为零；
- 默认权重可配置，但必须为有限非负数。

CandidateGain 完整表是唯一 gain authority；最终 planned 输入只携带同一 owning batch 的唯一
candidate index 和有限非负真实 path length，必须是该 gain 表的可达子集；允许空可达集合并直接
返回空。Ranker 从同一冻结机器人 pose 重新派生 heading，并从有限
completed-goal world XY、当前有限正 resolution 和 `platform_length_m` 内部派生
`revisit_radius=max(2*platform_length_m,2*resolution_m)`；半径必须有限，相切 `<=` 计重访，
`nextafter` 到外侧不计。调用方不能提交任意 distance、heading 或 revisit penalty。

四个权重分别必须是有限 double 且非负；以 long double 累加并要求总和严格大于零。四个有限
double 在批准的 amd64/Orin ABI 上不会令 long-double 累加溢出，因此不设置不可构造的“有限权重
令 long-double sum 溢出”RED，也不额外要求总和可表示为 double。默认总和精确为 1.0，但自定义
总和不要求等于 1，也不静默重归一化。gain/path 最大值为零时，相应 normalized component 精确
为零；每个加权 component 和 long-double 最终 score 在转换为 double 后必须有限且可表示，否则
抛 `std::overflow_error`。DBL_MAX 级有限权重使用两个或以上同号非零 penalty component 构造稳定
score-overflow RED。最终同分时依次选择：信息增益更大、真实路径更短、转向更小、世界坐标
字典序更小、规范化 yaw 更小，再以完整 `(frontier.canonical_key, CandidateKey)` 作最后总序。
结果必须对输入 permutation 确定，人工相同 display hash/不同完整 key 仍保持不同。

## 12. 目标承诺与失败恢复

一旦选定最终候选并发布运动参考，普通地图更新或出现得分更高的新候选不会切换目标。月表
`path_preview` 是到最终候选的完整全局粗路线，而 `trajectory/hops` 只覆盖当前局部图内的可执行段；
探索器必须分别保存最终承诺候选、当前 `plan_id` 和当前可执行段末端。只有以下事件触发重新选择：

- 到达目标；
- 当前目标格不再是 FREE；
- 当前旋转足迹不再完全位于已知自由区；
- 当前目标对应的前沿消失或预计信息增益变为零；
- 规划结论性返回 `kExhaustiveNoPath`（`NO_PATH` 或 `GOAL_OUTSIDE_LOCAL_MAP`）；
- 进度看门狗判定卡住；
- 收到 `PAUSE/CANCEL/START`。

到达当前可执行段末端但尚未到达最终候选时，对同一承诺候选发起新的串行月表规划请求；这属于
滚动重规划，不释放承诺，也不消耗卡住重试次数。执行阶段真正卡住时，对同一候选最多重新规划
两次；仍失败后，将候选加入当前任务失败记录并选择下一候选。TIMEOUT、取消、服务/传输故障或
非法响应不能写入物理不可达失败记录。

`ActiveGoal` 是不可聚合构造、构造器私有且只授权上述 factory/状态机的类型；factory 从同一冻结
span 中的 `frontier_index` 深拷贝完整 frontier canonical key，并同时保存 CandidateKey。状态机
`CommitGoal` 只接受已构建的 ActiveGoal，调用方不能绕过 factory 拼装一个未验证承诺。
candidate/frontier 显示 ID 都不参与相等判断。当前前沿是否消失必须在新快照中比较完整 canonical
key，不能只比较 FNV-1a-64。失败记录键使用 CandidateKey 的量化候选世界位置和 yaw，绝不使用
candidate display ID。只有执行阶段卡住且同一候选的两次重规划已经耗尽时，才能用 typed reason
`kExecutionReplansExhausted` 写入持久失败；候选评估 `NO_PATH`、TIMEOUT、取消、拒绝和传输/合同
错误都不能写入，`GOAL_OUTSIDE_LOCAL_MAP` 也只作为当前冻结 batch 的结论性穷尽证据而不持久化。
每次已接受 START（即使复用同一 task_id）都在 cancel-and-wait 完成后调用一次
`BeginTask` 并无条件清空失败记录；PAUSE/RESUME/普通地图重算不建立新 scope。

Task 8 的 core API 必须把滚动分段与卡滞恢复做成不同的类型，不允许调用方用字符串或隐含状态
猜测重规划原因：`BeginReplanning(kRollingSegment)` 保留 ActiveGoal 且不改变 `replan_count`；
`BeginReplanning(kStuckRecovery)` 在计数仍小于 `maximum_replans_per_candidate` 时先加一再进入
`REPLANNING`。当计数已等于上限时，下一次卡滞调用必须原子清除 ActiveGoal、进入
`SELECTING_FRONTIER`、返回 `kExhausted`，并报告 `STUCK_RETRIES_EXHAUSTED`。因此默认上限为 2
时，前两次卡滞分别以计数 1、2 开始重规划，第三次卡滞才耗尽。`maximum_replans=0` 合法，第一次
卡滞即耗尽。成功取得新可执行段后只能由 `ResumeExecution()` 从 `REPLANNING` 回到
`EXECUTING`。进度监测器只检测卡滞，不保存或消费重规划次数。

`MakeActiveGoal` 要求非空 `request_id`、有效 `frontier_index`、有限候选 pose、有限非负
`frontier_distance_m`、非空 shared canonical key，且候选显示 frontier id 和 shared key 必须分别
等于被索引 cluster 的显示 id 与完整 key；异常不得留下半构造对象。factory 深拷贝 key、CandidateKey、
target 和 request id，输入 span/shared owner 随后销毁或修改原始 frontier 存储不影响 ActiveGoal。
显示 id 相同但完整 key 不同的两个合法快照都可构造，交叉配对必须拒绝，且
`MatchesAnyFrontier(current_frontiers)` 只按完整 canonical key 判断当前前沿是否仍存在。
ActiveGoal 只可移动构造，不可复制、移动赋值或默认构造。自定义 move constructor 转移私有
`owns_payload` token 并把源对象永久标为 consumed；不得依赖 moved-from string/vector 恰好为空。
`CommitGoal(ActiveGoal&&)` 在验证合法源状态和 `owns_payload` 后才移动，无 payload/已消费对象抛
`std::invalid_argument` 且状态机保持强保证。因此调用方既不能复制带旧 `replan_count` 的目标，也
不能重复提交同一 moved-from 句柄，factory 仍是计数为零的新 payload 唯一来源。

失败 entry 保存完整 CandidateKey、首次记录的有限世界 XY、记录时固定的
`failure_radius=max(platform_length_m,2*resolution_m)`、完整有限 `GridGeometry` 和规范 occupancy
patch。数值 authority 固定为 double-then-promote：先以 double 计算
`twice_resolution=2.0*resolution_m`、failure radius 和
`radius_cells=failure_radius/resolution_m`，每一步都要求有限正值；保存的 world center 必须调用
TaskRaster/OccupancyGridView 共享 `WorldToGrid` 得到 double `(cx,cy)`，不得复制旋转/平移公式。随后
才把这三个 double 提升为 long double，并计算未裁剪 tight cell-center AABB：

```text
xmin = ceil(cx - radius_cells - 0.5)
xmax = floor(cx + radius_cells - 0.5)
ymin = ceil(cy - radius_cells - 0.5)
ymax = floor(cy + radius_cells - 0.5)
```

理论未裁剪 width/height/cell count 先在 long double 中与
`maximum_patch_cells_per_entry` 比较；若 width、height 或 checked product 超限，优先在任何 int32
裁剪/转换、分配或循环前抛 `std::length_error`。不允许把 AABB 裁到地图或 GridIndex 域来减少资源
work。只有理论 work 准入后才检查四个 bound 是否位于 int32；任一越界抛
`std::overflow_error`，不生成虚拟 cell，也不 wrap。

准入后按 y/x 升序枚举完整 AABB，每格在分类前消费一个 work unit。闭圆成员关系只在去除世界
平移/旋转后的 logical grid 中以 long double 计算：cell center 为 `(x+0.5L,y+0.5L)`，与提升后的
double `(cx,cy)` 之差平方和 `<= promoted_radius_cells*promoted_radius_cells` 才入 patch；无 epsilon，
相切包含，double `nextafter` 到外侧排除。旋转 map 不在 world frame 重算距离。保存
`kOutsideTask/kOutsideMap/kUnknown/kFree/kOccupied` 全五态。后续同 CandidateKey 始终以 entry 保存的
世界中心、double radius 和同一算术重建，不因同一毫米量化桶内的细小 pose 差改变区域。

只有当前 GridGeometry 的 width/height/resolution/origin x/y/yaw 与 entry 完全相同，逻辑 GridIndex
才可比较。geometry 任一字段变化时，不能拿旧 index 对应新物理区域，必须保守删除 entry 并返回
不抑制；geometry 相同时，patch 内任一 index/分类或 patch 长度改变也删除该 entry。远处分类变化
不影响 entry；不读取消息时间戳、地图 revision 或版本。

命名空间级 `FailureMemoryLimits` 的 `maximum_entries`、`maximum_patch_cells_per_entry` 和
`maximum_total_patch_cells` 都由部署调用方显式传入正数，核心无生产默认值。tight AABB 理论 cell
数按上述未裁剪 long-double authority 与 per-entry limit 比较；准入后的循环/index/size_t 算术仍
全部受检，超限抛 `std::length_error`、不可表示抛 `std::overflow_error`。新增、同 key 替换及总量计算均先完成
预检和临时 patch，再原子提交；异常时原 entries/size/total 不变。达到上限不得静默逐出旧项、
解除抑制或支持正常完成。

## 13. 参考执行与车载进度看门狗

探索节点只把最终选中候选已计算的 `MotionReference` 发布到 `/Car/T4/execution/motion_reference`。发布后以课题三 Odometry 提供的 `odom -> base` 位姿和 `/tf` 中最新直接
`map -> odom` 解析 `map` 位姿；`/tf` 中即使存在 `odom -> base` 也不得覆盖 Odometry。

候选路径代价使用完整 `path_preview`；执行进度采用当前可执行 `trajectory` 或 `hops` 段折线上的
最近投影弧长，保存该段历史最大弧长，避免路径弯曲时用直线目标距离误判。到达段末端但未到达
最终候选时滚动重规划同一 ActiveGoal，完整 frontier key 和 CandidateKey 保持不变。

默认参数：

```text
stuck_timeout_s: 30.0
minimum_progress_m: 0.2
maximum_replans_per_candidate: 2
goal_yaw_tolerance_deg: 11.25
planner_result_timeout_s: 2.0
```

`maximum_position_probes`、`maximum_candidate_views`、`maximum_collision_work_units`、
`maximum_visibility_work_units`、`maximum_path_preview_poses`、`maximum_executable_path_points`、
`maximum_failure_entries`、`maximum_failure_patch_cells_per_entry` 和
`maximum_failure_total_patch_cells` 是部署调用方必须显式提供的九个正数资源预算，本设计不提供生产
默认值，也不把未测量的容器/Orin 能力写入上述默认 YAML。Task 14/16 只冻结参数名、必填、错误语义
和部署 profile 前置条件；缺失、零值或超限均进入 `ERROR`，不得以空候选或零增益继续。具体部署
值必须来自 Humble 压力测试及 Orin 时延/内存验收并由批准的部署 profile 显式提供。Task 6 的
visibility limit 是单次 Evaluate 上限，不能替代整周期验收；Task 14/16 必须按同一 profile 实测
`maximum_candidate_views * maximum_visibility_work_units` 所界定的最坏候选周期，记录整周期时延/
内存证据。failure-memory 证据不是 profile 中三组可自由改写的正浮点数：entry、per_entry、total
三个独立 fixture 各自产生一个仓库外、发布后不可变的 content-addressed JSON artifact。profile 对
每项保存 artifact **绝对路径和实际 JSON bytes 的 SHA-256**，并继续导出该项 elapsed/RSS 供逐值
交叉检查。三个 artifact 的固定 schema 为：

```json
{
  "schema": "lunar-pure-exploration/failure-saturation-evidence/v1",
  "fixture_kind": "entry | per_entry | total",
  "configured_limits": {
    "maximum_entries": 1,
    "maximum_patch_cells_per_entry": 1,
    "maximum_total_patch_cells": 1
  },
  "target_limit_name": "maximum_entries | maximum_patch_cells_per_entry | maximum_total_patch_cells",
  "reached_value": 1,
  "over_one_attempted_value": 2,
  "over_one_rejected": true,
  "elapsed_ms": 1.0,
  "peak_retained_rss_mib": 1.0,
  "authority": {
    "environment": "nonempty authority environment",
    "image": "nonempty immutable image digest or ID"
  },
  "generated_at": "record only; nonempty string"
}
```

三个 artifact 都包含同一批准 profile 的全部三个 configured limits；entry/per_entry/total 的
`target_limit_name` 分别固定映射到对应 limit，`reached_value` 必须等于该 target，
`over_one_attempted_value=reached_value+1` 且 rejected 为 true。三条绝对路径及 SHA 必须互异，实际
bytes 只读一次后先验 SHA 再按无重复 key、无未知/缺失字段的严格 schema 解析；fixture kind 或完整
path/hash record 互换会因 slot/kind/target 错配失败。artifact 中 elapsed/RSS 必须逐一等于 profile
对应导出值，且为有限正数；因此交换 entry/per_entry 测量、替换任意旧正测量或使用 stale-limit
artifact 都失败。三个 artifact 的 authority environment/image 必须非空、彼此相同并等于 profile
声明的 authority。`generated_at` 仅保存生成记录：验证器只检查它是非空字符串，**不得解析、比较
当前时间或执行年龄/新鲜度准入**。

三项仍使用三个独立 fixture：entry fixture 触达 entry limit、per_entry fixture 触达单 patch 理论
AABB limit、total fixture 触达累计 retained cells limit；各自把其他限制设为可达值，并验证 exact
成功/over-one fail closed，不要求一个实例同时触达三个可能互斥的边界。修改任一 failure limit 后，
旧 artifact 因 configured-limits 不相等而失效；不得把 artifact 或测量写入仓库，也不得据此拍出
新的生产默认值。

`maximum_path_preview_poses` 只界定完整全局 preview 的遍历/保存；
`maximum_executable_path_points` 复用于 WHEELED/LEGGED 的 `trajectory.points` 与 HOPPER 的 `hops`，
在复制、遍历或 ProgressMonitor reset 前检查 exact/over-one。不得另加 trajectory/hops 私有默认或
旁路 limit。platform/reference 组合若同时声称不相容的执行载荷属于合同错误；合法 reference 保存时
保持所有 stamp 和字段原样。PlannerTimingAccumulator 的 FIFO capacity 复用
`maximum_candidate_views`，不引入第十个部署资源参数。批准 profile 必须显式导出全部九个值，
包括 `PURE_EXPLORATION_MAXIMUM_PATH_PREVIEW_POSES` 和
`PURE_EXPLORATION_MAXIMUM_EXECUTABLE_PATH_POINTS`。

`planner_result_timeout_s` 是与上述九个无默认资源值不同的 transport 配置，仓库默认精确为有限正数
`2.0`。它只转换为 `steady_clock::duration` 并驱动第 11.2 节客户端 result watchdog；不与 ROS time、
header stamp、地图/状态年龄或 diagnostics stamp 比较，不能解释成 freshness。

车载实现使用 `std::chrono::steady_clock`：

- 测量进度窗口经过时间和 planner 客户端 transport result deadline；两种 timer 各自独立；
- 不依赖 ROS time、GPS、网络授时或系统日期；
- 系统时间校正不会造成倒退或跳变；
- 节点重启后重新建立进度窗口。

连续 `30 s` 内历史最大路径弧长增加不足 `0.2 m` 时判定卡住。暂停期间不累计该时间。
`ProgressMonitor` 构造时要求 timeout（类型为 `steady_clock::duration`）严格为正、minimum progress
为有限正数，且调用方必填的 `maximum_executable_path_points` 为正。`Reset` 在读取、分配或循环前
先检查折线非空且点数不超过该上限；exact limit 成功，over-one 抛 `std::length_error`。单点折线、
连续或非连续重复点合法。Reset 必须深拷贝/拥有化后续投影所需的全部折线数据，调用方修改或销毁
输入容器不改变语义；因此 retained memory 和每次 Update 的 O(N) 投影 work 都由同一点数上限约束。
投影到等距的多个线段/点时选累计弧长更大的投影，从而对自交和重复折线确定。段差、点积和距离
等中间量以 long double 计算；只要最终累计弧长和投影弧长有限且可表示为 double 就接受，有限的
极端位置本身不得因 double 中间溢出而误拒绝。路径段长或累计弧长不可表示时抛
`std::overflow_error`。所有输入检查和临时构建在提交前完成，异常保持旧 owned path、窗口、暂停、
卡滞和历史最大值不变。

`Update` 和 historical-max accessor 只能在首次成功 Reset 后调用。正常更新保存历史最大弧长；历史
最大值相对活动窗口基线增加 `>=0.2 m` 时把当前时刻和该最大值作为新基线。未达进度且经过时间
`>=30 s` 返回卡滞；卡滞为 latch，在成功 Reset 或一次 pause/resume 重建前持续返回 true。
Reset 可在未初始化、active 或 stuck 阶段建立新可执行段，清除 stuck latch 和旧窗口；暂停阶段
Reset 非法，防止收到新路径时隐式恢复执行。首次 Reset 可接受任意 steady time，后续 Reset/Update
都要求 `now >= last_accepted_time`，相等合法，倒退抛 `std::invalid_argument` 且保持强保证。

第一次 `paused=true` 更新记录暂停入口且忽略该位置；暂停中的时间和空间移动均不进入活动窗口。
第一次恢复更新以恢复时的当前位置重新投影，将历史最大值和窗口基线都重建为该投影，并从该时刻
重新开始完整 timeout 窗口、清除已有 stuck latch，因此恢复调用本身不报告卡滞。活动、首次暂停、
已暂停和首次恢复四条 Update 路径都拒绝倒退时间及非有限 position，并在异常时保持强保证。
elapsed 判断不得直接执行可能溢出的 time_point/duration 有符号相减。批准的 Ubuntu
22.04/libstdc++ amd64 与 Orin ABI 要求 `steady_clock::duration::rep` 是有符号整数；先用 time_point
关系拒绝倒退，再把两个 epoch count 转为对应 unsigned rep 并做模减，所得值就是 `[min,max]` 内的
精确非负 tick 差，随后与正 timeout count 的 unsigned 值比较。该方法必须覆盖
`time_point::min()/max()` 的合法前向跨度且不读取真实系统、ROS、GPS 或网络时间；不满足 signed
integral rep 的平台编译期拒绝，不伪造替代时间源。

目标到达必须同时满足：

```text
position_error_m <= max(0.5 * resolution_m, 0.25 * platform_width_m)
absolute_yaw_error_deg <= 11.25
```

取消、暂停、目标失效和重规划前，若参考已下发，先向 `/Car/T4/execution/cancel` 发布当前
`plan_id`；若规划 Action 仍在途，再发送 Action cancel 并等待终态，之后才能发送下一 Goal。
候选评估阶段没有控制器参考，只取消并等待 Action。正常串行评估、滚动重规划和卡住重规划均在
前一 PlannerClient single-winner terminal completion 后使用 `replace_active_request=false`；
`async_cancel_goal` response 不是终态，取消结果不写失败记忆。

## 14. 状态机

```text
IDLE
  -> WAITING_FOR_INPUT
  -> SELECTING_FRONTIER
  -> PLANNING
  -> EXECUTING
       |- reached --------------------> SELECTING_FRONTIER
       |- segment reached -----------> ROLLING_REPLAN -> EXECUTING
       |- goal invalid ---------------> SELECTING_FRONTIER
       |- stuck and retry available --> REPLANNING -> EXECUTING
       `- retry exhausted ------------> SELECTING_FRONTIER
  -> COMPLETED_NO_REACHABLE_FRONTIER
```

独立控制状态：

- `PAUSED`：任务保留，不选点、不规划、不累计活动时间；
- `CANCELED`：取消参考并回到 `IDLE`；
- `ERROR`：算法必需的输入结构或内部数值非法，不能冒充探索完成。

`WAITING_FOR_INPUT` 用于尚未收到全局图、位姿或可用 TF。缺输入只等待，不检查已收到消息的年龄，也不能报告完成。

状态机由构造参数独占 `maximum_replans`。每个 public 事件有唯一冻结的合法源状态集合，且下表是
唯一 authority；非法调用抛 `std::logic_error` 且 task、reason、ActiveGoal 和计数全部不变。
构造后公开初值固定为 `IDLE`、空 task id、空 reason、无 ActiveGoal。
`Start(nonempty_task_id)` 可从任意状态原子替换任务并进入 `WAITING_FOR_INPUT`；空 task id 拒绝。
`Cancel()` 可从任意状态幂等回到 `IDLE` 并清空任务/目标；`Fail(nonempty_reason)` 可从任意状态
进入 `ERROR`、清除 ActiveGoal 但保留现有 task id；`COMPLETED` 同样保留 task id。只读
`task_id()` 使替换、保留和异常强保证可测试。其余合法矩阵固定为：

| 事件 | 合法源状态 | 目标状态与效果 |
|---|---|---|
| `WaitForInput` | `WAITING_FOR_INPUT`、`SELECTING_FRONTIER` | `WAITING_FOR_INPUT` |
| `BeginSelection` | `WAITING_FOR_INPUT` | `SELECTING_FRONTIER` |
| `BeginPlanning` | `SELECTING_FRONTIER` | `PLANNING` |
| `CommitGoal(ActiveGoal&&)` | `PLANNING` | 保存 factory 的 move-only ActiveGoal，进入 `EXECUTING` |
| `BeginReplanning(rolling/stuck)` | `EXECUTING` 且有目标 | 按 typed cause 进入 `REPLANNING`，或卡滞耗尽后进入 `SELECTING_FRONTIER` |
| `ResumeExecution` | `REPLANNING` 且有目标 | 保留目标与计数，进入 `EXECUTING` |
| `ReleaseGoal` | `EXECUTING`/`REPLANNING` 且有目标 | 仅接受 arrived/no-path/candidate-invalid/frontier-disappeared/information-gain-zero，清除目标并进入 `SELECTING_FRONTIER` |
| `Pause` | `WAITING_FOR_INPUT`、`SELECTING_FRONTIER`、`PLANNING`、`EXECUTING`、`REPLANNING` | 清除活动目标并进入 `PAUSED` |
| `Resume` | `PAUSED` | 进入 `WAITING_FOR_INPUT`，由最新输入重新选点 |
| `CompleteNoReachableFrontier` | `SELECTING_FRONTIER` 且无目标 | `COMPLETED`，reason=`COMPLETED_NO_REACHABLE_FRONTIER` |

普通地图更新和更高分候选不得替换已承诺目标；`MayReplaceGoalForMapUpdate()` 仅在不存在 ActiveGoal
时返回 true。稳定 reason code 固定使用 `STARTED`、`WAITING_FOR_INPUT`、`SELECTING_FRONTIER`、
`PLANNING`、`EXECUTING`、`ROLLING_SEGMENT`、`STUCK_RETRY`、`STUCK_RETRIES_EXHAUSTED`、
`ARRIVED`、`NO_PATH`、`CANDIDATE_INVALID`、`FRONTIER_DISAPPEARED`、
`INFORMATION_GAIN_ZERO`、`PAUSED`、`RESUMED`、`CANCELED` 与
`COMPLETED_NO_REACHABLE_FRONTIER`；`Fail` 保留调用方给定的非空错误码。
在合法源状态与 ActiveGoal 前置条件成立时，伪造的 `ReplanCause` 或 `GoalReleaseReason` 底层值抛
`std::invalid_argument` 且不改变任何状态；若源状态本身非法，则先按矩阵抛 `std::logic_error`。
空 task id/Fail reason 也抛 `std::invalid_argument` 并保持强保证。

## 15. 完成语义

唯一正常完成原因：

```text
COMPLETED_NO_REACHABLE_FRONTIER
```

进入该状态必须同时证明：

1. 已拥有任务、全局地图和 `map` 位姿；
2. 当前任务逻辑栅格已建立；
3. 当前所有前沿簇已完成候选生成；
4. 所有足迹安全候选已完成信息增益计算；
5. 所有正增益候选已经获得 typed `kExhaustiveNoPath`（且 reason 只能为 `NO_PATH` 或
   `GOAL_OUTSIDE_LOCAL_MAP`），或已由当前有效失败记录抑制；
6. 没有任何候选拥有有效路径。

覆盖率不是完成条件。障碍包围、平台过宽、全部候选均获得上述两种结论性穷尽证据或任务区域部分
位于地图外，都可能以较低覆盖率正常结束；TIMEOUT、客户端 transport deadline、取消、
服务/传输故障、合同错误或非法规划结果不能推动
正常完成。状态必须同时报告剩余未知、地图外面积和失败候选数量。

## 16. 覆盖与任务统计

任务逻辑栅格面积：

```text
task_raster_area_m2 = task_cell_count * resolution_m^2
known_area_m2 = (free_cell_count + occupied_cell_count) * resolution_m^2
unknown_area_m2 = unknown_cell_count * resolution_m^2
outside_map_area_m2 = outside_map_cell_count * resolution_m^2
coverage_ratio = known_area_m2 / task_raster_area_m2
```

`CalculateCoverage` 只遍历 `TaskRaster::task_cells()`；`FREE/OCCUPIED/UNKNOWN/OUTSIDE_MAP` 分别
计数，若该 span 中出现 `OUTSIDE_TASK` 则是内部不变量破坏并抛 `std::logic_error`。polygon area
原样取 TaskRaster 已验证的 shoelace area；raster area 和 coverage 分母只用 task-cell count，禁止
用连续 polygon area 代替。

cell area 和各 count×cell-area 先用 long double 检查；最终 double 必须有限，数学上非零的正面积
若下溢成 0 或超出 double 均抛 `std::overflow_error`，不得返回 Inf/NaN/伪零。合法非零 polygon
可能没有任何 cell center：此时 task raster area、四类 raster area 和 coverage ratio 都精确为
`0.0`。非空时 ratio 以整数计数 `double(free+occupied)/double(task_cell_count)` 计算并检查位于
`[0,1]`。`UNKNOWN` 和 `OUTSIDE_MAP` 都不计入已知覆盖。

`CoverageStats` 只承载统计，禁止增加 completed/success/terminal/threshold 字段；核心和 ROS 层都
不得将 coverage ratio 与任何阈值比较。coverage 为 1.0 不触发完成，coverage 低于 1.0 也可在所有
可达前沿已按第15节穷尽后完成。

为使 `OUTSIDE_TASK` defensive branch 和 positive-area-to-double-zero 分支可直接构造 RED，Task 7 在
`src/detail/coverage_statistics.hpp` 提供非安装、非导出、仅核心实现与白盒测试可见的 private seam：

```cpp
namespace lunar::pure_exploration::detail {
struct CoverageCounts {
  std::size_t free;
  std::size_t occupied;
  std::size_t unknown;
  std::size_t outside_map;
};
inline void AccumulateCoverageState(CellState state, CoverageCounts& counts);
inline double CheckedCoverageArea(std::size_t cell_count, double resolution_m);
inline CoverageStats FinalizeCoverage(double polygon_area_m2,
                                      double resolution_m,
                                      const CoverageCounts& counts);
}
```

该头不属于稳定 API，不能安装/export，也不能被 ROS/Task 8+ include。生产 `CalculateCoverage` 流式
遍历 `task_cells()`，每格调用共享 `AccumulateCoverageState`，最后调用 `FinalizeCoverage`；不得另写
状态 switch 或面积逻辑。白盒测试直接向 accumulator 注入 `kOutsideTask` 并断言 `logic_error`。
`CheckedCoverageArea` 固定先以 long double 计算
`area_ld=long double(count)*long double(resolution_m)*long double(resolution_m)`；count 非零时要求
`area_ld>0`，且转换到 double 后必须有限并大于零，否则抛 `std::overflow_error`。因此测试可直接以
正 double resolution 构造 `area_ld>0` 但 double cast 为 0 的 underflow RED，而无需伪造 TaskRaster。
`test_task7_implementation_contract.py` 必须抽取真实 `CalculateCoverage` 与上述 inline seam body，验证
生产调用链，并以删除 shared call、把 `OUTSIDE_TASK` 改为普通计数、删除 positive-area cast-to-zero
检查三类 mutation 证明测试会失败；不得通过扩展 TaskRaster 或 coverage 稳定公共 API 获得入口。

`PureExplorationStatus.msg` 至少包含：

```text
uint8 IDLE=0
uint8 WAITING_FOR_INPUT=1
uint8 SELECTING_FRONTIER=2
uint8 PLANNING=3
uint8 EXECUTING=4
uint8 REPLANNING=5
uint8 PAUSED=6
uint8 COMPLETED=7
uint8 ERROR=8

std_msgs/Header header
string task_id
uint8 state
string reason_code
float64 polygon_area_m2
float64 task_raster_area_m2
float64 known_free_area_m2
float64 known_occupied_area_m2
float64 unknown_area_m2
float64 outside_map_area_m2
float64 coverage_ratio
uint32 frontier_cluster_count
uint32 candidate_count
uint32 reachable_candidate_count
uint32 failed_candidate_count
uint32 completed_goal_count
uint32 replan_count
string current_plan_id
geometry_msgs/Pose current_goal
float64 active_elapsed_s
```

发布状态使用节点当前时间填充 header 仅供观察；订阅者不得把它解释为探索准入证据。

## 17. 计时与诊断

探索节点使用车载单调时钟记录：

- 前沿检测调用次数、最后一次和累计耗时；
- 信息增益调用次数、最后一次和累计耗时；
- 候选粗排与最终评分耗时；
- 一次决策从快照到选出最终目标的总耗时；
- 规划候选调用数量和重规划数量；
- 当前任务活动时间，暂停时间不计入。

纯规划器按配套设计在 `/Car/T4/planning/diagnostics` 发布：

```text
request_id
platform_type
environment_mode
planning_outcome
reason_code
global_elapsed_ms
global_call_count
local_elapsed_ms
local_call_count
total_elapsed_ms
```

探索节点按 `request_id` 关联并累计：

- 全局规划调用次数、最后一次和总耗时；
- 局部规划调用次数、最后一次和总耗时；
- 总规划耗时；
- 候选验证规划耗时与执行中重规划耗时。

`PlannerTimingAccumulator` 构造时要求正 capacity，并复用部署必填的
`maximum_candidate_views`。每次 `Ingest` 只原子接收一个 `DiagnosticStatus`，其 values 必须恰含上述
十个唯一 key，不得有零/多 status、未知/缺失/重复 key。`request_id/reason_code` 必须非空，
`platform_type` 精确为 `WHEELED/LEGGED/HOPPER`，`environment_mode` 精确为十进制 `1/2`，
`planning_outcome` 是一个已定义 PlanMotion outcome；environment/outcome/count 必须是无空白、无符号、
无尾随字符且范围可表示的完整十进制，
三个 elapsed 必须是同样 full-string 严格解析的 finite nonnegative 十进制；uint64 overflow、NaN/Inf、
负值或尾随字符全部拒绝。`count>0,elapsed=0` 合法，因为 steady-clock 分辨率可以得到零时长。
header/status stamp 完全忽略。

```cpp
enum class TimingIngestResult : std::uint8_t {
  kAccepted,
  kDuplicate,
  kRejected,
};
// PlannerTimingAccumulator::Ingest returns TimingIngestResult.
```

首条合法 request ID 在记录 resident 期间胜出并返回 `kAccepted`；同 ID 后续合法重复或冲突返回
`kDuplicate`，不得改值或刷新淘汰年龄。新的唯一合法 ID 按到达顺序加入 FIFO，超过 capacity 时淘汰
最早唯一记录。capacity=1 时，`A1 kAccepted -> B kAccepted/evict A -> A2 kAccepted/evict B`，随后
`Find(A)==A2` 且 `Find(B)==nullopt`；淘汰后 A 再摄入视为 new unique，缓存不得保留 tombstone 或无界
seen-ID 集合。
任何非法 message 返回 `kRejected` 且 records/FIFO 顺序完全不变。diagnostics 可晚于 Action Result
到达并按 request ID 查询。没有 diagnostics 或整条被拒绝时 timing 作为整体 unavailable；不得从
`total_elapsed_ms` 合成 global/local，也不得用 stamp 排序或判 freshness。

关联只用于统计。Action Result 在完全没有 diagnostics 时仍保持原 typed kind；diagnostics 迟到、
重复、淘汰或合同事件都不能调用、否定或改变 planning completion。

## 18. 参数默认值

```yaml
global_occupied_threshold: 50
maximum_task_raster_cells: 1048576
sensor_range_m: 10.0
sensor_fov_deg: 90.0
yaw_offsets_deg: [-45.0, -22.5, 0.0, 22.5, 45.0]
candidates_per_planning_batch: 16
score_weights:
  information_gain: 0.60
  global_path_length: 0.30
  heading_change: 0.05
  revisit: 0.05
stuck_timeout_s: 30.0
minimum_progress_m: 0.2
maximum_replans_per_candidate: 2
goal_yaw_tolerance_deg: 11.25
planner_result_timeout_s: 2.0
```

平台尺寸和净空不得在探索配置中重复填写，必须来自规划器拥有的同一份平台配置；规划及到达
位置容差由当前全局图分辨率和平台宽度按第7节公式派生，不增加第二个配置值。

## 19. 错误处理边界

算法只拒绝无法进行数学计算的输入：

- 空任务 ID 或非法命令；
- START 多边形少于三个有效顶点、自交、零面积或非有限；
- 地图 resolution 非正、尺寸与 data 长度不匹配或 origin 非有限；
- Odometry 位置或四元数非有限；
- 无法形成 `map` 位姿；
- 平台足迹、净空或评分权重非法；
- 九个候选/gain/preview/executable/failure 必填资源预算缺失或非正，或者工作量超过预算；
- 规划 Result wrapper/typed payload/union invariant 矛盾，成功缺少有效参考/路径，或 preview/executable
  sequence 超过已批准资源上限；
- planner client transport timeout 配置非有限/非正，或 diagnostics 结构/严格数值合同非法；后者只
  拒绝该统计记录，不改变 Action 结果分类。

以下事实不得拒绝算法：

- 时间戳为零、旧值或彼此不一致；
- 协方差很大或未提供有效估计；
- 没有地图 revision、LocalizationStatus 或执行反馈；
- 没有 semantic、观测年龄、质量、计数或 variance 层；
- 覆盖率低；
- 部分任务区域超出当前地图。

## 20. 验证策略

### 20.1 ROS 无关核心单元测试

- 凸、凹和边界对齐多边形栅格化；
- 地图范围外任务栅格统计；
- `-1/0..100` 分类和阈值参数；
- WFD 只从机器人可连通自由区搜索；
- 四邻域前沿判定和八邻域聚类；
- 米制最小前沿长度在不同 resolution 下等价；
- 旋转足迹、净空和候选退距；
- `OccupancyGridView` 唯一实现、`TaskRaster` 委托的旋转 world/grid 变换，以及闭 cell 方块与
  Minkowski 净空相切；
- canonical 接口边 25/50/75% 代表、位置组间距、同 XY 全部安全 yaw、量化越界和 factory
  深拷贝完整承诺身份；
- 窄前沿不能容纳平台时被过滤；
- 10 m、90 度无扩张闭边界、穿角/沿线 supercover、同 t 优先级、凹任务接触终止和障碍遮挡；
- 信息增益以平方米表达，跨分辨率使用对齐 fixture 或各自黄金计数，不使用通用一格误差断言；
- position/view/collision/visibility/preview/executable/failure 九类资源超限 fail closed，ROS 为
  ERROR，不产生完成证据；
- zero gain 在 Task 7 过滤，粗排加入最小 heading change，归一化、最终评分和 tie-break 确定；
- 失败候选只在附近地图分类改变后恢复；
- 覆盖率只统计，不参与终止。

### 20.2 状态机与规划替身测试

- START、PAUSE、RESUME、CANCEL；
- 无地图、无位姿和无 TF 时等待；
- 目标承诺，不因普通地图更新或更高分候选切换；
- 目标变障碍、前沿消失和增益归零时取消；
- 每批 16 个 typed `kExhaustiveNoPath`（只允许 `NO_PATH` 或 `GOAL_OUTSIDE_LOCAL_MAP`）后继续下一批；
- wrapper/payload 全矩阵逐 outcome/directive/reason/unknown/contradiction 验证；typed TIMEOUT、client
  2 s deadline、取消和 Action/传输错误不计入完成证据，合同错误进入 ERROR；
- GoalRegion 精确 map/goal-id/POINT/XY/+0.0 z/yaw/tolerance，且所有 stamp 变体不改变分类；
- empty/one-point/finite XY preview、long-double 累计、负零规范、preview/executable exact/over 资源边界；
- cancel-before-goal-response、重复 cancel、reject/cancel 交错、cancel accepted/rejected、result-before-
  cancel-response、cancel-no-result/watchdog 和 unsolicited canceled 的完整 single-winner 矩阵；
- callbacks 只 weak-own transport state，terminal claim 在锁内先清 active state、锁外调用 completion；
  reentrant Evaluate、throwing completion、析构和全部 late callback 安全；Feedback 不消费；
- diagnostics 精确十个唯一 key/严格解析，typed ingest result、resident FIFO first-valid-wins、duplicate
  不刷新、evict/reingest 无 tombstone、late ingest、无诊断不改 Result、无 total-to-global/local 合成且
  所有 stamp ignored；
- owning FrozenCandidateBatch 在异步结果/取消终态前保持存活，request_id 是唯一 callback 回关联键；
- 地图 callback 只替换 latest cache；in-flight batch 不变，内容变化使旧 batch 的完成结论作废；
- 未穷尽候选不得完成；
- 最终选中参考是唯一发布给控制器的参考；到达局部段末端后对同一最终候选滚动重规划；
- 30 s/0.2 m 单调时钟进度判定；
- 两次重规划后屏蔽候选；
- 无可达前沿时以正常原因完成；
- 零时间戳、高协方差和缺少 revision 不阻塞。

### 20.3 合成地图端到端测试

- 单房间；
- 长走廊；
- 多房间与回廊；
- 凹多边形任务区；
- 障碍完全包围的未知区；
- 多个近前沿与一个高增益远前沿；
- 部分任务区域在全局图外，随后地图扩大；
- 全部前沿候选不可达；
- `1.0 m` 全局 preview 与 `0.2 m` 局部可执行段边界，至少两个局部段到达最终候选；
- `/Car/T4/planning/diagnostics` 的十键关联及候选、滚动、卡住重规划分桶计时。

### 20.4 环境与证据边界

权威构建环境为 Ubuntu 22.04 amd64 + ROS 2 Humble。实现完成至少验证：

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
colcon build --base-paths pure_planner/ros2_ws/src ros2_ws/src/lunar_planning_msgs \
  --packages-up-to lunar_pure_exploration_ros
colcon test --packages-select \
  lunar_pure_planner_core \
  lunar_pure_planner_ros \
  lunar_pure_exploration_msgs \
  lunar_pure_exploration_core \
  lunar_pure_exploration_ros
```

容器构建和合成地图测试只能证明算法与 ROS 闭环可运行。实车部署、控制器执行、在线 Task3 地图扩展和完整任务完成必须在车辆上独立验证，不能由离线测试替代。

## 21. 实施顺序

统一准入门为：Task 1--9 可基于已验证的规划 ROS 预集成推进，且用户已授权 Task 9 并行；Task
10--16 必须等待 `bf1bb79` 或合同等价后继进入当前分支并重新通过最终 planner gate。

1. 验证固定规划集成 SHA 的 Action、十键诊断和平台配置单一来源；
2. 建立精简探索消息；
3. 以测试驱动实现 ROS 无关任务栅格和前沿算法；
4. 实现平台/分辨率感知的候选和信息增益；
5. 在 `bf1bb79` 或合同等价后继进入当前分支并重新通过最终 planner gate 后，实现规划 Action 客户端、
   评分与目标承诺；fake-only 不算完成；
6. 实现分段参考、双取消面、滚动重规划和单调时钟进度监控；
7. 实现覆盖、计时、状态和 RViz 可视化；
8. 以真实规划 server 加 fake 场景完成 ROS Humble 集成验证；
9. 在车辆上接入真实 `/Car/T3/...` 与 `/Car/T4/...` 完成在线验收。

## 22. 完成定义

实现只有同时满足以下条件才可声明“纯探索算法容器验证完成”：

- 新旧源码、包名、配置和启动入口物理隔离；
- 任务多边形和地图编码按本文工作，平台几何来自被测规划集成 SHA 的唯一配置；
- 候选不会进入未知或无法容纳完整车体的位置；
- 10 m/90 度增益、真实路径代价和确定性评分通过测试；
- 目标承诺、卡住恢复、失败解除和完整候选穷尽通过测试；
- 覆盖率只统计且无可达前沿是唯一正常结束条件；
- 最终参考和取消接口闭合；
- 异步 planner callback weak-own transport state，只通过 request_id 回到存活的 owning batch，不跨
  snapshot 使用 CandidateView；single-winner completion 支持 reentrant Evaluate 且 late callback no-op；
- 全局、局部和探索算法计时分别可见；
- ROS Humble 构建与测试通过。

只有真实车辆完成至少一个多前沿任务，且 Topic、规划、控制、位姿进度、地图更新和最终结束原因全部符合本文，才可进一步声明“在线探索端到端完成”。
