# 高分辨率可通行地图规划器 V1 设计

## 状态

本设计于 2026-08-24 在对话中确认。它定义课题三地图输入、高分辨率可通行地图维护、统一的全局/局部二维规划、轻量轨迹生成以及诊断验收边界。

本文档本身不改变生产代码。V1 先以显式配置启用并完成离线、ROS 接口与性能诊断验收；在取得目标 Jetson Orin 和实车证据前，不替换现有轮式规划器的生产默认值。腿式和跳跃规划不在本设计范围内。

## 决策

项目边界接收课题三提供的原始全局地图、随车滚动局部地图、里程计和坐标变换。规划算法不直接分别解释这些原始地图，而是统一读取由地图维护器生成的不可变高分辨率可通行地图快照。

```text
/Car/T3/mapping/global_overview ──────┐
/Car/T3/mapping/grid_map ─────────────┼─> 高分辨率可通行地图维护器
/Car/T3/localization/odometry + /tf ──┘              │
                                                     ▼
                                      TraversabilitySnapshot N
                                        ├─ 全局二维搜索
                                        ├─ 局部滚动二维搜索
                                        └─ 路径简化后复检
                                                     │
                                                     ▼
                           MotionReference + Path + TimedPath + diagnostics
```

全局规划、局部规划和最终路径复检必须使用同一个 `traversability_revision` 和同一套 `FREE/BLOCKED/UNKNOWN` 语义。车辆能力假设为：只要中心轨迹完全位于已经按车辆包络膨胀后的 `FREE` 区域，车辆就能够执行。因此，规划层不再重复完整车体扫掠、运动基元、支撑平面、底盘净空、曲率可行性和逐姿态地形认证。

## 与现有设计和实现的关系

当前实现分别维护：

- 低分辨率全局 `OccupancyGrid` 投影；
- 当前滚动窗口内的局部 `GridMap`/地形表示；
- 只覆盖当前局部几何范围的 `/Car/T4/planning/local_traversability`。

现有 `IncrementalTraversability` 在局部地图几何变化时整图重建，不会将车辆驶过区域拼接成持久的 `map` 坐标系高分辨率地图。全局和局部规划因而可能基于不同分辨率、不同膨胀边界和不同连通性结论。

本文档为轮式 V1 引入新的统一地图与简化规划路径。它不修改共享 ARA*、腿式或跳跃路径。`docs/superpowers/specs/2026-08-24-planner-core-incremental-optimization-design.md` 中已实现的复杂轮式 Hybrid/认证路径在 V1 诊断阶段保留为对照模式；本设计只在显式选择 `grid_traversability_v1` 时取代该请求的轮式局部搜索和重复认证。不得自动回退旧规划器并把回退结果计为 V1 成功。

## 项目输入契约

| 输入 | ROS 类型 | 坐标系 | V1 用途 |
| --- | --- | --- | --- |
| `/Car/T3/mapping/global_overview` | `nav_msgs/msg/OccupancyGrid` | `map` | 全局边界和未近距离观测区域的粗分辨率可通行先验 |
| `/Car/T3/mapping/grid_map` | `grid_map_msgs/msg/GridMap` | `odom` | 随车更新的高分辨率 `occupancy`、`elevation` 观测 |
| `/Car/T3/localization/odometry` | `nav_msgs/msg/Odometry` | `odom -> base_link` | 当前车辆位姿、速度、局部窗口起点和执行进度 |
| `/tf` | `tf2_msgs/msg/TFMessage` | `map -> odom` | 将局部地图和车辆状态转换到统一 `map` 坐标系 |
| `/Car/T4/plan_motion` | `lunar_planning_msgs/action/PlanMotion` | 由环境模式和 Goal header 决定 | 目标点/区域、位置容差和可选目标航向 |

V1 不新增要求课题三发布“已经生成好的可通行地图”的外部接口。可通行地图是项目内部派生状态。`/Car/T4/planning/local_traversability` 可继续用于 RViz 和诊断，但不是规划算法的权威订阅输入。

## 高分辨率地图分辨率契约

1. 第一个通过接口校验的 T3 局部 `GridMap.info.resolution` 冻结本次进程/任务的 `canonical_resolution_m`。
2. 该分辨率必须有限且大于零；全局搜索、局部搜索、路径简化复检和诊断统计均使用原始分辨率，不降采样。
3. 后续局部地图分辨率必须在 `max(1e-9 m, canonical_resolution_m * 1e-6)` 容差内相同。
4. 超出容差的变化返回 `MAP_RESOLUTION_MISMATCH`，不修改已有地图，也不发布新轨迹。V1 不在运行中隐式重采样或清空地图。
5. 高分辨率栅格轴与 `map` 坐标轴对齐，原点锚定到全局 `OccupancyGrid.info.origin`。全局图无效或带非零平面旋转时返回 `GLOBAL_MAP_GEOMETRY_INVALID`。
6. 全局图的分辨率可以不同于局部图。映射按世界坐标几何关系完成，不要求两者分辨率为整数倍。

## 地图存储架构

### 粗全局先验

全局 `OccupancyGrid` 保持原始分辨率存储，不物理展开为巨大的高分辨率稠密数组。高分辨率单元查询落到尚无局部覆盖的区域时，通过世界坐标找到对应全局单元：

- `data == -1`：`UNKNOWN`；
- `data >= global_occupancy_threshold_percent`：`BLOCKED`；
- 其余合法值：`FREE` 初始先验。

用户已明确允许全局 `FREE` 作为远距离、未近距离观测区域的初始可通行先验。局部高分辨率已知观测随后覆盖和修正该先验。

全局 `BLOCKED` 和 `UNKNOWN` 单元按其完整世界坐标矩形面积参与膨胀，不按粗栅格格心距离近似。这样高分辨率访问器在粗全局障碍边界和局部覆盖边界上保持同一车辆包络语义。

### 高分辨率局部覆盖层

高分辨率状态使用稀疏、固定 `256 x 256` 单元 tile 存储。只为实际收到局部观测或规划访问后需要缓存派生结果的区域分配 tile。全局图范围外没有局部观测的单元为 `UNKNOWN`。

每个已分配的高分辨率局部覆盖单元至少保存：

- 最新有效 occupancy；
- 最新有效 elevation；
- 数据来源 `LOCAL_OBSERVATION`；
- 最后一次本地观测序号；
- 派生可通行状态；
- 派生状态所使用的 profile/revision。

查询落入未分配覆盖区域时，访问器直接从粗全局先验计算结果并报告来源 `GLOBAL_PRIOR`，不为该查询强制分配高分辨率原始单元。

地图存储不得为整张高分辨率范围预分配搜索节点、父节点、代价或访问标志。规划搜索同样使用按发现分配的稀疏状态。

## 融合与可通行性生成

### 融合规则

每帧有效局部 `GridMap` 通过同一帧可用的 `map <- odom` 变换投影到高分辨率全局格。V1 接受有限的平面刚体变换，并使用确定性的保守栅格化：已知障碍覆盖所有与源单元世界矩形相交的 canonical 单元；已知自由观测只更新格心被源单元覆盖的 canonical 单元；UNKNOWN 不写入覆盖层。这样非整数原点偏移或非零 `map <- odom` yaw 不会在障碍边界留下自由缝隙。

融合顺序为：

1. 局部有限、合法的 occupancy/elevation 观测覆盖同一位置的粗全局先验和旧局部观测。
2. 新的已知 `FREE` 可以清除旧的局部 `BLOCKED`，新的已知 `BLOCKED` 可以阻断旧的 `FREE`；静态地形下以最新有效高分辨率观测修正旧证据。
3. 新的 `UNKNOWN` 或非法单元不擦除历史已知局部单元。
4. 相同内容的重复帧不增加 `traversability_revision`。
5. 全局图更新替换粗先验，但不删除仍有效的局部高分辨率覆盖。
6. 全局先验与局部观测冲突按局部观测取值，并计入诊断。

V1 先融合原始语义，再统一生成膨胀后的可通行状态。不得把已经膨胀的滚动窗口直接拼接，否则会在窗口重叠和边界处重复膨胀。

### 唯一可通行性策略

V1 地图维护器是轮式安全语义的唯一生产者。首版复用当前轻量分类策略：

- occupancy/elevation 非有限或超出合法范围：`UNKNOWN`；
- occupancy 大于等于 `local_occupancy_threshold`：原始障碍；
- 中心坡度大于轮式能力配置上限：原始不可通行；
- 原始障碍和未知区域按“轮式包络外接半径 + minimum clearance”膨胀；
- 膨胀后安全且中心地形合法：`FREE=1.0`；
- 膨胀后被阻断：`BLOCKED=0.0`；
- 证据不足：`UNKNOWN=NaN`。

地图维护器可以复用线性距离变换和 dirty halo 机制。每次原始单元变化只重算所在 tile 以及由膨胀半径决定的相邻 halo。tile 边界查询必须读取相邻 tile 或粗全局先验，不能把 tile 边界自动当作自由空间。

## 不可变地图快照

每次发生语义变化后生成单调递增的 `traversability_revision`。规划请求捕获一个不可变 `TraversabilitySnapshot`，至少包含：

- canonical resolution、原点和可查询范围；
- 粗全局先验 revision；
- 局部覆盖 revision；
- 可通行性 profile hash；
- tile 索引和只读单元访问器；
- 本 revision 的 changed tile 集合。

同一规划周期内，全局搜索、局部搜索、捷径简化和输出复检不得切换 snapshot。发布前对最终路径的 supercover 单元使用最新 snapshot 做一次廉价复检：如果新 revision 将任一路径单元变为非 `FREE`，丢弃结果并触发新周期；否则可以发布，并同时记录 planning revision 和 publish-check revision。

## 规划算法 V1

### 全局搜索

- 在统一高分辨率快照上执行 8 邻域二维 A*。
- 只扩展 `FREE` 单元，并禁止对角穿越两个相邻阻挡角。
- 使用欧氏距离启发，返回第一条完整路径，不进行 anytime 最优性细化。
- OPEN、节点、父节点和 CLOSED 按发现稀疏分配，不按整张高分辨率地图预分配。
- tile 的 FREE/BLOCKED/UNKNOWN 汇总仅可作为连通性剪枝、搜索走廊和启发加速索引；最终路径的每个单元仍来自 canonical resolution 搜索与复检。
- 全局路线按目标、地图 profile 和相关 revision 缓存。新局部观测未触及当前路线及一个膨胀单元 halo 时，可以复用全局路线；相交时重新运行完整全局 A*，V1 不实现 D* Lite/LPA*。

### 局部滚动搜索

- 默认使用可配置的 8 m 路线前瞻窗口。
- 在同一 snapshot 的局部视图中执行 8 邻域二维 A*。
- 沿全局路线从远到近生成最多 4 个局部接入候选；逐个尝试并记录尝试次数，首个可达的最大进度候选胜出。
- 新的局部地图 revision、接近当前局部参考末端或偏离路线超过配置阈值时触发局部重规划。
- 不搜索 yaw、曲率、运动模式或轮式运动基元。

### 方向、路径简化和速度

1. 对原始栅格折线执行 supercover 视线捷径；任何捷径经过非 `FREE` 单元时拒绝该捷径。
2. 捷径处理异常时回退到原始 A* 折线，不把后处理失败误报为 `NO_PATH`。
3. 按固定空间间距重采样，并用切线确定姿态；保留锐角时插入同位置原地转向点。
4. V1 只比较整段前进与整段倒车。选择初始转向代价、方向保持惩罚和行驶时间代价之和较小者；不允许一条局部段内多次前后换向。
5. 平移段使用受能力配置限制的固定名义速度，倒车速度沿车体朝向投影为负；原地转向段线速度为零并使用受限固定角速度。
6. 起点和终点速度为零。`time_from_start` 根据段长/速度和转角/角速度严格递增生成。
7. V1 不做样条、弹性带、曲率连续、加速度/加加速度最优或时间最优参数化。

## 项目输出契约

| 输出 | ROS 类型 | 语义 |
| --- | --- | --- |
| `/Car/T4/planning/wheeled_reference` | `lunar_planning_msgs/msg/MotionReference` | 控制器权威输入；包含轨迹位姿、时间、正负线速度和角速度 |
| `/Car/T4/planning/wheeled_path` | `nav_msgs/msg/Path` | ROS 标准几何路径，用于 RViz、rosbag 和通用观察者 |
| `/Car/T4/planning/wheeled_path_timing` | `lunar_planning_msgs/msg/TimedPath` | 同一 `Path` 加 `builtin_interfaces/msg/Duration planning_time` |
| `/Car/T4/planning/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | 地图维护、规划阶段、结果和失败原因 |

`MotionReference.path_preview`、独立 `Path` 和 `TimedPath.path` 必须使用相同 `map` frame、header stamp 和位姿序列。`planning_time` 表示从读取规划快照到本次输出完成的总规划计算时间，不是车辆执行轨迹的持续时间。

成功时三个路径/轨迹输出均非空。失败时不得发布新的可执行 `MotionReference`；`Path` 和 `TimedPath.path` 为空，但 `TimedPath.planning_time` 保留失败耗时。控制器根据 `MotionReference + 最新 T3 odometry` 发布 `/Car/T5/Car_Cmd_Vel`，T5 `Twist` 不是规划器核心直接输出。

## 诊断契约

每个规划周期只产生一条可关联的最终诊断摘要，至少包含：

### 输入与地图

- request/cycle ID；
- global、local、odometry 输入序号；
- planning `traversability_revision` 和 publish-check revision；
- profile hash、canonical resolution、地图原点；
- 已分配 tile 数和估算地图内存；
- 本帧更新单元、dirty tile、halo 重算单元；
- FREE/BLOCKED/UNKNOWN 统计；
- 全局先验与局部观测冲突数；
- 地图融合和可通行性重算耗时。

### 搜索与输出

- 全局路线是否复用；
- 全局/局部调用次数、展开状态、OPEN 峰值和耗时；
- 局部候选数量、尝试次数和选中进度；
- 原始、捷径、重采样和最终轨迹点数；
- 前进/倒车选择及两种代价；
- 最终 supercover 检查单元数；
- 后处理模式 `SHORTCUT` 或 `RAW_GRID_FALLBACK`；
- `planning_outcome`、`has_reference`、`reason_code` 和总耗时。

V1 原因码至少区分：

- `PLAN_FOUND`；
- `INVALID_INPUT`；
- `GLOBAL_MAP_GEOMETRY_INVALID`；
- `MAP_RESOLUTION_MISMATCH`；
- `START_NOT_FREE`；
- `GOAL_NOT_FREE`；
- `GLOBAL_NO_PATH`；
- `LOCAL_NO_CANDIDATE`；
- `LOCAL_NO_PATH`；
- `POSTCHECK_FAILED`；
- `STALE_PATH_INVALIDATED`；
- `TIMEOUT`；
- `REQUEST_CANCELED`；
- `PLANNER_ERROR`。

## 错误处理

- 非法 frame、尺寸、分辨率、数组布局、变换或非有限车辆状态不改变持久地图，并返回明确原因。
- `UNKNOWN` 不可被搜索或路径简化穿越。
- 地图更新、取消和请求替换优先于尚未发布的旧结果。
- 单条捷径失败只拒绝该捷径；完整后处理异常回退原始栅格路径。
- 最新地图复检失败时不发布旧结果，触发下一次滚动周期。
- V1 失败不得自动调用旧规划器并返回其轨迹，否则诊断无法证明 V1 的成功率。

## V1 诊断验收

### 地图维护硬门槛

1. 使用一个固定静态真值地图生成全局粗图和至少 5 个随车辆移动的局部窗口；拼接结果在所有已知高分辨率单元上与离线参考逐单元一致。
2. 全局 `FREE` 在局部尚未覆盖时可通过高分辨率访问器查询为 `FREE`；局部已知观测到达后按局部结果覆盖。
3. 局部 `UNKNOWN` 不擦除历史已知，局部新的已知 FREE/BLOCKED 能修正旧值。
4. 车辆驶离区域后已知单元仍保留，除非收到新的有效已知观测覆盖。
5. tile 边界、局部窗口重叠和全局/局部来源边界不存在膨胀缝隙或重复膨胀。
6. 相同输入帧不增加 revision；任一语义单元变化恰好生成一个新 revision。
7. 原始局部分辨率从输入到规划与输出复检保持不变；分辨率变化产生 `MAP_RESOLUTION_MISMATCH`。

### 规划硬门槛

1. 全局、局部、简化和复检诊断报告同一 planning revision。
2. 连通 FREE 场景产生路径；BLOCKED/UNKNOWN 断路场景产生阶段明确的失败原因。
3. 禁止对角穿角，最终每个线段的 supercover 单元均为 FREE。
4. 新观测揭示全局先验中的障碍后，当前路径相交时必须重规划或返回 `STALE_PATH_INVALIDATED`。
5. 正向场景产生正向规划速度；背向目标场景能够选择整段倒车并产生负向规划速度。
6. 后处理故障场景回退原始 A* 折线且仍能成功发布。
7. 相同输入、地图和配置重复运行产生相同路径、方向、原因码和非时间计数。

### ROS 输出硬门槛

1. 正式成功同时满足 `planning_outcome=0`、`has_reference=true` 和非空 `MotionReference.trajectory`。
2. 每个轨迹点恰好包含一个位姿和一个速度；所有数值有限，`time_from_start` 严格递增且速度不超能力配置。
3. `MotionReference.path_preview`、`Path` 和 `TimedPath.path` 的 frame、stamp 和位姿序列一致。
4. `TimedPath.planning_time >= 0`，并与 diagnostics `total_elapsed_ms` 在单位转换/输出收尾容差内一致。
5. 失败输出为空路径但保留规划耗时，且不得留下新的可执行参考。
6. 每次成功或失败的必需诊断字段完整率为 100%。

### 性能记录而非首版硬门槛

在固定 Release 构建、固定地图和固定硬件上运行至少 30 次，报告 map fusion、inflation、global、local、postprocess 和 total 的 p50/p95/max，以及展开节点、tile 数和峰值内存。V1 不预设 Jetson Orin 的绝对时延合格线；目标机实时门槛必须依据此基线另行确认。

历史全局或容器测试不能替代统一高分辨率地图 V1 的证明。Jetson Orin DDS、rosbag 连续滚动、控制器跟踪和实车通行能力在实际执行前均保持 `NOT_RUN`。

## 实施边界

V1 预计按以下独立单元实施，具体文件和测试顺序由后续实施计划确定：

1. 持久高分辨率原始地图与稀疏 tile 存储；
2. 全局先验/局部观测融合和 dirty-halo 可通行性更新；
3. 不可变 snapshot 与统一高分辨率访问器；
4. 全局/局部稀疏二维 A*；
5. 视线捷径、整段前后向选择和最小速度时序；
6. ROS 输出、诊断和完整验收场景。

## 非目标

- 不降采样，不把低分辨率路线直接当最终可执行路径。
- 不实现 D* Lite、LPA* 或跨请求增量搜索修复。
- 不实现 SE(2) 状态格点、运动基元或逐姿态车体扫掠。
- 不实现一条局部路径内的多次前进/倒车切换。
- 不实现贝塞尔、B 样条、弹性带、曲率连续或时间最优轨迹。
- 不修改纯跟踪控制器或让规划器直接发布 `/Car/T5/Car_Cmd_Vel`。
- 不删除旧轮式规划器，不改变腿式和跳跃规划器。
- 不把主机/容器诊断验收表述为目标 Orin 或实车就绪。

## 文档与迁移要求

实施时必须同步更新：

- `README.md` 的 T3 输入、统一可通行地图、输出和诊断说明；
- `config/external_interfaces.yaml` 的项目边界说明；
- planner 配置中的显式 V1 模式、地图分辨率锁定和诊断参数；
- `docs/validation/` 下的地图逐单元对比、规划结果和性能证据；
- 旧增量优化设计的关系说明，明确哪些轮式路径仅在 legacy 模式生效。
