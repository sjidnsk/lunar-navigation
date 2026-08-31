# 足式 Grid V1 分层路径规划设计

## 状态

本设计于 2026-08-31 在对话中确认，并在 RViz 计算效率复核后补充局部边认证优化。
后续连续滚动实验暴露出足式局部规划只消费第一个门户、使用地图原点量化状态、
仅以单目标欧氏距离引导并受固定 `131072` 状态上限约束的问题，因此本设计进一步
将足式局部搜索契约与现有轮式规划器对齐。它定义足式平台如何复用 Grid V1
持久化可通行性地图，保留现有全局 ARA*、足式动作原语、地形与机身边认证以及
`kLeggedBodyReference` 输出契约，同时支持完整局部目标集合、请求级状态格、足式
多目标距离场和动态搜索状态。

本文档本身不修改规划代码。后续实施必须先形成独立实施计划，并保持现有轮式
Grid V1 行为不变。

## 决策摘要

采用“Grid V1 地图 + 现有足式分层规划器”的方案：

```text
全局 occupancy ───────────────────┐
                                  │
局部 occupancy + elevation ───────┼─> 足式 TraversabilitySnapshot
                                  │           │
map <- odom TF ───────────────────┘           ▼
                                      全局可行性投影
                                               │
                                               ▼
                                      现有八邻域 ARA*
                                               │
                                               ▼
                                      前方局部目标集合
                                               │
                                               ▼
                                足式多目标距离场与请求级状态格
                                               │
                                               ▼
                                现有足式 lattice/anytime 搜索
                                               │
                                               ▼
                                    足式机身位姿轨迹
```

Grid V1 负责统一地图语义和全局路线约束；足式局部规划器继续负责坡度、台阶、
沟宽、机身高度和机身几何有效性，但不得在每个搜索状态的每条候选边上重复计算
逐格台阶邻域并执行密集完整机身扫掠。足式局部搜索复用轮式已经使用的完整
`LocalGoalSet`、目标距离场和动态状态编号语义，但不复制轮式运动学、足迹、样条
后处理或时间参数化。下游足式控制器继续负责步态和逐脚控制。

足式默认模式为：

```yaml
legged_global_mode: grid_traversability_v1
```

`legacy_occupancy` 仅保留为显式部署回退，不是默认值。Grid V1 失败时不得在同一
请求内静默调用 legacy 并把 legacy 结果报告为 Grid V1 成功。

## 与现有实现的关系

现有足式规划已经具备完整的分层结构：

1. `PlanSurfaceGlobal` 将全局 occupancy 生成 `GlobalOccupancyProjection`；
2. `SearchSurfaceGlobal` 在投影上执行八邻域 ARA*；
3. 分层规划器从全局路线选择局部目标；
4. `PlanLegged` 使用足式状态格、动作原语和局部高程生成机身轨迹；
5. `ComposeSurfaceReference` 组合全局 preview 与局部可执行 reference。

现有全局投影只解释全局 occupancy。其默认硬膨胀函数仅为轮式计算 footprint，
足式请求得到零硬膨胀；局部 elevation 也不会修正全局路线。因而当前全局路线
可能穿过局部足式规划器随后会拒绝的陡坡或狭窄区域。

本设计不开发另一套足式局部规划器，也不改变足式动作原语或 anytime 搜索框架。
全局阶段将可行性来源从 `GlobalOccupancyProjection` 替换为由足式 Grid V1 快照
生成的投影；局部阶段在现有 `PlanLegged` 内部增加足式专属地形投影和两级边
认证，并将局部目标接口、状态量化坐标系、搜索引导和状态资源模型对齐到现有轮式
搜索契约。轮式请求路由、配置、搜索结果和输出行为保持不变。

## 局部效率问题与设计依据

当前 `PlanLegged` 对每条候选边按 `local_resolution / 4` 插值。每个插值位姿重新
计算旋转矩形 AABB，遍历覆盖栅格，并为栅格重复执行 `3 x 3` 台阶邻域检查。当前
冻结配置的动作步长和地图分辨率均为 `0.2 m`，一次平移动作至少检查约五个机身
位姿；机身尺寸加净空后的矩形约为 `1.28 m x 0.93 m`。同一批地图栅格因此会在
同一条边和相邻边中被重复访问。

代表性足式导航工作采用分层和缓存化处理：先从高程图生成逐格可通行性，再缓存
圆形或足迹级结果；开阔区域优先使用简化足迹，精确多边形只用于验证或修复复杂
路径。本文采用相同原则，但不引入论文中的方差、学习模型或新传感器数据：

- [Wermelinger et al., *Navigation Planning for Legged Robots in Challenging Terrain*,
  IROS 2016](https://doi.org/10.3929/ethz-a-010686519)；
- [ETH `traversability_estimation`](https://github.com/leggedrobotics/traversability_estimation)
  的 footprint map 与 footprint path 检查；
- [Wellhausen and Hutter, *ArtPlanner: Robust Legged Robot Navigation in the Field*,
  Field Robotics 2023](https://arxiv.org/abs/2303.01420)。

## 输入契约

| 输入 | 数据内容 | 用途 |
| --- | --- | --- |
| 全局地图 | `occupancy`，`map` frame | 大范围粗占据先验 |
| 局部地图 | `occupancy`、`elevation`，`odom` frame | 近场障碍和地形坡度 |
| 坐标变换 | `map <- odom` | 将局部单元投影到统一地图坐标 |
| 足式状态 | 机身 pose、velocity | 规划起点和局部轨迹初态 |
| 规划目标 | `map` frame 点目标及容差 | 全局搜索终点 |
| 足式能力 | 尺寸、坡度、台阶、沟宽、速度和动作原语 | 地图 profile 与局部可执行性 |

本设计不增加观测时间、高程方差、语义类别、学习模型或新传感器。局部足式规划器
可以继续从现有 elevation 内部派生坡度、台阶和辅助地形代价，但不得要求上游新增
地图层。

## 足式可通行性 profile

足式 profile 复用现有 `TraversabilityProfile` 字段：

- `global_occupancy_threshold`：沿用规划器全局占据阈值；
- `local_occupancy_threshold`：沿用当前局部占据阈值；
- `maximum_slope_rad`：取 `LeggedCapability.maximum_slope_rad`；
- `inflation_radius_m`：取足式机身平面外接半径与最小机身净空之和。

膨胀半径定义为：

```text
hypot(body_extent_m.x, body_extent_m.y) / 2
  + minimum_body_clearance_m
```

按当前冻结配置，足式机身尺寸为 `0.68 m x 0.33 m`，最小机身净空为
`0.3 m`，对应全局圆形近似半径约 `0.678 m`。局部规划器使用保守扫掠包络快速
判定，并仅在包络包含危险或未知栅格时回退带朝向矩形检查；两条分支共同构成局部
可执行性的权威，不增加规划完成后的第二次认证。

## 地图分类与融合

### 全局先验

全局 occupancy 按当前 Grid V1 语义解释：

- 小于零：`UNKNOWN`；
- 大于等于全局阈值：`BLOCKED`；
- 其余合法值：`FREE`。

### 局部分类

局部单元按当前 Grid V1 轻量规则分类：

1. occupancy 或 elevation 非有限、occupancy 超出 `[0, 1]`：`UNKNOWN`；
2. occupancy 大于等于局部阈值：`BLOCKED`；
3. 由四邻域最大高程梯度计算的坡度超过足式上限：`BLOCKED`；
4. 其余：`FREE`。

坡度计算为：

```text
atan(max(abs(neighbor_elevation - center_elevation) / resolution))
```

不在全局地图中增加粗糙度、台阶或连续风险分数。台阶和沟宽仍由现有足式局部
规划器从同一份 elevation 判断。

### 融合语义

局部有效已知结果覆盖同一世界位置的全局先验和旧局部结果。局部非法或
`UNKNOWN` 单元不写入已知覆盖层，保留既有局部已知结果；没有局部已知覆盖时
回退到全局先验。局部与全局冲突时以局部已知结果为准并保留冲突诊断。

局部 `BLOCKED` 采用保守相交栅格化，局部 `FREE` 只写入源单元覆盖目标格心的
栅格。全局图和局部图可以使用不同分辨率，但局部 canonical resolution 在进程内
保持一致。

地图语义只由既有输入决定，不增加时间衰减或证据老化。

## 不可变快照

持久地图在语义变化时生成单调递增的 `traversability_revision`。足式规划请求使用
服务端现有的一次输入捕获，将有效 `TraversabilitySnapshot` 放入该请求的
`WorldSnapshot`。全局和局部规划均使用本次请求携带的输入。

本设计不增加规划完成后的最新地图重取、supercover 发布前复核、局部高程二次
认证或足式 `STALE_PATH_INVALIDATED` 流程。现有轮式 Grid V1 的发布检查不在本次
范围内，不因足式扩展而修改。

## 全局可行性投影

现有 ARA* 要求 `HardFeasible(cell)` 和 `ClearanceMeters(cell)` 为常数时间查询。
因此不得在 ARA* 邻居扩展中直接调用会扫描膨胀邻域的逐单元 Grid V1 查询。

每个新的 `(traversability_revision, profile_hash)` 只构建一次足式全局投影：

1. 将快照中的 `BLOCKED` 和不可搜索的 `UNKNOWN` 作为危险单元；
2. 构建危险单元距离场；
3. 根据足式膨胀半径生成 `hard_feasible`；
4. 保存每格 `clearance_m`；
5. 缓存只读投影供 ARA* 复用。

投影构建器必须读取地图维护器内部未膨胀的分类单元，或使用地图维护器提供的批量
投影接口；不得读取已经包含邻域膨胀的公共逐点查询结果后再次膨胀。膨胀只在投影
构建时执行一次。投影构建复杂度保持 `O(Ng)`，ARA* 单格查询保持 `O(1)`。

## 全局搜索

保留现有 `SearchSurfaceGlobal` 搜索框架：

- 八邻域 ARA*；
- 只扩展 `hard_feasible` 单元；
- 禁止对角穿越两个相邻阻挡角；
- 欧氏距离启发；
- 边代价继续包含几何距离和净空代价；
- 保留 deadline、取消、稳定边顺序和 anytime 候选语义；
- 不增加坡度软代价，坡度只参与 Grid V1 硬分类。

全局路线缓存键必须包含足式 capability fingerprint、
`traversability_revision` 和 `profile_hash`，不得按旧 global occupancy sequence 独立
复用已经不匹配新地形结论的路线。

## 局部目标选择

沿用现有月面分层规划逻辑：

1. 从全局路线当前位置向前选择位于局部地图范围内的候选；
2. 足式默认 horizon 为 `3.0 m`；
3. 候选按路线进度降序、同进度横向偏移绝对值升序、净空降序和稳定序号排序；
4. 最终目标已经进入局部范围时使用精确最终目标；
5. 没有局部候选或全部候选不可达时返回现有局部失败结果。

中间门户转换为足式目标时，将目标吸附到其对应的局部安全栅格中心，目标容差为
`0.5 * local_resolution - epsilon`，且不要求门户朝向。这样目标区域保持在已知安全
栅格内部，并避免全局栅格中心恰好落到局部栅格边界时产生零容差。精确最终目标
保持用户给定的位置、容差和朝向，不做栅格吸附。

足式局部规划器必须消费完整 `LocalGoalSet`，不得只取 `goals_odom.front()`。中间
滚动目标最多保留现有的 `32` 个候选；精确最终目标仍只能有一个。门户排序用于
稳定优先，不增加人工中心线惩罚，也不串行重启多个完整搜索。成功结果返回
`selected_goal_index`。

本设计不运行轮式 Grid V1 自带的第二次二维局部路径搜索。新增目标距离场只提供
连通性判断和搜索引导，足式请求仍只运行一次现有足式 anytime 局部搜索。

## 足式局部地形投影

`LocalTerrainProjection` 仍只从局部 `occupancy` 和 `elevation` 派生基础地形数据。
在其后构建足式专属只读 `LeggedTraversalProjection`，字段固定为：

```text
hard_feasible[cell]
step_feasible[cell]
slope_rad[cell]
roughness_m[cell]
clearance_m[cell]
hard_infeasible_prefix_sum
```

其中 `hard_feasible` 和 `step_feasible` 必须与当前局部边验证消费的占据、有限高程、
最大坡度和 `3 x 3` 邻域台阶规则一致。台阶邻域在投影构建时对每格只计算一次，
不得在搜索边验证中重复计算。`roughness_m` 继续由 elevation 内部派生，仅保留为
现有地形代价；它不是新增输入层，也不成为新的硬准入条件。

投影缓存键至少包含：

```text
local map sequence
map width/height
map resolution
local occupancy threshold
legged capability fingerprint
```

构建复杂度为 `O(Nl)`，其中 `Nl` 为局部地图栅格数。相同地图和 capability 的请求
复用只读投影。

## 请求级足式状态格

每次 `PlanLegged` 请求以精确起点位置和起始 yaw 构建不可变的请求级状态格坐标系。
地图原点只用于栅格查询，不再作为足式状态量化原点。状态键保持现有字段：

```text
x_index
y_index
yaw_index
motion_mode
narrow
```

普通区域继续使用局部地图分辨率和 `64` 个相对航向格，狭窄区域继续使用半分辨率
和 `128` 个相对航向格。`motion_mode` 继续参与状态键，因为模式切换参与边代价。
本阶段保持现有单键状态逻辑，不增加多标签、优势裁剪、状态签名或新的动作原语。

请求级坐标系只改变量化基准，不改变状态保存的实际世界位姿、地图查询坐标或动作
边几何。相同场景整体平移或连同起终点整体旋转后，应生成等价的状态键序列、规划
结果和代价。

## 足式多目标距离场与搜索

`LeggedPlanRequest` 改为接收完整 `LocalGoalSet` 和只读
`GoalDistanceField`；`LeggedPlanResult` 增加 `selected_goal_index`。距离场以全部
目标格为源点，在 `LeggedTraversalProjection` 的
`hard_feasible && step_feasible` 掩码上执行现有八邻域传播并禁止对角穿角，记录：

```text
distance_m[cell]
nearest_goal_index[cell]
```

起点距离为无穷时，说明起点与全部目标在必要可行掩码上断开，直接返回
`LEGGED_NO_PATH`。距离有限时，距离场只作为现有 ARA* 的非约束性 `guidance`；
主启发改为所有目标的现有欧氏代价下界最小值。距离场不得替代足式动作边认证，
也不得把二维连通性当作足式路径成功。

距离场缓存键至少包含足式局部投影身份和有序目标格集合。相同局部地图、capability
和目标集合可以复用；任一项变化后必须重建。

搜索图为每个实际到达的目标建立带目标索引的终端状态。每次状态展开只对一个动作
原语可达范围内的目标尝试现有缩放终端连接，所有连接仍执行足式边认证。一次
ARA* 同时搜索全部目标，并通过终端状态返回实际 `selected_goal_index`。

删除足式搜索固定的 `131072` 状态上限，状态和终端编号按创建顺序连续动态增长，
与现有轮式搜索使用同一动态编号语义。正常资源边界继续由有限局部地图、现有
deadline 和取消机制提供；不增加路线走廊、状态数量参数或新的内存预算配置。
搜索 OPEN 耗尽且无解时返回 `LEGGED_NO_PATH`，实际内存分配失败时返回
`LEGGED_RESOURCE_EXHAUSTED`。

## 足式局部两级边认证

现有 `PlanLegged` 保留：

- `x/y/yaw` 离散状态、普通区域与狭窄区域分辨率；
- 前进、后退、左右横移和原地旋转六种动作原语；
- 速度、转动、地形、净空和运动模式代价；
- 现有 anytime 搜索、deadline、取消和边结果缓存。

每条动作边按以下顺序认证：

1. 检查源点、终点、地图边界，并沿动作中心线执行现有机身高度等局部条件；
2. 用源点、终点、动作转角和机身外接半径构造覆盖整个动作的保守 swept AABB；
3. 通过 `hard_infeasible_prefix_sum` 常数时间查询 swept AABB 中是否存在不可行格；
4. 若不存在，执行快速接受，不再进行带朝向矩形逐格扫掠；
5. 若存在危险或未知格，执行现有带朝向矩形精确回退，并直接读取预计算地形字段；
6. 精确回退遇到首个相交的不可行栅格立即拒绝。

快速分支中的坡度、粗糙度和净空代价沿动作中心线 supercover 栅格聚合。它们用于
路径优选，不增加新的硬准入条件。精确分支保留当前相交矩形内的代价聚合。该差异
允许开阔地形以少量顺序数组查询完成边评估，同时仍由精确分支处理障碍边缘、狭窄
通道和原地旋转。

当前冻结能力中的最大台阶高度为 `0.5 m`，最大沟宽为 `0.3 m`。这些参数只在
局部动作可执行性中使用，不写入全局三值地图。

本设计不增加候选路径生成后的完整二次扫掠。每条输出边已经满足“保守 swept AABB
全部可行”或“精确带朝向矩形检查通过”之一，因此不存在未经机身约束的输出边。

## 后续优化门槛

首版不预计算 `(yaw_bin, primitive)` 完整扫掠掩码。搜索状态存在普通 `64` 航向格、
狭窄区 `128` 航向格和亚栅格位姿，相位处理会扩大实现和测试范围。只有本设计的
生产配置基准仍不能达到性能验收时，才单独设计动作原语扫掠掩码；不得在首版中
增加多标签状态、复杂优势裁剪或新的动作原语集合。

## 输出契约

成功结果继续使用现有 `MotionReference`：

```text
platform_type = LEGGED
preview = 全局参考路线
data = TrajectoryReference
data.semantics = kLeggedBodyReference
data.points = 局部可执行机身轨迹
```

全局路线只作为 preview 和滚动规划参考，不作为未经局部足式验证的完整可执行
轨迹。下游控制器根据机身 reference 生成步态和关节控制，本规划器不输出逐脚落足
序列。

## 模式与路由

增加足式全局模式字段：

```text
legged_global_mode = grid_traversability_v1 | legacy_occupancy
```

规则如下：

1. 参数缺省值为 `grid_traversability_v1`；
2. `grid_traversability_v1` 要求有效足式 capability 和可通行性快照；
3. `legacy_occupancy` 显式选择现有全局 occupancy 投影；
4. Grid V1 请求失败时不自动回退 legacy；
5. 当前 `wheel_planner_mode`、轮式默认值和轮式请求路由保持不变；
6. Hopper 路由保持不变。

## 失败处理

不为足式增加一套平行的安全错误体系。继续使用现有规划状态和阶段原因码：

- 非法或缺失输入：`INVALID_INPUT`；
- 起点或目标不在可搜索自由区域：对应现有起终点不可行原因；
- 全局搜索无路：全局 `NO_PATH`；
- 局部候选不存在或足式局部规划无路：局部 `NO_PATH`；
- 截止时间到达：`TIMEOUT`；
- 请求取消：`REQUEST_CANCELED`；
- 实际搜索内存分配失败：`LEGGED_RESOURCE_EXHAUSTED`；
- 内部契约异常：`PLANNER_ERROR`。

固定状态槽耗尽原因 `LEGGED_SEARCH_CAPACITY_EXHAUSTED` 不再由新足式搜索产生。
多目标距离场判定断连或动态搜索 OPEN 耗尽都属于正常不可达，返回局部 `NO_PATH`。

地图分类、ARA* 的 `HardFeasible` 和现有足式局部边验证属于算法求解条件，必须
保留。本设计不增加规划完成后的二次安全校验。

## 计算量约束

设全局投影栅格数为 `Ng`、局部更新影响栅格数为 `deltaNl`、局部地图栅格数为
`Nl`、全局展开状态数为 `Vg`、局部足式展开状态数为 `Vl`、进入精确回退的边数
为 `Efallback`、局部目标数为 `K`：

```text
增量地图更新       O(deltaNl)
全局投影构建       O(Ng)，按 revision 缓存
全局 ARA*          O(Vg log Vg)
足式局部投影       O(Nl)，按局部地图和 capability 缓存
多目标距离场       O(Nl log Nl)，按局部投影和目标格缓存
足式局部 ARA*      O(Vl log Vl)，动态状态
普通局部边判定     O(1) 包络查询 + O(L) 中心线代价聚合
终端连接候选       每状态仅检查动作一步范围内的 K 个目标
精确局部边判定     仅 Efallback 条边执行矩形扫掠
发布前新增复核     无
```

强制性能边界：

- 地图融合在输入更新路径完成；
- 规划请求只捕获只读快照；
- ARA* 的可行性和净空查询必须为 `O(1)`；
- 禁止在 ARA* 循环中执行足式半径邻域扫描；
- 禁止在普通局部边中重复执行 `3 x 3` 台阶邻域检查；
- 开阔区域不得进入精确矩形扫掠；
- 多目标距离场只构建一次，不为每个门户重复构建或重启足式搜索；
- 局部足式规划不得因本设计重复执行一套二维局部路径搜索。

## 诊断

复用现有 Grid V1 和分层规划诊断，至少保留：

- `legged_global_mode`；
- global/local/odometry 输入序号；
- `traversability_revision`、`profile_hash` 和 canonical resolution；
- 地图更新单元数、tile 数和 FREE/BLOCKED/UNKNOWN 统计；
- 全局投影与路线缓存命中；
- 全局和局部展开状态、OPEN 峰值与耗时；
- 局部候选数量、选中候选和足式目标距离场缓存命中；
- 足式局部投影构建时间与缓存命中；
- 快速接受边数、精确回退边数、精确扫掠栅格数和边认证缓存命中；
- 最终轨迹点数、总耗时、status、reason code 和是否存在 reference。

不新增 publish-check revision、足式 supercover 复核计数或足式二次认证耗时字段。
局部规划因总 deadline 被外层转换为 `TIMEOUT` 时，必须保留局部展开状态和上述边
认证统计，不得用全零诊断覆盖真实搜索工作量。

## 验收

### 地图与投影

1. 平坦、占据低于阈值的已知区域为可搜索自由区域。
2. 占据达到阈值的区域不可搜索。
3. `25°` 坡面按当前冻结能力对轮式不可行、对足式可行。
4. `35°` 坡面对足式不可行。
5. 无效局部 occupancy/elevation 不得直接写入新的 `FREE`；该位置按既有局部已知
   结果或全局先验处理。
6. 足式约 `0.678 m` 膨胀边界正确阻断过窄通道。
7. 同一 revision 重复请求命中全局投影缓存。
8. ARA* 单格查询不触发邻域半径扫描。

### 全局与局部规划

1. 全局 ARA* 禁止对角穿角。
2. 存在陡坡捷径和缓坡绕行时，足式全局路线选择缓坡绕行。
3. 没有可行全局路线时返回全局无路，不调用 legacy 自动回退。
4. 全局路线生成后只运行现有足式局部规划器。
5. `0.4 m` 台阶场景允许足式局部通过，`0.6 m` 台阶拒绝。
6. `0.2 m` 沟宽场景允许足式局部通过，`0.4 m` 沟宽拒绝。
7. 横移和原地旋转动作仍可生成合法机身 reference。
8. 开阔平地动作走快速包络分支，不执行精确矩形扫掠。
9. 中心线自由但 swept AABB 含危险格时进入精确回退；危险格与实际机身矩形相交时
   拒绝，不相交时允许通过。
10. 精确回退的平移、横移和旋转边有效性与优化前保持一致。
11. 中心线第一门户不可达、旁侧门户可达时，一次足式搜索选中旁侧门户并成功。
12. 中间门户位于局部栅格边界时，转换后的目标位于安全格中心且容差非零。
13. 相同场景整体平移或随起终点整体旋转后，足式状态键序列和结果等价。
14. 起点与全部目标在足式必要可行掩码上断连时，距离场直接返回局部无路。
15. 可达场景不再因固定 `131072` 状态槽返回
    `LEGGED_SEARCH_CAPACITY_EXHAUSTED`。

### 模式与输出

1. 未设置 `legged_global_mode` 时实际模式为 `grid_traversability_v1`。
2. 显式 `legacy_occupancy` 时保持现有足式全局路径行为。
3. Grid V1 失败不静默回退 legacy。
4. 成功结果的 `platform_type` 为 `LEGGED`。
5. 成功结果包含非空 `TrajectoryReference`，语义为
   `kLeggedBodyReference`。
6. 全局 preview 与局部机身轨迹坐标系正确。
7. 足式成功结果的 `selected_goal_index` 与实际终端门户一致。
8. 轮式 Grid V1、轮式 legacy 和 Hopper 现有测试不回归。

### 性能记录

在相同地图、起终点、配置和 Release 构建下对比现有足式 legacy 与新模式，至少
记录：

- 地图更新时间；
- 全局投影构建与缓存命中时间；
- 全局展开状态和耗时；
- 距离场构建时间与缓存命中；
- 局部展开状态、OPEN 峰值、选中目标、快速接受边、精确回退边、精确扫掠单元数
  和耗时；
- 总规划时间和峰值内存。

新增生产局部性能场景：`64 m` 局部地图、`0.2 m` 分辨率、`3 m` 局部目标、生产
六动作原语和非零起点 yaw。开阔可行场景的目标是 `local_search < 1 s`；所有已知
可行或不可行场景必须在 `3 s` 硬截止前给出 `PLAN_FOUND` 或 `NO_PATH`，不得仅因
局部边认证重复计算返回 `TIMEOUT`。用户 RViz 复现场景必须记录相同阶段耗时和边
认证统计。连续滚动复现场景还必须证明不再出现固定状态容量耗尽，并保留失败或
成功时的真实展开状态和选中目标诊断。

复杂度分析不替代实测，也不得把本机 Jazzy 结果表述为 Humble、Orin 或实车通过。

## 实施边界

后续实施计划应按以下独立单元组织：

1. 足式模式参数、默认值和请求路由；
2. 足式 `TraversabilityProfile` 与服务端地图输入；
3. 按 revision 缓存的足式全局可行性投影；
4. 现有 `PlanSurfaceGlobal` 对足式 Grid V1 投影的适配；
5. 足式局部地形投影、前缀和与缓存；
6. `PlanLegged` 两级边认证和超时诊断保留；
7. 中间门户安全格中心转换与完整 `LocalGoalSet` 传递；
8. 请求级足式状态格和动态状态编号；
9. 足式多目标距离场、一次多目标搜索和选中目标传播；
10. 单元测试、集成测试、生产配置性能基准和 RViz 复现。

每个单元必须先有失败测试，再实施最小改动。不得借本设计重构无关平台路径。

## 非目标

本设计不包含：

- 逐脚落足规划；
- 步态生成或关节轨迹；
- 动力学稳定性优化；
- 观测时间、地图老化或高程方差；
- 语义或学习型可通行性；
- 新传感器或新外部地图层；
- 发布前最新地图重取和足式二次认证；
- 足式 `STALE_PATH_INVALIDATED`；
- 控制器修改；
- Hopper 修改；
- 轮式 Grid V1 行为变更；
- 首版动作原语完整扫掠掩码；
- 每键多标签、复杂优势裁剪或终端状态签名；
- 路线走廊、新状态数量参数或新内存预算配置；
- 足式优选路径模板或通用 lattice 引擎重构；
- 航向格数或动作原语集合调整；
- 自动 legacy fallback。

## 2026-08-31 足式局部边认证实施证据

本次实现位于独立分支 `feat/legged-local-edge-evaluation`，对应提交：

- `a2dcb03`：从现有局部 occupancy/elevation 投影构建并缓存
  `LeggedTraversalProjection`；
- `ab66b4d`：在生产 `PlanLegged` 中加入保守 swept AABB 快速接受与现有定向矩形
  精确回退；
- `733ea71`：向规划结果和 ROS 请求诊断传播投影缓存、快/慢分支与超时工作量统计。

本机 ROS 2 Jazzy、Release 构建的生产形态局部场景为：`320 x 320` 栅格、`0.2 m`
分辨率（`64 m`）、`3.0 m` 局部目标、`1.03242 rad` 起点 yaw、冻结的
`0.68 x 0.33 m` 机身、`0.3 m` 净空、`0.5 m` 台阶、`0.3 m` 沟宽和六个
`0.2 m` 动作原语。单次结果为：

```text
local_status                   kSolved
reason_code                    LEGGED_PLAN_SOLVED
local_search_elapsed_ms        23.460992
expanded_states                6227
fast_path_accepts              14925
exact_sweep_fallbacks          0
exact_sweep_cell_checks        0
whole_gtest_elapsed_ms         50
```

这组数据证明开阔地形已跳过密集定向机身扫掠，并在本机达到 `local_search < 1 s`；
它不代表障碍场景总是走快速分支。单元测试另覆盖 swept AABB 含危险格时进入精确
回退且定向矩形不相交仍可通过，以及坡度、台阶、狭窄通道、取消和超时语义。

同一 Release 构建下的验证结果：

```text
lunar_pure_planner_core CTest   22/22 passed, 48.42 s, serial
lunar_pure_planner_ros CTest    18/18 passed, 48.05 s, serial
repository contract pytest      6/6 passed, 0.02 s
git diff --check                passed
```

证据边界：本次未重新运行交互 RViz 场景，`RViz = NOT_RUN`；
`Humble = NOT_RUN`、`Orin = NOT_RUN`、`DDS = NOT_RUN`、`rosbag = NOT_RUN`、
`vehicle = NOT_RUN`。本机未安装 `clang-format`，自动格式化检查为 `NOT_RUN`，但
源码已通过 Release 编译、全部受影响包测试和差异空白检查。

## 2026-08-31 足式滚动与 RViz 输出补充

足式 `grid_traversability_v1` 接入现有月面滚动调度，同一 Action 复用一次全局路线并
按 `3.0 m` 前瞻发布足式局部分段。当前分段到达前，局部图刷新和全局路线偏离量
不会抢占该分段；每个新周期使用当次冻结的可通行性快照。轮式滚动触发规则保持
不变。

足式局部轨迹和全局预览分别发布为 `legged_path_topic` 与
`legged_global_path_topic`。正常路径及失败时的空 `Path` 均带 `map` frame，避免
RViz Message Filter 因空 frame 丢弃清空消息。demo 每个 0.5 s 周期最多沿路径移动
0.5 m，并同步更新运动方向 yaw，防止密集路径点被一次全部吞掉或移动后仍使用旧
朝向。终端每条 `[PLAN nnn]` 前固定输出 `----------------------------`。

本机 Jazzy 无 RViz 实节点以固定起终点运行时，首周期为 `786.5 ms`，随后五个连续
滚动周期为 `198.0 ms`、`221.7 ms`、`185.0 ms`、`298.9 ms` 和 `161.4 ms`；六个
周期均为 `planning_outcome=0`、`reason_code=PLAN_FOUND`、
`has_reference=true`，未再出现局部图刷新导致的 `STALE_INPUT`。这是本机 demo
证据，不外推为 Humble、Orin、DDS 或实车证据。

## 2026-08-31 连续滚动失败与设计修订依据

后续交互 RViz 连续滚动暴露了与边认证计算量不同的局部搜索问题。第 `064` 次规划
仍成功，但局部搜索已经展开 `65633` 个状态并耗时约 `1402.9 ms`；第 `065` 次在
局部地图更新后返回 `LEGGED_SEARCH_CAPACITY_EXHAUSTED`，局部搜索约
`1676.8 ms`，快速边接受 `256075` 次、精确扫掠回退 `0` 次。随后多个新目标均在
约 `0.7 s` 至 `0.8 s` 局部搜索后返回同一固定容量错误。

该证据说明失败不是密集机身扫掠、全局搜索或 deadline，而是现有足式路径只消费
第一个门户、单目标欧氏引导和固定 `131072` 状态槽共同造成。顶层失败结果重新构造
时还丢失了局部 `expanded_states`，因此日志中的 `expanded_states=0` 不是实际未展开。

本次设计修订只处理这些直接原因：完整多目标接口、足式目标距离场、请求级状态格、
动态状态编号以及失败工作量传播。它不引入路线走廊、多标签状态、优选模板或新的
发布安全流程。

## 2026-08-31 局部搜索对齐实施结果

修订已在 `feat/legged-local-edge-evaluation` 实施：足式一次搜索接收完整有序门户集合，
以 `hard_feasible && step_feasible` 构建多源距离场，状态键改为请求起点与起始航向的
相对量化，终端和普通状态改为动态编号，并保留 `std::bad_alloc ->
LEGGED_RESOURCE_EXHAUSTED`。轮式继续使用原距离场入口和原门户转换行为。

RViz demo 首次复核还发现：足式搜索允许 `tolerance + 1e-9 m` 的数值边界，而滚动
调度器只检查裸 `tolerance`，导致路径停在半格边界后不触发下一段。实现仅对足式
滚动到达判定补齐同一 `1e-9 m` 数值容差，并增加边界回归测试，没有增加新参数或
安全流程。

本机 Jazzy 最终证据为 core CTest `22/22`、ROS CTest `18/18`、对应契约测试
`64 passed`。实际 RViz 足式 demo 记录 `18` 个连续滚动成功分段，前两轮总耗时分别
为 `214.4 ms` 和 `110.1 ms`；各轮均满足 `planning_outcome=0`、
`reason_code=PLAN_FOUND`、`has_reference=true`，未出现
`LEGGED_SEARCH_CAPACITY_EXHAUSTED` 或空 frame Message Filter 警告。
`Humble`、`Orin`、`DDS`、`rosbag` 和实车均为 `NOT_RUN`。
