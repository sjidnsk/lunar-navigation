# 足式 Grid V1 分层路径规划设计

## 状态

本设计于 2026-08-31 在对话中确认。它定义足式平台如何复用 Grid V1
持久化可通行性地图，并保留现有全局 ARA*、局部足式 lattice/anytime
规划器和 `kLeggedBodyReference` 输出契约。

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
                                      前方局部目标选择
                                               │
                                               ▼
                                现有足式 lattice/anytime 规划
                                               │
                                               ▼
                                    足式机身位姿轨迹
```

Grid V1 负责统一地图语义和全局路线约束；现有足式局部规划器继续负责坡度、
台阶、沟宽、机身高度和完整机身扫掠。下游足式控制器继续负责步态和逐脚控制。

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

本设计不开发另一套足式局部规划器。它只将现有足式全局阶段的可行性来源从
`GlobalOccupancyProjection` 替换为由足式 Grid V1 快照生成的投影，并继续调用
现有 ARA*、局部目标选择、足式局部规划和 reference 组合逻辑。

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
`0.3 m`，对应全局圆形近似半径约 `0.678 m`。局部规划器仍使用带朝向的矩形
机身扫掠，因此它是最终局部可执行性的权威。

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
3. 从进度较大的候选向较近候选尝试；
4. 最终目标已经进入局部范围时使用精确最终目标；
5. 没有局部候选或全部候选不可达时返回现有局部失败结果。

本设计不再运行轮式 Grid V1 自带的第二次二维局部 A*。足式请求只运行现有足式
局部规划器，避免重复局部搜索。

## 足式局部规划

现有 `PlanLegged` 保持不变。它继续使用：

- `x/y/yaw` 离散状态和可达机身高度区间；
- 前进、后退、左右横移和原地旋转动作原语；
- 局部 occupancy 与 elevation；
- 最大坡度、最大台阶高度和最大沟宽；
- 带朝向的机身矩形全边扫掠；
- 速度、转动、地形、净空和运动模式代价；
- 现有 anytime 搜索、deadline 和取消语义。

当前冻结能力中的最大台阶高度为 `0.5 m`，最大沟宽为 `0.3 m`。这些参数只在
局部动作可执行性中使用，不写入全局三值地图。

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
- 内部契约异常：`PLANNER_ERROR`。

地图分类、ARA* 的 `HardFeasible` 和现有足式局部边验证属于算法求解条件，必须
保留。本设计不增加规划完成后的二次安全校验。

## 计算量约束

设全局投影栅格数为 `Ng`、局部更新影响栅格数为 `deltaNl`、全局展开状态数为
`Vg`、局部足式展开状态数为 `Vl`：

```text
增量地图更新       O(deltaNl)
全局投影构建       O(Ng)，按 revision 缓存
全局 ARA*          O(Vg log Vg)
局部足式搜索       保持现有复杂度
发布前新增复核     无
```

强制性能边界：

- 地图融合在输入更新路径完成；
- 规划请求只捕获只读快照；
- ARA* 的可行性和净空查询必须为 `O(1)`；
- 禁止在 ARA* 循环中执行足式半径邻域扫描；
- 局部足式规划不得因本设计重复执行一套二维局部 Grid V1 搜索。

## 诊断

复用现有 Grid V1 和分层规划诊断，至少保留：

- `legged_global_mode`；
- global/local/odometry 输入序号；
- `traversability_revision`、`profile_hash` 和 canonical resolution；
- 地图更新单元数、tile 数和 FREE/BLOCKED/UNKNOWN 统计；
- 全局投影与路线缓存命中；
- 全局和局部展开状态、OPEN 峰值与耗时；
- 局部候选数量和选中候选；
- 最终轨迹点数、总耗时、status、reason code 和是否存在 reference。

不新增 publish-check revision、足式 supercover 复核计数或足式二次认证耗时字段。

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
8. 中心线自由但机身矩形扫掠碰撞时，现有局部规划器拒绝该动作。

### 模式与输出

1. 未设置 `legged_global_mode` 时实际模式为 `grid_traversability_v1`。
2. 显式 `legacy_occupancy` 时保持现有足式全局路径行为。
3. Grid V1 失败不静默回退 legacy。
4. 成功结果的 `platform_type` 为 `LEGGED`。
5. 成功结果包含非空 `TrajectoryReference`，语义为
   `kLeggedBodyReference`。
6. 全局 preview 与局部机身轨迹坐标系正确。
7. 轮式 Grid V1、轮式 legacy 和 Hopper 现有测试不回归。

### 性能记录

在相同地图、起终点、配置和 Release 构建下对比现有足式 legacy 与新模式，至少
记录：

- 地图更新时间；
- 全局投影构建与缓存命中时间；
- 全局展开状态和耗时；
- 局部展开状态、机身扫掠单元数和耗时；
- 总规划时间和峰值内存。

复杂度分析不替代实测。首版不预设未经基准验证的相对加速百分比。

## 实施边界

后续实施计划应按以下独立单元组织：

1. 足式模式参数、默认值和请求路由；
2. 足式 `TraversabilityProfile` 与服务端地图输入；
3. 按 revision 缓存的足式全局可行性投影；
4. 现有 `PlanSurfaceGlobal` 对足式 Grid V1 投影的适配；
5. 现有局部目标、`PlanLegged` 和 reference 组合复用；
6. 诊断、单元测试、集成测试和性能基准。

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
- 自动 legacy fallback。
