# 完整搜索、跳跃弹道认证与三平台滚动协调设计

日期：2026-08-06

状态：交互决策已批准，待正式文档审阅

## 1. 适用范围与文档优先级

本文修订并补充
`docs/superpowers/specs/2026-08-05-hierarchical-global-planning-design.md`，解决以下两个已经在
50 m × 50 m、0.2 m 合成地图实验中暴露的问题：

1. 全局搜索以固定展开数、候选数、节点数和出度作为提前终止条件，导致本来可能存在的路线
   被报告为资源不足，尤其使跳跃式平台很难规划成功；
2. `PlanMotion` 只返回当前一次局部段，RViz 演示既不推进平台，也没有根据执行进度自动请求
   后续局部段，因此用户看不到完整的滚动规划过程。

当本文与 2026-08-05 设计冲突时，下列内容以本文为准：

- 全局搜索的完备性、自然边界、停止条件和失败语义；
- 跳跃式候选落区、邻接生成、名义抛物线和完整飞行管认证；
- 全局路线复用、执行反馈、三平台滚动协调和 RViz 动态展示；
- 本轮性能基线和 Release 构建要求。

原设计中的多分辨率地图规则、平台能力边界、局部物理认证、Action 并发规则、只授权一个
跳跃段、外部输入所有权和 C++ v3 唯一生产内核继续有效。

## 2. 已批准约束

### 2.1 本轮必须完成

- 三个平台的全局搜索不再因固定 expanded/generated/open 数量提前宣告失败；
- 轮式和足式在有限全局栅格上完成确定性二维 A* 搜索；
- 跳跃式使用有限安全落区集合上的完整惰性 A*，不再预截断为 64 个候选、128 个节点或
  每节点 8 条边；
- 跳跃式仍求解名义抛物线，并使用平台外形与安全余量构成的完整飞行管检查地形和月岩；
- 三个平台共享“完整全局路线 + 当前局部执行段 + 执行反馈触发下一段”的滚动机制；
- RViz 能动态显示平台当前位置、完整全局路线、当前段、后续段以及成功或失败原因；
- 仓库内和仓库外实验构建均显式使用 Release；
- 增加正确性、完备性、滚动状态机和性能回归。

### 2.2 本轮明确不做

- 不修改三类平台的正式运动能力值；当前代理能力只能用于带明确标记的测试 fixture；
- 不把当前 `2.2 m/s`、`3 s` 或局部视距等实验值冻结为真实数字孪生参数；
- 不负责跳跃执行器的闭环轨迹跟踪、姿态控制、推力分配或硬件落地控制；
- 不取消地图最大尺寸、地图层级、输入有效性、停止令牌和数值求解迭代边界；
- 不用 Nav2、旧 Python A*、随机重试或增大目标容差掩盖规划问题；
- 不让一次 `PlanMotion` Action 持续占用直到任务最终完成，也不改变现有
  `PlanMotion.action` 或 `MotionReference.msg` schema；
- 不把系统取消、内存分配失败、数值不确定或客户端等待超时误报为物理无路。

## 3. 当前故障证据与根因

当前跳跃全局规划器存在相互叠加的拓扑与资源截断问题：

- `HopperPlannerConfig` 默认把落区、图节点和出度限制为 `64/128/8`；
- `BuildReachableOutgoingEdges` 对已经插入的节点只连接目标节点，无法形成完整的已发现非目标
  节点网络；
- 每次只保留排序后的前 8 条边，剩余物理可达边不会进入 Open 集；
- 一旦候选、节点或展开计数触顶，规划器返回
  `HOPPER_GLOBAL_ROUTE_RESOURCE_LIMIT`，即使 Open 集仍有未探索方向；
- 单独把节点数提高到 512 仍不能修复失败，说明根因不仅是节点数量，还包括候选生成、出度
  截断和错误的邻接拓扑；
- 现有仓库外构建脚本只打开测试，没有强制 `CMAKE_BUILD_TYPE=Release`，导致轮式、足式和
  跳跃式的调试构建耗时被放大。

已观察到的本机诊断样本中，腿式同一用例从非优化构建约 3.86 s 降至 Release 约 0.50 s；
一个已知跳跃正例从约 19.3 s 降至约 1.51 s。它们只说明构建类型是重要因素，不能证明现有
跳跃拓扑正确，也不能替代本设计的算法修复与正式基准。

## 4. 正确性与完备性定义

### 4.1 有限规划域

全局地图继续遵守已冻结的多分辨率规则，最多 1,048,576 个栅格、单轴最多 4,096 个栅格。
平台安全投影把每个栅格判定为安全、不可用或未知。全局搜索域由以下有限集合自然确定：

- 轮式/足式：安全投影中的所有有效栅格；
- 跳跃式：安全着陆支撑场中的所有安全落区中心，加上实际起点这个特殊节点。

算法不再另设小于该有限域的“搜索预算”。因此“完整”是指对当前离散地图、当前能力版本和
当前安全规则定义的有限状态/边集合完整，而不是对连续世界做无限精度证明。

### 4.2 成功、无路与搜索未完成

只有满足以下条件才能返回 `GLOBAL_NO_KNOWN_SAFE_ROUTE`：

- 轮式/足式 A* 的 Open 集为空；或
- 跳跃式惰性图搜索在所有可能到达的落区和候选边上耗尽 Open 集，并且所有曾进入候选路线
  但未通过完整认证的边都已被确定证明无效。

以下情况一律不允许返回物理无路：

- 收到 `stop_token` 或 Action cancel；
- 客户端自行达到等待截止时间并取消请求；
- 内存分配失败或输入地图超过已冻结的地图规模契约；
- 抛物线或飞行管数值认证无法收敛到确定结论；
- 输入、TF、定位、地图或执行反馈缺失、过期、不一致；
- 局部规划器的数值迭代边界触发。

这些情况分别返回 canceled、resource、numerical、stale/invalid 或 local-incomplete 类结果，
并明确表示本次没有完成可达性证明。

### 4.3 确定性

相同地图内容、能力版本、配置、起点和目标必须产生相同结果。所有稳定顺序使用：

```text
(f_cost, h_cost, g_cost, hop_count, row_major_node_id)
```

浮点代价相等时使用固定容差，再按跳数和行优先节点编号打破平局。候选生成、边认证、无效边
缓存和路线重建都不得依赖哈希容器迭代顺序或线程调度顺序。

## 5. 总体架构

系统分为纯规划核心、ROS 单次规划服务、路线续用存储和外部滚动协调器：

```text
RViz 2D Goal / 任务最终目标
             │
             ▼
RollingGoalCoordinator（外部执行侧）
  保留最终目标、当前平台、plan/segment 身份和触发条件
             │ repeated PlanMotion goals
             ▼
PlanMotionServer ───── ExecutionFeedback + Odometry
  冻结快照              │
  注入 previous_execution
  读取/更新 RollingRouteStore
             │
             ▼
Planner::Plan(input)（纯函数）
  新全局搜索或验证后续路线
  生成当前局部段/下一跳
  返回不可变 continuation
             │
             ├────────► 执行器
             └────────► RViz 路线、弹道、落区、状态与诊断
```

`Planner::Plan(const PlannerInput&)` 继续不持有跨请求可变状态。路线续用候选作为显式输入进入
核心，新的 continuation 作为显式输出离开核心；ROS 层的 `RollingRouteStore` 只保存最后一次
成功结果的不可变副本。这样既能避免每个局部段重复做完整全局搜索，又能保持核心测试可重复。

`RollingGoalCoordinator` 是 `PlanMotion` 的 Action 客户端，不是第二套规划器。它不得修改
路线、放宽安全约束或自行寻找替代路径，只负责在合法事件到来时用相同最终目标发起下一次
请求。生产环境由外部执行系统实现该角色；仓库外 RViz/ROS 实验桥提供同语义的参考实现。

## 6. 全局搜索资源模型

### 6.1 删除的提前终止条件

以下配置不再参与全局可达性判定，并从生产全局搜索配置、参数 schema 和实验配置中移除：

- `GlobalSearchConfig.resources.maximum_expanded_states`；
- `maximum_reopened_states`；
- `maximum_generated_candidates`；
- `maximum_open_states`；
- `maximum_memory_bytes`；
- 跳跃全局候选使用的 `maximum_landing_regions`；
- `maximum_graph_nodes`；
- `maximum_graph_out_degree`；
- 作为可达性截断使用的 `maximum_nominal_aim_points_per_region`；
- 作为路线搜索截断使用的 `maximum_certification_attempts`。

如果同名字段仍存在于旧配置，configure 必须以明确的 unknown/deprecated 参数错误拒绝，不能
静默接受后继续截断。局部高维 ARA*、走廊构建、平滑器和连续碰撞检查仍可保留自己的数值
迭代边界，但触发时必须报告局部搜索或数值认证未完成，不能形成全局物理无路结论。

### 6.2 保留的自然与安全边界

- 全局地图最大栅格数、最大单轴栅格数和 L0–L4 层级契约；
- 起点、目标、地图、TF、能力和观测新鲜度校验；
- `stop_token` 检查；
- 数值求根、飞行管细分、局部优化和连续碰撞验证的收敛边界；
- 输出预览点数上限，仅用于消息抽稀，不参与内部路线搜索或路线存在性判断；
- 由进程可用内存决定的真实分配失败。

轮式/足式使用按地图栅格数一次分配的紧凑数组，跳跃式使用按安全落区数量分配的节点数组、
流式邻居枚举器和只保存已认证/已否决边的证书缓存。不得预建完整跳跃邻接矩阵，也不得为了
重规划永久保存每个节点的全部潜在邻接。`maximum_flight_tube_sections` 只作为连续认证的数值
收敛边界；触发时返回 numerical-incomplete，不得把边判为碰撞。分配失败返回
`GLOBAL_SEARCH_ALLOCATION_FAILED`/`RESOURCE_EXHAUSTED`，表示搜索未完成。

### 6.3 取消与截止时间

核心没有会改变可达性结论的内部墙钟超时。Action 客户端可以设置响应截止时间，但达到截止
时间后必须发送 cancel，并把结果记为 `REQUEST_CANCELED` 或
`PLANNING_DEADLINE_CANCELED`，不能显示“无路”。所有 O(N) 建场、邻居枚举、弹道候选求解和
飞行管细分循环都按固定工作粒度检查 `stop_token`。

## 7. 轮式与足式完整全局搜索

轮式和足式继续使用平台各自的安全投影，并在有限二维栅格上运行惰性 8 邻域 A*：

- 节点在展开时才生成邻居，不预建图；
- 对角边必须通过禁止切角规则；
- `g`、parent、closed 和 Open handle 使用按行优先栅格编号索引的紧凑存储；
- 启发函数保持可采纳，风险代价只进入非负边代价；
- 节点可以 reopen，直到以一致的最优条件闭合或 Open 集耗尽；
- 不再在 expanded/generated/open 达到某个固定计数时停止；
- 原始路线先经过确定性 supercover 视线简化，再由局部后端生成曲率、朝向连续且碰撞复检
  通过的执行轨迹；
- `path_preview` 的第一个点固定为实际起点，最后一个点位于用户目标区域内；目标容差只定义
  合法终点集合，不能让路径脱离实际起点。

全局简化只减少预览和局部航点，不改变原始栅格链的存在性证据。任何简化线段检查失败都保留
原节点，不得以简化失败否定原始路线。

## 8. 跳跃式安全落区与空间索引

### 8.1 一次建场

每个请求只构建一次 `LandingSupportField`。它按当前跳跃能力、平台碰撞外形和地图安全层，
为每个全局栅格记录：

- 基础表面是否有效；
- 以该栅格为中心时完整着陆区域是否满足有效面积；
- 坡度、粗糙度、平面残差、侧向净空和顶部净空是否合格；
- 保守表面高度、风险代价和行优先节点编号。

所有安全中心进入确定性的二维空间索引。索引桶宽由当前能力推导出的最大水平可达距离和地图
分辨率确定；查询一个起点的邻居时只枚举与可达圆相交的桶，再做精确必要条件检查。空间索引
只改变枚举速度，不能丢弃任何物理可能可达的安全中心。

实际起点不要求恰好位于栅格中心。它作为特殊节点保存真实位置和表面高度；只有当前支撑区
确实安全时才能进入图。目标区域不预先占用固定数量的节点，任一安全中心与目标区域相交时
即带有 goal 标记。

### 8.2 直接跳优先

在进入图搜索前，先查询目标区域内且位于单跳包络内的所有安全中心，按稳定代价排序并尝试
名义抛物线与完整飞行管认证。任一候选通过即可返回一跳路线。直接跳失败的边进入请求内无效
边缓存，然后继续多跳搜索；直接跳失败不能阻止其他中间落区路线。

## 9. 名义抛物线与完整飞行管

### 9.1 名义轨迹

对于起点 `p0`、候选落点 `p1`、月面重力向量 `g` 和候选飞行时间 `T`，名义质心轨迹为：

```text
p(t) = p0 + v0 * t + 0.5 * g * t^2
v0   = (p1 - p0 - 0.5 * g * T^2) / T
v(t) = v0 + g * t
```

飞行时间求解必须覆盖能力资料允许的完整闭区间，并检查：

- 最小/最大飞行时间；
- 最大发射速度与最大等效冲量；
- 最小向下着陆速度和最大着陆速度；
- 起点/落点高差；
- 有限值、单位和数值残差。

可以使用有界解析区间、根隔离和确定性细分求解，但不得只抽取少量 aim point 后把未抽中的
时间解释为不可达。数值边界用于获得确定证书；若无法证明可行或不可行，返回
`HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE`，本次搜索未完成。

### 9.2 飞行管定义

本项目不负责执行器怎样跟踪该抛物线，但必须证明名义轨迹在当前快照中具有安全几何通道。
飞行管由平台碰撞几何的保守支撑体沿 `p(t)` 扫掠，再增加以下安全余量：

- 能力文件或项目配置中已有的几何净空；
- 地图分辨率和高程不确定性；
- 允许的执行跟踪误差；
- 数值细分误差。

对第二跳及后续 hop，证书还必须覆盖上一跳允许的实际落点偏差：先把上一落区收缩为
`promotion_region`，再把该区域相对名义落点的最大平移、定位协方差和稳定后残余速度纳入下一
跳飞行管余量。名义抛物线仍从名义落点生成，但只有实际姿态落在这个已认证的
`promotion_region` 内时才能直接提升下一跳；否则必须从实际姿态重规划。不能只验证“落在较大
候选落区内”便假设下一条名义抛物线仍然有效。

认证覆盖 `t ∈ [0,T]` 的完整区间，并同时检查地形表面、障碍高度、独立月岩碰撞体投影、
禁入区、起跳脱离段和着陆接近段。不能只检查弹道采样点的质心，也不能只检查最高点或落区。

连续检查使用保守包围体与自适应细分。若一个区间能证明与所有障碍分离则通过；能证明相交则
该边无效；达到数值细分边界仍不能判定时返回 numerical-incomplete，不能把该边静默视为
碰撞，也不能在最终 Open 集耗尽后声称无路。

所有远端 hop 首先针对同一次冻结快照中的保守 `global_map` 认证。每个占用/障碍栅格按完整
栅格横向范围和 `obstacle_height` 竖向范围构成障碍体，地形表面再按分辨率、高程方差和配置
余量扩张；因此粗层可以更保守地拒绝路线，但不能漏掉被聚合规则保留下来的月岩。当前授权
hop 与 `local_map` 覆盖区域相交的部分还必须通过 L0 一致性门控；局部图出现更新障碍或与全局
证书冲突时不授权并触发重规划。Isaac/真实环境中的独立月岩碰撞体必须先由外部地图融合系统
进入 `obstacle`/`obstacle_height`，规划器不能直接看到未栅格化的物理碰撞体。

### 9.3 两级边检查

为了控制规划时间，搜索采用两级检查：

1. **必要条件粗筛**：水平/高程包络、速度下界、保守高度范围和空间索引。粗筛只允许排除
   已被必要条件证明不可能的边；不确定边必须保留。
2. **完整认证**：A* 找到一条候选落区链后，按路线顺序求解每一跳的名义抛物线并认证完整
   飞行管。

完整认证失败且原因是确定碰撞或能力违反时，把有向边 `(source_id,target_id)` 记录为无效，
从当前搜索图移除。首版实现重新运行最短路阶段，但复用落区场、空间索引、惰性邻接、无效边
集合和已通过证书，不重复建场或认证同一有向边；以后可以在不改变语义的前提下替换为增量
最短路修复。认证通过的边及其抛物线、飞行管证书进入通过缓存。

只有整条候选链的每条边均通过，才生成成功结果。这个 lazy validate–invalidate–resume 循环一直
持续到获得完整已认证路线、收到取消、遇到无法判定的系统/数值错误，或 Open 集真正耗尽。

## 10. 跳跃完整惰性 A*

### 10.1 节点与邻接

节点全集是有限地图上的全部安全落区中心，加真实起点特殊节点。节点编号由行优先栅格编号
确定，不按“被发现先后”改变。展开节点时：

- 从空间索引取出最大水平可达圆内的全部安全中心；
- 包含已发现和未发现的非目标节点，也包含目标节点；
- 排除自身与已缓存无效边；
- 对每个候选执行必要条件粗筛；
- 按稳定边代价和目标启发值排序，但不截断出度；
- 邻接由可重复的流式迭代器产生；允许使用可丢弃的性能缓存，但缓存淘汰后必须能够无损重算，
  且不能限制为固定前 N 条。

这修复当前“已有节点只连目标节点”的拓扑缺口，并保证安全落区图中每一条可能边最终都有被
考虑的机会。

### 10.2 代价与启发函数

边代价继续由非负项构成：飞行时间、冲量/发射裕量、着陆风险和稳定等待代价。启发函数使用
到目标区域的水平距离除以保守最大单跳距离，再乘不高于最小单跳代价的下界，保持可采纳。
如果无法构造严格正的单跳代价下界，则启发值退化为零，算法成为完整 Dijkstra，不能使用
可能高估的经验权重换取速度。

### 10.3 终止与输出

- 成功：返回完整已认证落区链、每跳名义抛物线和飞行管证书；
- Action `path_preview`：显示完整落区链；第 17 节的规划器 Marker Topic 绘制全部名义弹道、
  飞行管和 promotion region；
- Action `hops`：仍只包含当前唯一授权跳跃；
- 无路：完整搜索和所有候选路线认证结束后 Open 集为空；
- 取消/分配/数值失败：返回对应 incomplete 结果。

`maximum_authorized_hops=1` 保留，因为它是执行安全授权规则，不是搜索资源上限。

## 11. 路线续用数据模型

核心增加不进入 ROS 公共消息的不可变 `RouteContinuation`：

```text
RouteContinuation
  route_id
  current_reference_plan_id
  platform_type
  mission_id + mission_revision
  normalized_goal_hash
  capability_version
  global_map_generation
  local_map_generation_at_issue
  map_from_odom_generation
  full_global_route
  raw_route_evidence
  current_route_cursor
  certified_hops[]             # hopper only
  invalid_hopper_edges[]       # request/continuation-local
  route_corridor_certificate   # ground platforms
```

`PlannerOutput` 可携带 continuation，`PlannerInput` 可携带一个续用候选。核心必须在使用前验证
所有身份字段和当前状态，不能因为 ROS 层提供了缓存就跳过安全检查。`RollingRouteStore` 最多为
当前活动平台保存一个 continuation；目标替换、任务 revision 变化、Lifecycle deactivate、
能力版本变化或全局地图 generation 变化时原子失效。

`global_map_generation` 由 ROS 快照存储根据规划相关内容身份生成。适配 GridMap 时对 frame、
几何、分辨率、地图姿态、层名和规范化后的全部规划层计算确定性指纹；相同内容的周期性重发
沿用 generation，任一规划相关内容变化才递增。Header stamp 仍独立参与新鲜度校验，不能因
指纹相同绕过 stale 检查。以后若要忽略某些层或使用增量瓦片，必须另行设计和验收。
`map_from_odom_generation` 保存生成路线时的基准变换；小幅更新只有在把当前姿态与剩余路线
重新投影后仍通过走廊/落区检查时才能复用，超限变化立即使 continuation 失效。

## 12. 三平台统一滚动状态机

### 12.1 通用会话

滚动会话由 `(platform_id, mission_id, mission_revision, final_goal_hash)` 唯一标识。状态为：

```text
IDLE
  └─新最终目标──> PLANNING
PLANNING
  ├─成功────────> READY
  ├─可重试输入缺失> WAITING_FOR_INPUT
  └─确定失败────> HOLDING_FAILED
READY
  └─执行器接受──> EXECUTING
EXECUTING
  ├─滚动触发────> PLANNING_NEXT       # 仅轮式/足式
  ├─段完成──────> PLANNING_NEXT
  ├─偏离/地图变更> REPLANNING
  └─失败────────> HOLDING_FAILED
PLANNING_NEXT / REPLANNING
  ├─成功────────> READY 或 EXECUTING
  └─失败────────> 安全停车/保持并报告
```

新目标总是取消旧会话并生成新 session generation。迟到的 Action 结果和执行反馈只有同时匹配
session、`plan_id`、`segment_id` 和平台类型时才可生效。

现有 ROS `MotionReference` 没有轮式/足式独立的公开 `segment_id`。因此本设计冻结以下身份
映射，避免暗中扩展消息：每次发出的 `MotionReference.plan_id` 都是唯一执行参考 ID；ground
反馈同时令 `plan_id=segment_id=MotionReference.plan_id`；hopper 反馈令
`plan_id=MotionReference.plan_id`、`segment_id=HopSegment.segment_id`。跨局部段保持不变的
路线血缘使用内部 `RouteContinuation.route_id`，不伪装成 ROS 公共字段。

### 12.2 路线复用条件

轮式/足式只有同时满足以下条件才复用旧全局路线：

- mission、最终目标、平台类型和 capability version 完全相同；
- global map generation 未变化；
- 当前实际位置能投影到尚未执行路线的安全走廊内；
- 当前航向/位置偏差没有超过配置的复用阈值；
- 当前 local map 能覆盖下一局部前缀且没有新障碍切断它。

复用时从投影点推进 route cursor，只重新选择和认证局部段。任何条件失败都从真实当前位置对
同一最终目标执行完整全局重规划。不得从上一个局部段的名义终点假定起点。

跳跃式复用条件更严格：global map generation、capability version、mission/goal 必须相同，
且稳定着陆实际位置必须位于已认证的预期落区。满足时推进到下一条预认证 hop；否则从真实
着陆姿态对同一最终目标重规划。

## 13. 轮式与足式滚动规则

### 13.1 轮式

- 首次请求得到完整全局路线和当前局部轨迹；
- 当前实验默认局部视距仍为 4.0 m，但该数值是配置，不是正式平台能力；
- 当执行器报告 segment complete、平台到当前轨迹剩余弧长低于滚动阈值，或 local map 更新
  影响未执行前缀时，协调器发起下一次 `PlanMotion`；
- 平台仍在执行时允许计算并原子替换下一局部段；新轨迹必须从收到请求时的真实状态锚定；
- 全局路线复用失败时重新做完整 A*；新结果就绪前继续旧段仅限旧段剩余部分仍被确认安全，
  否则安全停车。

### 13.2 足式

- 使用与轮式相同的会话、身份和路线复用规则；
- 当前实验默认局部视距仍为 3.0 m，但不冻结为正式能力；
- 滚动触发包括 segment complete、接近局部前沿、身体状态偏离、地形/落足相关局部图更新；
- 新局部身体参考必须从真实身体姿态与速度开始，并经过足式局部运动学、地形和连续碰撞认证；
- 允许在当前段执行中准备/替换下一段，失败时按足式安全保持语义处理。

轮式和足式不是“一次授权后自动沿全局折线运动”。全局路线只指导连续生成局部物理可执行
轨迹，每个新段都由同一 C++ 局部后端认证。

## 14. 跳跃式滚动规则

跳跃式首次成功时已经得到完整落区链，并对链上每一跳保存名义抛物线与完整飞行管证书，
但只把第一跳写入 `MotionReference.hops`。其执行状态为：

```text
GROUND_HOLD -> JUMP_READY -> JUMP_COMMITTED -> IN_FLIGHT
            -> LANDED_HOLD -> JUMP_READY（下一跳）
```

规则如下：

- `JUMP_COMMITTED` 和 `IN_FLIGHT` 期间禁止替换参考、重定向落点或启动新规划；
- 执行器负责按已授权 segment 执行并发布状态，本项目用 Odometry 独立验证时间、位置和速度；
- 进入稳定 `LANDED_HOLD` 后，必须校验反馈的 `plan_id/segment_id`、实际位置位于预认证的
  `promotion_region`、
  稳定等待条件满足、global map generation 与 capability version 未变；
- 全部匹配时，route cursor 推进，下一条已认证 hop 被提升为唯一授权段；
- 实际落点越界、地图或能力变化时，废弃剩余证书，从真实稳定姿态到同一最终目标重规划；
- 反馈缺失、身份不匹配或飞行结束后状态无法确认时进入 unresolved/hold，不能猜测已经着陆；
- 最后一跳稳定着陆并位于最终目标区域后会话完成。

“一次只授权一个局部路径”对跳跃式意味着一次只允许执行一跳，不意味着只规划一跳。后续跳
已经在 continuation 中存在，只有稳定着陆门控通过后才逐跳授权。

## 15. 执行反馈接口与所有权

### 15.1 新增暂定输入

为避免从轨迹时间或 Odometry 猜测执行器是否接受、完成或拒绝一个段，新增外部所有的输入
Topic：

```text
/execution/motion_feedback
lunar_navigation_msgs/msg/MotionExecutionFeedback
```

暂定 schema：

```text
uint8 WHEELED=1
uint8 LEGGED=2
uint8 HOPPER=3
uint8 IDLE=0
uint8 ACCEPTED=1
uint8 EXECUTING=2
uint8 SEGMENT_COMPLETE=3
uint8 LANDED_HOLD=4
uint8 FAILED=5
uint8 CANCELED=6
std_msgs/Header header
uint64 sequence
uint8 platform_type
string plan_id
string segment_id
uint8 state
string reason_code
```

发布者属于外部运动执行/控制系统；本仓只暂定同名 schema、订阅、校验和适配。上游正式定义
后必须按既有 external-input 原子替换流程迁移，不能让两个同名包共存。该变更实施时同步更新
`docs/interfaces/external-input-baseline.md`、`external_interfaces.yaml` 和边界检查器。

### 15.2 校验与 QoS

- `header.frame_id` 固定为平台 `base_frame_id`，stamp 非零；
- `sequence` 在同一 `plan_id` 内从 1 开始严格递增；新 plan 可以重新从 1 开始；
- `plan_id`、`segment_id` 非空且与当前活动参考完全一致；
- 平台类型与 Lifecycle 配置能力一致；
- `LANDED_HOLD` 只对 hopper 合法，ground 使用 `SEGMENT_COMPLETE`；
- `reason_code` 在 FAILED/CANCELED 时必须非空；
- 使用 reliable、volatile、depth 10 QoS；过期或倒序反馈拒绝并发布诊断；
- 执行状态不能取代 Odometry。完成、着陆和偏离判断必须同时使用反馈身份与新鲜 Odometry。

`PlanMotionServer` 把最后一个已验证反馈映射为现有
`GroundExecutionContext`/`HopperExecutionContext`，替换当前硬编码的
`previous_execution = std::nullopt`。核心和 `ReferenceGuard` 对同一状态进行一致的 fail-closed
检查。

## 16. Action、并发与替换语义

`PlanMotion.action` 保持单次请求/单次结果：

- 外部协调器为每个滚动段生成唯一 `request_id`，mission 和最终 goal 保持不变；
- 轮式/足式滚动更新使用 `replace_active_request=true`，服务器仍保证单 worker 和旧请求取消后
  再激活新请求；
- 跳跃式在 committed/in-flight 时即使收到 replace 也返回
  `CONTINUE_COMMITTED_HOP`，不会开始另一个规划 worker；
- `LANDED_HOLD` 门控完成后，下一请求可以复用 continuation 并返回新的唯一 hop；
- 用户在 RViz 选择新终点时，协调器取消当前滚动会话；ground 可安全替换，hopper 必须等待
  已承诺跳跃结束后才应用新目标；
- 迟到结果不得覆盖更新后的 session generation。

路线 continuation 的内部存储不改变 ROS Action schema。进程重启会丢失缓存，此时从最新真实
状态做完整重规划，安全性不依赖缓存持久化。

## 17. RViz/ROS 实验行为

仓库外 `isaac_ros_action_regression` 的交互节点承担参考
`RollingGoalCoordinator + simulated executor`，本次实验与 Isaac Sim 解耦时使用合成地图和
模拟平台运动。它必须：

- 左侧面板只显示当前选中平台，并显示 session、plan、segment、execution state；
- 2D Goal Pose 只更新该平台的最终目标；
- 按当前局部轨迹或跳跃名义抛物线推进模拟 Odometry，并发布与 plan/segment 匹配的反馈；
- 自动触发并发送后续 `PlanMotion` 请求，直到到达最终目标、取消或确定失败；
- 全局障碍完整显示，局部危险只作额外高亮；
- 平台 Marker 与障碍使用不同 namespace、形状、颜色和文字标签，平台位置随 Odometry 更新；
- 轮式全局路线浅蓝、当前段深蓝；足式全局路线浅绿、当前段深绿；
- 跳跃式显示全部名义弹道、飞行管和落区，当前授权跳为高亮实线，后续预认证跳为浅色；
- 成功路线始于平台实际位置，最终进入用户设定的目标容差区域；
- 失败保留目标 Marker，显示 outcome/reason，并且不绘制虚假成功路线；
- 面板显示全局/局部/认证耗时、路线是否复用、候选边、完整飞行管认证和缓存命中。

为了显示未来 hop 的精确抛物线、飞行管和落区，同时保持 `MotionReference` schema 不变，
`PlanMotionServer` 增加规划器自有的 transient-local 可视化输出：

```text
/planning/certified_route_markers
visualization_msgs/msg/MarkerArray
```

Marker 由内部 continuation 生成，每个 Marker 使用自身正确的 `map` 或 `odom` Header，并在
namespace/文本中携带 route、reference 和 segment 身份。该 Topic 只提供可视化证据，执行器
不得消费它作为控制指令。目标替换、continuation 失效、失败或 Lifecycle deactivate 时，服务
器发布针对自身已创建 Marker ID 的 DELETE；不得清除其他节点的 Marker。

模拟执行器仅用于验证滚动协议和可视化，不能作为真实控制系统验收，也不能把模拟运动结果
写入正式能力资料。

## 18. 失败语义

| 情况 | PlanningOutcome | reason_code/处理 |
|---|---|---|
| ground A* 完整耗尽 Open | `NO_KNOWN_SAFE_ROUTE` | `GLOBAL_NO_KNOWN_SAFE_ROUTE` |
| hopper 完整惰性图与边认证耗尽 Open | `NO_KNOWN_SAFE_ROUTE` | `GLOBAL_NO_KNOWN_SAFE_ROUTE` |
| 目标区没有任何安全终点/落区 | `GOAL_INFEASIBLE` | 平台对应 `GLOBAL_GOAL_INFEASIBLE` |
| 输入地图超过冻结尺寸或层级 | `RESOURCE_EXHAUSTED`/`INVALID_REQUEST` | 保留现有 map scale/level 原因码 |
| 真实内存分配失败 | `RESOURCE_EXHAUSTED` | `GLOBAL_SEARCH_ALLOCATION_FAILED` |
| Action/stop token 取消 | `CANCELED` | `REQUEST_CANCELED` |
| 客户端 deadline 后取消 | `CANCELED` | `PLANNING_DEADLINE_CANCELED` |
| 抛物线无法得到确定数值结论 | `NUMERICAL_FAILURE` | `HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE` |
| 飞行管细分无法得到确定结论 | `NUMERICAL_FAILURE` | `HOPPER_FLIGHT_TUBE_NUMERICAL_INDETERMINATE` |
| 局部搜索/优化边界触发 | `RESOURCE_EXHAUSTED` 或 `NUMERICAL_FAILURE` | `LOCAL_SEARCH_INCOMPLETE` 或具体数值码 |
| 执行反馈缺失/过期/身份不匹配 | hold，不激活新参考 | `EXECUTION_FEEDBACK_*` |
| hopper 承诺或飞行中收到替换 | 保持当前执行 | `HOP_JUMP_COMMITTED`/`HOP_IN_FLIGHT` |
| hopper 落点偏离已认证落区 | hold 后重规划 | `HOP_LANDING_DEVIATION_REPLAN_REQUIRED` |

`GLOBAL_SEARCH_RESOURCE_LIMIT` 和 `HOPPER_GLOBAL_ROUTE_RESOURCE_LIMIT` 从新生产路径退役。历史
测试可以验证旧配置被拒绝，但新规划结果不得再产生这两个码。

## 19. 诊断与可观测性

在不修改 `PlannerDiagnostics.msg` 的前提下，内部指标和标准 `/diagnostics` 至少增加：

- `global_search_elapsed_s`、`local_planning_elapsed_s`；
- `landing_field_elapsed_s`、`spatial_index_elapsed_s`；
- `ballistic_solve_elapsed_s`、`flight_tube_certification_elapsed_s`；
- `global_expanded_nodes`、`open_peak`；
- `safe_landing_nodes`、`candidate_edges_evaluated`；
- `coarse_edges_rejected`、`full_edges_certified`、`full_edges_invalidated`；
- `edge_certificate_cache_hits`；
- `route_reused`、`route_cursor`、`rolling_request_count`；
- `active_plan_id`、`active_segment_id`、`execution_state`；
- 每阶段 p50/p95 基准结果和构建类型。

诊断计数是观测信息，不得被重新用作隐藏终止条件。Release/Debug 必须在诊断和性能报告中
显式标记，避免再次把调试构建耗时误当作生产性能。

## 20. 构建与性能门槛

### 20.1 Release 要求

仓库内权威 Ubuntu 构建和仓库外 RViz 实验构建均显式传入：

```text
-DCMAKE_BUILD_TYPE=Release
```

构建脚本必须把构建类型写入 manifest/日志。性能测试检测到非 Release 时直接跳过并报告
`PERFORMANCE_BUILD_NOT_RELEASE`，不能产生可比较的性能结论。

### 20.2 基准方法

固定基准使用 50 m × 50 m、0.2 m、250 × 250 栅格的确定性地图。每个 fixture 预热后运行
30 次，报告 p50、p95、最大值、展开节点、候选边、认证次数、缓存命中和峰值常驻内存。
计时范围从核心接收完整 `PlannerInput` 到返回 `PlannerOutput`，另记录各阶段拆分。

本轮门槛：

| 用例 | Ubuntu Release p95 |
|---|---:|
| 轮式确定性正例 | ≤ 2.0 s |
| 足式确定性正例 | ≤ 2.0 s |
| 跳跃式直接一跳正例 | ≤ 1.0 s |
| 跳跃式代表性多跳正例 | 目标 ≤ 2.0 s，硬门槛 ≤ 5.0 s |
| 跳跃式完整无路反例 | ≤ 5.0 s |

测试能力文件必须放在 test fixture 目录并标注
`test-only/non-authoritative`。它们用于构造已知可达与不可达几何，不能覆盖或更新生产能力文件。
AGX Orin 的正式门槛必须在设备上另行测量；Ubuntu 结果不能冒充设备验收。

## 21. 测试策略

### 21.1 单元与性质测试

- ground A* 在最坏蛇形通道中展开超过旧上限仍成功；
- ground 完整障碍隔断只有 Open 耗尽后返回无路；
- 取消在建场、搜索和路线重建各阶段均返回 canceled；
- 跳跃空间索引查询结果与小地图全枚举结果完全一致；
- 跳跃邻接包含已发现非目标节点，不截断出度；
- 直接跳优先，失败边缓存后仍可找到多跳替代；
- 名义抛物线端点、速度、冲量、飞行时间和高差约束逐项验证；
- 飞行管检测中间月岩、掠地段、顶部障碍、起跳段和着陆段；
- 确定碰撞边 invalidation 后 A* 恢复并找到替代路线；
- 数值不确定返回 numerical failure，不返回 no-route；
- 相同输入多次运行路线、代价、诊断计数和原因码一致；
- 输出路线首点严格锚定真实起点，终点满足配置化目标容差。

### 21.2 滚动状态机测试

- wheel/legged segment complete 自动请求下一段；
- ground 剩余弧长触发时允许安全替换，迟到结果被 session generation 丢弃；
- 小偏差复用全局路线，大偏差或 global map generation 变化强制重规划；
- hopper committed/in-flight 拒绝替换；
- hopper `LANDED_HOLD`、身份、落区和稳定条件全满足时才授权下一跳；
- hopper 越界着陆从实际姿态重规划，不沿用名义落点；
- 执行反馈过期、倒序、平台不匹配、plan/segment 不匹配均 fail closed；
- Lifecycle deactivate、目标替换和任务 revision 变化清空 continuation。

### 21.3 ROS 与 RViz 自动化回归

- Action schema 保持不变，执行反馈新 schema 与外部接口基线一致；
- `PlanMotionServer` 不再硬编码 `previous_execution=nullopt`；
- 三个平台都能在 RViz 选最终目标并自动滚动到终点；
- RViz 只显示当前平台，平台 Marker 随 Odometry 移动；
- 路线、当前段、跳跃弹道、飞行管、落区、障碍和失败目标可区分；
- 三平台正反例固定起终点，不在测试时扫描“容易成功”的目标；
- 旧六案例、Action cancel/replace、地图/TF 新鲜度和 ReferenceGuard 回归继续通过。

## 22. 实施顺序与原子切换

实施按以下依赖顺序进行，详细文件级任务在设计批准后的实施计划中给出：

1. 先添加会失败的完整搜索、拓扑、弹道/飞行管和性能测试；
2. 强制 Release 构建，建立未修复基线；
3. 移除 ground 全局固定计数终止并验证完整 A*；
4. 重构跳跃建场、空间索引、完整惰性邻接和 direct-first；
5. 实现抛物线求解、全飞行管认证、边缓存与 invalidate–resume；
6. 增加 continuation、路线存储和 execution feedback 适配；
7. 实现三平台滚动协调器及 RViz 动态执行展示；
8. 运行核心、ROS、仓库边界、外部桥和 30 次 Release 性能回归；
9. 在同一切换中删除旧资源码生产路径和旧截断配置。

新旧跳跃算法不得通过运行时 fallback 并存。切换提交必须原子更新核心、配置 schema、ROS
适配、测试和仓库外实验脚本；任何阶段失败都保留旧可构建版本，不发布部分迁移状态。

## 23. 验收条件

1. 三个平台的全局规划不会因固定 expanded/generated/open/node/outdegree 计数提前结束。
2. `GLOBAL_NO_KNOWN_SAFE_ROUTE` 只来自当前有限离散域的完整 Open 集耗尽。
3. 原先因 64/128/8 截断失败的跳跃多跳 fixture 能规划成功，且不靠放宽目标容差。
4. 跳跃每条成功边都有满足能力约束的名义抛物线和完整平台飞行管证书。
5. 跳跃路线完整规划并认证，但任何时刻只授权一跳；下一跳只在稳定
   `LANDED_HOLD` 后推进。
6. 轮式和足式在局部段执行过程中按真实状态滚动生成其余局部段，并在允许条件下复用全局
   路线。
7. 地图、能力、任务或落点偏差导致证书失效时，从真实状态重规划，绝不沿用过期 continuation。
8. RViz 可从单次用户选点持续显示平台、全局路线、当前段、跳跃弹道/飞行管/落区和最终结果。
9. Ubuntu Release 的 30 次基准满足第 20 节门槛，且诊断能解释时间消耗位置。
10. 正式平台能力文件没有被本轮测试参数修改或重新定义。
11. 仓库边界测试、核心/ROS 单元测试、Action 回归和外部 RViz 回归全部通过。
