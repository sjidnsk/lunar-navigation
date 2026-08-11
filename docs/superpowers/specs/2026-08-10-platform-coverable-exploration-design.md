# 平台运动原语可达探索与候选终止闭环设计

> **部分取代说明（2026-08-11）：** 本文中以 truth/observed 运动原语图作为覆盖率分母、候选和
> oracle 权威的条款，`PLANNER_REJECTED_ALL` 终止语义，以及 cache v5 / training semantics v10
> 身份，已由
> [`2026-08-11-physical-opportunity-planner-failure-semantics-design.md`](2026-08-11-physical-opportunity-planner-failure-semantics-design.md)
> 原子取代。本文未被新设计明确修改的 source/split、传感器、宏动作、路径观测、成功阈值和奖励边界继续有效。

**日期：** 2026-08-10
**修订：** 2026-08-11
**状态：** 已部分取代；未被 2026-08-11 修订明确修改的条款继续有效
**范围：** 正式 PPO 训练的覆盖率分母、场景资格、平台运动原语可达性、候选生成、候选耗尽审计、宏动作闭环和全新训练身份

## 1. 权威关系

本文原子取代以下现行设计中与覆盖率和候选终止冲突的部分：

- `2026-08-09-unbounded-formal-episode-design.md` 中“共同可行动子集”、以完整任务 ROI
  作为覆盖率分母、随机安全起点和无候选自然终止的定义；
- `2026-08-08-formal-polar-training-environment-closure-design.md` 中 cache v3 的起点资格、
  完整 ROI 覆盖累计和三平台共同 eligible 约束；
- `2026-08-10-platform-aware-candidate-reachability-design.md` 中 Hopper 只做静态落点预检、
  修复前 checkpoint 可严格续跑以及不改变训练语义的结论。

未被本文明确取代的 source/split、确定性场景生成、`1024 m × 1024 m` 全局画布、
`4.0 m` 网络全局分辨率、`0.2 m` 精细观测、`30 m/360°` 传感器、无固定 episode
决策上限、共享 PPO 网络和 C++ v3 最终安全裁决继续有效。

本文参考旧仓冻结提交 `7309e93fdb85c60ff3736efe1a7f3c7eb640ee78` 中
`exact_reachable_safe_pose_range_los/v1` 的核心原则，但不复制其地面二维 BFS：三平台必须使用
当前 capability v2 和当前 C++ v3 的平台语义。

## 2. 问题与结论

当前正式环境使用如下任务区域：

```text
mission_roi = valid_mask - 64m 边界 - forbidden
```

该区域同时被当作候选任务域和覆盖率分母，却没有绑定平台、起点、可达性、传感器距离或 LOS。
起点资格只证明初始 reveal 后至少有一个候选。候选生成随后使用另一套平台预过滤；下一候选为空时
环境直接结束 episode。因此系统没有保证：

```text
覆盖率分母中的每个栅格
都能从该平台、该起点的某个合法可达观测位置被传感器看到。
```

修复结论是同时建立四个相互独立的合同：

1. `mission_target_detail_mask`：任务希望覆盖且可作为成功目标的 `0.2 m` 自由栅格；
2. `truth_primitive_reachability_graph`：该平台从冻结起点按正式运动原语可达且可恢复的状态图；
3. `reachable_pose_mask`：上述状态图投影得到的 `4.0 m` 可达观测位置，而不是独立二维连通判定；
4. `coverable_detail_mask`：能从某个可达观测状态在 `30 m` 和 LOS 合同内看到的任务目标栅格。

`coverable_detail_mask` 是成功分母。运行时另行维护同一算法的
`observed_primitive_reachability_graph`，候选只能从该 observed-only 状态图生成，不得读取真值图或
真值掩码。

### 2.1 新旧源码闭环审计结论

冻结旧仓能达到 `0.99` 的关键不是单独某个阈值，而是同一套语义贯穿候选、规划、执行和观测：

- 一个 action 选择一个候选目标，规划器执行到该目标，而不是每走 `3-4 m` 就丢弃目标；
- 沿整条已认证路径按 `<= 1.0 m` 间距进行观测；
- 预计信息增益允许射线穿过未知区域，只由已经观测到的障碍截断；
- 候选增益按当前候选集归一化，策略实际看到的是可比较的 `O(1)` 信号；
- 成功阈值与成功奖励分别为 `0.99` 和 `100.0`。

当前实现的失败证据与之相反：场景 `b230277e...` 在第 241 个决策边界只有
`6.419432%` 覆盖；机器人所在 `4 m` 单元已有 `378/400` 个精细格证据，局部 `0.2 m`
投影确认起点安全连通，但粗图按合同仍是 unknown，导致所有正式请求被
`GLOBAL_NO_KNOWN_SAFE_ROUTE` 拒绝。此前 20 步中候选目标平均距离为 `17.54 m`，实际每次只前进
`3.94 m`，目标在滚动参考后被重新选择。首步预计增益只相当于实际目标观测增量的约 `1/80`，
原因是预计增益把未知当成射线阻挡并仅计算未知边界薄壳。

因此，`coverable_detail_mask` 的物理定义保留；不得通过缩小分母掩盖失败。剩余修复必须使运行时
规划、动作、观测和预计增益与该分母闭合，并由自然终止闭环而不是训练曲线证明。

## 3. 精确可探索栅格合同

### 3.1 任务目标自由栅格

每个平台分别、逐 `64 m × 64 m` 精细瓦片计算：

```text
mission_target_detail_mask(platform) =
    inside_formal_mission_roi
    AND detail_valid
    AND forbidden == false
    AND physical_hard_obstacle == false
    AND platform_intrinsic_terrain_feasible
```

`platform_intrinsic_terrain_feasible` 使用当前 capability v2 与 C++ `SafeProjection` 的内在地形
限制，包括对应平台的坡度、粗糙度、支撑和落脚面限制，但不包含“机器人中心必须能安全驻留于此”
所需的机体净空。驻留净空由原语状态/边认证决定，`reachable_pose_mask` 只记录其空间投影。

已批准的成功语义固定为：

- 硬障碍和坡度/地形禁行栅格可以被传感器观测并更新 observed evidence；
- 它们不进入覆盖成功分母，也不产生任务覆盖增量；
- 未知真值不得通过该分类进入策略输入或候选预计增益。

### 3.2 平台运动原语可达状态图

二维 `hard_feasible + connected_component` 只能证明栅格拓扑连通，不能证明平台能够以合法朝向、
运动模式、机身高度或跳跃包络到达。因此它只能作为廉价的静态剪枝，不能再作为覆盖率分母或候选
可达性的权威。

每个平台使用 C++ 正式规划器的同源运动原语构建有向状态图：

```text
G(platform, world_evidence, start_state, capability) = (states, certified_edges)
forward_states    = states reachable from start_state
returnable_states = states that can reach the frozen safe anchor
recoverable_states = forward_states AND returnable_states
```

状态不得过早压缩成二维栅格：

- WHEELED 状态至少包含离散位置、`yaw_bin`、`START/FORWARD/REVERSE` 运动模式，并保留连续代表位姿；
- LEGGED 状态至少包含离散位置、`yaw_bin`、可达机身高度区间和支撑状态；
- HOPPER 状态包含精确落点位姿及已认证落区。当前 capability 没有累计燃料，因此暂不包含剩余燃料；
  一旦引入累计 `delta-v`，剩余资源必须进入状态并升级算法身份。

边必须调用与正式 Planner 同源的原语生成及硬约束认证：WHEELED/LEGGED 使用对应 lattice 的扫掠
净空、坡度、粗糙度、支撑、朝向和运动模式约束；HOPPER 使用落区、`delta-v`、ballistic envelope、
flight tube、forbidden、净空和落脚支撑认证。Python 不得复制一套近似运动学规则。

真值 `truth_primitive_reachability_graph` 使用完整冻结 truth 和冻结起点，允许通过任意数量的合法
原语覆盖整幅场景；`30.0 m` 只约束单次策略候选和传感器，不得截断真值多步可达图。图绑定：

```text
scene_id
platform_type
qualified_start_state
capability_sha256
v3_source_commit
primitive_state_schema
primitive_set_sha256
world_evidence_sha256
global_and_detail_geometry
reachability_algorithm_id
```

`reachable_pose_mask[row,column]` 仅是 `recoverable_states` 的空间投影：只要至少一个可作为观测位姿
的可恢复状态落入该 `4.0 m` 单元，该位为真。权威仍是带朝向和平台内部状态的图；同一单元中一个
状态可达，不能推导所有朝向或模式都可达。候选必须携带原图中的精确 `x/y/z/yaw` 和状态身份，不能
从二维 mask 的格心重新构造目标。

首版算法与状态 schema 身份固定为：

| 平台 | `reachability_algorithm_id` | `primitive_state_schema` |
|---|---|---|
| WHEELED | `cpp-wheel-motion-primitive-recoverable-graph/v1` | `wheel-lattice-state/v1` |
| LEGGED | `cpp-legged-motion-primitive-recoverable-graph/v1` | `legged-lattice-state/v1` |
| HOPPER | `cpp-hopper-certified-recoverable-state-graph/v4` | `hopper-landing-state/v2` |

truth 与 observed-only 实例使用相同 ID；`world_evidence_sha256` 和 snapshot revision 区分输入证据。
图 canonical hash 按状态 canonical tuple 的字典序写状态记录，再按
`(source_state_id, primitive_id, target_state_id)` 写边记录。整数使用定宽 little-endian，有限浮点使用
IEEE-754 binary64 little-endian，`-0.0` 归一为 `+0.0`，字符串使用 UTF-8 长度前缀；任何非有限值
使 materialization 硬失败。`primitive_set_sha256` 对原语定义的 canonical stream 计算；
`reachability_graph_sha256` 对包含该 primitive-set hash 的状态/边 stream 计算。Python 只校验
结果，不重新定义序列化。

可恢复语义保证进入成功分母的观测位置能由同一探索轨迹访问并回到冻结安全锚点。只单向可达的状态
不进入分母或运行时候选。数值不确定、取消或资源耗尽使场景—平台资格计算失败，不能降级为普通
不可达边。实现必须进行批量多目标图展开；禁止为每个栅格单独调用一次完整 Planner。

### 3.3 可覆盖精细栅格

对每个 `0.2 m` 任务目标栅格，若至少存在一个 `recoverable_states` 中的精确观测位姿满足：

```text
distance <= 30.0 m
AND two_dimensional_detail_los/v1 可见
```

则该目标栅格进入 `coverable_detail_mask`。LOS 使用与真实 reveal 相同的 `0.2 m` 障碍真值和
射线离散规则。正式传感器为 `360°`，因此 yaw 不改变单个观测位姿的可见扇区，但 yaw 和运动模式
仍决定该位姿是否能由运动原语到达。执行后的实际 reveal 继续使用完整传感器合同。

正式覆盖率唯一计算式为：

```text
coverage =
    popcount(observed_detail_mask AND coverable_detail_mask)
    / popcount(coverable_detail_mask)
```

分母必须大于零。初始 reveal 计入 numerator，但不得在初始状态达到当前批准的 `0.95`；首次从
`<0.95` 跨到 `>=0.95` 仍是唯一 `SUCCESS` 事件。旧仓的 `0.99` 能力保留为后续诊断目标，
不作为本轮恢复训练的成功门槛。

## 4. 真值冻结图与 observed-only 增量图

系统维护同一 C++ `PrimitiveReachabilityEngine` 的两个隔离实例。两者必须使用相同平台状态 schema、
运动原语集、能力资料、边认证和确定性排序，只允许 world evidence 不同：

1. `truth_primitive_reachability_graph` 使用完整 truth、冻结起点和冻结 capability，离线完整计算并随
   cache 身份冻结；
2. `observed_primitive_reachability_graph[t]` 只使用时刻 `t` 已观测的 `0.2 m` 证据、保守 global
   聚合和当前执行状态，随每次 reveal 增量更新。

真值图、其 `reachable_pose_mask`、`coverable_detail_mask` 及剩余未覆盖真值只能用于：

- cache 资格和完整性校验；
- 覆盖率 numerator/denominator；
- reward 与成功事件；
- 环境诊断和正式评估。

以下路径禁止读取真值图或真值 coverable mask：

- `prior_channels`、`coverage_summary` 的稠密策略输入；
- frontier 提取、可达状态查询和候选落点生成；
- 候选预计信息增益；
- 候选排序或策略 action mask；
- observed-only opportunity oracle。

运行时 observed-only 图只能用于候选、action mask、oracle 和诊断，不能修改已经冻结的覆盖率分母。
否则分母会随机器人观测顺序增长或收缩，使 coverage 非单调并允许策略通过改变自身可达域来“提高”
分数。相反，真值图不得参与候选空间定位，否则会泄漏未知地形。

现有 `pose_features[:,4]` 继续向策略提供一个累计覆盖率标量，但改为本文的精确 coverable
分母。它只透露机器人已取得的任务进度，不透露未观测区域的空间位置。候选预计信息增益继续使用
机器人当时已经掌握的稀疏 `0.2 m` 观测地图，不使用完整 detail truth。未知格不得被当作运动可通行
证据，但预计传感器射线可以穿过未知格，直到遇到已观测物理障碍。

## 5. Cache v5 与固定起点

正式 cache schema 升级为 `lunar-formal-training-cache/v5`。旧 v3/v4 cache 不可原地修改或作为新
训练输入；必须在仓库外新目录完整重建。

每个场景按平台保存：

```text
qualified_start_state                   platform-specific canonical state
reachable_pose_mask                    uint8/bit-packed [256,256]
coverable_detail_mask                  uint8/np.packbits row-major [5120,5120]
coverable_ratio                        float32 [256,256]
primitive_state_count                  int
certified_edge_count                   int
recoverable_state_count                int
mission_target_detail_cell_count       int
coverable_detail_cell_count            int
mission_coverable_fraction             float
initial_coverable_fraction              float
reachability_algorithm_id              string
primitive_state_schema                 string
primitive_set_sha256                   sha256
reachability_graph_sha256              sha256
visibility_algorithm_id                string
reachable_mask_sha256                  sha256
coverable_mask_sha256                  sha256
exact                                  true
eligible                               bool
ineligible_reason                      enum|null
```

`ineligible_reason` 固定为：`UNSAFE_START`、`ZERO_MISSION_TARGET`、
`MISSION_COVERABLE_BELOW_95`、`INITIAL_ALREADY_SUCCESS` 或 `NO_INITIAL_CANDIDATE`。
资格计算只能写入其中一个最先发生的普通任务不合格原因，并保留各阶段诊断计数；不得使用自由文本
决定调度。reachability/visibility 数值不确定、取消、资源耗尽或 mask identity/hash 错误不是
普通不合格场景，必须中止整个 cache materialization，不能发布仍标记 formal eligible 的 manifest。

`reachability_graph_sha256` 对按平台稳定状态键和稳定边键排序后的 canonical stream 计算；cache 不必
持久化完整状态图，但必须保存图计数、投影 mask 和图身份，使实现无法悄悄退回二维连通分量。

`coverable_ratio[row,column]` 等于该 `4.0 m` 单元内 `20 × 20` 个精细子格中 coverable 子格的比例，
仅供汇总、审计和报告；实际 numerator 必须与 bit-packed 精细掩码逐位相交，不能用比例近似部分
观测。

完整 `5120 × 5120` 多层 truth 不常驻内存。预计算逐 `64 m` tile 执行，只保留一个 tile 的工作
数组和约 `3.3 MB/platform/scene` 的最终 bit-packed 掩码；manifest 记录压缩前语义哈希、文件
大小和内容哈希。

### 5.1 起点与资格

为避免起点相关掩码失效，每个场景—平台 v1 只绑定一个确定性起点：继续使用当前按距画布中心、
行、列排序后找到的第一个安全且初始 observed-only 原语图能生成候选的 canonical 起点状态。正式
worker 必须始终使用该缓存起点状态，不再先尝试随机安全起点。增加多起点需要新的 start profile
schema，不在本文范围。

canonical 起点状态沿用当前正式环境并冻结为：`x/y` 是合格 `4.0 m` 单元格心，地形 `z` 来自该格，
`yaw=0.0`；WHEELED 的 motion mode 为 `START`；LEGGED 的 body `z` 为地形 `z` 加 capability
`body_height_m` 区间中点，并携带该起始可达高度区间；HOPPER 使用格心附近通过精细落区认证的实际
aim pose。任一字段变化都必须改变 start-state identity 和图 hash。

场景—平台 `eligible=true` 必须同时满足：

1. 起点状态安全且属于 recoverable states；
2. `coverable_detail_cell_count > 0`；
3. `mission_coverable_fraction = coverable / mission_target >= 0.95`；
4. `initial_coverable_fraction < 0.95`；
5. 初始 reveal 后 observed-only 原语图至少能生成一个合法候选；
6. 原语状态图、detail target、LOS 和掩码哈希全部精确完成。

该 `0.95` 资格门防止只覆盖任务区域中的小孤岛却轻易获得成功。被排除的场景仍保留在 source
和 scenario manifest 中，并带明确平台化 `ineligible_reason`，不得删除或改写原始数据。

### 5.2 调度与报告

训练 lane 使用各平台自己的 eligible 场景集合和确定性 permutation，不再要求三平台共享同一
训练场景子集。场景 schedule identity 必须绑定平台、eligible scene ID 列表及其哈希。

跨平台同场景比较使用三平台 eligible 集合的交集。正式报告同时给出：

- 每个 split×platform 的总场景数、eligible 数和 feasibility rate；
- eligible 场景上的 policy success rate；
- 每个 ineligible reason 的数量；
- 三平台 exact-common 评估分母。

不得只报告 eligible success rate 而隐藏平台对完整任务库的 feasibility rate。JAXA holdout 中
不满足资格的场景保留为显式 task-infeasible evidence，不通过换 seed 或降低阈值挪入评估。

## 6. 运行时 observed-only 原语图与候选生成

运行时流程从“先生成 frontier 候选、再做平台过滤”改为“先更新平台可达状态图、再从图中生成
候选”。frontier 只可作为查找信息机会的空间索引或排序提示，不能创建不在原语图中的目标。
`[64,12]` 张量形状、动作索引和稳定排序合同保持不变。

统一 C++ 边界为：

```text
PrimitiveReachabilityEngine.update(
    observed_world_revision,
    current_platform_state,
    frozen_safe_anchor,
    platform_capability,
    maximum_action_distance_m_or_none,
) -> PrimitiveReachabilitySnapshot

CandidateBuilder.build(
    primitive_reachability_snapshot,
    observed_detail_map,
    frontier_hints,
    visited_state_ids,
) -> CandidateBatch[64] + CandidateDiagnostics
```

`PrimitiveReachabilitySnapshot` 至少包含稳定状态 ID、精确代表位姿、已认证边、从当前状态正向可达
标记、能返回冻结安全锚点的反向可达标记、路径代价、图 revision、算法身份和增量更新计数。二维
reachable mask 只用于快速空间查询和可视化，不能替代状态 ID。

每次 reveal 后按以下顺序增量更新：

1. 引擎比较前后 world revision 的定坐标 tile 内容哈希，根据变化的 `0.2 m` tile 找出扫掠包围盒
   与之相交的原语边，并使这些边失效；调用方不得用遗漏变化的提示绕过该比较；
2. 对新增已观测安全区域附近的状态和边执行正式平台原语认证；
3. 保留未受影响且输入证据哈希未变的认证结果；
4. 在更新后的有向图上确定性重算 forward/reverse reachability 标签；
5. 发布新的不可变 snapshot 后，才允许构建候选。

首版允许增量维护边、但每个 revision 重新执行图上的 BFS/反向 BFS；图遍历通常远低于重新认证
全部原语边的成本，也比一开始实现动态 SCC 更容易证明正确。边 cache 必须以平台 capability、原语
ID、状态 schema、地图 tile 内容哈希和图 revision 为键；禁止跨不兼容 revision 复用。

运行时 `4 m` global 规划格仍只有在所属 `20 × 20` 个 `0.2 m` 细格全部已观测后才置为 known。
但 local 已观测证据可以直接认证精细运动原语，不能因其所属 global 格只有 `399/400` 个细格已知就
清零。实现由 local 原语子图、global 原语子图和同源认证的 portal 边拼接；禁止无条件相交两个二维
connected-component mask。HOPPER 的精细落区证据同样优先于粗格近似，flight tube 仍使用其需要的
完整 observed evidence。未知格不能作为可通行证据。

候选生成固定为：

1. WHEELED/LEGGED 查询相对当前精确位姿在 `30.0 m` 动作包络内、同时 forward reachable 与
   returnable 的状态；HOPPER 只查询当前状态的一条已认证出边所到达且可返回的直接后继，不能把
   truth/observed 图中的多跳远端节点冒充为一个策略动作；
2. 只保留可作为安全观测位姿的精确状态，并用其 `30 m/360°` footprint 查找未观测任务 ROI；
3. 使用当前 `0.2 m` observed map 计算预计 mission/priority gain；
4. 对正增益状态按空间分段、路径代价和稳定状态 ID 去重排序；
5. 正增益候选稀少时继续扫描可达图的其他距离层和观测位姿，而不是放宽平台约束；
6. 若仍有 observed-only frontier、但到达新信息区域需要先经过零即时增益状态，允许加入可恢复中转
   状态；直接父落点回退必须使用实际执行过的精确状态；
7. 最终截断到 64 个真实候选；不足部分使用 `candidate_mask=false` 填充，不伪造重复或不可达目标。

候选始终携带精确 `x/y/z/yaw`、平台状态 ID 和生成它的图 revision。正式 `Planner::Plan()` 继续
拥有最终 reference 安全裁决权，以覆盖连续目标容差和最新地图 revision；但使用同源原语后，规划器
拒绝必须记为生成器—执行器一致性诊断，不能静默当作普通 frontier 消失。运行时 C++ 数值失败、取消
或资源耗尽属于基础设施错误，不得写成候选不可达并让 episode 正常结束。

### 6.1 地面平台的局部—全局原语连接权威

WHEELED/LEGGED 的正式 Planner 不得在 local 原语图已证明当前精确状态安全时，仅因当前所属 `4 m`
单元未达到 `400/400` 观测就拒绝请求。全局图继续保持全观测才 known 的保守规则，不伪造粗格
证据；Planner 与 reachability engine 共享以下连接逻辑：

1. 从当前精确状态展开 local 运动原语，得到可认证 local 状态；
2. 找到能由一条已认证原语边进入 global 状态图的稳定 portal 状态；
3. 按原语路径代价和稳定状态 ID 排序 portal；
4. 组合 local 原语段、global 原语段和最终 local 目标段；
5. 没有任何完整原语链连接目标时，才返回 `GLOBAL_NO_KNOWN_SAFE_ROUTE`。

conditional corridor 仍只作用于对应 global 段。二维局部连通分量可做候选 portal 的静态剪枝，但
不能证明 portal 边或生成执行 reference。

### 6.2 一个策略动作对应一个持续候选目标

WHEELED/LEGGED 的一个 PPO action 固定候选的精确 `x/y/z`、目标容差和 theta。C++ 每次仍只发布
`wheel_horizon_m=4.0` 或 `legged_horizon_m=3.0` 的安全滚动 reference；每段执行后环境更新真实
观测地图，并在不再次调用策略的情况下对同一目标重新规划。满足以下任一条件才回到下一策略边界：

- 目标容差满足；
- `SUCCESS` 或硬终止发生；
- 刷新地图后该目标被正式 Planner 明确判为不可行或无安全路径。

所有内部 reference 的覆盖增量、优先级增量、规划/执行代价、时间和执行事件聚合成一个
`PlannerTransition`。目标坐标不得从刷新后的候选索引重建。最多允许 64 个内部 reference，且每段
必须产生有限、可验证的目标距离进展；违反者是环境不变量错误并丢弃 rollout，不是普通 episode
失败。该上限是单个 `<=30 m` action 的防失控保护，不是 episode 决策上限。Hopper 保持一个 action
对应一个已认证单跳，直到 `LANDED_HOLD`。

### 6.3 沿已认证路径累积真实传感器观测

WHEELED/LEGGED 对执行 reference 的折线按累计路程插值，观测样本间距不得超过 `1.0 m`，并始终
包含最终位姿。每个样本使用与实际 reveal 相同的 `0.2 m` truth、30 m/360° 和 LOS；样本间 elapsed
按 reference 时间戳插值。一个内部 reference 只发布一次最终策略 observation，但其覆盖增量是所有
路径样本新观测精细格的去重并集。replay state 必须保存完整路径样本和逐段 elapsed，以便恢复后
逐位重建相同 observed bits。Hopper 仍只在 `LANDED_HOLD` 观测，飞行中不得采样。

### 6.4 预计信息增益与归一化

候选预计增益只使用机器人当时掌握的 `0.2 m` 稀疏观测地图：候选必须是已观测安全位姿；对传感器
圆盘内所有未观测 ROI 端点，射线可以穿过未知格，仅在遇到“已观测且为物理障碍”的格时截断。
因此它是 observed-only 的乐观预计，不是真值 reveal，也不读取 coverable mask。

从 observed-only 原语图生成候选后，`frontier_features[:,5]` 使用本批正增益候选的最大 mission gain 归一化，
`frontier_features[:,6]` 同样按最大 priority gain 归一化；最大值为零时保持零。原始增益继续用于
零增益诊断，稳定排序仍使用确定性 tie-break。该变化不改变 `[64,12]` 形状，但属于策略输入语义
变化，必须升级训练语义并从新模型开始。

### 6.5 64 项动作容量与可达候选储备

64 是动作张量容量，不是每轮必须达到的真实候选数量。生成器必须先遍历当前动作包络内所有可恢复
原语状态的信息机会，再按确定性空间分段截断；不能因为已找到一个 frontier 点就停止枚举。正增益
候选少于 8 个时，按距离层继续补入其他正增益可达观测状态，再补入 6 节定义的可恢复中转状态。

不足 64 项时其余槽位必须为 `candidate_mask=false`。规划器拒绝一个候选时只屏蔽该候选，继续使用
同一 snapshot 中尚未尝试的候选；只有全部真实候选都被拒绝时才允许进入
`PLANNER_REJECTED_ALL`。不得调用 oracle 候选填充策略 batch，也不得通过复制同一状态掩盖真实
候选不足。

## 7. 候选耗尽与独立 Oracle

`CandidateDiagnostics` 扩展为分阶段计数：

```text
primitive_state_count
forward_reachable_state_count
returnable_state_count
recoverable_observation_state_count
frontier_hint_count
visited_excluded_count
zero_gain_count
positive_gain_state_count
transit_state_count
emitted_count
planner_rejected_count
invalidated_edge_count
revalidated_edge_count
```

新增 observed-only `PrimitiveOpportunityOracle`，独立判断当前已知地图上是否仍存在合法探索机会：

- oracle 可以读取同一 snapshot 的 certified edges，但必须独立重算 forward/returnable 状态和信息机会；
- 三个平台都从可恢复原语状态枚举能看到未知 ROI 的精确观测位姿，并覆盖必要的中转机会；
- oracle 不调用生产 CandidateBuilder、生产去重或生产排序，不读取 truth 图或 coverable mask，也不把
  结果输入策略或 reward。

候选为空时必须区分“生产端从未发出候选”和“生产端候选已被规划器逐个拒绝”。执行以下状态机：

```text
producer_emitted_count == 0 AND oracle_opportunity_count > 0
    -> ORACLE_CONTRADICTION
    -> raise EnvironmentInvariantError
    -> 不进入 rollout

producer_emitted_count == 0 AND oracle_opportunity_count == 0
    -> LEGAL_EXHAUSTION(<最后实际耗尽候选的阶段>)
    -> 未达到 0.95 时按该阶段原因合法失败终止

producer_emitted_count > 0
AND planner_rejected_count == producer_emitted_count
    -> PLANNER_REJECTED_ALL
    -> 未达到 0.95 时作为可审计探索失败终止

coverage 首次达到 0.95
    -> SUCCESS
```

规划器逐个拒绝全部候选时仍运行 oracle，但此时 oracle 只证明 observed-only 原语图中的机会，不等价于
完整规划器能够生成 reference；其正数结果作为诊断保留，不得反过来误报为生产生成器为空。
只有 `producer_emitted_count == 0` 时，oracle 正数才构成真正的生成器矛盾。硬安全、合同错配和
非有限数值继续走 `HARD_FAILURE`，不能与普通探索耗尽合并。

每个终止 episode 必须携带一个完整主原因：`SUCCESS`、
`NO_RECOVERABLE_OBSERVATION_STATE`、`VISITED_EXHAUSTED`、`ZERO_GAIN`、
`NO_TRANSIT_OPPORTUNITY`、`PLANNER_REJECTED_ALL`、`HARD_FAILURE` 或 `CANCELED`。除
`PLANNER_REJECTED_ALL` 外，
`oracle_opportunity_count=0` 是允许候选耗尽原因成为 terminal 的审计事实，不是另一个主原因；
`PLANNER_REJECTED_ALL` 可携带正 oracle 计数以说明“有几何机会、但完整规划失败”。若多个候选阶段
同时为空，使用最晚一个实际消耗候选的阶段作为主原因，同时保留全部计数。

环境可记录 `remaining_unobserved_coverable_count` 作为 truth-side 诊断，但不能用它生成候选或
替代 observed-only oracle。

## 8. Reward、checkpoint 与全新训练身份

奖励升级为 `lunar-reward/v4`。仍保持 coverage-first，固定共享权重为：

```text
mission_observed_delta                  100.0
executed_without_new_mission_coverage    0.10
goal_infeasible                          0.20
no_known_safe_route                      0.30
unsuccessful_remaining_coverage          1.00
success_first_crossing                  100.0
```

`normalized_plan_or_execution_cost`、`normalized_macro_step_time` 和
`priority_observed_delta` 继续作为有限性合同与诊断，不进入本轮覆盖优先奖励；这样不在修复覆盖闭环
时混入新的效率目标。成功奖励恢复到旧仓量级，避免最后几个百分点只得到弱于普通覆盖增量的信号。

cache、候选特征、动作聚合、成功阈值、reward 和 replay state 均发生语义变化，因此旧 checkpoint
既不能严格 resume，也不用于 warm start。本次正式运行必须：

- 随机初始化完整 policy 与 value 网络；
- 新建 optimizer、学习率调度、归一化统计、rollout/GAE 和 RNG；
- 从冻结 schedule 的第一个场景及 `global_step=0` 开始；
- run manifest 中 `resume_parent` 与 `warm_start_parent` 均为 null，并记录旧运行仅作为历史审计。

训练语义升级为
`lunar-training-semantics/sensor-30m-360-platform-primitive-coverable-detail95-observed-incremental-primitive-candidates-option-path-observation-auditable-failure/v10`。
cache v5 保留主要 mask 数组形状，但增加原语状态图身份；旧 manifest 的 schema、算法、语义哈希、
源码提交和 reward 哈希必须失配。cache 必须在仓库外完整重建，禁止就地改写旧 cache。

## 9. 验证与训练解锁门

### 9.1 C++ 单元测试

必须覆盖：

1. 二维 hard-feasible 连通、但受转弯半径或朝向约束不可达的 WHEELED 单元不进入 reachable mask；
2. 同一 WHEELED 单元的不同 `yaw_bin` 或 FORWARD/REVERSE 模式具有独立可达性；
3. LEGGED 的机身高度区间或支撑状态不能连续传播时拒绝对应原语边；
4. 只正向可达、不能返回冻结安全锚点的状态不进入 recoverable states；
5. Hopper 能通过认证单跳跨越地面不连通间隙，而正向或反向落区、`delta-v`、flight tube 任一失败
   都拒绝对应边；
6. `30.0 m` 只限制运行时候选查询，真值图能通过多次合法宏动作到达更远状态；
7. HOPPER 真值多跳图可以覆盖远端落点，但运行时候选只包含当前状态的一跳直接后继；
8. 新观测障碍与原语扫掠区域相交时只失效受影响边，并更新 forward/returnable 标签；
9. 新增已观测安全证据能够加入原语边和新的可恢复状态；
10. 同一输入重复计算的状态、边、reachable mask、诊断计数和哈希逐位一致；
11. 数值失败、取消和资源耗尽不会被降级为普通不可达；
12. global 起点 unknown、local 起点安全时只能通过已认证原语 portal 连接；
13. portal 原语链不完整或 global 目标不可达时仍 fail closed。

### 9.2 Python 单元与集成测试

使用小型确定性多分辨率 fixture 覆盖：

1. 障碍封闭且不可见的自由岛不进入 coverable mask；
2. 不能驻留但可从安全位置看到的目标自由格进入 coverable mask；
3. LOS 遮挡目标不进入 coverable mask；
4. 障碍和坡度禁行格可进入 observed evidence，但不增加 coverage；
5. 起点变化会改变 reachability/cache identity，旧掩码被拒绝；
6. 部分观测按 detail bitmask 精确累计，不使用 coarse ratio 近似；
7. 相同 observed state 下改变 truth 图或 coverable 空间分布，不得改变候选 features/mask 或策略
   稠密空间通道；只有获准的覆盖率标量、reward 和环境诊断可以随精确分母变化；
8. 更新 observed-only 原语图不得改变已经冻结的 coverable 分母，coverage 只随 observed bits 单调；
9. 每个有效候选都引用当前 snapshot 中 forward reachable 且 returnable 的状态 ID；
10. 超过 64 个真实候选时稳定截断，不足时只用 `candidate_mask=false` 填充且没有重复伪候选；
11. 新障碍失效原语边后对应候选消失，新安全证据认证边后对应候选出现；
12. `production empty + oracle opportunity` 必须抛出环境不变量错误；
13. scheduler 不会给某平台分配其 ineligible 场景；
14. 旧 v3/v4 cache、旧 checkpoint 和随机非绑定起点全部 fail closed；
15. 一次地面 action 跨多个滚动 reference 仍只产生一个 transition，精确状态和目标坐标不漂移；
16. 路径样本间距 `<=1.0 m`、replay 位相同且 Hopper 飞行中不观测；
17. 未知不阻挡预计增益、已知障碍阻挡，且候选最大正增益归一化为 `1.0`；
18. `0.95` 首次越界只触发一次 `SUCCESS`，reward v4 的成功奖励为 `100.0`。

### 9.3 固定场景闭环门

先生成仓库外 preflight v5 cache，在冻结 common schedule 的首个物理场景上对三平台执行
observed-only 确定性 gain-over-cost 基线到自然 terminal。该启动门证明闭环可运行，而不是证明
未训练网络或规则基线已经学会达到任务成功阈值。每个场景—平台必须满足：

- `exact=true`；
- `mission_coverable_fraction >= 0.95`；
- 最终覆盖率有限且位于 `[0,1]`，并与 `SUCCESS`/合法失败原因一致；
- `ORACLE_CONTRADICTION=0`；
- 至少成功执行过一条真实 reference；
- 安全违规、无效 action、执行失败和硬错误均为零；
- 每个 terminal reason 完整；
- `SUCCESS` 时覆盖率必须 `>=0.95`；未达到 `0.95` 时允许
  `NO_RECOVERABLE_OBSERVATION_STATE`、`VISITED_EXHAUSTED`、`ZERO_GAIN`、
  `NO_TRANSIT_OPPORTUNITY` 或 `PLANNER_REJECTED_ALL`。

不得按闭环结果替换首个 common 场景，也不得降低当前明确批准的 `0.95` 任务可行性资格门和
episode 的 `0.95` 成功阈值。24 个 exact-common 场景 × 三平台的覆盖分布与成功率改为训练启动后
的首轮评估证据，不再阻止 step-0 启动。启动闭环使用独立的
`closed-loop-gate` 命令和仓库外报告；现有
`formal-preflight` 的一步非代理探针与启动身份检查仍属于 full cache 生成后的启动前校准门，
不能冒充本节的自然终止闭环门。

随后才允许生成 full v5 cache。full manifest 必须公布三平台各 split 的资格率和排除原因；不得以
共同平均掩盖 Hopper 或任一 split。

### 9.4 仓库与 ROS 验证

实现必须运行训练包定向测试和完整测试；涉及 C++ 公共边界后，还必须在 Ubuntu 22.04 + ROS 2
Humble 环境、仓库外 build/install/log 目录中构建并运行 `lunar_planner_core` 与
`lunar_planner_training_bridge` 测试。最后运行仓库边界检查和 foundation 边界测试。

## 10. 实施顺序与停点

实施按以下独立审查单元推进：

1. 冻结 v10 训练语义、cache v5 schema、平台状态 schema 与失败测试；
2. 实现共享 C++ 原语状态图、forward/reverse 可恢复投影和 canonical hash；
3. 接入 WHEELED lattice 状态/边批量展开，不再以 connected component 作为权威；
4. 接入 LEGGED 高度区间/支撑状态图；
5. 将 HOPPER 已认证落区/单跳图接入统一原语图接口；
6. 使用 truth 原语图生成 tile-wise detail target、LOS union、bit-packed coverable mask 和 cache 资格；
7. 接入 observed-only 图的 tile 边失效、局部重认证、snapshot revision 与 Python bridge；
8. 改为从 observed-only 原语图生成候选，并接入分阶段诊断与独立 oracle；
9. 保留并复核持续地面目标、路径观测、一个 action 的 transition 聚合和预计增益归一化；
10. 完成新 run manifest、随机初始化边界以及旧 cache/checkpoint fail-closed；
11. 完成 C++、Python、ROS、确定性和仓库边界验证；
12. 生成 preflight v5 cache，执行首个 exact-common 场景的三平台启动闭环门；
13. 生成 full v5 cache 并复核三平台各 split 的资格分布；
14. 使用本次已获授权的随机初始化配置启动新的 step-0 正式训练并检查首批指标。

在第 12 项全部通过前不得启动新正式训练；失败时停在对应组件修复，不顺带调整网络、课程、
传感器范围或场景尺寸。

## 11. 非目标

本文不修改：

- PPO 网络结构、七输入名称、`[64,12]` 候选合同或动作分布；
- `1024 m × 1024 m` 任务窗、`30 m/360°` 传感器或 `0.2 m` 实际 reveal；
- 无固定决策上限和跨 update 保持 episode 的生命周期；
- C++ v3 对最终 reference 的安全所有权；
- source/split 原始数据和 hazard 生成器。

首版不实现通用动态 SCC，也不要求把完整 truth 状态图写入 cache；增量复用边认证并在每个 revision
重算 forward/reverse 标签即可。本文同样不要求每轮恰好产生 64 个真实候选。

本文也不把旧仓地面实现直接复制给 Hopper。当前 `0.95` 是用户明确批准的 episode 成功阈值，
仍要求 `mission_coverable_fraction >=0.95` 和自然终止闭环通过；它不是未训练基线的逐案例启动
门槛，不能靠继续缩小分母或伪造成功掩盖任务不可行。

## 12. 完成标准

本设计完成必须同时满足：

- 每个 eligible 场景—平台—起点都有 exact、内容寻址的原语图身份和 coverable mask；每个 ineligible 组合有
  完整、确定性的资格证据和原因；
- WHEELED、LEGGED 和 HOPPER 都以正式平台运动原语状态图为可达性权威，二维 connected component
  只作静态剪枝；
- 成功分母只包含 truth 原语图可恢复观测状态能够看到的任务目标自由栅格，并在 episode 内冻结；
- observed-only 原语图按 reveal 增量更新，且任何有效候选都引用其当前可恢复状态；
- truth 图变化不能影响相同 observed state 的候选，observed 图变化不能修改冻结分母；
- 训练和评估只调度对应平台 eligible 场景，并公开完整 feasibility rate；
- 候选耗尽能够区分生成、访问、平台、增益和规划器原因；
- observed-only oracle 能阻止错误的静默早停；
- 单场景三平台启动闭环门、完整测试、ROS 构建和仓库边界检查全部通过；
- 新训练使用全新模型、全新 optimizer、新身份和 step 0；
- 单场景、三平台闭环、完整测试和 cache 门通过后，按本次用户授权直接启动并核验首批指标。
