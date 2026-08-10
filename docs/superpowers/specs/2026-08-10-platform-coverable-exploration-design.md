# 平台化精确可探索栅格与候选终止闭环设计

**日期：** 2026-08-10
**状态：** 已批准，增量闭环修复实施中
**范围：** 正式 PPO 训练的覆盖率分母、场景资格、平台可达性、候选耗尽审计、宏动作闭环和全新训练身份

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

修复结论是同时建立三个相互独立的合同：

1. `mission_target_detail_mask`：任务希望覆盖且可作为成功目标的 `0.2 m` 自由栅格；
2. `reachable_pose_mask`：该平台从冻结起点理论上可到达的 `4.0 m` 观测位置；
3. `coverable_detail_mask`：能从某个可达观测位置在 `30 m` 和 LOS 合同内看到的任务目标栅格。

`coverable_detail_mask` 是成功分母。候选仍从 observed-only 状态生成，不得读取该真值掩码。

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
所需的机体净空。驻留净空只属于 `reachable_pose_mask`。

已批准的成功语义固定为：

- 硬障碍和坡度/地形禁行栅格可以被传感器观测并更新 observed evidence；
- 它们不进入覆盖成功分母，也不产生任务覆盖增量；
- 未知真值不得通过该分类进入策略输入或候选预计增益。

### 3.2 平台可达观测位置

`reachable_pose_mask` 使用 `4.0 m` 全局规划栅格，因为正式策略动作的候选落点和当前 C++
全局请求均以该栅格为宏观位置合同。掩码绑定：

```text
scene_id
platform_type
qualified_start_cell
capability_sha256
v3_source_commit
global_geometry
maximum_candidate_distance_m = 30.0
```

WHEELED 和 LEGGED 直接使用 C++ `ProjectTraversability()` 已生成的
`connected_component`：起点所在分量内的 `hard_feasible` 单元为可达观测位置。Python 不再复制
四邻域或直线连通规则。

HOPPER 使用新的 C++ 批量可恢复可达性投影。节点是通过正式落脚区域检查的 `4.0 m` 单元；从起点
执行确定性 BFS。对每个已发现节点，按稳定的行列顺序检查距离不超过 `30.0 m` 的目标节点；只有
正向与反向的当前正式 Hopper 单跳链都全部认证通过时才建立无向可恢复边：

```text
exact landing region
AND available delta-v >= required delta-v
AND ballistic envelope certified
AND flight tube certified
AND forbidden/clearance/landing support certified
```

当前 Hopper 冻结合同没有累计燃料，因此节点状态只需位置，不把历史跳数或剩余燃料加入图状态。
如果未来 capability 恢复累计燃料，本文的 Hopper reachability algorithm ID 必须失配并拒绝旧
cache，不得继续把位置图称为精确。

这是 `coverable` 可由同一条探索轨迹逐步访问并回退的保证；只从起点单向可达、却无法返回的落点
不得进入覆盖率分母或运行时候选。实现可以使用空间索引、严格的 delta-v 下界和确定性批处理减少
认证次数，但任何启发式不得删除可能可恢复的边。数值不确定、取消或资源耗尽使该场景—平台资格
计算失败，不能当作普通不可达边。

### 3.3 可覆盖精细栅格

对每个 `0.2 m` 任务目标栅格，若至少存在一个 `reachable_pose_mask` 中的单元中心满足：

```text
distance <= 30.0 m
AND two_dimensional_detail_los/v1 可见
```

则该目标栅格进入 `coverable_detail_mask`。LOS 使用与真实 reveal 相同的 `0.2 m` 障碍真值和
射线离散规则。正式传感器为 `360°`，因此 coverable 预计算不受 yaw 限制；执行后的实际 reveal
仍使用完整传感器合同。

正式覆盖率唯一计算式为：

```text
coverage =
    popcount(observed_detail_mask AND coverable_detail_mask)
    / popcount(coverable_detail_mask)
```

分母必须大于零。初始 reveal 计入 numerator，但不得在初始状态达到当前批准的 `0.95`；首次从
`<0.95` 跨到 `>=0.95` 仍是唯一 `SUCCESS` 事件。旧仓的 `0.99` 能力保留为后续诊断目标，
不作为本轮恢复训练的成功门槛。

## 4. 真值与策略边界

`coverable_detail_mask`、`reachable_pose_mask` 的真值版本及剩余未覆盖真值只能用于：

- cache 资格和完整性校验；
- 覆盖率 numerator/denominator；
- reward 与成功事件；
- 环境诊断和正式评估。

以下路径禁止读取真值 coverable mask：

- `prior_channels`、`coverage_summary` 的稠密策略输入；
- frontier 提取和候选落点生成；
- 候选预计信息增益；
- 候选排序或策略 action mask；
- observed-only frontier oracle。

现有 `pose_features[:,4]` 继续向策略提供一个累计覆盖率标量，但改为本文的精确 coverable
分母。它只透露机器人已取得的任务进度，不透露未观测区域的空间位置。候选预计信息增益继续使用
机器人当时已经掌握的稀疏 `0.2 m` 观测地图，不使用完整 detail truth。

## 5. Cache v4 与固定起点

正式 cache schema 升级为 `lunar-formal-training-cache/v4`。旧 v3 cache 不可原地修改或作为新
训练输入；必须在仓库外新目录完整重建。

每个场景按平台保存：

```text
qualified_start_cell
reachable_pose_mask                    uint8/bit-packed [256,256]
coverable_detail_mask                  uint8/np.packbits row-major [5120,5120]
coverable_ratio                        float32 [256,256]
mission_target_detail_cell_count       int
coverable_detail_cell_count            int
mission_coverable_fraction             float
initial_coverable_fraction              float
reachability_algorithm_id              string
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

`coverable_ratio[row,column]` 等于该 `4.0 m` 单元内 `20 × 20` 个精细子格中 coverable 子格的比例，
仅供汇总、审计和报告；实际 numerator 必须与 bit-packed 精细掩码逐位相交，不能用比例近似部分
观测。

完整 `5120 × 5120` 多层 truth 不常驻内存。预计算逐 `64 m` tile 执行，只保留一个 tile 的工作
数组和约 `3.3 MB/platform/scene` 的最终 bit-packed 掩码；manifest 记录压缩前语义哈希、文件
大小和内容哈希。

### 5.1 起点与资格

为避免起点相关掩码失效，每个场景—平台 v1 只绑定一个确定性起点：继续使用当前按距画布中心、
行、列排序后找到的第一个安全且初始 observed-only 候选非空的起点。正式 worker 必须始终使用
该缓存起点，不再先尝试随机安全起点。增加多起点需要新的 start profile schema，不在本文范围。

场景—平台 `eligible=true` 必须同时满足：

1. 起点安全且属于 reachable mask；
2. `coverable_detail_cell_count > 0`；
3. `mission_coverable_fraction = coverable / mission_target >= 0.95`；
4. `initial_coverable_fraction < 0.95`；
5. 初始 reveal 后至少有一个经平台检查的 observed-only 候选；
6. reachability、detail target、LOS 和掩码哈希全部精确完成。

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

## 6. 运行时平台候选可达性

候选发现的 observed-only 边界、64 项张量形状和稳定排序保持不变；预计增益的射线语义与数值
归一化按 6.4 节修复。平台检查改为统一
接口，避免 `CandidateBuilderV2` 内继续扩展互不一致的平台分支：

```text
PlatformCandidateReachability.filter(
    observed_world,
    current_pose,
    candidate_cells,
    platform_capability,
) -> accepted_mask + reason_counts
```

WHEELED/LEGGED 在当前已观测、安全投影的 C++ 连通分量中筛选候选。HOPPER 对当前位姿到候选
落点执行与 cache 预计算相同的正反向单跳认证，但运行时结果必须逐候选直接绑定当前执行位姿，
不得把候选之间的中间边组成多跳路径后误报为当前 action 可达。cache 使用
`cpp-hopper-certified-bidirectional-bfs/v3` 构建可恢复多跳分母图，运行时使用独立的
`cpp-hopper-certified-bidirectional-direct/v1` 批量投影。两者输入都只能包含当前 observed
global/local map；未知单元不能从 truth 补全。认证后的候选仍由正式 `Planner::Plan()` 生成最终
reference，预筛结果不替代执行时认证。

运行时 `4 m` global 规划格只有在所属 `20 x 20` 个 `0.2 m` 细格全部已观测后才置为 known，并在
此时固定聚合高程、最大障碍、禁区、质量和计数。只看见中心四个细格不能代表整格已知，否则后续
观测会改变同一粗格并使此前认证的返程边失效。局部落脚与候选预计增益仍直接使用当时掌握的稀疏
`0.2 m` 地图，不要求整格观测后才计算。

上述保守 global 聚合不能反过来否定 local 已经证明的地面连通性。WHEELED/LEGGED 候选的精确
目标位姿若落在当前 `0.2 m` local map 内，以该 local map 中当前位姿所在的
`hard_feasible + connected_component` 为可达性权威；只有目标不在 local map 内时才使用 `4 m`
global 投影。禁止把两者无条件相交：机器人可能位于仅观测 `399/400` 个细格的粗格内，此时 global
起点按合同仍是 unknown，global reachable 会全零，但 local map 已经拥有当前位姿、已执行父落点
及二者之间的精细安全路径。该情形必须保留 local 可达候选，最终仍由正式 `Planner::Plan()` 验证。
HOPPER 不采用该地面覆盖规则，继续使用本节冻结的精细落脚证据与双向单跳认证。

闭环闸门补充：传统粗分辨率 frontier 候选为空不等于没有观测机会。若传统路径最终没有发出
候选，生成器可在当前动作包络内扫描已观测、安全的观测位姿，仍先执行同一平台认证，再用机器人
当时掌握的 `0.2 m` 稀疏地图计算预计增益，只补入严格正增益位姿。这是 frontier standoff 的
退化补全。若传统路径因没有锚点或没有平台可达锚点而耗尽、但 observed-only 粗图仍存在
frontier，且传统候选和退化扫描都没有发出正增益候选，可补入当前单跳认证的零即时增益中转位姿；
这也包括传统 frontier 锚点存在但其当前预计增益全部为零的情况。先使用未访问位姿，无候选时才允许已访问
回退。回退只能指向确定性导航栈中的直接父落点；栈必须保存父落点实际认证并执行的精确
`x/y/z`，不得只保存其所属 `4 m` 粗格并在返程时改瞄粗格中心。返回父落点即弹栈，栈从已有
reveal history 重建，禁止在两个已访问落点间振荡。增益字段必须如实为零，执行成本和零覆盖惩罚
照常计算。不得延长 Hopper 单跳、读取 truth coverable mask 或改变
`[64,12]` 策略张量合同；精确返程位姿和高程只作为候选批及 replay 审计元数据。独立 oracle 必须覆盖
这类机会，但不得调用生产排序。oracle 的动作包络必须与生产生成器完全同源：相对当前执行位姿
使用目标的精确世界坐标（直接父节点使用其实际 `x/y/z`）计算 `30 m` 距离和 FOV，禁止用粗格
索引距离近似，否则位于格内边缘的机器人会产生假机会。

Hopper 落点存在精细/粗格冲突时，以已经通过算法 ID、几何和值校验的 `0.2 m` 精细落脚区域证据
为落点权威，不得再因其所属 `4 m` 聚合格的硬可行性为假而清零；粗格仍用于飞行管道与没有精细
落脚证据的保守路径。该变化必须升级 Hopper reachability algorithm ID，使旧 cache 自动失配。

运行时 C++ 数值失败或资源耗尽属于基础设施错误，不得写成候选不可达并让 episode 正常结束。

### 6.1 地面平台的局部起点连接权威

WHEELED/LEGGED 的正式 Planner 不得在局部 `0.2 m` 已证明当前位姿安全时，仅因当前所属 `4 m`
单元未达到 `400/400` 观测就拒绝请求。全局图仍保持全观测才 known 的保守规则，不伪造或提升
粗格证据；Planner 改为建立确定性的局部起点连接：

1. 在 local `hard_feasible + connected_component` 中确认精确当前位姿；
2. 找到同时属于该局部分量和全局 `hard_feasible` 的最近确定性 portal；
3. 用局部投影搜索并认证从精确起点到 portal 的连接路径；
4. 以 portal 作为全局搜索起点，再把局部连接和全局路径合成为一条 route preview；
5. 若没有同时满足局部连接和全局到目标路径的 portal，才返回 `GLOBAL_NO_KNOWN_SAFE_ROUTE`。

portal 按局部路径代价、行、列稳定排序；排除的 conditional corridor 仍只作用于全局段。任何
局部起点连接都必须经过现有 local planner 的最终安全认证，不能把局部连通分量当作执行 reference。

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

通过平台可达性筛选后，`frontier_features[:,5]` 使用本批正增益候选的最大 mission gain 归一化，
`frontier_features[:,6]` 同样按最大 priority gain 归一化；最大值为零时保持零。原始增益继续用于
零增益诊断，稳定排序仍使用确定性 tie-break。该变化不改变 `[64,12]` 形状，但属于策略输入语义
变化，必须升级训练语义并从新模型开始。

## 7. 候选耗尽与独立 Oracle

`CandidateDiagnostics` 扩展为分阶段计数：

```text
frontier_anchor_count
visited_excluded_count
static_infeasible_count
platform_unreachable_count
zero_gain_count
emitted_count
planner_rejected_count
```

新增 observed-only `FrontierOpportunityOracle`，独立判断当前已知地图上是否仍存在合法探索机会：

- WHEELED/LEGGED 在当前已观测安全连通域中寻找能看到未知 ROI 的观测位置；
- HOPPER 独立生成稳定的 frontier 落脚代表点，并用当前已观测地图执行真实单跳认证；
- oracle 不调用生产候选排序，不读取 truth coverable mask，不把结果输入策略或 reward。

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

规划器逐个拒绝全部候选时仍运行 oracle，但此时 oracle 只证明 observed-only 几何机会，不等价于
完整规划器能够生成 reference；其正数结果作为诊断保留，不得反过来误报为生产生成器为空。
只有 `producer_emitted_count == 0` 时，oracle 正数才构成真正的生成器矛盾。硬安全、合同错配和
非有限数值继续走 `HARD_FAILURE`，不能与普通探索耗尽合并。

每个终止 episode 必须携带一个完整主原因：`SUCCESS`、`NO_FRONTIER_ANCHOR`、
`VISITED_EXHAUSTED`、`PLATFORM_UNREACHABLE`、`ZERO_GAIN`、`PLANNER_REJECTED_ALL`、
`HARD_FAILURE` 或 `CANCELED`。除 `PLANNER_REJECTED_ALL` 外，
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
`lunar-training-semantics/sensor-30m-360-platform-coverable-detail95-ground-option-path-observation-auditable-failure/v8`。
cache v4 的数组 schema 不变，但旧 manifest 的语义哈希、源码提交和 reward 哈希必须失配；必须在
仓库外完整重建，禁止就地改写旧 cache。

## 9. 验证与训练解锁门

### 9.1 C++ 单元测试

必须覆盖：

1. WHEELED/LEGGED reachable mask 与起点 C++ connected component 一致；
2. Hopper 能通过认证单跳跨越地面不连通间隙；
3. Hopper 的正向或反向落脚区域、delta-v、flight tube 任一失败都会拒绝对应边；
4. 超过 `30.0 m` 的目标不进入当前策略可达图；
5. 同一输入重复计算的 reachable mask、诊断计数和哈希逐位一致；
6. 数值失败、取消和资源耗尽不会被降级为不可达；
7. 粗图起点 unknown、局部起点安全时使用经过认证的局部 portal 连接；
8. portal 不属于局部起点分量或全局目标不可达时仍 fail closed；
9. 已知粗图起点继续产生与修复前逐位相同的全局 route。

### 9.2 Python 单元与集成测试

使用小型确定性多分辨率 fixture 覆盖：

1. 障碍封闭且不可见的自由岛不进入 coverable mask；
2. 不能驻留但可从安全位置看到的目标自由格进入 coverable mask；
3. LOS 遮挡目标不进入 coverable mask；
4. 障碍和坡度禁行格可进入 observed evidence，但不增加 coverage；
5. 起点变化会改变 reachability/cache identity，旧掩码被拒绝；
6. 部分观测按 detail bitmask 精确累计，不使用 coarse ratio 近似；
7. 相同 observed state 下改变 truth coverable 空间分布，不得改变候选 features/mask 或策略稠密
   空间通道；只有获准的覆盖率标量、reward 和环境诊断可以随精确分母变化；
8. `production empty + oracle opportunity` 必须抛出环境不变量错误；
9. scheduler 不会给某平台分配其 ineligible 场景；
10. 旧 cache、旧 checkpoint 和随机非绑定起点全部 fail closed；
11. 一次地面 action 跨多个滚动 reference 仍只产生一个 transition，目标坐标不漂移；
12. 路径样本间距 `<=1.0 m`、replay 位相同且 Hopper 飞行中不观测；
13. 未知不阻挡预计增益、已知障碍阻挡，且候选最大正增益归一化为 `1.0`；
14. `0.95` 首次越界只触发一次 `SUCCESS`，reward v4 的成功奖励为 `100.0`。

### 9.3 固定场景闭环门

先生成仓库外 preflight v4 cache，在冻结 common schedule 的首个物理场景上对三平台执行
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
  `NO_FRONTIER_ANCHOR`、`VISITED_EXHAUSTED`、`PLATFORM_UNREACHABLE`、`ZERO_GAIN` 或
  `PLANNER_REJECTED_ALL`。

不得按闭环结果替换首个 common 场景，也不得降低当前明确批准的 `0.95` 任务可行性资格门和
episode 的 `0.95` 成功阈值。24 个 exact-common 场景 × 三平台的覆盖分布与成功率改为训练启动后
的首轮评估证据，不再阻止 step-0 启动。启动闭环使用独立的
`closed-loop-gate` 命令和仓库外报告；现有
`formal-preflight` 的一步非代理探针与启动身份检查仍属于 full cache 生成后的启动前校准门，
不能冒充本节的自然终止闭环门。

随后才允许生成 full v4 cache。full manifest 必须公布三平台各 split 的资格率和排除原因；不得以
共同平均掩盖 Hopper 或任一 split。

### 9.4 仓库与 ROS 验证

实现必须运行训练包定向测试和完整测试；涉及 C++ 公共边界后，还必须在 Ubuntu 22.04 + ROS 2
Humble 环境、仓库外 build/install/log 目录中构建并运行 `lunar_planner_core` 与
`lunar_planner_training_bridge` 测试。最后运行仓库边界检查和 foundation 边界测试。

## 10. 实施顺序与停点

实施按以下独立审查单元推进：

1. 冻结 v6 基础语义、cache v4 schema 与失败测试；
2. 暴露 WHEELED/LEGGED connected component，并实现 Hopper C++ reachability projection；
3. 实现 tile-wise detail target、LOS union 与 bit-packed coverable mask；
4. 重构 cache 资格、固定起点和 per-platform schedule；
5. 接入精确 detail coverage numerator/denominator；
6. 接入统一平台候选可达性、分阶段诊断和 observed-only oracle；
7. 接入持续地面目标、路径观测和一个 action 的 transition 聚合；
8. 升级预计增益、候选归一化、`0.95` 成功和 reward v4；
9. 完成新 run manifest 的随机初始化边界；
10. 生成 preflight v4 cache，执行首个 exact-common 场景的三平台启动闭环门；
11. 生成 full v4 cache并复核数据分布；
12. 使用本次已获授权的随机初始化配置启动新的 step-0 正式训练并检查首批指标。

在第 10 项全部通过前不得启动新正式训练；失败时停在对应组件修复，不顺带调整网络、课程、
传感器范围或场景尺寸。

## 11. 非目标

本文不修改：

- PPO 网络结构、七输入名称、`[64,12]` 候选合同或动作分布；
- `1024 m × 1024 m` 任务窗、`30 m/360°` 传感器或 `0.2 m` 实际 reveal；
- 无固定决策上限和跨 update 保持 episode 的生命周期；
- C++ v3 对最终 reference 的安全所有权；
- source/split 原始数据和 hazard 生成器。

本文也不把旧仓地面实现直接复制给 Hopper。当前 `0.95` 是用户明确批准的 episode 成功阈值，
仍要求 `mission_coverable_fraction >=0.95` 和自然终止闭环通过；它不是未训练基线的逐案例启动
门槛，不能靠继续缩小分母或伪造成功掩盖任务不可行。

## 12. 完成标准

本设计完成必须同时满足：

- 每个 eligible 场景—平台—起点都有 exact、内容寻址的 coverable mask；每个 ineligible 组合有
  完整、确定性的资格证据和原因；
- 成功分母只包含可达观测位置能够看到的任务目标自由栅格；
- Hopper reachability 使用真实单跳认证而非通用地面连通规则；
- 训练和评估只调度对应平台 eligible 场景，并公开完整 feasibility rate；
- 候选耗尽能够区分生成、访问、平台、增益和规划器原因；
- observed-only oracle 能阻止错误的静默早停；
- 单场景三平台启动闭环门、完整测试、ROS 构建和仓库边界检查全部通过；
- 新训练使用全新模型、全新 optimizer、新身份和 step 0；
- 单场景、三平台闭环、完整测试和 cache 门通过后，按本次用户授权直接启动并核验首批指标。
