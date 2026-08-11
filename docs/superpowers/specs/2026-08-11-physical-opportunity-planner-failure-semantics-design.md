# 物理机会、固定局部搜索域与规划失败终止语义修订设计

**日期：** 2026-08-11
**状态：** 已批准，待实施
**范围：** 正式 PPO 训练的覆盖率分母、物理候选、observed-only oracle、地面局部搜索域、规划失败刷新、候选临时抑制、终止语义以及训练身份迁移

## 1. 权威关系

本文原子取代
[`2026-08-10-platform-coverable-exploration-design.md`](2026-08-10-platform-coverable-exploration-design.md)
中以下结论：

- truth/observed 运动原语图是覆盖率分母、候选生成和 oracle 的可达性权威；
- 候选必须引用原语状态 ID 才能成为合法策略动作；
- `PLANNER_REJECTED_ALL` 可以作为候选耗尽主原因；
- cache v5 和 training semantics v10 是下一轮正式训练身份；
- `primitive_state_count`、`recoverable_state_count` 和原语图哈希是探索终止所需的核心诊断。

本文不撤销旧文档中的以下已批准边界：

- 冻结 source/split、确定性场景生成和正式 capability 材料；
- `1024 m × 1024 m` 全局画布、`4.0 m` 网络全局分辨率和 `0.2 m` 精细观测；
- `30 m/360°` 传感器、任务 ROI、LOS 和 observed-only 策略输入；
- 覆盖成功首次跨过 `0.95` 才能产生 `SUCCESS`；
- 地面一个策略动作保持同一全局目标并跨多个滚动 reference 执行；
- 沿已执行路径采样传感器证据，不能只在 reference 终点观测；
- 无固定 episode 决策上限；
- truth 侧覆盖分母不能泄漏到在线候选、策略特征或 oracle；
- C++ v3 仍是最终路径和运动安全裁决者。

运动原语图可以继续作为规划器内部搜索结构、性能缓存或诊断材料，但不得再决定覆盖率分母、生产候选、策略 action mask 或 observed-only oracle。

若本文未明确修改某一旧设计条款，则原条款继续有效。本文与旧文档冲突时，以本文为准。

## 2. 问题陈述与根因

正式轮式案例已经证明以下链路：

1. 候选点位于平台物理可通行区域，全局规划能够生成路线；
2. `BuildLocalView()` 将局部路线走廊和 horizon 外的栅格改写成 `valid=0`、`forbidden=1`；
3. 轮式 `SafeProjection` 在这张被裁剪的地图上执行车体净空膨胀；
4. 局部目标被人工窗口边界挤出可行集，在 lattice 展开前返回 `WHEEL_GOAL_INFEASIBLE`，展开数为零；
5. `Planner::Plan()` 将局部失败汇总为 `LOCAL_SEGMENT_INFEASIBLE`；
6. 环境在首次规划失败时立即按候选数组下标关闭 action mask；
7. 所有当前候选被关闭后，环境将规划器失败包装成 `PLANNER_REJECTED_ALL`；
8. 历史失败终点仍有 `146` 个 oracle 机会，证明该终止不是物理前沿耗尽。

单变量测试中，额外走廊 margin 为 `0.4 m` 和 `0.8 m` 时失败，`4.0 m` 时同一目标通过。这一结果证明人工窗口边界参与了物理可行性判定，但不能证明全局固定 `4.0 m` 是正确设计。正确修复是分离地图语义与搜索范围，再使用已批准的固定 `2.0 m` 额外 margin。

另一个独立问题是正式环境把运动原语可达图同时用于覆盖分母、候选和 oracle。这样会让路径规划器所选离散动作集改变“哪些任务区域理论上应该被覆盖”，违背探索任务的物理定义。覆盖率和候选必须先由平台物理条件定义，规划器随后裁决到达某个候选的具体路径。

## 3. 冻结决策

本文冻结以下四项设计：

1. WHEELED 和 LEGGED 的 `additional_corridor_margin_m` 全局固定为 `2.0 m`；
2. 局部物理地图与 `search_domain_mask` 完全分离；
3. 覆盖率分母、生产候选和 oracle 使用平台物理机会语义，运动原语只属于路径规划；
4. 永久候选淘汰取消，只保留当前物理快照内的临时规划失败抑制；`PLANNER_REJECTED_ALL` 被删除。

本文不引入 `0.4 m → 4.0 m` 动态扩宽、宽走廊恢复状态机或环境层自动重试规划器。

## 4. 术语与权威数据流

### 4.1 术语

`physical_observation_pose_mask`
: 在给定平台 capability、地图证据和起点下，满足物理安全、平台支撑/落脚和物理连通性的观测位姿空间投影。

`coverable_detail_mask`
: 能从 truth 侧某个物理可达观测位姿，在正式传感器距离和 LOS 合同内看到的任务 detail 栅格。

`physical_candidate_universe`
: 当前 observed-only 物理快照下，生产候选生成器枚举出的全部有界、确定性物理候选，位于 top-M 截断之前。

`available_policy_candidates`
: 从 `physical_candidate_universe` 去除当前物理快照已经最终规划失败的候选后，供策略选择的候选。

`physical_snapshot_id`
: 绑定当前位置、observed-map evidence、任务 revision、平台和 capability 的稳定身份。仅重建 observation revision 不得改变它。

`search_domain_mask`
: 本次局部搜索允许节点中心展开的空间范围。它是计算范围，不是世界障碍证据。

### 4.2 权威数据流

```text
frozen truth + capability + start
    -> physical_observation_pose_mask
    -> sensor range/LOS projection
    -> frozen coverable_detail_mask
    -> coverage denominator

observed map + current pose + capability
    -> physical_candidate_universe
    -> subtract planner_failed_candidate_ids[physical_snapshot_id]
    -> deterministic rank/reserve refill/top-M
    -> available_policy_candidates
    -> policy action

observed map + current pose + capability
    -> independent physical opportunity oracle
    -> exhaustion and mismatch audit

selected target + original physical maps + fixed search_domain_mask
    -> C++ v3 planner
    -> reference OR structured candidate disposition
```

Truth 侧只允许输出冻结分母及其身份。它不得参与在线候选位置、候选排序、candidate mask 或 oracle。

## 5. 固定 `2.0 m` 局部搜索域

### 5.1 配置合同

地面平台的局部走廊固定为：

```text
additional_corridor_margin_m = 2.0

corridor_half_width_m =
    platform_support_radius_m
    + platform_minimum_clearance_m
    + additional_corridor_margin_m
```

`2.0 m` 是额外 margin，不是包含车体支撑半径和最小净空在内的总半宽。

首版实现保留现有配置字段以减少接口迁移，但所有正式入口必须显式写入 `2.0`，C++ 配置校验必须拒绝其他有限值。diagnostics 必须记录实际 `corridor_half_width_m` 和冻结的 additional margin。Hopper 不适用该合同。

### 5.2 `LocalPlanningProblem` 边界

`LocalPlanningProblem` 必须分别携带：

- 未修改的 observed `local_map`；
- 与 `local_map` 几何完全一致的不可变 `search_domain_mask`；
- 局部目标；
- 路线前缀和 frontier 身份；
- 当前平台 capability 和 planner config。

`BuildLocalFrontiers()` 继续根据路线前缀、horizon 和固定走廊计算 `search_domain_mask`，但不得修改 `valid_mask`、`forbidden` 或其他地图 layer。

### 5.3 物理投影与搜索限制

Wheel 和 legged planner 必须遵守：

1. `SafeProjection`、净空、支撑和 footprint sweep 基于原始 `local_map`；
2. 目标物理可行性基于原始地图；
3. lattice/A* 的状态中心和连接器中心必须位于 `search_domain_mask`；
4. footprint 可以延伸到搜索域外，只要原始物理地图证明覆盖区域安全；
5. 最终 reference 继续通过完整物理 sweep/支撑校验；
6. projection cache 只以物理地图和 capability 为身份，不把 search domain 伪装成地图 generation；
7. 搜索结果 diagnostics 单独记录 search-domain 身份和范围。

### 5.4 失败分类

以下语义固定：

| reason | 含义 | candidate disposition |
| --- | --- | --- |
| `WHEEL_GOAL_INFEASIBLE` / 对应 legged 原因 | 原始物理地图上没有安全局部目标位姿 | planner 完成正常汇总后可抑制当前候选 |
| `LOCAL_SEARCH_DOMAIN_EXHAUSTED` | 固定搜索域中没有找到局部路径 | planner 完成正常汇总后可抑制当前候选 |
| `LOCAL_SEGMENT_INFEASIBLE` | 所有正常 local frontier 尝试均失败的汇总诊断 | 不能由环境单独解析为抑制指令 |
| canceled/resource/numerical/invalid/exception | 请求或基础设施失败 | `KEEP`，按现有硬失败路径处理 |

`WHEEL_GOAL_INFEASIBLE` 只能在原始物理投影上成立。人工 search-domain 边界不得导致搜索展开前的物理目标失败。

### 5.5 不新增恢复状态机

规划器保留已有的 global route、local frontier backoff 和 conditional corridor retry。本文不新增动态扩宽、多轮全局重规划或环境层 planner retry。规划器完成既有正常尝试后返回一个最终结果，环境只处理该最终结果。

## 6. 平台物理机会合同

### 6.1 任务目标 detail 栅格

每个平台分别计算：

```text
mission_target_detail_mask(platform) =
    inside_formal_mission_roi
    AND detail_valid
    AND forbidden == false
    AND physical_hard_obstacle == false
    AND platform_intrinsic_terrain_feasible
```

它只描述任务希望观测的自由目标栅格，不要求机器人中心能够驻留于每个目标栅格。机器人观测位姿的 footprint、净空、支撑和落脚条件由 `physical_observation_pose_mask` 决定。

### 6.2 WHEELED 和 LEGGED 物理观测位姿

地面平台使用 capability-specific C++ 物理安全投影：

- 有效地形和禁止区域；
- 硬障碍；
- 坡度、粗糙度和平台内在地形限制；
- footprint 净空；
- WHEELED 支撑和接地条件；
- LEGGED 支撑、落脚和允许的机身高度条件。

在该安全投影上，从冻结起点或当前 observed 起点计算 corner-safe 物理连通分量。该连通分量可以采用确定性的栅格邻接投影，但不得读取 planner motion primitive ID、yaw-mode graph 或 planner primitive-set hash。

### 6.3 HOPPER 物理观测位姿

Hopper 使用：

- 安全落点和落区支撑；
- capability 允许的物理飞行包络；
- 飞行管与已知障碍约束；
- 从当前落点或冻结起点出发的物理落点连通性。

该连通性使用连续 capability 包络和认证结果，不以规划器离散 primitive-set 身份定义覆盖分母。运行时候选仍必须是当前状态下物理可发起的落点；最终轨迹和单跳/多跳选择由 Hopper planner 裁决。

### 6.4 Truth 侧覆盖分母

```text
coverable_detail_mask(platform) =
    mission_target_detail_mask(platform)
    AND visible_from_any(
        physical_observation_pose_mask(platform),
        range = 30.0 m,
        field_of_view = 360 degrees,
        frozen LOS contract
    )
```

分母在 episode reset 前冻结，episode 内不得因为 observed evidence、planner 成功率或候选失败而改变。初始 reveal 计入 numerator，但不得直接产生策略动作奖励。

### 6.5 在线 observed-only 候选

生产候选固定执行：

1. 只读取当前 observed map、当前位姿、mission 和 capability；
2. 计算 observed-only 物理安全与物理连通观测位姿；
3. 排除已访问、静止且无收益、零观测收益等探索语义不合法位置；
4. 使用当前 observed evidence 估算 `30 m/360°` 观测收益；
5. 按稳定空间分段、收益、物理距离和稳定 candidate ID 排序；
6. 保留 top-M 之前的有界候选全集和 reserve；
7. 应用当前快照临时失败集合；
8. 从未失败候选中补位并截断到当前 `64` 项动作容量；
9. 不足槽位使用 `candidate_mask=false` padding，不复制候选。

候选可以被 planner 拒绝，因为物理机会不承诺固定搜索域中的具体路径一定成功。该拒绝只影响当前物理快照的可选动作，不回写物理候选定义。

### 6.6 运动原语隔离测试

在 truth、observed state、capability 和起点完全相同时，改变 planner motion primitive 集合必须满足：

- `coverable_detail_mask` 逐位不变；
- `physical_candidate_universe` 的稳定 ID 集合不变；
- oracle opportunity 集合或 canonical hash 不变；
- planner outcome、路径和轨迹允许变化。

## 7. 独立 observed-only oracle

Oracle 与生产 CandidateBuilder 共享物理安全定义和 sensor/LOS 合同，但必须独立执行以下步骤：

- 重算当前起点的物理安全连通观测位置；
- 独立检查这些位置是否能观察尚未观测的任务 ROI；
- 独立统计机会，不调用生产排序、top-M、reserve 或临时失败集合；
- 不读取 truth coverable mask、truth 可达投影或剩余 truth 目标位置；
- 不读取 planner primitive graph 或 planner-failed candidate IDs。

Oracle 证明的是“物理机会仍存在”，不是“当前 planner 必然能找到路径”。因此 oracle 为正且所有生产候选都在当前快照规划失败时，结果是 planner blocked，而不是 oracle 错误或探索完成。

## 8. 候选身份与当前快照临时抑制

### 8.1 稳定 `candidate_id`

每个真实候选必须携带稳定 ID，canonical 输入至少包含：

```text
platform_type
mission_revision
candidate_position_grid_key
candidate_height_or_landing_key
candidate_yaw_bin_or_heading_key
candidate_goal_tolerance_key
```

canonical stream 使用固定字段顺序、定宽整数和有限浮点量化；frontier、fallback 或 reserve 等生成路径
不得进入物理候选身份。同一目标位姿跨 observation rebuild 必须产生同一 ID。candidate 数组下标只在
当前 `ObservationIdentity` 中有效，不能作为失败身份。

### 8.2 `physical_snapshot_id`

snapshot canonical 输入至少包含：

```text
platform_type
platform_id
capability_version/content hash
mission_revision
current physical pose key
observed world evidence hash
observed evidence generation
```

这里的 evidence generation 只在传感器或其他权威物理证据变化时递增，不是规划地图对象的重建次数。
仅 candidate revision、policy observation revision、规划地图 materialization revision、top-M 排序或
tensor padding 变化不得改变 physical snapshot。真实移动、observed evidence 变化或 capability/mission
变化必须产生新 snapshot。

### 8.3 临时失败集合

环境维护：

```text
planner_failed_candidate_ids: mapping[physical_snapshot_id, set[candidate_id]]
```

首版只保留当前 snapshot 的集合。进入新 snapshot 后旧集合自动丢弃，不形成跨地图永久黑名单。

### 8.4 结构化 candidate disposition

Planner bridge 输出新增：

```text
CandidateDisposition:
    KEEP
    SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT
```

只有 planner 完成全部正常尝试后确认所选目标在当前输入下无法产生安全 reference，才返回 `SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT`。

Canceled、invalid request、resource exhausted、numerical failure、异常和 active reference invalidation 必须返回 `KEEP`。环境不得再通过 `(outcome, directive)` 或 reason string 列表猜测候选处置。

### 8.5 Reserve 补位

临时抑制必须发生在 top-M 选择前。若最高排名候选失败，CandidateBuilder 必须从同一 `physical_candidate_universe` 的未失败 reserve 中补入下一候选。

以下状态不能终止：

```text
selected top-M slots are all suppressed
AND untried reserve candidates remain
```

只有生产候选全集中每个候选都已在当前 snapshot 实际规划失败，才算 planner-failed universe exhausted。

## 9. 规划失败后的观测与候选刷新

### 9.1 新接口

正式环境新增 episode-owned 回调：

```python
refresh_after_planning_failure(
    candidate_id: str,
    disposition: CandidateDisposition,
) -> BoundaryObservationResult
```

`FormalEpisode` 负责当前位姿、deferred candidate 状态和 map materialization；`ObservationBoundaryController` 负责 observation identity、零增量边界和当前 observed state 的权威重建。`V3ExplorationEnvironment` 不得在 sensor-closed-loop 模式下绕过 controller 使用通用 observation provider。

### 9.2 首次规划失败且未移动

处理顺序固定为：

1. 验证 planner 输出和 disposition；
2. 若 disposition 要求抑制，将所选稳定 candidate ID 加入当前 snapshot；
3. 不调用 sensor reveal；
4. 不推进 state time；
5. coverage、priority coverage 和 success crossing 增量均为零；
6. 从同一 observed evidence 重建候选全集；
7. 保持同一 `physical_snapshot_id`，生成新的 policy observation revision；
8. 应用临时失败集合并从 reserve 补位；
9. 若仍有候选，返回非终止策略边界。

本次 planner wall time 和已有规划成本按现行合同计入一次。本文不新增 invalid-action 奖励项。

### 9.3 执行若干滚动 reference 后失败

处理顺序固定为：

1. 保留此前 reference 已产生的真实位移、路径传感器 evidence、时间、覆盖增量和奖励；
2. 中止当前 ground option，清除持续目标；
3. 将 `_defer_candidate_rebuild` 设为 false；
4. 不再次 reveal 已执行路径；
5. 使用当前真实位姿和已经累积的 observed map 完整重建候选；
6. 以最新 pose/evidence 生成 `physical_snapshot_id`；
7. 将失败全局目标的稳定 candidate ID 加入该最新 snapshot；
8. 从未失败候选和 reserve 形成新的 policy observation；
9. 聚合 transition 时，已执行覆盖、时间和成本只计算一次。

若执行过程中已经首次跨过成功阈值，则 success boundary 在发起下一次滚动规划前终止宏动作，不会再以随后的规划失败覆盖成功。

### 9.4 刷新不变量

无新物理 evidence 的刷新必须满足：

- state time 不变；
- pose 不变；
- observed world evidence hash 不变；
- physical snapshot ID 不变；
- coverage numerator 不变；
- 刷新本身的 reward 增量为零；承载该刷新的 transition 可以按现行合同计入本次既有 planner 成本；
- 同一稳定候选的 candidate ID 不变；
- 仅 available mask、reserve 选择、policy observation revision 和失败诊断允许变化。

## 10. 重写终止语义

### 10.1 成功优先且唯一

只有真实 sensor boundary 使正式覆盖率首次跨过 `0.95`，才能产生：

```text
TerminalReason.SUCCESS
```

无候选、planner failure、oracle 为零、step/stagnation 计数或模型输出均不能产生成功。

### 10.2 终止矩阵

在每次候选刷新后按以下顺序审计：

| production 状态 | oracle | 处理 |
| --- | ---: | --- |
| 存在 available candidate | 合法 | `DECISION_READY`，继续策略决策 |
| available 为空，但有未尝试 reserve | 正数 | reserve 补位后继续 |
| `physical_candidate_universe` 为空 | `0` | 合法物理耗尽，选择细分原因，未达阈值则失败终止 |
| `physical_candidate_universe` 为空 | `>0` | `CANDIDATE_ORACLE_MISMATCH`，fail closed |
| universe 非空且全部在当前 snapshot 规划失败 | `>0` | `PLANNER_BLOCKED_WITH_OPPORTUNITY`，有效失败终止 |
| universe 非空 | `0` | 生产候选/oracle 定义不一致，fail closed |

### 10.3 合法物理耗尽细分

只有同时满足：

```text
physical_candidate_universe_count == 0
AND oracle_opportunity_count == 0
```

才允许从以下原因中选择一个：

- `ZERO_GAIN`；
- `VISITED_EXHAUSTED`；
- `NO_RECOVERABLE_OBSERVATION_STATE`；
- `NO_TRANSIT_OPPORTUNITY`。

主原因使用生产候选流水线中最后一个实际耗尽阶段；全部阶段计数保留为 diagnostics。若 remaining truth-side coverable detail 仍为正，它只能用于离线诊断，不能改变 observed-only 主原因或生成候选。

### 10.4 Planner blocked

`PLANNER_BLOCKED_WITH_OPPORTUNITY` 必须同时满足：

```text
physical_candidate_universe_count > 0
planner_failed_current_snapshot_count == physical_candidate_universe_count
untried_reserve_count == 0
available_candidate_count == 0
oracle_opportunity_count > 0
```

它表示“物理上仍有观测机会，但当前 planner 无法为任何生产候选生成安全 reference”。它是可训练、可审计的失败 episode，不是 success、frontier exhaustion 或基础设施异常。closed-loop gate 必须单独统计并限制该原因。

### 10.5 Fail closed

以下状态是实现或合同错误，必须丢弃 rollout 并停止正式训练入口：

- 生产物理候选为空而 oracle 为正；
- 生产物理候选非空而 oracle 为零；
- 候选 ID 在无物理变化的刷新中漂移；
- 失败 candidate ID 不属于当前 physical snapshot；
- reserve 未耗尽却进入 terminal；
- 刷新重复增加 sensor、coverage、time 或 reward；
- planner hard failure 被记录成普通 candidate failure。

### 10.6 删除旧语义

下列标识和判断从正式环境合同删除：

- `PLANNER_REJECTED_ALL`；
- `planner_rejected_count == emitted_count` 直接终止；
- planner rejection 参与 `NO_RECOVERABLE_OBSERVATION_STATE`、`ZERO_GAIN` 或 `VISITED_EXHAUSTED` 计数；
- oracle 正数仍允许解释为探索完成。

## 11. Diagnostics 与 replay 合同

### 11.1 Candidate diagnostics

正式 diagnostics 至少包含：

```text
physical_snapshot_id
physical_reachability_algorithm_id
physical_candidate_universe_count
selected_policy_candidate_count
available_candidate_count
untried_reserve_count
planner_failed_current_snapshot_count
zero_gain_count
visited_excluded_count
physical_unreachable_count
oracle_opportunity_count
remaining_coverable_detail_cell_count
```

Primitive state/edge count 可以作为 planner diagnostics 保留，但不能参与候选合法性、耗尽状态机或 formal cache eligibility。

### 11.2 Planner diagnostics

地面 planner 至少记录：

```text
additional_corridor_margin_m = 2.0
corridor_half_width_m
search_domain_cell_count
search_domain_sha256
local_search_runs
local_frontier_attempts
global_replans
physical_goal_feasible
candidate_disposition
```

### 11.3 Replay state

新 worker/checkpoint state 必须持久化：

- 当前 `physical_snapshot_id`；
- 当前 snapshot 的 planner-failed candidate ID 集合；
- 物理候选 universe identity 和排序身份；
- 当前 observation identity；
- ground option 必须已完成或明确处于不可快照状态；
- reveal history 和 deferred candidate rebuild 状态。

恢复后必须逐位重现 candidate mask、reserve 补位、oracle count 和终止结果。

## 12. Cache 与训练身份迁移

### 12.1 Formal cache v6

新 schema 固定为：

```text
lunar-formal-training-cache/v6
```

平台 coverability 身份至少包含：

```text
physical_reachability_algorithm_id
physical_projection_schema
physical_safe_pose_count
physically_reachable_pose_count
physical_projection_sha256
mission_target_detail_mask_sha256
coverable_detail_mask_sha256
sensor_visibility_algorithm_id
capability content identity
start identity
```

cache eligibility 不再要求 primitive-set、primitive-state-schema 或 primitive-graph hash。若 planner 另行缓存原语图，该缓存属于 planner artifact，不属于 formal coverability 身份。

### 12.2 Training semantics v11

新 canonical identity 固定为：

```text
lunar-training-semantics/sensor-30m-360-platform-physical-coverable-detail95-observed-physical-candidates-fixed2m-search-domain-snapshot-planner-failure-option-path-observation-auditable-failure/v11
```

### 12.3 不兼容迁移

以下旧材料必须 fail closed：

- formal cache v5；
- training semantics v10；
- 依赖 primitive candidate/state identity 的 checkpoint；
- 旧 worker replay state；
- 旧 closed-loop gate report。

不得兼容续训或通过字段补默认值伪装兼容。新正式训练必须重新生成 cache、冻结 worker/micro-batch，并从 step 0 开始。

正式入口仍必须先验证真实 WHEELED、LEGGED、HOPPER capability 材料和 capability freeze；测试/代理 capability 不得解锁正式训练。

## 13. 实施边界

### 13.1 C++

主要修改：

- `planner_config.hpp`：冻结 additional margin；
- `LocalPlanningProblem`：增加独立 search domain；
- `local_frontier.cpp`：停止改写地图 layer；
- wheel/legged search：物理投影与搜索范围分离；
- `planner.cpp`：最终 candidate disposition；
- training bridge 与 Python bindings：导出 disposition 和新增 diagnostics。

### 13.2 Python

主要修改：

- `coverability.py`：物理 reachability 身份和分母；
- `candidate_builder.py`：物理候选、稳定 ID、universe/reserve；
- `formal_cache.py`：cache v6；
- `formal_start_qualification.py`：物理起点和候选资格；
- `formal_builder.py`：移除 primitive candidate 主路径并接入失败刷新；
- `observation_boundary.py`：无新 reveal 的权威 observation rebuild；
- `v3_environment.py`：按 disposition 更新 snapshot failure、刷新和新终止矩阵；
- checkpoint/replay/closed-loop gate：新身份和状态。

### 13.3 原子集成

实现可以分成多个提交，但合入 `integration` 的功能分支必须同时满足：

- 新物理分母与新物理候选；
- 新 oracle；
- 新 candidate failure/refill；
- 新终止语义；
- cache v6、semantics v11 和 checkpoint fail-closed；
- 完整回归。

不得发布以下中间组合：

- 新物理候选 + 旧 primitive oracle；
- 新 oracle + 旧 `PLANNER_REJECTED_ALL`；
- 新终止语义 + 旧 cache/checkpoint identity；
- search-domain 分离只在 WHEELED 生效但 LEGGED 仍改写地图。

## 14. 验证计划

### 14.1 C++ 单元与回归

1. 固定 additional margin 恰为 `2.0 m`，其他值拒绝；
2. BuildLocalFrontiers 后原始 local map 所有 layer 逐位不变；
3. search domain 几何、horizon 和 route corridor 正确；
4. footprint 可读取搜索域外的原始安全地图，不把域边界当障碍；
5. 历史窗口边界案例不再在展开前返回假 `WHEEL_GOAL_INFEASIBLE`；
6. 原始地图真实目标不安全时仍返回物理 goal infeasible；
7. 固定搜索域真实耗尽时返回 `LOCAL_SEARCH_DOMAIN_EXHAUSTED`；
8. hard/cancel/resource/numerical 输出 disposition 为 `KEEP`；
9. 正常最终目标级失败输出 snapshot suppression disposition；
10. WHEELED 和 LEGGED 使用同一 map/domain 分离合同。

### 14.2 物理机会语义

1. truth 分母只由物理安全、连通性和 sensor/LOS 决定；
2. observed 候选不读取 truth coverable mask；
3. oracle 不调用生产 CandidateBuilder；
4. 改变 planner primitive set 时分母、候选 universe 和 oracle 逐位不变；
5. WHEELED/LEGGED 的障碍、坡度、净空和支撑负例被排除；
6. Hopper 不安全落点、飞行管冲突和物理不可连通落点被排除；
7. 三平台同一输入重复计算的身份和排序逐位一致。

### 14.3 候选失败与刷新

1. 首个候选失败后不永久删除，当前 snapshot 内被抑制；
2. 未尝试 reserve 自动补位；
3. 候选重排后失败仍绑定原 stable ID；
4. 无物理变化刷新时 snapshot ID 不变；
5. 真实移动/map evidence 更新后 snapshot ID 改变，旧失败目标恢复资格；
6. planner hard failure 不修改失败集合；
7. 初始规划失败刷新不增加 sensor/time/coverage/reward；
8. 三段执行后第四段失败，前三段 evidence、time、coverage 和 reward 只计一次；
9. ground option 清除后立即使用当前真实位姿重建候选；
10. checkpoint/replay 精确恢复失败集合和 reserve。

### 14.4 终止语义

1. 覆盖率首次跨 `0.95` 才成功；
2. available candidate 或 reserve 存在时不得 terminal；
3. production empty + oracle positive 必须 fail closed；
4. production nonempty + oracle zero 必须 fail closed；
5. 所有候选规划失败且 oracle 为 `146` 时只能是 `PLANNER_BLOCKED_WITH_OPPORTUNITY`；
6. `PLANNER_BLOCKED_WITH_OPPORTUNITY` 不得标记 success/frontier exhausted；
7. 合法物理耗尽必须同时满足 production empty 和 oracle zero；
8. `PLANNER_REJECTED_ALL` 在正式枚举、指标和测试中不存在；
9. remaining truth coverable count 只作为诊断，不改变 observed-only 终止原因。

### 14.5 集成与门禁

1. 精确历史目标在 map/search 分离后使用固定 `2.0 m` 通过；
2. 若该目标仍因真实搜索域不足失败，停止并报告设计门槛，不得悄悄扩大到 `4.0 m`；
3. 完成 ROS core、training bridge、Python environment、formal cache、checkpoint/replay 和 closed-loop gate；
4. 运行 repository boundary 检查；
5. 所有 build/install/log、cache、训练输出和诊断 artifact 保持在仓库外；
6. 新 formal entrypoint 明确拒绝 cache v5、semantics v10、旧 checkpoint 和不完整 capability 材料。

## 15. 实施顺序

1. 将历史滚动失败和 `oracle=146` 固化为失败回归；
2. 实现固定 `2.0 m` 和物理地图/search-domain 分离；
3. 完成 WHEELED/LEGGED 真实正负例；
4. 将 coverability、候选和 oracle 切换为物理机会；
5. 增加 stable candidate ID、physical snapshot、universe/reserve；
6. 增加 structured candidate disposition；
7. 接通 initial/rolling planning-failure refresh；
8. 重写终止矩阵和 diagnostics；
9. 升级 cache v6、semantics v11、checkpoint/replay 和 gate；
10. 执行三平台全链验证并形成仓库外证据。

## 16. 非目标

本文首版不实现：

- 动态走廊扩宽；
- `0.4 m → 4.0 m` 恢复状态机；
- 永久候选黑名单；
- 环境层自动选择下一个候选并隐藏策略决策；
- oracle 候选直接填入策略 batch；
- 运动原语图删除或 planner 内部算法重写；
- 网络结构、七输入名称或动作容量变更；
- 从旧 cache/checkpoint 兼容续训；
- 代理 capability 解锁正式训练；
- 正式训练本身。

## 17. 完成标准

本文完成的必要且充分条件是：

- 地面 additional margin 在所有正式入口固定为 `2.0 m`；
- 局部搜索域不再修改物理地图；
- 物理可行性不再被人工窗口边界污染；
- 三平台覆盖分母、候选和 oracle 只由物理机会定义；
- planner primitives 不影响探索机会身份；
- 规划失败只在当前 physical snapshot 临时抑制 stable candidate；
- reserve 能补位，真实移动后候选恢复资格；
- initial 和 rolling failure 都在权威 sensor-closed-loop 边界刷新；
- `PLANNER_REJECTED_ALL` 被完全删除；
- oracle 正数永远不能被解释为探索完成；
- cache v6、semantics v11 和新 replay identity fail closed；
- 历史失败目标和 `oracle=146` 回归通过；
- 正式训练仍保持未启动，直到 capability、cache、worker/micro-batch 和 closed-loop gate 全部重新冻结。
