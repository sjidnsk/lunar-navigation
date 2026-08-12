# 地面平台全局机会目标与锁定执行设计

日期：2026-08-12
状态：已批准，作为
`2026-08-11-physical-opportunity-planner-failure-semantics-design.md`
的地面平台增量覆盖；Hopper 合同不变。

## 1. 问题与结论

当前 WHEELED/LEGGED 候选生成与独立 Oracle 都把传感器半径
`30.0 m` 同时当作候选目标相对当前位姿的最大距离。该绑定会把
“当前 30 m 邻域没有正收益”错误解释成“整幅地图没有正收益”。当机器人位于一片已观测的百米
平原内部，而远端前沿仍位于物理可达连通分量中时，生产候选可能只剩普通零收益站位，独立 Oracle
也看不到远端机会，最终导致错误候选或过早终止。

批准的修复是：地面平台使用全局正收益目标，不再设置固定候选目标距离上限。策略从整幅
`1024 m × 1024 m` 的已观测物理可达区域中选择一个正收益目标；全局规划器规划到该远目标，局部
规划器在每次传感器更新后继续滚动规划，直到抵达目标或出现明确失败/终止条件。传感器范围仍为
`30 m/360°`，只决定“站在候选位置能看到多少新信息”，不限制候选位置离当前机器人多远。

地面平台不以普通零收益 `TRANSIT` 代替全局正收益目标。Hopper 仍使用既有
`OBSERVATION → TRANSIT → BACKTRACK` 单跳分层语义。

## 2. 四个距离边界

实现必须把以下四个量分开，禁止再次共用一个 `sensor.range_m` 隐式表达：

| 边界 | WHEELED/LEGGED | HOPPER |
| --- | --- | --- |
| 传感器观测半径 | 冻结 `30.0 m` | 冻结 `30.0 m` |
| 候选搜索域 | 整幅当前 observed-only 物理可达连通分量 | 当前认证单跳后继 |
| 全局目标距离 | 无固定欧氏上限；受地图、证据与物理可达性约束 | 受单跳能力包络约束 |
| 单次局部 reference 长度 | 由既有局部 horizon 决定并滚动续接 | 一次认证 hop segment |

`ProjectReachability()` 对地面平台已返回整幅 global safe connected component；其
`maximum_edge_distance_m` 当前只进入投影诊断字段，不截断 ground mask。实现不得通过传入一个巨大
浮点数伪造“无限范围”，而应在 Python 地面候选和 Oracle 边界删除错误的传感器距离过滤。Hopper
继续把该参数作为单跳边距离约束。

## 3. 地图职责

### 3.1 4 m global 地图

固定几何为 `256 × 256 @ 4.0 m`，覆盖 `1024 m × 1024 m`。地面候选位置、全局安全连通分量、
frontier segment、全局路径代价和空间分段均在该层表达。一个 global 单元只有其对应
`20 × 20` 个 `0.2 m` detail 单元全部已观测时才可作为 global known 通行证据。

候选位置必须满足：

1. 位于当前 observed-only 物理安全、从当前精确状态可达的地面连通分量；
2. 位于 mission ROI 中，且不是当前机器人单元；
3. 使用候选位置的精确代表 `x/y/z`；
4. 经 0.2 m 收益核验后具有严格正 mission gain；
5. 未被当前物理快照的 planner-failure 集合屏蔽。

frontier 只用于空间分段和排序提示。生成器必须同时枚举全局可达安全观测站位，不能只保留 frontier
边界自身，也不能因已找到一批近端候选便提前停止。

### 3.2 0.2 m detail 地图

detail evidence 负责：

- 在每个 4 m 候选代表位置计算 `30 m/360°` observed-only mission/priority gain；
- LOS、障碍、净空、支撑、坡度和局部运动认证；
- 每段 reference 执行后的 reveal、coverage numerator 和物理证据 revision。

收益计算继续遍历传感器圆盘内所有 detail endpoint，并使用既有 Bresenham 遮挡合同。不得在 4 m
层用“一个 coarse 格未知”近似 400 个 detail 单元的收益，也不得读取冻结 truth coverable mask。

## 4. 全局候选生成

每次策略决策边界按以下顺序构造地面候选：

1. 从地面 `PhysicalReachabilityResult` 取得整幅 observed-only 物理可达 mask 和精确代表位置；
2. 排除当前单元、静态不安全、非 ROI、已访问和当前快照已失败的位置；
3. 对剩余全部位置批量计算 0.2 m 精确增益；
4. 严格排除 `mission_gain == 0` 的地面候选，但将数量保留为 `zero_gain_count`；
5. 先形成截断前的完整正收益 canonical universe；
6. 按 frontier segment 覆盖、预计收益、全局路径代价、物理距离和稳定 candidate ID 确定性排序；
7. 从完整 universe 生成 top-64 batch 和 reserve；不足槽位使用 false mask，不复制候选。

候选的距离 feature 仍按整幅 global canvas 对角线归一化，因此远目标可在现有 `[64,12]` 张量中
表达。候选身份继续绑定平台、精确目标、mission revision、目标容差和物理快照；生成路径、top-64
名次和 active/reserve 状态不进入 candidate ID。

普通零收益地面站位不能进入 universe、reserve 或 batch。由于策略已能直接选择远端正收益目标，
本设计不引入地面 `TRANSIT` 角色。

## 5. 独立 Oracle 与终止

地面 Oracle 必须独立扫描同一整幅物理可达位置集合，但继续独立计算收益，不能复用生产候选的 gain
数组、排序结果或截断结果。删除 Oracle 对机器人距离 `<= 30 m` 的过滤；`30 m` 只进入每个候选
位置的 visibility footprint。

终止矩阵保持 fail-closed：

| 生产正收益全集 | Oracle 正收益全集 | 结果 |
| ---: | ---: | --- |
| `>0` | `>0` | 正常选择/继续 |
| `0` | `0` | 合法 `ZERO_GAIN`/物理机会耗尽 |
| `0` | `>0` | `CANDIDATE_ORACLE_MISMATCH` |
| `>0` | `0` | `CANDIDATE_ORACLE_MISMATCH` |

合法耗尽表示整幅当前 observed-only 地面可达连通分量中不存在正收益观测位置，不再表示当前
30 m 邻域为空。覆盖率达到冻结 success threshold 的规则不变。

## 6. 远目标锁定与滚动执行

一个地面策略 action 选择一个全局目标，并创建一个锁定 ground option。环境不得在中间 reference
边界重新调用 policy 或暗中换目标。

执行循环固定为：

1. 对锁定目标执行 global + local 规划；
2. 执行返回的一个安全 local reference；
3. 应用该 reference 的传感器 reveal，更新 pose、coverage 和 evidence identity；
4. 以相同目标和最新证据构造 continuation request；
5. planner 可复用仍有效的 global route，或在证据变化使其失效时重新全局规划；
6. 重复直到到达目标或产生本节定义的终态。

现有固定 `_MAX_GROUND_OPTION_REFERENCES = 64` 必须移除，因为它会成为与地图尺度和 reference
horizon 相关的隐式距离上限。不得用更大的固定次数替代。

ground option 只能在以下条件结束：

- 当前位姿进入冻结 `0.2 m` 目标容差；
- episode 首次跨越 success threshold；
- 新证据使目标退出物理候选定义；
- planner 完成正常 global/local 尝试后返回结构化最终失败；
- hard safety、取消、资源耗尽或数值失败；
- 触发与目标距离无关的卡死保护。

卡死保护不得重新要求每段 reference 单调降低目标欧氏距离，因为合法绕障可能暂时远离目标。它必须
使用结构化证据，并至少检测：

- 同一 evidence identity 下重复出现相同 route cursor、相同 reference endpoint 和相同机器人状态；
- 连续执行后 pose/evidence/route cursor 均无变化；
- continuation identity 或目标 candidate identity 发生不合法漂移。

卡死属于 `GROUND_OPTION_STALLED`/基础设施或规划失败，不能写成 frontier exhaustion。外部 wall
watchdog 仍用于防止进程失控，但 timeout 不是有效探索终止。

一次锁定远目标产生一个 PPO 宏 transition。沿途所有 reference 的 mission gain、priority gain、
路径/执行成本和时间逐项累计；最终 observation 是抵达或失败边界的最新 observation。宏动作期间的
中间 reveal 不产生新的 policy action，但必须更新规划证据并接受 success/safety 检查。

## 7. 性能与有界性

取消候选距离限制不等于逐候选调用 Planner。候选阶段必须：

- 对全局可达 mask 一次行主序枚举；
- 对候选位置批量调用 detail visibility kernel；
- 使用 frontier segment 和 deterministic top-M 压缩；
- 只为策略实际选择的目标调用 Planner；
- 保留当前单次规划 `< 1 s` 的 WHEELED/LEGGED 性能门。

整幅 global map 最多 `65,536` 个 coarse 单元；候选收益批处理规模受当前 observed-safe reachable
单元数约束，不构造 `5120 × 5120` 的完整 dense detail 世界。若真实闭环证据显示批量 visibility
超出预算，只允许做等价的 tile/endpoint 缓存或向量化，不能恢复 30 m 候选截断。

## 8. 身份与迁移

该变化修改地面候选全集、Oracle、宏动作持续时间和策略 transition 语义，必须同时升级：

- training semantics identity；
- physical candidate/cache schema identity；
- checkpoint schema identity；
- formal cache 与 run manifest；
- closed-loop deterministic evidence。

旧 checkpoint 不得直接 resume，旧 cache 不得混用。Hopper 的候选、Oracle、单跳执行和已有成功
闭环证据保持原合同，但仍需在共享 schema 升级后完成回归。

## 9. 验证门

### 9.1 定向 RED/GREEN

1. 地面远端正收益位置距机器人超过 30 m 时仍进入 production universe 和 Oracle；
2. 近端全零、远端正收益时不得 `ZERO_GAIN`；
3. 全局所有地面可达位置均零收益时 universe/available/selected/reserve 为零，`zero_gain_count > 0`，
   Oracle 为零并合法终止；
4. top-64 截断不改变截断前 universe/Oracle 身份；
5. HOPPER 仍只暴露 direct `OBSERVATION/TRANSIT/BACKTRACK`；
6. 一个超过 64 个 local references 的远目标可以自然抵达；
7. 合法绕障可暂时增加欧氏目标距离而不被拒绝；
8. 重复状态/游标/endpoint 的真实卡死 fail-closed；
9. 远目标沿途所有 gain/cost/time 汇总到一个宏 transition；
10. 同一输入重复构建候选、Oracle 和执行结果逐位确定。

### 9.2 回归与发布门

- C++ planner/core、training bridge 与 ROS 全量测试；
- Python 全量 regression；
- fresh formal cache、identity、historical replay；
- WHEELED/LEGGED/HOPPER 独立完整闭环，单个平台失败不得停止其他平台；
- 两次正式 deterministic closed-loop gate 及 artifact hash 对比；
- WHEELED/LEGGED 每次规划 `< 1 s` 性能门；
- 不允许以 watchdog、skip、阈值放宽或旧 artifact 代替通过。

## 10. 非目标

- 不改变传感器 `30 m/360°` 合同；
- 不把候选位置改为全量 0.2 m 网格；
- 不放宽 4 m global known、固定 2 m corridor/search-domain 或平台安全约束；
- 不改变 Hopper 单跳策略边界；
- 不让 Oracle 候选填充 policy batch；
- 不把 timeout、planner blocked 或 candidate/oracle mismatch 解释为成功。
