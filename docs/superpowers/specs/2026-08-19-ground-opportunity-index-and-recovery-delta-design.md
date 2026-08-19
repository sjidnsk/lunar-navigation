# 地面机会索引与增量恢复状态设计

## 状态

已在讨论中批准设计方向；本文在实施前冻结接口、语义和性能边界。

## 背景与问题

`VALID_INCOMPLETE_TERMINAL` 的修复把“普通前沿候选为空”接到
`CandidateBuilderV2.build()` 中的 `GROUND_EXHAUSTION` 全量回退。该回退会在
一个 worker 内枚举当前全部物理可达的 0.2 m 观测站位，进行端点可行性、精确
可见性收益和排序。24 个 worker 拓扑早于该修复已存在；新增的是每个 worker
都可能同时进入一次全任务区重计算。因此，问题不是把 worker 数量误当作根因，
而是把完整性审计放进了同步候选刷新热路径。

恢复路径还有独立但耦合的成本：每个宏动作边界先把完整 worker 状态写入 journal，
正常封存时又反序列化和校验整组 payload，checkpoint 随后再次序列化等价环境状态。
观测轨迹、地图和候选历史随 episode 增长，导致采集、封存和恢复快照成本随 update
增长，而不是随本次实际观测工作增长。

## 目标

1. 保留地面候选完备性：不能因三个前沿候选都无收益而错误给出
   `VALID_INCOMPLETE_TERMINAL`。
2. 普通宏动作不得执行全任务区 exhaustion scan；其计算量只与本次脏区域、
   可达性变化和实际候选数相关。
3. 不向策略或候选选择泄漏隐藏真值；覆盖率分母仍不参与候选选择。
4. 保留宏动作边界原子提交和精确恢复；正常运行不重复解码完整恢复状态。
5. 让恢复状态大小由当前任务的已变更 tile 与分块历史决定，而不是由 update
   编号线性放大。

## 非目标

- 不改 Reward V4、PPO 动作空间、传感器语义、全局/局部规划器或平台能力。
- 不增加每段前沿的策略候选数量：普通路径仍是每段三个候选。
- 不把局部规划拒绝、隐藏可覆盖区域或任意连续站位当作探索完成证据。
- 不用简单降低 worker 数或全局 CPU 限流代替根因修复；资源上限只可作为保护栏。

## 总体架构

```text
轨迹观测提交
      │
      ├─ 脏 detail/coarse tiles + 可达掩码差分
      │             │
      │             ▼
      │      GroundOpportunityIndex（增量更新）
      │             │
普通三候选前沿 ─────┼─> 正收益候选查询 ─> 全局规划过滤 ─> 策略
      │             │
      │             └─> 普通候选为空时的完整性查询
      │                    ├─ 有正收益：形成候选，不终止
      │                    └─ 已追平且无正收益：合法耗尽终止
      │
宏动作边界 ─> 内容寻址状态块 ─> journal 引用 ─> checkpoint 引用
```

### 1. `GroundOpportunityIndex`

索引分为两层。

- `TaskOpportunityGeometry` 是任务级、内容寻址的静态几何：任务 ROI、0.2 m
  格点坐标、30 m 查询邻域和 tile 到观察站位的空间反向索引。它不得含有隐藏
  地形、障碍、可覆盖分母、真实收益或真值可达性。
- `GroundOpportunityIndexState` 是 episode 级、仅基于已观测数据的动态状态：
  当前物理可达站位掩码、端点认证结果、正收益站位集合、已观察 tile 版本和
  `index_generation`。

每次宏动作提交后，环境给索引提供：

1. 本次传感器写入的脏 tile；
2. 前后物理可达掩码的差分；
3. 当前观察证据版本与精确端点认证接口。

索引只重算两类站位：可达性差分中的站位，以及传感器 30 m 影响范围与脏 tile
相交的站位。端点局部窗口受影响时才重做端点认证；精确收益也只对受影响的、
端点可行站位批量计算。索引维护按稳定候选 ID 排序的正收益集合。

普通前沿候选仍从固定三点安全条带生成并由全局规划器过滤。只有普通集合为空时，
才查询已追平到当前观察版本的索引正收益集合；它不是重新枚举所有可达 0.2 m
站位的扫描。

若索引尚未追平当前观察版本，环境返回内部 `INDEX_PENDING` 边界，不生成 PPO
样本，也不宣布耗尽。后台维护按确定顺序处理有限 tile 批次；其它 worker 可以继续。
只有 `index_generation == observation_generation` 且正收益集合为空，才允许地面
平台产生合法无候选终态。全图重建仅允许用于离线一致性测试或故障诊断，不得由
`CandidateBuilderV2.build()` 在训练热路径调用；任务缓存只可构建不含真值的静态
几何索引，不得借缓存预先计算观测收益或端点可行性。

### 2. 并发边界

每个 worker 可以执行其增量 tile 更新；不得让所有 worker 为同一类型事件执行全
任务区扫描。索引初始化、缓存失配重建和离线一致性验证由独立协调器串行或采用小的
主机级 in-flight 预算执行，并记录为非 PPO 工作。这个预算是防止异常扇出的保护栏，
不是通过降低训练 worker 数量来掩盖算法成本。

运行时指标必须拆分并保留：`normal_candidate_s`、`opportunity_index_s`、
`endpoint_certification_s`、`exact_gain_s`、`journal_seal_s`、`checkpoint_s`。
这些是 transient telemetry，不进入 observation identity、恢复状态或候选排序。

### 3. 内容寻址的恢复状态

恢复状态拆成不可变对象块和轻量边界引用。

- 任务静态对象块：任务几何、能力、场景和 `TaskOpportunityGeometry`；按 SHA
  去重。
- episode tile 块：地图/观测状态只为变化的 tile 写新块；未变 tile 复用旧 SHA。
- 分块事件历史：轨迹和 reveal 历史按固定大小块追加，边界只保存最后块及游标，
  不把全部历史重复展开。
- 动态索引块：保存 `GroundOpportunityIndexState` 的 tile 版本、可达性差分和正
  收益集合引用；不保存可重新计算的临时数组或计时器。

提交顺序为：先原子写入并 fsync 新对象块，再写 journal slot 的对象引用和宏动作
结果，最后封存 update。checkpoint 保存模型、优化器和同一组对象引用/哈希；它不再
复制 journal 已拥有的完整 worker payload。

正常运行使用内存中已验证的已提交对象生成 rollout 与 seal：不得为 seal 前后各做
一次完整 `torch.load`。重启恢复仍必须对引用对象完整解码、校验 SHA、重建状态并与
journal/checkpoint identity 对照。只有确认没有 checkpoint 或 journal 引用的对象块
才可在 checkpoint 保留期之后回收。

## 终止语义

地面平台的无候选终止必须满足全部条件：

1. 普通前沿候选和 reserve 都没有可执行正收益候选；
2. 当前 `GroundOpportunityIndex` 已追平观察与物理可达性版本；
3. 索引正收益集合为空，或其候选已被全局规划器在同一物理快照中正式拒绝；
4. 没有硬错误、索引失配或未完成认证。

条件 1 不再直接触发全图 scan；条件 2 不满足时为 `INDEX_PENDING`；条件 1--3
同时满足才是合法耗尽。这样既不会把仍有机会的任务误标为
`VALID_INCOMPLETE_TERMINAL`，也不会让完整性检查使所有 worker 同时满负荷。

## 迁移与恢复

该设计需要新的 formal state/checkpoint/journal identity。已封存的旧 update 继续
可只读审计，但不能混合新的对象引用格式。生产切换从一个已应用 update 的宏动作
边界以 fresh episode 方式开始：保留策略与优化器状态，开始新的 episode，并从新格式
写入首个边界。这样不需要重放或重建旧 episode 的不断增长状态。

## 必要验证

1. 用相同 observed-only 输入，索引追平后的正收益集合、候选 ID、排序、端点认证
   和耗尽结论与当前全量 scan 逐项一致。
2. 普通三候选路径和普通候选为空后的索引查询都不得调用全任务区 scan。
3. 脏 tile、可达性增加、可达性减少、端点失效和零收益五类更新均与全量基线一致。
4. 索引未追平不能生成 `VALID_INCOMPLETE_TERMINAL` 或 PPO transition。
5. 正常 seal 不读取已提交 slot payload；模拟重启时必须完整读取并校验引用对象。
6. 同一宏动作序列的连续运行与崩溃恢复运行在 observation identity、候选、规划
   请求、奖励、覆盖率和 terminal reason 上逐项相同。
7. 24 worker 实测中，全任务区 scan 调用数为零；采集时间随连续 update 不出现由
   历史状态长度导致的单调增长。性能报告分别给出索引、封存和 checkpoint 耗时。
