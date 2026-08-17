# Reward V4 Training Evaluation Throughput Design

## Goal

在不改变 Reward V4、策略结构、候选动作、完整宏动作采样语义、课程门或 checkpoint 选择规则的前提下，把当前每 3 小时触发的 24 个自然终态任务评估拆成快速哨兵与低频完整评估，避免评估长期占住训练；同时让评估可按任务恢复，并补齐宏动作内部耗时证据。

## Confirmed cause

当前更新的采样约百秒、PPO 更新约半秒，但候选 update 进入固定评估后会同时运行三种 seed、四种尺度和全部活跃平台，并等待每个任务自然终止。评估任务没有宏动作上限，也没有逐任务持久化，因此一次评估可持续数小时；进程中断后还会丢失已经完成的任务。这是当前训练停顿的主要来源，不是 GPU PPO、场景预取或传感器计算。

## Design

### Two evaluation tiers

训练仍按冻结配置的 `evaluation_interval_gpu_s` 触发评估边界，但边界分为两类：

1. **快速哨兵评估**：使用冻结 seed 列表中的第一个 seed、所有活跃平台和四个尺度；每个任务恰好执行一个完整宏动作。它检查候选 checkpoint 是否能完成真实策略选择、规划、执行、奖励和状态提交，并对硬错误 fail closed。
2. **完整评估**：保留现有三个 seed、所有活跃平台、四个尺度和自然终态。只有完整评估可以推进课程、比较最佳 checkpoint 或触发回滚。

完整评估按累计 GPU 预算每 43,200 秒触发一次；其他原有 10,800 秒评估边界执行快速哨兵。该调度是运行时资格策略，不加入冻结 Reward V4 配置，因此现有 checkpoint 的 config hash 和奖励身份不变。

快速哨兵的报告只作为运行健康证据。它通过后更新照常原子应用，但不调用课程评估、不改变 best-checkpoint 状态，也不把一宏动作覆盖率误当成自然终态成绩。

### Durable evaluation mode and task progress

每个候选 update 在其评估目录内原子写入与 checkpoint payload 绑定的 `evaluation-mode.json`，冻结本次是 `sentinel` 还是 `full`。恢复时必须复用该模式，不能因进程重启或代码路径重入改变任务集合。

每个评估任务完成后，以平台、尺度和 seed 为身份原子写入独立 episode 文件。文件同时绑定 checkpoint payload 和评估模式。恢复时严格验证内容并只启动尚未完成的任务；损坏、重复或身份不一致的文件 fail closed。全部任务完成后仍生成现有 content-addressed Reward V4 报告。

### Timing evidence

宏动作总耗时继续保留。新增诊断只记录已有阶段的耗时和计数，至少区分策略推理、候选刷新、全局候选可达搜索、滚动局部规划调用及其累计耗时。一次局部规划请求的耗时与一个完整宏动作的耗时不得混用。

本轮只补证据，不据此改变搜索域、规划器阈值、候选或奖励。后续优化必须由这些计时证明单一热点后再单独实施。

## Recovery and atomic cutover

新代码验证完成前不停止旧训练。切换时先停止旧 controller，保留 run、journal、候选 checkpoint 和评估目录；再从最新 applied checkpoint 或已经封存的 candidate update 恢复。若候选 update 已存在但旧完整报告未完成，新运行将为它冻结快速哨兵模式，执行八个单宏动作任务，通过后直接应用该 update，不重新采样或 PPO。

## Non-goals

- 不减少每个训练 worker 的完整宏动作数，也不改变 12 宏动作 update 边界。
- 不把宏动作改成按墙钟截断，不改变规划成功、探索完成或奖励定义。
- 不提高 worker 数量，不优化 PPO/GPU，不修改冻结 Reward V4 配置。
- 不允许快速哨兵推动课程、选择最佳 checkpoint 或替代发布资格评估。

## Acceptance

- 快速哨兵任务在一个完整宏动作后结束，完整评估仍等待自然终态。
- 快速哨兵硬错误阻止候选应用；无硬错误时不改变课程与 best-checkpoint。
- 评估中断后只重跑未完成任务，已完成 episode 可验证复用。
- 旧 candidate update 可直接进入快速哨兵，不重新采样和 PPO。
- focused tests、最小真实哨兵烟测和 `git diff --check` 通过；切换后出现 applied update 和下一轮新采样证据。
