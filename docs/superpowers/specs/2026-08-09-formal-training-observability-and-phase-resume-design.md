# 正式训练阶段恢复、逐更新指标与周期评估设计

## 背景与目标

seed 4080 正式训练在轮式预热结束时保存了 `global_step=118` 的完整 v6 检查点，随后课程分配由
`WHEELED×24` 切换为 `LEGGED×24`。现有代码把轮式 worker 的活动 episode 状态直接交给足式
worker 池恢复，被 worker 身份校验正确拒绝。与此同时，PPO 更新已经计算 loss、entropy、KL、
梯度等指标，但正式训练调用方丢弃返回值；设计中的 `metrics/train.jsonl` 和联合阶段每三小时
评估没有接入运行循环。

本设计完成三个闭环：

1. 跨平台或改变 worker 分配时能够从安全检查点继续，不伪造 episode 成功或失败；
2. 每个已完成 PPO update 都留下可流式读取、语义稳定的训练指标；
3. 联合阶段按累计 GPU 工作时间每三小时自动评估最新不可变候选，并继续训练或在门通过时结束。

## 阶段切换与恢复语义

检查点中的状态分成两组：

- 全局训练状态：模型、优化器、scheduler、normalization、Python/NumPy/Torch RNG、global step、
  累计预算和候选/最新检查点计时器；
- worker 活动 episode 状态：平台、lane、scene/start/cursor、pose、观测历史和 observation identity。

同一课程阶段、相同 worker allocation 的普通恢复必须同时恢复两组状态，继续保持现有逐字节恢复
语义。只有当累计预算推导出的下一阶段与检查点阶段相邻，且目标 allocation 与检查点 allocation
不同，才允许只恢复全局训练状态并丢弃旧 worker 活动状态。目标平台按其独立场景 schedule 从
cursor 0 建立新 episode。

这不是给旧 episode 写入 terminal：不生成 reward、`done`、成功、失败或覆盖统计，也不修改旧
episode cursor；旧 worker 池只是随课程分配退役。任何非相邻阶段、同阶段 allocation 漂移或
外部配置漂移仍然 fail closed。

从当前 step 118 恢复时，模型、优化器、RNG、global step 和 7269.1511 GPU seconds 继续累计；
`WHEELED×24` 活动状态不注入 `LEGGED×24`，足式预热从新的足式 worker episode 开始。轮式预热
不重跑。

## 逐 update 指标日志

正式训练在 `<artifact-root>/metrics/train.jsonl` 写入 schema
`lunar-training-update-metrics/v1`。只有 trainer 完成更新、scheduler 前进且 policy version 成功
提交后才写一行；失败或被丢弃的 rollout 不写。

每行至少包含：

- `global_step`、UTC 时间、课程阶段、平台 allocation、rollout horizon 和 transition 数；
- 原始 reward 的 mean/std/min/max，并按平台给出 mean；
- update 起始/结束任务覆盖率均值和覆盖增量；
- terminal 数、首次成功跨越数、planning outcome 计数和 planner 成功率；
- total/policy/value loss、frontier/theta entropy、approximate KL、gradient norm、裁剪后梯度、
  参数变化 L2、optimizer steps、early-stop 标志；
- rollout 收集耗时、optimizer 耗时和 worker 等待耗时。

日志由主训练进程单写，使用 `O_APPEND`、单次 UTF-8 行写入和 `fsync`。启动或恢复时读取已有
日志并验证 schema 与严格递增的 `global_step`；最后一步不得超过恢复检查点，避免把未持久化更新
误认为已恢复状态。当前 step 118 之前没有逐 update 数据，首次新增行从 step 119 开始，不能用
参数差分伪造历史曲线。

`run-manifest.json` 继续只保存当前状态摘要和 artifact 身份；每次 latest checkpoint 时增加最后
已记录指标步和 JSONL 文件哈希，避免把大量时间序列复制进 manifest。

## 三小时自动评估

自动评估仅在 `joint` 阶段启用。第一次评估目标为 joint 起点之后 10800 累计 GPU seconds，后续
以最近一次评估开始点再加 10800 seconds。训练在完整 update 边界保存 `latest.pt`、关闭训练
worker 池，然后评估最近的 `candidate-step-*.pt`；不存在候选属于运行时错误，不能退化为评估
可覆盖的 `latest.pt`。

评估使用 validation、test、holdout 三个冻结 split 和三种方法，继续执行 split×platform 独立
门控。评估与训练串行共用同一个 `TrainingBudget`，正式评估 watchdog 的 3600 seconds 是单次
预算预留上界；不得创建第二个预算对象或让训练与评估同时占用 worker/GPU。

每次结果写入不可变目录
`<artifact-root>/evaluation/candidate-step-<N>/`，同时更新顶层 latest report 兼容现有命令。
manifest 的 `last_evaluation` 记录 checkpoint、评估开始/完成 GPU seconds、报告哈希和门控结果。
门通过则训练以 `release-gate-passed` 结束；未通过则从评估前 latest checkpoint 恢复同一 joint
活动 episode，评估造成的 RNG 变化由恢复检查点覆盖，而评估消耗的预算继续累计。

## 错误处理与恢复

- 指标 JSON 非有限、schema 不符、步号倒退或同一步重复时立即停止训练；
- 自动评估异常或 watchdog 超时保留 latest/candidate 和错误日志，但不把它记为策略不可行；
- 阶段切换只接受冻结课程中的相邻变化；其他 worker 身份不匹配仍由 `ParallelEnvPool` 拒绝；
- 修复提交改变了严格检查点中的 `source_commit`，因此不能绕过校验直接恢复。先把 step 118 原文件
  复制为不可变备份，再生成一个只改变 `source_commit` 和随之变化的 payload hash 的 v6 resume
  副本；模型、优化器、scheduler、normalization、RNG、episode、step、预算和所有正式 run identity
  字段必须逐项保持不变；
- source 迁移只接受旧提交到当前后代提交、显式 step 118 和权威 `latest.pt`，并在 manifest 记录
  新旧提交、两个 payload hash、原文件 hash、备份/副本路径和实际变更文件清单。正式 `resume`
  使用迁移副本；原 step 118 文件在成功写出 step 119 前不被修改；
- 训练、指标、评估报告和检查点全部留在仓库外，不提交运行 artifact。

## 验证

测试必须覆盖：

1. wheel→legged、legged→hopper、hopper→joint 时不注入旧活动 episode，但恢复模型/RNG；
2. 同 allocation resume 继续精确恢复 worker state；非法 allocation 漂移仍失败；
3. 每个成功 update 恰好写一行，失败 update 不写，恢复步号连续且 JSON 值有限；
4. 指标中的 reward、覆盖、success、planner outcome 和 PPO 数值来自真实 rollout/metrics；
5. joint 三小时边界选择最新 candidate、关闭训练池后共用预算评估；未通过继续，通过则结束；
6. step 118 真实检查点预检能够建立 `LEGGED×24` 池而不消耗一个训练 update；
7. source 迁移前后除提交字段和 payload hash 外的语义 body 完全相同，原文件有字节级备份；
8. 训练相关 Python 测试、CUDA 短恢复测试、仓库边界检查全部通过。
