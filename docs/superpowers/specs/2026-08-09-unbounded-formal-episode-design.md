# 无固定决策上限的正式探索回合设计

日期：2026-08-09  
状态：已批准，取代旧正式环境中的 8-step 预算与 optimizer-boundary episode rollover

## 问题与结论

当前正式 builder 没有显式传入 `total_decision_budget`，因此继承旧代理环境的默认值 `8`。
另一方面，PPO 每收集 `rollout_horizon=32` 个宏步完成一次更新后，训练入口调用
`rollover_all_workers()`，把全部 worker 换成新 episode。结果是正式回合同时受到显式 8-step
上限和隐式 32-step 上限，平方公里任务不可能在同一回合持续积累观测。

本设计作出以下冻结决定：

- 正式探索回合没有固定决策次数上限；不把 `8` 换成另一个更大的哨兵值；
- `rollout_horizon` 只决定每次 PPO update 收集多少 transition，不结束或重置回合；
- 完成的 worker 自动进入下一 episode，未完成的 worker 跨 update 保持同一 scene、start、pose、
  已观测状态和候选状态；
- 正式成功由同一任务 ROI 的累计实际观测率首次达到 `0.95` 唯一定义；
- 正式训练前比较 horizon `16/32/64`，`32` 只是初始候选，不是任务能力；
- checkpoint 保存并恢复活动 episode 的确定性动态状态，而不只保存 episode cursor；
- 正式评估运行至自然 terminal；技术 watchdog 超出时拒绝报告并标记
  `EVALUATION_INCOMPLETE`，不能伪装成任务不可行。

## 三个独立生命周期

| 生命周期 | 边界 | 是否结束 episode |
| --- | --- | --- |
| 探索宏步 | 候选选择、C++ 规划、参考执行、传感 reveal | 仅当产生自然 terminal |
| PPO rollout | 每个 worker 固定收集 `T` 个宏步并计算 GAE | 否 |
| 正式 episode | 从初始 reveal 到成功、无候选或异常 | 是 |

一个 episode 可以跨任意多个 rollout。每段 rollout 中的动作、value 和 `old_log_prob` 都由该段
开始时的同一 policy version 产生；段末使用 `V(s_T)` bootstrap。下一段允许使用更新后的 policy
继续同一环境状态，这是标准 on-policy 采集，不需要为了更新网络而重置世界。

## ObservationContractV3

旧 `pose_features [B,6]` 的第六项 `remaining_decision_budget_ratio` 在无固定预算任务中没有真实
含义。正式训练尚未开始，因此现在原子升级为：

```text
ObservationContractV3 = lunar-observation-contract/v3
pose_features float32 [B,5]

x_norm
y_norm
sin_yaw
cos_yaw
mission_observed_ratio
```

其余六个输入名称、shape 和语义不变，网络仍有七个输入。policy pose encoder 输入宽度同步从
6 改为 5。active 训练、PyTorch、ONNX、TensorRT、golden I/O、manifest 和 checkpoint 只接受
V3；V2 只作为历史 schema 常量保留，不能恢复为正式训练候选。
新生成的训练 checkpoint 同步升级为 `lunar-ppo-checkpoint/v6`；V5 只能作为 V2 历史格式读取，
不得直接恢复为正式 V3 训练。

collector 的“是否可行动”只读取 `candidate_mask` 和显式 environment terminal，不再读取
`pose_features[:,5]`。内部 `decision_budget_consumed` 更名为
`policy_decisions_consumed`，它只统计本次边界是否消费一个策略动作，不代表剩余资源。

## 正式回合状态机

生产 `V3ExplorationEnvironment` 删除 `total_decision_budget`、
`remaining_decision_budget`、`DECISION_BUDGET_EXHAUSTED` 和相关网络写回。测试若需要有界动作数，
使用 test-only vector environment，不允许重新向生产构造函数加入默认预算。

自然 terminal 固定为：

1. `SUCCESS`：ROI 加权实际观测率首次从 `<0.95` 跨到 `>=0.95`；
2. `NO_CANDIDATES`：最新真实 reveal 和被规划器拒绝的候选遮罩生效后，不存在可选择候选；
3. `HARD_FAILURE`：硬安全、合同、非有限数值或执行状态机错误；
4. `CANCELED`：外部明确取消。

`NO_CANDIDATES` 在未成功时设置 `episode_ended_without_success=True`。单个规划候选不可行只遮罩
该候选；同一 observation identity 中仍有候选时继续决策。候选生成继续排除当前位置和零潜在
增益，有限 ROI 由新增信息耗尽形成自然边界。

## ROI 覆盖与唯一成功事件

`ObservationBoundaryController` 是累计覆盖的唯一权威。设：

```text
mission_area = sum(mission_roi_ratio) * global_cell_area
observed_area = sum(actual newly revealed detail-cell area weighted by ROI)
coverage = observed_area / mission_area
```

初始 reveal 计入覆盖但不产生动作 reward。每个后续 reveal 先保存 `previous_coverage`，再更新累计
面积；只有 `previous_coverage < 0.95 <= coverage` 时返回一次
`success_first_crossing=True`。V3 环境把该事实写入最终 `PlannerTransition` 并终止回合；地面和
飞跃式 `LANDED_HOLD` 使用同一逻辑。执行器不得再自行决定正式覆盖成功。

## 跨 update 环境保持

完成 PPO update 后只推进 `policy_version`，不得调用 episode rollover。pool 继续持有当前 worker
进程和动态状态。宏步内飞跃式仍同步排空至 `LANDED_HOLD`，因此 rollout、checkpoint 和 policy
version 切换都只发生在稳定决策边界。

自然 terminal 的 worker 由现有 auto-reset 单独前进一个 episode cursor；其 `done=True` 截断
GAE。未 terminal 的 worker 保持 cursor 不变。一个 batch 可以包含跨不同 episode 的 transition，
`done` mask 是唯一的 GAE 分段依据。

## 可重放活动 episode checkpoint

cursor-only schema 升级为 `lunar-formal-environment-state/v2`。父进程在完整 PPO update 边界向
每个 worker 请求一个 JSON-compatible state。每个 worker state 固定包含：

- schedule ID、platform、worker lane、episode cursor、scene ID 和 start cell；
- 当前 pose、足式 body z、稳定 execution state、observation revision 和 state time；
- 初始 reveal 之后每个已完成宏步的 reveal pose、elapsed time、边界状态和足式 body z；
- 当前 observation identity、PolicyBatch digest、被拒绝候选索引和最近一次飞跃可用 delta-v；
- 派生 start/episode seed 身份；PPO/Python/NumPy/Torch RNG 仍由顶层 checkpoint 保存。

静态 DEM、hazard 和 capability 投影不复制进 checkpoint。恢复时按 scene/start 创建 episode，
从冻结 truth 重放 reveal 历史，重建 0.2 m 已观测瓦片、4 m 汇总、候选和 request snapshot，再应用
被拒绝候选。恢复结果必须与所存 observation identity 和 tensor digest 完全相等，否则 fail
closed。checkpoint 只允许稳定 `DECISION_BOUNDARY`、`GROUND_HOLD` 或 `LANDED_HOLD`；飞行中或
半执行状态不能保存。

正式 preflight 必须执行真实 checkpoint save/load，并证明 uninterrupted update 2 与 resumed
update 2 的模型、优化器、RNG、scene、start、observation、candidate tensors 和首个 planner
request digest 相等。

## 完整回合评估

正式 evaluation 不得复用 proxy 的固定三步循环。每个 scenario 从初始 reveal 运行到自然
terminal，覆盖率按
`sum(observed_ratio * mission_roi_ratio) / sum(mission_roi_ratio)` 计算。已经完成的 row
冻结其 evidence，不得用后续自动重置 episode 覆盖结果。

为防止基础设施无限挂起，评估器可使用独立 watchdog；watchdog 不进入观测、不改变策略、不
产生失败 reward。超出后整个对应评估批次以 `EVALUATION_INCOMPLETE` 退出，不生成可通过发布门
的报告。

## Horizon 校准

用相同 cache、scene/start/episode seed、24-worker 分配和相同 transition 总量比较
`T in {16,32,64}`。记录：

- transitions/s、单 update wall time、GPU 峰值显存；
- worker 等待比例、planner timeout 数；
- KL、value loss、advantage/return 有限性；
- update-boundary checkpoint/resume digest。

候选必须通过相同语义回归和资源稳定性门；在通过者中选吞吐最高者。选择结果写入 calibration
manifest 和 frozen config。任何 horizon 都不得出现在 episode terminal 判断中。

## 完成标准

- 正式环境可执行第 9 和第 33 次动作而保持相同 episode identity；
- 第 33 次动作使用更新后的 policy version，但 scene/start/累计观测不重置；
- `ObservationContractV3` 在训练、checkpoint、导出和运行时一致；
- ROI 95% 首次跨越只产生一次成功，正式评估可得到非零成功率；
- 无候选自然失败，不存在 budget exhaustion 状态；
- 活动 episode checkpoint/resume 与连续运行 update 2 完全相等；
- 16/32/64 校准后冻结唯一 horizon；
- 更新所有正式训练就绪文档并重新生成 preflight/calibration 证据；
- 不启动 seed 4080 正式训练。
