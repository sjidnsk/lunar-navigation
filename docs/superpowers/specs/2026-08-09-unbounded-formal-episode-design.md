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

## 正式任务场景全集与共同可行动子集

冻结 source/split 仍完整保存 1734 个物理场景，不因平台能力删除数据。正式三平台训练与评估的
任务全集定义为其中的“共同可行动子集”：同一场景必须能按 capability v2 为轮式、足式和飞跃式
各确定一个安全、初始 reveal 后可产生候选的起点。只有三平台全部满足时，
`start_qualification.common_eligible=true`。

该筛选是任务定义，不是用训练结果挑简单样本：资格在 PPO 之前由静态投影确定，精确场景 ID、
三平台起点及排除原因写入 content-addressed cache。NASA 的 train/validation/test 每个 split
共同可行动比例必须至少为 90%，六个 JAXA holdout 必须全部可行动；达不到即禁止正式训练。
当前 cache 的共同可行动数量为 train 1475、validation 90、test 95、holdout 6，共 1666；其余
68 个均因飞跃式没有合格起点而处于任务域外，不能计为成功或失败，也不能进入评估分母。

每个平台对应 lane 使用同一 seeded permutation，按一轮不放回、轮末再循环的方式遍历共同子集；
评估对每个 eligible scene 恰好执行一次。报告必须按 split×platform 给出分母，validation、test
和 holdout 分别过门，禁止用合并平均掩盖任一 split。

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

preflight 的 evaluation 检查是有界的真实链路探针：validation、test、holdout × PPO、nearest、
gain-over-cost × 三平台各执行一个真实宏步，共 27 条 evidence；它证明非 proxy 场景、候选、
规划、执行与观测链可运行，但不计算发布成功率。只有公开正式 `evaluate` 才运行全部 eligible
scene 到自然 terminal 并执行发布门，不能用该探针摘要替代。

为使该等价性具有明确物理语义，正式 reward 的 `normalized_macro_step_time` 只使用认证轨迹或
单跳参考中的物理执行时长，并统一除以 `2.0 s`；没有 reference 的拒绝动作对应零物理宏步时长。
C++ diagnostics 中的 planner 墙钟耗时受线程与主机调度影响，只进入独立性能统计，禁止进入
PPO reward。正式 CUDA 更新固定 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、PyTorch deterministic
algorithms 和 deterministic cuDNN；地图编码器使用与 V3 固定输入尺寸对应的定长平均池化，
禁止使用 CUDA backward 不可精确重放的 adaptive average pooling。

## 完整回合评估

正式 evaluation 不得复用 proxy 的固定三步循环。每个 scenario 从初始 reveal 运行到自然
terminal，覆盖率按
`sum(observed_ratio * mission_roi_ratio) / sum(mission_roi_ratio)` 计算。已经完成的 row
冻结其 evidence，不得用后续自动重置 episode 覆盖结果。

场景身份使用 cache 中共同可行动场景的固定 seeded permutation。报告中的 `scenario_seed` 必须
与 worker 实际执行的 `scene_id` 一一对应；不得再对 episode cursor 叠加带放回 hash 偏移。
成功率只读取 controller 产生的 `success_first_crossing` 事件，不能从 float32 最终覆盖率按容差
反推成功。

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

候选必须通过相同语义回归和资源稳定性门。先找实测最高吞吐；吞吐达到最高值 99% 的候选视为
工程近似等价，再优先选择单次 update wall time 更短、最后选择较小 horizon，避免亚百分之一的
测量噪声把训练冻结到更长更新边界。选择结果写入 calibration manifest 和 frozen config。
任何 horizon 都不得出现在 episode terminal 判断中；formal preflight 必须读取该 calibration
root，不能把配置文件中的 bootstrap 值 `32` 写成已校准结果。

## 正式训练计划与“训练完成”定义

训练使用 seed 4080 和累计 86400 GPU seconds 上限。校准阶段最多 2 小时；随后轮式、足式、
飞跃式各最多 2 小时预热；节余时间全部进入三平台 8+8+8 联合训练，联合阶段至少保留 16 小时。
`latest.pt` 每 30 分钟保存恢复状态，联合阶段每 1 小时保存不可变候选。episode 不因阶段、保存、
PPO update 或 86400 秒边界伪造成功。

以下三个状态必须区分：

1. `training-running`：预算未结束，也没有冻结的通过候选；
2. `release-gate-passed`：某个冻结 checkpoint 在 validation、test、holdout 的每个
   split×platform 上分别满足覆盖成功率不低于 0.95、零安全/非法/平台错配/飞跃承诺违规、
   finite rate 1.0、observed-safe rate 1.0 和 deterministic repeat rate 1.0；这才表示本轮 PPO
   训练成功完成；
3. `not-converged`：累计 GPU 预算耗尽仍无 checkpoint 通过。此时训练运行已经结束，但不能说
   模型训练成功，也不能进入 ONNX/TensorRT/AGX 发布链。

训练曲线变平、loss 下降、达到 24 小时或生成 `latest.pt` 都不是完成标准。ONNX 等价、TensorRT
构建和 AGX 验收是通过候选之后的独立阶段，不反向改变 PPO 是否收敛的判定。

## 完成标准

- 正式环境可执行第 9 和第 33 次动作而保持相同 episode identity；
- 第 33 次动作使用更新后的 policy version，但 scene/start/累计观测不重置；
- `ObservationContractV3` 在训练、checkpoint、导出和运行时一致；
- ROI 95% 首次跨越只产生一次成功，正式评估可得到非零成功率；
- 无候选自然失败，不存在 budget exhaustion 状态；
- 活动 episode checkpoint/resume 与连续运行 update 2 完全相等；
- 正式 reward 不读取 planner 墙钟时间，CUDA update 2 的 rollout、模型和优化器逐字节相等；
- 16/32/64 校准后冻结唯一 horizon；
- 更新所有正式训练就绪文档并重新生成 preflight/calibration 证据；
- preflight 通过前不得启动 seed 4080 正式训练；通过后是否启动由后续明确授权决定，运行状态
  记录在仓库外 calibration `run-manifest.json`，不复制为静态文档事实。
