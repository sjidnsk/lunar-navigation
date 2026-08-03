# 规划算法 v3 三平台单网络 PPO 重训练设计

## 背景与决策

U80 上已有 PPO 模型只使用默认 A* 规划结果训练，定位为历史保底参考，不代表本项目 C++ 规划算法 v3 的行为，也不再作为当前模型迁移或发布基线。本次重新训练直接面向已经迁入本仓的轮式、足式和飞跃式 v3 规划能力，在 Ubuntu 22.04 amd64、RTX 4080 SUPER 上生成一个同时支持三类平台的 PPO 网络。

本设计新增训练闭环和模型输入合同，替代卷三现有计划中“沿用旧六输入网络并只做训练冒烟”的假设。卷三实施计划必须在执行前按本文统一修订；本文本身不修改代码、当前卷三计划或正式发布状态。

关联基线为[卷三策略流水线计划](../plans/2026-08-02-lunar-navigation-volume-3-policy-pipeline.md)、[外部输入权威基线](../../interfaces/external-input-baseline.md)和[三平台参数化物理代理设计](2026-08-03-lunar-exploration-platform-proxies-design.md)。本文改变卷三模型与训练设计，不改变外部 Topic、静态能力资料的所有权或平台代理的实验性质。

## 目标与非目标

目标：

- 训练一个 PPO 网络、一套权重、一个最终 checkpoint 和一个 ONNX 模型，不拆分平台专用网络或输出头。
- 在每个训练宏步中调用真实的 C++ `lunar_planner_core` v3，而不是 Python A* 或规划结果替身。
- 通过固定平台类型输入，让同一网络能针对轮式、足式和飞跃式规划行为学习不同的高层前沿选择。
- 使用当前主机的 CPU、内存和 RTX 4080 SUPER 做激进但可回退的并行训练。
- 在总计 24 小时 GPU 工作预算内得到三平台分别达到 95% 覆盖成功率的训练候选。
- 支持正常暂停、进程中断和断电后的 checkpoint 恢复。

非目标：

- 不迁移或继续训练 U80/A* checkpoint，不承诺与旧 checkpoint 参数兼容。
- 不在 PPO 中复制 v3 的完整平台能力结构，也不让 PPO 覆盖 v3 的安全或可行性裁决。
- 不通过 ROS 2 Action 或 DDS 承载训练热循环；ROS Action 只用于部署和系统集成验证。
- 不把当前参数化物理代理宣称为真实平台数字孪生或实机能力证明。
- 不在本阶段执行 AGX TensorRT 验收、四小时稳定性测试或自动激活部署。
- 不引入 artifact registry、authority/repair graph、多级签名或人工审批链。

## 单网络与七输入合同

部署和训练共享以下七项输入：

```text
prior_channels      float32 [B,Cp,Hg,Wg]
coverage_summary    float32 [B,Cc,Hg,Wg]
local_crop          float32 [B,Cl,Hl,Wl]
frontier_features   float32 [B,M,22]
pose_features       float32 [B,P]
candidate_mask      bool    [B,M]
platform_context    float32 [B,3]
```

`platform_context` 只编码平台类型，不包含能力数值：

```text
WHEELED = [1,0,0]
LEGGED  = [0,1,0]
HOPPER  = [0,0,1]
```

每行必须全部有限、元素只能为 `0.0` 或 `1.0`，且恰好有一个元素为 `1.0`。训练、PyTorch 推理、ONNX Runtime 和 TensorRT 在 shape、dtype 或 one-hot 语义不合法时均直接拒绝输入。

`platform_context` 经过小型线性编码器投影到模型隐空间，再与现有 `pose_features` 的全局表示融合。模型仍使用一个共享骨干和一组共享策略/价值输出：

```text
frontier_logits     float32 [B,M]
theta_mu            float32 [B,M]
theta_kappa         float32 [B,M]
value               float32 [B]
```

不增加平台专用 head、平台专用 checkpoint 或额外 ONNX 包。运行时 one-hot 来自配置阶段已经验证的平台类型；它必须与随后 v3 返回的 `MotionReference.platform_type` 一致。

部署时的平台类型以外部输入权威基线中的 `platform-control-capability-source/v1` 为来源；本项目只校验并编码该类型，不取得静态能力资料的所有权。训练时的平台类型来自当前回合的代理配置，仍标记为 `proxy`，不能反向写入外部输入基线。

## v3 训练桥与数据流

训练包新增一个仅用于 Ubuntu 训练环境的 `pybind11` 适配层，直接链接 ROS 无关的 `lunar_planner_core`。适配层只负责在 Python 训练对象与 `PlannerInput`、`PlannerOutput` 之间做有界类型转换，不重新实现规划算法，也不依赖 `rclpy`、ROS Action 或 DDS。

每个环境进程独占一个 planner 实例和环境状态，不共享可变 planner 对象。进入 C++ 规划时释放 Python GIL，使多个环境进程能够并行执行 v3。

一个训练回合固定为一种平台，数据流为：

1. 课程采样器选择平台、代理能力、地形和固定场景 seed，并生成对应 `platform_context`。
2. PPO 从七项观测中选择前沿候选和方向。
3. 环境把动作转换为 v3 `GoalRegion`，连同当前状态、不可变世界快照、对应平台能力和配置构造 `PlannerInput`。
4. 进程内桥调用 `Planner::Plan()`，返回规划结果、执行指令、可选运动参考和诊断。
5. 轮式和足式环境按轨迹参考推进到下一探索决策边界；飞跃式环境按准备、已承诺、飞行、着陆保持的状态生命周期推进。
6. 环境用覆盖增量、推进、代价、耗时和规划结果生成奖励与下一观测，写入同一个 on-policy rollout。

飞跃一旦进入已承诺或飞行状态，环境不再请求新的 PPO 动作，直到着陆保持并重新具备决策条件。飞跃期间的覆盖、耗时和结果累积成一个时间归一化宏步，不能用新策略动作改写已承诺参考。

部署数据流仍为“PPO 选择目标 -> `PlanMotion` Action -> v3”。训练直接调用 core 只是去掉热循环中的 ROS 开销，不改变生产规划内核或部署职责。

## 激进并行配置

本机基线为 Intel Core i7-14700KF、20 个物理核心、28 个逻辑 CPU、62 GiB 内存和 RTX 4080 SUPER 16 GiB。训练首先在同一小型 workload 上比较 18 与 24 个环境进程；只要 24 workers 未超过内存和规划时限，且总吞吐不低于 18 workers，就固定使用 24。

- 单平台课程预热时，全部 workers 运行当前平台。
- 三平台联合训练时固定为轮式、足式、飞跃式各 8 workers。
- 每个 worker 内部把 OpenMP、MKL 等通用计算线程限制为 1，禁止嵌套扩线程。
- CPU workers 负责地形、v3 和状态推进；一个中央 GPU learner 负责批量策略推理和 PPO 更新。
- 观测通过共享内存、页锁定缓冲区和双缓冲传递，避免为每步复制大型高分辨率张量。
- rollout 严格绑定一个策略版本；一次 PPO 更新不接受旧策略版本产生的数据。

GPU micro-batch 在正式训练前逐级向上探测，显存上限为总显存约 90%。确定的 worker 数、micro-batch 和 rollout 配置写入解析后的 run manifest，正式训练与恢复期间不再自动漂移。

worker 崩溃、C++ 异常、共享内存错误或非有限张量会废弃当前未完成 rollout 并停止训练，不允许静默减少 workers 后继续。正常的 v3 目标不可行、无已知安全路线或资源受限仍属于有语义的规划结果。

## 奖励与规划结果分类

三类平台共用同一组无量纲奖励项和权重，不维护三套平台奖励函数：

- 新增有效覆盖面积是主要正奖励。
- 朝任务目标和未探索前沿的有效推进是辅助正奖励。
- 规划代价、模拟耗时和重复访问是归一化负奖励。
- `GOAL_INFEASIBLE` 和 `NO_KNOWN_SAFE_ROUTE` 是可学习的策略选择结果，给予明确负奖励。
- 安全前沿或降级参考可以执行，但收益低于正常新参考。
- `RESOURCE_LIMITED` 计入负奖励和独立诊断率；其比例过高表示训练或 planner 预算配置不合适。

U80/A* 的最终训练成绩和 checkpoint 不决定新奖励权重。24 小时总预算的校准阶段使用三个短随机种子、均衡的三平台小场景，按“三个平台中最低校准得分最高”选择一组权重。权重、归一化和配置哈希在正式种子开始前冻结，中途不能改奖励后继续接跑。

非法请求、陈旧输入、数值故障、非预期取消、C++ 异常和非有限张量不是策略失败；这些样本不得写入 rollout。碰撞、禁入区侵入、非法飞跃或绕过 v3 安全裁决属于训练硬失败，不能用更高探索奖励抵消。

## 24 小时训练课程与候选门槛

总 GPU 工作预算固定为 24 小时，包含并行调优、GPU micro-batch 探测、奖励短校准、一个正式种子的训练和阶段评估。代码实现、CPU 单元测试、普通构建和 ONNX 导出时间不计入这 24 小时。

预算按上限分配：

- 最多 2 小时用于并行参数和三个短种子奖励校准。
- 最多 6 小时用于一个正式种子的三平台课程预热，每个平台最多 2 小时。
- 至少 16 小时用于轮式、足式、飞跃式各 8 workers 的均衡联合训练和评估。

前一阶段提前结束时，剩余预算全部转入联合训练。预热按轮式、足式、飞跃式依次运行，并从平缓稀疏障碍提高到完整平台代理场景；达到各自两小时上限后必须进入联合阶段，不能挤占联合训练的最低预算。

联合阶段每小时保存候选 checkpoint，每三小时在冻结验证场景上运行一次评估。任一 checkpoint 一次完整评估同时满足以下规则即可提前停止，并记为“三平台训练候选达到本轮收敛门槛”：

- 轮式、足式和飞跃式的 `success_coverage_rate` 分别不低于 `0.95`。
- 每个平台的安全违规数和非法动作数均为零。
- 网络有限输出率为 `1.0`。
- v3 参考平台类型全部匹配，飞跃承诺状态违规数为零。

本轮不要求连续三次通过，也不要求 PPO 必须胜过最佳规则基线。`nearest_frontier` 和 `gain_over_cost_frontier` 继续在相同场景、相同 v3 内核下运行，但只作为诊断比较。

最终选择所有候选中“三个平台最低覆盖成功率最高”的 checkpoint；并列时依次选择规划失败率较低、完成时间较短者。24 小时耗尽仍未通过时，保留最佳 checkpoint 和报告并明确标记为未收敛，不导出正式 AGX 发布包。

`0.95` 是本次有限时间训练候选门槛，不覆盖卷三现有 `0.99` 正式模型发布规则。候选只有通过后续正式发布评估、ONNX 等价验证和包校验后才可进入 `model-package-ready`；AGX 上完成独立验证后才能进入 `device-verified`。

所有训练地形、能力范围和平台代理结果继续显式标记为 `proxy`，不得写成真实平台能力结论。

## 暂停、恢复与 artifact

训练每 30 分钟在完整 PPO update 边界原子更新 `latest.pt`。收到 `SIGINT` 或 `SIGTERM` 时停止领取新 rollout，丢弃当前未完成 rollout，在最后一个完整 update 后立即保存并退出，不等待下一个 30 分钟周期。意外断电最多损失约 30 分钟的已运行训练。

checkpoint 包含模型、优化器、学习率状态、累计环境步数、课程阶段、随机数状态、归一化统计、冻结配置、source commit 和累计 GPU 工作时间。恢复命令校验 checkpoint schema、七输入合同、配置哈希和 source commit 后，重新创建 workers 并从最后一个完整 update 继续。

暂停期间的墙钟时间不计入 24 小时预算；所有恢复会累加实际 GPU 工作时间，不能通过重启重新获得预算。恢复不承诺操作系统调度层面的逐位相同轨迹，但必须保持 checkpoint 状态完整、平台样本配比不变且不重复计入已完成 update。

训练产物只写到仓库外的绝对 artifact root：

```text
<artifact-root>/
├── checkpoints/latest.pt
├── checkpoints/best.pt
├── metrics/train.jsonl
├── evaluation/report.json
└── run-manifest.json
```

不把 checkpoint、优化器、训练日志、共享内存 dump 或场景缓存提交到 Git。`run-manifest.json` 只保存解析配置、提交、主机训练指纹、累计时间和输出哈希，不构造跨阶段治理图。

## 最小安全检查与验证

训练热路径只保留五项硬安全检查：

1. `platform_context` 是合法三维 one-hot。
2. 网络输入、输出和损失全部有限。
3. v3 输出平台类型与当前回合平台一致。
4. 环境不执行 v3 拒绝或未认证的运动参考。
5. 飞跃承诺状态不能被新的策略动作覆盖。

实现验证保持聚焦：

- 三类 `platform_context` 编码、非法编码拒绝和共享网络前向测试。
- 三类 `PlannerInput`/`PlannerOutput` 的 pybind 转换与 v3 bridge 冒烟。
- 轮式、足式轨迹宏步和飞跃承诺宏步的状态推进测试。
- 18/24 worker 小型吞吐探测与三平台 8+8+8 配比测试。
- RTX 4080 SUPER 短 CUDA 训练、正常暂停和中断恢复测试。
- PyTorch 与 ONNX Runtime 七输入、四输出等价测试。
- 95% 候选门槛按平台独立聚合、任一平台失败则整体失败的测试。

修改模型合同和卷三迁移规则后，仍运行仓库边界检查与 foundation 测试；这不扩张为四小时稳定性、AGX 性能验收或多级人工审核。

## 对卷三计划的影响

下一步实施计划需要原子修订卷三，而不是在现有任务末尾附加一个孤立训练脚本：

- Task 1 的旧模型前向基线改为“保留可维护骨干并新增 `platform_context` 融合”，不再承诺旧六输入数值完全相同。
- Task 2 增加 v3 bridge 构建、激进并行配置、30 分钟 checkpoint、累计预算和恢复合同。
- Task 3 区分本轮 95% 训练候选评估与原有 99% 正式发布评估；三平台始终分别聚合。
- Task 4 的 model manifest 和 `ObservationContract` 固定七项输入。
- Task 5 的 inference wrapper、ONNX、黄金输入输出和等价测试加入 `platform_context`。
- Task 6 的 TensorRT binding 和 cache 校验加入第七输入。
- Task 7 的探索输入构建在配置阶段生成 one-hot，并与 planner 平台类型做一致性检查。
- 新增的完整训练至候选收敛步骤位于 ONNX 正式导出之前，24 小时训练产物始终留在仓库外。

卷四的一键安装、默认不启动、独立验收和手动激活边界不变。设备包仍只发布一个 PPO 模型；不会因为三平台训练拆成三个发布目录。

## 风险与控制

- **CPU v3 成为瓶颈：** 用 18/24 workers 实测选择最大可持续吞吐，并禁止嵌套线程。
- **三平台负迁移：** 固定 one-hot 条件、平台独立指标和 8+8+8 样本配比，以最低平台得分选择 checkpoint。
- **代理到真实平台存在差距：** 所有结论保持 `proxy` 标签，正式发布和设备验证仍独立执行。
- **暂停造成部分数据丢失：** 只保存完整 PPO update，正常暂停立即保存，断电损失上限约 30 分钟。
- **有限预算未收敛：** 如实输出未收敛报告和最佳 checkpoint，不降低安全硬规则，也不冒充正式模型包。

本设计以快速获得可复现的三平台 v3 训练候选为目标，在时间受限情况下只简化统计和治理门槛，不简化 v3 安全所有权。
