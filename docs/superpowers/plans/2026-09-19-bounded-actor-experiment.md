# 有界 Actor 配对实验实施计划

> 用户已在对话中审阅设计并明确要求“落实此计划”。在现有隔离 feature worktree 实施，不再次请求执行许可，不提交、合并、删除旧产物。

**Goal:** 实施并运行 A=原分数、B=10*tanh((z-mean(z))/10) 对照，保留新决策图、候选对应Critic与SAC损失。
**Architecture:** Actor独占分数参数化；SAC消费同一策略分布并输出诊断；训练场景序列由槽位/回合确定；实验驱动串行运行8环境GPU任务并保存有界汇总。
**Spec:** 已确认的本轮对话方案及 docs/validation/2026-09-19-drl-graph-critic-implementation.md。

## 约束

- C=0为未限幅基线，C=10为实验；同状态有效动作均值可微分。保持Actor参数张量结构，Critic不做分数变换。
- 所有Actor采集/TD目标/损失/导出/推理使用同一分布；配置与checkpoint须包含参数化身份，禁止恢复时悄悄换C。
- gamma=.995，目标熵系数.10，alpha上限1e-4，学习率1e-5，batch64/microbatch16，预热1024，更新比.25。
- 8环境，40–80m小场景，512决策，传感器10m/90°，30倍率，.2m/s，1800s覆盖保存。
- 初次约3000转移工程筛查；正式3个配对种子各10000转移；相同参数初值和可复现槽位/回合场景序列。训练/验证地图分离。
- 只比较Actor；折扣/熵阶段待此项证据后决定，不预先修改。
- 不更改规划/控制/碰撞判定。碰撞与初始化等失败保留；若故障使比较失真，显式中止实验并记录。
- 单GPU串行各组；每组一份resume，无默认bag/视频/历史模型，自动生成紧凑结果文件。

## Task 1: Actor参数化与诊断

文件：config.py(ModelConfig), model.py, sac.py, checkpoint.py/runtime.py/evaluation.py必要兼容处理；新test_bounded_actor.py。
- [x] 写红测试：跨状态独立、共同偏移不变、排列等变、C=0等价、空动作与单动作、double gradcheck、三条策略消费路径一致。
- [x] ModelConfig.actor_score_bound默认0.，有限非负；Actor forward_packed 中唯一入口按action_groups可微减均值→tanh，最终统一计算概率。
- [x] SAC日志记录位置熵、条件朝向熵、位置最大概率、原始分数跨度、饱和比例、原始分数梯度摘要；不改损失。
- [x] 规范化缺少新字段的旧v3配置为C0，C不一致拒绝恢复；推理artifact携带C，Critic层参数不变。
- [x] 模型/梯度/恢复/导出测试通过。

## Task 2: 公平场景序列与实验运行

文件：training.py Curriculum及新paired_experiment.py/test_paired_experiment.py；config.py新增paired_scene_sequence字段由主会话加入。
- [x] 写红测试：交错reset顺序不影响(env,episode)场景；恢复序号接续；验证种子不进入训练区间。
- [x] paired_scene_sequence=False保持原课程；True使用(seed,env,slot_episode)生成种子和独立RNG，计数存curriculum状态。固定小图课程用于组间比较。
- [x] 串行实验驱动使用同一seed及C0/C10，首种子3000检查后续训至10000，再另两个种子10000；记录实际超额与更新数，不伪称精确相同。
- [x] 有界评估按相同地图及预算；结果包含失败、coverage/80/99/exhaustion/path、零增益和往返。每组固定几何统计不能用变动node id算往返。
- [x] 驱动遇到非正常训练退出停止且记录；恢复现有完成组不重跑；相同seed初值checksum；总输出不增无关文件。

## Task 3: 机制试验与集成（主会话）

文件：ros2_ws/src/lunar_drl_exploration/lunar_drl_exploration/actor_probe.py；配置A/B；实验说明与验证报告。
- [x] 新结构样本上的冻结Q试验区分变换t0与300更新后，控制状态、Q翻转与不同动作数；报告梯度饱和，不把拟合Q当真值最优。
- [x] 六包构建、包测试、真实短GPU采集/更新/导出/恢复；独立review。
- [x] 运行机制试验，启动持久化串行配对实验；监测故障并保留诊断。记录完成与仍运行的边界，不声称未得出的胜负。
- [x] 文档写明运行/恢复/停止方法、checkpoint目录、当前参数与Humble/Orin/收敛NOT_RUN。

本清单勾选表示实现、验证和启动已完成；60k 配对实验仍需等待执行结果，不能视作全部采集和评估已经完成。
