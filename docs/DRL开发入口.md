# 强化学习统一开发入口

2026-09-19：按用户要求先整合开发线，再开展实验。本文件是当前开发位置、版本和操作入口的索引；早期文档中的运行状态与命令不能覆盖本文件。

## 唯一活动开发线

- 分支：`dev/drl-exploration`（由原 `feat/drl-exploration-redesign` 改名，保留提交历史）。
- 工作树：`/home/kai/WS/lunar-navigation/lunar-runtime/.worktrees/drl-exploration-redesign`。
- 构建、安装、日志：`/home/kai/.cache/lunar-drl-redesign/jazzy/`。
- Python 依赖：`/home/kai/.cache/lunar-drl-training/venv/bin/python`。这里的 training 只是共享依赖目录名称，不表示使用旧训练分支。
- 模型、回放、指标、实验结果：本工作树的 `training-output/<实验名>/`，已被 Git 忽略。

复用现有工作树，避免复制大模型或破坏已有构建路径。目录中的 redesign 是保留的物理路径，不再表示另一条活动开发线。

## 分支层级与流向

```text
integration/pure-planner-orin      公共地图、规划、控制和通用接口的集成主线
└── dev/drl-exploration           强化学习模块开发整合线
    ├── feat/drl-<主题>           专项功能
    ├── fix/drl-<问题>            专项修复
    ├── test/drl-<实验>           需要修改代码的试验
    └── docs/drl-<主题>           专项设计与文档
```

其他模块或场景开发线与 `dev/drl-exploration` 同级，不隶属于强化学习。上图的专项分支是命名和归属示例，本次不创建空分支或工作树。

- 专项从所属开发线派生，在独立工作树完成，验证后合回所属开发线；开发线负责整理可共同运行的模块版本。
- 纯超参数对比无需改代码，用同一提交、不同配置和输出子目录运行；需要改变算法的实验才派生测试分支。
- 共性修复在集成主线的专项分支完成并合入主线，再由开发线同步。若首先在模块中发现，提取公共部分回到主线，不将整个训练模块一并上送。
- 当前已同步的净空代价和路径简化器通过合并主线记录共同祖先；不只保留内容相同却彼此不相认的复制提交。
- 后续合并、发布和删除仍遵守用户授权；本次整合授权不等于后续所有分支自动合并或清理。

## 固定的代码基线

| 层次 | 当前实现与边界 |
| --- | --- |
| 地图与任务分析 | 原生增量地图、二维射线遮挡、任务内覆盖；任务外可通行连接和观测站位保留。真值可覆盖面积只用于训练/评估。 |
| 决策图 | `task_graph_v3`：净空优先基础选点、2 m 合法路径覆盖、8 m 局部连接、1.2 路程伸长稀疏化；保留必要补点、回退和任务外绕行。 |
| 动作 | `local_metric_pose_v3`：相邻候选位置与八方向观测朝向；探索层选目标，导航层生成路径。 |
| 学习 | `candidate_graph_sac_v2`；候选对应的 `candidate_truth_v1` 训练侧 Critic；Actor 不读取真值。 |
| Actor 实验 | 默认 C=0；有界 C=10 可通过配对驱动比较，尚未确定胜者。SAC 损失保持不变。 |
| 导航与控制 | 正式规划器、保留原路径净空的简化器、轮式净空代价权重 2.0、公共轮式跟踪器、轻量运动学平台。 |
| 运行 | 8 环境、30 倍目标倍率、0.2 m/s、10 m/90°；batch64/microbatch16、gamma=.995、目标熵因子 .10、alpha 上限 1e-4；每30分钟保存。 |

整合后按用户要求采用净空优先基础选点B版，并保持观测补点规则。C=5敏感性探针仍不是正式算法。
当前决策图仍允许必要近邻节点。实现、对比与单线训练协议见[净空优先基线](validation/2026-09-19-drl-clearance-baseline.md)。

职责与设计详见：

- [完整架构](superpowers/specs/2026-09-12-drl-exploration-redesign.md)
- [决策图](superpowers/specs/2026-09-18-drl-decision-graph-consolidated.md)
- [候选对应 Critic](superpowers/specs/2026-09-18-drl-critic-action-alignment.md)
- [Actor 配对实验](superpowers/plans/2026-09-19-bounded-actor-experiment.md)
- [导航修正运行验证](validation/2026-09-19-clearance-runtime-rollout.md)

## 历史分支如何处理

以下是保留来源，不再作为当前 DRL 开发或实验入口。保留不代表其工作区改动已经全部合入。

| 分支 / 工作树 | 用途及本次处理 |
| --- | --- |
| `feat/drl-exploration-training` / `drl-exploration-training` | 早期循环 SAC 与训练闭环；含未提交修改，原样保留。不将旧模型、奖励与恢复协议并入新图策略。 |
| `fix/observability-coverage` / `observability-coverage` | 早期覆盖算法及运行优化；含未提交修改，原样保留。其 HEAD 是祖先不能证明这些修改已合入。 |
| `feat/drl-exploration-complete-design` / `drl-exploration-complete-design` | 早期连续动作设计，只作历史参考。 |
| `feat/exploration-coverage-definition` / `exploration-coverage-definition` | 早期覆盖定义，只作历史参考。 |
| `fix/path-simplifier-clearance`、`feat/clearance-cost-calibration` | 公共导航修正来源；正式简化器和净空权重已同步到当前训练线，不使用这些树的实验库覆盖训练安装目录。 |

不复制旧回放、不删模型、不删除脏工作树，不以形式上的全分支 merge 冒充兼容整合。后续如果需要回收旧工作树空间，应另行核对其中独有修改和产物。

## 统一命令

```bash
cd /home/kai/WS/lunar-navigation/lunar-runtime/.worktrees/drl-exploration-redesign
scripts/drl/build.sh
scripts/drl/train.sh --help
scripts/drl/compare_actor.sh --help
```

正式训练、恢复、导出、评估和推理的参数见[操作指令](操作指令.md#drl-稀疏图探索独立-redesign-分支)。启动脚本清除继承的旧 overlay，再加载同一份 Jazzy 安装；不要手动注入旧工作树的库。

当前实验改为一条C=0策略正常训练、在固定月表/洞穴地图上分阶段冻结评估：

```bash
scripts/drl/baseline.sh
scripts/drl/baseline.sh --resume
```

旧 `training-output/actor-bounded-20260919/` 保留碰撞修正前的记录，目前为 interrupted，配对实验未完成，不能据此确定原分数或有界分数胜出。切换导航算法后不要继续把该目录的早期组与新运行组当成同一条件的对照。

模型恢复是否兼容由观察、动作、Critic 和配置协议决定，不由分支名字决定。净空优先基础选点已经写入完整恢复语义，旧行列顺序检查点不静默续训；新基线用新目录、新模型和新回放。历史结果仍可用于诊断。

## 验证记录

本次重新执行的构建和测试见[整合验证](validation/2026-09-19-drl-development-consolidation.md)。算法单元测试和本机 Jazzy 闭环不能证明长期收敛、两节点循环已消除、Humble/Orin 或实车可用。

2026-09-20 的[覆盖完成判据试验](validation/2026-09-20-coverage-completion.md)位于
`test/drl-coverage-completion`，尚未合入或采用。它实现测量侧覆盖下界，但固定月表/洞穴的剩余需求估计仍偏保守；正式开发和训练入口继续使用本文件上方的开发线。

同日按用户确认的职责边界，试验分支在 `cad9565` 重构为[当前观测机会耗尽](validation/2026-09-20-current-task-opportunities.md)：取消未来站位/通道推演和在线覆盖下界，80%/99% 仅由真值评估。观测补点由决策图在可执行已知位置中选择。正式开发线未切换；本轮为测试、固定快照及历史目标重放，不更新模型、不重启训练。
