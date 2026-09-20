# 探索策略开发线整合与正式开训

## 整合范围

用户明确要求整理合并探索策略开发分支并开始训练。2026-09-20 将 `test/drl-coverage-completion` 快进合入 `dev/drl-exploration`：`296d004a → 96f8ba1f`。唯一开发和运行工作树仍为 `drl-exploration-redesign`。

- `cad9565` 的当前观测机会耗尽逻辑与补点修正已进入开发线；旧的未来通道/站位推演不再使用。
- 净空优先决策图、候选对应 Critic、公共路径简化器与净空代价在合入前已位于开发线，继续复用。
- `integration/pure-planner-orin` 的 `6868236f` 是开发线祖先，公共导航修正已包含；本轮没有将 DRL 专用训练代码反向合入公共主线。
- `test/drl-clearance-baseline` 与合入前开发线同版本。早期训练/覆盖工作树包含独有或未提交内容，保留为历史来源；没有删除、重置或批量合并这些旧实现。
- 未推送、未创建 PR、未删除分支或工作树。

## 训练设置与恢复

当前配置 `config/drl_exploration.yaml` 将输出设为项目内 `training-output/drl-current-opportunities-v1/`。结束语义改变，因此本轮新建模型、优化器、回放和课程状态，旧产物原样保留。

| 项目 | 本轮设置 |
| --- | --- |
| 环境 | 8 个，月表/洞穴各4个，正式课程，无有限转移上限 |
| 动作与模型 | 净空优先图；相邻位置×8朝向；原始 Actor 分数 C=0，权重正常学习；候选对应 Critic |
| 传感器与运动 | 10 m/90°，目标30倍倍率，速度上限0.2 m/s |
| 学习 | batch64/microbatch16，lr=1e-5，gamma=.995，目标熵因子.10，alpha上限1e-4 |
| 调度 | 1024条预热，每4条新增转移一次更新额度，每16次更新发布Actor |
| 课程 | 小/中/大40–80/80–150/100–300 m，预算512/2048/8192，阶段边界20k/60k |
| 保存 | 每1800墙钟秒保存，单份原子替换resume.pt；指标最多8 MiB；不额外录制视频或bag |

本轮保留既有奖励和折扣/熵设置，不表示这些参数的后期探索效果已经得到验证。完整恢复身份为 `current_reachable_task_opportunities_v1`，不跨旧结束语义导入回放。

```bash
cd /home/kai/WS/lunar-navigation/lunar-runtime/.worktrees/drl-exploration-redesign
# 本轮已经启动，不要重复启动同一输出目录。
# 在训练终端 Ctrl-C 有序保存退出后，恢复使用：
scripts/drl/train.sh --config config/drl_exploration.yaml --resume
```

## 本轮验证

- 合入前包内测试：311 passed、2 skipped，51.48 s。
- 合入后使用统一入口重新构建6个相关包：通过，1.89 s。
- 合入后启用真实 ROS 测试，包内完整测试：313 passed，74.27 s。
- 安装目录的37个 Python 源文件与统一开发工作树逐字节一致。
- UTF-8文档读取、配置解析和 `git diff --check` 通过。
- 以上为本机 Jazzy 证据。Humble、Orin、实车：NOT_RUN；没有以短期运行宣称收敛或往返循环消失。

## 启动证据

2026-09-20 已打开 GNOME 训练终端，标题为 `DRL 探索训练 · current-opportunities-v1`。进程 PID 498240，运行记录的代码版本为 `96f8ba1f`，CUDA 已初始化，安装包来自 `/home/kai/.cache/lunar-drl-redesign/jazzy/install/`。`resume=false`，没有 probe 参数或有限转移上限。

启动约33秒快照：234条转移，8个环境均已完成动作并继续执行；更新0，处于1024条预热阶段；记录中没有失败或停止事件。初始恢复检查点已写入。该快照仅证明启动和采集正常，不是长期稳定或策略性能结论。

继续检查至约163秒：1084条转移、15次SAC更新，更新额度0，Critic/Actor/温度损失均为有限值，GPU利用率采样72%，显存采样1767 MiB，记录中仍无失败或停止事件。已确认越过预热并实际开始学习，不只启动了采集进程。

现场硬件：32逻辑CPU、约30.5 GiB内存、RTX5070 Ti Laptop GPU（12227 MiB显存）；启动前磁盘可用约228 GiB。后续硬件和运行状态以 `run.json`、有界 `metrics.jsonl` 及实时进程为准。

合入前后测试日志位于 `/home/kai/.cache/lunar-drl-coverage-completion/`：`premerge-tests.log`、`integrated-build.log`、`integrated-tests.log`。本轮训练产物仅写入上述项目输出目录；旧实验和检查点未修改。
