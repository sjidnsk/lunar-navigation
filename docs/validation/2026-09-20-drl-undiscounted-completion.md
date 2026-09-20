# γ=1 与正常耗尽完成奖励

用户要求落实此前讨论的取消折扣和完成奖励。本轮修改从 `dev/drl-exploration` 的 `86c2f00f` 派生，复用已有独立工作树，专项分支 `fix/drl-undiscounted-completion`，完成后合回统一开发线。

## 最终行为

```text
r = ΔA/100 - 0.02·ΔL/10 - 0.005·ΔΘ/π - 0.001
    + completion_bonus · 1[正常探索耗尽]
gamma = 1.0
completion_bonus = 5.0
```

完成奖励本轮初值为5，可通过配置调整；这是待训练评估的权重，不是已验证最优值。

- TaskAnalyzer/DecisionCore 负责原有当前观测机会耗尽判定；环境根据最终 report.completed 在最后一条转移上加奖金。Actor 不读取真值完成阈值。
- 达到80%或99%、目标到达、导航失败本身、预算截断均不另发完成奖金。碰撞沿用原异常路径，不构造奖励转移。
- 同一步既耗尽又到预算，按正常终止处理，奖金仅加一次；之后 step 必须先重置。初始化即耗尽且没有动作时，不伪造有奖励转移。
- SAC γ=1 保留全部后继 soft value；真正终止仍不自举，预算截断仍使用重置前最终状态自举，不把下一张地图当作后继状态。
- 面积、距离、转角、固定步代价、熵强度和模型结构均保持原值。这里只改学习目标，不新增探索行为机制。
- 配置与完整恢复身份均记录 completion_bonus，奖励版本更新；gamma 仍属于学习恢复身份。旧奖励/折扣检查点不能静默续训，新配置中断后可正常恢复。

## 切换方式

原训练已通过 SIGINT 有序停止并保存：3706条转移、669次更新，final.stop_reason=SIGINT，owned_children_closed=true。复查原进程树27个PID均已退出，旧检查点保留在 `training-output/drl-current-opportunities-v1/resume.pt`，约680 MiB。

新目标使用新模型、优化器与回放，不迁移旧Q值或奖励。正式输出改为 `training-output/drl-undiscounted-completion-v1/`；小场景配置也更新γ与奖金，独立输出为 `drl-undiscounted-completion-small-v1/`。旧文件没有删除或覆盖。

```bash
cd /home/kai/WS/lunar-navigation/lunar-runtime/.worktrees/drl-exploration-redesign
scripts/drl/train.sh --config config/drl_exploration.yaml
# 同一新配置中断保存后：
scripts/drl/train.sh --config config/drl_exploration.yaml --resume
```

8环境、30倍目标倍率、10 m/90°、batch64/microbatch16和30分钟保存保持原值。不要将旧目录的检查点复制进新目录。

## 验证

- 修改前相关测试26 passed；新增奖金与γ=1测试在旧实现上7 failed、1 passed，失败原因为不接受奖金参数、拒绝γ=1。
- 测试涵盖奖金取5/2/0、正常非终止/预算截断/耗尽/耗尽且到预算、环境参数传递与终止后拒绝重复step。
- γ=.995和γ=1均对照实际网络期望验证终止不自举、截断自举；奖励权重变化检查点拒绝恢复。
- 源码包全套318 passed、2 skipped，60.31 s（2项为需显式启用的ROS闭环）。旧默认值测试按新要求更新，没有放宽实际行为判据。
- 独立代码审查未发现Critical/Important，相关四组短测51 passed；按审查建议补充了完整恢复时跨γ设置的拒绝断言。
- 长期收敛、循环是否改善、达到耗尽时的真实覆盖表现尚待训练和冻结评估；不能由数值测试推出。Humble/Orin/实车：NOT_RUN。

测试和停止检查的临时记录位于 `/home/kai/.cache/lunar-drl-coverage-completion/reward-*.log` 与 `reward-old-pids.json`，不提交模型或大体积运行产物。
