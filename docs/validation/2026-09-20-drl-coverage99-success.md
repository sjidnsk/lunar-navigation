# 真实可探索覆盖率99%成功条件

按用户确认，训练/仿真达到 `A_cov/A_E >= 0.99` 即成功终止，当前转移额外奖励5。γ保持1，其余奖励、网络、导航、观测、参考集合均不变。

- 覆盖达标：completed、terminated为真，成功奖励仅发一次。
- 未达标但当前观测机会耗尽：terminated为真、completed为假，不发完成奖励。
- 仅决策预算达到上限：truncated为真，使用重置前状态自举。
- 同步达到覆盖和预算：成功优先。碰撞不能领取成功奖励。
- 初始即达标：零决策结束，不伪造奖励转移。

真值仅用于环境评分、训练终止和评估；Actor输入不变。部署无全局真值，仍由TaskAnalyzer基于当前测量机会耗尽停止。audit_completion.py中的report.completed仍指在线耗尽，不等于训练成功。

复审发现0.2 m栅格中1683/1700的等面积运算可能得到0.9899999999999999。覆盖比例改用有效格数之比，不引入epsilon、不降低99%要求；面积报告继续使用m²。

源码测试：332 passed，2 skipped（显式启用ROS的测试）。覆盖恰好99%、未达标耗尽、成功时仍有机会、奖励单次发放、预算优先级、碰撞排除、评估终止、worker换图和检查点身份。

新恢复语义：truth_coverage_or_current_opportunity_exhaustion_v1；奖励版本coverage_success_v3，身份包含success_coverage。旧终止/奖励版本不直接混入新回放。本次核对时旧训练已正常SIGINT退出，5415条转移、1097次更新、owned_children_closed=true；旧检查点及输出保留。

新正式输出为training-output/drl-coverage99-v1，8环境、30倍目标倍率、batch64/micro16、每1800秒保存。

集成构建：6包通过；安装后的37个Python源文件逐字节一致。安装全量测试初次332通过、2失败：旧导航测试的小场景初始化已达到99%，新规则正确拒绝继续step。扩大该测试的可达纵向场景后，两个实际Jazzy导航/控制/预算截断用例均通过（15.22秒），没有修改生产成功条件。

固定月表记录目标回放（seed2026091901，40 m）在第58步达到99.0994885%，152.44 m，completed/terminated为真，exhausted为假，仍有70个观测接口；所有58步GOAL_REACHED，无错误。此回放绕开Actor，不是策略性能评估；奖励发放由env.step回归测试验证。证据位于仓库外 ~/.cache/lunar-drl-coverage-completion/coverage99-{integrated-build,integrated-tests,native-tests,moon-replay}.log及moon-replay.json。

正式训练已在独立终端启动，run.json代码版本e7020edc、resume=false、CUDA、8环境，gamma=1、success_coverage=0.99、completion_bonus=5、保存间隔1800秒与新终止/奖励身份均已核对。首批185条转移覆盖全部8环境，处于1024条预热采集阶段；初始化检查点已写入。旧输出保留。Humble/Orin/真实课题三/实车均NOT_RUN；本次验证不证明策略收敛或循环已解决。
