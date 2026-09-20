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

集成构建、安装测试、导航回放及启动结果随后记录。Humble/Orin/真实课题三/实车均NOT_RUN；本次验证不证明策略收敛或循环已解决。
