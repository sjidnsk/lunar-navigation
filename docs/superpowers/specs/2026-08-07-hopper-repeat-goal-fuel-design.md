# 飞跃式连续 Goal、独立地图代次与燃料递减设计

> **已取代（2026-08-08）：** 本文的完成态静默清理、独立地图代次和连续 Goal 状态机仍有效；
> 燃料 Topic、`0.200 -> 0.189` 递减、fuel commit 和提交失败状态全部失效。现行飞跃式每个
> Goal 使用同一个固定单跳 Δv 包络，见
> `2026-08-08-formal-capability-and-hopper-no-fuel-budget-design.md`。

日期：2026-08-07

状态：已按用户“前面的推荐修复也一并处理”授权冻结

## 1. 已确认根因

飞跃完成后，外部 RViz 控制器在接收下一次选点时清理旧滚动会话。现有清理函数无条件调用
`RollingExecutionSession.cancel()`；即使旧会话已经处于 `COMPLETED`，仍发布一次
`MotionExecutionFeedback.CANCELED`。主仓把飞跃式 `CANCELED` 映射为
`HopperExecutionState::kEmergencyDelegated`，下一次 `PlanMotion` 因而以
`HOPPER_EMERGENCY_CONTROL_DELEGATED` 结束。这解释了“第一次跳到目标，第二次有时有旧路线但不动，
第三次状态不一致”的现象；它不是落区搜索失败。

另有两个独立问题：

- `HopperReference` 错误要求 `global_map_generation == local_map_generation`；全局地图不变而目标落区局部
  窗口重建时，两者本来就应独立递增。
- 交互桥每次都发布固定 `0.2 kg` 可用燃料，完成一跳后没有把核心认证结果中的
  `expected_remaining_usable_fuel_kg` 提交给下一次规划。

## 2. 终态清理合同

`cancel()` 只表示撤销仍在活动的引用。允许发布 `CANCELED` 的状态限定为
`PLANNING`、`PLANNING_NEXT`、`JUMP_READY`、`EXECUTING`、`IN_FLIGHT`、`LANDED_HOLD`；
`IDLE`、`COMPLETED`、`HOLDING_FAILED`、`CANCELED` 的清理必须静默、幂等，并保持原终态。

控制器在新 Goal、平台切换和关闭时都使用这一合同。完成第一跳后选择第二个目标时：

1. 保留第一跳最后发布的 `LANDED_HOLD`；
2. 不生成旧 plan/segment 的 `CANCELED`；
3. 清空旧可视化和会话对象；
4. 在同一已激活 hopper 进程中刷新目标落区并提交一个全新的 Action；
5. 新 Action 使用上一次真实着陆位姿和更新后的燃料。

非滚动的“只规划、不执行”模式没有稳定着陆证据，仍保留重启 hopper 会话的只读保护流程。

## 3. 地图代次合同

`global_map_generation` 与 `local_map_generation` 分别验证为正整数，不再要求相等。引用接受仍同时
核对 capability version、全局代次和局部代次均非零；主仓 `ReferenceGuard` 的逐字段一致性检查不变。
目标局部窗口更新只改变局部内容身份时，主仓可以看到相同全局代次和新的局部代次。

## 4. 测试燃料链

该链只属于仓库外 RViz/ROS 测试桥，不宣称替代真实推进剂遥测。

- 交互桥声明原子参数 `simulated_remaining_usable_fuel_mass_kg`，初值 `0.2`。
- 冻结干质量为 `19.8 kg`；发布消息满足
  `total_mass_kg = 19.8 + remaining_usable_fuel_mass_kg`。
- 只有在抛物线飞行结束、规范里程计确认落地、settle guard 完成后，控制器才提交本跳引用中的
  `expected_remaining_usable_fuel_kg`。
- 参数更新只能保持或减少燃料，必须为有限非负数；增加燃料、非 hopper 调用、混合落区参数与燃料
  参数的原子请求均拒绝。
- 同一个 plan/segment 只提交一次。提交失败时保持 hopper commitment，发布
  `HOPPER_FUEL_COMMIT_FAILED`，不得开始下一跳。
- 成功提交后才清除 `_hopper_session_committed`；下一 Goal 的快照必须读取新燃料值。

示例资格链为 `0.200 -> 0.189 -> 下一跳按 0.189 求解`，实际扣减量完全采用核心认证引用，
外部控制器不重复计算火箭方程。

## 5. 测试与验收

- 纯状态机：`COMPLETED.cancel()` 静默且保持 `COMPLETED`；活动状态取消仍发布一次 `CANCELED`；
  重复取消不重复发布。
- 引用：不同的正全局/局部代次可接受，零值和错误类型拒绝。
- 桥参数：单调扣减、总质量派生、增加/负数/非有限/错误平台拒绝、重复相同值幂等。
- 控制器：第一跳完成并提交燃料后，第二次 Goal 不产生旧 `CANCELED`，不重启已落地会话，
  使用新位姿并发送第二个 Action。
- 真实进程回归：在隔离 ROS domain 中执行“选 hopper -> Goal A -> 完整飞行与落地 -> Goal B ->
  完整飞行与落地”，要求两个不同 plan/segment 都出现 ADD 路线、平台两次移动、无
  `HOPPER_EMERGENCY_CONTROL_DELEGATED`，第二次 propellant 输入等于第一次预期余量。

不修改飞跃平台能力值、单跳落区语义、抛物线认证算法或真实飞控职责。
