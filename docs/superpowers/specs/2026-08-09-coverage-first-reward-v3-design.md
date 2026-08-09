# 覆盖率优先 Reward V3 设计

**状态：** 2026-08-09 已批准，立即实施。

## 目标与边界

正式 PPO 只优化真实任务区域的新增覆盖率。路径/执行代价、宏观执行时间、燃料和
`priority_observed_delta` 继续作为遥测事实保留，但不进入奖励。此次不修改
`parallel_pool.py`、规划器、传感器、平台能力、PPO 网络、折扣因子或 rollout horizon。

本设计仅替代既有文档中的 V2 奖励段落；其他冻结接口保持不变。

## 权威奖励

令 `C_after` 为执行后 `next_observation.pose_features[0, 4]` 中的权威
`mission_observed_ratio`，`delta_C` 为控制器给出的非负
`mission_observed_delta`：

```text
reward = 100.0 * delta_C
       - 0.10 * I(delta_C == 0)
       - 0.20 * I(GOAL_INFEASIBLE)
       - 0.30 * I(NO_KNOWN_SAFE_ROUTE or ACTIVE_REFERENCE_INVALIDATED)
       - (1.0 - C_after) * I(episode_ended_without_success)
       + 5.00 * I(success_first_crossing)
```

10% 至 90% 覆盖率节点不增加奖励；连续 `100 * delta_C` 已提供密集引导。
只有首次跨越 95% 成功阈值获得一次 `+5.0`。

## 契约与无效样本

- 新契约命名为 `RewardWeightsV3`、`RewardInputsV3`，身份包含
  `lunar-reward/v3` 和全部有效权重。
- 时间、路径和 priority 改变不得改变同一 transition 的奖励。
- C++ 异常、非法/陈旧输入、数值失败、`RESOURCE_EXHAUSTED` 和硬约束破坏不得进入
  PPO rollout；这些事实不是模型可学习的普通负样本。
- 奖励、累计覆盖和终止事实必须有限、范围合法且相互一致。
- reward hash 改变后，V2 检查点不得用于正式续训。

## 最小验证

1. Reward V3 精确项、符号和边界值单元测试。
2. 时间、路径、priority 不影响奖励的行为测试。
3. 重复覆盖、首次成功、覆盖相关失败终止和无效样本测试。
4. 奖励 hash 平台无关且与 V2 不同。
5. 三平台短 rollout/训练 smoke；不运行与本变更无关的系统验收。

当前 V2 训练只保存为历史试运行证据；在安全 update 边界停止。Reward V3 正式训练从
新初始化开始，不继承 V2 模型或优化器状态。
