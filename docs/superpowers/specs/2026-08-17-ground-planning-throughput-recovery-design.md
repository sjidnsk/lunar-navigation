# Ground Planning Throughput Recovery Design

## Goal

在不改变策略动作、候选、奖励、规划器搜索域或地图语义的前提下，修复地面平台滚动执行中的误失败，并减少观测轨迹中重复的整图可见性工作，使训练从 update 26 的原子快照安全恢复。

## Confirmed causes

1. 每个地面滚动续段都会用“当前位置到当前位置、零容差”再调用一次完整规划器。单点合法结果在全局路径转换处可能被解释为 `LOCAL_DETAIL_ROUTE_RESULT_INVALID`，造成 `GROUND_START_QUALIFICATION_FAILED`；该探针还与随后真正的滚动规划重复。
2. 路径证据按稠密样本逐点提交。连续样本大量落在同一 0.2 m 栅格，但每个样本都重新执行可见性计算和整幅观测层复制。现有样本中约 87% 的 pose cell 重复。

## Design

### Certified continuation boundary

初始任务边界仍执行现有地面起点资格检查。一个宏动作开始后，只有规划器给出的已认证 reference endpoint 才能成为下一滚动段起点。续段刷新地图和物理快照，但不再对这个 endpoint 执行“原地到原地”的资格规划。真正的下一次滚动规划继续 fail-closed；若 endpoint 已不安全或后续路径不可规划，正常规划结果仍会拒绝它。

### Same-cell observation batching

按路径顺序对连续且位于同一 0.2 m pose cell 的样本做 run-length 分组。每组只调用一次 `reveal_from_pose` 并只复制一次观测层；组内每个 elapsed 值仍按原顺序更新观测年龄与计数，确保最终数组、覆盖增量、优先级增量、可见次数和状态时间与逐样本执行一致。

该优化只对位置栅格完全相同的连续样本生效。传感器合同为 360 度，visibility 输入只包含 `pose_cell`，因此组内 yaw 不参与可见性结果。

## Non-goals

- 不伪造或冻结 map generation 来强制复用全局路径。
- 不改变 WHEELED 4 m / LEGGED 3 m 局部 horizon。
- 不改候选生成、Oracle/无 Oracle 合同、Reward V4、checkpoint 内容或 PPO 更新频率。
- 不实现新的 C++ 路径批量 visibility kernel；本轮只做低风险 Python 语义等价优化。

## Acceptance

- 续段构建不调用 `_qualify_ground_start`，初始边界行为不变。
- 批量观测与逐样本观测的全部观测数组和聚合 delta 精确一致。
- 同一 cell 的 N 个连续路径样本仅触发一次 visibility reveal。
- 相关 focused suites 全绿，`git diff --check` 通过。
- 从 update 26 恢复；旧 update 27 未封口的 6/24 worker 数据不与新语义混用，并以可恢复方式归档。
