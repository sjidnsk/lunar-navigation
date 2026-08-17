# Ground Observation Throughput V2 Design

## Goal

在不改变 1 m 地面采样、0.2 m 观测、30 m 传感器、候选/奖励/规划语义和完整恢复合同的前提下，消除地面宏动作中重复的观测提交、age 老化和地图投影工作。一个局部地面轨迹仍按原顺序产生完全相同的细节观测与覆盖结果，但作为一次原子事务提交。

已有实现已经完成 `latest.pt` 到不可变 `update-N.pt` 的原子 hardlink alias；V2 不再重新实现或改变该恢复机制。

## Confirmed remaining costs

当前 `observe_world_path()` 仍会：

- 为每条路径复制全部 detail tile；
- 为每个不同 0.2 m pose 单独读取重叠的 64 m 窗口、单独调用 native visibility；
- 只在单条路径内惰性 age，提交末尾仍物化全部历史 tile；
- 在滚动局部规划边界重建完整 planner map 的全部层，即使只有小片观测区域改变。

这些操作位于地面宏动作的执行/状态提交路径，而不是局部规划器本身；不能以降低采样密度、放宽传感器或改变候选语义来换取速度。

## Invariants

1. 对同一有序 `(Pose2, elapsed_s)` 路径，V2 完整物化后的 detail/coarse 数组、coverage、priority coverage、observation count、evidence generation、physical-evidence SHA 与逐样本基准逐位一致。
2. 任一准备、native 可见性或应用异常发生时，所有权威状态保持原样；事务不能暴露半条路径。
3. 每个 1 m 样本的 elapsed 顺序、float64 累加和 float32 存储时机不改变。V2 仅推迟尚未读取的 tile 的写入，不重排 age 加法。
4. 滚动局部规划读取的 tile 必须先补齐到当前全局 age cursor；候选决策、physical evidence hash、checkpoint 和完整策略 observation 必须先物化全部 tile 与 dirty coarse cells。
5. map generation、请求 stamp 和规划器缓存失效语义不伪造、不跨真实 sensor evidence reuse。
6. 不删除不可变 checkpoint，不降低 checkpoint 频率，也不改变 worker state 或恢复 schema。

## A. One atomic ground-trajectory transaction

新增公开接口：

```python
observe_ground_trajectory(samples: tuple[tuple[Pose2, float], ...]) -> ObservationDelta
```

既有 `observe_world_path()` 兼容地转发到它。事务分为三个阶段：

1. **prepare**：验证所有输入；按精确 0.2 m cell 去重 visibility 请求，同时保留原样的时间顺序和重复 cell 事件序列。
2. **stage**：以 copy-on-write 临时状态执行所有事件。detail tile 字典先浅复制；仅第一次写某 tile 时复制该 tile。coarse/coverable 计数数组只在实际被触及后复制一次。
3. **commit**：一次性交换 staged 引用、dirty 集与累积 delta。任何异常只丢弃 staging，不触及正式状态。

同一 cell 即使在路径中非连续地重复出现，visibility mask 也只准备一次；每一条原始事件仍按原顺序应用 elapsed、更新观测次数并增加 evidence generation。这保证“批量”是事务边界，而不是改变传感器事件的语义。

### 64 m window reuse

每个 pose 仍使用准确的 64 m × 64 m 0.2 m 窗口。路径级 `DetailWindowComposer` 将这些滑动窗口从现有的对齐 64 m fixed tile 投影拼接；底层 fixed tile 在一条路径中按 identity 缓存。拼接结果必须与 `SceneTileProvider.read_window()` 的全部 `ProjectedScene` 字段逐位相同。这样重叠窗口复用已读取 tile，而不是复用或近似可见性。

### Native visibility batch

bridge 增加批量 `reveal_from_poses(truth_obstacle_ratio, pose_cells)`：输入为一组相同形状、C-contiguous 的 0.2 m truth 窗口和 cell，输出同顺序的 bool mask。内部复用单 pose 的同一权威 kernel；Python 侧仅按有界 batch 切块，避免大轨迹临时内存无界。单 pose API 保留并与批量单元素结果逐位一致。

## B. Persistent lazy observation age

age 惰性化从“路径内部”扩大到整个 episode：

- 状态保存有序 `elapsed_ledger`、全局 cursor，以及每个 detail tile 的 `last_age_cursor`；
- 每个传感器事件只把 elapsed 追加到 ledger。被读取、写入、局部规划或 checkpoint/完整 hash 使用的 tile 才从自身 cursor 补算至当前 cursor；
- 写入 tile 前先补算该 tile，随后按既有语义将本事件可见 cell 的 age 置零；
- 物化一个 tile 后更新其 cursor，绝不反复给同一历史 interval 加 age。

单独的轻量 rolling identity 以有序事件 chain digest 表示当前观测进展，避免滚动局部规划为了获取 `physical_evidence_sha256()` 而强制物化全图。最终物理 SHA 的定义不变：它始终基于完整物化后的真实数组。

## C. Dirty-tile incremental planner projection

`MultiresSensorObservationState` 维护 dirty detail/coarse cell 集。`FormalEpisode` 维护长期 planner-global map 和 source-to-planner 反向投影索引：

- 局部滚动边界先物化当前 64 m planning window 所需的 detail tile；
- 仅把 dirty coarse cells 的 elevation/obstacle/age/quality/count 等层写入已有 planner map；
- 全局 planner map 对象长期存在，bridge 提供受限的按 flat-index patch 接口，避免为每次滚动重建所有 layer；
- local map 仍由当前位置准确重建；
- 候选刷新/策略决策边界先全量物化 age 和 dirty coarse cells，随后构造完整策略 observation。

任何没有 dirty cell 的滚动边界不得复制整张 global map。若 dirty 区域无法用固定投影精确表示，回退到完整投影并记录测试可见的 fallback，而不是近似。

## D. Checkpoint alias

保持现有合同：完整不可变 `update-N.pt` 是恢复权威；`latest.pt` 只是指向它的原子 hardlink alias。V2 只保留并重新验证此行为，不引入第二次序列化、软链接或缩减 worker state 的新路径。

## Non-goals

- 不改变地面平台局部规划算法、候选生成、Reward V4、PPO、worker 数量或 update 边界。
- 不改变 0.2 m 栅格、1 m 路径采样、30 m 传感器或可覆盖分母。
- 不把跨 sensor evidence generation 的结果当成同一张地图，也不修改 cache/replay checkpoint schema。

## Acceptance gates

1. 单 pose native mask = batch 单元素 mask；batch 内每个结果 = 旧单 pose API。
2. 一个含重复/non-consecutive pose 与远端历史 tile 的路径，V2 与逐样本基准完整物化后所有权威 bytes 完全一致；中途强制异常保持原子性。
3. 远端 tile 在多个路径之间不因新事件被写入；首次局部读取/最终 checkpoint 物化后与 eager 基准逐位一致。
4. 滚动 planner map 仅 patch dirty cells，且与完整投影请求的 layer、路径输出、reason、coverage 和 diagnostics 完全一致。
5. `latest.pt` 与不可变 update checkpoint 同 inode，加载 payload 相同。
6. 在同一冻结任务、相同 checkpoint 下，地面宏动作语义完全一致，并以外部性能 artifact 记录提升；未达门不得切换 live training source。
