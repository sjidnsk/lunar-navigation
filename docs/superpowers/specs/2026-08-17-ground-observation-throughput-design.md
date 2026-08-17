# Ground Observation Throughput Design

## Goal

在不改变正式训练的传感器、地图、奖励、候选、规划器和恢复语义的前提下，消除地面宏动作滚动执行中随已探索 tile 数量增长的重复工作，并在同一完整 update 边界快速恢复训练。

## Confirmed bottlenecks

update 50–58 的 216 个地面宏动作中，单宏动作平均 86.72 s，规划器平均仅 6.29 s，占 7.25%；其余 80.43 s 位于环境执行和状态提交。24-worker 同步长尾本轮明确不改。

地面路径按不大于 1 m 的间隔产生传感器样本。每个不同 0.2 m pose cell 都读取一个 64 m × 64 m 细节窗口并计算可见性；每次样本提交还遍历全部历史细节 tile 更新 observation age。其成本随路径样本数和累计已探索 tile 数共同增长。

每个 update 已写入不可变 `update-N.pt`，随后又对同一 checkpoint 完整序列化、散列和 fsync 为 `latest.pt`。正式 resume 已以 manifest 指向的不可变 update checkpoint 为权威，因此第二份完整数据写入是冗余物理工作。

## Design

### Atomic ground-path observation

`MultiresSensorObservationState.observe_world_path()` 接收一个有序、非空的 `(Pose2, elapsed_s)` 元组。它先完成所有 pose、时间、细节窗口和 visibility 验证，之后才创建一次事务副本并提交，确保任一准备或应用错误都不会留下部分传感器状态。

连续落在同一 0.2 m cell 的样本继续共享一次 visibility。不同 cell 仍各自执行权威 visibility，不合并射线、不改变 30 m 传感器范围，也不改变路径上不大于 1 m 的采样间隔。

事务内按原顺序应用每个 elapsed 和 visibility，因此 observation count、首次覆盖、优先级覆盖、evidence generation 与逐样本执行一致。

### Lazy detail-tile aging within one path

事务为每个已有细节 tile 保存一个 elapsed cursor。一次样本只将当前 visibility 触及的 tile 补算到当前 cursor；远端历史 tile 不再被每个样本反复扫描。事务提交前统一补算所有剩余 tile。

每次补算仍按原始 elapsed 顺序执行 `float64 addition -> float32 storage`，保持原有舍入和溢出检查。coarse cell 仍在每次实际受 visibility 影响时按当时的细节状态更新，未触及 coarse cell 的旧 age 行为不变。

### Rolling map materialization

锁定地面候选期间继续跳过候选重建。地图构造只消费路径事务最终提交后的 coarse state 和当前位置 64 m detail window。固定的 task-to-planner 最近包含栅格索引只计算一次并复用；每次边界仅投影变化值并重新封装带新 stamp 的 bridge map。

不缓存或伪造 map generation，不跨 sensor evidence generation 复用规划器结果。

### Immutable checkpoint alias

不可变 `update-N.pt` 仍是 optimizer、环境状态和恢复日志的提交权威。保存完成并 fsync 后，在同一 checkpoint 目录创建指向它的临时 hardlink，再以 `os.replace()` 原子替换 `latest.pt` 并 fsync 目录。

这样 `latest.pt` 保持普通 checkpoint 文件兼容性，但不再第二次执行 `torch.save`。若目标不在同一目录、源不是普通不可变 update checkpoint或 hardlink 创建失败，操作 fail-closed，不回退到非原子复制。

## Non-goals

- 不修改 worker 数量、24-worker 同步屏障、分层等权或 policy lag。
- 不改变 0.2 m 细节分辨率、30 m 传感器范围、360 度视场或不大于 1 m 的地面采样间隔。
- 不改变候选、Reward V4、规划器搜索、宏动作终止或 PPO update 频率。
- 不删除完整 update 快照，不降低 checkpoint 频率，不修改既有 checkpoint 内容 schema。
- 不顺手修复当前 HEAD 上三个任务边缘 padding 的旧测试断言。

## Acceptance

- 批量路径与逐样本执行的全部 detail/coarse 数组、覆盖 delta、evidence generation 和 physical evidence SHA 精确一致。
- 路径准备或提交失败后，全部权威传感器状态保持原子不变。
- 同一 cell 的连续样本只执行一次 visibility；远端历史 tile 每条路径最多补算一次 elapsed 序列。
- 固定投影索引与旧投影函数逐数组精确一致。
- `latest.pt` 与对应不可变 `update-N.pt` 具有相同 inode 和 checkpoint payload。
- WHEELED、LEGGED 使用同一恢复 checkpoint 的单种子代表性复测，行为等价且端到端采集或宏动作单位工作量至少提升 35%。
- 达门后只在完整 update checkpoint 边界执行 source migration；模型、优化器、RNG、curriculum、budget 和已提交环境状态不重置。
