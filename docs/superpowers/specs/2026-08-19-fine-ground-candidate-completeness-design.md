# 地面候选精细完备性设计

## 目标

修复 WHEELED 与 LEGGED 在任务区域仍有可观测未知区时，因为把 0.2 m 站位投影回 4 m 父格而错误丢弃候选、或三个固定锚点恰好错过有效站位，而过早进入 `VALID_INCOMPLETE_TERMINAL` 的问题。

策略接口保持不变：每段 4 m 前沿最终最多输出三个 PPO 候选。内部可枚举多个 0.2 m 见证站位，但不得把它们扩展为更多策略动作。

## 不可变边界

- 4 m 地图负责任务 ROI、前沿分段与每段的三个固定锚点；0.2 m 地图负责安全站位、精确端点与传感器收益。
- 在线候选流程不得读取冻结 coverability 分母、未知真值或 Oracle。
- 不改变 PPO 张量、Reward V4、候选槽位、传感器范围、HOPPER 语义或正式滚动局部规划的硬安全裁决。
- 每个物理快照只构建一次 C++ 全局成本树；端点查询不得为每一个候选重跑全局规划。
- **禁止恢复第三级“全任务残余区域／残余分量细扫”**，也不保存其分页游标或将其接入训练路径。
- `SUCCESS` 仍只由已提交覆盖率首次达到 95% 产生；候选耗尽仍要显式记录，不得由隐藏真值补候选。

## 第一级：三锚点安全条带

1. 从 `observed & roi & adjacent(roi & ~observed)` 构造 4 m 前沿段。
2. 每段固定取 1/4、1/2、3/4 三个锚点；前沿法线仅由 `mission_roi & ~observed` 的相邻格推导。
3. 每个锚点在已观测侧的 0.2 m 安全条带中枚举见证：退让 1–20 个细格，横向半宽 1 个细格，最多 60 个；C++ 精细足迹、净空和安全余量必须通过。
4. 将同一刷新中的见证位置组成一个批次，依次进行：批量精确端点全局连通、批量 0.2 m 端点安全、批量精确可见性收益。
5. 只有正收益见证有资格输出。优先每锚点一个，空槽从同段其余合格见证稳定回填；最终每段最多三个。

## 精确端点全局连通权威

新增 C++/pybind 两阶段接口：

```text
project_ground_endpoint_context(request, maximum_edge_distance_m)
    -> GroundEndpointReachabilityContext
       { projection: ReachabilityProjection, immutable global cost tree }

query_ground_exact_endpoints(context, positions[N,3], tolerance_m=0.2)
    -> { reachable[N], minimum_cost_m[N], reason_code[N] }
```

上下文一次性构造 `MapSnapshot`、`SafeProjection` 与 global cost tree。查询以精确坐标和 0.2 m 点目标容差查找与目标球相交的、硬安全且树成本有限的全局栅格，返回最小成本；它不再把 `world.canvas` 的 4 m 父格采样值当作端点证书。

这个查询与已有 `_ground_endpoint_feasibility` 相交：前者证明同一全局树可到达目标容差区域，后者在目标附近的 observed-only 0.2 m 窗口验证精细端点安全。两者均通过才进入收益计算。正式滚动局部规划仍是执行硬门。

## 第二级：完整前沿段补扫

仅当某一段的三级固定锚点经过精确端点、精细安全和收益认证后 **没有任何正收益见证** 时，才对该段的全部 4 m 前沿格按固定顺序建立相同的 0.2 m 安全条带。

- 第二级只处理该段，不跨段、不扫描任务剩余区域、不读取真值。
- 所有补扫见证仍按批次进入同一 immutable global context、端点安全与收益查询；不新增全局树。
- 第二级结果与第一级结果合并，再按每段最多三个的同一稳定压缩规则输出。
- 若整段仍没有正收益，候选生成器只报告该事实；不会升级为全任务残余扫描。

## 性能、可审计性与恢复

- 正常刷新：一次 global tree、一次第一级精确端点批、一次第一级收益批；第二级只对第一级零正收益段增加一个段内批次。
- 诊断需分别计数第一级/第二级原始见证数、精确可达数、端点安全数与正收益数，但不得将第三级扫描状态写入 checkpoint。
- 所有输出顺序、候选 ID、成本与限制必须确定；同输入的两次运行产生相同候选集合。
- 切换时保留模型、优化器、调度器、RNG 与 global update，从一个已原子提交 update 边界启动 fresh worker episodes。
