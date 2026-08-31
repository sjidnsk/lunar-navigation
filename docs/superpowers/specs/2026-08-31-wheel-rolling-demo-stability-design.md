# Wheel Rolling Demo Stability Design

## 1. 背景与已确认事实

本设计修复本机 ROS 2 Jazzy 轮式月表 RViz Demo 中已经复现的三类问题：

1. 首个滚动周期可能在全局路线已经生成后、局部搜索尚未启动前返回 `NO_PATH`。诊断特征是
   `global_call_count=1`、`local_goal_elapsed_ms>0`、`local_call_count=0`，对应局部门户集合为空。
2. Demo 每 500 ms 重发一张随车辆移动的局部 GridMap。滚动规划发布前要求全局图、局部图和 TF
   序列与规划快照一致，因此耗时超过一个局部图周期的结果可能被标记为 `STALE_INPUT`。
3. 已经发布过有效滚动段后，后续单周期 `NO_PATH` 或 3 s `TIMEOUT` 会立即终止整个 Action；
   Demo 又会忽略空路径，导致显示车辆仍可能沿旧路径前进。

同一运行中已经多次得到 `planning_outcome=0`、`reason_code=PLAN_FOUND`、
`has_reference=true`，所以本设计不替换全局搜索或轮式局部搜索算法。修复重点是安全门户覆盖、滚动生命周期和
Demo 输入节奏。

## 2. 目标

- 让 12 m 滚动前瞻能够回退搜索到车辆前方至少一个全局栅格，而不是只覆盖前瞻末端附近 8 个栅格。
- 已有成功滚动段之后，允许瞬时 `NO_PATH` 或 `TIMEOUT` 在严格上限内重新尝试；重试期间发布空轮式
  参考和空路径，禁止继续执行旧段。
- 保留当前严格地图版本合同：输入版本变化后不直接发布基于旧局部图计算的路径。
- Demo 只在车辆相对上一张局部图中心移动达到 4 m 时发布新局部 GridMap；odometry 和可视化仍按
  现有 2 Hz 更新。
- 失败时输出带合法 frame 的空 Path；Reporter 的每个滚动周期显示该周期诊断到达时的最新车辆位置。
- 保持 `/lunar_demo/*` 隔离，不启动控制器，不发布 `/Car/T5/Car_Cmd_Vel`。

## 3. 非目标与安全边界

- 不取消 `SamePlanningIdentity` 对 `local_sequence` 的比较，不允许未经最新地图重新认证的旧路径继续执行。
- 不延长 3 s 单周期硬截止时间；超时重试使用新的独立周期预算，不能把一次搜索无限延长。
- 不改变轮式足迹、净空、坡度、碰撞认证或终点姿态约束。
- 不增加 `/tf_static`。当前仓库隔离合同明确禁止 pure planner ROS wrapper 出现该接口；TF 可视化时效问题
  不在本次行为修复范围内。
- 不修改生产 `/Car/T3/*`、`/Car/T4/*` Topic 名称、Action 定义或控制器接口。
- Jazzy 结果不证明 Humble、Orin、DDS、rosbag 或实车就绪；未执行层级继续标记 `NOT_RUN`。

## 4. 总体数据流

```text
RViz goal
  -> 一次全局路线
  -> 12 m 滚动决策
  -> 动态门户回退采样
  -> 轮式局部搜索
  -> 发布前输入版本检查
  -> 成功：发布新 MotionReference/Path
     失败且已有成功段：发布空输出、记录重试诊断、有界重试
     失败且没有成功段或重试耗尽：终止 Action

Demo vehicle
  -> odometry 2 Hz
  -> 相对上一局部图中心移动 >= 4 m 时才重建并发布 local GridMap
  -> 空 Path 立即清空 active_path 并停止推进
```

## 5. 安全门户回退

`BuildSurfacePortalSet()` 当前固定使用 `0..7` 个全局栅格的 backoff。Demo 的前瞻为 12 m、全局图
分辨率为 1 m，因此只检查约 5–12 m 的路径进度，可能漏掉车辆前方 1–4 m 的安全入口。

修改后根据
`desired_horizon_progress_m - projected_route_progress_m` 和全局分辨率生成 backoff：

- 从目标前瞻进度开始，每次回退一个全局栅格；
- 最远回退到 `projected_route_progress_m + global_resolution`；
- backoff 步数最多 32，保持计算有界；
- 每个纵向样本继续使用现有 9 个横向偏移、全局膨胀占据检查、局部
  `free_with_height` 和 clearance 检查；
- 候选仍按路线进度、全局净空、局部净空和稳定序排序，最终最多保留 32 个；
- 最终目标位于本地窗口时仍使用原有精确目标分支，不做位置或姿态放宽。

若所有候选仍不安全，Action 继续使用公共 `NO_PATH` 结果合同，同时诊断增加
`rolling_failure_stage=PORTAL_SET`，用于区分“没有局部门户”和“局部搜索无解”。

## 6. 滚动瞬时失败恢复

新增 ROS 参数：

```yaml
rolling_transient_retry_limit: 2
```

数值表示已有至少一个成功滚动段后允许的额外连续重试次数，必须位于 `0..10`。默认 `2` 表示首次瞬时
失败后最多再尝试两次；任何成功滚动周期都会把连续失败计数清零。

可恢复失败仅包括：

- 门户或局部搜索返回 `NO_PATH`；
- 单个滚动周期返回 `TIMEOUT`。

以下结果始终立即终止，不进入恢复：

- `REQUEST_CANCELED`；
- `INVALID_INPUT`；
- `PLANNER_ERROR`；
- 全局路线首次建立失败；
- 尚未产生任何成功滚动段时的局部失败；
- 连续失败次数超过参数上限。

每个可恢复失败周期必须：

1. 发布本周期原始诊断，并增加 `rolling_recovery=true`、
   `rolling_recovery_attempt=<1-based>`；
2. 发布空 `MotionReference`、空局部 Path 和空 TimedPath，使控制端和 Demo 不再沿旧路径推进；
3. 保持 Action 为 active，等待 `rolling_poll_period_ms` 后用新的 3 s 周期预算重试；
4. 不把上一成功段当作当前安全证明，也不重新发布上一成功段。

恢复耗尽后返回最后一次原始失败原因并按现有 Action 结果合同终止。

`STALE_INPUT` 保留现有无计数重算语义：它说明结果基于旧身份，不能发布，但并不立即终止 Action。

## 7. Demo 局部图发布节奏

`lunar_surface_demo_node` 保留 2 Hz timer 和 0.5 m/周期的测试车辆推进。拆分发布职责：

- global map：保留当前启动阶段有限重发；
- odometry、默认目标和显示数据：每个 timer 周期发布；
- local GridMap 与 local OccupancyGrid 可视化：首次发布，之后仅当车辆相对上一局部图中心的平面距离
  不小于 `4.0 m` 时成对发布；
- 设置新起点后清除上一局部图中心，使新起点下一周期必定发布新局部图；
- 收到非空轮式/足式路径时替换 active path；收到空路径时清空 active path，车辆立即停止推进。

4 m 小于 64 m 局部窗口半宽 32 m，也小于 12 m 路线前瞻，不改变局部覆盖边界。它把静止或小位移时
无内容价值的 2 Hz 地图版本更新降为按位移更新，同时仍允许真实滚动位置生成新局部地图。

## 8. 输出、Reporter 与 RViz

- 失败的轮式局部 Path 使用 `frame_id=odom`；失败的全局 Path 使用 `frame_id=map`；TimedPath 内部
  Path 与对应局部 Path header 一致。
- 空输出仍保持 `poses.empty()`，不会被误认为有效参考。
- Reporter 在处理每条 diagnostics 时，用当时最新 odometry 更新本周期 `start=(x,y)`；目标仍保持
  当前 RViz goal。滚动日志由此反映车辆实际进度，不再重复最初点击目标时的位置。
- 不通过关闭 RViz Message Filter、放宽 Fixed Frame 或隐藏错误来消除告警。

## 9. 测试策略

所有行为修改按测试先行实施。

### 9.1 Core 门户测试

- 构造 12 m 前瞻，令 5–12 m 候选均不可行、4 m 候选可行；修改前必须 `NO_PATH`，修改后必须选中
  4 m 安全门户。
- 验证最终精确目标分支仍拒绝全局或局部不安全目标。
- 验证动态 backoff 不超过 32 个纵向样本，最终候选不超过 32。

### 9.2 ROS 滚动生命周期测试

- 已有成功段后，第一次局部 `NO_PATH` 发布空输出但 Action 保持 active，下一次成功后重新发布非空路径。
- 已有成功段后，第一次 `TIMEOUT` 进入恢复；恢复成功后连续失败计数清零。
- 连续失败超过 `rolling_transient_retry_limit` 后返回最后一次原始失败并终止。
- 首段 `NO_PATH`、`INVALID_INPUT`、`PLANNER_ERROR` 和取消仍立即终止。
- 地图身份变化仍生成 `STALE_INPUT`，且旧结果不发布。

### 9.3 Demo 与报告测试

- 小于 4 m 的连续运动不重发局部图；累计达到 4 m 时恰好重发一次。
- 新起点强制下一周期重发局部图。
- 空路径清空 active path，后续 timer 不再改变车辆位置。
- 空失败 Path 的 frame 分别为 `odom` 和 `map`。
- Reporter 的连续滚动摘要使用最新 odometry 起点。

### 9.4 验证层级

1. 根目录 `/usr/bin/python3 -m pytest`，显式忽略当前工作树生成的 build/install/log 目录；
2. Jazzy 独立 build/install/log 目录下构建 `lunar_pure_planner_ros`；
3. `lunar_pure_planner_core` 和 `lunar_pure_planner_ros` 包级 CTest 串行执行；
4. `tests/test_launch_contract.py` 与受影响静态合同；
5. 隔离 `ROS_DOMAIN_ID`、`start_rviz:=false` 的 headless Demo：复现已记录起点和目标，检查
   `PLAN_FOUND`、合法空路径 frame、滚动恢复和车辆停止行为；
6. `git diff --check`。

本机 Jazzy runtime 只能记为 `PASS_JAZZY_DEMO`。Humble、Orin、DDS、rosbag 和实车均记为
`NOT_RUN`，除非在对应环境实际执行。

## 10. 文档与验收标准

更新 `README.md`、`docs/操作指令.md` 和 `VERIFICATION.md`，说明：

- Demo 局部图按 4 m 位移更新；
- `rolling_transient_retry_limit` 的安全语义；
- `STALE_INPUT` 是未发布旧结果的重算诊断；
- `ROLLING_RETRY` 期间输出为空且车辆不得继续推进；
- 正式单周期成功仍只接受 `planning_outcome=0`、`reason_code=PLAN_FOUND` 或
  `PLAN_FOUND_LATE`、`has_reference=true`；
- Demo 最终完成还必须有 Action 正常完成和 odometry 进入最终目标容差，不能用中间一次
  `PLAN_FOUND` 代替。

验收时必须满足：

1. 新增回归测试完成 RED→GREEN 证据；
2. 受影响包和静态合同全部通过；
3. 记录实际 Jazzy Demo 结果与剩余 `NOT_RUN` 边界；
4. 不出现失败后车辆继续沿旧 active path 前进；
5. 不接入 `/Car/T5/Car_Cmd_Vel`，不启动轮式控制器；
6. 仅提交本设计、对应计划、源码、测试和正式文档，不提交构建、日志或 runtime artifact。
