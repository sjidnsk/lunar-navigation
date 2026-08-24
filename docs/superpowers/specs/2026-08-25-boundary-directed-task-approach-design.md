# 基于任务边界的区外入区探索设计

日期：2026-08-25

状态：已批准（2026-08-25）

## 1. 目标

扩展现有纯前沿探索器，使车辆接收 `PureExplorationTask.boundary` 后，即使初始位姿位于任务区域外，也能沿未知区域方向安全推进，进入任务区域后再继续现有 WFD 自主探索。

本设计采用“未知意图目标、已知安全执行目标”语义：任务区内 UNKNOWN 可以决定推进方向，但发布给 `/Car/T4/plan_motion` 的实际目标必须位于当前已知、map-backed、满足完整平台足迹和净空约束的 FREE 区域。规划器、控制器和 UNKNOWN 不可通行合同不变。

## 2. 非目标

- 不在 `PureExplorationTask.msg` 中新增人工入口点。
- 不允许规划器或控制器把 UNKNOWN 当作可通行区域。
- 不修改 WFD、现有区内候选信息增益或 `COMPLETED_NO_REACHABLE_FRONTIER` 的算法含义。
- 不把任务区外观测计入任务覆盖率。
- 不添加 SLAM、真值地图泄漏、地图降采样或安全足迹缩小。
- 不借本功能重构无关规划、RViz 场景或 rosbag 代码。
- 本机 Jazzy 自动测试不替代 Humble/Orin 的 DDS、Action、控制器和 RViz 实机闭环验收。

## 3. 当前限制

当前 `FrontierDetector` 只在 `TaskRaster` 内从机器人所在 FREE cell 或其四邻 FREE cell 启动。车辆位于 `boundary` 外时，机器人 cell 被分类为 `kOutsideTask`，检测结果为 `kNoReachableFreeStart`。ROS 节点保持 `SELECTING_FRONTIER`，但不会产生目标或运动参考。

直接把任务区内 UNKNOWN cell 发送给当前规划器也不可行：全局和局部安全投影都把 UNKNOWN 视为 hazard，目标本身必须 hard-feasible。入区能力因此必须在探索器中产生安全的部分推进目标。

## 4. 路径规划算法权威

本功能的实现基线固定为 `feat/grid-v1-exploration-adaptation` 的
`3f5b0f746a15d48edced0c045387154eb808e59c`。必要审查确认：

- `200dad9` 导入了 `feat/high-res-traversability-v1@bf9172f` 的完整 Grid
  Traversability V1 规划器，导入点两者的
  `lunar_pure_planner_core` 和 `lunar_pure_planner_ros` 树一致；
- `bad57cc` 在该规划器上暴露完整全局路线代价；
- `3f5b0f7` 继续保留 Grid V1 的细粒度规划原因，是审查时已完成的
  最新集成提交。

所有入区候选和区内 WFD 候选都必须通过现有 `PlannerClient ->
/Car/T4/plan_motion` 路径调用该基线的正式 `PlanMotion` 实现。不得在探索包内复制
A*、使用直线可达性代替规划、回退到旧规划器或更改 Grid V1 搜索、车辆包络、路径复检与
失败原因语义。

会启动规划器的探索运行入口必须显式设置：

```yaml
wheel_planner_mode: grid_traversability_v1
rolling_surface_enabled: false
```

不得依赖当前仍为 `legacy_certified` 的默认值。测试必须从请求诊断中证明
`grid_v1_active=true`，并保留 `PLAN_FOUND`、`GLOBAL_NO_PATH`、
`LOCAL_NO_CANDIDATE`、`LOCAL_NO_PATH`、`START_NOT_FREE` 和
`STALE_PATH_INVALIDATED` 等详细原因。探索排序使用该版本返回的完整全局路线
`best_cost`，不用约 8 m 的局部 `path_preview` 代替总路径代价。

本次功能分支对该精确提交做可复现冻结。实施期间若其他规划器分支继续前进，不自动合并未审查的
WIP 变更；后续升级必须另行确认权威提交和回归证据。

## 5. 总体架构

探索器内部增加与生命周期状态正交的导航阶段：

```text
START + boundary + map + pose
  -> 当前膨胀足迹是否全部位于任务区内的已知 FREE cell？
       yes -> EXPLORE_TASK -> 现有 TaskRaster/WFD 流程
       no  -> APPROACH_TASK -> boundary 定向引导与安全截断
                                     |
                                     +-> 到达/地图更新后重新判断
```

`ExplorationState` 继续负责 `WAITING_FOR_INPUT / SELECTING_FRONTIER / PLANNING / EXECUTING / REPLANNING / PAUSED / COMPLETED / ERROR`。导航阶段只选择哪一种候选流水线，不新增 ROS 状态枚举。

两个阶段共享：

- `PlanMotion` Action 客户端和结果分类；
- 最终参考发布、执行监视、暂停、恢复、取消和任务替换；
- 请求身份、超时、资源错误及异步 epoch/generation 防护；
- 平台足迹、净空和候选 pose 的安全判定实现。

两个阶段不共享语义不同的排序分数、覆盖统计或完成判定。

## 6. 阶段判定

`APPROACH_TASK -> EXPLORE_TASK` 的唯一判定是：当前车辆 pose 下，平台原始足迹与 `minimum_clearance_m` 闭圆盘的 Minkowski 和所接触的每个 logical cell 都同时满足：

1. cell center 属于任务多边形；
2. cell 位于当前全局图内；
3. cell 分类为 FREE。

该判定复用现有候选生成器的闭边界碰撞数学；相切任务边界、UNKNOWN、OUTSIDE_MAP、OCCUPIED 或 OUTSIDE_TASK 均不能通过。只检查机器人中心是否在多边形内不构成入区成功。

阶段只在没有活动执行目标时切换。入区目标执行途中即使已满足完全入区条件，也先完成当前安全运动段，避免地图更新引发频繁抢占。目标释放后的下一轮进入 WFD。车辆后续意外离开任务区域时，下一次无活动目标的构建重新进入 `APPROACH_TASK`。

## 7. BoundaryGuidance 输入与输出

ROS 无关核心新增 `BoundaryGuidance`。输入为：

- 完整全局 `OccupancyGridView`；
- 当前任务 `TaskRaster`；
- 当前机器人 `Pose2`；
- 平台足迹和净空；
- 已批准的 10 m / 90 度传感器模型；
- 显式资源限制。

输出为 owning `BoundaryGuidanceResult`：

- 稳定有序的 `ApproachIntent`；
- 去重后的 `ApproachCandidate`；
- 每个候选的剩余引导代价、任务区内可见 UNKNOWN 面积、引导路径上可见 UNKNOWN 数量和确定性身份；
- 无候选时的非终态原因；
- 已消费的 cell/work 计数。

`BoundaryGuidance` 不调用规划器、不修改地图、不发布 ROS 消息，也不决定任务完成。

## 8. 入区意图目标

任务栅格中满足以下条件的 cell 是边界意图 cell：

1. cell center 位于任务多边形内或边界上；
2. 至少一个四邻 cell 为 `kOutsideTask`；
3. 当前全局图中该 cell 为 FREE 或 UNKNOWN。

OCCUPIED 和 OUTSIDE_MAP 不能成为意图目标。边界意图 cell 按世界 x/y，再按 logical index 排序，保证旋转地图和输入角点顺序不改变物理选择结果。

若车辆已经在任务区中心附近但膨胀足迹跨界，仍使用同一边界意图集合推进到完全入区。若任务边界尚未落入当前地图，返回 `WAITING_FOR_TASK_MAP_COVERAGE`，不推断地图外真值。

## 9. 未知引导搜索

从机器人当前 global-map cell 对完整全局图执行一次确定性多目标搜索。搜索仅产生意图方向，不产生可执行路径。

- FREE 和 UNKNOWN 可加入引导图；
- OCCUPIED、非法原生占据值和地图外禁止加入；
- 使用四邻域，禁止对角穿越障碍角；
- 进入 UNKNOWN cell 时 `unknown_count += 1`；
- 每一步 `path_length_m += resolution_m`；
- 路径代价按 `(unknown_count, path_length_m)` 字典序比较；
- 同代价依次按前驱世界 x/y 和 logical index 决定唯一 predecessor。

因此搜索优先利用完全已知路线；必须进入未知时，优先未知 cell 更少、随后物理距离更短的方向。UNKNOWN 在这里只是排序依据，不能被提升为 FREE。

对所有可达边界意图 cell 保存 predecessor route。按总代价、意图世界坐标和 logical index 排序后依次构建候选；重复实际 pose 被稳定去重。已知障碍封闭最近入口时，其他边界方向仍可产生候选。

## 10. 安全截断与候选姿态

每条引导 route 从机器人向意图 cell 顺序扫描。route cell 对应 pose 的 yaw 指向下一 route cell；最后一个 cell 使用前一段切向 yaw。每个 pose 使用共享的 `SafePoseValidator` 检查膨胀足迹：所有接触 cell 必须位于全局图内并为 FREE。引导 route 中首个验证失败的 pose 及其后续 pose 都不能成为本轮实际目标。

最后一个验证成功且相对当前 pose 有严格推进的 pose 成为平移候选。严格推进指候选的剩余引导代价小于机器人当前代价；不能用浮点 epsilon 把零推进解释为前进。

若安全前缀只包含当前 XY，可在同位置生成朝向下一 UNKNOWN 的旋转候选，但必须同时满足：

- yaw 变化超过现有到达 yaw 容差；
- 旋转后的完整膨胀足迹安全；
- 10 m / 90 度传感器模型能看到至少一个引导 route UNKNOWN cell。

平移候选必须满足“完全进入任务区”或能看到至少一个更靠近任务区的引导 UNKNOWN cell；这样不会发布既不入区也不产生新观测的零价值目标。

可见性沿完整全局图检查 OCCUPIED 遮挡，只把 map-backed UNKNOWN 当作潜在新观测。`task_unknown_area_m2` 只累计 `TaskRaster` 内 UNKNOWN；区外 UNKNOWN 可以证明入区推进价值，但不进入 coverage 或任务信息增益统计。

## 11. 身份与共享安全逻辑

不得把入区候选伪装成 WFD `FrontierCluster`。核心引入带类型的目标身份：

```text
GoalKind = BOUNDARY_APPROACH | TASK_FRONTIER
```

`BOUNDARY_APPROACH` 身份至少包含意图 cell、实际 `CandidateKey` 和候选类型（平移或旋转）；`TASK_FRONTIER` 保留现有完整 frontier canonical key 与 `CandidateKey`。显示 ID 仍只用于诊断，回关联继续依赖唯一 `request_id`。

`ActiveGoal` 保存 `GoalKind` 和对应 owning identity。任务前沿的跨周期有效性仍比较完整 canonical key；入区目标的有效性重新检查实际 pose 安全、意图仍合法以及推进价值，不能比较虚构 frontier key。

现有 `CandidateGenerator` 中的闭足迹、Minkowski clearance、cell 接触和资源计数数学提取为 `SafePoseValidator`，两个候选流水线共同调用。提取前后现有候选测试必须逐项等价；禁止复制或简化车辆包络。

## 12. 入区候选排序与规划验证

入区候选不复用区内 `information_gain / path_cost` 分数。粗排按下列稳定顺序：

1. 完整膨胀足迹已经位于任务区内；
2. 剩余 `(unknown_count, path_length_m)` 引导代价更小；
3. `task_unknown_area_m2` 更大；
4. 候选到当前 pose 的欧氏距离更小；
5. 完整 `ApproachCandidateKey` 字典序。

候选继续使用现有每批最多 16 个的串行 `PlanMotion` 验证，并穷尽资源上限内的候选。最终只在 `kReachable` 候选中按下列顺序选择：

1. 完整入区；
2. 剩余引导代价更小；
3. 任务区 UNKNOWN 增益更大；
4. 实际规划路径更短；
5. heading change 更小；
6. 完整候选身份字典序。

发送给规划器的 `GoalRegion` 继续是 map-frame POINT，并且 point/yaw 精确取安全候选 pose。PlannerClient 不发送 UNKNOWN 意图 cell。

## 13. 目标承诺与地图更新

入区目标一旦获得规划参考并开始执行，只因以下事件释放或抢占：

- 正常到达；
- PAUSE、CANCEL 或新 START；
- 实际目标 pose 在新地图上不再安全；
- 当前意图变为 OCCUPIED/OUTSIDE_MAP 或不再属于任务边界；
- 规划/执行重试按现有上限耗尽。

仅仅出现更远或分数更高的新候选不能抢占仍安全的活动目标。地图更新造成活动目标失效时，发布一次精确 plan ID cancel，清除活动参考并从最新 map/pose 重建。

到达后释放目标并等待最新 map/pose 构建下一轮。没有输入内容变化时不得立即重复发送相同请求或忙循环。

## 14. 非终态等待与错误边界

以下情况不是完成，也不是结构错误：

| 条件 | lifecycle state | reason_code |
| --- | --- | --- |
| 缺地图、位姿、TF 或任务 | `WAITING_FOR_INPUT` | `WAITING_FOR_INPUT` |
| 当前膨胀足迹没有已知安全起点 | `SELECTING_FRONTIER` | `WAITING_FOR_SAFE_APPROACH_START` |
| 任务边界尚未进入地图 | `SELECTING_FRONTIER` | `WAITING_FOR_TASK_MAP_COVERAGE` |
| 已知障碍阻断全部引导方向 | `SELECTING_FRONTIER` | `APPROACH_NO_GUIDANCE_ROUTE` |
| 所有实际候选均结论性无路径 | `SELECTING_FRONTIER` | `APPROACH_NO_REACHABLE_TARGET` |
| 重规划耗尽且 map/pose 未变化 | `SELECTING_FRONTIER` | `APPROACH_STALLED` |

这些等待状态记录当前 map geometry/data、pose generation 和任务 identity 的签名，只由新 map、新 pose、RESUME 或新 START 唤醒。retryable transport 结果继续使用现有有界重试逻辑。

非法 polygon/map/pose、数值溢出、资源上限、Action 合同矛盾和内部身份不一致仍进入现有 `ERROR`，不能降级为无候选等待。

## 15. 完成与覆盖合同

`CompleteNoReachableFrontier()` 之前新增导航阶段 guard。只有同时满足下列条件才能发布 `COMPLETED_NO_REACHABLE_FRONTIER`：

1. 当前阶段为 `EXPLORE_TASK`；
2. 当前膨胀足迹完全位于任务区内已知 FREE；
3. 使用最新且内容未变化的地图穷尽全部区内 WFD 候选；
4. 没有候选获得有效规划参考。

任何 `APPROACH_*` 无目标或无路径结果都不能证明任务完成。

coverage 继续由现有 `TaskRaster` 计算 `(known_free + known_occupied) / task_raster_total`。区外引导图、区外 UNKNOWN 观测和引导路径长度不改变分母或分子。核心仍不按覆盖率阈值终止；20 m 合成验收在外部同时检查终态和 `coverage_ratio > 0.80`。

## 16. 状态、诊断与可视化

不修改 `PureExplorationTask.msg` 或 `PureExplorationStatus.msg`。现有 lifecycle state 保持兼容，稳定 reason_code 增加第 14 节的入区原因以及：

- `SELECTING_APPROACH_TARGET`
- `PLANNING_APPROACH`
- `EXECUTING_APPROACH`

现有 diagnostics 增加：

- `navigation_phase`
- `fully_inside_task`
- `approach_intent_count`
- `approach_candidate_count`
- `remaining_guidance_m`
- `guidance_unknown_cell_count`
- `approach_wait_reason`

现有 MarkerArray 新增独立 namespace 显示 UNKNOWN 意图点、引导 route 和实际安全候选。`current_goal` 只发布真实执行目标，不能显示 UNKNOWN 意图点冒充控制目标。

## 17. 资源限制

不降采样全局图。新增限制均在分配和循环前或逐 work unit fail-closed：

- `maximum_guidance_grid_cells`：默认复用 `maximum_task_raster_cells`；
- `maximum_guidance_work_units`：默认 `8 * maximum_guidance_grid_cells`，乘法使用 checked size arithmetic；
- `maximum_approach_candidates`：默认复用 `maximum_candidate_views`；
- predecessor、distance 和 queue storage 均在 cell limit 通过后分配；
- route 重建、可见性和候选安全检查计入 work limit。

恰好等于 limit 的 fixture 成功，少一个 work unit 的 fixture 以 `std::length_error` 失败并映射到稳定资源 reason。资源耗尽不能返回空候选并进入等待或完成。

## 18. 验证策略

### 18.1 核心测试

新增 `test_boundary_guidance.cpp`，覆盖：

- 区外机器人生成朝任务区的安全候选；
- 意图为 UNKNOWN 而执行目标始终为 FREE；
- 膨胀足迹、净空和闭边界检查；
- 最近入口被阻挡时选择其他入口；
- 已知绕行优先于未知捷径；
- 旋转 origin、负逻辑索引、凹 polygon 和输入排列确定性；
- 原地旋转候选及零观测拒绝；
- OUTSIDE_MAP、不安全起点和全边界 OCCUPIED；
- 精确资源边界、溢出和候选去重。

`SafePoseValidator` 提取必须保留现有 `test_candidate_generator` 全部结果，并增加提取前批准 fixture 的逐候选身份等价断言。

### 18.2 ROS 协调测试

扩展 `test_exploration_node.cpp`：

- 区外 START 的首个 Action goal 是安全 FREE；
- 到达和地图显露后目标单调向任务区推进；
- 完全入区后切换到现有 WFD；
- 活动目标失效时精确 cancel 并重建；
- 全部入区候选 NO_PATH 时不完成；
- 无输入变化时不忙循环或重复请求；
- PAUSE/RESUME/CANCEL/替换 START 在两个阶段一致；
- diagnostics、Marker 和 current_goal 区分意图与执行目标；
- 任务初始位姿已在区内时，现有请求顺序和输出不变。

### 18.3 合成场景

扩展现有 `test_synthetic_scenarios.cpp` 和 YAML fixture，增加 `outside_to_inside`：20 m × 20 m 任务区、车辆从区外开始、最近边界有障碍、另一路径可入区，地图随脚本化运动逐步显露。状态轨迹必须证明：

```text
APPROACH selection/planning/execution
  -> footprint fully inside task
  -> normal WFD selection/planning/execution
  -> COMPLETED_NO_REACHABLE_FRONTIER
```

外部场景断言最终 `coverage_ratio > 0.80`，但不修改核心终止条件。

### 18.4 验证边界

- 本机 `/opt/ros/jazzy`：构建相关消息、探索 core 和探索 ROS 包，运行全部相关 CTest，并执行 `git diff --check`。
- 结果必须分别报告 focused 新测试、完整探索回归和构建状态。
- Humble/Orin：真实 T3 global/local map、odometry/TF、PlanMotion Action、控制输出和 RViz 闭环属于后续目标机验收；没有该证据时明确标记为未验证。

## 19. 完成定义

实现完成需要同时满足：

1. 车辆区外时能产生安全入区候选并通过现有规划/执行链推进；
2. 任何发送给规划器的实际目标都不是 UNKNOWN、OCCUPIED、OUTSIDE_MAP 或足迹不安全 pose；
3. 车辆完全入区后继续现有 WFD，不改变区内算法和完成合同；
4. 入区无路、等待或停滞永不误报 `COMPLETED`；
5. coverage 只统计任务区；
6. 新增核心、ROS、合成场景和现有探索回归全部通过；
7. 项目文档说明新阶段、reason_code、诊断、测试证据和仍待执行的 Orin 验收。
