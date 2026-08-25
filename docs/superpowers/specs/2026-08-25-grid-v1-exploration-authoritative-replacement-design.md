# Grid Traversability V1 探索权威替换与薄适配设计

## 状态

本设计于 2026-08-25 在对话中批准，采用“方案 A：权威替换 + 薄适配层”。用户已授权在设计文档完成、自检并提交后直接制定实施计划和执行，不设置额外人工复核停顿点。

本文档定义最新版轮式规划器接入现有纯前沿探索闭环的代码权威、数据语义、错误分类、验证方法和回滚边界。它不改变前沿算法、规划算法、车辆能力或探索终止规则。

## 权威基线

- 探索基线：`wip/exploration-readiness-20260824` 的 `9a90ab2340a81905d471ba6fa1951bb9403e048b`。
- 规划器权威基线：`feat/high-res-traversability-v1` 的 `bf9172f09dba9f9bcf5bcf6477ef1b10f29ddec6`。
- `bf9172f` 包含 `41d1bc7` 的完整全局路线输出，并补齐 Orin 发布包中的 Grid V1 生产源文件。
- 适配工作必须从探索基线建立仓库根目录之外的独立工作树和功能分支，不直接改写上述两个来源分支。

规划器权威基线在本次适配期间冻结。实施过程中若规划器来源分支再次前进，不自动吸收新提交；新提交必须单独审查并形成明确的后续同步。

## 目标与硬边界

### 目标

1. 现有探索闭环只调用最新版 `grid_traversability_v1` 轮式规划路径。
2. 前沿候选继续作为探索方向和信息增益目标；规划器返回完整全局引导路线以及当前局部可执行路径。
3. 控制器连续跟踪一次规划产生的完整局部轨迹，不在轨迹内部的每个运动原语之间停车。
4. 探索候选排序使用完整全局路线代价，不把约 8 m 的局部路径长度误当成远距离候选总代价。
5. 保留全局规划、局部规划和总规划耗时统计，并兼容 V1 新增诊断字段。
6. 只有障碍格参与车辆包络膨胀；未知格不可搜索，但不作为相邻自由格的膨胀源。
7. 任务区域内不存在可达前沿时结束；覆盖率只计算和记录，不设置完成阈值。

### 非目标

- 不修改 Grid V1 的全局 A*、局部 A*、路径简化、方向选择或速度时序算法。
- 不降低地图分辨率、车辆足迹、最小净空、障碍阈值或最终路径复检要求。
- 不减少探索候选的规划调用次数。
- 不新增 DWA、TEB、MPC、SE(2) 格点或新的运动原语。
- 不修改课题三 `/Car/T3/...` 输入和课题四 `/Car/T4/...` 输出命名。
- 不新增时间戳、地图版本、协方差、地图新鲜度或观测时间准入校验。
- 不以本机 Jazzy 或 Humble 容器结果替代 Jetson AGX Orin、DDS 或实车验收。

## 选定架构

### 1. 路径级权威替换

不得对探索分支和规划器分支执行普通整树合并，也不得把两个实现手工拼成混合规划器。适配分支以规划器权威提交中的下列内容为准：

- `ros2_ws/src/lunar_pure_planner_core/**`
- `ros2_ws/src/lunar_pure_planner_ros/**`
- `ros2_ws/src/lunar_planning_msgs/**`
- `config/pure_planner.yaml`
- `config/external_interfaces.yaml`
- 规划器 launch、Grid V1 接口检查、单元测试和验证文档
- `tools/create_car_orin_bundle.py` 及其 Grid V1 发布包测试

探索侧保留：

- `lunar_pure_exploration_core`、`lunar_pure_exploration_ros` 和 `lunar_pure_exploration_sim`
- `config/pure_exploration.yaml`
- 300 m 仿真 launch、RViz 配置、结果记录脚本和操作说明
- `lunar_pure_wheeled_controller`
- 停车后规划、局部段连续执行、终点刷新和探索完成规则

若同一路径同时存在来源差异，先恢复为规划器权威版本，再以独立的适配提交添加本文定义的最小变化。最终必须能列出全部允许偏离 `bf9172f` 的规划器文件及差异原因。

### 2. 运行模式

300 m 探索 launch 必须显式设置：

```yaml
wheel_planner_mode: grid_traversability_v1
rolling_surface_enabled: false
```

`wheel_planner_mode` 决定使用最新版 V1；不得依赖仍为 `legacy_certified` 的默认值。测试必须证明 V1 已激活，且没有静默回退旧规划器。

`rolling_surface_enabled` 只控制旧 `legacy_certified` 的长生命周期滚动规划会话。它在 V1 模式下不参与算法分支。保持 `false` 是为了让探索节点继续拥有“停车—候选规划—局部路径执行—刷新”的编排权；它不关闭局部地图、局部规划或后续重规划。

### 3. 数据流和路径语义

```text
T3 全局图 + T3 局部 GridMap + odometry + tf
  -> Grid V1 持久高分辨率可通行地图
  -> 完整全局二维路线
  -> 约 8 m 局部二维路线与轮式轨迹
  -> PlanMotion 结果
       |- MotionReference.path_preview：当前局部几何路径
       |- MotionReference.trajectory：控制器当前可执行轨迹
       |- PlannerDiagnostics.best_cost：完整全局路线长度
       `- /Car/T4/planning/wheeled_global_path：完整全局路线可视化
  -> 探索候选排序
  -> 发布获选候选的 MotionReference
  -> 控制器连续跟踪完整局部轨迹
  -> 局部段完成或失效后停车并进入下一探索周期
```

前沿坐标不是要求车辆一次抵达的执行终点。局部段正常完成属于探索进展，不写入候选失败记忆。新地图到达后重新检测前沿，原前沿可以移动、分裂、消失或再次获选。

本文档覆盖旧的 `2026-08-24-stop-gated-local-segment-exploration-design.md` 中“`path_preview` 是完整全局路线”的描述。对 Grid V1 而言，`path_preview` 与发布到 `wheeled_path` 的当前局部路径一致；完整全局路线通过独立 `wheeled_global_path` 发布。

## 薄适配层

### 1. 完整全局路线代价

当前探索 `PlannerEvaluation.path_length_m` 从 `MotionReference.path_preview` 计算。Grid V1 的 `path_preview` 是有界局部路径，因此远距离候选会在约 8 m 附近产生相近代价，改变原有 gain-over-cost 排序。

适配规则：

1. Grid V1 在完整全局 A* 路线生成后，以相邻世界坐标点的欧氏长度之和计算完整路线长度。
2. 将有限且非负的结果写入已有 `PlanningResult.best_cost`。
3. ROS 层沿用已有 `has_best_cost/best_cost` 诊断或反馈字段，不修改 `PlanMotion.action`。
4. 探索 PlannerClient 在同一请求的有效 `best_cost` 可用时将其作为 `path_length_m`。
5. 只有兼容旧规划器且没有有效 `best_cost` 时，才允许回退为 `path_preview` 长度。
6. 不把完整全局路线塞回 `path_preview`，避免控制器误把全局引导路线当成当前可执行路径。

路线长度必须来自规划器实际生成的全局格路径，而不是候选直线距离，也不引入新的候选预筛选。

### 2. 原因码和失败记忆

ROS Action 结果必须保留已知 Grid V1 详细原因码，不能仅按粗粒度状态统一折叠为 `NO_PATH` 或 `INVALID_INPUT`。探索侧按以下类别处理：

| 类别 | 原因码 | 探索行为 |
| --- | --- | --- |
| 成功 | `PLAN_FOUND` | 参与候选排序并保留局部执行参考 |
| 候选穷尽 | `GOAL_NOT_FREE`、`GLOBAL_NO_PATH`、`LOCAL_NO_CANDIDATE`、`LOCAL_NO_PATH` | 当前规划 epoch 中记为该候选不可达 |
| 可重试失效 | `STALE_PATH_INVALIDATED`、`TIMEOUT`、`REQUEST_CANCELED` | 不写永久失败记忆；停车后刷新快照并重试或重建候选批次 |
| 输入/系统错误 | `START_NOT_FREE`、`INVALID_INPUT`、`MAP_RESOLUTION_MISMATCH`、`GLOBAL_MAP_GEOMETRY_INVALID`、`POSTCHECK_FAILED`、`PLANNER_ERROR` | fail-closed，发布明确错误，不得计入“无可达前沿”完成条件 |

`START_NOT_FREE` 说明当前车辆起点不满足规划地图自由条件，是车辆/地图/膨胀配置问题，而不是某个前沿自身不可达。把它计入所有候选失败会造成候选过早耗尽，因此必须阻止该批次产生正常完成结论。

Action `planning_outcome` 与详细原因码保持一致：候选穷尽使用 `GOAL_INFEASIBLE`，`STALE_PATH_INVALIDATED` 使用 `ACTIVE_REFERENCE_INVALIDATED`，输入错误使用 `INVALID_REQUEST`，`POSTCHECK_FAILED/PLANNER_ERROR` 使用 `NUMERICAL_FAILURE`，超时和取消继续使用各自现有 outcome。结果无论属于哪一类都保留原始详细 `reason_code`。

只有在车辆已停车、规划 epoch 当前有效、所有候选均得到候选穷尽类结果且任务区已无其他可达前沿时，才能输出 `COMPLETED_NO_REACHABLE_FRONTIER`。

### 3. 诊断兼容

现有探索计时解析器必须从“字段集合完全相等”改为“基础必需字段完整且唯一，允许额外字段”。

必需语义继续包括：

- 全局规划调用次数和耗时；
- 局部候选/局部搜索调用次数和耗时；
- 总规划耗时；
- `planning_outcome`、`has_reference` 和 `reason_code`；
- 可用时的 `has_best_cost/best_cost`。

Grid V1 的 `grid_v1_*`、地图 revision、tile、融合、膨胀和搜索统计字段可以作为扩展字段存在。重复必需字段、缺少必需字段、数值格式非法或请求关联不一致仍然是协议错误。

停车等待时间继续用本机单调时钟独立统计，不计入全局、局部或总规划算法耗时。部署到车上后，单调时钟只度量本进程内实际经过时间，不依赖 ROS 时间、系统日期或上游消息时间戳。

### 4. 障碍膨胀与未知格

车辆中心可搜索条件保持为膨胀后 `FREE`，但膨胀源只包括真实不可通行格：

- 局部 occupancy 达到阈值的障碍；
- 超过平台坡度能力而派生的不可通行地形；
- 全局 OccupancyGrid 中达到阈值的占据格。

`UNKNOWN` 本身仍不可被搜索、捷径或最终路径穿越，但不向相邻 `FREE` 扩张禁行范围。不得把局部 `NaN` 或全局 `-1` 当作障碍膨胀源。

本文条款覆盖旧 Grid V1 设计文档中“原始障碍和未知区域共同膨胀”的文字。实施必须同步修正文档，并增加跨局部未知边界和全局未知边界的回归测试。

## 外部接口

保持下列接口不变：

- 输入：`/Car/T3/mapping/global_overview`、`/Car/T3/mapping/grid_map`、`/Car/T3/localization/odometry`、`/tf`
- Action：`/Car/T4/plan_motion`
- 局部参考：`/Car/T4/planning/wheeled_reference`
- 局部路径：`/Car/T4/planning/wheeled_path`
- 完整全局路径：`/Car/T4/planning/wheeled_global_path`
- 带规划耗时路径：`/Car/T4/planning/wheeled_path_timing`
- 诊断：`/Car/T4/planning/diagnostics`
- 探索执行参考：`/Car/T4/execution/motion_reference`
- 控制命令：`/Car/T5/Car_Cmd_Vel`

`PlanMotion.action` 和 `MotionReference.msg` 不增加字段。最新版新增的 `TimedPath.msg` 随规划消息包权威替换一起进入探索分支。

## 实施与提交边界

实施分为可独立回退的三组提交：

1. **权威规划器导入**：从 `bf9172f` 路径级替换规划器核心、ROS、消息、配置、launch、测试和 Orin bundle 生产清单，不添加探索适配。
2. **薄适配层**：显式 V1 模式、完整全局路线 `best_cost`、原因码分类、诊断字段超集解析和未知格非膨胀回归。
3. **文档与验证**：更新被覆盖的旧设计语义、运行说明及验证证据，不把未执行的平台验证写成通过。

任一提交不得夹带无关重构、候选调用削减、控制器算法修改或实车 launch 改动。

## 测试与验收

### 结构与静态检查

1. 对权威替换路径逐文件比较 `bf9172f`；除适配白名单外内容一致。
2. 检查探索、仿真和控制器目录没有被旧规划器分支覆盖。
3. 检查 `PlanMotion.action`、`MotionReference.msg` 兼容，`TimedPath.msg` 已纳入构建和发布包。
4. 检查 300 m launch 显式选择 `grid_traversability_v1` 且旧 rolling 会话关闭。
5. 检查 repository boundary 和外部接口脚本通过。

### 自动化测试

测试先于生产修改，至少覆盖：

1. V1 激活且没有 legacy 回退。
2. 长度大于局部前瞻的全局路线产生完整 `best_cost`；探索使用该值，而不是局部 `path_preview` 长度。
3. 旧规划器无 `best_cost` 时兼容回退仍可用。
4. 基础诊断字段完整时允许额外 `grid_v1_*` 字段；重复、缺失或非法基础字段仍拒绝。
5. `STALE_PATH_INVALIDATED` 不进入失败记忆并触发刷新。
6. 候选穷尽类原因只影响对应候选。
7. `START_NOT_FREE`、`INVALID_INPUT` 和 `PLANNER_ERROR` 不会触发假完成。
8. 未知格不膨胀相邻自由格；障碍格按“平台外接半径 + 最小净空”正常膨胀。
9. 停车准入、规划 epoch、局部段连续执行和终点刷新行为保持不变。
10. 全局、局部、总规划计时以及独立停车等待计时保持可观测。

### 构建环境

1. 本机 ROS 2 Jazzy：完成规划器、探索器、控制器和仿真相关包的干净构建及测试。
2. Ubuntu 22.04 + ROS 2 Humble：使用项目指定环境做干净外部构建和测试，并验证 Orin bundle 包含 Grid V1 生产源文件。
3. Jetson AGX Orin、真实 DDS、rosbag 连续输入和实车：未实际运行前保持 `NOT_RUN`。

### 运行验收

先运行与历史问题同配置的 120 s smoke，再决定是否进入完整 300 m 运行：

1. 规划诊断证明使用 `grid_traversability_v1`。
2. 候选均在停车准入后调用规划器。
3. 至少一个大于 8 m 的候选报告完整全局路线代价，并生成当前局部执行轨迹。
4. 控制器连续执行包含多个轨迹点或运动阶段的局部段，轨迹内部不出现探索强制停车。
5. 局部段完成后停车、刷新地图和前沿，再构建下一候选批次。
6. 不再出现因详细原因码折叠造成的 planner contract mismatch。
7. RViz 同时显示完整全局路径和当前局部可执行路径。
8. 记录覆盖率、完成局部段数、规划次数、全局/局部/总规划耗时和失败原因分布。

完整 300 m 探索只有在终态为 `COMPLETED_NO_REACHABLE_FRONTIER`、完成局部段数大于零、结果文件完整且不存在输入/系统错误被误计为候选穷尽时才算算法闭环完成。覆盖率必须报告，但不设最低百分比。

## 风险与处置

- **来源分支再次变化**：本次冻结到 `bf9172f`，后续变化另行审查，不在实施中追逐移动 HEAD。
- **路径语义混淆**：局部 `path_preview`、可执行 `trajectory` 和独立全局路径分别测试，不复用同一 Topic 表达不同语义。
- **候选代价饱和**：使用规划器完整路线 `best_cost`，以远距离回归场景证明没有约 8 m 饱和。
- **假无前沿完成**：输入/系统错误 fail-closed，只有当前停车 epoch 的候选穷尽结果参与完成判定。
- **安全边界退化**：不降低障碍膨胀半径；只纠正 UNKNOWN 不作为膨胀源，并保留 UNKNOWN 不可搜索。
- **跨平台结果误读**：Jazzy、Humble、Orin 和实车证据分别记录，禁止由低一级环境推断更高一级就绪。

## 回滚

三组实施提交必须保持可独立识别。若适配失败，可回退薄适配和权威导入提交，恢复到探索基线 `9a90ab2`；不得通过删除工作树、重置共享分支或覆盖其他用户改动来回滚。运行产物写入仓库外的既定 `~/CodexDownloads/lunar_navigation/...` 路径，不进入 Git。
