# 规划正确性、性能受控平滑与 RViz 证据设计

日期：2026-08-05

状态：已批准，待实施

## 1. 背景

分层全局规划已经让三类平台使用 `global_map` 生成完整路线预览，并只在当前
`local_map` 内认证可执行段。当前 50 m × 50 m、0.2 m 合成地图暴露出以下后续问题：

1. 跳跃式全局层把单个栅格面积与 `minimum_landing_region_area_m2` 比较。在
   0.2 m 地图中单格面积只有 0.04 m²，小于能力文件要求的 1.327322 m²，导致所有
   候选以及起点被拒绝。
2. 轮式、足式局部格点把真实状态投影到格心和离散朝向，导致输出轨迹首点没有经过
   定位给出的真实起点。
3. 现有位置优化只移动离散顶点，时间参数化仍逐段插值；输出可以通过碰撞检查，却仍表现
   为明显折线，且优化、无需优化和离散回退在 RViz 面板中无法区分。
4. 仓库外 RViz 桥的全局 PointCloud2 只编码高程，障碍颜色只来自当前平台的局部窗口，
   因而规划器实际使用的大部分全局障碍不可见。
5. 平台代理与局部执行轨迹共用 Marker 层，真实尺寸的平台在全图视角下难以辨认。
6. 交互目标容差硬编码为轮式/足式 0.5 m、跳跃式 0.75 m；跳跃式又错误地把落点中心
   容差与整个安全落区面积绑定。

## 2. 已批准决策

1. 采用“全局一次线性预处理、局部有限窗口精细认证”的性能受控方案。
2. 不把完整全局路线转成可执行平滑轨迹；全局路线继续是安全拓扑预览。
3. 不为每个跳跃候选重复扫描整张地图。
4. 跳跃目标位置容差只约束落点中心；安全落区可以延伸到目标容差之外，但其全部边界仍须
   通过地形和净空认证。
5. 三平台交互位置容差默认改为 0.20 m，朝向容差默认保持 15°，两者独立配置且不得被
   静默放大。
6. 轮式和足式输出必须从真实定位状态开始，不允许通过在输出前面拼接未经认证的直线来
   掩盖格心投影。
7. 平滑只作用于当前 3–4 m 局部执行段；平滑结果必须重新通过连续扫掠和运动学约束。
8. 默认允许回退到已经认证的离散运动原语，并在诊断和 RViz 中显式显示；部署可启用
   `require_smoothed_execution` 使平滑失败转为失败结果。
9. RViz 继续使用标准 `sensor_msgs/PointCloud2`，不恢复 `grid_map_rviz_plugin`。
10. 不修改 `PlanMotion.action`、`MotionReference.msg` 或外部地图 Topic schema；新增内容仅为
    内部 C++ 诊断、标准 `/diagnostics` 键和仓库外 RViz 专用显示 Topic。
11. 现有按全局栅格数划分的 p95 和内存门槛不得因本工作放宽。

## 3. 范围与所有权

### 3.1 主仓 `/mnt/data/WS/lunar-navigation`

主仓负责：

- 跳跃式全局落区支持域和拓扑搜索；
- 第一跳 L0 落区认证的目标语义；
- 轮式、足式真实起点状态和终端连续连接；
- 局部曲线优化、连续碰撞复检和生成模式诊断；
- ROS 标准诊断键和核心性能基准。

### 3.2 仓库外 `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression`

仓库外工具负责：

- 交互目标策略 YAML；
- 完整全局障碍和局部危险 PointCloud2；
- 当前平台独立 Marker 层；
- RViz 左侧面板中的容差、轨迹模式、误差和警告码；
- 50 m 合成地图与三平台交互集成回归。

RViz 显示代码不重新实现规划或碰撞判定，只显示规划器和冻结地图提供的证据。

## 4. 跳跃式落区支持域

### 4.1 两级认证

跳跃式使用两级落区认证：

1. 全局层在当前 `global_map` 上生成保守落区支持域，只用于构建名义多跳拓扑。
2. 局部层只对实际将要授权的第一跳，在 L0 `local_map` 上生成完整落区多边形并认证弹道、
   飞行管和落点。

全局层的成功不能替代局部层认证；局部失败时不得返回可执行 `HopSegment`。

### 4.2 全局线性预处理

全局层先为每个栅格生成 `landing_base_safe` 掩码。一个单元只有在以下条件全部满足时才为真：

- 已知、有效、非障碍、非禁入；
- 坡度、粗糙度和平面残差满足跳跃能力；
- 横向、顶部和最低着陆净空满足能力；
- 对应跳跃能力和地图安全配置均有效。

随后对该二值掩码执行一次确定性的二维欧氏距离变换。设：

```text
r_area = sqrt(minimum_landing_region_area_m2 / pi)
```

距离场必须按单元方形边界保守修正；仅当以候选中心为圆心、半径 `r_area` 的闭圆盘不会与
任何不安全单元相交时，该中心才进入 `landing_center_safe`。这使面积判断由错误的
`resolution² >= required_area` 改为实际连续支持域判断，同时保持时间复杂度 O(N)、工作
内存 O(N)。

所有长循环检查 `stop_token`。不允许为每个候选执行 flood fill、矩形扩张或完整地图复扫。

### 4.3 候选发现与资源上限

起点保持区和目标容差内候选必须显式进入候选集合，不受普通采样顺序影响。其余候选采用从
起点可达前沿按稳定栅格顺序惰性发现，不再简单截取全图 row-major 的前 128 个安全格。

每个已展开落区只在物理单跳包络内生成候选边，并按稳定代价保留最多 8 条。继续保持：

```text
maximum_graph_nodes = 128
maximum_graph_out_degree = 8
maximum_authorized_hops = 1
```

达到节点、边或搜索资源上限但尚未证明无路时返回
`HOPPER_GLOBAL_ROUTE_RESOURCE_LIMIT`，不得返回物理无路。

### 4.4 目标容差与局部落区

局部 L0 认证把目标分成两个独立条件：

- `aim_position_on_surface_m` 必须位于 `position_tolerance_m` 内；
- `landing_region` 多边形面积必须不小于 `minimum_landing_region_area_m2`。

落区多边形无需完全落在目标容差圆内。其每个单元仍须满足地形、平面、净空和不确定性检查，
实际落点必须位于多边形内部。外部正例筛选器和 `hop_checks.py` 使用同一语义。

## 5. 真实起点和精确终端

### 5.1 轮式虚拟起点

轮式格点图增加唯一的虚拟起点状态。它保存真实位置和真实 yaw，不由 `CellCenter()` 或 yaw
bin 重新构造。第一批运动原语直接从该真实状态应用，所得转移必须通过现有
`WheelSweepValidator`。通过第一条转移后，目标状态才进入格心和 yaw bin 表示。

如果当前状态已经满足局部目标，静止轨迹直接使用真实状态。

### 5.2 足式虚拟起点

足式格点图同样增加真实身体位置、真实 yaw 和真实身体高度的虚拟起点。第一条身体运动原语
必须使用真实状态及当前可达高度区间，并通过 `ValidateLeggedBodySweep`。不得直接把身体移动到
格心或高度区间中点。

### 5.3 全局预览与终端

全局路线首个 Pose 使用冻结状态转换到 `map` 后的真实位置和朝向。首点到下一简化点的
supercover 必须安全；不能仅为视觉效果改写路径。

最终局部段优先生成到用户点击目标的连续终端连接。终端连接必须满足平台运动学、目标 yaw、
连续碰撞和目标容差；无法精确到达时只允许落在明确的容差范围内。所有输出均记录实际终端
误差，不允许修改请求中的容差。

## 6. 有界局部平滑

### 6.1 共同边界

平滑只消费已经搜索成功的当前局部离散段，不处理完整全局预览。默认边界为：

| 配置 | 默认值 |
|---|---:|
| `maximum_smoothing_control_points` | 64 |
| `maximum_smoothing_samples` | 512 |
| `maximum_iterations` | 128 |
| `maximum_trust_region_reductions` | 8 |
| 连续验证最大细分 | 32 |
| `require_smoothed_execution` | false |

控制点超过上限时，在保留真实起点、目标、方向切换点和走廊拐点的前提下确定性降采样。达到
资源上限时不得无界增加采样或迭代。

### 6.2 轮式

轮式按前进、倒车、原地旋转和停车切换点分段。每段采用夹持三次曲线，至少约束：

- 起点位置和切向与真实状态一致；
- 终点在局部目标容差内，最终段满足用户 yaw；
- 曲率不超过 `maximum_curvature_per_m`；
- 控制点位于已认证凸走廊内；
- 方向切换点速度为零。

### 6.3 足式

足式分别平滑身体平面位置、身体高度和 yaw：

- 平面位置二阶连续；
- yaw 一阶连续；
- 身体高度始终位于地形可达区间；
- 平移、垂向和 yaw 的速度、加速度满足能力文件；
- 允许横移，不把身体朝向强制等同于路径切线。

### 6.4 复检与回退

曲线按自适应间距采样，样本总数不超过 512。平滑后依次执行：

1. 有限值和 frame 检查；
2. 走廊包含检查；
3. 轮式足迹或足式身体连续扫掠；
4. 坡度、粗糙度、净空和地图边界检查；
5. 曲率、速度、加速度和时间单调性检查；
6. 真实起点和终端误差检查。

若平滑或复检失败：

- `require_smoothed_execution=false`：回退到从真实起点生成且已认证的离散运动原语，并记录
  稳定警告码；
- `require_smoothed_execution=true`：返回平台相关的平滑要求失败原因，不发布参考。

安全认证不得因性能优化而跳过。

## 7. 诊断语义

内部 `PlannerDiagnostics` 增加不进入 ROS Action schema 的局部轨迹诊断：

```text
trajectory_mode = STATIONARY | OPTIMIZED | DISCRETE_FALLBACK | CERTIFIED_HOP
start_anchor_error_m
endpoint_error_m
maximum_curvature_per_m
collision_validation = CERTIFIED | NOT_APPLICABLE
smoothing_elapsed_s
landing_field_elapsed_s
```

现有 Action `warning_codes` 保持稳定警告语义。`lunar_planner_ros` 把以上字段和逗号分隔的
警告码发布到标准 `/diagnostics`；仓库外控制器透传到交互状态，RViz 面板显示：

- 实际位置和 yaw 容差；
- 轨迹模式；
- 起点与终端误差；
- 最大曲率；
- 碰撞认证状态；
- 完整警告码。

不能通过“无警告”推断优化成功，必须使用 `trajectory_mode`。

## 8. 交互目标策略

仓库外包新增版本化 UTF-8 YAML，例如 `config/interactive_goal.yaml`：

```yaml
schema_version: lunar-interactive-goal-policy/v1
platforms:
  wheel:
    position_tolerance_m: 0.20
    yaw_tolerance_rad: 0.2617993877991494
  legged:
    position_tolerance_m: 0.20
    yaw_tolerance_rad: 0.2617993877991494
  hopper:
    position_tolerance_m: 0.20
    yaw_tolerance_rad: 0.2617993877991494
```

YAML 必须键集合精确、数值有限且位置容差大于零。节点启动时一次加载并冻结；命令行显式传入
路径。配置小于地图半格误差时仍可使用，但成功依赖连续终端或连续落点认证，规划器不得静默
把容差扩大到格心误差。

## 9. RViz 地图和平台层

### 9.1 完整障碍

`/lunar_isaac_validation/global_surface` 仍为一个包含全部有效格心的 PointCloud2：

- 普通地形按高程使用灰度；
- `obstacle` 使用红色，z 为 `elevation + obstacle_height`；
- `forbidden` 使用紫色并优先于 obstacle；
- 无效或非有限单元不发布。

`/lunar_isaac_validation/local_hazards` 保留当前平台 L0 窗口，使用更高亮颜色和更大的 RViz
点尺寸。全局点云在桥会话建立时生成一次，后续只更新时间戳并重新发布，不进入每次
`PlanMotion` Action 的关键路径。

### 9.2 当前平台

新增 RViz 专用 `/lunar_isaac_validation/platform_state` MarkerArray，独立于
`local_execution`：

- 使用包内代理 mesh 和实际物理尺寸；
- 显示不改变碰撞尺寸的高对比轮廓或光环；
- 显示朝向箭头和平台类型文字；
- 显示真实起点圆环；
- 平台切换时使用稳定 namespace/id 原子清空旧 Marker；
- 任何时刻只显示当前平台。

Marker 仅为显示证据；平台碰撞仍由能力文件和规划器连续扫掠决定。

## 10. 性能和资源门槛

新增逻辑不得改变以下既有门槛：

| 全局栅格数 | Ubuntu 全局搜索 p95 | AGX 全局搜索 p95 | Ubuntu 完整 Action p95 | AGX 完整 Action p95 |
|---:|---:|---:|---:|---:|
| ≤65,536 | 0.5 s | 1.0 s | 2.0 s | 3.0 s |
| ≤262,144 | 1.0 s | 2.0 s | 3.0 s | 4.0 s |
| ≤1,048,576 | 2.0 s | 4.0 s | 4.0 s | 6.0 s |

全局落区支持域必须为 O(N)；局部平滑由控制点、采样点、迭代和连续细分上限共同限定。性能
失败时按以下顺序处理：

1. 复用同一请求内已经生成的安全投影和距离场；
2. 减少非关键控制点；
3. 使用曲率和障碍距离驱动的自适应采样；
4. 在允许离散回退时使用已认证离散轨迹。

不得降低落区、碰撞、飞行管或 frame/时间一致性检查。不得使用受主机瞬时负载影响的内部
墙钟超时改变规划结论。

## 11. 稳定失败和警告

新增或明确使用以下语义：

| 条件 | 类型 | 代码 |
|---|---|---|
| 起点单元安全但支持面积不足 | 失败 | `HOPPER_START_REGION_AREA_INSUFFICIENT` |
| 目标容差内没有合格落点中心 | 失败 | `HOPPER_GLOBAL_GOAL_INFEASIBLE` |
| 跳跃候选发现达到资源上限 | 失败 | `HOPPER_GLOBAL_ROUTE_RESOURCE_LIMIT` |
| 轮式真实起点无法连接到格点图 | 失败 | `WHEEL_START_CONNECTOR_INFEASIBLE` |
| 足式真实起点无法连接到格点图 | 失败 | `LEGGED_START_CONNECTOR_INFEASIBLE` |
| 要求平滑但轮式平滑或复检失败 | 失败 | `WHEEL_SMOOTHED_EXECUTION_REQUIRED` |
| 要求平滑但足式平滑或复检失败 | 失败 | `LEGGED_SMOOTHED_EXECUTION_REQUIRED` |
| 轮式回退到离散轨迹 | 警告 | `WHEEL_OPTIMIZATION_DISCRETE_FALLBACK` 或现有更具体代码 |
| 足式回退到离散轨迹 | 警告 | `LEGGED_OPTIMIZATION_DISCRETE_FALLBACK` 或现有更具体代码 |

已有更具体的飞行管、弹道、走廊和 sweep 原因码优先保留。

## 12. 测试与验收

### 12.1 主仓单元测试

- 0.2 m 空旷图、1.327322 m² 要求能够生成跳跃落区；不足面积和被障碍切断的区域被拒绝；
- 相同输入的落区掩码、候选节点、路线和警告顺序完全一致；
- 目标容差 0.05 m 时只要求落点中心位于容差内，落区边界允许在容差外；
- 全局落区预处理检查取消和内存/节点资源上限；
- 轮式和足式轨迹首点与真实状态位置、朝向一致；
- 首段不安全时失败，不产生瞬移连接；
- 平滑轨迹满足曲率、速度、加速度和连续扫掠；
- 构造优化失败时验证离散回退与 `require_smoothed_execution` 两种结果；
- 标准诊断完整发布轨迹模式、误差、曲率、认证和阶段耗时。

### 12.2 仓库外测试

- 目标策略 YAML 的正确加载、缺键、多键、非有限值和非正容差；
- `global_surface` 中红色/紫色点数与全局 obstacle/forbidden 分类一致；
- `local_hazards` 仍只包含当前局部窗口；
- `platform_state` 包含 mesh、轮廓、朝向、标签和真实起点，平台切换后无旧 Marker；
- RViz 面板显示精确容差、`OPTIMIZED`、`DISCRETE_FALLBACK`、误差和警告码；
- 保持现有 PointCloud2 RViz 启动和安全退出回归。

### 12.3 集成与性能

- 使用独立确定性 fixture 验证三平台远距离正例和物理反例；
- 跳跃式至少覆盖多跳成功、落区面积不足、目标中心不合格、飞行管碰撞和资源不足；
- 50 m × 50 m、0.2 m 随机交互地图不因本修复改变随机生成或强制连通语义；
- 每个性能档预热后至少执行 30 次，报告 p50、p95、最大值、阶段耗时、展开状态和峰值内存；
- 62,500 栅格 Ubuntu 完整 Action p95 仍须不超过 2.0 s；
- AGX 指标必须在设备上测量，Ubuntu 结果不得冒充设备验收。

## 13. 完成条件

以下条件全部满足才可声明完成：

1. 跳跃式不再因单格面积比较而确定性拒绝所有 L0 候选。
2. 三平台默认交互位置容差为 0.20 m，且请求、Marker、面板和终端误差一致。
3. 轮式和足式局部轨迹从真实状态开始，首段经过连续认证。
4. 平滑结果或离散回退状态在 RViz 中明确可见，显示路线都具备碰撞认证证据。
5. 完整全局障碍、局部危险和当前平台可以同时区分。
6. ROS Action 和 MotionReference schema 未改变，生产内核未引入 Nav2 或 Python fallback。
7. 主仓、仓库外工具、集成测试和性能门槛全部通过。
