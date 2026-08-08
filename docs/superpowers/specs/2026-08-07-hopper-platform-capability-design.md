# 飞跃式平台单跳落区规划与能力基线设计

> **历史设计，燃料部分已取代（2026-08-08）：** 实时总质量、实时剩余燃料、认证燃料需求和
> 预计剩余燃料不再属于本项目活动合同。质量、比冲和参考推进剂只定义每次相同的固定单跳 Δv
> 包络。落区、抛物线和飞行管内容继续有效；现行解释见
> `2026-08-08-formal-capability-and-hopper-no-fuel-budget-design.md`。

**状态：** 已完成逐段设计确认，等待书面规格审阅

**适用平台：** `HOPPER`

**能力版本建议：** `hopper-engineering-baseline-v1`
**来源等级：** `project_engineering_baseline`，不是厂家实测或飞行鉴定值

## 1. 目标与范围

本设计把飞跃式平台从现有的“固定速度/冲量上限、多候选落区、多跳图搜索”模型，收敛为
一个职责明确的单跳认证器：上游探索决策网络或 RViz 给出一个精确目标，本项目判断该目标及
其周围凸落区是否具有足够地形证据、是否可由当前燃料到达，以及是否存在无碰撞的名义抛物线。

每个成功请求只授权一次飞跃，并输出：

- 包含精确目标的安全凸落区；
- 精确名义落点；
- 名义起飞速度、飞行时间、抛物线和飞行管；
- 理想燃料需求、含裕量认证燃料需求和预计剩余燃料；
- 能力版本、全局地图版本、L0 地图版本和认证诊断。

本项目不生成发动机推力、控制力矩、姿态轨迹或实际软着陆制导，不接管下游飞行控制，也不
规划一次请求中的多跳链。名义抛物线只用于可达性和碰撞认证，不是执行器指令。

## 2. 决策依据与来源边界

### 2.1 用户提供的控制模型资料

`/home/kai/文档/feiyue输入输出.md` 给出了质点和刚体动力学、质量消耗关系以及下游控制器的
输入输出。该资料证明质量和比冲会影响飞跃能力，同时明确实际飞行控制需要位置、速度、质量、
姿态、惯量、推力扰动等更多状态。本规划项目只消费其中与简化可达性有关的质量和比冲关系；
推力、力矩、姿态和扰动属于下游控制单位。

### 2.2 论文参数的使用方式

陈上上、关轶峰、黄翔宇《多约束月面盘旋飞跃轨迹优化控制方法》，
DOI `10.15982/j.issn.2096-9287.2024.20230005` 的算例使用约 `348.9 kg` 质量和
`2951.5 N·s/kg` 有效排气速度，后者约等于 `300.97 s` 常规比冲。该论文研究的是大型动力
盘旋飞跃器；本设计只把约 `301 s` 用作推进系统工程锚点，不采用 `348.9 kg` 质量。

### 2.3 小型平台质量锚点

已批准的 Quad48 足式平台设计 `legged-platform-capability-design@bb4327d` 记录 URDF 总质量
`15.89 kg`、标称载荷约 `8 kg`，对应约 `23.89 kg` 标称带载质量。飞跃式平台因此按同一
数量级设计，选择 `20 kg` 整机湿质量作为百米参考工况。

历史 Isaac Sim 代理也曾使用 `20 kg`，但代理资料是可视化与碰撞测试材料，不能作为正式
能力证据。本设计中的 `20 kg` 是用户重新批准、经百米燃料反推后的项目工程基线；二者来源
等级必须分别记录，禁止用历史代理文件替代本设计的正式能力资料。

## 3. 职责边界

### 3.1 本项目负责

- 接收当前平台位置、实时总质量、实时可用剩余燃料和精确目标；
- 从正式静态能力资料读取比冲、支撑半径、飞行碰撞半径和着陆地形限制；
- 使用静止起飞、静止着陆的双理想脉冲加无动力抛物线模型认证燃料可达性；
- 使用保守多分辨率全局地图认证飞行管；
- 使用 `0.2 m` L0 地图认证精确目标和完整凸落区；
- 输出一个且仅一个 `HopSegment`，并维持已承诺/飞行中的不可替换保护。

### 3.2 本项目不负责

- 推力时序、推力上限、发动机点火和关机过程；
- 姿态角、角速度、控制力矩、惯量和推力扰动；
- 真实软着陆轨迹跟踪；
- 一次请求中的多跳搜索、落点图、下一跳晋升或续跳状态；
- 为不安全或不可达目标自动吸附、放宽容差或寻找替代目标；
- 为推进剂保留着陆或应急储备。运行时给出的全部可用剩余燃料均可参与本跳认证。

下游飞行控制可根据真实执行器约束重新生成轨迹，但不得把未经本项目认证的落点或飞行走廊
冒充为本项目输出。若实际控制轨迹超出认证飞行管或落区，必须重新申请认证。

## 4. 正式工程能力基线

### 4.1 静态平台能力

| 字段 | 冻结值 | 来源等级 | 规划语义 |
|---|---:|---|---|
| `specific_impulse_s` | `301.0 s` | `project_engineering_baseline` | 火箭方程使用的常规比冲 |
| `landing_support_radius_m` | `0.45 m` | `project_engineering_baseline` | 完整圆形支撑足迹半径 |
| `flight_collision_radius_m` | `0.55 m` | `project_engineering_baseline` | 不含地图裕量的全姿态包络球半径 |
| `maximum_landing_slope_rad` | `0.174533 rad` | `planning_safety_baseline` | 约 `10°`，超过时硬拒绝 |
| `maximum_landing_plane_residual_m` | `0.05 m` | `planning_safety_baseline` | 最佳拟合平面的最大绝对残差 |

能力资料必须同时携带 `platform_id`、`platform_type=HOPPER`、唯一递增的
`capability_version`、`base_frame_id`、几何来源和来源等级。数值必须为有限正数，坡度必须
位于 `[0, π/2)`。

### 4.2 规划安全策略

| 字段 | 冻结值 | 语义 |
|---|---:|---|
| `landing_lateral_margin_m` | `0.20 m` | 一格 L0 横向安全裕量 |
| `flight_map_margin_m` | `0.20 m` | 飞行管地图与模型裕量 |
| `reachability_delta_v_margin_ratio` | `0.10` | 所需总 Δv 增加 `10%` |
| `gravity_mps2` | `[0, 0, -1.62]` | 月面环境参数，不是平台能力 |
| `standard_gravity_mps2` | `9.80665` | 比冲秒与有效排气速度换算常数 |

因此落区完整检查半径固定为 `0.45 + 0.20 = 0.65 m`，飞行管认证半径固定为
`0.55 + 0.20 = 0.75 m`。地图裕量与实体几何必须分别保留，不能预先合并后又重复膨胀。

### 4.3 参考工况

以下字段只进入能力资料的 `reference_conditions` 和测试 fixture，不得成为运行时回退值：

| 字段 | 值 |
|---|---:|
| `reference_total_mass_kg` | `20.0 kg` |
| `reference_remaining_usable_fuel_mass_kg` | `0.20 kg` |
| `reference_horizontal_range_m` | `100.0 m` |
| `reference_elevation_delta_m` | `0.0 m` |

规划请求缺少实时质量或实时燃料时必须失败，不能静默使用参考工况。

## 5. 运行时输入接口

### 5.1 独立推进剂状态 Topic

新增外部所有的暂定接口：

```text
topic: /platform/hopper_propellant_state
type: lunar_navigation_msgs/msg/HopperPropellantState
owner: external propulsion or flight-control system
qos: reliable, volatile, depth 10
```

暂定消息字段为：

```text
std_msgs/Header header
string platform_id
string capability_version
float64 total_mass_kg
float64 remaining_usable_fuel_mass_kg
```

字段语义如下：

- `header.stamp` 非零且不得来自未来；`header.frame_id` 等于当前能力资料的 `base_frame_id`；
- `platform_id` 和 `capability_version` 必须与本次 Action 及已加载能力完全一致；
- `total_mass_kg` 是包含当前可用燃料和载荷的当前整机质量；
- `remaining_usable_fuel_mass_kg` 是上游已扣除自身不可用余量后允许本项目使用的燃料；
- 两个质量均为有限值，且 `0 < remaining_usable_fuel_mass_kg < total_mass_kg`。

ROS 快照策略新增显式 `propellant_state_max_age=0.5 s`；推进剂状态还必须与里程计、L0 地图
和 TF 满足现有 `max_pairwise_skew=0.25 s`。缺失、过期、平台不匹配或能力版本不匹配时，
不得调用 C++ 规划核心。与其余暂定 `lunar_navigation_msgs` schema 一样，上游正式定义出现后
必须逐字段对比并原子切换，不能让两个同名 provider 共存。

### 5.2 Action 目标

飞跃式只接受：

- `GoalRegion.goal_type=POINT`；
- `position_tolerance_m=0.0`；
- `has_yaw_constraint=false`；
- 有限、位于地图范围内的精确 `x/y`。

目标 `x/y` 是权威水平坐标，不得吸附到栅格中心或邻近安全点。输入 `z` 只要求有限；规划器
必须从绑定的 L0 地形快照取得该精确 `x/y` 的着陆高程，并在输出中记录地图派生 `z`。
`PLANAR_REGION`、非零位置容差或偏航约束对飞跃式均为非法请求。

### 5.3 静止起飞语义

里程计仍用于确定真实发射位置和执行状态，但弹道求解器不把当前线速度或角速度代入模型。
新飞跃只能在协调器确认 `GROUND_HOLD` 后授权；已承诺或飞行中的请求继续由现有引用保护拒绝
普通替换。着陆稳定等待属于执行协调策略，`minimum_settle_guard` 不再是平台物理能力字段，
本设计不改变协调器当前采用的稳定等待策略。

## 6. 简化弹道与燃料认证

### 6.1 双理想脉冲模型

设发射位置为 `r0`、精确目标为 `r1`、位移为 `Δr=r1-r0`、月面重力向量为 `g`、无动力
飞行时间为 `T>0`。静止起飞后所需名义发射速度为：

\[
v_0(T)=\frac{\Delta r}{T}-\frac{1}{2}gT
\]

无动力段末速度为：

\[
v_f(T)=v_0(T)+gT
\]

为了表达静止起飞和静止着陆，理想总速度增量为：

\[
\Delta v_{ideal}(T)=\|v_0(T)\|+\|v_f(T)\|
\]

认证总速度增量为：

\[
\Delta v_{cert}(T)=1.10\,\Delta v_{ideal}(T)
\]

该模型把真实动力上升和动力减速近似为两个瞬时脉冲，中间为无动力抛物线。它比只计算
起飞速度更保守，但仍不等价于真实推力受限轨迹。

### 6.2 火箭方程

对当前总质量 `m0`、当前可用燃料 `mf`、比冲 `Isp`：

\[
\Delta v_{available}=I_{sp}g_0\ln\frac{m_0}{m_0-m_f}
\]

某飞行时间可认证的必要条件是：

\[
\Delta v_{cert}(T)\leq\Delta v_{available}
\]

对应燃料输出为：

\[
m_{ideal}=m_0\left(1-e^{-\Delta v_{ideal}/(I_{sp}g_0)}\right)
\]

\[
m_{cert}=m_0\left(1-e^{-\Delta v_{cert}/(I_{sp}g_0)}\right)
\]

\[
m_{remaining}=m_f-m_{cert}
\]

计算必须使用至少 `double` 精度，并在对数、指数和质量差附近进行有限性与定义域检查。

### 6.3 百米参考结果

在同高差、`g=1.62 m/s²`、水平距离 `100 m` 时，最低发射速度的抛物线为：

- 发射仰角 `45°`；
- 发射速度 `12.727922 m/s`；
- 飞行时间 `11.111111 s`；
- 最高点相对发射面 `25.0 m`；
- 静止起飞与静止着陆合计理想 Δv `25.455844 m/s`；
- 加 `10%` 后认证 Δv `28.001429 m/s`。

对 `20 kg`、`301 s`，认证燃料约 `0.188827 kg`。参考工况取 `0.20 kg`，理想同高差
能力约 `135.8 m`，计入 `10%` Δv 裕量后的认证边界约 `112.2 m`，因此百米目标仍有
明确裕量。`100 m` 只是参数标定点，不是
`maximum_hop_distance_m`；任意目标都必须按实时质量、燃料、高差和地图重新认证。

## 7. 飞行时间搜索与飞行管认证

### 7.1 完整燃料可行时间集合

规划器先求出全部满足 `Δv_cert(T) <= Δv_available` 的飞行时间集合，再在整个集合内寻找
耗油最少且无碰撞的抛物线。不得只检查最低能耗时间，也不得使用固定候选时间数组。

搜索采用确定性的自适应区间分支定界：

1. 使用保守数值界确定燃料可行时间区间；
2. 为每个时间子区间计算燃料下界和该区间所有抛物线的保守空间包络；
3. 包络与膨胀障碍明确相交时淘汰该区间；
4. 包络完全安全且燃料下界不优于当前解时剪枝；
5. 其余区间继续细分，直到可以证明安全、碰撞或数值不确定；
6. 在全部已证明安全的区间中选择认证燃料最少者，以飞行时间和坐标字典序打破并列。

没有候选数、开放列表数、认证次数或飞行时间采样数上限。地图有限边界、有限燃料可行时间
集合、活动地图分辨率和浮点数值精度构成自然边界。区间空间包络缩小到活动地图分辨率的
四分之一后仍无法证明安全或碰撞时，该区间标为数值不确定；若不存在已认证解但仍有这种
不确定区间，返回 `NUMERICAL_FAILURE`，不能报告不可达。

### 7.2 飞行管

名义轨迹按：

\[
r(t)=r_0+v_0t+\frac{1}{2}gt^2,\qquad 0\leq t\leq T
\]

重建。认证使用 `0.75 m` 半径的连续飞行管，不能只检查固定数量的离散点。远离着陆点的
飞行管可使用按既定规则生成的保守多分辨率全局图；任一未知、禁入或可能占据的单元均不能
证明安全。目标附近的落区认证必须切换到 `0.2 m` L0 地图。

## 8. 精确目标与安全凸落区

### 8.1 精确目标硬检查

以精确目标 `x/y` 为圆心检查半径 `0.65 m` 的完整闭圆盘。圆盘必须：

- 全部位于同一新鲜 L0 快照的有效范围内；
- 不包含未知、无数据、障碍、禁入区或岩石占据；
- 使用圆盘内全部 L0 高程样本拟合平面；
- 最佳拟合平面坡度不超过 `0.174533 rad`；
- 相对该平面的最大绝对残差不超过 `0.05 m`；
- 对该精确点至少存在一条燃料可行且飞行管无碰撞的抛物线。

通用粗糙度和高程方差继续进入诊断与 RViz 着色，但不单独作为飞跃式硬拒绝条件。目标本身
任何硬检查失败时直接失败，不搜索替代点，不改变精确 `x/y`。

### 8.2 强保证凸区域

输出落区必须满足强保证：多边形内每一个可能着陆质心点都同时具备完整支撑圆地形安全、
燃料可达性和至少一条可认证飞行管。不能只验证顶点、栅格中心或名义点。

构造过程为：

1. 对目标附近 L0 单元进行保守区间认证，只有整个单元内任一点都满足强保证才标为认证单元；
2. 目标所在粗单元不能整体认证时，以目标为种子自适应细分；
3. 无法得到包含目标的正面积认证子区域时返回 `LANDING_REGION_NOT_CERTIFIABLE`；
4. 从种子区域按距目标、栅格坐标的固定顺序向外增长；
5. 每次把新单元并入凸包前，重新检查凸包覆盖的全部单元和连续边界；
6. 凸包跨越岩石、陨石坑、未知孔洞、不可达区域或未认证飞行走廊时拒绝该次扩张；
7. 重复到没有可接受扩张，输出逆时针、无重复、至少三个不共线顶点的凸多边形。

不设置 `minimum_landing_region_area_m2`。区域只需具有由地图证据支持的正面积。名义落点始终
等于精确目标，不替换为多边形质心。

## 9. 输出接口与执行语义

### 9.1 单段输出

为减少无关 wire break，`MotionReference.hops[]` 暂时保留数组类型，但成功的飞跃式输出长度
必须严格等于 `1`。零段或多段输出均在 ROS 转换边界拒绝。数组不再表达多跳链。

现有 `HopSegment` 保留：

- `segment_id`；
- `launch_pose`；
- `landing_region`；
- `flight_time`；
- `launch_velocity`；
- `flight_tube_radius_m`。

新增：

```text
geometry_msgs/Point nominal_landing_point
float64 ideal_fuel_required_kg
float64 certified_fuel_required_kg
float64 expected_remaining_usable_fuel_kg
float64 required_delta_v_mps
float64 available_delta_v_mps
string capability_version
uint64 global_map_generation
uint64 local_map_generation
```

`required_delta_v_mps` 表示含 `10%` 裕量的认证值。`launch_velocity + flight_time + gravity`
足以重建名义抛物线，无需发布固定采样点数组。

### 9.2 承诺保护

一次飞跃被激活后，现有 `ReferenceGuard` 继续保护该单段引用：已承诺和飞行状态拒绝普通取消
或替换；着陆后必须结合身份匹配的 `MotionExecutionFeedback` 与新鲜 Odometry 判断稳定。由于
没有续跳，同一 Action 不生成 promotion region、route continuation 或下一段。新目标必须由
新的 Action 请求发起。

## 10. 旧字段与旧路径退役

### 10.1 `HopperCapability` 迁移

| 旧字段 | 新处理 |
|---|---|
| `body_half_extent_m` | 用 `flight_collision_radius_m` 和正式几何来源替代 |
| `platform_mass_kg` | 改为实时推进剂状态；参考质量只进 `reference_conditions` |
| `gravity_mps2` | 移到月面环境/规划安全策略 |
| `maximum_landing_roughness_m` | 取消硬门限，粗糙度仅诊断 |
| `maximum_plane_residual_m` | 重命名为 `maximum_landing_plane_residual_m` |
| `minimum_overhead_clearance_m` | 由飞行碰撞半径和地图裕量统一表达 |
| `minimum_lateral_clearance_m` | 由支撑半径、飞行碰撞半径及各自地图裕量表达 |
| `minimum_landing_region_area_m2` | 删除 |
| `maximum_launch_speed_mps` | 删除 |
| `maximum_launch_impulse_newton_seconds` | 删除 |
| `minimum_flight_time` / `maximum_flight_time` | 删除，由燃料可行集合自然确定 |
| 着陆速度、冲击速度字段 | 删除，真实软着陆由下游控制负责 |
| 角速度、角加速度、初始角速度字段 | 删除，姿态不属于本规划模型 |
| `minimum_landing_clearance_m` | 删除，由完整支撑圆规则替代 |
| `minimum_settle_guard` | 移到执行协调策略，不作为平台物理能力 |
| `actuator_or_impulse_profile`、飞跃 motion primitives | 从规划能力删除，实际执行器属于下游控制 |

旧 YAML 中出现这些字段时必须明确报 schema 版本不兼容，不能悄悄忽略并继续运行。

### 10.2 搜索路径迁移

以下生产语义退役：

- hopper 多跳全局图、落点索引图和 route cursor；
- `maximum_authorized_hops` 配置；
- `maximum_landing_regions`、`maximum_graph_nodes`、`maximum_graph_out_degree`；
- 固定名义瞄准点数、认证次数和飞行管段数；
- 下一跳 promotion、continuation 和自动滚动授权；
- `HOPPER_GLOBAL_ROUTE_RESOURCE_LIMIT` 等固定资源上限错误码。

飞跃成功路径是“精确目标认证 → 单条抛物线认证 → 单段输出”，不再先构建全局多跳路线。

## 11. 失败语义

| 情况 | `PlanningOutcome` | `reason_code` |
|---|---|---|
| 不是精确点、容差非零或带偏航约束 | `INVALID_REQUEST` | `HOPPER_EXACT_POINT_REQUIRED` |
| 推进剂状态缺失、平台或版本不匹配 | `INVALID_REQUEST` | `HOPPER_PROPELLANT_STATE_INVALID` |
| 推进剂状态过期或快照时差过大 | `STALE_INPUT` | `HOPPER_PROPELLANT_STATE_STALE` |
| 总质量非法 | `INVALID_REQUEST` | `HOPPER_TOTAL_MASS_INVALID` |
| 可用燃料非法 | `INVALID_REQUEST` | `HOPPER_USABLE_FUEL_INVALID` |
| 比冲或静态能力非法 | `INVALID_REQUEST` | `HOPPER_CAPABILITY_INVALID` |
| 精确目标缺少完整 L0 证据 | `GOAL_INFEASIBLE` | `LANDING_EVIDENCE_INSUFFICIENT` |
| 目标支撑圆与障碍/禁区相交 | `GOAL_INFEASIBLE` | `HOPPER_LANDING_TARGET_OCCUPIED` |
| 着陆坡度超限 | `GOAL_INFEASIBLE` | `HOPPER_LANDING_SLOPE_EXCEEDED` |
| 着陆平面残差超限 | `GOAL_INFEASIBLE` | `HOPPER_LANDING_PLANE_RESIDUAL_EXCEEDED` |
| 当前燃料不足以到达精确目标 | `GOAL_INFEASIBLE` | `HOPPER_FUEL_INSUFFICIENT` |
| 无法形成正面积强保证凸区域 | `GOAL_INFEASIBLE` | `LANDING_REGION_NOT_CERTIFIABLE` |
| 全部燃料可行飞行管均已证明碰撞 | `NO_KNOWN_SAFE_ROUTE` | `HOPPER_ALL_FLIGHT_TUBES_BLOCKED` |
| 抛物线数值无法确定 | `NUMERICAL_FAILURE` | `HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE` |
| 飞行管数值无法确定 | `NUMERICAL_FAILURE` | `HOPPER_FLIGHT_TUBE_NUMERICAL_INDETERMINATE` |
| 凸落区数值无法确定 | `NUMERICAL_FAILURE` | `HOPPER_LANDING_REGION_NUMERICAL_INDETERMINATE` |
| stop token 或客户端取消 | `CANCELED` | `REQUEST_CANCELED` |
| 客户端截止时间后取消 | `CANCELED` | `PLANNING_DEADLINE_CANCELED` |
| 真实内存分配失败 | `RESOURCE_EXHAUSTED` | `HOPPER_SEARCH_ALLOCATION_FAILED` |

取消、超时、分配失败、输入陈旧和数值不确定均表示搜索未完成，绝不能转换为燃料不足、目标
不可行或无安全路线。只有整个燃料可行时间集合都得到确定认证结论后，才能报告全部飞行管阻塞。

## 12. RViz 证据

RViz 左侧平台选择继续只显示当前平台。飞跃式至少发布并标注：

- 按正式几何比例绘制的当前飞跃平台；
- 上游精确目标点；
- 半径 `0.65 m` 的完整着陆检查圆；
- 半透明填充并带轮廓的安全凸落区；
- 与目标重合、颜色独立的精确名义落点；
- 名义抛物线；
- 直径 `1.50 m` 的认证飞行管；
- 完整全局障碍、局部危险、禁入区和未知证据；
- `ideal/certified/available fuel`、`required/available Δv`、地图/能力版本和最终错误码。

粗糙度只使用诊断色图，不得复用硬障碍颜色。正例必须同时看见平台、目标、落区、抛物线和
飞行管；负例必须保留目标与失败证据，不能只显示“规划失败”文字。

## 13. 验证策略

### 13.1 数学与单元测试

- 固定验证百米同高差参考数值；
- 覆盖正负高差、近距离、燃料边界和质量/燃料非法值；
- 验证可用燃料增加不会降低可用 Δv，总质量增加不会提高同燃料可达性；
- 验证理想燃料、认证燃料和预计剩余燃料的内部一致性；
- 验证整个燃料可行时间集合被覆盖，固定采样之间的安全弹道不会漏检；
- 构造最低能耗弹道碰撞、另一飞行时间安全的正例；
- 构造所有可行弹道均碰撞的完整反例；
- 验证取消和数值不确定不发布引用。

### 13.2 落区性质测试

- 支撑圆边缘岩石、禁入区、未知单元和地图边界分别拒绝；
- `10°` 坡度和 `0.05 m` 残差的边界内外测试；
- 高粗糙度但坡度、残差和障碍检查合格时仍通过；
- 目标处于安全区边缘时不吸附、不改目标；
- 凹形安全区和带危险孔洞区域不得被普通凸包跨越；
- 对随机地图输出多边形内的顶点、边、栅格交点和随机内点逐点复验强保证；
- 同一输入重复运行必须得到逐顶点相同的落区和相同名义弹道。

### 13.3 ROS 接口测试

- 推进剂 Topic 缺失、过期、未来时间戳、平台不匹配和能力版本不匹配；
- 参考质量/燃料不能在运行时回退；
- `POINT + tolerance=0` 成功，区域目标、非零容差和偏航约束失败；
- 目标 `x/y` 保持精确，`z` 来自绑定 L0 快照；
- 成功结果严格包含一个 `HopSegment`，零个和多个均在转换边界拒绝；
- 新增燃料、Δv 和版本字段无损转换；
- 已承诺/飞行中的单跳继续受引用保护，着陆后新目标必须新建 Action。

### 13.4 Release 性能回归

性能测试必须强制使用 `Release` 构建。时间阈值是回归验收条件，不是生产搜索停止条件：

| 场景 | 目标时间 |
|---|---:|
| `100 m` 平坦直接正例 | `≤ 1.0 s` |
| 最低能耗弹道受阻、存在另一安全弹道 | `≤ 2.0 s` |
| 所有燃料可行弹道均受阻的完整反例 | `≤ 5.0 s` |

超过阈值使性能回归失败并记录阶段耗时，但生产规划器仍继续搜索，直到自然边界耗尽、客户端
取消、外部截止时间取消、真实分配失败或数值无法确定。

## 14. 完成条件

实现只有同时满足以下条件才算完成：

1. 正式能力 schema、动态推进剂状态和单段输出字段全部实现并校验；
2. 旧固定速度/冲量/时间窗/多跳字段在新 schema 中被明确拒绝；
3. 单跳燃料可行集合、飞行管和强保证凸落区测试全部通过；
4. ROS 快照、Action、引用保护和 RViz 正反例通过；
5. Release 性能回归满足目标，且没有生产搜索资源上限；
6. 仓库边界检查、ROS/C++ 测试和 UTF-8 文档检查通过；
7. 能力资料记录 `project_engineering_baseline` 来源，未把论文大型质量或历史代理写成实测值。

本设计文档本身不解锁正式 PPO 训练。只有实现通过、三平台能力资料共同完成 schema 校验和
正式 capability freeze 后，才能据此重新生成训练缓存并进入正式训练门禁。
