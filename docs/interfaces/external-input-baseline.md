# 外部输入接收基线

> **这是本项目暂定消息字段与语义的唯一权威基线。** `lunar_navigation_msgs` 上游尚未定义期间，本仓 `.msg`、配置和检查器必须与本文一致；Topic 数据生产者仍由外部项目拥有。未来切换上游必须执行固定版本、schema 对比和原子替换，不能叠加同名包。

当前接收合同版本为 `lunar-external-interfaces/v5`。

旧 checkpoint 的接口联调实现、固定 YAML 替换方式和启动顺序见
[`interface-v1-quickstart.md`](../deployment/interface-v1-quickstart.md)。该实现严格绑定 fed9/step 251
旧定义，资格为 `integration_only`；它不会覆盖本文件后续版本的接口所有权，也不得被当作新语义模型。

## 来源与所有权

- 外部/legacy 交接来源（不复制入本仓）：`课题四未知场景无人平台自主探索与规划外部输入.md`
- SHA-256：`a4c2db0a6647d59fa7cee5cf8048d18f7bca7c7b11a591a32d33d237c0c06e78`
- `grid_map_msgs`、`nav_msgs` 和 `tf2_msgs` 仍来自 ROS/外部系统；定位与任务系统仍负责消息数据的发布和演进协商。
- 本仓暂定提供五个 `lunar_navigation_msgs` schema；Topic 数据生产者仍由外部项目拥有。
- 本项目只声明依赖、直接订阅、适配和运行时校验；上游尚未定义时，本仓 `.msg`、配置和检查器以本文为准。未来切换必须固定唯一上游 tag 或 commit、执行 schema 对比并原子替换，不得与上游同名包共存。

## Topic 与接收字段

| 输入 | 类型 | 外部所有权 | 接收字段/约束 |
|---|---|---|---|
| `/environment/map_global` | `grid_map_msgs/msg/GridMap` | 外部地图融合系统 | `header`、`info`、`layers`、`basic_layers`、`data`、`outer_start_index`、`inner_start_index`；`frame_id=map`；发布满足下述资源规则的最精细二倍层级 |
| `/environment/map_local` | `grid_map_msgs/msg/GridMap` | 外部地图融合系统 | 同上；`frame_id=odom`；始终发布当前平台附近的 L0 窗口，不得复制整张全局图 |
| `/localization/odometry` | `nav_msgs/msg/Odometry` | 外部定位系统 | `header`、`child_frame_id`、`pose`、`twist`；`odom -> base_link` |
| `/localization/status` | `lunar_navigation_msgs/msg/LocalizationStatus` | 外部定位系统 | `header`、`status`；状态为 `UNKNOWN/VALID/DEGRADED/INVALID/RELOCALIZING` |
| `/tf` | `tf2_msgs/msg/TFMessage` | 外部 TF 发布者 | `transforms[]`，形成 `map -> odom -> base_link` |
| `/mission/exploration_task` | `lunar_navigation_msgs/msg/ExplorationTask` | 外部任务系统 | `header`、`mission_id`、`revision`、`desired_state`、ROI、`science_regions`；状态为 `ACTIVE/PAUSED/CANCELED` |
| `/execution/motion_feedback` | `lunar_navigation_msgs/msg/MotionExecutionFeedback` | 外部运动执行/控制系统 | `header`、`sequence`、`platform_type`、`plan_id`、`segment_id`、`state`、`reason_code`；状态为 `IDLE/ACCEPTED/EXECUTING/SEGMENT_COMPLETE/LANDED_HOLD/FAILED/CANCELED` |
| `science_regions[]` | `lunar_navigation_msgs/msg/ScienceTargetRegion` | 外部任务系统 | `region_id`、`objective_id`、`boundary`、`priority` |

旧 checkpoint 联调节点还提供一个明确的执行输出：

| 输出 | 类型 | 消费者 | 语义 |
|---|---|---|---|
| `/lunar/motion_reference` | `lunar_planning_msgs/msg/MotionReference` | 外部运动执行/控制系统 | 原样转发当前 `PlanMotion` 成功结果；`reliable + volatile + depth 1`，禁止执行器重启后重放旧命令 |

外部执行器只执行该 Topic 上最新收到的参考，并用同一 `plan_id/segment_id` 发布
`MotionExecutionFeedback`。Topic 可用 ROS remap 调整；若外部项目更换消息类型，应在外部适配器边界新增显式
converter，策略、规划器和旧模型合同内部仍保持上述类型，不能用 YAML 静默改变字段语义。

接口 YAML 只允许替换 Topic、类型、frame 和显式 converter，不拥有 QoS。内部通道的 DDS
语义固定为：地图 `reliable + transient_local + depth 1`，里程计采用 ROS 2 sensor-data
QoS，TF 为 `best_effort + volatile + depth 100`，任务为
`reliable + transient_local + depth 1`，定位状态与执行反馈均为
`reliable + volatile + depth 10`。更换外部项目时必须在 adapter 边界满足这些固定通道合同，
不能通过替换 YAML 悄悄改变消费者的 QoS。

## 暂定消息 schema

### `LocalizationStatus.msg`

```text
uint8 UNKNOWN=0
uint8 VALID=1
uint8 DEGRADED=2
uint8 INVALID=3
uint8 RELOCALIZING=4
std_msgs/Header header
uint8 status
```

### `ScienceTargetRegion.msg`

```text
string region_id
string objective_id
geometry_msgs/Polygon boundary
float64 priority
```

### `ExplorationTask.msg`

```text
uint8 ACTIVE=1
uint8 PAUSED=2
uint8 CANCELED=3
std_msgs/Header header
string mission_id
uint64 revision
uint8 desired_state
float64 roi_min_x_m
float64 roi_min_y_m
float64 roi_max_x_m
float64 roi_max_y_m
lunar_navigation_msgs/ScienceTargetRegion[<=64] science_regions
```

### `MotionExecutionFeedback.msg`

```text
uint8 WHEELED=1
uint8 LEGGED=2
uint8 HOPPER=3
uint8 IDLE=0
uint8 ACCEPTED=1
uint8 EXECUTING=2
uint8 SEGMENT_COMPLETE=3
uint8 LANDED_HOLD=4
uint8 FAILED=5
uint8 CANCELED=6
std_msgs/Header header
uint64 sequence
uint8 platform_type
string plan_id
string segment_id
uint8 state
string reason_code
```

### `HopperPropellantState.msg`（历史兼容，不是活动输入）

```text
std_msgs/Header header
string platform_id
string capability_version
float64 total_mass_kg
float64 remaining_usable_fuel_mass_kg
```

该消息仅为旧 Isaac/ROS 证据的源码兼容而继续生成和校验；v5 接收合同不声明对应 Topic，
规划节点不订阅、快照不冻结、训练桥不暴露该状态。新实现不得发布或消费它。

## 地图与定位字段

`GridMap` 接收 `header.stamp`（非零）、`header.frame_id`、`info.resolution`、`info.length_x`、`info.length_y`、`info.pose`、唯一的 `layers`、`basic_layers`、与层顺序一致的 `data`、`outer_start_index`、`inner_start_index`。本项目依赖的地图层为：

`elevation`、`valid_mask`、`obstacle`、`obstacle_height`、`observation_age_s`、`observation_quality`、`elevation_variance`、`obstacle_variance`、`observation_count`、`forbidden`。

### 多分辨率地图契约

地图金字塔契约版本为 `lunar-conservative-grid-aggregation/v1`。设最高精度规划分辨率为
`r0=0.2 m`，允许的尺度因子固定为：

```text
q = [1, 2, 4, 8, 16, 20]
r_l = r0 * q_l,  l in {0, 1, 2, 3, 4, 5}
```

即默认分辨率依次为 `0.2 m`、`0.4 m`、`0.8 m`、`1.6 m`、`3.2 m` 和 `4.0 m`。
L5 是为固定栅格预算增加的终端尺度，不把层级规则错误表述成无限二倍扩展。设地图物理宽高
为 `Sx`、`Sy`，目标单轴格数为 `T=min(target_axis_cells,
maximum_axis_cells)`，当前默认 `T=256`。外部地图生产方必须发布满足下式的最小层级，
也就是满足目标工作集和硬资源条件的最精细层级：

```text
ceil(Sx / r_l) <= T
ceil(Sy / r_l) <= T
ceil(Sx / r_l) * ceil(Sy / r_l) <= 1048576
max(ceil(Sx / r_l), ceil(Sy / r_l)) <= 4096
```

不得按任务名称或“50 m”“1 km”等特定范围选择固定分辨率。收到的全局图不是二倍层级、
比应选层级更粗，或比应选层级更细而超过资源上限时，消费者以
`GLOBAL_MAP_LEVEL_INVALID` 拒绝快照。L5 仍不满足条件时以
`GLOBAL_MAP_SCALE_UNSUPPORTED` 拒绝；不允许继续静默降低分辨率。局部图分辨率必须始终
等于 `r0`。例如当前 `1024 m × 1024 m`、`T=256` 的正式训练窗由同一公式得到
`256 × 256 @ 4.0 m`，该结果不是按“千米任务”名称特判。

局部图的空间语义按平台区分，但 Topic 和消息类型不变：

- 轮式、足式的 `local_map` 是以平台当前位置附近为中心的 L0 执行证据窗，随平台滚动更新；
- 飞跃式的 `local_map` 是覆盖本次精确目标及完整着陆支撑区域的 L0 着陆证据窗，不是以起点
  为中心的搜索窗口，也不得用窗口半径限制单跳距离；
- 飞跃式从起点到目标的完整抛物线飞行管始终使用同一冻结快照中的 `global_map` 认证。

飞跃式普通障碍按有限高度三维几何解释：单元水平范围取完整栅格方形，顶面为
`elevation + max(obstacle_height, resolution)`。飞行管下表面能证明高于顶面时允许从上方
穿越；相交时该候选抛物线无效，并可在固定单跳 Δv 包络与速度约束内继续尝试更高弹道。`forbidden` 在当前
schema 中仍是绝对禁入层；在正式拆分“禁止着陆”和“禁止进入空域”图层之前不得把它按普通
有限高度障碍放行。

每个父单元由最多四个 L(l-1) 子单元按下表保守聚合；奇数边界缺少的子单元视为无效且
禁入，不能视为自由空间：

| 图层 | `lunar-conservative-grid-aggregation/v1` 规则 |
|---|---|
| `valid_mask` | 所有四个子单元有效才为 1（AND） |
| `obstacle` | 任一子单元为 1 即为 1（OR） |
| `forbidden` | 任一子单元为 1 即为 1（OR） |
| `obstacle_height` | 四个子单元的最大值 |
| `observation_age_s` | 四个子单元的最大值 |
| `observation_quality` | 四个子单元的最小值 |
| `observation_count` | 四个子单元的最小值 |
| `obstacle_variance` | 四个子单元的最大值 |
| `elevation` | 仅当所有子单元有效时取四个高程的算术均值；否则父单元无效 |
| `elevation_variance` | 最大子单元自身方差，加四个有效子单元高程的总体方差 |

外部生产方必须保存聚合版本与逐层验证证据。规划核心校验收到层级、形状与数值，但单凭
一张粗图无法证明生产方遵守了聚合算法，因此聚合证据属于接口集成验收的一部分。

`Odometry` 接收非零 `header.stamp`、`header.frame_id=odom`、`child_frame_id=base_link`、`pose.pose.position`、单位四元数 `pose.pose.orientation`、`pose.covariance`、`twist.twist.linear`、`twist.twist.angular`、`twist.covariance`。协方差必须为有限值且主对角线非负。

`TFMessage.transforms[]` 接收每个变换的 `header.stamp`、`header.frame_id`、`child_frame_id`、`transform.translation` 和单位四元数 `transform.rotation`。可用链必须是 `map -> odom -> base_link`。

## 任务与科学目标字段

`ExplorationTask` 接收非零的 `header.stamp`、`header.frame_id=map`、非空 `mission_id`、从 1 开始严格递增的 `revision`、`desired_state`、有限且有序的 `roi_min_x_m`、`roi_min_y_m`、`roi_max_x_m`、`roi_max_y_m`，以及 0 至 64 项 `science_regions`。

每个 `ScienceTargetRegion` 接收任务内唯一非空的 `region_id`、非空 `objective_id`、3 至 256 个不同顶点的简单非自交 `boundary`、顶点 `x/y/z` 和 `(0,1]` 的 `priority`。边界位于 `map`，二维边界的 `z=0.0`。

## 运动执行反馈字段

`MotionExecutionFeedback` 的发布者属于外部运动执行/控制系统；本仓只暂定同名 schema、订阅、校验和适配，不接管数据生产。`header.stamp` 必须非零且 `header.frame_id` 必须等于当前平台能力资料的 `base_frame_id`。`sequence` 在同一 `plan_id` 内从 1 开始严格递增，新 `plan_id` 可以重新从 1 开始；迟到、重复或倒序消息一律拒绝。

`platform_type` 必须与当前活动平台一致；`plan_id` 和 `segment_id` 必须非空并与当前授权参考完全匹配。轮式和足式反馈固定使用 `plan_id=segment_id=MotionReference.plan_id`；飞跃式反馈使用 `plan_id=MotionReference.plan_id`、`segment_id=HopSegment.segment_id`。`LANDED_HOLD` 只对飞跃式合法，轮式和足式使用 `SEGMENT_COMPLETE`；`FAILED` 或 `CANCELED` 时 `reason_code` 必须非空。执行反馈不能替代 Odometry，完成、稳定着陆和偏离判断必须同时使用身份匹配的反馈与新鲜状态估计。

订阅 QoS 固定为 `reliable`、`volatile`、`depth 10`。上游正式定义该消息后，必须与其余暂定 schema 一样执行固定版本、逐字段对比和同名包原子替换，不得让两个 provider 共存。

## 飞跃式推进剂状态边界

本项目不接收实时推进剂状态，不累计单跳燃料消耗，也不因历史跳跃次数终止探索 episode。
飞跃可达性只使用正式能力 v2 中固定的参考总质量、参考推进剂质量、比冲和安全裕量，计算每次
相同且可重复的单跳 Δv 包络；该参考量不是会随规划次数递减的库存。

## 静态能力资料字段

运行时只接收一个完整 `lunar-platform-profile/v1` YAML，固定路径为：

```text
/etc/lunar_navigation/platform_profile.yaml
```

文件同时包含 `platform`、`observation`、`assets`、`sources` 和当前平台专属段。平台类型只允许
`WHEELED`、`LEGGED` 或 `HOPPER`；观测段固定承载 `sensor_range_m`、`sensor_fov_deg`，不再通过
第二个观测能力文件拼接。轮式段包含正式几何、轮胎/轴距/轨距、底盘净空、连续运动约束和真实
运动原语；足式段包含 Quad48 机身、质量/载荷、机身高度、坡度/台阶/方向沟隙、速度和加速度；
飞跃式段包含比冲、固定单跳参考质量/推进剂量、着陆支撑、飞行碰撞包络及规划裕量。

URDF/mesh 位于可选 `assets` 段。缺少资源只影响可视化或几何调试，不能阻止依赖 YAML 数值的
规划节点启动，也不能用资源中的值覆盖已审查数值。替换文件后必须重启，运行中不热加载。

仓库中的 `platform_profiles/wheeled.yaml`、`legged.yaml`、`hopper.yaml` 是当前已批准工程基线。
正在运行的本轮正式训练仍以 `three_platform_capability_freeze_v1.yaml` 和既有摘要作为不可变历史
输入；`check_platform_profiles.py` 证明三份新文件的运行数值与该冻结等价。训练完成后，后续训练、
规划和部署直接读取三份 profile，旧聚合文件只保留为本次模型证据。

静态能力和真实设备数据的所有权仍属于外部平台与传感系统。正式 provider 出现后必须先逐字段
核对，再原子替换固定路径上的完整文件；不得让 provisional 文件与正式 provider 同时生效。

## 外部接口适配 profile

外部项目的 Topic 名称或自定义消息类型与本基线不一致时，只替换以下固定文件并重启：

```text
/etc/lunar_navigation/interface_profile.yaml
```

该文件使用 `lunar-interface-profile/v1`，完整声明七个输入通道、外部/内部 Topic、消息类型、
frame、QoS 和固定 converter 名称。`GridMap`、`Odometry`、`TFMessage` 等标准类型只通过 launch
remap 接入，不经过 Python 复制；三种本仓暂定消息只允许调用编译进代码的显式转换函数。配置
不得包含字段表达式、`eval`、按同名字段自动复制或运行时动态代码。converter 的输入与输出 Topic
必须不同，转换失败的单条消息只记录诊断并丢弃，不能向稳定内部 Topic 发布半有效数据。

当前工程基线为 `config/interface_profiles/default.yaml`。未来对接正式 provider 时，ROS 内部
`PlanMotion`、地图语义及规划器输入保持不变；只替换 profile，并在确有字段差异时修改对应的
一个显式转换函数。接口 profile 属于本仓部署配置，不改变外部系统对原始 Topic 数据的所有权。

## 消费边界

本项目在运行时校验字段、时间、坐标系和范围，并以双方确认的 rosbag 执行兼容性测试。缺少派生地图层的前提时必须拒绝快照，不能用零值、空数组或 synthetic 数据冒充真实输入；概率障碍层不能直接标为确定物理障碍。
