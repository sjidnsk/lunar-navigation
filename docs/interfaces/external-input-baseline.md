# 外部输入接收基线

> **这是本项目暂定消息字段与语义的唯一权威基线。** `lunar_navigation_msgs` 上游尚未定义期间，本仓 `.msg`、配置和检查器必须与本文一致；Topic 数据生产者仍由外部项目拥有。未来切换上游必须执行固定版本、schema 对比和原子替换，不能叠加同名包。

## 来源与所有权

- 只读输入：`docs/外部输入/课题四未知场景无人平台自主探索与规划外部输入.md`
- SHA-256：`a4c2db0a6647d59fa7cee5cf8048d18f7bca7c7b11a591a32d33d237c0c06e78`
- `grid_map_msgs`、`nav_msgs` 和 `tf2_msgs` 仍来自 ROS/外部系统；定位与任务系统仍负责消息数据的发布和演进协商。
- 本仓暂定提供三个 `lunar_navigation_msgs` schema；Topic 数据生产者仍由外部项目拥有。
- 本项目只声明依赖、直接订阅、适配和运行时校验；上游尚未定义时，本仓 `.msg`、配置和检查器以本文为准。未来切换必须固定唯一上游 tag 或 commit、执行 schema 对比并原子替换，不得与上游同名包共存。

## Topic 与接收字段

| 输入 | 类型 | 外部所有权 | 接收字段/约束 |
|---|---|---|---|
| `/environment/map_global` | `grid_map_msgs/msg/GridMap` | 外部地图融合系统 | `header`、`info`、`layers`、`basic_layers`、`data`、`outer_start_index`、`inner_start_index`；`frame_id=map` |
| `/environment/map_local` | `grid_map_msgs/msg/GridMap` | 外部地图融合系统 | 同上；`frame_id=odom` |
| `/localization/odometry` | `nav_msgs/msg/Odometry` | 外部定位系统 | `header`、`child_frame_id`、`pose`、`twist`；`odom -> base_link` |
| `/localization/status` | `lunar_navigation_msgs/msg/LocalizationStatus` | 外部定位系统 | `header`、`status`；状态为 `UNKNOWN/VALID/DEGRADED/INVALID/RELOCALIZING` |
| `/tf` | `tf2_msgs/msg/TFMessage` | 外部 TF 发布者 | `transforms[]`，形成 `map -> odom -> base_link` |
| `/mission/exploration_task` | `lunar_navigation_msgs/msg/ExplorationTask` | 外部任务系统 | `header`、`mission_id`、`revision`、`desired_state`、ROI、`science_regions`；状态为 `ACTIVE/PAUSED/CANCELED` |
| `science_regions[]` | `lunar_navigation_msgs/msg/ScienceTargetRegion` | 外部任务系统 | `region_id`、`objective_id`、`boundary`、`priority` |

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

## 地图与定位字段

`GridMap` 接收 `header.stamp`（非零）、`header.frame_id`、`info.resolution`、`info.length_x`、`info.length_y`、`info.pose`、唯一的 `layers`、`basic_layers`、与层顺序一致的 `data`、`outer_start_index`、`inner_start_index`。本项目依赖的地图层为：

`elevation`、`valid_mask`、`obstacle`、`obstacle_height`、`observation_age_s`、`observation_quality`、`elevation_variance`、`obstacle_variance`、`observation_count`、`forbidden`。

`Odometry` 接收非零 `header.stamp`、`header.frame_id=odom`、`child_frame_id=base_link`、`pose.pose.position`、单位四元数 `pose.pose.orientation`、`pose.covariance`、`twist.twist.linear`、`twist.twist.angular`、`twist.covariance`。协方差必须为有限值且主对角线非负。

`TFMessage.transforms[]` 接收每个变换的 `header.stamp`、`header.frame_id`、`child_frame_id`、`transform.translation` 和单位四元数 `transform.rotation`。可用链必须是 `map -> odom -> base_link`。

## 任务与科学目标字段

`ExplorationTask` 接收非零的 `header.stamp`、`header.frame_id=map`、非空 `mission_id`、从 1 开始严格递增的 `revision`、`desired_state`、有限且有序的 `roi_min_x_m`、`roi_min_y_m`、`roi_max_x_m`、`roi_max_y_m`，以及 0 至 64 项 `science_regions`。

每个 `ScienceTargetRegion` 接收任务内唯一非空的 `region_id`、非空 `objective_id`、3 至 256 个不同顶点的简单非自交 `boundary`、顶点 `x/y/z` 和 `(0,1]` 的 `priority`。边界位于 `map`，二维边界的 `z=0.0`。

## 静态能力资料字段

| 资料 | 类型 | 外部所有权 | 接收字段 |
|---|---|---|---|
| 观测能力 | YAML/JSON | 传感与融合系统 | `sensor_range_m`、`sensor_fov_deg` |
| 平台能力 | `platform-control-capability-source/v1`，YAML/JSON/URDF/mesh | 平台控制单位 | `platform.platform_id`、`platform.platform_type`、`platform.capability_version`、`platform.base_frame_id`、`geometry_source.urdf_file` 与 URDF 引用 mesh |

平台类型为 `WHEELED`、`LEGGED` 或 `HOPPER`。轮式资料还接收净空、最大坡度、障碍高度、速度/加速度、曲率和 `motion_primitives`；足式资料还接收参考点、坡度/粗糙度/台阶/净空、机体高度、速度/加速度和 `motion_primitives`；飞跃式资料还接收着陆坡度/粗糙度/净空、着陆区域、发射/飞行/着陆约束及 `actuator_or_impulse_profile`。字段的单位、范围和完整语义以已列来源文档为准。

## 消费边界

本项目在运行时校验字段、时间、坐标系和范围，并以双方确认的 rosbag 执行兼容性测试。缺少派生地图层的前提时必须拒绝快照，不能用零值、空数组或 synthetic 数据冒充真实输入；概率障碍层不能直接标为确定物理障碍。
