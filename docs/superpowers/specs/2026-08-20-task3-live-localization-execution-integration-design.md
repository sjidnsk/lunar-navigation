# 课题三实时局部地图、车体位姿与执行闭环联调设计

## 1. 决策

本项目保持课题三既有的地图、位姿和 SQLite 接口不变；新增本项目侧的运行时适配层，形成可审计的 ROS 规划闭环。

- 课题三的 `/Car/T3/mapping/grid_map` 已为 `0.2 m` 分辨率，局部图适配器不得重采样、插值或改变栅格数量。
- 课题三直接提供 `map -> odom -> base_link` TF 链及 `odom -> base_link` 车体 Odometry；本项目直接消费，不再转换相机位姿或发布新的车体 TF。
- 课题三局部图必须为 `frame_id=odom`、0.2 m 且栅格轴对齐 odom；局部图适配器只做图层派生和有效性校验，输出现有规划器消费的 `/environment/map_local`。
- `PlanMotion` Action 是唯一规划请求和高层运动输出接口；底层控制、路径跟踪和执行反馈仍由外部项目拥有。
- 第一阶段使用 `fallback` 策略运行真实规划联调；训练模型不得因“已安装”而被伪装为已接入 ROS 推理。

现有 `luna_t3_map_adapter` 的只读 SQLite 全局图职责不扩大。本设计只补齐其已保留的局部图、定位和执行联调边界。

## 2. 范围与非目标

### 范围

1. 将课题三已经处于 `odom`、60 m x 60 m、0.2 m 的局部图适配为同几何的十层局部执行证据图；不做坐标变换。
2. 直接校验课题三车体 Odometry 与 TF 链，并从其新鲜度和协方差生成定位状态。
3. 定义外部 Action client、控制器和执行反馈的最小联调合同。
4. 提供一个同时启动全局图适配器、局部图适配器、定位状态校验器和规划服务的联调入口。
5. 用纯转换测试、ROS 回放测试和一次安全的真实地面参考请求验证该链路。

### 非目标

- 不修改课题三的 `/Car/T3/*` topic、SQLite 文件、WAL/SHM 或其发布频率。
- 不把全局 0.2 m 大图复制成局部图或反复发布为稠密 ROS 消息。
- 不接管车辆底层控制、步态控制、推进器命令或外部任务系统。
- 不新增模型 ROS 推理接线，不改变 `fallback` 的 fail-closed 模型边界。
- 不以零值、常量低方差或虚构的高质量观测填充缺失地图证据。

## 3. 系统数据流和责任边界

```text
Task3 SQLite + /Car/T3/mapping/global_map_revision
                 |
                 v
       luna_t3_map_adapter (existing)
                 |
                 +---- /environment/map_global (map, bounded pyramid)

/Car/T3/mapping/grid_map (0.2 m, odom)
Task3 L0 tile cache + safety mapping
                 |
                 v
       luna_t3_local_map_adapter (new)
                 |
                 +---- /environment/map_local (0.2 m, odom)

/Car/T3/semantic/current_pose (odom -> base_link)
/tf (map -> odom -> base_link)
                 |
                 +---- direct planner odometry/TF input
                 |
                 v
  luna_t3_localization_status_adapter (new, validation only)
                 |
                 +---- /localization/status

external task supervisor ---- /mission/exploration_task
external action client ------ /plan_motion
lunar_planner_ros ---------- MotionReference in Action result
external controller -------- /execution/motion_feedback
```

本项目拥有全局图、局部图和定位状态三个适配节点以及 `PlanMotion` 服务；课题三拥有源地图、车体 TF/里程计和 SQLite 写入；外部任务/控制项目拥有任务发布、Action 调用、低层运动执行和反馈发布。

## 4. `luna_t3_local_map_adapter`

### 4.1 输入与输出

输入：

| 输入 | 类型 | 约束 |
| --- | --- | --- |
| `/Car/T3/mapping/grid_map` | `grid_map_msgs/msg/GridMap` | `frame_id=odom`；单位朝向；分辨率严格为 `0.2 m`；非零时间戳；课题三现有 QoS `RELIABLE + TRANSIENT_LOCAL + depth=1` |
| `/tf` | `tf2_msgs/msg/TFMessage` | 在局部图时间戳存在 `map -> odom`；仅用于把局部格中心查询到全局 L0 evidence，绝不改变局部图几何或重新发布 TF |
| 全局 L0 tile cache | 课题三 SQLite 只读快照 | 用于补充确有来源的高度范围、方差和累计观测资料 |
| 安全映射配置 | YAML | occupancy 阈值、语义障碍/禁入类别和外部禁区 |

输出 `/environment/map_local`：

- 类型为 `grid_map_msgs/msg/GridMap`；
- `frame_id=odom`；
- 分辨率与源图完全相同，均为 `0.2 m`；
- 空间长度、层维度、原点、单位朝向和有效源格数量均与源图一致；
- QoS 固定为 `RELIABLE + TRANSIENT_LOCAL + depth=1`；
- 必须含外部接口 v5 规定的十个层。

若源图不是 `odom`、姿态不是单位朝向、分辨率不是 0.2 m、时间戳为零、层缺失，或没有同时间的 `map -> odom` 以认证所需 L0 evidence，适配器不发布新图并输出稳定诊断原因。适配器不得把 `map` 图简单改名为 `odom` 图；TF 查询只服务于 evidence 索引，绝不变更本地 GridMap 的几何。

### 4.2 图层派生

| 输出层 | 规则 |
| --- | --- |
| `elevation` | 直接采用课题三局部高程。 |
| `valid_mask` | occupancy、高程、roughness 均为有效源值，且坐标可与当前全球 L0 evidence 对齐时为 1。 |
| `obstacle` | `occupancy >= configured_threshold` 与配置的语义障碍类取 OR。 |
| `obstacle_height` | 使用同坐标、同可验证 source tile 的 `height_range`；缺失可信高度时该格不得作为安全站位或飞跃飞行/着陆证据。 |
| `elevation_variance` | 使用同坐标 source tile 的 `elevation_variance`；缺失即无效。 |
| `obstacle_variance` | 使用 source tile 的 roughness/不确定性保守映射；缺失即无效。 |
| `observation_count` | source tile 有值时保留；只有 source tile 缺失且收到新的有效局部图时间戳时，适配器本地计数从 1 开始递增。 |
| `observation_age_s` | 由每个有效格最后一次接收的局部图源时间戳计算；进程重启前未知历史年龄不宣称新鲜。 |
| `observation_quality` | 由源字段完整性、局部图新鲜度和可用的语义置信度保守计算；无证据时为无效而不是高质量。 |
| `forbidden` | 无效格、明确任务禁区、配置的语义禁区或不能证明安全所需字段的格取 OR。 |

任务 ROI 只定义任务收益和全局工作集，不能自动写入 `forbidden`；平台可以在认证的安全前提下从 ROI 外绕行。

适配器维护固定容量的按 0.2 m map-cell 编址观测账本和全局 tile LRU cache。它只更新本次局部窗口触及的格，不能在每帧扫描完整地图。未知、失配或超出缓存可证明范围的格 fail-closed。

### 4.3 平台差异

- WHEELED 和 LEGGED：`obstacle` 与 `forbidden` 是局部路径的硬约束；高度/方差字段仍需完整，用于坡度、净空和风险判定。
- HOPPER：任何起飞、着陆或飞行管相交的格都必须有可信高度范围和方差证据；不能以局部 occupancy 可见作为高度安全证明。

## 5. `luna_t3_localization_status_adapter`

### 5.1 直接消费车体状态

课题三 `/Car/T3/semantic/current_pose` 现在直接表示 `odom -> base_link`；本项目的运行时接口把 `interfaces.odometry` 配置为该 topic，不复制、不转换 Odometry。课题三 `/tf` 直接提供 `map -> odom -> base_link`，本项目也不重新广播该链。

`luna_t3_localization_status_adapter` 仅订阅该 Odometry，并发布：

```text
/localization/status        UNKNOWN/VALID/DEGRADED/INVALID/RELOCALIZING
```

该节点不发布 `/localization/odometry`、`/tf` 或任何车体命令。

### 5.2 有效性与失败语义

- 源 Odometry 的 `header.frame_id=odom`、`child_frame_id=base_link`，且时间戳、四元数、位置、速度及协方差均有限并满足配置阈值时发布 `VALID`；
- 可用但协方差较大时发布 `DEGRADED`；
- frame 不匹配、非单位四元数或过期消息时发布 `INVALID`，且规划器只能 HOLD；`map -> odom -> base_link` 缺失由直接消费该链的规划器拒绝并 HOLD，状态适配器不伪造或重发 TF；
- 不允许由本项目把相机 pose 重命名为车体 pose；课题三必须发布真实车体 Odometry。

## 6. 任务、规划和执行适配

### 6.1 任务和规划请求

外部任务系统发布现有 `/mission/exploration_task`，提供递增的 `revision`、`ACTIVE` 状态、`map` 系 ROI 及科学区域。

外部任务协调器或集成测试 Action client 使用现有 `/plan_motion`：

```text
request_id, mission_id, mission_revision,
GoalRegion(goal_id, POINT/PLANAR_REGION, point/region, tolerance, yaw constraint),
replace_active_request
```

`ExplorationTask` 本身不会自动产生 `PlanMotion` 请求。部署首版以外部 Action client 驱动单次安全目标规划；未来自主探索协调器必须作为独立上层组件接入，不能偷偷改变规划器的 Action 语义。

### 6.2 MotionReference 和执行反馈

`PlanMotion` Result 内的 `MotionReference` 是唯一高层运动输出：地面平台由 `path_preview`/`trajectory` 表示，HOPPER 还包含 `hops`。外部控制器/跟踪器负责将其变成实际控制命令。

控制器通过现有 `/execution/motion_feedback` 回传：

```text
ACCEPTED -> EXECUTING -> SEGMENT_COMPLETE
FAILED(reason_code) or CANCELED(reason_code)
```

WHEELED/LEGGED 固定 `segment_id=plan_id`；HOPPER 使用 HopSegment 的 `segment_id` 并只允许在真实稳定着陆后发送 `LANDED_HOLD`。每条反馈必须匹配当前平台、`plan_id`、`segment_id` 和严格递增的 sequence。反馈不能替代新鲜 Odometry。

可选 `luna_execution_bridge` 仅作为控制器适配插件边界：它可以充当 `PlanMotion` Action client、向某个具体控制系统转发 MotionReference、再标准化反馈；在控制器 topic/命令契约明确前不得猜测或发布底层命令。

## 7. 启动、配置与可观测性

新增 `task3_live_integration.launch.py`，启动：

1. `luna_t3_map_adapter`；
2. `luna_t3_local_map_adapter`；
3. `luna_t3_localization_status_adapter`；
4. `lunar_planner_ros`；
5. 可选的已明确控制器契约的 execution bridge。

配置最少包含：课题三源 topic、SQLite 绝对路径、固定 frame 名称、语义/occupancy 安全映射、局部图与里程计新鲜度阈值、能力文件和 planner snapshot policy。

每个适配器发布 diagnostics，至少包含源帧率/时间戳、有效格/禁入格数量、cache hit/miss、局部图几何合同状态、最后拒绝原因及最近已发布版本。诊断与 route markers 是观察接口，不是控制命令。

## 8. 验证和联调顺序

### 8.1 纯逻辑与单元验证

1. 0.2 m 输入以完全相同的栅格尺寸、索引和坐标间距输出；任何重采样尝试为失败。
2. 车体 Odometry 的 frame、时间戳、四元数和协方差违反合同即产生 `INVALID` 定位状态。
3. 缺失高度范围、方差、有效掩码或过期证据的格变为无效/禁入，绝不变为可通行。
4. 同一源时间戳不重复增加 observation count；新时间戳只更新窗口内触及格。
5. 错误 frame、非单位局部图姿态、错误分辨率、过期时间和不匹配 feedback 都得到稳定拒绝原因。

### 8.2 ROS 集成验证

以记录的 Task3 地图、位姿、revision 和标定 replay：

1. 激活任务后，现有全局适配器发布一个有界 `/environment/map_global`；
2. 新局部适配器发布十层、0.2 m、`odom` 系 `/environment/map_local`；
3. `tf2_echo map base_link` 与课题三车体 Odometry 一致；
4. planner 输入均在配置新鲜度和 pairwise skew 内；
5. 发出一个近距离安全 `PlanMotion` POINT 请求，验证结果、诊断和参考 ID；
6. 注入严格匹配的 `ACCEPTED/EXECUTING/SEGMENT_COMPLETE` 与新鲜里程计，验证下一次请求可继续；注入错误 ID/旧 sequence 验证其被拒绝。

### 8.3 真实联调阶段

先 WHEELED，依次完成“静止输入链”“一次不执行规划”“受控短路径执行闭环”。随后 LEGGED 仅切换能力文件。HOPPER 需另行完成飞行控制器和稳定着陆反馈验证，不能复用地面执行器作为替代。

## 9. 验收条件

满足以下条件才可称为“可与外部项目开展端到端地面联调”：

1. 所有规划输入 topic 与自定义消息均有唯一 provider，且满足 `lunar-external-interfaces/v5`；
2. 本地地图未经重采样、十层完整、缺证据格 fail-closed；
3. `map -> odom -> base_link` 与车体 Odometry 来自课题三，且不存在相机/车体 frame 冒名；
4. 对安全 Action 请求能得到可解释结果，并能传递一个身份匹配的 MotionReference；
5. 外部控制器可以回传完整的身份匹配反馈，规划器拒绝重复、倒序及错误身份反馈；
6. 所有 live 验证在目标 Ubuntu 22.04/ROS 2 Humble 或 Orin Humble 环境完成，不能以 Windows 或离线训练环境代替。

模型推理、自主候选选择、路径跟踪实现和 HOPPER 真实飞行不属于上述地面联调验收条件。
