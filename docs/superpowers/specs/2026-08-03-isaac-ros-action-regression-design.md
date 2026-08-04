# 仓库外 Isaac→ROS Action 回归设计

## 目标

在不修改 Isaac Sim 场景、不向本仓库写入运行产物的前提下，实现一套仓库外测试系统：从当前月球极区 USD 只读提取地形、月岩碰撞和三类平台初始状态，将其转换为 ROS 2 Humble 规划输入，加载三份平台能力资料，并通过 `/plan_motion` 对轮式、足式和跳跃式规划器分别执行一个正例和一个反例。

本阶段只验证规划 Action 的结果语义、参考几何和能力约束，不驱动 Isaac Sim 中的平台运动，也不开发轮式、足式或跳跃式控制器。

## 已选方案

采用双运行时的确定性快照桥：

- Isaac Sim 6.0.1 的 Python 3.12 运行时通过 `isaacsim.code_editor.python_server` 执行只读采集器。
- 采集器输出不含 ROS 类型的 JSON/NPZ 快照。
- Ubuntu 22.04 的系统 Python 3.10 与 ROS 2 Humble 运行外部 ROS 测试包，加载本项目的自定义消息并发布规划输入。
- 每个正反例使用全新的规划节点生命周期会话；六个会话严格顺序运行。

不采用以下方案：

- 不在 Isaac Python 3.12 中直接加载由系统 Python 3.10 生成的项目自定义 ROS 消息。
- 不用 OmniGraph 和外部节点分别发布同一份规划快照，以免引入双时钟和双发布源。
- 不把实时运动闭环、执行反馈或控制器开发并入本阶段。

## 范围与硬边界

- 所有实现源码、能力资料、构建目录、日志和测试报告位于 `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/`。
- 本仓库只保存本设计和后续实施计划；桥接实现、能力资料及 artifact 不进入本仓库。
- 当前 USD 保持只读；采集器不得播放时间轴、修改 prim、保存 layer 或导出派生 USD。
- 测试包是外部输入的实验生产者，不接管 `external_interfaces.yaml` 中声明的上游数据所有权。
- 三份能力资料是 `proxy-v1` 验证资料，不替代未来由平台控制项目交付的正式能力文件。
- 不修改现有 Topic、消息、Action、生命周期节点或 `/plan_motion` 绝对名称。
- 不启动 Nav2，不做平台运动执行、性能、功耗或稳定性验收。

## 外部目录结构

外部根目录采用以下结构：

```text
/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/
├── isaac/
│   └── collect_stage_snapshot.py
├── ros2_ws/
│   └── src/
│       └── lunar_isaac_validation/
│           ├── config/
│           │   ├── capabilities/{wheeled,legged,hopper}.yaml
│           │   ├── observation.json
│           │   ├── capability_provenance.json
│           │   └── planner/{wheeled,legged,hopper}.yaml
│           ├── meshes/{wheeled,legged,hopper}_proxy_collision.stl
│           ├── urdf/{wheeled,legged,hopper}_proxy.urdf
│           ├── scenarios/scenario_lock.json
│           ├── lunar_isaac_validation/
│           ├── test/
│           ├── package.xml
│           ├── setup.cfg
│           └── setup.py
├── scripts/{build_external,qualify_fixtures,run_action_regression}.sh
├── build/
├── install/
├── log/
└── artifacts/<run-id>/
```

`build/`、`install/`、`log/` 和 `artifacts/` 均为运行目录，不得链接或复制回仓库。

## 组件职责

### Isaac 只读采集器

采集器只接受当前已打开 stage，不自行打开其它 USD。它必须：

1. 读取 stage identifier、根 layer 路径、`metersPerUnit` 和 `upAxis`，要求 1 米单位和 Z-up。
2. 读取 `/World/LunarPolarTerrain` 的世界空间三角网格。
3. 枚举 `/World/LunarPolarRock_*`，确认每块月岩都有独立 `PhysicsCollisionAPI`，记录碰撞网格与世界变换。
4. 读取 `/World/LunarExplorationPlatforms/{WheeledScout,LeggedScout,HopperScout}` 的世界变换、质量、碰撞子 prim 和 `customData`。
5. 排除相机、灯光、材质以及三类平台自身，不把它们栅格化为环境障碍。
6. 在采集前后计算根 USD 的 SHA-256；若发生变化，拒绝输出有效快照。

采集器通过 loopback Python Server 连接。若服务器启用认证，令牌只从进程环境或权限为 `0600` 的外部文件读取，不写入命令行、报告或日志。

### 中立快照

`snapshot_manifest.json` 至少包含：

- `schema_version=isaac-ros-planning-snapshot/v1`
- stage identifier、USD SHA-256、Isaac Sim 版本、米制单位和 up axis
- 地形与月岩碰撞 prim 清单及数量
- 三个平台的根位姿、规划参考位姿、能力元数据和碰撞包络
- 全局图与三个局部图的原点、分辨率、尺寸、frame 和数组文件名
- `snapshot_arrays.npz` 的 SHA-256

NPZ 中每张图包含十个同尺寸数组：

`elevation`、`valid_mask`、`obstacle`、`obstacle_height`、`observation_age_s`、`observation_quality`、`elevation_variance`、`obstacle_variance`、`observation_count`、`forbidden`。

所有数组先写入同目录临时文件，完成哈希和 schema 校验后再原子重命名，ROS 侧不得读取半成品。

### ROS 输入桥

ROS 输入桥只读取已验证的中立快照，并发布：

| Topic | 类型 | QoS 与行为 |
| --- | --- | --- |
| `/environment/map_global` | `grid_map_msgs/msg/GridMap` | reliable、transient-local |
| `/environment/map_local` | `grid_map_msgs/msg/GridMap` | reliable、transient-local |
| `/localization/odometry` | `nav_msgs/msg/Odometry` | SensorDataQoS |
| `/localization/status` | `lunar_navigation_msgs/msg/LocalizationStatus` | reliable、depth 10 |
| `/tf` | `tf2_msgs/msg/TFMessage` | best-effort、depth 100 |
| `/mission/exploration_task` | `lunar_navigation_msgs/msg/ExplorationTask` | reliable、transient-local |

桥以一个定时回调为六类消息生成同一 ROS 当前时间戳。地图和任务至少 2 Hz 发布，Odometry、定位状态和 TF 以 10 Hz 发布，确保输入年龄和 pairwise skew 满足规划节点策略。测试不使用 Isaac 仿真时钟，也不发布 `/clock`。

`map→odom` 为单位变换；局部图采用 `odom` frame，但其坐标与 stage 世界坐标保持一致。`odom→base_link` 与 Odometry 使用相同平台规划参考位姿。轮式参考位姿使用代理根；足式参考位姿为地形接触根加 `0.55 m`；跳跃式参考位姿为地形接触根加 `0.60 m`。

### Action 回归运行器

运行器负责进程编排、生命周期转换、Action 调用、结果断言和报告。它不实现规划算法，也不从 Action 实际结果生成期望值。

每个场景启动独立的 `/lunar_planner`：

1. 启动对应场景的 ROS 输入桥。
2. 启动 `lunar_planner_ros/lunar_planner_node` 并传入对应参数文件。
3. 执行 `configure`，要求状态为 `inactive`；此时能力资料必须成功加载。
4. 执行 `activate`，要求状态为 `active`。
5. 等待 `/plan_motion` Action server、全部订阅以及至少三批同代输入就绪。
6. 发送 `replace_active_request=false` 的单个目标并收集全部 feedback 和 result。
7. 完成断言后执行 `deactivate → cleanup → shutdown`，再结束输入桥。

由于 `/plan_motion` 是绝对名称，任何时刻只能存在一个被测规划节点。正例产生的 active reference 不得泄漏到反例，因此总计运行六个独立会话。

## 地图生成规则

### 栅格范围与采样

- 全局图覆盖完整地形 XY 包围盒，分辨率为 `0.5 m`。
- 每个平台生成一张分辨率为 `0.25 m` 的局部图；范围至少覆盖起点、目标、完整候选路线或飞行管投影，并在地形边界允许范围内保留平台包络余量。
- 每个栅格使用中心和四个四分之一格偏移点，共五个固定采样点。
- 地形 elevation 通过世界空间三角形上的竖直相交确定；有效采样高度的中位数写入 `elevation`，方差写入 `elevation_variance`，有效数量写入 `observation_count`。
- 无有效地形相交的单元设置 `valid_mask=0`，其余设置为 1；无效单元的 elevation 和 variance 填入 `0.0`，保证所有浮点层有限。

### 月岩障碍

每块月岩使用带 CollisionAPI 的实际碰撞网格进行 XY 栅格化。与单元相交时：

- `obstacle=1`
- `obstacle_height=max(0, collision_top_z-terrain_elevation)`
- `obstacle_variance=0`

没有碰撞交叠时三者分别为 `0`、`0.0`、`0.0`。不得仅根据渲染网格或 prim 名称伪造碰撞。

### 固定观测层

- `observation_age_s=0.0`
- `observation_quality=1.0`
- `forbidden=0`

GridMap 编码必须符合现有 adapter：十层名称唯一，二维布局标签、stride、尺寸和 start index 合法；binary、unit interval 和 count 层必须使用可无损转换的 float 值。

## 平台能力资料

三份 YAML 使用 `platform-control-capability-source/v1`，`capability_version=proxy-v1`，`base_frame_id=base_link`，`geometry_source.urdf_file` 指向包 share 内对应 URDF。每份 URDF 的 `base_link` 必须引用至少一个与 USD 代理碰撞包络一致的外部 STL，满足现有能力加载器的几何校验。

共享 `observation.json` 固定为 `sensor_range_m=30.0`、`sensor_fov_deg=120.0`。

### 轮式

- `platform_id=proxy-wheeled-scout-v1`
- footprint：`[-0.61,-0.54]`、`[0.61,-0.54]`、`[0.61,0.54]`、`[-0.61,0.54]` 米
- body z：`[0.0,0.555] m`
- 最大坡度 `18°`，最大障碍高度 `0.18 m`，最小净空 `0.22 m`
- 前进 `0.70 m/s`，倒车 `0.35 m/s`，原地旋转 `0.60 rad/s`
- 加速 `0.40 m/s²`，制动 `0.50 m/s²`，偏航加速度 `0.60 rad/s²`，横向加速度 `0.35 m/s²`
- 最大曲率 `1/0.70 m⁻¹`
- primitives：`FORWARD`、`REVERSE`、左右 `FORWARD_ARC`、`SPIN_CLOCKWISE`、`SPIN_COUNTERCLOCKWISE`、`STOP_AND_SWITCH`；直线位移按 `0.25 m` 栅格对齐，圆弧半径不小于 `0.70 m`，所有 nominal duration 必须满足上述速度限制

### 足式

- `platform_id=proxy-legged-scout-v1`
- `reference_point=base_link`
- body half extent：`[0.475,0.325,0.14] m`
- 最大坡度 `28°`，最大粗糙度 `0.12 m`，最大台阶 `0.22 m`，最大间隙 `0.35 m`
- 最小 confidence `0.80`，最小身体净空 `0.25 m`，身体高度区间 `[0.50,0.60] m`
- 前向速度 `[-0.20,0.45] m/s`，横向速度 `[-0.30,0.30] m/s`，竖直速度 `[-0.15,0.15] m/s`
- 偏航速度 `[-0.60,0.60] rad/s`，线加速度 `0.35 m/s²`，偏航加速度 `0.60 rad/s²`
- primitives：`FORWARD`、`BACKWARD`、`LATERAL_LEFT`、`LATERAL_RIGHT` 和两条 yaw change 分别为 `±π/4` 的 `SPIN`；平移以 `0.25 m` 为基本步长

### 跳跃式

- `platform_id=proxy-hopper-scout-v1`
- body half extent：`[0.275,0.275,0.60] m`
- 质量 `20.0 kg`，重力 `[0.0,0.0,-1.62] m/s²`
- 最大落区坡度 `15°`，最大粗糙度 `0.08 m`，最大平面残差 `0.05 m`
- 最小 overhead clearance `0.10 m`、lateral clearance `0.10 m`、landing clearance `0.25 m`
- 最小落区面积 `π×0.65²=1.327322 m²`
- 最大发射速度 `2.20 m/s`，最大冲量 `44.0 N·s`
- 飞行时间 `[0.50,3.00] s`，最大落地速度 `2.20 m/s`，最小向下接触速度 `0.10 m/s`
- 最大角速度 `0.80 rad/s`，最大角加速度 `1.00 rad/s²`，最大初始角速度 `0.10 rad/s`
- settle guard `1.00 s`，profile `proxy_impulse_v1`，primitive `nominal-hop`

`capability_provenance.json` 对每个字段记录 `usd_custom_data`、`usd_collision_geometry`、`derived` 或 `validation_only_assumption`。倒车、角速度、间隙、置信度、平面残差和未在 USD 中直接声明的动力学上限必须标为验证专用假设；不得把它们写回 USD 或正式外部接口基线。

## 场景资格与锁定

首次实施时，资格检查器依据快照和上述能力约束生成候选报告，再按固定规则选择六个坐标并写入 `scenario_lock.json`。锁文件使用 `lunar-scenario-lock/v2`，包含 stage SHA-256、快照数组 SHA-256、平台连续请求起点、目标、目标 tolerance、选择证据以及精确期望的 Action outcome、directive 和 reason code。旧版 `v1` 锁不得被新版回归静默接受。

锁文件一旦存在，普通回归只能读取，不能重选坐标、修改期望值或覆盖锁文件。stage 或快照哈希不匹配时返回 `FIXTURE_STAGE_MISMATCH`；重新基线必须显式运行资格命令并产生独立审阅报告。

### 2026-08-04 方案 A 语义修订

本修订保持生产规划器、能力资料、快照数组和场景 USD 不变，只使外部资格检查与断言遵循现有离散规划契约。不得从 Action 实际输出反推锁值。

轮式和足式仍以平台规划参考位姿作为 Odometry、TF 和 Action 请求的连续起点；锁证据另外保存 `start_cell_xy`、`projected_start_position_m` 和 `start_projection_maximum_xy_error_m`。投影按局部图执行：

- `cell_x=floor((start_x-origin_x)/resolution)`，Y 轴同理。
- 投影 X/Y 为该单元中心；每轴与连续请求起点的偏差不得超过 `resolution/2+1e-9 m`。
- 轮式投影 Z 为起点单元的 `elevation`。
- 足式投影 Z 为起点单元的 `elevation + (body_height_lower+body_height_upper)/2`。

资格检查器必须从锁定快照和能力资料独立计算这些字段。Action 断言先重新计算并核对锁证据，再要求轮式或足式轨迹首点在 `1e-3 m` 内等于投影起点；不得把连续请求起点直接当作离散轨迹首点。

`wheel-positive`、`wheel-negative`、`legged-positive`、`legged-negative` 和 `hopper-negative` 的点目标 tolerance 保持 `0.50 m`；`hopper-positive` 固定使用 `0.75 m`。跳跃正例对每个 1–2 m 候选目标执行与生产规划器相同的目标裁剪：只保留中心距点目标不超过 `0.75 m` 的认证安全单元，以距目标中心最近且按 `(distance,y,x)` 稳定排序的单元为 seed，再按 `minimum_x → maximum_x → minimum_y → maximum_y` 顺序反复扩展安全矩形，直到不能扩展。只有该目标裁剪矩形面积不小于 `1.327322 m²` 且弹道检查通过时才能锁定；证据保存 tolerance、候选单元数、seed、矩形边界和面积。在固定 `0.25 m` 栅格上，完整 `5×5` 矩形面积为 `1.5625 m²`，提供高于最低面积的离散余量。

六个场景为：

| 场景 | 资格条件 | 精确期望 |
| --- | --- | --- |
| wheel-positive | 实际连通路线无月岩碰撞且坡度不超过 18°，目标距起点 2–4 m | `NEW_REFERENCE_AVAILABLE / ACTIVATE_NEW_REFERENCE / WHEEL_PLAN_AVAILABLE` |
| wheel-negative | 目标 tolerance 内所有单元均被高度超过 0.18 m 的月岩碰撞覆盖 | `GOAL_INFEASIBLE / HOLD_POSITION / WHEEL_GOAL_INFEASIBLE` |
| legged-positive | 实际连通路线至少经过一个坡度大于 18°且不超过 28°的单元，同时满足足式其它约束 | `NEW_REFERENCE_AVAILABLE / ACTIVATE_NEW_REFERENCE / LEGGED_BODY_PLAN_AVAILABLE` |
| legged-negative | 目标 tolerance 内所有单元均超过 28°坡度或 0.22 m 邻域台阶限制 | `GOAL_INFEASIBLE / HOLD_POSITION / LEGGED_GOAL_INFEASIBLE` |
| hopper-positive | 目标距起点 1–2 m，`0.75 m` 点目标裁剪后存在面积不小于 1.327322 m² 的认证落区且可完成一次受约束跳跃 | `NEW_REFERENCE_AVAILABLE / ACTIVATE_NEW_REFERENCE / HOPPER_FIRST_HOP_AVAILABLE` |
| hopper-negative | 目标 tolerance 内所有候选单元均因坡度、粗糙度、残差或净空不安全，不能形成任何落区 | `GOAL_INFEASIBLE / HOLD_POSITION / HOPPER_GOAL_INFEASIBLE` |

找不到任一真实场景时资格检查失败；不得通过注入合成障碍、改写地图层或放宽能力参数让资格检查通过。

## Action 输入和断言

每个会话发布一个 `ACTIVE` mission，revision 为 1，ROI 覆盖全局图边界；Action goal 使用相同 mission id 和 revision。平台初始速度为零，定位状态为 `VALID`，位姿和速度协方差使用有限且低于 degraded 阈值的固定值。

规划节点参数固定为：

- map max age：`2.0 s`
- Odometry、定位状态和 TF max age：`0.5 s`
- max pairwise skew：`0.25 s`
- degraded pose/twist covariance limit：`0.5`
- maximum transform samples：`256`
- capability package：`lunar_isaac_validation`

正例共同断言：

- outcome、directive 和 `reason_code` 与锁文件一致
- `has_reference=true`、plan id 非空、platform type 正确
- result 中地图、状态 stamp 与本次输入代一致，mission revision 为 1
- 轮式和足式 reference 起点与锁定快照独立计算的离散投影起点一致；跳跃式 launch 起点与平台规划参考位姿一致；终点均落入各自目标 tolerance
- 所有数值有限，时间严格递增，速度和加速度不超过能力资料

轮式还要验证 `WHEELED_BASE` 轨迹语义和连续碰撞安全；足式验证 `LEGGED_BODY_REFERENCE`、身体高度和三轴速度区间；跳跃式要求恰好一个 hop，并独立复算飞行时间、弹道终点、发射速度、冲量、落地速度、向下接触速度和落区面积。

反例共同断言 outcome、directive 和 `reason_code` 精确相等，`has_reference=false`，且不得出现可执行 path、trajectory 或 hop。

## 构建与进程编排

构建脚本先 source `/opt/ros/humble/setup.bash` 并要求 `ROS_DISTRO=humble`、系统 Python 为 3.10。它使用仓库的 `ros2_ws/src` 和外部 package source，但通过显式 `--build-base`、`--install-base` 和 `--log-base` 将 colcon 产物写入外部根目录。

每次运行记录当前 Git commit 和 `git status --short`，但不修改、提交或清理用户工作区。旧的外部 install 目录不能作为权威输入；正式回归使用本次构建的 install overlay。

编排器只保存自己启动的 PID。正常退出先请求生命周期 shutdown 和进程 `SIGTERM`，等待限定时间后才对该明确 PID 使用 `SIGKILL`；禁止使用无目标 `pkill`、递归删除或批量清理。

## 故障模型

预检失败、快照不完整、USD 哈希变化、缺失月岩碰撞、schema 错误或场景锁不匹配属于硬失败，不能启动 Action 测试。

单个 Action 的生命周期、输入就绪、超时或断言失败必须记录后清理该会话，再继续剩余用例；最终总结果为失败。默认 lifecycle 超时为 30 秒，Action 超时为 60 秒。

Isaac Sim 在完整快照通过哈希校验后退出时，ROS 回归允许继续重放该快照，但报告 `ISAAC_DISCONNECTED_AFTER_SNAPSHOT` 警告；快照完成前断开则失败。编排器不得自动重启、关闭或保存 Isaac Sim。

退出码分组：

- `10`：环境或依赖预检
- `20`：Isaac 连接或快照
- `30`：schema、碰撞或场景资格
- `40`：ROS 构建或生命周期
- `50`：Action 结果或几何断言
- `60`：受管进程清理

## 验证与通过标准

外部包先执行纯 Python 单元测试，覆盖 manifest/NPZ schema、十层类型与布局、能力资料来源、场景锁只读行为和结果断言器。随后在外部 colcon 目录构建并运行：

1. `lunar_planner_core` 与 `lunar_planner_ros` 现有测试。
2. 三份能力资料的真实生命周期 configure/cleanup 测试。
3. 六个隔离的 `/plan_motion` Action 会话。
4. 仓库边界检查与对应 foundation 测试。

一次正式运行只有同时满足以下条件才为通过：

- 三份能力资料均可加载，六个会话都完成预期生命周期。
- 三个正例返回新引用并通过平台专属几何与动力学断言。
- 三个反例返回锁定的精确结果且不包含引用。
- 无 stale input、TF unavailable、pairwise skew 或消息布局错误。
- 同一锁定快照连续运行两次时，六个场景均生成完整归一化语义摘要，且 outcome、directive、reason code 和归一化 reference 摘要逐例一致；缺失任一摘要即失败。
- 采集前后 USD SHA-256 不变。
- 仓库内没有新增 build、install、log、快照或测试报告。
- 现有 planner 和仓库边界测试通过。

每次运行在 `artifacts/<run-id>/` 保存 `environment.json`、`snapshot_manifest.json` 副本、`scenario_results.json`、JUnit XML、受管进程日志和总摘要。报告不得包含认证令牌。

## 非目标

- 不在 Isaac Sim 中执行规划轨迹或 hop。
- 不建立平台代理关节、轮胎、足端或推进器控制器。
- 不评估真实平台动力学、传感器噪声、闭环定位或实时性能。
- 不修改冻结接口、平台迁移顺序、Nav2 适配或外部输入所有权。
- 不为满足外部回归而改变生产规划器的网格投影、起点或着陆区契约。
- 不把测试代理能力资料当作设备验收或生产能力声明。
