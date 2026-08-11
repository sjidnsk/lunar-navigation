# Unreal TCP 轮式路径规划接入设计

日期：2026-08-11

状态：已完成对话评审，等待书面规范复核

## 1. 目标

在同一局域网内连接两台独立计算机：

- Windows 计算机运行 Unreal Engine 5.0.1 与 AGX Dynamics for Unreal，不安装 ROS；
- Ubuntu 22.04 计算机运行 ROS 2 Humble、地图处理和现有 C++ v3 路径规划系统。

Unreal 只模拟传感器与单台轮式机器人，ROS 主机完成坐标转换、已观测地图融合、
障碍与占据计算、单目标滚动路径规划，并把带时间戳的轮式车体轨迹传回 Unreal 执行。

本设计追求的首个可交付闭环是：

~~~text
Unreal 已观测高程与机器人状态
  -> TCP
  -> ROS 地图与状态
  -> 单个目标位姿
  -> PlanMotion
  -> WHEELED MotionReference
  -> TCP
  -> Unreal 执行与反馈
  -> 后续滚动规划
~~~

现有 ROS 合同继续以 docs/interfaces/external-input-baseline.md、
ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml 和
ros2_ws/src/lunar_planning_msgs 为权威；部署入口参考
docs/deployment/interface-v1-quickstart.md。

## 2. 已冻结决策

| 项目 | 决策 |
|---|---|
| TCP 拓扑 | Unreal 是服务端，ROS 主机是主动连接的客户端 |
| 运维入口 | 启动、暂停、恢复、重置和目标下发均从 ROS 主机发起 |
| 连接数量 | 一个 Unreal 进程、一个机器人、一个活动 TCP 会话 |
| 首版平台 | 只支持 WHEELED |
| 目标来源 | ROS 主机通过单个 geometry_msgs/PoseStamped 目标位姿指定 |
| 地图来源 | Unreal 发送传感器已经重建出的局部高程与有效掩码 |
| 地图计算 | 占据、障碍、障碍高度、统计层和地图金字塔由 ROS 主机计算 |
| 局部图 | 64 m x 64 m、0.2 m/格、320 x 320，机器人附近滚动更新 |
| 全局图 | ROS 内部稀疏保存已观测 L0 图块，向现有规划器发布不超过约 1 km x 1 km 的活动区域 |
| 坐标 | TCP 线上保持 Unreal 原生坐标和厘米单位，ROS 入口统一转换 |
| 时间 | Unreal 仿真时间是观测时间权威；网络安全看门狗使用单调墙钟 |
| 规划边界 | 保持现有 PlanMotion、MotionReference 和 C++ v3 规划器接口 |
| 探索 | 不启动候选点、覆盖率、PPO、训练或探索策略闭环 |

## 3. 范围

### 3.1 本期包含

- Unreal Runtime C++ 插件的 TCP 服务端行为和外部项目交付要求；
- ROS 侧 TCP 客户端、协议编解码、会话与看门狗；
- Unreal 原生坐标到 ROS map/odom 坐标的统一变换；
- 已观测局部高程图的稀疏融合和 WHEELED 障碍/占据派生；
- 符合现有多分辨率契约的 local_map 与活动 global_map；
- ROS 单目标入口、PlanMotion Action 客户端和滚动执行状态机；
- MotionReference 到 Unreal 轨迹的转换与执行反馈闭环；
- 假 Unreal 服务端、Windows 插件和两机局域网验证。

### 3.2 本期不包含

- 探索候选点生成、信息增益、覆盖率、PPO 推理或训练；
- LEGGED、HOPPER 或多机器人调度；
- 一次覆盖完整约 11.34 km 场景的稠密全局图；
- 跨图块的 11 km 级稀疏全局规划器；
- Unreal 侧 ROS 运行时；
- 轮速、转矩、悬架或电机级控制协议；
- 公网部署、TLS、远程代码执行或任意脚本通道；
- 把 Unreal 项目源码、构建目录或运行 artifact 导入本仓库。

## 4. 所有权与仓库边界

Windows Unreal 项目是外部项目，拥有：

- AGX 机器人、传感器和物理执行；
- LunarTcpServer Runtime 插件实现；
- 机器人根组件到 base_link/base_footprint 的标定；
- Windows 构建、打包和插件自动化测试。

本 lunar-navigation 仓库拥有：

- TCP wire protocol 的规范和 ROS 侧黄金向量；
- lunar_unreal_tcp_bridge；
- lunar_observed_map；
- lunar_goal_coordinator；
- 现有 PlanMotion/MotionReference 适配和 ROS 集成测试。

现有 lunar_external_adapter 继续只做显式 ROS 消息转换，不承载 TCP、地图融合或路径规划。
本设计不修改 PlanMotion.action、MotionReference.msg 或冻结的外部 Topic 名称。

外部 Unreal 插件必须在其独立 Git 根实现。本仓只保存协议、ROS 端和跨端测试资料，
不得复制 Unreal 项目的 Content、Binaries、Intermediate、Saved 或 DerivedDataCache。

运行 artifact 使用：

- Ubuntu：~/CodexDownloads/lunar_navigation/unreal_tcp_path_planning
- Windows：D:/CodexDownloads/lunar_navigation/unreal_tcp_path_planning

## 5. 总体架构

~~~text
Windows / Unreal + AGX                         Ubuntu / ROS 2 Humble

传感器局部高程
机器人位姿与速度
执行状态
      |
      v
LunarTcpServer  <------ 单条 TCP ------>  lunar_unreal_tcp_bridge
      ^                                      |       |       |
      |                                      |       |       +--> /clock
轮式轨迹跟踪器                               |       +----------> Odometry / TF / feedback
      ^                                      +------------------> 原始局部高程
      |                                                               |
      +----------- WHEELED MotionReference                             v
                                                          lunar_observed_map
                                                           |             |
                                                           v             v
                                                  /environment/     /environment/
                                                    map_local         map_global
                                                           \             /
                                                            v           v
/goal_pose --> lunar_goal_coordinator --> /plan_motion --> lunar_planner
                     ^                           |
                     +------ feedback -----------+
~~~

### 5.1 Unreal LunarTcpServer

LunarTcpServer 是 Unreal Runtime 插件，不依赖 Editor 模块。它：

- 在配置的局域网 IPv4 地址与 TCP 端口监听；
- 同时只允许一个活动 ROS 客户端，第二个连接返回 BUSY 后关闭；
- 在游戏/物理线程取得不可变传感器和状态快照；
- 在专用网络线程完成序列化、收发、CRC 和队列调度；
- 向 AGX 轨迹跟踪器投递已校验的 WHEELED 车体参考；
- 发布 ACCEPTED、EXECUTING、SEGMENT_COMPLETE、FAILED 或 CANCELED；
- 在断线、看门狗超时或协议错误时使机器人进入 HOLD。

网络线程不得直接修改 Actor、Component 或 AGX 对象；接收结果必须通过线程安全队列回到
游戏/物理线程。

### 5.2 lunar_unreal_tcp_bridge

该组件是 C++20 ROS 2 LifecycleNode，使用独立 I/O 线程主动连接 Unreal。它：

- 管理连接、握手、会话、心跳、重连与协议限额；
- 发布 /clock，并分别记录仿真时间与单调墙钟接收时间；
- 把 Unreal 原生坐标、姿态和速度变换到 ROS；
- 在 /lunar/unreal/observed_elevation 发布 frame_id=map、只含 elevation 与
  valid_mask 的私有 GridMap，并发布外部接口 Odometry/TF 和执行反馈；
- 订阅 /lunar/motion_reference，并把合法 WHEELED 轨迹转换为 Unreal 原生坐标；
- 对 HOLD、RESET_SESSION 等控制命令提供 ROS 侧入口；
- 只保留最新地图与状态，不丢弃控制、轨迹、反馈和错误。

### 5.3 lunar_observed_map

该组件是独立 C++20 ROS 2 LifecycleNode。它：

- 订阅经过坐标转换的原始局部高程与有效掩码；
- 在 0.2 m L0 稀疏图块中融合已观测栅格；
- 计算占据、障碍、障碍高度和接口要求的统计层；
- 按现有保守聚合契约生成活动 global_map；
- 从 L0 稀疏存储裁剪机器人附近的 local_map；
- 保证同一代 local_map、global_map、Odometry 和 TF 可组成一致快照。

### 5.4 lunar_goal_coordinator

该组件是独立 C++20 ROS 2 LifecycleNode 和 PlanMotion Action 客户端，不是第二套规划器。
它：

- 接收 ROS 主机上的 /goal_pose；
- 校验目标位于活动、已观测和非禁行区域；
- 发布当前路径规划 mission 上下文，science_regions 为空；
- 重复请求现有 /plan_motion；
- 只有 PlanMotion 结果携带 reference 且 directive=ACTIVATE_NEW_REFERENCE 时，才把当前
  局部 reference 发布到既有 /lunar/motion_reference；
- 等待匹配执行反馈和更新后的地图/状态后再请求下一段；
- 到达目标容差、收到失败或用户取消时结束目标。

lunar_exploration_policy 和 lunar_interface_v1_policy 不参与本期启动。
因此本启动图中 /lunar/motion_reference 的唯一发布者是 lunar_goal_coordinator；
不得与现有 exploration policy 同时运行。

### 5.5 ROS 私有运维接口

首版只使用标准 ROS 2 类型：

- /lunar/unreal/start、/hold、/resume、/reset_session：
  std_srvs/srv/Trigger；
- /lunar/unreal/status：diagnostic_msgs/msg/DiagnosticArray；
- /lunar/path_planning/cancel：std_srvs/srv/Trigger；
- /lunar/path_planning/status：diagnostic_msgs/msg/DiagnosticArray；
- /goal_pose：geometry_msgs/msg/PoseStamped；
- /lunar/unreal/observed_elevation：grid_map_msgs/msg/GridMap。

这些名称属于本仓内部运维面，不修改冻结的外部输入 Topic。

## 6. base_link 与 base_footprint 适配

冻结外部接口要求 Odometry 使用 odom -> base_link，而当前 WHEELED 能力资料要求规划参考
frame 为 base_footprint。二者不能通过错误改名解决。

ROS bridge 必须由同一 Unreal 状态快照生成：

1. /localization/odometry：遵守冻结外部接口，child_frame_id=base_link；
2. /lunar/unreal/wheeled_odometry：应用已标定的刚体变换后，
   child_frame_id=base_footprint；
3. TF：map -> odom -> base_link，以及静态 base_link -> base_footprint。

WHEELED 启动文件将 lunar_planner 的绝对 /localization/odometry 订阅 remap 到
/lunar/unreal/wheeled_odometry。这样保持外部接口不变，同时满足当前规划器按 capability
base_frame_id 校验 Odometry 的要求。

MotionReference 进入 Unreal 时，同样使用已冻结的 base_link/base_footprint/Unreal 根组件
刚体标定，不能把三个参考点当作同一个点。标定缺失、非刚体或哈希与握手声明不一致时，
会话停留在 SYNCING，不允许规划。

对于 WHEELED，活动 capability profile 中的 reference_point=base_footprint 是轨迹几何
语义权威。当前兼容 MotionReference 中的 trajectory.joint_names 标签不得被 bridge
错误解释为改变参考点；bridge 始终先按 base_footprint 几何解释，再用已冻结标定转换到
Unreal 根组件。

## 7. TCP 拓扑与生命周期

### 7.1 地址

- Unreal 参数 listen_address 必须显式配置为仿真机的局域网 IPv4；
- 默认端口为 47001，可在两端配置中一致修改；
- ROS 参数 server_host 必须显式填写，不从广播自动发现；
- ROS 参数 server_port 默认 47001；
- TCP_NODELAY 和操作系统 keepalive 开启，但安全判定以应用心跳为准。

Unreal 不绑定公网接口。Windows 防火墙仅允许已配置 ROS 主机 IP 访问该端口。

### 7.2 启动顺序

1. Unreal 启动并在 HOLD 状态监听；
2. lunar_unreal_tcp_bridge configure/activate，按 0.5 s 到 5 s 有上限退避重连；
3. HELLO/HELLO_ACK 成功后进入 SYNCING；
4. 收到有效 ROBOT_STATE 和 LOCAL_ELEVATION_MAP 后激活地图输出；
5. 地图、Odometry 与 TF 满足新鲜度后，lunar_goal_coordinator 进入 READY；
6. 用户才可提交目标。

Unreal RESET_SESSION、场景切换或仿真时间倒退都会结束当前 session。ROS 取消活动目标、
清空当前地图代次并重新同步，禁止把旧会话数据拼入新场景。

## 8. Wire protocol v1

### 8.1 帧布局

协议名为 lunar-unreal-tcp/v1。每帧由固定 48 byte 帧头、UTF-8 JSON 元数据和可选二进制
payload 组成。所有整数和浮点数均为 IEEE 754/二进制小端序；实现不得通过直接发送原生
C++ struct 依赖编译器 padding。

| Offset | 长度 | 字段 |
|---:|---:|---|
| 0 | 4 | magic，ASCII LNT1 |
| 4 | 2 | protocol_version，固定为 1 |
| 6 | 2 | header_size，固定为 48 |
| 8 | 2 | message_type |
| 10 | 2 | flags |
| 12 | 8 | sequence |
| 20 | 8 | simulation_time_ns，有符号 |
| 28 | 4 | metadata_length |
| 32 | 4 | payload_length |
| 36 | 4 | body_crc32 |
| 40 | 4 | header_crc32 |
| 44 | 4 | reserved，必须为 0 |

CRC 使用 CRC-32/ISO-HDLC。body_crc32 覆盖元数据原始字节后紧接 payload；
header_crc32 在自身四字节置零后覆盖完整 48 byte 帧头。

硬限额：

- metadata_length 不超过 65,536 bytes；
- metadata_length + payload_length 不超过 8 MiB；
- JSON 顶层必须为对象；重复键、非法 UTF-8、缺失必填字段、未知字段和非有限数值均拒绝；
- 未知 flags、非零 reserved、长度溢出或 CRC 不匹配立即结束当前会话。

TCP 解码器必须支持任意拆包和粘包。每个方向的 sequence 在新会话内从 1 开始严格递增；
重复、倒序和跳回序号均拒绝；因“仅保留最新地图/状态”而产生的向前跳号合法。

flags 的 bit 0 表示存在 JSON 元数据，bit 1 表示存在二进制 payload，其余 bit 在 v1
必须为 0。所有 v1 消息都设置 bit 0；只有携带二进制 body 的消息设置 bit 1。

### 8.2 消息类型

| ID | 名称 | 方向 | 作用 |
|---:|---|---|---|
| 0x0001 | HELLO | ROS -> Unreal | 提议协议、客户端 nonce、平台与频率 |
| 0x0002 | HELLO_ACK | Unreal -> ROS | 建立 session 并冻结场景、机器人、坐标和标定摘要 |
| 0x0010 | CONTROL | ROS -> Unreal | START、HOLD、RESUME、RESET_SESSION |
| 0x0011 | CONTROL_ACK | Unreal -> ROS | 控制命令结果 |
| 0x0020 | ROBOT_STATE | Unreal -> ROS | 位姿、速度、协方差来源 |
| 0x0021 | LOCAL_ELEVATION_MAP | Unreal -> ROS | 高程和有效掩码 |
| 0x0030 | MOTION_REFERENCE | ROS -> Unreal | 当前唯一授权 WHEELED 轨迹 |
| 0x0031 | EXECUTION_FEEDBACK | Unreal -> ROS | 轨迹接收、执行和终态 |
| 0x0040 | HEARTBEAT | 双向 | 活性、会话和最近收发序号 |
| 0x00ff | ERROR | 双向 | 稳定原因码和可恢复性 |

HELLO_ACK 生成新的 UUID session_id。除 HELLO/HELLO_ACK 外，所有会话内消息元数据都
必须携带该 session_id；重连不复用 session_id。只有在握手建立 session 之前拒绝第二客户端
或不支持版本时，ERROR 才可以省略 session_id，并分别使用 BUSY 或 VERSION_UNSUPPORTED。

### 8.3 HELLO/HELLO_ACK

HELLO 至少携带：

- protocol=lunar-unreal-tcp/v1；
- role=ROS_CLIENT；
- platform_type=WHEELED；
- robot_count=1；
- client_nonce；
- state_rate_hz=20；
- map_rate_hz=5；
- heartbeat_rate_hz=1；
- maximum_body_bytes=8388608。

HELLO_ACK 至少携带：

- session_id、scene_id、robot_id；
- engine_version=5.0.1 和实际 AGX 插件版本字符串；
- coordinate_convention、length_unit_to_m、handedness 和 up_axis；
- T_base_link_from_unreal_root 与 T_base_footprint_from_base_link 的摘要；
- local_map_width=320、local_map_height=320、resolution_native=20 cm；
- server_nonce 和协议限额。

AGX 插件版本未知不是握手失败条件，但必须传输实际检测结果或明确的 UNKNOWN，并写入诊断。

CONTROL 语义固定为：

- START：从初始 HOLD 进入可接收 reference 的 READY，不产生运动；
- HOLD：取消活动 reference，发布 CANCELED 后保持位置；
- RESUME：只从 HOLD 回到 READY，不恢复被取消的旧 reference；
- RESET_SESSION：取消 reference、清空会话状态并关闭连接，由下一次握手创建新 session。

### 8.4 ROBOT_STATE

ROBOT_STATE 使用 JSON 元数据承载：

- Unreal world 中机器人根组件的位置与单位四元数；
- Unreal world 表达的线速度和角速度；
- 机器人控制状态；
- pose/twist covariance，或 simulation_covariance_profile 的标识。

若 Unreal 不提供估计协方差，ROS bridge 使用启动配置中显式冻结的仿真协方差 profile；
不得静默生成 NaN、负对角线或无限确定性声明。

### 8.5 LOCAL_ELEVATION_MAP

元数据至少包含：

- width=320、height=320、resolution_cm=20.0；
- cell_zero_center_world_cm；
- u_axis_world、v_axis_world；
- sensor_pose_world 与 robot_pose_world；
- elevation_encoding=float32_le；
- valid_encoding=bitset_lsb0。

payload 固定为：

1. 102,400 个 row-major float32 高程，单位厘米；
2. 12,800 byte 有效位图，bit i 对应第 i 个高程单元。

有效单元高程必须有限；无效单元的高程字节固定编码为 0.0，但 ROS 只依据 valid bit 使用。
帧只包含传感器已观测内容，不能包含未观测 Landscape 真值。

### 8.6 MOTION_REFERENCE

元数据至少包含：

- session_id、plan_id、platform_type=WHEELED；
- input_time_ns；
- point_count；
- source_frame 和 target_unreal_reference；
- execution_directive；
- calibration_hash。

lunar_goal_coordinator 只发布 ACTIVATE_NEW_REFERENCE 的结果，因此 TCP v1 的
execution_directive 固定为 ACTIVATE_NEW_REFERENCE。HOLD 使用 CONTROL，不用空轨迹表达。

每个 payload 轨迹点固定为：

- int64 time_from_start_ns；
- float64 position_cm[3]；
- float64 quaternion_xyzw[4]；
- float64 linear_velocity_cmps[3]；
- float64 angular_velocity_radps[3]。

每条记录固定 112 bytes。time_from_start_ns 从 0 开始严格递增。Unreal 返回 ACCEPTED
的仿真时刻定义为本轨迹 t=0；
未返回 ACCEPTED 前不能进入 EXECUTING。首版一次只缓存一个授权 reference。

### 8.7 EXECUTION_FEEDBACK

反馈严格映射现有 MotionExecutionFeedback：

- platform_type=WHEELED；
- plan_id；
- segment_id=plan_id；
- sequence 在该 plan_id 内从 1 开始递增；
- state 为 ACCEPTED、EXECUTING、SEGMENT_COMPLETE、FAILED 或 CANCELED；
- FAILED/CANCELED 必须携带非空 reason_code。

反馈不替代 ROBOT_STATE。SEGMENT_COMPLETE 只有与更新后的位姿和地图共同出现时，才能触发
下一次滚动规划。

## 9. 时间、频率与队列

### 9.1 时间

- Unreal simulation_time_ns 是 ROBOT_STATE、地图和执行反馈的权威时间；
- ROS bridge 从其发布 /clock，相关 ROS 节点统一 use_sim_time=true；
- 同一 session 内 simulation_time_ns 不得倒退；
- 网络连接和安全超时只使用 std::chrono::steady_clock；
- ROS 额外记录 receive_steady_time，用于诊断网络停滞。

### 9.2 频率

- ROBOT_STATE：20 Hz；
- LOCAL_ELEVATION_MAP：5 Hz；
- HEARTBEAT：1 Hz；
- CONTROL、MOTION_REFERENCE、EXECUTION_FEEDBACK 和 ERROR：事件触发。

### 9.3 背压

发送与接收分别使用有界优先级队列：

1. HOLD/RESET/ERROR；
2. MOTION_REFERENCE/EXECUTION_FEEDBACK/CONTROL_ACK；
3. ROBOT_STATE；
4. LOCAL_ELEVATION_MAP。

地图和状态槽只保留最新一帧。高优先级消息达到队列限额属于不可恢复会话错误，必须 HOLD
并断开，不能静默丢弃。首版不使用 Base64 或压缩；5 Hz 地图的名义吞吐约 2.1 MiB/s。

## 10. 坐标与 TF

### 10.1 基本原则

Unreal 端保持场景原生坐标和厘米单位。ROS bridge 配置并在 HELLO_ACK 后冻结
T_map_from_unreal_world。该变换包含：

- Unreal world 原点在 ROS map 中的位置；
- 单位缩放，默认 0.01 m/cm；
- 左右手系和轴向变换；
- 项目需要的水平 yaw 对齐。

位置、姿态、线速度、角速度、栅格原点、栅格基向量和下行轨迹都由同一刚体基变换处理。
姿态使用旋转矩阵/四元数的完整基变换，禁止只靠手写某一个分量取反。

Landscape Actor 位置 (-205160, -511559, 50) cm 只属于场景元数据，不能被假定为高程图
左下角。每个局部图必须给出 cell_zero_center_world_cm 与 u/v 基向量。

### 10.2 TF 链

ROS 对外保持：

~~~text
map -> odom -> base_link -> base_footprint
~~~

map -> odom 表达会话内全局与连续里程计关系；odom -> base_link 来自机器人状态；
base_link -> base_footprint 是会话内不变的已标定静态变换。

## 11. 已观测地图与障碍计算

### 11.1 稀疏 L0 存储

- 基础分辨率固定 0.2 m；
- 每个稀疏图块为 256 x 256，物理尺寸 51.2 m x 51.2 m；
- 仅在至少一个有效观测落入图块时分配；
- 每格保存 valid、elevation、elevation_variance、observation_count、
  last_observed_time、obstacle 和 obstacle_height；
- 使用在线均值/方差更新有效观测；
- 无效输入不得擦除已有有效观测；
- 新 session/scene 不继承旧图块。

首版会话最多保留 512 个 L0 图块，不在会话内静默遗忘已观测图块。下一帧若需要分配第
513 个图块，返回 MAP_TILE_BUDGET_EXCEEDED，拒绝扩大活动区域并保持当前已知地图。

### 11.2 占据与障碍

内部占据状态固定为 UNKNOWN、FREE、OCCUPIED：

- valid=0 -> UNKNOWN；
- valid=1 且未被判为离散障碍 -> FREE；
- valid=1 且局部支撑面残差或不连续台阶超过 WHEELED
  maximum_local_obstacle_relief_m -> OCCUPIED。

首版使用当前已冻结 WHEELED profile 的 maximum_local_obstacle_relief_m=0.20 m。
obstacle_height 是相对局部支撑面的正向高度；负台阶或沟槽仍可令 obstacle=1，但高度至少按
当前 resolution 提供保守占位。

坡度和粗糙度保留在高程邻域中，由现有规划器按照 maximum_slope_rad 等 capability 字段
判定。地图生产方发布原始障碍单元，不再次按车体半径膨胀；现有 safe projection 继续负责
footprint_xy_m、minimum_clearance_m 和安全边界，避免双重膨胀。

forbidden 只来自 ROS 显式禁行配置。UNKNOWN 通过 valid_mask 表达，不能伪装成 forbidden。
没有进入传感器高程图的悬空物、动态物或垂直结构不属于首版已知障碍。

### 11.3 接口图层

发布给规划器的 GridMap 必须含：

- elevation；
- valid_mask；
- obstacle；
- obstacle_height；
- observation_age_s；
- observation_quality；
- elevation_variance；
- obstacle_variance；
- observation_count；
- forbidden。

首版传感器未提供独立质量时，单次有效样本的 observation_quality=1.0，无效单元为 0；
方差、次数和年龄仍按实际融合状态发布。obstacle_variance 对 obstacle 单元取其局部支撑
邻域的最大 elevation_variance，对有效非障碍单元取 0；无效单元由 valid_mask 屏蔽。

### 11.4 local_map

- frame_id=odom；
- 分辨率固定 0.2 m；
- 物理范围固定 64 m x 64 m；
- 以当前 WHEELED base_footprint 附近为中心；
- 从稀疏 L0 存储裁剪并重采样；
- 只包含实际已观测证据。

### 11.5 active global_map

没有活动目标时，地图节点发布以机器人附近为中心、由当前已观测图块界定的初始
global_map，使系统能够进入 READY。

活动全局区域必须同时覆盖：

- 当前机器人；
- 当前单目标；
- 二者之间已观测连通走廊；
- 规划安全边界。

单轴物理范围不得超过约 1,024 m。区域按 external-input-baseline 的
lunar-conservative-grid-aggregation/v1 选择满足预算的最精细层级：
0.2、0.4、0.8、1.6、3.2 或 4.0 m。

聚合继续使用 valid AND、obstacle/forbidden OR、最大障碍高度/年龄/方差、最小质量/次数
和有效子单元高程均值。目标超出活动区域、目标未观测或起终点不存在已知连通证据时，
lunar_goal_coordinator 在调用 PlanMotion 前拒绝。

lunar_observed_map 与 lunar_goal_coordinator 同时订阅 /goal_pose。地图节点先按该消息的
header.stamp 尝试生成覆盖机器人、目标和已观测走廊的新 active global_map；协调器必须等到
收到时间不早于该目标、边界确实包含目标的 global_map 后才做目标有效性校验并调用
PlanMotion。若目标对应 L0 栅格从未观测，地图仍以 valid_mask=0 表达并由协调器拒绝。

## 12. 单目标滚动规划

### 12.1 目标入口

/goal_pose 使用 geometry_msgs/PoseStamped：

- frame_id 必须为 map；
- 位置与四元数必须有限，四元数可归一化；
- XY 必须位于当前 active global_map；
- 对应目标必须有效、非 obstacle、非 forbidden；
- 目标 Z 从已观测高程填写；
- 默认位置容差为 0.5 m；
- 默认启用 yaw 约束，yaw 容差为 15 度。

同一时刻只允许一个活动目标。执行中提交新目标不会隐式替换；调用方必须显式取消，
等待旧 reference 进入 CANCELED/HOLD 后再提交。

### 12.2 mission 兼容

现有 PlanMotion 需要 mission_id 与 mission_revision。lunar_goal_coordinator 为路径规划
会话发布现有 ExplorationTask schema 的最小兼容消息：

- mission_id 为当前 TCP session 与目标 UUID 的组合；
- revision 从 1 开始严格递增；
- ROI 等于活动全局区域；
- science_regions 为空；
- desired_state=ExplorationTask.ACTIVE。

这只是现有规划输入合同的兼容上下文，不启动探索、候选点或覆盖率逻辑。

### 12.3 滚动流程

1. READY 时接收目标；
2. 地图节点以目标 header.stamp 生成活动 global_map；
3. 协调器等待包含目标的同代 global_map、local_map、Odometry 和 TF；
4. 构造 PlanMotion.Goal，replace_active_request=false；
5. 规划成功且 directive=ACTIVATE_NEW_REFERENCE 时，协调器发布当前 MotionReference；
6. bridge 把该 reference 作为 TCP ACTIVATE_NEW_REFERENCE 发送；
7. Unreal 依次确认 ACCEPTED 与 EXECUTING；
8. 收到匹配 SEGMENT_COMPLETE；
9. 等待严格晚于完成反馈的地图和机器人状态；
10. 未进入目标容差则以同一最终目标发起下一次 PlanMotion；
11. 进入容差后发送 HOLD 并报告 SUCCEEDED。

NO_KNOWN_SAFE_ROUTE、GOAL_INFEASIBLE、STALE_INPUT、RESOURCE_EXHAUSTED、FAILED 或 CANCELED
均不产生空轨迹；机器人保持 HOLD，并向 ROS 诊断返回稳定原因码。

## 13. 状态机与故障安全

~~~text
DISCONNECTED -> HANDSHAKING -> SYNCING -> READY
                                            |
                                            v
                                        EXECUTING
                                            |
                          +-----------------+-----------------+
                          v                 v                 v
                        READY           SUCCEEDED            HOLD
~~~

强制 HOLD 条件：

- TCP 关闭或连续 3 s 没有合法心跳；
- 协议版本、CRC、长度、sequence 或 session 不合法；
- 地图或状态超过 1 s；
- 同一规划快照的地图/状态时间差超过 0.2 s；
- 轨迹 platform_type 不是 WHEELED；
- plan_id 空、重复、非当前授权或 feedback 身份不匹配；
- 轨迹时间不单调、数值非有限或坐标/标定哈希不匹配；
- Unreal 执行器返回 FAILED/CANCELED；
- ROS lifecycle deactivate、场景 reset 或仿真时间倒退。

重连总是创建新 session。旧会话的目标、reference、序号、地图代次和反馈全部失效，
不得自动恢复断线前轨迹。

## 14. 性能与安全边界

- ROBOT_STATE 目标接收频率 20 Hz，验收下限 18 Hz；
- LOCAL_ELEVATION_MAP 目标接收频率 5 Hz，验收下限 4.5 Hz；
- 有线局域网 MOTION_REFERENCE 到 ACCEPTED 的 P95 不高于 200 ms；
- 连续运行 30 min 不得出现无界队列、内存持续增长或控制消息被地图饿死；
- TCP 断开/心跳丢失后 3 s 内 Unreal 必须 HOLD；
- 端口只对指定 ROS 主机开放，不得暴露公网；
- 本协议不提供认证或保密，因此仅允许受控实验室 LAN。跨不可信网络必须另行设计 TLS
  或 PSK，不得把明文 token 冒充安全认证。

## 15. 测试设计

### 15.1 协议单元测试

- C++ ROS codec 与 Unreal codec 使用相同黄金字节；
- header、CRC、JSON、大小端、有限值和长度边界；
- 每个消息类型的正例和字段缺失反例；
- 任意 TCP 拆包、粘包、半帧和连续多帧；
- duplicate、out-of-order、wrong-session 和 reconnect；
- 8 MiB 上限、整数溢出和恶意长度。

### 15.2 坐标测试

- Unreal 原点、单位轴、任意 yaw、负坐标和 Landscape 给定位置；
- 位置、四元数、线速度、角速度的往返黄金向量；
- base_link/base_footprint/Unreal root 三者标定；
- 栅格 cell_zero 与 u/v 轴映射；
- 下行 MotionReference 再转换回 ROS 后与原轨迹误差有界。

### 15.3 地图测试

- 320 x 320、0.2 m 输入布局；
- valid bit 与 row-major 索引；
- 无效输入不覆盖有效历史；
- 图块边缘与跨图块融合；
- 在线均值、方差、次数和年龄；
- 0.20 m 障碍 relief 阈值两侧；
- UNKNOWN/FREE/OCCUPIED 和 forbidden 独立性；
- 512 图块预算；
- local_map 裁剪；
- global_map L0-L5 选择与保守聚合；
- 未观测真值永不出现在输出图层。

### 15.4 ROS 集成测试

仓库内提供 fake Unreal TCP server，自动验证：

- lifecycle 启停、握手、重连与 /clock；
- 原始地图到最终十层 GridMap；
- 外部 base_link Odometry 与私有 WHEELED base_footprint Odometry；
- /goal_pose 到最小 mission 和 PlanMotion；
- MotionReference 下行和执行反馈上行；
- 只有匹配 SEGMENT_COMPLETE 加更新快照才触发下一段；
- stale、skew、wrong plan、wrong session 和断线均 fail closed；
- 启动图中不存在探索 policy、PPO 或训练节点。

### 15.5 Windows/Unreal 测试

- Unreal Engine 5.0.1 Editor 与 packaged build 都能加载 Runtime 插件；
- 监听配置、单客户端限制和 Windows 防火墙说明；
- 网络线程不直接访问游戏线程对象；
- AGX 轮式执行器完成 ACCEPTED/EXECUTING/SEGMENT_COMPLETE；
- HOLD、重置、断线和心跳超时；
- 实际 AGX 插件版本被握手与日志记录。

### 15.6 两机验收

至少执行：

1. 平坦已观测区域直达目标；
2. 已观测离散障碍区域绕行；
3. 未知、活动区外或不可达目标被拒绝且机器人保持 HOLD；
4. 执行中拔断网络，3 s 内 HOLD，重连后不重放旧轨迹。

最终位置必须进入目标请求的 0.5 m 容差；如启用 yaw，进入 15 度容差。

## 16. 仓库验证

设计和后续实现至少保持：

~~~bash
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
~~~

实施阶段还必须运行新协议、地图、ROS 集成测试和现有受影响的 planner/interface-v1 回归。
ROS 命令在 Ubuntu 22.04 上先 source /opt/ros/humble/setup.bash，并把 build/install/log
全部定向到仓库外。

Windows Unreal 插件的实际编译开始前，必须在外部 Unreal 项目中确认项目根路径、AGX
插件安装位置和实际版本；缺少该外部 checkout 不阻塞仓库内协议、fake server 和 ROS 侧
实现，但会阻塞 Windows 插件构建与两机验收。

## 17. 完成定义

本期只有同时满足以下条件才算完成：

- ROS 主机能够主动连接 Windows Unreal；
- 持续接收传感器局部高程、位姿和速度；
- ROS 只用已观测数据生成合法 local_map/global_map 和障碍/占据；
- 用户从 ROS 主机发送一个目标位姿；
- 现有 WHEELED PlanMotion 产生轨迹并由 Unreal 执行；
- 匹配反馈驱动至少两段滚动规划或直接到达目标；
- 三个功能案例、断线案例和 30 min 稳定性通过；
- 未启动探索、覆盖率、PPO 或训练；
- 没有改变 PlanMotion.action、MotionReference.msg 和冻结外部 Topic 名；
- Unreal 外部项目源码与运行 artifact 未进入本仓。
