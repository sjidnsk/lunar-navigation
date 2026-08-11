# lunar-unreal-tcp/v1 线协议

本文冻结首版 Unreal Engine 5.0.1 与 ROS 2 Humble 之间的 TCP 字节协议。Windows Unreal
进程是 TCP 服务端，Ubuntu ROS 主机是唯一客户端；首版只允许一个 WHEELED 机器人和一个
活动会话。协议不依赖 ROS、C++ ABI、结构体 padding、Base64 或压缩。

权威实现位于 `lunar_unreal_tcp_bridge`，跨语言黄金向量位于
`tests/fixtures/unreal_tcp/protocol_v1_vectors.json`。外部 Unreal 插件必须逐字节通过这些向量，
不能另行解释字段顺序、大小端或坐标符号。

## 1. TCP 帧

每帧为 `48 byte header + UTF-8 JSON metadata + binary payload`。整数与 IEEE 754 浮点数
均为小端序。

| Offset | 长度 | 类型 | 字段 | v1 约束 |
|---:|---:|---|---|---|
| 0 | 4 | byte[4] | magic | ASCII `LNT1` |
| 4 | 2 | uint16 | protocol_version | `1` |
| 6 | 2 | uint16 | header_size | `48` |
| 8 | 2 | uint16 | message_type | 见第 2 节 |
| 10 | 2 | uint16 | flags | bit0=JSON；bit1=payload；其余为 0 |
| 12 | 8 | uint64 | sequence | 每方向、每会话严格递增，从 1 开始 |
| 20 | 8 | int64 | simulation_time_ns | Unreal 仿真时间，不得回退 |
| 28 | 4 | uint32 | metadata_length | 最大 65,536 |
| 32 | 4 | uint32 | payload_length | 与 bit1 一致 |
| 36 | 4 | uint32 | body_crc32 | metadata 原始字节再接 payload |
| 40 | 4 | uint32 | header_crc32 | 本字段置 0 后覆盖完整 48 bytes |
| 44 | 4 | uint32 | reserved | `0` |

CRC 算法固定为 CRC-32/ISO-HDLC：反射多项式 `0xEDB88320`、初值和末异或均为
`0xFFFFFFFF`。metadata 与 payload 合计不得超过 8 MiB。JSON 顶层必须是对象，重复键、
未知键、缺失键、非法 UTF-8、非有限数值、CRC 错误、长度溢出和非零 reserved 都是协议错误。

TCP 接收器必须支持任意拆包和粘包。向前跳过 sequence 合法，重复或倒序不合法。HELLO 与
HELLO_ACK 分别是各自方向的 sequence=1；重连重新从 1 开始。

## 2. 消息目录

| ID | 名称 | 方向 | payload |
|---:|---|---|---|
| `0x0001` | HELLO | ROS → Unreal | 无 |
| `0x0002` | HELLO_ACK | Unreal → ROS | 无 |
| `0x0010` | CONTROL | ROS → Unreal | 无 |
| `0x0011` | CONTROL_ACK | Unreal → ROS | 无 |
| `0x0020` | ROBOT_STATE | Unreal → ROS | 无 |
| `0x0021` | LOCAL_ELEVATION_MAP | Unreal → ROS | 固定高程和有效位图 |
| `0x0030` | MOTION_REFERENCE | ROS → Unreal | 112-byte 轨迹点数组 |
| `0x0031` | EXECUTION_FEEDBACK | Unreal → ROS | 无 |
| `0x0040` | HEARTBEAT | 双向 | 无 |
| `0x00ff` | ERROR | 双向 | 无 |

除 HELLO/HELLO_ACK 外，会话内消息都必须携带当前 UUID `session_id`。只有尚未建立会话时的
`BUSY` 或 `VERSION_UNSUPPORTED` ERROR 可以省略它。

## 3. JSON schema

实现对字段集合做封闭校验：表中没有列出的字段一律拒绝。

### 3.1 HELLO

| 字段 | 固定值或类型 |
|---|---|
| `protocol` | `lunar-unreal-tcp/v1` |
| `role` | `ROS_CLIENT` |
| `platform_type` | `WHEELED` |
| `robot_count` | `1` |
| `client_nonce` | 非空字符串 |
| `state_rate_hz` | `20` |
| `map_rate_hz` | `5` |
| `heartbeat_rate_hz` | `1` |
| `maximum_body_bytes` | `8388608` |

### 3.2 HELLO_ACK

必填字段为：

- UUID `session_id`，以及非空 `scene_id`、`robot_id`、`server_nonce`；
- `engine_version`，首版现场目标为 `5.0.1`；
- 非空 `agx_plugin_version`；不能检测时必须明确写 `UNKNOWN`；
- `coordinate_convention`、`handedness`、`up_axis`；
- 正有限数 `length_unit_to_m`；当前配置为 `0.01`；
- 两个 16 元有限数组 `T_base_link_from_unreal_root` 和
  `T_base_footprint_from_base_link`；
- `local_map_width=320`、`local_map_height=320`、`resolution_native_cm=20.0`；
- `maximum_body_bytes=8388608` 和非空 `calibration_hash`。

HELLO_ACK 创建新会话并冻结场景、机器人、单位、坐标和标定。`UNKNOWN` 插件版本可以传输，
但不能作为真实 AGX 插件资格通过的证据。

### 3.3 CONTROL 与 CONTROL_ACK

CONTROL 必填 `session_id`、非空 `command_id`、`command`。command 只允许：

- `START`：进入可接收新 reference 的 READY；
- `HOLD`：取消活动 reference 并保持位置；
- `RESUME`：从 HOLD 回到 READY，不恢复旧 reference；
- `RESET_SESSION`：取消、清空并断开，由下次握手创建新 session。

CONTROL_ACK 必填 `session_id`、`command_id`、非空 `status` 和字符串 `reason_code`。

### 3.4 ROBOT_STATE

必填：`session_id`、`position_cm[3]`、`quaternion_xyzw[4]`、
`linear_velocity_cmps[3]`、`angular_velocity_radps[3]`、非空 `control_state`。数组元素必须有限，
四元数范数必须非零。

协方差必须使用以下二选一：

- `pose_covariance[36]` 与 `twist_covariance[36]` 同时提供；或
- 非空 `simulation_covariance_profile`，由 ROS 配置解析为冻结协方差。

### 3.5 LOCAL_ELEVATION_MAP

必填：`session_id`、`width=320`、`height=320`、`resolution_cm=20.0`、
`cell_zero_center_world_cm[3]`、`u_axis_world[3]`、`v_axis_world[3]`、
`sensor_pose_world`、`robot_pose_world`、`elevation_encoding=float32_le` 和
`valid_encoding=bitset_lsb0`。两个 pose 都严格包含 `position_cm[3]` 与
`quaternion_xyzw[4]`。

payload 固定为：

| Offset | 长度 | 内容 |
|---:|---:|---|
| 0 | 409,600 | 102,400 个 row-major float32 高程，单位 cm |
| 409,600 | 12,800 | 有效位图；bit i 为第 i 格，LSB-first |

总 payload 为 422,400 bytes。无效格的高程字节必须是 `0.0f`，ROS 仍只以 valid bit 为准。
插件只能发送传感器已经观测到的单元，禁止把完整 Landscape 真值伪装成观测。

### 3.6 MOTION_REFERENCE

必填：`session_id`、非空 `plan_id`、`platform_type=WHEELED`、整数
`input_time_ns`、正整数 `point_count`、`source_frame=base_footprint`、
`target_unreal_reference=robot_root`、`execution_directive=ACTIVATE_NEW_REFERENCE` 和与握手一致的
`calibration_hash`。

ROS `MotionReference.trajectory.header.frame_id` 必须为 `odom`；`source_frame` 表示每个轨迹点
授权的是 `base_footprint` 参考点。bridge 使用冻结标定把它转换成 Unreal robot root。每个轨迹点
固定 112 bytes：

| 点内 Offset | 长度 | 类型 | 内容 |
|---:|---:|---|---|
| 0 | 8 | int64 | `time_from_start_ns` |
| 8 | 24 | float64[3] | `position_cm` |
| 32 | 32 | float64[4] | `quaternion_xyzw` |
| 64 | 24 | float64[3] | `linear_velocity_cmps` |
| 88 | 24 | float64[3] | `angular_velocity_radps` |

第一点时间必须为 0，之后严格递增；payload 长度必须等于 `point_count * 112`。Unreal 以返回
ACCEPTED 的仿真时刻作为轨迹 t=0，未 ACCEPTED 不能进入 EXECUTING。

### 3.7 EXECUTION_FEEDBACK、HEARTBEAT 与 ERROR

EXECUTION_FEEDBACK 必填 `session_id`、`platform_type=WHEELED`、`plan_id`、
`segment_id=plan_id`、从 1 递增的正整数 `sequence`、`state`、字符串 `reason_code`。state 只允许
`ACCEPTED`、`EXECUTING`、`SEGMENT_COMPLETE`、`FAILED`、`CANCELED`；后两者 reason_code 非空。

HEARTBEAT 必填 `session_id` 和无符号 `last_received_sequence`。ERROR 必填非空
`reason_code` 和布尔 `recoverable`，可选 UUID `session_id`。

## 4. 时间、队列与 fail-closed

- ROBOT_STATE 目标 20 Hz，LOCAL_ELEVATION_MAP 目标 5 Hz，HEARTBEAT 目标 1 Hz。
- `/clock`、地图、Odometry 和反馈均以 `simulation_time_ns` 为时间源。
- 网络超时只用 steady clock；3 秒心跳超时、1 秒状态/地图停滞或 0.2 秒状态/地图偏差进入 HOLD。
- 高优先级为 HOLD/RESET/ERROR，其次 reference/feedback/control ACK，再其次状态，最低为地图。
- 状态和地图只保留最新未发送帧；高优先级队列耗尽不得静默丢弃。
- wrong plan、wrong session、旧 sequence、时间回退、CRC 错误、断线均使旧 reference 失效。
- 重连必须创建新 session，不能重放旧轨迹、旧目标或旧反馈。

## 5. 坐标、标定与 TF

Unreal 保持厘米、Z-up、左手场景坐标。当前 ROS 配置使用：

```text
p_map_m = origin_map_m + diag(1,-1,1) * p_unreal_cm * 0.01
R_map = B * R_unreal * inverse(B)
```

线速度同时做轴变换和 `0.01` 缩放；角速度是轴向量，额外乘 `det(B)`。姿态必须完整做基变换，
不能只反转某个四元数分量。下行位置、姿态、线速度和角速度使用严格逆变换。

两个 4x4 标定数组按行主序表达刚体变换，末行必须为 `[0,0,0,1]`，旋转必须正交且
det=+1。ROS 启动参数、HELLO_ACK 数组和 `calibration_hash` 必须一致；会话中不得热改。

ROS TF 链固定为：

```text
map -> odom -> base_link -> base_footprint
```

Landscape Actor 的位置不是地图原点。每帧地图自己的 `cell_zero_center_world_cm` 和 u/v 轴才是
栅格落点依据。

## 6. 黄金向量与外部实现检查

权威 fixture 当前冻结两帧：

| 向量 | sequence / sim time | body CRC32 | header CRC32 |
|---|---|---:|---:|
| `hello` | 1 / 0 | 3757364800 | 1483146940 |
| `robot_state` | 7 / 123456789 | 791375228 | 3627478954 |

完整 hex 不在本文重复维护，必须直接读取
`tests/fixtures/unreal_tcp/protocol_v1_vectors.json`。Unreal 自动化测试至少做到：

1. fixture hex 解码后字段与 JSON 完全一致；
2. 从 fixture metadata 编码出的完整 bytes 与 hex 完全一致；
3. 分别篡改 header/body 一个 bit 后拒绝；
4. 以 1-byte 到整帧的不同分片方式喂入，结果一致；
5. 连续拼接两帧能够分别解码。

ROS 侧同一约束由 `lunar_unreal_tcp_bridge_protocol_test`、stream decoder、metadata、session 和
TCP client 测试覆盖；一键入口见 `scripts/run_unreal_tcp_regression.sh`。
