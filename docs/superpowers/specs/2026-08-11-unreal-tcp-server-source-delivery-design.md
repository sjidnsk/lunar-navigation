# Unreal TCP 服务端源码交付设计

日期：2026-08-11
状态：待用户书面审阅
适用范围：Unreal Engine 5.0.1、Windows、单台 WHEELED、`lunar-unreal-tcp/v1`

## 1. 背景与目标

`lunar-navigation` 已实现 ROS 2 Humble 客户端、观测地图融合、单目标滚动规划和仓库内
fake Unreal server，但真实 Windows Unreal Runtime 服务端仍为 `EXTERNAL_UNREAL_PENDING`。

本设计补齐一个可独立交付的纯源码 Runtime 插件。源码包能够直接复制到：

```text
<UnrealProject>/Plugins/LunarTcpServer/
```

然后由 Unreal Engine 5.0.1 的 Windows 工具链本机编译。首个交付同时满足：

1. 不要求 Windows 安装 ROS；
2. 不依赖具体 AGX C++ 版本即可编译通用服务端；
3. 本机回环即可验证插件加载、监听、握手、协议错误处理和断线 HOLD；
4. 通过稳定适配接口接入实际 AGX 机器人、传感器和轨迹执行；
5. Unreal 源码保持独立 Git 根，不进入 `lunar-navigation` 仓库。

## 2. 已选方案与取舍

### 2.1 采用：独立 Runtime 插件加项目适配器

服务端拆成两层：

- 通用层负责 TCP、帧、CRC、JSON schema、会话、心跳、队列、节流和 watchdog；
- 项目适配层负责从真实 Unreal/AGX 对象取得状态与已观测高程，并执行 WHEELED reference。

插件通过 `ILunarTcpRobotAdapter` 与项目代码连接。网络线程只处理普通值对象，不访问
`UObject`、Actor、Component 或 AGX 对象；适配器只能在游戏线程调用。

该方案允许先在未知 AGX 版本的情况下编译、加载和验证协议核心，之后只补一个项目侧适配器，
不重写网络协议。

### 2.2 不采用：把 Unreal 源码放入 lunar-navigation

这会破坏仓库的单 Git 根和外部项目所有权边界，还会把 Windows/Unreal 构建职责混入 Ubuntu
ROS 仓库。因此 `lunar-navigation` 只继续保存协议、黄金向量、跨端说明和资格状态。

### 2.3 不采用：直接复用飞跃器 TCP 协议

飞跃器示例是 Python 客户端主动连接 Unreal 服务端，协议只有长度前缀、flag 和定长 float。
它没有 `lunar-unreal-tcp/v1` 所需的 CRC、封闭 metadata、session、sequence、心跳、重连失效、
单客户端仲裁和 fail-closed 语义。它只能作为 Unreal Socket 生命周期参考，不能作为线协议基础。

## 3. 仓库与交付位置

实现时创建独立 Git 根：

```text
/mnt/data/WS/lunar-unreal-tcp-server
```

Ubuntu 生成的可传输源码包写入仓库外：

```text
~/CodexDownloads/lunar_navigation/unreal_tcp_path_planning/unreal-server-source/
```

源码包不得包含：

```text
Binaries/
Intermediate/
DerivedDataCache/
Saved/
.vs/
*.sln
*.dll
*.pdb
```

交付 ZIP 必须附带：

- 插件源码；
- 默认回环配置；
- `lunar-unreal-tcp/v1` 黄金向量快照；
- 来源 commit 与 SHA-256 manifest；
- Windows 复制、生成工程、编译和测试说明。

## 4. 源码结构

```text
lunar-unreal-tcp-server/
├── AGENTS.md
├── CMakeLists.txt
├── README.md
├── LunarTcpServer/
│   ├── LunarTcpServer.uplugin
│   ├── Config/
│   │   └── DefaultLunarTcpServer.ini
│   ├── Source/
│   │   └── LunarTcpServer/
│   │       ├── LunarTcpServer.Build.cs
│   │       ├── Public/
│   │       │   ├── LunarTcpRobotAdapter.h
│   │       │   ├── LunarTcpServerSettings.h
│   │       │   ├── LunarTcpServerSubsystem.h
│   │       │   └── LunarTcpTypes.h
│   │       └── Private/
│   │           ├── LunarTcpConnectionWorker.cpp
│   │           ├── LunarTcpConnectionWorker.h
│   │           ├── LunarTcpProtocol.cpp
│   │           ├── LunarTcpProtocol.h
│   │           ├── LunarTcpServerModule.cpp
│   │           ├── LunarTcpServerSettings.cpp
│   │           ├── LunarTcpServerSubsystem.cpp
│   │           └── Tests/
│   │               └── LunarTcpProtocolAutomationTest.cpp
│   └── Tests/
│       └── fixtures/
│           └── protocol_v1_vectors.json
├── native/
│   ├── include/lunar_tcp/protocol_core.hpp
│   ├── src/protocol_core.cpp
│   └── tests/protocol_core_test.cpp
├── scripts/
│   ├── package_source.sh
│   ├── verify_source_package.py
│   └── windows_smoke_test.ps1
└── docs/
    ├── windows-build-and-test.md
    └── superpowers/
        ├── specs/
        └── plans/
```

`native/` 是不依赖 Unreal 的协议内核，使用同一份帧布局、CRC 和流解码逻辑；Unreal 模块通过
薄适配调用它。Ubuntu 可以用 CMake/CTest 验证该内核，但不能把这一结果写成 UE 5.0.1 编译通过。

## 5. Unreal 模块边界

### 5.1 `ULunarTcpServerSubsystem`

使用 Runtime `UGameInstanceSubsystem` 管理服务生命周期：

- 游戏实例建立后按配置启动服务；
- 初始控制状态固定为 `HOLD`；
- 每个游戏线程 tick 拉取适配器状态、提交最新不可变快照并派发网络命令；
- PIE/游戏实例退出时先 HOLD、停止 worker、关闭 socket 并 join 线程；
- 不使用 Editor-only API。

服务状态至少公开：`STOPPED`、`LISTENING`、`HANDSHAKING`、`SYNCING`、`READY`、`HOLD`、
`FAULTED`。

### 5.2 `ILunarTcpRobotAdapter`

项目侧实现并向 subsystem 注册唯一适配器。接口固定为：

```cpp
virtual bool CaptureRobotState(FLunarRobotStateSnapshot& OutState) = 0;
virtual bool CaptureObservedElevation(FLunarElevationSnapshot& OutMap) = 0;
virtual void ExecuteMotionReference(const FLunarMotionReference& Reference) = 0;
virtual void CancelMotion(const FString& ReasonCode) = 0;
virtual void EnterHold(const FString& ReasonCode) = 0;
```

约束：

- 所有调用发生在游戏线程；
- `CaptureObservedElevation` 只返回传感器已观测单元，未知区域的 valid bit 必须为 0；
- 适配器不得把完整 Landscape 真值伪装成传感器观测；
- `ExecuteMotionReference` 不得在旧 session 或 HOLD 状态执行；
- 未注册适配器时服务仍可监听和握手，但保持 `HOLD`，并以稳定原因码拒绝 `START` 和 reference。

### 5.3 网络 worker

专用 `FRunnable` 线程拥有 listen/client socket、流解码器和发送队列。它：

- 最多允许一个活动客户端；
- 对第二个客户端发送 `BUSY` 后关闭；
- 不读取或写入任何 `UObject`；
- 只在有界队列中交换普通值对象；
- 控制、reference、feedback 和错误不得按 latest-only 丢弃；
- 状态和地图队列只保留最新快照；
- 停止时关闭 socket 解除阻塞，再 join worker。

## 6. 协议与状态机

线协议严格复用
[`lunar-unreal-tcp/v1`](../../interfaces/lunar-unreal-tcp-v1.md)，不得为 Unreal 端另建方言。

### 6.1 帧约束

- 48-byte 小端帧头；
- magic `LNT1`、version `1`、reserved `0`；
- CRC-32/ISO-HDLC；
- metadata 最大 65,536 bytes；
- body 最大 8 MiB；
- 任意拆包和粘包；
- 重复、倒序 sequence、CRC 错误、未知字段和超限长度均 fail closed。

### 6.2 建连

```text
LISTENING
  -> 接收合法 HELLO(sequence=1)
  -> 创建新 UUID session
  -> 返回 HELLO_ACK(sequence=1)
  -> SYNCING
  -> 适配器状态和地图均有效
  -> 等待 CONTROL START
  -> READY
```

每次 TCP 重连都创建新 session。旧 session 的 CONTROL、reference 和 feedback 不得重放或接受。

### 6.3 频率与时间

- ROBOT_STATE：目标 20 Hz；
- LOCAL_ELEVATION_MAP：目标 5 Hz；
- HEARTBEAT：1 Hz；
- `simulation_time_ns` 来自 Unreal 仿真时钟，并在会话内单调不回退；
- 状态/地图生产慢时不得发送伪造重复仿真时间的数据。

### 6.4 HOLD 与 watchdog

以下任一事件必须通过游戏线程命令使适配器进入 HOLD：

- TCP 断开；
- 3 秒未收到客户端心跳或合法流量；
- 帧、CRC、schema、session 或 sequence 协议错误；
- `CONTROL HOLD` 或 `RESET_SESSION`；
- worker/队列不可恢复错误；
- PIE/游戏实例结束。

网络线程不得直接调用 `EnterHold`。它只入队高优先级 HOLD 命令，由下一个游戏线程 tick 执行。

## 7. 配置与安全默认值

源码交付默认只允许本机测试：

```ini
[/Script/LunarTcpServer.LunarTcpServerSettings]
bAutoStart=True
ListenAddress=127.0.0.1
ListenPort=47001
HeartbeatTimeoutSeconds=3.0
StateRateHz=20
MapRateHz=5
MaximumBodyBytes=8388608
```

两机联调前必须在外部 Unreal 项目配置中把 `ListenAddress` 改为 Windows 实际局域网 IPv4，
并用 Windows Private 防火墙规则只允许 ROS 主机 IP。协议没有 TLS 和身份认证，禁止绑定公网
接口或创建 `Any` 来源的公网入站规则。

## 8. 测试设计

### 8.1 Ubuntu 原生协议测试

CTest 必须覆盖：

- 权威 HELLO 黄金向量逐字节解码与重编码；
- CRC 单 bit 篡改拒绝；
- 1 byte 到整帧的拆包；
- 两帧粘包；
- metadata/body 限额；
- sequence 重复与倒序拒绝；
- reconnect 后旧 session 失效的纯状态机行为。

该测试证明跨平台协议内核，不证明 Unreal 模块能被 UBT 编译。

### 8.2 Unreal Automation 测试

Windows UE 5.0.1 必须运行：

- 插件 Runtime 模块在 Editor 和无 Editor 依赖的目标中加载；
- Unreal codec 与黄金向量 bytes 完全一致；
- loopback listener 启停和端口释放；
- 第二客户端 BUSY；
- network worker 不在网络线程触碰 UObject 的结构审查；
- 非法 CRC、错误 session 和断连触发游戏线程 HOLD。

### 8.3 Windows 本机 smoke

`windows_smoke_test.ps1` 必须：

1. 确认 47001 的 OwningProcess 是 `UnrealEditor` 或 packaged executable；
2. 发送权威 HELLO，校验返回 `LNT1`、message type `HELLO_ACK` 和 CRC；
3. 记录 session、engine version、AGX version、calibration hash；
4. 主动断开并确认服务继续存活且控制状态为 HOLD；
5. 停止 PIE 后确认端口释放。

### 8.4 AGX 项目适配测试

实际项目适配器到位后才执行：

- 20 Hz 位姿/速度；
- 5 Hz 仅已观测高程与 valid mask；
- reference 的 ACCEPTED、EXECUTING、SEGMENT_COMPLETE；
- 断开客户端 3 秒内 HOLD；
- 重连不重放旧 reference。

在取得实际 `.uproject`、AGX 插件版本和机器人组件 API 前，这组测试保持明确的
`EXTERNAL_AGX_ADAPTER_PENDING`，不得由 mock 结果关闭。

## 9. Windows 复制与编译流程

1. 关闭 Unreal Editor；
2. 解压后把完整 `LunarTcpServer/` 放到 `<Project>/Plugins/`；
3. 使用 UE 5.0.1 的 `GenerateProjectFiles.bat` 重新生成工程；
4. 使用同一引擎的 `Build.bat` 构建 `<ProjectName>Editor Win64 Development`；
5. 启动 Editor，启用插件并重启；
6. 进入 PIE，运行 Automation 测试和 `windows_smoke_test.ps1`；
7. 保存编译日志、Automation 结果、监听输出和 smoke JSON 到仓库外目录。

Linux 编译出的对象、库或 DLL 不传到 Windows。所有 UE 二进制必须由目标 Windows 的 UE 5.0.1
和其认可的 MSVC 工具链本机生成。

## 10. 失败处理

- 插件编译失败：保留完整 UBT 首个错误和引擎版本，不改用预编译 DLL规避；
- 端口无法绑定：检查地址是否属于当前 Windows；本机 smoke 回退到 `127.0.0.1`；
- HELLO 无 ACK：区分未 accept、帧拒绝、schema 拒绝和配置/标定拒绝；
- 适配器缺失：监听和握手可继续，但 START/reference 稳定拒绝且保持 HOLD；
- AGX 调用崩溃：隔离为项目适配器问题，不允许网络线程直接调用 AGX 作为绕过；
- 任一 watchdog 或旧 session 测试失败：阻止现场联调，不以重试成功掩盖。

## 11. 完成条件与资格措辞

### 11.1 源码交付完成

只有同时满足以下条件，才可写“Unreal 服务端源码包已生成”：

- 独立 Git 根和目录边界检查通过；
- Ubuntu 原生协议测试全绿；
- 源码包 allowlist、manifest 和 SHA-256 验证通过；
- ZIP 中不存在构建、缓存和运行 artifact；
- Windows 编译与 smoke 命令完整可复制。

### 11.2 Windows 编译完成

只有目标 Windows UE 5.0.1 的 UBT 构建、Automation 和本机 smoke 均有现场日志，才可写
“UE 5.0.1 插件已编译并验证监听”。Ubuntu 静态检查不得替代该结论。

### 11.3 系统资格

本机 loopback 通过只能关闭插件加载、协议和监听项；真实 AGX adapter、两机网络、路径执行、
断网和 30 分钟稳定性仍按现有资格清单分别关闭。任何阶段都不得仅凭端口 `LISTEN` 宣称系统
qualified。
