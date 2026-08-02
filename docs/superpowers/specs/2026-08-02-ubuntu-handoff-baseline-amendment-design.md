# Lunar Navigation Ubuntu 交接基线修订设计

**状态：** 已实施

**日期：** 2026-08-02

**适用范围：** 卷一 Task 3–5 及所有后续引用外部消息所有权或训练 GPU 型号的计划

## 1. 背景与修订依据

Ubuntu 22.04 amd64 现场复验确认了两项与原冻结设计不同的事实：

1. 计划依赖的 `lunar_navigation_msgs` 上游项目尚未定义或发布该包。本项目需要先提供同名暂定接口，后续再与上游协商并执行兼容迁移。
2. 权威训练主机的实际 GPU 是 NVIDIA GeForce RTX 4080 SUPER，PCI Device ID 为 `10de:2702`；原设计中的 RTX 4080 是设备型号记录错误。

用户已明确批准以上修订，并允许在当前 `volume-1-foundation` 分支原地实施。本文只修订消息 schema 的暂定所有权和训练 GPU 型号，不改变 Topic、frame、AGX 职责、C++ v3 主线、PPO 发布边界或 Task 1 冻结迁移清单。

## 2. 目标与非目标

### 2.1 目标

- 在本仓创建唯一的暂定 `lunar_navigation_msgs` ROS 2 接口包，使 Ubuntu 构建不再等待尚未存在的上游包。
- 以 `docs/interfaces/external-input-baseline.md` 作为暂定消息字段和语义的唯一权威来源。
- 保持外部定位和任务系统对 Topic 数据的所有权，同时明确 schema 暂由本仓提供。
- 建立禁止同名包共存、阻止接口漂移并支持未来原子切换上游的自动门槛。
- 把权威训练平台统一修正为 Ubuntu 22.04 amd64 + NVIDIA GeForce RTX 4080 SUPER。
- 如实记录 GPU 驱动、CUDA 与训练 readiness 的现场探测结果，不用计划值或事后补写掩盖探测时序。

### 2.2 非目标

- 本次不实现外部定位、任务、地图或 TF 发布节点。
- 本次不实现运行时任务语义校验器；该逻辑仍属于后续接收适配层。
- 本次不创建独立消息仓库，也不代表上游项目接受了本仓 schema。
- 本次不自动安装 GPU 驱动、修改 GRUB 或重启主机。
- 本次不改变 AGX Orin 原生构建、TensorRT engine 和设备发布门槛。

## 3. 消息所有权架构

本仓在 `ros2_ws/src/lunar_navigation_msgs` 提供同名暂定接口包。Topic 与 schema 的所有权分开记录：

- `/localization/status` 的数据生产者仍是外部定位系统。
- `/mission/exploration_task` 及其科学目标数据的生产者仍是外部任务系统。
- `lunar_navigation_msgs` 的 schema 提供方暂时是本仓，状态为 `in_repository_provisional`。

暂定阶段不在 `dependencies.repos` 引入同名上游，也不增加适配节点或改变 Topic/type 名称。所有消费者继续使用：

- `lunar_navigation_msgs/msg/LocalizationStatus`
- `lunar_navigation_msgs/msg/ScienceTargetRegion`
- `lunar_navigation_msgs/msg/ExplorationTask`

`external_interfaces.yaml` 升级为 `lunar-external-interfaces/v2`。既有 Topic 项继续使用 `owner: external` 表示数据生产者；新增顶层 `interface_packages` 独立记录 schema 来源：

```yaml
schema_version: lunar-external-interfaces/v2
interface_packages:
  lunar_navigation_msgs:
    schema_provider: in_repository_provisional
    source_path: ros2_ws/src/lunar_navigation_msgs
    upstream_status: undefined
    replacement_policy: atomic
```

## 4. 暂定消息合同

### 4.1 `LocalizationStatus.msg`

```text
uint8 UNKNOWN=0
uint8 VALID=1
uint8 DEGRADED=2
uint8 INVALID=3
uint8 RELOCALIZING=4
std_msgs/Header header
uint8 status
```

### 4.2 `ScienceTargetRegion.msg`

```text
string region_id
string objective_id
geometry_msgs/Polygon boundary
float64 priority
```

### 4.3 `ExplorationTask.msg`

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

`ExplorationTask` 的 `0` 状态值保持非法，避免零初始化消息被误判为 ACTIVE。字段顺序、类型、常量值和数组上限一经提交即冻结；暂定 schema 用包版本和 Git commit 标识，不给每条消息增加额外 schema 字段。

## 5. 权威语义约束

`docs/interfaces/external-input-baseline.md` 从单纯接收说明升级为暂定消息字段和语义的权威基线。实际 `.msg` 文件是 rosidl 构建输入，但不得偏离该文档。

接收适配层后续必须验证：

- `LocalizationStatus.header.stamp` 非零，frame 为 `odom`，`status` 是五个已定义常量之一。
- `ExplorationTask.header.stamp` 非零，`header.frame_id=map`，`mission_id` 非空。
- `revision` 从 1 开始，并在同一 mission 内严格递增。
- `desired_state` 是 ACTIVE、PAUSED 或 CANCELED。
- ROI 四个坐标均为有限值，且 min/max 顺序合法。
- `science_regions` 为 0 至 64 项；`region_id` 在任务内唯一且非空，`objective_id` 非空。
- 每个 boundary 含 3 至 256 个不同顶点，是简单非自交边界；顶点位于 `map`，二维边界的 `z=0.0`。
- `priority` 位于 `(0, 1]`。

任何约束失败时拒绝整条输入并产生确定性诊断。不得钳制值、补默认值、修复边界或合成替代内容。过期或乱序 revision 不得改变当前有效任务。

## 6. 构建与现场检查

Ubuntu 构建顺序为：

1. 从 `/opt/ros/humble/setup.bash` 加载 ROS 2 Humble。
2. 在仓库外的显式 `LUNAR_VOLUME1_OUTPUT` 下创建 colcon build/install/log 目录。
3. 先生成 `lunar_navigation_msgs`，再构建配置包、`lunar_planning_msgs` 和后续消费者。
4. 从本次 install 前缀加载 overlay。
5. 运行现场接口与生成类型测试。

现场检查器由调用方显式传入本次暂定包的期望 install 前缀。它必须确认：

- `grid_map_msgs`、`nav_msgs` 和 `tf2_msgs` 来自 `/opt/ros/humble`。
- `lunar_navigation_msgs` 来自本次工作区的仓库外 install 前缀。
- `AMENT_PREFIX_PATH` 中不存在第二个提供 `lunar_navigation_msgs` 的 ament resource index。
- 三个生成接口的字段和常量与本设计及外部输入基线一致。

任何包遮蔽、来源偏差、额外同名实现或接口差异都使门槛失败。

## 7. 上游切换合同

上游发布 `lunar_navigation_msgs` 后，不允许简单叠加其 overlay。切换流程必须：

1. 固定唯一上游 URL 和 tag/commit。
2. 比较规范化字段顺序、类型、常量值、数组边界和生成接口。
3. 使用双方确认的 rosbag/fixture 执行兼容验证。
4. 若完全兼容，在同一提交中删除本仓暂定包并把固定上游加入 `dependencies.repos`。
5. 若不兼容，停止切换并协商显式版本或适配方案；不得静默更改当前合同。

任何时刻同一工作区只能存在一个 `lunar_navigation_msgs` 实现。

## 8. 测试设计

实施遵循测试先行：

1. 静态 pytest 逐字验证三个 `.msg` 的字段顺序、类型、常量值和 64 项上限。
2. 静态测试验证 `package.xml`、`CMakeLists.txt` 和 rosidl 依赖完整。
3. 配置测试验证 Topic 数据所有者仍为 external、schema 来源为 `in_repository_provisional`，并与外部输入基线一致。
4. 仓库边界测试只允许批准路径下的三个 `.msg`，拒绝第二个同名包、额外 msg/action/service、gitlink 和构建产物。
5. colcon 构建后，使用生成的 Python 类型验证字段映射、常量和 bounded sequence。
6. `ros2 interface show` 和包前缀检查验证现场定义及来源。
7. 最终运行 foundation pytest、仓库边界、接口检查、消息构建、生成类型 pytest、`colcon test` 和 `colcon test-result --verbose`。

本卷测试不把静态 schema 检查冒充运行时语义验证。revision、ROI、Polygon 和任务状态拒绝路径在后续适配层实现后再用 fixture/rosbag 验证。

## 9. RTX 4080 SUPER 基线修订

所有仍具约束力的 README、架构、设计、路线图和四卷计划统一使用 `RTX 4080 SUPER`。尚未创建的平台文件直接采用：

```yaml
schema_version: lunar-platform-baseline/v1
profile: train_amd64_rtx4080_super
os: Ubuntu 22.04 LTS
architecture: amd64
ros_distro: humble
python: "3.10"
gpu_model: NVIDIA GeForce RTX 4080 SUPER
gpu_pci_device_id: "10de:2702"
responsibilities: [ros_integration, ppo_training, onnx_export, rosbag_replay]
```

对应目录为 `platform/train_amd64_rtx4080_super/`。原 `train_amd64_rtx4080` 名称尚无已发布 artifact，因此不保留别名。

环境指纹必须记录 OS、architecture、kernel、CPU、内存、ROS、Python、GCC、CMake、GPU model、PCI ID、driver、CUDA、TensorRT、L4T、JetPack、功耗模式和时钟状态。探测优先级为：

1. 驱动正常时使用 `nvidia-smi` 识别型号、显存和驱动。
2. 驱动不可用时记录 `lspci`/sysfs 的 PCI ID、原始错误和 `available: false`。
3. `nvcc` 只证明 toolkit 存在，不能替代驱动或 GPU compute readiness。

本轮指纹工具实现前，当前训练主机已经完成 RTX 4080 SUPER 驱动维护，因此不存在由该工具生成的安装前 JSON，也不得事后伪造。最终现场采集必须写入调用方指定的仓库外验证根。当前主机探测到 NVIDIA GeForce RTX 4080 SUPER、PCI Device ID `10de:2702`、驱动 595.84 和 CUDA 13.2，训练 profile 的 `readiness.ready=true`；TensorRT 未安装不阻塞训练主机 readiness，因为 TensorRT engine 生成与设备验收仍属于 AGX Orin 权威环境。

## 10. 现有改动保护与实施顺序

`AGENTS.md` 当前含用户未提交改动。实施时只在工作副本中把其中的设备型号修正为 RTX 4080 SUPER，保留其他内容；提交必须排除不属于当前任务的用户改动。

实施顺序为：

1. 修订权威文档、计划和检查合同。
2. 以 TDD 创建暂定 `lunar_navigation_msgs` 并通过 Ubuntu 接口构建。
3. 继续卷一 Task 4 的 `lunar_planning_msgs`。
4. 实现 Task 5 的平台基线与环境指纹工具。
5. 在调用方指定的仓库外目录保存现场 GPU 指纹和原始探测结果。
6. 仅在训练 profile 的 compute readiness 通过后进入后续训练相关任务。

该顺序保留原卷一边界，同时把已经不存在的“等待上游消息包”阻塞替换为可验证的暂定接口迁移路径。

## 11. 实施结果

卷一 Task 3–5 已按本设计完成：本仓暂定提供 `LocalizationStatus`、`ScienceTargetRegion` 和 `ExplorationTask` 三个外部输入接口；`lunar-external-interfaces/v2` 的 provider/source 现场门槛、内部规划接口以及训练与部署平台基线和环境指纹工具均已落地。最终集成验证从干净的仓库外构建根重建 `lunar_navigation_msgs`、`lunar_navigation_config` 和 `lunar_planning_msgs`，并核对生成类型、接口来源和仓库边界。

RTX 4080 SUPER 驱动维护早于环境指纹工具实现，本仓没有也不会伪造安装前 JSON。最终训练主机指纹保存在调用方指定的仓库外验证根；本次采集结果为 `readiness.ready=true`，TensorRT 缺失不阻塞训练主机基线。该结论只覆盖当前 Ubuntu 22.04 amd64 训练主机，不构成训练 smoke、运行时语义适配器或 Jetson AGX Orin 部署能力验证。

AGX Orin 的环境指纹、原生构建、TensorRT engine、性能、功耗和稳定性仍必须在后续实机门槛中采集和验收；在该权威设备验证完成前，不得写成已验证能力。
