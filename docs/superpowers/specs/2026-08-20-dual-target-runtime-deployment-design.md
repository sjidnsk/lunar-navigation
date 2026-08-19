# 双目标源码运行时部署与统一 `luna` 命令设计

**状态：** 已批准，待实施

**日期：** 2026-08-20

**适用范围：** Ubuntu 22.04 + ROS 2 Humble amd64 目标主机，以及 Jetson AGX Orin R36 + ROS 2 Humble aarch64 目标主机。

## 1. 决策与目的

本设计将当前单一 Git 根仓库交付为两份**目标机原生构建的源码运行时包**。它们服务于独立的部署主机，而不是当前训练主机；不得假定目标 amd64 主机具有当前训练主机的 GPU、Python 环境、编译产物或缓存。

两份包来自同一源码提交和同一接口契约，只因目标 profile、依赖解析和推理后端不同而分开发布：

| 源码运行时包 | 目标 | 默认策略后端 | 已安装模型后的后端 |
| --- | --- | --- | --- |
| `lunar-runtime-ubuntu22-humble-amd64-src` | Ubuntu 22.04、x86_64、ROS 2 Humble | 确定性安全回退 | ONNX Runtime CPU；可选兼容 GPU provider |
| `lunar-runtime-jetson-orin-r36-aarch64-src` | Jetson AGX Orin、Ubuntu 22.04、aarch64、L4T R36、ROS 2 Humble | 确定性安全回退 | 由 ONNX 在设备本机生成并加载 TensorRT engine |

部署运行时始终保留规划、安全投影、足迹/净空、精确端点可行性、路径认证和输入时序校验。所谓“去除硬门”只指默认命令不运行训练资格、formal preflight、闭环评估或 release gate；绝不允许通过配置关闭物理安全或输入一致性检查。

## 2. 范围与非目标

### 范围

- 生成两份可部署源码包、各自 profile、依赖入口和原生构建路径。
- 提供一个短而统一的命令 `luna`，包装初始化、构建、配置、启动、状态、日志和模型切换。
- 将训练中的 PPO checkpoint 与运行时模型交接解耦，允许规划器在模型尚未完成时安全运行。
- 清楚冻结设备运行时的 ROS 输入、输出、模型文件接口、配置接口和扩展接口。
- 为后续全局地图处理与路径跟踪新增可选包边界，而不重构规划核心或修改既有外部消息。
- 在部署包根目录保留面向操作者的 `README.md` 和 `COMMANDS.md`。

### 非目标

- 不复制训练数据、训练缓存、训练日志、PPO optimizer state、恢复 checkpoint 或当前训练机的 Python 环境。
- 不将 amd64 ELF、Linux wheel、`build/`、`install/`、`log/`、TensorRT engine 传递给另一架构机器。
- 不自动将训练目录的最新 checkpoint 直接上线。
- 不通过运行时包直接向轮式、足式或飞跃式执行器发送低级控制指令。
- 不包含仓库的测试、设计/计划文档、迁移资料、历史 artifact 或开发日志；实现仓库中的测试仍保留，用于开发验证，但最终部署源码包不携带它们。
- 不在本设计中定义尚未拥有的执行器命令消息；路径跟踪适配器必须等待外部控制接口获批准。

## 3. 源码包与目标机文件边界

### 3.1 交付物

每个压缩包包含一个 `release-manifest.json`，其中冻结：

- `runtime_schema_version`；
- 精确 `source_commit` 与源码清单 SHA-256；
- `target_profile`；
- 支持的 OS、架构、ROS 发行版与（Orin）L4T/JetPack 基线；
- 外部接口、模型 observation/action contract 的版本；
- 包含的可选扩展清单。

两个 manifest 使用同一 `source_commit`，但目标 profile 不同。`luna init` 和 `luna build` 必须拒绝架构、OS 或 ROS 发行版不匹配的包；不能借由修改 profile 伪装跨架构构建成功。

部署包只保留以下源码：

```text
README.md
COMMANDS.md
luna/                         # 命令实现与 profile
config/                       # 默认运行时配置及 schema
platform/                     # 已批准平台能力资料
model_contract/               # 观测/动作契约与模型 manifest 校验
ros2_ws/src/
  lunar_navigation_msgs/
  lunar_planning_msgs/
  lunar_navigation_config/
  lunar_planner_core/
  lunar_planner_ros/
  lunar_nav2_adapter/         # 可选，轮式 Nav2 集成
  lunar_policy_runtime/       # 本设计新增的可选运行时策略包
```

`lunar_planner_training_bridge`、`training/` 和训练专用 Python 依赖不进入部署包。训练桥继续属于训练机的高吞吐内部接口，不是设备 ROS 接口。

### 3.2 本机构建与运行目录

源码包可解压到任意用户可写目录。`luna init` 默认把可变状态写在源码树外：

```text
~/.config/luna/runtime.yaml             # 操作者配置
~/.local/share/luna/<release-id>/        # build、install、logs、PID 与本机 engine cache
```

可以通过 `LUNA_HOME` 显式改写根目录。`luna` 不在源码目录中生成构建、日志、模型或 TensorRT 产物。

`luna` 本身以用户级可执行文件安装；若系统已有同名命令，初始化必须报告冲突，不能覆盖现有可执行文件。包内 `./luna` 始终可用。

## 4. 模型仍在训练时的交接

### 4.1 默认行为

训练与部署有不同职责：训练机持有可恢复 checkpoint；部署端只接受已导出的不可变运行时模型包。因此两份源码运行时初始均写入：

```yaml
policy:
  mode: fallback
```

在 `fallback` 下，系统使用确定性候选排序/选择策略，仍经由同一候选生成、规划和安全认证链工作；不会试图加载 checkpoint，也不会等待训练结束才能启动规划服务。

### 4.2 模型包接口

训练完成一个可交付候选后，训练侧导出独立目录或压缩包：

```text
model-manifest.json
policy.onnx
normalization.npz
```

`model-manifest.json` 至少冻结：模型 ID、模型包 SHA-256、`source_commit` 或兼容范围、配置摘要、ObservationContract 版本、ActionContract 版本、ONNX 输入/输出名、shape、dtype、归一化文件 hash 与导出时间。原始 `.pt` checkpoint、优化器和训练状态不属于模型包。

安装顺序为：

1. `luna model install <模型包>` 验证 manifest、hash、输入/输出形状和运行时契约；失败时不改变当前模型。
2. amd64 直接保存 ONNX 并用 ONNX Runtime 进行加载冒烟。
3. Orin 在本机从同一 ONNX 生成 engine，并绑定 ONNX hash、TensorRT 版本、设备指纹和精度配置；任何一项变化都使旧 engine 失效并重建。
4. `luna model activate <模型-id>` 原子切换活动模型，并使新的规划会话使用 `policy` 模式。
5. `luna model rollback` 回到上一个有效模型；没有模型时回到 `fallback`。

运行时绝不轮询训练目录，也不根据“latest.pt”自动替换模型。模型包与源码运行时可独立更新，只要 manifest 宣称且验证通过兼容的接口版本。

## 5. 外部输入、输出与所有权

外部 ROS 接口固定采用当前权威合同 `lunar-external-interfaces/v5`。本节是部署包的摘要；精确字段、坐标系、时间与多分辨率地图语义以随包携带的 `external_interfaces.yaml` 为唯一机器可读来源。

### 5.1 输入

| 名称 | ROS 类型 | 所有者 | 必要语义 |
| --- | --- | --- | --- |
| `/environment/map_global` | `grid_map_msgs/msg/GridMap` | 外部地图融合系统 | `map` frame 的已选全局层级；包含冻结要求的地图图层 |
| `/environment/map_local` | `grid_map_msgs/msg/GridMap` | 外部地图融合系统 | `odom` frame 的 L0、0.2 m 局部执行证据窗 |
| `/localization/odometry` | `nav_msgs/msg/Odometry` | 外部定位系统 | 新鲜的 `odom -> base_link` 位姿与协方差 |
| `/localization/status` | `lunar_navigation_msgs/msg/LocalizationStatus` | 外部定位系统 | `VALID/DEGRADED/INVALID/...` 定位状态 |
| `/tf` | `tf2_msgs/msg/TFMessage` | 外部 TF 发布者 | 有效 `map -> odom -> base_link` 变换链 |
| `/mission/exploration_task` | `lunar_navigation_msgs/msg/ExplorationTask` | 外部任务系统 | 版本化任务 ROI、任务状态和高优先级区域 |
| `/execution/motion_feedback` | `lunar_navigation_msgs/msg/MotionExecutionFeedback` | 外部执行/控制系统 | 与计划 ID、段 ID 匹配的执行反馈 |
| 平台/观测能力文件 | YAML/JSON/URDF | 外部能力提供方 | 已批准的平台能力 v2 与传感器范围/FOV |

`HopperPropellantState` 仅是历史源码兼容消息，不是活动输入，不订阅、不冻结也不暴露给模型。

输入校验失败时，规划器输出保持/无安全参考与诊断，不以零值、陈旧数据或合成地图继续规划。

### 5.2 规划请求、结果与诊断输出

`/plan_motion` 是双向的 ROS Action endpoint：外部任务协调器、Nav2 适配器或未来路径跟踪适配器提交 `request_id`、`mission_id`、`mission_revision`、`GoalRegion` 和显式替换标志；运行时返回规划结果和反馈。它不是由部署端主动发送的控制 Topic。

| 名称 | ROS 类型 | 方向 | 语义 |
| --- | --- | --- | --- |
| `/plan_motion` | `lunar_planning_msgs/action/PlanMotion` | client → runtime：goal；runtime → client：feedback/result | 唯一的规划请求/结果接口；反馈报告规划 phase/耗时，结果包含 outcome、directive、`MotionReference` 和诊断 |
| `/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | runtime → 监控系统 | 输入拒绝、规划状态、资源或安全原因 |
| `/planning/certified_route_markers` | `visualization_msgs/msg/MarkerArray` | runtime → RViz/调试工具 | 已认证路径可视化，不是控制命令 |
| `/planning/provisional_route_markers` | `visualization_msgs/msg/MarkerArray` | runtime → RViz/调试工具 | 仅规划过程可视化，不是可执行参考 |

`PlanMotion` 结果中的 `MotionReference` 是对外唯一的高层运动输出。轮式和足式可包含路径/轨迹；飞跃式可包含经认证的 `HopSegment`。下游控制器仍拥有执行命令、底层跟踪和 `MotionExecutionFeedback` 的生产权。

### 5.3 策略模型 I/O

模型不通过 ROS Topic 接收原始 Tensor。`lunar_policy_runtime` 仅由本机运行时调用，输入/输出冻结为：

```text
ObservationContractV4
  prior_channels, coverage_summary, local_crop,
  frontier_features, pose_features, candidate_mask, platform_context
        ->
ActionContractV2
  frontier_logits, theta_mu, theta_kappa, value
```

运行时只能把通过候选生成、精确可达性和规划安全认证的候选交给模型。模型输出只用于高层候选/方向选择，不能绕过规划器、替代路径认证或直接生成执行器命令。

## 6. 单一 `luna` 操作入口

操作者无需手写 `source /opt/ros/...`、`colcon`、环境变量或长 Python 模块路径。所有常用功能经由 `luna`：

| 命令 | 行为 |
| --- | --- |
| `luna init --profile ubuntu22-humble-amd64` | 创建 amd64 运行时配置并检查目标基本身份 |
| `luna init --profile jetson-orin-r36` | 创建 Orin 配置并检查 R36/架构身份 |
| `luna doctor` | 检查 ROS、依赖、能力文件、配置与外部接口可解析性 |
| `luna config check` | 只校验 `runtime.yaml`，不启动服务 |
| `luna build` | 在 `LUNA_HOME` 中原生构建必要 ROS 包 |
| `luna start` / `luna stop` | 启动或停止部署规划服务 |
| `luna status` | 显示进程、活动 profile、模型模式和 ROS 接口状态 |
| `luna logs [--follow]` | 读取或跟随本运行时日志 |
| `luna model install <path>` | 安装并校验独立模型包 |
| `luna model activate <id>` / `rollback` / `status` | 原子切换、回退或查看模型状态 |
| `luna extension list` | 列出可用及已启用扩展 |
| `luna extension enable <name>` / `disable <name>` | 修改配置中的可选扩展开关；下次启动生效 |
| `luna bundle --target <profile>` | 在开发机从干净源码树创建对应源码运行时包 |

`luna doctor` 与 `luna config check` 是快速本地配置检查，不是训练资格或发布门。`luna start` 不自动运行训练 replay、formal preflight、覆盖率评估、AGX release gate 或 service 激活。

## 7. 唯一运行时配置

`luna init` 生成一个 `runtime.yaml`。配置的顶级区块固定为：

```yaml
profile: ubuntu22-humble-amd64 | jetson-orin-r36
interfaces:                 # Topic 名称与 frame 绑定；默认 v5 值
capabilities:               # 观测与平台能力文件
planner:                    # 公开预算、日志级别、Nav2 适配器开关
policy:                     # fallback / onnx / tensorrt，模型 ID
extensions:                 # map_pipeline、path_tracking 的启用状态
runtime:                    # LUNA_HOME、进程与日志设置
```

允许修改 Topic 名称、文件位置、公开性能预算、日志和扩展选择；不允许用配置移除地图层、改变模型 shape/dtype、伪造平台类型或关闭安全投影。配置 schema 与 profile 一同随包发布，并由 `luna config check` 验证。

## 8. 可扩展模块边界

### 8.1 全局地图处理

默认地图路径为外部地图融合系统直接发布 canonical 的 `/environment/map_global` 和 `/environment/map_local`。未来 `lunar_map_pipeline` 是可选的独立 ROS 包，而非对 `lunar_planner_core` 的修改：

```text
外部原始/融合地图
        -> lunar_map_pipeline（可选）
        -> canonical GridMap v5
        -> lunar_planner_ros
```

它可以处理、过滤或汇聚全局地图，但输出必须仍满足 `GridMap` 层、frame、时间和保守聚合合同。任一时刻只能有一个 publisher 产生 canonical map Topic；启用该扩展不改变 planner 的输入类型。

### 8.2 路径跟踪

默认部署只提供 `PlanMotion` 与 `MotionReference`，由外部执行系统跟踪。未来 `lunar_path_tracking` 作为独立扩展，充当 `PlanMotion` action client 或消费其封装后的 `MotionReference`，并经由**另行批准的外部控制器适配器**发送平台特定命令。

在该控制器契约冻结前，路径跟踪扩展不得自行猜测速度、姿态或推进器命令的 Topic/消息。其必须将身份匹配的执行结果通过现有 `/execution/motion_feedback` 合同返回，不能修改规划核心、候选模型或 MotionReference 语义。

## 9. 文档和操作者体验

每份部署源码包根目录只提供以下使用文档：

- `README.md`：项目目的、组件图、支持 profile、输入/输出摘要、模型状态、五分钟首次运行和安全边界；
- `COMMANDS.md`：逐条 `luna` 命令、最少参数、输出解释、配置例子、模型切换和常见诊断处理。

内部设计文档、测试、训练报告、cache、checkpoint、编译目录和历史日志均不进入包。所有常见操作必须以单条 `luna` 命令完成；只有模型包路径和显式 profile 等不可推导输入才需要参数。

## 10. 失败处理与恢复

- `luna build` 在独立目录构建；失败不会覆盖上一次可用安装。
- `luna model install`、`activate` 和 `rollback` 使用 manifest/hash 验证和原子指针切换；失败保持旧模型或 `fallback`。
- 运行时输入失效、模型契约不匹配、路径不可行或安全认证失败时，输出明确的 `PlanMotion` 结果/诊断，不自动降级为未认证运动。
- `luna stop` 仅停止该运行时的受管进程；`luna start` 可从已有构建和活动模型恢复，无需训练 cache。
- `luna logs`、`status` 和 `doctor` 提供人类可读原因以及稳定的机器可读状态 JSON。

## 11. 验收标准

- 两份源码包在各自目标机身份上可通过 `luna init`、`luna build`、`luna config check` 和 `luna start` 启动规划 ROS 服务。
- amd64 包不依赖训练主机的 GPU 或构建物；Orin 包不接受非本机 TensorRT engine。
- 无模型时 `fallback` 可以运行规划；模型包安装失败不改变正在使用的模型。
- 外部输入/输出与 `lunar-external-interfaces/v5`、`PlanMotion`、ObservationContract 和 ActionContract 精确匹配。
- 包中不含 tests、docs（`README.md` 与 `COMMANDS.md` 除外）、训练、cache、checkpoint、build/install/log 或历史 artifact。
- `luna` 覆盖初始化、构建、配置、运行、诊断、日志、模型和扩展的常用操作，无需操作者拼接底层 ROS/colcon 命令。
- 未来地图处理和路径跟踪可以通过可选包接入，而不修改规划核心或已冻结的外部消息。

## 12. 与既有部署设计的关系

本设计取代 `2026-08-03-agx-git-one-command-deployment-design.md` 中关于 Git 拉取式单一 AGX 安装入口、自动 qualification/activation 提示和只面向 AGX 的操作者流程；新的默认交付是两个显式源码运行时包和 `luna` 命令。

该旧设计中仍有效的约束被保留：不同架构必须原生构建、TensorRT engine 必须在 Orin 本机生成、训练 checkpoint 不进入设备运行时、普通用户不以 root 运行构建器，以及安全/输入验证不能被部署快捷入口移除。
