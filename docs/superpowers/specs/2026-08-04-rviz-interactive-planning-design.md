# RViz2 单平台交互规划与实时证据显示设计

日期：2026-08-04

状态：已批准；仓库外实现与 Ubuntu 验证已完成

选择：实时交互模式；RViz 左侧面板选择平台；只显示当前平台

## 目标

在仓库外的 Isaac→ROS 验证工程中增加一个独立的 RViz2 交互规划入口。用户在
RViz 左侧面板选择轮式、腿式或跳跃式平台，待当前平台就绪后使用标准
`2D Goal Pose` 在月面地图上选择目标。系统调用现有 `/plan_motion` Action，并实时
显示当前平台的规划阶段、路线、跳跃弹道、飞行管、落区或不可行原因。

交互结果用于探索和人工检查，不替代六案例自动化 Action 回归，也不进入正式
JSON/JUnit 验收证据。

## 已选方案与备选方案

采用自定义 RViz 左侧 `Lunar Planner` 面板，提供三个互斥的平台按钮、当前状态、
取消规划和错误信息。面板与仓库外的交互控制器通信，控制器负责平台会话和
`PlanMotion`；RViz 插件不直接启动或管理规划器进程。

未选择以下方案：

1. 启动命令传入 `--platform`：实现简单，但每次切换都需要终止并重新启动会话。
2. 点击场景中的 Interactive Marker：交互含义不够明显，容易误点，并依赖当前已标记
   deprecated 的 `interactive_markers` 包。
3. 为三个平台分别配置 Goal 工具：会把平台选择和目标选择混在同一次点击中，不符合
   “先明确当前平台，再选一个目标”的要求。

## 范围与所有权

实现位于仓库外目录
`~/CodexDownloads/lunar_navigation/isaac_ros_action_regression`。主仓只保存本设计、
实施计划以及现有规划器接口；不得为调试界面修改 `PlanMotion.action`、规划核心、能力
所有权或外部 Topic 基线。

本增量使用已校验快照中的地图、平台 `planning_position_m` 和姿态。它不会连接或移动
Isaac Sim 中的实时 Actor，不修改或保存 USD，也不会改变场景锁、能力文件、快照和正式
六案例期望。此前的离线正例可见性设计保留为历史方案，但不是本增量的实施入口。

## 组件边界

### RViz 面板插件

仓库外新增一个小型 `ament_cmake` RViz 插件包。面板只负责：

- 显示轮式、腿式和跳跃式三个互斥按钮；
- 异步请求平台切换；
- 显示 `NO_PLATFORM`、`SWITCHING`、`READY`、`PLANNING`、`FEASIBLE`、
  `INFEASIBLE`、`CANCELED` 或 `ERROR`；
- 在 `PLANNING` 时锁定平台按钮；
- 提供取消当前 Action 的按钮；
- 显示当前平台、规划阶段和最近一个稳定原因码。

面板不解析地图、不构造 Action、不验证规划结果，也不拥有子进程生命周期。

### 交互控制器

在既有 `lunar_isaac_validation` Python 包中增加独立控制器。它是平台状态、Action
客户端和可视化发布的唯一所有者，负责：

- 一次只维护一个 bridge/planner 子进程对；
- 按平台选择相应能力文件、规划参数、局部地图和快照平台姿态；
- 依次完成旧会话清理、新会话启动、Lifecycle configure/activate 和端点就绪检查；
- 接收 `/goal_pose`，校验并构造 `PlanMotion.Goal`；
- 处理反馈、取消和终态结果；
- 在发布图形前执行显示所需的结构与有限值校验；
- 原子清除旧显示，使 RViz 始终只呈现当前平台和当前请求。

正式 `RegressionRunner` 不承担交互职责，其默认 CLI、报告结构和判定逻辑保持不变。

### 交互快照桥

交互会话从快照 manifest 读取所选平台的 `planning_position_m`、`orientation_wxyz` 和
对应局部 GridMap，持续发布现有六项规划输入、TF 与当前交互 mission。每次成功切换
生成新的 session mission id；控制器发送的 Action goal 必须携带同一 mission id 和
revision。

该桥与正式场景桥共享无副作用的快照编码函数，但不伪装成
`wheel-positive`、`legged-positive` 或 `hopper-positive` 案例。

### RViz 显示配置

随仓库外安装空间提供固定 RViz 配置，Fixed Frame 为 `map`，包含：

- 从全局 GridMap 派生的灰度高程 `PointCloud2`，以及从当前平台局部 GridMap 派生的
  红色障碍/禁行 `PointCloud2`；规划器继续直接使用原始 GridMap，显示适配不改变规划输入；
- 高程、障碍物和禁行层；
- 当前平台代理标记、起点、目标和容差；
- 当前轮式或腿式 `nav_msgs/Path`；
- 当前跳跃式弹道、飞行管、落区和着陆点 `MarkerArray`；
- `Lunar Planner` 面板与标准 `2D Goal Pose` 工具。

## ROS 通信

不新增主仓消息接口。仓库外组件使用标准 ROS 2 类型：

- 三个平台切换入口分别为 `/lunar_isaac_validation/select_wheel`、
  `/lunar_isaac_validation/select_legged` 和
  `/lunar_isaac_validation/select_hopper`，使用 `std_srvs/Trigger`；
- 取消入口为 `/lunar_isaac_validation/cancel_plan`，使用 `std_srvs/Trigger`；
- `/goal_pose` 使用 `geometry_msgs/PoseStamped`；
- `/lunar_isaac_validation/interactive_status` 使用
  `diagnostic_msgs/DiagnosticArray`，包含 session、active platform、state、Action phase
  和 reason code；
- `/lunar_isaac_validation/current_path` 使用 `nav_msgs/Path`；
- `/lunar_isaac_validation/global_surface` 和
  `/lunar_isaac_validation/local_hazards` 使用 `sensor_msgs/PointCloud2`，只属于仓库外
  RViz 显示面；
- 目标、文字、弹道、飞行管、落区和错误使用
  `/lunar_isaac_validation/current_markers` 的
  `visualization_msgs/MarkerArray`。

平台切换 Trigger 成功只表示请求已被控制器接受；面板必须继续等待状态 Topic 报告
`READY`，不能把 Service response 当作规划端点已就绪。诊断数组使用固定 status name
`lunar_isaac_validation/interactive_planner`，`message` 保存状态枚举，其余字段放入
key/value，避免面板解析自由格式日志。

状态和可视化使用 reliable + transient-local QoS，使晚启动的 RViz 能取得当前稳定状态，
同时不保留已退出控制器的跨会话历史。

## 平台选择与状态机

初始状态为 `NO_PLATFORM`。选择平台后执行：

1. 拒绝新的 Goal，并发布 `SWITCHING`；
2. 发布 Marker `DELETEALL` 和空 Path；
3. 按 Lifecycle 顺序停用并关闭旧规划器，再停止旧 bridge；
4. 从 manifest 装载所选平台起点和姿态；
5. 用对应参数启动 bridge 与规划器；
6. configure、activate，并检查地图、状态、TF、mission 和 Action 端点；
7. 全部就绪后发布 `READY`。

规划期间平台按钮不可用，控制器也会独立拒绝切换请求。用户必须等待终态，或先点击
“取消规划”。切换失败进入 `ERROR`，不恢复旧路线或旧平台标识；用户可重新选择平台
重试。

`FEASIBLE`、`INFEASIBLE` 和 `CANCELED` 是带最近结果的稳定状态，均允许发送下一个
Goal；收到新 Goal 后进入 `PLANNING`。

## RViz 目标到 PlanMotion 的映射

控制器只在当前会话已就绪时接受目标，并按以下顺序 fail closed：

1. 要求 `frame_id == map`，坐标和四元数全部有限且四元数可归一化；
2. 要求 XY 位于当前平台局部地图范围内；
3. 要求对应单元的 `valid_mask` 有效，并有有限高程；
4. 对明确的 obstacle 或 forbidden 目标在客户端拒绝，不发送 Action；
5. 从 GridMap 高程层填写目标 Z；
6. 从 RViz 箭头姿态计算 yaw，并启用 yaw 约束；
7. 轮式和腿式位置容差默认 `0.5 m`，跳跃式默认 `0.75 m`，yaw 容差统一为
   `15°`；
8. 使用当前 session 的 mission id/revision，且
   `replace_active_request == false`。

收到任何新的、基本格式有效的点击后，先清除上一请求的路线、弹道、落区和文字，避免
旧结果被误认为属于新目标。地图外、无高程、障碍或禁行目标显示红色目标和本地拒绝
原因；它们不冒充规划器的 `reason_code`。

同一时刻只允许一个 Action goal。`PLANNING` 期间再次点击返回 `BUSY`，不替换或取消
活动请求。

本地拒绝使用稳定原因码 `GOAL_FRAME_INVALID`、`GOAL_NONFINITE`、
`GOAL_OUTSIDE_LOCAL_MAP`、`GOAL_ELEVATION_INVALID`、`GOAL_OBSTACLE`、
`GOAL_FORBIDDEN` 或 `BUSY`。这些原因码只属于交互控制器，不写入或伪装成
`PlanMotion.Result.reason_code`。

## Action 结果与实时显示

Action feedback 驱动面板显示 `VALIDATING_INPUT`、`BUILDING_SNAPSHOT`、`SEARCHING`、
`OPTIMIZING` 和 `CERTIFYING`。只有终态为 succeeded 且结果通过平台、frame、有限值、
容器长度与基本几何结构校验后，才允许绘制结果。

- 轮式：蓝色地形跟随 Path、方向箭头、目标容差和成功 reason code；
- 腿式：绿色地形跟随 Path、方向箭头、目标容差和成功 reason code；
- 跳跃式：橙色三维弹道中心线，以 hop 的 launch pose、launch velocity、flight time
  和能力重力采样；用半透明采样截面表达 flight tube；同时显示实际 landing region
  边界、半透明落区、计算着陆点和成功 reason code；
- 无安全参考或目标不可行：不绘制路线，显示红色叉号、目标容差和 Action
  `reason_code`；
- Action rejected、aborted、timeout 或结构异常：进入 `ERROR`，只显示错误状态，不绘制
  未验证几何。

平台切换和新请求使用稳定 namespace/id 删除旧 Marker，并发布空 Path。任何时刻都不得
同时保留两个平台或两个请求的路径证据。

## 启动与退出

仓库外提供单独的交互启动脚本。脚本负责 source ROS 2 Humble 与指定 install overlay、
租用非 ambient ROS domain、启动交互控制器和带固定配置的 RViz。退出时对已核验的
RViz PID 发送 X11 `WM_DELETE_WINDOW`，再按身份清理该脚本启动的控制器和子进程并释放
domain lease；不得按名称批量终止其他 ROS、Isaac Sim 或 RViz 进程。

Ubuntu 实测确认 ROS Humble `grid_map_rviz_plugin` 即使不加载本项目面板，也会在正常
窗口析构时触发段错误；保留本项目面板而移除该显示插件则正常退出。因此 RViz 只使用
标准 `PointCloud2` 显示，原始 `grid_map_msgs/GridMap` Topic 与规划端语义保持不变。

正式六案例脚本保持原命令和无界面默认行为。交互会话不得写入正式 artifact 目录；若
需要日志，只能写入仓库外本次 session 的明确目录。

## 测试策略

实现严格按 TDD 推进。

### 单元测试

- 面板三个按钮互斥，切换请求、锁定和状态渲染正确；
- 控制器状态转换合法，非法转换和重复切换 fail closed；
- 平台到能力、参数、局部地图和 manifest 姿态的映射准确；
- `PoseStamped` 的 frame、有限值、地图范围、valid mask、高程、obstacle 和 forbidden
  检查完整；
- XY、Z、yaw、位置容差、yaw 容差、mission id/revision 和 replace 标志正确写入 Goal；
- `BUSY`、取消、超时、子进程退出和 Lifecycle 失败产生稳定状态；
- 新点击、平台切换和错误路径均清除旧 Path/Marker；
- 轮式/腿式 Path、跳跃弹道/飞行管/落区及不可行标记内容正确；
- 非有限、frame 错误、平台错误或不完整结果不产生几何消息。

### ROS 集成测试

在独立 ROS domain 中按轮式、腿式、跳跃式顺序分别验证：

1. 选择平台并达到 `READY`；
2. 发送一个已知可行目标并取得对应 Path 或 hop 显示；
3. 发送一个未被本地筛除、但由 Action 判定不可行的目标，取得无路线的规划器原因；
4. 另行发送 obstacle、forbidden 或地图外目标，取得稳定的本地拒绝原因；
5. 覆盖规划中重复 Goal、取消、切换清屏和子进程异常；
6. 断言任一时刻只存在当前平台的显示 namespace。

集成测试使用的交互可行与 Action 不可行目标必须从当前 hash-bound 快照预先筛选，并
保存在仓库外测试 fixture 中；它们不得写回正式六案例 lock，也不得改变正式期望。

随后运行外部工程完整 source-first 测试和原六案例正式 Action 回归；后者必须继续
`6/6`，退出码为 0，且 JSON/JUnit 的既有语义不变。

### 人工 RViz 验收

- 左侧面板可完成三类平台选择，当前平台和状态无需查看终端即可确认；
- `READY` 后可用标准 `2D Goal Pose` 选取目标；
- 轮式和腿式路线贴合高程显示；
- 跳跃式弹道、飞行管、落区和着陆点在三维视图中可区分；
- 可行、不可行、本地目标拒绝和运行错误具有不同且清楚的标识；
- 切换平台或发送新 Goal 后不存在旧平台或旧请求残留；
- 退出交互启动脚本不会终止用户自行启动的 Isaac Sim 或其他 ROS/RViz 会话。

## 完成判据

只有在单元测试、ROS 集成、完整外部测试、原六案例 `6/6` 和人工 RViz 验收全部通过后，
才可报告本功能完成。Ubuntu 验证只能证明仓库外交互与规划可视化就绪，不得替代真实
Isaac Sim 动态联调、AGX 性能验收或完整导航系统切换验收。
