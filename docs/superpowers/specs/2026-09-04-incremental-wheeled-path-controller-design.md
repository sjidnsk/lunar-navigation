# 增量轮式路径跟踪控制器设计

**状态：** 已获设计批准，待按本说明实施
**范围：** `incremental_v2` 的轮式路径执行边界

## 目标

使现有 `lunar_pure_wheeled_controller` 能够显式接收增量导航器发布的局部
`nav_msgs/msg/Path`，基于实时位姿产生唯一的
`/Car/T5/Car_Cmd_Vel`（`geometry_msgs/msg/Twist`）输出。控制器仅负责路径
跟踪，不承担规划、地图处理、探索决策、重规划或 Action 管理。

## 既定接口与边界

| 方向 | 接口 | 语义 |
| --- | --- | --- |
| 路径输入 | `/Car/T4/planning/local_path`，`nav_msgs/msg/Path` | `ACTIVE` 段对应非空局部几何路径；空路径表示立即清空跟踪目标。QoS 为 Reliable + Transient Local + KeepLast(1)。 |
| 状态输入 | `/Car/T3/localization/odometry`，`nav_msgs/msg/Odometry` | 位姿位于 `odom` 坐标系，子坐标系为 `base_link`。 |
| 坐标变换 | `/tf` | 使用直接 `map <- odom` 将车辆位姿转换到路径坐标系。 |
| 唯一输出 | `/Car/T5/Car_Cmd_Vel`，`geometry_msgs/msg/Twist` | 仅填充 `linear.x` 与 `angular.z`；不发布横向速度。 |

`/Car/T4/planning/path_reference` 仍是带会话/revision 的规划可观测接口；控制器
不订阅、不解析其中的 session、state 或 revision。`/Car/T4/planning/global_route`
不作为控制输入。

## 架构

不创建第二个轮式控制器包。现有 `lunar_pure_wheeled_controller` 新增互斥的
`input_mode` 参数：

- `motion_reference`：保留默认 legacy 行为，订阅原有
  `/Car/T4/planning/wheeled_reference`，不改变兼容性。
- `incremental_path`：只订阅 `path_topic`（默认
  `/Car/T4/planning/local_path`），不建立 `MotionReference` 或 legacy cancel
  订阅。

两种模式都只保留一个命令发布者，禁止由同一节点同时消费两种路径源。增量模式中，
ROS 适配层负责把 odometry 位姿由 `odom` 变换到 `Path.header.frame_id`（正式
契约为 `map`）；纯函数跟踪层只接收已处于同一坐标系的有限 `(x, y, yaw)` 路径和
状态。这避免把 TF、ROS 消息和车辆控制数学混入规划器或跟踪算法。

增量控制器仍须作为单独节点显式启动。`exploration_navigation.launch.py` 只启动
导航和探索，不添加控制器节点，也不因规划/RViz 启动而取得底盘控制权。

## 运行行为

1. 收到非空、几何有效且 `frame_id=map` 的路径时，以该路径原子替换当前跟踪目标。
2. 定时控制周期取得最新有效 odometry 和直接 `map <- odom` 变换，生成 map-frame
   `TrackingState`，再调用现有 Pure Pursuit 跟踪函数。
3. 线速度、角速度沿用现有可配置限幅和前视距离。路径末端位置已到、但 yaw 未在
   容差内时，保持 `linear.x=0`，以已有 `spin_kp` 和角速度上限完成原地转向。
4. 收到空路径、无效路径、路径偏离、路径完成、legacy cancel（仅 legacy 模式）或
   节点销毁时，清空活动目标并发布零速度。odometry 或直接 TF 暂时不可用时只发布
   零速度并保留最近有效路径；状态恢复后由下一控制周期继续跟踪。

路径的 header 时间戳、规划时间、地图版本、会话 revision、协方差和新鲜度不进入
控制器判定；这些不是该路径执行边界的输入。TF 查询使用当前可用的直接变换，不建立
时间同步或历史变换缓存策略。

## 实施边界

- 新增 Path 解析模块及其单元测试；解析只验证所需的有限平面位置、合法四元数和
  固定 `map` 坐标系，不生成新的路径协议。
- 扩展 ROS 节点、launch 参数和包依赖以支持 `incremental_path` 与 TF；legacy 模式
  的订阅、取消和轨迹跟踪保持原样。
- 在 `config/exploration_navigation.yaml`、
  `config/incremental_navigation_interfaces.yaml` 中补全 `local_path` 的 topic、类型、
  frame 与 QoS 事实，并让已有 launch 将配置传递给导航器。
- 更新 `README.md` 与 `docs/操作指令.md`：记录独立启动命令、唯一命令发布者、
  TF 前提和本机/目标验证边界。

不修改增量规划算法、`NavigateToPose`、探索器、`PathReference` 定义、全局路径、
车辆底盘协议或自动启动策略。

## 验收与验证

1. 测试先行覆盖：有效/空/无效 `Path`、路径替换、末端 yaw 对齐、map/odom 变换、
   TF 缺失停车、路径偏离停车，以及 legacy 模式回归。
2. 接口合同测试确认 `local_path` 在配置、发布者、文档与 launch 中一致，且
   `incremental_v2` launch 不包含控制器或 `/Car/T5/Car_Cmd_Vel`。
3. 在独立 Jazzy 构建目录完成受影响接口和控制器构建/测试；随后在 Humble 环境复建并
   运行受影响包测试。
4. Jetson AGX Orin、DDS、真实 `/tf`、底盘、急停和车辆闭环均单列为 `NOT_RUN`，除非
   本任务实际在相应环境执行。
