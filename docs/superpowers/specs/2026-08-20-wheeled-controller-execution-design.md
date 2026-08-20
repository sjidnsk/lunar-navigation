# WHEELED 参考轨迹执行控制器设计

**状态：** 已批准，待实施  
**日期：** 2026-08-20  
**范围：** 将已认证的 WHEELED `MotionReference` 安全地跟踪为底盘 `geometry_msgs/msg/Twist` 命令，并把执行状态反馈给 `lunar_planner_ros`。

## 1. 决策与边界

新增独立 ROS 2 Python 包 `luna_wheeled_controller`。它只消费规划器已经产出的 WHEELED 参考；不生成候选点、不重新规划、不发布 TF、不改写地图，也不处理 LEGGED 或 HOPPER。

底盘命令唯一输出为：

```text
/Car/T5/Car_Cmd_Vel   geometry_msgs/msg/Twist
```

该 Topic 与已提供的手动键盘控制器一致。自动控制器和键盘控制器不得同时运行；启动时应由操作者二选一。运行时开关默认为关闭：`controller.wheeled.enabled: false`。关闭时，`luna` 不启动该节点，规划仍可输出 `MotionReference` 供外部控制器消费。

## 2. 输入、输出与数据流

```text
/plan_motion Action Result
  → MotionReference(WHEELED, plan_id, trajectory/path_preview)
  → luna_wheeled_controller
  → /Car/T5/Car_Cmd_Vel : Twist

/Car/T3/semantic/current_pose : Odometry
  → luna_wheeled_controller

luna_wheeled_controller
  → /execution/motion_feedback : MotionExecutionFeedback
```

控制器通过 `PlanMotion` Action 客户端顺序请求目标，而不是订阅、猜测或重建规划器内部状态。收到有参考的结果后，验证：

- `reference.platform_type == WHEELED`；
- `plan_id` 非空；
- `path_preview` 或地面 `trajectory` 至少包含可跟踪的平面点；
- 参考、里程计和 TF 都在配置的最大新鲜度内；
- 坐标为有限数，路径单调推进且终点有效。

每条反馈都复制 `platform_type`、`plan_id` 和当前 `segment_id`，并为该控制器进程单调增加 `sequence`。执行控制器不伪造成功：只有位置和航向均进入终点容差时才发 `SEGMENT_COMPLETE`。

## 3. 跟踪算法

采用有限状态机：`IDLE → ACCEPTED → EXECUTING → SEGMENT_COMPLETE`，失败或停止可转入 `FAILED`/`CANCELED`。

在 `EXECUTING`，控制器以当前底盘 `base_link` 位姿将参考路径投影到车辆前方，选取距离为 `lookahead_m` 的路径前视点。使用 Pure Pursuit 曲率：

```text
curvature = 2 * lateral_error / lookahead_m^2
angular.z = clamp(linear.x * curvature, -max_angular_radps, +max_angular_radps)
```

`linear.x` 受最大速度、最大加速度和终点距离约束；接近终点时以制动距离收敛到零。路径点本身不直接下发速度，车体以定位闭环跟踪路径。默认参数从配置读取，不内嵌到代码：`control_rate_hz`、`lookahead_m`、`goal_position_tolerance_m`、`goal_yaw_tolerance_rad`、`max_linear_mps`、`max_angular_radps`、`max_cross_track_error_m`、`reference_max_age_s` 和 `odometry_max_age_s`。

## 4. 安全与故障语义

下列任一情形必须在同一控制周期先发布零 `Twist`，再发布对应反馈；不等待下一次规划：

| 条件 | 反馈状态 | `reason_code` |
| --- | --- | --- |
| 参考无效、非 WHEELED、路径不可跟踪 | `FAILED` | `INVALID_REFERENCE` |
| 参考、里程计或 TF 过期/缺失 | `FAILED` | `STALE_INPUT` |
| 横向偏离超过阈值 | `FAILED` | `PATH_DEVIATION` |
| 上游 Action 返回无参考或安全保持 | `CANCELED` | `REFERENCE_WITHDRAWN` |
| 节点被停用、停止或替换已有参考 | `CANCELED` | `CONTROLLER_STOPPED` |

控制器进程退出、异常处理和 ROS lifecycle 停用都必须尝试发布一次零 `Twist`。任何检测失败均保持零速度；不会改由键盘控制或允许旧参考继续运行。

## 5. `luna` 集成

`runtime.yaml` 增加可选段：

```yaml
controller:
  wheeled:
    enabled: false
    command_topic: /Car/T5/Car_Cmd_Vel
    odometry_topic: /Car/T3/semantic/current_pose
    feedback_topic: /execution/motion_feedback
    control_rate_hz: 20.0
    lookahead_m: 1.0
    max_linear_mps: 0.2
    max_angular_radps: 0.5
    max_cross_track_error_m: 1.0
    goal_position_tolerance_m: 0.25
    goal_yaw_tolerance_rad: 0.35
    reference_max_age_s: 1.0
    odometry_max_age_s: 0.5
```

默认关闭保持既有部署行为。`luna start` 在启用时启动统一联调 launch，其中规划器、课题三适配器和控制器具有单一受管生命周期；配置检查拒绝非正频率、非正时效/容差、空 Topic 与速度上限缺失。`luna status` 应显示控制器 PID/状态与最近反馈原因。

## 6. 测试与验收

纯函数测试：前视点选择、直线/弯道控制、终点减速、速度/角速度限制和非有限输入拒绝。ROS 节点测试：有效参考的 `ACCEPTED → EXECUTING → SEGMENT_COMPLETE` 反馈；非 WHEELED、过期里程计、过期参考、路径偏离、替换/停止均先发零 `Twist` 后反馈。

集成测试只使用假的 `Twist` 订阅者和里程计发布者，绝不接触真实底盘。实机联调另行执行：验证底盘命令 Topic、线/角速度方向、控制超时行为和急停路径；在实机验证完成前，默认配置保持 `enabled: false`。

## 7. 非目标

- 不改变 `PlanMotion`、`MotionReference` 或 `MotionExecutionFeedback` 消息定义。
- 不把键盘控制脚本打包、复用为自动控制器或与其同时运行。
- 不实现全局地图处理、障碍物绕行、Nav2 行为树、低级驱动、LEGGED 执行或 HOPPER 飞行控制。
