# 增量规划真实控制闭环：仿真器与记录器

本包仅用于独立 ROS 2 Jazzy 仿真域。完整链路启动见仓库操作文档。
`vehicle_sim` 只订阅 `cmd_vel` 决定运动；不订阅路径或目标，不发布 TrackingStatus。
TrackingStatus 由正式控制器发布，本包只读用于 HUD 和证据。

车辆按真实墙钟 50 Hz 积分 SE(2)，正反向速度限制均为 0.2 m/s，
加速 0.3 m/s²、制动 0.5 m/s²，角速度限制 0.6 rad/s，角加速度 0.5 rad/s²。
命令超过 0.3 s 未更新则制动。实际 odometry twist 来自积分状态。
碰撞使用 1.182 × 0.818 m 矩形足迹的分离轴测试，接触时阻止穿障并记录失败。
原始命令的最大正反速度和超限次数在截断之前记录，不能由仿真限速掩盖控制器错误。
这是确定性平面运动仿真，不含轮滑、动力学、感知噪声或真实传感器。

所有业务话题前缀为 `/lunar_demo/controller`；TF 使用 `map -> odom -> base_link`，
必须用独立 ROS_DOMAIN_ID。不要与其他仿真器共用一个域。默认实时，不使用 `/clock`。
地图 2 Hz 发布，分辨率 0.2 m、默认 32 m 方形。`multiple_exits` 仅提供车体周围
10 m 半径的 360° 合成观测，其他位置是 NaN，靠正式持久地图积累；其余场景完全已知。
RViz 的 Synthetic terrain 是场景真值，仅用于显示，不是规划器占据图输入。

| case | 目标 x,y,yaw | 检查内容 |
|---|---|---|
| forward | 3,0,无 | 前进、停稳完成 |
| reverse | -3,0,0 | 实际倒车速度大于 0.05 m/s，保留朝向 |
| final_yaw | 3,0,π/2 | 终点角误差 ≤0.2 rad |
| detour | 6,0,无 | 绕过 x∈[2,3],y∈[-1.8,1.8] 岩石 |
| multiple_exits | 12,0,无 | 多通道、滚动未知边界，必须出现非最终 PathReference |
| manual | RViz 点选 | 通过正式 goal bridge 发送目标 |

记录器只发送 NavigateToPose Action，不发送速度。每个案例需重新启动整套链路以清零状态。
启动完整链路且 source 同一构建 overlay 后，在第二个终端运行：

```bash
export ROS_DOMAIN_ID=72
ros2 run lunar_incremental_controller_demo run_case --case forward \
  --timeout 240 --output /tmp/lunar-controller-forward.json
```

记录器等待地图、车辆状态和 Action 服务最多 30 s，再预留 2 s 地图处理时间。
失败输出也保存 JSON，退出码为 1；不把 Action 成功单独视为验收通过。
通过要求 PLAN_FOUND 与活动引用、同 session/revision 的最终 COMPLETED、Action GOAL_REACHED、
实际停稳、几何误差合格、无原始超速/无效命令/碰撞，另加倒车或非最终引用场景约束。
运行超时将请求取消并保存证据。默认场景 240 s 上限用于实时低速执行。

纯车辆测试（仓库根目录）：

```bash
python3 -m pytest -q ros2_ws/src/lunar_incremental_controller_demo/test
```

源码单测不能代替完整链路证据。本包不证明 Humble、Orin、DDS 实机或实车就绪；这些层级 `NOT_RUN`。
