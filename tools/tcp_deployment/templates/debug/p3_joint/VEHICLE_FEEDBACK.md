# P4 复用旧车辆接口反馈

日期：2026-09-16。只修改 P4 联调目录，不修改 P1/P2/P3 文件。

## 当前链路

旧 lunar_car_node 独占 192.168.10.22:6668，ROS 域 10 发布 /car/telemetry。
P4 legacy_feedback_input.py 订阅它的原始 x_lh/y_lh/z_lh、qx_lh/qy_lh/qz_lh/qw_lh，
直接导入已安装 lunar_obj_tcp_sim.coordinates 的 world_position、orientation、PoseRates。
包含原 P4 的 Actor 车体轴转换，速度由位姿变化计算；不使用旧接口的原始 vx/wx 作为控制速度。
不创建 TCP 客户端，不发布车辆控制指令，不写入共享 /tf 或 /tf_static。

启动时等待约 6 秒车辆静止和多帧稳定 P3 位姿，以平面刚体变换对齐 P3 odom，
然后在域 57 发布 /P4/input/odometry（odom -> base_link）。
原静态 map->odom 适配保持不变。导航器、探索器、控制器及 RViz Odometry 显示均已切换。
P3 地图与 P3 本身的里程计发布不变。
旧接口接收时间戳保留；导数使用本地单调接收时间；约 0.2 秒窗口计算线速度。
导数无效时发布 NaN 和高协方差，不能伪装成已经停稳；断流不重复发布旧位姿。
P4 控制器 odometry_timeout_s 恢复为 0.5 秒。TF 阈值仍为 5 秒。
线速度上限 0.2、角速度上限 0.1、终点容差 0.1 m / 0.05 rad、盲区余量 3.0 m。

## 验证

本机 Python 4 项单元测试通过；原生 Humble/Orin 同样 4 项通过。
覆盖原桥车体轴转换、P3 对齐旋转、速度保持在车体系、断流导数无效、非法四元数。
三个运行节点参数均确认 odometry_topic=/P4/input/odometry。
输出实测约 15.5～15.7 Hz；一次 8 秒停车观测中最大接收间隔约 0.437 秒。
初次停车验证末帧时间差约 6 ms，此数不是端到端传感器延迟证明。

有界运动测试：规划 PLAN_FOUND、ACTIVE 路径；采样状态均 TRACKING，
倒车指令达到 -0.2 m/s，反馈线速度符号为负。
设置 0.5 m 位移阈值，最终旧接口位姿净位移约 0.537 m 后停车（含停止响应余量）。
未出现 CURVATURE_INFEASIBLE；测试主动限距结束，不是 GOAL_REACHED。
运动期间抽样的最新反馈接收年龄最大约 0.176 秒，不等同于完整消息最大间隔。
停车后适配位姿与稳定 P3 位姿平面差约 0.0114 m，速度为零。
ss 检查仍仅 lunar_car_node PID 124754 持有唯一 6668 连接。
已请求 park，末次四轮指令为零。探索当前空闲，不自动恢复此前任务。
RViz 和反馈输入保持运行；未留下常驻运动转发脚本。

## 使用与边界

start_preview.sh 已包含新反馈输入和探索节点；它拒绝已有 P4 进程时重复启动。
启动需要旧车辆接口运行，以及 P3 odometry 和 map->odom 可用；车辆须停稳以完成初始对齐。
输入状态：/P4/input/feedback_status；校准记录每次启动写 feedback-alignment.json。
状态 READY 仅表示输入准备好，不代表完整定位标定或自动导航验收。
启动脚本会 source 旧接口 overlay 以读取 VehicleTelemetry 消息，再 source P4 overlay。
适配器只有读反馈权限；实际运动仍需显式运行既有 P4 车辆联调转发工具。

重要限制：
- 当前是停车起点的平面坐标对齐，尚非地标标定或长距离漂移修正；不会不断拉动位姿来追逐延迟的 P3。
- P3 若重置定位、改变 odom 原点或 UE 车辆重置，须停车并重启 P4 反馈适配器重新对齐。
- 未解决车辆速度增益、轮径配置差异；本轮反馈短窗速度幅值曾约 0.29 m/s，而指令上限 0.2 m/s。
- 不能把导数速度当成 IMU，也不能把接收端时间戳当成仿真测量时间。
- 完整终点位置/朝向收敛、长距离探索、实物车辆均未验证。

回退配置保存在 *.before-vehicle-feedback；请先停止 P4 再恢复这些文件，
不要触碰旧车辆接口或 P3 进程。
本次未重打包 P4.tar.gz。
