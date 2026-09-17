# 验证记录（2026-09-17）

- 原实现回归：心跳中断测试失败，实际产生 park；其余 3 项通过。
- 修改后本机 Python：4/4，通过失联 2/60/3600 秒不抢占、不发 park，以及明确释放、人工急停和接管前手动控制。
- Orin /usr/bin/python3：4/4 同一组回归通过。
- Orin ROS Humble 原生隔离域 183、localhost：真实 ROS Bool 接管一次，静默 4 秒；没有 park、没有 GUI cmd_vel；随后人工急停和明确释放均输出 park/零速，PASS。不连接 Unreal。
- Orin 现场替换：P4 GUI PID 379811，启动身份和源文件 SHA 校验通过；重复 ensure_gui 调用复用同一 PID。
- 原 P3 GUI 文件 SHA 保持一致。只退出并替换旧 GUI，没有重启 P3 建图、P4 导航或 TCP 接口。
- start_nodes.sh 增加 ensure_gui 调用，bash -n 通过；未为了测试此入口而重启现场整套导航。
- GUI 关闭时原有急停生效，既有外部停车锁定未自动解除，没有自动重发旧目标。
- 现场最终检查 Unreal TCP 处于 SYN-SENT、输入过期，因此真实车辆定位中断后的恢复跟踪 NOT_RUN。此状态不能由隔离 ROS 测试推断通过。

运行日志、环境备份等保存在 Orin 的 vehicle_interface/gui 下，不纳入源码。
