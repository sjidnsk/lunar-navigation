# P4 车辆界面：取消控制权心跳超时

旧界面在 1.5 秒未收到 `/car/external_control_lease` 后发送 `park`，被 P4 解释为外部停止并取消导航。此副本保留原界面，只取消控制权的时间期限。

- 接收到 Bool true 后，P4 持续拥有速度输出权，GUI 不再根据时间接回控制或自动发送 park。
- Bool false 是明确释放：界面清零滑条并停车。人工停车、急停保持有效。
- P4 仍发送兼容原接口的 Bool；它现在表示接管/释放，不再是本界面的超时条件。
- P4 临时定位失效时零速度等待的逻辑不变。车辆接口原有 cmd_vel 中断停车仍保留，避免沿用旧的非零命令；它不等同于 GUI 的外部取消。
- P4 异常退出且没有释放时，GUI 不自动抢回速度输出。人工急停仍可用，恢复 P4 或重启界面才能恢复控制权流程。

Orin 安装目录 `/home/yanfa/P4/vehicle_interface/gui`。P4 的 `debug/p3_joint/start_nodes.sh` 在启动各节点前调用本目录 `ensure_gui.py`。它仅替换明确识别的旧车辆 GUI，复用原进程域和桌面环境；不会修改 P3 文件、重启建图或 TCP 接口。P4 GUI 已运行时不重复启动。

`ensure_gui.py` 保存原 GUI 进程信息到本目录 backups（含环境，权限受限，不纳入源码包）。运行日志 gui.log；runtime.json 记录 PID、启动身份和代码哈希。替换 GUI 会通过原退出逻辑停车，不会自动恢复已取消的导航目标。

验证：本机 `python3 -m pytest -q deployment/p4_vehicle_gui/test/test_ownership.py`；Orin ROS Humble 隔离域 183 执行 `test/native_ros_check.py`，禁止在域 183 已有任务时运行。真实车辆定位中断及自动恢复需另行验证。
