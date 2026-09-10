# 增量规划控制闭环 demo 执行计划

Goal: 完成用户批准的三项改动隔离测试 demo。
Architecture: 复用生产核心和控制器，通过独立模拟车辆闭环，不另写演示规划器。
Tech Stack: ROS 2 Jazzy / C++17 / Python / RViz / pytest / GTest。
Spec: ../specs/2026-09-10-incremental-controller-rviz.md
Global Constraints: 独立 worktree；无 merge/push/PR；保护其他会话；构建和证据写 /tmp；硬件话题零发布者。

1. 选择性移植 LocalGoalRegion（bd39d26）：core local/wheel/legged 和测试；避免返航依赖。运行相应核心测试，保留粗指导。
2. 选择性移植控制器及消息（51927e6）、0.2 m/s（a1cb8bf）：controller、TrackingStatus、配置及验证器；运行 Python 测试。
3. 集成 coordinator 和 ROS navigator：选择性加入停稳反馈、session/revision 防串扰、真实区域选点和调试 marker；运行相关 GTest。
4. 添加独立演示包：cmd_vel 驱动的 SE2 车辆、加速度限制、实际 odometry、固定地图/场景、任务客户端、RViz 显示与启动脚本。先测试物理积分及场景判定再实现；保留旧 demo。
5. 独立 Jazzy 构建受影响包及测试，串行运行核心 CTest；启动新域运行固定场景，记录成功和失败原因、命令/实际限速、反馈匹配、硬件话题隔离。
6. 审查整体差异及范围，修复问题；补充操作与验证记录，标注 Humble/Orin/实车 NOT_RUN，不将仅有路径当作完成。

执行结果（2026-09-10）：六项任务已完成。最终包测试 693 项无失败；相关静态契约 57 项通过、4 项旧 live 用例跳过。五个正向场景通过，多出口实际执行两段；4.5/6.5 m 短观测边界保留。新增 RViz 在域 72 运行，旧域 71 保留。详细修改与失败修复见 ../../增量规划控制闭环demo验证.md。未合并、推送或创建 PR，Humble/Orin/实车 NOT_RUN。
