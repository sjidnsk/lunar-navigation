# 增量规划真实控制闭环 RViz demo

用户于 2026-09-10 批准执行。基线 integration/pure-planner-orin@df469d1。

目标：在独立分支、独立 ROS 域中验证 LocalGoalRegion、incremental_reference PathExecutor + TrackingStatus、正反向 0.2 m/s 上限三项修改。

链路：模拟高程及定位 → 正式增量地图/粗指导/平台局部搜索 → PathReference → 正式 PathExecutor → /lunar_demo/controller/cmd_vel → 受加减速约束的车辆积分 → 实际模拟 odometry + 控制器 TrackingStatus → 导航。

保持原粗指导路线和旧 demo；不导入 KnownSpaceRoutePlanner、返航、DRL 策略；不发布硬件速度话题。候选集合不是可达性证明，必须保留平台边和终态认证。模拟器不能直接沿 PathReference 移动，也不能靠截断掩盖控制器速度超限。

提供固定绕障、多局部出口、前进/倒退、终点朝向场景及手工 RViz 目标。RViz 展示候选、选中目标、路径、实际轨迹、速度、执行阶段，不显示 OPEN/CLOSED。每个场景记录规划、执行、命令/实际速度和结果。正式规划成功须 PLAN_FOUND、有有效引用；到达须匹配 session/revision 且停稳后完成。保持失败证据。

范围只证明本机 Jazzy 仿真，Humble、Orin、实车明确 NOT_RUN。提供构建启动操作文档、变更来源和验证记录。
