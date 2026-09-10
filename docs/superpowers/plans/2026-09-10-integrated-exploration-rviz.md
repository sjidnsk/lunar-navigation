# Integrated Exploration RViz Implementation Plan

> For agentic workers: use superpowers:subagent-driven-development for scoped implementation and review.

**Goal:** 保留简单版，交付可视完整地图/探索/规划/控制的复杂地形demo。
**Architecture:** 新演示包复用正式增量栈与控制器，合成传感器/车辆只接标准接口；统一模拟时钟，粗细图独立可视。
**Tech Stack:** ROS2 Jazzy, C++20, Python/numpy, RViz, GTest/pytest.
**Spec:** ../specs/2026-09-10-integrated-exploration-rviz-design.md

## Global Constraints

独立 worktree、域73；0.2m fine、1m coarse、64m规划窗、双向0.2m/s；用户指定默认30倍可调。禁止真值输入规划，禁止另加速度发布者。不得修改保留的旧工作树；产物 /tmp；Humble/Orin/实车 NOT_RUN。

- [x] 1. 复杂解析地形与范围/遮挡观测：新 lunar_integrated_exploration_demo/terrain.py + tests；验证确定性、复杂度、坐标、遮挡、起点及物理碰撞。
- [x] 2. 正式探索调度修复：incremental_exploration_node.cpp 及测试；先复现不变地图停等，再保留失败/重复成功抑制下修复。
- [x] 3. 新模拟车辆/时钟：复用已验证Plant，cmd_vel驱动、统一/clock和动态倍率；观测/odometry/TF、遥测，保持模拟时间速度约束。
- [x] 4. 完整launch、自动任务启动、全局/局部RViz和HUD；使用正式explorer/navigator/executor，不引入KnownSpace/DRL/返航。
- [ ] 5. 串行构建测试、独立域live探测与小区域闭环、大场景多目标推进；审查并修复实际失败。
- [x] 6. 操作及验证文档，默认启动完整demo，记录域/进程/结果/边界，不影响原有demo。

验证进度：用户反馈驱动修复了局部传感器的首命中表面漏测、正式FineBuilder扩图后新格漏派生。主构建11包通过，本轮core 233、navigation_ros 76、新demo 111，共420项通过，无本轮xfail。300m有界多目标闭环通过；第5项仍有明确未通过结果：20m任务最新95.25%、19m²未知、8目标后等待，COMPLETED门禁未通过；此前全量旧test_exploration_node超时/缺结果仍未定位。观测、通行性、探索投影的语义区别和失败记录在正式验证文档，不能将此勾选为全部验证通过。局部0.2m探索/全局1m调度分层尚未实现，当前传感器/细图是0.2m而探索器仍用1m。
