# 双尺度 RViz 路径规划实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建可在 RViz 交互设置起终点、同时显示 1 km 全局图与 64 m 局部图、分色展示全局/局部路径并在终端输出简洁规划摘要的隔离 Jazzy demo。

**Architecture:** 保留现有 `/lunar_demo/*` planner Action 和 1 km 场景，扩展场景为可在任意世界坐标高分辨率采样；demo node 接受 RViz 起点并发布 `1000×1000 @ 1 m` 全局 GridMap、`320×320 @ 0.2 m` 局部 GridMap 与显示用 OccupancyGrid。两个 GridMap 经过同一轮式可通行性核心，visualizer 再按当前起点的四邻域连通分量生成同色图例的经典地图风格 MarkerArray，以规避本机 RViz Map shader 冲突。地图统一置于路径以下；规划器原生 global/local Path 直接进入 RViz，独立 reporter 对路径和 diagnostics 做单行汇总。

**Tech Stack:** C++20、ROS 2 Jazzy、rclcpp/rclcpp_action、nav_msgs、grid_map_msgs、visualization_msgs、ament/colcon、GoogleTest、pytest。

**Spec:** `docs/superpowers/specs/2026-08-26-lunar-surface-rviz-dual-scale-planning-design.md`

## Global Constraints

- 所有实验接口必须位于 `/lunar_demo/*`；不得启动控制器或使用 `/Car/T4/*`、`/Car/T5/*`。
- 全局图固定 `1000×1000 @ 1.0 m`；局部图固定 `320×320 @ 0.2 m`，即 `64×64 m`。
- 起点必须校验边界与占据，不得清障、造走廊或削弱 planner 安全检查。
- RViz 必须直接消费 planner 的 global/local Path，保留空 Path 清屏语义。
- 成功展示必须核对 `planning_outcome=0`、`PLAN_FOUND`/`PLAN_FOUND_LATE` 和 `has_reference=true`。

---

### Task 1: 场景高分辨率查询和 64 m 局部地图

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_ros/include/lunar_pure_planner_ros/lunar_surface_scenario.hpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_scenario.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_demo_node.cpp`
- Test: `ros2_ws/src/lunar_pure_planner_ros/test/lunar_surface_scenario_test.cpp`

**Interfaces:**
- Produces: `LunarSurfaceScenario::Sample(double x_m, double y_m)`，返回占据与高程；demo 发布 320×320 GridMap 和 local visualization OccupancyGrid。

- [ ] 写失败测试：验证 64 m 几何、合法/越界采样，以及同一 global cell 内的亚米级高程变化。
- [ ] 运行 `lunar_surface_scenario_test`，确认因缺少 Sample API/旧 100 m 常量失败。
- [ ] 实现场景查询和 320×320 局部消息构造。
- [ ] 重跑测试并确认通过。

### Task 2: RViz 起点重置与交互标记

**Files:**
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_demo_node.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_visualizer_node.cpp`
- Modify: `rviz/lunar_surface_demo.rviz`
- Test: `tests/launch/test_lunar_surface_demo_contract.py`

**Interfaces:**
- Consumes: `/lunar_demo/start_pose` `PoseWithCovarianceStamped`。
- Produces: 更新后的 `/lunar_demo/odometry`、局部地图中心、`/lunar_demo/pose_markers` 和 `/lunar_demo/local_window`。

- [ ] 写失败契约测试：要求 start topic、双 map、双 path 和 pose/window markers。
- [ ] 运行 pytest，确认旧 launch/RViz 配置失败。
- [ ] 实现合法起点更新、非法起点拒绝、活动模拟 path 清空和标记发布。
- [ ] 更新 RViz 工具、地图和路径 display，重跑契约测试。

### Task 3: 简洁规划 reporter

**Files:**
- Create: `ros2_ws/src/lunar_pure_planner_ros/include/lunar_pure_planner_ros/lunar_surface_reporter.hpp`
- Create: `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_reporter.cpp`
- Create: `ros2_ws/src/lunar_pure_planner_ros/src/lunar_surface_reporter_main.cpp`
- Create: `ros2_ws/src/lunar_pure_planner_ros/test/lunar_surface_reporter_test.cpp`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/CMakeLists.txt`

**Interfaces:**
- Produces: `PathLength(const nav_msgs::msg::Path&)` 与规划摘要格式器；节点订阅 goal/odom/global path/local path/timed path/diagnostics。

- [ ] 写失败测试：手算 3-4-5 路径长度、成功行和失败 `N/A`。
- [ ] 构建测试，确认 reporter API 缺失。
- [ ] 实现纯函数和节点关联状态，只在每个 diagnostics 周期输出一行。
- [ ] 重跑 reporter 测试并确认通过。

### Task 4: Launch、依赖与完整验证

**Files:**
- Modify: `launch/lunar_surface_rviz_demo.launch.py`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/package.xml`
- Modify: `ros2_ws/src/lunar_pure_planner_ros/CMakeLists.txt`
- Modify: `tests/launch/test_lunar_surface_demo_contract.py`

**Interfaces:**
- Produces: `auto_goal` launch argument，默认 false；完整 demo 一条 launch 命令启动。

- [ ] 将 reporter 和新参数接入 launch，补齐 ROS 依赖。
- [ ] 运行 pytest 契约测试和相关 CTest。
- [ ] 使用 `colcon build --packages-up-to lunar_pure_planner_ros` 做 Jazzy 构建。
- [ ] 启动 `start_rviz:=false auto_goal:=false` 的隔离 demo，核对 topic 类型和地图尺寸。
- [ ] 发送合法 start/goal，捕获 diagnostics、global/local Path 和 reporter 输出。
- [ ] 运行 `git diff --check` 并复核未出现 `/Car/T5` 或生产 Action 接口。
