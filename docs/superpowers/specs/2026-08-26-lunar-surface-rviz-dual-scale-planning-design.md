# 双尺度 RViz 路径规划实验设计

## 目标

在本机 ROS 2 Jazzy 中提供一个完全隔离的轮式路径规划实验：RViz 同时展示
`1000 m × 1000 m` 低分辨率全局地图和以车辆为中心的 `64 m × 64 m`
高分辨率局部地图；用户通过 RViz 设置起点和终点；终端对每次规划周期输出一行
起点、终点、规划时间、全局路径长度、局部段长度和结果；RViz 分色展示完整全局路径与
当前局部可执行段。

## 安全与范围

- 所有新增接口都位于 `/lunar_demo/*`，不得发布或订阅 `/Car/T5/Car_Cmd_Vel`。
- 不启动控制器、探索器或生产 Action `/Car/T4/plan_motion`。
- 任意 RViz 起点必须通过地图边界和占据检查；不得为选中的起点清障或创建人工通道。
- UNKNOWN 不得作为可执行起点或局部可执行目标。
- 生产规划器继续负责发布空路径来清除失败后的陈旧路径；demo 显示链不得削弱该语义。

## 地图模型

全局图保持 `1000 × 1000 cells @ 1.0 m/cell`，原点为 `(-500,-500)`。
场景由固定 seed 的连续解析高程、陨石坑和岩石集合描述。全局 OccupancyGrid 在 1 m
cell 中保守聚合占据状态。

局部 GridMap 固定为 `320 × 320 cells @ 0.2 m/cell`，物理范围 `64 m × 64 m`，
中心跟随当前车辆位置。局部 `occupancy` 和 `elevation` 直接在 0.2 m cell 中查询同一场景
模型，而不是把 1 m 全局 cell 重复五次。局部范围外或全局边界外为 NaN。

本机 RViz 即使单独使用 `Map` display 也会触发 GLSL sampler 冲突，因此显示链使用
`MarkerArray` 绘制经典地图。全局和局部都使用同一套图例：与当前规划起点四邻域连通的
安全区域为浅绿色，单格安全但不属于起点连通分量的区域为灰绿色，阻塞/风险区为橙红色，
UNKNOWN 为灰色，障碍核心为黑色。连续底色加稀疏覆盖层避免重新形成棋盘格。全局图由
`1000×1000 @ 1.0 m` GridMap 经过与局部图相同的轮式可通行性核心生成；局部图保持
`320×320 @ 0.2 m`。连通性只改变显示分类，不把 UNKNOWN 或断开区域变成规划可执行目标。
所有地图 Marker 都放在场景最低高程以下，保证全局与局部路径位于地图之上、不会被遮挡。

demo 仍发布 `/lunar_demo/local_map_viz` `nav_msgs/OccupancyGrid` 作为可检查的显示数据，并从
原始 `/lunar_demo/traversability` 生成 `/lunar_demo/traversability_viz`；规划器继续消费原始
`/lunar_demo/grid_map`，所有显示转换均不参与规划。

## RViz 交互

- `2D Pose Estimate` 发布 `geometry_msgs/PoseWithCovarianceStamped` 到
  `/lunar_demo/start_pose`。
- demo node 校验起点；合法时取消/失效当前模拟路径，更新车辆 pose、odometry 和局部地图。
- `2D Goal Pose` 发布 `geometry_msgs/PoseStamped` 到 `/lunar_demo/rviz_goal`，由现有
  goal bridge 转成 `/lunar_demo/plan_motion` Action 请求。
- 手动实验默认不自动发送目标；默认目标仅保留为可视参考，可通过 `auto_goal:=true` 恢复。

起点和终点通过单独 MarkerArray 显示：起点蓝色、终点橙色；车辆保持洋红色大尺度标记；
局部窗口使用青色方框显示。

## 路径与终端输出

- `/lunar_demo/global_path`：完整全局路径，RViz 使用橙黄色粗线。
- `/lunar_demo/wheeled_path`：当前滚动局部段，RViz 使用青色实线。
- 两个 RViz Path display 直接消费规划器输出，使空 Path 能清除旧显示。

新增 `lunar_surface_reporter_node`。它记录目标到达时的最新 odometry 作为本次起点，订阅
global/local Path、TimedPath 和 request diagnostics。路径长度定义为相邻 pose XY 欧氏距离之和。
每个成功规划周期输出一行：

```text
[PLAN 003] OK start=(-349.50,0.50) goal=(350.50,0.50) time=1284.6 ms path=712.3 m local=12.1 m reason=PLAN_FOUND
```

失败输出 `path=N/A local=N/A`。成功判据必须同时满足 `planning_outcome=0`、
`has_reference=true` 且 reason 为 `PLAN_FOUND` 或 `PLAN_FOUND_LATE`。

## 启动与就绪边界

`lunar_surface_rviz_demo.launch.py` 启动 scene、planner、全局/局部两个 traversability 节点、
goal bridge、visualizer、reporter 和 RViz。全局图与 `map→odom` 仅在启动窗口重复发布，随后冻结；
局部图和 odometry 持续发布。手动目标必须在地图、TF、odometry 和 local map 均被 planner
消费后设置。

## 验收标准

1. 单元/契约测试证明全局图为 1000 m、局部图为 64 m/0.2 m/320 cells。
2. 两个落在同一 1 m global cell 内的 0.2 m 查询可以得到不同高程，证明局部图不是简单复制。
3. 合法 RViz 起点更新 odometry 和局部图中心；障碍或越界起点不改变车辆状态。
4. RViz 同时展示同色图例的全局/局部可通行性；与起点不连通的安全格不得显示为绿色；地图层
   位于路径以下，并包含起点、终点、障碍物、全局路径和局部段路径。
5. reporter 的手算路径长度、成功/失败格式和 N/A 语义有自动化测试。
6. Jazzy 构建成功，相关测试全部通过。
7. 隔离 E2E 观察到全局 `1000×1000 @ 1.0`、局部 `64×64 @ 0.2`、
   `planning_outcome=0`、`reason_code=PLAN_FOUND`、`has_reference=true`，RViz 两类路径 topic
   均收到非空 Path，终端出现一行正确摘要。

## 非目标

- 不验证真实车辆执行、控制器、Orin/Humble 部署或底盘指令。
- 不改变纯规划算法、生产 topic 名称或地图安全阈值。
- 不为保证演示成功而创建隐藏走廊或绕过 footprint/obstacle 检查。
