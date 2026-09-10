# 增量规划控制闭环 RViz demo

本 demo 原开发于独立分支 `feat/incremental-controller-rviz`，基线为 `df469d1`；本次集成到 `integration/pure-planner-orin`。
旧 `incremental_exploration_navigation_rviz.launch.py` 继续保留。本 demo 使用正式增量规划和正式轮式控制器。

## 改动及来源

| 内容 | 来源 | 效果 |
|---|---|---|
| LocalGoalRegion / SelectRolling / wheel 和 legged 搜索 | `bd39d26` 的定向提取 | 在候选集合中由平台搜索同时选点与求路，保留粗指导路线、64 m 局部窗口和平台边认证 |
| PathExecutor / TrackingStatus / session 与段确认 | `51927e6` 的定向提取 | 正式控制器执行引用；仅当前任务当前段的停稳反馈允许完成/切段，失败停稳后请求重规划 |
| 双向 0.2 m/s | `a1cb8bf` 配置及严格配置校验器 | 同时限制规划能力配置和控制输出；记录器另外检查截断前命令及实际速度 |
| 演示包 | 本次新增 | 速度驱动模拟车辆、合成高程/定位、固定场景和证据记录、RViz 显示 |

没有导入 KnownSpaceRoutePlanner、返航、历史路径复用或 DRL 策略。本 demo 不启用前沿探索器；固定案例和手工目标通过同一个 NavigateToPose 接口调用正式规划。
本次集成包含正式规划/控制行为修改；原 demo 工作树和已安装程序保留，现有运行实例不会自动切换到新源码。

## 构建

本机 Jazzy；产物保存在工作树之外，不与 Humble overlay 混用。

```bash
cd /home/kai/WS/lunar-navigation/lunar-runtime/.worktrees/pure-planner-orin
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/lunar-mainline-controller-demo/log build \
  --base-paths ros2_ws/src \
  --build-base /tmp/lunar-mainline-controller-demo/build \
  --install-base /tmp/lunar-mainline-controller-demo/install \
  --packages-up-to lunar_incremental_controller_demo \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DLUNAR_INCREMENTAL_NAVIGATION_BUILD_DEBUG=ON
source /tmp/lunar-mainline-controller-demo/install/setup.bash
```

## 手工操作

终端 1：选用一个没有其他节点的独立域。本文用 72，旧 demo 用 71。

```bash
source /opt/ros/jazzy/setup.bash
source /tmp/lunar-mainline-controller-demo/install/setup.bash
export ROS_DOMAIN_ID=72 ROS_LOCALHOST_ONLY=1
ros2 launch lunar_incremental_controller_demo controller_rviz.launch.py \
  case:=manual start_rviz:=true
```

RViz 点击 **2D Goal Pose**，在地图上按下设定位置、拖动箭头设定朝向。目标经 goal bridge 送往正式导航。
青色点是可选局部目标；紫色球是已搜索得到的端点；规划路径与实际轨迹分开显示；HUD 显示实际速度、方向、执行阶段和段编号。
车辆会按控制器命令移动并在终点停稳。未启动自动探索器，因此不会有第二个任务源覆盖手工目标。

所有业务话题使用 `/lunar_demo/controller/*`。`cmd_vel` 只有正式控制器一个发布者，只供模拟车辆订阅；没有接到 `/Car/T5/Car_Cmd_Vel`。
TF 名称保持 map/odom/base_link，因此域隔离仍然必要。停止这套 demo 用其启动终端 Ctrl+C，不要批量结束其他 ROS 进程。

## 固定场景验证

每个场景重新启动整套链路，以清零车辆、持久地图、引用和累计统计。以倒车为例，终端 1 改为：

```bash
ros2 launch lunar_incremental_controller_demo controller_rviz.launch.py case:=reverse
```

终端 2 使用相同 overlay 和 ROS 域：

```bash
source /opt/ros/jazzy/setup.bash
source /tmp/lunar-mainline-controller-demo/install/setup.bash
export ROS_DOMAIN_ID=72 ROS_LOCALHOST_ONLY=1
ros2 run lunar_incremental_controller_demo run_case --case reverse \
  --timeout 240 --output /tmp/lunar-controller-reverse.json
```

| case | 操作/验证目的 |
|---|---|
| forward | 3 m 前进，速度上限和停稳完成 |
| reverse | 目标在车后且保持初始朝向，检查实际负向运动及反向上限 |
| final_yaw | 前进到目标后完成指定 π/2 朝向，验证 FINAL_ALIGN 与停稳反馈 |
| detour | 固定障碍挡住直线，检查绕障路径和实际足迹没有碰撞 |
| multiple_exits | 10 m 观测随车推进，检查多通道、非最终引用和停稳后的段切换 |

记录器会自动发目标，无需在 RViz 点选。JSON 包含命令/实际最大正反向速度、碰撞/超限计数、任务和段引用、阶段变化、导航诊断及最终判定。`passed: true` 要求有效规划引用、PLAN_FOUND、匹配最终停稳完成、Action GOAL_REACHED、几何误差和场景检查全部通过。失败保留 JSON，进程返回 1。

## 仿真边界

车辆按墙钟实时运行，平面 SE(2) 加减速模型不代表实车动力学。地图来自合成高程；多出口用 360° 圆形观测，无遮挡、噪声或真实传感器模拟，其余场景给定完整静态地图。详细模型见演示包 README。

源码测试、本机 Jazzy ROS 闭环和 RViz 显示分别记录在 [验证记录](增量规划控制闭环demo验证.md)。Humble、Orin、实车均 `NOT_RUN`。
