# Orin 可选 RViz 与新 Demo 操作

正式运行接收外部 T3 高程、定位与 TF；demo 自动提供模拟输入和轮式执行闭环。
RViz 目标桥接只把 PoseStamped 转成正式 NavigateToPose Action，不发布速度，不更换规划算法。

## 构建与依赖

进入复制到 Orin 的部署文件夹，ROS 2 Humble 已安装时执行：

```bash
# 7 个正式运行包，包含目标桥接；不构建模拟器
bash scripts/orin/build.sh
# 9 个包：正式运行包 + 简单控制 demo + 复杂探索 demo
bash scripts/orin/build.sh demos
```

脚本只构建，不自动下载依赖。`rosdep install` 是可选的依赖安装方式，会调用 apt 等包管理器，可能联网；已有依赖时跳过即可。
RViz 需要 `rviz2`，demo 还需要 numpy 和 ROS Python/消息包。
如构建报告缺包，按缺失项安装或从离线软件源补齐。允许联网且已配置 ROS apt 源时，例如：

```bash
sudo apt install ros-humble-rviz2 python3-numpy
```

无桌面/显示服务器的 SSH 终端使用 `false` 关闭 RViz。显示 RViz 需要 Orin 图形桌面、远程桌面或可用的 X11 转发；拖入源码本身不提供图形通道。

## 正式导航与 RViz 选点

导航脚本参数：`[wheel|legged] [use_sim_time] [config_path] [start_rviz]`。默认不打开 RViz。

```bash
# 无 RViz
bash scripts/orin/start_navigation.sh wheel false
# 同时打开 RViz 与目标桥接
bash scripts/orin/start_navigation.sh wheel false config/exploration_navigation.yaml true
```

另一个终端启动正式轮式控制器：

```bash
bash scripts/orin/start_controller.sh false
```

导航/探索/RViz 不自动启动正式底盘控制器。该控制器输出 `/Car/T5/Car_Cmd_Vel`。
点击 RViz 工具栏 `2D Goal Pose`，在自由区域按下鼠标确定位置，拖动确定最终朝向，松开发送。
桥接接收 `/Car/T4/rviz_goal`，调用配置中的导航 Action；再次点选会请求取消旧目标并等待其结束后发送新目标。
RViz 固定坐标系、地图/路径/探索话题、地图 QoS 根据同一主配置生成；目标必须位于配置的 map 坐标系。

已有导航进程时，可以单独启动界面和桥接：

```bash
bash scripts/orin/start_rviz.sh false config/exploration_navigation.yaml
# 只开桥接，RViz 在另一台已配置 DDS 的机器运行
bash scripts/orin/start_rviz_goal_bridge.sh false config/exploration_navigation.yaml
```

同一 ROS 域只启动一个目标桥接，勿同时用两种入口启动重复实例。
自动探索运行时先暂停探索、等待当前导航取消及车辆停稳，再使用鼠标目标：

```bash
bash scripts/orin/control_exploration.sh area-01 pause
```

正式 RViz 默认显示 **Local fine map**：以机器人为中心、默认 28 m 的细分辨率通行性图，细分辨率沿用 T3 输入（输入 0.2 m 时显示 0.2 m）。它已考虑平台尺寸/地形能力，是机器人中心是否可通行的图，不是原始高程或原始障碍像素图。白色可通行、黑色不可通行、灰色未知。

**Exploration map** 是默认 1 m 的全局粗探索图，默认关闭，可勾选作全局参考。粗指导路线、局部路径、定位、探索前沿和目标同时可显示。界面不加载模拟地形。

`navigation.publish_local_fine_map` 控制正式细图发布，`local_fine_map_topic` 与 `local_fine_map_window_m` 配置话题和显示窗口。没有订阅者时不生成显示栅格；地图变化或机器人移动至少一个细格距离后更新。显示图来自已有细图，不参与规划输入，不改变全局粗分辨率或搜索窗口。

鼠标坐标不吸附粗格，但细图 FREE 也不保证全局粗规划一定接受目标；细图自由、粗图占据导致的全局拒绝仍是独立算法问题。小场景可按任务需要调整 `common.coarse_resolution_m` 并重启，例如从 1.0 改为 0.2；这会提高全局搜索与探索计算量，属于算法配置调整，界面不会自动修改它。

## 简单控制闭环 Demo

```bash
# 默认手动选点，打开 RViz，独立 ROS_DOMAIN_ID=72
bash scripts/orin/start_controller_demo.sh
# 指定绕障场景，无 RViz
bash scripts/orin/start_controller_demo.sh detour false
# 指定多出口场景，打开 RViz
bash scripts/orin/start_controller_demo.sh multiple_exits true
```

场景：`manual`、`forward`、`reverse`、`final_yaw`、`detour`、`multiple_exits`。
脚本启动场景地图和车辆、正式导航和控制器、目标桥接。选择场景不会自动下发目标；可用 `2D Goal Pose`，或在第二个终端用记录器发送该场景的预设目标：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=72
ros2 run lunar_incremental_controller_demo run_case --case detour --timeout 240 --output /tmp/detour.json
```

实时模拟，速度上限 0.2 m/s；业务话题 `/lunar_demo/controller/*`。
每个预设测试重新启动该场景，以清零车辆与证据状态。

## 复杂地图探索 Demo

```bash
# 300 m 场景、0.2 m 观测/细图、1 m 粗图、30 倍目标仿真时钟、双 RViz
bash scripts/orin/start_exploration_demo.sh
# 无 RViz，10 倍目标倍率
bash scripts/orin/start_exploration_demo.sh false 10
# 较小任务区域，仍保留大场景
bash scripts/orin/start_exploration_demo.sh true 30 task_size_m:=40
# 手动选点模式，不自动启动探索
bash scripts/orin/start_exploration_demo.sh true 30 auto_start:=false
```

默认独立 ROS_DOMAIN_ID=73，业务话题 `/lunar_demo/integrated/*`。默认自动启动探索。
该脚本会启动模拟车辆、正式导航/探索/控制器、观测显示和目标桥接。
机器人速度始终按模拟时间限为 0.2 m/s，30 倍是目标仿真时钟倍率；目标机负载可能限制实际倍率。
默认细观测与局部地图窗口 28 m，导航搜索窗口 64 m；全局视图与局部视图分别显示粗图和细图。
可用额外参数 `start_local_rviz:=false` 只开全局窗口。

运行中调速（第二个终端）：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=73
ros2 param set /integrated_vehicle time_scale 10.0
```

范围 1～60。两个 RViz 窗口都可用 `2D Goal Pose`；自动探索开启时不要同时发送手动目标。
`auto_start:=false` 时手动选择已观测自由区域作为目标，探索器仍启动但不下发探索任务。

两个 demo 都只输出各自 `/lunar_demo/*/cmd_vel`。默认脚本覆盖继承的 ROS_DOMAIN_ID，避免落入正式运行域。
需要更换域时使用 `DEMO_ROS_DOMAIN_ID=74 bash scripts/orin/start_exploration_demo.sh`，其余观测终端使用相同域。
同一域只运行一套模拟器；在主启动终端 Ctrl+C 停止整套 demo。

## 验证边界

具体交付验证见 README。仿真、x86 Humble 容器结果不代表原生 ARM64 Orin、真实传感器 DDS 或实车验证。
