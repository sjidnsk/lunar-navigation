# OBJ / TCP 数字仿真验证记录

当前协议已按用户最新 C++ 参考更新；下文早期 C# 数据仅保留历史，现行协议与本次复测见文末“VehicleID 协议修订”。

日期：2026-09-11。工作树 `obj-tcp-simulation`，分支 `feat/obj-tcp-simulation`，开发基线 `5c23c13`。本次未提交、合并或更新部署包。完整操作见 [使用说明](OBJ_TCP数字仿真使用说明.md)。

## 环境与证据边界

本机 ROS 2 Jazzy；新包及其依赖共 10 个包完成独立构建。验证 overlay 位于 `/tmp/obj-tcp-validation/install`，默认启动缓存也已由 `scripts/simulation/build.sh` 构建通过（10 包，15.9 s），位于 `~/.cache/lunar_obj_tcp_sim/jazzy/install`，已验证可发现包和 launch 参数。日志、测试产物、地图均位于仓库外。

远端 Unreal IP 未提供，网络实测只连接 `127.0.0.1` 的测试服务端。该服务端独立解码/校验控制包，并以有加速度的简化车辆模型返回状态；它证明本机协议、ROS 算法与反馈闭环接通，不证明 Unreal 物理参数、地形碰撞或现场坐标对齐。

| 层级 | 状态 |
| --- | --- |
| C# 协议对应、配置、几何、ROS 接线与源码审查 | 已执行 |
| 本机 Jazzy 构建、单元/ROS 集成、TCP 回环闭环 | 已执行，详见下文 |
| 实际 4.4 GB OBJ 的 1 km 转换与局部观测 | 已执行 |
| 远端 Unreal TCP / 物理闭环 / 坐标对齐 | NOT_RUN |
| 1 km 完整探索与长时间资源稳定性 | NOT_RUN |
| RViz 真实窗口显示 / 目标话题 | 双窗口实际渲染已检查，目标话题闭环通过；鼠标人工点选未单独记录 |
| Humble、原生 Orin、DDS 跨机、实车 | NOT_RUN |

## 实际 OBJ 与观测性能

源文件 `/home/kai/WS/lunar-navigation/ue_land/5555/OpenScene.OBJ`：4,399,375,768 字节、19,063,287 顶点、36,807,764 三角面。以源模型车辆附近 `(1214,-1110) m` 裁剪 1 km 正方形，外加 12 m 观测外圈。采用明确标为假设的 `x,-z,y` / 0.01 变换，未把假设写成实测对齐。

结果：5120 × 5120、0.2 m 高程，417,442 个保留三角形；26,214,400 格有效高程。转换耗时约 168.14 s，目录约 318 MiB。通过显式列表排除了该源文件的 269 个车辆组。输出已复制到 `/home/kai/WS/lunar-navigation/simulation-maps/open-scene-1km`。

观测默认 12 m、120°、近场 1.4 m、窗口 28 m。原始岩石附近遮挡查询曾达 1.06–2.34 s；加入保守角域预筛后，同一组查询可见格数和高度保持一致，三处岩石查询为 24.37 / 37.50 / 39.55 ms，平缓区示例 5.23 ms。这里是几何函数测量，不是整个 ROS 地图/规划周期耗时，也不是最坏情况保证。状态 `query_ms` 含定时器完成轮询等待，不能与纯几何耗时混用。

对应证据：`/tmp/obj-tcp-validation/{prepare.log,observation_benchmark.json,rock_observation_benchmark.json,rock_observation_optimized.json}`。单元测试覆盖障碍表面、背后遮挡、缺面、跨块、过滤以及三角面限定。

## 回归与闭环

新包常规测试：63 项通过，覆盖精确字节、拆/粘包、超时零命令、部分发送、退出串行写、参数/脚本、OBJ 几何、GridMap 布局、旧位姿不刷新、迟到 Action 接受取消、显示订阅和探索启动就绪。TCP、几何、ROS/launch 分别审查并修正，最终复审无剩余阻塞问题。

重跑常规测试（先 source 同发行版与构建 overlay）：

```bash
PYTHONPATH="$PWD/ros2_ws/src/lunar_obj_tcp_sim:$PYTHONPATH" ROS_DOMAIN_ID=175 \
  python3 -m pytest -q ros2_ws/src/lunar_obj_tcp_sim/test \
  --ignore=ros2_ws/src/lunar_obj_tcp_sim/test/test_closed_loop.py
```

闭环验收订阅正式诊断、非空 ACTIVE PathReference、TrackingStatus；导航需 Action `GOAL_REACHED`，同时检查控制封包速度与服务端返回速度。退出需任务终态、新鲜反馈停稳及最后完整 TCP 包为零。测试服务端不会截断车速来掩盖控制器超速。

```bash
ROS_DOMAIN_ID=174 LUNAR_RUN_TCP_E2E=1 \
LUNAR_REAL_OBJ_MAP=/home/kai/WS/lunar-navigation/simulation-maps/open-scene-1km \
  python3 -m pytest -q ros2_ws/src/lunar_obj_tcp_sim/test/test_closed_loop.py
```

最终四项闭环测试全部通过，共 127.55 s。全程订阅观测、HUD、场景和地形预览显示消息，覆盖显示开启后的发布路径。

| 场景 | 实测结果 |
| --- | --- |
| 已知小地图导航 | 3 个目标均 GOAL_REACHED，含目标朝向与返回目标 |
| 8 m 小区域探索 | 正式 COMPLETED / COVERAGE_TARGET_REACHED，粗图覆盖 1.0，观测覆盖超过 0.95 |
| 实际 1 km OBJ 的局部导航 | 初始位置附近采样，1 m 目标 GOAL_REACHED；不是 1 km 全量探索 |
| 已知地图岩石绕障 | 对侧目标 GOAL_REACHED；反馈轨迹绕开障碍，检查障碍外 0.4 m 的矩形净空 |

四项均确认任务终态、新鲜零速反馈、最后 TCP 控制包为零，未出现退出未确认告警。最大指令和测试服务端反馈线速度均约 0.2000000053 m/s（float32 表示误差，测试容差 1e-6）。本轮导航路径不强制选择倒车；负向轮速的协议/运动学由单元测试覆盖，不声称四项中每项都实际倒车。

证据目录 `/tmp/obj-tcp-validation/e2e-reviewed/`，各场景保存 `receipt.json` 和 `launch.log`。正式导航 ROS 包另行串行回归：13 个 CTest 全部通过，colcon 汇总 93 tests / 0 errors / 0 failures / 0 skipped；其中包含路径分段、停稳反馈与 RViz 目标桥接测试。没有把独立 CTest 结果冒充本次 TCP 持续分段实测。

实际远端物理绕障、持续分段和大区域多目标探索仍需现场验证；不以本机积分轨迹替代。

## 发现并修复的问题

| 问题 | 修正及证据 |
| --- | --- |
| 正式调试发布器限定两个 demo 前缀，合法 `/lunar_sim/planning` 启动失败 | 改为 ROS 名称合法性校验；调试发布器回归测试 |
| TCP 退出与工作线程并发写、部分发送可能截断封包 | 工作线程独占 socket 写，退出串行零包，按已发送偏移完成部分写；专门回归 |
| 默认 OBJ 组绕过过滤、非三角形简单扇形化可能产生错误表面 | 默认组应用过滤，多边形明确要求先三角化 |
| 环视超时/停止后迟到的目标接受可能继续执行 | 保存异步取消意图，接受后立即取消 |
| operator 自定义 Context 使用全局 executor、向整个进程组发 SIGINT 造成二次中断 | 专属 executor、空 ROS 参数、正常只中断 launch；超时才进程组回收 |
| 未开始探索时 CANCEL 结果 task_id 被清空，退出误报未确认 | 按正式 `IDLE/CANCELED` 状态确认，不接受初始 `IDLE/IDLE` |
| 显示开启时局部变量覆盖 marker 函数 | 增加真实 MarkerArray 订阅回归，修正名称；闭环也订阅显示主题 |
| 节点自身任务订阅被误认为探索器已经就绪 | 等待实际探索状态及额外任务订阅后才发布 START |

早期失败日志保留在 `/tmp/obj-tcp-validation/e2e-initial`、`e2e-ordered`、`e2e-final`。修复前失败不得与最终结果合并为通过记录。当前不改变正式粗图拒绝目标、不可达或等待地图的算法语义。

## 本机交互入口补充

新增 `run_local_rviz_demo.sh`、`local_demo.py`、`local_vehicle.py`，不依赖测试目录里的服务端。只允许回环监听，轮径/轮距读取有效平台配置，运动记录使用有上限的队列。原有独立测试服务端保留，避免协议测试仅依赖同一套模拟实现。

新增本地服务端回归覆盖拆包、配置车轮几何、地形 Z 跟随、命令中断后的实际反馈归零和无效包拒绝。交互入口集成测试通过 RViz 使用的 `goal_pose` 话题发目标，并读取正式 Action 结果；测试环境变量 `LUNAR_LOCAL_RVIZ=1` 会同时开启真实 RViz 窗口。


补充实测结果：常规测试 65 passed（3.56 s）；新本地入口 + 双 RViz + goal_pose 目标桥 + Action GOAL_REACHED + 有序停止集成测试 1 passed（9.15 s）。Jazzy 新包重建通过。证据 `/tmp/obj-tcp-local-rviz-validation/test_local_script_rviz_goal_an0/local-demo.log`。

随后启动实际 1 km OBJ 默认探索入口，RViz 两个窗口 Global Status: Ok；截屏人工检查地图、视野和路径显示。状态采样：292 次观测、14,721 个观测格 / 25,000,000 个任务格，正式探索 `NAVIGATION_GOAL_SUBMITTED`，控制 `TRACKING`，命令 0.2 m/s 且反馈新鲜。证据 `live-status.txt`、`rviz-screen.png` 位于上述验证目录；运行日志 `/tmp/obj-tcp-local-rviz-live.log`。这是短时间运行证据，不是完整大区域探索或 UE 物理证明。


探索范围调整：地图仍为原 1 km 输出，新增 exploration_size_m=100，任务固定在地图中心。任务 START、RViz 边界和观测统计同步采用 `[1164,-1160,1264,-1060]`；实时状态证实 task_cells=250000（500×500），而地图仍为 5120×5120（含观测外圈）。67 项常规测试通过，Jazzy 包构建通过。演示重启时反馈新鲜，初始环视正常进行。历史 1 km 任务的覆盖数据保留其原始含义。

缩小任务后的 TCP 自动探索闭环回归：1 passed（28.98 s），3 个无关场景未重复运行；结果目录 `/tmp/obj-tcp-task100-regression`。

任务边长随后按用户要求更新为 300 m，仍使用相同 1 km 地图。默认 YAML、配置对象与节点默认值同步为 300；7 项相关测试通过，Jazzy 构建通过。重启实时状态确认边界 `[1064,-1260,1364,-960]`、task_cells=2250000，初始环视运行且反馈新鲜。日志 `/tmp/obj-tcp-local-rviz-300m.log`。之前 100 m 记录作为历史保留。

任务范围可视化改进：粗紫红色边界、边长文字，全局视图按任务边界生成临时 RViz 配置，固定 map 中心并随任务大小缩放；退出删除临时配置。67 项常规测试通过，包重建通过。实际双窗口截屏检查完整 300 m 正方形、标签和局部视图；图像 `/tmp/obj-task-global-fixed.png`。修改仅作用于显示，不改变任务边界或规划算法。


## VehicleID 协议修订

依据本次 COMStructType.h、ControlComponent.cpp、TCPComponent.cpp、main.py，用户确认 VehicleID=0。控制帧为 16+4+37=57 字节，反馈为 16+4+88=108 字节。控制 Command=3 / PayloadSize=41，反馈 Command=4 / PayloadSize=92，外头 XOR 校验。内层控制校验不包含 VehicleID。位置以发送代码已乘0.01后的米值直接使用，覆盖头文件过时的 cm 注释。未修改用户参考文件。

协议、桥接、离线自动中心工具、本地交互模拟端、独立回环测试端均同步。vehicle_id 在配置中可修改，非目标车反馈不刷新本车位姿或新鲜度。新增 ID 区分、外头损坏恢复测试；69 项常规测试通过（3.64 s），5 项 TCP/ROS 闭环通过（140.23 s）：导航、自动探索、实际 OBJ 局部导航、绕障、一键入口目标桥与有序停止。构建通过，证据 `/tmp/obj-tcp-vehicle-id-e2e`。远端 UE 仍为 NOT_RUN。

参考 UE 接收实现使用 Nodivision 且 OnReceive 只处理一个完整包，缺少拆包剩余缓存/多包循环；这是源码发现，不冒充已复现的远端故障。未将本机处理拆粘包的能力描述成 UE 服务端已经修复。


RViz 空 frame 联调诊断：现场只读订阅确认 /lunar_sim/planning/fine_state 连续发布 width=height=0、frame_id为空的地图。同期 UE 位置约 (3163.626,-46.019,3.018)m，超出当前 1 km 裁剪区域 [714,-1610,1714,-610]；导航在分块加载地图。根因是 ProjectFineVisualization 的窗口无交集分支返回默认消息，发布器又赋新时间戳。已改为有效窗口无交集时发布带 snapshot frame 的全 UNKNOWN 显示栅格，不生成通行证据；同时补全 LocalGoalRegion DELETEALL marker 的 header。运行中的远端联调进程未重启，也未覆盖其正在加载的默认安装库。修复在 /tmp/obj-tcp-validation 独立构建/测试。地图重裁剪或坐标对齐仍需处理；显示修复不等于位置问题消失。


传感器/箭头错位现场只读诊断：当前远端 nav 模式在 LOADING_KNOWN_MAP，未运行按视场 TerrainMap.observe，不能以观测覆盖统计 0 判断传感器失效。位姿 (3163.626,-46.019,3.017574)m，四元数 (-0.123056,-0.644315,0.712255,0.249817)，归一长度约1；按当前约定换算 RPY≈(-81.897,-8.431,148.664)deg。旋转后的 [0,0,1.5] 安装偏移为 (-0.745823,-1.284526,0.209147)m，水平偏移1.48535m。传感器Z=3.22672m，位置处OBJ地面Z=2.67969m；相同位姿离线虚拟观测返回3873个有限高程格。原始采样 `/tmp/obj-sensor-alignment-state.json`。尚需确认UE实体是否侧翻；如车辆直立，应核查VehicleActor坐标轴/安装变换，不以任意抬高Z或清零roll/pitch掩盖。


## 2026-09-11：世界/车体坐标转换修正

用户确认世界左手系、Z 上；车体左手系 X 前、Y 下、Z 右；两种反馈速度均为车体系。新增 coordinates.py，桥接统一转换位姿、TF 和 twist；协议解析仍保留原始字段。测试模拟车按逆变换发送，独立 TCP E2E 对端使用独立公式，避免继续用 ROS 姿态冒充 Unreal 数据。角速度按轴向量变换，Vortex 实际旋转符号仍待远端验证。

- 新增 3 项坐标测试：直立安装、100 个姿态的独立矩阵对照与往返、记录姿态的传感器高度；常规测试共 72 passed、5 个显式 E2E skipped。
- 隔离 Jazzy 构建 `/tmp/obj-tcp-validation`：10 包成功；没有更新正在使用的默认缓存 overlay，也没有重启远端联调进程。
- 4 项本机 TCP/ROS 闭环 114.56 s 通过：nav、explore、obstacle、本地启动脚本与 RViz 目标话题。日志 `/tmp/obj-tcp-coordinates-e2e.log`。
- 新 OBJ 地图 real 闭环 8.73 s 通过：日志 `/tmp/obj-tcp-coordinates-real.log`。闭环测试包含正式导航结果、路径参考与停机断言。
- 源 OBJ 重建 `/home/kai/WS/lunar-navigation/simulation-maps/unreal-start-1km-ros`，346110 个三角形、26224641 个有效细格，141.75 s。轴映射 x,z,y，中心 (3163.6262,46.0188)，物理范围 1 km，任务范围仍 300 m。
- 记录姿态转换后 RPY 约 (-8.10,-8.43,-148.66) 度；车体上方 1.5 m 的安装点在世界系偏移 (0.29594,-0.06735,1.46897) m。新地图离线虚拟扫描 3873 个有效细格，脚下地形高程 2.679688 m。证据 `/tmp/obj-tcp-coordinates-observation.json`。

上述为源码、本机 Jazzy 与本机 TCP 模拟证据。新版本远端 Unreal 配准、实际角速度符号、Humble/Orin：NOT_RUN。地图元数据继续保留 NOT_EXTERNALLY_ALIGNED，不以仿真闭环替代现场配准。


## 2026-09-11：Unreal 原地转向实测、控制校验根因与 Actor 轴更正

本节覆盖上一节车体“前 X、下 Y、右 Z”假设。用户最终确认 VehicleActor 为后 X、下 Y、左 Z；四元数对应该 Actor 局部轴，不存在单独的旋转轴定义。

1. 正常停止旧 operator PID 117912 及其 launch，确认 TCP 连接释放。第一轮 13 秒序列连续收到反馈，但四元数和位置不变，轮速约 1e-9；用户也确认车辆没动。保留 `/tmp/obj-tcp-turn-calibration/rejected-checksum-raw.json`，不能记为角速度校准通过。
2. 源码追溯发现接收路径 `TCPComponent.cpp:437` 不包含 Head 的 checksum，与本机误用的 Head-inclusive 规则相差 0x92。两个回归用例先失败，修复后协议、传输及本机车共 23 项通过。原本本机接收器使用了同一错误假设，现已一并纠正。
3. 重复同一 ±0.1 rad/s 各 3 秒指令，收到约 20 Hz 的真实远端反馈。正向实际 +8.1918°，反向 -9.1908°；AngularVelocity.Y 稳态均值约 +2.1767/-2.2110；轮速均值约 ±0.20653 rad/s。每段有 0.076/0.091 m 平移，属于实际物理反馈，未当成纯运动学无漂移证明。停机后轮速近零，最终零速发送并关闭测试连接。
4. 角速度原始 Y 的轴和符号通过实测；单位仍不闭合。两段姿态增量/原始 Y 积分分别约 0.020925、0.020936。未硬编码经验比例；反馈头文件 rad/s 注释不能作为已验证依据。临时从相邻姿态、本机接收时间估计 ROS angular；原始速度保留在测试记录。
5. 已确认 Actor 轴的矩阵测试、100 个随机姿态往返及记录姿态前向/上向检查通过。测试后的 ROS yaw 约 30.30°，传感器世界偏移 `(0.28899,-0.070997,1.47018)` m，新地图离线扫描 3875 个有效细格。见 `/tmp/obj-tcp-turn-calibration/corrected-pose.json`。

证据：`/tmp/obj-tcp-turn-calibration/raw.json`、`summary.json`；可重复工具 `scripts/simulation/test_turn_feedback.sh`。转向动作和 TCP 协议修复有 Unreal 实测证据；三轴速度完整标定、原始角速度单位、Humble/Orin 仍 NOT_RUN/待确认，不能用本机差分替代声称通过。


最终复核：默认 Jazzy 构建缓存 10 包构建成功（`/tmp/obj-tcp-actor-coordinate-build.log`）。常规测试 75 passed、5 个显式闭环用例 skipped；另行执行这 5 项 TCP/ROS 闭环全部通过，144.71 s（`/tmp/obj-tcp-final-actor-e2e.log`）。这些闭环使用本机模型，不是另一次 Unreal 导航到达验证。

已重新启动远端 nav + 双 RViz，ROS 域 74，新 `unreal-start-1km-ros` 地图，未下发目标。只读抓取 4 秒得到 82 条 Odometry，frame=odom、child=base_link，静止速度接近零；ROS 位姿与转换后的原始反馈一致（`live-ros-odometry.json`）。截图 `/tmp/obj-tcp-turn-calibration/rviz-current.png` 可见车体与绿色视场；nav 仍在分块加载整张已知地图，因此当前位置细图仍可能显示 UNKNOWN，不能误认为虚拟扫描未工作。启动时 RViz 记录了 GLSL sampler 警告；截图 Global Status OK，未把此警告当作已经修复。当前任务只确认坐标/TCP 转向，不声明全部地图渲染或原始角速度单位已验证。


## 2026-09-11：500 m 裁剪与已知地图加载优化

- 原始输入为已处理的 `unreal-start-1km-ros`。新 `unreal-start-500m-ros` 从原数组裁剪，不重采样；500 m 正文范围加 12 m 外圈，2621×2621 细格、0.2 m、91162 三角形，裁剪 0.674 s。保留旧目录，探索任务区仍 300 m。
- 旧 nav 默认为 128×128，按地图角落逐行加载；1 km 约 1681 个块。新默认 512×512，按首个有效车辆位置对齐首块、由近至远无重叠分区；仍等待正式地图接收及派生完成。nav 不再生成逐格黄色“观测点”，explore 保留实际观测显示。
- 实测先发现只增大块仍然慢。检查旧 CMake flags 为 `-std=gnu++20 -fPIC`，没有编译优化。仿真 build.sh 改为默认 Release（`-O3 -DNDEBUG`），可用 LUNAR_CMAKE_BUILD_TYPE 覆盖。正式地图语义与通行性算法未改。
- 未优化对照 `/tmp/obj-tcp-500m-loading-unoptimized.json`：首个车辆细格可用 5.964 s，174.774 s 时发送到第 7 块，未完成整图。该记录不是完整加载耗时，不据此捏造精确整图倍数；构建负载和交互导航也可能影响现场计时。
- Release 实测 `/tmp/obj-tcp-500m-loading.json`：从启动含双 RViz，到车辆所在细格 FREE 为 2.249 s；时间探针同时检查坐标落在细图内以及该格非 UNKNOWN，非单纯“收到一张地图”。全图以 KNOWN_MAP_READY 及正式接收/派生完成计。
- 新增无重叠全覆盖、首块包含车辆周边、边界/地图外聚焦、裁剪高程与采样点不变、遮挡查询等测试；首轮常规 80 passed、5 个显式闭环 skipped。独立构建 `/tmp/obj-tcp-loading-validation` 及最终默认 Release 构建均为 10 包成功；Release 日志 `/tmp/obj-tcp-loading-release-build.log`。

1 km 使用同一加载优化，但本轮完整加载验收以用户选择的 500 m 为准；不推断 1 km 的整图秒数。Humble/Orin 性能 NOT_RUN。


Release 500 m 实测最终结果：首个车辆细格 FREE **2.249 s**，全图 `KNOWN_MAP_READY` **167.243 s**（约 2 分 47 秒，包含启动双 RViz），49 块全部经正式地图接收和派生完成。测试期间有交互导航 TRACKING，故该结果是实际联调耗时，不是空载 CPU 微基准。日志 `/tmp/obj-tcp-500m-rviz.log`，时序 `/tmp/obj-tcp-500m-loading.json`。当前启动保留，使用 500 m 地图；未额外下发导航目标。

Release 回归（按包串行）：incremental_navigation_core 249 项、incremental_navigation_ros 94 项均通过；obj_tcp_sim 80 项通过、5 个显式 E2E 跳过，合计 423 项通过，0 errors/0 failures。日志 `/tmp/obj-tcp-loading-release-tests.log`。本轮没有重跑完整 E2E；500 m 远端实时加载与交互 TRACKING 已单独记录，不混作 Action 到达成功证据。


## 2026-09-11：初始盲区诊断与传感器安装定义更正（待安装偏移数值）

实测 explore 在起始转向前返回 START_BLIND_ZONE_UNRESOLVED。只读位姿 `(3152.892334,34.869129,5.359138)`；按当时 `[0,0,1.5]` ROS 安装偏移计算，传感器横移约 0.44 m，车体中心 fine=UNKNOWN，车体 1.4 m 范围内有 22 个未观测格。隔离域 185、无控制器及 TCP 的重放：near=1.4 复现失败；near=2.0 得到 cycle_result=PLAN_FOUND、ACTIVE PathReference、start_patch_used=false。证据 `/tmp/obj-blind-zone-state.json`、`/tmp/obj-blind-zone-replay-results.json`。没有执行该转向路径。

随后用户明确：传感器应装在 Actor -X（车头）方向，而不是之前假设的 ROS +Z 上方。因而 **2.0 m 近场修改已撤回，保持原 1.4 m**；以上对照仅说明旧安装假设下的盲区来源，不是对正确安装后的修复验收。需要 Actor 安装偏移 (x,y,z) 的实际米数，按 ROS `(-x,z,-y)` 转换；光轴朝向 Actor -X 对应 ROS +X。当前运行栈未重启，四轮独立转向未完成前没有再发运动指令。

### 车头上方 1 m 安装配置

用户指定车头上方 1 m。默认 ROS 安装偏移改为 `[0.591, 0, 1.0]`，Actor 对应 `[-0.591, -1.0, 0]`；前缘取项目 footprint，Actor 原点与车体中心一致仍属虚拟安装假设。同步 YAML、节点和几何默认值及使用说明，保留近场 1.4 m。Jazzy 受影响包构建通过。首次 pytest 未加载项目 overlay，7 项因消息包缺失失败；加载后一次 TCP 时序测试未捕获 80 ms 内的非零指令，79 通过、1 失败、5 跳过，保留为间歇时序问题。未向 Unreal 发送运动指令，未重启现场探索，启动盲区修复效果 NOT_RUN。

同环境完整复跑结果：80 passed、5 skipped（3.69 s）；上述 TCP 时序波动未修改或隐藏。

### 四轮独立转向/驱动替换差速分配（2026-09-11）

- 根因：C++ 接收八个控制浮点数并分别设置轮速和转角；本机旧适配器只计算左右轮差速，transport 不接收转角参数。本次移除旧差速函数，新增独立 `kinematics.py`；协议编码保持 57 字节和原校验规则。
- 轮径 0.319 m、轮距 0.67 m、轴距 0.8175 m 来自 wheel.yaml。四轮速度向量使用刚体平面运动学；反向滚动使转角保持 ±90°，统一轮速缩放保留曲率。分别配置 Wheels/Turns 数组映射、符号、转角零点、转角限制与到位误差。
- 桥接在转角反馈到位后释放驱动，停车、超时和退出保留最后目标转角并发送零轮速。本地车辆及独立 E2E 接收器能接收非零转角、反馈有限速度转向；独立接收器用四轮速度向量反推车体运动。转向测试工具也使用新模型及 --config。
- 红灯：旧转向结果没有 turn_angles，新增刚体轮速度验证失败。实现后本机 Jazzy Python 测试 86 passed、5 skipped（6.88 s），包含正反行驶、两向原地转、弯道、轮序/符号/零点、超限曲率、转角到位释放驱动、TCP 转角与停车保持、本地真实 socket 闭环。
- Jazzy 受影响包构建通过。正式 ROS + TCP 独立接收器 E2E：导航（前进、终点转向、返回另一目标）和绕障两场景通过，4 次 GOAL_REACHED，存在 ACTIVE PathReference 和 cycle_result=PLAN_FOUND。最高车体速度约 0.200000009 m/s（线协议 float32 误差）；此次正式导航最小线速度为 0，倒车仅由运动学测试验证，不能写成正式倒车闭环已验证。
- 探索 E2E 未通过：当前车头安装偏移和近场 1.4 m 的默认观测在初始扫描前复现 START_BLIND_ZONE_UNRESOLVED，BOOTSTRAP_FAILED，命令 v=ω=0、探索 IDLE。未为通过控制器测试而改动观测范围、绕过未知格或改写成功状态。完整结果为 2 passed、1 failed、1 deselected（176.30 s）。
- 证据：/tmp/obj-four-wheel-e2e-20260911.log；/tmp/obj-four-wheel-e2e-20260911/ 下导航和绕障 receipt.json、探索失败 launch.log。该目录为本机产物，不提交。
- 本次没有发送远端运动指令或重启用户运行栈。Unreal 的轮序、关节符号/零点/转角能力尚未由用户或场景数据确认；当前默认仅为本机约定。Unreal 四轮物理联调、Humble/Orin 均 NOT_RUN。

最终 Jazzy 包测试：colcon test lunar_obj_tcp_sim，86 passed、5 skipped（6.97 s）。用户入口 run_local_rviz_demo.sh + RViz goal_pose 话题桥接 + 有序停车验证：1 passed（7.11 s，未打开 GUI）；证据 /tmp/obj-four-wheel-script-20260911.log。包构建、Python 编译检查和 git diff --check 通过。

### Unreal 转向通道和正负方向实测核对（2026-09-11）

用户授权立即测试。先向旧 operator PID 147576 发 SIGINT，由原流程停车退出；确认 TCP 连接释放、反馈轮速接近零，再独占连接 192.168.10.23:6668，VehicleID=0。

1. 轮速全零，依次对 TurnAngle[0..3] 发送 +0.2、0、−0.2、0 rad。四通道末段反馈分别为 ±0.199999988 rad，其他通道接近零。用户根据 Unreal 画面确认 0 左前、1 右前、2 左后、3 右后，均先向右再向左。因此 steering_order=[0,1,2,3]，steering_signs=[−1,−1,−1,−1]。配置、ROS 参数默认、本地模拟器入口与转向测试工具同步更新；独立 E2E 接收模型采用相同线协议方向。
2. 转角回零，四轮共同 +0.15/−0.15 rad/s，各 2 秒，中间停车。车体前向位移分别 +0.05614/−0.05522 m，原始 LinearVelocity.X 均值 +0.02363/−0.02318 m/s，四轮反馈达到 ±0.15 rad/s。确认共同正轮速前进、负轮速后退。WheelSpeed 数组的前后槽没有逐轮目视辨认；当前 v/ω 分配同侧前后速度相同，不把该试验写成八个关节的完整场景绑定验证。
3. 采用修正后的四轮运动学发送 +0.1/−0.1 rad/s 车体角速度，各 3 秒。姿态反馈换算 yaw 分别 +17.9856°/−21.3215°，即左转/右转正确；结束轮速绝对值小于 1e−8 rad/s。最后保持原地转向轮向（接口角约 [+0.8842,−0.8842,−0.8842,+0.8842] rad），未自动回正或恢复导航。正负角度幅值不对称、原始角速度单位/尺度仍未完整标定。

证据目录：/tmp/ue-steering-channels-20260911-1789118485008647301/feedback.json；/tmp/ue-drive-sign-20260911-1789118581901092517/feedback.json；/tmp/ue-four-wheel-turn-confirm-20260911/turn-feedback.json。三组远端测试 errors 均为空。测试脚本在 /tmp，不提交实验原始数据。

回归中再次复现已有 TCP 测试时序缺陷：连接前创建的 80 ms 指令可能在首次发送前到期。仅修正测试同步，让接收端先确认非零指令，再停止刷新并验证 80 ms 超时零速及转角保持；未放宽生产超时或修改传输逻辑。转向测试的成功判断同步要求正指令 yaw>0、负指令 yaw<0，防止仅观察到转动就误判方向正确。

方向修正后的最终验证：Jazzy 包构建通过；Python 86 passed、5 skipped（6.88 s）；本地脚本/goal_pose 桥接/有序停止 1 passed（6.98 s），日志 /tmp/ue-calibrated-local-script-20260911.log；git diff --check 通过。测试后没有残留到 6668 的本机 TCP 连接，未恢复探索启动。

### 当前视场错位与启动盲区分析（只读现场、隔离回放）

采集 /tmp/ue-alignment-analysis.json：ROS 车体原点 (3152.9314,34.8841,5.3479)，sensor_origin 与 FOV 顶点均为 (3153.2634,35.4333,6.3161)。二者遵循 p+R*[0.591,0,1]，不存在本次数据中的显示/观测安装公式不一致。姿态 RPY 为 (16.995°,−1.170°,85.940°)，在水平航向坐标下传感器投影位于前方 0.571 m、右侧 0.292 m。箭头长度固定 1 m，不代表车头轮廓前缘；通行图 z=0，箭头和传感器保留反馈高度，显示语义混用易误读。

OBJ 在车体和传感器 XY 处均为 2.679688 m，Actor 原点高于 OBJ 约 2.668 m，虚拟传感器高于 OBJ 约 3.636 m。不能由这些数值直接断言高程偏移：Actor 原点实际位置、OBJ 与 UE 地形对应点以及静止横滚都需交叉核对，不能自动将真实反馈 Z 替换为地形高程。

车体周围 1.4 m 的诊断圈内共 151 个 sample_xy，其中 42 个被水平 FOV/近场筛选排除，圈内通过候选筛选的点无遮挡；该 1.4 m 为诊断范围，不是声明规划器固定认证半径。正式细图起点为 UNKNOWN。当前观测为水平 ±60° 扇区并集传感器周围 1.4 m 圆，再做三角网格遮挡，不包含垂直 FOV。BOOTSTRAP_FAILED 使 tick 提前返回，后续采样也停止，导致恢复链路中断。

隔离域 189，仅正式导航 + 当前冻结位姿和 OBJ 观测，无 TCP/执行器：near_field=1.4 复现 START_BLIND_ZONE_UNRESOLVED、无 ACTIVE reference；仅将近场改为 2.1 后 cycle_result=PLAN_FOUND、ACTIVE PathReference、start_patch_used=false。证明该位置观测支持不足是直接原因，不证明 2.1 m 对所有姿态均足够，也不证明传感器外参/高程已标定。未修改现场近场配置或恢复运动。证据 /tmp/ue-current-start-replay-results.json、/tmp/ue-current-start-replay.log；诊断图 /tmp/ue-current-observation-diagnosis.png。

建议：二维显示统一投影平面并标明车体原点/物理轮廓/安装点，三维实际安装点分开展示；先核实原点、姿态与高程对应关系；为现有虚拟模型明确近场补盲能力并按需要补充真实射线观测，不直接写 FREE；启动失败仍继续观测，仅在相关地图证据或位姿变化后重试。真实前向相机模型则需另行定义俯仰/垂直 FOV 与可实现的初始扫描能力，不能将扩大近场伪装成单相机视场。

### 近场补足与扫描失败持续观测实现（2026-09-11）

按用户指定两项修改：YAML、节点及几何默认 near_field_radius_m 由 1.4 改为 2.1 m，保留射线遮挡；BOOTSTRAP_FAILED 从 tick 的观测禁用条件移除，改为仅在 advance_task 阻止继续任务。没有实现证据触发自动重试，不跳过失败扫描或自动发布探索任务。

红灯验证：真实 ROS 传感器在 BOOTSTRAP_FAILED 收到新 Odometry 时没有发布可见高程。修复后同一测试通过，同时确认旧时间戳不会重复采样、失败状态和任务暂停保持不变。Jazzy 包构建通过，普通测试 87 passed、6 skipped（8.20 s）。两个完整 ROS/TCP 测试通过（40.85 s）：默认近场下完成初始扫描和探索，coarse coverage=1.0、COVERAGE_TARGET_REACHED；故意 near=1.4 的故障案例失败后 samples 从 4 增至 9，始终零运动且无 ACTIVE reference。证据 /tmp/ue-nearfield-recovery-e2e-20260911.log 及同名目录下 receipt.json。

有序退出旧 operator 159068，启动新 operator 161920（domain 74、500 m 地图、300 m 任务、双 RViz、已标定四轮方向）。现场不再报 START_BLIND_ZONE_UNRESOLVED，实际执行扫描并增加到约 12912 个观测格；随后扫描转向停留在 ALIGNING，约 1789119827 时报告 Initial scan timeout，尚未进入自动探索。这是新的控制/扫描完成问题，未提高超时或伪报探索启动。

现场失败后继续观测也已验证：5 秒内 samples 从 434 增至 457，phase 保持 BOOTSTRAP_FAILED，command_v=command_w=0，末次反馈 vx≈1.08e−8 m/s、wz=0。日志 /tmp/ue-nearfield-exploration-1789119780852094245.log；状态 /tmp/ue-nearfield-live-status.json 和 /tmp/ue-nearfield-live-failed-continuation.json。栈和 RViz 保留，车辆已停止。Humble/Orin 未运行。本次没有修改姿态转换、安装偏移、地图高度或控制器。

### 低速分段扫描、滑移重新定点及转向减速修复（2026-09-11）

按用户要求，初始扫描角速度命令上限 0.15 rad/s、角加速度命令上限 0.1 rad/s²，每段 30°。每段使用实时位置申请正式导航；滑移超过 0.1 m 后取消旧目标，等待 Action 终态和连续 0.3 s 停稳，再从当前细图可认证的新位置继续同一朝向，不返回最初候选位置。重新定点不重置单段 45 s 超时。扫描完成恢复普通控制策略，失败仍继续观测。扫描适配器复用正式控制节点及唯一速度发布者。

首次远端低速试验仍失败：/tmp/ue-scan-reanchor-live-1789120913765354808.log，重新定点 2 次，第一段 INITIAL_SCAN_TIMEOUT。复现正式执行器以反馈角速度为加减速起点的缺陷：反馈 0.18 rad/s、输出上限 0.15 rad/s 时，即便朝向误差为零，输出仍持续 0.15。改为以最近发出的指令进行转向加减速，真实反馈仍决定停稳；节点额外发布零速时同步更新指令记录。新增响应增益 1.2 的回归证明能完成转向并停稳，未拟合 Unreal 比例、未放宽位置/朝向容差。

本机 Jazzy 控制器和仿真模块测试 243 passed、7 skipped（9.77 s），受影响两包构建通过。用户脚本、RViz goal_pose 桥接和有序停止 E2E 1 passed（7.23 s）。此前注入世界 X 方向 0.04 m/s 转向滑移的完整 TCP/ROS 回归通过，重新定点 23 次，完成扫描及测试任务；证据 /tmp/ue-scan-slip-final-20260911/ 下 receipt.json。退出过程修复为保留 ROS context 到节点完成停车和线程退出；E2E 检查没有 Traceback 或 process has died。

重新启动 Unreal operator 167427（domain 74，192.168.10.23:6668，VehicleID 0，500 m 地图/300 m 任务，双 RViz）。约 58 s 内完成全部 12 段，EXPLORATION_TASK_STARTED、NAVIGATION_EXECUTING，随后恢复普通控制参数。已观测细格 12379，本次没有触发超过 0.1 m 的扫描重新定点。扫描最大命令 0.15 rad/s；位姿差分角速度样本峰值约 0.919 rad/s，不能将命令限幅写成仿真物理速度/加速度已标定。现场日志仍有既有 RViz GLSL sampler 报错，不影响本次 Action 完成证据。证据 /tmp/ue-scan-command-ramp-1789121316479974887.log、/tmp/ue-scan-command-ramp-status.json。完整 300 m 探索未验证，Humble/Orin NOT_RUN。保留自动探索运行，未提交或合并。

转向指令减速修复后的最终滑移 E2E：1 passed、5 deselected（78.53 s），重新定点 12 次，实际模拟位置 X 从 0 漂移到 2.4187 m，仍完成全部扫描并达到测试任务 COVERAGE_TARGET_REACHED，未返回原起点。证据 /tmp/ue-scan-ramp-slip-final-20260911.log 与同名目录 receipt.json；属于本地注入滑移测试，不等同于 Unreal 物理滑移标定。最终 git diff --check、文档 UTF-8 读取通过。

### 前向速度来源修复（2026-09-11）

现场只读诊断：前进命令约 +0.025 m/s，位姿沿车头方向移动约 +0.0298 m/s，但桥接采用的原始 LinearVelocity.X 为 −0.013 m/s。执行器每周期将负的前进反馈截为零，再增加一步约 0.025 m/s，造成无法累积加速。证据 /tmp/ue-current-speed-probe.json；该原始轴不能继续视为已标定的 ROS 前进速度。

桥接 linear.xyz 统一采用已有 PoseRates：转换到 ROS 的相邻世界位置差分，再乘当前姿态矩阵转置，得到当前 base_link 的速度。原始线速度字段仍按协议解析，但不参与 ROS Twist。角速度来源及协议保持不变，不取绝对值、不仅翻转符号、不估计比例。首次或无有效时间间隔时保留零估计和大 covariance；受接收时间抖动影响，仍不是带仿真时间戳的物理速度标定结果。

新增桥接回归覆盖原始 X 符号错误时的前进、后退和静止。修复前首帧仍发布 raw X=4.0 的断言失败，修复后模块 92 passed、7 skipped（8.08 s）。Jazzy 包构建通过；完整 TCP/正式导航 E2E 1 passed、5 deselected（15.88 s），证据 /tmp/ue-pose-linear-e2e-20260911.log 和同名目录。Humble/Orin NOT_RUN。

现场重启 operator 173864，保留 120 m × 120 m 任务和低速扫描。12 段扫描完成并进入 NAVIGATION_EXECUTING；随后 15 s 样本中前向命令已达到 0.2 m/s，不再固定于 0.025 m/s。该窗口包含转向/加减速，平均反馈前向速度约 0.0582 m/s，净位移约 0.743 m，不能宣称持续匀速 0.2 m/s 或物理速度已标定。证据 /tmp/ue-pose-linear-live-speed.json、/tmp/ue-pose-linear-live-scan.json、/tmp/ue-pose-linear-live-1789122517381281615.log。当前保持自动探索运行；未提交或合并。

### 行进速度顿挫修复（2026-09-11）

现场 15 s 样本：命令 66 次为零、单次最大变化 0.2 m/s；位姿速度峰值 6.6247 m/s，接收间隔最短约 0.634 ms，典型约 50 ms。TCP 批量/抖动使相邻位置差分被极小时间间隔放大。行进 tracking_speed 又以测得速度建立加减速可行区间，反馈越过指令上限时区间可为空，触发 BRAKING 零速。证据 /tmp/ue-stutter-before.json。新增两个回归分别复现突发帧速度尖峰 16.67 m/s 与跟踪输出误入 BRAKING。

修改：PoseRates 的线速度改用约 0.2 s 位移窗口并投影到当前车体系，断档清空窗口；角速度暂保持相邻姿态差分。正式执行器记录最近发布的线/角指令，以此约束下一条行进命令；真实状态继续用于路径进度、几何跟踪与停稳检查，节点级零速同步记录。没有提高速度上限或删去碰撞、换段、输入失效停车。窗口带来短暂速度滞后，不等于远端仿真时间标定。

Jazzy 控制器与仿真模块 245 passed、7 skipped（9.39 s），两包构建通过；TCP 正式导航 E2E 1 passed、5 deselected（16.56 s），证据 /tmp/ue-smooth-e2e-20260911.log 及同名目录。Humble/Orin NOT_RUN，未提交或合并。

现场 operator 175855：初始扫描完成后进入自动探索。15 s 采样净位移约 2.594 m、平均车体前向反馈约 0.171 m/s，最后状态 TRACKING，command_v=0.2。修复前同长度窗口净位移约 1.401 m、平均反馈约 0.124 m/s（含尖峰），两个窗口位置和转向过程不同，不能作为严格同路段性能对照。证据 /tmp/ue-stutter-before.json、/tmp/ue-stutter-after.json、/tmp/ue-smooth-live-1789122841331147285.log。继续保持 120 m 任务运行。

末尾约 5 s 持续跟踪窗口：修复前命令在 0～0.2 m/s 间变化；修复后命令稳定 0.2 m/s，反馈均值约 0.240 m/s。实际响应约高于命令 20%，仍需单独标定 Unreal 速度/时间尺度，不能宣称实际车速被限制在 0.2。完整窗口修复后零命令共 82/301（包含启动转向），不能以总零速次数宣称减少；持续跟踪窗口才体现当前改善。

### 30 m 全向观测与普通导航低速转向（2026-09-11）

默认 YAML 改为 sensor_range_m=30、sensor_fov_deg=360、observation_window_m=64，initial_scan=false；全向虚拟观测不再通过起步转圈补盲。保留网格遮挡，任务 120 m、地图 500 m 不变。普通控制 max_angular_radps 从 0.5 降为 0.15，新增仿真配置 max_angular_accel_radps2=0.1 并通过 launch 传入正式控制器；公共 wheel.yaml 未修改。

本机 Jazzy 仿真包 93 passed、7 skipped（8.14 s），包构建通过。额外平坦地图检查：320×320 的 0.2 m 观测窗口内前后左右 25 m 地面均被揭示；真实控制节点实例的普通 policy 角速度/角加速度实际为 0.15/0.1。git diff --check 通过。本次未启动远端车辆；Unreal 上新视场性能、滑移改善与 Humble/Orin NOT_RUN。新配置在下次按现有命令启动时生效。

### 候选点附近左右调整诊断与修改（2026-09-11）

现场 20 s 采样：ALIGNING 251 帧、FINAL_ALIGN 110 帧、TRACKING 25 帧、BRAKING 13 帧、FAILED 1 帧。失败原因为 GOAL_POSITION_LOST，同一 session 的路径段从 revision 31 改为 32；终端路径位置约 (3163.1567,76.7433)，采样结束位置约 (3163.3408,76.7482)，距离约 0.184 m，但仍要求约 1.766 rad 终点朝向调整。证明此时主要在做朝向对准，且发生转向滑移后位置丢失重规划；不是仅由 wheel steering alignment 日志证明故障。证据 /tmp/ue-candidate-turn-diagnosis.json。

仿真控制器位置到达容差从默认 0.2 放宽为 0.3 m，与正式轮式导航 profile 一致。探索器对 360° 传感器发送无终端朝向约束的目标，窄视场保持原约束；正式平台规划仍认证实际路径和姿态。新增 ROS Action 服务测试覆盖全向/窄视场 has_target_yaw。未在当前运行栈热修改，需重启后验证 Unreal 效果。

验证：本机 Jazzy 两包构建通过，仿真模块 93 passed、7 skipped（8.22 s），探索 ROS 包 CTest 12/12 测试目标通过（18.68 s），包含全向/窄视场 Action 朝向约束回归。git diff --check 通过。当前用户运行进程 201152 未重启，修改尚未在现场生效；新行为 Unreal 与 Humble/Orin NOT_RUN，未提交合并。

### 恢复探索目标的终点朝向要求（2026-09-12）

按用户要求撤销上述全向传感器的朝向豁免：探索器恢复使用 NavigationTarget 的默认 has_target_yaw=true，全向和窄视场均要求满足候选目标的终点朝向。只恢复目标提交逻辑，并更新回归测试与使用说明；导航器、控制器、TCP 桥、消息接口、到达状态机、0.3 m 位置容差和仿真观测/限速配置均未修改。探索器提交位姿，导航器负责规划和任务结果，控制器跟踪路径，职责沿用原接口。

红灯验证：将已有全向测试改为 RequiresTerminalViewingYaw 后，真实探索节点通过 ROS Action 发出的 has_target_yaw 仍为 false，测试按预期失败。恢复源码后，本机 Jazzy 受影响包 lunar_pure_exploration_ros 构建通过，包内 CTest 串行运行 12/12 测试目标通过（18.21 s），包含全向与窄视场朝向要求。git diff --check 与修改文件 UTF-8 读取通过。

验证产物位于 /tmp/lunar-restore-terminal-yaw-97x4n_ad/：red.log、red.xml、ctest.log、log-green/、build/、install/；测试域为 ROS_DOMAIN_ID=191、仅本机发现。构建使用独立目录，未覆盖默认缓存 overlay，未重启或切换运行中的仿真；常规启动前应按使用说明重新构建默认 overlay，或显式指定此次验证 overlay。此次恢复的 Unreal 闭环、Humble、原生 Orin 与实车均为 NOT_RUN；不宣称坡面滑移或反复微调已解决。未提交、合并或推送。

### 严格终点容差与停稳反馈修正（2026-09-12）

用户选择直接试验 0.10 m／0.05 rad（约 2.86°）。simulation.yaml 和 SimulationConfig 使用相同默认值，launch 向导航器和控制器分别传入同一配置；导航节点新增对应启动参数，拒绝非正或非有限值，公共平台默认值保持原样。现有协调器按 max(配置位置容差, 半个细栅格) 判定，因此 0.20 m 细图下参数的有效位置下限是 0.10 m；这不是物理精度下限。终点朝向要求保留，本轮不再修改探索器。

控制器的内部 TrackingState 接收 Odometry 的横向速度，既有停稳判断改为 hypot(vx, vy) 加既有角速度门槛；不会把前向为零、横向仍滑动判为停稳。TCP 桥首帧或断档后无有效差分速度时，六个 Twist 分量发布 NaN，保留原大 covariance 和有效位姿；控制器沿用已有无效输入停车逻辑。没有改变 Action/消息接口、模块职责、速度限幅、制动/超时机制或增加重试状态机；角速度差分滤波未修改。

红灯回归分别复现：横向运动被误报 COMPLETED、非有限横向反馈被接受、桥接未知速度被发布成零，以及导航器忽略更严格容差和非法覆盖值。修复后这些回归通过。独立目录构建 lunar_incremental_navigation_ros、lunar_pure_wheeled_controller、lunar_obj_tcp_sim 三包通过；本机 Jazzy 导航 ROS 包 CTest 串行 13/13 通过（3.00 s），控制器与仿真模块 Python 测试 250 passed、7 skipped（9.48 s，包含默认跳过的显式启用闭环测试）。

随后显式启用本机 TCP 正式导航闭环：1 passed、5 deselected（58.26 s）。三个目标为 (1,0,0)、(1,0,0.8)、(0.2,0,0)，均返回 GOAL_REACHED；测试在每次返回时检查模拟车辆平面误差不超过 0.10 m、航向误差不超过 0.05 rad（仅加 1e-6 浮点断言余量），并检查 PLAN_FOUND 诊断、ACTIVE PathReference、跟踪反馈和有序停车。该模型为本机平地运动学回环，不包含 Unreal 重力、摩擦或真实坡面滑移。

产物目录 /tmp/lunar-tight-arrival-cgcnwxu5/：log-green/、ctest-navigation.log、pytest-unit.log、e2e.log、pytest-e2e/test_formal_closed_loop_throug0/receipt.json 与 launch.log。ROS 测试域 193/194/195，仅本机发现；TCP 连接仅本地回环。构建没有覆盖默认缓存 overlay，也未重启远端运行栈；常规使用前需重新构建相关包并重启。Unreal 坡面闭环、Humble、原生 Orin 和实车为 NOT_RUN；尚不能宣称 0.10 m／0.05 rad 是坡面稳定可达精度，或反复微调已完全消除。未提交、合并或推送。


### TCP 分支整理及 P4 部署包（2026-09-16）

整理前发现 16 个已跟踪文件修改，另有 TCP 包、仿真脚本、设计与使用/验证文档未跟踪。按文件审阅并保留原修改；原始补丁、文件哈希及内容备份在 `/tmp/tcp-audit-20260916-103436/`。新增可重复导出器 `tools/create_obj_tcp_deployment.py`，构建模板与源码白名单来自 Orin 1.0 部署提交 `418ea95`；导出当前 TCP 运行源码，不合并部署分支、不推送。旧规划器运行包排除，当前控制器依赖的兼容消息保留。

验证通过：Jazzy 开发链构建 10 包；TCP/控制器 Python 250 passed、7 skipped；导航 ROS CTest 13/13、探索 ROS CTest 12/12；nav 与 obstacle 本机 TCP 闭环 2 passed。初次 CTest 发现 `EnabledDebugPublishesCycleSnapshotAndActualStartPatchView` 只等待任意诊断即断言 EXECUTING，实际先收到 PLAN_FOUND；修正为等待 EXECUTING 后，导航包全量复测及该用例重复 10 次通过，未修改车辆逻辑。

候选部署包在 `phase2-humble-gridmap:local`（Ubuntu 22.04/Humble x86_64）容器只读挂载源码，全新编译 8 包通过；部署包安装目录的 nav 回环测试 1 passed，三个目标均 GOAL_REACHED，位置和朝向满足测试的 0.10 m / 0.05 rad 断言，退出停车确认。依赖闭合、Shell 语法、启动 dry-run 通过。最终包包含完整 500 m 预处理地图、使用说明、来源提交及文件哈希；地图仍 NOT_EXTERNALLY_ALIGNED。本轮 Orin 原生运行、远端 Unreal 和坡面精度 NOT_RUN，不能由 Humble x86 回环推断。部署入口与验证详情以 `TCP仿真部署说明.md` 为准。
