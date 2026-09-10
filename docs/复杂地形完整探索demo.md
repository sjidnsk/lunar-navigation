# 复杂地形完整探索 demo

本 demo 原开发于 `feat/integrated-exploration-rviz`，本次集成到 `integration/pure-planner-orin`，保留域 71 的旧探索演示和域 72 的[简单控制闭环演示](增量规划控制闭环demo.md)。默认使用 ROS 2 Jazzy、域 73 和两个 RViz 窗口。只操作模拟车辆。

## 场景和实际算法

默认 300×300 m，固定种子 20260910；共有 447 块岩石和 149 段脊线，含缓坡、洼地、交叉宽通路和局部狭缝。不是两块矩形障碍的放大版。

```text
有限视野、遮挡的合成高程 + 模拟车辆真实 odometry / TF
  → 正式 PersistentElevationMap：持久 0.2 m 细图、1 m 全局粗图
  → 正式前沿探索：前沿检测、观测候选、信息增益排序
  → NavigateToPose
  → 正式粗指导 + LocalGoalRegion + 平台局部搜索与通行性认证
  → PathReference
  → 正式 PathExecutor（incremental_reference）
  → /lunar_demo/integrated/cmd_vel
  → 速度、加减速受限的模拟车辆
  → 实际 odometry + 匹配 session / revision 的 TrackingStatus
```

模拟传感器每帧覆盖 28 m 方形窗口内的 12 m / 120° 前向扇区，另含 1.4 m 近场；岩石、脊线遮挡后方观测，未观测位置为 NaN。导航持久保存观测证据，64 m 局部规划窗从细图提取，粗图随观测扩张。规划窗不等于传感器视野，也不表示窗内都已知。

全地图地形预览与障碍轮廓是**显示专用真值**，只提供给 RViz 和模拟器碰撞检查。正式地图、探索器和规划器收不到全图真值。`GetPolicyMap`、DRL、KnownSpaceRoutePlanner 和返航不在本演示中。

### 高程、通行性、占据与场景真值

| 数据 | 当前含义 | 是否机器人测到的地图 |
|---|---|---|
| 棕色场景轮廓 | 模拟器预设的实体岩石/脊线形状，完整场景从启动时就存在 | 否；仅供显示和物理模拟 |
| 输入 GridMap 的 `elevation` | 传感器测到的表面高度，单位 m；NaN 表示本次没有测量 | 是；黄色点仅显示最近批次测量位置，没有用颜色表达高度 |
| 0.2 m fine state | 根据高程邻域、坡度/起伏和车辆安全范围计算的 FREE/BLOCKED/UNKNOWN 通行性 | 是；UNKNOWN 也包含已经测到部分高程但不足以认证通行性的情况 |
| 1 m exploration map | fine state 的三态投影，供前沿检测使用 | 是；用 OccupancyGrid 消息承载，但不是射线更新的占据概率地图 |
| 1 m global guidance | 粗规划使用的 CANDIDATE/PROVEN_BLOCKED/UNKNOWN 方向指导 | 是；与 exploration map 共用粗几何，但状态投影规则不同 |

场景只有一套地形：地面、缓坡、洼地、岩石、脊线共用 Terrain 中的地面函数和实体几何；没有另行输入一张“黑色障碍物真值占据图”。点位高程查询、显示预览、模拟碰撞和局部传感器都来自这套场景。黑色格是正式高程物理评估和安全膨胀生成的 BLOCKED，因此其范围可能大于实体轮廓。高度大不等于障碍：平缓高地可能可通行，陡坎或超过平台能力的高度差才构成阻塞依据。

当前合成观测采用一致的局部 0.2 m 离散场景：与实体相交的栅格承担该实体的表面高程，遮挡判断也使用这些格。沿传感器到目标格的射线，只返回遮挡前的可见地面和首个实体格，后方仍为 NaN。边界格的格心可能位于连续实体外，返回高度取自格子与实体交集中的固定代表点；同一全局格在不同窗口、不同视角下高度一致。这个局部栅格场景只在模拟传感器内部使用，发布给正式节点的仍是经过视野和遮挡过滤的标准 GridMap/elevation。

这是明确的栅格尺度表面近似，没有根据传感器高度和俯仰做完整三维光线追踪。连续实体几何与车辆碰撞检查保持不变。正确行为是：可见的陡峭前缘提供高程差，由正式地图生成 BLOCKED；被遮住的内部或后方保留未知。既不能因为格心落在实体外就丢掉实际命中，也不能因为知道模拟器真值就把整块岩石或墙后填成已知。

`ExplorationMapProjector` 的实际规则是：1 m 格中只要有一个 FREE 细格就输出 0；没有 FREE 且整个面积都由 BLOCKED 细格覆盖才输出 100；其他情况输出 -1。例如 25 个细格中 24 个 BLOCKED、1 个 UNKNOWN，粗格仍是 UNKNOWN。该接口把“未测到”和“测量不足以完成通行性分类”合并了，前沿信息增益无法仅凭它区分这两类未知。

因此当前 HUD 的百分比标为 `classified coarse cells`：它统计已分类粗格面积，并非原始高程测量覆盖率。局部观测记忆解决同一证据下的重复访问，不完成观测/占据/通行性/可观测性四类语义的分层。小区域仍可能保留未知并进入等待，不能从目标数增长宣称探索完成。

### 传感器已经是 0.2 m，探索器目前仍用 1 m

当前传感器在 28 m 局部窗口内以 0.2 m 格采样；12 m 视距、120° 视场决定哪些格有实际观测。持久高程图与局部通行性计算也保留 0.2 m。范围、分辨率和测量准确度是不同概念，这里的 0.2 m 指栅格间距。

损失细节的环节在后面：正式探索器只订阅 1 m exploration map，前沿、候选与信息增益都在这张投影图上计算。局部细图虽然已经存在，目前主要由局部规划和平台认证使用，探索器尚未用它完成局部精细决策。

面向大地图的后续分层应是：1 m 全局图安排探索区域和粗路线；0.2 m 局部图识别实际观测前沿、选择可达观测姿态并评估局部收益；执行仍通过正式细图认证。这是尚未实现的探索架构修改，不能仅把公共 coarse_resolution_m 改成 0.2 就称为完成。后者会连全局图一起变细：300×300 m 从 90,000 格变成 2,250,000 格，也没有解决真实观测与通行性 UNKNOWN 混合的问题。

## 与实际算法及简单 demo 的差别

三项已接受的修改继续保留：LocalGoalRegion 由平台搜索选出可达候选；正式控制器停稳反馈驱动切段/完成；前进和后退均限制为 0.2 m/s。没有第二个速度发布者，没有沿规划路径直接移动位姿的替代执行器。

演示配套机制有：合成传感器和车辆；统一可调 `/clock`；开局通过正式导航执行四个原地朝向目标以建立初始观测；自动提交探索边界；显示与记录。开局扫描不会计入探索目标验收。可视化节点把最新原始 odometry/TF 按 5 Hz 墙钟原样转发给探索器，减少高倍速下重复构造探索显示的开销；导航和控制器直接接收高频原始状态。

本分支还修复了两个正式探索调度问题，均不改变规划器的可达性判定：

1. 目标到达后允许用当前有效地图继续选其他候选；成功候选保存对应传感器范围内的局部地图证据。只有该局部证据变化时才重新允许同一候选，远处地图更新或时间戳变化不解禁。受抑制候选显示灰色，待选为绿色，当前目标为红色；HUD 区分生成总数与实际待选数。该记忆不把 UNKNOWN 改成已知。
2. 导航明确反馈 `WAITING_FOR_MAP` 时，探索器保留真实等待状态。可选 `navigation_map_wait_timeout_s` 限制无新证据的等待；正式节点与本 demo 均默认 30 秒（use_sim_time=true 时为模拟秒），显式设置 0 可关闭。超时后取消该导航，等取消终态确认，再按现有失败记忆抑制该候选并选择其他候选。取消不会记为 `GOAL_REACHED` 或证明 `NO_PATH`。

若所有候选都被抑制而地图不变，状态为 `WAITING_FOR_MAP_CHANGE`，不会伪报探索完成。`COMPLETED_NO_REACHABLE_FRONTIER` 表示当前地图没有可选前沿，并不保证遮挡内部、岩石内部或整个真值地图被观测。

## 启动

已构建产物位于 `/tmp/lunar-mainline-demo`，重启或清理临时目录后需要重新构建。每个新终端先执行：

```bash
source /opt/ros/jazzy/setup.bash
source /tmp/lunar-mainline-demo/install/setup.bash
export ROS_DOMAIN_ID=73
export ROS_LOCALHOST_ONLY=1
```

首次启动或停止本 demo 后重新启动：

```bash
ros2 launch lunar_integrated_exploration_demo integrated_rviz.launch.py
```

车辆先原地扫描，然后自动探索 300×300 m 边界；默认请求 30 倍仿真时钟。已经运行时不要重复启动同一域。关闭 RViz 只关闭显示；启动终端按 `Ctrl+C` 才停止整套 demo。

可改为小任务区验证完整闭环，世界地形仍为 300 m：

```bash
ros2 launch lunar_integrated_exploration_demo integrated_rviz.launch.py \
  task_size_m:=50.0 time_scale:=30.0
```

无界面运行加 `start_rviz:=false start_local_rviz:=false`。`auto_start:=false` 保留节点但不执行初始扫描、不提交任务。

## 调整倍率、暂停和恢复

运行时调整，不必重启；允许 1～60 倍，浮点或整数均可：

```bash
ros2 param set /integrated_vehicle time_scale 10.0
ros2 param set /integrated_vehicle time_scale 30.0
ros2 param get /integrated_vehicle time_scale
```

车速始终按模拟秒为 0.2 m/s；30 倍意味着理想情况下每墙钟秒走 6 m。HUD 同时显示请求倍率与实际倍率。计算资源不足时实际倍率会低于请求值，时钟不通过一次大跳追赶；最大单次时钟步长 0.02 模拟秒，物理积分步长不超过 0.01 模拟秒。规划耗时仍是墙钟耗时，高倍速不是 Orin 性能证据。

传感器按 1 Hz 模拟时间采样；地图发布上限为 5 Hz 墙钟。30 倍下，一条地图消息会合并约 6 次真实采样，只合并有限高程值，保留遮挡处的 NaN；不会因为降低发布频率而扔掉转向过程中已采到的证据。HUD 中的实际倍率可能低于请求值，采样数、发布数、每批采样数和实际采样间隔单独记录。该模型仍是离散 1 Hz 采样，不是连续光学或激光扫描。

暂停探索（控制器执行取消并停稳）：

```bash
ros2 topic pub --once /lunar_demo/integrated/exploration/task \
  lunar_pure_exploration_msgs/msg/PureExplorationTask \
  "{task_id: integrated-complex-terrain, command: 2}"
```

继续同一任务：

```bash
ros2 topic pub --once /lunar_demo/integrated/exploration/task \
  lunar_pure_exploration_msgs/msg/PureExplorationTask \
  "{task_id: integrated-complex-terrain, command: 3}"
```

暂停探索不暂停 `/clock`。不要向模拟速度话题手动发速度，否则会破坏“正式控制器是唯一速度来源”的验证条件。

## 两个窗口怎么看

| 窗口/显示项 | 看什么 |
|---|---|
| Global exploration | 完整地形、任务边界、1 m 粗地图扩张、前沿和探索轨迹；滚轮可放大已探索区域 |
| Local detail | 跟随车辆，0.2 m 细图、未知边缘、局部候选和选中目标、路径认证后的绕障 |
| COARSE guidance route，橙色 | 粗地图上的方向指导，不能直接执行 |
| FINE certified local path，绿色 | 正式细分辨率搜索给出的当前可执行路径 |
| ACTUAL trajectory，蓝色 | 模拟车辆实际走过的轨迹 |
| 青色扇形 / 紫色方框 | 12 m 观测范围 / 64 m 局部规划范围 |
| ACTUAL measurement coverage - yellow | 最近发布批次真正含有限高程值的格子；黄色只表示测到，不表示可通行 |
| Fine cost / Fine traversability risk / Start blind-zone patch | 可在左侧勾选的代价、通行性风险、起点盲区补丁 |
| HUD | 已分类粗格比例、完成目标数、前沿数、执行阶段、实际速度、路径段版本、实际倍率 |

默认不显示 A* OPEN/CLOSED。局部目标标记只在当前局部规划产生候选时有内容；不是每一个探索动作都会触发多个滚动候选。全图预览可取消勾选，以专看逐步建立的地图。

### 为什么扇形扫过后还有未知，棕色矩形是什么

- `SIMULATOR truth - rocks and terrain` 的棕色小矩形、长条，是岩石和脊线的真值轮廓；未探索处也会显示。取消该图层即可只看机器人建立的地图。紫色大方框是随车移动的 64 m 规划窗口。
- 整片灰色矩形还可能是 OccupancyGrid 的矩形显示范围：消息必须有矩形宽高，但其中灰格仍为 UNKNOWN，矩形范围并不等于观测范围。
- 青色扇形只是最大几何视野，并不表示其中每格都被测到。岩石后方、岩石内部和离散采样间隙可能没有有限高程；以黄色实际观测点为准。
- 细图显示的是**通行性**。即使某点有高程，周围 3×3 高程邻域或车辆安全膨胀范围内仍缺证据时，细图仍可能是 UNKNOWN。测到一点不等于证明车辆可以经过。
- 全局窗口的 `GLOBAL exploration projection 1.0 m` 显示探索用的通行性三态投影，不是原始观测掩膜，也不等于内部 global guidance 状态。正式局部搜索仍会独立检查细图。

初次 GUI 验收版本曾把“采样”也限制为 5 Hz 墙钟，30 倍转向时约 6 模拟秒才采一次，确实会漏掉中间朝向；最终版本将采样和发布分离，保留这一问题及修复验证，不能将它全部归因于真实遮挡。

后续还发现障碍边界上的数值错误：同一全局格心在不同移动窗口中出现不同浮点坐标，导致静态高程在“地面”和“脊线”之间翻转。最终使用全局整数格索引统一生成格心，要求同一格在重叠窗口里的坐标、高程一致。这类伪地图变化会反复解禁候选，不能解释为正常探索行为。

用户指出的“实体挡住车，扫描后却没有障碍证据”还复现了第三个错误：旧传感器先用膨胀实体截断射线，再独立取格心高程，可能在到达障碍面前就结束；仅去掉膨胀也不够，例如墙面 y=6.25 m 穿过格心 y=6.3 m 所在格，按格心采样仍会误取地面。上述统一栅格场景修复了命中与高度来源不一致。回归同时检查真实传感器输出、正式细图/粗探索图和未知阴影，具体结果见验证记录。

同一验证还修复了正式 FineBuilder 的扩图漏更新：远处的新观测扩大地图范围后，旧障碍附近新纳入的格子也需要按已有证据计算通行性。现在从稀疏候选瓦片中补算新旧范围的差集，和同一最终高程快照的完整计算保持一致。原始高程仍可能是 NaN，但车辆安全范围已与已知障碍相交时，通行性可以证明为 BLOCKED；这不表示该格已经获得新的高程测量。扩图时仍需检查已分配的候选瓦片，计算量不按整个新增外包矩形面积增长。

## 记录和重新构建

记录从当前开始收到的正式规划、最终段停稳反馈与探索进展：

```bash
ros2 run lunar_integrated_exploration_demo record \
  --output /tmp/integrated-progress.json --timeout 180 --minimum-goals 5
```

记录器只读，结束不停止 demo。`--require-completed` 额外要求探索状态到达 COMPLETED；超时保留失败证据，不修改目标或覆盖阈值。匹配正式 `PLAN_FOUND`、有效 PathReference 的细图版本、同 session/segment 的最终段停稳反馈，并排除开局扫描；看到一条 RViz 路径不算验收。

重新构建（源码目录与安装目录保持分离）：

```bash
cd /home/kai/WS/lunar-navigation/lunar-runtime/.worktrees/pure-planner-orin/ros2_ws
source /opt/ros/jazzy/setup.bash
CMAKE_BUILD_PARALLEL_LEVEL=4 colcon --log-base /tmp/lunar-mainline-demo/log build \
  --base-paths src --build-base /tmp/lunar-mainline-demo/build \
  --install-base /tmp/lunar-mainline-demo/install \
  --packages-up-to lunar_integrated_exploration_demo --parallel-workers 3 \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
```

本机验证及限制见[验证记录](复杂地形完整探索demo验证.md)。Humble、Jetson AGX Orin、真实传感器和实车均为 `NOT_RUN`。
