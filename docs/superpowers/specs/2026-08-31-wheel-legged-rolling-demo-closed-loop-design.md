# 轮式与足式滚动 RViz Demo 闭环设计

## 状态与基线

- 状态：已在 `feat/wheel-legged-demo-delivery-closed-loop` 实现，待人工评审与合并。
- 实现基线：`integration/pure-planner-orin` 的 `3e01d1b`；实现提交从 `750ff8e` 开始。
- 已合并能力：轮式滚动恢复、足式 Grid V1 局部搜索优化、32 航向格、机身中心可行图、
  多目标距离场、边认证缓存及轮式/足式独立 RViz 路径输出。
- 已实现：显式地图 ACK、请求/token/分段绑定、12 m 统一门户、4 m 流水刷新、轮式/足式
  恢复闭环和 Demo 专用诊断。实际完成范围与未运行验收项以第 12 节为准。

本文统一以下两份既有设计，但不改变其中已经验证的平台内部搜索语义：

- [Wheel Rolling Demo Stability Design](2026-08-31-wheel-rolling-demo-stability-design.md)
- [足式 Grid V1 分层路径规划设计](2026-08-31-legged-grid-v1-planning-design.md)

## 1. 问题定义

轮式与足式最近出现的失败表象不同，但 Demo 层的共同根因不是某一个局部搜索算法，而是
局部图、里程计、规划请求和可执行路径之间没有交付闭环：

```text
Demo publish(local map)
  -> 立即把 publish 当作规划器已经消费
  -> 继续沿旧路径移动并更新 odometry

规划器 InputStore
  -> 分别保存最新局部图和最新 odometry
  -> 可能组合成“新位置 + 旧窗口”
  -> 起点落到局部图外、门户不可达或局部搜索直接失败
```

`publish()` 只代表消息交给 DDS，不代表规划器回调已经消费；BEST_EFFORT 下尤其不能据此推进
Demo 状态。现有诊断还可能把底层具体原因压缩成 `INVALID_INPUT` 或 `NO_PATH`，使交付失败看起来
像搜索失败。

因此需要解决的是一个共同的 Demo 协议问题：规划必须使用同一次停止位置生成的 odometry 和
局部图，路径必须绑定生成它的请求与地图，车辆只能执行当前已确认路径。

## 2. 目标与边界

### 2.1 目标

1. 轮式、足式共享一套滚动会话和输入交付状态机。
2. 规划器明确确认已接收并准备好某一版局部图后，才允许该版地图进入规划。
3. 每段路径绑定 `request_id + map_token`，旧请求或旧地图产生的迟到路径不能再次驱动车辆。
4. 两类平台统一采用 12 m 名义局部目标距离和最多 32 个动态门户。
5. 每移动 4 m 刷新局部图并在后台重新规划，继续利用旧段剩余前视距离。
6. 保留轮式与足式各自的运动学和边可行性判断，不以降低足式规划距离换取速度。
7. 失败原因按真实阶段上报，Demo 可恢复失败与任务终止失败明确分开。

### 2.2 输入边界

两类平台继续只使用现有数据：

- 全局占据图；
- 局部占据图；
- 局部高程图；
- odometry、现有 TF 和任务目标。

`map_token` 只是已有消息头时间戳的身份标识，不用于判断地图新鲜度。本文不新增观测时间窗、
variance、置信度、新传感器层或额外地形语义。

### 2.3 工程边界

- 新协议只用于 `/lunar_demo/*` 隔离演示。
- 生产规划接口、默认 topic、Task3 数据协议和控制接口保持不变。
- 规划器中的 Demo 协议支持必须默认关闭，仅由 `lunar_surface_rviz_demo.launch.py` 显式启用。
- 不接入 `/Car/T5/Car_Cmd_Vel`，不增加真实底盘连续控制。
- 不增加发布前密集机身扫掠、二次全轨迹认证或地图时间新鲜度门禁。

## 3. 共享框架与平台差异

| 层级 | 轮式与足式共享 | 保留的平台专用逻辑 |
|---|---|---|
| 输入 | 同版占据图、高程图、odometry、TF、目标 | 足式从相同输入派生机身中心可行图 |
| 全局 | Grid V1 全局路线、滚动进度、会话复用 | 各自的平台全局可行性投影 |
| 局部目标 | 12 m 名义前视、1–12 m 回退、最多 32 门户、稳定排序 | 门户须通过对应平台可行图 |
| 启发 | 面向同一门户集合的多目标距离场 | 足式距离场在机身中心可行图上计算 |
| 状态 | 请求级状态键、资源统计、失败分级 | 足式固定 32 航向格；轮式保留现有运动原语状态 |
| 边认证 | 搜索内认证、请求级缓存、成功路径直接输出 | 轮式连续车体扫掠；足式快速可行图检查与必要的精确边回退 |
| Demo | 同一交付确认、执行、刷新、重试和抢占协议 | 发布到各自的路径与时序 topic |

共享框架不要求两类平台使用相同运动原语。差异只应来自平台运动学和地形可行性，不应来自
地图是否交付、路径是否过期或滚动生命周期的不同处理。

## 4. 权威滚动状态机

每个 Action 目标在 `pure_plan_motion_server` 内只有一个权威滚动会话。Demo 节点负责生成地图和
按已接受路径推进，`rviz_goal_bridge` 只负责目标转发与抢占，不再各自推断会话是否已准备好。

```text
IDLE
  -> PREPARING_GOAL
  -> PREPARING_LOCAL_MAP
  -> WAITING_MAP_ACK
  -> PLANNING
  -> EXECUTING
       -> PREPARING_LOCAL_MAP -> WAITING_MAP_ACK -> PLANNING -> EXECUTING
       -> COMPLETED

任意活动状态 + 新目标 -> PREPARING_GOAL（新 session generation）
可恢复失败             -> PREPARING_LOCAL_MAP（最多 2 次）
不可恢复失败/取消       -> FAILED 或 IDLE
```

状态语义如下：

1. `PREPARING_GOAL`：停止推进，清空上一请求的可执行路径，取消旧规划，建立新的
   `request_id` 和 session generation。
2. `PREPARING_LOCAL_MAP`：在当前停止位置生成局部占据图、高程图和 odometry；这些消息使用
   同一个 header stamp，形成 `map_token`。
3. `WAITING_MAP_ACK`：保持当前快照位姿，并重复发布同一 token 的局部图与 odometry。首个目标或
   恢复 STOP 还会交付全局图与 `map->odom` TF；普通刷新保留旧执行段，ACK 后即可继续执行。
   重复输入必须幂等，不得重复重建输入序列或同一足式投影。
4. `PLANNING`：只冻结并使用 ACK 对应的输入序列；若旧段尚未耗尽，Demo 同时继续执行其剩余
   部分。此时收到新目标或新 generation，旧结果作废。
5. `EXECUTING`：Demo 只接受与当前 `request_id + map_token + segment_index` 完全匹配的路径；
   新段激活时跳过已经行驶过的前缀，从当前位置附近的前向段接续。
6. `COMPLETED`：最终精确目标到达后清空可执行路径并结束 Action。

## 5. 最小交付协议

### 5.1 地图身份

Demo 在当前快照位姿同周期发布局部地图与 odometry，并复用同一 header stamp。该 stamp 是
`map_token`，只用于关联消息，不进行年龄或超时新鲜度判断。等待 ACK 期间重复交付同一个位姿
快照；ACK 后恢复旧段执行，规划器继续使用已经冻结的该版输入。

规划器只有在以下条件成立时发布 ACK：

- 已经获得全局图和直接 `map->odom` TF；
- odometry 与局部地图 token 相同；
- 起点位于该局部图覆盖范围内；
- 轮式已完成该 token 的局部地图适配；
- 足式已完成该 token 的机身中心可行图和距离场输入投影。

这些检查直接解决“新位置 + 旧窗口”，不扩展为额外安全认证。

### 5.2 DemoMapAck

新增 Demo 专用确认消息，最小字段为：

```text
string request_id
builtin_interfaces/Time map_token
uint64 local_input_sequence
uint64 projection_revision
uint8 platform_type
```

同一 token 的重复地图只能重复返回相同就绪语义；新 token 到达后，旧 token 的 ACK 不再推进
状态机。足式 `projection_revision` 必须对应该 token，而不是此前缓存的投影。

### 5.3 DemoPlanSegment

新增 Demo 专用可执行段信封，最小字段为：

```text
string request_id
builtin_interfaces/Time map_token
uint32 segment_index
uint8 platform_type
uint8 command  # STOP 或 EXECUTE
nav_msgs/Path executable_path
```

Demo 的运动状态只订阅该信封。现有 `/lunar_demo/wheeled_path`、
`/lunar_demo/wheeled_path_timing` 和 `/lunar_demo/legged_path` 继续作为 RViz/诊断输出；它们本身
不能解除停止状态。新目标到达时，规划器先发布带新 `request_id` 的 `STOP` 信封；Demo 采用该
generation、停止并生成第一版地图。只有字段完全匹配的 `EXECUTE` 信封才能解除停止状态。
这样既保持可视化兼容，也避免旧的普通 `Path` 被误当成当前执行授权。

## 6. 统一 12 m 局部目标

### 6.1 距离定义

- 轮式和足式的名义局部目标进度统一为沿全局路线前方 12 m。
- 12 m 是规划并认证的局部前视段目标，不表示必须在刷新地图前走完全部 12 m。
- Demo 每执行 4 m 就刷新局部图并重规划；ACK 后继续执行剩余约 8 m 的已规划前视段，以覆盖
  后台规划时间。
- 距离最终任务目标不足 12 m 时，只使用未经修改的最终任务目标，完成精确终端连接。

### 6.2 门户生成与排序

门户生成对两类平台使用同一流程：

1. 从 12 m 向当前进度回退，覆盖 1–12 m 的可执行范围；
2. 每个纵向进度先检查路线中心，再检查成对的左右横向候选；
3. 使用平台可行图过滤后，稳定排序为：前进进度更大、横向偏移更小、净空更大、创建序更早；
4. 去重后最多保留 32 个门户；
5. 一次局部搜索接收整个门户集合，而不是逐门户重复运行完整搜索。

中心线被障碍物占据时可以选择侧向门户；12 m 门户不可行时可以回退到较短进度。足式不再用
不同的门户生命周期或较短固定 horizon 掩盖搜索效率问题。

### 6.3 平台内部搜索

- 轮式保留现有连续车体边认证、运动原语和状态标签策略。
- 足式保持 32 航向格、机身中心必要可行图、多目标距离场、请求级边缓存以及快速检查/精确
  回退两级边认证。
- 成功结果已经在搜索中逐边认证，直接作为可执行段输出；不在发布前增加密集机身扫掠。
- 若足式 12 m 搜索超出目标时延，后续只优化启发函数、状态复用、缓存命中和边检查热点，不把
  horizon 降回 4 m。

## 7. 执行、地图刷新与全局路线

1. 全局路线属于整个 `request_id`，只在目标变化、全局图变化或路线已失效时重算。
2. 每个局部周期从当前全局进度选择门户，并生成一条实际认证的局部可执行段。
3. Demo 沿该段累计移动 4 m 后生成新 token；ACK 前保持快照位姿，避免局部图与 odometry
   失配，但不清除旧执行段。
4. ACK 后立即恢复旧段并在后台规划下一段；新段到达时从当前位置附近的前向路径段原子切换。
   在 Demo 协议中，被接受的新 token 本身就是两类平台的显式重规划请求；足式不再额外
   等待到达旧门户或路线步长。协议关闭时的生产足式判据保持不变。
5. 旧段先耗尽时保持停止等待；局部规划失败时清空旧可执行段。RViz 可保留全局路线用于观察。
6. 最终局部周期以原始任务目标为唯一终点；不能以“到达全局路径中点/门户”代替任务完成。

## 8. 失败恢复与目标抢占

### 8.1 可恢复失败

以下局部周期失败允许在车辆停止后刷新地图并重试：

- ACK 未到或地图消息丢失；
- 局部 `NO_PATH`；
- 局部 `TIMED_OUT`；
- 起点与当前 token 不匹配或位于旧窗口外。

恢复顺序固定为：清空旧段、保持停止、重新发布/生成当前停止位置地图、等待 ACK、重新规划。
同一任务连续最多重试 2 次。重试次数不能通过继续执行旧路径来换取。

### 8.2 终止失败

全局无路、非法配置、不可转换目标、显式取消和超过局部重试上限直接结束当前 Action。返回值保留
底层真实 reason code，例如 `WHEEL_POSE_OUTSIDE_LOCAL_MAP`、`MAP_ACK_TIMEOUT`、
`LOCAL_NO_PATH` 或 `LOCAL_TIMED_OUT`，不得统一压缩成无信息的 `INVALID_INPUT`。

### 8.3 新目标

新 RViz 目标从任意活动状态触发 session generation 递增：

1. 立即停止并清空当前可执行段；
2. 取消旧 Action/搜索；
3. 丢弃旧请求迟到的 ACK 和路径；
4. 在当前停止位置为新目标生成地图并等待新 ACK；
5. 只执行新 `request_id` 的第一个 segment。

## 9. 输出与诊断

每个周期至少记录：

- `request_id`、session generation、`map_token`、segment index；
- local input sequence、traversability/projection revision；
- 当前状态、失败阶段、真实 reason code、重试次数；
- 门户候选数、选中门户索引与路线进度；
- 全局路线是否复用、局部扩展状态数、open peak；
- 足式快速边接受数、精确回退数、检查单元数和缓存命中数；
- 局部段长度、当前周期实际移动距离和累计任务进度。

Reporter 每次规划输出前保留一行 `----------------------------`。成功证据仍必须同时满足：

```text
planning_outcome: 0
reason_code: PLAN_FOUND
has_reference: true
```

Action `SUCCEEDED` 或 RViz 中出现路径不能单独作为规划成功证据。

## 10. 实施顺序

1. 在消息包和 ROS 层增加 Demo 专用 ACK/segment 信封及纯状态机单元测试。
2. 让 InputStore 以 token 形成同版 odometry/局部图快照；足式 ACK 延后到投影就绪。
3. 在滚动 Action 中实现初始停止、地图 ACK、流水规划、执行授权、4 m 刷新和最多 2 次恢复。
4. 统一轮式与足式 12 m 局部目标及最多 32 门户，保留平台专用可行性和边认证。
5. 实现 request/token/segment 迟到结果过滤与新目标抢占。
6. 完成确定性非简单场景测试、RViz 实验和操作文档，再考虑性能热点优化。

每一步都先增加失败测试，再实现最小代码使其通过。不得顺带修改 HOPPER、生产控制、Task3 接口
或与本闭环无关的地图语义。

## 11. 验收方案

### 11.1 单元与组件测试

- publish 不等于 ACK；ACK 前状态不能进入规划/执行。
- 同 token 重复地图幂等，不重复构建足式投影。
- 新 token 后的旧 ACK、旧路径和旧 Action 结果均被丢弃。
- 轮式地图适配完成后 ACK；足式机身中心可行图完成后 ACK。
- 两类平台均以 12 m 为首选门户，首选受阻时可在 1–12 m 内回退。
- 门户最多 32 个，一次搜索处理全部候选，最终目标保持精确。
- 移动累计达到 4 m 后保留旧段并交付下一版地图；ACK 后继续执行旧段，直至新段接续或旧段耗尽。
- 两次可恢复重试后终止，终止失败不重试。
- 新目标能抢占正在等待、规划或执行的旧目标。

### 11.2 非简单确定性 RViz 场景

轮式与足式各自至少连续完成 3 个可达目标，整个进程不重启。目标点不能选择为开阔地上的短直线，
每类平台的场景必须同时满足：

- 单个任务全局路线长度至少 100 m，或至少触发 10 次局部图刷新；
- 起终点直线 supercover 穿过障碍/不可行单元，必须发生真实绕行；
- 路径累计转向至少 90°，或包含两次明显主转弯；
- 全局路径中段包含邻近障碍物的局部规划，不只在终点附近放置障碍；
- 记录每段 `request_id + map_token + segment_index` 连续性，并确认没有旧段被再次执行。

轮式至少包含一个 12 m 首选门户受阻、较短或侧向门户成功的场景。足式至少包含一个受机身中心
可行图/高程约束影响的障碍邻近场景，并实际选中非中心或回退门户。另设一个不可达目标，验证
清空路径、停止和真实失败原因；不可达用例不计入 3 个可达目标。

还必须注入一次地图 ACK 丢失和一次规划中目标抢占，验证重发幂等、迟到路径丢弃和新目标闭环。

### 11.3 验证层级

- 本机 Jazzy：构建、单元测试、组件测试、契约测试和 RViz Demo。
- Humble/Orin、DDS、真实 rosbag 和实车：必须分别执行并记录；未执行时标为 `NOT_RUN`。
- 750 m 轮式随机障碍性能场景保留在测试代码中，但按用户决定不作为本次闭环整合的验收门禁，
  记录为 `SKIPPED_BY_USER`，不得删除或放宽原测试。

## 12. 当前整合验证记录

`feat/wheel-legged-demo-delivery-closed-loop` 基于 `3e01d1b` 实现。2026-08-31 至
2026-09-01 本机 ROS 2 Jazzy、`RelWithDebInfo` 的当前结果如下：

- `lunar_planning_msgs`、`lunar_pure_planner_core`、`lunar_pure_planner_ros` 构建通过；仅保留既有
  `TraversabilityInputSnapshot{}` explicit-constructor 编译警告；
- 轮式 GoogleTest 排除唯一 750 m 性能用例后 75/75 通过；该用例保留在源码中并明确记为
  `SKIPPED_BY_USER`；
- 其余 core CTest 21/21 通过，门户测试 15/15 通过；
- 当前 ROS 包级串行 CTest 19/19 通过；其中 server 63/63、InputStore 7/7、
  TraversabilityInput 6/6、Reporter 5/5、确定性场景 6/6 均通过；
- 完整 Python 回归 353 passed、1 skipped；launch contract 34/34 通过；`git diff --check` 和
  冲突标记检查通过；
- 本机未安装 `clang-format`，因此格式器检查为 `NOT_RUN`，不把缺少工具写成通过。

确定性场景测试证明默认起点 `(-349.5, 0.5)` 至默认目标 `(350.5, 0.5)` 的 700 m 直线
supercover 穿过障碍。实际 RViz 短程观察中，轮式连续产生至少 10 个正式成功执行段，并在局部
`TIMEOUT` 后通过 STOP、新 token 和重建路线恢复；足式连续产生 11 个正式成功执行段，在障碍邻域
`NO_PATH` 后恢复，并从 `y=0.5` 绕行到 `y=-3.5`。正式成功段均以
`planning_outcome=0 + reason_code=PLAN_FOUND + has_reference=true` 判定。

上述 RViz 证据只证明滚动闭环、障碍邻域恢复和真实绕行已被短程观察，不证明整条 700 m Action
完成，也不满足“不重启进程连续完成 3 个困难目标”的完整人工验收。Humble/Orin、DDS、真实
rosbag 和实车均为 `NOT_RUN`。

2026-09-01 将普通 4 m 周期改为 ACK 后继续执行旧段并后台规划；新增状态机回归测试覆盖旧段
保留、规划期间推进及新段前缀跳过。本机重新构建通过，ROS 包级串行 CTest 19/19、相关 Python
契约测试 48/48 通过。该次修改后的 RViz 连续运动观察尚未执行，标为 `NOT_RUN`；上面的完整
Python 353 项结果属于修改前基线，本次未重复运行。
