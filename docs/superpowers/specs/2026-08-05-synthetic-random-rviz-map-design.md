# 50 m 合成随机地图与 RViz2 交互规划设计

日期：2026-08-05

状态：对话设计已批准；待书面规格审阅

选择：独立合成地图包；固定随机种子；纯 ROS 2/RViz2；不检查连通性

## 目标

在仓库外的 ROS 验证工程
`~/CodexDownloads/lunar_navigation/isaac_ros_action_regression` 中增加一个确定性合成地图
入口。地图覆盖 `50 m × 50 m`，分辨率为 `0.2 m`，使用固定随机种子生成中等难度
障碍。用户继续在 RViz2 左侧面板选择轮式、腿式或跳跃式平台，再使用标准
`2D Goal Pose` 选择目标并观察路线、弹道、落区或不可行原因。

本实验与 Isaac Sim 无关，不启动或连接 Isaac Sim，不读取、修改或保存 USD，也不把合成
数据伪装成 Isaac 快照。合成地图只用于交互探索和对应的自动化测试，不改变正式六案例
回归及其证据语义。

## 已批准选择与未选方案

采用独立、可哈希的合成地图包。生成器写出明确标识为 synthetic 的 manifest 和数组文件，
交互控制器通过单独的命令行参数加载它。固定 seed 的地图只生成一次，后续交互会话读取同一
份不可变数据。

未选择以下方案：

1. RViz2 每次启动时在内存中重新生成。即使 seed 固定，这也会把生成、发布、进程同步和
   证据保存耦合到一次交互会话中。
2. 复用 `isaac-ros-planning-snapshot/v1` 并填充假的 stage、prim 或 USD 哈希。该方案会
   污染来源证据，因此禁止使用。
3. 生成障碍后搜索连通区域或按规划结果重采样。用户已明确不检查连通性；地图只满足几何、
   密度和起点安全区约束。

## 范围与所有权

- 主仓只保存本设计、后续实施计划和既有规划器接口，不新增或修改正式 ROS 消息。
- 代码实现位于仓库外验证工程
  `~/CodexDownloads/lunar_navigation/isaac_ros_action_regression`。
- 生成结果写入
  `~/CodexDownloads/lunar_navigation/ros_random_maps/<map-id>/`，不得进入主仓。
- 原 `--manifest` Isaac 快照入口、正式场景锁和六案例自动回归保持兼容且语义不变。
- 新入口不得探测、启动或清理 Isaac Sim 进程。

## 组件与数据流

```text
固定 seed 合成地图生成器
        │
        ▼
map_manifest.json + map_arrays.npz
        │
        ▼
RViz2 交互控制器 / 合成地图桥
        ├── GridMap ──────► 三类 PlanMotion 规划器
        └── PointCloud2 ──► RViz2 地面和障碍显示
```

### 合成地图生成器

新增无 ROS 运行时依赖的命令行生成器。它校验参数、生成所有必需图层、写入临时目录、重新
加载并校验结果，最后以原子目录发布方式生成一个新 `<map-id>`。目标目录已经存在时必须
失败，不得覆盖既有地图。

### 通用规划地图加载边界

交互路径增加通用的 planning-map bundle 内部表示。它包含栅格描述、十层数组、平台起点
和来源元数据。既有 Isaac loader 与新的 synthetic loader 分别完成自身 schema 的严格
校验，然后转换为该内部表示。

正式回归仍直接使用既有 Isaac 快照和场景锁，不改成 synthetic loader，也不允许自动
schema 降级。交互启动参数必须明确选择 `--manifest` 或 `--synthetic-map`，两者互斥。

### 合成地图桥与 RViz 显示

合成地图桥复用既有六 Topic 发布、TF、mission、GridMap 编码和交互状态机。规划器继续
消费原始 `grid_map_msgs/msg/GridMap`；RViz2 继续只使用标准
`sensor_msgs/msg/PointCloud2` 显示地面和障碍，避免重新引入
`grid_map_rviz_plugin` 的退出崩溃路径。

## 合成地图契约

manifest schema 固定为 `lunar-synthetic-planning-map/v1`，至少包含以下严格字段：

- `schema_version`
- `map_id`
- `source_kind=deterministic_random`
- `generator`：生成器名称、版本、随机算法和 seed
- `geometry`：frame、原点、尺寸、分辨率、宽和高
- `obstacle_parameters`：类型、尺寸范围、目标和允许占用率、障碍高度、起点安全半径
- `actual_obstacle_occupancy`
- `platform_starts`
- `arrays_file`
- `arrays_sha256`

未知字段、缺少字段、非有限数值、数组尺寸不符或哈希不符均 fail closed。生成器采用固定
版本的 NumPy `PCG64` 随机算法；数组键按稳定顺序写入，manifest 使用排序键和 UTF-8，
使同一版本、参数和 seed 可重建同一数组内容与哈希。

`map-id` 由 schema、生成器版本、几何、障碍参数、平台起点、seed 和数组内容共同绑定，
不得使用时间戳作为唯一身份。

## 地图几何与图层

地图几何固定为：

| 字段 | 值 |
|---|---:|
| XY 范围 | `[-25.0, 25.0) m × [-25.0, 25.0) m` |
| 原点 | `[-25.0, -25.0] m` |
| 分辨率 | `0.2 m` |
| 宽、高 | `250 × 250` |
| 全局 frame | `map` |
| 局部 frame | `odom`，且本实验 `map→odom` 为单位变换 |

为保持现有规划输入契约，bundle 包含 `global`、`wheel`、`legged` 和 `hopper` 四个 grid。
四者覆盖同一完整 `50 m × 50 m` 范围并使用相同合成数据；bridge 仅发布全局图和当前所选
平台图。完整范围使 RViz2 可在整张地图上选点，不引入动态裁剪或地图换页。

十层数据固定如下：

| 图层 | 合成规则 |
|---|---|
| `elevation` | 全部为 `0.0 m` |
| `valid_mask` | 全部为 `1` |
| `obstacle` | 随机图元栅格化得到的二值掩码 |
| `obstacle_height` | 障碍单元为 `0.5 m`，其余为 `0.0 m` |
| `observation_age_s` | 全部为 `0.0 s` |
| `observation_quality` | 全部为 `1.0` |
| `elevation_variance` | 全部为 `0.0` |
| `obstacle_variance` | 全部为 `0.0` |
| `observation_count` | 全部为 `1.0` |
| `forbidden` | 全部为 `0` |

所有层使用当前 contract 可接受的数值 dtype，形状严格为 `(250, 250)`，且所有值必须
有限。

## 随机障碍生成

默认 seed 为 `20260805`，默认目标占用率为 `0.12`，允许最终范围为 `[0.11, 0.13]`。
生成器以稳定顺序从 PCG64 采样位置、类型和尺寸，圆形与矩形默认等概率混合：

- 圆形半径范围为 `[0.4, 1.2] m`；
- 矩形 X/Y 边长分别从 `[0.6, 2.4] m` 采样；
- 图元允许互相重叠，实际占用率按去重后的栅格单元计算；
- 会令实际占用率超过 `0.13` 的候选被拒绝；
- 达到目标占用率后停止；有限次候选后仍未进入允许范围则生成失败。

障碍按栅格单元中心是否落在图元内部进行确定性栅格化。该规则、边界包含方式和浮点比较
方式由生成器版本固定，避免不同运行产生边界单元漂移。

三类平台起点如下：

| 平台 | 起点 `x,y,z` | 姿态 `w,x,y,z` |
|---|---|---|
| 轮式 | `[-18.0, -16.0, 0.0]` | `[1.0, 0.0, 0.0, 0.0]` |
| 腿式 | `[-18.0, 0.0, 0.55]` | `[1.0, 0.0, 0.0, 0.0]` |
| 跳跃式 | `[-18.0, 16.0, 0.60]` | `[1.0, 0.0, 0.0, 0.0]` |

每个起点 XY 周围半径 `2.0 m` 的闭圆盘是障碍保留区。候选图元只要覆盖该区域中的栅格
中心就拒绝。生成器不检查三个安全区与地图其他区域是否连通，不遍历可达域，不调用规划器，
也不因后续规划不可行而重新生成。

## RViz2 操作与显示

生成命令设计为：

```bash
./scripts/generate_synthetic_map.py \
  --seed 20260805 \
  --size-m 50 \
  --resolution-m 0.2 \
  --obstacle-occupancy 0.12 \
  --output-root /home/kai/CodexDownloads/lunar_navigation/ros_random_maps
```

生成成功后输出唯一的 manifest 绝对路径和 map id。交互启动命令设计为：

```bash
./scripts/run_interactive_rviz.sh \
  --synthetic-map \
    /home/kai/CodexDownloads/lunar_navigation/ros_random_maps/<map-id>/map_manifest.json \
  --install install/<build-id>
```

现有 `--manifest` 参数继续加载 Isaac 快照；传入两个来源参数或均未传入时显示明确用法并
失败。

RViz2 行为保持为：

- 左侧 `Lunar Planner` 面板提供轮式、腿式和跳跃式三个互斥按钮；
- 面板显示只读 source、map id、seed、尺寸、分辨率和实际障碍占用率；
- 切换平台后只显示当前平台和对应起点；
- 平坦地面显示为灰色，当前平台地图的障碍显示为红色；
- 使用标准 `2D Goal Pose` 在整个地图内选择 XY 和 yaw；
- 障碍目标或地图外目标由交互控制器本地拒绝，显示红色目标和稳定原因；
- 空闲目标发送给所选平台的 `PlanMotion`；轮式路线为蓝色、腿式路线为绿色，跳跃式
  显示橙色弹道、飞行管、落区和着陆点；
- 无可行路线时保留目标并显示规划器不可行原因，不绘制虚假路径；
- 不增加“重新随机”按钮，避免同一实验会话中地图身份发生变化。

平台切换、Goal 并发、取消、状态显示、旧 Marker 清理和只显示当前平台等规则继续服从
已批准的 RViz2 单平台交互设计。

## 错误处理

- seed、尺寸、分辨率或占用率非法时，生成器在创建发布目录前退出。
- `size_m / resolution_m` 不能无误差地形成正整数栅格时拒绝生成。
- 输出 `<map-id>` 已存在时拒绝覆盖；用户必须显式选择已有 manifest 或改变参数。
- 临时输出只有在 manifest、数组及哈希完成回读校验后才能发布。
- synthetic loader 对 schema、几何、图层、平台起点、占用率和哈希执行严格校验。
- 加载失败不得静默回退到 Isaac loader，也不得启动规划器或 RViz2。
- 交互启动、退出和进程清理继续只作用于脚本拥有的 PID，不按名称批量终止 ROS、RViz
  或其他进程。

## 测试与验收

实现按 TDD 推进。

### 单元测试

- 同一生成器版本、参数和 seed 产生相同数组内容、数组哈希和 map id；不同 seed 的障碍
  掩码不同；
- 几何严格为 `50 m × 50 m`、`0.2 m`、`250 × 250`；
- 实际障碍占用率位于 `[0.11, 0.13]`；
- 三个 `2.0 m` 起点安全区没有障碍；
- 十层键、形状、dtype、有限值和固定层语义正确；
- 栅格化的圆形、矩形和边界单元规则有固定小样例；
- 未知或缺失字段、错误数组哈希、错误形状、非法占用率和越界起点均被拒绝；
- `--manifest` 与 `--synthetic-map` 互斥，旧 Isaac 入口的默认行为保持不变；
- 面板能显示合成地图元数据，切换平台时不会残留其他平台或旧请求的显示。

### ROS 集成测试

在独立 ROS domain 中：

1. 加载生成的固定地图并依次选择三个平台，均达到 `READY`；
2. 核对 `/environment/map_global`、`/environment/map_local` 的 frame、尺寸、分辨率、
   图层顺序和障碍单元数；
3. 核对 `/lunar_isaac_validation/global_surface` 和
   `/lunar_isaac_validation/local_hazards` 的标准 PointCloud2 显示数据；
4. 每个平台选择一个起点安全区内的空闲目标，确认请求进入 Action 规划流程；
5. 选择一个已知障碍单元，确认本地拒绝原因和红色标记；
6. 退出 RViz2，确认只清理本会话拥有的控制器、规划器和 RViz PID。

这些测试不得扫描全图连通性，不得断言存在长距离通路，也不得为得到可行路线而重采样。
起点安全区内的短目标只验证交互链路可操作，不代表全图可达性。

### 兼容性与人工验收

- 外部工程全部 source-first 单元测试和 ROS 集成测试通过；
- 既有正式六案例 Action 回归继续 `6/6`；
- 既有 Isaac manifest 交互入口仍可加载；
- 人工启动 RViz2，依次选择三类平台，确认地图信息、随机障碍、起点、目标和至少一次规划
  终态可见；
- 保存地图、障碍、当前平台、目标及规划结果的可视化证据到本次仓库外 artifact 目录。

## 完成定义

只有当固定地图成功生成并通过 hash-bound contract 校验、RViz2 中三平台均可选择和选点、
自动测试与原六案例回归均通过、退出无 RViz 段错误且可视化证据可审阅时，才能报告本功能
完成。任何长距离目标是否可达由该固定随机分布和平台规划器自然决定，不属于生成器验收
条件。
