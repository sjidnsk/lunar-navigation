# 月球极区正式训练环境闭环设计

日期：2026-08-08
状态：已批准，作为正式训练前数据、场景、缓存和运行入口的现行权威

> **2026-08-09 生命周期修订：** 本文关于固定决策预算、`pose_features` 剩余预算字段、
> optimizer update 后统一 episode rollover、仅保存 episode cursor 和固定三步正式评估的内容，
> 已由
> [`2026-08-09-unbounded-formal-episode-design.md`](2026-08-09-unbounded-formal-episode-design.md)
> 原子取代。数据、场景、地图分辨率、传感器和平台能力部分继续有效。

## 目标与结论

本设计定义并已闭合 seed `4080` 正式 PPO 训练前的数据与运行链：把已经锁定的月球极区数据、
确定性随机障碍、三平台 capability v2、`30 m/360°` 观测抽象和当前 C++ v3 规划器装配成
同一个可复现训练环境，并让公开命令完整支持：

```text
prepare-data -> calibrate -> train -> checkpoint -> resume -> evaluate
```

完成本设计不等于启动 24 小时训练。本阶段只生成正式场景/cache、执行短时 preflight、冻结
worker 与 micro-batch，并证明上述公开入口使用同一份正式世界身份。历史
`formal-seed-4080` proxy 产物继续保留为开发证据，不能续跑或重命名为正式训练。

## 权威输入

正式环境只接受以下现行权威：

- 仓库外 `polar_source_lock_v1.json` 所锁定的 NASA LOLA 南极 DEM、count 栅格和 JAXA
  LUPEX DataS1；
- 仓库外 `polar_split_v2.json`，文件 SHA-256 为
  `4434b342dddd184ac7ce2cd5b5c2e5e199ef28247f2602c7d9cfa5dfc51e1e2d`，内部
  split SHA-256 为
  `6da68f54d7349142f787d0e82e3cc08b44770f99ae6deddb9f98b7cc5bfcbd18`；
- 仓库内项目正式 capability v2，包含轮式、足式、飞跃式和统一 `30 m/360°` 观测能力；
- `integration` 当前 C++ v3 planner 与 `lunar_planner_training_bridge`；
- 当前 reward、七输入观测合同和飞跃式无累计燃料语义。

旧 Volume 3 路径算法、旧平台能力、旧 proxy 场景、旧 cache 和累计燃料字段均不允许进入
formal 命令。

## 三层空间分辨率合同

“地图分辨率”不再用一个数同时表示网络、传感器和规划器：

| 层 | 范围与尺寸 | 分辨率 | 用途 |
| --- | --- | --- | --- |
| 网络全局画布 | `1024 m × 1024 m`, `256 × 256` | `4.0 m` | 全局先验、覆盖摘要、候选位置 |
| 规划/reveal 局部瓦片 | 默认 `64 m × 64 m`, `320 × 320` | `0.2 m` | C++ v3 局部规划、物理障碍、实际可见性 |
| 网络局部裁剪 | `6.4 m × 6.4 m`, `32 × 32` | `0.2 m` | 当前位姿附近的四通道局部输入 |

网络全局分辨率不随 50 m、1 km 等实验名称变化；它由固定 256×256 网络合同和当前
1024 m 任务窗共同决定。局部 0.2 m 数据按需生成和缓存，禁止为每个 1024 m 窗口分配完整
`5120 × 5120` 多层数组。

NASA DEM 的原始分辨率为 5 m。因此：

- 4 m 全局高程是对真实 5 m 产品的受约束重采样；
- 0.2 m 局部高程中的大尺度趋势来自该 DEM；
- 陨石坑、月岩、禁入区域和亚栅格起伏由冻结生成器确定性补充；
- cache manifest 必须写入 `local_detail_provenance=synthetic_subgrid_on_locked_dem`，不得把
  0.2 m 细节表述成 NASA 实测精度。

## 正式场景库与随机性

### NASA 主训练集

物理场景对三平台共享，平台差异只来自 capability 投影和 C++ v3 规划：

- 192 个 train 窗口，每窗 8 个独立 hazard seed，共 1536 个训练场景；
- 48 个 validation 窗口，每窗 2 个固定 seed，共 96 个验证场景；
- 48 个 test 窗口，每窗 2 个与验证集不重合的固定 seed，共 96 个测试场景。

seed 集固定为：

```text
train      = 408000..408007
validation = 409000..409001
test       = 410000..410001
```

每个场景的最终 seed 仍按
`SHA256(window_sha256:scenario_seed:generator_version)` 派生。月岩、陨石坑和禁入区分别使用
独立随机流，改变一种对象的数量不会重排另两种对象。

### JAXA 独立 holdout

CR1、GR1、GR2、LP1、MP1、MP2 的真实 1 m DTM 只用于 holdout，不进入参数更新。主 holdout
不叠加随机障碍；如需随机压力测试，必须使用独立的 `holdout-stress` 集合且不参与发布门。

### 训练时随机项

场景障碍不是每个 episode 临时任意生成。正式训练从冻结场景库抽样，确保可重复和可复审。
以下随机流彼此独立并写入 schedule 身份：

- `scene_seed`：决定地形和障碍，已在场景 manifest 中冻结；
- `start_seed`：从当前平台的安全起点集合中选择起点；
- `episode_seed`：决定场景遍历顺序和 episode 重置；
- PPO 自身 RNG：由 checkpoint 保存和恢复。

resume 必须恢复同一场景游标和 RNG，不能从 worker index 重新开始场景序列。

## 场景 manifest

`prepare-data` 首先生成 canonical JSON，schema 为
`lunar-formal-scenario-manifest/v1`。manifest 包含：

- aggregate source lock 文件 SHA-256、三个源 SHA-256；
- split manifest 文件 SHA-256、内部 split SHA-256；
- generator version、三组 seed、对象分布参数；
- 三层几何合同和 0.2 m 来源说明；
- 每个场景的 `scene_id/source/split/window_id/window_sha256/scenario_seed/scene_seed`；
- JAXA 成员路径和成员 SHA-256；
- canonical `scenario_manifest_sha256`。

条目按 `(split, source, window_id, scenario_seed)` 稳定排序。manifest 不记录本机绝对路径；
路径由调用命令显式提供，身份只由内容哈希决定。

## 多分辨率场景与 cache

### 几何真值

生成器先产生与分辨率无关的矢量场景定义：圆形月岩、陨石坑参数和禁入多边形。4 m 全局图、
0.2 m 局部瓦片和网络局部裁剪都从同一个定义投影，禁止为不同分辨率重新抽随机数。

全局 cache 每个场景至少保存：

- `elevation_m`；
- `physical_obstacle_ratio`；
- `forbidden_ratio`；
- 三平台由当前 C++ v3 `project_traversability()` 产生的 `hard_feasible` 与
  `clearance_margin_norm`；
- 数组 shape、dtype、字节序和内容 SHA-256。

静态 cache 只保存不会随探索状态变化的地形投影。以下内容必须在线计算，不能缓存：

- `valid_mask`、观测年龄、质量、次数；
- 当前已观测连通域；
- 候选信息增益、candidate mask 和 reward；
- 当前请求、路径、执行状态和飞跃次数。

### 0.2 m 瓦片

局部瓦片由 content-addressed LRU provider 按需生成。key 至少包含
`scene_id/tile_row/tile_column/generator_sha256`。瓦片使用固定世界坐标网格，重叠读取必须逐位
一致。每个 worker 只保留有限数量瓦片；淘汰只删除内存副本，不改变场景身份。

### cache manifest

cache schema 为 `lunar-formal-training-cache/v1`，绑定：

```text
source + split + scenario manifest + generator + geometry
+ capability bundle + reward + training semantics + C++ v3 source commit
```

manifest 还记录每个静态文件的相对路径、大小和 SHA-256。formal 命令必须验证完整 inventory；
缺文件、额外文件、hash 漂移、旧 capability 或旧源码身份均在创建 worker 前失败。cache 和运行
artifact 都位于仓库外。

## 0.2 m 实际观测与 4 m 网络状态

每次真实 reveal 在 0.2 m 局部瓦片上执行 `30 m/360°` 遮挡计算，并用全局 0.2 m cell id
去重。新增覆盖奖励按 `0.04 m²` 单元面积累计，重复观测不重复奖励。

高分辨率结果再保守汇总到 4 m 网络状态：

- 只有被实际 reveal 的 0.2 m 单元才能贡献已知信息；
- 4 m 单元在其中心子单元已观测后才标记为全局 `valid`，避免“看见一个角落即知道整格”；
- 高程、障碍与不确定度只由已观测子单元聚合；
- 网络 32×32 局部裁剪直接读取 0.2 m 已观测状态，未知单元仍为零且由 mask 表达。

候选潜在增益继续在 4 m observed-only 图上快速估算，不能修改上述观测状态或 reward。机器人
当前所在的 4 m 全局栅格必须在估算前排除；原地不动不构成探索候选，也不能触发平台规划。

## 平台投影、起点和请求

每个物理场景只生成一次；三平台分别使用 capability v2 计算静态投影。起点从各平台
`hard_feasible`、清障裕度和边界裕度均满足的单元中确定性选择，并要求初始 reveal 后至少存在
一个 observed-only 候选。找不到起点属于场景构建错误，不能静默换成平地。

每个策略动作的请求固定经过：

```text
PolicyBatch candidate + theta
  -> 同一 observation identity
  -> 4 m 已观测 global map + 0.2 m 已观测 local map
  -> 正式 capability v2
  -> 当前 C++ v3 PlannerBridge
```

轮式和足式执行认证轨迹的末点；飞跃式执行单条参考的名义落点并在 `LANDED_HOLD` 触发一次
观测。飞跃式每个请求都使用相同 capability 推导出的单跳 delta-v，不记录、不扣减 episode
燃料，也不限制累计跳数。

## 正式环境 builder

唯一生产 builder 接收 cache manifest、split 选择、场景游标和正式 capability bundle，返回
`FrozenCapabilityEnvironmentFactory` 与一个真实 `PolicyBatch` 模板。calibrate、train、resume、
evaluate 和 formal-preflight 全部调用这一入口，不允许各自复制场景装配逻辑。

worker reset 时按冻结 schedule 前进到下一场景。checkpoint/run manifest 记录场景 schedule ID、
各 worker episode cursor 和训练 RNG；resume 对这些字段进行完全相等校验。

## 公开命令

### prepare-data

验证 source/split 后生成场景 manifest 和静态 cache：

```text
lunar-policy-training prepare-data \
  --source-lock <absolute> \
  --split-manifest <absolute> \
  --cache-root <absolute-outside-repo> \
  --materialization full
```

`preflight` materialization 仅用于测试，不具备 formal 资格；正式 calibrate 只接受 `full`。

### calibrate

formal calibrate 必须提供 config、cache manifest 和当前主机/源码绑定的 Release 传感性能报告。
它用正式 worker 执行 18/24 worker 与 micro-batch 探测，并在固定验证场景上完成三组 reward
校准，不再调用 proxy。

### train/resume

两者从同一 cache manifest 构造正式工厂。train 只接受已冻结 calibrate 根；resume 还必须恢复
场景游标。正式命令继续禁止 `--max-updates`，短时验证由 formal-preflight 负责。

### evaluate

evaluate 使用 validation、test 和 JAXA holdout 的固定场景，不进入训练 split。PPO、nearest 和
gain-over-cost 三种方法必须在同一物理场景、同一起点和同一请求序列上比较。正式报告
`proxy=false`，并携带 checkpoint 的完整 run identity。

### formal-preflight

该命令不创建正式 checkpoint、不消耗 24 小时预算，只验证：

1. cache inventory 与所有身份；
2. 三平台各一个 worker 能创建、初始 reveal、构造候选和调用 C++ v3；
3. 同一场景/同一请求重复结果一致；
4. 0.2 m reveal、4 m 汇总和局部裁剪来自同一世界；
5. 飞跃式连续两次 episode 不出现燃料递减；
6. worker 进程可退出，18/24 配置可由正式环境执行；
7. checkpoint/resume 的场景游标与 RNG 可确定恢复；
8. formal evaluate 能生成非 proxy 报告。

## 错误边界

单个候选规划不可行仍是环境内普通结果；以下情况属于正式运行基础设施错误，立即停止 rollout：

- source/split/cache/capability/源码或训练语义 hash 不一致；
- 真值被直接传入策略、候选或 observed-only 规划；
- 0.2 m 与 4 m 世界坐标、tile identity 或 observation identity 不一致；
- worker 场景游标漂移、resume 重排场景；
- 正式入口落回 proxy、旧 cache 或累计燃料；
- JAXA holdout 进入训练抽样。

## 完成标准

正式训练前准备完成必须同时满足：

- source/split 在当前主机重新验证；
- full 场景 manifest 与三平台静态 cache 生成在仓库外并通过完整 inventory 校验；
- 公开 formal calibrate 能产生冻结 run root；
- train/resume/evaluate 都能从该 root 和同一 cache 构造正式环境；
- formal-preflight 通过三平台、同世界、无 proxy、无累计燃料和确定恢复检查；
- 24 worker 若通过则冻结为 24；只有既有性能门明确失败时才冻结已资格的 18 worker；
- 文档状态更新为 `formal-training-ready / training-not-started`；
- 未启动正式 seed 4080 的 24 小时训练。
