# 月球极区三平台 v3 单网络 PPO 完整设计

## 状态与权威关系

本文冻结 2026-08-04 已确认的卷三训练架构，覆盖月球极区数据、观测、候选、动作、共享网络、C++ v3 闭环、奖励、训练、恢复和验收。与下列旧文档冲突时以本文为准：

- [规划算法 v3 三平台单网络 PPO 重训练设计](2026-08-03-lunar-v3-three-platform-ppo-retraining-design.md)
- [卷三策略流水线实施计划](../plans/2026-08-02-lunar-navigation-volume-3-policy-pipeline.md)
- [月球极区 Blender 环境设计](2026-08-03-lunar-polar-environment-design.md)中的训练地形部分

旧设计仍可说明历史决策和已完成实现，但其可变尺寸地图、22 维候选、程序正弦代理地形、旧奖励权重及已具备正式训练条件的假设不再有效。外部 Topic、静态能力资料和任务字段仍以[外部输入接收基线](../../interfaces/external-input-baseline.md)为唯一权威来源。

当前平台运动能力尚未确定。本阶段允许实现并验证所有能力无关的前置组件，也允许用显式测试 fixture 验证能力相关接口；禁止把 fixture 或旧 `proxy-v1` 参数冻结成正式能力，禁止启动正式 PPO rollout。正式训练必须通过本文的能力冻结门。

## 目标与范围

目标是用一个共享 PPO 网络在轮式、足式和飞跃式三类 C++ v3 规划闭环中训练收敛，并最终生成一个 ONNX 模型包。三类平台共享网络、权重和输出 head，通过平台类型和平台化观测学习不同的高层探索策略；C++ v3 始终拥有运动可行性和安全裁决权。

本设计包括：

- NASA/JAXA 月球南极真实高程数据和确定性程序障碍；
- 固定物理尺寸的全局与局部地图输入；
- 64 个前沿候选和连续朝向动作；
- 一个共享 cross-attention PPO 网络；
- 三类平台 C++ v3 宏步环境；
- 以成功为主导的奖励；
- 可暂停、可恢复的 RTX 4080 SUPER 正式训练；
- 按平台分别计算的 95% 验收和后续 ONNX/TensorRT 发布边界。

当前实施只推进到“正式训练就绪但能力未冻结”的状态。正式训练、正式 checkpoint、正式评估、四文件模型候选、AGX TensorRT 和设备状态标签均不在本阶段执行。

## 单网络与固定七输入合同

新合同命名为 `ObservationContractV2`。输入名称和顺序仍为七项，但 shape、通道和语义固定如下：

```text
prior_channels      float32 [B,4,256,256]
coverage_summary    float32 [B,3,256,256]
local_crop          float32 [B,4,32,32]
frontier_features   float32 [B,64,12]
pose_features       float32 [B,6]
candidate_mask      bool    [B,64]
platform_context    float32 [B,3]
```

除 `candidate_mask` 外均为 FP32；所有浮点输入必须有限。输入不得根据 batch、地图来源或平台改变 shape。训练、checkpoint、PyTorch、ONNX Runtime、TensorRT 和探索节点共同消费同一份合同常量，不复制另一套字段定义。

`platform_context` 只编码平台类型，不增加 capability 数值：

```text
WHEELED = [1,0,0]
LEGGED  = [0,1,0]
HOPPER  = [0,0,1]
```

每行必须恰好一个 `1.0`。平台类型不是全部运动能力；它只选择共享网络中的行为条件。实际能力通过平台化可通行观测、候选约束、完整 v3 capability 和闭环结果体现。

## 固定网络地图几何

### 全局画布

- 覆盖范围固定为 `1024 m × 1024 m`。
- 分辨率固定为 `4 m/cell`，因此张量固定为 `256 × 256`。
- 画布在一次 mission revision 内固定，使用 `map` 坐标轴，不随机器人移动或旋转。
- 画布中心取任务 ROI 包围盒中心；ROI 超出画布时拒绝该任务，不静默裁剪。
- 任意合法源分辨率通过有语义的面积、最近邻或高程重采样进入固定画布；重采样到 4 m 不代表产生了新的 4 m 实测细节。

### 局部画布

- 覆盖范围固定为 `8 m × 8 m`。
- 分辨率固定为 `0.25 m/cell`，因此张量固定为 `32 × 32`。
- 机器人位于画布中心，但画布轴仍与 `map` 对齐，不随机器人 yaw 旋转。
- `/environment/map_local` 位于 `odom` 时，先用有效的 `map -> odom` TF 转换，再裁剪和重采样；不得直接把 odom 栅格当作 map 轴。

输入地图可以具有不同原始范围和分辨率，但网络所见的物理范围、分辨率、轴向和 shape 不变。超出源有效范围的单元保持零值，同时由观测掩码明确区分，不能把 padding 当作已知平坦地形。

## 地图通道

### `prior_channels [B,4,256,256]`

1. `observed_relative_elevation`：仅已观测区域的相对高程；未观测和 padding 为零。
2. `mission_priority`：由任务 ROI 和 `science_regions[].priority` 栅格化得到，范围 `[0,1]`。
3. `observed_physical_obstacle_ratio`：已观测确定物理障碍在单元内的面积比例，范围 `[0,1]`。
4. `active_platform_traversable_ratio`：当前平台在单元内的可通行面积比例，范围 `[0,1]`。

### `coverage_summary [B,3,256,256]`

1. `observed_ratio`：单元内已观测面积比例。
2. `mission_roi_ratio`：单元与任务 ROI 相交面积比例。
3. `unobserved_priority_ratio`：尚未观测的任务优先区域比例。

### `local_crop [B,4,32,32]`

1. `relative_elevation`：相对当前机器人高程。
2. `observed_mask`：局部观测有效比例。
3. `observed_physical_obstacle`：确定物理障碍占比。
4. `active_platform_traversable`：当前平台的可通行比例。

比例通道均限制在 `[0,1]`。位置由固定画布归一化，角度使用正余弦。高程归一化只使用训练 split 统计并写入冻结 manifest；验证、测试和部署禁止重新拟合。未观测区域不能使用完整 DEM 真值填充网络输入；完整 DEM 只供仿真世界和结果判定使用。

物理障碍面积使用障碍 footprint 与栅格单元的二维面积交集并对重叠 footprint 求并集，再除以单元面积。程序月岩是物理障碍；坑、坡和粗糙地形保留为高程几何，由平台能力决定是否可通行，不因某一平台不能通过就改写为通用物理障碍。概率障碍不得直接升级为确定物理障碍。

## `pose_features [B,6]`

```text
x_norm
y_norm
sin_yaw
cos_yaw
mission_observed_ratio
remaining_decision_budget_ratio
```

`x_norm/y_norm` 相对固定全局画布归一化；yaw 是 `map` 中的绝对朝向；覆盖率和剩余预算位于 `[0,1]`。

## 前沿候选

候选数量固定 `M=64`。不足 64 项时按零填充并置 `candidate_mask=false`；超过 64 项时使用确定性代表点与最远点选择，不依赖容器遍历顺序。禁止把当前位置伪造为候选填满数组。

每个候选固定 12 个字段：

```text
0  x_norm
1  y_norm
2  distance_from_robot_norm
3  bearing_sin
4  bearing_cos
5  potential_coverage_gain_ratio
6  priority_weighted_gain_ratio
7  normal_sin
8  normal_cos
9  normal_confidence
10 clearance_margin_norm
11 region_remaining_ratio
```

候选生成沿用原版设计中可维护的轮廓分段思路，并统一为：

1. 只从已观测与未观测交界提取前沿，不查看未观测 DEM 真值。
2. 使用观测范围和视场参数确定 anchor 间距与 standoff。
3. 采用 observed-only LOS 估计潜在覆盖；禁止穿透已知物理障碍。
4. 使用面积加权计算覆盖收益和优先收益，不用栅格个数代替物理面积。
5. 每段先选代表点，再以确定性最远点补齐，最终稳定排序。
6. 预筛只排除字段不完整、终端区域明显不合法或当前平台快速投影明确不可行的候选；完整路线仍由选中后的 C++ v3 认证。

`clearance_margin_norm` 是当前平台最紧净空约束的归一化余量。能力未冻结时可以实现接口和测试，但不得生成正式候选缓存。`candidate_mask` 全 false 时不调用策略采样，也不生成当前位置 fallback；环境报告无候选决策边界并按任务状态处理。

候选不再提供 `recommended_theta`。前沿法向只作为几何特征，最终朝向由策略动作产生。

## 动作合同

动作由两部分组成：

1. 对 64 个候选的 masked categorical 选择；
2. 以所选候选为条件的连续绝对 `map` yaw。

每个候选的朝向 head 内部输出 raw sine、raw cosine 和 concentration；公开输出固定为：

```text
frontier_logits  float32 [B,64]
theta_mu         float32 [B,64]
theta_kappa      float32 [B,64]
value            float32 [B]
```

`theta_mu = atan2(raw_sin, raw_cos)`；`theta_kappa = clamp(softplus(raw), 0.05, 64.0)`。训练时从条件 Von Mises 分布采样，部署时使用 masked argmax 候选和对应 `theta_mu`。策略熵由候选 categorical 熵和条件角度熵分别加权。

目标 yaw 容差固定为 `π/24`。轮式和足式 v3 在正式训练前使用 64 个 yaw bin；飞跃式继续使用其着陆和承诺语义，不强行套用地面平台 yaw lattice。

v3 拒绝动作时，该次决策消耗一个决策预算单位，在当前观测快照内屏蔽该候选并重新请求策略；地图、任务 revision 或机器人状态变化后重新构建 mask。PPO 不能执行拒绝态携带的 reference，也不能覆盖已承诺飞跃。

## 共享网络

三类平台只使用一个网络、一套权重和一组 head：

- 全局 7 通道编码为 `32 × 32` tokens；
- 局部 4 通道编码为 `16 × 16` tokens；
- token dimension 为 `128`；
- cross-attention 使用 4 heads、2 blocks；
- 64 个候选 token 作为 query；
- `pose_features` 与 `platform_context` 各自编码后加入共享上下文；
- 策略和价值 head 全部共享，不增加平台专用分支。

候选 mask 在 attention、动作分布、value pooling 和导出路径中都必须生效。模型参数、输入和输出统一 FP32；训练时不使用混合精度作为默认基线。

## 平台运动能力如何进入闭环

正式能力不作为第八项网络输入。三层机制共同表达运动能力：

1. `platform_context` 告诉策略当前是轮式、足式还是飞跃式。
2. 同一地形根据当前 `PlatformCapability` 生成不同的全局/局部可通行通道、净空余量和候选 mask。
3. 完整 capability 传入对应 C++ v3，由 v3 生成和认证轨迹或飞跃参考；结果、代价、耗时和下一状态返回 PPO。

因此只切换 one-hot、却让三类平台看到相同可通行输入和相同执行结果，是合同违规。

### 能力冻结门

正式训练命令必须接收仓库外的绝对 `capability-bundle`。该 bundle 必须包含三类平台各一份能力资料及一个清单，清单至少记录：

```text
bundle_schema
platform_id
platform_type
capability_version
base_frame_id
capability_file
geometry_files
sha256
```

训练启动器要求三种 `platform_type` 恰好各一份、文件存在、哈希匹配、字段可由 v3 loader 完整解析，并把 bundle 总哈希写入冻结 run manifest、checkpoint 和评估报告。测试 fixture 和旧 Isaac `proxy-v1` 明确带 `test_only/proxy` 标记，正式模式必须拒绝。

在 bundle 未提供或未冻结时允许：CPU 单元测试、C++ v3 fixture 测试、数据下载与切片、网络前向、虚构能力的开发 smoke、checkpoint 机制测试和未训练 ONNX smoke。禁止：正式 seed rollout、正式 checkpoint、正式评估、模型候选、`model-package-ready` 或更后状态。

能力冻结后如果改变任一参数、运动基元、几何、v3 源码或 capability 版本，旧正式 checkpoint 不可恢复；必须形成新的配置哈希并重新训练，或至少建立独立完整验收，不能沿用旧发布结论。

## 月球极区数据

不再复用旧 `MOON_LRO_NAC_DEM_89S210E_4mp.tif` 作为训练基线。

### 主训练数据

主数据采用 NASA Goddard [South Pole LOLA DEM Mosaic](https://pgda.gsfc.nasa.gov/products/81)：

- `ldem_87s_5mpp.tif`：87°S–90°S、5 m/pixel、高程 GeoTIFF，约 3.3 GB；
- `ldec_87s_5mpp.tif`：对应 LOLA return count，约 140 MB；
- 坐标为南极立体投影 X/Y 米制，参考框架为 MOON_ME DE421。

不下载约 4.9 GB 的预计算坡度文件；坡度和粗糙度由锁定的 DEM 处理实现统一计算。count map 只用于数据质量、来源追踪和 split 分层，不作为隐藏策略输入，也不凭任意阈值删除区域。

从 NASA mosaic 确定性抽取 192 个训练窗口、48 个验证窗口和 48 个测试窗口，每个窗口为 `1024 m × 1024 m`。选择算法固定 seed `4080`，窗口不重叠，不同 split 的窗口中心至少相距 2 km；窗口坐标、源像素范围和哈希写入 split manifest。

### 高分辨率留出数据

高分辨率留出采用 JAXA [LUPEX Candidate Sites DTMs and Orthomosaics](https://jlpeda.jaxa.jp/en/product/archive/detail_14/index.html)及其 [Zenodo 数据集](https://zenodo.org/records/17153447)：六处地点为 CR1、GR1、GR2、LP1、MP1、MP2，DTM 分辨率为 1 m/pixel，并带 ROI 和高程不确定度等辅助数据。

六处 JAXA 地点不得进入训练、奖励校准或普通验证。最终每个平台 64 个正式评估回合中，16 个来自 JAXA 留出地形及固定程序障碍变体，48 个来自 NASA test split。JAXA 高程不确定度用于相应回合的 `elevation_variance`，不作为策略可偷看的未来真值。

### 数据存放与锁定

原始数据、派生瓦片、场景缓存和下载临时文件只放在：

```text
~/CodexDownloads/lunar_navigation/volume3/data/
```

CLI 实际接收解析后的绝对路径，仓库只提交 schema、下载器、清单模板和小型合成测试 fixture。受控下载完成后，source lock 记录最终 URL、文件名、字节数、SHA-256、CRS、transform、NoData 和许可/引用；后续复用逐项校验。Git 边界检查必须拒绝 GeoTIFF、DEM、DTM、派生训练瓦片、checkpoint 和运行日志进入仓库。

## 真实大尺度地形与程序小障碍

NASA 5 m 和 JAXA 1 m 数据提供真实大尺度高程、坡度和坑貌。局部 `0.25 m` 网格通过高程插值承载大尺度表面，但插值部分不宣称具有新的真实细节；小于源分辨率的月岩、小坑缘破碎和局部起伏由确定性程序生成器补充。

程序生成器输入为 `source_window_hash + scenario_seed + generator_version`，输出至少包括障碍 footprint、高程增量、物理障碍 union、禁入区和生成参数清单。随机月岩/月坑不能改变 train/validation/test 的源归属；同一正式 scenario id 必须逐位复现相同危险层。

大坑首先作为高程几何存在；只有明确的岩石实体、不可穿越空洞或任务禁入多边形进入物理障碍/forbidden 层。平台不能通过的坡面由平台化可通行性表达，不污染通用障碍定义。

## C++ v3 训练闭环

训练热路径继续通过 `pybind11` 进程内调用 ROS 无关的 `lunar_planner_core`，禁止 Python A*、ROS Action、DDS 或规划结果缓存替代 v3。

每个宏步：

1. 场景构建七输入观测和 64 个候选；
2. PPO 选择候选和绝对 yaw；
3. 环境构造包含状态、世界快照、目标、完整 capability、planner config 和 previous execution 的请求；
4. C++ v3 返回 outcome、directive、可选 reference 和诊断；
5. 轮式/足式沿认证轨迹推进到下一决策边界；飞跃式在 committed/in-flight 期间不再调用策略，直至真实 `LANDED_HOLD` 反馈；
6. 环境生成覆盖、优先覆盖、代价、耗时、结果和下一观测。

基础设施、合同、非有限值和非法 outcome/directive/reference 组合使未完成 rollout 作废并停止当前 run，不能转成普通负奖励。正常 `GOAL_INFEASIBLE`、`NO_KNOWN_SAFE_ROUTE` 和 `RESOURCE_EXHAUSTED` 是可学习结果。

## 奖励

三平台共享同一奖励，不使用 goal progress，也不维护平台专用权重：

```text
r_t =
  20.0 * delta_mission_observed_ratio
+  5.0 * delta_priority_weighted_observed_ratio
-  0.10 * normalized_plan_or_execution_cost
-  0.05 * normalized_macro_step_time
-  0.20 * I(executed_without_new_coverage)
-  outcome_penalty
+ 50.0 * I(first_crossing_95_percent_success)
- 10.0 * I(episode_ends_without_success)
```

`outcome_penalty` 为：

- `GOAL_INFEASIBLE = 2.0`
- `NO_KNOWN_SAFE_ROUTE` 或 `ACTIVE_REFERENCE_INVALIDATED = 3.0`
- `RESOURCE_EXHAUSTED = 1.5`

覆盖和优先覆盖增量按物理面积计算，并在回合内望远镜式累加，因此正常正向塑形总量最多约 25；首次成功奖励 50，明确高于全部正常塑形奖励。成功奖励每回合只发放一次。回合失败终止额外扣 10。碰撞、禁入区侵入、未认证 reference 或飞跃承诺违规立即终止并额外扣 50，且同时导致发布安全 gate 失败，不能用成功奖励抵消。

非法请求、陈旧输入、数值故障、C++ 异常、合同错误和非有限张量不进入 rollout。奖励配置哈希写入 checkpoint 和评估报告。

## PPO 与并行训练

固定 PPO 基线：

```text
gamma                    0.995
gae_lambda               0.95
policy_clip              0.20
value_clip               0.20
learning_rate            3e-4
optimizer                AdamW
weight_decay             1e-4
adam_epsilon             1e-5
epochs_per_update        4
target_kl                0.03
value_loss_coefficient   0.5
frontier_entropy_coef    0.01
theta_entropy_coef       0.001
max_grad_norm            0.5
dtype                    float32
rollout_horizon          32
```

正式训练前比较 18 与 24 workers；24 workers 的吞吐不低于 18 且资源稳定时使用 24，联合训练固定轮式、足式、飞跃式各 8 workers。18 workers 回退为各 6。每个 worker 内部计算线程固定 1。24 workers 的 update batch 为 768 transitions，18 workers 为 576 transitions；micro-batch 在 RTX 4080 SUPER 上探测后冻结。

正式 seed 只有 `4080`。24 小时累计 GPU 预算为初始上限：最多 2 小时校准、最多 6 小时三平台预热、至少 16 小时联合训练。训练可暂停和中断；如需延长，由同一配置按明确的 6 小时增量增加累计预算，不重置 checkpoint 或已消耗时间。

正式训练开始前必须同时冻结：数据 source/split/generator、观测合同、候选合同、奖励、PPO 配置、三平台 capability bundle、v3 source commit、worker/micro-batch 和正式 seed。

## Checkpoint、暂停与自动监督

- 每 30 分钟在完整 PPO update 边界原子更新 `latest.pt`。
- 每小时保存不可覆盖的 candidate checkpoint。
- 每 3 小时在冻结验证集运行一次评估。
- `SIGINT`/`SIGTERM` 停止领取新 rollout，丢弃未完成 rollout，保存最后一个完整 update 后退出。
- 暂停期间不计 GPU 工作时间；恢复后累计时间不能清零。
- checkpoint 包含模型、优化器、scheduler、归一化、RNG、课程、全局步数、配置/数据/split/generator/capability/reward/v3 哈希及累计 GPU 秒。
- 任一冻结身份不匹配时拒绝恢复。

watchdog 每 10 分钟检查进程、最新完整 update、GPU 可见性、磁盘余量和有限指标。存在显式 pause marker 时不自动重启；否则只从最近完整 `latest.pt` 恢复。监督不持续占用人工会话，也不改变训练配置或自行延长预算。

训练 artifact 只写仓库外绝对目录，不提交 checkpoint、日志、数据、场景缓存或设备生成物。

## 评估与发布门

正式评估每个平台固定 64 回合：NASA test 48、JAXA 留出 16。三平台分别满足：

- `success_coverage_rate >= 0.95`，即至少 61/64 成功；
- 硬安全违规数为 0；
- 非法动作数为 0；
- 输出有限率为 1.0；
- reference 平台不匹配为 0；
- 飞跃承诺违规为 0。

成功定义为任务 ROI 的有效观测面积首次达到 95%。不同平台不能互相平均掩盖失败。一次完整评估通过即可产生训练候选，不要求连续三次通过，也不要求胜过规则基线；规则基线只作诊断。

能力未冻结前的测试 fixture、程序正弦 proxy 或未训练网络结果不得进入上述 64 回合统计，也不得产生正式候选记录。

训练通过后，卷三后续仍只生成一个四文件模型包：

```text
policy.onnx
manifest.json
golden_inputs.npz
golden_outputs.npz
```

AGX 原生 TensorRT、性能与设备验证发生在模型候选之后；安装、独立验收和手动激活边界保持不变。本阶段不得提前标记 `model-package-ready`、`device-verified` 或 `policy-pipeline-v1`。

## 当前前置实施范围

能力未确定期间直接实施：

1. `ObservationContractV2` 的固定 shape、通道、候选和平台 one-hot；
2. 共享网络对 4/3/4/12 通道和固定 token geometry 的适配；
3. 连续 theta 合同、Von Mises 参数范围和 64 候选 mask；
4. NASA/JAXA 数据 source manifest、仓库外路径、下载校验和固定 split；
5. 固定全局/局部几何、坐标变换、面积重采样和观测 builder；
6. 确定性月岩/月坑生成、物理障碍 union 和 scene manifest；
7. 12 维候选生成器及能力注入接口；
8. 新奖励、PPO 配置、checkpoint 身份和评估数据溯源；
9. capability bundle schema、校验器及正式训练硬拒绝门；
10. 能力无关单元测试、C++ fixture 回归、CPU/CUDA smoke 和边界检查。

当前阶段明确不执行：

- 选择或提交三平台正式运动能力数值；
- 用测试能力生成正式 traversability/candidate 缓存；
- 正式 worker 吞吐和 micro-batch 冻结；
- seed 4080 正式 rollout、预热或联合训练；
- 正式评估、ONNX 模型候选、TensorRT 或 AGX 状态推进。

前置阶段完成的定义是：所有能力无关测试通过；给定一份合法但标记为测试用途的三平台 capability fixture，可以跑通数据窗口到 v3 宏步、PPO update、checkpoint/resume 的短 smoke；正式命令使用同一 fixture 必须明确拒绝，并准确报告缺少正式 capability bundle。

## 验证

前置实现至少验证：

- 合同 shape、dtype、字段顺序、物理尺寸和全 false mask；
- NASA/JAXA source lock、哈希、split 互斥和 JAXA 禁止训练；
- CRS、TF、NoData、count、面积重采样及固定 crop；
- 程序障碍可复现、footprint union 和 train/test 来源不泄漏；
- 三平台化 traversability 接口在不同测试 capability 下产生不同结果；
- 12 维候选稳定排序、面积收益和无当前位置 fallback；
- categorical + Von Mises 动作采样与重算 log-prob 一致；
- 奖励成功 50、失败 10、覆盖 20、优先 5 和硬安全 50；
- checkpoint 对数据、split、能力、奖励和 v3 哈希不匹配时拒绝恢复；
- 正式训练在能力未冻结时拒绝，开发 smoke 仍可执行；
- RTX 4080 SUPER 短 CUDA 前向/update/中断恢复 smoke；
- ROS 2 Humble C++ bridge 和 planner fixture 回归；
- UTF-8、`git diff --check`、仓库边界和 foundation 测试。

验证保持直接，不增加签名体系、审批图或训练 artifact 治理平台。安全边界依靠 v3 所有权、能力冻结、输入合同、正式启动硬门和按平台独立验收。
