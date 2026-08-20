# 参数化平台能力与正式训练对齐设计

日期：2026-08-20  
状态：待书面审核  
目标分支基线：`feat/global-ground-anchor-budget` @ `972eb49ba222676c95936a502c9f2070be6775c3`  
能力来源：`github/feat/wheel-parametric-capability` @ `d068faf7c79071416e658bcd32fa0a8cf19ed2f1`

## 1. 目标

把已经进入部署主线的 WHEELED、LEGGED、HOPPER 参数化能力接入最新地面正式训练链，同时保持已经确认的 PPO 动作语义、候选容量、Reward V4、训练观测抽象和轻量评估机制。

本设计解决以下错配：

- 部署规划器读取 `deployment/config/{wheel,legged,hopper}.yaml`；
- 正式训练仍读取旧 `three_platform_capability_freeze_v1.yaml`；
- 部署 `observation.yaml` 是 10 m、90°，正式训练抽象仍是 30 m、360°；
- 候选位置由生成器产生，地面最终朝向由 PPO 的连续 `theta` 输出，但候选预期收益尚未明确绑定前沿法向。

## 2. 已确认决策

### 2.1 动作边界

地面策略动作保持为：

```text
PolicyAction = (candidate_index, theta_rad)
```

- 候选生成器负责给出安全、物理可达、位于任务区域内的候选位置；
- PPO 通过 `candidate_index` 选择位置；
- PPO 通过连续 `theta_rad` 学习到达后的观测朝向；
- PPO 不连续回归候选位置的 `x/y/z`；
- 候选数量和模型张量形状不变。

### 2.2 候选预期收益

候选预期收益使用前沿法向的标准观测朝向计算，但该标准朝向不约束 PPO 的最终 `theta_rad`。

对每个前沿锚点：

1. 仅使用任务 ROI 内、当前未观测的邻域确定未知侧；
2. 由候选位置指向未知侧局部质心得到单位法向 `n_unknown`；
3. 定义 `canonical_yaw = atan2(n_unknown.y, n_unknown.x)`；
4. 在候选位置和 `canonical_yaw` 下计算 `expected_gain_m2` 与 `expected_priority_gain_m2`；
5. 用标准收益完成资格检查、排序和候选特征构造；
6. 执行规划使用 PPO 输出的 `theta_rad`；
7. Reward V4 使用最终执行轨迹实际提交的新增覆盖，不使用标准收益替代真实奖励。

法向不可确定、不是有限值或未知侧为空时，该锚点不产生正式候选；不得使用机器人当前位置方向、全图真值或任务 ROI 外未知格补造法向。

### 2.3 训练观测与部署接口

正式训练继续使用：

```text
sensor_range_m = 30.0
sensor_fov_deg = 360.0
```

`deployment/config/observation.yaml` 中的 10 m、90° 仅属于部署运行配置，不改变本次正式训练语义。Topic 名称、QoS、T3 地图适配、相机到车体 TF 和底盘控制接口均属于部署边界，不进入离线正式训练身份。

若未来把正式训练抽象改成 10 m、90°，必须作为新的训练语义版本、缓存身份和新运行单独设计；不得在本次对齐中静默切换。

### 2.4 当前训练范围

- 训练 worker：12 WHEELED + 12 LEGGED；
- HOPPER worker：0；
- 每个平台仍使用共享 Cross-Attention PPO；
- 全局锚点上限 96；
- 精细窗口上限 32；
- 策略候选 64；
- reserve 32；
- Reward V4、80% 成功阈值、95% 任务物理可覆盖资格门不变；
- 路径规划算法不调整，仅替换能力输入；
- 评估保持地面平台、轻量、非阻塞。

## 3. 能力权威与冻结

### 3.1 单一参数来源

平台参数的项目权威改为：

- `deployment/config/wheel.yaml`；
- `deployment/config/legged.yaml`；
- `deployment/config/hopper.yaml`。

训练不得继续从旧三平台冻结 YAML 重建不同数值的能力。旧文件仅保留为历史兼容输入，不再是新运行的能力权威。

### 3.2 运行级不可变快照

训练启动时必须：

1. 读取三个参数化能力文件；
2. 使用现有严格 schema 和解析器验证；
3. 对每个平台生成 canonical JSON 语义摘要；
4. 将原始能力文件复制到 run 外部产物目录；
5. 在 run manifest 中记录源路径、文件 SHA256、平台内容 SHA256 和组合摘要；
6. worker 仅接收已冻结的不可变能力对象，不在宏动作中重新读取文件。

平台文件在训练期间发生变化时，新 worker 启动必须拒绝加入旧 run；现有 worker 不热更新能力。

### 3.3 平台作用域身份

地面训练的资格、缓存和 worker 身份分别绑定 WHEELED 与 LEGGED 的平台内容摘要。HOPPER 能力仍需通过加载器和训练桥测试，但 HOPPER 内容变化不得单独使一个只含地面 worker 的 checkpoint 失效。

## 4. 规划与训练桥

参数化能力分支已经完成 C++ 核心、ROS 加载器和 Python 训练桥的类型扩展。本次对齐只做以下接线：

- WHEELED 使用新足迹、车体包络、轮径、轴距、轮距、净空、坡度、速度、曲率、64 航向和运动基元；
- LEGGED 使用参数化车体包络、速度、坡度、台阶、沟宽、净空和显式运动基元；
- HOPPER 参数继续可加载、可桥接，但不创建训练 worker；
- `TrainingPlanRequest.capability` 必须与候选可达投影所使用的平台内容摘要一致；
- 能力摘要不一致时 fail closed，不得回退旧工程基线；
- 不修改全局规划、滚动局部规划、碰撞检查和失败处置算法。

## 5. 缓存与恢复

### 5.1 可复用内容

以下内容与新平台参数无关，可以保留：

- 原始场景源文件；
- 任务区域和场景划分；
- 地形高程、障碍和语义基础栅格；
- 与平台无关的 tile 压缩数据；
- update 457 checkpoint 中形状兼容的策略网络参数。

### 5.2 必须重新生成或认证的内容

WHEELED 数值能力已经改变，必须重新生成：

- 物理可达投影；
- 端点安全投影；
- 平台可覆盖分母；
- 候选可达性相关派生索引。

LEGGED 和 HOPPER 只有在 canonical 物理字段与旧缓存逐字段相等、派生投影算法身份相同且 bit-exact 复核通过时才可迁移旧派生缓存；任一条件不满足则只重建对应平台，不重建原始场景。

正式训练传感器仍是 30 m、360°，因此传感器可见性缓存不因部署 10 m、90° 文件失效。

### 5.3 checkpoint 使用方式

update 457 不作为精确恢复点。新 run 采用 fresh episode，并且只热启动形状和语义兼容的共享策略参数：

- 不恢复旧 worker 环境；
- 不恢复未完成宏动作；
- 不恢复 optimizer；
- 不恢复 GAE；
- 不恢复 value head；
- 不恢复旧运行归一化统计；
- 保留旧 checkpoint 和 run manifest 作为来源证据。

## 6. 分支集成策略

实施分支必须从最新训练基线 `972eb49` 创建，而不是切换到部署文档分支。将参数化能力提交 `d068faf` 合入该分支时：

- 保留 `972eb49` 的 ground-only worker、异步轻量评估、96/32/64+32 候选预算和缓存兼容修复；
- 接受参数化能力分支对 C++ 核心、ROS 加载器、训练桥、schema 和能力类型的改动；
- 对 `project_capability.py`、`capability_freeze.py` 和相关测试做语义合并，不使用整文件 ours/theirs 覆盖；
- 不把 Task3 Topic、SQLite、TF 或底盘控制适配混入训练对齐提交。

部署主线随后只合入已经验证的训练能力快照与候选法向收益改动，不反向覆盖现有 Task3 适配器。

## 7. 最小验证边界

本次不运行全场景正式门，不启动 HOPPER 训练，也不执行三种子评估。完成条件为：

1. 参数化能力 schema、WHEELED/LEGGED/HOPPER 加载器和训练桥聚焦测试通过；
2. WHEELED 新能力逐字段进入 C++ 规划请求；
3. LEGGED 新能力逐字段进入 C++ 规划请求；
4. 前沿法向只使用 ROI 内未知侧，标准收益确定且可重复；
5. PPO 输出的地面 `theta_rad` 不被标准法向覆盖；
6. 30 m、360° 正式训练语义未被部署 10 m、90° 配置改变；
7. 旧缓存对新 WHEELED 能力 fail closed；
8. 经等价认证的平台缓存可按平台复用；
9. update 457 仅提取兼容策略权重并创建 fresh-episode 新 run；
10. 单个 WHEELED 和 LEGGED 场景各完成一个完整宏动作，无硬错误、非有限值或身份漂移；
11. `git diff --check` 与目标 Python 编译检查通过。

验证通过后才恢复 24 个地面 worker。训练启动、首次 checkpoint 和首个指标 update 分别报告，不能把进程存活表述为训练健康或收敛。

## 8. 明确不做

- 不修改 PPO 网络结构；
- 不让 PPO 连续回归候选位置；
- 不增加候选数量；
- 不修改 Reward V4；
- 不降低安全、碰撞、净空或规划失败约束；
- 不修改路径规划算法；
- 不把部署 Topic/QoS/TF 写进训练身份；
- 不恢复第三级全任务残余区域扫描；
- 不启动或评估 HOPPER；
- 不运行全场景长门禁；
- 不删除旧 checkpoint、原始场景或历史缓存。

