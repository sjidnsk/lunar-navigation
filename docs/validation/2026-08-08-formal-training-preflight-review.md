# 正式训练前逻辑与语义复审

日期：2026-08-08

## 结论

现行 C++ v3 规划、三平台 capability v2、`30 m/360°` 观测闭环、平台动作语义、飞跃式无累计
燃料合同、正式极区场景/cache 和公开 formal 运行链已经一致闭合。当前状态定义为：

```text
formal-training-ready / training-not-started
```

本结论只表示启动正式训练前的输入、环境、身份、恢复和评估入口已经准备完毕。校准 manifest
的 `global_step=0`，未产生正式 checkpoint，也未执行 seed 4080 的 PPO 参数更新。

## 复审范围

本次只检查四项：

1. 前后数据流和控制流是否闭环；
2. 同一个术语是否只有一种现行解释；
3. 现行算法、能力和训练边界组合是否合理；
4. 操作文档、设计文档、迁移交接和资格报告是否指向同一权威。

不新增签名、审批、安全平台、通用质量体系或其他与正式训练闭环无关的工作。

## 已闭合的现行链

### 规划与能力

```text
three_platform_capability_freeze_v1.yaml
  -> schema 与规范化摘要校验
  -> FrozenWheeled/Legged/HopperCapability
  -> lunar_planner_training_bridge
  -> 当前 C++ v3 planner
```

仓库内 capability v2 是本项目正式训练和规划的唯一能力权威，规范化 SHA-256 为
`60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95`。外部
capability lock、旧 `proxy-v1`、旧 Volume 3 能力字段和旧固定 search limit 均不能进入
formal 命令。

### 探索观测与奖励

```text
训练真值
  -> 初始真实 reveal
  -> 已观测世界
  -> observed-only 候选与信息增益
  -> 七项策略输入
  -> 候选位置与 theta
  -> 当前 C++ v3 规划和认证
  -> 执行到探索决策边界
  -> 单次真实 reveal
  -> 实际新增覆盖奖励与下一观测
```

`30 m` 是候选估算与真实 reveal 的几何上限，不会自动把圆内地图标为已知。实际已知状态由
`valid_mask`、`observation_age_s`、`observation_quality` 和 `observation_count` 表达；
候选潜在增益不直接写入状态或奖励。

### theta

`theta` 只表示候选位置处的绝对 `map` 机体 yaw。`360°` FOV 只消除观测增益对 yaw 的依赖；
轮式和足式仍通过终端朝向、运动学代价和执行时间学习 `theta`，飞跃式的 `theta` 在动作、熵和
PPO loss 中统一 mask，并以无目标 yaw 的请求调用规划器。

### 飞跃式

飞跃式一次策略动作只认证一个安全凸落区和一条简单抛物线。质量、比冲和参考推进剂只换算每次
相同的单跳 `available_delta_v_mps`；它们不是 episode 库存。规划请求、ROS 快照、策略输入、
奖励、终止条件和下一 Goal 都不存在上一跳剩余燃料。外部 RViz/Isaac 桥同样不发布推进剂 Topic
或执行燃料提交。

## 已闭合的正式训练链

### 1. 正式场景与 cache

`prepare-data` 已按 source/split v2、确定性场景生成器、capability v2、训练语义和当前 C++ v3
规划器生成 full cache。生产加载器校验完整 inventory、每文件摘要和统一 RunIdentity；旧 Volume 3
cache 不能通过该身份门。

### 2. formal calibrate 可执行

公开 `calibrate` 只接受正式配置、full cache 和当前主机/源码的传感器性能报告。当前主机实测
18/24 worker 与 micro-batch 1/2/4 后，冻结为 24 workers、micro-batch 4，并把 cache、场景
schedule、能力、训练语义和源码身份写入 run manifest。

### 3. formal train/resume 使用同一环境

唯一 `FormalEnvironmentBuilder` 从 cache manifest 构造正式数据窗口、当前 capability、传感器
闭环、请求身份和执行器；公开 `train/resume` 均从已校准 run root 重新装配同一 factory 与
observation template。update boundary 保存/恢复场景游标和 RNG，不允许从半个 episode 恢复。

### 4. formal evaluate 与 preflight

公开 `evaluate` 固定使用 validation、test 和六站点 JAXA holdout，不读取 train split；PPO 与
基线在同一场景、起点和请求序列上比较。进程级 `formal-preflight` 已验证非 proxy evaluation、
三平台 worker、同世界/同请求、4 m/0.2 m 映射、无累计燃料和确定恢复，且不创建 checkpoint。

## 启动正式训练的边界

训练前准备门已经满足，但本轮授权明确不启动 24 小时训练。后续启动必须显式运行公开
`train`，复用本资格报告记录的 calibrated root；任何 cache、能力、训练语义、C++ v3 或传感器
报告身份漂移都应先重新生成 cache、preflight 和校准证据，而不是绕过校验。

## 本次复审资格证据

- 功能源码提交：`e7c0c3b0c1563c99ec8d6b59454c42aa8e58aa98`；
- full cache：1734 场景，内部 manifest SHA-256
  `e8b2321507217fbb69a30ba7948d18a8d7309eb52ed13ef8e3ecbdedeca5d01c`；
- 正式传感器报告内部 SHA-256：
  `95ba584aa18b9f57ec087f295215ecc6c185dd0a3ad725ebb27ecbb525a9a1d4`；
- formal preflight 九项检查全部通过，报告内部 SHA-256：
  `14ab85bc7f977da962452bce663202320c1fcf49806e4d0a95c4326053ef9c1d`；
- 校准选择 24 workers、micro-batch 4，`global_step=0`。

所有路径、外部文件 SHA-256、分场景数量、Release 结果和复现命令统一记录在
`docs/validation/2026-08-08-formal-training-environment-qualification.md`。

## 文档权威边界

不同文档分别回答不同问题，不使用一个含混的全局先后顺序：

- 规划能力与飞跃式无累计燃料接口以
  `docs/superpowers/specs/2026-08-08-formal-capability-and-hopper-no-fuel-budget-design.md` 为准；
- 探索观测、theta 和 reward 语义以
  `docs/superpowers/specs/2026-08-08-sensor-observation-capability-design.md` 为准；
- 能否启动正式训练以本复审为准，`docs/migration/volume-3-pretraining-readiness.md` 记录同一状态；
- 自动化数值和性能证据以 `docs/validation/sensor-observation-capability-qualification.md` 为准。

2026-08-07 及更早的外部 capability closure、实时/累计燃料和旧 Volume 3 规划表述只保留为历史
证据；与上述文档冲突时一律不具有现行操作效力。
