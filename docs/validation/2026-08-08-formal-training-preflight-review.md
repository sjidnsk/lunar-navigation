# 正式训练前逻辑与语义复审

日期：2026-08-08

## 结论

当前功能分支可以合入 `integration`，因为现行 C++ v3 规划、三平台 capability v2、
`30 m/360°` 观测闭环、平台动作语义和飞跃式无累计燃料合同已经一致实现并完成资格验证。

正式 PPO 训练仍不能启动。阻断原因不是能力资料、传感器标定、安全审计或通用代码质量，而是
正式训练自身的数据与运行链还没有闭合：正式 cache 没有生成/加载入口，公开 `calibrate` 不接受
formal 配置，公开 `train/resume` 没有构造正式环境工厂和正式观测模板，公开 `evaluate` 明确拒绝
formal 评估。因此本次合并状态定义为：

```text
integration-qualified / formal-training-blocked-on-environment-and-cache
```

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

## 尚未闭合的正式训练链

### 1. 正式 cache 没有可执行生产入口

文档要求按当前 capability v2、数据 source/split、观测语义和当前 C++ v3 规划器重新生成三平台
traversability/candidate cache，但仓库中还没有对应生成命令、manifest schema、正式加载器和
身份校验。旧 Volume 3 cache 不允许复用。

### 2. formal calibrate 不可执行

公开 `calibrate` 调用 `_calibrate_training_run()`，该函数只接受 `development-smoke`，formal
配置会以 `formal runtime calibration requires the future non-proxy environment` 拒绝。与此同时，
公开 `train` 又要求已存在且已冻结的 calibrated artifact root，形成无法通过公开命令闭合的前后
依赖。

### 3. formal train/resume 缺少环境绑定

`_start_training_run()` 和 `_resume_training_run()` 支持内部注入
`FrozenCapabilityEnvironmentFactory` 与 `PolicyBatch`，但公开 CLI 只传入 capability bundle。
`_run_updates()` 因而会拒绝缺失的正式环境工厂或正式观测模板。当前只有测试代码构造该工厂，
没有生产 builder 把正式数据窗口、当前 capability、传感器闭环、请求身份和执行器装配成 worker。

### 4. formal evaluate 明确未实现

`_evaluate_checkpoint()` 对 formal run 无条件返回
`formal evaluation environment is not configured yet`。因此即使绕过入口启动训练，也无法完成同一
正式环境上的评估闭环。

## 启动正式训练前必须满足的唯一后续门

下一阶段只实施以下直接闭环，不扩展范围：

1. 生成并冻结当前三平台正式 traversability/candidate cache，manifest 同时绑定 data、split、
   generator、capability、训练语义和当前规划器源码身份；
2. 提供唯一的正式环境 builder，并由公开 CLI 构造
   `FrozenCapabilityEnvironmentFactory` 和正式观测模板；
3. 让 `calibrate -> train -> checkpoint -> resume -> evaluate` 全部通过同一数据、世界、请求、
   capability 和传感器边界；
4. 增加一个不消耗正式 24 小时预算的进程级 formal preflight，证明公开命令不会落回 proxy、旧
   planner、旧 capability 或旧 cache；
5. 完成上述闭环后再冻结 worker/micro-batch，并启动 seed 4080。

在这五项完成前，不得把状态写成 `formal-training-ready`，也不得启动正式 rollout。

## 本次复审资格证据

- 源码提交：`c8cf14e167345150ba64dd141cfec5470e09ba41`；
- 干净 Release 构建：8 packages；
- ROS/C++：36 tests，0 errors，0 failures，0 skipped；
- Python 契约、训练、差分与性能：645 passed，16 skipped；
- 正式 24-worker 性能报告：
  `/home/kai/CodexDownloads/lunar_navigation/formal_training_preflight/c8cf14e/sensor-performance.json`；
- 报告内部 SHA-256：`201f872a3b67e8fc24b5d8c5fc1c24ff97f1515e6918b31de975fec861d7ebcf`；
- JSON 文件 SHA-256：`2f05a5f6d46a051801aa61a19ea3befc1ddb8925429bb5a8bccc111a97a4862d`；
- 候选 p95 `0.126391 ms`，reveal p95 `1.328467 ms`，24-worker 吞吐降幅
  `2.808589%`，全部通过冻结门限。

## 文档权威顺序

现行解释按以下顺序读取：

1. `docs/superpowers/specs/2026-08-08-formal-capability-and-hopper-no-fuel-budget-design.md`；
2. `docs/superpowers/specs/2026-08-08-sensor-observation-capability-design.md`；
3. `docs/migration/volume-3-pretraining-readiness.md`；
4. `docs/validation/sensor-observation-capability-qualification.md`；
5. 本复审。

2026-08-07 及更早的外部 capability closure、实时/累计燃料和旧 Volume 3 规划表述只保留为历史
证据；与上述文档冲突时一律不具有现行操作效力。
