# Volume 3 月球极区 PPO 训练前状态交接

状态：`formal-training-ready / training-not-started`

## 当前结论

训练代码已具备 30 m/360° 保守观测、truth/observed 隔离、三平台宏步闭环、平台正确的
`theta` 语义和 Release 性能前置门。正式 seed 4080 rollout 尚未启动，也没有生成 ONNX、
TensorRT engine、四文件模型候选、`model-package-ready` 或 AGX 标签。

仓库内已批准的 capability v2 就是正式运动能力源，不再等待外部
`lunar-training-capability-freeze/v1`，也不要求用 URDF/mesh 重复闭合训练能力。正式入口直接
校验 `three_platform_capability_freeze_v1.yaml` 的规范化摘要并转换为当前 C++ v3 planner 的
typed capability。URDF/mesh 继续服务于仿真外观、碰撞代理和设备集成，但不是 PPO 训练门。

飞跃式能力只保留每次独立规划的固定单跳 `delta-v` 包线。参考质量、比冲和参考推进剂用于
计算该包线，不是 episode 燃料状态；训练请求、观测、奖励和终止条件均不累计燃料，也不会因
先前跳跃次数降低下一跳可达性。

2026-08-08 最终训练前复审确认：正式 full cache、统一环境 builder、公开
`calibrate/train/resume/evaluate`、非 proxy 评估与进程级 formal preflight 已闭合。当前主机已
冻结 `24 workers（WHEELED/LEGGED/HOPPER 各 8）+ micro-batch 4`；校准 manifest 的
`global_step=0`，没有正式 rollout、checkpoint 或参数更新。完整证据见
`docs/validation/2026-08-08-formal-training-environment-qualification.md`。

## 现行权威覆盖规则

自 2026-08-08 起，本交接中的规划与平台能力按以下优先级解释：

1. `integration` 当前 C++ v3 分层全局规划、有限地图完整搜索、局部物理认证、滚动协调和
   ROS Action 行为是唯一路径规划权威。
2. `platform-control-capability-source/v2` 及
   `three_platform_capability_freeze_v1.yaml` 是当前轮式、足式、飞跃式批准工程能力权威；
   规范化 SHA-256 为
   `60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95`。
3. 传感器训练语义是三平台共享 `30 m/360°`，语义版本
   `lunar-training-semantics/sensor-30m-360-theta-mask/v1`，SHA-256 为
   `739ce3e1f6eaab4ee44a0136ff848f246d158a15ebb7f8b6c8b2c7ec1bca8cb5`。

旧 Volume 3 的路径算法、平台代理、运动能力字段、固定 search limit、80 m 观测默认值和
旧 checkpoint v3 身份均已过时。它们只允许作为历史迁移证据或显式 development-smoke
fixture，不得进入现行规划请求、正式 cache、性能报告、训练 manifest、resume 或模型发布。
从旧分支合入的 policy、数据、PPO、checkpoint 编排和 C++ bridge 代码只有在适配上述现行
权威后才继续有效；一次 Git 合并不会把旧规划语义恢复为权威。

## 仍有效的数据与模型合同

NASA/JAXA 极区数据 source/split 锁仍然有效，因为它们固定的是数据身份和划分，而不是路径
算法或平台运动能力。seed `4080` 的 split 为：NASA train 192、validation 48、test 48；
JAXA `CR1`、`GR1`、`GR2`、`LP1`、`MP1`、`MP2` 六站点只进入 holdout。内部
`split_sha256` 为
`6da68f54d7349142f787d0e82e3cc08b44770f99ae6deddb9f98b7cc5bfcbd18`；现行外部文件是
`polar_split_v2.json`，文件 SHA-256 为
`4434b342dddd184ac7ce2cd5b5c2e5e199ef28247f2602c7d9cfa5dfc51e1e2d`。v1 只保留为历史
划分证据，正式 cache 不读取它。

当前 `ObservationContractV2` 保持七输入：

```text
prior_channels    float32 [B,4,256,256]
coverage_summary  float32 [B,3,256,256]
local_crop        float32 [B,4,32,32]
frontier_features float32 [B,64,12]
pose_features     float32 [B,6]
candidate_mask    bool    [B,64]
platform_context  float32 [B,3]
```

`platform_context` 只表示平台类别，运动能力通过当前 capability v2 注入 C++ 规划器，不把
能力数值复制进网络输入。候选不足 64 项时只做零填充并置 mask；全 false 时绕过策略。

动作继续由 masked 64-way frontier 和候选条件下的连续 `theta` 组成。`theta` 是候选点处的
绝对 `map` 机体 yaw：轮式和足式参与训练，飞跃式 mask 掉该维的策略、熵和 PPO loss。
360° FOV 只让即时观测增益与 yaw 无关，不会抹掉地面平台的运动学代价。

checkpoint 已升级为 `lunar-ppo-checkpoint/v4`。`RunIdentity` 除 data、split、generator、
capability、reward 和源码 SHA-256 外，还绑定训练语义 SHA-256；旧 v3 checkpoint 只允许
显式 development-smoke 读取，正式 resume 必须拒绝。

## 正式环境与观测闭环资格

2026-08-08 在 Ubuntu 22.04 amd64 + ROS 2 Humble + RTX 4080 SUPER 主机完成以下现行验证：

- full cache 共 1734 个冻结场景：NASA train 1536、validation 96、test 96，另有 6 个 JAXA
  holdout；cache manifest 内部摘要为
  `e8b2321507217fbb69a30ba7948d18a8d7309eb52ed13ef8e3ecbdedeca5d01c`；
- 全局候选和网络状态使用 `256×256 @ 4.0 m`，局部规划/reveal 使用
  `320×320 @ 0.2 m`，网络局部裁剪为 `32×32 @ 0.2 m`；三者来自同一场景定义；
- 最新传感器报告中，候选 p95 为 `0.132299 ms`，30 m reveal p95 为 `1.381889 ms`，
  24-worker 观测吞吐降幅为 `6.899653%`，均通过冻结门限；
- formal preflight 的九项检查全部通过，包含三平台、同世界、4 m/0.2 m、无累计燃料、
  update-boundary resume 和非 proxy evaluation；
- 正式校准选择 24 workers 与 micro-batch 4，`global_step=0`。

完整主机信息、命令、性能 fixture 和正式/测试证据边界见
`docs/validation/sensor-observation-capability-qualification.md`。

## 正式训练启动边界

正式训练前的项目内准备已经完成。历史 Isaac/ROS `capability_provenance.json` 是
30 m/120° 旧 schema 证据，只保留为历史快照；正式工具不接收可替换项目权威的
`--capability-lock`。下一步只能由用户另行授权执行 seed 4080 的正式 `train`；启动时必须复用
已经校准的 full cache、传感器报告和 run root，不得重用旧 Volume 3 cache/checkpoint，也不得
回退到 proxy。

本交接不声称正式训练、完整模型评估、ONNX/TensorRT、AGX 或实际平台资格已经完成。
