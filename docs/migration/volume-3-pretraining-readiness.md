# Volume 3 月球极区 PPO 训练前就绪交接

状态：`sensor-closed-loop-ready / blocked-on-formal-capability-closure`

## 当前结论

训练代码已具备 30 m/360° 保守观测、truth/observed 隔离、三平台宏步闭环、平台正确的
`theta` 语义和 Release 性能前置门。正式 seed 4080 rollout 尚未启动，也没有生成 ONNX、
TensorRT engine、四文件模型候选、`model-package-ready` 或 AGX 标签。

当前唯一未闭合的训练输入门是外部 `lunar-training-capability-freeze/v1`：它必须同时封装
现行三平台能力 v2、共享 30 m/360° 观测文档以及 URDF/mesh 资源哈希。仓库内已批准的工程
能力值已经确定并通过 Ubuntu 规划资格，但不能用缺少外部资源 closure 的仓库配置文件或
测试夹具冒充正式训练 bundle。

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
`5d458081972e1ee767c5f91dd5cb42d519214a0111a6e28dfaa7283025ec99e2`。

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

## 新观测闭环资格

2026-08-08 在 Ubuntu 22.04 amd64 + ROS 2 Humble + RTX 4080 SUPER 主机完成以下验证：

- 当前规划器、bridge、ROS 的外置 Release 构建成功；`colcon test-result` 为 292 tests，
  0 errors，0 failures，0 skipped；
- model contract、训练和性能 Python 回归为 622 passed，7 skipped；
- 原生可见性候选 p95 为 0.124662 ms，30 m reveal p95 为 1.334282 ms；
- 24-worker 当前规划器/schema-valid v2 测试 closure 回归的观测吞吐降幅为 4.191261%，
  低于 10% 门限；该 fixture 不具正式授权能力；
- 仓库边界专项 14 passed，能力 v2 冻结校验通过。

完整主机信息、命令、性能 fixture 和正式/测试证据边界见
`docs/validation/sensor-observation-capability-qualification.md`。

## 正式能力门与解锁顺序

当前没有满足新语义的外部正式能力 lock，所以没有正式
`sensor-observation-performance/v1` 路径或 SHA-256。历史 Isaac/ROS
`capability_provenance.json` 是 30 m/120° 旧 schema 证据，输入正式工具时会在 benchmark
前被拒绝，且不会创建报告。

唯一解锁顺序是：

1. 外部平台资料提供方把当前三平台 capability v2、30 m/360° observation、URDF 和 mesh
   组成完整 `lunar-training-capability-freeze/v1`，并通过内容一致性和 SHA-256 校验。
2. 用该 lock 在当前传感器源码提交和同一 Release 主机运行 24-worker 性能工具，生成并验证
   `sensor-observation-performance/v1`。
3. 用同一 bundle 重新生成三平台 traversability/candidate cache，并重新冻结 worker 与
   micro-batch；旧 Volume 3 cache/checkpoint 不得沿用。
4. 对 `train/resume/evaluate` 重新执行 formal preflight，之后才允许启动一个正式 seed
   4080 的可暂停训练。

正式训练、完整评估、ONNX/TensorRT、AGX 和实际平台资格仍分别受后续门控制；本交接不会
把开发态或 Ubuntu 仿真结果升级成这些状态。
