# Volume 3 月球极区 PPO 训练前状态交接

状态：`formal-training-qualified / runtime-state-external`

## 当前结论

项目内正式训练输入和运行链已经闭合。seed 4080 是否已开始、当前步数和预算只以仓库外
calibration manifest 为准；本交接不复制易漂移的实时状态。训练必须使用仓库内 capability v2、
当前 C++ v3 planner、30 m/360° 观测、formal cache v3、ObservationContractV3 和 checkpoint v6；早期 Volume 3
规划算法、平台代理、燃料库存、固定 8-step 和旧 checkpoint/cache 不得进入 formal 命令。

飞跃式每次规划重新由正式能力换算相同的单跳可用 delta-v；参考推进剂不是可累计资源。PPO
episode 没有固定决策次数，`rollout_horizon` 仅定义一次更新收集多少 transition。未完成 episode
可跨多个 update、保存和恢复继续同一 scene/start/累计观测。

## 现行合同

七项网络输入为：

```text
prior_channels    float32 [B,4,256,256]
coverage_summary  float32 [B,3,256,256]
local_crop        float32 [B,4,32,32]
frontier_features float32 [B,64,12]
pose_features     float32 [B,5]
candidate_mask    bool    [B,64]
platform_context  float32 [B,3]
```

`pose_features` 依次为归一化 x/y、sin yaw、cos yaw、实际 mission observed ratio，不含剩余决策
预算。动作仍是 masked frontier 与 theta；轮式/足式 theta 是候选处绝对 map yaw，飞跃式 theta
在策略、熵和 loss 中 mask。

正式 checkpoint 是 `lunar-ppo-checkpoint/v6`，同时保存 model、optimizer、scheduler、RNG 和
`lunar-formal-environment-state/v2` 活动 episode。恢复后的 update 2 必须与连续 update 2 逐项
完全一致；V5 及以前不能恢复为正式 V3 训练。

## 正式场景与任务域

source/split v2 的物理 inventory 共 1734：train 1536、validation 96、test 96、JAXA holdout 6。
三平台 capability v2 都能确定合格起点的共同可行动子集共 1666：train 1475、validation 90、
test 95、holdout 6。其余 68 个均因 HOPPER 无合格起点而处于三平台任务域外；精确 ID 和原因
由 cache manifest 冻结，不能计入成功/失败分母。

训练按 fixed seeded permutation 一轮不放回后循环，对应平台 lane 使用同一物理场景；评估对
每个 eligible scene 恰好一次并按 validation/test/holdout × 三平台分别门控。成功只认
controller 的 95% ROI 首次跨阈值事件。

正式 full cache：

```text
/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/de51731/cache/cache-manifest.json
```

schema 为 `lunar-formal-training-cache/v3`，内容 SHA-256 为
`44ab21c19c4c22b7c6feb9123d7ed88f02b5f3e282d71fe5f0f774ee4090dacc`。

## 训练前冻结配置

正式资格产物位于：

```text
/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/qualified-current/
```

正式选择由同一 root 的 `runtime_calibration.selected_workers`、`selected_micro_batch` 和
`selected_rollout_horizon` 唯一给出；配置中的 32 只是 bootstrap，不能覆盖校准结果。formal
seed 为 4080，episode decision limit 为 none。formal preflight 必须消费同一 calibration root，
用真实 V6 文件证明 update-2 resume 等价，并在生成时保持 `training_started=false`、
`global_step=0`；训练启动后实时状态仍由 manifest 接管。

## 后续正式训练计划

累计 GPU 工作上限为 86400 秒：校准最多 2 小时；轮式、足式、飞跃式各最多 2 小时预热；联合
训练至少 16 小时。每 30 分钟保存 `latest.pt`，联合阶段每小时保存候选。

只有某个冻结 checkpoint 在 validation、test、holdout 的每个 split×platform 上分别达到覆盖
成功率不低于 0.95，并通过现行发布 gate，才记为 `release-gate-passed` 并称本轮训练成功完成。
若累计预算耗尽仍无通过候选，必须记为 `not-converged`；这只是训练运行结束，不允许进入正式
模型包导出。

本交接只声明训练前资格，不声称模型已收敛，也不声称 ONNX/TensorRT、AGX 或实际平台资格完成。
