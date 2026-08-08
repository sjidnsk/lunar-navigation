# 月球极区正式训练环境资格报告

日期：2026-08-09

状态：`formal-training-ready / training-not-started`

## 结论

正式训练前的数据、场景、三平台能力、观测、规划、无上限 episode、活动回合恢复和自然终态
评估入口已经闭环。当前网络合同是 `ObservationContractV3`，`pose_features [B,5]`；活动
checkpoint 是 `lunar-ppo-checkpoint/v6`。`rollout_horizon` 仅定义 PPO update 的 transition
批长，不结束 episode，也不限制探索决策次数。

资格产物统一写在仓库外：

```text
/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/qualified-current/
├── sensor-performance.json
├── calibration/run-manifest.json
└── preflight/
    ├── formal-preflight.json
    └── resume-equivalence/update-1.pt
```

这些 JSON/checkpoint 内部记录完整 source commit、内容摘要和当前能力/训练语义身份，是精确数值
证据；本文不复制会因重新资格运行而漂移的文件哈希。资格运行必须在干净提交上生成，正式
`train` 只能消费同一 `calibration` root。当前没有执行 seed 4080 策略更新，校准 manifest 的
`global_step` 必须为 0。

## 唯一现行数据流

```text
NASA/JAXA source lock + split v2
  -> 1734 个冻结物理场景
  -> capability-v2 三平台共同起点资格
  -> 1666 个共同可行动任务场景
  -> 256x256 @ 4.0 m 全局真值/静态投影
  -> 按需 320x320 @ 0.2 m 局部瓦片
  -> 30 m/360 deg 实际 reveal
  -> observed-only 候选与网络观测
  -> 当前 C++ v3 planner 与参考执行
  -> 自然 terminal 或继续同一 episode
```

全局多分辨率仍由通用尺度 `[1,2,4,8,16,20] * 0.2 m` 选择；1024 m 窗使用
`4.0 m` 全局层。局部规划、reveal 和局部认证保持 `0.2 m`，没有因全局尺度降采样。

## 场景、起点与评估分母

正式 cache：

```text
/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/de51731/cache/cache-manifest.json
```

冻结事实：

- schema：`lunar-formal-training-cache/v3`；
- 物理场景 1734：train 1536、validation 96、test 96、JAXA holdout 6；
- 共同可行动场景 1666：train 1475、validation 90、test 95、holdout 6；
- 排除 68：train 61、validation 6、test 1，均为 HOPPER 没有合格起点；
- cache manifest 内容 SHA-256：
  `44ab21c19c4c22b7c6feb9123d7ed88f02b5f3e282d71fe5f0f774ee4090dacc`；
- NASA 各 split 的共同资格比例必须至少 90%，holdout 必须 6/6；
- 精确排除 ID、三平台起点和缺失平台由 manifest 内容摘要冻结。

1734 是数据 inventory，1666 是三平台共享任务域。排除场景既不计成功也不计失败，不能进入
评估分母；这不是训练后筛样本。训练按固定 seeded permutation 一轮不放回、轮末循环；正式
评估对每个 eligible scene 恰好执行一次，并按 validation/test/holdout × 三平台分别门控。

## 传感器与校准

正式传感器报告必须同时证明：4 m 候选信息增益 p95 不高于 5 ms、0.2 m reveal p95 不高于
2 ms、24-worker 观测吞吐降幅不高于 10%，并绑定当前 RTX 4080 SUPER、Release native
benchmark、capability v2、训练语义和干净 source commit。

runtime calibration 固定比较 18/24 workers、micro-batch 候选和 horizon 16/32/64。三个
horizon 使用每 worker 64 transitions 的等工作量，并验证有限 advantage/return、无 OOM/IPC/
planner timeout、跨 update 连续和 resume digest。吞吐达到最佳值 99% 的候选视为近似等价，
优先较短 update；当前资格选择为：

```text
workers = 24
WHEELED/LEGGED/HOPPER = 8/8/8
micro_batch = 4
rollout_horizon = 16
formal_seed = 4080
global_step = 0
episode_decision_limit = none
```

## Formal preflight

preflight 必须在 calibration 之后运行并读取同一 calibrated root。报告至少证明：

- full cache、RunIdentity、三平台 worker 与同一物理场景闭合；
- 4 m global / 0.2 m local、多平台当前 capability 和飞跃式无累计燃料语义；
- seeded scene permutation 与报告 scenario seed 一一对应，不重复、不漏评；
- 非 proxy 自然终态 evaluation 使用 controller 的 `success_first_crossing`；
- 实际创建 V6 `update-1.pt`，保存/加载后 uninterrupted update 2 与 resumed update 2 的模型、
  optimizer、RNG、环境状态、observation、candidate 和首个 planner request 摘要完全相等；
- 报告选择与 calibration 同为 24 workers、micro-batch 4、horizon 16；
- `training_started=false`。

## 正式训练与完成判定

训练累计 GPU 上限为 86400 秒：校准最多 2 小时，三平台各最多 2 小时预热，联合训练至少
16 小时。每 30 分钟更新 `latest.pt`，联合阶段每小时保存候选；episode 可跨任意多个 update。

只有冻结 checkpoint 在 validation、test、holdout 的每个 split×platform 上分别达到覆盖成功率
不低于 0.95，并通过既定有限输出、动作/参考一致性及确定性门，才记为
`release-gate-passed`，表示本轮训练成功完成。达到 86400 GPU seconds 但未通过时只能记为
`not-converged`：运行结束但训练不成功，不得导出正式模型包。

本资格不代表训练、ONNX/TensorRT、AGX 或实际平台验收完成。

## 复现入口

执行顺序固定为：干净 Release 构建与全量测试 -> 传感器性能报告 -> 复用并验证 full cache ->
`calibrate` -> `formal-preflight --calibration-root ...` -> 生产 loader 重开全部证据。不得先用配置
文件的 bootstrap horizon 生成 preflight，也不得在 preflight 通过前运行 `train`。
