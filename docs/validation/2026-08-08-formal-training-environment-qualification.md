# 月球极区正式训练环境资格报告

日期：2026-08-09

状态：`formal-training-qualified / runtime-state-external`

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
├── calibration/metrics/train.jsonl
├── calibration/evaluation/candidate-step-*/
└── preflight/
    ├── formal-preflight.json
    └── resume-equivalence/update-1.pt
```

这些 JSON/checkpoint 内部记录完整 source commit、内容摘要和当前能力/训练语义身份，是精确数值
证据；本文不复制会因重新资格运行而漂移的文件哈希或选择值。资格运行必须在干净提交上生成，
正式 `train` 只能消费同一 `calibration` root。preflight 完成时 manifest 的 `global_step` 必须为
0；训练启动后的实时 phase、global step、预算和 checkpoint 只以该外部 manifest 为准。

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
优先较短 update。配置中的 horizon 32 只是 bootstrap；当前 source commit 的唯一正式选择必须
从 `calibration/run-manifest.json` 的以下字段读取：

```text
runtime_calibration.selected_workers
runtime_calibration.selected_micro_batch
runtime_calibration.selected_rollout_horizon
run_identity.formal_seed
global_step
```

`episode_decision_limit` 必须为 `none`；任何文档复制值都不能覆盖 manifest。

## Step 118 阶段恢复复审

正式运行在 `global_step=118`、`7269.1511032514145` 累计 GPU seconds 的轮式预热安全边界
停止。原因是旧恢复逻辑把 `WHEELED×24` 活动 episode 注入下一阶段 `LEGGED×24`，worker
身份门正确拒绝。现行实现只在相邻课程阶段且 allocation 改变时退役旧平台 episode；模型、
optimizer、scheduler、normalization、RNG、global step 和预算仍从 v6 checkpoint 恢复。同阶段
同 allocation 的恢复仍要求完整活动 episode 精确一致。

训练实现提交 `6585c3d0bc72e5c6e5db0c464d4268f9699d2c70` 完成以下复审：

- 非 CUDA 训练测试：`732 passed, 1 skipped, 3 deselected`；
- CUDA 正式标记测试：`3 passed, 733 deselected`，包含真实中断后恢复回归；
- repository boundary：脚本 `OK`，foundation 测试 `14 passed`；
- step 118 生产 loader 预演推导出 `warmup_legged / LEGGED×24`，没有注入旧 episode，reset
  得到 24 条观测，snapshot 也是 24 个 `LEGGED` 状态；没有执行 rollout 或 optimizer update；
- 预演前后原 `latest.pt` 文件 SHA-256 均为
  `43f7b3717ccf264a505f8a5e0c871b14893dfce4efcde254142178fa9cdf01a0`。

修复后逐 update 指标写入 `metrics/train.jsonl`，历史 step 1--118 不伪造，第一条必须是 step
119。联合阶段首次在 joint 起点后 10800 累计 GPU seconds、之后按最近一次评估开始点每 10800
seconds 串行评估最新 immutable candidate；评估使用同一预算，单次保留 3600 seconds watchdog，
并保存 validation/test/holdout 独立门控证据。

由于严格 checkpoint 和传感器性能报告均绑定 source，恢复前必须运行受控迁移：保留 step 118
原文件字节级备份，生成只改变 `source_commit`/payload hash 的独立 resume 副本，同时验证新旧
传感器报告仍绑定同一 host、capability 和 training semantics。最终新旧提交、文件和 payload
hash、性能报告 hash、变更路径及预算由外部 manifest 的 `source_migrations` 精确记录。

## Formal preflight

preflight 必须在 calibration 之后运行并读取同一 calibrated root。报告至少证明：

- full cache、RunIdentity、三平台 worker 与同一物理场景闭合；
- 4 m global / 0.2 m local、多平台当前 capability 和飞跃式无累计燃料语义；
- seeded scene permutation 与报告 scenario seed 一一对应，不重复、不漏评；
- validation/test/holdout × 三种方法 × 三平台各完成一个真实宏步，共 27 条非 proxy probe
  evidence；该探针不计算发布成功率；
- 实际创建 V6 `update-1.pt`，保存/加载后 uninterrupted update 2 与 resumed update 2 的模型、
  optimizer、RNG、环境状态、observation、candidate 和首个 planner request 摘要完全相等；
- 报告选择字段与同一 calibration manifest 完全一致；
- `training_started=false` 只描述 preflight 报告生成时刻，不是启动后的实时状态。

公开正式 `evaluate` 仍对 validation 90、test 95、holdout 6 的每个 eligible scene 恰好执行一次，
运行到自然 terminal，并只使用 controller 的 `success_first_crossing` 计算成功率。

## 正式训练与完成判定

训练累计 GPU 上限为 86400 秒：校准最多 2 小时，三平台各最多 2 小时预热，联合训练至少
16 小时。每 30 分钟更新 `latest.pt`，联合阶段每小时保存候选；episode 可跨任意多个 update。
每个完成的 update 立即追加指标，因此健康趋势不再等到 30 分钟 checkpoint 才可见。

只有冻结 checkpoint 在 validation、test、holdout 的每个 split×platform 上分别达到覆盖成功率
不低于 0.95，并通过既定有限输出、动作/参考一致性及确定性门，才记为
`release-gate-passed`，表示本轮训练成功完成。达到 86400 GPU seconds 但未通过时只能记为
`not-converged`：运行结束但训练不成功，不得导出正式模型包。

本资格只表示训练入口可启动；是否正在训练及是否收敛由外部 manifest 与发布评估判定。它不代表
ONNX/TensorRT、AGX 或实际平台验收完成。

## 复现入口

执行顺序固定为：干净 Release 构建与全量测试 -> 传感器性能报告 -> 复用并验证 full cache ->
`calibrate` -> `formal-preflight --calibration-root ...` -> 生产 loader 重开全部证据。不得先用配置
文件的 bootstrap horizon 生成 preflight，也不得在 preflight 通过前运行 `train`。
