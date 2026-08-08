# 正式训练前逻辑与语义复审

日期：2026-08-09

状态：`formal-training-qualified / runtime-state-external`

## 复审结论

本复审只判断前后逻辑闭环、语义唯一、设计合理和文档一致，不扩展新的治理或设备验收要求。
现行链已经统一为：C++ v3 规划、capability v2、30 m/360° 实际观测、无累计燃料、
ObservationContractV3、无固定决策上限 episode、cache v3 和 checkpoint v6。旧 Volume 3 的
pose `[B,6]`、8-step episode、checkpoint v4、cache v1 及旧平台代理均为历史证据。

## 闭环核对

### 规划、能力和飞跃式

`three_platform_capability_freeze_v1.yaml` 是训练所用正式能力源，经 typed adapter 进入当前
C++ v3 planner。飞跃式每次动作提供独立单跳 delta-v 包线和安全凸落区；质量、比冲及参考
推进剂不构成 episode 库存，不会因先前跳跃减少后续训练可达性。

### 观测、动作和成功

30 m 是信息增益估算与真实 reveal 的几何上限，不会自动把圆内栅格设为已知。实际状态由
valid/age/quality/count 闭合；奖励读取真实新增覆盖。地面平台学习候选位置与绝对 map yaw，
飞跃式 theta 在策略和 loss 中 mask。

`ObservationBoundaryController` 是成功的唯一权威。只有累计 ROI 覆盖从 `<0.95` 首次跨到
`>=0.95` 才产生一次 `success_first_crossing`；评估不得从 float32 终值加容差反推成功。

### episode、rollout 与恢复

正式 episode 没有决策次数上限。`rollout_horizon` 只是 PPO update 批长；未完成 worker 跨
update 保持 scene/start/pose/累计观测，已自然结束的 worker 才单独 auto-reset。正式策略使用
随机采样动作，只有部署式 development smoke 使用 deterministic argmax。

preflight 必须真实保存 V6 update-1 checkpoint，并逐项证明连续 update 2 与恢复 update 2 的
model、optimizer、RNG、活动 episode、observation、candidate 和首 planner request 完全相等。

preflight 的 evaluation 检查只执行 validation/test/holdout × 三种方法 × 三平台各一个真实宏步，
形成 27 条非 proxy evidence；它验证链路，不计算发布成功率。正式 `evaluate` 才负责全部 eligible
scene 的自然终态评估与 split×platform 发布门。

### 场景调度与评估

1734 个冻结数据场景保留不变；1666 个三平台都有合格起点的共同可行动场景构成正式任务域。
训练以固定 seeded permutation 一轮不放回后循环；对应平台 lane 得到同一物理场景。正式评估
对 validation 90、test 95、holdout 6 每个 eligible scene 恰好一次。报告按 split×platform
分别门控，不能通过合并 191 个 NASA 与 6 个 JAXA 样本掩盖 holdout 失败。

### 校准与 preflight 顺序

配置文件的 `rollout_horizon=32` 只是 bootstrap。校准用相同 64 transitions/worker 比较
16/32/64；通过者中先取最高吞吐的 99% 近似等价集，再优先更短 update。formal preflight 必须
读取同一 calibration root，并逐项复用 manifest 选择；不允许把 bootstrap 32 或旧资格运行的
数值写成当前校准结论。

## 训练完成的唯一解释

- `formal-training-qualified`：训练前报告完成且当时 `global_step=0`；这是本静态文档状态。
- `training-running`：seed 4080 已产生参数更新，但没有冻结的通过候选。
- `release-gate-passed`：同一 checkpoint 在 validation/test/holdout 的每个 split×platform 均达到
  95% 成功率及既定一致性门；才表示本轮训练成功完成。
- `not-converged`：86400 GPU seconds 耗尽仍无通过候选；表示运行结束，不表示训练成功。

loss 下降、生成 checkpoint、完成预热或耗尽时间都不能单独称为训练完成。ONNX、TensorRT 与
AGX 是通过候选之后的独立阶段。

实时状态只读取仓库外 calibration `run-manifest.json`，不由本静态文档复制。

## 权威文件

- episode、V3、恢复、场景任务域和完成语义：
  `docs/superpowers/specs/2026-08-09-unbounded-formal-episode-design.md`；
- 平台能力与飞跃式无累计燃料：
  `docs/superpowers/specs/2026-08-08-formal-capability-and-hopper-no-fuel-budget-design.md`；
- 传感器与 theta：
  `docs/superpowers/specs/2026-08-08-sensor-observation-capability-design.md`；
- 数值和外部 artifact 入口：
  `docs/validation/2026-08-08-formal-training-environment-qualification.md`；
- 迁移交接：`docs/migration/volume-3-pretraining-readiness.md`。

上述 2026-08-09 文档和当前 `integration`/功能提交优先于更早 Volume 3 历史表述。
