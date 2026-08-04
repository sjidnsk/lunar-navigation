# Volume 3 Task 4 实施报告

## 状态与边界

- 起点：`e8ce6b84fb32f9fd2c0c71e7689773ef5b185dbf`
- 分支：`volume-3-policy-pipeline`
- 实现范围仅限 Task 4：共享 v3 reward、确定性 proxy curriculum、逐平台 95% gate、评估报告、`calibrate|train|resume|evaluate` 生命周期，以及 Task 5 正式训练所需的 checkpoint/课程编排合同。
- 未启动 24 小时正式训练，未导出 ONNX，未执行 AGX/runtime 工作。所有测试 checkpoint、manifest、pycache 和运行 artifact 均在仓库外。

## TDD RED / GREEN 记录

测试使用固定 venv `/home/kai/CodexDownloads/lunar_navigation/volume3/venv`；涉及 v3 的路径先 source ROS 2 Humble 与 Task 2 外置 bridge install。

1. 首轮四个计划测试模块在收集阶段得到 `4 errors`：`reward`、`curriculum`、`evaluation.release_gate` 和 `evaluation.report` 尚不存在。实现最小 API 后，Task 4 计划测试得到 `22 passed, 1 deselected`。
2. 真实 proxy 评估测试先因 `proxy_scenario` 与 `evaluate_proxy_policy` 缺失失败；加入固定 scenario/action→GoalRegion/C++ v3/reference execution 路径后得到 `8 passed, 1 deselected`。
3. 收敛语义测试先发现 action 会随 step 旋转且单步错误完成；改成固定 action→固定目标、两个有效 frontier 加一个零收益 distractor 后，重复 action 零收益、两个唯一目标达到 100% 覆盖，相关测试 `4 passed`。
4. CLI lifecycle 测试先因 Task 4 manifest freeze、权威 `checkpoints/` 路径和 phase API 缺失失败；补齐后对应焦点测试通过。
5. terminal auto-reset 测试先看到终止 step 返回 100% observation，不能直接继续训练；默认训练模式改为 worker 内自动重建真实 v3 episode 后通过。
6. 评估终局 observation 测试先以 `auto_reset` 参数缺失失败；新增显式 `auto_reset=False` 后，评估直接读取终局 100% coverage，训练默认仍返回新回合 5% observation。
7. 单平台 warmup 的真实 CUDA checkpoint 首次失败为 `checkpoint worker allocation is invalid`；checkpoint 校验改为只接受合法非空平台子集且总 worker 仍必须为 18 或 24，checkpoint 回归 `12 passed`，CUDA 中断恢复随后通过。
8. 无 `--max-updates` 的课程编排测试验证一次调用实际依次进入 `warmup_wheeled → warmup_legged → warmup_hopper → joint` 并耗尽同一累计预算，未在 phase 边界停滞。

## 实现结果

- `reward.py`
  - 三平台共用一个 `RewardWeights` 与稳定 SHA-256。
  - 奖励包含有效覆盖增益、目标进展、规划成本、模拟时间、重复访问及批准的 planner outcome 惩罚。
  - invalid/stale/numerical/unexpected canceled/C++ exception/non-finite transition 抛 `InvalidTransition`，不会进入 rollout。
- `curriculum.py` 与 `proxy_scenario.py`
  - 固定 seed `4080`，reward calibration seeds `4081/4082/4083`。
  - 冻结最多 2h calibration、按 W/L/H 各最多 2h warmup、至少 16h joint；早停节余只进入 joint。
  - joint 按校准结果分配 18→6+6+6 或 24→8+8+8；warmup 将全部 worker 分给当前平台。
  - scenario 固定 platform、proxy capability、terrain、seed；每步构造有效 `TrainingPlanRequest`，PPO action 映射 `GoalRegion`，通过 `ParallelEnvPool → create_v3_environment → C++ v3` 执行 reference 并生成变化 observation。
- `evaluation/`
  - 报告 schema 为 `lunar-policy-release-evaluation/v1`，固定 seeds 和 canonical hash。
  - PPO、`nearest_frontier`、`gain_over_cost_frontier` 使用同一 scenario/v3 路径；baseline 只作诊断。
  - candidate/release gate 对 WHEELED、LEGGED、HOPPER 分别判定，不使用均值掩盖失败；失败名精确到平台与规则。
  - candidate 排序先最大化最低平台覆盖，再比较失败率和完成时间。
- CLI 与恢复合同
  - `calibrate` 实际执行 18/24 CUDA 校准及三个 reward seed 评估，并在同一 run manifest 冻结 reward hash、schedule identity、seed 和累计预算。
  - `train` 消费已校准 root；`resume --checkpoint` 只接受 `<artifact-root>/checkpoints/` 下显式 checkpoint；`evaluate` 输出到 `<artifact-root>/evaluation/` 并复用同一预算。
  - 正式 `train|resume` 默认无 update 数上限；测试才使用显式 `--max-updates`。
  - `latest.pt` 每 30 分钟完整 PPO update 边界原子更新；joint candidate 使用 `candidate-step-<n>.pt`。
  - 600 秒完整 update reserve 使 phase 边界不会启动可能越界的单元；resume 根据冻结 calibration end 与累计 GPU 秒继续下一阶段。

## 最终验证

```text
Task 4 focused CPU:
50 passed, 1 deselected in 16.53s

terminal observation / curriculum focused:
3 passed in 9.47s

RTX 4080 SUPER public CUDA calibrate + interrupt/resume:
1 passed in 23.97s

RTX 4080 SUPER deterministic CUDA evaluation:
1 passed, 4 deselected in 4.90s

training/lunar_policy_training/tests（含 CUDA）：
213 passed, 1 skipped in 45.69s

model_contract/tests:
7 passed

tests/foundation:
113 passed

python tools/check_repository_boundaries.py .:
repository boundaries: OK
```

最终静态检查包括 UTF-8 读取、`git diff --check`、无跟踪 artifact、无残留 `lunar-env` worker；均通过。正式 24 小时 seed 仍是 Task 5，Task 4 没有生成可发布模型。

## Fix round 1（2026-08-04）

本轮只处理 scoped review 的三个 Important finding：proxy reference 执行、完整冻结场景日程、执行事件指标；未调整 PPO 架构、gate 阈值、训练预算或 Task 5 边界。

### RED 证据

- 真实 C++ reference 聚焦测试首次为 `4 failed`：有效 wheel/legged 结果没有 reference-consumption 事件；空 reference 仍错误获得 `0.475` coverage；HOPPER 直接返回 `GROUND_HOLD`，没有 commitment 生命周期。
- 冻结评估日程测试首次为 `1 failed`：`CurriculumSchedule` 没有公开完整 evaluation scenario indices。
- 完整 seeds 的真实评估测试首次为 `1 failed`：WHEELED 实际只有 `(10000,)`，而冻结日程要求 `(10000, 10001, 10002)`；其他平台同样仅执行 index 0。
- 事件聚合/gate 测试首次在收集阶段为 `2 errors`：`_ScenarioEvidence` 与 `_aggregate_platform_metrics` 尚不存在，证明原报告指标没有事件聚合入口。

### 最小修复

- proxy executor 现在验证并消费 C++ `MotionReference`：
  - WHEELED/LEGGED 校验平台、trajectory semantics、plan id、至少两个轨迹点、有限且单调的时间、起点、地图边界和实际终点；使用实际末端位置推进下一决策边界。
  - HOPPER 校验 `HopReference`、segment、launch、landing boundary、flight time、launch velocity 和 tube radius；按实际落区中心推进，并记录 `JUMP_COMMITTED → IN_FLIGHT → LANDED_HOLD`。
  - 不可执行 reference 保持原位置、coverage/progress 均为零，记录 execution failure 及对应 safety/platform/commitment 事件，不再给合成成功奖励。
- `ExecutionEvents` 随 `PlannerTransition` 穿过 worker queue 到 `ParallelRolloutStep`；评估读取真实 action、planner、reference executor 与 HOPPER lifecycle 事件。
- 评估使用 9 个 worker 一次覆盖三平台 × scenario index `0/1/2`；PPO、nearest 与 gain/cost 使用完全相同的 row schedule。报告持久化全部实际 seeds：WHEELED `10000..10002`、LEGGED `11000..11002`、HOPPER `12000..12002`。
- coverage/finite/safe/deterministic/planner failure 以 scenario 或实际 step 为分母聚合；四个 violation 指标求真实事件计数，不再在报告构造处写死为零。HOPPER 接受 reference 但未观察到完整 commitment 序列时额外计一次 violation。

### Fix round 1 验证

```text
指定三个 Task 4 测试文件（CPU）：
22 passed, 1 deselected in 13.40s

v3 environment / hopper / parallel pool 回归：
30 passed in 4.93s

training/lunar_policy_training/tests（含 CUDA）：
220 passed, 1 skipped in 54.25s

独立 CUDA 短 smoke：
3 passed, 218 deselected in 33.98s

model_contract/tests + tests/foundation：
120 passed in 0.62s

python tools/check_repository_boundaries.py .：
repository boundaries: OK

git diff --check：
通过
```

上述仍是 `proxy: true` 的 Task 4 证据，不替代真实平台或 AGX 验收；未启动 Task 5 正式 24 小时训练。
