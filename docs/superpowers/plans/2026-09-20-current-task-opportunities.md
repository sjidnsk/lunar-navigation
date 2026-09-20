# Current Task Opportunities Implementation Plan

**Goal:** 落实已确认的当前观测机会语义，移除未来需求及在线覆盖下界，并完成固定失败场景验证。

**Spec:** ../specs/2026-09-20-current-task-opportunities.md

**Architecture:** TaskAnalyzer 只以当前 R 的射线产生前沿/见证；GraphBuilder 复用；训练和 Runtime 共用耗尽；真值只评估。

## Global Constraints

复用 test/drl-coverage-completion；不合并、不推送、不干扰正式训练。真值 E/K、网络结构、奖励/折扣/熵、导航与控制保持不变。使用现有隔离 Jazzy 构建和训练 venv；临时产物写 ~/.cache/lunar-drl-coverage-completion/，使用 current-* 文件名。

## Review Focus

区外远起点不能立即结束；区外直接观测与未知未来绕行必须区分；M 不是 B；八方向扫描及累计观测不能丢失；空图不能冒充完成；移除下界参数必须覆盖 CLI、训练、Runtime、评估和恢复；历史失败记录不得改写。

### Task 1: 当前观测机会与统一报告

**Files:** task_analysis.py、contracts.py、remaining.py、graph.py、test_task_analysis.py、test_coverage_completion.py、test_current_opportunities.py（均位于 ros2_ws/src/lunar_drl_exploration 对应源码或 test 目录）。

**Interfaces:** 消费现有 MeasuredWorkspace、direct_witnesses、first_pending_cells；产出原形状 frontier_cells/witnesses 及 available/exhausted/completed，不再提供面积界限。

1. 编写行为测试：先观察任务小区，任务另一未知区只能经远处未知区外通道到达，此时当前无直接任务机会则耗尽；区外远起点未开始任务观测则不完成；已知区外站位仍能支持任务观测。运行测试观察旧实现失败。
2. 用 `pending = task_mask & ~known` 的当前边界和当前 `reachable` 做范围、光学必要支持及原生见证查询；按 source 批量提取首个未测量接口，去重。删除 potential/exterior/future-stance 处理。GraphBuilder 对基础点尚未表示的接口从超过到达容差的已知 R 中选补点，保持既有选点/连边/覆盖要求。
3. `completed = available and exhausted`；`exhausted = not witnesses and known_area_m2 > 0`，无任务观测时保持可决策。删除 remaining.py。替换旧上界测试为当前机会的独立小图观测对照。
4. 运行新旧相关行为测试，期望全部通过；记录有意改变的旧测试语义。

### Task 2: 消费端与恢复语义

**Files:** decision.py、config.py、cli.py、runtime.py、ros_env.py、evaluation.py、paired_experiment.py 及对应测试。

**Interfaces:** 消费 Task 1 的报告；输出完成状态、真值覆盖统计、带新终止语义的完整恢复身份。

1. 删除覆盖下界参数和输出。Runtime 正常结束为 EXHAUSTED，真值评估继续独立记录 reached_80/reached_99。
2. 完整恢复身份使用 `current_reachable_task_opportunities_v1`，Actor-only 结构兼容不变。用真实配置/恢复校验测试证明旧终止身份不能静默恢复。
3. 运行整个 lunar_drl_exploration/test，期望通过；构建受影响 Python 包并验证安装版本。

### Task 3: 固定场景验证与文档

**Files:** scripts/drl/audit_completion.py、scripts/drl/replay_completion.py、docs/validation/2026-09-20-current-task-opportunities.md、docs/DRL开发入口.md。

**Interfaces:** 消费当前报告及旧 frozen NPZ/goal JSON；产出新文件名的快照/闭环结果，无学习写入。

1. 审计脚本移除面积界限断言，继续独立统计真值覆盖/剩余；修正审计真值缓存需包含 sensor 配置。
2. 在固定 8 快照和已保存月表/洞穴历史目标上验证，报告真实覆盖和耗尽分别为何；不得把重放称为新策略效果。
3. 自检所有语义和测试结果，完成独立代码审查；记录未验证边界。提交当前任务的显式文件路径，保留试验分支，不合并。

## Commands

```bash
bash /home/kai/.cache/lunar-drl-coverage-completion/run.sh -m pytest ros2_ws/src/lunar_drl_exploration/test -q
bash /home/kai/.cache/lunar-drl-coverage-completion/build.sh
bash /home/kai/.cache/lunar-drl-coverage-completion/run.sh scripts/drl/audit_completion.py --snapshot-dir /home/kai/.cache/lunar-drl-completion-audit --output /home/kai/.cache/lunar-drl-coverage-completion/current-snapshot-results.json
bash /home/kai/.cache/lunar-drl-coverage-completion/run.sh scripts/drl/replay_completion.py --config config/drl_exploration.yaml --recorded-dir /home/kai/.cache/lunar-drl-completion-audit --output /home/kai/.cache/lunar-drl-coverage-completion/current-closed-loop-results.json
```
