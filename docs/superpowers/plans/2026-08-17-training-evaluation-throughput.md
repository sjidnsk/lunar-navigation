# Reward V4 Training Evaluation Throughput Implementation Plan

> Execute with test-driven development and verification before completion.

**Goal:** 将频繁评估改为可恢复的单宏动作哨兵，低频保留完整自然终态评估，并从现有 sealed candidate 原子恢复训练。

**Constraints:** 不改冻结 Reward V4 配置、奖励、模型、候选、规划器、训练宏动作或课程/最佳 checkpoint 的完整评估语义；生成 artifact 留在仓库外；只做必要定向验证。

## Task 1: Freeze evaluation scheduling

**Files:**
- Add: `training/lunar_policy_training/lunar_policy_training/evaluation/reward_v4_schedule.py`
- Test: `training/lunar_policy_training/tests/test_reward_v4_schedule.py`

- [ ] RED: 10,800 秒边界选择 sentinel，跨越 43,200 秒边界选择 full；模式文件与 candidate payload 绑定且可幂等恢复。
- [ ] GREEN: 实现纯调度和原子模式 artifact，不改变 frozen config。

## Task 2: Bound sentinel evaluation and resume per task

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/evaluation/reward_v4_runtime.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/reward_evaluation.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/evaluation/report.py`
- Test: `training/lunar_policy_training/tests/test_reward_v4_runtime_evaluation.py`

- [ ] RED: 单宏动作 cap 会完成非终态任务；自然终态模式不受影响。
- [ ] RED: 已完成任务原子持久化，恢复只运行缺失任务；损坏或错 checkpoint 的进度 fail closed。
- [ ] GREEN: 实现 cap、episode 严格解码和逐任务恢复。

## Task 3: Route sentinel and full boundaries

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Test: `training/lunar_policy_training/tests/test_cli.py`

- [ ] RED: sentinel 使用单 seed/四尺度/活跃平台、单宏动作和独立报告路径；通过时不改变课程或 best state，硬错误阻止应用。
- [ ] RED: full 保持原三 seed、自然终态，并继续驱动课程、checkpoint 选择和回滚。
- [ ] GREEN: 接入调度、进度目录和两类边界；兼容已存在 candidate update。

## Task 4: Add actionable timing evidence

**Files:**
- Modify only existing environment/collector metric types that own the measured stages.
- Test the corresponding focused metric serialization and stage boundary suites.

- [ ] RED: 宏动作诊断可分别报告局部规划调用数/累计耗时与候选刷新/全局搜索耗时，且旧 checkpoint/metrics 可读取。
- [ ] GREEN: 用 `perf_counter` 在现有阶段边界累加，不改变控制流和结果。

## Task 5: Verify and cut over

- [ ] Run the new schedule/runtime tests, focused CLI recovery tests, and metric serialization tests.
- [ ] Run one real eight-task sentinel smoke against a copied candidate checkpoint and confirm task-progress reuse.
- [ ] Run `git diff --check`, review the exact diff, and commit only task files.
- [ ] Stop the old controller only after the new controller is fully assembled.
- [ ] Preserve the old evaluation artifacts, resume from the existing candidate/applied checkpoint, and confirm the sealed update applies without recollection.
- [ ] Confirm the next update starts collecting and new evaluation/timing evidence is written.
