# Ground Planning Throughput Recovery Implementation Plan

> Execute with test-driven development and verification before completion.

**Goal:** 修复地面滚动续段误失败，合并同一 0.2 m cell 的重复观测，然后从 update 26 原子恢复正式训练。

**Constraints:** 不改规划器、候选、奖励、HOPPER、局部 horizon；不混用旧 update 27 的部分采样；所有生成运行产物保留在仓库外。

## Task 1: Remove the redundant continuation qualification probe

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Test: `training/lunar_policy_training/tests/test_formal_builder.py`

- [x] Add a regression that enters a locked ground continuation and makes `_qualify_ground_start` fail if called.
- [x] Run the focused test and record RED.
- [x] Stop invoking `_qualify_ground_start` from `_build_ground_continuation_observation`; retain initial-boundary qualification.
- [x] Run focused continuation and start-qualification tests GREEN.

## Task 2: Batch consecutive same-cell sensor samples

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/sensor_observation.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/observation_boundary.py`
- Test: `training/lunar_policy_training/tests/test_sensor_observation.py`
- Test: `training/lunar_policy_training/tests/test_sensor_closed_loop.py`

- [x] Add a state-level RED comparing sequential observations to a missing batched API.
- [x] Add a controller-level RED proving N same-cell samples require one reveal.
- [x] Implement ordered batched state updates and controller run-length grouping.
- [x] Run sensor suites GREEN and compare all observed arrays and deltas.

## Task 3: Verify, commit, and recover

- [x] Run focused formal/sensor suites and existing ground-start regression.
- [x] Run a representative 820-cell same-pose performance comparison without touching active run state.
- [ ] Run `git diff --check`, review diff, and commit only task files.
- [ ] Fast-forward the clean emergency runtime to the verified commit.
- [ ] Archive the unsealed update 27 journal, migrate source identity on a copied update 26 checkpoint, and verify it before switching `latest.pt`.
- [ ] Resume the existing run and confirm live workers create fresh update 27 macro-boundary records without the prior qualification error.
