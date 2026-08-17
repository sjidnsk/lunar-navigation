# Ground Observation Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除地面路径观测、滚动地图物化和 checkpoint alias 的重复工作，并从完整 update 边界恢复当前训练。

**Architecture:** Multires 状态新增一次原子地面路径事务，事务内用 per-tile cursor 延迟 observation age 更新；固定 planner-level 索引在 episode 内缓存；不可变 update checkpoint 通过原子 hardlink 暴露为 `latest.pt`。所有逻辑保持现有正式语义与恢复权威。

**Tech Stack:** Python 3.10, NumPy, PyTorch, pybind11 training bridge, pytest, Ubuntu 22.04, RTX 4080 SUPER.

## Global Constraints

- 不改 worker 同步/长尾调度。
- 不改 0.2 m、30 m、360 度、地面路径采样间隔不大于 1 m。
- 不改候选、奖励、规划器、PPO、checkpoint schema 和完整 update 快照频率。
- 生产代码必须先有有效 RED；只运行 focused suites 和一个固定 checkpoint 的单种子性能门。
- 训练在旧 source 上继续；新 source 只在完整 update 边界且 ≥35% 性能门通过后切换。

---

### Task 1: Atomic ground-path observation with lazy tile aging

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/multires_observation.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/observation_boundary.py`
- Test: `training/lunar_policy_training/tests/test_multires_observation.py`
- Test: `training/lunar_policy_training/tests/test_sensor_closed_loop.py`

**Interfaces:**
- Consumes: ordered map-frame `Pose2` values and per-sample elapsed seconds already carried by `SensorBoundaryEvidence.path_samples`.
- Produces: `MultiresSensorObservationState.observe_world_path(samples: tuple[tuple[Pose2, float], ...]) -> ObservationDelta`.

- [ ] **Step 1: Write the state-level failing equivalence test**

Add a test that creates sequential and batched states, applies a literal path containing distinct cells and one repeated cell, then asserts exact equality of `_authoritative_observation_bytes`, aggregated `ObservationDelta`, evidence generation, and reveal count. It must fail because `observe_world_path` does not exist.

- [ ] **Step 2: Run the focused RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_multires_observation.py::test_ground_path_batch_preserves_sequential_authoritative_updates
```

Expected: FAIL with missing `observe_world_path`.

- [ ] **Step 3: Implement the minimal atomic transaction**

Add validation and consecutive-cell preparation, clone authoritative mutable state once, apply samples in order, restore references on exception, aggregate literal deltas, and commit only after all samples succeed.

- [ ] **Step 4: Add per-tile elapsed cursors**

Inside the transaction, append ordered elapsed values to one ledger. Materialize a tile only before it is affected or at final commit using the existing float64-to-float32 addition loop. Update coarse cells after each affected sample from the materialized transaction tile.

- [ ] **Step 5: Write and run the controller RED**

Add a real multires controller test proving one ground boundary calls the path API once and produces the same policy observation identity as sequential evidence. Run the exact node and confirm failure before routing the ground branch through `observe_world_path`.

- [ ] **Step 6: Route the controller and run GREEN**

Ground multires evidence uses `observe_world_path`; generic sensor doubles retain the existing same-cell fallback. Run focused multires, sensor, and boundary tests, excluding only the three recorded task-edge baseline assertions.

- [ ] **Step 7: Commit Task 1**

```bash
git add training/lunar_policy_training/lunar_policy_training/environment/multires_observation.py \
  training/lunar_policy_training/lunar_policy_training/environment/observation_boundary.py \
  training/lunar_policy_training/tests/test_multires_observation.py \
  training/lunar_policy_training/tests/test_sensor_closed_loop.py
git commit -m "perf(training): batch ground path observations"
```

### Task 2: Reuse fixed task-to-planner projection indices

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Test: `training/lunar_policy_training/tests/test_formal_builder.py`

**Interfaces:**
- Consumes: immutable source and planner `MapCanvas` geometry.
- Produces: a cached row/column projection index pair reused by every layer and rolling boundary.

- [ ] **Step 1: Write the failing projection-cache test**

Add a fixed 100–500 m geometry table that compares every projected layer against `_project_task_grid_to_planner_level` and asserts the episode computes projection indices once across two rolling observations. The count assertion must fail before caching exists.

- [ ] **Step 2: Run the focused RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_formal_builder.py -k planner_projection_indices
```

- [ ] **Step 3: Implement immutable index reuse**

Extract the source row/column and target validity calculation into a frozen value owned by `FormalEpisode`; reuse it for every layer while still allocating a fresh output array and a fresh bridge stamp.

- [ ] **Step 4: Run focused GREEN and commit**

```bash
git add training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py \
  training/lunar_policy_training/tests/test_formal_builder.py
git commit -m "perf(training): reuse planner map projection indices"
```

### Task 3: Atomically alias the immutable update checkpoint

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/checkpoint.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Test: `training/lunar_policy_training/tests/test_checkpoint.py`

**Interfaces:**
- Consumes: one existing regular immutable `update-%08d.pt` in the same directory as `latest.pt`.
- Produces: `replace_checkpoint_alias_atomic(immutable_path: Path, latest_path: Path) -> None`.

- [ ] **Step 1: Write the failing filesystem behavior test**

Save two literal checkpoints, alias the first then the second, and assert `latest.pt` loads the expected payload, shares inode with the selected immutable file, never shares inode with the old file after replacement, and leaves no temporary file. It must fail because the helper is absent.

- [ ] **Step 2: Run the focused RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_checkpoint.py -k checkpoint_alias
```

- [ ] **Step 3: Implement same-directory atomic hardlink replacement**

Validate regular files and parent identity, create one exclusive temporary hardlink, `os.replace` it over `latest.pt`, fsync the directory, and remove only the exact temporary path after an error.

- [ ] **Step 4: Replace the second serialization and run GREEN**

After `commit_or_recover_reward_v4_update` returns, alias its immutable `checkpoint_path` instead of calling `save_checkpoint_atomic(latest, ...)`. Run checkpoint and Reward V4 commit/recovery tests.

- [ ] **Step 5: Commit Task 3**

```bash
git add training/lunar_policy_training/lunar_policy_training/checkpoint.py \
  training/lunar_policy_training/lunar_policy_training/cli.py \
  training/lunar_policy_training/tests/test_checkpoint.py
git commit -m "perf(training): alias immutable update checkpoints"
```

### Task 4: Minimal qualification and atomic training recovery

**Files:**
- Modify: `docs/superpowers/specs/2026-08-17-ground-observation-throughput-design.md`
- Modify: `docs/superpowers/plans/2026-08-17-ground-observation-throughput.md`
- Runtime artifacts: outside Git under `~/CodexDownloads/lunar_navigation/ground-observation-throughput/`

**Interfaces:**
- Consumes: the latest complete old-source checkpoint and one frozen WHEELED/LEGGED task pair.
- Produces: exact-equivalence evidence, one performance report, a descendant source-migration checkpoint, and resumed live worker evidence.

- [ ] **Step 1: Run focused verification**

Run the modified test nodes, checkpoint commit/recovery nodes, `git diff --check`, Python compilation, and repository boundary checks. Record the three unrelated task-edge baseline failures separately.

- [ ] **Step 2: Run one fixed-checkpoint performance comparison**

From a copy of the same complete checkpoint, run one WHEELED and one LEGGED macro action on old and new source. Compare transition identity, coverage, reward, terminal class, planner counts, evidence SHA and checkpoint replay. Require at least 35% lower combined environment/macro unit-work wall time.

- [ ] **Step 3: Commit documentation evidence**

Update only the acceptance checkboxes and external artifact SHA references, then commit the spec and plan without adding generated reports.

- [ ] **Step 4: Migrate at a complete update boundary**

Wait for the current old-source run to publish a complete immutable update checkpoint, stop only the controller-owned process tree, create the audited descendant source-migration checkpoint, atomically replace `latest.pt`, and start the existing resume controller against the same artifact root.

- [ ] **Step 5: Verify resumed training**

Confirm source identity, manifest/global step, 24 workers, fresh macro-boundary journals, a subsequent complete update checkpoint, finite metrics, and no hard error. Do not claim convergence from liveness.
