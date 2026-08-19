# 地面机会索引与增量恢复状态实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让地面候选完整性检查不再触发 24 个 worker 的全任务区同步扫描，并消除正常 update 对完整恢复状态的重复序列化与重读。

**Architecture:** 新增 observed-only `GroundOpportunityIndex`，以任务级几何索引和 episode 级脏 tile/可达性差分维护正收益站位。索引冷启动、失配重建和增量追平都切成固定站位预算的工作片，由父进程协调器最多同时授予两个 worker；普通三候选路径只查询已追平索引，索引未追平时不给 PPO 样本也不终止。恢复数据移入内容寻址对象存储，journal 与 checkpoint 共享对象引用；正常运行保留已验证对象，重启才完整解码校验。

**Tech Stack:** Python 3.10、NumPy、PyTorch、pytest、现有 task cache、TransitionJournal、FormalWorkerState。

**Spec:** `docs/superpowers/specs/2026-08-19-ground-opportunity-index-and-recovery-delta-design.md`

## Global Constraints

- 不改 Reward V4、PPO 动作空间、传感器语义、全局/局部规划器或平台能力。
- 普通地面前沿每段仍只输出三个策略候选；HOPPER 不使用本索引。
- 候选选择仅使用 observed-only 数据；隐藏覆盖分母和真值不得进入索引或 task cache。
- `VALID_INCOMPLETE_TERMINAL` 只能在索引追平当前观察/物理快照且无正收益站位时产生。
- 索引冷启动、缓存失配与追平必须由父进程的确定性协调器按固定站位片处理，`max_index_in_flight=2`；任何 worker 都不得在一次候选刷新中自行完成全任务区扫描。
- 宏动作边界仍原子提交；连续运行和崩溃恢复必须逐项等价。
- 运行 artifact、对象块、性能 JSON 均写到仓库外的 `/home/kai/CodexDownloads/lunar_navigation/`。

---

## 文件结构

- 新建 `environment/ground_opportunity_index.py`：任务几何反向索引、episode 状态和增量查询。
- 修改 `environment/candidate_builder.py`：全量 scan 保留为离线参考，不再在 `build()` 热路径调用。
- 修改 `environment/formal_builder.py`、`formal_episode_state.py`、`v3_environment.py`、`parallel_pool.py`：维护分片索引工作、处理 `INDEX_PENDING`、由父进程协调器限流调度并持久化索引引用。
- 新建 `recovery/state_object_store.py`，修改 `transition_journal.py` 与 `cli.py`：共享内容寻址对象。
- 新建 `tests/test_ground_opportunity_index.py`，扩展 `test_transition_journal.py`、`test_checkpoint_resume.py`、`test_v3_environment.py` 与 `test_formal_resume_state.py`。

## Task 1: observed-only 增量机会索引

**Files:**

- Create: `training/lunar_policy_training/lunar_policy_training/environment/ground_opportunity_index.py`
- Create: `training/lunar_policy_training/tests/test_ground_opportunity_index.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class GroundOpportunityIndexState:
    observation_generation: int
    physical_snapshot_id: str
    indexed_generation: int
    positive_pose_cells: tuple[tuple[int, int], ...]
    positive_gain_pairs: tuple[tuple[float, float], ...]
    reachable_mask_sha256: str
    observed_mask_sha256: str
    pending_pose_cells: tuple[tuple[int, int], ...]
    next_pending_pose_offset: int

@dataclass(frozen=True, slots=True)
class GroundOpportunityIndexUpdate:
    state: GroundOpportunityIndexState
    endpoint_certified_pose_count: int
    exact_gain_evaluated_pose_count: int
    pending_pose_count: int
```

`GroundOpportunityIndex.update` 接收前一状态、当前 observed mask、可达站位 mask、
30 m 反向索引命中的脏 tile、可达性差分、端点认证回调、当前 generation/snapshot 和
显式 `max_pose_count`。它一次最多认证和精确评估 `max_pose_count` 个稳定排序的站位；
剩余站位以 `pending_pose_cells` 与 `next_pending_pose_offset` 留在状态中。只有 offset
到达队列末尾时，`indexed_generation` 才可更新到当前 generation。

- [ ] **Step 1: Write the failing test**

Add `test_dirty_tile_update_matches_full_observed_only_reference`. Use a small real `ObservedWorld`/`MissionRaster` fixture with two reachable poses. Change one observed tile within only the first pose's 30 m influence area. Assert that the index's final positive cells and gain pairs equal the existing `scan_ground_exhaustion_candidates()` reference, and that the second update certifies exactly one affected endpoint.

Add `test_cold_index_reaches_reference_in_bounded_slices`. Start with five eligible poses and `max_pose_count=2`; assert each invocation certifies no more than two poses, emits `pending_pose_count` until the third invocation, and only the completed state equals the full observed-only reference.

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
cd training/lunar_policy_training
PYTHONPATH=. /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -q tests/test_ground_opportunity_index.py -v
```

Expected: FAIL because `GroundOpportunityIndex` does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement deterministic pose coding and a 30 m reverse-neighbour lookup. A cold index creates a stable sorted work queue but evaluates only its first fixed-size slice. Later updates append only `reachable_delta_mask` plus poses whose stencil intersects `dirty_observed_mask`, deduplicate by pose cell, and process only the next slice. Sort output by `(row, column)`; keep timing arrays and hidden data outside `GroundOpportunityIndexState`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
cd training/lunar_policy_training
PYTHONPATH=. /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -q tests/test_ground_opportunity_index.py tests/test_exhaustion_upgrade_scan.py -v
```

Expected: PASS; the old scan remains an isolated reference only.

- [ ] **Step 5: Commit**

```bash
git add training/lunar_policy_training/lunar_policy_training/environment/ground_opportunity_index.py \
  training/lunar_policy_training/tests/test_ground_opportunity_index.py
git diff --cached --check
git commit -m "feat(training): index ground opportunities incrementally"
```

## Task 2: 从 rollout 热路径移除全量 exhaustion scan

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_episode_state.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py`
- Test: `training/lunar_policy_training/tests/test_ground_opportunity_index.py`
- Test: `training/lunar_policy_training/tests/test_v3_environment.py`

**Interfaces:**

`GroundOpportunityQueryResult` 有三个确定值：`READY_WITH_CANDIDATES`、
`READY_EMPTY`、`INDEX_PENDING`。查询仅在 `indexed_generation`、观察 generation 和
物理 snapshot 全部匹配且没有待处理站位时返回前两个值，否则返回 `INDEX_PENDING`。

`ParallelPool` 新增父进程拥有的 `GroundOpportunityIndexCoordinator`。它按
`(target_generation, worker_index)` 排序等待项，每一轮至多授予两个 worker 一个
`max_pose_count` 固定的索引工作片；未获授予者保持 `INDEX_PENDING`，不得运行该片、
不得产生 PPO transition。该 coordinator 的队列与授予游标纳入 journal/recovery state，
因此恢复后的下一次授予顺序与连续运行一致。

- [ ] **Step 1: Write failing integration tests**

Add `test_empty_frontier_queries_ready_index_without_full_scan`: replace the legacy full scan with a raising callable, build an empty ordinary frontier with one ready positive index witness, and assert the episode emits that candidate.

Add `test_stale_index_cannot_emit_valid_incomplete_terminal`: give the episode an index whose `indexed_generation` is behind the observation. Assert it emits `INDEX_PENDING`, produces no PPO transition, and does not set `VALID_INCOMPLETE_TERMINAL`.

Add `test_parent_index_coordinator_limits_concurrent_work_slices`: use four pending ground-worker fixtures, grant two slices, and assert exactly workers 0 and 1 receive the first deterministic grants while workers 2 and 3 remain pending. Complete worker 0, grant again, and assert worker 2 receives the next grant; at no point may more than two grants be active.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
cd training/lunar_policy_training
PYTHONPATH=. /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -q tests/test_ground_opportunity_index.py tests/test_v3_environment.py \
  -k 'index or stale' -v
```

Expected: FAIL because `CandidateBuilderV2.build()` still calls the synchronous fallback and no pending boundary exists.

- [ ] **Step 3: Write minimal implementation**

Have `FormalEpisode` retain prior observed/reachability masks, calculate deltas after each sensor commit, and create a pending index-work request. `ParallelPool` owns the deterministic two-slot coordinator and dispatches `advance_ground_opportunity_index` only to granted workers, each call using the fixed pose budget. Remove the training call from `CandidateBuilderV2.build()` to `build_ground_exhaustion_universe()`. With no normal candidate, query only a state whose generations match; otherwise send `INDEX_PENDING` through the pool without advancing PPO tensors, reward, coverage, candidate identity, or policy version. A cache miss never promotes a worker to a whole-map scan.

- [ ] **Step 4: Run focused integration tests**

Run:

```bash
cd training/lunar_policy_training
PYTHONPATH=. /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -q tests/test_ground_opportunity_index.py tests/test_exhaustion_upgrade_scan.py \
  tests/test_v3_environment.py -v
```

Expected: PASS; HOPPER and fixed three-candidate safe strips are unchanged, and the parent grants no more than two index slices at once.

- [ ] **Step 5: Commit**

```bash
git add training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py \
  training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py \
  training/lunar_policy_training/lunar_policy_training/environment/formal_episode_state.py \
  training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py \
  training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py \
  training/lunar_policy_training/tests/test_ground_opportunity_index.py \
  training/lunar_policy_training/tests/test_v3_environment.py
git diff --cached --check
git commit -m "fix(training): remove exhaustive ground scan from rollout"
```

## Task 3: 内容寻址对象存储与 journal 去重

**Files:**

- Create: `training/lunar_policy_training/lunar_policy_training/recovery/state_object_store.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/recovery/transition_journal.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Test: `training/lunar_policy_training/tests/test_transition_journal.py`
- Test: `training/lunar_policy_training/tests/test_checkpoint_resume.py`

**Interfaces:**

`StateObjectRef` 是不可变的 `{sha256, byte_count, kind}` 三元组。`StateObjectStore`
提供三个明确操作：将 mapping 写入对象并返回引用；按引用完整解码 mapping；仅校验对象
存在性、字节数、SHA-256 和 kind。所有对象路径都由 SHA-256 唯一确定，调用者不能提供
任意路径；引用不匹配或对象缺失均以恢复错误失败关闭。

- [ ] **Step 1: Write failing normal/restart tests**

Add `test_normal_seal_uses_in_memory_object_refs_but_restart_validates_all`. Commit four real slots to a temporary journal/object store. Spy on the real read boundary. Assert normal `seal_update()` reads neither already committed slot nor object; create a new journal instance and assert restart `load_update()` validates all four object references exactly once.

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
cd training/lunar_policy_training
PYTHONPATH=. /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -q tests/test_transition_journal.py -k 'object_refs or normal_seal' -v
```

Expected: FAIL because objects/refs do not exist and sealing reloads payloads.

- [ ] **Step 3: Write minimal implementation**

Store each state mapping at `<journal-root>/objects/<sha256>.pt` using temporary-file write, fsync, atomic rename, and parent fsync. Journal slots store the reference, transition, and identity only. Keep validated open-update payloads in memory; `seal_update()` seals that cache and returns it. A new process, cache miss, or recovery validates object SHA and decodes the mapping.

- [ ] **Step 4: Share refs with checkpoint**

Change `_checkpoint_environment_state_from_sealed_journal()` to persist scenario identity and ordered worker `StateObjectRef` values. On restore, resolve refs through `StateObjectStore` before the existing `FormalWorkerState.from_dict()` validation. The checkpoint must not embed duplicate `worker_episode_states` mappings.

- [ ] **Step 5: Run journal/checkpoint verification**

Run:

```bash
cd training/lunar_policy_training
PYTHONPATH=. /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -q tests/test_transition_journal.py tests/test_checkpoint_resume.py -v
```

Expected: PASS; normal seal does not reread, restart validates fully, checkpoint identity remains strict.

- [ ] **Step 6: Commit**

```bash
git add training/lunar_policy_training/lunar_policy_training/recovery/state_object_store.py \
  training/lunar_policy_training/lunar_policy_training/recovery/transition_journal.py \
  training/lunar_policy_training/lunar_policy_training/cli.py \
  training/lunar_policy_training/tests/test_transition_journal.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py
git diff --cached --check
git commit -m "perf(training): share journal recovery state objects"
```

## Task 4: 分块 worker 状态、迁移与性能门

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_episode_state.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/recovery/state_object_store.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Test: `training/lunar_policy_training/tests/test_formal_resume_state.py`
- Test: `training/lunar_policy_training/tests/test_checkpoint_resume.py`
- Test: `training/lunar_policy_training/tests/test_training_smoke.py`
- Test: `training/lunar_policy_training/tests/test_cli.py`

**Interfaces:**

`FormalWorkerStateRef` 包含任务静态块引用、有序 tile 块引用、有序 reveal/replay 历史块
引用、机会索引块引用和小型 terminal scalars。`materialize_worker_state(reference, store)`
按该顺序读取、拼接后构造现有 `FormalWorkerState`，再调用既有验证逻辑。它拒绝缺块、
SHA 错误、kind 错误、重复引用和不连续历史块。

- [ ] **Step 1: Write failing chunk/recovery tests**

Add `test_worker_state_ref_reuses_unchanged_tiles_and_history_chunks`: create two consecutive boundaries where only one tile and one reveal sample change. Assert the second reference reuses the first static and unchanged-tile refs and adds only one event/tile object. Materialize both and compare their dictionaries to the original worker states.

Add a checkpoint roundtrip test that runs uninterrupted and resumed tracks, then compares observation identity, candidate IDs, reward, coverage, request digest, and terminal reason.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
cd training/lunar_policy_training
PYTHONPATH=. /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -q tests/test_formal_resume_state.py tests/test_checkpoint_resume.py \
  -k 'chunk or exact_active' -v
```

Expected: FAIL because every state currently expands full reveal history and map mapping.

- [ ] **Step 3: Write minimal implementation**

Encode task-static fields once, changed map tiles separately, and reveal/replay history in fixed-size immutable blocks. `FormalWorkerStateRef` contains object refs and terminal scalars; `materialize_worker_state()` rebuilds a normal `FormalWorkerState` and uses its existing validation. Missing object, SHA mismatch, duplicate reference, or wrong kind must fail closed before live worker mutation.

- [ ] **Step 4: Add transient stage timings and external benchmark**

Record `opportunity_index_s`, `endpoint_certification_s`, `exact_gain_s`, `journal_seal_s`, and `checkpoint_s` outside all identities and reward state. Run 20 consecutive boundaries against one task and write medians for boundaries 1--10 and 11--20 to `/home/kai/CodexDownloads/lunar_navigation/ground_opportunity_recovery_perf/`. The later group must not be slower solely because reveal history is longer.

- [ ] **Step 5: Add fresh-episode migration and run required suite**

Bump formal state/checkpoint/journal identities. Old artifacts remain read-only; a new run begins from an applied update boundary with weights/optimizer/normalization/global step preserved and fresh worker episodes. Reject an unsealed update or mixed reference format.

Run:

```bash
cd training/lunar_policy_training
PYTHONPATH=. /home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  -m pytest -q tests/test_ground_opportunity_index.py tests/test_exhaustion_upgrade_scan.py \
  tests/test_formal_resume_state.py tests/test_transition_journal.py \
  tests/test_checkpoint_resume.py tests/test_v3_environment.py \
  tests/test_training_smoke.py tests/test_cli.py
cd /home/kai/CodexDownloads/lunar_navigation/worktrees/narrow-frontier-safe-belt
git diff --check
```

Expected: PASS; this is the bounded candidate/recovery/launch suite, not a whole-repository gate.

- [ ] **Step 6: Commit**

```bash
git add training/lunar_policy_training/lunar_policy_training/environment/formal_episode_state.py \
  training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py \
  training/lunar_policy_training/lunar_policy_training/recovery/state_object_store.py \
  training/lunar_policy_training/lunar_policy_training/cli.py \
  training/lunar_policy_training/tests/test_formal_resume_state.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py \
  training/lunar_policy_training/tests/test_training_smoke.py \
  training/lunar_policy_training/tests/test_cli.py
git diff --cached --check
git commit -m "perf(training): chunk formal recovery state"
```
