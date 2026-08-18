# Recovery Throughput and Retention Implementation Plan

> Execute with test-driven development and verification before switching a live run.

**Goal:** Eliminate redundant disk deserialization during normal macro-action journal commits, retain the existing crash-recovery contract, and remove only training artifacts proven outside the active recovery dependency chain.

**Constraints:** Do not change PPO, rewards, candidate generation, planner semantics, map resolution, worker count, or macro-action boundaries.  Keep each committed transition durable before it is acknowledged.  The live run stays on its current source until the optimized build passes focused verification and can switch at an update boundary.

## Task 1: Cache the open journal update in memory

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/recovery/transition_journal.py`
- Test: `training/lunar_policy_training/tests/test_transition_journal.py`

- [ ] Add a regression proving successive normal commits do not reload already committed immutable payload files.
- [ ] Preserve retry, corruption, fault-injection, sealing, and process-restart validation behavior.
- [ ] Implement an invalidated in-memory cache for an open update; mutate it only after the durable index append.
- [ ] Run the focused journal suite and a bounded disk-I/O comparison.

## Task 2: Reuse sealed terminal worker states for checkpoints

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/recovery/transition_journal.py`
- Test: `training/lunar_policy_training/tests/test_cli.py`

- [ ] Prove that the sealed journal contains the last committed state for each worker and reconstruct the existing full `environment_state` checkpoint schema from it.
- [ ] Reject any missing, duplicated, nonterminal, or non-serializable worker state before checkpoint publication.
- [ ] Replace the redundant all-worker `snapshot_episode_states()` request with the sealed journal state only after the update has been durably sealed.
- [ ] Keep the full independent checkpoint payload; do not introduce a delta checkpoint format or weaken recovery.

## Task 3: Verify switch compatibility and retention targets

- [ ] Validate that a new process reloads the journal produced by the cached writer exactly.
- [ ] Inspect the live run manifest, checkpoint references, and journal state to produce an explicit retention manifest.
- [ ] Remove no active or recovery-referenced artifact.  Because repository policy forbids bulk recursive deletion, present any large batch deletion for manual execution after the manifest is verified.

## Task 4: Commit and atomically resume

- [ ] Run focused recovery and CLI resume checks, `git diff --check`, and review only task files.
- [ ] Commit the verified source change.
- [ ] Wait for a live update boundary, migrate the source identity without changing model or worker state, then resume and confirm a new journal update progresses.
