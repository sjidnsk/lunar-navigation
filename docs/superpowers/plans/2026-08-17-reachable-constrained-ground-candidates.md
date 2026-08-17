# Reachability-constrained ground candidates implementation plan

> **For Codex:** Execute this plan with `superpowers:executing-plans`, preserving the active training process until the atomic cutover step.

**Goal:** Stop WHEELED and LEGGED from losing an otherwise usable frontier because three fixed samples are rejected; emit at most three candidates per frontier chain only after global reachability, observed-only 0.2 m endpoint feasibility, and positive-gain checks.

**Architecture:** Keep 4 m frontier extraction and the single global reachability projection. Enumerate deterministic pose options along each complete frontier chain, batch-check their exact endpoints with the existing 0.2 m observed-only traversability projector, evaluate gain only for certified options, then spatially select at most three candidates per chain. The PPO sees only final candidate identities. HOPPER, Oracle-free termination, planner revalidation, reward V4, and planner algorithms remain unchanged.

**Tech stack:** Python 3.10, NumPy, pytest, existing `lunar_planner_core_py` bridge.

---

### Task 1: Pin the regression with focused RED tests

**Files:**
- Add: `training/lunar_policy_training/tests/test_reachable_constrained_ground_candidates.py`
- Add: `training/lunar_policy_training/tests/test_ground_endpoint_feasibility.py`
- Add: `training/lunar_policy_training/tests/test_formal_ground_endpoint_authority.py`

1. Add a frontier-chain fixture whose fixed 25/50/75 percent cells are endpoint-infeasible while other cells on the same chain are feasible and have positive gain.
2. Assert the ground universe backfills up to three deterministic certified positions, never emits an infeasible position, and retains the three-per-segment bound.
3. Add exact point-goal tests matching the C++ cell-box/tolerance rule for feasible, infeasible, and outside-window targets.
4. Assert `FormalEpisode` supplies the ground endpoint-feasibility batch callback while HOPPER does not use it.
5. Run only the new tests and observe failures caused by the missing production behavior.

### Task 2: Implement deterministic whole-chain constrained generation

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`

1. Replace fixed-sample-first mapping with deterministic whole-chain pose-option enumeration and deduplication.
2. Accept a ground-only batch endpoint-feasibility callback in `build_physical_universe`.
3. Apply physical/global reachability, exact endpoint feasibility, and exact positive gain before selecting positions.
4. Select at most three spatially distributed positions per frontier segment and only then create canonical `PhysicalCandidate` identities.
5. Keep decision snapshot counts internally consistent and leave HOPPER behavior unchanged.

### Task 3: Reuse the observed-only 0.2 m projector efficiently

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/platform_reachability.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`

1. Add a pure exact point-goal feasibility helper that mirrors `GoalIntersectsCell` over `hard_feasible` cells.
2. In `FormalEpisode`, group candidate endpoints into bounded 64 m observed-detail windows, call the existing traversability projection once per occupied group, and return a boolean result in input order.
3. Fail closed for uncovered or invalid endpoints. Do not read hidden truth and do not invoke a full planner per candidate.

### Task 4: Focused verification and atomic training cutover

**Files:**
- Verify only the touched candidate, reachability, and formal-builder suites.
- Preserve artifacts under `/home/kai/CodexDownloads/lunar_navigation/emergency-live-training-recovery/`.

1. Run the new tests, then the relevant candidate/reachability/formal focused suites and `git diff --check`.
2. Commit only the scoped source and test changes.
3. Wait for a complete atomic update checkpoint from the active run.
4. Create a source-migrated checkpoint that preserves model, optimizer, scheduler, RNG, curriculum, and global update, but restarts worker episode state because candidate identity semantics changed.
5. Stop the old controller, fast-forward the runtime checkout, update the controller checkpoint/expected SHA atomically, and resume immediately.
6. Verify new PIDs, manifest source SHA, restored global step, and the first committed macro-action boundary; roll back to the preserved checkpoint if startup fails.
