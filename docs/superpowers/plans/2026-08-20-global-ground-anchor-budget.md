# Global Ground Anchor Budget Implementation Plan

> **For agentic workers:** Execute this plan task by task with test-driven development. Do not restore complete-frontier or residual-area scans.

**Goal:** Replace the unbounded ground candidate refresh with one deterministic global pass capped at 96 anchors and 32 fine windows, while preserving the 64 policy plus 32 reserve contract and resuming training without rebuilding task-area caches.

**Architecture:** The 4 m frontier map remains the global discovery layer. It contributes exactly three normal anchors per segment. A deterministic spatial selector chooses at most 32 aligned 64 m fine windows and at most 96 anchors. Each selected anchor enumerates a bounded 0.2 m safe strip, but emits at most one certified positive-gain candidate. Fine projection, endpoint reachability, and gain evaluation are reused per selected window. Deferred anchors are represented as honest budget truncation, never as physical exhaustion.

**Tech stack:** Python 3.10, NumPy, pytest, pybind-backed planner bridge, ROS 2 Humble runtime.

**Specification:** `docs/superpowers/specs/2026-08-20-global-ground-anchor-budget-design.md`

## Non-negotiable boundaries

- Ground only: WHEELED and LEGGED. HOPPER remains unchanged.
- Sensor remains 30 m and 360 degrees.
- Normal anchors remain segment positions 1/4, 1/2, and 3/4.
- Global limits: 96 anchors, 32 fine windows, 64 policy candidates, 32 reserve candidates.
- One anchor emits at most one candidate; one segment therefore emits at most three.
- Each fine strip is at most 20 longitudinal by 3 lateral 0.2 m witnesses.
- No complete-frontier second-level scan, no residual-area third-level scan, no paging cursor.
- No Oracle, Reward V4, PPO tensor, C++ planner algorithm, ROS interface, task-area, or sensor changes.
- Existing physical safety and exact execution-planner hard gates remain.
- A same-snapshot planner rejection may promote reserve candidates only; it must not rebuild candidates.
- Runtime semantics may change, but the unchanged task-area cache must retain its former compatibility identity.

## Task 1: Freeze limits and snapshot schema

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/training_semantics.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/candidate_builder.py`
- Test: `training/lunar_policy_training/tests/test_coverability.py`
- Test: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`

1. Add RED tests for exact constants and strict snapshot round-trip fields.
2. Run the two focused test files and observe missing constants/fields.
3. Add:

```python
FORMAL_GROUND_ANCHOR_LIMIT = 96
FORMAL_GROUND_DETAIL_WINDOW_LIMIT = 32
FORMAL_POLICY_CANDIDATE_LIMIT = 64
FORMAL_GROUND_RESERVE_LIMIT = 32
```

4. Extend `CandidateDecisionSnapshot` with strict, default-free persisted fields:

```python
ground_anchor_total_count: int
ground_anchor_selected_count: int
ground_anchor_deferred_count: int
ground_detail_window_count: int
ground_detail_witness_count: int
ground_exact_endpoint_query_count: int
ground_positive_gain_count: int
ground_anchor_budget_truncated: bool
ground_candidate_generation_elapsed_s: float
ground_exact_endpoint_elapsed_s: float
ground_gain_evaluation_elapsed_s: float
```

5. Update all constructors and exact serializers in the same patch; do not add permissive decoding.
6. Run focused tests and commit `test(training): freeze bounded ground candidate schema`.

## Task 2: Implement a pure deterministic anchor/window selector

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/candidate_builder.py`
- Test: `training/lunar_policy_training/tests/test_reachable_constrained_ground_candidates.py`

1. Add RED fixtures containing more than 96 anchors and more than 32 windows. Assert:
   - each segment contributes only 1/4, 1/2, 3/4 anchors;
   - selected anchors are at most 96;
   - selected windows are at most 32;
   - repeated and input-permuted runs are identical;
   - spatially separated frontier components survive selection.
2. Add immutable internal types `_GroundFrontierAnchor`, `_GroundAnchorWindow`, and `_GroundAnchorBudget`. Keep `anchor_budget_rank` internal; do not add it to public `PhysicalCandidate`.
3. Implement `_select_ground_anchor_budget(...)`:
   - map each anchor to a 32 m lattice-aligned 64 m window;
   - seed the window set with the lowest finite coarse global cost, then coordinates;
   - repeatedly select the window maximizing distance to selected windows, breaking ties by coarse cost and coordinates (implemented as `min()` over a key whose first item is negative minimum squared distance);
   - visit middle anchors, then quarter anchors, then three-quarter anchors, round-robin across selected windows until 96;
   - preserve selected anchor order as internal budget rank.
4. Return exact total, selected, deferred, and window counts.
5. Run the focused file and commit `feat(training): select bounded ground frontier anchors`.

## Task 3: Build one bounded fine page per selected window

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/formal_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/candidate_builder.py`
- Test: `training/lunar_policy_training/tests/test_formal_ground_endpoint_candidates.py`
- Test: `training/lunar_policy_training/tests/test_formal_builder.py`

1. Add RED tests proving one projection per selected window and no complete-segment recovery callback.
2. Introduce `_GroundDetailCandidatePage` containing window identity, ordered anchors, witnesses, projection, and timing/count diagnostics.
3. Replace the flat detail-candidate provider with a bounded page provider driven by `_GroundAnchorBudget`.
4. For every selected window:
   - build one 64 m fine traversability projection;
   - enumerate at most 20 by 3 safe witnesses for every anchor in that window;
   - use the same projection for strip `hard_feasible`, footprint/clearance, and exact endpoint-safety tolerance;
   - deduplicate witnesses deterministically.
5. Remove `ground_detail_segment_recovery_provider` wiring and all complete-segment fallback calls. Do not leave a dormant alternate path.
6. Keep the immutable global cost-tree context shared across all pages; do not rebuild global search per witness or page.
7. Run focused tests and commit `refactor(training): build bounded ground detail pages`.

## Task 4: Certify pages and compress to one candidate per anchor

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/candidate_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/formal_builder.py`
- Test: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`
- Test: `training/lunar_policy_training/tests/test_formal_candidate_builder.py`

1. Add RED tests asserting:
   - endpoint and gain batch calls are at most the number of selected windows;
   - every selected anchor contributes at most one result;
   - result universe is at most 96;
   - policy batch is at most 64 and reserve at most 32;
   - all emitted candidates are exact-endpoint reachable and have positive exact task-ROI gain;
   - promoting reserve after a rejection does not call the page provider again.
2. Within each page, call the existing exact endpoint query once and exact gain evaluation once.
3. Select one witness per anchor with the stable key:

```text
positive gain descending
fine clearance descending
global cost ascending
exact x, y, z ascending
```

4. Rank the resulting candidates by internal anchor budget rank, then existing candidate identity. Store the rank in existing `rank_key`; do not change model features or public candidate schema.
5. Slice the canonical ground universe into first 64 policy candidates and next 32 reserve candidates.
6. Persist exact counts and three stage timings in `CandidateDecisionSnapshot`.
7. Run focused tests and commit `feat(training): certify bounded ground candidate pages`.

## Task 5: Add honest budget-truncation termination

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/v3_environment.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/macro_step.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/formal_episode_state.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/closed_loop_gate.py`
- Test: `training/lunar_policy_training/tests/test_v3_environment.py`
- Test: `training/lunar_policy_training/tests/test_macro_step.py`
- Test: `training/lunar_policy_training/tests/test_checkpoint_resume.py`

1. Add RED tests for an empty/all-rejected selected page with deferred anchors.
2. Add `GROUND_CANDIDATE_BUDGET_TRUNCATED` as a valid incomplete terminal reason.
3. In boundary audit, give truncation precedence over zero-gain, no-route, planner-exhausted, and physical-exhaustion classifications when `ground_anchor_deferred_count > 0` and no executable selected candidate remains.
4. Persist strict diagnostics and restore them exactly through checkpoint/resume.
5. Verify that an actually complete scan with no deferred anchors retains the existing physical terminal classification.
6. Run focused tests and commit `fix(training): report bounded ground scan truncation honestly`.

## Task 6: Separate task-cache compatibility from runtime semantics

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/training_semantics.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/task_cache.py`
- Test: `training/lunar_policy_training/tests/test_formal_cache.py`
- Test: `training/lunar_policy_training/tests/test_checkpoint_resume.py`

1. Before editing, extract the exact former v18 training-semantics string and SHA used by the existing task caches.
2. Add RED tests proving:
   - runtime/checkpoint semantics changes to v19;
   - task-cache validation still expects the exact former v18 SHA;
   - a cache with any different task/sensor/coverability identity is rejected.
3. Add a frozen `FORMAL_TASK_CACHE_SEMANTICS_VERSION` equal to the exact former v18 semantic string and `task_cache_semantics_sha256()` over it.
4. Retain the existing serialized cache field name `training_semantics_sha256` for on-disk compatibility, but produce/validate it with `task_cache_semantics_sha256()`.
5. Keep checkpoints and runtime manifests on the new v19 `training_semantics_sha256()`.
6. Run focused tests and commit `fix(training): preserve task cache compatibility across candidate refresh`.

## Task 7: Focused qualification and one performance gate

**Files:** No production changes unless a focused test exposes a defect.

1. Run `python3 -m py_compile` on all modified Python modules.
2. Run only the focused suites named above plus existing ground candidate identity/determinism tests.
3. Run `git diff --check`.
4. Create external artifacts under `/home/kai/CodexDownloads/lunar_navigation/global-ground-anchor-budget/`; do not write generated artifacts into Git.
5. On one fixed stress scene, compare baseline `d7b7678` and the feature commit with identical source snapshot and capability inputs. Record:
   - total/selected/deferred anchors;
   - windows, witnesses, endpoint calls, gain calls;
   - policy/reserve counts;
   - candidate-generation, endpoint, gain, and total wall time;
   - candidate identity hash for two repeated feature runs.
6. Qualification requires counts within 96/32/5760/32/32/64/32, repeated identity equality, and feature wall time at most 65% of the old complete-frontier path. If this fails, do not cut over.
7. Do not run multi-seed, full-scene, replay, or formal three-platform gates.

## Task 8: Atomic cutover and fresh-episode training resume

**Files:** External runtime/checkpoint/cache directories only.

1. Record source SHA, last sealed global update, model/optimizer/scheduler/RNG hashes, task-cache identity, and rollback command.
2. Verify no old training process remains before switching source.
3. Resume from the last sealed global update while preserving model, optimizer, scheduler, RNG, and compatible task caches; discard only worker episode state and start fresh episodes.
4. Start the existing 24-worker command with evaluation disabled and the same Reward V4/task-scale configuration.
5. Verify process liveness, candidate-stage diagnostics, no repeated page-provider rebuild, and the first newly committed update.
6. Report the first update's collection wall, PPO wall, per-platform success/incomplete/failure counts, per-worker coverage, and the three candidate-stage timings. Do not claim convergence from one update.

## Final acceptance

- Source is committed on an isolated feature branch based on `d7b7678`.
- No second-level or third-level fallback remains callable.
- Ground generation respects 96 anchors, 32 windows, 64 policy, and 32 reserve.
- Deferred work is auditable as `GROUND_CANDIDATE_BUDGET_TRUNCATED`.
- Existing task-area caches validate without rebuild; runtime semantics are v19.
- The single fixed performance gate improves wall time by at least 35%.
- Training resumes from the last sealed update with fresh worker episodes and commits one new update.
