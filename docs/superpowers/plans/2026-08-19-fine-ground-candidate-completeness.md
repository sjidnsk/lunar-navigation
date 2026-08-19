# Fine Ground Candidate Completeness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the three-candidate ground PPO interface while making its internal 0.2 m candidate witnesses complete within fixed work bounds.

**Architecture:** `FormalEpisode` emits bounded detailed strip witnesses instead of one clearance-ranked pose. `CandidateBuilderV2` batch-certifies all witnesses through the existing single global reachability tree, endpoint callback and exact gain evaluator, then compresses positive witnesses to at most three candidates per segment. This implementation is deliberately limited to normal-path witness loss; a future whole-segment/paged residual fallback requires a new fine global-reachability authority and is not hidden behind the present 4 m tree.

**Tech Stack:** Python 3.10, NumPy, pytest, ROS 2 Humble, existing C++ training bridge.

**Spec:** `docs/superpowers/specs/2026-08-19-fine-ground-candidate-completeness-design.md`

## Global Constraints

- No online access to coverability truth, Oracle output or unknown terrain.
- No PPO shape, Reward V4, HOPPER, sensor or planner-safety change.
- One global tree per refresh; batch endpoint and gain checks; final output no more than three candidates per frontier segment.
- Native build, benchmark and training artifacts remain outside the repository.
- Runtime switch preserves model, optimizer, scheduler, RNG and global step but starts fresh worker episodes.

---

### Task 1: Preserve all bounded safe strip witnesses

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Test: `training/lunar_policy_training/tests/test_reachable_constrained_ground_candidates.py`

**Interfaces:**
- `_select_narrow_frontier_strip_positions(...)` returns all safe, deduplicated cells in canonical `(distance, lateral, row, column)` order.
- `_ground_detail_frontier_candidates(...)` converts each cell to `_RawFrontierCandidate`; `CandidateBuilderV2` remains the sole gain gate.

- [ ] **Step 1: Write RED tests**

```python
def test_positive_safe_neighbour_survives_when_max_clearance_pose_has_zero_gain():
    assert universe.candidates[0].target_position_m == positive_position

def test_missing_anchor_slot_is_backfilled_from_same_segment_witnesses():
    assert len(segment_candidates) == 3
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_reachable_constrained_ground_candidates.py -k 'positive_safe_neighbour or backfilled'`

Expected: failure because production emits only one high-clearance pose per anchor.

- [ ] **Step 3: Implement minimal witness emission**

```python
safe_cells.append((distance, lateral_offset, row, column))
# Sort and emit every unique safe cell; do not rank on clearance before gain.
```

- [ ] **Step 4: Verify GREEN**

Run: the two tests and existing `narrow_frontier_strip` tests.

Expected: all pass; maximum enumeration is 51 cells per anchor and final segment output is still at most three.

- [ ] **Step 5: Commit**

Commit message: `fix(training): retain fine frontier pose witnesses`

### Task 2: Use task-local unknown direction and compress after qualification

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Test: `training/lunar_policy_training/tests/test_reachable_constrained_ground_candidates.py`

**Interfaces:**
- The selector accepts an aligned local `unknown_roi_mask`.
- Positive witnesses are selected anchor-first; unused positive witnesses deterministically fill missing segment slots.

- [ ] **Step 1: Write RED tests**

```python
def test_strip_direction_ignores_unobserved_cells_outside_task_roi():
    assert selected_pose == authorized_side_pose

def test_segment_compression_caps_output_at_three_after_backfill():
    assert len(segment_candidates) == 3
```

- [ ] **Step 2: Verify RED**

Run: focused tests above.

Expected: failure because direction derives from every non-observed neighbour.

- [ ] **Step 3: Implement and verify GREEN**

```python
unknown_roi = (mission.roi_ratio > 0.0) & ~world.observed_mask
# Convert into each aligned 64 m detail window and select only after positive gain.
```

Run: full reachable-constrained candidate suite.

- [ ] **Step 4: Commit**

Commit message: `fix(training): constrain fine strips to task frontier`

### Deferred follow-up: Upgrade zero-positive fallback to fine paged witnesses

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py`
- Test: `training/lunar_policy_training/tests/test_exhaustion_upgrade_scan.py`
- Test: `training/lunar_policy_training/tests/test_v3_environment.py`

**Interfaces:**
- `ExhaustionUpgradeDiagnostics` gains fine witness/page/closed state with validated monotonic counts.
- A nonclosed scan cannot return `ZERO_EXPECTED_GAIN`.

- [ ] **Prerequisite:** Add a native exact fine endpoint-connectivity authority. Do not start this task by treating a coarse parent cell as an exact path certificate.

- [ ] **Step 1: Write RED tests**

```python
def test_fine_exhaustion_scan_finds_positive_witness_missed_by_coarse_representative():
    assert result.diagnostics.positive_pose_count == 1

def test_incomplete_scan_is_not_zero_expected_gain():
    assert _audit_candidate_boundary(snapshot, "WHEELED") is TerminalReason.TRUNCATED
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_exhaustion_upgrade_scan.py tests/test_v3_environment.py -k 'fine_exhaustion or incomplete_scan'`

Expected: failure because the current coarse scan reports a completed zero-gain result.

- [ ] **Step 3: Implement fixed-page scan and verify GREEN**

```python
for page in canonical_fine_witness_pages(page_size=4096):
    endpoint_mask = endpoint_feasibility(page.positions)
    gains = exact_gain(page.positions[endpoint_mask])
# Only completed, empty pages across the full library can classify zero gain.
```

Run: exhaustion and v3 focused suites.

- [ ] **Step 4: Commit**

Commit message: `fix(training): scan fine ground exhaustion witnesses`

### Task 3: Verify and atomically cut normal-path witness retention over

**Files:**
- Modify: both Task 1 specification and this plan with measured evidence.

- [ ] **Step 1: Build external Release native prefix**

Build only `lunar_planner_core` and `lunar_planner_training_bridge` into `/home/kai/CodexDownloads/lunar_navigation/fine-candidates-{build,install}`.

- [ ] **Step 2: Verify focused Python and native suites**

Run candidate, exhaustion, formal-builder, v3 environment, bridge and relevant native reachability tests against the new prefix.

- [ ] **Step 3: Measure bounded normal-path work**

Require one C++ detail projection per shared 64 m window, one endpoint batch and one gain batch per qualified witness set, no additional global-tree call, deterministic output and a final policy cardinality of at most three per segment.

- [ ] **Step 4: Commit verified scoped source and documentation**

Run `git diff --check`, stage only feature source/tests/docs, and commit with a focused message.

- [ ] **Step 5: Cut over only at an atomic update boundary**

Preserve current immutable checkpoint and SHA, update runtime source and external native prefix, resume with fresh worker episodes, then verify PID, source SHA, restored global step and the first new committed update. Restore the preserved runtime/checkpoint if startup fails.
