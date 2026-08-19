# Narrow Frontier Safe Belt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate at most three platform-safe 0.2 m ground observation poses per frontier segment without scanning an unbounded map area.

**Architecture:** Candidate construction keeps coarse frontier segmentation and policy shape. A bounded, deterministic detail-strip selector finds one observed-safe pose for each of three anchors. `FormalEpisode` supplies only current observed detail and a platform traversability projection; candidate identity v3 distinguishes the resulting exact positions. Existing gain, endpoint and planner checks remain mandatory.

**Tech Stack:** Python 3.10, NumPy, pytest, existing `lunar_planner_training_bridge` C++ projection.

**Spec:** `docs/superpowers/specs/2026-08-19-narrow-frontier-safe-belt-design.md`

## Global Constraints

- Ground platforms only; HOPPER behavior and candidate identity remain unchanged.
- One complete frontier segment produces at most three candidates.
- Online candidate generation must not read coverability denominator or unobserved truth.
- Detail work is bounded to three 0.2 m strips of at most 153 cells per segment.
- A source change starts worker episodes fresh but preserves optimizer and completed update state.

---

### Task 1: Deterministic bounded detail-strip selector

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Modify: `training/lunar_policy_training/tests/test_reachable_constrained_ground_candidates.py`

**Interfaces:**
- Produces `_select_narrow_frontier_strip_positions(...) -> _NarrowFrontierStripSelection`.
- Consumes contiguous coarse observed mask and same-shape detailed observed/safe/clearance arrays.

- [x] **Step 1: Write the failing tests**

  Test an observed safe high-clearance detailed pose is selected over an unobserved or unsafe neighbor; test a 39-cell segment evaluates exactly 27 cells for a 3-distance, 3-wide strip.

- [x] **Step 2: Verify RED**

  Run: `pytest -q training/lunar_policy_training/tests/test_reachable_constrained_ground_candidates.py`

  Expected: import failure because `_select_narrow_frontier_strip_positions` is absent.

- [x] **Step 3: Implement the bounded selector**

  Use quarter/half/three-quarter anchors, four-neighbour unknown normal, observed-side strips and lexicographic `(negative clearance, distance, lateral offset, row, column)` selection.

- [x] **Step 4: Verify GREEN and benchmark**

  Run the focused pytest file and a 1000-segment in-memory benchmark. Require exactly 153 examined cells per segment for a 17-cell, width-three strip and no full-map loop.

### Task 2: Observed-detail strip authority and candidate identity v3

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Modify: `training/lunar_policy_training/tests/test_formal_ground_endpoint_authority.py`
- Modify: `training/lunar_policy_training/tests/test_reachable_constrained_ground_candidates.py`

**Interfaces:**
- Produces a ground-only provider that maps coarse frontier anchors to exact observed-safe detail target positions.
- Candidate ID v3 contains exact ground `x_mm`, `y_mm`, and `z_mm`.

- [x] **Step 1: Write failing integration tests**

  Build a real observed-detail window containing a narrow safe strip and assert that a ground universe emits its exact target, rejects an unobserved high-clearance point, and keeps the candidate count at three. Assert two targets in one coarse cell have different IDs.

- [x] **Step 2: Verify RED**

  Run the two focused test files and confirm the current coarse-only endpoint provider cannot supply a detailed target or unique identity.

- [x] **Step 3: Add provider and identity migration**

  Group anchors by the existing 48 m window bucket, construct one observed-only detail map/projection per group, invoke Task 1, and batch-return exact positions. Pass those positions through global cost, endpoint, gain and candidate construction. Bump only ground candidate schema to v3.

- [x] **Step 4: Verify GREEN**

  Run candidate and formal endpoint focused suites; assert no provider is called for HOPPER. The historical full formal-builder module currently has an unrelated collection defect (`scope_formal_task_area` is absent), so this change verifies the focused builder-boundary suite instead.

### Task 3: Performance and recovery cutover

**Files:**
- Verify only: touched focused tests, source diff, and source-migrated resume artifact outside Git.

- [ ] **Step 1: Measure real candidate refresh**

  Compare fixed-scene ground refresh before and after. Accept only if detail provider stays bounded and no candidate refresh regression exceeds the current per-macro action budget.

- [ ] **Step 2: Verify final source**

  Run focused candidate/formal tests, `python -m py_compile` on touched Python files, and `git diff --check`.

- [ ] **Step 3: Commit and migrate**

  Commit only the spec, plan, source and focused tests. Generate a source-migrated checkpoint from the most recent completed update, preserving model/optimizer/scheduler/RNG/global step and clearing worker episode state.

- [ ] **Step 4: Atomic training cutover**

  Stop only at an update boundary, switch source and resume from the migrated checkpoint with fresh episodes. Verify source SHA, global step and first worker macro boundary.
