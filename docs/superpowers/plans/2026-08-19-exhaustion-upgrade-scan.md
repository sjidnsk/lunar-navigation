# Ground Candidate Exhaustion Upgrade Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and measure a ground-only observed-data exhaustion scan without changing live episode termination.

**Architecture:** Add a pure candidate-builder API that derives residual authorized-unobserved components, exhaustively evaluates the finite reachable observer-pose library near those components, and returns the deterministic positive subset plus diagnostics. Keep the existing normal three-per-frontier builder untouched; integration with `v3_environment` is explicitly deferred pending measurement.

**Tech Stack:** Python 3.10, NumPy, existing `CandidateBuilderV2`, existing ground global-reachability and batched visibility-estimator interfaces, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-exhaustion-upgrade-scan-design.md`

## Global Constraints

- Ground platforms only; do not alter HOPPER candidate or termination semantics.
- Never use hidden coverability/reward truth to generate or rank a candidate.
- Reuse the existing exact-gain and endpoint-feasibility contracts.
- Do not call the scan from normal candidate refresh in this change.
- Runtime artifacts and benchmark outputs remain outside Git.

---

### Task 1: Define isolated scan value types and RED tests

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Test: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`

**Interfaces:**
- Produces `ExhaustionUpgradeDiagnostics` and `ExhaustionUpgradeResult`.
- Produces `CandidateBuilderV2.build_ground_exhaustion_upgrade(...)`.

- [ ] **Step 1: Write failing tests**

```python
def test_ground_exhaustion_upgrade_emits_positive_residual_observer():
    result = builder.build_ground_exhaustion_upgrade(...)
    assert result.diagnostics.residual_component_count == 1
    assert result.diagnostics.positive_pose_count == 1
    assert len(result.candidates) == 1
```

```python
def test_ground_exhaustion_upgrade_records_reachable_zero_gain_without_emission():
    result = builder.build_ground_exhaustion_upgrade(...)
    assert result.diagnostics.reachable_pose_count == 1
    assert result.diagnostics.positive_pose_count == 0
    assert result.candidates == ()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest -q training/lunar_policy_training/tests/test_candidate_builder_v2.py -k exhaustion_upgrade`

Expected: FAIL because `build_ground_exhaustion_upgrade` does not exist.

- [ ] **Step 3: Implement immutable diagnostics and the pure residual-component helper**

```python
@dataclass(frozen=True, slots=True)
class ExhaustionUpgradeDiagnostics:
    residual_component_count: int
    reachable_pose_count: int
    endpoint_feasible_pose_count: int
    exact_gain_evaluated_pose_count: int
    positive_pose_count: int
    selected_candidate_count: int
    elapsed_s: float
```

Implement a canonical four-neighbour component function over
`(mission.roi_ratio > 0) & ~world.observed_mask`; it must sort cells and
components lexicographically.

- [ ] **Step 4: Implement the minimal ground scan**

For each component, find globally reachable physical observation cells inside
the sensor-radius stencil, apply existing endpoint feasibility, call existing
batched exact gain once over the deduplicated pose list, then select at most
three positive candidates per component using deterministic gain/cost/risk/ID
ordering.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m pytest -q training/lunar_policy_training/tests/test_candidate_builder_v2.py -k exhaustion_upgrade`

Expected: PASS.

Commit: `git add training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py training/lunar_policy_training/tests/test_candidate_builder_v2.py && git commit -m "feat(training): add ground exhaustion upgrade scan"`

### Task 2: Verify exclusion, determinism and no normal-path invocation

**Files:**
- Modify: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`
- Test: `training/lunar_policy_training/tests/test_v3_environment.py`

**Interfaces:**
- Consumes Task 1 API.
- Ensures existing `build_physical_universe` output and call path are unchanged.

- [ ] **Step 1: Write failing tests**

```python
def test_ground_exhaustion_upgrade_excludes_disconnected_and_endpoint_infeasible():
    result = builder.build_ground_exhaustion_upgrade(...)
    assert result.diagnostics.reachable_pose_count == 2
    assert result.diagnostics.endpoint_feasible_pose_count == 1
    assert result.candidates == ()
```

```python
def test_ground_exhaustion_upgrade_is_deterministic():
    assert builder.build_ground_exhaustion_upgrade(...) == builder.build_ground_exhaustion_upgrade(...)
```

```python
def test_normal_ground_refresh_never_calls_exhaustion_upgrade(monkeypatch):
    monkeypatch.setattr(CandidateBuilderV2, "build_ground_exhaustion_upgrade", fail)
    environment.build_policy_observation()
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest -q training/lunar_policy_training/tests/test_candidate_builder_v2.py training/lunar_policy_training/tests/test_v3_environment.py -k exhaustion_upgrade`

Expected: FAIL until exclusions and equality semantics are implemented.

- [ ] **Step 3: Implement exclusions and stable equality-compatible output**

Use only finite global costs, preserve endpoint-feasibility validation, and
derive IDs/ranks from component ID and canonical pose. Do not edit normal
`build_physical_universe` logic.

- [ ] **Step 4: Run focused tests and commit**

Run: `python -m pytest -q training/lunar_policy_training/tests/test_candidate_builder_v2.py training/lunar_policy_training/tests/test_v3_environment.py -k exhaustion_upgrade`

Expected: PASS.

Commit: `git add training/lunar_policy_training/tests/test_candidate_builder_v2.py training/lunar_policy_training/tests/test_v3_environment.py && git commit -m "test(training): certify exhaustion scan boundaries"`

### Task 3: Measure the isolated scan on update-307 terminal snapshots

**Files:**
- Create: external artifact script under `~/CodexDownloads/lunar_navigation/exhaustion-upgrade-scan-benchmark/`
- Test: existing focused Python tests from Tasks 1–2

**Interfaces:**
- Consumes the Task 1 scan API and the existing 300 m task runner's terminal worker states.
- Produces JSON outside Git with WHEELED/LEGGED counts, elapsed time and memory sample.

- [ ] **Step 1: Run focused tests and inspect the real terminal snapshots**

Run: `python -m pytest -q training/lunar_policy_training/tests/test_candidate_builder_v2.py -k exhaustion_upgrade`

Expected: PASS before benchmarking.

- [ ] **Step 2: Run the external benchmark**

Restore each group-1 terminal worker state, invoke only
`build_ground_exhaustion_upgrade`, record elapsed time, exact-gain batch
count, candidate counts and peak temporary array bytes. Do not invoke PPO,
planner action execution or train-state mutation.

- [ ] **Step 3: Compare against normal refresh and record decision**

Report whether the scan found positive candidates, its one-time terminal cost,
and whether normal refresh remains unchanged. If the scan violates its stated
performance gate, optimize the scan implementation before any environment
integration.

- [ ] **Step 4: Run minimal regression and commit only source/tests/docs**

Run: `python -m pytest -q training/lunar_policy_training/tests/test_candidate_builder_v2.py training/lunar_policy_training/tests/test_v3_environment.py`

Expected: PASS.

Commit: `git add docs/superpowers/specs/2026-08-19-exhaustion-upgrade-scan-design.md docs/superpowers/plans/2026-08-19-exhaustion-upgrade-scan.md && git commit -m "docs(training): specify exhaustion upgrade scan"`
