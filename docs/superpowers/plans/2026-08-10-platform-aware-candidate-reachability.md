# Platform-Aware Candidate Reachability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the universal straight-ground-ray candidate prefilter with deterministic platform-aware filtering, preserve exact pre-repair checkpoint restoration, expose per-platform diagnostics, and resume formal training without restarting from step 0.

**Architecture:** Frontier discovery and `0.2 m` observed-only gain estimation remain shared. Ground platforms use one observed platform-traversable connected-component calculation per observation; Hopper skips ground connectivity and uses its landing projection plus the existing C++ ballistic planner for final certification. Diagnostics travel out-of-band through the worker protocol and never enter policy tensors, rewards, or observation digests.

**Tech Stack:** Python 3.10, NumPy, PyTorch multiprocessing, pytest, ROS 2 Humble bridge bindings, user-level systemd.

## Global Constraints

- Work only in `/mnt/data/WS/.lunar-navigation-worktrees/formal-training-environment-closure`.
- Preserve `30 m / 360 deg`, 64 candidates, the `0.2 m` observed-only gain estimator, PPO, rewards, curriculum, 24 workers, formal identities, and the 86400 GPU-second budget.
- Do not use unobserved terrain to certify reachability or duplicate the C++ Hopper flight-tube planner in Python.
- Keep current training running until code, tests, and the new sensor report pass.
- Resume through `_prepare_source_migrated_resume_checkpoint(...)` from authoritative `latest.pt`; never edit checkpoints or manifests directly.
- Keep runtime artifacts outside the repository and do not start from step 0.

---

### Task 1: Platform-Aware Candidate Prefilter

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Test: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`

**Interfaces:**
- Produces: `CandidateDiagnostics(frontier_anchor_count, platform_filter_rejected_count, emitted_count)`.
- Produces: `CandidateBuilderV2.build(..., *, platform_type: str, platform_reachability_filter_enabled: bool = True, excluded_cells=()) -> CandidateBatch`.

- [ ] **Step 1: Write failing tests**

Create obstacle layouts proving a ground candidate survives when an observed detour exists and a Hopper candidate survives when only the ground ray is blocked. Also assert unknown platforms fail, unsafe/unobserved Hopper landings are rejected, diagnostics are consistent, and repeated builds are byte-identical.

```python
ground = builder.build(
    world, mission, pose, projection, platform_type="WHEELED"
)
hopper = builder.build(
    world, mission, pose, projection, platform_type="HOPPER"
)
assert ground.count > 0
assert hopper.count > 0
assert hopper.diagnostics.emitted_count == hopper.count
```

- [ ] **Step 2: Run RED test**

```bash
/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python -m pytest -q \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py
```

Expected: failure because the platform interface and diagnostics do not exist.

- [ ] **Step 3: Implement minimal deterministic filter**

Build the raw stable anchor set once. Preserve the old `_clear_observed(..._ray_cells...)` behavior only when the compatibility flag is false. For new ground behavior, flood-fill the intersection of observed, obstacle-free, and platform-traversable cells once using fixed four-neighbor order. For Hopper, require only observed, obstacle-free, ROI, and Hopper projection feasibility before gain estimation; leave ballistic certification to C++.

- [ ] **Step 4: Run GREEN test and commit**

```bash
/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python -m pytest -q \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py
git add training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py
git commit -m "fix(training): filter candidates by platform reachability"
```

---

### Task 2: Formal Wiring and Checkpoint Compatibility

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Test: `training/lunar_policy_training/tests/test_formal_builder.py`
- Test: `training/lunar_policy_training/tests/test_formal_resume_state.py`

**Interfaces:**
- Consumes: Task 1 platform/filter inputs.
- Produces: `FormalEpisode.current_candidate_diagnostics() -> CandidateDiagnostics` and one-boundary legacy restore.

- [ ] **Step 1: Write failing wiring and restore tests**

Assert FormalEpisode passes its exact platform. Build stored state with the legacy filter, then assert restore reproduces that exact digest and the next observation enables the new filter.

- [ ] **Step 2: Run RED test**

```bash
/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python -m pytest -q \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_formal_resume_state.py
```

- [ ] **Step 3: Implement wiring and ordered fallback**

Add `platform_candidate_reachability_enabled: bool = True` to FormalEpisode. Pass platform and flag to the builder. Attempt restore in this order:

```python
(True, True, True)    # visited, detail gain, platform reachability
(True, True, False)   # current checkpoint legacy boundary
(True, False, False)
(False, False, False)
```

Only the two existing compatibility errors may fall through. After a legacy match, flip all repairs to true without rebuilding the verified boundary.

- [ ] **Step 4: Run GREEN test and commit**

```bash
/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python -m pytest -q \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_formal_resume_state.py
git add training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_formal_resume_state.py
git commit -m "fix(training): migrate platform candidate boundaries"
```

---

### Task 3: Per-Platform Diagnostics

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/training_metrics.py`
- Test: `training/lunar_policy_training/tests/test_parallel_pool.py`
- Test: `training/lunar_policy_training/tests/test_training_metrics.py`
- Test: `training/lunar_policy_training/tests/test_cli.py`

**Interfaces:**
- Consumes: current candidate diagnostics and ordered worker platforms.
- Produces: additive `candidate.by_platform` and `planner.outcome_counts_by_platform` metrics.

- [ ] **Step 1: Write failing protocol and metric tests**

Require one typed diagnostics record per action row and a no-candidate flag during boundary preparation. Assert per-platform anchor, filter, emitted, no-candidate termination, planner-rejected exhaustion, and planner outcome counts.

- [ ] **Step 2: Run RED test**

```bash
/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python -m pytest -q \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_training_metrics.py \
  training/lunar_policy_training/tests/test_cli.py
```

- [ ] **Step 3: Implement out-of-band transport**

Capture diagnostics before an action mutates the observation and before auto-reset replaces a no-candidate episode. Extend worker messages, `_await_step`, `_await_resolution`, `ParallelRolloutStep`, and `_ParallelPoolVectorEnv` with strict validation. Keep shared policy buffers unchanged. Aggregate action rows by `worker_platforms[index % worker_count]` and derive rejected exhaustion from aligned dones plus non-reference outcomes. Retain metrics schema v1 because fields are additive and old journal rows remain valid.

- [ ] **Step 4: Run GREEN test and commit**

```bash
/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python -m pytest -q \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_training_metrics.py \
  training/lunar_policy_training/tests/test_cli.py
git add training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py \
  training/lunar_policy_training/lunar_policy_training/cli.py \
  training/lunar_policy_training/lunar_policy_training/training_metrics.py \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_training_metrics.py \
  training/lunar_policy_training/tests/test_cli.py
git commit -m "feat(training): report platform candidate diagnostics"
```

---

### Task 4: Regression and Performance Gate

**Files:**
- Verify all Task 1-3 files.
- External output: a new timestamped directory under `/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/`.

- [ ] **Step 1: Run focused, package, and boundary tests**

```bash
/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python -m pytest -q \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_formal_resume_state.py \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_training_metrics.py \
  training/lunar_policy_training/tests/test_training_smoke.py
/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python -m pytest -q \
  training/lunar_policy_training/tests
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
git diff --check
```

Expected: all pass.

- [ ] **Step 2: Generate formal sensor evidence**

After the source is clean and committed, run:

```bash
source /opt/ros/humble/setup.bash
source /home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/df27f02/native/install/setup.bash
export PYTHONPATH=/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/detail-candidate-gain/native/install/local/lib/python3.10/dist-packages:$PWD/model_contract:$PWD/training/lunar_policy_training
NEW_COMMIT=$(git rev-parse HEAD)
EXTERNAL_DIR=/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/platform-reachability-${NEW_COMMIT:0:12}
mkdir -p "$EXTERNAL_DIR"
/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python \
  training/tools/benchmark_sensor_observation.py \
  --native-benchmark /home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/detail-candidate-gain/native/install/lib/lunar_planner_training_bridge/lunar_training_visibility_benchmark \
  --output "$EXTERNAL_DIR/sensor-performance-${NEW_COMMIT:0:12}.json"
```

Expected: `passed=true`, candidate p95 below 5 ms, reveal p95 below 2 ms, and throughput drop no more than 0.10.

- [ ] **Step 3: Verify descendant and clean status**

```bash
git merge-base --is-ancestor 3251aefc4379b848628ffc0c648ba63d27eae213 HEAD
git status --short --branch
```

---

### Task 5: Controlled Migration and Resume

**Files:**
- Consume external `calibration/checkpoints/latest.pt` and `run-manifest.json`.
- Create a timestamped external recovery launcher/log directory.

- [ ] **Step 1: Gracefully stop after all gates pass**

```bash
systemctl --user stop lunar-formal-training-detail-gain-3251aef-resume.service
```

Wait up to the configured six-minute timeout. Verify the final metrics row, `latest.pt` and manifest have the same step/source/budget identity.

- [ ] **Step 2: Run audited source migration**

Invoke `_prepare_source_migrated_resume_checkpoint` with authoritative `latest.pt`, old source `3251aefc4379b848628ffc0c648ba63d27eae213`, the stopped step read from that checkpoint, previous report `sensor-performance-3251aef.json`, and the new report. Expected: immutable old backup, a resume checkpoint whose filename contains the stopped step and the first 12 characters of `git rev-parse HEAD`, and an appended manifest migration.

- [ ] **Step 3: Start the new user service**

Create an external absolute-path launcher matching the current ROS/native/PYTHONPATH/CUDA/thread environment. Start with:

```bash
NEW_COMMIT=$(git rev-parse HEAD)
UNIT="lunar-formal-training-platform-reachability-${NEW_COMMIT:0:7}-resume"
RECOVERY_DIR="/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/reward-v3-69453ec/calibration/recovery/platform-reachability-${NEW_COMMIT:0:12}"
LAUNCHER="$RECOVERY_DIR/resume-${NEW_COMMIT:0:12}.sh"
systemd-run --user \
  --unit="$UNIT" \
  --property=WorkingDirectory=/home/kai \
  --property=KillMode=mixed \
  --property=TimeoutStopSec=6min \
  "$LAUNCHER"
```

- [ ] **Step 4: Verify one new update**

Confirm `active/running`, zero restarts, one main plus 24 workers and resource tracker, GPU presence, empty/error-free runtime log, no OOM/Xid, contiguous metrics, and the first row beyond the stopped step containing the new candidate and per-platform planner fields. Report exact commits, steps, checkpoint, service and gates without claiming convergence from one update.
