# Formal Training Observability and Phase Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume the formal seed-4080 run from step 118 across platform curriculum boundaries while recording every completed PPO update and automatically evaluating joint candidates every three cumulative GPU hours.

**Architecture:** Separate allocation-aware environment restoration from global checkpoint restoration, add a single-writer JSONL update journal, and make joint training run in evaluation-sized chunks. Training and evaluation execute serially against one `TrainingBudget`; immutable candidate evaluation directories preserve trend evidence.

**Tech Stack:** Python 3.10, PyTorch PPO, NumPy, pytest, ROS 2 Humble native planner bridge, atomic filesystem writes, systemd user service.

## Global Constraints

- Source commit and formal cache/capability/reward/training-semantics identities remain frozen.
- Runtime artifacts stay below `/home/kai/CodexDownloads/lunar_navigation`; no checkpoint, JSONL, report, or service log enters Git.
- A curriculum allocation transition does not emit reward, done, success, or failure for retired worker episodes.
- Same-allocation checkpoint resume remains exact and fail-closed.
- Formal training and evaluation share the same 86400-second cumulative `TrainingBudget`.
- Training remains on Ubuntu 22.04, ROS 2 Humble, RTX 4080 SUPER, Python 3.10 and the frozen native bridge.
- Existing `.vscode/` and unrelated worktrees are not modified.

---

### Task 1: Allocation-aware episode restoration

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/curriculum.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Test: `training/lunar_policy_training/tests/test_curriculum.py`
- Test: `training/lunar_policy_training/tests/test_cli.py`

**Interfaces:**
- Produces: `resume_worker_episode_states(checkpoint: object, *, target_phase: str, target_allocation: Mapping[str, int]) -> tuple[Mapping[str, object], ...] | None`.
- Consumes: checkpoint `curriculum_phase`, `worker_allocation`, and `environment_state["worker_episode_states"]`.

- [ ] **Step 1: Write failing phase-transition tests**

Add tests proving same-phase/same-allocation returns the stored states, each adjacent allocation-changing transition returns `None`, and non-adjacent or same-phase allocation drift raises `ValueError`.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  training/lunar_policy_training/tests/test_curriculum.py \
  -k resume_worker_episode_states
```

Expected: FAIL because `resume_worker_episode_states` does not exist.

- [ ] **Step 3: Implement the pure selection helper**

Use the frozen order `warmup_wheeled`, `warmup_legged`, `warmup_hopper`, `joint`. Return stored states only for an exact phase/allocation match; return `None` only for the next phase with a changed allocation.

- [ ] **Step 4: Write a failing `_run_updates` integration test**

Construct a wheel checkpoint and request a legged allocation. Assert the pool constructor receives `initial_episode_states=None` while `restore_training_state` still receives the checkpoint.

- [ ] **Step 5: Verify RED, integrate, and verify GREEN**

Call the helper before `ParallelEnvPool` construction and keep top-level model/optimizer/RNG restoration unchanged. Run the two targeted test files and confirm all pass.

- [ ] **Step 6: Commit**

```bash
git add training/lunar_policy_training/lunar_policy_training/curriculum.py \
  training/lunar_policy_training/lunar_policy_training/cli.py \
  training/lunar_policy_training/tests/test_curriculum.py \
  training/lunar_policy_training/tests/test_cli.py
git commit -m "fix(training): reset episodes across platform phases"
```

### Task 2: Durable per-update training metrics

**Files:**
- Create: `training/lunar_policy_training/lunar_policy_training/training_metrics.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Test: `training/lunar_policy_training/tests/test_training_metrics.py`
- Test: `training/lunar_policy_training/tests/test_cli.py`

**Interfaces:**
- Produces: `TrainingMetricsJournal(path: Path, *, resume_global_step: int)` and `append(record: Mapping[str, object]) -> None`.
- Produces: `build_training_update_record(...) -> dict[str, object]` using `CollectedRollout`, final observations, `PPOUpdateMetrics`, planner outcomes and timing.
- Consumes: `PPOTrainer.update()` metrics and `_ParallelPoolVectorEnv` rollout diagnostics.

- [ ] **Step 1: Write failing journal contract tests**

Cover UTF-8 JSONL append, schema validation, finite-number rejection, strictly increasing steps, resume continuity, and refusal when the journal is ahead of the checkpoint.

- [ ] **Step 2: Verify RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  training/lunar_policy_training/tests/test_training_metrics.py
```

Expected: FAIL because the module is absent.

- [ ] **Step 3: Implement the minimal journal**

Write one sorted compact JSON object per completed update with `os.open(..., O_APPEND|O_CREAT)`, one `os.write`, and `os.fsync`. Re-read existing rows during construction and validate their schema and step order.

- [ ] **Step 4: Write failing update-record tests**

Use a two-platform synthetic rollout to assert raw reward summaries, coverage start/end/delta, terminal/success counts, planning outcome counts, planner success rate, PPO metrics, and wall-time fields.

- [ ] **Step 5: Implement record construction and adapter diagnostics**

Extend `_ParallelPoolVectorEnv` to retain raw rewards, success-first-crossing flags and execution outcomes for the current policy version. Keep these diagnostics outside `RolloutBatch` so the seven-input/checkpoint contract does not change.

- [ ] **Step 6: Write failing training-loop integration test**

Assert one successful update writes exactly one step, a failed update writes none, and resumed training begins at checkpoint step plus one.

- [ ] **Step 7: Integrate metrics and verify GREEN**

Return `CollectedRollout` from the collector closure, feed `.rollout` to PPO, append only after trainer/scheduler/policy-version success, and create `<artifact-root>/metrics/train.jsonl` outside the repository.

- [ ] **Step 8: Commit**

```bash
git add training/lunar_policy_training/lunar_policy_training/training_metrics.py \
  training/lunar_policy_training/lunar_policy_training/cli.py \
  training/lunar_policy_training/tests/test_training_metrics.py \
  training/lunar_policy_training/tests/test_cli.py
git commit -m "feat(training): journal every PPO update"
```

### Task 3: Joint-phase automatic evaluation scheduler

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/curriculum.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/evaluation/report.py`
- Test: `training/lunar_policy_training/tests/test_cli.py`
- Test: `training/lunar_policy_training/tests/test_formal_evaluation.py`

**Interfaces:**
- Produces: `next_joint_evaluation_gpu_seconds(manifest, *, schedule, calibration_end_gpu_seconds) -> float`.
- Produces: `_latest_candidate_checkpoint(artifact_root: Path) -> Path`.
- Extends: `_evaluate_checkpoint(..., shared_budget: TrainingBudget | None = None, evaluation_started_gpu_seconds: float | None = None)`.
- Consumes: a callback `evaluate_candidate(checkpoint: Path, budget: TrainingBudget, started_gpu_seconds: float) -> GateResult` injected into `_run_curriculum_training`.

- [ ] **Step 1: Write failing scheduler tests**

Assert no warmup evaluation, first joint due at joint start plus 10800 seconds, subsequent due from the previous evaluation start, and latest immutable candidate selection by numeric global step.

- [ ] **Step 2: Verify RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  training/lunar_policy_training/tests/test_cli.py \
  -k 'joint_evaluation or latest_candidate'
```

Expected: FAIL because scheduling helpers are absent.

- [ ] **Step 3: Add a soft joint pause boundary**

Keep the existing 600-second hard phase reserve, but add an optional evaluation target that stops only after a completed update reaches the target. Save `latest.pt`, close the pool, and return control to the curriculum driver.

- [ ] **Step 4: Write failing shared-budget evaluation tests**

Assert the evaluator mutates the exact `TrainingBudget` instance supplied by training, reserves the formal 3600-second watchdog bound, and does not construct a second authoritative budget.

- [ ] **Step 5: Integrate automatic evaluation**

Build validation/test/holdout batches from the calibrated cache, evaluate the latest candidate with `candidate_gate_v1.yaml`, resume the same joint episode state when the gate fails, and return immediately when it passes.

- [ ] **Step 6: Preserve immutable evaluation history**

Write each candidate under `evaluation/candidate-step-<N>/formal-report.json` and `formal-result.json`; also refresh the existing top-level latest files. Add started/completed GPU seconds to `last_evaluation`.

- [ ] **Step 7: Verify GREEN and commit**

```bash
git add training/lunar_policy_training/lunar_policy_training/curriculum.py \
  training/lunar_policy_training/lunar_policy_training/cli.py \
  training/lunar_policy_training/lunar_policy_training/evaluation/report.py \
  training/lunar_policy_training/tests/test_cli.py \
  training/lunar_policy_training/tests/test_formal_evaluation.py
git commit -m "feat(training): evaluate joint candidates automatically"
```

### Task 4: Real checkpoint recovery rehearsal and full verification

**Files:**
- Modify: `docs/validation/2026-08-08-formal-training-environment-qualification.md`
- Runtime-only: `/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/qualified-current`

**Interfaces:**
- Consumes: step-118 `latest.pt`, frozen manifest, native install and sensor report.
- Produces: verified resume command and live metric/evaluation paths; no runtime artifact enters Git.

- [ ] **Step 1: Run all CPU training tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  training/lunar_policy_training/tests
```

- [ ] **Step 2: Run repository boundary checks**

```bash
python3 tools/check_repository_boundaries.py .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/foundation/test_repository_boundaries.py
git diff --check
```

- [ ] **Step 3: Run the bounded CUDA resume regression**

Use the formal venv, ROS Humble setup and frozen native bridge. Run the CUDA interruption/resume test selected by `-m cuda` and verify update-boundary equivalence.

- [ ] **Step 4: Preflight the real step-118 checkpoint without an optimizer update**

Load it with the production restricted loader, derive `warmup_legged`, construct the `LEGGED×24` pool with no old worker states, reset to 24 valid decision observations, close the pool, and verify the checkpoint file hash did not change.

- [ ] **Step 5: Update qualification evidence and commit**

Record the source commit, test commands, step-118 hash, derived phase, allocation and artifact paths without copying runtime outputs into Git.

- [ ] **Step 6: Fast-forward integration and restart**

Fast-forward `integration` to the qualified feature branch. Start a transient user service with the formal `resume` command, the step-118 latest checkpoint, ROS/native setup and append-only runtime log.

- [ ] **Step 7: Verify live continuation**

Confirm the service is active, allocation becomes `LEGGED×24`, the first metric row is step 119, manifest/checkpoint progress increases, GPU/CPU/memory remain finite, and no `initial episode state worker identity differs` traceback recurs.

---

Plan self-review: all three approved changes have a RED/GREEN test cycle, runtime artifacts remain outside Git, same-allocation exact resume is preserved, and the final operational step resumes rather than restarts the formal budget.
