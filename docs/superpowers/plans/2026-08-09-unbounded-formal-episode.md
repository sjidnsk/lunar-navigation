# Unbounded Formal Episode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the formal eight-decision and optimizer-boundary reset limits while preserving fixed-size PPO rollouts, exact active-episode resume, and finite auditable evaluation.

**Architecture:** Upgrade the active observation schema to V3 without a remaining-budget feature, make the sensor boundary own ROI success, and let each worker retain a variable-length episode across policy updates. Add a JSON-compatible reveal-replay state exchanged through the existing worker protocol so update-boundary checkpoints restore the exact observation, candidates, and first request. Formal evaluation runs to natural terminal; rollout horizon is selected independently by a 16/32/64 calibration.

**Tech Stack:** Python 3.10, NumPy, PyTorch PPO/GAE, multiprocessing spawn/shared tensors, pybind11 C++20 planner bridge, pytest, ROS 2 Humble, canonical JSON checkpoints.

## Global Constraints

- Work only in `/mnt/data/WS/.lunar-navigation-worktrees/formal-training-environment-closure` on `feature/formal-training-environment-closure`.
- Use `apply_patch` for repository edits; preserve the main checkout and every unrelated worktree.
- Keep cache, benchmark, checkpoint and runtime artifacts under `/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure`.
- Source ROS 2 Humble and the current external Release bridge before bridge-backed tests; temporarily disable nounset while sourcing.
- Use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for Python tests.
- Do not start the formal seed-4080 training run.
- Every production behavior change follows RED -> observed failure -> GREEN -> focused regression -> commit.

---

### Task 1: Upgrade the active observation contract to V3

**Files:**

- Modify: `model_contract/lunar_model_contract/observation.py`
- Modify: `model_contract/lunar_model_contract/__init__.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/policy/observation.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/policy/observation_core.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/policy/cross_attention.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/checkpoint.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/observation_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/eval/baselines.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/formal_preflight.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/ppo/collector.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/ppo/rollout.py`
- Test: `model_contract/tests/test_observation_contract.py`
- Test: `training/lunar_policy_training/tests/test_collector.py`
- Test: `training/lunar_policy_training/tests/test_hopper_macro_step.py`
- Test: `training/lunar_policy_training/tests/test_policy_forward.py`
- Test: `training/lunar_policy_training/tests/test_ppo_training.py`
- Test: `training/lunar_policy_training/tests/test_reward.py`
- Test: `training/lunar_policy_training/tests/test_sensor_closed_loop.py`
- Test: `training/lunar_policy_training/tests/test_training_smoke.py`
- Test: `training/lunar_policy_training/tests/test_v3_environment.py`
- Test: `training/lunar_policy_training/tests/test_observation_builder_v2.py`
- Test: `training/lunar_policy_training/tests/test_checkpoint_resume.py`

**Interfaces:**

- Produces: `ObservationContractV3.version == "lunar-observation-contract/v3"`.
- Produces: `ObservationContractV3.shapes["pose_features"] == (None, 5)` and five `pose_fields`.
- Produces: active checkpoint schema `lunar-ppo-checkpoint/v6`; V5 remains an explicit V2 legacy reader and cannot resume a formal V3 run.
- Preserves: seven input tensor names and every non-pose shape.

- [ ] **Step 1: Write failing contract and forward tests**

```python
assert ObservationContractV3.pose_fields == (
    "x_norm", "y_norm", "sin_yaw", "cos_yaw", "mission_observed_ratio"
)
assert ObservationContractV3.shapes["pose_features"] == (None, 5)
assert CrossAttentionPolicy().pose_encoder[0].in_features == 5
```

- [ ] **Step 2: Run the RED tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  model_contract/tests/test_observation_contract.py \
  training/lunar_policy_training/tests/test_policy_forward.py \
  training/lunar_policy_training/tests/test_observation_builder_v2.py
```

Expected: failure because `ObservationContractV3` does not exist and the pose encoder still consumes six values.

- [ ] **Step 3: Implement the V3 schema and five-value builder**

Keep `ObservationContractV2` as a legacy constant, introduce `ObservationContractV3`, make active validators use V3, build:

```python
pose_features = np.asarray([[
    x_norm, y_norm, math.sin(yaw), math.cos(yaw), observed_ratio
]], dtype=np.float32)
```

and change `nn.Linear(6, TOKEN_DIM)` to `nn.Linear(5, TOKEN_DIM)`.
Set new checkpoints to `lunar-ppo-checkpoint/v6`; retain V5 only for explicit legacy inspection or migration, never as an active formal V3 resume format.

- [ ] **Step 4: Migrate active imports and fixtures, then run GREEN tests**

Update the active consumers and fixtures listed in this task to import V3. Test-only tensors use shape `(B,5)`. Run the RED command plus:

```bash
rg -n "ObservationContractV2|pose_features.*6|remaining_decision_budget_ratio" \
  training/lunar_policy_training/lunar_policy_training model_contract
```

Only explicit legacy declarations/readers may remain.

- [ ] **Step 5: Commit the contract migration**

```bash
git add model_contract training/lunar_policy_training
git commit -m "feat(training): remove decision budget from observation contract"
```

### Task 2: Make ROI coverage the formal success authority

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/training_semantics.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/observation_boundary.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py`
- Test: `training/lunar_policy_training/tests/test_sensor_closed_loop.py`
- Test: `training/lunar_policy_training/tests/test_v3_environment.py`
- Test: `training/lunar_policy_training/tests/test_hopper_macro_step.py`

**Interfaces:**

- Produces: `FORMAL_SUCCESS_COVERAGE_RATIO = 0.95`.
- Produces: `BoundaryObservationResult.mission_observed_ratio: float`.
- Produces: `BoundaryObservationResult.success_first_crossing: bool`.

- [ ] **Step 1: Write failing first-crossing tests**

Use a real `SensorObservationState` fixture whose first action moves coverage from `0.94` to `0.96`. Assert:

```python
assert first.success_first_crossing is True
assert first.mission_observed_ratio == pytest.approx(0.96)
assert repeated.success_first_crossing is False
```

Add a hopper test proving only `LANDED_HOLD` can trigger the crossing.

- [ ] **Step 2: Run the RED tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_sensor_closed_loop.py \
  training/lunar_policy_training/tests/test_v3_environment.py \
  training/lunar_policy_training/tests/test_hopper_macro_step.py
```

Expected: failure because the boundary does not expose cumulative coverage and formal transitions copy executor `False`.

- [ ] **Step 3: Implement cumulative coverage and transition ownership**

Track actual mission area and revealed mission area in the controller. At every reveal compute:

```python
crossing = previous < FORMAL_SUCCESS_COVERAGE_RATIO <= current
```

When applying a sensor boundary, replace executor fields with the boundary result and set `terminated=crossing`.

- [ ] **Step 4: Run GREEN and reward tests**

Run the RED command plus:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_reward.py
```

- [ ] **Step 5: Commit the success state machine**

```bash
git add training/lunar_policy_training
git commit -m "fix(training): derive formal success from observed ROI"
```

### Task 3: Remove production decision exhaustion

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/ppo/collector.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/proxy_scenario.py`
- Test: `training/lunar_policy_training/tests/test_v3_environment.py`
- Test: `training/lunar_policy_training/tests/test_parallel_pool.py`
- Test: `training/lunar_policy_training/tests/test_collector.py`

**Interfaces:**

- Produces: `DecisionBoundaryResult.policy_decisions_consumed: int`.
- Produces: `ParallelRolloutStep.policy_decisions_consumed: torch.Tensor`.
- Removes: production `total_decision_budget`, `remaining_decision_budget` and `DECISION_BUDGET_EXHAUSTED`.

- [ ] **Step 1: Write a failing greater-than-eight action test**

Use a real V3 environment fixture that returns a refreshed positive candidate after every action. Execute nine actions and assert the ninth result consumes one policy decision and is not terminal because of a count.

- [ ] **Step 2: Write a failing no-candidate natural-terminal test**

After a valid action returns an all-false next candidate mask, assert `episode_ended_without_success=True`, `terminated=True`, and no budget-exhausted state.

- [ ] **Step 3: Run RED tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_v3_environment.py \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_collector.py
```

Expected: the ninth action reaches `DECISION_BUDGET_EXHAUSTED`; collector still indexes `pose_features[:,5]`.

- [ ] **Step 4: Remove budget branches and rename the count**

`refresh_decision_boundary()` returns only `DECISION_READY` or `NO_CANDIDATES`. Collector availability becomes:

```python
return ~observations.candidate_mask.any(dim=1)
```

Keep bounded proxy/test behavior inside test-only vector environments, not the production V3 constructor.

- [ ] **Step 5: Run GREEN tests and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_v3_environment.py \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_collector.py \
  training/lunar_policy_training/tests/test_curriculum.py
git add training/lunar_policy_training
git commit -m "fix(training): remove formal decision exhaustion"
```

### Task 4: Preserve active episodes across PPO updates

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py`
- Test: `training/lunar_policy_training/tests/test_training_smoke.py`
- Test: `training/lunar_policy_training/tests/test_parallel_pool.py`
- Test: `training/lunar_policy_training/tests/test_collector.py`

**Interfaces:**

- Produces: policy-version advancement without episode replacement.
- Preserves: per-worker scene identity and episode cursor until natural terminal.

- [ ] **Step 1: Write a failing cross-update identity test**

Collect one two-step rollout, perform an optimizer update, collect another two-step rollout, and assert:

```python
assert second_identity.episode_id == first_identity.episode_id
assert second_cursor == first_cursor
assert second_policy_version == first_policy_version + 1
```

- [ ] **Step 2: Run RED tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_training_smoke.py \
  training/lunar_policy_training/tests/test_parallel_pool.py
```

Expected: formal update invokes `rollover_all_workers()` and changes the episode identity.

- [ ] **Step 3: Delete optimizer-boundary rollover**

After `trainer.update()` and `scheduler.step()`, increment policy version and retain `_ParallelPoolVectorEnv._current`. Keep per-worker auto-reset only for terminal transitions. Remove the formal calibration rollover for the same reason.

- [ ] **Step 4: Prove step 33 continuity**

Use a lightweight real vector environment and horizon 32 followed by horizon 1. Assert action 33 sees the same episode identity and accumulated observation value while using the next policy version.

- [ ] **Step 5: Run GREEN tests and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_training_smoke.py \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_collector.py
git add training/lunar_policy_training
git commit -m "fix(training): keep formal episodes across PPO updates"
```

### Task 5: Checkpoint and restore an active episode

**Files:**

- Create: `training/lunar_policy_training/lunar_policy_training/environment/formal_episode_state.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/observation_boundary.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/capability_freeze.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/checkpoint.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Create: `training/lunar_policy_training/tests/test_formal_resume_state.py`
- Modify: `training/lunar_policy_training/tests/test_checkpoint_resume.py`

**Interfaces:**

- Produces: `FormalWorkerState.to_dict() -> dict[str, object]` and `from_dict()` strict validation.
- Produces: `ParallelEnvPool.snapshot_episode_states(policy_version: int) -> tuple[dict[str, object], ...]`.
- Produces: `lunar-formal-environment-state/v2` with `worker_episode_states`.

- [ ] **Step 1: Write failing state schema tests**

Build a literal worker state with scene/start/current pose/reveal history/rejected indices and reject missing, extra, non-finite, mismatched platform/cursor or non-stable execution-state fields.

- [ ] **Step 2: Write the failing uninterrupted-versus-resume test**

Run update 1, snapshot/save/load, then compare uninterrupted and restored update 2 for exact model, optimizer, RNG, observation identity, all seven tensors, candidate tensors and first prepared planner-request digest.

- [ ] **Step 3: Run RED tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_formal_resume_state.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py
```

Expected: no active-worker snapshot command or v2 environment-state schema exists.

- [ ] **Step 4: Implement reveal-replay state**

Record each completed boundary as literal pose, elapsed time, execution state and legged body z. On restore, load the frozen scene/start, replay reveals without replaying planners, rebuild observation/candidates, apply rejected candidate indices, and verify stored identity/digest.

- [ ] **Step 5: Implement the safe-boundary worker RPC**

Add a `snapshot_episode_state` command. Reject any identity outside `DECISION_BOUNDARY`, `GROUND_HOLD`, or `LANDED_HOLD`. Return JSON-compatible mappings through the existing result queue.

- [ ] **Step 6: Bump the formal environment-state schema and wire CLI save/restore**

Use `lunar-formal-environment-state/v2` inside the V6 checkpoint introduced by Task 1. Formal restore passes the saved worker states before processes start; V2 observation/V5 checkpoints cannot resume a formal V3 run.

- [ ] **Step 7: Remove nondeterministic time and CUDA kernels from formal reward replay**

Use certified trajectory/hop duration divided by `2.0 s` as the formal macro-step time contribution. Exclude planner wall time from reward while retaining it in performance evidence. Replace adaptive average pooling with fixed-grid average pooling and require the fixed cuBLAS workspace plus deterministic PyTorch/cuDNN algorithms. Prove the uninterrupted/restored rollout fields, model, optimizer and RNG are exact at update two.

- [ ] **Step 8: Run GREEN tests and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_formal_resume_state.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_formal_builder.py
git add training/lunar_policy_training
git commit -m "feat(training): resume active formal episodes exactly"
```

### Task 6: Run formal evaluation to natural terminal

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/evaluation/report.py`
- Modify: `training/lunar_policy_training/tests/test_formal_evaluation.py`
- Modify: `training/lunar_policy_training/tests/test_evaluation_determinism.py`

**Interfaces:**

- Produces: ROI-weighted `mission_coverage_ratio(PolicyBatch) -> np.ndarray`.
- Produces: formal batch failure reason `EVALUATION_INCOMPLETE` when the watchdog fires.

- [ ] **Step 1: Write failing variable-length evaluation tests**

Use a real deterministic evaluation environment that terminates at step 5. Assert completion step 5 and ROI-weighted coverage `0.95`; separately prove padding outside ROI does not lower coverage.

- [ ] **Step 2: Write a failing watchdog classification test**

Use a never-terminal fixture and assert the evaluator raises/returns `EVALUATION_INCOMPLETE` without emitting a normal failed scenario metric.

- [ ] **Step 3: Run RED tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_formal_evaluation.py \
  training/lunar_policy_training/tests/test_evaluation_determinism.py
```

Expected: fixed `range(1,4)` ends at three and coverage averages the whole canvas.

- [ ] **Step 4: Implement terminal-driven formal evaluation**

Loop until every original row has terminal evidence; freeze completed rows while reset rows are used only to keep the synchronous pool actionable. Compute:

```python
coverage = (observed * roi).sum((1, 2)) / roi.sum((1, 2))
```

Use the watchdog only to abort report construction.

- [ ] **Step 5: Run GREEN tests and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_formal_evaluation.py \
  training/lunar_policy_training/tests/test_evaluation_determinism.py \
  training/lunar_policy_training/tests/test_release_gate.py
git add training/lunar_policy_training
git commit -m "fix(training): evaluate complete formal episodes"
```

### Task 7: Calibrate rollout horizon and regenerate readiness evidence

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/formal_preflight.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/tests/test_formal_preflight.py`
- Modify: `training/configs/rtx4080_super_v3_joint.yaml`
- Modify: `docs/validation/2026-08-08-formal-training-preflight-review.md`
- Modify: `docs/validation/2026-08-08-formal-training-environment-qualification.md`
- Modify: `docs/migration/volume-3-pretraining-readiness.md`

**Interfaces:**

- Produces: calibration evidence for horizon candidates `(16,32,64)` under equal total transitions.
- Produces: one selected horizon in the formal calibration manifest and frozen config.

- [ ] **Step 1: Write failing equal-work horizon calibration tests**

Assert every candidate consumes the same total transition count, records throughput/memory/update latency/KL/resume digest, and selection rejects any candidate with semantic or resume failure.

- [ ] **Step 2: Run RED tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_formal_preflight.py \
  training/lunar_policy_training/tests/test_cli.py
```

- [ ] **Step 3: Implement candidate calibration and manifest selection**

Keep horizon out of environment constructors and terminal logic. Find the highest-throughput qualified candidate, treat candidates within 1% of that throughput as engineering near-ties, then prefer the smaller update latency and smaller horizon.

- [ ] **Step 4: Run all affected and full test suites**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  model_contract/tests training/lunar_policy_training/tests
python3 tools/check_repository_boundaries.py .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/foundation/test_repository_boundaries.py
git diff --check
```

- [ ] **Step 5: Regenerate external preflight/calibration evidence**

Use a new source-commit artifact root. Run calibration before formal preflight, require preflight to consume the calibrated root and record the same workers, micro-batch and horizon, reopen every JSON/checkpoint with production loaders, and do not run `train`.

- [ ] **Step 6: Update readiness documents and commit**

Document the selected horizon, active V3 contract, unbounded episode semantics, exact artifact paths/hashes and `training-not-started` status.

```bash
git add training model_contract docs
git commit -m "docs: qualify unbounded formal episode training"
```

### Task 8: Review and integration gate

**Files:**

- Review: `model_contract/lunar_model_contract/observation.py`
- Review: `training/lunar_policy_training/lunar_policy_training/environment/`
- Review: `training/lunar_policy_training/lunar_policy_training/ppo/`
- Review: `training/lunar_policy_training/lunar_policy_training/checkpoint.py`
- Review: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Review: `training/lunar_policy_training/lunar_policy_training/evaluation/report.py`
- Review: `training/configs/rtx4080_super_v3_joint.yaml`
- Review: `docs/validation/2026-08-08-formal-training-preflight-review.md`
- Review: `docs/validation/2026-08-08-formal-training-environment-qualification.md`
- Review: `docs/migration/volume-3-pretraining-readiness.md`

**Interfaces:**

- Produces: a clean feature branch that is eligible for fast-forward integration only after all gates pass.

- [ ] **Step 1: Run focused mutation review**

Verify tests fail if each of these regressions is reintroduced: default 8, optimizer rollover, pose width 6, cursor-only restore, fixed three-step evaluation, whole-canvas coverage.

- [ ] **Step 2: Request code review and address substantiated findings**

Review specifically for lifecycle closure, semantic uniqueness, checkpoint determinism and documentation consistency. Do not expand into unrelated safety or quality work.

- [ ] **Step 3: Run final verification**

Run the full Python suite, current Release colcon suite, external-interface check, capability-freeze check, repository boundaries, UTF-8 reads and `git diff --check`.

- [ ] **Step 4: Finish the branch**

If the branch is clean and every gate is green, fast-forward into local `integration`, rerun merged-head smoke/boundary checks, and leave remote push/publication untouched.
