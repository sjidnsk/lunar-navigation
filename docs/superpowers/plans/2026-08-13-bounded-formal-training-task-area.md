# Bounded Formal Training Task Area Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every formal PPO episode use a deterministic random 100-500 m task area and start a fresh step-zero training run without a whole-scene gate.

**Architecture:** Freeze the task size in training config, apply a platform-specific runtime view over immutable cached scene evidence, bind the view into formal identities, and use a stratified 128-scene bounded cache with exact common three-platform evaluation lanes.

**Tech Stack:** Python 3.10, NumPy, PyYAML, pytest, ROS 2 Humble, PyTorch/CUDA.

## Global Constraints

- Task side is sampled uniformly from `100.0, 104.0, ..., 500.0 m`, aligned to `25-125` global `4.0 m` cells and `500-2500` detail `0.2 m` cells.
- Sampling uses `deterministic-uniform-square/v1` keyed by episode seed and is exact across restore.
- The physical source map is not cropped or mutated.
- Candidate, Oracle, reveal, reward and coverage use one identical scoped ROI.
- WHEELED, LEGGED and HOPPER are all scoped.
- Training starts at global step zero; no old checkpoint resume.
- Do not read, modify or stage the protected 2026-08-11 plan file.

---

### Task 1: Freeze the task-area config

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/config.py`
- Modify: `training/configs/rtx4080_super_v3_joint.yaml`
- Modify: `training/configs/rtx4080_super_smoke.yaml`
- Test: `training/lunar_policy_training/tests/test_ppo_training.py`

**Interfaces:**
- Produces: `ResolvedTrainingConfig.task_area: TaskAreaConfig`.

- [ ] Write tests requiring exact built-in float bounds `100.0` and `500.0`, the frozen sampling ID, and rejection of missing/drifted values.
- [ ] Run the focused test and observe failure because the field is unsupported.
- [ ] Add the field to parsing, validation, serialization and both supported YAML files.
- [ ] Run the focused config tests to GREEN.

### Task 2: Apply the scoped task view

**Files:**
- Create: `training/lunar_policy_training/lunar_policy_training/environment/task_area.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Test: `training/lunar_policy_training/tests/test_formal_builder.py`

**Interfaces:**
- Produces: `sample_task_area_span_cells(...) -> int`, `scope_formal_task_area(...) -> ScopedTaskArea`, and a builder-bound `TaskAreaConfig`.

- [ ] Write tests asserting deterministic inclusive 25-125-cell sampling, exact detail bounds, platform coverability hash/count, source immutability and three-platform worker use.
- [ ] Run the tests and observe failure because the scoped API is missing.
- [ ] Implement the immutable scoped arrays/entry view and route it through load/restore.
- [ ] Bind the task size into the combined scenario schedule digest.
- [ ] Run formal-builder tests to GREEN.

### Task 3: Route config through formal CLI

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Test: `training/lunar_policy_training/tests/test_cli.py`

**Interfaces:**
- Consumes: `ResolvedTrainingConfig.task_area`.
- Produces: calibrated/train/evaluation assemblies with identical task scope.

- [ ] Write tests showing calibrate, preflight, train, resume and evaluate preserve the scoped environment identity.
- [ ] Run focused CLI tests to RED.
- [ ] Pass the field into direct environment builds and reconstruct it from calibrated frozen config.
- [ ] Run focused CLI tests to GREEN.

### Task 4: Verify and launch

**Files:**
- External artifacts only under `~/CodexDownloads/lunar_navigation/formal_training_environment_closure/`.

**Interfaces:**
- Produces: fresh calibration root and live seed-4080 step-zero training root.

- [ ] Run config, formal-builder and CLI focused suites, `py_compile`, UTF-8 reads and `git diff --check`.
- [ ] Materialize the frozen 98/12/12/6 bounded cache and require exact common three-platform samples in every split.
- [ ] Build one real worker boundary for WHEELED, LEGGED and HOPPER from the bounded cache.
- [ ] Commit only task-related source, tests, configs, design and this plan.
- [ ] Generate current sensor-performance evidence if source identity requires it.
- [ ] Run formal calibration with the scoped config.
- [ ] Start formal training from global step zero without a whole-scene gate.
- [ ] Verify live process, GPU use, frozen task config, cache identity and the first finite training metrics.
