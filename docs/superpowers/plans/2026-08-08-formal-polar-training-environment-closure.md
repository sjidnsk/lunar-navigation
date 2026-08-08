# Formal Polar Training Environment Closure Implementation Plan

> **2026-08-09 supersession:** Task 6's all-worker optimizer-boundary rollover,
> cursor-only resume evidence, fixed-budget episode behavior, and fixed-step formal
> evaluation are superseded by
> [`2026-08-09-unbounded-formal-episode.md`](2026-08-09-unbounded-formal-episode.md).
> Completed data/cache/sensor/platform work in this plan remains authoritative.

> **For Codex:** execute this plan task by task with TDD. Generated data and runtime artifacts must stay under `/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure`; do not start the 24-hour seed-4080 run.

**Goal:** Make the locked lunar-polar data and current three-platform C++ v3 stack executable through one formal `prepare-data -> calibrate -> train/resume -> evaluate` environment, then produce current-host preflight evidence and mark the repository ready for—but not running—formal training.

**Architecture:** A canonical scenario manifest freezes the NASA/JAXA split and deterministic hazard bank. A content-addressed external cache stores 4 m global truth plus current capability-v2 C++ projections, while a deterministic 0.2 m tile provider supplies local planning, reveal, and the 32×32 network crop. One picklable formal worker builder owns observed-only state, request construction, reference execution, and sensor boundaries. Every public formal CLI path loads the same cache/builder and binds its digest into `RunIdentity`.

**Tech Stack:** Python 3.10, NumPy, rasterio 1.4.4, Shapely 2.1.2, PyTorch, ROS 2 Humble, pybind11 C++20 `lunar_planner_training_bridge`, pytest, colcon Release.

---

## Execution invariants

- Work only in `/mnt/data/WS/.lunar-navigation-worktrees/formal-training-environment-closure` on `feature/formal-training-environment-closure`.
- Preserve `/mnt/data/WS/lunar-navigation/.vscode/` and all unrelated worktrees.
- Source `/opt/ros/humble/setup.bash` and the current external Release bridge before bridge-backed tests; source with nounset temporarily disabled.
- Set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` so ROS `launch_testing` does not pollute the Volume 3 venv.
- Use `apply_patch` for repository edits. Cache, manifests, benchmark JSON, checkpoints and logs go outside Git.
- A `preflight` cache may be used by unit/process tests, but only a verified `full` cache may authorize formal calibrate.

## Task 1: Freeze scenario and geometry contracts

**Files:**

- Create: `training/lunar_policy_training/lunar_policy_training/polar_data/scenario_manifest.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/raster.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/__init__.py`
- Create: `training/lunar_policy_training/tests/test_formal_scenario_manifest.py`
- Modify: `training/lunar_policy_training/tests/test_polar_raster.py`

1. Write failing tests for exact NASA counts `1536/96/96`, the six non-overlay JAXA holdouts, stable ordering, independent seed families, source/split hash validation and canonical manifest SHA-256.
2. Write failing tests for `1024 m @ 4 m`, `64 m @ 0.2 m`, and `6.4 m @ 0.2 m / 32×32`; reject ambiguous or mismatched geometry.
3. Implement immutable scenario entry/manifest types, canonical JSON read/write, manifest inventory validation and deterministic seed derivation.
4. Change the network local geometry from historical `8 m @ 0.25 m` to the approved `6.4 m @ 0.2 m`, and add a distinct 64 m local-tile geometry.
5. Run the focused tests and `git diff --check`; commit the contract.

## Task 2: Decouple vector hazards and implement deterministic 0.2 m tiles

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/hazards.py`
- Create: `training/lunar_policy_training/lunar_policy_training/polar_data/multires_scene.py`
- Create: `training/lunar_policy_training/tests/test_multires_scene.py`
- Modify: `training/lunar_policy_training/tests/test_procedural_hazards.py`

1. Write failing tests proving rock/crater/no-go streams do not reorder one another, 4 m and 0.2 m projections share one vector scene, overlapping tiles are bit-identical and the 0.2 m output is labelled synthetic subgrid.
2. Add generator v2 vector definitions with deterministic circle/crater parameters. Preserve the v1 API as a compatibility wrapper for existing tests and historical artifacts.
3. Implement vectorized global projection and fixed-coordinate 0.2 m tile generation; avoid per-cell Shapely loops on the formal path.
4. Implement a bounded per-worker LRU tile provider and exact world/grid conversions.
5. Benchmark one scene/tile locally, run focused tests, and commit.

## Task 3: Build and verify the external formal cache

**Files:**

- Create: `training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py`
- Create: `training/lunar_policy_training/tests/test_formal_cache.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`

1. Write failing tests for external-root enforcement, `preflight` versus `full` eligibility, missing/extra/hash-drift inventory failures and source/split/capability/v3 identity mismatches.
2. Implement `FormalCacheIdentity`, canonical cache manifest and atomic per-scene `.npz` writer/loader.
3. Construct truth bridge requests and call current `PlannerBridge.project_traversability()` for WHEELED, LEGGED and HOPPER. Store only static hard feasibility and normalized clearance; never cache observed masks, gains or rewards.
4. Add `prepare-data` CLI with explicit absolute source lock, split manifest and cache root. `full` materializes every NASA scene plus six JAXA holdouts; `preflight` accepts an explicit bounded scenario count and is permanently ineligible for formal calibrate.
5. Prove a second prepare against the same completed root is verification-only and produces the same manifest digest.
6. Run focused tests and commit.

## Task 4: Implement sparse 0.2 m reveal and conservative 4 m aggregation

**Files:**

- Create: `training/lunar_policy_training/lunar_policy_training/environment/multires_observation.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/observation_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/sensor_observation.py`
- Create: `training/lunar_policy_training/tests/test_multires_observation.py`
- Modify: `training/lunar_policy_training/tests/test_observation_builder_v2.py`

1. Write failing tests for 0.04 m² reward increments, overlap deduplication, obstacle occlusion, centre-subcell 4 m validity, unknown-value isolation and the exact 6.4 m local crop.
2. Implement a sparse global high-resolution observation index backed by on-demand scene tiles; do not allocate a 5120×5120 multilayer world per worker.
3. Implement a `SensorObservationState` subtype that reveals with the native 0.2 m visibility kernel, updates exact-area reward and conservatively aggregates only observed samples into the existing 4 m `TrainingObservedGrid`.
4. Let the policy builder request the current 0.2 m local crop without receiving high-resolution truth directly.
5. Run visibility, sensor closed-loop and focused tests; commit.

## Task 5: Implement the one formal worker builder

**Files:**

- Create: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Create: `training/lunar_policy_training/tests/test_formal_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/capability_freeze.py`

1. Write failing tests for deterministic platform-safe starts, same physical scene across platforms, observed-only candidates, identity-bound requests, 4 m global plus 0.2 m local maps and current capability-v2 injection.
2. Write failing execution tests for wheeled/legged trajectory endpoints and hopper nominal landing; require `JUMP_COMMITTED -> IN_FLIGHT -> LANDED_HOLD` evidence and prove two hopper episodes use identical available delta-v without cumulative fuel.
3. Implement `FormalEnvironmentBuilder`/`FormalWorkerBuilder`, `FormalEpisode`, bridge map conversion, static projection masking, candidate/observation building, request construction and reference execution.
4. Select safe starts deterministically from platform projections and reject scenes whose initial reveal cannot produce an observed-only candidate.
5. Wrap the builder in `FrozenCapabilityEnvironmentFactory`; expose one real observation template from the same code path.
6. Run all environment/capability tests and commit.

## Task 6: Make update-boundary resume deterministic

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/tests/test_parallel_pool.py`
- Create: `training/lunar_policy_training/tests/test_formal_resume_state.py`

1. Write failing tests for an explicit all-worker episode rollover at an optimizer boundary and for exact scenario cursor recovery from observation identities.
2. Add a worker command that discards the completed rollout environment and recreates every worker at a fresh episode boundary. Do not permit checkpointing a half-executed formal episode.
3. Store formal episode cursors in the checkpointed trainer state and configure the formal builder with them before spawned workers start. Development-smoke behavior remains unchanged.
4. Prove uninterrupted update 2 and resume-at-update-2 select the same scene IDs, starts, candidate tensors and first planner requests.
5. Run parallel, checkpoint, collector and resume tests; commit.

## Task 7: Close formal calibrate/train/resume

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/curriculum.py`
- Modify: `training/lunar_policy_training/tests/test_cli.py`
- Modify: `training/lunar_policy_training/tests/test_training_smoke.py`

1. Write failing CLI tests showing formal calibrate accepts only a full cache and current sensor report, while every formal command rejects a preflight/proxy/old-identity cache before CUDA or worker creation.
2. Generalize Task 4 calibration metadata so formal runs record `proxy=false` and the cache scenario schedule ID; preserve the development-smoke proxy path.
3. Build the formal factory/template once per command and inject it into calibration, train and resume. Runtime calibration must use real formal workers, not `_CudaPlannerCalibrationWorkload` proxy workers.
4. Bind source lock SHA, split SHA, cache/generator SHA, capability SHA, reward SHA, v3 source SHA and training semantics SHA into one `RunIdentity` and run manifest.
5. Add a bounded internal calibration mode only for tests/preflight; public formal train/resume retain the unbounded 24-hour budget contract.
6. Run CLI, budget, checkpoint/resume and CUDA smoke tests; commit.

## Task 8: Close formal evaluation and preflight

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/evaluation/report.py`
- Create: `training/lunar_policy_training/lunar_policy_training/formal_preflight.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Create: `training/lunar_policy_training/tests/test_formal_evaluation.py`
- Create: `training/lunar_policy_training/tests/test_formal_preflight.py`

1. Generalize the real evaluation loop to accept a formal fixed-scenario factory while retaining proxy evaluation unchanged.
2. Evaluate validation/test/JAXA holdout outside the train split, comparing PPO and both existing baselines on identical scene/start/request sequences. Emit `proxy=false` reports.
3. Add `formal-preflight` to check three-platform worker construction, same-world/request hashes, deterministic repeat, 0.2 m reveal, no cumulative hopper fuel, update-boundary resume and non-proxy evaluation without producing a formal checkpoint.
4. Persist a canonical preflight JSON under the explicit external artifact root; include timings and selected worker/micro-batch recommendation without claiming formal training ran.
5. Run focused process tests and commit.

## Task 9: Generate current-host cache and qualification evidence

**Artifacts outside Git:**

- `/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/<source-commit>/cache`
- `/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/<source-commit>/preflight`
- `/home/kai/CodexDownloads/lunar_navigation/formal_training_environment_closure/<source-commit>/calibration`

1. Build the current bridge in a clean external Release build/install/log root and run its C++/Python tests.
2. Run `prepare-data --materialization full` against the locked real NASA/JAXA inputs. Record elapsed time, bytes, counts and manifest SHA-256.
3. Run formal-preflight with the current host-bound sensor performance evidence.
4. Run formal calibrate only far enough to freeze the validated worker/micro-batch choice. Do not run `train`.
5. Re-open every produced manifest/checkpoint with production loaders and verify that no file was written under Git.

## Task 10: Full verification, documentation and integration

**Files:**

- Modify: `docs/validation/2026-08-08-formal-training-preflight-review.md`
- Modify: `docs/migration/volume-3-pretraining-readiness.md`
- Create: `docs/validation/2026-08-08-formal-training-environment-qualification.md`
- Modify only if needed: `docs/superpowers/specs/2026-08-08-sensor-observation-capability-design.md`

1. Run all model-contract and policy-training pytest with plugin autoload disabled.
2. Run repository boundary checker/tests, UTF-8 read, `git diff --check`, and the clean Release colcon suite.
3. Review only logic closure, semantic uniqueness, design consistency and documentation authority. Correct any ambiguity or stale formal gate wording.
4. Request code review, address substantiated findings, then rerun the affected and full gates.
5. Update status to `formal-training-ready / training-not-started`, including exact artifact paths and hashes; explicitly state that seed 4080 training has not started.
6. Use the finishing-branch workflow: verify the feature branch is clean, fast-forward it into `integration`, rerun merged-head smoke/boundary checks, and leave remote push/publication untouched.
