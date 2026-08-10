# Platform-Coverable Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the full-ROI coverage denominator with an exact platform/start-specific `0.2 m` coverable mask, make platform reachability and candidate exhaustion auditable, and prepare a new step-0 run that warm-starts only reusable policy weights.

**Architecture:** C++ remains the authority for platform traversability and Hopper single-hop certification. Cache v4 binds one deterministic start, an exact `4.0 m` reachable-pose mask, and a bit-packed `0.2 m` coverable-detail mask to every scene/platform. Runtime coverage intersects observed detail bits with that frozen mask, while candidate gain and an independent frontier oracle remain observed-only. Training uses per-platform eligible scene lanes and a new run identity; old checkpoints are inert warm-start sources rather than resumable environment state.

**Tech Stack:** C++20, ROS 2 Humble, pybind11, Python 3.10, NumPy, PyTorch PPO, pytest, GoogleTest, canonical JSON/NPZ cache artifacts.

## Global Constraints

- Work only in `/mnt/data/WS/.lunar-navigation-worktrees/formal-training-environment-closure` on `feature/formal-training-environment-closure`.
- Use `apply_patch` for repository edits and preserve unrelated user changes.
- Keep build, install, log, cache, checkpoint, report and benchmark artifacts under `/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration`; never write generated artifacts into the repository.
- Source `/opt/ros/humble/setup.bash` and verify `ROS_DISTRO=humble` before native ROS commands. Use production-like `--merge-install` for final native verification.
- Use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for Python tests and the existing Volume 3 virtual environment when available.
- Every production behavior change follows RED -> observe the intended failure -> GREEN -> focused regression -> commit.
- Do not alter the PPO network topology, reward weights, `1024 m` scene, `30 m / 360 deg` sensor, `0.2 m` reveal, `[64,12]` candidate contract, or unbounded episode lifetime.
- Do not start formal training until the v4 preflight cache and 24-scene x 3-platform closed-loop gate pass. A later training launch must create a new step-0 run and may only warm-start approved policy prefixes.

---

### Task 1: Freeze v6 semantics and the coverability data contracts

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/training_semantics.py`
- Create: `training/lunar_policy_training/lunar_policy_training/environment/coverability.py`
- Create: `training/lunar_policy_training/tests/test_coverability.py`
- Modify: `training/lunar_policy_training/tests/test_checkpoint_resume.py`
- Modify: `training/lunar_policy_training/tests/test_formal_cache.py`

**Interfaces:**

- Produces: `FORMAL_TRAINING_SEMANTICS_VERSION == "lunar-training-semantics/sensor-30m-360-platform-coverable-detail95-unbounded-per-platform-subset/v6"`.
- Produces: `PlatformCoverability` with validated start cell, reachable mask, packed detail mask, coarse ratio, counts, fractions, algorithm IDs, hashes, exact flag, eligibility and enum reason.
- Produces: row-major `pack_detail_mask()` / `unpack_detail_mask()` that reject padding-bit, shape, count or hash drift.
- Produces: fixed ordinary ineligibility reasons and fail-closed separation from infrastructure errors.

- [ ] **Step 1: Write failing contract tests**

Add literal expectations for the v6 semantic string, a hand-built `4 x 5` detail mask round-trip, rejection of non-zero packbits padding, and the five allowed ineligibility reasons. Add existing-cache and strict-resume tests that prove v3 cache identities and v5 semantics cannot enter a v6 formal run.

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_coverability.py \
  training/lunar_policy_training/tests/test_formal_cache.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py
```

Expected: import/constant failures because the coverability contract and v6 identity do not exist.

- [ ] **Step 3: Implement the minimal immutable contract**

Validate arrays as C-contiguous exact dtypes, compute semantic mask hashes over unpacked row-major boolean bytes, and require:

```python
eligible == (
    exact
    and coverable_detail_cell_count > 0
    and mission_coverable_fraction >= 0.95
    and initial_coverable_fraction < 0.95
    and initial_candidate_count > 0
)
```

An ineligible object has exactly one enum reason; an exact eligible object has no reason.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_coverability.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py
git add training/lunar_policy_training
git commit -m "feat(training): define platform coverability v6 contracts"
```

---

### Task 2: Add exact C++ platform reachability projection and Python binding

**Files:**

- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/traversability_projection.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/shared/traversability_projection.cpp`
- Create: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/reachability_projection.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/shared/reachability_projection.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/include/lunar_planner_training_bridge/request.hpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/conversions.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/reachability_projection_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py`

**Interfaces:**

- Extends: `TraversabilityProjection.intrinsic_feasible`, distinct from body-clearance `hard_feasible`.
- Produces: `ProjectReachability(const PlannerInput&, double maximum_edge_distance_m)` returning an exact mask, algorithm ID and deterministic diagnostic counts.
- Ground: selects the C++ `connected_component` containing the transformed start pose and never runs Python flood-fill.
- Hopper: stable row-major BFS over safe landing cells; every edge within `30.0 m` is accepted only after the same exact landing-region and `CertifySingleHop` chain succeeds in both directions, so every denominator pose remains returnable in one exploration trajectory.
- Fail closed: cancellation, bad allocation, invalid/numerically indeterminate certification and resource exhaustion return a non-empty failure reason, never a false mask.
- Bridge: `PlannerBridge.project_reachability(request, maximum_edge_distance_m)` releases the GIL and returns owned NumPy arrays.

- [ ] **Step 1: Write failing GoogleTests and bridge tests**

Fixtures must independently assert: ground start-component equality; Hopper crosses a ground-disconnected gap; landing, delta-v and flight-tube failures reject an edge; a `>30 m` target is absent; repeat results and hashes match; cancellation/numerical/resource failures do not become success.

- [ ] **Step 2: Run RED native tests in an external build root**

```bash
set +u
source /opt/ros/humble/setup.bash
set -u
test "$ROS_DISTRO" = humble
NATIVE_ROOT=/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration/native-red
colcon --log-base "$NATIVE_ROOT/log-build" build --merge-install \
  --base-paths ros2_ws/src \
  --packages-select lunar_planner_core lunar_planner_training_bridge \
  --build-base "$NATIVE_ROOT/build" --install-base "$NATIVE_ROOT/install" \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
```

Expected: the new tests fail to compile because the projection API is absent.

- [ ] **Step 3: Implement exact projection with bounded spatial enumeration**

Use integer row/column offsets whose center distance is `<= maximum_edge_distance_m`; enumerate source nodes and offsets in stable order. Reuse one validated map/projection snapshot. For Hopper, distinguish ordinary `kInfeasible` edge rejection from invalid, canceled, numerical and resource statuses that abort the entire projection.

- [ ] **Step 4: Run GREEN native tests and commit**

```bash
colcon --log-base "$NATIVE_ROOT/log-test" test --base-paths ros2_ws/src \
  --packages-select lunar_planner_core lunar_planner_training_bridge \
  --build-base "$NATIVE_ROOT/build" --event-handlers console_direct+
colcon test-result --test-result-base "$NATIVE_ROOT/build" --verbose
git add ros2_ws/src/lunar_planner_core ros2_ws/src/lunar_planner_training_bridge
git commit -m "feat(planner): project exact platform reachability"
```

---

### Task 3: Compute exact tile-wise target and coverable detail masks

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/coverability.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/multires_scene.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py`
- Modify: `training/lunar_policy_training/tests/test_coverability.py`
- Modify: `training/lunar_policy_training/tests/test_multires_scene.py`

**Interfaces:**

- Produces: tile-wise `build_mission_target_detail_mask(...)` using detail valid, forbidden, hard-obstacle and C++ intrinsic-terrain feasibility.
- Produces: `build_coverable_detail_mask(...)` as the union of `two_dimensional_detail_los/v1` visibility from exact reachable-pose centers.
- Keeps at most one `64 m` truth tile plus packed output in working memory.
- Produces exact `coverable_ratio[256,256]` only for reporting; numerator never consumes this coarse ratio.

- [ ] **Step 1: Write failing deterministic miniature fixtures**

Use literal small grids proving: an enclosed invisible free island is excluded; a free cell that cannot be occupied but is visible from a safe pose is included; LOS-blocked targets are excluded; obstacle and slope-infeasible cells are target-false even if observed; repeat packed bytes and hashes match.

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_coverability.py \
  training/lunar_policy_training/tests/test_multires_scene.py
```

- [ ] **Step 3: Implement tile streaming and visibility union**

Map each reachable `4 m` cell center to affected `0.2 m` tiles, call the native truth visibility kernel with exactly the reveal ray discretization, intersect only mission-target bits, and append row-major output bits. Validate total cell count and semantic hash before returning.

- [ ] **Step 4: Run GREEN, memory-bound fixture and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_coverability.py \
  training/lunar_policy_training/tests/test_multires_scene.py
git add training/lunar_policy_training
git commit -m "feat(training): compute exact detail coverability masks"
```

---

### Task 4: Upgrade formal cache to v4 and per-platform eligibility

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_start_qualification.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/tests/test_formal_cache.py`
- Modify: `training/lunar_policy_training/tests/test_formal_builder.py`
- Modify: `training/lunar_policy_training/tests/test_cli.py`

**Interfaces:**

- Produces: `FORMAL_CACHE_SCHEMA == "lunar-formal-training-cache/v4"`.
- Scene NPZ stores per-platform reachable bits, packed detail bits and coarse coverable ratio; manifest stores counts, fractions, exactness, algorithms, hashes, fixed start, eligibility, reason and stage diagnostics.
- Training schedule uses each platform's own eligible scene IDs and hashes; paired evaluation uses the exact three-way eligible intersection.
- Worker always uses the cache-bound start and rejects a replay/random start mismatch.
- Full manifest reports split x platform total/eligible/feasibility, reason counts and exact-common count.

- [ ] **Step 1: Replace v3/common-subset tests with failing v4 tests**

Assert exact manifest literals, one scene eligible for WHEELED but not HOPPER, platform-local scheduling, exact-common evaluation intersection, fixed start use, old v3 rejection, and deterministic first-reason ordering:

```text
UNSAFE_START -> ZERO_MISSION_TARGET -> MISSION_COVERABLE_BELOW_95
-> INITIAL_ALREADY_SUCCESS -> NO_INITIAL_CANDIDATE
```

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_formal_cache.py \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_cli.py
```

- [ ] **Step 3: Implement v4 serialization, validation and scheduler**

Remove `common_eligible` as the training authority. Validate every NPZ array against manifest shape/dtype/byte-order/semantic hash, and abort publication on any non-ordinary projection/visibility failure. Bind `scenario_schedule_id` to platform plus ordered eligible IDs.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_formal_cache.py \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_cli.py
git add training/lunar_policy_training
git commit -m "feat(training): materialize per-platform formal cache v4"
```

---

### Task 5: Make exact detail coverage the runtime success authority

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/multires_observation.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/observation_boundary.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_episode_state.py`
- Modify: `training/lunar_policy_training/tests/test_multires_observation.py`
- Modify: `training/lunar_policy_training/tests/test_sensor_closed_loop.py`
- Modify: `training/lunar_policy_training/tests/test_formal_resume_state.py`

**Interfaces:**

- `MultiresSensorObservationState` receives the validated packed coverable mask and tracks exact `observed_detail & coverable_detail` bits.
- `mission_observed_delta_m2`, coverage ratio, first crossing, reward and `pose_features[0,4]` derive from exact coverable detail cells.
- Physical obstacles and terrain-infeasible cells may update observed evidence but never increment coverage.
- Active state records the coverability hash; any old active episode or different mask fails closed.

- [ ] **Step 1: Write failing exact-bit tests**

Reveal part of a `4 m` coarse cell and assert the exact hand-counted detail ratio differs from the coarse ratio approximation. Add tests for obstacle evidence without coverage, truth-mask spatial changes leaving candidate tensors unchanged, and replay mask-hash mismatch rejection.

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_multires_observation.py \
  training/lunar_policy_training/tests/test_sensor_closed_loop.py \
  training/lunar_policy_training/tests/test_formal_resume_state.py
```

- [ ] **Step 3: Implement bit-accurate accounting**

At each reveal, set newly observed detail bits as today, count only newly set coverable bits, and divide by the fixed positive coverable count. Keep truth mask out of `ObservedWorld`, candidate gain inputs and dense policy channels.

- [ ] **Step 4: Run GREEN, reward regressions and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_multires_observation.py \
  training/lunar_policy_training/tests/test_sensor_closed_loop.py \
  training/lunar_policy_training/tests/test_reward.py \
  training/lunar_policy_training/tests/test_formal_resume_state.py
git add training/lunar_policy_training
git commit -m "fix(training): count exact platform-coverable detail coverage"
```

---

### Task 6: Unify runtime platform reachability and stage diagnostics

**Files:**

- Create: `training/lunar_policy_training/lunar_policy_training/environment/platform_reachability.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`
- Modify: `training/lunar_policy_training/tests/test_formal_builder.py`

**Interfaces:**

- Produces: `PlatformCandidateReachability.filter(...) -> accepted_mask + reason_counts`.
- Ground runtime filtering uses the C++ observed-map start component; Hopper uses
  C++ observed-map exact current-pose-to-candidate bidirectional single-hop batch
  certification and never consumes the cache BFS result as an action mask.
- `CandidateDiagnostics` fields are exactly `frontier_anchor_count`, `visited_excluded_count`, `static_infeasible_count`, `platform_unreachable_count`, `zero_gain_count`, `emitted_count`, `planner_rejected_count`.
- Infrastructure errors propagate; they never increment `platform_unreachable_count`.

- [ ] **Step 1: Write failing stage-isolation tests**

Create one fixture per rejection stage and assert only the correct counter changes. Prove Hopper accepts an observed certified hop across a ground break and rejects landing/delta-v/tube failures. Prove the same observed state is byte-deterministic.

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py \
  training/lunar_policy_training/tests/test_formal_builder.py
```

- [ ] **Step 3: Implement the shared filter and diagnostic pipeline**

Candidate discovery first records anchors, then visited removal, static feasibility, platform reachability, observed-only detail gain, 64-item selection and planner rejection. The filter owns no truth-mask reference.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py \
  training/lunar_policy_training/tests/test_formal_builder.py
git add training/lunar_policy_training
git commit -m "fix(training): certify runtime candidates per platform"
```

---

### Task 7: Add the observed-only frontier oracle and terminal reasons

**Files:**

- Create: `training/lunar_policy_training/lunar_policy_training/environment/frontier_oracle.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/parallel_pool.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/training_metrics.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Create: `training/lunar_policy_training/tests/test_frontier_oracle.py`
- Modify: `training/lunar_policy_training/tests/test_v3_environment.py`
- Modify: `training/lunar_policy_training/tests/test_parallel_pool.py`
- Modify: `training/lunar_policy_training/tests/test_training_metrics.py`
- Modify: `training/lunar_policy_training/tests/test_cli.py`

**Interfaces:**

- Produces: independent `FrontierOpportunityOracle` that scans observed safe poses/unknown ROI opportunity and uses platform certification without calling production candidate ranking.
- `production empty + oracle > 0` raises `EnvironmentInvariantError` and discards the rollout.
- `production empty + oracle == 0` ends with the latest real stage reason.
- Terminal reason enum: `SUCCESS`, `NO_FRONTIER_ANCHOR`, `VISITED_EXHAUSTED`, `PLATFORM_UNREACHABLE`, `ZERO_GAIN`, `PLANNER_REJECTED_ALL`, `HARD_FAILURE`, `CANCELED`.
- Worker protocol and metrics carry terminal reason, oracle opportunity count, full stage diagnostics and remaining-coverable truth diagnostics out of band.

- [ ] **Step 1: Write failing oracle/state-machine/protocol tests**

Assert contradiction raises before rollout insertion; legal zero-opportunity exhaustion terminates with the correct stage; all-planner-rejected reruns the oracle; every successful and failed terminal has exactly one reason; metric aggregation remains platform-aligned across auto-reset.

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_frontier_oracle.py \
  training/lunar_policy_training/tests/test_v3_environment.py \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_training_metrics.py \
  training/lunar_policy_training/tests/test_cli.py
```

- [ ] **Step 3: Implement oracle and audited terminal transport**

Run the oracle only at an empty production boundary or planner-rejected exhaustion. Never expose oracle positions or truth remaining cells to policy/reward. Treat canceled, non-finite, invalid and resource errors as hard outcomes rather than legal exploration exhaustion.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_frontier_oracle.py \
  training/lunar_policy_training/tests/test_v3_environment.py \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_training_metrics.py \
  training/lunar_policy_training/tests/test_cli.py
git add training/lunar_policy_training
git commit -m "feat(training): audit exploration exhaustion with oracle"
```

---

### Task 8: Add policy-only warm start and new-run manifest boundary

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/checkpoint.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/tests/test_checkpoint.py`
- Modify: `training/lunar_policy_training/tests/test_checkpoint_resume.py`
- Modify: `training/lunar_policy_training/tests/test_cli.py`

**Interfaces:**

- Produces: inert `load_policy_warm_start(...)` that verifies the parent checkpoint hash and loads only approved policy prefixes.
- Approved prefixes include global/local/pose/platform/frontier encoders, frontier position encoder, cross-attention blocks, action MLP and action heads; `value_mlp` is excluded and reinitialized from the new run seed.
- New optimizer, scheduler, normalization, RNG, worker state, episode cursors, metrics journal and global step all start fresh at zero.
- Run manifest records parent checkpoint SHA-256, parent global step, exact loaded prefixes and value-head reinitialization digest.
- Strict resume rejects old semantic/cache identities; source migration cannot bypass the semantic change.

- [ ] **Step 1: Write failing warm-start boundary tests**

Fill parent policy/value tensors with distinct literals. Assert approved tensors transfer exactly, value tensors do not, optimizer/scheduler remain untouched, new global step is zero, malformed/missing/unexpected prefixes fail, and the manifest contains the independently computed parent file hash.

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_checkpoint.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py \
  training/lunar_policy_training/tests/test_cli.py
```

- [ ] **Step 3: Implement explicit warm-start CLI path**

Add a formal-only `--warm-start-checkpoint` option mutually exclusive with `--resume`. Load onto CPU through the restricted checkpoint reader, copy an exact allow-list with strict shape/dtype checks, reinitialize `value_mlp`, then create the new immutable run manifest before workers start.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_checkpoint.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py \
  training/lunar_policy_training/tests/test_cli.py
git add training/lunar_policy_training
git commit -m "feat(training): warm start policy into new formal runs"
```

---

### Task 9: Run repository-wide regression and native release verification

**Files:**

- Verify: all files changed by Tasks 1-8.
- External artifacts: `VERIFY_ROOT=/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration/verification-$(git rev-parse --short=12 HEAD)`.

- [ ] **Step 1: Run the full Python package**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests
```

- [ ] **Step 2: Run production-like native build and tests**

```bash
set +u
source /opt/ros/humble/setup.bash
set -u
test "$ROS_DISTRO" = humble
VERIFY_ROOT=/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration/verification-$(git rev-parse --short=12 HEAD)
colcon --log-base "$VERIFY_ROOT/log-build" build --merge-install \
  --base-paths ros2_ws/src \
  --packages-select lunar_planner_core lunar_planner_training_bridge \
  --build-base "$VERIFY_ROOT/build" --install-base "$VERIFY_ROOT/install" \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
colcon --log-base "$VERIFY_ROOT/log-test" test --base-paths ros2_ws/src \
  --packages-select lunar_planner_core lunar_planner_training_bridge \
  --build-base "$VERIFY_ROOT/build" --event-handlers console_direct+
colcon test-result --test-result-base "$VERIFY_ROOT/build" --verbose
```

- [ ] **Step 3: Run repository boundaries and source hygiene**

```bash
python3 tools/check_repository_boundaries.py .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/foundation/test_repository_boundaries.py
git diff --check
git status --short
```

- [ ] **Step 4: Commit any verification-only fixes**

Only if a real regression required a production/test correction, repeat its RED/GREEN proof and commit the scoped fix. Do not commit generated build, cache or report artifacts.

---

### Task 10: Materialize and pass the external preflight gate before training

**Files:**

- Modify after the observed Hopper exhaustion failure: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Modify after the observed Hopper exhaustion failure: `training/lunar_policy_training/lunar_policy_training/environment/frontier_oracle.py`
- Modify after observing unstable coarse-map return certification: `training/lunar_policy_training/lunar_policy_training/environment/multires_observation.py`
- Modify after the observed Hopper coarse/detail conflict: `ros2_ws/src/lunar_planner_core/src/shared/reachability_projection.cpp`
- Modify after the observed Hopper coarse/detail conflict: `ros2_ws/src/lunar_planner_core/test/reachability_projection_test.cpp`
- Modify after the Hopper reachability algorithm bump: `ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py`
- Modify if required by observed gate failures: `training/lunar_policy_training/lunar_policy_training/formal_preflight.py`
- Modify if required by observed gate failures: `training/lunar_policy_training/lunar_policy_training/eval/baselines.py`
- Create after discovering that `formal-preflight` is only a one-step probe: `training/lunar_policy_training/lunar_policy_training/closed_loop_gate.py`
- Modify after discovering that `formal-preflight` cannot prove natural terminal: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify after the observed Hopper exhaustion failure: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`
- Modify after the observed Hopper exhaustion failure: `training/lunar_policy_training/tests/test_frontier_oracle.py`
- Modify after observing unstable coarse-map return certification: `training/lunar_policy_training/tests/test_multires_observation.py`
- Modify if required by observed gate failures: `training/lunar_policy_training/tests/test_formal_preflight.py`
- Create: `training/lunar_policy_training/tests/test_closed_loop_gate.py`
- Modify: `training/lunar_policy_training/tests/test_cli.py`
- External artifacts only: v4 cache, native install, closed-loop report and prospective run manifest.

**Interfaces:**

- A minimum of 24 frozen physical scenes is evaluated for all three platforms using deterministic observed-only gain-over-cost until natural terminal.
- Every selected scene/platform has `exact=true`, mission coverable fraction `>=0.95`, final exact coverage `>=0.95`, zero oracle contradictions, complete terminal reason and repeat-identical masks/candidates/requests/final coverage.
- Full v4 cache publishes split x platform feasibility and reason counts before a training command is prepared.
- `closed-loop-gate` selects exactly the first 24 entries of the frozen exact-common schedules, emits per-scene/platform terminal evidence plus candidate/request/planner sequence hashes, and never starts training.

The first real gate run found a Hopper legal-exhaustion false negative: at 20.18%
exact coverage, no coarse frontier anchor was within the current 30 m hop, while
113 observed-safe landing cells remained directly certifiable and seven of them
had positive observed-only `0.2 m` gain. The bounded repair is therefore:

- keep the existing coarse-frontier path as the primary candidate path;
- only when it emits no candidate, scan observed-safe cells inside the current
  action envelope, apply the same platform certification, then retain only cells
  with positive observed-only detail gain;
- make the independent oracle scan the same class of observation opportunities
  without calling production ranking;
- admit a zero-immediate-gain transit only after the primary frontier path and
  positive-gain fallback are empty while an observed coarse frontier still
  exists, including when all otherwise feasible frontier anchors currently have
  zero predicted gain; prefer unvisited certified poses and
  permit only the direct parent on a deterministic navigation stack when no
  unvisited transit survives. Store and recertify the parent's executed exact
  `x/y/z`, not its enclosing `4 m` cell center; pop on return and rebuild the
  stack from reveal history;
- make the independent oracle use the same exact current/target world positions,
  `30 m` range and FOV as production; a coarse-cell index distance is not a
  valid action-envelope approximation near cell boundaries;
- do not admit truth-derived targets, a longer Hopper edge, a lower coverage
  threshold, or a new policy tensor field.
- when exact detail landing evidence is supplied, do not erase that certified
  landing solely because its enclosing 4 m aggregate cell is hard-infeasible;
  keep the coarse hard filter for the no-evidence path, require forward and
  reverse certification, and bump the Hopper reachability algorithm identity.
- keep the cache denominator's recoverable multi-hop BFS separate from runtime
  action filtering: runtime must certify each current-pose-to-candidate edge
  directly in both directions and may not accept a candidate merely because it
  is reachable through another candidate node.

The first exact WHEELED replay then exposed a separate cross-resolution authority
failure on scene `b230277e...`: after 241 safe decisions, the current `4 m` cell
contained 378/400 observed detail cells. The global start cell was consequently
unknown, `cpp-ground-start-connected-component/v1` returned zero reachable cells,
and the just-executed parent was rejected even though the `0.2 m` local component
accepted it. The bounded ground repair is therefore:

- inside the observed `0.2 m` local-map window, use the local hard-feasible
  connected component as the WHEELED/LEGGED candidate authority;
- outside that window, retain the conservative `4 m` global component result;
- do not unconditionally intersect the two masks, and do not weaken the global
  rule that a coarse cell becomes known only after all 400 detail cells are known;
- keep Hopper on its independently certified landing/hop path;
- cover the split authority with a regression where global reachability is empty,
  one in-window target is local-reachable, another is in a different local
  component, and an out-of-window target follows the global result.

- [ ] **Step 1: Commit the clean implementation source**

```bash
git status --short
git diff --check
```

Expected: empty status before cache identity is calculated.

- [ ] **Step 2: Build a repository-external preflight v4 cache**

```bash
PREFLIGHT_ROOT=/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration/preflight-$(git rev-parse --short=12 HEAD)
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m lunar_policy_training.cli prepare-data \
  --source-lock /home/kai/CodexDownloads/lunar_navigation/volume3/data/locks/polar_source_lock_v1.json \
  --split-manifest /home/kai/CodexDownloads/lunar_navigation/volume3/data/splits/polar_split_v2.json \
  --cache-root "$PREFLIGHT_ROOT/cache-v4" \
  --materialization preflight \
  --preflight-scenario-limit 24
```

The first commit-bound 24-raw-scene cache produced only six exact-common
scenes (five train and one validation), so it is diagnostic evidence rather
than a passable gate input. Build the next deterministic raw prefix with 128
scenes; the gate still selects exactly 24 exact-common scenes and fails closed
if that prefix remains insufficient. This expands qualification discovery only
and does not select scenes using closed-loop outcomes.

- [ ] **Step 3: Run the three-platform closed-loop gate twice**

Run the dedicated natural-terminal command twice; `formal-preflight` remains a
separate post-full-cache calibration/resume probe and is not a substitute for
this gate:

Before the external run, add and pass these exact regressions:

- `test_select_closed_loop_gate_cases_requires_24_exact_common_scenes`;
- `test_select_closed_loop_gate_cases_binds_schedule_cursor_and_coverability`;
- `test_closed_loop_gate_report_is_canonical_and_repeat_comparable`;
- parameterized rejection of non-exact masks, `<0.95` mission/final coverage,
  missing success, non-success terminal, oracle contradiction, planner failure
  and safety violation;
- `test_closed_loop_gate_cli_runs_without_starting_training`.

Run them together with all CLI regressions:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  training/lunar_policy_training/tests/test_closed_loop_gate.py \
  training/lunar_policy_training/tests/test_cli.py
```

```bash
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m lunar_policy_training.cli closed-loop-gate \
  --cache-manifest "$PREFLIGHT_ROOT/cache-v4/cache-manifest.json" \
  --artifact-root "$PREFLIGHT_ROOT/closed-loop-run-1" \
  --minimum-scenes 24 --max-workers 8
```

Repeat with `closed-loop-run-2`. Compare
`closed_loop_evidence_sha256`; timings and artifact paths are deliberately
excluded from this canonical digest. Fail on any semantic difference.

- [ ] **Step 4: Materialize full v4 cache only after the gate passes**

```bash
FULL_ROOT=/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration/full-$(git rev-parse --short=12 HEAD)
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m lunar_policy_training.cli prepare-data \
  --source-lock /home/kai/CodexDownloads/lunar_navigation/volume3/data/locks/polar_source_lock_v1.json \
  --split-manifest /home/kai/CodexDownloads/lunar_navigation/volume3/data/splits/polar_split_v2.json \
  --cache-root "$FULL_ROOT/cache-v4" \
  --materialization full
```

- [ ] **Step 5: Stop at the formal launch gate**

Report the exact v4 manifest SHA, feasibility table, exclusion reasons, preflight report SHA, warm-start parent SHA/step and prospective new artifact root. Do not invoke the training command until all gates pass and the user explicitly authorizes the new step-0 run.

## Execution Status (2026-08-10)

- Tasks 1-8: complete with scoped RED/GREEN commits.
- Task 9: complete; full Python suite reports `812 passed, 1 skipped`, native
  Release verification reports `234 tests, 0 failures`, and repository boundary
  verification reports `14 passed`.
- Task 10 step 1: completed by the scoped source commit containing this status.
- Task 10 step 2: the first commit-bound 24-raw-scene v4 cache was generated,
  but it contains only six exact-common scenes and therefore cannot pass the
  24-scene gate. The deterministic 128-scene qualification prefix is next.
- Task 10 step 3 tooling: implemented as a separate natural-terminal command
  after live inspection proved that existing `formal-preflight` executes only
  a one-step probe. The two real runs remain pending the enlarged cache.
- Task 10 steps 4-5: pending both repeat-identical closed-loop passes, full-cache
  publication and explicit launch authorization. Formal training remains
  stopped until those gates pass.

## Completion Evidence

The implementation is complete only when:

1. Tasks 1-8 each have observed RED/GREEN evidence and scoped commits.
2. Task 9 full Python, native Release and repository boundary suites pass.
3. Task 10 preflight and full-cache evidence is exact, deterministic and external to Git.
4. `git status --short` is clean and no generated artifact is tracked.
5. The old training checkpoint remains immutable and is referenced only as a policy warm-start parent.
