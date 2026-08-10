# Exploration Closure Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining planner/action/sensor/gain mismatch, prove deterministic 95% platform-coverable exploration, then launch a brand-new formal PPO run at step 0.

**Architecture:** Keep cache v4's exact platform/start-specific `0.2 m` coverable mask as the coverage authority. Add a locally certified start connector before conservative global search, persist one selected ground candidate across rolling C++ references, accumulate real sensor evidence along the certified path, and compute observed-only optimistic gain with candidate-set normalization. Bump success/reward/training identities, rebuild external cache artifacts, pass one-scene and 24-scene x three-platform natural-terminal gates, then start with randomly initialized policy/value/optimizer state.

**Tech Stack:** C++20, ROS 2 Humble, GoogleTest, pybind11, Python 3.10, NumPy, PyTorch PPO, pytest, canonical JSON/NPZ artifacts.

## Global Constraints

- Work only in `/mnt/data/WS/.lunar-navigation-worktrees/formal-training-environment-closure` on `feature/formal-training-environment-closure`.
- Preserve unrelated changes. The pre-existing uncommitted `AnchorsAPartiallyObservedGlobalStartFromLocalDetail` regression is Task 1's RED test and must not be discarded.
- Use `apply_patch` for repository edits. Keep build/install/log/cache/checkpoint/report artifacts under `/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration`.
- Source `/opt/ros/humble/setup.bash`, assert `ROS_DISTRO=humble`, and use external `--merge-install` roots for native builds.
- Use `/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for training tests.
- Every behavior change follows RED -> observe intended failure -> minimal GREEN -> focused regression -> scoped commit.
- Do not change the PPO topology, seven input names, `[64,12]` action shape, `1024 m` scene, `30 m / 360 deg` sensor, `0.2 m` reveal, capability, source/split, or unbounded episode lifetime.
- `mission_coverable_fraction >= 0.95` remains task feasibility. Episode success is exact coverage `>=0.95` of the platform-coverable denominator; historical 99% performance is not a launch gate for this run.
- Training stays stopped until Task 7's gates pass. The authorized launch uses neither resume nor warm start.

---

### Task 1: Anchor a partially observed coarse start through certified local detail

**Files:**

- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/global_route_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hierarchical/global_route_planner.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/global_route_planner_test.cpp`

**Interfaces:**

- Produces internal `ResolveStartAnchor(...)`: hard-feasible global portal, local safe connector, cost and deterministic diagnostics.
- Uses the original global start directly when hard-feasible and preserves that route byte-for-byte.
- Otherwise accepts only portals whose centers lie in the local start's `hard_feasible + connected_component` and whose local connector search succeeds.
- Tries portals in `(connector_cost, global_row, global_column)` order and chooses the first with a global path to the goal.
- Keeps the exact current map pose as `route.poses_map.front()`, global indices in `raw_cells`, and never marks the partial coarse start known.

- [ ] **Step 1: Keep and extend failing tests**

The existing regression must fail if the coarse start gate returns:

```cpp
PlannerInput input = GroundInput(PlatformType::kWheeled);
SetKnown(input.world.global_map, 1U, 5U, false);
const auto result = PlanGroundGlobalRoute(input);
ASSERT_TRUE(result.ok()) << result.reason_code;
EXPECT_NE(result.route->raw_cells.front(), (shared::GridCell{1, 5}));
EXPECT_EQ(result.route->poses_map.front().position_m.x, 1.5);
```

Add literal cases for no portal in the local component, nearest portal globally disconnected but a later portal valid, and exact equality of the known-start route before/after repair.

- [ ] **Step 2: Run RED**

```bash
set +u
source /opt/ros/humble/setup.bash
set -u
test "$ROS_DISTRO" = humble
NATIVE_ROOT=/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration/native-red-cross-resolution
colcon --log-base "$NATIVE_ROOT/log-build" build --merge-install \
  --base-paths ros2_ws/src --packages-select lunar_planner_core \
  --build-base "$NATIVE_ROOT/build" --install-base "$NATIVE_ROOT/install" \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
ctest --test-dir "$NATIVE_ROOT/build/lunar_planner_core" \
  -R lunar_planner_core_global_route_planner_test --output-on-failure
```

Expected: the partial-start cases fail with `GLOBAL_NO_KNOWN_SAFE_ROUTE`; established cases pass.

- [ ] **Step 3: Implement connector and portal search**

Use existing `MapSnapshot`, `BuildSafeProjection`, `SearchGlobalGrid`, `TransformPose` and `SimplifyRouteSupercover`. Build a one-cell local goal mask per portal, prepend transformed connector poses, and apply excluded conditional cells only to the global segment.

- [ ] **Step 4: Run GREEN**

```bash
colcon --log-base "$NATIVE_ROOT/log-build" build --merge-install \
  --base-paths ros2_ws/src --packages-select lunar_planner_core \
  --build-base "$NATIVE_ROOT/build" --install-base "$NATIVE_ROOT/install" \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
ctest --test-dir "$NATIVE_ROOT/build/lunar_planner_core" \
  -R 'global_route_planner|hierarchical_planner|route_continuation' \
  --output-on-failure
```

- [ ] **Step 5: Commit**

```bash
git add ros2_ws/src/lunar_planner_core/src/hierarchical/global_route_planner.cpp \
  ros2_ws/src/lunar_planner_core/src/hierarchical/global_route_planner.hpp \
  ros2_ws/src/lunar_planner_core/test/global_route_planner_test.cpp
git commit -m "fix(planner): anchor partial global starts through local detail"
```

---

### Task 2: Restore observed-only optimistic detail information gain

**Files:**

- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/visibility.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/test/visibility_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/visibility.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Create: `training/lunar_policy_training/tests/test_visibility.py`
- Modify: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`
- Modify: `training/lunar_policy_training/tests/test_multires_observation.py`

**Interfaces:**

- `VisibilityKernel::EstimateCandidateGains` evaluates every unobserved ROI endpoint in range.
- Rays stop only at `observed[cell] == 1 && obstacle_ratio[cell] > 0`; unknown cells remain transparent to prediction.
- Candidate pose remains required to be observed and free; no truth/coverable input is added.
- `SlowVisibilityReference` independently implements the same semantics.
- Raw gain still controls zero-gain filtering; feature 5 is `gain/max_positive_gain`, feature 6 is `priority_gain/max_positive_priority_gain`, with zero maxima mapped to zero.

- [ ] **Step 1: Write failing native/Python tests**

Use a literal seven-cell ray: farther unknown ROI cells contribute through an unknown predecessor; changing that predecessor to observed-obstacle blocks them. For raw candidate gains `2` and `8`, assert literal normalized values `0.25` and `1.0`.

- [ ] **Step 2: Run RED**

```bash
PYTHON=/home/kai/CodexDownloads/lunar_navigation/volume3/venv/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q \
  training/lunar_policy_training/tests/test_visibility.py \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py \
  training/lunar_policy_training/tests/test_multires_observation.py
```

- [ ] **Step 3: Implement minimal kernel and normalization**

Reuse precomputed ray/reverse-occurrence tables to form visibility under known obstacles, sum `visible && !observed` ROI/priority weights, and normalize only after platform reachability.

- [ ] **Step 4: Run GREEN and native parity**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q \
  training/lunar_policy_training/tests/test_visibility.py \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py \
  training/lunar_policy_training/tests/test_multires_observation.py \
  ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py
ctest --test-dir "$NATIVE_ROOT/build/lunar_planner_training_bridge" \
  -R 'visibility|bridge' --output-on-failure
```

- [ ] **Step 5: Commit**

```bash
git add ros2_ws/src/lunar_planner_training_bridge \
  training/lunar_policy_training/lunar_policy_training/environment/visibility.py \
  training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py \
  training/lunar_policy_training/tests/test_visibility.py \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py \
  training/lunar_policy_training/tests/test_multires_observation.py
git commit -m "fix(training): estimate detail gain through unknown space"
```

---

### Task 3: Observe every certified ground path at one-metre spacing

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/observation_boundary.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_episode_state.py`
- Modify: `training/lunar_policy_training/tests/test_sensor_closed_loop.py`
- Modify: `training/lunar_policy_training/tests/test_formal_builder.py`
- Modify: `training/lunar_policy_training/tests/test_formal_resume_state.py`

**Interfaces:**

- Adds immutable `SensorPathSample(pose_map: Pose2, elapsed_s: float)`.
- Extends `SensorBoundaryEvidence` with path samples; final sample equals `pose_map`, sample elapsed values sum to total elapsed, empty samples preserve one-pose compatibility.
- Controller applies all samples, accumulates de-duplicated deltas and emits one final `PolicyBatch` revision.
- Formal ground execution interpolates the certified trajectory at `<=1.0 m`, always including endpoint and interpolated timestamps.
- `FormalRevealState` serializes all samples so replay restores identical detail bits, time, coverage and observation identity.

- [ ] **Step 1: Write failing path/replay tests**

Create a `4.0 m` route whose endpoint-only reveal misses a literal side cell. Assert samples at maximum `1.0 m`, one observation revision, aggregate delta, exact serialized samples, replay byte equality, and no Hopper in-flight observation.

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q \
  training/lunar_policy_training/tests/test_sensor_closed_loop.py \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_formal_resume_state.py
```

- [ ] **Step 3: Implement validation, interpolation and replay**

Interpolate cumulative distance/time and shortest-angle yaw; update navigation stack only with final endpoints, not intermediate sensor samples.

- [ ] **Step 4: Run GREEN**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q \
  training/lunar_policy_training/tests/test_sensor_closed_loop.py \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_formal_resume_state.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py
```

- [ ] **Step 5: Commit**

```bash
git add training/lunar_policy_training/lunar_policy_training/environment \
  training/lunar_policy_training/tests/test_sensor_closed_loop.py \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_formal_resume_state.py
git commit -m "fix(training): observe certified ground paths"
```

---

### Task 4: Persist one ground candidate through rolling references

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/environment/formal_builder.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/macro_step.py`
- Modify: `training/lunar_policy_training/tests/test_formal_builder.py`
- Modify: `training/lunar_policy_training/tests/test_v3_environment.py`
- Modify: `training/lunar_policy_training/tests/test_parallel_pool.py`
- Modify: `training/lunar_policy_training/tests/test_collector.py`

**Interfaces:**

- `begin_ground_option(action, identity)` freezes exact target/tolerance/theta/stable goal ID.
- `continue_ground_option(identity)` rebuilds only maps/state/time around that target.
- `ground_option_distance_m()` and `clear_ground_option()` own lifecycle without policy-visible state.
- Environment loops plan -> execute -> observe -> replan until target tolerance, success/hard terminal, or formal refreshed-goal rejection.
- All internal contributions form one transition and consume one policy decision; non-finite/no-progress or more than 64 references raises `EnvironmentInvariantError` and discards rollout.

- [ ] **Step 1: Write failing identity/aggregation tests**

Use three fake references ending at `x=4,8,10` for literal target `x=10`; mutate intermediate candidate arrays and assert every request still targets 10, policy is called once, deltas/cost/time/events sum once, and last observation wins. Add rejected-after-progress, no-progress and 65-reference cases.

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q \
  training/lunar_policy_training/tests/test_v3_environment.py \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_collector.py
```

- [ ] **Step 3: Implement option loop and ground aggregation**

Mirror committed-Hopper aggregation while retaining stable sensor boundaries between internal references. Clear active option in every return/exception path; snapshots require no active option.

- [ ] **Step 4: Run GREEN**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q \
  training/lunar_policy_training/tests/test_v3_environment.py \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_collector.py \
  training/lunar_policy_training/tests/test_curriculum.py
```

- [ ] **Step 5: Commit**

```bash
git add training/lunar_policy_training/lunar_policy_training/environment \
  training/lunar_policy_training/tests/test_v3_environment.py \
  training/lunar_policy_training/tests/test_formal_builder.py \
  training/lunar_policy_training/tests/test_parallel_pool.py \
  training/lunar_policy_training/tests/test_collector.py
git commit -m "fix(training): persist ground goals across rolling references"
```

---

### Task 5: Freeze 95% success, reward v4 and fully fresh run identity

**Files:**

- Modify: `training/lunar_policy_training/lunar_policy_training/training_semantics.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/reward.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/polar_data/formal_cache.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/formal_preflight.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/closed_loop_gate.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/checkpoint.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/tests/test_reward.py`
- Modify: `training/lunar_policy_training/tests/test_sensor_closed_loop.py`
- Modify: `training/lunar_policy_training/tests/test_formal_cache.py`
- Modify: `training/lunar_policy_training/tests/test_formal_preflight.py`
- Modify: `training/lunar_policy_training/tests/test_closed_loop_gate.py`
- Modify: `training/lunar_policy_training/tests/test_checkpoint_resume.py`
- Modify: `training/lunar_policy_training/tests/test_cli.py`

**Interfaces:**

- Success ratio is exactly `0.95`; semantics are `lunar-training-semantics/sensor-30m-360-platform-coverable-detail95-ground-option-path-observation/v7`.
- Reward schema is `lunar-reward/v4`; coverage scale stays `100.0`, first-success bonus becomes `100.0`, and cost/time/priority remain validated telemetry.
- Initial eligibility is `<0.95`, mission feasibility remains `>=0.95`, and closed-loop final coverage is `>=0.95`.
- Pre-v7/pre-v4 resume and warm start fail closed.
- Launch without either flag records null parents and initializes complete model/value/optimizer/RNG/normalization at step 0.

- [ ] **Step 1: Write failing literal tests**

Assert `0.949999` is not success, `0.949 -> 0.95` fires once, a later step has no second bonus, zero-gain success reward is `99.9`, `0.949999` gate coverage fails, `0.95` passes, and fresh manifest parents are null at step 0.

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q \
  training/lunar_policy_training/tests/test_reward.py \
  training/lunar_policy_training/tests/test_sensor_closed_loop.py \
  training/lunar_policy_training/tests/test_formal_cache.py \
  training/lunar_policy_training/tests/test_formal_preflight.py \
  training/lunar_policy_training/tests/test_closed_loop_gate.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py \
  training/lunar_policy_training/tests/test_cli.py
```

- [ ] **Step 3: Implement v7/v4 and fresh-launch boundary**

Keep the existing inert warm-start reader but make it incompatible with this formal v7 launch. Do not introduce efficiency reward terms.

- [ ] **Step 4: Run GREEN**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q \
  training/lunar_policy_training/tests/test_reward.py \
  training/lunar_policy_training/tests/test_sensor_closed_loop.py \
  training/lunar_policy_training/tests/test_formal_cache.py \
  training/lunar_policy_training/tests/test_formal_preflight.py \
  training/lunar_policy_training/tests/test_closed_loop_gate.py \
  training/lunar_policy_training/tests/test_checkpoint.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py \
  training/lunar_policy_training/tests/test_cli.py
```

- [ ] **Step 5: Commit**

```bash
git add training/lunar_policy_training
git commit -m "fix(training): set current formal success gate to 95 percent"
```

---

### Task 6: Verify the complete source before materialization

**Files:** Verify all Task 1-5 files; write generated artifacts only below `/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration`.

- [ ] **Step 1: Run all training tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q training/lunar_policy_training/tests
```

- [ ] **Step 2: Run native Release build/tests**

```bash
set +u
source /opt/ros/humble/setup.bash
set -u
test "$ROS_DISTRO" = humble
VERIFY_ROOT=/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration/verification-$(git rev-parse --short=12 HEAD)
colcon --log-base "$VERIFY_ROOT/log-build" build --merge-install \
  --base-paths ros2_ws/src --packages-select lunar_planner_core lunar_planner_training_bridge \
  --build-base "$VERIFY_ROOT/build" --install-base "$VERIFY_ROOT/install" \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
colcon --log-base "$VERIFY_ROOT/log-test" test --base-paths ros2_ws/src \
  --packages-select lunar_planner_core lunar_planner_training_bridge \
  --build-base "$VERIFY_ROOT/build" --install-base "$VERIFY_ROOT/install" \
  --merge-install --event-handlers console_direct+
colcon test-result --test-result-base "$VERIFY_ROOT/build" --verbose
```

- [ ] **Step 3: Verify boundaries, UTF-8 and Git hygiene**

```bash
python3 tools/check_repository_boundaries.py .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q tests/foundation/test_repository_boundaries.py
python3 - <<'PY'
from pathlib import Path
for path in (
    Path('docs/superpowers/specs/2026-08-10-platform-coverable-exploration-design.md'),
    Path('docs/superpowers/plans/2026-08-10-exploration-closure-completion.md'),
):
    path.read_text(encoding='utf-8')
PY
git diff --check
git status --short
```

- [ ] **Step 4: Commit only verified corrections**

Any correction gets its own RED/GREEN. Source must be committed and clean before computing cache identity.

---

### Task 7: Prove 95% closure, rebuild cache and launch fresh training

**Files:** Verify `closed_loop_gate.py`, `formal_preflight.py` and `cli.py`; all resulting caches, traces, reports, logs and checkpoints remain external.

**Interfaces:**

- Replays `b230277e630f34f194195f87321ea1ae5be7c25e40b597e429150c375dceeaf7`/WHEELED to natural success `>=0.95`, then representative exact LEGGED and HOPPER cases.
- Requires zero oracle contradictions/safety violations/systematic planner rejection and target/reference progress evidence.
- Runs the first 24 exact-common frozen scenes x three platforms twice; both reports require `72/72` success and identical canonical digests.
- Full cache and final preflight precede launch.
- Launch omits resume/warm-start, records PID/command externally, and is healthy only after liveness, step-0 manifest, finite first metrics and first valid checkpoint.

- [ ] **Step 1: Build commit-bound 128-scene preflight cache**

```bash
SOURCE_COMMIT=$(git rev-parse --verify HEAD)
PREFLIGHT_ROOT=/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration/preflight-detail95-${SOURCE_COMMIT:0:12}
set +u
source /opt/ros/humble/setup.bash
source "/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration/verification-${SOURCE_COMMIT:0:12}/install/setup.bash"
set -u
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  "$PYTHON" -m lunar_policy_training.cli prepare-data \
  --source-lock /home/kai/CodexDownloads/lunar_navigation/volume3/data/locks/polar_source_lock_v1.json \
  --split-manifest /home/kai/CodexDownloads/lunar_navigation/volume3/data/splits/polar_split_v2.json \
  --cache-root "$PREFLIGHT_ROOT/cache-v4" --materialization preflight \
  --preflight-scenario-limit 128
```

- [ ] **Step 2: Run single-case natural-terminal diagnostics**

Use the closed-loop case runner and frozen schedule. Do not replace cases based on outcomes. Require b230/WHEELED plus the first exact-common LEGGED and HOPPER cases to terminate `SUCCESS >=0.95`.

- [ ] **Step 3: Run 24x3 gate twice**

```bash
for run in 1 2; do
  PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
    "$PYTHON" -m lunar_policy_training.cli closed-loop-gate \
    --cache-manifest "$PREFLIGHT_ROOT/cache-v4/cache-manifest.json" \
    --artifact-root "$PREFLIGHT_ROOT/closed-loop-run-$run" \
    --minimum-scenes 24 --max-workers 8
done
```

Compare `closed_loop_evidence_sha256`; only timing and artifact paths already excluded by canonicalization may differ.

- [ ] **Step 4: Materialize full cache and run formal preflight**

```bash
FULL_ROOT=/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration/full-detail95-${SOURCE_COMMIT:0:12}
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  "$PYTHON" -m lunar_policy_training.cli prepare-data \
  --source-lock /home/kai/CodexDownloads/lunar_navigation/volume3/data/locks/polar_source_lock_v1.json \
  --split-manifest /home/kai/CodexDownloads/lunar_navigation/volume3/data/splits/polar_split_v2.json \
  --cache-root "$FULL_ROOT/cache-v4" --materialization full
```

Run existing `formal-preflight` with the full manifest. Retain report SHA and verify every split/platform feasibility row and exclusion total.

- [ ] **Step 5: Launch detached fresh step-0 training and inspect**

Resolve exact arguments from `lunar_policy_training.cli train --help` and the full manifest. Use formal seed `4080`, validated worker/micro-batch freeze and a new external root; omit both parent flags. Start with `nohup`, record PID/command externally, then conditionally poll until:

```text
process alive
manifest global_step=0 and both parent fields null
first finite metrics row with three-platform coverage/terminal/candidate diagnostics
first checkpoint present and manifest-consistent
```

If a gate fails, return to its owning task with a new RED test and do not launch training.

## Completion Evidence

1. Tasks 1-5 have observed RED/GREEN evidence and scoped commits.
2. Full Python, native Release and repository-boundary checks pass at the final source commit.
3. b230/WHEELED and representative LEGGED/HOPPER traces naturally reach `>=0.95`.
4. Both 24x3 reports pass with identical canonical digests.
5. Full cache and formal-preflight identities are exact and external.
6. A randomly initialized formal run is alive from step 0 with finite first metrics and a valid first checkpoint.
7. `git status --short` is clean and no generated artifact is tracked.
