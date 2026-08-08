# Sensor Observation Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved 30 m/360° conservative sensor abstraction, truth-isolated exploration observation loop, native visibility acceleration, and platform-correct theta training semantics on the `integration` line.

**Architecture:** Bring the qualified Volume 3 policy/bridge material into the current integration-based feature branch while preserving all newer complete-search, hierarchical-planning, capability-v2, and rolling-reliability behavior. A reusable Release C++ visibility kernel provides deterministic Bresenham batch gains and single-pose reveal through pybind11; Python owns typed truth/observed state, candidate construction, macro-step boundaries, PPO semantics, manifests, and fail-closed gates.

**Tech Stack:** C++20, CMake/ament, pybind11, ROS 2 Humble, Python 3.10, NumPy, PyTorch PPO, pytest, GoogleTest, YAML/JSON capability closures.

## Global Constraints

- Work only in `/mnt/data/WS/.lunar-navigation-worktrees/sensor-observation-capability-design` on `feature/sensor-observation-capability-design`; preserve `/mnt/data/WS/lunar-navigation/.vscode/`.
- Source `/opt/ros/humble/setup.bash` and verify `ROS_DISTRO=humble` before every ROS build or test.
- Put `build/`, `install/`, `log/`, benchmark reports, checkpoints, and generated artifacts under `/home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability`, never in Git.
- Formal observation capability is exactly `sensor_range_m=30.0` and `sensor_fov_deg=360.0`; production paths have no 80 m fallback.
- `30 m` limits geometric estimation/reveal only; only actual reveal updates observed state and reward.
- Policy/candidate construction never receives truth; a truth object reaching a public policy-facing API is a fail-closed error.
- Any positive physical-obstacle ratio blocks LOS; `forbidden` never blocks LOS; the first blocking cell is visible during reveal.
- Reuse deterministic Bresenham semantics; 360° uses a yaw-independent fast path with no trigonometric filtering.
- Keep the four model outputs and one shared head set; theta is active for `WHEELED` and `LEGGED`, inactive for `HOPPER`.
- Do not add fixed search-resource ceilings removed by the integration branch.
- Native visibility code is internally single-threaded, releases the GIL, uses no GPU/OpenMP, and is qualified only in Release.
- Do not alter or rebaseline the external Isaac/ROS `30 m/120°` historical snapshot.
- Do not launch formal 24-hour PPO training in this plan; completion makes the code eligible for the separately controlled formal-training gate.

---

### Task 1: Integrate the qualified Volume 3 source baseline without planner regressions

**Files:**
- Merge source: `volume-3-policy-pipeline` at `f8cea6a`
- Resolve: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`
- Resolve: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_config.hpp`
- Resolve: `ros2_ws/src/lunar_planner_core/test/public_api_test.cpp`
- Resolve: `tests/foundation/test_repository_boundaries.py`
- Resolve: `tools/check_repository_boundaries.py`
- Verify: `ros2_ws/src/lunar_planner_core/test/public_header_boundary_test.py`

**Interfaces:**
- Consumes: current `integration` planner behavior at `b33d437` and qualified Volume 3 training tree at `f8cea6a`.
- Produces: one integration-based tree containing `training/`, `model_contract/`, `lunar_planner_training_bridge`, and traversability projection without restoring obsolete planner limits.
- Authority rule: current `integration` hierarchical planning, exact single-hop fuel chain, and platform capability schema v2 override every stale planner, request, and motion-capability assumption imported from Volume 3. Volume 3 contributes only still-valid policy/data/training structure.

- [x] **Step 1: Start the explicit history-preserving merge**

```bash
git merge --no-ff volume-3-policy-pipeline
```

Expected: conflicts only in the known planner CMake/config/API and repository-boundary files; do not advance `volume-3-policy-pipeline`.

- [x] **Step 2: Resolve planner files by preserving integration behavior and adding only required Volume 3 surfaces**

Keep every current hierarchical, propellant, projection-cache, rolling, smoothing, benchmark, and no-fixed-resource-limit source/test. Add:

```cmake
src/shared/traversability_projection.cpp
```

and its `lunar_planner_core_traversability_projection_test` target. Preserve all current CMake targets. In `planner_config.hpp`, keep current structs and fields, change only:

```cpp
std::size_t yaw_bin_count{64U};
```

for wheel and legged defaults. Do not import `SearchResourceLimits`, `maximum_terminal_candidates`, graph-node caps, or validation-subdivision caps.

- [x] **Step 3: Resolve API and repository-boundary tests as unions**

Keep the integration smoothing/diagnostic assertions and approved `MotionExecutionFeedback.msg`/`HopperPropellantState.msg`. Add the traversability public include/default-64-yaw test and forbidden raster/archive suffixes:

```python
".tif", ".tiff", ".dem", ".dtm", ".img", ".vrt", ".zip"
```

- [x] **Step 4: Build the merged baseline in Release outside the repository**

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
artifact_root=/home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/baseline
colcon --log-base "$artifact_root/log" build \
  --merge-install \
  --build-base "$artifact_root/build" \
  --install-base "$artifact_root/install" \
  --base-paths ros2_ws/src \
  --packages-up-to lunar_planner_training_bridge \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Expected: `lunar_planner_core` and `lunar_planner_training_bridge` build successfully.

If the imported bridge references retired planner or capability fields, migrate the bridge and its training adapters to the current public API. Do not add compatibility aliases for retired duration, graph-cap, impulse, roughness, or resource-limit fields.

- [x] **Step 5: Run baseline boundary, native, bridge, and Python tests**

```bash
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
source /home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/baseline/install/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/baseline/test-log test \
  --merge-install \
  --build-base /home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/baseline/build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/baseline/install \
  --packages-select lunar_planner_core lunar_planner_training_bridge \
  --event-handlers console_direct+
colcon test-result --test-result-base /home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/baseline/build --verbose
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m pytest -q model_contract/tests training/lunar_policy_training/tests
```

Expected: repository boundaries OK; all selected tests pass. Any inherited failure is fixed before sensor work begins.

- [x] **Step 6: Finish the merge commit**

```bash
git add ros2_ws model_contract training migration tools tests docs .superpowers
git commit -m "merge: integrate qualified volume 3 training baseline"
```

Expected commit subject: `merge: integrate qualified volume 3 training baseline`.

---

### Task 2: Freeze formal observation and training-action semantics

**Files:**
- Create: `training/lunar_policy_training/lunar_policy_training/training_semantics.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/capability_freeze.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/checkpoint.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Test: `training/lunar_policy_training/tests/test_capability_freeze.py`
- Test: `training/lunar_policy_training/tests/test_checkpoint.py`
- Test: `training/lunar_policy_training/tests/test_checkpoint_resume.py`
- Test: `training/lunar_policy_training/tests/test_cli.py`

**Interfaces:**
- Consumes: `FrozenObservationCapability`, `FrozenCapabilityBundle`, and `RunIdentity` from the imported baseline.
- Produces: `FORMAL_SENSOR_RANGE_M`, `FORMAL_SENSOR_FOV_RAD`, `TRAINING_SEMANTICS_VERSION`, `training_semantics_sha256()`, and checkpoint schema `lunar-ppo-checkpoint/v4`.

- [x] **Step 1: Write failing capability and checkpoint identity tests**

Add tests proving that a formal bundle accepts three identical 30 m/360° observation documents, rejects 30 m/120°, rejects per-platform observation drift, and that a v4 resume rejects a changed training-semantics hash. The central expectation is:

```python
assert bundle.platforms[0].observation_capability == FrozenObservationCapability(
    sensor_range_m=30.0,
    sensor_fov_rad=2.0 * math.pi,
)
with pytest.raises(CapabilityFreezeError, match="formal observation capability"):
    load_frozen_capability_bundle(lock_120_deg, run_kind="formal")
```

- [x] **Step 2: Run the focused tests and verify RED**

```bash
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m pytest -q \
  training/lunar_policy_training/tests/test_capability_freeze.py \
  training/lunar_policy_training/tests/test_checkpoint.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py
```

Expected: failures for missing formal observation validation and missing `training_semantics_sha256` identity.

- [x] **Step 3: Implement immutable semantics constants and hash**

```python
FORMAL_SENSOR_RANGE_M = 30.0
FORMAL_SENSOR_FOV_RAD = 2.0 * math.pi
TRAINING_SEMANTICS_VERSION = (
    "lunar-training-semantics/sensor-30m-360-theta-mask/v1"
)

def training_semantics_sha256() -> str:
    return hashlib.sha256(TRAINING_SEMANTICS_VERSION.encode("utf-8")).hexdigest()
```

Formal bundle loading validates exact shared observation values after resource/hash closure verification. Development-smoke fixtures may use explicit alternative values but cannot become formal eligible.

- [x] **Step 4: Upgrade run identity/checkpoint semantics**

Add `training_semantics_sha256` to `RunIdentity`, `_RUN_IDENTITY_FIELDS`, serialized bodies, CLI manifest construction, and resume equality checks; bump the complete-run schema to `lunar-ppo-checkpoint/v4`. Keep v3 read-only for explicit development smoke and reject it for formal runs.

- [x] **Step 5: Run focused tests and commit**

```bash
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m pytest -q \
  training/lunar_policy_training/tests/test_capability_freeze.py \
  training/lunar_policy_training/tests/test_checkpoint.py \
  training/lunar_policy_training/tests/test_checkpoint_resume.py \
  training/lunar_policy_training/tests/test_cli.py
git add training/lunar_policy_training
git commit -m "feat: freeze formal sensor training semantics"
```

---

### Task 3: Implement the deterministic native visibility kernel

**Files:**
- Create: `ros2_ws/src/lunar_planner_training_bridge/include/lunar_planner_training_bridge/visibility.hpp`
- Create: `ros2_ws/src/lunar_planner_training_bridge/src/visibility.cpp`
- Create: `ros2_ws/src/lunar_planner_training_bridge/test/visibility_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/python/lunar_planner_training_bridge/__init__.py`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/CMakeLists.txt`

**Interfaces:**
- Consumes: C-contiguous `bool`/`float32` grids and `int32 [N,2]` row-column candidates.
- Produces: `VisibilityKernel(resolution_m: float, range_m: float)`, `estimate_candidate_gains(...) -> float32[N,2]`, and `reveal_from_pose(...) -> bool[H,W]`.

- [x] **Step 1: Write failing C++ tests for exact ray and occlusion semantics**

Cover all octants, map edges, repeated rays, any-positive obstacle ratio, first-obstacle visibility, behind-obstacle invisibility, and forbidden independence. Assert deterministic offset order and this public surface:

```cpp
VisibilityKernel kernel(1.0, 3.0);
const auto visible = kernel.RevealFromPose(
    GridShape{.height = 7U, .width = 7U},
    GridCell{.row = 3U, .column = 3U}, truth_obstacle_ratio);
EXPECT_TRUE(visible[CellIndex(3U, 5U, 7U)]);
EXPECT_FALSE(visible[CellIndex(3U, 6U, 7U)]);
```

- [x] **Step 2: Build the test and verify RED**

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
artifact_root=/home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/native-red
colcon --log-base "$artifact_root/log" build --merge-install \
  --build-base "$artifact_root/build" --install-base "$artifact_root/install" \
  --base-paths ros2_ws/src --packages-select lunar_planner_training_bridge \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Expected: compile failure because `visibility.hpp`/`VisibilityKernel` is absent.

- [x] **Step 3: Implement the C++ kernel and precomputed stencil**

Implement one immutable stencil per `(resolution_m, range_m)`. Sort endpoint offsets row-major; build each Bresenham sequence once; validate finite positive geometry; use checked multiplication for `height*width`; never read beyond the provided spans. Candidate gain accepts every candidate in one call and follows observed-only intermediate-cell rules. Reveal walks truth-free cells, marks the first blocking cell, and stops that ray.

Build `src/visibility.cpp` as a position-independent `lunar_training_visibility` static library. Link both the pybind module and `visibility_test.cpp` against this target so tests exercise the exact implementation exported to Python.

- [x] **Step 4: Add pybind wrappers with validation before GIL release**

Validate dtype, dimensionality, equal shape, finite floats, candidate bounds, and C-contiguity while holding the GIL; then execute:

```cpp
{
  py::gil_scoped_release release;
  gains = kernel.EstimateCandidateGains(...);
}
```

Expose no implicit dtype casting and no Python fallback from the production wrapper.

- [x] **Step 5: Build, run native and binding tests, then commit**

```bash
source /opt/ros/humble/setup.bash
artifact_root=/home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/native-green
colcon --log-base "$artifact_root/log" build --merge-install \
  --build-base "$artifact_root/build" --install-base "$artifact_root/install" \
  --base-paths ros2_ws/src --packages-select lunar_planner_training_bridge \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source "$artifact_root/install/setup.bash"
colcon --log-base "$artifact_root/test-log" test \
  --build-base "$artifact_root/build" --install-base "$artifact_root/install" \
  --packages-select lunar_planner_training_bridge --event-handlers console_direct+
colcon test-result --test-result-base "$artifact_root/build" --verbose
git add ros2_ws/src/lunar_planner_training_bridge
git commit -m "feat: add native sensor visibility kernel"
```

---

### Task 4: Replace full-grid Python candidate scans with one native batch

**Files:**
- Create: `training/lunar_policy_training/lunar_policy_training/environment/visibility.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/candidate_builder.py`
- Modify: `training/lunar_policy_training/tests/test_candidate_builder_v2.py`
- Create: `training/lunar_policy_training/tests/test_visibility_equivalence.py`
- Modify: `training/lunar_policy_training/tests/test_training_smoke.py`

**Interfaces:**
- Consumes: `FrozenObservationCapability` and native `VisibilityKernel`.
- Produces: required `SensorGeometry`, `NativeVisibilityEstimator`, and a test-only `SlowVisibilityReference` with identical batch signatures.

- [x] **Step 1: Write failing tests for required injection, one-call batching, and equivalence**

Tests must prove `CandidateBuilderV2()` raises `TypeError`, formal construction uses 30 m/2π, 360° gains are yaw invariant, all feasible anchors are sent in one estimator call, and random small-grid native outputs exactly equal the slow reference.

```python
with pytest.raises(TypeError):
    CandidateBuilderV2()
np.testing.assert_array_equal(native.visible_mask, slow.visible_mask)
np.testing.assert_allclose(native.gains, slow.gains, rtol=0.0, atol=1e-7)
```

- [x] **Step 2: Run focused tests and verify RED**

```bash
source /home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/native-green/install/setup.bash
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m pytest -q \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py \
  training/lunar_policy_training/tests/test_visibility_equivalence.py
```

- [x] **Step 3: Implement the explicit reference/native boundary**

Keep `_ray_cells()` as the readable reference. `SlowVisibilityReference` is importable only from a test-support path or under an explicit test-only constructor. `NativeVisibilityEstimator` constructs the bridge kernel once and accepts all current anchors in one call.

- [x] **Step 4: Refactor candidate feature construction**

Generate feasible standoff anchors first, invoke native gains once, then construct the existing 12 fields and deterministic representative/farthest subset. Remove the `unknown_points` nested loop and `SensorGeometry(80.0, 2*pi)` default. In the 360° fast path, retain the 30 m robot-to-candidate bound but skip angular/trigonometric filtering.

- [x] **Step 5: Run tests and commit**

```bash
source /home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/native-green/install/setup.bash
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m pytest -q \
  training/lunar_policy_training/tests/test_candidate_builder_v2.py \
  training/lunar_policy_training/tests/test_visibility_equivalence.py \
  training/lunar_policy_training/tests/test_training_smoke.py
git add training/lunar_policy_training
git commit -m "perf: batch observed-only candidate visibility"
```

---

### Task 5: Implement typed truth and observed-map reveal state

**Files:**
- Create: `training/lunar_policy_training/lunar_policy_training/environment/sensor_observation.py`
- Create: `training/lunar_policy_training/tests/test_sensor_observation.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/observation_builder.py`
- Modify: `training/lunar_policy_training/tests/test_observation_builder_v2.py`

**Interfaces:**
- Consumes: native `VisibilityKernel`, map geometry, mission ROI/priority, and a map-frame pose.
- Produces: `TrainingWorldTruth`, `TrainingObservedGrid`, `ObservationDelta`, and `SensorObservationState.observe(pose_cell, elapsed_s)`.

- [x] **Step 1: Write failing reveal-state tests**

Test initial observation, free-space continuation, first-obstacle visibility, behind-obstacle unknown, forbidden non-occlusion, repeated observation zero delta, age increment, quality reset, count saturation, geometry mismatch, out-of-map pose, and truth rejection by `ObservationBuilderV2`.

```python
delta = state.observe((10, 10), elapsed_s=2.0)
assert state.valid_mask[10, 12]
assert state.observation_quality[10, 12] == 1.0
assert state.observation_age_s[10, 12] == 0.0
assert delta.mission_observed_delta_m2 > 0.0
assert state.observe((10, 10), elapsed_s=0.0).mission_observed_delta_m2 == 0.0
```

- [x] **Step 2: Run focused tests and verify RED**

```bash
source /home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/native-green/install/setup.bash
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m pytest -q training/lunar_policy_training/tests/test_sensor_observation.py
```

- [x] **Step 3: Implement strict truth/observed types and state updates**

`TrainingWorldTruth` is never a subclass of `ObservedWorld`. `TrainingObservedGrid` owns finite arrays for elevation, obstacle, age, quality, variances, count, and valid mask. `observe()` ages prior known cells, invokes native reveal, copies truth only for visible cells, sets quality to `1.0`, resets age, saturates `uint32` counts, and computes new ROI/priority physical area from previously unknown cells only.

- [x] **Step 4: Add explicit conversion to policy-facing `ObservedWorld`**

Expose only:

```python
def to_observed_world(
    self, *, local: LocalObservation
) -> ObservedWorld:
    ...
```

No builder overload accepts `TrainingWorldTruth`. Geometry/canvas mismatch raises before any partial mutation.

- [x] **Step 5: Run tests and commit**

```bash
source /home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/native-green/install/setup.bash
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m pytest -q \
  training/lunar_policy_training/tests/test_sensor_observation.py \
  training/lunar_policy_training/tests/test_observation_builder_v2.py
git add training/lunar_policy_training
git commit -m "feat: close truth-to-observed sensor updates"
```

---

### Task 6: Enforce observation updates at exploration macro-step boundaries

**Files:**
- Create: `training/lunar_policy_training/lunar_policy_training/environment/observation_boundary.py`
- Create: `training/lunar_policy_training/tests/test_sensor_closed_loop.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/environment/v3_environment.py`
- Modify: `training/lunar_policy_training/tests/test_v3_environment.py`
- Modify: `training/lunar_policy_training/tests/test_hopper_macro_step.py`
- Modify: `training/lunar_policy_training/tests/test_training_smoke.py`

**Interfaces:**
- Consumes: `SensorObservationState`, pose at an execution boundary, elapsed physical macro-step time, mission/projection builders, and platform type.
- Produces: `SensorBoundaryEvidence`, `ObservationBoundaryController.reset(...)`, and `.after_execution(...) -> BoundaryObservationResult` with the next `PolicyBatch` and normalized coverage deltas.

- [x] **Step 1: Write failing boundary and end-to-end tests**

Prove one reset reveal before the first candidate; one reveal per ground exploration decision boundary; zero reveal for hopper `JUMP_COMMITTED` and `IN_FLIGHT`; exactly one reveal at `LANDED_HOLD`; and reward deltas equal newly observed area rather than candidate predicted gain.

```python
assert controller.after_execution(
    platform_type="HOPPER", execution_state="IN_FLIGHT", ...
).updated is False
assert controller.after_execution(
    platform_type="HOPPER", execution_state="LANDED_HOLD", ...
).updated is True
```

Extend the same-world training smoke through: truth -> reset reveal -> candidate -> policy action -> C++ v3 request -> certified reference -> boundary reveal -> actual coverage reward.

- [x] **Step 2: Run focused tests and verify RED**

```bash
source /home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/native-green/install/setup.bash
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m pytest -q \
  training/lunar_policy_training/tests/test_sensor_closed_loop.py \
  training/lunar_policy_training/tests/test_hopper_macro_step.py \
  training/lunar_policy_training/tests/test_training_smoke.py
```

- [x] **Step 3: Implement the boundary controller and V3 invariant**

The controller owns observation revision and map-snapshot identity. `V3ExplorationEnvironment` accepts an optional typed boundary controller only from formal/sensor-closed builders; if present, executable references cannot supply self-reported coverage. A ground executor reports final pose/time, while hopper feedback reports pose/time only at `LANDED_HOLD`. The controller builds the next observation and reward facts atomically.

Add this optional evidence to both execution result types without weakening legacy validation:

```python
@dataclass(frozen=True)
class SensorBoundaryEvidence:
    pose_map: Pose2
    elapsed_s: float

@dataclass(frozen=True)
class BoundaryObservationResult:
    next_observation: PolicyBatch
    mission_observed_delta: float
    priority_observed_delta: float
    updated: bool
```

`ReferenceExecutionResult` and `CommittedHopExecutionFeedback` gain `sensor_boundary_evidence: SensorBoundaryEvidence | None = None`. When `require_sensor_closed_loop=True`, ground decision boundaries and hopper `LANDED_HOLD` require evidence; `JUMP_COMMITTED`/`IN_FLIGHT` forbid it. The environment replaces executor-supplied observation/deltas with the controller result.

- [x] **Step 4: Preserve legacy development-smoke isolation**

The existing handcrafted proxy remains explicitly `development-smoke` and cannot satisfy the formal sensor-closed flag. It may keep test-only coverage behavior, but formal factory construction without `ObservationBoundaryController` fails before worker startup.

- [x] **Step 5: Run tests and commit**

```bash
source /home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/native-green/install/setup.bash
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m pytest -q \
  training/lunar_policy_training/tests/test_sensor_closed_loop.py \
  training/lunar_policy_training/tests/test_v3_environment.py \
  training/lunar_policy_training/tests/test_hopper_macro_step.py \
  training/lunar_policy_training/tests/test_training_smoke.py
git add training/lunar_policy_training
git commit -m "feat: bind observations to exploration boundaries"
```

---

### Task 7: Mask theta correctly for the hopper while preserving shared outputs

**Files:**
- Create: `training/lunar_policy_training/lunar_policy_training/policy/action_semantics.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/policy/cross_attention.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/ppo/collector.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/ppo/trainer.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/ppo/trainer_core.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/proxy_scenario.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/evaluation/report.py`
- Modify: `training/lunar_policy_training/tests/test_policy_forward.py`
- Modify: `training/lunar_policy_training/tests/test_collector.py`
- Modify: `training/lunar_policy_training/tests/test_ppo_training.py`
- Modify: `training/lunar_policy_training/tests/test_curriculum.py`
- Modify: `training/lunar_policy_training/tests/test_evaluation_determinism.py`

**Interfaces:**
- Consumes: validated `[B,3]` platform context and the existing four policy outputs.
- Produces: `theta_action_mask(platform_context) -> bool[B]`, masked joint log probability, active-row theta entropy, and hopper planner requests with `yaw_rad=None`.

- [x] **Step 1: Write failing mixed-platform action/loss tests**

Use one row per platform. Assert wheel/legged masks true, hopper false; changing hopper theta does not change joint log probability, PPO ratio, KL, or theta entropy; changing wheel theta does. Assert no-active-row theta entropy is exactly FP32 zero.

```python
mask = theta_action_mask(platform_context)
assert mask.tolist() == [True, True, False]
assert evaluation.log_prob_total[2] == evaluation.log_prob_frontier[2]
```

Also assert hopper request `goal.yaw_rad is None` while wheel/legged receive sampled theta with `pi/24` tolerance.

- [x] **Step 2: Run focused tests and verify RED**

```bash
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m pytest -q \
  training/lunar_policy_training/tests/test_policy_forward.py \
  training/lunar_policy_training/tests/test_collector.py \
  training/lunar_policy_training/tests/test_ppo_training.py \
  training/lunar_policy_training/tests/test_curriculum.py
```

- [x] **Step 3: Implement platform-derived masking**

Require `platform_context` in `sample_action()` and `recompute_action_log_probs()`. For inactive rows set selected theta deterministically to zero, set `log_prob_theta=0`, and compute:

```python
log_prob_total = log_prob_frontier + torch.where(
    theta_active, raw_log_prob_theta, torch.zeros_like(raw_log_prob_theta)
)
```

Pass `theta_active` into the PPO loss and compute theta entropy as `sum(active * entropy) / sum(active)`, or exact zero when the active count is zero. Old and new policy paths derive the mask from the rollout's stored `platform_context`.

- [x] **Step 4: Remove the hopper yaw constraint at every request adapter**

Centralize:

```python
def apply_goal_theta(goal, platform_type: str, theta_rad: float) -> None:
    goal.yaw_rad = None if platform_type == "HOPPER" else float(theta_rad)
    goal.yaw_tolerance_rad = 0.0 if platform_type == "HOPPER" else math.pi / 24.0
```

The zero hopper tolerance is required by the authoritative current exact-point
single-hop planner; do not restore the stale Volume 3 yaw-tolerance assumption.

Use it in proxy, calibration, smoke, and formal request builders; do not alter the exported output tensor names/shapes.

- [x] **Step 5: Add yaw-sensitive diagnostics and commit**

Add deterministic wheel/legged fixtures where the same candidate with different theta changes planner cost/time or next yaw. Report per-platform theta concentration and fixed-yaw counterfactual values in evaluation diagnostics without adding a new release threshold.

```bash
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m pytest -q \
  training/lunar_policy_training/tests/test_policy_forward.py \
  training/lunar_policy_training/tests/test_collector.py \
  training/lunar_policy_training/tests/test_ppo_training.py \
  training/lunar_policy_training/tests/test_curriculum.py
git add training/lunar_policy_training
git commit -m "fix: mask inactive hopper theta training"
```

---

### Task 8: Add Release performance reports and a formal-training gate

**Files:**
- Create: `ros2_ws/src/lunar_planner_training_bridge/test/visibility_benchmark.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/CMakeLists.txt`
- Create: `training/lunar_policy_training/lunar_policy_training/sensor_performance.py`
- Create: `training/tools/benchmark_sensor_observation.py`
- Create: `training/lunar_policy_training/tests/test_sensor_performance.py`
- Create: `tests/performance/test_sensor_visibility_benchmark.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/tests/test_cli.py`

**Interfaces:**
- Consumes: Release native benchmark executable, formal capability hash, source commit, 64-candidate and 0.2 m reveal fixtures.
- Produces: immutable JSON `sensor-observation-performance/v1` and `validate_sensor_performance_report(...)` required by formal CLI startup.

- [ ] **Step 1: Write failing report-contract and CLI-gate tests**

Require finite p50/p95 values, exact host/build/capability/source identity, candidate p95 <= 5 ms, reveal p95 <= 2 ms, and 24-worker throughput drop <= 10%. Formal CLI rejects a missing, stale, Debug, wrong-host, wrong-capability, or failing report; development smoke does not claim formal eligibility.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m pytest -q \
  training/lunar_policy_training/tests/test_sensor_performance.py \
  training/lunar_policy_training/tests/test_cli.py \
  tests/performance/test_sensor_visibility_benchmark.py
```

- [ ] **Step 3: Implement native and 24-worker benchmark collection**

The native executable reports build type and warmed p50/p95 for `256x256 x 64` candidate gains plus one `0.2 m/30 m` reveal. The Python tool compares the same fixed-seed 24-worker workload with the observation controller enabled and disabled, then writes outside the repository using atomic replace.

- [ ] **Step 4: Implement strict report validation and CLI preflight**

Validate exact schema fields, SHA-256 identities, host architecture, Release build, sample count, thresholds, and current source/capability hashes before constructing formal workers. A failure raises a preflight error; it never becomes a PPO negative reward and never reduces candidate count.

- [ ] **Step 5: Build Release, run the benchmark gate, test, and commit**

```bash
source /opt/ros/humble/setup.bash
artifact_root=/home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/performance
colcon --log-base "$artifact_root/log" build --merge-install \
  --build-base "$artifact_root/build" --install-base "$artifact_root/install" \
  --base-paths ros2_ws/src --packages-select lunar_planner_training_bridge \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source "$artifact_root/install/setup.bash"
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 training/tools/benchmark_sensor_observation.py \
  --output "$artifact_root/sensor-performance.json" --workers 24
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m pytest -q \
  training/lunar_policy_training/tests/test_sensor_performance.py \
  tests/performance/test_sensor_visibility_benchmark.py \
  training/lunar_policy_training/tests/test_cli.py
git add ros2_ws/src/lunar_planner_training_bridge training tests/performance
git commit -m "test: gate sensor observation performance"
```

---

### Task 9: Complete cross-layer qualification and integration readiness

**Files:**
- Modify: `docs/migration/volume-3-pretraining-readiness.md`
- Create: `docs/validation/sensor-observation-capability-qualification.md`
- Modify: `docs/superpowers/plans/2026-08-08-sensor-observation-capability.md`

**Interfaces:**
- Consumes: all task commits and external performance report path.
- Produces: reproducible qualification evidence and a clean feature branch ready for review/merge, not a formal trained model.

- [ ] **Step 1: Run UTF-8, diff, boundary, and Python qualification**

```bash
git diff --check integration...HEAD
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
source /home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/performance/install/setup.bash
PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training" \
  python3 -m pytest -q model_contract/tests training/lunar_policy_training/tests tests/performance
```

- [ ] **Step 2: Run native Release qualification**

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
artifact_root=/home/kai/CodexDownloads/lunar_navigation/sensor_observation_capability/final
colcon --log-base "$artifact_root/log" build --merge-install \
  --build-base "$artifact_root/build" --install-base "$artifact_root/install" \
  --base-paths ros2_ws/src \
  --packages-up-to lunar_planner_training_bridge lunar_planner_ros \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source "$artifact_root/install/setup.bash"
colcon --log-base "$artifact_root/test-log" test \
  --build-base "$artifact_root/build" --install-base "$artifact_root/install" \
  --packages-select lunar_planner_core lunar_planner_training_bridge lunar_planner_ros \
  --event-handlers console_direct+
colcon test-result --test-result-base "$artifact_root/build" --verbose
```

- [ ] **Step 3: Record exact evidence without committing artifacts**

The qualification document records commit SHA, branch, Ubuntu/ROS/compiler, Release build paths, test totals, performance report path/SHA, measured p50/p95/throughput, capability/training-semantics hashes, and remaining external gates. It explicitly states that formal training has not started and the Isaac snapshot remains historical.

- [ ] **Step 4: Mark all plan checkboxes and commit qualification**

```bash
git add docs/migration/volume-3-pretraining-readiness.md \
  docs/validation/sensor-observation-capability-qualification.md \
  docs/superpowers/plans/2026-08-08-sensor-observation-capability.md
git commit -m "docs: qualify sensor observation capability"
```

- [ ] **Step 5: Run final branch audit**

```bash
git status --short --branch
git log --oneline --decorate integration..HEAD
git diff --stat integration...HEAD
git diff --check integration...HEAD
```

Expected: clean feature branch; only approved source, tests, plans, and qualification docs; no generated artifacts, external repo changes, nested Git roots, pushes, or merges into `integration`.
