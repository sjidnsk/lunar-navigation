# Project Formal Capability and Hopper No-Fuel-Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the approved in-repository three-platform capability v2 the formal training authority and remove cumulative hopper fuel state from planning, ROS, and exploration training while retaining fixed single-hop ballistic reachability.

**Architecture:** A focused Python adapter validates and converts the canonical capability freeze into immutable training capabilities. The C++ hopper planner derives a fixed per-hop delta-v envelope from reference mass, specific impulse, and reference propellant stored in `HopperCapability`; no runtime fuel snapshot enters `PlannerInput`, and `HopSegment` exposes only required and available delta-v. Formal CLI and performance evidence always load the repository authority and cannot substitute an external lock.

**Tech Stack:** Ubuntu 22.04 amd64, ROS 2 Humble, C++20, CMake/colcon, Python 3.10, pytest, PyYAML, pybind11.

## Global Constraints

- Current C++ v3 hierarchical planning and capability schema v2 are the only path-planning and motion-capability authorities.
- Formal capability source is `ros2_ws/src/lunar_navigation_config/config/three_platform_capability_freeze_v1.yaml`; its current normalized platform digest is `60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95`.
- Formal sensor capability remains exactly `30.0 m / 360.0 deg`.
- Hopper planning is one safe convex landing region and one certified parabola per policy action; no multi-hop route is introduced.
- Reference total mass, specific impulse, and reference propellant define a repeatable single-hop delta-v envelope; they are not episode state and never decrement.
- No fuel input, fuel reward, fuel termination, or next-request remaining-fuel handoff is allowed.
- Do not restore old Volume 3 planners, proxy capabilities, cached traversability, checkpoints, or fixed search-resource cutoffs.
- Run ROS commands only after sourcing `/opt/ros/humble/setup.bash` and confirming `ROS_DISTRO=humble`.
- Build and test artifacts stay outside the repository under `/home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/`.
- Preserve the root checkout's untracked `.vscode/` and all unrelated user changes.

---

### Task 1: Canonical project capability adapter

**Files:**
- Create: `training/lunar_policy_training/lunar_policy_training/project_capability.py`
- Create: `training/lunar_policy_training/tests/test_project_capability.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/capability_freeze.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/__init__.py`

**Interfaces:**
- Consumes: `platform_capability_schema_v2.yaml`, `three_platform_capability_freeze_v1.yaml`, existing `Frozen*Capability` value types, and `training_semantics.py`.
- Produces: `load_project_formal_capability(repository_root: str | Path) -> FrozenCapabilityBundle` and `project_capability_paths(repository_root: str | Path) -> tuple[Path, Path]`.
- Invariant: `FrozenCapabilityBundle.bundle_sha256` equals the canonical YAML `freeze_digest_sha256`; `formal_eligible` is always `True`.

- [ ] **Step 1: Write failing canonical-source tests**

```python
def test_project_formal_capability_uses_approved_freeze() -> None:
    bundle = load_project_formal_capability(REPOSITORY_ROOT)
    assert bundle.formal_eligible is True
    assert bundle.bundle_sha256 == (
        "60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95"
    )
    assert tuple(item.platform_type for item in bundle.platforms) == (
        "WHEELED", "LEGGED", "HOPPER"
    )
    assert all(item.observation_capability.sensor_range_m == 30.0
               for item in bundle.platforms)


def test_project_formal_capability_rejects_digest_drift(tmp_path: Path) -> None:
    root = copy_project_capability_tree(tmp_path)
    rewrite_yaml(root, "platforms.WHEELED.kinematics.maximum_forward_speed_mps", 9.0)
    with pytest.raises(CapabilityFreezeError, match="freeze_digest_sha256"):
        load_project_formal_capability(root)
```

In this test module, implement `copy_project_capability_tree` by copying only the two canonical YAML files into
their repository-relative paths under `tmp_path`, and implement `rewrite_yaml` with `yaml.safe_load`, explicit
dotted-key traversal, and `yaml.safe_dump(sort_keys=False)`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/training/lunar_policy_training" \
python3 -m pytest -q \
  training/lunar_policy_training/tests/test_project_capability.py
```

Expected: collection fails because `project_capability` and `load_project_formal_capability` do not exist.

- [ ] **Step 3: Implement strict YAML validation and typed conversion**

Implement a loader that:

```python
def load_project_formal_capability(
    repository_root: str | Path,
) -> FrozenCapabilityBundle:
    root = Path(repository_root).resolve(strict=True)
    schema_path, freeze_path = project_capability_paths(root)
    schema = _read_yaml_mapping(schema_path, "project capability schema")
    freeze = _read_yaml_mapping(freeze_path, "project capability freeze")
    _validate_project_schema_and_exact_values(schema, freeze)
    digest = _canonical_platform_digest(freeze["platforms"])
    if freeze["freeze_digest_sha256"] != digest:
        raise CapabilityFreezeError("freeze_digest_sha256 does not match platform payload")
    return FrozenCapabilityBundle(
        schema=CAPABILITY_FREEZE_SCHEMA,
        platforms=_convert_project_platforms(freeze["platforms"]),
        bundle_sha256=digest,
        formal_eligible=True,
    )
```

Derive wheel and legged motion primitives exactly from the current approved 0.2 m local-planner geometry; do not read any retired Volume 3 primitive values. Store hopper `reference_total_mass_kg=20.0` and `reference_propellant_mass_kg=0.2` in `FrozenHopperCapability` for single-hop envelope conversion.

- [ ] **Step 4: Run adapter and existing capability tests**

Run:

```bash
PYTHONPATH="$PWD/training/lunar_policy_training" \
python3 -m pytest -q \
  training/lunar_policy_training/tests/test_project_capability.py \
  training/lunar_policy_training/tests/test_capability_freeze.py \
  tests/foundation/test_platform_capability_freeze.py
```

Expected: all pass; legacy development-smoke bundle parsing remains test-only.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  training/lunar_policy_training/lunar_policy_training/project_capability.py \
  training/lunar_policy_training/lunar_policy_training/capability_freeze.py \
  training/lunar_policy_training/lunar_policy_training/__init__.py \
  training/lunar_policy_training/tests/test_project_capability.py
git commit -m "feat: load approved project capability as formal authority"
```

### Task 2: Replace hopper fuel accounting with a fixed single-hop envelope

**Files:**
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/platform_capability.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/planner_io.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/motion_reference.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/ballistic_envelope.hpp`
- Create: `ros2_ws/src/lunar_planner_core/src/hopper/ballistic_envelope.cpp`
- Delete: `ros2_ws/src/lunar_planner_core/src/hopper/propellant_model.hpp`
- Delete: `ros2_ws/src/lunar_planner_core/src/hopper/propellant_model.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hop_certifier.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hop_certifier.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/hopper/hopper_planner.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/CMakeLists.txt`
- Delete: `ros2_ws/src/lunar_planner_core/test/propellant_model_test.cpp`
- Create: `ros2_ws/src/lunar_planner_core/test/ballistic_envelope_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_core/test/hopper_planner_test.cpp`
- Modify: affected C++ test fixtures and benchmarks that construct `HopperCapability` or `HopSegment`.

**Interfaces:**
- Consumes: `HopperCapability.reference_total_mass_kg`, `reference_propellant_mass_kg`, `specific_impulse_s`, `standard_gravity_mps2`, and `reachability_delta_v_margin_ratio`.
- Produces: `AvailableSingleHopDeltaV(const HopperCapability&)` and `EvaluateSingleHopEnvelope(const BallisticArc&, const HopperCapability&)`.
- `PlannerInput` no longer contains `hopper_propellant`; `HopSegment` contains only `required_delta_v_mps` and `available_delta_v_mps` for reachability evidence.

- [ ] **Step 1: Write failing core behavior tests**

Add tests equivalent to:

```cpp
TEST(HopperPlanner, RepeatsTheSameSingleHopEnvelopeWithoutFuelState) {
  PlannerInput first = MakeValidHopperInput();
  first.hopper_propellant.reset();
  const PlannerOutput one = HopperPlanner{}.Plan(first);
  PlannerInput second = first;
  second.request_id = "second-hop";
  const PlannerOutput two = HopperPlanner{}.Plan(second);
  ASSERT_EQ(one.outcome, PlanningOutcome::kNewReferenceAvailable);
  ASSERT_EQ(two.outcome, PlanningOutcome::kNewReferenceAvailable);
  EXPECT_EQ(RequiredDeltaV(one), RequiredDeltaV(two));
}

TEST(BallisticEnvelope, RejectsOnlyWhenFixedSingleHopDeltaVIsExceeded) {
  const auto result = EvaluateSingleHopEnvelope(too_demanding_arc, capability);
  EXPECT_EQ(result.reason_code, "HOPPER_SINGLE_HOP_ENVELOPE_EXCEEDED");
}
```

- [ ] **Step 2: Build the focused tests and verify RED**

Run:

```bash
source /opt/ros/humble/setup.bash
test "$ROS_DISTRO" = humble
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/red-log \
  build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --base-paths "$PWD/ros2_ws/src" \
  --build-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/red-build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/red-install \
  --packages-select lunar_navigation_msgs lunar_planning_msgs lunar_planner_core
```

Expected: compilation fails on the missing fixed-envelope interface or the old mandatory propellant assertion fails.

- [ ] **Step 3: Implement the fixed envelope**

Use the rocket equation only to derive a repeatable capability envelope:

```cpp
available_delta_v = Isp * g0 * log(
    total_mass / (total_mass - reference_propellant_mass));
required_delta_v = (1.0 + margin) *
    (norm(launch_velocity) + norm(landing_velocity));
```

Validate both reference masses in `ValidCapability`. Remove the `HopperPropellantState*` member from `SingleHopCertificationProblem`, remove all remaining-fuel arithmetic, and map an envelope miss to `HOPPER_SINGLE_HOP_ENVELOPE_EXCEEDED`. Preserve cancellation, numerical-indeterminate, landing-region, and flight-tube semantics.

- [ ] **Step 4: Remove fuel fields from planner inputs and references**

Delete the planner-core `HopperPropellantState` input type and `PlannerInput.hopper_propellant`. Delete the three `*_fuel_*_kg` fields from core `HopSegment`; keep `required_delta_v_mps` and `available_delta_v_mps`. Update every aggregate initializer and validation test deliberately—do not add default fallback values that hide an incomplete capability.

- [ ] **Step 5: Run the core test closure**

Run:

```bash
source /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/red-install/setup.bash
colcon test \
  --test-result-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/red-results \
  --build-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/red-build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/red-install \
  --packages-select lunar_planner_core
colcon test-result \
  --test-result-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/red-results \
  --verbose
```

Expected: all core tests pass and no active core source contains `remaining_usable_fuel` or `HOPPER_FUEL_INSUFFICIENT`.

- [ ] **Step 6: Commit Task 2**

```bash
git add ros2_ws/src/lunar_planner_core tests/performance
git commit -m "refactor: make hopper reachability a repeatable single-hop envelope"
```

### Task 3: Remove propellant from the active ROS and bridge contract

**Files:**
- Modify: `ros2_ws/src/lunar_planning_msgs/msg/HopSegment.msg`
- Modify: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/snapshot_store.hpp`
- Modify: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/snapshot_builder.hpp`
- Modify: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/message_conversion.hpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/snapshot_builder.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/message_conversion.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/config/planner_ros.schema.json`
- Modify: `ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml`
- Modify: `docs/interfaces/external-input-baseline.md`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/include/lunar_planner_training_bridge/request.hpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/conversions.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp`
- Modify: ROS, bridge, public API, marker, reference-guard, and message-conversion tests affected by these fields.

**Interfaces:**
- Consumes: hopper pose, map snapshots, transform, mission, current capability v2, and exact point goal.
- Produces: a propellant-independent `PlannerInput` and `HopSegment` wire result with delta-v evidence only.
- The legacy `lunar_navigation_msgs/msg/HopperPropellantState.msg` remains generated but is not subscribed to or consumed.

- [ ] **Step 1: Write failing ROS and bridge tests**

Add assertions that:

```cpp
TEST(SnapshotBuilder, BuildsHopperWithoutPropellantTopic) {
  SnapshotView view = CompleteHopperViewWithoutPropellant();
  const auto result = builder.Build(view, request, now);
  ASSERT_TRUE(result.ok());
}

TEST(PlanMotionServer, HopperDoesNotCreatePropellantSubscription) {
  EXPECT_EQ(count_subscribers("/platform/hopper_propellant_state"), 0U);
}
```

Update Python bridge tests so a `TrainingPlanRequest` has no `hopper_propellant` property and can plan a hopper request using only the typed capability.

- [ ] **Step 2: Run focused tests and verify RED**

Run the ROS/bridge package tests from the existing Release build roots. Expected: the snapshot still reports `HOPPER_PROPELLANT_STATE_INVALID`, the subscription exists, and old message fields are still required.

- [ ] **Step 3: Remove the active propellant data flow**

Remove the propellant callback group, subscription, age parameter, snapshot-store member, validation, pairwise-skew check, planner input conversion, bridge property, and reference guard's remaining-fuel identity. Remove fuel fields from `HopSegment.msg` and conversion code. Keep the legacy message definition only for source compatibility and mark it unused in the external-input baseline.

- [ ] **Step 4: Extend capability loading with formal reference conditions**

Require these two fields in v2 hopper content and map them to `HopperCapability`:

```yaml
reference_total_mass_kg: 20.0
reference_propellant_mass_kg: 0.2
```

The repository canonical adapter maps them from `reference_conditions.reference_total_mass_kg` and `reference_conditions.reference_remaining_usable_fuel_mass_kg`; the latter YAML name remains frozen to avoid silently changing its approved digest, but the active typed field is explicitly a non-decrementing reference quantity.

- [ ] **Step 5: Run ROS and bridge test closure**

Run:

```bash
source /opt/ros/humble/setup.bash
colcon --log-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/main-log \
  build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --base-paths "$PWD/ros2_ws/src" \
  --build-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/main-build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/main-install
colcon test \
  --build-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/main-build \
  --install-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/main-install \
  --test-result-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/main-results
colcon test-result \
  --test-result-base /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/main-results \
  --verbose
```

Expected: zero failures across message, core, bridge, and ROS packages.

- [ ] **Step 6: Commit Task 3**

```bash
git add ros2_ws/src docs/interfaces/external-input-baseline.md
git commit -m "refactor: remove hopper fuel from active planning interfaces"
```

### Task 4: Wire formal training and performance to the project authority

**Files:**
- Modify: `training/lunar_policy_training/lunar_policy_training/cli.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/sensor_performance.py`
- Modify: `training/tools/benchmark_sensor_observation.py`
- Modify: `training/lunar_policy_training/lunar_policy_training/proxy_scenario.py`
- Modify: `training/lunar_policy_training/tests/test_cli.py`
- Modify: `training/lunar_policy_training/tests/test_sensor_performance.py`
- Modify: `training/lunar_policy_training/tests/test_training_smoke.py`
- Modify: `training/lunar_policy_training/tests/test_curriculum.py`
- Modify: `training/lunar_policy_training/tests/test_v3_environment.py`
- Modify: `training/lunar_policy_training/tests/test_hopper_macro_step.py`

**Interfaces:**
- Consumes: `load_project_formal_capability(repository_root)` and the current bridge API.
- Produces: formal preflight and sensor performance evidence bound directly to the project capability digest; no formal `--capability-lock` input.
- Development-smoke may retain its explicit test-only bundle loader but cannot be promoted to formal.

- [ ] **Step 1: Write failing formal-entry tests**

```python
@pytest.mark.parametrize("command", ("train", "resume", "evaluate"))
def test_formal_entry_uses_project_capability_without_external_lock(command):
    args = formal_args(command)
    parsed = build_parser().parse_args(args)
    assert not hasattr(parsed, "capability_lock")


def test_formal_benchmark_uses_project_capability_digest(tmp_path):
    report = run_benchmark_without_capability_lock(tmp_path)
    assert report["capability_sha256"] == APPROVED_FREEZE_SHA256
```

Build `formal_args` from the existing per-command fixtures in `test_cli.py`. Implement
`run_benchmark_without_capability_lock` by invoking the tool's `main()` with its native-runner and throughput
calls patched to deterministic successful evidence; assert the real canonical loader is not patched.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
source /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/main-install/setup.bash
export PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pytest -q \
  training/lunar_policy_training/tests/test_cli.py \
  training/lunar_policy_training/tests/test_sensor_performance.py
```

Expected: parser and benchmark still require `--capability-lock`.

- [ ] **Step 3: Replace formal external-lock loading**

Change `_formal_capability_preflight` to accept `repository_root`, call `load_project_formal_capability`, and fail before artifacts/CUDA/workers only on canonical schema/digest errors. Remove `--capability-lock` from formal CLI and from the performance tool. Keep the sensor performance report gate because it measures runtime performance rather than supplying capability values.

- [ ] **Step 4: Remove hopper fuel from training requests and state assertions**

Delete all construction, mutation, observation, reward, checkpoint, or episode assertions involving `HopperPropellantState`. The request builder sets the typed capability's fixed reference conditions; repeated hopper macro steps must produce identical `available_delta_v_mps` and never carry a previous result field into the next request.

- [ ] **Step 5: Run all training tests**

Run:

```bash
source /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/main-install/setup.bash
export PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pytest -q \
  model_contract/tests \
  training/lunar_policy_training/tests \
  tests/differential \
  tests/performance
```

Expected: all non-device tests pass; device-gated skips remain explicitly reported.

- [ ] **Step 6: Commit Task 4**

```bash
git add training model_contract tests
git commit -m "feat: bind formal training to project capability authority"
```

### Task 5: Generate current formal sensor performance evidence

**Files:**
- External artifact: `/home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/sensor-performance.json`
- Modify: `docs/validation/sensor-observation-capability-qualification.md`
- Modify: `docs/migration/volume-3-pretraining-readiness.md`
- Modify: `docs/superpowers/specs/2026-08-08-sensor-observation-capability-design.md`
- Modify: `docs/superpowers/plans/2026-08-08-sensor-observation-capability.md`

**Interfaces:**
- Consumes: clean committed sensor/planner source, Release native benchmark, canonical project capability, and 24 workers.
- Produces: a formal `sensor-observation-performance/v1` JSON bound to current host, source commit, capability digest, and sensor semantics digest.

- [ ] **Step 1: Build a clean Release closure**

Run the full `colcon build` command from Task 3 in a new source-commit-named external directory. Confirm the native runner reports `build_type=Release`.

- [ ] **Step 2: Run formal performance qualification without a capability lock**

```bash
source /opt/ros/humble/setup.bash
source /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/main-install/setup.bash
export PYTHONPATH="$PWD/model_contract:$PWD/training/lunar_policy_training${PYTHONPATH:+:$PYTHONPATH}"
python3 training/tools/benchmark_sensor_observation.py \
  --output /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/sensor-performance.json \
  --native-benchmark /home/kai/CodexDownloads/lunar_navigation/formal_capability_no_fuel/main-install/lunar_planner_training_bridge/lib/lunar_planner_training_bridge/lunar_training_visibility_benchmark \
  --workers 24
```

Expected: `passed: true`; report capability SHA equals the canonical freeze digest. Do not start PPO training in this task.

- [ ] **Step 3: Update qualification and migration documents**

Record that capability closure is complete from the approved project freeze, remove the false external-provider blocker, record the formal report path/SHA and actual performance metrics, and state that training is eligible for its next independent environment/data gate. Mark the old external-lock and cumulative-fuel sections as superseded rather than deleting historical evidence.

- [ ] **Step 4: Run repository and UTF-8 checks**

```bash
python3 tools/check_platform_capability_freeze.py \
  --schema ros2_ws/src/lunar_navigation_config/config/platform_capability_schema_v2.yaml \
  --freeze ros2_ws/src/lunar_navigation_config/config/three_platform_capability_freeze_v1.yaml
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
git diff --check
```

Expected: all checks pass and no generated artifact is tracked.

- [ ] **Step 5: Commit Task 5**

```bash
git add docs
git commit -m "docs: qualify built-in formal capability and no-fuel hopper"
```

### Task 6: Full main-repository regression and handoff

**Files:**
- Modify: this plan only to check completed steps.

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: one clean feature branch ready for review and later fast-forward integration.

- [ ] **Step 1: Run the full Release ROS result check**

Run `colcon test` for all built packages and require `colcon test-result --verbose` to report zero errors and zero failures.

- [ ] **Step 2: Run the complete Python and boundary suite**

Run the Task 4 full pytest command plus platform-freeze and repository-boundary checks.

- [ ] **Step 3: Audit retired semantics**

```bash
if rg -n \
  "HOPPER_FUEL_INSUFFICIENT|expected_remaining_usable_fuel|hopper_propellant" \
  ros2_ws/src/lunar_planner_core \
  ros2_ws/src/lunar_planner_ros \
  ros2_ws/src/lunar_planner_training_bridge \
  training/lunar_policy_training/lunar_policy_training; then
  exit 1
fi
```

Expected: no active production occurrence. Historical design documents and the unused legacy message are excluded from this audit.

- [ ] **Step 4: Confirm Git boundaries**

Confirm the feature worktree is clean, root `integration` still only has its pre-existing `.vscode/`, and the external Isaac/RViz repository has not been imported into this repository.

- [ ] **Step 5: Commit plan completion**

```bash
git add docs/superpowers/plans/2026-08-08-project-formal-capability-and-hopper-no-fuel-budget.md
git commit -m "docs: close formal capability and no-fuel implementation plan"
```
