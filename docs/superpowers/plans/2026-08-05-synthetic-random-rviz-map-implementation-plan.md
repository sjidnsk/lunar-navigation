# Synthetic Random RViz Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one deterministic `50 m × 50 m`, `0.2 m` synthetic obstacle map that all three planners can use interactively from RViz2 without any Isaac Sim dependency.

**Architecture:** Add a source-neutral interactive `PlanningMapBundle` and a strict `lunar-synthetic-planning-map/v1` generator/loader beside the unchanged Isaac snapshot contract. The existing interactive controller, bridge and RViz panel consume the neutral bundle; the formal six-case regression continues to consume only the Isaac snapshot. A new `--synthetic-map` CLI selects the synthetic source explicitly and publishes the same planner GridMap topics plus standard PointCloud2 visualization topics.

**Tech Stack:** Ubuntu 22.04, ROS 2 Humble, Python 3.10, NumPy PCG64, rclpy, grid_map_msgs, sensor_msgs/PointCloud2, RViz2/Qt5 C++17, pytest, ament/colcon, Bash.

## Global Constraints

- Implement only in `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression`; the main repository receives design and plan documents only.
- Preserve the existing `--manifest` Isaac interaction path and the formal six-case regression without semantic changes.
- Do not start, inspect, modify or clean up Isaac Sim; the synthetic path has no USD, stage or prim fields.
- Generate exactly `50 m × 50 m` at `0.2 m`, which is `250 × 250` cells with origin `[-25.0, -25.0]`.
- Default seed is `20260805`; obstacle occupancy must be within `[0.11, 0.13]`; do not perform connectivity, reachability or replanning-based resampling.
- Keep `2.0 m` obstacle-free discs around wheel `(-18,-16)`, legged `(-18,0)` and hopper `(-18,16)` starts.
- Keep the planner input as `grid_map_msgs/msg/GridMap` and the RViz map displays as standard `sensor_msgs/msg/PointCloud2`.
- Run all ROS commands after sourcing `/opt/ros/humble/setup.bash` and verifying `ROS_DISTRO=humble`; append to `PYTHONPATH` rather than replacing it.
- Follow strict RED→GREEN TDD for every production behavior. Generated map, build, log and visual artifacts stay under `/home/kai/CodexDownloads/lunar_navigation`.
- Preserve unrelated worktree changes and never use recursive deletion or name-wide process termination.

---

### Task 1: Deterministic synthetic map contract and generator

**Files:**
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/planning_map.py`
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/synthetic_map.py`
- Create: `ros2_ws/src/lunar_isaac_validation/test/test_synthetic_map.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/constants.py`

**Interfaces:**
- Produces: `PlanningPlatform`, `PlanningMapMetadata`, `PlanningMapBundle` dataclasses.
- Produces: `SyntheticMapConfig(seed: int = 20260805, size_m: float = 50.0, resolution_m: float = 0.2, obstacle_occupancy: float = 0.12)`.
- Produces: `generate_synthetic_map(config: SyntheticMapConfig, output_root: Path) -> Path` returning the published `map_manifest.json`.
- Produces: `load_synthetic_map(manifest_path: Path) -> PlanningMapBundle`.
- Produces: `load_interactive_map(*, manifest_path: Path | None, synthetic_map_path: Path | None) -> PlanningMapBundle` and rejects zero or two sources.

- [ ] **Step 1: Write failing geometry, reproducibility and source-separation tests**

```python
def test_default_map_is_exact_and_reproducible(tmp_path: Path) -> None:
    first = generate_synthetic_map(SyntheticMapConfig(), tmp_path / "one")
    second = generate_synthetic_map(SyntheticMapConfig(), tmp_path / "two")
    one = load_synthetic_map(first)
    two = load_synthetic_map(second)
    assert one.metadata.source_kind == "synthetic"
    assert one.metadata.seed == 20260805
    assert one.grids["global"] == GridDescriptor(
        "global", "map", (-25.0, -25.0), 0.2, 250, 250
    )
    assert one.manifest["arrays_sha256"] == two.manifest["arrays_sha256"]
    assert one.metadata.map_id == two.metadata.map_id

def test_generator_preserves_starts_without_connectivity_search(tmp_path: Path) -> None:
    bundle = load_synthetic_map(
        generate_synthetic_map(SyntheticMapConfig(), tmp_path)
    )
    obstacle = bundle.arrays["wheel__obstacle"]
    assert 0.11 <= float(obstacle.mean()) <= 0.13
    rows, columns = np.indices(obstacle.shape)
    center_x = -25.0 + (columns + 0.5) * 0.2
    center_y = -25.0 + (rows + 0.5) * 0.2
    for x, y in ((-18.0, -16.0), (-18.0, 0.0), (-18.0, 16.0)):
        protected = (center_x - x) ** 2 + (center_y - y) ** 2 <= 4.0
        assert not obstacle[protected].any()
```

The named break is a wrong map geometry, nondeterministic seed, fake Isaac source, wrong occupancy, or obstacle leakage into a start disc. Hand-check the geometry and safe-disc cells independently in the test; do not call generator helpers to compute expected values.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test/test_synthetic_map.py
```

Expected: collection fails because `lunar_isaac_validation.synthetic_map` and its public interfaces do not exist.

- [ ] **Step 3: Implement the minimal neutral bundle and generator**

Use these stable public shapes:

```python
@dataclass(frozen=True)
class PlanningPlatform:
    key: str
    platform_id: str
    planning_position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]

@dataclass(frozen=True)
class PlanningMapMetadata:
    source_kind: str
    map_id: str
    seed: int | None
    size_m: float
    resolution_m: float
    obstacle_occupancy: float

@dataclass(frozen=True)
class PlanningMapBundle:
    manifest_path: Path
    arrays_path: Path
    manifest: dict[str, object]
    grids: dict[str, GridDescriptor]
    platforms: dict[str, PlanningPlatform]
    arrays: dict[str, np.ndarray]
    metadata: PlanningMapMetadata
```

The generator must use `np.random.Generator(np.random.PCG64(config.seed))`, sample circle/rectangle with equal probability, rasterize by cell center, reject any candidate intersecting a protected start disc or taking occupancy above `0.13`, and stop at or above `0.12`. Store ten canonical float32 layers once in `map_arrays.npz`; the loader exposes the same arrays under all `global__*`, `wheel__*`, `legged__*` and `hopper__*` keys. Compute `map_id` from canonical JSON parameters plus `arrays_sha256`, validate the completed temporary output by reloading it, then publish the explicit final directory with `os.replace`.

- [ ] **Step 4: Add strict corruption and boundary tests, verify RED then GREEN**

Add literal mutations for a missing layer, wrong SHA-256, unknown manifest key, non-integral `size_m/resolution_m`, output collision, different seed and out-of-bounds platform start. Each mutation must raise one stable `SyntheticMapError` code. Run the focused test before each matching validation branch, observe the expected failure, then implement only that branch.

- [ ] **Step 5: Run Task 1 tests and commit**

```bash
python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test/test_synthetic_map.py
git add ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/planning_map.py \
  ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/synthetic_map.py \
  ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/constants.py \
  ros2_ws/src/lunar_isaac_validation/test/test_synthetic_map.py
git commit -m "feat: add deterministic synthetic planning map"
```

Expected: focused tests pass and only the four Task 1 files enter the commit.

---

### Task 2: Source-neutral interactive controller and bridge

**Files:**
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/bridge_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_bridge_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_goal.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_session.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_goal.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_session.py`

**Interfaces:**
- Consumes: `load_interactive_map()` and `PlanningMapBundle` from Task 1.
- Produces: mutually exclusive `--manifest PATH` and `--synthetic-map PATH` on both interactive executables.
- Produces: `SessionSupervisor(map_path: Path, map_option: str, ...)`, where `map_option` is exactly `--manifest` or `--synthetic-map`.

- [ ] **Step 1: Write failing parser and child-command tests**

```python
def test_controller_cli_accepts_exactly_one_map_source() -> None:
    parser = _parser()
    required = [
        "--artifact-dir", "/tmp/artifacts",
        "--wheel-params", "/tmp/wheel-params.yaml",
        "--wheel-capability", "/tmp/wheel-capability.yaml",
        "--legged-params", "/tmp/legged-params.yaml",
        "--legged-capability", "/tmp/legged-capability.yaml",
        "--hopper-params", "/tmp/hopper-params.yaml",
        "--hopper-capability", "/tmp/hopper-capability.yaml",
    ]
    synthetic = parser.parse_args(required + ["--synthetic-map", "/tmp/map.json"])
    assert synthetic.synthetic_map == Path("/tmp/map.json")
    with pytest.raises(SystemExit):
        parser.parse_args(required)
    with pytest.raises(SystemExit):
        parser.parse_args(required + [
            "--manifest", "/tmp/isaac.json",
            "--synthetic-map", "/tmp/map.json",
        ])

def test_synthetic_session_starts_bridge_with_synthetic_flag(supervisor) -> None:
    value, _events, _transport, factory, _params = supervisor
    value.map_option = "--synthetic-map"
    value.switch_platform("wheel")
    assert factory.calls[0][0][4:6] == [
        "--synthetic-map", str(value.map_path)
    ]
```

The named break is an ambiguous source or a synthetic map accidentally passed through the Isaac flag.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_goal.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_session.py
```

Expected: failures show the missing `--synthetic-map`, `map_path` and neutral bundle behavior.

- [ ] **Step 3: Implement source-neutral loading with legacy preservation**

Change interactive-only type annotations and constructors from `SnapshotBundle` to `PlanningMapBundle`. Convert an Isaac snapshot to the neutral bundle inside `load_interactive_map`; do not change `SnapshotBridge`, scenario qualification, action assertions or the formal runner. Construct bridge and controller with the same already validated neutral bundle semantics. Ensure all source selection occurs before rclpy node or child-process creation.

- [ ] **Step 4: Verify both source paths and commit**

Run the focused tests, then the existing snapshot contract, bridge and session tests:

```bash
python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_snapshot_contract.py \
  ros2_ws/src/lunar_isaac_validation/test/test_bridge_node.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_goal.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_session.py
git add ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_goal.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_session.py
git commit -m "feat: load synthetic maps in interactive planning"
```

Expected: both source paths pass while formal snapshot tests remain green.

---

### Task 3: Map provenance in the RViz left panel

**Files:**
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_integration.py`
- Modify: `ros2_ws/src/lunar_isaac_rviz_plugins/include/lunar_isaac_rviz_plugins/lunar_planner_panel.hpp`
- Modify: `ros2_ws/src/lunar_isaac_rviz_plugins/src/lunar_planner_panel.cpp`
- Modify: `ros2_ws/src/lunar_isaac_rviz_plugins/test/lunar_planner_panel_test.cpp`

**Interfaces:**
- Consumes: `PlanningMapMetadata` from Task 1.
- Produces: six additional exact diagnostic keys: `source_kind`, `map_id`, `seed`, `size_m`, `resolution_m`, `obstacle_occupancy`.
- Produces: read-only Qt labels with object names `source_label`, `map_id_label`, `seed_label`, `size_label`, `resolution_label`, `occupancy_label`.

- [ ] **Step 1: Write failing Python diagnostic test**

```python
def test_synthetic_diagnostic_contains_map_identity() -> None:
    message = _diagnostic_message(
        ControllerStatus(InteractiveState.READY, "s", "wheel", "", ""),
        Time(sec=3),
        PlanningMapMetadata("synthetic", "map-abcd", 20260805, 50.0, 0.2, 0.12),
    )
    values = {item.key: item.value for item in message.status[0].values}
    assert values["source_kind"] == "synthetic"
    assert values["seed"] == "20260805"
    assert values["size_m"] == "50"
    assert values["resolution_m"] == "0.2"
    assert values["obstacle_occupancy"] == "0.12"
```

- [ ] **Step 2: Run the test and verify RED, then implement Python metadata publication**

Run `python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py`; expected failure is the old `_diagnostic_message` signature and four-key payload. Add canonical finite-number formatting and always publish all ten keys for both source types.

- [ ] **Step 3: Write failing Qt panel behavior test**

Extend the real diagnostic fixture with all ten keys, apply it to a real offscreen panel, and assert literal rendered values:

```cpp
EXPECT_EQ(panel->findChild<QLabel *>("source_label")->text(), "synthetic");
EXPECT_EQ(panel->findChild<QLabel *>("seed_label")->text(), "20260805");
EXPECT_EQ(panel->findChild<QLabel *>("size_label")->text(), "50 m x 50 m");
EXPECT_EQ(panel->findChild<QLabel *>("resolution_label")->text(), "0.2 m/cell");
EXPECT_EQ(panel->findChild<QLabel *>("occupancy_label")->text(), "12.0%");
```

The named break is missing, stale or malformed provenance in the left panel. Run the package gtest and observe missing label failures before adding widgets or parser branches.

- [ ] **Step 4: Implement strict ten-key parsing and panel labels**

Reject duplicate, missing or extra keys as `DIAGNOSTIC_KEYS_INVALID`. Validate source as `isaac` or `synthetic`, parse finite positive size/resolution, parse occupancy in `[0,1]`, and require an integer seed for synthetic. Keep platform buttons, state, phase and reason behavior unchanged.

- [ ] **Step 5: Build/test the plugin and commit**

```bash
colcon build --merge-install \
  --base-paths /mnt/data/WS/lunar-navigation/ros2_ws/src ros2_ws/src \
  --build-base build/tdd-panel --install-base install/tdd-panel \
  --packages-select lunar_isaac_rviz_plugins lunar_isaac_validation \
  --cmake-args -DBUILD_TESTING=ON
colcon test --merge-install \
  --base-paths /mnt/data/WS/lunar-navigation/ros2_ws/src ros2_ws/src \
  --build-base build/tdd-panel --install-base install/tdd-panel \
  --packages-select lunar_isaac_rviz_plugins lunar_isaac_validation
colcon test-result --test-result-base build/tdd-panel --verbose
git add ros2_ws/src/lunar_isaac_validation ros2_ws/src/lunar_isaac_rviz_plugins
git commit -m "feat: show synthetic map identity in rviz"
```

Expected: Qt gtest and Python diagnostic tests pass with no change to platform-selection behavior.

---

### Task 4: User-facing generator and launch commands

**Files:**
- Create: `scripts/generate_synthetic_map.py`
- Modify: `scripts/run_interactive_rviz.sh`
- Modify: `README.md`
- Create: `ros2_ws/src/lunar_isaac_validation/test/test_synthetic_cli.py`

**Interfaces:**
- Consumes: `generate_synthetic_map()` from Task 1.
- Produces: the exact generation CLI from the approved design and prints `map_id` plus absolute manifest path.
- Produces: `run_interactive_rviz.sh --synthetic-map PATH --install PATH`, mutually exclusive with `--manifest`.

- [ ] **Step 1: Write failing real-process CLI tests**

Use `subprocess.run` against the real scripts. Assert that generation into a temporary output root exits `0`, stdout contains the absolute manifest, and loading that manifest reports seed `20260805`. Assert that passing both source flags exits `10` before ROS startup and that `--help` documents both exclusive forms. The named break is a script that accepts ambiguous provenance or cannot reproduce the approved one-command parameters.

- [ ] **Step 2: Run CLI tests and verify RED**

```bash
python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test/test_synthetic_cli.py
```

Expected: the generator script is absent and the launch script rejects `--synthetic-map` as unknown.

- [ ] **Step 3: Implement the wrapper and launch selection**

The Python wrapper prepends the source package path exactly like `render_action_visualization.py`, parses finite numeric arguments, calls `generate_synthetic_map`, then prints compact JSON followed by the manifest path. The Bash launcher accepts exactly one source, resolves legacy manifests only below the external checkout and synthetic manifests only below `/home/kai/CodexDownloads/lunar_navigation/ros_random_maps`, and passes the corresponding flag to the controller. It must retain PID-scoped RViz shutdown.

- [ ] **Step 4: Update copyable README workflow and commit**

Document generation, manifest inspection, RViz launch, panel selection, `2D Goal Pose`, expected colors, no-connectivity semantics and safe exit. Then run the CLI tests plus `shellcheck` if available and `bash -n` unconditionally.

```bash
bash -n scripts/run_interactive_rviz.sh
python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test/test_synthetic_cli.py
git add scripts/generate_synthetic_map.py scripts/run_interactive_rviz.sh README.md \
  ros2_ws/src/lunar_isaac_validation/test/test_synthetic_cli.py
git commit -m "feat: launch synthetic maps in rviz"
```

---

### Task 5: Synthetic ROS integration coverage

**Files:**
- Create: `ros2_ws/src/lunar_isaac_validation/test/test_synthetic_interactive_integration.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_integration.py`

**Interfaces:**
- Consumes: installed `lunar_isaac_interactive`, generated manifest and existing platform parameter/capability files.
- Produces: process-level proof that all three platforms reach `READY`, map topics have the approved geometry, short safe-zone goals enter planning, and an obstacle goal is rejected locally.

- [ ] **Step 1: Write the skipped-by-default process integration test**

The test must run only when both `LUNAR_INTERACTIVE_INTEGRATION=1` and `LUNAR_SYNTHETIC_MAP_MANIFEST` are set. Subscribe to the ten-key diagnostic, `/environment/map_global`, `/environment/map_local`, `/lunar_isaac_validation/global_surface` and `/lunar_isaac_validation/local_hazards`. Hand-assert `250 × 250`, `0.2`, `map/odom`, synthetic seed `20260805`, and an obstacle count within `6875..8125` cells. Select wheel, legged and hopper in order; publish one literal target inside each start disc and one obstacle-cell target. Do not scan connectivity or search for a feasible long route.

- [ ] **Step 2: Run it before the final wiring and verify RED**

```bash
set +u
source /opt/ros/humble/setup.bash
source install/tdd-panel/setup.bash
set -u
export LUNAR_INTERACTIVE_INTEGRATION=1
export LUNAR_SYNTHETIC_MAP_MANIFEST="$(python3 scripts/generate_synthetic_map.py \
  --seed 20260805 --size-m 50 --resolution-m 0.2 \
  --obstacle-occupancy 0.12 --output-root "$PWD/artifacts/test-maps" \
  | tail -n 1)"
python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_synthetic_interactive_integration.py
```

Expected: fail at the first unimplemented or incorrectly installed synthetic source behavior, not at test setup.

- [ ] **Step 3: Make only the integration-level fixes required and rerun GREEN**

If a real defect appears, first reduce it to a focused failing unit test in the owning Task 1–4 test file, then fix production code and rerun both focused and process tests. Do not weaken geometry, occupancy, source or process-lifecycle assertions.

- [ ] **Step 4: Run legacy and synthetic process tests and commit**

Run both integration tests against the same install overlay, then commit only the integration test and any test-first fixes:

```bash
python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_integration.py \
  ros2_ws/src/lunar_isaac_validation/test/test_synthetic_interactive_integration.py
git add ros2_ws/src/lunar_isaac_validation
git commit -m "test: cover synthetic rviz planning sessions"
```

---

### Task 6: Authoritative build, regression, RViz evidence and integration

**Files:**
- Modify if verification exposes a tested defect: only the owning Task 1–5 file and its focused test.
- Generate outside Git under the configured `ros_random_maps`, `build`, `install`, `log` and `artifacts` roots, each keyed by the emitted map id or UTC run id.

**Interfaces:**
- Consumes: all Task 1–5 commits.
- Produces: a locally integrated external `main`, a generated hash-bound map, full test/build evidence, unchanged six-case `6/6`, and a live RViz2 session with reviewable screenshot.

- [ ] **Step 1: Review the complete feature diff against the approved specification**

Check every design requirement, run `git diff --check`, scan for unfinished-marker comments, confirm no USD/Isaac dependency entered the synthetic modules, and confirm formal regression files are unchanged except reusable type annotations if required.

- [ ] **Step 2: Run the full source-first suite**

```bash
set +u
source /opt/ros/humble/setup.bash
set -u
test "$ROS_DISTRO" = humble
export PYTHONPATH="$PWD/ros2_ws/src/lunar_isaac_validation${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test
```

Expected: all tests pass, with only environment-gated process tests skipped.

- [ ] **Step 3: Fast-forward the verified feature branch into external `main`**

The user explicitly authorized direct execution and no further confirmation gates. Verify the original checkout is clean, then from the original external checkout run `git merge --ff-only feat/synthetic-random-rviz-map`. Keep the worktree and feature branch until every post-merge verification completes.

- [ ] **Step 4: Generate the authoritative map and build a fresh overlay**

```bash
python3 scripts/generate_synthetic_map.py \
  --seed 20260805 --size-m 50 --resolution-m 0.2 \
  --obstacle-occupancy 0.12 \
  --output-root /home/kai/CodexDownloads/lunar_navigation/ros_random_maps
LUNAR_BUILD_RUN_ID="$(date -u +%Y%m%dT%H%M%S%6NZ)"
bash scripts/build_external.sh --run-id "$LUNAR_BUILD_RUN_ID"
```

Record the emitted manifest path and installed overlay path without using a broad filesystem search.

- [ ] **Step 5: Run colcon, both process integrations and formal six-case regression**

Use the fresh install overlay. Run `colcon test` for `lunar_planner_core`, `lunar_planner_ros`, `lunar_isaac_rviz_plugins` and `lunar_isaac_validation`, then `colcon test-result --verbose`. Run both interactive process tests with the generated manifest. Finally run `scripts/run_action_regression.sh` against the frozen Isaac manifest and existing lock, and require exact summary `{"exit_code":0,"failed":0,"passed":6,"total":6}`.

- [ ] **Step 6: Launch RViz2 and collect visible evidence**

Start `scripts/run_interactive_rviz.sh --synthetic-map "$LUNAR_SYNTHETIC_MAP_MANIFEST" --install "$LUNAR_FRESH_INSTALL"` in a managed terminal session, where both variables are the exact paths emitted in Step 4. Select each platform through its Trigger service or the left panel, publish one safe-zone goal, and capture a screenshot showing the `50 m × 50 m` map, red obstacles, current platform, target, route or terminal reason, and the synthetic map metadata labels. Store the screenshot and a small UTF-8 evidence note under the run's external artifact directory. Leave the interactive RViz2 window open for the user unless it exits unexpectedly.

- [ ] **Step 7: Final verification and status report**

Immediately before reporting completion, rerun `git status --short`, `git log -5 --oneline`, the focused synthetic tests, `colcon test-result --verbose`, inspect the formal `summary.json`, inspect RViz/controller logs for traceback, segfault or context errors, and verify the live RViz PID belongs to this launch. Report exact commit IDs, map id/manifest, install path, artifact/screenshot paths, test counts, regression summary, ROS domain and the copyable launch command.
