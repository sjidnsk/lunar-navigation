# RViz Goal Policy and Planning Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the external interactive RViz harness use explicit 0.20 m goal tolerances, show all global obstacles, display one clearly distinguishable current platform, and expose the planner's actual trajectory/collision evidence.

**Architecture:** A strict versioned YAML is loaded once at process startup and frozen into platform-specific Action goals. The bridge caches one complete global PointCloud2 with semantic colors while continuing to publish a brighter platform-local hazard cloud. Platform visualization moves to its own MarkerArray layer. Standard planner diagnostics are parsed into immutable interactive status and rendered by the existing RViz panel; the harness never recomputes planning or collision validity.

**Tech Stack:** Python 3.10, ROS 2 Humble/rclpy, RViz2, sensor_msgs/PointCloud2, visualization_msgs/MarkerArray, Qt/C++, pytest, GoogleTest.

## Global Constraints

- Work in an isolated Git worktree of `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression`; do not edit its clean `main` checkout in place.
- Keep `/lunar_isaac_validation/global_surface` and `/lunar_isaac_validation/local_hazards` as PointCloud2. Do not add `grid_map_rviz_plugin`.
- Add only the RViz-specific `/lunar_isaac_validation/platform_state` MarkerArray. Do not change the planner Action or map Topic schemas.
- Load `interactive_goal.yaml` once at node startup. Require exact keys, finite numbers, positive position tolerance, and positive yaw tolerance no greater than pi. Never silently enlarge a request tolerance.
- Default all three platforms to 0.20 m position tolerance and 0.2617993877991494 rad yaw tolerance.
- Build the complete global cloud once per frozen map/session and only refresh its header timestamp on republish; it must not enter the Action critical path.
- Display one current platform only. Platform visibility effects must not alter physical collision dimensions.
- Close only the RViz PID owned by the test session: send one `WM_DELETE_WINDOW`, wait, and kill only that exact PID on timeout.
- Keep generated maps, colcon outputs, screenshots, and evidence under `/home/kai/CodexDownloads/lunar_navigation/planner_correctness/external/`.

---

### Task 1: Add a strict versioned interactive goal policy

**Files:**
- Create: `ros2_ws/src/lunar_isaac_validation/config/interactive_goal.yaml`
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_goal.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/setup.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_goal.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py`

- [ ] **Step 1: Write strict loader tests first**

Require schema `lunar-interactive-goal-policy/v1`, exact platform keys `wheel`, `legged`, `hopper`, exact numeric keys `position_tolerance_m` and `yaw_tolerance_rad`, and immutable returned data. Add rejection tests for missing/extra keys, booleans, strings, NaN/Inf, non-positive values, and yaw above pi.

- [ ] **Step 2: Prove hard-coded tolerances fail**

Run:

```bash
python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_goal.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py
```

Expected: no loader exists and the current 0.5/0.75 m constants disagree with 0.20 m.

- [ ] **Step 3: Implement and freeze the loader**

Use `yaml.safe_load`, reject `bool` as a numeric subtype, convert accepted values to float only after finite/range checks, and return frozen dataclasses/mapping proxies. Remove both duplicate hard-coded tolerance maps. Make the policy path a required explicit node/CLI argument.

- [ ] **Step 4: Install the YAML and pass it from the launcher**

Include `config/interactive_goal.yaml` in `setup.py`; update the executable argument parser and node constructor so every generated Action goal receives exactly the selected platform policy.

- [ ] **Step 5: Run tests and commit**

```bash
git add ros2_ws/src/lunar_isaac_validation
git commit -m "feat: configure strict interactive goal tolerances"
```

### Task 2: Align external hopper checks with aim-point tolerance semantics

**Files:**
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/hop_checks.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/scenario_qualifier.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_hop_checks.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_scenario_qualifier.py`

- [ ] **Step 1: Replace the old goal-clipping expectation with failing physical tests**

Add a valid 1.327322 m² landing region whose center is inside a 0.05 m goal tolerance but whose boundary crosses it. Add negatives for a center outside tolerance, unsafe polygon cell, insufficient region area, and a point outside the certified polygon.

- [ ] **Step 2: Implement one shared semantic helper**

Validate `distance(aim, goal) <= position_tolerance`, `region_area >= minimum_landing_region_area_m2`, `aim in polygon`, and every represented region cell's terrain/clearance safety. Do not intersect or clip the certified polygon with the goal circle.

- [ ] **Step 3: Run tests and commit**

```bash
python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_hop_checks.py \
  ros2_ws/src/lunar_isaac_validation/test/test_scenario_qualifier.py
git add ros2_ws/src/lunar_isaac_validation
git commit -m "fix: qualify hopper aim and landing region independently"
```

### Task 3: Render complete global obstacle semantics without slowing Actions

**Files:**
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/visual_map_cloud.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_bridge_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_visual_map_cloud.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py`

- [ ] **Step 1: Add failing binary PointCloud2 color/count tests**

Construct a global map with normal, obstacle, forbidden, obstacle-plus-forbidden, invalid, and non-finite cells. Decode every `<fffI>` point and assert: normal terrain is gray; obstacle is red at `elevation + obstacle_height`; forbidden is purple and wins conflicts; invalid/non-finite cells are absent. Assert local hazards remain window-limited and brighter.

- [ ] **Step 2: Implement semantic point generation**

Keep the existing cloud layout and endianness. Use stable row-major point order and fixed RGBA constants. Validate all source layer lengths before iterating so malformed maps fail closed.

- [ ] **Step 3: Cache the global cloud payload**

Build the global point data once when the frozen planning map is installed. On periodic publication, replace only `header.stamp`; do not rebuild colors or traverse cells in a goal callback. Continue rebuilding local hazards only when platform/local-window state changes.

- [ ] **Step 4: Run tests and commit**

```bash
python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_visual_map_cloud.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_bridge_node.py
git add ros2_ws/src/lunar_isaac_validation
git commit -m "feat: show complete global obstacle semantics"
```

### Task 4: Move the current platform into an independent RViz Marker layer

**Files:**
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/rviz_evidence.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/config/rviz/lunar_interactive_planning.rviz`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_rviz_evidence.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py`

- [ ] **Step 1: Write failing layer and atomic-switch tests**

Require a seventh `platform_state` layer containing one mesh at physical scale, one high-contrast outline/halo, one yaw arrow, one type label, and one true-start ring. Assert `local_execution` no longer owns platform/start markers. Switch wheel -> hopper and require DELETE markers for every old stable namespace/id before new ADD markers.

- [ ] **Step 2: Implement stable Marker IDs and visual-only scale separation**

Use package URIs for existing proxy meshes and their real physical scale. Add a non-colliding halo/outline only as a separate RViz Marker. Store the original frozen start independently from the moving current pose. Keep marker frame and timestamp consistent with the pose source.

- [ ] **Step 3: Wire the dedicated publisher and RViz display**

Publish `/lunar_isaac_validation/platform_state` as MarkerArray with transient-local durability matching other evidence layers. Add one RViz MarkerArray display named `Current platform`; retain PointCloud2 displays and remove no working standard plugin.

- [ ] **Step 4: Run tests and commit**

```bash
python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_rviz_evidence.py \
  ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py
git add ros2_ws/src/lunar_isaac_validation
git commit -m "feat: display current platform independently"
```

### Task 5: Show actual planner tolerances and execution evidence in the RViz panel

**Files:**
- Modify: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/interactive_node.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_node.py`
- Modify: `ros2_ws/src/lunar_isaac_rviz_plugins/include/lunar_isaac_rviz_plugins/lunar_planner_panel.hpp`
- Modify: `ros2_ws/src/lunar_isaac_rviz_plugins/src/lunar_planner_panel.cpp`
- Modify: `ros2_ws/src/lunar_isaac_rviz_plugins/test/lunar_planner_panel_test.cpp`

- [ ] **Step 1: Add failing diagnostic parser tests**

Feed standard DiagnosticArray entries containing all approved planner keys. Assert exact pass-through of `OPTIMIZED`, `DISCRETE_FALLBACK`, `STATIONARY`, `CERTIFIED_HOP`, start/endpoint error, curvature, collision state, phase timings, full ordered warning list, and configured position/yaw tolerances. Missing or malformed fields must display `unknown`, not imply success.

- [ ] **Step 2: Extend the immutable interactive status payload**

Whitelist the new keys, parse only finite numeric values, retain stable strings, and include the active platform's frozen policy values. Do not infer optimization from an empty warning list and do not recompute collision validity.

- [ ] **Step 3: Extend the C++ panel**

Add compact rows for position/yaw tolerance, trajectory mode, start error, endpoint error, maximum curvature, collision certification, and warning codes. Use a visible warning color for `DISCRETE_FALLBACK`, unknown collision evidence, or non-empty warnings; keep planner success/failure separate from optimization mode.

- [ ] **Step 4: Run Python and C++ panel tests and commit**

Build/test `lunar_isaac_rviz_plugins` under the external artifact prefix, run the focused pytest, then:

```bash
git add ros2_ws/src/lunar_isaac_validation ros2_ws/src/lunar_isaac_rviz_plugins
git commit -m "feat: expose planner execution evidence in rviz"
```

### Task 6: Wire the launcher and qualify the interactive 50 m workflow

**Files:**
- Modify: `scripts/run_interactive_rviz.sh`
- Modify: `README.md`
- Modify: `test/test_cli_contract.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_interactive_integration.py`
- Modify: `ros2_ws/src/lunar_isaac_validation/test/test_synthetic_interactive_integration.py`

- [ ] **Step 1: Add launcher contract tests first**

Require the strict goal-policy argument, existing planner/capability/map paths, no `set -u` around ROS setup, and no `grid_map_rviz_plugin`. Verify the script never manufactures or enlarges tolerances.

- [ ] **Step 2: Update launcher and usage documentation**

Pass the installed or source policy path explicitly. Document platform selection, `2D Goal Pose`, expected seven display layers, obstacle/forbidden colors, platform marker contents, mode meanings, and the fact that success inside 0.20 m is valid while the exact endpoint error is shown.

- [ ] **Step 3: Run the complete external unit suite**

```bash
python3 -m pytest -q test ros2_ws/src/lunar_isaac_validation/test
```

Build and test both external ROS packages with ROS 2 Humble using output directories under `/home/kai/CodexDownloads/lunar_navigation/planner_correctness/external/`.

- [ ] **Step 4: Run live integration against the newly built main planner**

Launch the deterministic 50 m × 50 m, 0.2 m map; switch through wheel, legged, and hopper; send one far positive and one physical negative for each; verify all seven layers and panel evidence. Save screenshots/logs outside both repositories. Use the owned-PID RViz close helper exactly once and wait for clean process exit.

- [ ] **Step 5: Verify repository cleanliness and commit**

```bash
git diff --check
git status --short
git add README.md scripts/run_interactive_rviz.sh test ros2_ws/src
git commit -m "test: qualify rviz planning evidence workflow"
```

## Completion Gate

- [ ] All three generated Action goals carry exactly 0.20 m and 15-degree defaults from YAML.
- [ ] Hopper landing-region area can extend beyond the target tolerance while the aim remains inside it.
- [ ] Global obstacle/forbidden counts and colors match the frozen full map; local hazards remain visually separate.
- [ ] Exactly one current platform is visible with mesh, halo, yaw, label, and true-start evidence.
- [ ] The panel distinguishes optimized, discrete fallback, stationary, and certified-hop references and shows actual errors/warnings.
- [ ] No Action/map schema or planning logic was duplicated in the external harness.
- [ ] All Python, plugin, launcher, integration, safe-exit, and `git diff --check` validations pass.
