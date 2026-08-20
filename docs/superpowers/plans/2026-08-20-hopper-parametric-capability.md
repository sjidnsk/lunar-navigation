# Hopper Parametric Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load, preserve, validate, and deploy the approved `hopper-v1` HOPPER capability from numeric parameters without URDF or mesh assets.

**Architecture:** Add a tracked v2 HOPPER document using `parametric_envelope`, then extend the existing typed capability and strict ROS loader so every approved frozen HOPPER field survives projection into planning. Preserve the already-approved fixed single-hop reference semantics: no runtime fuel subscription, inventory, or decrement is introduced.

**Tech Stack:** C++20, yaml-cpp, ROS 2 Humble, pybind11, Python 3, pytest, ament/colcon, YAML.

**Spec:** `docs/superpowers/specs/2026-08-07-hopper-platform-capability-design.md`, superseded for fuel semantics by `docs/superpowers/specs/2026-08-08-formal-capability-and-hopper-no-fuel-budget-design.md`, with the user-approved deployment identity and parametric geometry recorded in this plan.

## Global Constraints

- Deployment identity is exactly `platform_id: hopper`, `platform_type: HOPPER`, `capability_version: hopper-v1`, `base_frame_id: base_link`.
- Geometry is exactly `geometry_source: {type: parametric_envelope}`; do not create or require URDF/mesh files.
- Preserve these exact values: `301.0 s`, `0.45 m`, `0.17453292519943295 rad`, `0.05 m`, `0.55 m`, `0.2 m`, `0.2 m`, `0.1`, gravity `[0.0, 0.0, -1.62] m/s2`, standard gravity `9.80665 m/s2`, reference total mass `20.0 kg`, reference usable fuel `0.2 kg`, reference horizontal range `100.0 m`, reference elevation delta `0.0 m`, and `runtime_fallback_allowed: false`.
- `reference_remaining_usable_fuel_mass_kg` remains the approved source-field spelling but means a non-decrementing fixed single-hop reference quantity. Do not add a propellant-state subscription, live inventory, cross-hop decrement, or runtime fallback.
- HOPPER goals remain exact POINT goals with zero position tolerance and no yaw constraint. Do not add motion primitives.
- Strictly reject wrong `hopper/hopper-v1` identity, extra/missing root or HOPPER fields, partial/missing/extra sources, unsupported source values, non-finite/invalid values, and mixed parametric/URDF geometry.
- Do not modify `ros2_ws/src/lunar_navigation_config/config/three_platform_capability_freeze_v1.yaml` or its digest.
- Preserve legacy URDF HOPPER fixtures and the external legacy capability-bundle compatibility already supported by the repository.
- Keep tracked `wheel.yaml` and `legged.yaml`; installing HOPPER changes only the active `/opt/luna/capabilities/platform.yaml`.
- Do not invent observation capability values. Full lifecycle startup remains blocked when `/opt/luna/capabilities/observation.yaml` is absent.
- Run ROS builds and C++ tests in `ros2-humble` after sourcing `/opt/ros/humble/setup.bash`.

---

### Task 1: Define and integrate the formal `hopper-v1` capability

**Files:**
- Create: `deployment/config/hopper.yaml`
- Create: `tests/deployment/test_hopper_capability.py`
- Modify: `ros2_ws/src/lunar_planner_core/include/lunar_planner_core/types/platform_capability.hpp`
- Modify: `ros2_ws/src/lunar_planner_core/src/shared/projection_cache.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/capability_loader.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/capability_loader_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp`
- Modify only the existing formal freeze adapter and its tests where required to carry the complete typed HOPPER projection without changing the freeze payload.
- Modify: `deployment/docs/README.runtime.md`
- Modify: `deployment/docs/COMMANDS.runtime.md`

**Interfaces:**
- Consumes: the canonical HOPPER entry in `three_platform_capability_freeze_v1.yaml` and the fixed single-hop semantics from the superseding 2026-08-08 design.
- Produces: `deployment/config/hopper.yaml`; a `HopperCapability` that carries `gravity_mps2`, `reference_horizontal_range_m`, `reference_elevation_delta_m`, and `runtime_fallback_allowed` in addition to existing planning fields; strict `hopper/hopper-v1` parametric loading; stable projection hashing and pybind/freeze projection for the new typed fields.

- [ ] **Step 1: Write failing deployment and loader contracts**

The deployment test must parse the real YAML and assert the complete literal document behavior, including:

```python
assert document["schema_version"] == "platform-control-capability-source/v2"
assert document["platform"]["platform_id"] == "hopper"
assert document["platform"]["platform_type"] == "HOPPER"
assert document["platform"]["capability_version"] == "hopper-v1"
assert document["platform"]["base_frame_id"] == "base_link"
assert document["platform"]["unknown_fields"] == []
assert document["geometry_source"] == {"type": "parametric_envelope"}
assert document["hopper"]["specific_impulse_s"] == 301.0
assert document["hopper"]["gravity_mps2"] == [0.0, 0.0, -1.62]
assert document["hopper"]["reference_remaining_usable_fuel_mass_kg"] == 0.2
assert document["hopper"]["reference_horizontal_range_m"] == 100.0
assert document["hopper"]["reference_elevation_delta_m"] == 0.0
assert document["hopper"]["runtime_fallback_allowed"] is False
assert "motion_primitives" not in document["hopper"]
```

Add real loader tests that load this tracked YAML with a temporary valid observation document and assert parametric geometry, empty URDF/mesh paths, exact identity, all typed numeric values, and `runtime_fallback_allowed == false`. Add mutation cases for wrong identity/version/type/base frame, an extra root key, an extra HOPPER key, each missing required source, an extra source, an unsupported source type, mixed `urdf_file`, invalid reference mass/fuel, non-lunar or non-finite gravity, and `runtime_fallback_allowed: true`.

- [ ] **Step 2: Run the new focused tests and record RED evidence**

```bash
python3 -m pytest -q tests/deployment/test_hopper_capability.py
```

Inside `ros2-humble`, build the current packages if required and run only the new HOPPER loader filter. Expected failure reason: the tracked document and approved parametric HOPPER loader do not yet exist; the tests must not fail from an invalid fixture or missing dependency.

- [ ] **Step 3: Add the exact tracked HOPPER document and minimal typed implementation**

Use these exact source mappings:

```yaml
specific_impulse_s: project_engineering_baseline
landing_support_radius_m: project_engineering_baseline
flight_collision_radius_m: project_engineering_baseline
maximum_landing_slope_rad: planning_safety_baseline
maximum_landing_plane_residual_m: planning_safety_baseline
landing_lateral_margin_m: planning_safety_baseline
flight_map_margin_m: planning_safety_baseline
reachability_delta_v_margin_ratio: planning_safety_baseline
gravity_mps2: planning_safety_baseline
standard_gravity_mps2: planning_safety_baseline
reference_total_mass_kg: project_engineering_baseline
reference_remaining_usable_fuel_mass_kg: project_engineering_baseline
reference_horizontal_range_m: project_engineering_baseline
reference_elevation_delta_m: project_engineering_baseline
```

Flatten the approved sections into the existing runtime `hopper:` block, keep the approved source spelling `reference_remaining_usable_fuel_mass_kg`, and map it internally to the existing non-decrementing reference propellant quantity. Validate exact keys and exact source keys for the approved parametric identity. Preserve legacy HOPPER parsing only in legacy URDF/package-relative mode. Hash and bind every new typed field so ROS and formal-freeze projections cannot silently disagree.

- [ ] **Step 4: Run focused GREEN verification**

Run the deployment contract, the complete capability loader test binary, core HOPPER/ballistic tests, and the affected formal-freeze/project-adapter Python tests. Expected: all selected tests pass with zero new warnings; the immutable freeze checker reports the same payload/digest.

- [ ] **Step 5: Run integration verification and update operator docs**

Inside `ros2-humble`, copy the clean worktree to a dedicated non-repository path, source Humble, and run `colcon build --packages-up-to lunar_planner_ros lunar_planner_training_bridge`. Run the loader and PlanMotionServer HOPPER tests against the built artifacts. Update runtime docs to list wheel, legged, and hopper selection commands and explicitly state that only one becomes active.

- [ ] **Step 6: Commit the implementation**

```bash
git add deployment/config/hopper.yaml tests/deployment/test_hopper_capability.py \
  ros2_ws/src/lunar_planner_core ros2_ws/src/lunar_planner_ros \
  ros2_ws/src/lunar_planner_training_bridge deployment/docs
git commit -m "feat(planner): deploy hopper parametric capability"
```

- [ ] **Step 7: Install only after review**

After independent review is clean, install the reviewed tracked file as `/opt/luna/capabilities/platform.yaml`, compare SHA-256 values byte-for-byte, and verify wheel/legged tracked files remain present. Report observation capability separately; do not claim full planner lifecycle readiness without it.
