# Wheel Parametric Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load and deploy the formal `wheel-v1` WHEELED capability from numeric parameters without creating or requiring URDF/mesh assets.

**Architecture:** Extend the v2 platform capability document with a mutually exclusive `parametric_envelope` geometry source while preserving the legacy package-relative URDF/mesh path. Add one loader entry point that selects either absolute external files or package-relative files, then deploy the reviewed `wheel-v1` document to `/opt/luna/capabilities/platform.yaml`.

**Tech Stack:** C++20, yaml-cpp, ROS 2 Humble lifecycle nodes, ament/colcon, Python 3, pytest, YAML.

**Spec:** `docs/superpowers/specs/2026-08-20-wheel-parametric-capability-design.md`

## Global Constraints

- Platform identity is exactly `platform_id: wheel`, `platform_type: WHEELED`, `capability_version: wheel-v1`, `base_frame_id: base_footprint`.
- Geometry is exactly the reviewed `1.301 x 0.808 x 1.363 m` full-vehicle envelope; do not generate URDF or mesh files.
- Use `maximum_curvature_per_m: 5.0`; the `0.2 m` minimum moving radius is speed-limited by `maximum_spin_rate_radps: 0.389923188554511`.
- Do not enable lateral or diagonal primitives; retain forward, reverse, moving arcs, spin, and stop/switch only.
- Do not modify `three_platform_capability_freeze_v1.yaml` or its digest.
- Do not invent observation capability values. Full runtime startup remains blocked until `sensor_range_m` and `sensor_fov_deg` are provided.
- Preserve the existing user changes in deployment docs, `deployment/luna_runtime/cli.py`, and `tests/deployment/test_cli.py`.
- Run ROS build and C++ tests inside `ros2-humble` after sourcing `/opt/ros/humble/setup.bash`.

---

### Task 1: Replace the operator worksheet with the formal `wheel-v1` document

**Files:**
- Modify: `deployment/config/wheel.yaml`
- Create: `tests/deployment/test_wheel_capability.py`

**Interfaces:**
- Consumes: The exact reviewed values in the design specification.
- Produces: A tracked `platform-control-capability-source/v2` YAML document at `deployment/config/wheel.yaml`.

- [ ] **Step 1: Write the failing document contract test**

Create `tests/deployment/test_wheel_capability.py` with an exact identity, geometry, motion, provenance, source, and primitive check:

```python
from pathlib import Path
import math
import yaml


ROOT = Path(__file__).resolve().parents[2]
CAPABILITY = ROOT / "deployment/config/wheel.yaml"


def test_wheel_v1_is_complete_parametric_runtime_capability() -> None:
    document = yaml.safe_load(CAPABILITY.read_text(encoding="utf-8"))
    assert document["schema_version"] == "platform-control-capability-source/v2"
    assert document["platform"] == {
        "platform_id": "wheel",
        "platform_type": "WHEELED",
        "capability_version": "wheel-v1",
        "base_frame_id": "base_footprint",
        "provenance": {
            "level": "approved_user_parameter_baseline",
            "design_document": "docs/superpowers/specs/2026-08-20-wheel-parametric-capability-design.md",
        },
        "unknown_fields": [
            "bare_mass_kg", "nominal_payload_kg", "maximum_payload_kg",
            "center_of_mass_height_m", "suspension_type", "maximum_drive_effort",
            "manufacturer_rated_speed_mps", "longitudinal_traction_coefficient",
            "lateral_traction_coefficient", "verified_cross_slope_limit_rad",
        ],
    }
    assert document["geometry_source"] == {"type": "parametric_envelope"}
    wheel = document["wheeled"]
    assert wheel["body_extent_m"] == [1.301, 0.808, 1.363]
    assert wheel["footprint_xy_m"] == [
        [0.6505, 0.404], [0.6505, -0.404],
        [-0.6505, -0.404], [-0.6505, 0.404],
    ]
    assert wheel["wheel_count"] == 4
    assert wheel["wheel_center_xy_m"] == [
        [0.4075, 0.3115], [0.4075, -0.3115],
        [-0.4075, -0.3115], [-0.4075, 0.3115],
    ]
    assert wheel["maximum_curvature_per_m"] == 5.0
    assert math.isclose(wheel["maximum_spin_rate_radps"], 0.389923188554511, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(wheel["maximum_yaw_acceleration_radps2"], 0.389923188554511, rel_tol=0.0, abs_tol=1e-15)
    assert [item["kind"] for item in wheel["motion_primitives"]] == [
        "FORWARD", "REVERSE", "FORWARD_ARC", "FORWARD_ARC",
        "REVERSE_ARC", "REVERSE_ARC", "SPIN_COUNTERCLOCKWISE",
        "SPIN_CLOCKWISE", "STOP_AND_SWITCH",
    ]
    required_sources = {
        "reference_point", "body_extent_m", "footprint_xy_m", "wheel_count",
        "wheel_diameter_m", "wheel_width_m", "wheelbase_m", "track_width_m",
        "wheel_center_xy_m", "minimum_underbody_clearance_m", "minimum_clearance_m",
        "maximum_forward_speed_mps", "maximum_reverse_speed_mps",
        "maximum_spin_rate_radps", "maximum_acceleration_mps2",
        "maximum_braking_deceleration_mps2", "maximum_yaw_acceleration_radps2",
        "maximum_lateral_acceleration_mps2", "maximum_curvature_per_m",
        "maximum_surface_slope_rad", "maximum_local_obstacle_relief_m",
        "allow_unsupported_gap", "roughness_handling", "motion_primitives",
    }
    assert set(document["sources"]) == required_sources
```

- [ ] **Step 2: Run the test and verify the worksheet fails parsing**

Run:

```bash
python3 -m pytest -q tests/deployment/test_wheel_capability.py
```

Expected: FAIL because the current Chinese worksheet is not the v2 runtime document and has empty identity/motion fields.

- [ ] **Step 3: Replace `deployment/config/wheel.yaml` with the complete capability document**

Write the exact values below, including all nine motion primitives.

```yaml
schema_version: platform-control-capability-source/v2
platform:
  platform_id: wheel
  platform_type: WHEELED
  capability_version: wheel-v1
  base_frame_id: base_footprint
  provenance:
    level: approved_user_parameter_baseline
    design_document: docs/superpowers/specs/2026-08-20-wheel-parametric-capability-design.md
  unknown_fields: [bare_mass_kg, nominal_payload_kg, maximum_payload_kg, center_of_mass_height_m, suspension_type, maximum_drive_effort, manufacturer_rated_speed_mps, longitudinal_traction_coefficient, lateral_traction_coefficient, verified_cross_slope_limit_rad]
geometry_source: {type: parametric_envelope}
wheeled:
  reference_point: base_footprint
  body_extent_m: [1.301, 0.808, 1.363]
  footprint_xy_m: [[0.6505, 0.404], [0.6505, -0.404], [-0.6505, -0.404], [-0.6505, 0.404]]
  wheel_count: 4
  wheel_diameter_m: 0.304
  wheel_width_m: 0.148
  wheelbase_m: 0.815
  track_width_m: 0.623
  wheel_center_xy_m: [[0.4075, 0.3115], [0.4075, -0.3115], [-0.4075, -0.3115], [-0.4075, 0.3115]]
  minimum_underbody_clearance_m: 0.214
  maximum_local_obstacle_relief_m: 0.2
  allow_unsupported_gap: false
  maximum_forward_speed_mps: 0.2
  maximum_reverse_speed_mps: 0.2
  maximum_spin_rate_radps: 0.389923188554511
  maximum_acceleration_mps2: 0.2
  maximum_braking_deceleration_mps2: 0.2
  maximum_yaw_acceleration_radps2: 0.389923188554511
  maximum_lateral_acceleration_mps2: 0.2
  maximum_curvature_per_m: 5.0
  maximum_slope_rad: 0.174532925199433
  minimum_clearance_m: 0.1
  roughness_handling: COST_SPEED_AND_LOCAL_RECHECK
  motion_primitives:
    - primitive_id: forward
      kind: FORWARD
      relative_end_pose: {position_m: [0.2, 0.0, 0.0], orientation_wxyz: [1.0, 0.0, 0.0, 0.0]}
    - primitive_id: reverse
      kind: REVERSE
      relative_end_pose: {position_m: [-0.2, 0.0, 0.0], orientation_wxyz: [1.0, 0.0, 0.0, 0.0]}
    - primitive_id: forward-arc-left
      kind: FORWARD_ARC
      relative_end_pose: {position_m: [0.039018064403226, 0.003842943919354, 0.0], orientation_wxyz: [0.995184726672197, 0.0, 0.0, 0.098017140329561]}
    - primitive_id: forward-arc-right
      kind: FORWARD_ARC
      relative_end_pose: {position_m: [0.039018064403226, -0.003842943919354, 0.0], orientation_wxyz: [0.995184726672197, 0.0, 0.0, -0.098017140329561]}
    - primitive_id: reverse-arc-left
      kind: REVERSE_ARC
      relative_end_pose: {position_m: [-0.039018064403226, 0.003842943919354, 0.0], orientation_wxyz: [0.995184726672197, 0.0, 0.0, -0.098017140329561]}
    - primitive_id: reverse-arc-right
      kind: REVERSE_ARC
      relative_end_pose: {position_m: [-0.039018064403226, -0.003842943919354, 0.0], orientation_wxyz: [0.995184726672197, 0.0, 0.0, 0.098017140329561]}
    - primitive_id: spin-left
      kind: SPIN_COUNTERCLOCKWISE
      relative_end_pose: {position_m: [0.0, 0.0, 0.0], orientation_wxyz: [0.995184726672197, 0.0, 0.0, 0.098017140329561]}
    - primitive_id: spin-right
      kind: SPIN_CLOCKWISE
      relative_end_pose: {position_m: [0.0, 0.0, 0.0], orientation_wxyz: [0.995184726672197, 0.0, 0.0, -0.098017140329561]}
    - primitive_id: stop-switch
      kind: STOP_AND_SWITCH
      relative_end_pose: {position_m: [0.0, 0.0, 0.0], orientation_wxyz: [1.0, 0.0, 0.0, 0.0]}
sources:
  reference_point: derived
  body_extent_m: user_provided_dimension
  footprint_xy_m: derived
  wheel_count: user_confirmed_capability
  wheel_diameter_m: user_provided_dimension
  wheel_width_m: user_provided_dimension
  wheelbase_m: user_provided_dimension
  track_width_m: user_provided_dimension
  wheel_center_xy_m: derived
  minimum_underbody_clearance_m: user_provided_dimension
  minimum_clearance_m: user_approved_planning_policy
  maximum_forward_speed_mps: user_confirmed_capability
  maximum_reverse_speed_mps: user_confirmed_capability
  maximum_spin_rate_radps: derived
  maximum_acceleration_mps2: user_confirmed_capability
  maximum_braking_deceleration_mps2: user_confirmed_capability
  maximum_yaw_acceleration_radps2: derived
  maximum_lateral_acceleration_mps2: user_approved_planning_policy
  maximum_curvature_per_m: user_confirmed_capability
  maximum_surface_slope_rad: user_confirmed_capability
  maximum_local_obstacle_relief_m: user_confirmed_capability
  allow_unsupported_gap: user_confirmed_capability
  roughness_handling: user_approved_planning_policy
  motion_primitives: user_approved_planning_policy
```

- [ ] **Step 4: Run the exact document test**

Run: `python3 -m pytest -q tests/deployment/test_wheel_capability.py`

Expected: `1 passed`.

- [ ] **Step 5: Commit only the formal document and its test**

```bash
git add deployment/config/wheel.yaml tests/deployment/test_wheel_capability.py
git commit -m "feat(deploy): define wheel v1 capability"
```

---

### Task 2: Add parameter-envelope parsing and strict geometry/source validation

**Files:**
- Modify: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/capability_loader.hpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/capability_loader.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/capability_loader_test.cpp`

**Interfaces:**
- Consumes: `geometry_source.type: parametric_envelope`, `wheeled.wheel_count`, `wheeled.wheel_center_xy_m`, and per-field `sources`.
- Produces: `GeometrySourceKind`, `ParametricWheeledGeometry`, and populated audit fields on `LoadedCapabilities`; legacy URDF fields remain populated only for `kUrdfMesh`.

- [ ] **Step 1: Add failing parameter-geometry loader tests**

Add these public result assertions to `capability_loader_test.cpp`:

Add `#include <iterator>` for `std::istreambuf_iterator`.

```cpp
std::filesystem::path RepositoryRoot() {
  return std::filesystem::path{__FILE__}.parent_path()
      .parent_path().parent_path().parent_path().parent_path();
}

std::string TrackedWheelYaml() {
  std::ifstream stream{RepositoryRoot() / "deployment/config/wheel.yaml"};
  return std::string{
      std::istreambuf_iterator<char>{stream},
      std::istreambuf_iterator<char>{}};
}

CapabilityLoadResult LoadParametricText(const std::string& document) {
  const auto share = UniqueShare("parametric-text");
  WriteObservation(share);
  Write(share / "config" / "platform.yaml", document);
  return CapabilityLoader{}.LoadFromShareDirectory(
      share, "config/platform.yaml", "config/observation.json");
}

std::string WithoutSource(std::string document, const std::string& field) {
  const std::string line = "  " + field + ": ";
  const auto begin = document.find(line, document.find("sources:\n"));
  const auto end = document.find('\n', begin);
  document.erase(begin, end - begin + 1U);
  return document;
}

TEST(CapabilityLoader, LoadsParametricWheelWithoutUrdfOrMesh) {
  const auto share = UniqueShare("parametric-wheel");
  WriteObservation(share);
  Write(share / "config" / "platform.yaml", TrackedWheelYaml());
  const auto result = CapabilityLoader{}.LoadFromShareDirectory(
      share, "config/platform.yaml", "config/observation.json");
  ASSERT_TRUE(result.ok())
      << (result.error ? result.error->detail : std::string{});
  EXPECT_EQ(result.capabilities->geometry_source_kind,
            GeometrySourceKind::kParametricEnvelope);
  EXPECT_TRUE(result.capabilities->urdf_path.empty());
  EXPECT_TRUE(result.capabilities->mesh_paths.empty());
  ASSERT_TRUE(result.capabilities->parametric_wheeled_geometry.has_value());
  EXPECT_EQ(result.capabilities->parametric_wheeled_geometry->wheel_count, 4U);
  EXPECT_EQ(result.capabilities->parametric_wheeled_geometry->wheel_center_xy_m.size(), 4U);
  const auto& wheel = std::get<WheeledCapability>(result.capabilities->platform);
  EXPECT_DOUBLE_EQ(wheel.maximum_curvature_per_m, 5.0);
}

TEST(CapabilityLoader, RejectsMixedOrInvalidParametricGeometry) {
  std::string mixed = TrackedWheelYaml();
  mixed.insert(mixed.find("wheeled:\n"), "  urdf_file: urdf/rover.urdf\n");
  EXPECT_EQ(LoadParametricText(mixed).error->reason_code,
            "CAPABILITY_SCHEMA_INVALID");
  std::string wrong_count = TrackedWheelYaml();
  wrong_count.replace(wrong_count.find("wheel_count: 4"), 14U, "wheel_count: 3");
  EXPECT_EQ(LoadParametricText(wrong_count).error->reason_code,
            "CAPABILITY_VALUE_INVALID");
  std::string outside = TrackedWheelYaml();
  const auto centers = outside.find("wheel_center_xy_m:");
  outside.replace(centers, outside.find('\n', centers) - centers,
                  "wheel_center_xy_m: [[2.0, 0.0], [0.4075, -0.3115], [-0.4075, -0.3115], [-0.4075, 0.3115]]");
  EXPECT_EQ(LoadParametricText(outside).error->reason_code,
            "CAPABILITY_VALUE_INVALID");
}

TEST(CapabilityLoader, RejectsMissingOrUnsupportedFieldSources) {
  EXPECT_EQ(LoadParametricText(
                WithoutSource(TrackedWheelYaml(), "maximum_curvature_per_m"))
                .error->reason_code,
            "CAPABILITY_SCHEMA_INVALID");
  std::string unsupported = TrackedWheelYaml();
  const std::string approved =
      "maximum_curvature_per_m: user_confirmed_capability";
  const auto source = unsupported.find(
      approved,
      unsupported.find("sources:\n"));
  unsupported.replace(source, approved.size(), "maximum_curvature_per_m: guess");
  EXPECT_EQ(LoadParametricText(unsupported).error->reason_code,
            "CAPABILITY_VALUE_INVALID");
}
```

The test reads the tracked Task 1 document, so the C++ parser and deployment artifact cannot drift into separate fixtures.

- [ ] **Step 2: Build the target and verify the new tests fail to compile**

Run inside the container:

```bash
docker exec ros2-humble bash -lc 'cd /root/lunar-runtime && source /opt/ros/humble/setup.bash && colcon build --packages-select lunar_planner_core lunar_planner_ros --cmake-args -DBUILD_TESTING=ON'
```

Expected: FAIL because `GeometrySourceKind` and `parametric_wheeled_geometry` do not exist.

- [ ] **Step 3: Add the exact audit types**

Add to `capability_loader.hpp`:

```cpp
enum class GeometrySourceKind : std::uint8_t { kUrdfMesh, kParametricEnvelope };

struct ParametricWheeledGeometry final {
  std::size_t wheel_count{};
  std::vector<lunar::planning::Vec2> wheel_center_xy_m;
};

// Add these members to LoadedCapabilities before urdf_path.
GeometrySourceKind geometry_source_kind{GeometrySourceKind::kUrdfMesh};
std::optional<ParametricWheeledGeometry> parametric_wheeled_geometry;
```

- [ ] **Step 4: Implement the discriminated geometry parser and validation**

In `capability_loader.cpp`, parse geometry before calling `ParseWheeled`:

```cpp
const YAML::Node geometry = RequireMap(platform_document, "geometry_source");
const bool parametric = geometry["type"] &&
    RequireString(geometry, "type") == "parametric_envelope";
if (parametric) {
  RejectUnexpectedKeys(geometry, {"type"}, "geometry_source");
  if (platform_type != "WHEELED") {
    ValueFailure("parametric_envelope is only approved for WHEELED");
  }
  loaded.geometry_source_kind = GeometrySourceKind::kParametricEnvelope;
} else {
  RejectUnexpectedKeys(geometry, {"urdf_file"}, "geometry_source");
  loaded.geometry_source_kind = GeometrySourceKind::kUrdfMesh;
  // Preserve the current ResolveFile + ValidateGeometry path exactly.
}
```

Extend the WHEELED allowed keys with `wheel_count` and `wheel_center_xy_m`. Add focused helpers with these exact contracts:

```cpp
std::size_t RequirePositiveSize(const YAML::Node&, const std::string& key);
bool IsStrictlyConvex(const std::vector<Vec2>& polygon) noexcept;
ParametricWheeledGeometry ParseParametricWheelGeometry(
    const YAML::Node& wheeled, const Vec3& body_extent,
    double wheel_diameter, double wheel_width);
void ValidateRequiredSources(
    const LoadedCapabilities&, const std::set<std::string, std::less<>>& required);
```

`ParseParametricWheelGeometry` must require four unique wheel centers for `wheel_count: 4`, require a strictly convex footprint, and enforce for each center:

```cpp
std::abs(center.x) + wheel_diameter / 2.0 <= body_extent.x / 2.0 + 1.0e-9;
std::abs(center.y) + wheel_width / 2.0 <= body_extent.y / 2.0 + 1.0e-9;
```

Remove the legacy equality assumption `track_width == body_extent.y - wheel_width` only for parametric geometry; retain the legacy behavior for URDF mode. Validate source values against the existing schema allowlist and require the exact WHEELED set from Task 1.

- [ ] **Step 5: Rebuild and run the focused C++ tests**

Run:

```bash
docker exec ros2-humble bash -lc 'cd /root/lunar-runtime && source /opt/ros/humble/setup.bash && colcon build --packages-select lunar_planner_core lunar_planner_ros --cmake-args -DBUILD_TESTING=ON && colcon test --packages-select lunar_planner_ros --ctest-args -R lunar_planner_ros_capability_loader_test --output-on-failure && colcon test-result --verbose'
```

Expected: capability loader target passes, including legacy URDF+mesh tests.

- [ ] **Step 6: Commit the parameter loader**

```bash
git add ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/capability_loader.hpp ros2_ws/src/lunar_planner_ros/src/capability_loader.cpp ros2_ws/src/lunar_planner_ros/test/capability_loader_test.cpp
git commit -m "feat(ros): load parametric wheel geometry"
```

---

### Task 3: Reconcile absolute deployment files with package-relative capability files

**Files:**
- Modify: `ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/capability_loader.hpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/capability_loader.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/capability_loader_test.cpp`
- Modify: `ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp`
- Modify: `tests/deployment/test_config.py`

**Interfaces:**
- Consumes: Optional `capability_package`, absolute external paths, or safe package-relative paths.
- Produces: `CapabilityLoader::LoadConfigured(package, platform, observation)` and stable `CAPABILITY_PATH_MODE_INVALID` failures.

- [ ] **Step 1: Write failing path-mode and tracked-file tests**

Add loader tests for all three branches:

```cpp
TEST(CapabilityLoader, SelectsExactlyOneConfiguredPathMode) {
  const auto absolute = UniqueShare("absolute-mode");
  WriteObservation(absolute);
  Write(absolute / "config" / "platform.yaml", TrackedWheelYaml());
  EXPECT_TRUE(CapabilityLoader{}.LoadConfigured(
      "", absolute / "config/platform.yaml",
      absolute / "config/observation.json").ok());
  EXPECT_EQ(CapabilityLoader{}.LoadConfigured(
                "package_that_does_not_exist", "config/platform.yaml",
                "config/observation.json")
                .error->reason_code,
            "CAPABILITY_PACKAGE_NOT_FOUND");
  EXPECT_EQ(CapabilityLoader{}.LoadConfigured(
                "", "config/platform.yaml", "config/observation.json")
                .error->reason_code,
            "CAPABILITY_PATH_MODE_INVALID");
  EXPECT_EQ(CapabilityLoader{}.LoadConfigured(
                "robot_pkg", "/tmp/platform.yaml", "/tmp/observation.json")
                .error->reason_code,
            "CAPABILITY_PATH_MODE_INVALID");
}

TEST(CapabilityLoader, LoadsTrackedWheelV1FromAbsolutePath) {
  const auto observation = UniqueShare("tracked-wheel") / "observation.yaml";
  Write(observation, "sensor_range_m: 25.0\nsensor_fov_deg: 90.0\n");
  const auto result = CapabilityLoader{}.LoadConfigured(
      "", RepositoryRoot() / "deployment/config/wheel.yaml", observation);
  ASSERT_TRUE(result.ok())
      << (result.error ? result.error->detail : std::string{});
  EXPECT_EQ(result.capabilities->capability_version, "wheel-v1");
}
```

Add a server test that supplies absolute temporary capability files with `preloaded_capabilities = std::nullopt` and expects lifecycle configuration to reach `INACTIVE`. Add this assertion to `test_valid_fallback_config_renders_planner_ros_parameters`:

```python
assert "platform_capability_file: /tmp/platform.yaml" in text
assert "observation_capability_file: /tmp/observation.yaml" in text
assert "capability_package" not in text
```

- [ ] **Step 2: Verify tests fail for the missing selector and empty package declaration**

Run the Task 2 build/test command plus:

```bash
python3 -m pytest -q tests/deployment/test_config.py
```

Expected: C++ compile/test failure because `LoadConfigured` is absent and lifecycle configuration cannot read absolute files.

- [ ] **Step 3: Implement the path selector**

Add this public method:

```cpp
[[nodiscard]] CapabilityLoadResult LoadConfigured(
    const std::string& package_name,
    const std::filesystem::path& platform_capability_file,
    const std::filesystem::path& observation_capability_file) const;

[[nodiscard]] CapabilityLoadResult LoadFromFiles(
    const std::filesystem::path& platform_capability_file,
    const std::filesystem::path& observation_capability_file) const;
```

Implement exact mode selection:

```cpp
if (package_name.empty() && platform.is_absolute() && observation.is_absolute()) {
  return LoadFromFiles(platform, observation);
}
if (!package_name.empty() && !platform.is_absolute() && !observation.is_absolute()) {
  return LoadFromPackageShare(package_name, platform, observation);
}
return Failure(LoadFailure{
    CapabilityLoadErrorCode::kPathModeInvalid,
    "CAPABILITY_PATH_MODE_INVALID",
    "capability package and file path modes are inconsistent"});
```

Add `kPathModeInvalid` to `CapabilityLoadErrorCode`. `LoadFromFiles` is public for focused loader verification; it must require two regular absolute files and pass no package share to document parsing. In absolute mode, reject `urdf_file` geometry with `CAPABILITY_PATH_MODE_INVALID`; only `parametric_envelope` is approved outside a package share.

- [ ] **Step 4: Wire the lifecycle node to the selector**

Change declaration and load call in `plan_motion_server.cpp`:

```cpp
node.declare_parameter<std::string>("capability_package", "");

const CapabilityLoadResult loaded = CapabilityLoader{}.LoadConfigured(
    package, platform_file, observation_file);
```

Do not add `capability_package` to deployment runtime YAML: the empty default selects its already documented absolute paths.

- [ ] **Step 5: Run deployment and ROS tests**

Run:

```bash
python3 -m pytest -q tests/deployment/test_config.py tests/deployment/test_wheel_capability.py
docker exec ros2-humble bash -lc 'cd /root/lunar-runtime && source /opt/ros/humble/setup.bash && colcon build --packages-select lunar_planner_core lunar_planner_ros --cmake-args -DBUILD_TESTING=ON && colcon test --packages-select lunar_planner_ros --ctest-args -R "lunar_planner_ros_(capability_loader|plan_motion_server)_test" --output-on-failure && colcon test-result --verbose'
```

Expected: all selected Python and C++ tests pass.

- [ ] **Step 6: Commit the deployment/ROS path reconciliation**

```bash
git add ros2_ws/src/lunar_planner_ros/include/lunar_planner_ros/capability_loader.hpp ros2_ws/src/lunar_planner_ros/src/capability_loader.cpp ros2_ws/src/lunar_planner_ros/src/plan_motion_server.cpp ros2_ws/src/lunar_planner_ros/test/capability_loader_test.cpp ros2_ws/src/lunar_planner_ros/test/plan_motion_server_test.cpp tests/deployment/test_config.py
git commit -m "fix(deploy): load absolute capability files"
```

---

### Task 4: Publish the parameter-geometry contract in schema and interface docs

**Files:**
- Modify: `ros2_ws/src/lunar_navigation_config/config/platform_capability_schema_v2.yaml`
- Modify: `ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml`
- Modify: `tools/check_external_interfaces.py`
- Modify: `tests/foundation/test_external_interface_config.py`
- Modify: `docs/interfaces/external-input-baseline.md`

**Interfaces:**
- Consumes: Runtime behavior completed in Tasks 2-3.
- Produces: Machine-checked declaration that v2 accepts `parametric_envelope` or legacy `urdf_mesh`, with no change to frozen platform values/digest.

- [ ] **Step 1: Make the interface fixture expect the geometry-source union**

Extend both the test fixture and checker constant under `static_inputs.platform_capability`:

```python
"geometry_sources": ["parametric_envelope", "urdf_mesh"],
```

Add a mutation case that deletes `geometry_sources`, runs the checker, asserts a nonzero exit code, and asserts `static_inputs` appears in stderr/stdout.

- [ ] **Step 2: Run the focused foundation test and verify failure**

Run:

```bash
python3 -m pytest -q tests/foundation/test_external_interface_config.py
```

Expected: FAIL because `external_interfaces.yaml` lacks `geometry_sources`.

- [ ] **Step 3: Extend the schema without changing the freeze payload**

Add at the root of `platform_capability_schema_v2.yaml`:

```yaml
geometry_source_variants:
  parametric_envelope:
    required_fields: [type]
    forbidden_fields: [urdf_file]
  urdf_mesh:
    required_fields: [urdf_file]
    forbidden_fields: [type]
```

Add to `external_interfaces.yaml`:

```yaml
geometry_sources: [parametric_envelope, urdf_mesh]
```

Update the static input paragraph/table in `external-input-baseline.md` to state that `parametric_envelope` uses the formal numeric footprint/body/wheel fields and provides no component geometry or visualization; URDF mode continues to require referenced meshes.

- [ ] **Step 4: Run schema, interface, and boundary verification**

Run:

```bash
python3 tools/check_platform_capability_freeze.py --schema ros2_ws/src/lunar_navigation_config/config/platform_capability_schema_v2.yaml --freeze ros2_ws/src/lunar_navigation_config/config/three_platform_capability_freeze_v1.yaml
python3 -m pytest -q tests/foundation/test_external_interface_config.py tests/foundation/test_platform_capability_freeze.py tests/foundation/test_repository_boundaries.py
python3 tools/check_repository_boundaries.py .
```

Expected: all commands succeed; the reported capability freeze SHA-256 remains `60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95`.

- [ ] **Step 5: Commit the contract and documentation**

```bash
git add ros2_ws/src/lunar_navigation_config/config/platform_capability_schema_v2.yaml ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml tools/check_external_interfaces.py tests/foundation/test_external_interface_config.py docs/interfaces/external-input-baseline.md
git commit -m "docs: define parametric capability geometry"
```

---

### Task 5: Install the reviewed file in the container and run final verification

**Files:**
- No additional repository files unless a verification failure requires a scoped fix.
- Create outside the repository: `/opt/luna/capabilities/platform.yaml` inside `ros2-humble`.

**Interfaces:**
- Consumes: The tracked `deployment/config/wheel.yaml` and completed loader/build.
- Produces: Container-local formal platform capability at the runtime-configured path.

- [ ] **Step 1: Verify the target does not overwrite an unexpected existing file**

Run:

```bash
docker exec ros2-humble bash -lc 'if [ -e /opt/luna/capabilities/platform.yaml ]; then sha256sum /opt/luna/capabilities/platform.yaml; else echo PLATFORM_CAPABILITY_NOT_PRESENT; fi'
```

Expected for the inspected current container: `PLATFORM_CAPABILITY_NOT_PRESENT`. If a file appears, stop and compare it before replacement.

- [ ] **Step 2: Install the reviewed source file with explicit permissions**

Run:

```bash
docker exec ros2-humble bash -lc 'install -d -m 0755 /opt/luna/capabilities && install -m 0644 /root/lunar-runtime/deployment/config/wheel.yaml /opt/luna/capabilities/platform.yaml'
```

- [ ] **Step 3: Verify byte identity and the absence of fabricated assets**

Run:

```bash
docker exec ros2-humble bash -lc 'sha256sum /root/lunar-runtime/deployment/config/wheel.yaml /opt/luna/capabilities/platform.yaml && find /opt/luna/capabilities -maxdepth 1 -type f -printf "%f\n" | sort'
```

Expected: identical hashes; only `platform.yaml` is present unless the user has separately supplied a real `observation.yaml`; no URDF or mesh files exist.

- [ ] **Step 4: Run final repository and ROS verification**

Run:

```bash
python3 -m pytest -q tests/deployment/test_config.py tests/deployment/test_wheel_capability.py tests/foundation/test_external_interface_config.py tests/foundation/test_platform_capability_freeze.py tests/foundation/test_repository_boundaries.py
python3 tools/check_repository_boundaries.py .
docker exec ros2-humble bash -lc 'cd /root/lunar-runtime && source /opt/ros/humble/setup.bash && colcon build --packages-select lunar_planner_core lunar_planner_ros --cmake-args -DBUILD_TESTING=ON && colcon test --packages-select lunar_planner_core lunar_planner_ros --event-handlers console_direct+ && colcon test-result --verbose'
```

Expected: all selected tests pass with zero failed tests.

- [ ] **Step 5: Record the honest runtime boundary**

Run:

```bash
docker exec ros2-humble bash -lc 'test -f /opt/luna/capabilities/observation.yaml && echo FULL_CAPABILITY_PAIR_PRESENT || echo OBSERVATION_CAPABILITY_MISSING'
```

Expected with current inputs: `OBSERVATION_CAPABILITY_MISSING`. Report platform capability installation as complete, but do not claim `./luna start` is ready until the user provides sensor range and field of view.

- [ ] **Step 6: Inspect final status without committing unrelated changes**

```bash
git status --short
git log -5 --oneline --decorate
```

Expected: the task commits are present; the pre-existing unrelated deployment doc/CLI changes remain preserved unless separately committed by their owner.
