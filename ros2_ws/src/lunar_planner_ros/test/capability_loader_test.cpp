#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numbers>
#include <string>
#include <variant>

#include <gtest/gtest.h>

#include "lunar_planner_ros/capability_loader.hpp"

namespace lunar::planning::ros {
namespace {

std::filesystem::path UniqueShare(const std::string& suffix) {
  const auto nonce = std::chrono::steady_clock::now()
      .time_since_epoch()
      .count();
  const auto path = std::filesystem::temp_directory_path() /
      ("lunar-capability-" + suffix + "-" + std::to_string(nonce));
  std::filesystem::create_directories(path / "config");
  std::filesystem::create_directories(path / "urdf" / "meshes");
  return path;
}

void Write(const std::filesystem::path& path, const std::string& content) {
  std::ofstream stream{path};
  ASSERT_TRUE(stream.good());
  stream << content;
  ASSERT_TRUE(stream.good());
}

void WriteGeometry(const std::filesystem::path& share, const bool mesh_exists) {
  Write(
      share / "urdf" / "rover.urdf",
      R"(<robot name="rover">
  <link name="base_link">
    <visual><geometry><mesh filename="meshes/body.stl"/></geometry></visual>
  </link>
  <link name="base_footprint"/>
  <joint name="base_footprint_to_base_link" type="fixed">
    <parent link="base_footprint"/>
    <child link="base_link"/>
  </joint>
</robot>)");
  if (mesh_exists) {
    Write(share / "urdf" / "meshes" / "body.stl", "solid body\nendsolid body\n");
  }
}

void WriteObservation(const std::filesystem::path& share) {
  Write(
      share / "config" / "observation.json",
      R"({"sensor_range_m": 25.0, "sensor_fov_deg": 90.0})");
}

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

std::string CommonHeader(const std::string& type) {
  const std::string base_frame = type == "WHEELED" ? "base_footprint" : "base_link";
  return "schema_version: platform-control-capability-source/v2\n"
      "platform:\n"
      "  platform_id: rover-1\n"
      "  platform_type: " + type + "\n"
      "  capability_version: capability-v1\n"
      "  base_frame_id: " + base_frame + "\n"
      "geometry_source:\n"
      "  urdf_file: urdf/rover.urdf\n"
      "sources:\n"
      "  baseline: project_engineering_baseline\n";
}

std::string WheeledYaml(const std::string& forward_speed = "1.0") {
  return CommonHeader("WHEELED") + R"(wheeled:
  footprint_xy_m: [[-0.591, -0.409], [0.591, -0.409], [0.591, 0.409], [-0.591, 0.409]]
  body_extent_m: [1.182, 0.818, 1.29996]
  reference_point: base_footprint
  wheel_diameter_m: 0.319
  wheel_width_m: 0.148
  wheelbase_m: 0.8175
  track_width_m: 0.67
  minimum_underbody_clearance_m: 0.21
  maximum_local_obstacle_relief_m: 0.2
  allow_unsupported_gap: false
  maximum_forward_speed_mps: )" + forward_speed + R"(
  maximum_reverse_speed_mps: 1.0
  maximum_spin_rate_radps: 1.0
  maximum_acceleration_mps2: 0.5
  maximum_braking_deceleration_mps2: 0.5
  maximum_yaw_acceleration_radps2: 0.5
  maximum_lateral_acceleration_mps2: 0.5
  maximum_curvature_per_m: 1.0
  maximum_slope_rad: 0.3490658503988659
  minimum_clearance_m: 0.2
  roughness_handling: COST_SPEED_AND_LOCAL_RECHECK
  motion_primitives:
    - primitive_id: forward
      kind: FORWARD
      relative_end_pose:
        position_m: [0.2, 0.0, 0.0]
        orientation_wxyz: [1.0, 0.0, 0.0, 0.0]
    - primitive_id: reverse
      kind: REVERSE
      relative_end_pose:
        position_m: [-0.2, 0.0, 0.0]
        orientation_wxyz: [1.0, 0.0, 0.0, 0.0]
    - primitive_id: forward-arc-left
      kind: FORWARD_ARC
      relative_end_pose:
        position_m: [0.19509032201612825, 0.01921471959676957, 0.0]
        orientation_wxyz: [0.9951847266721969, 0.0, 0.0, 0.0980171403295606]
    - primitive_id: reverse-arc-right
      kind: REVERSE_ARC
      relative_end_pose:
        position_m: [-0.19509032201612825, 0.01921471959676957, 0.0]
        orientation_wxyz: [0.9951847266721969, 0.0, 0.0, -0.0980171403295606]
    - primitive_id: spin-left
      kind: SPIN_COUNTERCLOCKWISE
      relative_end_pose:
        position_m: [0.0, 0.0, 0.0]
        orientation_wxyz: [0.9951847266721969, 0.0, 0.0, 0.0980171403295606]
    - primitive_id: spin-right
      kind: SPIN_CLOCKWISE
      relative_end_pose:
        position_m: [0.0, 0.0, 0.0]
        orientation_wxyz: [0.9951847266721969, 0.0, 0.0, -0.0980171403295606]
    - primitive_id: stop-switch
      kind: STOP_AND_SWITCH
      relative_end_pose:
        position_m: [0.0, 0.0, 0.0]
        orientation_wxyz: [1.0, 0.0, 0.0, 0.0]
)";
}

std::string LeggedYaml() {
  return CommonHeader("LEGGED") + R"(legged:
  reference_point: base_link
  body_extent_m: [0.68, 0.33, 0.35]
  platform_mass_kg: 15.89
  maximum_payload_kg: 10.0
  maximum_slope_rad: 0.5235987755982988
  maximum_step_height_m: 0.5
  maximum_gap_width_m: 0.3
  minimum_body_clearance_m: 0.3
  step_vertical_rate_mps: 0.1
  body_height_m: [0.28, 0.38]
  forward_speed_mps: [-1.5, 1.5]
  lateral_speed_mps: [-0.8, 0.8]
  yaw_rate_radps: [-1.0, 1.0]
  maximum_linear_acceleration_mps2: 1.0
  maximum_yaw_acceleration_radps2: 1.0
  roughness_handling: DIAGNOSTIC_ONLY
  motion_primitives:
    - primitive_id: forward
      kind: FORWARD
      body_frame_displacement_m: [0.2, 0.0, 0.0]
      yaw_change_rad: 0.0
)";
}

std::string HopperYaml() {
  return CommonHeader("HOPPER") + R"(hopper:
  specific_impulse_s: 301.0
  reference_total_mass_kg: 20.0
  reference_propellant_mass_kg: 0.2
  landing_support_radius_m: 0.45
  flight_collision_radius_m: 0.55
  maximum_landing_slope_rad: 0.17453292519943295
  maximum_landing_plane_residual_m: 0.05
  landing_lateral_margin_m: 0.2
  flight_map_margin_m: 0.2
  reachability_delta_v_margin_ratio: 0.1
  standard_gravity_mps2: 9.80665
)";
}

CapabilityLoadResult Load(
    const std::filesystem::path& share,
    const std::string& platform_document) {
  WriteGeometry(share, true);
  WriteObservation(share);
  Write(share / "config" / "platform.yaml", platform_document);
  return CapabilityLoader{}.LoadFromShareDirectory(
      share, "config/platform.yaml", "config/observation.json");
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
  EXPECT_EQ(
      result.capabilities->parametric_wheeled_geometry->wheel_center_xy_m.size(),
      4U);
  const auto& wheel = std::get<lunar::planning::WheeledCapability>(
      result.capabilities->platform);
  EXPECT_DOUBLE_EQ(wheel.maximum_curvature_per_m, 5.0);
}

TEST(CapabilityLoader, RejectsMixedOrInvalidParametricGeometry) {
  std::string mixed = TrackedWheelYaml();
  const std::string parametric_geometry =
      "geometry_source: {type: parametric_envelope}";
  mixed.replace(
      mixed.find(parametric_geometry), parametric_geometry.size(),
      "geometry_source:\n"
      "  type: parametric_envelope\n"
      "  urdf_file: urdf/rover.urdf");
  EXPECT_EQ(LoadParametricText(mixed).error->reason_code,
            "CAPABILITY_SCHEMA_INVALID");
  std::string wrong_count = TrackedWheelYaml();
  wrong_count.replace(
      wrong_count.find("wheel_count: 4"), 14U, "wheel_count: 3");
  EXPECT_EQ(LoadParametricText(wrong_count).error->reason_code,
            "CAPABILITY_VALUE_INVALID");
  std::string outside = TrackedWheelYaml();
  const auto centers = outside.find("wheel_center_xy_m:");
  outside.replace(
      centers, outside.find('\n', centers) - centers,
      "wheel_center_xy_m: [[2.0, 0.0], [0.4075, -0.3115], "
      "[-0.4075, -0.3115], [-0.4075, 0.3115]]");
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

TEST(CapabilityLoader, LoadsV2WheelGeometryAndSourcesWithoutProxyValues) {
  const CapabilityLoadResult result =
      Load(UniqueShare("wheel"), WheeledYaml());

  ASSERT_TRUE(result.ok())
      << (result.error.has_value() ? result.error->detail : std::string{});
  ASSERT_TRUE(result.capabilities.has_value());
  EXPECT_EQ(result.capabilities->platform_id, "rover-1");
  EXPECT_EQ(result.capabilities->base_frame_id, "base_footprint");
  EXPECT_NEAR(
      result.capabilities->observation.sensor_fov_rad,
      std::numbers::pi / 2.0,
      1.0e-12);
  ASSERT_EQ(result.capabilities->mesh_paths.size(), 1U);
  EXPECT_TRUE(std::filesystem::is_regular_file(
      result.capabilities->mesh_paths.front()));
  const auto& wheel = std::get<lunar::planning::WheeledCapability>(
      result.capabilities->platform);
  EXPECT_NEAR(wheel.maximum_slope_rad, 0.3490658503988659, 1.0e-12);
  EXPECT_EQ(wheel.body_extent_m, (lunar::planning::Vec3{1.182, 0.818, 1.29996}));
  EXPECT_DOUBLE_EQ(wheel.wheel_diameter_m, 0.319);
  EXPECT_DOUBLE_EQ(wheel.wheel_width_m, 0.148);
  EXPECT_DOUBLE_EQ(wheel.wheelbase_m, 0.8175);
  EXPECT_DOUBLE_EQ(wheel.track_width_m, 0.67);
  EXPECT_DOUBLE_EQ(wheel.minimum_underbody_clearance_m, 0.21);
  EXPECT_DOUBLE_EQ(wheel.maximum_local_obstacle_relief_m, 0.2);
  EXPECT_FALSE(wheel.allow_unsupported_gap);
  ASSERT_EQ(wheel.motion_primitives.size(), 7U);
  EXPECT_EQ(wheel.motion_primitives.front().primitive_id, "forward");
  EXPECT_EQ(wheel.motion_primitives.back().kind,
            lunar::planning::WheelPrimitiveKind::kStopAndSwitch);
  EXPECT_EQ(result.capabilities->field_source_types.at("baseline"),
            "project_engineering_baseline");
}

TEST(CapabilityLoader, AdaptsLeggedAndHopperSourcesToTypedCapabilities) {
  const CapabilityLoadResult legged =
      Load(UniqueShare("legged"), LeggedYaml());
  ASSERT_TRUE(legged.ok())
      << (legged.error.has_value() ? legged.error->detail : std::string{});
  EXPECT_TRUE(std::holds_alternative<lunar::planning::LeggedCapability>(
      legged.capabilities->platform));
  const auto& legged_typed = std::get<lunar::planning::LeggedCapability>(
      legged.capabilities->platform);
  EXPECT_EQ(legged.capabilities->reference_point, "base_link");
  EXPECT_EQ(legged_typed.body_extent_m,
            (lunar::planning::Vec3{0.68, 0.33, 0.35}));
  EXPECT_DOUBLE_EQ(legged_typed.platform_mass_kg, 15.89);
  EXPECT_DOUBLE_EQ(legged_typed.maximum_payload_kg, 10.0);
  EXPECT_DOUBLE_EQ(legged_typed.step_vertical_rate_mps, 0.1);

  const CapabilityLoadResult hopper =
      Load(UniqueShare("hopper"), HopperYaml());
  ASSERT_TRUE(hopper.ok())
      << (hopper.error.has_value() ? hopper.error->detail : std::string{});
  const auto& typed = std::get<lunar::planning::HopperCapability>(
      hopper.capabilities->platform);
  EXPECT_NEAR(
      typed.maximum_landing_slope_rad,
      0.17453292519943295,
      1.0e-12);
  EXPECT_DOUBLE_EQ(typed.specific_impulse_s, 301.0);
  EXPECT_DOUBLE_EQ(typed.reference_total_mass_kg, 20.0);
  EXPECT_DOUBLE_EQ(typed.reference_propellant_mass_kg, 0.2);
  EXPECT_DOUBLE_EQ(typed.landing_support_radius_m, 0.45);
  EXPECT_DOUBLE_EQ(typed.flight_collision_radius_m, 0.55);
  EXPECT_DOUBLE_EQ(typed.maximum_landing_plane_residual_m, 0.05);
  EXPECT_DOUBLE_EQ(typed.reachability_delta_v_margin_ratio, 0.1);
}

TEST(CapabilityLoader, RejectsV1HopperFieldsAsVersionIncompatible) {
  std::string retired = HopperYaml();
  retired.replace(
      retired.find("platform-control-capability-source/v2"),
      std::string{"platform-control-capability-source/v2"}.size(),
      "platform-control-capability-source/v1");

  const CapabilityLoadResult result = Load(UniqueShare("retired-v1"), retired);

  ASSERT_TRUE(result.error.has_value());
  EXPECT_EQ(result.error->code, CapabilityLoadErrorCode::kSchemaInvalid);
  EXPECT_EQ(
      result.error->reason_code,
      "CAPABILITY_SCHEMA_VERSION_INCOMPATIBLE");
}

TEST(CapabilityLoader, RejectsRetiredHopperFieldInsideV2Document) {
  std::string retired = HopperYaml();
  const std::string marker = "hopper:\n";
  retired.insert(
      retired.find(marker) + marker.size(),
      "  maximum_launch_speed_mps: 8.0\n");

  const CapabilityLoadResult result = Load(UniqueShare("retired-field"), retired);

  ASSERT_TRUE(result.error.has_value());
  EXPECT_EQ(result.error->code, CapabilityLoadErrorCode::kSchemaInvalid);
  EXPECT_EQ(
      result.error->reason_code,
      "CAPABILITY_SCHEMA_VERSION_INCOMPATIBLE");
}

TEST(CapabilityLoader, RejectsMissingMeshUnsafePathAndInvalidValues) {
  const auto missing_mesh = UniqueShare("missing-mesh");
  WriteGeometry(missing_mesh, false);
  WriteObservation(missing_mesh);
  Write(missing_mesh / "config" / "platform.yaml", WheeledYaml());
  const CapabilityLoadResult mesh_result =
      CapabilityLoader{}.LoadFromShareDirectory(
          missing_mesh,
          "config/platform.yaml",
          "config/observation.json");
  ASSERT_TRUE(mesh_result.error.has_value());
  EXPECT_EQ(mesh_result.error->code, CapabilityLoadErrorCode::kMeshMissing);

  const CapabilityLoadResult unsafe =
      CapabilityLoader{}.LoadFromShareDirectory(
          missing_mesh,
          "../platform.yaml",
          "config/observation.json");
  ASSERT_TRUE(unsafe.error.has_value());
  EXPECT_EQ(unsafe.error->code, CapabilityLoadErrorCode::kUnsafePath);

  const CapabilityLoadResult nonfinite =
      Load(UniqueShare("nonfinite"), WheeledYaml(".nan"));
  ASSERT_TRUE(nonfinite.error.has_value());
  EXPECT_EQ(nonfinite.error->code, CapabilityLoadErrorCode::kValueInvalid);
}

TEST(CapabilityLoader, RejectsMissingObservationFieldAndUnknownPackage) {
  const auto missing_field = UniqueShare("missing-field");
  WriteGeometry(missing_field, true);
  Write(
      missing_field / "config" / "observation.json",
      R"({"sensor_range_m": 25.0})");
  Write(missing_field / "config" / "platform.yaml", WheeledYaml());
  const CapabilityLoadResult field_result =
      CapabilityLoader{}.LoadFromShareDirectory(
          missing_field,
          "config/platform.yaml",
          "config/observation.json");
  ASSERT_TRUE(field_result.error.has_value());
  EXPECT_EQ(field_result.error->code, CapabilityLoadErrorCode::kSchemaInvalid);

  const CapabilityLoadResult package_result =
      CapabilityLoader{}.LoadFromPackageShare(
          "lunar_package_that_does_not_exist",
          "config/platform.yaml",
          "config/observation.yaml");
  ASSERT_TRUE(package_result.error.has_value());
  EXPECT_EQ(
      package_result.error->code,
      CapabilityLoadErrorCode::kPackageNotFound);
}

}  // namespace
}  // namespace lunar::planning::ros
