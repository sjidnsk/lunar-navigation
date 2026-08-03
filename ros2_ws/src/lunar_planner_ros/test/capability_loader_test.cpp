#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
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

std::string CommonHeader(const std::string& type) {
  return "schema_version: platform-control-capability-source/v1\n"
      "platform:\n"
      "  platform_id: rover-1\n"
      "  platform_type: " + type + "\n"
      "  capability_version: capability-v1\n"
      "  base_frame_id: base_link\n"
      "geometry_source:\n"
      "  urdf_file: urdf/rover.urdf\n";
}

std::string WheeledYaml(const std::string& forward_speed = "1.0") {
  return CommonHeader("WHEELED") + R"(wheeled:
  footprint_xy_m: [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]]
  minimum_body_z_m: -0.1
  maximum_body_z_m: 0.5
  maximum_slope_rad: 0.8
  maximum_obstacle_height_m: 0.2
  maximum_forward_speed_mps: )" + forward_speed + R"(
  maximum_reverse_speed_mps: 0.5
  maximum_spin_rate_radps: 1.0
  maximum_acceleration_mps2: 1.0
  maximum_braking_deceleration_mps2: 1.0
  maximum_yaw_acceleration_radps2: 1.0
  maximum_lateral_acceleration_mps2: 1.0
  maximum_curvature_per_m: 1.0
  minimum_clearance_m: 0.05
  motion_primitives:
    - primitive_id: forward
      kind: FORWARD
      relative_end_pose:
        position_m: [1.0, 0.0, 0.0]
        orientation_wxyz: [1.0, 0.0, 0.0, 0.0]
      nominal_duration_s: 1.0
)";
}

std::string LeggedYaml() {
  return CommonHeader("LEGGED") + R"(legged:
  reference_point: body
  body_half_extent_m: [0.2, 0.2, 0.3]
  maximum_slope_rad: 0.4
  maximum_roughness_m: 0.2
  maximum_step_height_m: 0.3
  maximum_gap_width_m: 0.4
  minimum_confidence: 0.8
  minimum_body_clearance_m: 0.1
  body_height_m: [0.4, 0.6]
  forward_speed_mps: [-0.5, 0.5]
  lateral_speed_mps: [-0.5, 0.5]
  vertical_speed_mps: [-0.1, 0.1]
  yaw_rate_radps: [-1.0, 1.0]
  maximum_linear_acceleration_mps2: 0.5
  maximum_yaw_acceleration_radps2: 1.0
  motion_primitives:
    - primitive_id: forward
      kind: FORWARD
      body_frame_displacement_m: [1.0, 0.0, 0.0]
      yaw_change_rad: 0.0
      nominal_duration_s: 2.0
)";
}

std::string HopperYaml() {
  return CommonHeader("HOPPER") + R"(hopper:
  body_half_extent_m: [0.35, 0.25, 0.5]
  platform_mass_kg: 10.0
  gravity_mps2: [0.0, 0.0, -1.62]
  maximum_landing_slope_rad: 0.7
  maximum_landing_roughness_m: 0.1
  maximum_plane_residual_m: 0.05
  minimum_overhead_clearance_m: 0.0
  minimum_lateral_clearance_m: 0.0
  minimum_landing_region_area_m2: 0.2
  maximum_launch_speed_mps: 8.0
  maximum_launch_impulse_newton_seconds: 100.0
  minimum_flight_time_s: 0.5
  maximum_flight_time_s: 10.0
  maximum_landing_speed_mps: 8.0
  minimum_downward_impact_speed_mps: 0.1
  minimum_landing_clearance_m: 0.0
  maximum_angular_speed_radps: 2.0
  maximum_angular_acceleration_radps2: 4.0
  maximum_initial_angular_speed_radps: 0.2
  minimum_settle_guard_s: 0.1
  actuator_or_impulse_profile:
    profile_id: nominal-impulse
  motion_primitives:
    - primitive_id: nominal-hop
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

TEST(CapabilityLoader, LoadsYamlJsonUrdfMeshAndClampsProjectSlope) {
  const CapabilityLoadResult result =
      Load(UniqueShare("wheel"), WheeledYaml());

  ASSERT_TRUE(result.ok())
      << (result.error.has_value() ? result.error->detail : std::string{});
  ASSERT_TRUE(result.capabilities.has_value());
  EXPECT_EQ(result.capabilities->platform_id, "rover-1");
  EXPECT_EQ(result.capabilities->base_frame_id, "base_link");
  EXPECT_NEAR(
      result.capabilities->observation.sensor_fov_rad,
      std::numbers::pi / 2.0,
      1.0e-12);
  ASSERT_EQ(result.capabilities->mesh_paths.size(), 1U);
  EXPECT_TRUE(std::filesystem::is_regular_file(
      result.capabilities->mesh_paths.front()));
  const auto& wheel = std::get<lunar::planning::WheeledCapability>(
      result.capabilities->platform);
  EXPECT_NEAR(wheel.maximum_slope_rad, std::numbers::pi / 6.0, 1.0e-12);
  ASSERT_EQ(wheel.motion_primitives.size(), 1U);
  EXPECT_EQ(wheel.motion_primitives.front().primitive_id, "forward");
  ASSERT_TRUE(result.capabilities->maximum_obstacle_height_m.has_value());
  EXPECT_DOUBLE_EQ(*result.capabilities->maximum_obstacle_height_m, 0.2);
}

TEST(CapabilityLoader, AdaptsLeggedAndHopperSourcesToTypedCapabilities) {
  const CapabilityLoadResult legged =
      Load(UniqueShare("legged"), LeggedYaml());
  ASSERT_TRUE(legged.ok())
      << (legged.error.has_value() ? legged.error->detail : std::string{});
  EXPECT_TRUE(std::holds_alternative<lunar::planning::LeggedCapability>(
      legged.capabilities->platform));
  EXPECT_EQ(legged.capabilities->reference_point, "body");

  const CapabilityLoadResult hopper =
      Load(UniqueShare("hopper"), HopperYaml());
  ASSERT_TRUE(hopper.ok())
      << (hopper.error.has_value() ? hopper.error->detail : std::string{});
  const auto& typed = std::get<lunar::planning::HopperCapability>(
      hopper.capabilities->platform);
  EXPECT_NEAR(
      typed.maximum_landing_slope_rad,
      std::numbers::pi / 6.0,
      1.0e-12);
  EXPECT_EQ(hopper.capabilities->actuator_profile_id, "nominal-impulse");
  EXPECT_EQ(
      hopper.capabilities->source_motion_primitive_ids,
      (std::vector<std::string>{"nominal-hop"}));
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
