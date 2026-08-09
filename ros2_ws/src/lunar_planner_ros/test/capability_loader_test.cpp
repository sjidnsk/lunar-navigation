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

std::filesystem::path ProfileRoot() {
  return std::filesystem::path{__FILE__}.parent_path().parent_path().parent_path() /
      "lunar_navigation_config" / "config" / "platform_profiles";
}

std::filesystem::path UniqueDirectory(const std::string& suffix) {
  const auto nonce = std::chrono::steady_clock::now()
      .time_since_epoch()
      .count();
  const auto path = std::filesystem::temp_directory_path() /
      ("lunar-profile-" + suffix + "-" + std::to_string(nonce));
  std::filesystem::create_directories(path);
  return path;
}

std::string Read(const std::filesystem::path& path) {
  std::ifstream stream{path, std::ios::binary};
  EXPECT_TRUE(stream.good()) << path;
  return {
      std::istreambuf_iterator<char>{stream},
      std::istreambuf_iterator<char>{}};
}

void Write(const std::filesystem::path& path, const std::string& content) {
  std::ofstream stream{path, std::ios::binary};
  ASSERT_TRUE(stream.good()) << path;
  stream << content;
  ASSERT_TRUE(stream.good()) << path;
}

std::filesystem::path CopyProfile(
    const std::string& platform,
    const std::string& suffix = {}) {
  const auto directory = UniqueDirectory(platform + suffix);
  const auto destination = directory / "platform_profile.yaml";
  Write(destination, Read(ProfileRoot() / (platform + ".yaml")));
  return destination;
}

TEST(CapabilityLoader, LoadsCompleteWheeledProfileAndHashesExactBytes) {
  const auto path = CopyProfile("wheeled");
  const CapabilityLoadResult result = CapabilityLoader{}.LoadFromFile(path);

  ASSERT_TRUE(result.ok())
      << (result.error.has_value() ? result.error->detail : std::string{});
  ASSERT_TRUE(result.capabilities.has_value());
  EXPECT_EQ(result.capabilities->platform_id, "wheeled-lunar-explorer");
  EXPECT_EQ(result.capabilities->base_frame_id, "base_footprint");
  EXPECT_EQ(
      result.capabilities->profile_sha256,
      "3a4f87e310cf7721be71818f5e4abc6cc73e78644bcdb0f8465426e8bcef9360");
  EXPECT_DOUBLE_EQ(result.capabilities->observation.sensor_range_m, 30.0);
  EXPECT_NEAR(
      result.capabilities->observation.sensor_fov_rad,
      2.0 * std::numbers::pi,
      1.0e-12);
  EXPECT_TRUE(result.capabilities->urdf_path.empty());
  EXPECT_TRUE(result.capabilities->mesh_paths.empty());
  EXPECT_TRUE(result.warnings.empty());

  const auto& wheel = std::get<lunar::planning::WheeledCapability>(
      result.capabilities->platform);
  EXPECT_NEAR(wheel.maximum_slope_rad, 0.3490658503988659, 1.0e-12);
  EXPECT_EQ(wheel.body_extent_m, (lunar::planning::Vec3{1.182, 0.818, 1.29996}));
  EXPECT_DOUBLE_EQ(wheel.wheel_diameter_m, 0.319);
  EXPECT_DOUBLE_EQ(wheel.wheel_width_m, 0.148);
  EXPECT_DOUBLE_EQ(wheel.wheelbase_m, 0.8175);
  EXPECT_DOUBLE_EQ(wheel.track_width_m, 0.67);
  EXPECT_DOUBLE_EQ(wheel.minimum_underbody_clearance_m, 0.21);
  EXPECT_DOUBLE_EQ(wheel.minimum_clearance_m, 0.2);
  EXPECT_DOUBLE_EQ(wheel.maximum_forward_speed_mps, 1.5);
  ASSERT_EQ(wheel.motion_primitives.size(), 9U);
  EXPECT_EQ(wheel.motion_primitives.front().primitive_id, "forward");
  EXPECT_EQ(
      wheel.motion_primitives.back().kind,
      lunar::planning::WheelPrimitiveKind::kStopAndSwitch);
}

TEST(CapabilityLoader, LoadsCompleteLeggedAndHopperProfiles) {
  const CapabilityLoadResult legged =
      CapabilityLoader{}.LoadFromFile(CopyProfile("legged"));
  ASSERT_TRUE(legged.ok())
      << (legged.error.has_value() ? legged.error->detail : std::string{});
  EXPECT_EQ(
      legged.capabilities->profile_sha256,
      "1f25b2fc4796e50ef7e966473c03cf902b1073291e2d1ccb09983ec90f0b17f0");
  const auto& legged_typed = std::get<lunar::planning::LeggedCapability>(
      legged.capabilities->platform);
  EXPECT_EQ(legged.capabilities->reference_point, "base_link");
  EXPECT_EQ(
      legged_typed.body_extent_m,
      (lunar::planning::Vec3{0.68, 0.33, 0.35}));
  EXPECT_DOUBLE_EQ(legged_typed.platform_mass_kg, 15.89);
  EXPECT_DOUBLE_EQ(legged_typed.maximum_payload_kg, 10.0);
  EXPECT_DOUBLE_EQ(legged_typed.maximum_step_height_m, 0.5);
  EXPECT_DOUBLE_EQ(legged_typed.forward_speed_mps.lower, -1.5);
  EXPECT_DOUBLE_EQ(legged_typed.forward_speed_mps.upper, 1.5);
  EXPECT_DOUBLE_EQ(legged_typed.lateral_speed_mps.lower, -0.8);
  EXPECT_DOUBLE_EQ(legged_typed.yaw_rate_radps.upper, 1.0);
  EXPECT_EQ(legged_typed.motion_primitives.size(), 6U);

  const CapabilityLoadResult hopper =
      CapabilityLoader{}.LoadFromFile(CopyProfile("hopper"));
  ASSERT_TRUE(hopper.ok())
      << (hopper.error.has_value() ? hopper.error->detail : std::string{});
  EXPECT_EQ(
      hopper.capabilities->profile_sha256,
      "1af41026d4c81500ce3351639d1fa7443f0c161b4de67f5da13d643b44dff841");
  const auto& hopper_typed = std::get<lunar::planning::HopperCapability>(
      hopper.capabilities->platform);
  EXPECT_DOUBLE_EQ(hopper_typed.specific_impulse_s, 301.0);
  EXPECT_DOUBLE_EQ(hopper_typed.reference_total_mass_kg, 20.0);
  EXPECT_DOUBLE_EQ(hopper_typed.reference_propellant_mass_kg, 0.2);
  EXPECT_DOUBLE_EQ(hopper_typed.landing_support_radius_m, 0.45);
  EXPECT_DOUBLE_EQ(hopper_typed.flight_collision_radius_m, 0.55);
  EXPECT_DOUBLE_EQ(hopper_typed.maximum_landing_plane_residual_m, 0.05);
  EXPECT_DOUBLE_EQ(hopper_typed.reachability_delta_v_margin_ratio, 0.1);
}

TEST(CapabilityLoader, ProfileHashChangesWhenAnyRawByteChanges) {
  const auto first = CopyProfile("wheeled", "-first");
  const auto second = CopyProfile("wheeled", "-second");
  Write(second, Read(second) + "\n");

  const auto first_result = CapabilityLoader{}.LoadFromFile(first);
  const auto second_result = CapabilityLoader{}.LoadFromFile(second);

  ASSERT_TRUE(first_result.ok());
  ASSERT_TRUE(second_result.ok());
  EXPECT_NE(
      first_result.capabilities->profile_sha256,
      second_result.capabilities->profile_sha256);
}

TEST(CapabilityLoader, MissingOptionalAssetsProduceWarningsNotFailure) {
  const auto profile = CopyProfile("wheeled", "-missing-assets");
  std::string content = Read(profile);
  const std::string old_assets = "  urdf: null\n  meshes: []";
  const std::string digest(64U, '0');
  const std::string new_assets =
      "  urdf:\n"
      "    path: missing/rover.urdf\n"
      "    sha256: " + digest + "\n"
      "  meshes:\n"
      "    - path: missing/body.stl\n"
      "      sha256: " + digest;
  const auto position = content.find(old_assets);
  ASSERT_NE(position, std::string::npos);
  content.replace(position, old_assets.size(), new_assets);
  Write(profile, content);

  const CapabilityLoadResult result = CapabilityLoader{}.LoadFromFile(profile);

  ASSERT_TRUE(result.ok())
      << (result.error.has_value() ? result.error->detail : std::string{});
  EXPECT_TRUE(result.capabilities->urdf_path.empty());
  EXPECT_TRUE(result.capabilities->mesh_paths.empty());
  ASSERT_EQ(result.warnings.size(), 2U);
  EXPECT_EQ(
      result.warnings[0].reason_code,
      "CAPABILITY_OPTIONAL_URDF_UNAVAILABLE");
  EXPECT_EQ(
      result.warnings[1].reason_code,
      "CAPABILITY_OPTIONAL_MESH_UNAVAILABLE");
}

TEST(CapabilityLoader, RejectsWrongSchemaMissingObservationAndNonfiniteValue) {
  const auto wrong_schema = CopyProfile("wheeled", "-wrong-schema");
  std::string content = Read(wrong_schema);
  content.replace(
      content.find("lunar-platform-profile/v1"),
      std::string{"lunar-platform-profile/v1"}.size(),
      "lunar-platform-profile/v0");
  Write(wrong_schema, content);
  const auto schema_result = CapabilityLoader{}.LoadFromFile(wrong_schema);
  ASSERT_TRUE(schema_result.error.has_value());
  EXPECT_EQ(schema_result.error->code, CapabilityLoadErrorCode::kSchemaInvalid);
  EXPECT_EQ(
      schema_result.error->reason_code,
      "CAPABILITY_SCHEMA_VERSION_INCOMPATIBLE");

  const auto missing_observation = CopyProfile("legged", "-no-observation");
  content = Read(missing_observation);
  const auto begin = content.find("observation:\n");
  const auto end = content.find("assets:\n", begin);
  ASSERT_NE(begin, std::string::npos);
  ASSERT_NE(end, std::string::npos);
  content.erase(begin, end - begin);
  Write(missing_observation, content);
  const auto observation_result =
      CapabilityLoader{}.LoadFromFile(missing_observation);
  ASSERT_TRUE(observation_result.error.has_value());
  EXPECT_EQ(
      observation_result.error->code,
      CapabilityLoadErrorCode::kSchemaInvalid);

  const auto nonfinite = CopyProfile("hopper", "-nonfinite");
  content = Read(nonfinite);
  const std::string finite = "specific_impulse_s: 301.0";
  content.replace(content.find(finite), finite.size(), "specific_impulse_s: .nan");
  Write(nonfinite, content);
  const auto value_result = CapabilityLoader{}.LoadFromFile(nonfinite);
  ASSERT_TRUE(value_result.error.has_value());
  EXPECT_EQ(value_result.error->code, CapabilityLoadErrorCode::kValueInvalid);
}

TEST(CapabilityLoader, RejectsUnknownPackageAndUnsafePackageRelativePath) {
  const CapabilityLoadResult package_result =
      CapabilityLoader{}.LoadFromPackageShare(
          "lunar_package_that_does_not_exist", "config/platform_profile.yaml");
  ASSERT_TRUE(package_result.error.has_value());
  EXPECT_EQ(
      package_result.error->code,
      CapabilityLoadErrorCode::kPackageNotFound);

  const CapabilityLoadResult unsafe = CapabilityLoader{}.LoadFromPackageShare(
      "lunar_navigation_config", "../platform_profile.yaml");
  ASSERT_TRUE(unsafe.error.has_value());
  EXPECT_EQ(unsafe.error->code, CapabilityLoadErrorCode::kUnsafePath);
}

}  // namespace
}  // namespace lunar::planning::ros
