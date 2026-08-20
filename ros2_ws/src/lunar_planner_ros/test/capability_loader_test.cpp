#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <numbers>
#include <string>
#include <variant>
#include <vector>

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

std::string ReplaceOnce(
    std::string document,
    const std::string& existing,
    const std::string& replacement) {
  const auto position = document.find(existing);
  EXPECT_NE(position, std::string::npos) << existing;
  if (position != std::string::npos) {
    document.replace(position, existing.size(), replacement);
  }
  return document;
}

void ExpectParametricFailure(
    const std::string& document,
    const std::string& reason_code,
    const std::string& detail) {
  const CapabilityLoadResult result = LoadParametricText(document);
  ASSERT_TRUE(result.error.has_value());
  EXPECT_EQ(result.error->reason_code, reason_code);
  EXPECT_EQ(result.error->detail, detail);
}

std::map<std::string, std::string, std::less<>> ApprovedWheelSources() {
  return {
      {"allow_unsupported_gap", "user_confirmed_capability"},
      {"body_extent_m", "user_provided_dimension"},
      {"footprint_xy_m", "derived"},
      {"maximum_acceleration_mps2", "user_confirmed_capability"},
      {"maximum_braking_deceleration_mps2", "user_confirmed_capability"},
      {"maximum_curvature_per_m", "user_confirmed_capability"},
      {"maximum_forward_speed_mps", "user_confirmed_capability"},
      {"maximum_lateral_acceleration_mps2",
       "user_approved_planning_policy"},
      {"maximum_local_obstacle_relief_m", "user_confirmed_capability"},
      {"maximum_reverse_speed_mps", "user_confirmed_capability"},
      {"maximum_spin_rate_radps", "derived"},
      {"maximum_surface_slope_rad", "user_confirmed_capability"},
      {"maximum_yaw_acceleration_radps2", "derived"},
      {"minimum_clearance_m", "user_approved_planning_policy"},
      {"minimum_underbody_clearance_m", "user_provided_dimension"},
      {"motion_primitives", "user_approved_planning_policy"},
      {"reference_point", "derived"},
      {"roughness_handling", "user_approved_planning_policy"},
      {"track_width_m", "user_provided_dimension"},
      {"wheel_center_xy_m", "derived"},
      {"wheel_count", "user_confirmed_capability"},
      {"wheel_diameter_m", "user_provided_dimension"},
      {"wheel_width_m", "user_provided_dimension"},
      {"wheelbase_m", "user_provided_dimension"},
  };
}

struct ExpectedWheelPrimitive final {
  std::string primitive_id;
  lunar::planning::WheelPrimitiveKind kind;
  lunar::planning::Vec3 position_m;
  lunar::planning::Quaternion orientation;
};

const std::array<ExpectedWheelPrimitive, 9U> kApprovedWheelPrimitives{{
    {"forward", lunar::planning::WheelPrimitiveKind::kForward,
     {0.2, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}},
    {"reverse", lunar::planning::WheelPrimitiveKind::kReverse,
     {-0.2, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}},
    {"forward-arc-left", lunar::planning::WheelPrimitiveKind::kForwardArc,
     {0.039018064403226, 0.003842943919354, 0.0},
     {0.995184726672197, 0.0, 0.0, 0.098017140329561}},
    {"forward-arc-right", lunar::planning::WheelPrimitiveKind::kForwardArc,
     {0.039018064403226, -0.003842943919354, 0.0},
     {0.995184726672197, 0.0, 0.0, -0.098017140329561}},
    {"reverse-arc-left", lunar::planning::WheelPrimitiveKind::kReverseArc,
     {-0.039018064403226, 0.003842943919354, 0.0},
     {0.995184726672197, 0.0, 0.0, -0.098017140329561}},
    {"reverse-arc-right", lunar::planning::WheelPrimitiveKind::kReverseArc,
     {-0.039018064403226, -0.003842943919354, 0.0},
     {0.995184726672197, 0.0, 0.0, 0.098017140329561}},
    {"spin-left", lunar::planning::WheelPrimitiveKind::kSpinCounterclockwise,
     {0.0, 0.0, 0.0},
     {0.995184726672197, 0.0, 0.0, 0.098017140329561}},
    {"spin-right", lunar::planning::WheelPrimitiveKind::kSpinClockwise,
     {0.0, 0.0, 0.0},
     {0.995184726672197, 0.0, 0.0, -0.098017140329561}},
    {"stop-switch", lunar::planning::WheelPrimitiveKind::kStopAndSwitch,
     {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}},
}};

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
  ASSERT_TRUE(result.capabilities.has_value());
  EXPECT_EQ(result.capabilities->platform_id, "wheel");
  EXPECT_EQ(result.capabilities->capability_version, "wheel-v1");
  EXPECT_EQ(result.capabilities->base_frame_id, "base_footprint");
  EXPECT_EQ(result.capabilities->reference_point, "base_footprint");
  EXPECT_EQ(result.capabilities->geometry_source_kind,
            GeometrySourceKind::kParametricEnvelope);
  EXPECT_TRUE(result.capabilities->urdf_path.empty());
  EXPECT_TRUE(result.capabilities->mesh_paths.empty());
  ASSERT_TRUE(result.capabilities->parametric_wheeled_geometry.has_value());
  const auto& audit_geometry =
      *result.capabilities->parametric_wheeled_geometry;
  EXPECT_EQ(audit_geometry.wheel_count, 4U);
  EXPECT_EQ(
      audit_geometry.wheel_center_xy_m,
      (std::vector<lunar::planning::Vec2>{
          {0.4075, 0.3115},
          {0.4075, -0.3115},
          {-0.4075, -0.3115},
          {-0.4075, 0.3115},
      }));
  EXPECT_EQ(result.capabilities->field_source_types, ApprovedWheelSources());
  EXPECT_EQ(
      result.capabilities->source_motion_primitive_ids,
      (std::vector<std::string>{
          "forward", "reverse", "forward-arc-left", "forward-arc-right",
          "reverse-arc-left", "reverse-arc-right", "spin-left", "spin-right",
          "stop-switch"}));
  EXPECT_DOUBLE_EQ(result.capabilities->observation.sensor_range_m, 25.0);
  EXPECT_NEAR(
      result.capabilities->observation.sensor_fov_rad,
      std::numbers::pi / 2.0,
      1.0e-12);

  const auto& wheel = std::get<lunar::planning::WheeledCapability>(
      result.capabilities->platform);
  EXPECT_EQ(
      wheel.footprint_xy_m,
      (std::vector<lunar::planning::Vec2>{
          {0.6505, 0.404},
          {0.6505, -0.404},
          {-0.6505, -0.404},
          {-0.6505, 0.404},
      }));
  EXPECT_EQ(wheel.body_extent_m, (lunar::planning::Vec3{1.301, 0.808, 1.363}));
  EXPECT_DOUBLE_EQ(wheel.wheel_diameter_m, 0.304);
  EXPECT_DOUBLE_EQ(wheel.wheel_width_m, 0.148);
  EXPECT_DOUBLE_EQ(wheel.wheelbase_m, 0.815);
  EXPECT_DOUBLE_EQ(wheel.track_width_m, 0.623);
  EXPECT_DOUBLE_EQ(wheel.minimum_underbody_clearance_m, 0.214);
  EXPECT_DOUBLE_EQ(wheel.maximum_local_obstacle_relief_m, 0.2);
  EXPECT_FALSE(wheel.allow_unsupported_gap);
  EXPECT_DOUBLE_EQ(wheel.minimum_body_z_m, 0.0);
  EXPECT_DOUBLE_EQ(wheel.maximum_body_z_m, 1.363);
  EXPECT_DOUBLE_EQ(wheel.maximum_forward_speed_mps, 0.2);
  EXPECT_DOUBLE_EQ(wheel.maximum_reverse_speed_mps, 0.2);
  EXPECT_NEAR(wheel.maximum_spin_rate_radps, 0.389923188554511, 1.0e-15);
  EXPECT_DOUBLE_EQ(wheel.maximum_acceleration_mps2, 0.2);
  EXPECT_DOUBLE_EQ(wheel.maximum_braking_deceleration_mps2, 0.2);
  EXPECT_NEAR(
      wheel.maximum_yaw_acceleration_radps2,
      0.389923188554511,
      1.0e-15);
  EXPECT_DOUBLE_EQ(wheel.maximum_lateral_acceleration_mps2, 0.2);
  EXPECT_DOUBLE_EQ(wheel.maximum_curvature_per_m, 5.0);
  EXPECT_NEAR(wheel.maximum_slope_rad, 0.174532925199433, 1.0e-15);
  EXPECT_DOUBLE_EQ(wheel.minimum_clearance_m, 0.1);
  ASSERT_EQ(wheel.motion_primitives.size(), kApprovedWheelPrimitives.size());
  for (std::size_t index = 0U; index < kApprovedWheelPrimitives.size(); ++index) {
    const auto& actual = wheel.motion_primitives[index];
    const auto& expected = kApprovedWheelPrimitives[index];
    EXPECT_EQ(actual.primitive_id, expected.primitive_id);
    EXPECT_EQ(actual.kind, expected.kind);
    EXPECT_NEAR(actual.relative_end_pose.position_m.x, expected.position_m.x,
                1.0e-15);
    EXPECT_NEAR(actual.relative_end_pose.position_m.y, expected.position_m.y,
                1.0e-15);
    EXPECT_NEAR(actual.relative_end_pose.position_m.z, expected.position_m.z,
                1.0e-15);
    EXPECT_NEAR(actual.relative_end_pose.orientation.w, expected.orientation.w,
                1.0e-15);
    EXPECT_NEAR(actual.relative_end_pose.orientation.x, expected.orientation.x,
                1.0e-15);
    EXPECT_NEAR(actual.relative_end_pose.orientation.y, expected.orientation.y,
                1.0e-15);
    EXPECT_NEAR(actual.relative_end_pose.orientation.z, expected.orientation.z,
                1.0e-15);
  }
}

TEST(CapabilityLoader, SelectsExactlyOneConfiguredPathMode) {
  const auto absolute = UniqueShare("absolute-mode");
  WriteObservation(absolute);
  Write(absolute / "config" / "platform.yaml", TrackedWheelYaml());

  EXPECT_TRUE(CapabilityLoader{}.LoadConfigured(
      "", absolute / "config/platform.yaml",
      absolute / "config/observation.json").ok());

  const auto missing_package = CapabilityLoader{}.LoadConfigured(
      "package_that_does_not_exist", "config/platform.yaml",
      "config/observation.json");
  ASSERT_TRUE(missing_package.error.has_value());
  EXPECT_EQ(missing_package.error->reason_code, "CAPABILITY_PACKAGE_NOT_FOUND");

  const auto relative_without_package = CapabilityLoader{}.LoadConfigured(
      "", "config/platform.yaml", "config/observation.json");
  ASSERT_TRUE(relative_without_package.error.has_value());
  EXPECT_EQ(
      relative_without_package.error->reason_code,
      "CAPABILITY_PATH_MODE_INVALID");

  const auto absolute_with_package = CapabilityLoader{}.LoadConfigured(
      "robot_pkg", "/tmp/platform.yaml", "/tmp/observation.json");
  ASSERT_TRUE(absolute_with_package.error.has_value());
  EXPECT_EQ(
      absolute_with_package.error->reason_code,
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

TEST(CapabilityLoader, RejectsUrdfGeometryFromAbsoluteFiles) {
  const auto share = UniqueShare("absolute-urdf");
  WriteGeometry(share, true);
  WriteObservation(share);
  Write(share / "config" / "platform.yaml", WheeledYaml());

  const auto result = CapabilityLoader{}.LoadFromFiles(
      share / "config/platform.yaml", share / "config/observation.json");

  ASSERT_TRUE(result.error.has_value());
  EXPECT_EQ(result.error->reason_code, "CAPABILITY_PATH_MODE_INVALID");
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

  std::string duplicate_center = TrackedWheelYaml();
  const auto wheel_centers = duplicate_center.find("wheel_center_xy_m:");
  duplicate_center.replace(
      wheel_centers,
      duplicate_center.find('\n', wheel_centers) - wheel_centers,
      "wheel_center_xy_m: [[0.4075, 0.3115], [0.4075, 0.3115], "
      "[-0.4075, -0.3115], [-0.4075, 0.3115]]");
  EXPECT_EQ(LoadParametricText(duplicate_center).error->reason_code,
            "CAPABILITY_VALUE_INVALID");

  std::string concave = TrackedWheelYaml();
  const auto concave_footprint = concave.find("footprint_xy_m:");
  concave.replace(
      concave_footprint,
      concave.find('\n', concave_footprint) - concave_footprint,
      "footprint_xy_m: [[0.6505, 0.404], [0.6505, -0.404], "
      "[0.0, -0.1], [-0.6505, -0.404], [-0.6505, 0.404]]");
  EXPECT_EQ(LoadParametricText(concave).error->reason_code,
            "CAPABILITY_VALUE_INVALID");

  std::string self_intersecting = TrackedWheelYaml();
  const auto star_footprint = self_intersecting.find("footprint_xy_m:");
  self_intersecting.replace(
      star_footprint,
      self_intersecting.find('\n', star_footprint) - star_footprint,
      "footprint_xy_m: [[0.0, 0.4], [0.2351, -0.3236], "
      "[-0.3804, 0.1236], [0.3804, 0.1236], [-0.2351, -0.3236]]");
  EXPECT_EQ(LoadParametricText(self_intersecting).error->reason_code,
            "CAPABILITY_VALUE_INVALID");
}

TEST(CapabilityLoader, RejectsWheelV1GeometryThatDriftsFromItsDimensions) {
  ExpectParametricFailure(
      ReplaceOnce(
          TrackedWheelYaml(),
          "wheel_center_xy_m: [[0.4075, 0.3115]",
          "wheel_center_xy_m: [[0.4, 0.3115]"),
      "CAPABILITY_VALUE_INVALID",
      "wheel_center_xy_m must match wheelbase_m and track_width_m");
  ExpectParametricFailure(
      ReplaceOnce(
          TrackedWheelYaml(),
          "minimum_underbody_clearance_m: 0.214",
          "minimum_underbody_clearance_m: 1.363"),
      "CAPABILITY_VALUE_INVALID",
      "minimum_underbody_clearance_m must be less than body_extent_m.z");
  ExpectParametricFailure(
      ReplaceOnce(
          TrackedWheelYaml(),
          "footprint_xy_m: [[0.6505, 0.404], [0.6505, -0.404], "
          "[-0.6505, -0.404], [-0.6505, 0.404]]",
          "footprint_xy_m: [[0.6, 0.35], [0.6, -0.35], "
          "[-0.6, -0.35], [-0.6, 0.35]]"),
      "CAPABILITY_VALUE_INVALID",
      "footprint_xy_m must match wheel-v1 full body envelope");
}

TEST(CapabilityLoader, RejectsUnapprovedWheelV1PrimitiveSemantics) {
  ExpectParametricFailure(
      ReplaceOnce(
          TrackedWheelYaml(),
          "position_m: [0.2, 0.0, 0.0]",
          "position_m: [0.2, 0.01, 0.0]"),
      "CAPABILITY_VALUE_INVALID",
      "motion primitive forward does not match wheel-v1 definition");
  ExpectParametricFailure(
      ReplaceOnce(
          TrackedWheelYaml(),
          "- primitive_id: spin-left\n"
          "      kind: SPIN_COUNTERCLOCKWISE\n"
          "      relative_end_pose: {position_m: [0.0, 0.0, 0.0]",
          "- primitive_id: spin-left\n"
          "      kind: SPIN_COUNTERCLOCKWISE\n"
          "      relative_end_pose: {position_m: [0.01, 0.0, 0.0]"),
      "CAPABILITY_VALUE_INVALID",
      "motion primitive spin-left does not match wheel-v1 definition");
  ExpectParametricFailure(
      ReplaceOnce(
          TrackedWheelYaml(),
          "orientation_wxyz: [1.0, 0.0, 0.0, 0.0]",
          "orientation_wxyz: [1.0001, 0.0, 0.0, 0.0]"),
      "CAPABILITY_VALUE_INVALID",
      "quaternion is not unit length: "
      "motion_primitives.relative_end_pose.orientation_wxyz");
}

TEST(CapabilityLoader, RejectsWheelV1IdentityAndAuditMetadataDrift) {
  ExpectParametricFailure(
      ReplaceOnce(TrackedWheelYaml(), "platform_id: wheel", "platform_id: rover"),
      "CAPABILITY_VALUE_INVALID",
      "platform.platform_id must be wheel");
  ExpectParametricFailure(
      ReplaceOnce(
          TrackedWheelYaml(),
          "platform_type: WHEELED",
          "platform_type: LEGGED"),
      "CAPABILITY_VALUE_INVALID",
      "platform.platform_type must be WHEELED");
  ExpectParametricFailure(
      ReplaceOnce(
          TrackedWheelYaml(),
          "capability_version: wheel-v1",
          "capability_version: wheel-v2"),
      "CAPABILITY_VALUE_INVALID",
      "platform.capability_version must be wheel-v1");
  ExpectParametricFailure(
      ReplaceOnce(
          TrackedWheelYaml(),
          "base_frame_id: base_footprint",
          "base_frame_id: base_link"),
      "CAPABILITY_VALUE_INVALID",
      "platform.base_frame_id must be base_footprint");
  ExpectParametricFailure(
      ReplaceOnce(
          TrackedWheelYaml(),
          "level: approved_user_parameter_baseline",
          "level: user_confirmed_capability"),
      "CAPABILITY_VALUE_INVALID",
      "platform.provenance.level must be approved_user_parameter_baseline");
  ExpectParametricFailure(
      ReplaceOnce(
          TrackedWheelYaml(),
          "2026-08-20-wheel-parametric-capability-design.md",
          "2026-08-07-wheeled-platform-capability-design.md"),
      "CAPABILITY_VALUE_INVALID",
      "platform.provenance.design_document must name the wheel-v1 design");
  ExpectParametricFailure(
      ReplaceOnce(
          TrackedWheelYaml(),
          "unknown_fields: [bare_mass_kg, ",
          "unknown_fields: ["),
      "CAPABILITY_VALUE_INVALID",
      "platform.unknown_fields must exactly match wheel-v1 unknown fields");
}

TEST(CapabilityLoader, RejectsMissingOrUnsupportedFieldSources) {
  ExpectParametricFailure(
      WithoutSource(TrackedWheelYaml(), "maximum_curvature_per_m"),
      "CAPABILITY_SCHEMA_INVALID",
      "sources missing fields: maximum_curvature_per_m");
  std::string unsupported = TrackedWheelYaml();
  const std::string approved =
      "maximum_curvature_per_m: user_confirmed_capability";
  const auto source = unsupported.find(
      approved,
      unsupported.find("sources:\n"));
  unsupported.replace(source, approved.size(), "maximum_curvature_per_m: guess");
  ExpectParametricFailure(
      unsupported,
      "CAPABILITY_VALUE_INVALID",
      "unsupported field source type for maximum_curvature_per_m: guess");

  ExpectParametricFailure(
      ReplaceOnce(
          TrackedWheelYaml(),
          approved,
          "maximum_curvature_per_m: derived"),
      "CAPABILITY_VALUE_INVALID",
      "sources invalid fields: maximum_curvature_per_m");

  std::string extra = TrackedWheelYaml();
  extra += "  baseline: derived\n";
  ExpectParametricFailure(
      extra,
      "CAPABILITY_SCHEMA_INVALID",
      "sources extra fields: baseline");
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
