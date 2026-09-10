#include "lunar_incremental_navigation_ros/platform_config.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace lunar::incremental_navigation_ros {
namespace {

using lunar::incremental_navigation::Interval;
using lunar::incremental_navigation::LeggedBodyPrimitive;
using lunar::incremental_navigation::LeggedCapability;
using lunar::incremental_navigation::LeggedPrimitiveKind;
using lunar::incremental_navigation::PlatformCapability;
using lunar::incremental_navigation::Pose3;
using lunar::incremental_navigation::Quaternion;
using lunar::incremental_navigation::Vec2;
using lunar::incremental_navigation::Vec3;
using lunar::incremental_navigation::WheeledCapability;
using lunar::incremental_navigation::WheelMotionPrimitive;
using lunar::incremental_navigation::WheelPrimitiveKind;

constexpr std::string_view kPlannerError{"PLANNER_ERROR"};

[[noreturn]] void Invalid() { throw std::runtime_error{"invalid platform config"}; }

void RequireKeys(const YAML::Node& node,
                 std::initializer_list<std::string_view> expected) {
  if (!node.IsMap() || node.size() != expected.size()) {
    Invalid();
  }
  for (const auto& entry : node) {
    if (!entry.first.IsScalar()) {
      Invalid();
    }
    const std::string key = entry.first.as<std::string>();
    bool found = false;
    for (const std::string_view candidate : expected) {
      if (key == candidate) {
        found = true;
        break;
      }
    }
    if (!found) {
      Invalid();
    }
  }
}

YAML::Node Require(const YAML::Node& node, std::string_view key) {
  const YAML::Node value = node[std::string{key}];
  if (!value) {
    Invalid();
  }
  return value;
}

std::string String(const YAML::Node& node, std::string_view key) {
  const YAML::Node& value = Require(node, key);
  if (!value.IsScalar()) {
    Invalid();
  }
  const std::string result = value.as<std::string>();
  if (result.empty()) {
    Invalid();
  }
  return result;
}

double Finite(const YAML::Node& node, std::string_view key) {
  const YAML::Node& value = Require(node, key);
  if (!value.IsScalar()) {
    Invalid();
  }
  const double result = value.as<double>();
  if (!std::isfinite(result)) {
    Invalid();
  }
  return result;
}

double Positive(const YAML::Node& node, std::string_view key) {
  const double value = Finite(node, key);
  if (value <= 0.0) {
    Invalid();
  }
  return value;
}

double NonNegative(const YAML::Node& node, std::string_view key) {
  const double value = Finite(node, key);
  if (value < 0.0) {
    Invalid();
  }
  return value;
}

bool Boolean(const YAML::Node& node, std::string_view key) {
  const YAML::Node& value = Require(node, key);
  if (!value.IsScalar()) {
    Invalid();
  }
  return value.as<bool>();
}

Vec3 Vec3Value(const YAML::Node& node, std::string_view key) {
  const YAML::Node& value = Require(node, key);
  if (!value.IsSequence() || value.size() != 3U) {
    Invalid();
  }
  const Vec3 result{value[0U].as<double>(), value[1U].as<double>(),
                    value[2U].as<double>()};
  if (!std::isfinite(result.x) || !std::isfinite(result.y) ||
      !std::isfinite(result.z)) {
    Invalid();
  }
  return result;
}

Interval IntervalValue(const YAML::Node& node, std::string_view key) {
  const YAML::Node& value = Require(node, key);
  if (!value.IsSequence() || value.size() != 2U) {
    Invalid();
  }
  const Interval result{value[0U].as<double>(), value[1U].as<double>()};
  if (!std::isfinite(result.lower) || !std::isfinite(result.upper) ||
      result.lower > result.upper) {
    Invalid();
  }
  return result;
}

Quaternion YawQuaternion(const double yaw) {
  return Quaternion{.w = std::cos(yaw / 2.0), .z = std::sin(yaw / 2.0)};
}

WheelPrimitiveKind WheelKind(const std::string& name) {
  if (name == "FORWARD") return WheelPrimitiveKind::kForward;
  if (name == "REVERSE") return WheelPrimitiveKind::kReverse;
  if (name == "FORWARD_ARC") return WheelPrimitiveKind::kForwardArc;
  if (name == "REVERSE_ARC") return WheelPrimitiveKind::kReverseArc;
  if (name == "SPIN_CLOCKWISE") return WheelPrimitiveKind::kSpinClockwise;
  if (name == "SPIN_COUNTERCLOCKWISE") return WheelPrimitiveKind::kSpinCounterclockwise;
  if (name == "STOP_AND_SWITCH") return WheelPrimitiveKind::kStopAndSwitch;
  Invalid();
}

LeggedPrimitiveKind LeggedKind(const std::string& name) {
  if (name == "FORWARD") return LeggedPrimitiveKind::kForward;
  if (name == "BACKWARD") return LeggedPrimitiveKind::kBackward;
  if (name == "LATERAL_LEFT") return LeggedPrimitiveKind::kLateralLeft;
  if (name == "LATERAL_RIGHT") return LeggedPrimitiveKind::kLateralRight;
  if (name == "SPIN") return LeggedPrimitiveKind::kSpin;
  Invalid();
}

std::vector<WheelMotionPrimitive> WheelPrimitives(const YAML::Node& node) {
  const YAML::Node& values = Require(node, "motion_primitives");
  if (!values.IsSequence() || values.size() != 9U) {
    Invalid();
  }
  std::vector<WheelMotionPrimitive> result;
  result.reserve(values.size());
  for (const YAML::Node& value : values) {
    RequireKeys(value, {"primitive_id", "kind", "position_m", "yaw_change_rad"});
    const std::string id = String(value, "primitive_id");
    const double yaw = Finite(value, "yaw_change_rad");
    result.push_back(WheelMotionPrimitive{
        .primitive_id = id,
        .kind = WheelKind(String(value, "kind")),
        .relative_end_pose = Pose3{.position_m = Vec3Value(value, "position_m"),
                                   .orientation = YawQuaternion(yaw)},
    });
  }
  return result;
}

std::vector<LeggedBodyPrimitive> LeggedPrimitives(const YAML::Node& node) {
  const YAML::Node& values = Require(node, "motion_primitives");
  if (!values.IsSequence() || values.size() != 6U) {
    Invalid();
  }
  std::vector<LeggedBodyPrimitive> result;
  result.reserve(values.size());
  for (const YAML::Node& value : values) {
    RequireKeys(value, {"primitive_id", "kind", "body_frame_displacement_m", "yaw_change_rad"});
    result.push_back(LeggedBodyPrimitive{
        .primitive_id = String(value, "primitive_id"),
        .kind = LeggedKind(String(value, "kind")),
        .body_frame_displacement_m = Vec3Value(value, "body_frame_displacement_m"),
        .yaw_change_rad = Finite(value, "yaw_change_rad"),
    });
  }
  return result;
}

void ValidateIdentity(const YAML::Node& root, const std::string_view platform,
                      const std::string_view id, const std::string_view type,
                      const std::string_view version, const std::string_view frame,
                      const bool surface_platform) {
  if (surface_platform) {
    if (root["start_blind_zone_margin_m"]) {
      RequireKeys(root, {"platform", "platform_id", "platform_type",
                         "capability_version", "base_frame_id",
                         "start_blind_zone_margin_m", "capability"});
    } else {
      RequireKeys(root, {"platform", "platform_id", "platform_type",
                         "capability_version", "base_frame_id", "capability"});
    }
  } else {
    RequireKeys(root, {"platform", "platform_id", "platform_type",
                       "capability_version", "base_frame_id", "capability"});
  }
  if (String(root, "platform") != platform || String(root, "platform_id") != id ||
      String(root, "platform_type") != type ||
      String(root, "capability_version") != version ||
      String(root, "base_frame_id") != frame) {
    Invalid();
  }
}

PlatformCapability ParseWheel(const YAML::Node& root) {
  ValidateIdentity(root, "wheel", "wheeled-lunar-explorer", "WHEELED",
                   "wheeled-engineering-baseline-v1", "base_footprint", true);
  const YAML::Node& node = Require(root, "capability");
  RequireKeys(node, {"footprint_xy_m", "body_extent_m", "wheel_diameter_m", "wheel_width_m",
                     "wheelbase_m", "track_width_m", "minimum_underbody_clearance_m",
                     "maximum_local_obstacle_relief_m", "allow_unsupported_gap",
                     "maximum_forward_speed_mps", "maximum_reverse_speed_mps",
                     "maximum_spin_rate_radps", "maximum_acceleration_mps2",
                     "maximum_braking_deceleration_mps2", "maximum_yaw_acceleration_radps2",
                     "maximum_lateral_acceleration_mps2", "maximum_curvature_per_m",
                     "maximum_slope_rad", "minimum_clearance_m", "motion_primitives"});
  const YAML::Node& footprint = Require(node, "footprint_xy_m");
  if (!footprint.IsSequence() || footprint.size() != 4U) {
    Invalid();
  }
  std::vector<Vec2> footprint_xy_m;
  footprint_xy_m.reserve(footprint.size());
  for (const YAML::Node& point : footprint) {
    if (!point.IsSequence() || point.size() != 2U) Invalid();
    const Vec2 value{point[0U].as<double>(), point[1U].as<double>()};
    if (!std::isfinite(value.x) || !std::isfinite(value.y)) Invalid();
    footprint_xy_m.push_back(value);
  }
  const Vec3 body_extent = Vec3Value(node, "body_extent_m");
  if (body_extent.x <= 0.0 || body_extent.y <= 0.0 || body_extent.z <= 0.0) Invalid();
  return WheeledCapability{
      .footprint_xy_m = std::move(footprint_xy_m), .body_extent_m = body_extent,
      .wheel_diameter_m = Positive(node, "wheel_diameter_m"), .wheel_width_m = Positive(node, "wheel_width_m"),
      .wheelbase_m = Positive(node, "wheelbase_m"), .track_width_m = Positive(node, "track_width_m"),
      .minimum_underbody_clearance_m = Positive(node, "minimum_underbody_clearance_m"),
      .maximum_local_obstacle_relief_m = NonNegative(node, "maximum_local_obstacle_relief_m"),
      .allow_unsupported_gap = Boolean(node, "allow_unsupported_gap"),
      .minimum_body_z_m = 0.0, .maximum_body_z_m = body_extent.z,
      .maximum_forward_speed_mps = Positive(node, "maximum_forward_speed_mps"),
      .maximum_reverse_speed_mps = Positive(node, "maximum_reverse_speed_mps"),
      .maximum_spin_rate_radps = Positive(node, "maximum_spin_rate_radps"),
      .maximum_acceleration_mps2 = Positive(node, "maximum_acceleration_mps2"),
      .maximum_braking_deceleration_mps2 = Positive(node, "maximum_braking_deceleration_mps2"),
      .maximum_yaw_acceleration_radps2 = Positive(node, "maximum_yaw_acceleration_radps2"),
      .maximum_lateral_acceleration_mps2 = Positive(node, "maximum_lateral_acceleration_mps2"),
      .maximum_curvature_per_m = Positive(node, "maximum_curvature_per_m"),
      .maximum_slope_rad = Positive(node, "maximum_slope_rad"), .minimum_clearance_m = NonNegative(node, "minimum_clearance_m"),
      .motion_primitives = WheelPrimitives(node),
  };
}

PlatformCapability ParseLegged(const YAML::Node& root) {
  ValidateIdentity(root, "legged", "yobotics-quad48", "LEGGED",
                   "quad48-approved-baseline-v2", "base_link", true);
  const YAML::Node& node = Require(root, "capability");
  RequireKeys(node, {"body_extent_m", "nominal_body_height_m", "body_height_m", "platform_mass_kg",
                     "nominal_payload_kg", "maximum_payload_kg", "maximum_forward_speed_mps",
                     "maximum_reverse_speed_mps", "maximum_lateral_speed_mps", "maximum_yaw_rate_radps",
                     "maximum_linear_acceleration_mps2", "maximum_yaw_acceleration_radps2",
                     "maximum_slope_rad", "maximum_step_height_m", "maximum_gap_width_m",
                     "minimum_body_clearance_m", "step_vertical_rate_mps", "unknown_is_traversable",
                     "motion_primitives"});
  const Vec3 body_extent = Vec3Value(node, "body_extent_m");
  if (body_extent.x <= 0.0 || body_extent.y <= 0.0 || body_extent.z <= 0.0) Invalid();
  const Interval body_height = IntervalValue(node, "body_height_m");
  if (body_height.lower < 0.0) Invalid();
  const double forward = Positive(node, "maximum_forward_speed_mps");
  const double reverse = Positive(node, "maximum_reverse_speed_mps");
  const double lateral = Positive(node, "maximum_lateral_speed_mps");
  const double yaw = Positive(node, "maximum_yaw_rate_radps");
  return LeggedCapability{
      .body_extent_m = body_extent, .nominal_body_height_m = Positive(node, "nominal_body_height_m"),
      .platform_mass_kg = Positive(node, "platform_mass_kg"), .nominal_payload_kg = Positive(node, "nominal_payload_kg"),
      .maximum_payload_kg = Positive(node, "maximum_payload_kg"), .maximum_slope_rad = Positive(node, "maximum_slope_rad"),
      .maximum_step_height_m = NonNegative(node, "maximum_step_height_m"), .maximum_gap_width_m = NonNegative(node, "maximum_gap_width_m"),
      .minimum_body_clearance_m = NonNegative(node, "minimum_body_clearance_m"), .step_vertical_rate_mps = Positive(node, "step_vertical_rate_mps"),
      .body_height_m = body_height, .forward_speed_mps = {-reverse, forward}, .lateral_speed_mps = {-lateral, lateral},
      .yaw_rate_radps = {-yaw, yaw}, .maximum_linear_acceleration_mps2 = Positive(node, "maximum_linear_acceleration_mps2"),
      .maximum_yaw_acceleration_radps2 = Positive(node, "maximum_yaw_acceleration_radps2"),
      .unknown_is_traversable = Boolean(node, "unknown_is_traversable"), .motion_primitives = LeggedPrimitives(node),
  };
}

bool SameVec2(const Vec2& actual, const Vec2& expected) {
  return actual.x == expected.x && actual.y == expected.y;
}

bool SameVec3(const Vec3& actual, const Vec3& expected) {
  return actual.x == expected.x && actual.y == expected.y && actual.z == expected.z;
}

bool SameQuaternion(const Quaternion& actual, const Quaternion& expected) {
  return actual.w == expected.w && actual.x == expected.x &&
      actual.y == expected.y && actual.z == expected.z;
}

bool SameWheel(const WheeledCapability& actual, const WheeledCapability& expected) {
  if (actual.footprint_xy_m.size() != expected.footprint_xy_m.size() ||
      actual.motion_primitives.size() != expected.motion_primitives.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < actual.footprint_xy_m.size(); ++index) {
    if (!SameVec2(actual.footprint_xy_m[index], expected.footprint_xy_m[index])) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < actual.motion_primitives.size(); ++index) {
    const auto& lhs = actual.motion_primitives[index];
    const auto& rhs = expected.motion_primitives[index];
    if (lhs.primitive_id != rhs.primitive_id || lhs.kind != rhs.kind ||
        !SameVec3(lhs.relative_end_pose.position_m, rhs.relative_end_pose.position_m) ||
        !SameQuaternion(lhs.relative_end_pose.orientation, rhs.relative_end_pose.orientation)) {
      return false;
    }
  }
  return SameVec3(actual.body_extent_m, expected.body_extent_m) &&
      actual.wheel_diameter_m == expected.wheel_diameter_m &&
      actual.wheel_width_m == expected.wheel_width_m &&
      actual.wheelbase_m == expected.wheelbase_m &&
      actual.track_width_m == expected.track_width_m &&
      actual.minimum_underbody_clearance_m == expected.minimum_underbody_clearance_m &&
      actual.maximum_local_obstacle_relief_m == expected.maximum_local_obstacle_relief_m &&
      actual.allow_unsupported_gap == expected.allow_unsupported_gap &&
      actual.minimum_body_z_m == expected.minimum_body_z_m &&
      actual.maximum_body_z_m == expected.maximum_body_z_m &&
      actual.maximum_forward_speed_mps == expected.maximum_forward_speed_mps &&
      actual.maximum_reverse_speed_mps == expected.maximum_reverse_speed_mps &&
      actual.maximum_spin_rate_radps == expected.maximum_spin_rate_radps &&
      actual.maximum_acceleration_mps2 == expected.maximum_acceleration_mps2 &&
      actual.maximum_braking_deceleration_mps2 == expected.maximum_braking_deceleration_mps2 &&
      actual.maximum_yaw_acceleration_radps2 == expected.maximum_yaw_acceleration_radps2 &&
      actual.maximum_lateral_acceleration_mps2 == expected.maximum_lateral_acceleration_mps2 &&
      actual.maximum_curvature_per_m == expected.maximum_curvature_per_m &&
      actual.maximum_slope_rad == expected.maximum_slope_rad &&
      actual.minimum_clearance_m == expected.minimum_clearance_m;
}

bool SameLegged(const LeggedCapability& actual, const LeggedCapability& expected) {
  if (actual.motion_primitives.size() != expected.motion_primitives.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < actual.motion_primitives.size(); ++index) {
    const auto& lhs = actual.motion_primitives[index];
    const auto& rhs = expected.motion_primitives[index];
    if (lhs.primitive_id != rhs.primitive_id || lhs.kind != rhs.kind ||
        !SameVec3(lhs.body_frame_displacement_m, rhs.body_frame_displacement_m) ||
        lhs.yaw_change_rad != rhs.yaw_change_rad) {
      return false;
    }
  }
  return SameVec3(actual.body_extent_m, expected.body_extent_m) &&
      actual.nominal_body_height_m == expected.nominal_body_height_m &&
      actual.platform_mass_kg == expected.platform_mass_kg &&
      actual.nominal_payload_kg == expected.nominal_payload_kg &&
      actual.maximum_payload_kg == expected.maximum_payload_kg &&
      actual.maximum_slope_rad == expected.maximum_slope_rad &&
      actual.maximum_step_height_m == expected.maximum_step_height_m &&
      actual.maximum_gap_width_m == expected.maximum_gap_width_m &&
      actual.minimum_body_clearance_m == expected.minimum_body_clearance_m &&
      actual.step_vertical_rate_mps == expected.step_vertical_rate_mps &&
      actual.body_height_m.lower == expected.body_height_m.lower &&
      actual.body_height_m.upper == expected.body_height_m.upper &&
      actual.forward_speed_mps.lower == expected.forward_speed_mps.lower &&
      actual.forward_speed_mps.upper == expected.forward_speed_mps.upper &&
      actual.lateral_speed_mps.lower == expected.lateral_speed_mps.lower &&
      actual.lateral_speed_mps.upper == expected.lateral_speed_mps.upper &&
      actual.yaw_rate_radps.lower == expected.yaw_rate_radps.lower &&
      actual.yaw_rate_radps.upper == expected.yaw_rate_radps.upper &&
      actual.maximum_linear_acceleration_mps2 == expected.maximum_linear_acceleration_mps2 &&
      actual.maximum_yaw_acceleration_radps2 == expected.maximum_yaw_acceleration_radps2 &&
      actual.unknown_is_traversable == expected.unknown_is_traversable;
}

PlatformCapability ExpectedCapability(const std::string_view platform) {
  if (platform == "wheel") {
    return WheeledCapability{
        .footprint_xy_m = {{0.591, 0.409}, {0.591, -0.409},
                           {-0.591, -0.409}, {-0.591, 0.409}},
        .body_extent_m = {1.182, 0.818, 1.29996},
        .wheel_diameter_m = 0.319,
        .wheel_width_m = 0.148,
        .wheelbase_m = 0.8175,
        .track_width_m = 0.67,
        .minimum_underbody_clearance_m = 0.21,
        .maximum_local_obstacle_relief_m = 0.2,
        .allow_unsupported_gap = false,
        .minimum_body_z_m = 0.0,
        .maximum_body_z_m = 1.29996,
        .maximum_forward_speed_mps = 0.2,
        .maximum_reverse_speed_mps = 0.2,
        .maximum_spin_rate_radps = 1.0,
        .maximum_acceleration_mps2 = 0.5,
        .maximum_braking_deceleration_mps2 = 0.5,
        .maximum_yaw_acceleration_radps2 = 0.5,
        .maximum_lateral_acceleration_mps2 = 0.5,
        .maximum_curvature_per_m = 1.0,
        .maximum_slope_rad = 0.3490658503988659,
        .minimum_clearance_m = 0.2,
        .motion_primitives = {
            {"forward", WheelPrimitiveKind::kForward,
             {.position_m = {0.2, 0.0, 0.0}}},
            {"reverse", WheelPrimitiveKind::kReverse,
             {.position_m = {-0.2, 0.0, 0.0}}},
            {"forward-arc-left", WheelPrimitiveKind::kForwardArc,
             {.position_m = {0.19509032201612825, 0.01921471959676957, 0.0},
              .orientation = YawQuaternion(0.19634954084936207)}},
            {"forward-arc-right", WheelPrimitiveKind::kForwardArc,
             {.position_m = {0.19509032201612825, -0.01921471959676957, 0.0},
              .orientation = YawQuaternion(-0.19634954084936207)}},
            {"reverse-arc-left", WheelPrimitiveKind::kReverseArc,
             {.position_m = {-0.19509032201612825, 0.01921471959676957, 0.0},
              .orientation = YawQuaternion(-0.19634954084936207)}},
            {"reverse-arc-right", WheelPrimitiveKind::kReverseArc,
             {.position_m = {-0.19509032201612825, -0.01921471959676957, 0.0},
              .orientation = YawQuaternion(0.19634954084936207)}},
            {"spin-left", WheelPrimitiveKind::kSpinCounterclockwise,
             {.orientation = YawQuaternion(0.19634954084936207)}},
            {"spin-right", WheelPrimitiveKind::kSpinClockwise,
             {.orientation = YawQuaternion(-0.19634954084936207)}},
            {"stop-switch", WheelPrimitiveKind::kStopAndSwitch, {}},
        },
    };
  }
  if (platform == "legged") {
    return LeggedCapability{
        .body_extent_m = {0.68, 0.33, 0.35},
        .nominal_body_height_m = 0.33,
        .platform_mass_kg = 15.89,
        .nominal_payload_kg = 8.0,
        .maximum_payload_kg = 10.0,
        .maximum_slope_rad = 0.5235987755982988,
        .maximum_step_height_m = 0.5,
        .maximum_gap_width_m = 0.3,
        .minimum_body_clearance_m = 0.3,
        .step_vertical_rate_mps = 0.1,
        .body_height_m = {0.28, 0.38},
        .forward_speed_mps = {-1.5, 1.5},
        .lateral_speed_mps = {-0.8, 0.8},
        .yaw_rate_radps = {-1.0, 1.0},
        .maximum_linear_acceleration_mps2 = 1.0,
        .maximum_yaw_acceleration_radps2 = 1.0,
        .unknown_is_traversable = false,
        .motion_primitives = {
            {"forward", LeggedPrimitiveKind::kForward, {0.2, 0.0, 0.0}, 0.0},
            {"backward", LeggedPrimitiveKind::kBackward, {-0.2, 0.0, 0.0}, 0.0},
            {"lateral-left", LeggedPrimitiveKind::kLateralLeft, {0.0, 0.2, 0.0}, 0.0},
            {"lateral-right", LeggedPrimitiveKind::kLateralRight, {0.0, -0.2, 0.0}, 0.0},
            {"spin-left", LeggedPrimitiveKind::kSpin, {0.0, 0.0, 0.0}, 0.19634954084936207},
            {"spin-right", LeggedPrimitiveKind::kSpin, {0.0, 0.0, 0.0}, -0.19634954084936207},
        },
    };
  }
  throw std::invalid_argument{"unsupported platform"};
}

bool MatchesExpectedCapability(const PlatformCapability& actual,
                               const std::string_view platform) {
  const PlatformCapability expected = ExpectedCapability(platform);
  if (platform == "wheel") {
    const auto* actual_wheel = std::get_if<WheeledCapability>(&actual);
    const auto* expected_wheel = std::get_if<WheeledCapability>(&expected);
    return actual_wheel != nullptr && expected_wheel != nullptr &&
        SameWheel(*actual_wheel, *expected_wheel);
  }
  const auto* actual_legged = std::get_if<LeggedCapability>(&actual);
  const auto* expected_legged = std::get_if<LeggedCapability>(&expected);
  return actual_legged != nullptr && expected_legged != nullptr &&
      SameLegged(*actual_legged, *expected_legged);
}

}  // namespace

PlatformConfigResult LoadPlatformConfig(const std::filesystem::path& path,
                                        const std::string_view platform) {
  if (platform != "wheel" && platform != "legged") {
    return {.capability = std::nullopt,
            .start_blind_zone_margin_m = std::nullopt,
            .reason_code = std::string{kPlannerError}};
  }
  try {
    const YAML::Node root = YAML::LoadFile(path.string());
    PlatformCapability capability =
        platform == "wheel" ? ParseWheel(root) : ParseLegged(root);
    if (!MatchesExpectedCapability(capability, platform)) {
      Invalid();
    }
    const std::optional<double> start_blind_zone_margin_m{
        root["start_blind_zone_margin_m"]
            ? NonNegative(root, "start_blind_zone_margin_m")
            : 0.2};
    return {.capability = std::move(capability),
            .start_blind_zone_margin_m = start_blind_zone_margin_m,
            .reason_code = {}};
  } catch (const std::exception&) {
    return {.capability = std::nullopt,
            .start_blind_zone_margin_m = std::nullopt,
            .reason_code = std::string{kPlannerError}};
  }
}

}  // namespace lunar::incremental_navigation_ros
