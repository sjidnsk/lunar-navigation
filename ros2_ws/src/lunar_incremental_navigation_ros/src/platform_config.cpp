#include "lunar_incremental_navigation_ros/platform_config.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <set>

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

[[noreturn]] void Invalid(const std::string& detail = "invalid platform config structure") { throw std::runtime_error{detail}; }

void RequireKeys(const YAML::Node& node,
                 std::initializer_list<std::string_view> expected) {
  if (!node.IsMap()) Invalid("expected configuration mapping");
  for (const auto key : expected) {
    if (!node[std::string{key}]) Invalid("missing field: " + std::string{key});
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
      Invalid("unknown field: " + key);
    }
  }
}

YAML::Node Require(const YAML::Node& node, std::string_view key) {
  const YAML::Node value = node[std::string{key}];
  if (!value) {
    Invalid("missing field: " + std::string{key});
  }
  return value;
}

std::string String(const YAML::Node& node, std::string_view key) {
  const YAML::Node& value = Require(node, key);
  if (!value.IsScalar()) {
    Invalid(std::string{key} + ": expected nonempty string");
  }
  const std::string result = value.as<std::string>();
  if (result.empty()) {
    Invalid(std::string{key} + ": expected nonempty string");
  }
  return result;
}

double Finite(const YAML::Node& node, std::string_view key) {
  const YAML::Node& value = Require(node, key);
  if (!value.IsScalar()) {
    Invalid(std::string{key} + ": expected finite number");
  }
  double result{};
  try {
    result = value.as<double>();
  } catch (const YAML::Exception&) {
    Invalid(std::string{key} + ": expected finite number");
  }
  if (!std::isfinite(result)) {
    Invalid(std::string{key} + ": expected finite number");
  }
  return result;
}

double Positive(const YAML::Node& node, std::string_view key) {
  const double value = Finite(node, key);
  if (value <= 0.0) {
    Invalid(std::string{key} + ": must be > 0");
  }
  return value;
}

double NonNegative(const YAML::Node& node, std::string_view key) {
  const double value = Finite(node, key);
  if (value < 0.0) {
    Invalid(std::string{key} + ": must be >= 0");
  }
  return value;
}

bool Boolean(const YAML::Node& node, std::string_view key) {
  const YAML::Node& value = Require(node, key);
  if (!value.IsScalar()) {
    Invalid(std::string{key} + ": expected boolean");
  }
  return value.as<bool>();
}

Vec3 Vec3Value(const YAML::Node& node, std::string_view key) {
  const YAML::Node& value = Require(node, key);
  if (!value.IsSequence() || value.size() != 3U) {
    Invalid(std::string{key} + ": expected three finite coordinates");
  }
  const Vec3 result{value[0U].as<double>(), value[1U].as<double>(),
                    value[2U].as<double>()};
  if (!std::isfinite(result.x) || !std::isfinite(result.y) ||
      !std::isfinite(result.z)) {
    Invalid(std::string{key} + ": expected three finite coordinates");
  }
  return result;
}

Interval IntervalValue(const YAML::Node& node, std::string_view key) {
  const YAML::Node& value = Require(node, key);
  if (!value.IsSequence() || value.size() != 2U) {
    Invalid(std::string{key} + ": expected finite [lower, upper], lower <= upper");
  }
  const Interval result{value[0U].as<double>(), value[1U].as<double>()};
  if (!std::isfinite(result.lower) || !std::isfinite(result.upper) ||
      result.lower > result.upper) {
    Invalid(std::string{key} + ": expected finite [lower, upper], lower <= upper");
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
  Invalid("motion_primitives.kind: unsupported " + name);
}

LeggedPrimitiveKind LeggedKind(const std::string& name) {
  if (name == "FORWARD") return LeggedPrimitiveKind::kForward;
  if (name == "BACKWARD") return LeggedPrimitiveKind::kBackward;
  if (name == "LATERAL_LEFT") return LeggedPrimitiveKind::kLateralLeft;
  if (name == "LATERAL_RIGHT") return LeggedPrimitiveKind::kLateralRight;
  if (name == "SPIN") return LeggedPrimitiveKind::kSpin;
  Invalid("motion_primitives.kind: unsupported " + name);
}

std::vector<WheelMotionPrimitive> WheelPrimitives(const YAML::Node& node) {
  const YAML::Node& values = Require(node, "motion_primitives");
  if (!values.IsSequence() || values.size() == 0U) {
    Invalid("motion_primitives: expected nonempty sequence");
  }
  std::vector<WheelMotionPrimitive> result;
  result.reserve(values.size());
  std::set<std::string> primitive_ids;
  for (const YAML::Node& value : values) {
    RequireKeys(value, {"primitive_id", "kind", "position_m", "yaw_change_rad"});
    const std::string id = String(value, "primitive_id");
    if (!primitive_ids.insert(id).second) Invalid("motion_primitives.primitive_id: duplicate " + id);
    const double yaw = Finite(value, "yaw_change_rad");
    const auto kind = WheelKind(String(value, "kind"));
    const auto displacement = Vec3Value(value, "position_m");
    if ((kind == WheelPrimitiveKind::kForward || kind == WheelPrimitiveKind::kForwardArc) && displacement.x <= 0.0)
      Invalid("motion_primitives.position_m: forward motion must have positive x");
    if ((kind == WheelPrimitiveKind::kReverse || kind == WheelPrimitiveKind::kReverseArc) && displacement.x >= 0.0)
      Invalid("motion_primitives.position_m: reverse motion must have negative x");

    result.push_back(WheelMotionPrimitive{
        .primitive_id = id,
        .kind = kind,
        .relative_end_pose = Pose3{.position_m = displacement,
                                   .orientation = YawQuaternion(yaw)},
    });
  }
  return result;
}

std::vector<LeggedBodyPrimitive> LeggedPrimitives(const YAML::Node& node) {
  const YAML::Node& values = Require(node, "motion_primitives");
  if (!values.IsSequence() || values.size() == 0U) {
    Invalid("motion_primitives: expected nonempty sequence");
  }
  std::vector<LeggedBodyPrimitive> result;
  result.reserve(values.size());
  std::set<std::string> primitive_ids;
  for (const YAML::Node& value : values) {
    RequireKeys(value, {"primitive_id", "kind", "body_frame_displacement_m", "yaw_change_rad"});
    if (!primitive_ids.insert(String(value, "primitive_id")).second) Invalid("motion_primitives.primitive_id: duplicate");
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
                      const std::string_view type,
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
  if (String(root, "platform") != platform || String(root, "platform_type") != type) {
    Invalid("platform/platform_type does not match selected platform " + std::string{platform});
  }
  // Identity labels describe a configured platform, not a frozen numeric profile.
  static_cast<void>(String(root, "platform_id"));
  static_cast<void>(String(root, "capability_version"));
  static_cast<void>(String(root, "base_frame_id"));
}

PlatformCapability ParseWheel(const YAML::Node& root) {
  ValidateIdentity(root, "wheel", "WHEELED", true);
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
  if (body_extent.x <= 0.0 || body_extent.y <= 0.0 || body_extent.z <= 0.0) Invalid("body_extent_m: dimensions must be > 0");
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
  ValidateIdentity(root, "legged", "LEGGED", true);
  const YAML::Node& node = Require(root, "capability");
  RequireKeys(node, {"body_extent_m", "nominal_body_height_m", "body_height_m", "platform_mass_kg",
                     "nominal_payload_kg", "maximum_payload_kg", "maximum_forward_speed_mps",
                     "maximum_reverse_speed_mps", "maximum_lateral_speed_mps", "maximum_yaw_rate_radps",
                     "maximum_linear_acceleration_mps2", "maximum_yaw_acceleration_radps2",
                     "maximum_slope_rad", "maximum_step_height_m", "maximum_gap_width_m",
                     "minimum_body_clearance_m", "step_vertical_rate_mps", "unknown_is_traversable",
                     "motion_primitives"});
  const Vec3 body_extent = Vec3Value(node, "body_extent_m");
  if (body_extent.x <= 0.0 || body_extent.y <= 0.0 || body_extent.z <= 0.0) Invalid("body_extent_m: dimensions must be > 0");
  const Interval body_height = IntervalValue(node, "body_height_m");
  if (body_height.lower < 0.0) Invalid("body_height_range_m: lower bound must be >= 0");
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
    const std::optional<double> start_blind_zone_margin_m{
        root["start_blind_zone_margin_m"]
            ? NonNegative(root, "start_blind_zone_margin_m")
            : 0.2};
    return {.capability = std::move(capability),
            .start_blind_zone_margin_m = start_blind_zone_margin_m,
            .reason_code = {}};
  } catch (const std::exception& error) {
    return {.capability = std::nullopt,
            .start_blind_zone_margin_m = std::nullopt,
            .reason_code = std::string{kPlannerError},
            .error_detail = path.string() + ": " + error.what()};
  }
}

}  // namespace lunar::incremental_navigation_ros
