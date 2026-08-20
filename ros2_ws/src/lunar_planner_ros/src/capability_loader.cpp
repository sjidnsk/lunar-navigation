#include "lunar_planner_ros/capability_loader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <numbers>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <urdf/model.h>
#include <urdf_model/link.h>
#include <yaml-cpp/yaml.h>

namespace lunar::planning::ros {
namespace {

constexpr std::string_view kPlatformSchema =
    "platform-control-capability-source/v2";
constexpr double kProjectMaximumSlopeRad = std::numbers::pi / 6.0;
const std::set<std::string, std::less<>> kAllowedSourceTypes{
    "upstream_model",
    "upstream_config",
    "user_spec_material",
    "user_confirmed_upgrade",
    "user_provided_dimension",
    "user_approximate_measurement",
    "user_confirmed_capability",
    "user_approved_planning_policy",
    "derived",
    "planning_policy",
    "project_engineering_baseline",
    "planning_safety_baseline",
    "unknown",
};
const std::set<std::string, std::less<>> kRequiredWheeledSources{
    "allow_unsupported_gap",
    "body_extent_m",
    "footprint_xy_m",
    "maximum_acceleration_mps2",
    "maximum_braking_deceleration_mps2",
    "maximum_curvature_per_m",
    "maximum_forward_speed_mps",
    "maximum_lateral_acceleration_mps2",
    "maximum_local_obstacle_relief_m",
    "maximum_reverse_speed_mps",
    "maximum_spin_rate_radps",
    "maximum_surface_slope_rad",
    "maximum_yaw_acceleration_radps2",
    "minimum_clearance_m",
    "minimum_underbody_clearance_m",
    "motion_primitives",
    "reference_point",
    "roughness_handling",
    "track_width_m",
    "wheel_center_xy_m",
    "wheel_count",
    "wheel_diameter_m",
    "wheel_width_m",
    "wheelbase_m",
};

class LoadFailure final : public std::runtime_error {
 public:
  LoadFailure(
      const CapabilityLoadErrorCode code,
      std::string reason_code,
      std::string detail)
      : std::runtime_error{detail},
        code(code),
        reason_code(std::move(reason_code)),
        detail(std::move(detail)) {}

  CapabilityLoadErrorCode code;
  std::string reason_code;
  std::string detail;
};

[[noreturn]] void SchemaFailure(const std::string& detail) {
  throw LoadFailure{
      CapabilityLoadErrorCode::kSchemaInvalid,
      "CAPABILITY_SCHEMA_INVALID",
      detail,
  };
}

[[noreturn]] void SchemaCompatibilityFailure(const std::string& detail) {
  throw LoadFailure{
      CapabilityLoadErrorCode::kSchemaInvalid,
      "CAPABILITY_SCHEMA_VERSION_INCOMPATIBLE",
      detail,
  };
}

[[noreturn]] void ValueFailure(const std::string& detail) {
  throw LoadFailure{
      CapabilityLoadErrorCode::kValueInvalid,
      "CAPABILITY_VALUE_INVALID",
      detail,
  };
}

[[nodiscard]] CapabilityLoadResult Failure(const LoadFailure& failure) {
  return CapabilityLoadResult{
      .capabilities = std::nullopt,
      .error = CapabilityLoadError{
          .code = failure.code,
          .reason_code = failure.reason_code,
          .detail = failure.detail,
      },
  };
}

[[nodiscard]] bool IsSafeRelativePath(
    const std::filesystem::path& path) noexcept {
  if (path.empty() || path.is_absolute()) {
    return false;
  }
  for (const auto& component : path) {
    if (component == "..") {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool IsWithin(
    const std::filesystem::path& root,
    const std::filesystem::path& candidate) {
  const auto relative = candidate.lexically_relative(root);
  return !relative.empty() && !relative.is_absolute() &&
      *relative.begin() != "..";
}

[[nodiscard]] std::filesystem::path ResolveFile(
    const std::filesystem::path& root,
    const std::filesystem::path& relative,
    const CapabilityLoadErrorCode missing_code,
    const std::string& missing_reason) {
  if (!IsSafeRelativePath(relative)) {
    throw LoadFailure{
        CapabilityLoadErrorCode::kUnsafePath,
        "CAPABILITY_PATH_UNSAFE",
        relative.string(),
    };
  }
  const auto canonical_root = std::filesystem::weakly_canonical(root);
  const auto candidate = std::filesystem::weakly_canonical(root / relative);
  if (!IsWithin(canonical_root, candidate)) {
    throw LoadFailure{
        CapabilityLoadErrorCode::kUnsafePath,
        "CAPABILITY_PATH_UNSAFE",
        relative.string(),
    };
  }
  if (!std::filesystem::is_regular_file(candidate)) {
    throw LoadFailure{missing_code, missing_reason, candidate.string()};
  }
  return candidate;
}

[[nodiscard]] YAML::Node RequireMap(
    const YAML::Node& parent,
    const std::string& key) {
  const YAML::Node node = parent[key];
  if (!node || !node.IsMap()) {
    SchemaFailure("missing map: " + key);
  }
  return node;
}

[[nodiscard]] YAML::Node RequireSequence(
    const YAML::Node& parent,
    const std::string& key,
    const std::size_t minimum_size = 0U) {
  const YAML::Node node = parent[key];
  if (!node || !node.IsSequence() || node.size() < minimum_size) {
    SchemaFailure("missing or short sequence: " + key);
  }
  return node;
}

[[nodiscard]] std::string RequireString(
    const YAML::Node& parent,
    const std::string& key) {
  const YAML::Node node = parent[key];
  if (!node || !node.IsScalar()) {
    SchemaFailure("missing string: " + key);
  }
  try {
    std::string value = node.as<std::string>();
    if (value.empty()) {
      SchemaFailure("empty string: " + key);
    }
    return value;
  } catch (const YAML::Exception&) {
    SchemaFailure("invalid string: " + key);
  }
}

[[nodiscard]] double RequireDouble(
    const YAML::Node& parent,
    const std::string& key) {
  const YAML::Node node = parent[key];
  if (!node || !node.IsScalar()) {
    SchemaFailure("missing number: " + key);
  }
  try {
    return node.as<double>();
  } catch (const YAML::Exception&) {
    SchemaFailure("invalid number: " + key);
  }
}

[[nodiscard]] std::size_t RequirePositiveSize(
    const YAML::Node& parent,
    const std::string& key) {
  const YAML::Node node = parent[key];
  if (!node || !node.IsScalar()) {
    SchemaFailure("missing positive size: " + key);
  }
  try {
    const auto value = node.as<std::size_t>();
    if (value == 0U) {
      ValueFailure("expected positive size: " + key);
    }
    return value;
  } catch (const YAML::Exception&) {
    SchemaFailure("invalid positive size: " + key);
  }
}

[[nodiscard]] bool RequireBool(
    const YAML::Node& parent,
    const std::string& key) {
  const YAML::Node node = parent[key];
  if (!node || !node.IsScalar()) {
    SchemaFailure("missing bool: " + key);
  }
  try {
    return node.as<bool>();
  } catch (const YAML::Exception&) {
    SchemaFailure("invalid bool: " + key);
  }
}

void RejectUnexpectedKeys(
    const YAML::Node& node,
    const std::set<std::string, std::less<>>& allowed,
    const std::string& section) {
  if (!node || !node.IsMap()) {
    SchemaFailure("expected map: " + section);
  }
  for (const auto& entry : node) {
    std::string key;
    try {
      key = entry.first.as<std::string>();
    } catch (const YAML::Exception&) {
      SchemaFailure("non-string key in: " + section);
    }
    if (!allowed.contains(key)) {
      SchemaCompatibilityFailure(
          "retired or unknown " + section + " field: " + key);
    }
  }
}

[[nodiscard]] double Finite(
    const double value,
    const std::string& field) {
  if (!std::isfinite(value)) {
    ValueFailure("non-finite value: " + field);
  }
  return value;
}

[[nodiscard]] double Positive(
    const double value,
    const std::string& field) {
  if (!std::isfinite(value) || value <= 0.0) {
    ValueFailure("expected positive value: " + field);
  }
  return value;
}

[[nodiscard]] double NonNegative(
    const double value,
    const std::string& field) {
  if (!std::isfinite(value) || value < 0.0) {
    ValueFailure("expected non-negative value: " + field);
  }
  return value;
}

[[nodiscard]] double UnitInterval(
    const double value,
    const std::string& field) {
  if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
    ValueFailure("expected unit interval value: " + field);
  }
  return value;
}

[[nodiscard]] double Slope(
    const double value,
    const std::string& field) {
  if (!std::isfinite(value) || value <= 0.0 ||
      value >= std::numbers::pi / 2.0) {
    ValueFailure("invalid slope: " + field);
  }
  return std::min(value, kProjectMaximumSlopeRad);
}

[[nodiscard]] lunar::planning::Vec2 Vec2(
    const YAML::Node& node,
    const std::string& field) {
  if (!node.IsSequence() || node.size() != 2U) {
    SchemaFailure("expected vec2: " + field);
  }
  try {
    return lunar::planning::Vec2{
        .x = Finite(node[0].as<double>(), field + "[0]"),
        .y = Finite(node[1].as<double>(), field + "[1]"),
    };
  } catch (const YAML::Exception&) {
    SchemaFailure("invalid vec2: " + field);
  }
}

[[nodiscard]] bool IsStrictlyConvex(
    const std::vector<lunar::planning::Vec2>& polygon) noexcept {
  if (polygon.size() < 3U) {
    return false;
  }
  double orientation = 0.0;
  for (std::size_t index = 0U; index < polygon.size(); ++index) {
    const auto& first = polygon[index];
    const auto& second = polygon[(index + 1U) % polygon.size()];
    const auto& third = polygon[(index + 2U) % polygon.size()];
    const double cross = (second.x - first.x) * (third.y - second.y) -
        (second.y - first.y) * (third.x - second.x);
    if (!std::isfinite(cross) || std::abs(cross) <= 1.0e-12) {
      return false;
    }
    if (orientation == 0.0) {
      orientation = cross;
    } else if ((orientation > 0.0) != (cross > 0.0)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] lunar::planning::Vec3 Vec3(
    const YAML::Node& node,
    const std::string& field) {
  if (!node.IsSequence() || node.size() != 3U) {
    SchemaFailure("expected vec3: " + field);
  }
  try {
    return lunar::planning::Vec3{
        .x = Finite(node[0].as<double>(), field + "[0]"),
        .y = Finite(node[1].as<double>(), field + "[1]"),
        .z = Finite(node[2].as<double>(), field + "[2]"),
    };
  } catch (const YAML::Exception&) {
    SchemaFailure("invalid vec3: " + field);
  }
}

[[nodiscard]] lunar::planning::Quaternion Quaternion(
    const YAML::Node& node,
    const std::string& field) {
  if (!node.IsSequence() || node.size() != 4U) {
    SchemaFailure("expected quaternion wxyz: " + field);
  }
  lunar::planning::Quaternion value;
  try {
    value = lunar::planning::Quaternion{
        .w = Finite(node[0].as<double>(), field + "[0]"),
        .x = Finite(node[1].as<double>(), field + "[1]"),
        .y = Finite(node[2].as<double>(), field + "[2]"),
        .z = Finite(node[3].as<double>(), field + "[3]"),
    };
  } catch (const YAML::Exception&) {
    SchemaFailure("invalid quaternion: " + field);
  }
  const double norm = std::sqrt(
      value.w * value.w + value.x * value.x +
      value.y * value.y + value.z * value.z);
  if (!std::isfinite(norm) || std::abs(norm - 1.0) > 1.0e-3) {
    ValueFailure("quaternion is not unit length: " + field);
  }
  value.w /= norm;
  value.x /= norm;
  value.y /= norm;
  value.z /= norm;
  return value;
}

[[nodiscard]] lunar::planning::Interval Interval(
    const YAML::Node& node,
    const std::string& field,
    const bool require_zero = false) {
  if (!node.IsSequence() || node.size() != 2U) {
    SchemaFailure("expected interval: " + field);
  }
  lunar::planning::Interval interval;
  try {
    interval = lunar::planning::Interval{
        .lower = Finite(node[0].as<double>(), field + "[0]"),
        .upper = Finite(node[1].as<double>(), field + "[1]"),
    };
  } catch (const YAML::Exception&) {
    SchemaFailure("invalid interval: " + field);
  }
  if (interval.lower > interval.upper ||
      (require_zero &&
       (interval.lower >= interval.upper || interval.lower > 0.0 ||
        interval.upper < 0.0))) {
    ValueFailure("invalid interval bounds: " + field);
  }
  return interval;
}

[[nodiscard]] std::chrono::nanoseconds Duration(
    const double seconds,
    const std::string& field,
    const bool allow_zero = false) {
  if (!std::isfinite(seconds) || seconds < 0.0 ||
      (!allow_zero && seconds == 0.0)) {
    ValueFailure("invalid duration: " + field);
  }
  const long double nanoseconds =
      static_cast<long double>(seconds) * 1'000'000'000.0L;
  if (nanoseconds >
      static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
    ValueFailure("duration exceeds representation: " + field);
  }
  return std::chrono::nanoseconds{
      static_cast<std::int64_t>(std::llround(nanoseconds))};
}

[[nodiscard]] lunar::planning::Pose3 Pose(
    const YAML::Node& node,
    const std::string& field) {
  if (!node || !node.IsMap()) {
    SchemaFailure("expected pose: " + field);
  }
  return lunar::planning::Pose3{
      .position_m = Vec3(
          RequireSequence(node, "position_m"), field + ".position_m"),
      .orientation = Quaternion(
          RequireSequence(node, "orientation_wxyz"),
          field + ".orientation_wxyz"),
  };
}

[[nodiscard]] lunar::planning::WheelPrimitiveKind WheelKind(
    const std::string& value) {
  static const std::map<std::string, lunar::planning::WheelPrimitiveKind>
      kinds{
          {"FORWARD", lunar::planning::WheelPrimitiveKind::kForward},
          {"REVERSE", lunar::planning::WheelPrimitiveKind::kReverse},
          {"FORWARD_ARC", lunar::planning::WheelPrimitiveKind::kForwardArc},
          {"REVERSE_ARC", lunar::planning::WheelPrimitiveKind::kReverseArc},
          {"SPIN_CLOCKWISE", lunar::planning::WheelPrimitiveKind::kSpinClockwise},
          {"SPIN_COUNTERCLOCKWISE", lunar::planning::WheelPrimitiveKind::kSpinCounterclockwise},
          {"STOP_AND_SWITCH", lunar::planning::WheelPrimitiveKind::kStopAndSwitch},
      };
  const auto found = kinds.find(value);
  if (found == kinds.end()) {
    ValueFailure("unknown wheel primitive kind: " + value);
  }
  return found->second;
}

[[nodiscard]] lunar::planning::LeggedPrimitiveKind LeggedKind(
    const std::string& value) {
  static const std::map<std::string, lunar::planning::LeggedPrimitiveKind>
      kinds{
          {"FORWARD", lunar::planning::LeggedPrimitiveKind::kForward},
          {"BACKWARD", lunar::planning::LeggedPrimitiveKind::kBackward},
          {"LATERAL_LEFT", lunar::planning::LeggedPrimitiveKind::kLateralLeft},
          {"LATERAL_RIGHT", lunar::planning::LeggedPrimitiveKind::kLateralRight},
          {"SPIN", lunar::planning::LeggedPrimitiveKind::kSpin},
          {"COUPLED", lunar::planning::LeggedPrimitiveKind::kCoupled},
      };
  const auto found = kinds.find(value);
  if (found == kinds.end()) {
    ValueFailure("unknown legged primitive kind: " + value);
  }
  return found->second;
}

void RecordPrimitiveId(
    const std::string& id,
    std::set<std::string, std::less<>>& seen,
    std::vector<std::string>& ordered) {
  if (!seen.insert(id).second) {
    ValueFailure("duplicate motion primitive id: " + id);
  }
  ordered.push_back(id);
}

[[nodiscard]] ParametricWheeledGeometry ParseParametricWheelGeometry(
    const YAML::Node& wheeled,
    const lunar::planning::Vec3& body_extent,
    const double wheel_diameter,
    const double wheel_width) {
  const std::size_t wheel_count = RequirePositiveSize(wheeled, "wheel_count");
  if (wheel_count != 4U) {
    ValueFailure("parametric wheeled geometry requires four wheels");
  }

  const YAML::Node center_nodes = RequireSequence(
      wheeled, "wheel_center_xy_m", wheel_count);
  if (center_nodes.size() != wheel_count) {
    ValueFailure("wheel_center_xy_m count must match wheel_count");
  }
  std::vector<lunar::planning::Vec2> centers;
  std::set<std::pair<double, double>> unique_centers;
  centers.reserve(wheel_count);
  for (std::size_t index = 0U; index < center_nodes.size(); ++index) {
    const auto center = Vec2(
        center_nodes[index],
        "wheeled.wheel_center_xy_m[" + std::to_string(index) + "]");
    if (!unique_centers.emplace(center.x, center.y).second) {
      ValueFailure("duplicate wheeled wheel center");
    }
    if (std::abs(center.x) + wheel_diameter / 2.0 >
            body_extent.x / 2.0 + 1.0e-9 ||
        std::abs(center.y) + wheel_width / 2.0 >
            body_extent.y / 2.0 + 1.0e-9) {
      ValueFailure("wheel center exceeds body envelope");
    }
    centers.push_back(center);
  }

  const YAML::Node footprint_nodes = RequireSequence(
      wheeled, "footprint_xy_m", 3U);
  std::vector<lunar::planning::Vec2> footprint;
  std::set<std::pair<double, double>> unique_vertices;
  footprint.reserve(footprint_nodes.size());
  for (std::size_t index = 0U; index < footprint_nodes.size(); ++index) {
    const auto vertex = Vec2(
        footprint_nodes[index],
        "wheeled.footprint_xy_m[" + std::to_string(index) + "]");
    if (!unique_vertices.emplace(vertex.x, vertex.y).second) {
      ValueFailure("duplicate wheeled footprint vertex");
    }
    footprint.push_back(vertex);
  }
  if (!IsStrictlyConvex(footprint)) {
    ValueFailure("parametric wheeled footprint must be strictly convex");
  }
  return ParametricWheeledGeometry{
      .wheel_count = wheel_count,
      .wheel_center_xy_m = std::move(centers),
  };
}

void ValidateRequiredSources(
    const LoadedCapabilities& loaded,
    const std::set<std::string, std::less<>>& required) {
  if (loaded.field_source_types.size() != required.size()) {
    SchemaFailure("sources do not match required fields");
  }
  for (const auto& field : required) {
    if (!loaded.field_source_types.contains(field)) {
      SchemaFailure("missing required source: " + field);
    }
  }
}

[[nodiscard]] lunar::planning::WheeledCapability ParseWheeled(
    const YAML::Node& root,
    LoadedCapabilities& loaded) {
  const YAML::Node node = RequireMap(root, "wheeled");
  RejectUnexpectedKeys(
      node,
      {"footprint_xy_m",
       "body_extent_m",
       "reference_point",
       "wheel_count",
       "wheel_diameter_m",
       "wheel_width_m",
       "wheelbase_m",
       "track_width_m",
       "wheel_center_xy_m",
       "minimum_underbody_clearance_m",
       "maximum_local_obstacle_relief_m",
       "allow_unsupported_gap",
       "maximum_forward_speed_mps",
       "maximum_reverse_speed_mps",
       "maximum_spin_rate_radps",
       "maximum_acceleration_mps2",
       "maximum_braking_deceleration_mps2",
       "maximum_yaw_acceleration_radps2",
       "maximum_lateral_acceleration_mps2",
       "maximum_curvature_per_m",
       "maximum_slope_rad",
       "minimum_clearance_m",
       "roughness_handling",
       "motion_primitives"},
      "wheeled");
  loaded.reference_point = RequireString(node, "reference_point");
  if (loaded.reference_point != "base_footprint") {
    ValueFailure("wheeled.reference_point must be base_footprint");
  }
  const YAML::Node footprint_node =
      RequireSequence(node, "footprint_xy_m", 3U);
  std::vector<lunar::planning::Vec2> footprint;
  std::set<std::pair<double, double>> vertices;
  for (std::size_t index = 0U; index < footprint_node.size(); ++index) {
    const auto vertex = Vec2(
        footprint_node[index],
        "wheeled.footprint_xy_m[" + std::to_string(index) + "]");
    if (!vertices.emplace(vertex.x, vertex.y).second) {
      ValueFailure("duplicate wheeled footprint vertex");
    }
    footprint.push_back(vertex);
  }

  const auto body_extent = Vec3(
      RequireSequence(node, "body_extent_m"), "body_extent_m");
  if (body_extent.x <= 0.0 || body_extent.y <= 0.0 || body_extent.z <= 0.0) {
    ValueFailure("body_extent_m must be positive");
  }
  const double wheel_diameter = Positive(
      RequireDouble(node, "wheel_diameter_m"), "wheel_diameter_m");
  const double wheel_width = Positive(
      RequireDouble(node, "wheel_width_m"), "wheel_width_m");
  const double wheelbase = Positive(
      RequireDouble(node, "wheelbase_m"), "wheelbase_m");
  const double track_width = Positive(
      RequireDouble(node, "track_width_m"), "track_width_m");
  if (wheel_width >= body_extent.y || wheelbase >= body_extent.x ||
      (loaded.geometry_source_kind == GeometrySourceKind::kUrdfMesh &&
       std::abs(track_width - (body_extent.y - wheel_width)) > 1.0e-6)) {
    ValueFailure("wheeled wheel geometry is inconsistent with body extent");
  }
  if (loaded.geometry_source_kind == GeometrySourceKind::kParametricEnvelope) {
    loaded.parametric_wheeled_geometry = ParseParametricWheelGeometry(
        node, body_extent, wheel_diameter, wheel_width);
  }
  const double underbody_clearance = Positive(
      RequireDouble(node, "minimum_underbody_clearance_m"),
      "minimum_underbody_clearance_m");
  const double local_relief = NonNegative(
      RequireDouble(node, "maximum_local_obstacle_relief_m"),
      "maximum_local_obstacle_relief_m");
  const bool allow_unsupported_gap =
      RequireBool(node, "allow_unsupported_gap");
  const double maximum_forward_speed = Positive(
      RequireDouble(node, "maximum_forward_speed_mps"),
      "maximum_forward_speed_mps");
  const std::string roughness_handling =
      RequireString(node, "roughness_handling");
  if (roughness_handling != "COST_SPEED_AND_LOCAL_RECHECK") {
    ValueFailure("unsupported wheeled roughness_handling");
  }

  std::vector<lunar::planning::WheelMotionPrimitive> primitives;
  const YAML::Node primitive_nodes =
      RequireSequence(node, "motion_primitives", 1U);
  std::set<std::string, std::less<>> primitive_ids;
  for (std::size_t index = 0U; index < primitive_nodes.size(); ++index) {
    const YAML::Node primitive = primitive_nodes[index];
    if (!primitive.IsMap()) {
      SchemaFailure("wheel motion primitive must be a map");
    }
    const std::string id = RequireString(primitive, "primitive_id");
    RecordPrimitiveId(id, primitive_ids, loaded.source_motion_primitive_ids);
    primitives.push_back(lunar::planning::WheelMotionPrimitive{
        .primitive_id = id,
        .kind = WheelKind(RequireString(primitive, "kind")),
        .relative_end_pose =
            Pose(RequireMap(primitive, "relative_end_pose"),
                 "motion_primitives.relative_end_pose"),
    });
  }

  return lunar::planning::WheeledCapability{
      .footprint_xy_m = std::move(footprint),
      .body_extent_m = body_extent,
      .wheel_diameter_m = wheel_diameter,
      .wheel_width_m = wheel_width,
      .wheelbase_m = wheelbase,
      .track_width_m = track_width,
      .minimum_underbody_clearance_m = underbody_clearance,
      .maximum_local_obstacle_relief_m = local_relief,
      .allow_unsupported_gap = allow_unsupported_gap,
      .minimum_body_z_m = 0.0,
      .maximum_body_z_m = body_extent.z,
      .maximum_forward_speed_mps = maximum_forward_speed,
      .maximum_reverse_speed_mps = NonNegative(
          RequireDouble(node, "maximum_reverse_speed_mps"),
          "maximum_reverse_speed_mps"),
      .maximum_spin_rate_radps = Positive(
          RequireDouble(node, "maximum_spin_rate_radps"),
          "maximum_spin_rate_radps"),
      .maximum_acceleration_mps2 = Positive(
          RequireDouble(node, "maximum_acceleration_mps2"),
          "maximum_acceleration_mps2"),
      .maximum_braking_deceleration_mps2 = Positive(
          RequireDouble(node, "maximum_braking_deceleration_mps2"),
          "maximum_braking_deceleration_mps2"),
      .maximum_yaw_acceleration_radps2 = Positive(
          RequireDouble(node, "maximum_yaw_acceleration_radps2"),
          "maximum_yaw_acceleration_radps2"),
      .maximum_lateral_acceleration_mps2 = Positive(
          RequireDouble(node, "maximum_lateral_acceleration_mps2"),
          "maximum_lateral_acceleration_mps2"),
      .maximum_curvature_per_m = Positive(
          RequireDouble(node, "maximum_curvature_per_m"),
          "maximum_curvature_per_m"),
      .maximum_slope_rad = Slope(
          RequireDouble(node, "maximum_slope_rad"),
          "maximum_slope_rad"),
      .minimum_clearance_m = NonNegative(
          RequireDouble(node, "minimum_clearance_m"),
          "minimum_clearance_m"),
      .motion_primitives = std::move(primitives),
  };
}

[[nodiscard]] lunar::planning::LeggedCapability ParseLegged(
    const YAML::Node& root,
    LoadedCapabilities& loaded) {
  const YAML::Node node = RequireMap(root, "legged");
  RejectUnexpectedKeys(
      node,
      {"reference_point",
       "body_extent_m",
       "platform_mass_kg",
       "maximum_payload_kg",
       "maximum_slope_rad",
       "maximum_step_height_m",
       "maximum_gap_width_m",
       "minimum_body_clearance_m",
       "step_vertical_rate_mps",
       "body_height_m",
       "forward_speed_mps",
       "lateral_speed_mps",
       "yaw_rate_radps",
       "maximum_linear_acceleration_mps2",
       "maximum_yaw_acceleration_radps2",
       "roughness_handling",
       "motion_primitives"},
      "legged");
  loaded.reference_point = RequireString(node, "reference_point");
  if (loaded.reference_point != "base_link") {
    ValueFailure("legged.reference_point must be base_link");
  }
  const auto body_extent = Vec3(
      RequireSequence(node, "body_extent_m"), "body_extent_m");
  if (body_extent.x <= 0.0 || body_extent.y <= 0.0 || body_extent.z <= 0.0) {
    ValueFailure("body_extent_m must be positive");
  }
  const double platform_mass = Positive(
      RequireDouble(node, "platform_mass_kg"), "platform_mass_kg");
  const double maximum_payload = Positive(
      RequireDouble(node, "maximum_payload_kg"), "maximum_payload_kg");
  const double step_vertical_rate = Positive(
      RequireDouble(node, "step_vertical_rate_mps"),
      "step_vertical_rate_mps");
  if (RequireString(node, "roughness_handling") != "DIAGNOSTIC_ONLY") {
    ValueFailure("unsupported legged roughness_handling");
  }

  std::vector<lunar::planning::LeggedBodyPrimitive> primitives;
  const YAML::Node primitive_nodes =
      RequireSequence(node, "motion_primitives", 1U);
  std::set<std::string, std::less<>> primitive_ids;
  for (std::size_t index = 0U; index < primitive_nodes.size(); ++index) {
    const YAML::Node primitive = primitive_nodes[index];
    if (!primitive.IsMap()) {
      SchemaFailure("legged motion primitive must be a map");
    }
    const std::string id = RequireString(primitive, "primitive_id");
    RecordPrimitiveId(id, primitive_ids, loaded.source_motion_primitive_ids);
    primitives.push_back(lunar::planning::LeggedBodyPrimitive{
        .primitive_id = id,
        .kind = LeggedKind(RequireString(primitive, "kind")),
        .body_frame_displacement_m = Vec3(
            RequireSequence(primitive, "body_frame_displacement_m"),
            "motion_primitives.body_frame_displacement_m"),
        .yaw_change_rad = Finite(
            RequireDouble(primitive, "yaw_change_rad"),
            "motion_primitives.yaw_change_rad"),
    });
  }

  const auto body_height = Interval(
      RequireSequence(node, "body_height_m"), "body_height_m");
  if (body_height.lower < 0.0) {
    ValueFailure("body_height_m must be non-negative");
  }
  return lunar::planning::LeggedCapability{
      .body_extent_m = body_extent,
      .platform_mass_kg = platform_mass,
      .maximum_payload_kg = maximum_payload,
      .maximum_slope_rad = Slope(
          RequireDouble(node, "maximum_slope_rad"), "maximum_slope_rad"),
      .maximum_step_height_m = NonNegative(
          RequireDouble(node, "maximum_step_height_m"),
          "maximum_step_height_m"),
      .maximum_gap_width_m = NonNegative(
          RequireDouble(node, "maximum_gap_width_m"),
          "maximum_gap_width_m"),
      .minimum_body_clearance_m = NonNegative(
          RequireDouble(node, "minimum_body_clearance_m"),
          "minimum_body_clearance_m"),
      .step_vertical_rate_mps = step_vertical_rate,
      .body_height_m = body_height,
      .forward_speed_mps = Interval(
          RequireSequence(node, "forward_speed_mps"),
          "forward_speed_mps", true),
      .lateral_speed_mps = Interval(
          RequireSequence(node, "lateral_speed_mps"),
          "lateral_speed_mps", true),
      .yaw_rate_radps = Interval(
          RequireSequence(node, "yaw_rate_radps"),
          "yaw_rate_radps", true),
      .maximum_linear_acceleration_mps2 = Positive(
          RequireDouble(node, "maximum_linear_acceleration_mps2"),
          "maximum_linear_acceleration_mps2"),
      .maximum_yaw_acceleration_radps2 = Positive(
          RequireDouble(node, "maximum_yaw_acceleration_radps2"),
          "maximum_yaw_acceleration_radps2"),
      .motion_primitives = std::move(primitives),
  };
}

[[nodiscard]] lunar::planning::HopperCapability ParseHopper(
    const YAML::Node& root,
    LoadedCapabilities& loaded) {
  static_cast<void>(loaded);
  const YAML::Node node = RequireMap(root, "hopper");
  RejectUnexpectedKeys(
      node,
      {"specific_impulse_s",
       "reference_total_mass_kg",
       "reference_propellant_mass_kg",
       "landing_support_radius_m",
       "flight_collision_radius_m",
       "maximum_landing_slope_rad",
       "maximum_landing_plane_residual_m",
       "landing_lateral_margin_m",
       "flight_map_margin_m",
       "reachability_delta_v_margin_ratio",
       "standard_gravity_mps2"},
      "hopper");
  const double specific_impulse = Positive(
      RequireDouble(node, "specific_impulse_s"), "specific_impulse_s");
  const double reference_total_mass = Positive(
      RequireDouble(node, "reference_total_mass_kg"),
      "reference_total_mass_kg");
  const double reference_propellant_mass = Positive(
      RequireDouble(node, "reference_propellant_mass_kg"),
      "reference_propellant_mass_kg");
  if (reference_propellant_mass >= reference_total_mass) {
    ValueFailure(
        "reference_propellant_mass_kg must be less than "
        "reference_total_mass_kg");
  }
  const double landing_support_radius = Positive(
      RequireDouble(node, "landing_support_radius_m"),
      "landing_support_radius_m");
  const double flight_collision_radius = Positive(
      RequireDouble(node, "flight_collision_radius_m"),
      "flight_collision_radius_m");
  const double landing_plane_residual = NonNegative(
      RequireDouble(node, "maximum_landing_plane_residual_m"),
      "maximum_landing_plane_residual_m");
  const double landing_lateral_margin = NonNegative(
      RequireDouble(node, "landing_lateral_margin_m"),
      "landing_lateral_margin_m");
  const double flight_map_margin = NonNegative(
      RequireDouble(node, "flight_map_margin_m"), "flight_map_margin_m");
  const double delta_v_margin = NonNegative(
      RequireDouble(node, "reachability_delta_v_margin_ratio"),
      "reachability_delta_v_margin_ratio");
  const double standard_gravity = Positive(
      RequireDouble(node, "standard_gravity_mps2"),
      "standard_gravity_mps2");

  return lunar::planning::HopperCapability{
      .specific_impulse_s = specific_impulse,
      .reference_total_mass_kg = reference_total_mass,
      .reference_propellant_mass_kg = reference_propellant_mass,
      .landing_support_radius_m = landing_support_radius,
      .flight_collision_radius_m = flight_collision_radius,
      .maximum_landing_plane_residual_m = landing_plane_residual,
      .landing_lateral_margin_m = landing_lateral_margin,
      .flight_map_margin_m = flight_map_margin,
      .reachability_delta_v_margin_ratio = delta_v_margin,
      .standard_gravity_mps2 = standard_gravity,
      .maximum_landing_slope_rad = Slope(
          RequireDouble(node, "maximum_landing_slope_rad"),
          "maximum_landing_slope_rad"),
  };
}

[[nodiscard]] std::filesystem::path ResolveMeshPath(
    const std::filesystem::path& share,
    const std::filesystem::path& urdf_path,
    const std::string& filename) {
  constexpr std::string_view kPackagePrefix = "package://";
  if (filename.starts_with(kPackagePrefix)) {
    const std::string remainder = filename.substr(kPackagePrefix.size());
    const std::size_t separator = remainder.find('/');
    if (separator == std::string::npos || separator == 0U) {
      throw LoadFailure{
          CapabilityLoadErrorCode::kUnsafePath,
          "MESH_URI_INVALID",
          filename,
      };
    }
    const std::string package_name = remainder.substr(0U, separator);
    const std::filesystem::path relative = remainder.substr(separator + 1U);
    try {
      const auto package_share =
          ament_index_cpp::get_package_share_directory(package_name);
      return ResolveFile(
          package_share, relative,
          CapabilityLoadErrorCode::kMeshMissing,
          "CAPABILITY_MESH_MISSING");
    } catch (const LoadFailure&) {
      throw;
    } catch (const std::exception&) {
      throw LoadFailure{
          CapabilityLoadErrorCode::kMeshMissing,
          "CAPABILITY_MESH_PACKAGE_MISSING",
          filename,
      };
    }
  }
  const std::filesystem::path mesh_path{filename};
  if (!IsSafeRelativePath(mesh_path)) {
    throw LoadFailure{
        CapabilityLoadErrorCode::kUnsafePath,
        "MESH_PATH_UNSAFE",
        filename,
    };
  }
  const auto relative =
      std::filesystem::relative(urdf_path.parent_path() / mesh_path, share);
  return ResolveFile(
      share, relative,
      CapabilityLoadErrorCode::kMeshMissing,
      "CAPABILITY_MESH_MISSING");
}

void AddMesh(
    const urdf::GeometrySharedPtr& geometry,
    const std::filesystem::path& share,
    const std::filesystem::path& urdf_path,
    std::set<std::filesystem::path>& mesh_paths) {
  if (geometry == nullptr || geometry->type != urdf::Geometry::MESH) {
    return;
  }
  const auto mesh = std::static_pointer_cast<urdf::Mesh>(geometry);
  if (mesh->filename.empty() || !std::isfinite(mesh->scale.x) ||
      !std::isfinite(mesh->scale.y) || !std::isfinite(mesh->scale.z) ||
      mesh->scale.x <= 0.0 || mesh->scale.y <= 0.0 || mesh->scale.z <= 0.0) {
    throw LoadFailure{
        CapabilityLoadErrorCode::kUrdfInvalid,
        "CAPABILITY_URDF_MESH_INVALID",
        urdf_path.string(),
    };
  }
  mesh_paths.insert(ResolveMeshPath(
      share, urdf_path, mesh->filename));
}

[[nodiscard]] std::vector<std::filesystem::path> ValidateGeometry(
    const std::filesystem::path& share,
    const std::filesystem::path& urdf_path,
    const std::string& base_frame_id) {
  urdf::Model model;
  if (!model.initFile(urdf_path.string()) ||
      model.getLink(base_frame_id) == nullptr) {
    throw LoadFailure{
        CapabilityLoadErrorCode::kUrdfInvalid,
        "CAPABILITY_URDF_INVALID",
        urdf_path.string(),
    };
  }
  std::vector<urdf::LinkSharedPtr> links;
  model.getLinks(links);
  std::set<std::filesystem::path> mesh_paths;
  for (const auto& link : links) {
    for (const auto& visual : link->visual_array) {
      if (visual != nullptr) {
        AddMesh(visual->geometry, share, urdf_path, mesh_paths);
      }
    }
    for (const auto& collision : link->collision_array) {
      if (collision != nullptr) {
        AddMesh(collision->geometry, share, urdf_path, mesh_paths);
      }
    }
  }
  if (mesh_paths.empty()) {
    throw LoadFailure{
        CapabilityLoadErrorCode::kUrdfInvalid,
        "CAPABILITY_URDF_MESH_REQUIRED",
        urdf_path.string(),
    };
  }
  return {mesh_paths.begin(), mesh_paths.end()};
}

[[nodiscard]] LoadedCapabilities LoadDocuments(
    const std::filesystem::path& share,
    const std::filesystem::path& platform_path,
    const std::filesystem::path& observation_path) {
  YAML::Node platform_document;
  YAML::Node observation_document;
  try {
    platform_document = YAML::LoadFile(platform_path.string());
    observation_document = YAML::LoadFile(observation_path.string());
  } catch (const YAML::Exception& error) {
    throw LoadFailure{
        CapabilityLoadErrorCode::kParseError,
        "CAPABILITY_DOCUMENT_PARSE_ERROR",
        error.what(),
    };
  }
  if (!platform_document.IsMap() || !observation_document.IsMap()) {
    SchemaFailure("capability documents must be objects");
  }
  if (RequireString(platform_document, "schema_version") != kPlatformSchema) {
    SchemaCompatibilityFailure("unsupported platform capability schema");
  }

  LoadedCapabilities loaded;
  const YAML::Node platform = RequireMap(platform_document, "platform");
  loaded.platform_id = RequireString(platform, "platform_id");
  const std::string platform_type = RequireString(platform, "platform_type");
  loaded.capability_version = RequireString(platform, "capability_version");
  loaded.base_frame_id = RequireString(platform, "base_frame_id");
  const std::string expected_base_frame =
      platform_type == "WHEELED" ? "base_footprint" : "base_link";
  if (loaded.base_frame_id != expected_base_frame) {
    ValueFailure(
        "platform.base_frame_id must be " + expected_base_frame);
  }

  const YAML::Node sources = RequireMap(platform_document, "sources");
  for (const auto& entry : sources) {
    std::string field;
    std::string source_type;
    try {
      field = entry.first.as<std::string>();
      source_type = entry.second.as<std::string>();
    } catch (const YAML::Exception&) {
      SchemaFailure("sources must map strings to strings");
    }
    if (field.empty() || source_type.empty() ||
        !loaded.field_source_types.emplace(field, source_type).second) {
      ValueFailure("invalid or duplicate field source: " + field);
    }
  }
  if (loaded.field_source_types.empty()) {
    SchemaFailure("sources must not be empty");
  }
  for (const auto& [field, source_type] : loaded.field_source_types) {
    static_cast<void>(field);
    if (!kAllowedSourceTypes.contains(source_type)) {
      ValueFailure("unsupported field source type: " + source_type);
    }
  }

  loaded.observation.sensor_range_m = Positive(
      RequireDouble(observation_document, "sensor_range_m"),
      "sensor_range_m");
  const double fov_degrees = Positive(
      RequireDouble(observation_document, "sensor_fov_deg"),
      "sensor_fov_deg");
  if (fov_degrees > 360.0) {
    ValueFailure("sensor_fov_deg exceeds 360 degrees");
  }
  loaded.observation.sensor_fov_rad =
      fov_degrees * std::numbers::pi / 180.0;

  const YAML::Node geometry = RequireMap(platform_document, "geometry_source");
  const bool parametric = geometry["type"] &&
      RequireString(geometry, "type") == "parametric_envelope";
  if (parametric) {
    if (geometry["urdf_file"]) {
      SchemaFailure("parametric geometry must not declare urdf_file");
    }
    RejectUnexpectedKeys(geometry, {"type"}, "geometry_source");
    if (platform_type != "WHEELED") {
      ValueFailure("parametric_envelope is only approved for WHEELED");
    }
    loaded.geometry_source_kind = GeometrySourceKind::kParametricEnvelope;
  } else {
    RejectUnexpectedKeys(geometry, {"urdf_file"}, "geometry_source");
    loaded.geometry_source_kind = GeometrySourceKind::kUrdfMesh;
    const std::filesystem::path urdf_relative =
        RequireString(geometry, "urdf_file");
    loaded.urdf_path = ResolveFile(
        share, urdf_relative,
        CapabilityLoadErrorCode::kFileMissing,
        "CAPABILITY_URDF_MISSING");
    loaded.mesh_paths =
        ValidateGeometry(share, loaded.urdf_path, loaded.base_frame_id);
  }

  if (platform_type == "WHEELED") {
    loaded.platform = ParseWheeled(platform_document, loaded);
    if (loaded.geometry_source_kind == GeometrySourceKind::kParametricEnvelope) {
      ValidateRequiredSources(loaded, kRequiredWheeledSources);
    }
  } else if (platform_type == "LEGGED") {
    loaded.platform = ParseLegged(platform_document, loaded);
  } else if (platform_type == "HOPPER") {
    loaded.platform = ParseHopper(platform_document, loaded);
  } else {
    ValueFailure("unknown platform.platform_type: " + platform_type);
  }
  return loaded;
}

}  // namespace

CapabilityLoadResult CapabilityLoader::LoadFromPackageShare(
    const std::string& package_name,
    const std::filesystem::path& platform_capability_file,
    const std::filesystem::path& observation_capability_file) const {
  try {
    if (package_name.empty()) {
      throw std::runtime_error{"package name is empty"};
    }
    const std::filesystem::path share =
        ament_index_cpp::get_package_share_directory(package_name);
    return LoadFromShareDirectory(
        share, platform_capability_file, observation_capability_file);
  } catch (const std::exception& error) {
    return CapabilityLoadResult{
        .capabilities = std::nullopt,
        .error = CapabilityLoadError{
            .code = CapabilityLoadErrorCode::kPackageNotFound,
            .reason_code = "CAPABILITY_PACKAGE_NOT_FOUND",
            .detail = error.what(),
        },
    };
  }
}

CapabilityLoadResult CapabilityLoader::LoadFromShareDirectory(
    const std::filesystem::path& package_share_directory,
    const std::filesystem::path& platform_capability_file,
    const std::filesystem::path& observation_capability_file) const {
  try {
    if (!std::filesystem::is_directory(package_share_directory)) {
      throw LoadFailure{
          CapabilityLoadErrorCode::kFileMissing,
          "CAPABILITY_SHARE_MISSING",
          package_share_directory.string(),
      };
    }
    const auto platform_path = ResolveFile(
        package_share_directory, platform_capability_file,
        CapabilityLoadErrorCode::kFileMissing,
        "PLATFORM_CAPABILITY_FILE_MISSING");
    const auto observation_path = ResolveFile(
        package_share_directory, observation_capability_file,
        CapabilityLoadErrorCode::kFileMissing,
        "OBSERVATION_CAPABILITY_FILE_MISSING");
    return CapabilityLoadResult{
        .capabilities = LoadDocuments(
            std::filesystem::weakly_canonical(package_share_directory),
            platform_path, observation_path),
        .error = std::nullopt,
    };
  } catch (const LoadFailure& failure) {
    return Failure(failure);
  } catch (const YAML::Exception& error) {
    return Failure(LoadFailure{
        CapabilityLoadErrorCode::kParseError,
        "CAPABILITY_DOCUMENT_PARSE_ERROR",
        error.what(),
    });
  } catch (const std::exception& error) {
    return Failure(LoadFailure{
        CapabilityLoadErrorCode::kParseError,
        "CAPABILITY_LOAD_ERROR",
        error.what(),
    });
  }
}

}  // namespace lunar::planning::ros
