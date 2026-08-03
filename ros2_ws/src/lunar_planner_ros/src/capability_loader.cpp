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
    "platform-control-capability-source/v1";
constexpr double kProjectMaximumSlopeRad = std::numbers::pi / 6.0;

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

[[nodiscard]] lunar::planning::WheeledCapability ParseWheeled(
    const YAML::Node& root,
    LoadedCapabilities& loaded) {
  const YAML::Node node = RequireMap(root, "wheeled");
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

  const double minimum_body_z =
      Finite(RequireDouble(node, "minimum_body_z_m"), "minimum_body_z_m");
  const double maximum_body_z =
      Finite(RequireDouble(node, "maximum_body_z_m"), "maximum_body_z_m");
  if (minimum_body_z > maximum_body_z) {
    ValueFailure("minimum_body_z_m exceeds maximum_body_z_m");
  }
  loaded.maximum_obstacle_height_m = NonNegative(
      RequireDouble(node, "maximum_obstacle_height_m"),
      "maximum_obstacle_height_m");

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
        .nominal_duration = Duration(
            RequireDouble(primitive, "nominal_duration_s"),
            "motion_primitives.nominal_duration_s"),
    });
  }

  return lunar::planning::WheeledCapability{
      .footprint_xy_m = std::move(footprint),
      .minimum_body_z_m = minimum_body_z,
      .maximum_body_z_m = maximum_body_z,
      .maximum_forward_speed_mps = Positive(
          RequireDouble(node, "maximum_forward_speed_mps"),
          "maximum_forward_speed_mps"),
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
  loaded.reference_point = RequireString(node, "reference_point");
  const auto body_half_extent = Vec3(
      RequireSequence(node, "body_half_extent_m"), "body_half_extent_m");
  if (body_half_extent.x <= 0.0 || body_half_extent.y <= 0.0 ||
      body_half_extent.z <= 0.0) {
    ValueFailure("body_half_extent_m must be positive");
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
        .nominal_duration = Duration(
            RequireDouble(primitive, "nominal_duration_s"),
            "motion_primitives.nominal_duration_s"),
    });
  }

  const auto body_height = Interval(
      RequireSequence(node, "body_height_m"), "body_height_m");
  if (body_height.lower < 0.0) {
    ValueFailure("body_height_m must be non-negative");
  }
  return lunar::planning::LeggedCapability{
      .body_half_extent_m = body_half_extent,
      .maximum_slope_rad = Slope(
          RequireDouble(node, "maximum_slope_rad"), "maximum_slope_rad"),
      .maximum_roughness_m = NonNegative(
          RequireDouble(node, "maximum_roughness_m"),
          "maximum_roughness_m"),
      .maximum_step_height_m = NonNegative(
          RequireDouble(node, "maximum_step_height_m"),
          "maximum_step_height_m"),
      .maximum_gap_width_m = NonNegative(
          RequireDouble(node, "maximum_gap_width_m"),
          "maximum_gap_width_m"),
      .minimum_confidence = UnitInterval(
          RequireDouble(node, "minimum_confidence"), "minimum_confidence"),
      .minimum_body_clearance_m = NonNegative(
          RequireDouble(node, "minimum_body_clearance_m"),
          "minimum_body_clearance_m"),
      .body_height_m = body_height,
      .forward_speed_mps = Interval(
          RequireSequence(node, "forward_speed_mps"),
          "forward_speed_mps", true),
      .lateral_speed_mps = Interval(
          RequireSequence(node, "lateral_speed_mps"),
          "lateral_speed_mps", true),
      .vertical_speed_mps = Interval(
          RequireSequence(node, "vertical_speed_mps"),
          "vertical_speed_mps", true),
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
  const YAML::Node node = RequireMap(root, "hopper");
  const auto body_half_extent = Vec3(
      RequireSequence(node, "body_half_extent_m"), "body_half_extent_m");
  if (body_half_extent.x <= 0.0 || body_half_extent.y <= 0.0 ||
      body_half_extent.z <= 0.0) {
    ValueFailure("body_half_extent_m must be positive");
  }
  const auto gravity =
      Vec3(RequireSequence(node, "gravity_mps2"), "gravity_mps2");
  if (gravity.z >= 0.0 ||
      std::hypot(gravity.x, gravity.y, gravity.z) <= 0.0) {
    ValueFailure("gravity_mps2 must point downward");
  }
  const auto minimum_flight = Duration(
      RequireDouble(node, "minimum_flight_time_s"),
      "minimum_flight_time_s");
  const auto maximum_flight = Duration(
      RequireDouble(node, "maximum_flight_time_s"),
      "maximum_flight_time_s");
  if (minimum_flight > maximum_flight) {
    ValueFailure("minimum_flight_time_s exceeds maximum_flight_time_s");
  }
  const YAML::Node profile =
      RequireMap(node, "actuator_or_impulse_profile");
  loaded.actuator_profile_id = RequireString(profile, "profile_id");

  const YAML::Node primitive_nodes =
      RequireSequence(node, "motion_primitives", 1U);
  std::set<std::string, std::less<>> primitive_ids;
  for (std::size_t index = 0U; index < primitive_nodes.size(); ++index) {
    if (!primitive_nodes[index].IsMap()) {
      SchemaFailure("hopper motion primitive must be a map");
    }
    const std::string id =
        RequireString(primitive_nodes[index], "primitive_id");
    RecordPrimitiveId(id, primitive_ids, loaded.source_motion_primitive_ids);
  }

  return lunar::planning::HopperCapability{
      .body_half_extent_m = body_half_extent,
      .platform_mass_kg = Positive(
          RequireDouble(node, "platform_mass_kg"), "platform_mass_kg"),
      .gravity_mps2 = gravity,
      .maximum_landing_slope_rad = Slope(
          RequireDouble(node, "maximum_landing_slope_rad"),
          "maximum_landing_slope_rad"),
      .maximum_landing_roughness_m = NonNegative(
          RequireDouble(node, "maximum_landing_roughness_m"),
          "maximum_landing_roughness_m"),
      .maximum_plane_residual_m = NonNegative(
          RequireDouble(node, "maximum_plane_residual_m"),
          "maximum_plane_residual_m"),
      .minimum_overhead_clearance_m = NonNegative(
          RequireDouble(node, "minimum_overhead_clearance_m"),
          "minimum_overhead_clearance_m"),
      .minimum_lateral_clearance_m = NonNegative(
          RequireDouble(node, "minimum_lateral_clearance_m"),
          "minimum_lateral_clearance_m"),
      .minimum_landing_region_area_m2 = Positive(
          RequireDouble(node, "minimum_landing_region_area_m2"),
          "minimum_landing_region_area_m2"),
      .maximum_launch_speed_mps = Positive(
          RequireDouble(node, "maximum_launch_speed_mps"),
          "maximum_launch_speed_mps"),
      .maximum_launch_impulse_newton_seconds = Positive(
          RequireDouble(node, "maximum_launch_impulse_newton_seconds"),
          "maximum_launch_impulse_newton_seconds"),
      .minimum_flight_time = minimum_flight,
      .maximum_flight_time = maximum_flight,
      .maximum_landing_speed_mps = Positive(
          RequireDouble(node, "maximum_landing_speed_mps"),
          "maximum_landing_speed_mps"),
      .minimum_downward_impact_speed_mps = NonNegative(
          RequireDouble(node, "minimum_downward_impact_speed_mps"),
          "minimum_downward_impact_speed_mps"),
      .minimum_landing_clearance_m = NonNegative(
          RequireDouble(node, "minimum_landing_clearance_m"),
          "minimum_landing_clearance_m"),
      .maximum_angular_speed_radps = Positive(
          RequireDouble(node, "maximum_angular_speed_radps"),
          "maximum_angular_speed_radps"),
      .maximum_angular_acceleration_radps2 = Positive(
          RequireDouble(node, "maximum_angular_acceleration_radps2"),
          "maximum_angular_acceleration_radps2"),
      .maximum_initial_angular_speed_radps = NonNegative(
          RequireDouble(node, "maximum_initial_angular_speed_radps"),
          "maximum_initial_angular_speed_radps"),
      .minimum_settle_guard = Duration(
          RequireDouble(node, "minimum_settle_guard_s"),
          "minimum_settle_guard_s", true),
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
    SchemaFailure("unsupported platform capability schema");
  }

  LoadedCapabilities loaded;
  const YAML::Node platform = RequireMap(platform_document, "platform");
  loaded.platform_id = RequireString(platform, "platform_id");
  const std::string platform_type = RequireString(platform, "platform_type");
  loaded.capability_version = RequireString(platform, "capability_version");
  loaded.base_frame_id = RequireString(platform, "base_frame_id");
  if (loaded.base_frame_id != "base_link") {
    ValueFailure("platform.base_frame_id must be base_link");
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
  const std::filesystem::path urdf_relative =
      RequireString(geometry, "urdf_file");
  loaded.urdf_path = ResolveFile(
      share, urdf_relative,
      CapabilityLoadErrorCode::kFileMissing,
      "CAPABILITY_URDF_MISSING");
  loaded.mesh_paths =
      ValidateGeometry(share, loaded.urdf_path, loaded.base_frame_id);

  if (platform_type == "WHEELED") {
    loaded.platform = ParseWheeled(platform_document, loaded);
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
