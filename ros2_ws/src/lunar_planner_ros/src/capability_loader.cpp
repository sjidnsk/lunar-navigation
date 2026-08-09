#include "lunar_planner_ros/capability_loader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <openssl/evp.h>
#include <urdf/model.h>
#include <yaml-cpp/yaml.h>

namespace lunar::planning::ros {
namespace {

constexpr std::string_view kPlatformSchema = "lunar-platform-profile/v1";
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

[[nodiscard]] std::string ReadBytes(const std::filesystem::path& path) {
  std::ifstream stream{path, std::ios::binary};
  if (!stream.good()) {
    throw LoadFailure{
        CapabilityLoadErrorCode::kFileMissing,
        "PLATFORM_PROFILE_FILE_MISSING",
        path.string(),
    };
  }
  return {
      std::istreambuf_iterator<char>{stream},
      std::istreambuf_iterator<char>{}};
}

[[nodiscard]] std::string Sha256(const std::string_view bytes) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context{
      EVP_MD_CTX_new(), EVP_MD_CTX_free};
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) != 1) {
    throw std::runtime_error{"unable to initialize SHA-256"};
  }
  unsigned char digest[EVP_MAX_MD_SIZE]{};
  unsigned int digest_size = 0U;
  if (EVP_DigestFinal_ex(context.get(), digest, &digest_size) != 1 ||
      digest_size != 32U) {
    throw std::runtime_error{"unable to finalize SHA-256"};
  }
  std::ostringstream encoded;
  encoded << std::hex << std::setfill('0');
  for (unsigned int index = 0U; index < digest_size; ++index) {
    encoded << std::setw(2) << static_cast<unsigned int>(digest[index]);
  }
  return encoded.str();
}

[[nodiscard]] bool IsSha256(const std::string_view value) noexcept {
  return value.size() == 64U &&
      std::ranges::all_of(value, [](const char character) {
        return (character >= '0' && character <= '9') ||
            (character >= 'a' && character <= 'f');
      });
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
       "maximum_surface_slope_rad",
       "minimum_clearance_m",
       "roughness_handling",
       "xy_resolution_m",
       "yaw_bin_count",
       "arc_radius_m",
       "arc_yaw_change_rad",
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
  const double wheel_count_value = Positive(
      RequireDouble(node, "wheel_count"), "wheel_count");
  if (std::floor(wheel_count_value) != wheel_count_value) {
    ValueFailure("wheel_count must be an integer");
  }
  const YAML::Node wheel_centers =
      RequireSequence(node, "wheel_center_xy_m", 1U);
  if (wheel_centers.size() != static_cast<std::size_t>(wheel_count_value)) {
    ValueFailure("wheel_center_xy_m count must equal wheel_count");
  }
  for (std::size_t index = 0U; index < wheel_centers.size(); ++index) {
    static_cast<void>(Vec2(
        wheel_centers[index],
        "wheeled.wheel_center_xy_m[" + std::to_string(index) + "]"));
  }
  if (wheel_width >= body_extent.y || wheelbase >= body_extent.x ||
      std::abs(track_width - (body_extent.y - wheel_width)) > 1.0e-6) {
    ValueFailure("wheeled wheel geometry is inconsistent with body extent");
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
  static_cast<void>(Positive(
      RequireDouble(node, "xy_resolution_m"), "xy_resolution_m"));
  const double yaw_bins = Positive(
      RequireDouble(node, "yaw_bin_count"), "yaw_bin_count");
  if (std::floor(yaw_bins) != yaw_bins) {
    ValueFailure("yaw_bin_count must be an integer");
  }
  static_cast<void>(Positive(
      RequireDouble(node, "arc_radius_m"), "arc_radius_m"));
  static_cast<void>(Finite(
      RequireDouble(node, "arc_yaw_change_rad"), "arc_yaw_change_rad"));

  std::vector<lunar::planning::WheelMotionPrimitive> primitives;
  const YAML::Node primitive_nodes =
      RequireSequence(node, "motion_primitives", 1U);
  std::set<std::string, std::less<>> primitive_ids;
  for (std::size_t index = 0U; index < primitive_nodes.size(); ++index) {
    const YAML::Node primitive = primitive_nodes[index];
    if (!primitive.IsMap()) {
      SchemaFailure("wheel motion primitive must be a map");
    }
    RejectUnexpectedKeys(
        primitive,
        {"primitive_id", "kind", "relative_end_pose"},
        "wheel motion primitive");
    const YAML::Node relative_end_pose =
        RequireMap(primitive, "relative_end_pose");
    RejectUnexpectedKeys(
        relative_end_pose,
        {"position_m", "orientation_wxyz"},
        "wheel relative_end_pose");
    const std::string id = RequireString(primitive, "primitive_id");
    RecordPrimitiveId(id, primitive_ids, loaded.source_motion_primitive_ids);
    primitives.push_back(lunar::planning::WheelMotionPrimitive{
        .primitive_id = id,
        .kind = WheelKind(RequireString(primitive, "kind")),
        .relative_end_pose = Pose(
            relative_end_pose, "motion_primitives.relative_end_pose"),
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
          RequireDouble(node, "maximum_surface_slope_rad"),
          "maximum_surface_slope_rad"),
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
       "nominal_body_height_m",
       "platform_mass_kg",
       "nominal_payload_kg",
       "maximum_payload_kg",
       "maximum_forward_speed_mps",
       "maximum_reverse_speed_mps",
       "maximum_lateral_speed_mps",
       "maximum_yaw_rate_radps",
       "maximum_surface_slope_rad",
       "maximum_step_height_m",
       "maximum_gap_width_m",
       "minimum_body_clearance_m",
       "step_vertical_rate_mps",
       "body_height_m",
       "maximum_linear_acceleration_mps2",
       "maximum_yaw_acceleration_radps2",
       "roughness_handling",
       "unknown_is_traversable",
       "local_xy_resolution_m",
       "output_semantics",
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
  const double nominal_payload = NonNegative(
      RequireDouble(node, "nominal_payload_kg"), "nominal_payload_kg");
  if (nominal_payload > maximum_payload) {
    ValueFailure("nominal_payload_kg exceeds maximum_payload_kg");
  }
  static_cast<void>(Positive(
      RequireDouble(node, "nominal_body_height_m"),
      "nominal_body_height_m"));
  const double step_vertical_rate = Positive(
      RequireDouble(node, "step_vertical_rate_mps"),
      "step_vertical_rate_mps");
  if (RequireString(node, "roughness_handling") != "DIAGNOSTIC_ONLY") {
    ValueFailure("unsupported legged roughness_handling");
  }
  if (RequireBool(node, "unknown_is_traversable")) {
    ValueFailure("legged unknown_is_traversable must be false");
  }
  static_cast<void>(Positive(
      RequireDouble(node, "local_xy_resolution_m"),
      "local_xy_resolution_m"));
  if (RequireString(node, "output_semantics") !=
      "LEGGED_BODY_REFERENCE") {
    ValueFailure("unsupported legged output_semantics");
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
    RejectUnexpectedKeys(
        primitive,
        {"primitive_id", "kind", "body_frame_displacement_m",
         "yaw_change_rad"},
        "legged motion primitive");
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
          RequireDouble(node, "maximum_surface_slope_rad"),
          "maximum_surface_slope_rad"),
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
      .forward_speed_mps = lunar::planning::Interval{
          .lower = -Positive(
              RequireDouble(node, "maximum_reverse_speed_mps"),
              "maximum_reverse_speed_mps"),
          .upper = Positive(
              RequireDouble(node, "maximum_forward_speed_mps"),
              "maximum_forward_speed_mps"),
      },
      .lateral_speed_mps = [&node]() {
        const double maximum = Positive(
            RequireDouble(node, "maximum_lateral_speed_mps"),
            "maximum_lateral_speed_mps");
        return lunar::planning::Interval{.lower = -maximum, .upper = maximum};
      }(),
      .yaw_rate_radps = [&node]() {
        const double maximum = Positive(
            RequireDouble(node, "maximum_yaw_rate_radps"),
            "maximum_yaw_rate_radps");
        return lunar::planning::Interval{.lower = -maximum, .upper = maximum};
      }(),
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
       "reference_horizontal_range_m",
       "reference_elevation_delta_m",
       "runtime_fallback_allowed",
       "landing_support_radius_m",
       "flight_collision_radius_m",
       "maximum_landing_slope_rad",
       "maximum_landing_plane_residual_m",
       "landing_lateral_margin_m",
       "flight_map_margin_m",
       "reachability_delta_v_margin_ratio",
       "gravity_mps2",
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
  static_cast<void>(Positive(
      RequireDouble(node, "reference_horizontal_range_m"),
      "reference_horizontal_range_m"));
  static_cast<void>(Finite(
      RequireDouble(node, "reference_elevation_delta_m"),
      "reference_elevation_delta_m"));
  if (RequireBool(node, "runtime_fallback_allowed")) {
    ValueFailure("hopper runtime_fallback_allowed must be false");
  }
  const auto gravity = Vec3(
      RequireSequence(node, "gravity_mps2"), "gravity_mps2");
  if (gravity.z >= 0.0 || std::hypot(gravity.x, gravity.y) > 1.0e-9) {
    ValueFailure("gravity_mps2 must be vertical and downward");
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

void AddOptionalAssetWarning(
    std::vector<CapabilityLoadWarning>& warnings,
    const std::string& reason_code,
    const std::string& detail) {
  warnings.push_back(CapabilityLoadWarning{
      .reason_code = reason_code,
      .detail = detail,
  });
}

[[nodiscard]] std::optional<std::filesystem::path> ResolveOptionalAsset(
    const YAML::Node& asset,
    const std::filesystem::path& root,
    const std::string& reason_code,
    std::vector<CapabilityLoadWarning>& warnings) {
  if (!asset || !asset.IsMap()) {
    AddOptionalAssetWarning(warnings, reason_code, "asset entry is not a map");
    return std::nullopt;
  }
  RejectUnexpectedKeys(asset, {"path", "sha256"}, "asset");
  std::string relative_text;
  std::string expected_sha256;
  try {
    relative_text = RequireString(asset, "path");
    expected_sha256 = RequireString(asset, "sha256");
  } catch (const LoadFailure& failure) {
    AddOptionalAssetWarning(warnings, reason_code, failure.detail);
    return std::nullopt;
  }
  const std::filesystem::path relative{relative_text};
  if (!IsSafeRelativePath(relative) || !IsSha256(expected_sha256)) {
    AddOptionalAssetWarning(warnings, reason_code, relative_text);
    return std::nullopt;
  }
  const auto canonical_root = std::filesystem::weakly_canonical(root);
  const auto candidate = std::filesystem::weakly_canonical(root / relative);
  if (!IsWithin(canonical_root, candidate) ||
      !std::filesystem::is_regular_file(candidate)) {
    AddOptionalAssetWarning(warnings, reason_code, candidate.string());
    return std::nullopt;
  }
  try {
    if (Sha256(ReadBytes(candidate)) != expected_sha256) {
      AddOptionalAssetWarning(
          warnings, reason_code, "SHA-256 mismatch: " + candidate.string());
      return std::nullopt;
    }
  } catch (const std::exception& error) {
    AddOptionalAssetWarning(warnings, reason_code, error.what());
    return std::nullopt;
  }
  return candidate;
}

void LoadOptionalAssets(
    const YAML::Node& root,
    const std::filesystem::path& profile_directory,
    LoadedCapabilities& loaded,
    std::vector<CapabilityLoadWarning>& warnings) {
  const YAML::Node assets = RequireMap(root, "assets");
  RejectUnexpectedKeys(assets, {"urdf", "meshes"}, "assets");
  const YAML::Node urdf_node = assets["urdf"];
  if (!urdf_node) {
    SchemaFailure("missing assets.urdf");
  }
  if (!urdf_node.IsNull()) {
    const auto urdf_path = ResolveOptionalAsset(
        urdf_node,
        profile_directory,
        "CAPABILITY_OPTIONAL_URDF_UNAVAILABLE",
        warnings);
    if (urdf_path) {
      urdf::Model model;
      if (model.initFile(urdf_path->string()) &&
          model.getLink(loaded.base_frame_id) != nullptr) {
        loaded.urdf_path = *urdf_path;
      } else {
        AddOptionalAssetWarning(
            warnings,
            "CAPABILITY_OPTIONAL_URDF_UNAVAILABLE",
            "URDF does not contain base frame: " + loaded.base_frame_id);
      }
    }
  }

  const YAML::Node meshes = assets["meshes"];
  if (!meshes || !meshes.IsSequence()) {
    SchemaFailure("assets.meshes must be a sequence");
  }
  for (const YAML::Node& mesh : meshes) {
    const auto mesh_path = ResolveOptionalAsset(
        mesh,
        profile_directory,
        "CAPABILITY_OPTIONAL_MESH_UNAVAILABLE",
        warnings);
    if (mesh_path) {
      loaded.mesh_paths.push_back(*mesh_path);
    }
  }
}

[[nodiscard]] LoadedCapabilities LoadDocument(
    const std::filesystem::path& profile_path,
    const std::string& bytes,
    std::vector<CapabilityLoadWarning>& warnings) {
  YAML::Node document;
  try {
    document = YAML::Load(bytes);
  } catch (const YAML::Exception& error) {
    throw LoadFailure{
        CapabilityLoadErrorCode::kParseError,
        "CAPABILITY_DOCUMENT_PARSE_ERROR",
        error.what(),
    };
  }
  if (!document.IsMap()) {
    SchemaFailure("platform profile must be an object");
  }
  RejectUnexpectedKeys(
      document,
      {"schema_version", "ownership", "platform", "observation", "assets",
       "sources", "wheeled", "legged", "hopper"},
      "platform profile");
  if (RequireString(document, "schema_version") != kPlatformSchema) {
    SchemaCompatibilityFailure("unsupported platform profile schema");
  }

  const YAML::Node ownership = RequireMap(document, "ownership");
  RejectUnexpectedKeys(
      ownership, {"producer", "consumer", "repository_role"}, "ownership");
  if (RequireString(ownership, "producer") !=
          "external_platform_and_sensor_systems" ||
      RequireString(ownership, "consumer") != "lunar_navigation" ||
      RequireString(ownership, "repository_role") !=
          "provisional_schema_and_approved_engineering_baseline") {
    ValueFailure("ownership boundary does not match the approved profile");
  }

  LoadedCapabilities loaded;
  loaded.profile_sha256 = Sha256(bytes);
  const YAML::Node platform = RequireMap(document, "platform");
  RejectUnexpectedKeys(
      platform,
      {"platform_id", "platform_type", "capability_version", "base_frame_id"},
      "platform");
  loaded.platform_id = RequireString(platform, "platform_id");
  const std::string platform_type = RequireString(platform, "platform_type");
  loaded.capability_version = RequireString(platform, "capability_version");
  loaded.base_frame_id = RequireString(platform, "base_frame_id");
  const std::string expected_base_frame =
      platform_type == "WHEELED" ? "base_footprint" : "base_link";
  if (loaded.base_frame_id != expected_base_frame) {
    ValueFailure("platform.base_frame_id must be " + expected_base_frame);
  }

  const YAML::Node sources = RequireMap(document, "sources");
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

  const YAML::Node observation = RequireMap(document, "observation");
  RejectUnexpectedKeys(
      observation, {"sensor_range_m", "sensor_fov_deg"}, "observation");
  loaded.observation.sensor_range_m = Positive(
      RequireDouble(observation, "sensor_range_m"), "sensor_range_m");
  const double fov_degrees = Positive(
      RequireDouble(observation, "sensor_fov_deg"), "sensor_fov_deg");
  if (fov_degrees > 360.0) {
    ValueFailure("sensor_fov_deg exceeds 360 degrees");
  }
  loaded.observation.sensor_fov_rad =
      fov_degrees * std::numbers::pi / 180.0;

  if (platform_type == "WHEELED") {
    if (document["legged"] || document["hopper"]) {
      SchemaCompatibilityFailure("wheeled profile contains another platform");
    }
    loaded.platform = ParseWheeled(document, loaded);
  } else if (platform_type == "LEGGED") {
    if (document["wheeled"] || document["hopper"]) {
      SchemaCompatibilityFailure("legged profile contains another platform");
    }
    loaded.platform = ParseLegged(document, loaded);
  } else if (platform_type == "HOPPER") {
    if (document["wheeled"] || document["legged"]) {
      SchemaCompatibilityFailure("hopper profile contains another platform");
    }
    loaded.platform = ParseHopper(document, loaded);
  } else {
    ValueFailure("unknown platform.platform_type: " + platform_type);
  }

  LoadOptionalAssets(document, profile_path.parent_path(), loaded, warnings);
  return loaded;
}

}  // namespace

CapabilityLoadResult CapabilityLoader::LoadFromFile(
    const std::filesystem::path& platform_profile_file) const {
  try {
    if (!std::filesystem::is_regular_file(platform_profile_file)) {
      throw LoadFailure{
          CapabilityLoadErrorCode::kFileMissing,
          "PLATFORM_PROFILE_FILE_MISSING",
          platform_profile_file.string(),
      };
    }
    const auto canonical_path =
        std::filesystem::weakly_canonical(platform_profile_file);
    const std::string bytes = ReadBytes(canonical_path);
    std::vector<CapabilityLoadWarning> warnings;
    LoadedCapabilities loaded = LoadDocument(canonical_path, bytes, warnings);
    return CapabilityLoadResult{
        .capabilities = std::move(loaded),
        .error = std::nullopt,
        .warnings = std::move(warnings),
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

CapabilityLoadResult CapabilityLoader::LoadFromPackageShare(
    const std::string& package_name,
    const std::filesystem::path& platform_profile_file) const {
  try {
    if (!IsSafeRelativePath(platform_profile_file)) {
      throw LoadFailure{
          CapabilityLoadErrorCode::kUnsafePath,
          "CAPABILITY_PATH_UNSAFE",
          platform_profile_file.string(),
      };
    }
    if (package_name.empty()) {
      throw std::runtime_error{"package name is empty"};
    }
    const std::filesystem::path share =
        ament_index_cpp::get_package_share_directory(package_name);
    const auto profile_path = ResolveFile(
        share, platform_profile_file,
        CapabilityLoadErrorCode::kFileMissing,
        "PLATFORM_PROFILE_FILE_MISSING");
    return LoadFromFile(profile_path);
  } catch (const LoadFailure& failure) {
    return Failure(failure);
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

}  // namespace lunar::planning::ros
