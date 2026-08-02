#include "lunar_path_planner/v3/codec/json_codec.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "lunar_path_planner/v3/codec/jcs_canonicalizer.hpp"
#include "lunar_path_planner/v3/crypto/sha256.hpp"
#include "lunar_path_planner/v3/map/immutable_snapshot.hpp"

namespace lunar::planning::v3 {
namespace {

using Json = nlohmann::json;

constexpr std::string_view kPlanningRequestSchemaVersion =
    "path-planner-v3-planning-request/v1";
constexpr std::string_view kPlanningResponseSchemaVersion =
    "path-planner-v3-planning-response/v1";
constexpr std::string_view kReferenceBundleSchemaVersion =
    "path-planner-v3-reference-bundle/v1";
constexpr std::string_view kWheeledReferenceSchemaVersion =
    "path-planner-v3-wheeled-reference/v1";
constexpr std::string_view kLeggedReferenceSchemaVersion =
    "path-planner-v3-legged-body-reference/v1";
constexpr std::string_view kHopperReferenceSchemaVersion =
    "path-planner-v3-hopper-reference/v1";

class DecodeFailure final : public std::exception {
 public:
  explicit DecodeFailure(Error error) : error_{std::move(error)} {}

  [[nodiscard]] const Error& error() const noexcept { return error_; }
  [[nodiscard]] const char* what() const noexcept override {
    return error_.message.c_str();
  }

 private:
  Error error_;
};

class DuplicateKeyFailure final : public std::exception {
 public:
  [[nodiscard]] const char* what() const noexcept override {
    return "duplicate JSON object key";
  }
};

[[noreturn]] void Fail(ErrorCode code,
                       std::string_view field_path,
                       std::string message) {
  throw DecodeFailure{
      Error{code, std::string{field_path}, std::move(message)}};
}

[[nodiscard]] bool Contains(
    std::initializer_list<std::string_view> names,
    std::string_view candidate) {
  return std::find(names.begin(), names.end(), candidate) != names.end();
}

void CheckObject(
    const Json& value,
    std::string_view field_path,
    std::initializer_list<std::string_view> required,
    std::initializer_list<std::string_view> optional = {}) {
  if (!value.is_object()) {
    Fail(ErrorCode::kSchemaMismatch, field_path, "expected JSON object");
  }
  for (const auto& [key, unused] : value.items()) {
    static_cast<void>(unused);
    if (!Contains(required, key) && !Contains(optional, key)) {
      Fail(ErrorCode::kSchemaMismatch,
           std::string{field_path} + "." + key,
           "unknown object member");
    }
  }
  for (const std::string_view key : required) {
    if (!value.contains(std::string{key})) {
      Fail(ErrorCode::kSchemaMismatch,
           std::string{field_path} + "." + std::string{key},
           "required object member is missing");
    }
  }
}

[[nodiscard]] const Json& Member(
    const Json& object,
    std::string_view key,
    std::string_view field_path) {
  const auto iterator = object.find(std::string{key});
  if (iterator == object.end()) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + "." + std::string{key},
         "required object member is missing");
  }
  return *iterator;
}

[[nodiscard]] bool IsIdentifierCharacter(char value) {
  const auto byte = static_cast<unsigned char>(value);
  return (byte >= static_cast<unsigned char>('A') &&
          byte <= static_cast<unsigned char>('Z')) ||
         (byte >= static_cast<unsigned char>('a') &&
          byte <= static_cast<unsigned char>('z')) ||
         (byte >= static_cast<unsigned char>('0') &&
          byte <= static_cast<unsigned char>('9')) ||
         value == '.' || value == '_' || value == ':' || value == '/' ||
         value == '-';
}

[[nodiscard]] bool IsAsciiAlphaNumeric(char value) {
  const auto byte = static_cast<unsigned char>(value);
  return (byte >= static_cast<unsigned char>('A') &&
          byte <= static_cast<unsigned char>('Z')) ||
         (byte >= static_cast<unsigned char>('a') &&
          byte <= static_cast<unsigned char>('z')) ||
         (byte >= static_cast<unsigned char>('0') &&
          byte <= static_cast<unsigned char>('9'));
}

[[nodiscard]] std::string DecodeString(const Json& value,
                                       std::string_view field_path) {
  if (!value.is_string()) {
    Fail(ErrorCode::kSchemaMismatch, field_path, "expected JSON string");
  }
  return value.get<std::string>();
}

[[nodiscard]] std::string DecodeIdentifier(const Json& value,
                                           std::string_view field_path) {
  std::string text = DecodeString(value, field_path);
  if (text.empty() || text.size() > 128U ||
      !IsAsciiAlphaNumeric(text.front()) ||
      !std::all_of(text.begin() + 1, text.end(), IsIdentifierCharacter)) {
    Fail(ErrorCode::kSchemaMismatch, field_path, "invalid identifier");
  }
  return text;
}

[[nodiscard]] std::string DecodeReasonCode(const Json& value,
                                           std::string_view field_path) {
  std::string text = DecodeString(value, field_path);
  const auto is_reason_character = [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') || character == '_';
  };
  if (text.empty() || text.size() > 128U ||
      text.front() < 'A' || text.front() > 'Z' ||
      !std::all_of(text.begin() + 1, text.end(), is_reason_character)) {
    Fail(ErrorCode::kSchemaMismatch, field_path, "invalid reason code");
  }
  return text;
}

[[nodiscard]] bool DecodeBoolean(const Json& value,
                                 std::string_view field_path) {
  if (!value.is_boolean()) {
    Fail(ErrorCode::kSchemaMismatch, field_path, "expected JSON boolean");
  }
  return value.get<bool>();
}

[[nodiscard]] double DecodeNumber(const Json& value,
                                  std::string_view field_path) {
  if (!value.is_number()) {
    Fail(ErrorCode::kSchemaMismatch, field_path, "expected JSON number");
  }
  const double result = value.get<double>();
  if (!std::isfinite(result)) {
    Fail(ErrorCode::kInvalidArgument,
         field_path,
         "floating-point value must be finite");
  }
  return result;
}

[[nodiscard]] std::uint64_t DecodeUnsignedInteger(
    const Json& value,
    std::string_view field_path,
    std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max()) {
  std::uint64_t result{};
  if (value.is_number_unsigned()) {
    result = value.get<std::uint64_t>();
  } else if (value.is_number_integer()) {
    const std::int64_t signed_value = value.get<std::int64_t>();
    if (signed_value < 0) {
      Fail(ErrorCode::kSchemaMismatch,
           field_path,
           "expected non-negative integer");
    }
    result = static_cast<std::uint64_t>(signed_value);
  } else {
    Fail(ErrorCode::kSchemaMismatch, field_path, "expected integer");
  }
  if (result > maximum) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "integer exceeds schema maximum");
  }
  return result;
}

[[nodiscard]] std::size_t DecodeSize(const Json& value,
                                     std::string_view field_path) {
  const auto decoded = DecodeUnsignedInteger(
      value,
      field_path,
      static_cast<std::uint64_t>(
          std::numeric_limits<std::size_t>::max()));
  return static_cast<std::size_t>(decoded);
}

[[nodiscard]] std::uint32_t DecodeRevision(
    const Json& value,
    std::string_view field_path) {
  const auto revision = DecodeUnsignedInteger(value, field_path, 2147483647U);
  if (revision == 0U) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "revision must be positive");
  }
  return static_cast<std::uint32_t>(revision);
}

[[nodiscard]] std::string DecodeHash(const Json& value,
                                     std::string_view field_path) {
  std::string hash = DecodeString(value, field_path);
  const auto is_lower_hex = [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  };
  if (hash.size() != 64U ||
      !std::all_of(hash.begin(), hash.end(), is_lower_hex)) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "expected lower-case SHA-256 hexadecimal digest");
  }
  return hash;
}

void RequireLiteral(const Json& value,
                    std::string_view expected,
                    std::string_view field_path) {
  if (!value.is_string() || value.get_ref<const std::string&>() != expected) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "unexpected schema constant or enum value");
  }
}

void RequireBooleanLiteral(const Json& value,
                           bool expected,
                           std::string_view field_path) {
  if (!value.is_boolean() || value.get<bool>() != expected) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "unexpected schema boolean constant");
  }
}

void RequireNumberLiteral(const Json& value,
                          double expected,
                          std::string_view field_path) {
  if (DecodeNumber(value, field_path) != expected) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "unexpected schema numeric constant");
  }
}

[[nodiscard]] std::chrono::nanoseconds ParseNanoseconds(
    const Json& value,
    std::string_view field_path,
    bool allow_negative,
    bool require_positive = false) {
  if (!value.is_string()) {
    Fail(ErrorCode::kInvalidArgument,
         field_path,
         "nanoseconds must be a decimal string");
  }
  const std::string& text = value.get_ref<const std::string&>();
  const bool negative = !text.empty() && text.front() == '-';
  const std::size_t first_digit = negative ? 1U : 0U;
  const bool canonical_zero = text == "0";
  const bool canonical_nonzero =
      first_digit < text.size() &&
      text[first_digit] >= '1' && text[first_digit] <= '9' &&
      std::all_of(
          text.begin() +
              static_cast<std::ptrdiff_t>(first_digit + 1U),
          text.end(),
          [](char character) {
            return character >= '0' && character <= '9';
          });
  if ((!canonical_zero && !canonical_nonzero) ||
      (negative && !allow_negative) ||
      (require_positive && (negative || canonical_zero))) {
    Fail(ErrorCode::kInvalidArgument,
         field_path,
         "nanoseconds must use the required canonical decimal form");
  }
  std::int64_t count{};
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), count);
  if (error != std::errc{} || end != text.data() + text.size()) {
    Fail(ErrorCode::kInvalidArgument,
         field_path,
         "invalid int64 nanoseconds decimal string");
  }
  return std::chrono::nanoseconds{count};
}

[[nodiscard]] std::string EncodeNanoseconds(
    std::chrono::nanoseconds value) {
  std::array<char, 32> buffer{};
  const auto [end, error] =
      std::to_chars(buffer.data(),
                    buffer.data() + buffer.size(),
                    value.count());
  if (error != std::errc{}) {
    throw std::invalid_argument{"failed to encode int64 nanoseconds"};
  }
  return {buffer.data(), end};
}

[[nodiscard]] std::string EncodeNonNegativeNanoseconds(
    std::chrono::nanoseconds value,
    std::string_view field_path) {
  if (value.count() < 0) {
    throw std::invalid_argument{
        std::string{field_path} +
        ": nanoseconds must be non-negative"};
  }
  return EncodeNanoseconds(value);
}

[[nodiscard]] std::string EncodePositiveNanoseconds(
    std::chrono::nanoseconds value,
    std::string_view field_path) {
  if (value.count() <= 0) {
    throw std::invalid_argument{
        std::string{field_path} + ": nanoseconds must be positive"};
  }
  return EncodeNanoseconds(value);
}

class StrictParseCallback final {
 public:
  bool operator()(int,
                  Json::parse_event_t event,
                  Json& parsed) {
    switch (event) {
      case Json::parse_event_t::object_start:
        object_keys_.emplace_back();
        break;
      case Json::parse_event_t::key: {
        if (object_keys_.empty()) {
          throw DuplicateKeyFailure{};
        }
        const std::string& key = parsed.get_ref<const std::string&>();
        if (!object_keys_.back().insert(key).second) {
          throw DuplicateKeyFailure{};
        }
        break;
      }
      case Json::parse_event_t::object_end:
        if (!object_keys_.empty()) {
          object_keys_.pop_back();
        }
        break;
      default:
        break;
    }
    return true;
  }

 private:
  std::vector<std::unordered_set<std::string>> object_keys_;
};

void CheckFiniteJsonTree(const Json& value,
                         std::string_view field_path) {
  if (value.is_number_float() &&
      !std::isfinite(value.get<double>())) {
    Fail(ErrorCode::kInvalidArgument,
         field_path,
         "floating-point value must be finite");
  }
  if (value.is_array()) {
    for (std::size_t index = 0; index < value.size(); ++index) {
      CheckFiniteJsonTree(
          value[index],
          std::string{field_path} + "[" + std::to_string(index) + "]");
    }
  } else if (value.is_object()) {
    for (const auto& [key, child] : value.items()) {
      CheckFiniteJsonTree(
          child, std::string{field_path} + "." + key);
    }
  }
}

[[nodiscard]] Json ParseStrictJson(std::string_view payload) {
  try {
    StrictParseCallback callback;
    Json parsed = Json::parse(
        payload.begin(), payload.end(), std::ref(callback), true, false);
    CheckFiniteJsonTree(parsed, "$");
    return parsed;
  } catch (const DuplicateKeyFailure&) {
    Fail(ErrorCode::kSchemaMismatch, "$", "duplicate JSON object key");
  } catch (const nlohmann::json::out_of_range& exception) {
    Fail(ErrorCode::kInvalidArgument, "$", exception.what());
  } catch (const nlohmann::json::parse_error& exception) {
    Fail(ErrorCode::kInvalidArgument, "$", exception.what());
  } catch (const nlohmann::json::exception& exception) {
    Fail(ErrorCode::kInvalidArgument, "$", exception.what());
  }
}

void CheckSchemaVersion(const Json& root,
                        std::string_view expected,
                        std::string_view field_path) {
  if (!root.is_object()) {
    Fail(ErrorCode::kSchemaMismatch, field_path, "expected JSON object");
  }
  const auto iterator = root.find("schema_version");
  if (iterator == root.end() || !iterator->is_string() ||
      iterator->get_ref<const std::string&>() != expected) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".schema_version",
         "schema version mismatch");
  }
}

[[nodiscard]] Vec2 DecodeVec2(const Json& value,
                              std::string_view field_path) {
  if (!value.is_array() || value.size() != 2U) {
    Fail(ErrorCode::kSchemaMismatch, field_path, "expected Vector2");
  }
  return {
      DecodeNumber(value[0], std::string{field_path} + "[0]"),
      DecodeNumber(value[1], std::string{field_path} + "[1]"),
  };
}

[[nodiscard]] Json EncodeVec2(const Vec2& value) {
  return Json::array({value.x, value.y});
}

[[nodiscard]] Vec3 DecodeVec3(const Json& value,
                              std::string_view field_path) {
  if (!value.is_array() || value.size() != 3U) {
    Fail(ErrorCode::kSchemaMismatch, field_path, "expected Vector3");
  }
  return {
      DecodeNumber(value[0], std::string{field_path} + "[0]"),
      DecodeNumber(value[1], std::string{field_path} + "[1]"),
      DecodeNumber(value[2], std::string{field_path} + "[2]"),
  };
}

[[nodiscard]] Json EncodeVec3(const Vec3& value) {
  return Json::array({value.x, value.y, value.z});
}

[[nodiscard]] Quaternion DecodeQuaternion(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value, field_path, {"w", "x", "y", "z"});
  return {
      DecodeNumber(Member(value, "w", field_path),
                   std::string{field_path} + ".w"),
      DecodeNumber(Member(value, "x", field_path),
                   std::string{field_path} + ".x"),
      DecodeNumber(Member(value, "y", field_path),
                   std::string{field_path} + ".y"),
      DecodeNumber(Member(value, "z", field_path),
                   std::string{field_path} + ".z"),
  };
}

[[nodiscard]] Json EncodeQuaternion(const Quaternion& value) {
  return {
      {"w", value.w}, {"x", value.x}, {"y", value.y}, {"z", value.z}};
}

[[nodiscard]] ContentRef DecodeContentRef(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value, field_path, {"id", "revision", "content_hash"});
  return {
      DecodeIdentifier(Member(value, "id", field_path),
                       std::string{field_path} + ".id"),
      DecodeRevision(Member(value, "revision", field_path),
                     std::string{field_path} + ".revision"),
      DecodeHash(Member(value, "content_hash", field_path),
                 std::string{field_path} + ".content_hash"),
  };
}

[[nodiscard]] Json EncodeContentRef(const ContentRef& value) {
  return {
      {"id", value.id},
      {"revision", value.revision},
      {"content_hash", value.content_hash},
  };
}

[[nodiscard]] ClockStamp DecodeClockStamp(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value, field_path, {"clock_id", "tick_ns"});
  return {
      DecodeIdentifier(Member(value, "clock_id", field_path),
                       std::string{field_path} + ".clock_id"),
      ParseNanoseconds(Member(value, "tick_ns", field_path),
                       std::string{field_path} + ".tick_ns",
                       true),
  };
}

[[nodiscard]] Json EncodeClockStamp(const ClockStamp& value) {
  return {
      {"clock_id", value.clock_id},
      {"tick_ns", EncodeNanoseconds(value.tick)},
  };
}

[[nodiscard]] PoseXyzYaw DecodePose(const Json& value,
                                    std::string_view field_path) {
  CheckObject(value, field_path, {"position_m", "yaw_rad"});
  return {
      DecodeVec3(Member(value, "position_m", field_path),
                 std::string{field_path} + ".position_m"),
      DecodeNumber(Member(value, "yaw_rad", field_path),
                   std::string{field_path} + ".yaw_rad"),
  };
}

[[nodiscard]] Json EncodePose(const PoseXyzYaw& value) {
  return {
      {"position_m", EncodeVec3(value.position_m)},
      {"yaw_rad", value.yaw_rad},
  };
}

[[nodiscard]] AxisAlignedBox3 DecodeAxisAlignedBox3(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value, field_path, {"shape", "center", "half_extent"});
  RequireLiteral(Member(value, "shape", field_path),
                 "axis_aligned_box",
                 std::string{field_path} + ".shape");
  return {
      DecodeVec3(Member(value, "center", field_path),
                 std::string{field_path} + ".center"),
      DecodeVec3(Member(value, "half_extent", field_path),
                 std::string{field_path} + ".half_extent"),
  };
}

[[nodiscard]] EuclideanBall3 DecodeEuclideanBall3(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value, field_path, {"shape", "center", "radius"});
  RequireLiteral(Member(value, "shape", field_path),
                 "euclidean_ball",
                 std::string{field_path} + ".shape");
  const double radius =
      DecodeNumber(Member(value, "radius", field_path),
                   std::string{field_path} + ".radius");
  if (radius < 0.0) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".radius",
         "radius must be non-negative");
  }
  return {
      DecodeVec3(Member(value, "center", field_path),
                 std::string{field_path} + ".center"),
      radius,
  };
}

[[nodiscard]] DeterministicVectorSet3 DecodeVectorSet3(
    const Json& value,
    std::string_view field_path) {
  if (!value.is_object() || !value.contains("shape")) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "vector set requires a shape discriminator");
  }
  const std::string shape = DecodeString(
      value.at("shape"), std::string{field_path} + ".shape");
  if (shape == "axis_aligned_box") {
    return DecodeAxisAlignedBox3(value, field_path);
  }
  if (shape == "euclidean_ball") {
    return DecodeEuclideanBall3(value, field_path);
  }
  Fail(ErrorCode::kSchemaMismatch,
       std::string{field_path} + ".shape",
       "unknown deterministic vector-set shape");
}

[[nodiscard]] Json EncodeVectorSet3(
    const DeterministicVectorSet3& value) {
  return std::visit(
      [](const auto& concrete) -> Json {
        using T = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<T, AxisAlignedBox3>) {
          return {
              {"shape", "axis_aligned_box"},
              {"center", EncodeVec3(concrete.center)},
              {"half_extent", EncodeVec3(concrete.half_extent)},
          };
        } else {
          return {
              {"shape", "euclidean_ball"},
              {"center", EncodeVec3(concrete.center)},
              {"radius", concrete.radius},
          };
        }
      },
      value);
}

[[nodiscard]] SymmetricScalarInterval DecodeSymmetricInterval(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value, field_path, {"shape", "center", "half_width"});
  RequireLiteral(Member(value, "shape", field_path),
                 "symmetric_interval",
                 std::string{field_path} + ".shape");
  const double half_width =
      DecodeNumber(Member(value, "half_width", field_path),
                   std::string{field_path} + ".half_width");
  if (half_width < 0.0) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".half_width",
         "half width must be non-negative");
  }
  return {
      DecodeNumber(Member(value, "center", field_path),
                   std::string{field_path} + ".center"),
      half_width,
  };
}

[[nodiscard]] Json EncodeSymmetricInterval(
    const SymmetricScalarInterval& value) {
  return {
      {"shape", "symmetric_interval"},
      {"center", value.center},
      {"half_width", value.half_width},
  };
}

[[nodiscard]] RotationVectorBall DecodeRotationVectorBall(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value, field_path, {"shape", "radius_rad"});
  RequireLiteral(Member(value, "shape", field_path),
                 "rotation_vector_ball",
                 std::string{field_path} + ".shape");
  const double radius =
      DecodeNumber(Member(value, "radius_rad", field_path),
                   std::string{field_path} + ".radius_rad");
  if (radius < 0.0 || radius > 3.141592653589793) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".radius_rad",
         "rotation-vector radius is outside schema bounds");
  }
  return {radius};
}

[[nodiscard]] Json EncodeRotationVectorBall(
    const RotationVectorBall& value) {
  return {
      {"shape", "rotation_vector_ball"},
      {"radius_rad", value.radius_rad},
  };
}

[[nodiscard]] WheeledOrLeggedErrorBounds
DecodeWheeledOrLeggedErrorBounds(const Json& value,
                                 std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"position_bound_m",
               "yaw_bound_rad",
               "linear_velocity_bound_mps",
               "yaw_rate_bound_radps"});
  return {
      DecodeVectorSet3(
          Member(value, "position_bound_m", field_path),
          std::string{field_path} + ".position_bound_m"),
      DecodeSymmetricInterval(
          Member(value, "yaw_bound_rad", field_path),
          std::string{field_path} + ".yaw_bound_rad"),
      DecodeVectorSet3(
          Member(value, "linear_velocity_bound_mps", field_path),
          std::string{field_path} + ".linear_velocity_bound_mps"),
      DecodeSymmetricInterval(
          Member(value, "yaw_rate_bound_radps", field_path),
          std::string{field_path} + ".yaw_rate_bound_radps"),
  };
}

[[nodiscard]] Json EncodeWheeledOrLeggedErrorBounds(
    const WheeledOrLeggedErrorBounds& value) {
  return {
      {"position_bound_m", EncodeVectorSet3(value.position_bound_m)},
      {"yaw_bound_rad", EncodeSymmetricInterval(value.yaw_bound_rad)},
      {"linear_velocity_bound_mps",
       EncodeVectorSet3(value.linear_velocity_bound_mps)},
      {"yaw_rate_bound_radps",
       EncodeSymmetricInterval(value.yaw_rate_bound_radps)},
  };
}

[[nodiscard]] HopperErrorBounds DecodeHopperErrorBounds(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"position_bound_m",
               "orientation_bound",
               "linear_velocity_bound_mps",
               "angular_velocity_bound_radps"});
  return {
      DecodeVectorSet3(
          Member(value, "position_bound_m", field_path),
          std::string{field_path} + ".position_bound_m"),
      DecodeRotationVectorBall(
          Member(value, "orientation_bound", field_path),
          std::string{field_path} + ".orientation_bound"),
      DecodeVectorSet3(
          Member(value, "linear_velocity_bound_mps", field_path),
          std::string{field_path} + ".linear_velocity_bound_mps"),
      DecodeVectorSet3(
          Member(value, "angular_velocity_bound_radps", field_path),
          std::string{field_path} + ".angular_velocity_bound_radps"),
  };
}

[[nodiscard]] Json EncodeHopperErrorBounds(
    const HopperErrorBounds& value) {
  return {
      {"position_bound_m", EncodeVectorSet3(value.position_bound_m)},
      {"orientation_bound",
       EncodeRotationVectorBall(value.orientation_bound)},
      {"linear_velocity_bound_mps",
       EncodeVectorSet3(value.linear_velocity_bound_mps)},
      {"angular_velocity_bound_radps",
       EncodeVectorSet3(value.angular_velocity_bound_radps)},
  };
}

[[nodiscard]] WheeledOrLeggedState
DecodeWheeledOrLeggedState(const Json& value,
                           std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"position_m",
               "yaw_rad",
               "linear_velocity_mps",
               "yaw_rate_radps",
               "error_bounds"});
  return {
      DecodeVec3(Member(value, "position_m", field_path),
                 std::string{field_path} + ".position_m"),
      DecodeNumber(Member(value, "yaw_rad", field_path),
                   std::string{field_path} + ".yaw_rad"),
      DecodeVec3(Member(value, "linear_velocity_mps", field_path),
                 std::string{field_path} + ".linear_velocity_mps"),
      DecodeNumber(Member(value, "yaw_rate_radps", field_path),
                   std::string{field_path} + ".yaw_rate_radps"),
      DecodeWheeledOrLeggedErrorBounds(
          Member(value, "error_bounds", field_path),
          std::string{field_path} + ".error_bounds"),
  };
}

[[nodiscard]] Json EncodeWheeledOrLeggedState(
    const WheeledOrLeggedState& value) {
  return {
      {"position_m", EncodeVec3(value.position_m)},
      {"yaw_rad", value.yaw_rad},
      {"linear_velocity_mps", EncodeVec3(value.linear_velocity_mps)},
      {"yaw_rate_radps", value.yaw_rate_radps},
      {"error_bounds",
       EncodeWheeledOrLeggedErrorBounds(value.error_bounds)},
  };
}

[[nodiscard]] HopperState DecodeHopperState(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"position_m",
               "orientation_body_to_frame",
               "linear_velocity_mps",
               "angular_velocity_radps",
               "error_bounds"});
  return {
      DecodeVec3(Member(value, "position_m", field_path),
                 std::string{field_path} + ".position_m"),
      DecodeQuaternion(
          Member(value, "orientation_body_to_frame", field_path),
          std::string{field_path} + ".orientation_body_to_frame"),
      DecodeVec3(Member(value, "linear_velocity_mps", field_path),
                 std::string{field_path} + ".linear_velocity_mps"),
      DecodeVec3(Member(value, "angular_velocity_radps", field_path),
                 std::string{field_path} + ".angular_velocity_radps"),
      DecodeHopperErrorBounds(
          Member(value, "error_bounds", field_path),
          std::string{field_path} + ".error_bounds"),
  };
}

[[nodiscard]] Json EncodeHopperState(const HopperState& value) {
  return {
      {"position_m", EncodeVec3(value.position_m)},
      {"orientation_body_to_frame",
       EncodeQuaternion(value.orientation_body_to_frame)},
      {"linear_velocity_mps", EncodeVec3(value.linear_velocity_mps)},
      {"angular_velocity_radps",
       EncodeVec3(value.angular_velocity_radps)},
      {"error_bounds", EncodeHopperErrorBounds(value.error_bounds)},
  };
}

[[nodiscard]] PlatformType DecodePlatformType(
    const Json& value,
    std::string_view field_path) {
  const std::string text = DecodeString(value, field_path);
  if (text == "WHEELED") {
    return PlatformType::kWheeled;
  }
  if (text == "LEGGED") {
    return PlatformType::kLegged;
  }
  if (text == "HOPPER") {
    return PlatformType::kHopper;
  }
  Fail(ErrorCode::kSchemaMismatch, field_path, "unknown platform_type");
}

[[nodiscard]] std::string_view EncodePlatformType(PlatformType value) {
  switch (value) {
    case PlatformType::kWheeled:
      return "WHEELED";
    case PlatformType::kLegged:
      return "LEGGED";
    case PlatformType::kHopper:
      return "HOPPER";
  }
  throw std::invalid_argument{"unknown PlatformType"};
}

[[nodiscard]] PlatformState DecodePlatformState(
    const Json& value,
    PlatformType platform_type,
    std::string_view field_path) {
  if (platform_type == PlatformType::kHopper) {
    return DecodeHopperState(value, field_path);
  }
  return DecodeWheeledOrLeggedState(value, field_path);
}

[[nodiscard]] CircularYawInterval DecodeCircularYawInterval(
    const Json& value,
    std::string_view field_path) {
  CheckObject(
      value,
      field_path,
      {"representation", "start_rad", "span_rad", "closed"});
  RequireLiteral(Member(value, "representation", field_path),
                 "canonical_ccw",
                 std::string{field_path} + ".representation");
  RequireBooleanLiteral(Member(value, "closed", field_path),
                        true,
                        std::string{field_path} + ".closed");
  return {
      CircularYawInterval::Representation::kCanonicalCcw,
      DecodeNumber(Member(value, "start_rad", field_path),
                   std::string{field_path} + ".start_rad"),
      DecodeNumber(Member(value, "span_rad", field_path),
                   std::string{field_path} + ".span_rad"),
      true,
  };
}

[[nodiscard]] Json EncodeCircularYawInterval(
    const CircularYawInterval& value) {
  return {
      {"representation", "canonical_ccw"},
      {"start_rad", value.start_rad},
      {"span_rad", value.span_rad},
      {"closed", value.closed},
  };
}

[[nodiscard]] LandingPlane DecodeLandingPlane(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"origin_m",
               "normal",
               "basis_u",
               "basis_v",
               "residual_bound_m"});
  return {
      DecodeVec3(Member(value, "origin_m", field_path),
                 std::string{field_path} + ".origin_m"),
      DecodeVec3(Member(value, "normal", field_path),
                 std::string{field_path} + ".normal"),
      DecodeVec3(Member(value, "basis_u", field_path),
                 std::string{field_path} + ".basis_u"),
      DecodeVec3(Member(value, "basis_v", field_path),
                 std::string{field_path} + ".basis_v"),
      DecodeNumber(Member(value, "residual_bound_m", field_path),
                   std::string{field_path} + ".residual_bound_m"),
  };
}

[[nodiscard]] Json EncodeLandingPlane(const LandingPlane& value) {
  return {
      {"origin_m", EncodeVec3(value.origin_m)},
      {"normal", EncodeVec3(value.normal)},
      {"basis_u", EncodeVec3(value.basis_u)},
      {"basis_v", EncodeVec3(value.basis_v)},
      {"residual_bound_m", value.residual_bound_m},
  };
}

[[nodiscard]] ConvexPolygonUv DecodeConvexPolygonUv(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value, field_path, {"vertices_uv", "winding"});
  RequireLiteral(Member(value, "winding", field_path),
                 "CCW",
                 std::string{field_path} + ".winding");
  const Json& vertices = Member(value, "vertices_uv", field_path);
  if (!vertices.is_array() || vertices.size() < 3U ||
      vertices.size() > 128U) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".vertices_uv",
         "polygon vertex count is outside schema bounds");
  }
  ConvexPolygonUv result;
  result.vertices_uv.reserve(vertices.size());
  for (std::size_t index = 0; index < vertices.size(); ++index) {
    result.vertices_uv.push_back(
        DecodeVec2(vertices[index],
                   std::string{field_path} + ".vertices_uv[" +
                       std::to_string(index) + "]"));
  }
  return result;
}

[[nodiscard]] Json EncodeConvexPolygonUv(
    const ConvexPolygonUv& value) {
  Json vertices = Json::array();
  for (const Vec2& vertex : value.vertices_uv) {
    vertices.push_back(EncodeVec2(vertex));
  }
  return {{"vertices_uv", std::move(vertices)}, {"winding", "CCW"}};
}

[[nodiscard]] PointGoal DecodePointGoal(
    const Json& value,
    std::string_view field_path) {
  CheckObject(
      value, field_path, {"kind", "position_m", "position_tolerance_m"});
  RequireLiteral(Member(value, "kind", field_path),
                 "POINT_WITH_TOLERANCE",
                 std::string{field_path} + ".kind");
  const double tolerance =
      DecodeNumber(Member(value, "position_tolerance_m", field_path),
                   std::string{field_path} + ".position_tolerance_m");
  if (tolerance < 0.0) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".position_tolerance_m",
         "position tolerance must be non-negative");
  }
  return {
      DecodeVec3(Member(value, "position_m", field_path),
                 std::string{field_path} + ".position_m"),
      tolerance,
  };
}

[[nodiscard]] PlanarRegionGoal DecodePlanarRegionGoal(
    const Json& value,
    std::string_view field_path) {
  CheckObject(
      value, field_path, {"kind", "plane", "polygon", "normal_tolerance_m"});
  RequireLiteral(Member(value, "kind", field_path),
                 "PLANAR_CONVEX_REGION",
                 std::string{field_path} + ".kind");
  const double tolerance =
      DecodeNumber(Member(value, "normal_tolerance_m", field_path),
                   std::string{field_path} + ".normal_tolerance_m");
  if (tolerance < 0.0) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".normal_tolerance_m",
         "normal tolerance must be non-negative");
  }
  return {
      DecodeLandingPlane(Member(value, "plane", field_path),
                         std::string{field_path} + ".plane"),
      DecodeConvexPolygonUv(Member(value, "polygon", field_path),
                            std::string{field_path} + ".polygon"),
      tolerance,
  };
}

[[nodiscard]] GoalTarget DecodeGoalTarget(
    const Json& value,
    std::string_view field_path) {
  if (!value.is_object() || !value.contains("kind")) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "goal target requires a kind discriminator");
  }
  const std::string kind =
      DecodeString(value.at("kind"), std::string{field_path} + ".kind");
  if (kind == "POINT_WITH_TOLERANCE") {
    return DecodePointGoal(value, field_path);
  }
  if (kind == "PLANAR_CONVEX_REGION") {
    return DecodePlanarRegionGoal(value, field_path);
  }
  Fail(ErrorCode::kSchemaMismatch,
       std::string{field_path} + ".kind",
       "unknown goal target kind");
}

[[nodiscard]] Json EncodeGoalTarget(const GoalTarget& target) {
  return std::visit(
      [](const auto& concrete) -> Json {
        using T = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<T, PointGoal>) {
          return {
              {"kind", "POINT_WITH_TOLERANCE"},
              {"position_m", EncodeVec3(concrete.position_m)},
              {"position_tolerance_m", concrete.position_tolerance_m},
          };
        } else {
          return {
              {"kind", "PLANAR_CONVEX_REGION"},
              {"plane", EncodeLandingPlane(concrete.plane)},
              {"polygon", EncodeConvexPolygonUv(concrete.polygon)},
              {"normal_tolerance_m", concrete.normal_tolerance_m},
          };
        }
      },
      target);
}

[[nodiscard]] MetadataEntry DecodeMetadataEntry(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value, field_path, {"key", "value"});
  const Json& wire_value = Member(value, "value", field_path);
  MetadataValue decoded_value;
  if (wire_value.is_string()) {
    const std::string text = wire_value.get<std::string>();
    if (text.size() > 256U) {
      Fail(ErrorCode::kSchemaMismatch,
           std::string{field_path} + ".value",
           "metadata string exceeds schema maximum");
    }
    decoded_value = text;
  } else if (wire_value.is_boolean()) {
    decoded_value = wire_value.get<bool>();
  } else if (wire_value.is_number()) {
    decoded_value = DecodeNumber(
        wire_value, std::string{field_path} + ".value");
  } else {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".value",
         "unsupported metadata value type");
  }
  return {
      DecodeIdentifier(Member(value, "key", field_path),
                       std::string{field_path} + ".key"),
      std::move(decoded_value),
  };
}

[[nodiscard]] Json EncodeMetadataEntry(
    const MetadataEntry& value) {
  Json wire_value = std::visit(
      [](const auto& concrete) -> Json { return Json{concrete}; },
      value.value);
  return {{"key", value.key}, {"value", std::move(wire_value)}};
}

[[nodiscard]] GoalRegion DecodeGoalRegion(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"goal_id", "target", "task_metadata"},
              {"optional_yaw_interval", "mission_direction_hint"});
  GoalRegion result{
      .goal_id = DecodeIdentifier(
          Member(value, "goal_id", field_path),
          std::string{field_path} + ".goal_id"),
      .target = DecodeGoalTarget(
          Member(value, "target", field_path),
          std::string{field_path} + ".target"),
  };
  if (value.contains("optional_yaw_interval")) {
    result.optional_yaw_interval = DecodeCircularYawInterval(
        value.at("optional_yaw_interval"),
        std::string{field_path} + ".optional_yaw_interval");
  }
  if (value.contains("mission_direction_hint")) {
    result.mission_direction_hint = DecodeVec3(
        value.at("mission_direction_hint"),
        std::string{field_path} + ".mission_direction_hint");
  }
  const Json& metadata = Member(value, "task_metadata", field_path);
  if (!metadata.is_array() || metadata.size() > 32U) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".task_metadata",
         "task metadata must be a bounded array");
  }
  result.task_metadata.reserve(metadata.size());
  for (std::size_t index = 0; index < metadata.size(); ++index) {
    result.task_metadata.push_back(DecodeMetadataEntry(
        metadata[index],
        std::string{field_path} + ".task_metadata[" +
            std::to_string(index) + "]"));
  }
  return result;
}

[[nodiscard]] Json EncodeGoalRegion(const GoalRegion& value) {
  Json metadata = Json::array();
  for (const MetadataEntry& entry : value.task_metadata) {
    metadata.push_back(EncodeMetadataEntry(entry));
  }
  Json result{
      {"goal_id", value.goal_id},
      {"target", EncodeGoalTarget(value.target)},
      {"task_metadata", std::move(metadata)},
  };
  if (value.optional_yaw_interval.has_value()) {
    result["optional_yaw_interval"] =
        EncodeCircularYawInterval(*value.optional_yaw_interval);
  }
  if (value.mission_direction_hint.has_value()) {
    result["mission_direction_hint"] =
        EncodeVec3(*value.mission_direction_hint);
  }
  return result;
}

struct MapSnapshotWireIdentity final {
  ContentRef snapshot_ref;
  std::string immutable_data_handle;
};

[[nodiscard]] MapSnapshotWireIdentity DecodeMapSnapshotWireIdentity(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"snapshot_ref",
               "frame_id",
               "source_time",
               "map_revision",
               "bounds",
               "resolution_m",
               "layer_manifest",
               "immutable_data_handle"});
  const ContentRef snapshot_ref = DecodeContentRef(
      Member(value, "snapshot_ref", field_path),
      std::string{field_path} + ".snapshot_ref");
  static_cast<void>(DecodeIdentifier(
      Member(value, "frame_id", field_path),
      std::string{field_path} + ".frame_id"));
  static_cast<void>(DecodeClockStamp(
      Member(value, "source_time", field_path),
      std::string{field_path} + ".source_time"));
  static_cast<void>(DecodeRevision(
      Member(value, "map_revision", field_path),
      std::string{field_path} + ".map_revision"));

  const Json& bounds = Member(value, "bounds", field_path);
  CheckObject(bounds,
              std::string{field_path} + ".bounds",
              {"minimum_m", "maximum_m"});
  static_cast<void>(DecodeVec3(
      Member(bounds, "minimum_m", std::string{field_path} + ".bounds"),
      std::string{field_path} + ".bounds.minimum_m"));
  static_cast<void>(DecodeVec3(
      Member(bounds, "maximum_m", std::string{field_path} + ".bounds"),
      std::string{field_path} + ".bounds.maximum_m"));

  const double resolution =
      DecodeNumber(Member(value, "resolution_m", field_path),
                   std::string{field_path} + ".resolution_m");
  if (resolution <= 0.0) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".resolution_m",
         "map resolution must be positive");
  }

  const Json& layers = Member(value, "layer_manifest", field_path);
  if (!layers.is_array() || layers.empty() || layers.size() > 32U) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".layer_manifest",
         "layer manifest size is outside schema bounds");
  }
  constexpr std::array<std::string_view, 8> kLayerNames{
      "KNOWN_MASK",
      "ELEVATION",
      "TERRAIN_NORMAL",
      "ROUGHNESS",
      "HARD_OBSTACLE",
      "CONFIDENCE",
      "ESDF",
      "STATIC_SPEED_LIMIT",
  };
  for (std::size_t index = 0; index < layers.size(); ++index) {
    const std::string layer_path =
        std::string{field_path} + ".layer_manifest[" +
        std::to_string(index) + "]";
    CheckObject(layers[index],
                layer_path,
                {"layer_name", "content_ref"});
    const std::string layer_name = DecodeString(
        Member(layers[index], "layer_name", layer_path),
        layer_path + ".layer_name");
    if (std::find(kLayerNames.begin(), kLayerNames.end(), layer_name) ==
        kLayerNames.end()) {
      Fail(ErrorCode::kSchemaMismatch,
           layer_path + ".layer_name",
           "unknown map layer name");
    }
    static_cast<void>(DecodeContentRef(
        Member(layers[index], "content_ref", layer_path),
        layer_path + ".content_ref"));
  }

  return {
      snapshot_ref,
      DecodeIdentifier(Member(value, "immutable_data_handle", field_path),
                       std::string{field_path} +
                           ".immutable_data_handle"),
  };
}

[[nodiscard]] ControllerStatus DecodeControllerStatus(
    const Json& value,
    std::string_view field_path) {
  const std::string text = DecodeString(value, field_path);
  if (text == "READY") {
    return ControllerStatus::kReady;
  }
  if (text == "EXECUTING") {
    return ControllerStatus::kExecuting;
  }
  if (text == "HOLDING") {
    return ControllerStatus::kHolding;
  }
  if (text == "COMMITTED") {
    return ControllerStatus::kCommitted;
  }
  if (text == "IN_FLIGHT") {
    return ControllerStatus::kInFlight;
  }
  if (text == "FAULT") {
    return ControllerStatus::kFault;
  }
  Fail(ErrorCode::kSchemaMismatch, field_path, "unknown controller status");
}

[[nodiscard]] std::string_view EncodeControllerStatus(
    ControllerStatus value) {
  switch (value) {
    case ControllerStatus::kReady:
      return "READY";
    case ControllerStatus::kExecuting:
      return "EXECUTING";
    case ControllerStatus::kHolding:
      return "HOLDING";
    case ControllerStatus::kCommitted:
      return "COMMITTED";
    case ControllerStatus::kInFlight:
      return "IN_FLIGHT";
    case ControllerStatus::kFault:
      return "FAULT";
  }
  throw std::invalid_argument{"unknown ControllerStatus"};
}

[[nodiscard]] JumpExecutionState DecodeJumpExecutionState(
    const Json& value,
    std::string_view field_path) {
  const std::string text = DecodeString(value, field_path);
  if (text == "GROUND_HOLD") {
    return JumpExecutionState::kGroundHold;
  }
  if (text == "JUMP_READY") {
    return JumpExecutionState::kJumpReady;
  }
  if (text == "JUMP_COMMITTED") {
    return JumpExecutionState::kJumpCommitted;
  }
  if (text == "IN_FLIGHT") {
    return JumpExecutionState::kInFlight;
  }
  if (text == "LANDED_HOLD") {
    return JumpExecutionState::kLandedHold;
  }
  Fail(ErrorCode::kSchemaMismatch, field_path, "unknown jump state");
}

[[nodiscard]] std::string_view EncodeJumpExecutionState(
    JumpExecutionState value) {
  switch (value) {
    case JumpExecutionState::kGroundHold:
      return "GROUND_HOLD";
    case JumpExecutionState::kJumpReady:
      return "JUMP_READY";
    case JumpExecutionState::kJumpCommitted:
      return "JUMP_COMMITTED";
    case JumpExecutionState::kInFlight:
      return "IN_FLIGHT";
    case JumpExecutionState::kLandedHold:
      return "LANDED_HOLD";
  }
  throw std::invalid_argument{"unknown JumpExecutionState"};
}

[[nodiscard]] ExecutionCursor DecodeExecutionCursor(
    const Json& value,
    std::string_view field_path) {
  if (!value.is_object() || !value.contains("kind")) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "execution cursor requires kind");
  }
  const std::string kind =
      DecodeString(value.at("kind"), std::string{field_path} + ".kind");
  if (kind == "TIME_OFFSET") {
    CheckObject(value,
                field_path,
                {"kind", "offset_ns"},
                {"segment_id"});
    TimeExecutionCursor cursor{
        DurationNanoseconds{ParseNanoseconds(
            Member(value, "offset_ns", field_path),
            std::string{field_path} + ".offset_ns",
            false)},
        std::nullopt,
    };
    if (value.contains("segment_id")) {
      cursor.segment_id = DecodeIdentifier(
          value.at("segment_id"),
          std::string{field_path} + ".segment_id");
    }
    return cursor;
  }
  if (kind == "JUMP_STATE") {
    CheckObject(value,
                field_path,
                {"kind", "jump_state"},
                {"boundary_id"});
    JumpExecutionCursor cursor{
        DecodeJumpExecutionState(
            Member(value, "jump_state", field_path),
            std::string{field_path} + ".jump_state"),
        std::nullopt,
    };
    if (value.contains("boundary_id")) {
      cursor.boundary_id = DecodeIdentifier(
          value.at("boundary_id"),
          std::string{field_path} + ".boundary_id");
    }
    return cursor;
  }
  Fail(ErrorCode::kSchemaMismatch,
       std::string{field_path} + ".kind",
       "unknown execution cursor kind");
}

[[nodiscard]] Json EncodeExecutionCursor(
    const ExecutionCursor& value) {
  return std::visit(
      [](const auto& concrete) -> Json {
        using T = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<T, TimeExecutionCursor>) {
          Json result{
              {"kind", "TIME_OFFSET"},
              {"offset_ns",
               EncodeNonNegativeNanoseconds(
                   concrete.offset.value, "execution_cursor.offset_ns")},
          };
          if (concrete.segment_id.has_value()) {
            result["segment_id"] = *concrete.segment_id;
          }
          return result;
        } else {
          Json result{
              {"kind", "JUMP_STATE"},
              {"jump_state",
               EncodeJumpExecutionState(concrete.jump_state)},
          };
          if (concrete.boundary_id.has_value()) {
            result["boundary_id"] = *concrete.boundary_id;
          }
          return result;
        }
      },
      value);
}

[[nodiscard]] CommitBoundary DecodeCommitBoundary(
    const Json& value,
    std::string_view field_path) {
  if (!value.is_object() || !value.contains("kind")) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "commit boundary requires kind");
  }
  const std::string kind =
      DecodeString(value.at("kind"), std::string{field_path} + ".kind");
  if (kind == "TIME") {
    CheckObject(
        value, field_path, {"kind", "committed_until_offset_ns"});
    return TimeCommitBoundary{DurationNanoseconds{ParseNanoseconds(
        Member(value, "committed_until_offset_ns", field_path),
        std::string{field_path} + ".committed_until_offset_ns",
        false)}};
  }
  if (kind == "JUMP") {
    CheckObject(value, field_path, {"kind", "boundary_id", "locked"});
    return JumpCommitBoundary{
        DecodeIdentifier(Member(value, "boundary_id", field_path),
                         std::string{field_path} + ".boundary_id"),
        DecodeBoolean(Member(value, "locked", field_path),
                      std::string{field_path} + ".locked"),
    };
  }
  Fail(ErrorCode::kSchemaMismatch,
       std::string{field_path} + ".kind",
       "unknown commit boundary kind");
}

[[nodiscard]] Json EncodeCommitBoundary(
    const CommitBoundary& value) {
  return std::visit(
      [](const auto& concrete) -> Json {
        using T = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<T, TimeCommitBoundary>) {
          return {
              {"kind", "TIME"},
              {"committed_until_offset_ns",
               EncodeNonNegativeNanoseconds(
                   concrete.committed_until_offset.value,
                   "commit_boundary.committed_until_offset_ns")},
          };
        } else {
          return {
              {"kind", "JUMP"},
              {"boundary_id", concrete.boundary_id},
              {"locked", concrete.locked},
          };
        }
      },
      value);
}

[[nodiscard]] PreviousExecutionContext DecodePreviousExecutionContext(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"active_bundle_ref",
               "active_bundle_handle",
               "commit_boundary",
               "execution_cursor",
               "controller_status",
               "source_map_snapshot_ref",
               "source_capability_ref"});
  return {
      DecodeContentRef(
          Member(value, "active_bundle_ref", field_path),
          std::string{field_path} + ".active_bundle_ref"),
      DecodeIdentifier(
          Member(value, "active_bundle_handle", field_path),
          std::string{field_path} + ".active_bundle_handle"),
      DecodeCommitBoundary(
          Member(value, "commit_boundary", field_path),
          std::string{field_path} + ".commit_boundary"),
      DecodeExecutionCursor(
          Member(value, "execution_cursor", field_path),
          std::string{field_path} + ".execution_cursor"),
      DecodeControllerStatus(
          Member(value, "controller_status", field_path),
          std::string{field_path} + ".controller_status"),
      DecodeContentRef(
          Member(value, "source_map_snapshot_ref", field_path),
          std::string{field_path} + ".source_map_snapshot_ref"),
      DecodeContentRef(
          Member(value, "source_capability_ref", field_path),
          std::string{field_path} + ".source_capability_ref"),
  };
}

[[nodiscard]] Json EncodePreviousExecutionContext(
    const PreviousExecutionContext& value) {
  return {
      {"active_bundle_ref", EncodeContentRef(value.active_bundle_ref)},
      {"active_bundle_handle", value.active_bundle_handle},
      {"commit_boundary", EncodeCommitBoundary(value.commit_boundary)},
      {"execution_cursor", EncodeExecutionCursor(value.execution_cursor)},
      {"controller_status",
       EncodeControllerStatus(value.controller_status)},
      {"source_map_snapshot_ref",
       EncodeContentRef(value.source_map_snapshot_ref)},
      {"source_capability_ref",
       EncodeContentRef(value.source_capability_ref)},
  };
}

[[nodiscard]] PlanningRequest DecodePlanningRequestValue(
    const Json& root,
    const ContractObjectRegistry& registry) {
  CheckSchemaVersion(
      root, kPlanningRequestSchemaVersion, "$");
  CheckObject(root,
              "$",
              {"schema_version",
               "request_id",
               "request_time",
               "state_time",
               "frame_id",
               "platform_type",
               "current_state",
               "goal",
               "map_snapshot",
               "safety_capability_ref",
               "algorithm_config_ref"},
              {"previous_execution_context", "learned_cost_snapshot"});

  PlanningRequest result;
  result.request_id =
      DecodeIdentifier(root.at("request_id"), "$.request_id");
  result.request_time =
      DecodeClockStamp(root.at("request_time"), "$.request_time");
  result.state_time =
      DecodeClockStamp(root.at("state_time"), "$.state_time");
  result.frame_id = DecodeIdentifier(root.at("frame_id"), "$.frame_id");
  result.platform_type =
      DecodePlatformType(root.at("platform_type"), "$.platform_type");
  result.current_state = DecodePlatformState(
      root.at("current_state"), result.platform_type, "$.current_state");
  result.goal = DecodeGoalRegion(root.at("goal"), "$.goal");

  const MapSnapshotWireIdentity map_identity =
      DecodeMapSnapshotWireIdentity(root.at("map_snapshot"),
                                    "$.map_snapshot");
  result.map_snapshot = registry.FindMapSnapshot(
      map_identity.snapshot_ref, map_identity.immutable_data_handle);
  if (!result.map_snapshot) {
    Fail(ErrorCode::kMissingRegistryObject,
         "$.map_snapshot",
         "immutable map snapshot is absent from the registry");
  }

  const ContentRef capability_ref = DecodeContentRef(
      root.at("safety_capability_ref"), "$.safety_capability_ref");
  result.safety_capability =
      registry.FindSafetyCapability(capability_ref);
  if (!result.safety_capability) {
    Fail(ErrorCode::kMissingRegistryObject,
         "$.safety_capability_ref",
         "safety capability is absent from the registry");
  }
  if (result.safety_capability->content_ref != capability_ref) {
    Fail(ErrorCode::kInconsistentSnapshot,
         "$.safety_capability_ref",
         "registry returned a different safety capability identity");
  }

  const ContentRef config_ref = DecodeContentRef(
      root.at("algorithm_config_ref"), "$.algorithm_config_ref");
  result.algorithm_config = registry.FindAlgorithmConfig(config_ref);
  if (!result.algorithm_config) {
    Fail(ErrorCode::kMissingRegistryObject,
         "$.algorithm_config_ref",
         "algorithm config is absent from the registry");
  }
  if (result.algorithm_config->content_ref != config_ref) {
    Fail(ErrorCode::kInconsistentSnapshot,
         "$.algorithm_config_ref",
         "registry returned a different algorithm config identity");
  }

  const auto bindings =
      registry.ResolveCapabilityBindings(*result.safety_capability);
  if (!IsOk(bindings)) {
    Error error = std::get<Error>(bindings);
    if (error.field_path.empty()) {
      error.field_path = "$.safety_capability_ref";
    }
    throw DecodeFailure{std::move(error)};
  }
  result.capability_bindings =
      std::get<ResolvedCapabilityBindings>(bindings);

  if (root.contains("previous_execution_context")) {
    result.previous_execution_context = DecodePreviousExecutionContext(
        root.at("previous_execution_context"),
        "$.previous_execution_context");
  }

  if (root.contains("learned_cost_snapshot")) {
    const Json& binding = root.at("learned_cost_snapshot");
    CheckObject(binding,
                "$.learned_cost_snapshot",
                {"snapshot_ref",
                 "registry_handle",
                 "ready_before_request",
                 "hard_feasibility_authority"});
    RequireBooleanLiteral(binding.at("ready_before_request"),
                          true,
                          "$.learned_cost_snapshot.ready_before_request");
    RequireLiteral(
        binding.at("hard_feasibility_authority"),
        "NONE",
        "$.learned_cost_snapshot.hard_feasibility_authority");
    const ContentRef snapshot_ref = DecodeContentRef(
        binding.at("snapshot_ref"),
        "$.learned_cost_snapshot.snapshot_ref");
    const std::string registry_handle = DecodeIdentifier(
        binding.at("registry_handle"),
        "$.learned_cost_snapshot.registry_handle");
    auto resolved =
        registry.FindLearnedCost(snapshot_ref, registry_handle);
    if (!resolved) {
      Fail(ErrorCode::kMissingRegistryObject,
           "$.learned_cost_snapshot",
           "learned-cost snapshot is absent from the registry");
    }
    result.learned_cost_snapshot = LearnedCostSnapshotBinding{
        snapshot_ref, registry_handle, std::move(resolved)};
  }
  return result;
}

}  // namespace

Result<PlanningRequest> JsonCodec::DecodePlanningRequest(
    std::string_view payload,
    const ContractObjectRegistry& registry) {
  try {
    return DecodePlanningRequestValue(ParseStrictJson(payload), registry);
  } catch (const DecodeFailure& failure) {
    return failure.error();
  } catch (const std::exception& exception) {
    return Error{ErrorCode::kInvalidArgument, "$", exception.what()};
  }
}

Result<std::string> JsonCodec::EncodeDuration(
    DurationNanoseconds value) {
  if (value.value.count() < 0) {
    return Error{ErrorCode::kInvalidArgument,
                 "$",
                 "duration nanoseconds must be non-negative"};
  }
  return Json(EncodeNanoseconds(value.value)).dump();
}

Result<DurationNanoseconds> JsonCodec::DecodeDuration(
    std::string_view json_string) {
  try {
    return DurationNanoseconds{
        ParseNanoseconds(ParseStrictJson(json_string), "$", false)};
  } catch (const DecodeFailure& failure) {
    return failure.error();
  } catch (const std::exception& exception) {
    return Error{ErrorCode::kInvalidArgument, "$", exception.what()};
  }
}

namespace {

[[nodiscard]] TimeInterval DecodeTimeInterval(
    const Json& value,
    std::string_view field_path) {
  CheckObject(
      value, field_path, {"start_offset_ns", "end_offset_ns"});
  return {
      DurationNanoseconds{ParseNanoseconds(
          Member(value, "start_offset_ns", field_path),
          std::string{field_path} + ".start_offset_ns",
          false)},
      DurationNanoseconds{ParseNanoseconds(
          Member(value, "end_offset_ns", field_path),
          std::string{field_path} + ".end_offset_ns",
          false)},
  };
}

[[nodiscard]] Json EncodeTimeInterval(const TimeInterval& value) {
  return {
      {"start_offset_ns",
       EncodeNonNegativeNanoseconds(value.start_offset.value,
                                    "time_interval.start_offset_ns")},
      {"end_offset_ns",
       EncodeNonNegativeNanoseconds(value.end_offset.value,
                                    "time_interval.end_offset_ns")},
  };
}

[[nodiscard]] Interval DecodeInterval(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value, field_path, {"lower", "upper"});
  return {
      DecodeNumber(Member(value, "lower", field_path),
                   std::string{field_path} + ".lower"),
      DecodeNumber(Member(value, "upper", field_path),
                   std::string{field_path} + ".upper"),
  };
}

[[nodiscard]] Json EncodeInterval(const Interval& value) {
  return {{"lower", value.lower}, {"upper", value.upper}};
}

[[nodiscard]] Vector3Bounds DecodeVector3Bounds(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value, field_path, {"lower", "upper"});
  return {
      DecodeVec3(Member(value, "lower", field_path),
                 std::string{field_path} + ".lower"),
      DecodeVec3(Member(value, "upper", field_path),
                 std::string{field_path} + ".upper"),
  };
}

[[nodiscard]] Json EncodeVector3Bounds(
    const Vector3Bounds& value) {
  return {
      {"lower", EncodeVec3(value.lower)},
      {"upper", EncodeVec3(value.upper)},
  };
}

[[nodiscard]] Halfspace3 DecodeHalfspace3(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value, field_path, {"normal", "offset_m"});
  return {
      DecodeVec3(Member(value, "normal", field_path),
                 std::string{field_path} + ".normal"),
      DecodeNumber(Member(value, "offset_m", field_path),
                   std::string{field_path} + ".offset_m"),
  };
}

[[nodiscard]] Json EncodeHalfspace3(const Halfspace3& value) {
  return {
      {"normal", EncodeVec3(value.normal)},
      {"offset_m", value.offset_m},
  };
}

[[nodiscard]] ConvexPolytope3 DecodeConvexPolytope3(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value, field_path, {"representation", "halfspaces"});
  RequireLiteral(Member(value, "representation", field_path),
                 "halfspace_intersection",
                 std::string{field_path} + ".representation");
  const Json& halfspaces = Member(value, "halfspaces", field_path);
  if (!halfspaces.is_array() || halfspaces.size() < 4U ||
      halfspaces.size() > 128U) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".halfspaces",
         "halfspace count is outside schema bounds");
  }
  ConvexPolytope3 result;
  result.halfspaces.reserve(halfspaces.size());
  for (std::size_t index = 0; index < halfspaces.size(); ++index) {
    result.halfspaces.push_back(DecodeHalfspace3(
        halfspaces[index],
        std::string{field_path} + ".halfspaces[" +
            std::to_string(index) + "]"));
  }
  return result;
}

[[nodiscard]] Json EncodeConvexPolytope3(
    const ConvexPolytope3& value) {
  Json halfspaces = Json::array();
  for (const Halfspace3& halfspace : value.halfspaces) {
    halfspaces.push_back(EncodeHalfspace3(halfspace));
  }
  return {
      {"representation", "halfspace_intersection"},
      {"halfspaces", std::move(halfspaces)},
  };
}

[[nodiscard]] CubicPolynomialSegment DecodeCubicPolynomialSegment(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"start_offset_ns",
               "end_offset_ns",
               "coefficients",
               "coefficient_basis"});
  RequireLiteral(Member(value, "coefficient_basis", field_path),
                 "local_tau_seconds",
                 std::string{field_path} + ".coefficient_basis");
  const Json& coefficients = Member(value, "coefficients", field_path);
  if (!coefficients.is_array() || coefficients.size() != 4U) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".coefficients",
         "cubic polynomial requires four coefficients");
  }
  CubicPolynomialSegment result{
      DurationNanoseconds{ParseNanoseconds(
          Member(value, "start_offset_ns", field_path),
          std::string{field_path} + ".start_offset_ns",
          false)},
      DurationNanoseconds{ParseNanoseconds(
          Member(value, "end_offset_ns", field_path),
          std::string{field_path} + ".end_offset_ns",
          false)},
      {},
  };
  for (std::size_t index = 0; index < result.coefficients.size();
       ++index) {
    result.coefficients[index] = DecodeNumber(
        coefficients[index],
        std::string{field_path} + ".coefficients[" +
            std::to_string(index) + "]");
  }
  return result;
}

[[nodiscard]] Json EncodeCubicPolynomialSegment(
    const CubicPolynomialSegment& value) {
  return {
      {"start_offset_ns",
       EncodeNonNegativeNanoseconds(
           value.start_offset.value,
           "cubic_segment.start_offset_ns")},
      {"end_offset_ns",
       EncodeNonNegativeNanoseconds(
           value.end_offset.value,
           "cubic_segment.end_offset_ns")},
      {"coefficients",
       Json::array({value.coefficients[0],
                    value.coefficients[1],
                    value.coefficients[2],
                    value.coefficients[3]})},
      {"coefficient_basis", "local_tau_seconds"},
  };
}

[[nodiscard]] std::vector<CubicPolynomialSegment>
DecodeCubicSegments(const Json& value,
                    std::string_view field_path) {
  if (!value.is_array() || value.empty() || value.size() > 4096U) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "cubic segment count is outside schema bounds");
  }
  std::vector<CubicPolynomialSegment> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    result.push_back(DecodeCubicPolynomialSegment(
        value[index],
        std::string{field_path} + "[" + std::to_string(index) + "]"));
  }
  return result;
}

[[nodiscard]] Json EncodeCubicSegments(
    const std::vector<CubicPolynomialSegment>& values) {
  Json result = Json::array();
  for (const CubicPolynomialSegment& segment : values) {
    result.push_back(EncodeCubicPolynomialSegment(segment));
  }
  return result;
}

[[nodiscard]] PiecewiseCubicScalarTrajectory
DecodePiecewiseCubicScalarTrajectory(
    const Json& value,
    std::string_view field_path) {
  CheckObject(
      value, field_path, {"representation", "value_semantics", "segments"});
  RequireLiteral(Member(value, "representation", field_path),
                 "piecewise_cubic",
                 std::string{field_path} + ".representation");
  std::string semantics = DecodeString(
      Member(value, "value_semantics", field_path),
      std::string{field_path} + ".value_semantics");
  if (semantics.empty() || semantics.size() > 64U) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".value_semantics",
         "trajectory value semantics is outside schema bounds");
  }
  return {
      std::move(semantics),
      DecodeCubicSegments(Member(value, "segments", field_path),
                          std::string{field_path} + ".segments"),
  };
}

[[nodiscard]] Json EncodePiecewiseCubicScalarTrajectory(
    const PiecewiseCubicScalarTrajectory& value) {
  return {
      {"representation", "piecewise_cubic"},
      {"value_semantics", value.value_semantics},
      {"segments", EncodeCubicSegments(value.segments)},
  };
}

[[nodiscard]] MonotoneTimeScaling DecodeMonotoneTimeScaling(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"representation",
               "value_semantics",
               "monotonicity",
               "start_value",
               "end_value",
               "segments"});
  RequireLiteral(Member(value, "representation", field_path),
                 "piecewise_cubic",
                 std::string{field_path} + ".representation");
  RequireLiteral(Member(value, "value_semantics", field_path),
                 "path_parameter_s",
                 std::string{field_path} + ".value_semantics");
  RequireLiteral(Member(value, "monotonicity", field_path),
                 "nondecreasing",
                 std::string{field_path} + ".monotonicity");
  RequireNumberLiteral(Member(value, "start_value", field_path),
                       0.0,
                       std::string{field_path} + ".start_value");
  RequireNumberLiteral(Member(value, "end_value", field_path),
                       1.0,
                       std::string{field_path} + ".end_value");
  return {
      DecodeCubicSegments(Member(value, "segments", field_path),
                          std::string{field_path} + ".segments")};
}

[[nodiscard]] Json EncodeMonotoneTimeScaling(
    const MonotoneTimeScaling& value) {
  return {
      {"representation", "piecewise_cubic"},
      {"value_semantics", "path_parameter_s"},
      {"monotonicity", "nondecreasing"},
      {"start_value", 0},
      {"end_value", 1},
      {"segments", EncodeCubicSegments(value.segments)},
  };
}

[[nodiscard]] ClampedCubicBSplinePath DecodeBsplinePath(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"representation",
               "degree",
               "clamped",
               "parameter_start",
               "parameter_end",
               "knots",
               "control_points"});
  RequireLiteral(Member(value, "representation", field_path),
                 "clamped_cubic_bspline",
                 std::string{field_path} + ".representation");
  RequireNumberLiteral(Member(value, "degree", field_path),
                       3.0,
                       std::string{field_path} + ".degree");
  RequireBooleanLiteral(Member(value, "clamped", field_path),
                        true,
                        std::string{field_path} + ".clamped");
  RequireNumberLiteral(Member(value, "parameter_start", field_path),
                       0.0,
                       std::string{field_path} + ".parameter_start");
  RequireNumberLiteral(Member(value, "parameter_end", field_path),
                       1.0,
                       std::string{field_path} + ".parameter_end");
  const Json& knots = Member(value, "knots", field_path);
  const Json& control_points = Member(value, "control_points", field_path);
  if (!knots.is_array() || knots.size() < 8U || knots.size() > 4100U) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".knots",
         "B-spline knot count is outside schema bounds");
  }
  if (!control_points.is_array() || control_points.size() < 4U ||
      control_points.size() > 4096U) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".control_points",
         "B-spline control-point count is outside schema bounds");
  }
  ClampedCubicBSplinePath result;
  result.knots.reserve(knots.size());
  for (std::size_t index = 0; index < knots.size(); ++index) {
    result.knots.push_back(DecodeNumber(
        knots[index],
        std::string{field_path} + ".knots[" +
            std::to_string(index) + "]"));
  }
  result.control_points.reserve(control_points.size());
  for (std::size_t index = 0; index < control_points.size(); ++index) {
    result.control_points.push_back(DecodePose(
        control_points[index],
        std::string{field_path} + ".control_points[" +
            std::to_string(index) + "]"));
  }
  return result;
}

[[nodiscard]] Json EncodeBsplinePath(
    const ClampedCubicBSplinePath& value) {
  Json knots = Json::array();
  for (double knot : value.knots) {
    knots.push_back(knot);
  }
  Json control_points = Json::array();
  for (const PoseXyzYaw& pose : value.control_points) {
    control_points.push_back(EncodePose(pose));
  }
  return {
      {"representation", "clamped_cubic_bspline"},
      {"degree", 3},
      {"clamped", true},
      {"parameter_start", 0},
      {"parameter_end", 1},
      {"knots", std::move(knots)},
      {"control_points", std::move(control_points)},
  };
}

[[nodiscard]] PrimitiveKind DecodePrimitiveKind(
    const Json& value,
    std::string_view field_path) {
  const std::string text = DecodeString(value, field_path);
  if (text == "DRIVE_FORWARD") {
    return PrimitiveKind::kDriveForward;
  }
  if (text == "DRIVE_REVERSE") {
    return PrimitiveKind::kDriveReverse;
  }
  if (text == "SPIN_CW") {
    return PrimitiveKind::kSpinCw;
  }
  if (text == "SPIN_CCW") {
    return PrimitiveKind::kSpinCcw;
  }
  if (text == "STOP_AND_SWITCH") {
    return PrimitiveKind::kStopAndSwitch;
  }
  if (text == "BODY_TRANSLATION") {
    return PrimitiveKind::kBodyTranslation;
  }
  if (text == "BODY_SPIN") {
    return PrimitiveKind::kBodySpin;
  }
  if (text == "BODY_COUPLED") {
    return PrimitiveKind::kBodyCoupled;
  }
  Fail(ErrorCode::kSchemaMismatch, field_path, "unknown primitive kind");
}

[[nodiscard]] std::string_view EncodePrimitiveKind(
    PrimitiveKind value) {
  switch (value) {
    case PrimitiveKind::kDriveForward:
      return "DRIVE_FORWARD";
    case PrimitiveKind::kDriveReverse:
      return "DRIVE_REVERSE";
    case PrimitiveKind::kSpinCw:
      return "SPIN_CW";
    case PrimitiveKind::kSpinCcw:
      return "SPIN_CCW";
    case PrimitiveKind::kStopAndSwitch:
      return "STOP_AND_SWITCH";
    case PrimitiveKind::kBodyTranslation:
      return "BODY_TRANSLATION";
    case PrimitiveKind::kBodySpin:
      return "BODY_SPIN";
    case PrimitiveKind::kBodyCoupled:
      return "BODY_COUPLED";
  }
  throw std::invalid_argument{"unknown PrimitiveKind"};
}

[[nodiscard]] ValidatedPrimitive DecodeValidatedPrimitive(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"primitive_id",
               "capability_primitive_id",
               "primitive_kind",
               "start_pose",
               "end_pose",
               "nominal_duration_ns",
               "validation_ref"});
  return {
      DecodeIdentifier(Member(value, "primitive_id", field_path),
                       std::string{field_path} + ".primitive_id"),
      DecodeIdentifier(
          Member(value, "capability_primitive_id", field_path),
          std::string{field_path} + ".capability_primitive_id"),
      DecodePrimitiveKind(
          Member(value, "primitive_kind", field_path),
          std::string{field_path} + ".primitive_kind"),
      DecodePose(Member(value, "start_pose", field_path),
                 std::string{field_path} + ".start_pose"),
      DecodePose(Member(value, "end_pose", field_path),
                 std::string{field_path} + ".end_pose"),
      DurationNanoseconds{ParseNanoseconds(
          Member(value, "nominal_duration_ns", field_path),
          std::string{field_path} + ".nominal_duration_ns",
          false)},
      DecodeContentRef(
          Member(value, "validation_ref", field_path),
          std::string{field_path} + ".validation_ref"),
  };
}

[[nodiscard]] Json EncodeValidatedPrimitive(
    const ValidatedPrimitive& value) {
  return {
      {"primitive_id", value.primitive_id},
      {"capability_primitive_id", value.capability_primitive_id},
      {"primitive_kind", EncodePrimitiveKind(value.primitive_kind)},
      {"start_pose", EncodePose(value.start_pose)},
      {"end_pose", EncodePose(value.end_pose)},
      {"nominal_duration_ns",
       EncodeNonNegativeNanoseconds(
           value.nominal_duration.value,
           "validated_primitive.nominal_duration_ns")},
      {"validation_ref", EncodeContentRef(value.validation_ref)},
  };
}

[[nodiscard]] ValidatedPrimitiveChain
DecodeValidatedPrimitiveChain(const Json& value,
                              std::string_view field_path) {
  CheckObject(value, field_path, {"representation", "primitives"});
  RequireLiteral(Member(value, "representation", field_path),
                 "validated_primitive_chain",
                 std::string{field_path} + ".representation");
  const Json& primitives = Member(value, "primitives", field_path);
  if (!primitives.is_array() || primitives.empty() ||
      primitives.size() > 4096U) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".primitives",
         "primitive-chain size is outside schema bounds");
  }
  ValidatedPrimitiveChain result;
  result.primitives.reserve(primitives.size());
  for (std::size_t index = 0; index < primitives.size(); ++index) {
    result.primitives.push_back(DecodeValidatedPrimitive(
        primitives[index],
        std::string{field_path} + ".primitives[" +
            std::to_string(index) + "]"));
  }
  return result;
}

[[nodiscard]] Json EncodeValidatedPrimitiveChain(
    const ValidatedPrimitiveChain& value) {
  Json primitives = Json::array();
  for (const ValidatedPrimitive& primitive : value.primitives) {
    primitives.push_back(EncodeValidatedPrimitive(primitive));
  }
  return {
      {"representation", "validated_primitive_chain"},
      {"primitives", std::move(primitives)},
  };
}

[[nodiscard]] GeometricPath DecodeGeometricPath(
    const Json& value,
    std::string_view field_path) {
  if (!value.is_object() || !value.contains("representation")) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "geometric path requires a representation discriminator");
  }
  const std::string representation = DecodeString(
      value.at("representation"),
      std::string{field_path} + ".representation");
  if (representation == "clamped_cubic_bspline") {
    return DecodeBsplinePath(value, field_path);
  }
  if (representation == "validated_primitive_chain") {
    return DecodeValidatedPrimitiveChain(value, field_path);
  }
  Fail(ErrorCode::kSchemaMismatch,
       std::string{field_path} + ".representation",
       "unknown geometric-path representation");
}

[[nodiscard]] Json EncodeGeometricPath(const GeometricPath& value) {
  return std::visit(
      [](const auto& concrete) -> Json {
        using T = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<T, ClampedCubicBSplinePath>) {
          return EncodeBsplinePath(concrete);
        } else {
          return EncodeValidatedPrimitiveChain(concrete);
        }
      },
      value);
}

[[nodiscard]] SafeStopAnchor DecodeSafeStopAnchor(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"anchor_id",
               "pose",
               "target_linear_velocity_mps",
               "target_yaw_rate_radps",
               "terrain_certification_ref"});
  RequireNumberLiteral(
      Member(value, "target_linear_velocity_mps", field_path),
      0.0,
      std::string{field_path} + ".target_linear_velocity_mps");
  RequireNumberLiteral(
      Member(value, "target_yaw_rate_radps", field_path),
      0.0,
      std::string{field_path} + ".target_yaw_rate_radps");
  return {
      DecodeIdentifier(Member(value, "anchor_id", field_path),
                       std::string{field_path} + ".anchor_id"),
      DecodePose(Member(value, "pose", field_path),
                 std::string{field_path} + ".pose"),
      0.0,
      0.0,
      DecodeContentRef(
          Member(value, "terrain_certification_ref", field_path),
          std::string{field_path} + ".terrain_certification_ref"),
  };
}

[[nodiscard]] Json EncodeSafeStopAnchor(
    const SafeStopAnchor& value) {
  return {
      {"anchor_id", value.anchor_id},
      {"pose", EncodePose(value.pose)},
      {"target_linear_velocity_mps", value.target_linear_velocity_mps},
      {"target_yaw_rate_radps", value.target_yaw_rate_radps},
      {"terrain_certification_ref",
       EncodeContentRef(value.terrain_certification_ref)},
  };
}

[[nodiscard]] DerivedKinematicCaches DecodeDerivedKinematicCaches(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"cache_authority",
               "consistency_tolerance",
               "signed_body_forward_speed_mps",
               "yaw_rate_radps"});
  RequireLiteral(Member(value, "cache_authority", field_path),
                 "DERIVED_NON_AUTHORITATIVE",
                 std::string{field_path} + ".cache_authority");
  const double tolerance =
      DecodeNumber(Member(value, "consistency_tolerance", field_path),
                   std::string{field_path} +
                       ".consistency_tolerance");
  if (tolerance < 0.0) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".consistency_tolerance",
         "cache tolerance must be non-negative");
  }
  auto speed = DecodePiecewiseCubicScalarTrajectory(
      Member(value, "signed_body_forward_speed_mps", field_path),
      std::string{field_path} + ".signed_body_forward_speed_mps");
  if (speed.value_semantics != "signed_body_forward_speed_mps") {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} +
             ".signed_body_forward_speed_mps.value_semantics",
         "wrong derived-cache value semantics");
  }
  auto yaw_rate = DecodePiecewiseCubicScalarTrajectory(
      Member(value, "yaw_rate_radps", field_path),
      std::string{field_path} + ".yaw_rate_radps");
  if (yaw_rate.value_semantics != "yaw_rate_radps") {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".yaw_rate_radps.value_semantics",
         "wrong derived-cache value semantics");
  }
  return {tolerance, std::move(speed), std::move(yaw_rate)};
}

[[nodiscard]] Json EncodeDerivedKinematicCaches(
    const DerivedKinematicCaches& value) {
  return {
      {"cache_authority", "DERIVED_NON_AUTHORITATIVE"},
      {"consistency_tolerance", value.consistency_tolerance},
      {"signed_body_forward_speed_mps",
       EncodePiecewiseCubicScalarTrajectory(
           value.signed_body_forward_speed_mps)},
      {"yaw_rate_radps",
       EncodePiecewiseCubicScalarTrajectory(value.yaw_rate_radps)},
  };
}

[[nodiscard]] DriveDirection DecodeDriveDirection(
    const Json& value,
    std::string_view field_path) {
  const std::string text = DecodeString(value, field_path);
  if (text == "FORWARD") {
    return DriveDirection::kForward;
  }
  if (text == "REVERSE") {
    return DriveDirection::kReverse;
  }
  Fail(ErrorCode::kSchemaMismatch, field_path, "unknown drive direction");
}

[[nodiscard]] std::string_view EncodeDriveDirection(
    DriveDirection value) {
  switch (value) {
    case DriveDirection::kForward:
      return "FORWARD";
    case DriveDirection::kReverse:
      return "REVERSE";
  }
  throw std::invalid_argument{"unknown DriveDirection"};
}

[[nodiscard]] DriveSegment DecodeDriveSegment(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"segment_type",
               "segment_id",
               "time_interval",
               "direction",
               "geometric_path",
               "time_scaling"},
              {"derived_caches"});
  RequireLiteral(Member(value, "segment_type", field_path),
                 "DRIVE",
                 std::string{field_path} + ".segment_type");
  DriveSegment result{
      DecodeIdentifier(Member(value, "segment_id", field_path),
                       std::string{field_path} + ".segment_id"),
      DecodeTimeInterval(
          Member(value, "time_interval", field_path),
          std::string{field_path} + ".time_interval"),
      DecodeDriveDirection(Member(value, "direction", field_path),
                           std::string{field_path} + ".direction"),
      DecodeGeometricPath(
          Member(value, "geometric_path", field_path),
          std::string{field_path} + ".geometric_path"),
      DecodeMonotoneTimeScaling(
          Member(value, "time_scaling", field_path),
          std::string{field_path} + ".time_scaling"),
      std::nullopt,
  };
  if (value.contains("derived_caches")) {
    result.derived_caches = DecodeDerivedKinematicCaches(
        value.at("derived_caches"),
        std::string{field_path} + ".derived_caches");
  }
  return result;
}

[[nodiscard]] Json EncodeDriveSegment(const DriveSegment& value) {
  Json result{
      {"segment_type", "DRIVE"},
      {"segment_id", value.segment_id},
      {"time_interval", EncodeTimeInterval(value.time_interval)},
      {"direction", EncodeDriveDirection(value.direction)},
      {"geometric_path", EncodeGeometricPath(value.geometric_path)},
      {"time_scaling", EncodeMonotoneTimeScaling(value.time_scaling)},
  };
  if (value.derived_caches.has_value()) {
    result["derived_caches"] =
        EncodeDerivedKinematicCaches(*value.derived_caches);
  }
  return result;
}

[[nodiscard]] SpinSegment DecodeSpinSegment(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"segment_type",
               "segment_id",
               "time_interval",
               "fixed_position_m",
               "unwrapped_yaw_rad"});
  RequireLiteral(Member(value, "segment_type", field_path),
                 "SPIN",
                 std::string{field_path} + ".segment_type");
  auto yaw = DecodePiecewiseCubicScalarTrajectory(
      Member(value, "unwrapped_yaw_rad", field_path),
      std::string{field_path} + ".unwrapped_yaw_rad");
  if (yaw.value_semantics != "unwrapped_yaw_rad") {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} +
             ".unwrapped_yaw_rad.value_semantics",
         "wrong spin-yaw value semantics");
  }
  return {
      DecodeIdentifier(Member(value, "segment_id", field_path),
                       std::string{field_path} + ".segment_id"),
      DecodeTimeInterval(
          Member(value, "time_interval", field_path),
          std::string{field_path} + ".time_interval"),
      DecodeVec3(Member(value, "fixed_position_m", field_path),
                 std::string{field_path} + ".fixed_position_m"),
      std::move(yaw),
  };
}

[[nodiscard]] Json EncodeSpinSegment(const SpinSegment& value) {
  return {
      {"segment_type", "SPIN"},
      {"segment_id", value.segment_id},
      {"time_interval", EncodeTimeInterval(value.time_interval)},
      {"fixed_position_m", EncodeVec3(value.fixed_position_m)},
      {"unwrapped_yaw_rad",
       EncodePiecewiseCubicScalarTrajectory(value.unwrapped_yaw_rad)},
  };
}

[[nodiscard]] WheeledSegment DecodeWheeledSegment(
    const Json& value,
    std::string_view field_path) {
  if (!value.is_object() || !value.contains("segment_type")) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "wheeled segment requires segment_type");
  }
  const std::string type = DecodeString(
      value.at("segment_type"),
      std::string{field_path} + ".segment_type");
  if (type == "DRIVE") {
    return DecodeDriveSegment(value, field_path);
  }
  if (type == "SPIN") {
    return DecodeSpinSegment(value, field_path);
  }
  Fail(ErrorCode::kSchemaMismatch,
       std::string{field_path} + ".segment_type",
       "unknown wheeled segment type");
}

[[nodiscard]] Json EncodeWheeledSegment(
    const WheeledSegment& value) {
  return std::visit(
      [](const auto& concrete) -> Json {
        using T = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<T, DriveSegment>) {
          return EncodeDriveSegment(concrete);
        } else {
          return EncodeSpinSegment(concrete);
        }
      },
      value);
}

[[nodiscard]] WheeledReference DecodeWheeledReference(
    const Json& value,
    std::string_view field_path) {
  CheckSchemaVersion(value, kWheeledReferenceSchemaVersion, field_path);
  CheckObject(value,
              field_path,
              {"schema_version",
               "reference_id",
               "reference_hash",
               "platform_type",
               "reference_time_origin",
               "segments",
               "safe_stop_anchor"});
  RequireLiteral(Member(value, "platform_type", field_path),
                 "WHEELED",
                 std::string{field_path} + ".platform_type");
  const Json& segments = Member(value, "segments", field_path);
  if (!segments.is_array() || segments.empty() ||
      segments.size() > 4096U) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".segments",
         "wheeled segment count is outside schema bounds");
  }
  WheeledReference result{
      .reference_id = DecodeIdentifier(
          Member(value, "reference_id", field_path),
          std::string{field_path} + ".reference_id"),
      .reference_hash = DecodeHash(
          Member(value, "reference_hash", field_path),
          std::string{field_path} + ".reference_hash"),
      .reference_time_origin = DecodeClockStamp(
          Member(value, "reference_time_origin", field_path),
          std::string{field_path} + ".reference_time_origin"),
  };
  result.segments.reserve(segments.size());
  for (std::size_t index = 0; index < segments.size(); ++index) {
    result.segments.push_back(DecodeWheeledSegment(
        segments[index],
        std::string{field_path} + ".segments[" +
            std::to_string(index) + "]"));
  }
  result.safe_stop_anchor = DecodeSafeStopAnchor(
      Member(value, "safe_stop_anchor", field_path),
      std::string{field_path} + ".safe_stop_anchor");
  return result;
}

[[nodiscard]] Json EncodeWheeledReference(
    const WheeledReference& value) {
  Json segments = Json::array();
  for (const WheeledSegment& segment : value.segments) {
    segments.push_back(EncodeWheeledSegment(segment));
  }
  return {
      {"schema_version", kWheeledReferenceSchemaVersion},
      {"reference_id", value.reference_id},
      {"reference_hash", value.reference_hash},
      {"platform_type", "WHEELED"},
      {"reference_time_origin",
       EncodeClockStamp(value.reference_time_origin)},
      {"segments", std::move(segments)},
      {"safe_stop_anchor", EncodeSafeStopAnchor(value.safe_stop_anchor)},
  };
}

[[nodiscard]] BodyFrameVelocityEnvelope
DecodeBodyFrameVelocityEnvelope(const Json& value,
                                std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"forward_mps",
               "lateral_mps",
               "vertical_mps",
               "yaw_rate_radps"});
  return {
      DecodeInterval(Member(value, "forward_mps", field_path),
                     std::string{field_path} + ".forward_mps"),
      DecodeInterval(Member(value, "lateral_mps", field_path),
                     std::string{field_path} + ".lateral_mps"),
      DecodeInterval(Member(value, "vertical_mps", field_path),
                     std::string{field_path} + ".vertical_mps"),
      DecodeInterval(Member(value, "yaw_rate_radps", field_path),
                     std::string{field_path} + ".yaw_rate_radps"),
  };
}

[[nodiscard]] Json EncodeBodyFrameVelocityEnvelope(
    const BodyFrameVelocityEnvelope& value) {
  return {
      {"forward_mps", EncodeInterval(value.forward_mps)},
      {"lateral_mps", EncodeInterval(value.lateral_mps)},
      {"vertical_mps", EncodeInterval(value.vertical_mps)},
      {"yaw_rate_radps", EncodeInterval(value.yaw_rate_radps)},
  };
}

[[nodiscard]] TerrainNormalEnvelope DecodeTerrainNormalEnvelope(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"maximum_normal_deviation_rad",
               "source_terrain_certification_ref"});
  const double maximum = DecodeNumber(
      Member(value, "maximum_normal_deviation_rad", field_path),
      std::string{field_path} + ".maximum_normal_deviation_rad");
  if (maximum < 0.0 || maximum > 1.5707963267948966) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".maximum_normal_deviation_rad",
         "normal deviation is outside schema bounds");
  }
  return {
      maximum,
      DecodeContentRef(
          Member(value, "source_terrain_certification_ref", field_path),
          std::string{field_path} +
              ".source_terrain_certification_ref"),
  };
}

[[nodiscard]] Json EncodeTerrainNormalEnvelope(
    const TerrainNormalEnvelope& value) {
  return {
      {"maximum_normal_deviation_rad",
       value.maximum_normal_deviation_rad},
      {"source_terrain_certification_ref",
       EncodeContentRef(value.source_terrain_certification_ref)},
  };
}

[[nodiscard]] RollPitchDiagnosticEnvelope
DecodeRollPitchDiagnosticEnvelope(const Json& value,
                                  std::string_view field_path) {
  CheckObject(value, field_path, {"authority", "roll_rad", "pitch_rad"});
  RequireLiteral(Member(value, "authority", field_path),
                 "NON_AUTHORITATIVE_DIAGNOSTIC",
                 std::string{field_path} + ".authority");
  return {
      DecodeInterval(Member(value, "roll_rad", field_path),
                     std::string{field_path} + ".roll_rad"),
      DecodeInterval(Member(value, "pitch_rad", field_path),
                     std::string{field_path} + ".pitch_rad"),
  };
}

[[nodiscard]] Json EncodeRollPitchDiagnosticEnvelope(
    const RollPitchDiagnosticEnvelope& value) {
  return {
      {"authority", "NON_AUTHORITATIVE_DIAGNOSTIC"},
      {"roll_rad", EncodeInterval(value.roll_rad)},
      {"pitch_rad", EncodeInterval(value.pitch_rad)},
  };
}

[[nodiscard]] LeggedBodyReference DecodeLeggedReference(
    const Json& value,
    std::string_view field_path) {
  CheckSchemaVersion(value, kLeggedReferenceSchemaVersion, field_path);
  CheckObject(value,
              field_path,
              {"schema_version",
               "reference_id",
               "reference_hash",
               "platform_type",
               "reference_point_id",
               "reference_time_origin",
               "geometric_path",
               "time_scaling",
               "body_frame_velocity_envelope",
               "terrain_normal_envelope",
               "roll_pitch_diagnostic_envelope",
               "safe_stop_anchor",
               "feasibility_scope",
               "footstep_feasibility_guaranteed"});
  RequireLiteral(Member(value, "platform_type", field_path),
                 "LEGGED",
                 std::string{field_path} + ".platform_type");
  RequireLiteral(Member(value, "feasibility_scope", field_path),
                 "body_geometry_and_terrain_thresholds_only",
                 std::string{field_path} + ".feasibility_scope");
  RequireBooleanLiteral(
      Member(value, "footstep_feasibility_guaranteed", field_path),
      false,
      std::string{field_path} +
          ".footstep_feasibility_guaranteed");
  return {
      .reference_id = DecodeIdentifier(
          Member(value, "reference_id", field_path),
          std::string{field_path} + ".reference_id"),
      .reference_hash = DecodeHash(
          Member(value, "reference_hash", field_path),
          std::string{field_path} + ".reference_hash"),
      .reference_point_id = DecodeIdentifier(
          Member(value, "reference_point_id", field_path),
          std::string{field_path} + ".reference_point_id"),
      .reference_time_origin = DecodeClockStamp(
          Member(value, "reference_time_origin", field_path),
          std::string{field_path} + ".reference_time_origin"),
      .geometric_path = DecodeGeometricPath(
          Member(value, "geometric_path", field_path),
          std::string{field_path} + ".geometric_path"),
      .time_scaling = DecodeMonotoneTimeScaling(
          Member(value, "time_scaling", field_path),
          std::string{field_path} + ".time_scaling"),
      .body_frame_velocity_envelope = DecodeBodyFrameVelocityEnvelope(
          Member(value, "body_frame_velocity_envelope", field_path),
          std::string{field_path} +
              ".body_frame_velocity_envelope"),
      .terrain_normal_envelope = DecodeTerrainNormalEnvelope(
          Member(value, "terrain_normal_envelope", field_path),
          std::string{field_path} + ".terrain_normal_envelope"),
      .roll_pitch_diagnostic_envelope =
          DecodeRollPitchDiagnosticEnvelope(
              Member(value,
                     "roll_pitch_diagnostic_envelope",
                     field_path),
              std::string{field_path} +
                  ".roll_pitch_diagnostic_envelope"),
      .safe_stop_anchor = DecodeSafeStopAnchor(
          Member(value, "safe_stop_anchor", field_path),
          std::string{field_path} + ".safe_stop_anchor"),
      .feasibility_scope =
          "body_geometry_and_terrain_thresholds_only",
      .footstep_feasibility_guaranteed = false,
  };
}

[[nodiscard]] Json EncodeLeggedReference(
    const LeggedBodyReference& value) {
  return {
      {"schema_version", kLeggedReferenceSchemaVersion},
      {"reference_id", value.reference_id},
      {"reference_hash", value.reference_hash},
      {"platform_type", "LEGGED"},
      {"reference_point_id", value.reference_point_id},
      {"reference_time_origin",
       EncodeClockStamp(value.reference_time_origin)},
      {"geometric_path", EncodeGeometricPath(value.geometric_path)},
      {"time_scaling", EncodeMonotoneTimeScaling(value.time_scaling)},
      {"body_frame_velocity_envelope",
       EncodeBodyFrameVelocityEnvelope(
           value.body_frame_velocity_envelope)},
      {"terrain_normal_envelope",
       EncodeTerrainNormalEnvelope(value.terrain_normal_envelope)},
      {"roll_pitch_diagnostic_envelope",
       EncodeRollPitchDiagnosticEnvelope(
           value.roll_pitch_diagnostic_envelope)},
      {"safe_stop_anchor", EncodeSafeStopAnchor(value.safe_stop_anchor)},
      {"feasibility_scope",
       "body_geometry_and_terrain_thresholds_only"},
      {"footstep_feasibility_guaranteed", false},
  };
}

[[nodiscard]] HopperKinematicState DecodeHopperKinematicState(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"position_m",
               "orientation_body_to_frame",
               "linear_velocity_mps",
               "angular_velocity_radps"});
  return {
      DecodeVec3(Member(value, "position_m", field_path),
                 std::string{field_path} + ".position_m"),
      DecodeQuaternion(
          Member(value, "orientation_body_to_frame", field_path),
          std::string{field_path} + ".orientation_body_to_frame"),
      DecodeVec3(Member(value, "linear_velocity_mps", field_path),
                 std::string{field_path} + ".linear_velocity_mps"),
      DecodeVec3(Member(value, "angular_velocity_radps", field_path),
                 std::string{field_path} + ".angular_velocity_radps"),
  };
}

[[nodiscard]] Json EncodeHopperKinematicState(
    const HopperKinematicState& value) {
  return {
      {"position_m", EncodeVec3(value.position_m)},
      {"orientation_body_to_frame",
       EncodeQuaternion(value.orientation_body_to_frame)},
      {"linear_velocity_mps", EncodeVec3(value.linear_velocity_mps)},
      {"angular_velocity_radps",
       EncodeVec3(value.angular_velocity_radps)},
  };
}

[[nodiscard]] GroundHoldAnchor DecodeGroundHoldAnchor(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"anchor_id",
               "hold_state",
               "allowed_hold_state_error_set",
               "terrain_certification_ref"});
  return {
      DecodeIdentifier(Member(value, "anchor_id", field_path),
                       std::string{field_path} + ".anchor_id"),
      DecodeHopperKinematicState(
          Member(value, "hold_state", field_path),
          std::string{field_path} + ".hold_state"),
      DecodeHopperErrorBounds(
          Member(value, "allowed_hold_state_error_set", field_path),
          std::string{field_path} +
              ".allowed_hold_state_error_set"),
      DecodeContentRef(
          Member(value, "terrain_certification_ref", field_path),
          std::string{field_path} + ".terrain_certification_ref"),
  };
}

[[nodiscard]] Json EncodeGroundHoldAnchor(
    const GroundHoldAnchor& value) {
  return {
      {"anchor_id", value.anchor_id},
      {"hold_state", EncodeHopperKinematicState(value.hold_state)},
      {"allowed_hold_state_error_set",
       EncodeHopperErrorBounds(value.allowed_hold_state_error_set)},
      {"terrain_certification_ref",
       EncodeContentRef(value.terrain_certification_ref)},
  };
}

[[nodiscard]] NextLandingRegion DecodeNextLandingRegion(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"region_id",
               "frame_id",
               "landing_plane",
               "convex_polygon",
               "allowed_yaw_interval",
               "terrain_certification_ref",
               "inward_safety_margin_m"});
  const double margin =
      DecodeNumber(Member(value, "inward_safety_margin_m", field_path),
                   std::string{field_path} +
                       ".inward_safety_margin_m");
  if (margin < 0.0) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".inward_safety_margin_m",
         "landing-region margin must be non-negative");
  }
  return {
      DecodeIdentifier(Member(value, "region_id", field_path),
                       std::string{field_path} + ".region_id"),
      DecodeIdentifier(Member(value, "frame_id", field_path),
                       std::string{field_path} + ".frame_id"),
      DecodeLandingPlane(Member(value, "landing_plane", field_path),
                         std::string{field_path} + ".landing_plane"),
      DecodeConvexPolygonUv(
          Member(value, "convex_polygon", field_path),
          std::string{field_path} + ".convex_polygon"),
      DecodeCircularYawInterval(
          Member(value, "allowed_yaw_interval", field_path),
          std::string{field_path} + ".allowed_yaw_interval"),
      DecodeContentRef(
          Member(value, "terrain_certification_ref", field_path),
          std::string{field_path} + ".terrain_certification_ref"),
      margin,
  };
}

[[nodiscard]] Json EncodeNextLandingRegion(
    const NextLandingRegion& value) {
  return {
      {"region_id", value.region_id},
      {"frame_id", value.frame_id},
      {"landing_plane", EncodeLandingPlane(value.landing_plane)},
      {"convex_polygon", EncodeConvexPolygonUv(value.convex_polygon)},
      {"allowed_yaw_interval",
       EncodeCircularYawInterval(value.allowed_yaw_interval)},
      {"terrain_certification_ref",
       EncodeContentRef(value.terrain_certification_ref)},
      {"inward_safety_margin_m", value.inward_safety_margin_m},
  };
}

[[nodiscard]] JumpBoundary DecodeJumpBoundary(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"boundary_id",
               "lock_event",
               "nominal_launch_state",
               "allowed_launch_state_error_set",
               "gravity_model_ref",
               "ballistic_time_origin",
               "ballistic_flight_time_ns",
               "actuator_or_impulse_profile_ref"});
  RequireLiteral(Member(value, "lock_event", field_path),
                 "JUMP_BOUNDARY_LOCK",
                 std::string{field_path} + ".lock_event");
  RequireLiteral(Member(value, "ballistic_time_origin", field_path),
                 "BALLISTIC_LAUNCH_EVENT",
                 std::string{field_path} + ".ballistic_time_origin");
  return {
      .boundary_id = DecodeIdentifier(
          Member(value, "boundary_id", field_path),
          std::string{field_path} + ".boundary_id"),
      .lock_event = JumpBoundary::LockEvent::kJumpBoundaryLock,
      .nominal_launch_state = DecodeHopperKinematicState(
          Member(value, "nominal_launch_state", field_path),
          std::string{field_path} + ".nominal_launch_state"),
      .allowed_launch_state_error_set = DecodeHopperErrorBounds(
          Member(value, "allowed_launch_state_error_set", field_path),
          std::string{field_path} +
              ".allowed_launch_state_error_set"),
      .gravity_model_ref = DecodeContentRef(
          Member(value, "gravity_model_ref", field_path),
          std::string{field_path} + ".gravity_model_ref"),
      .ballistic_time_origin =
          JumpBoundary::BallisticTimeOrigin::kBallisticLaunchEvent,
      .ballistic_flight_time = DurationNanoseconds{ParseNanoseconds(
          Member(value, "ballistic_flight_time_ns", field_path),
          std::string{field_path} + ".ballistic_flight_time_ns",
          false,
          true)},
      .actuator_or_impulse_profile_ref = DecodeContentRef(
          Member(value, "actuator_or_impulse_profile_ref", field_path),
          std::string{field_path} +
              ".actuator_or_impulse_profile_ref"),
  };
}

[[nodiscard]] Json EncodeJumpBoundary(const JumpBoundary& value) {
  return {
      {"boundary_id", value.boundary_id},
      {"lock_event", "JUMP_BOUNDARY_LOCK"},
      {"nominal_launch_state",
       EncodeHopperKinematicState(value.nominal_launch_state)},
      {"allowed_launch_state_error_set",
       EncodeHopperErrorBounds(value.allowed_launch_state_error_set)},
      {"gravity_model_ref", EncodeContentRef(value.gravity_model_ref)},
      {"ballistic_time_origin", "BALLISTIC_LAUNCH_EVENT"},
      {"ballistic_flight_time_ns",
       EncodePositiveNanoseconds(
           value.ballistic_flight_time.value,
           "jump_boundary.ballistic_flight_time_ns")},
      {"actuator_or_impulse_profile_ref",
       EncodeContentRef(value.actuator_or_impulse_profile_ref)},
  };
}

[[nodiscard]] PredictedLandingFootprint
DecodePredictedLandingFootprint(const Json& value,
                                std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"landing_plane",
               "convex_center_landing_polygon",
               "landing_time_window",
               "landing_velocity_bounds",
               "landing_yaw_interval",
               "source_error_model_ref",
               "outer_approximation_margin_m"});
  const double margin =
      DecodeNumber(Member(value,
                          "outer_approximation_margin_m",
                          field_path),
                   std::string{field_path} +
                       ".outer_approximation_margin_m");
  if (margin < 0.0) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} +
             ".outer_approximation_margin_m",
         "footprint margin must be non-negative");
  }
  return {
      DecodeLandingPlane(Member(value, "landing_plane", field_path),
                         std::string{field_path} + ".landing_plane"),
      DecodeConvexPolygonUv(
          Member(value, "convex_center_landing_polygon", field_path),
          std::string{field_path} +
              ".convex_center_landing_polygon"),
      DecodeTimeInterval(
          Member(value, "landing_time_window", field_path),
          std::string{field_path} + ".landing_time_window"),
      DecodeVector3Bounds(
          Member(value, "landing_velocity_bounds", field_path),
          std::string{field_path} + ".landing_velocity_bounds"),
      DecodeCircularYawInterval(
          Member(value, "landing_yaw_interval", field_path),
          std::string{field_path} + ".landing_yaw_interval"),
      DecodeContentRef(
          Member(value, "source_error_model_ref", field_path),
          std::string{field_path} + ".source_error_model_ref"),
      margin,
  };
}

[[nodiscard]] Json EncodePredictedLandingFootprint(
    const PredictedLandingFootprint& value) {
  return {
      {"landing_plane", EncodeLandingPlane(value.landing_plane)},
      {"convex_center_landing_polygon",
       EncodeConvexPolygonUv(value.convex_center_landing_polygon)},
      {"landing_time_window",
       EncodeTimeInterval(value.landing_time_window)},
      {"landing_velocity_bounds",
       EncodeVector3Bounds(value.landing_velocity_bounds)},
      {"landing_yaw_interval",
       EncodeCircularYawInterval(value.landing_yaw_interval)},
      {"source_error_model_ref",
       EncodeContentRef(value.source_error_model_ref)},
      {"outer_approximation_margin_m",
       value.outer_approximation_margin_m},
  };
}

[[nodiscard]] FlightTubeSection DecodeFlightTubeSection(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value, field_path, {"time_interval", "envelope"});
  return {
      DecodeTimeInterval(Member(value, "time_interval", field_path),
                         std::string{field_path} + ".time_interval"),
      DecodeConvexPolytope3(Member(value, "envelope", field_path),
                            std::string{field_path} + ".envelope"),
  };
}

[[nodiscard]] Json EncodeFlightTubeSection(
    const FlightTubeSection& value) {
  return {
      {"time_interval", EncodeTimeInterval(value.time_interval)},
      {"envelope", EncodeConvexPolytope3(value.envelope)},
  };
}

[[nodiscard]] CertifiedFlightTube DecodeCertifiedFlightTube(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"frame_id",
               "sections",
               "source_map_snapshot_ref",
               "body_rotation_envelope_ref",
               "error_model_ref",
               "minimum_certified_clearance_m"});
  const Json& sections = Member(value, "sections", field_path);
  if (!sections.is_array() || sections.empty() ||
      sections.size() > 4096U) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".sections",
         "flight-tube section count is outside schema bounds");
  }
  const double clearance = DecodeNumber(
      Member(value, "minimum_certified_clearance_m", field_path),
      std::string{field_path} + ".minimum_certified_clearance_m");
  if (clearance < 0.0) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} +
             ".minimum_certified_clearance_m",
         "certified clearance must be non-negative");
  }
  CertifiedFlightTube result{
      .frame_id = DecodeIdentifier(
          Member(value, "frame_id", field_path),
          std::string{field_path} + ".frame_id"),
  };
  result.sections.reserve(sections.size());
  for (std::size_t index = 0; index < sections.size(); ++index) {
    result.sections.push_back(DecodeFlightTubeSection(
        sections[index],
        std::string{field_path} + ".sections[" +
            std::to_string(index) + "]"));
  }
  result.source_map_snapshot_ref = DecodeContentRef(
      Member(value, "source_map_snapshot_ref", field_path),
      std::string{field_path} + ".source_map_snapshot_ref");
  result.body_rotation_envelope_ref = DecodeContentRef(
      Member(value, "body_rotation_envelope_ref", field_path),
      std::string{field_path} + ".body_rotation_envelope_ref");
  result.error_model_ref = DecodeContentRef(
      Member(value, "error_model_ref", field_path),
      std::string{field_path} + ".error_model_ref");
  result.minimum_certified_clearance_m = clearance;
  return result;
}

[[nodiscard]] Json EncodeCertifiedFlightTube(
    const CertifiedFlightTube& value) {
  Json sections = Json::array();
  for (const FlightTubeSection& section : value.sections) {
    sections.push_back(EncodeFlightTubeSection(section));
  }
  return {
      {"frame_id", value.frame_id},
      {"sections", std::move(sections)},
      {"source_map_snapshot_ref",
       EncodeContentRef(value.source_map_snapshot_ref)},
      {"body_rotation_envelope_ref",
       EncodeContentRef(value.body_rotation_envelope_ref)},
      {"error_model_ref", EncodeContentRef(value.error_model_ref)},
      {"minimum_certified_clearance_m",
       value.minimum_certified_clearance_m},
  };
}

[[nodiscard]] TargetAttitudeSet DecodeTargetAttitudeSet(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"nominal_orientation_body_to_frame",
               "orientation_error_set",
               "allowed_yaw_interval"});
  return {
      DecodeQuaternion(
          Member(value, "nominal_orientation_body_to_frame", field_path),
          std::string{field_path} +
              ".nominal_orientation_body_to_frame"),
      DecodeRotationVectorBall(
          Member(value, "orientation_error_set", field_path),
          std::string{field_path} + ".orientation_error_set"),
      DecodeCircularYawInterval(
          Member(value, "allowed_yaw_interval", field_path),
          std::string{field_path} + ".allowed_yaw_interval"),
  };
}

[[nodiscard]] Json EncodeTargetAttitudeSet(
    const TargetAttitudeSet& value) {
  return {
      {"nominal_orientation_body_to_frame",
       EncodeQuaternion(value.nominal_orientation_body_to_frame)},
      {"orientation_error_set",
       EncodeRotationVectorBall(value.orientation_error_set)},
      {"allowed_yaw_interval",
       EncodeCircularYawInterval(value.allowed_yaw_interval)},
  };
}

[[nodiscard]] AttitudeBoundary DecodeAttitudeBoundary(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"initial_orientation_error_set",
               "initial_angular_velocity_error_set_radps",
               "target_attitude_set",
               "landing_angular_velocity_bounds_radps",
               "settle_guard_ns",
               "center_of_mass_translation_authority",
               "certification_ref"});
  RequireLiteral(
      Member(value, "center_of_mass_translation_authority", field_path),
      "NONE",
      std::string{field_path} +
          ".center_of_mass_translation_authority");
  return {
      .initial_orientation_error_set = DecodeRotationVectorBall(
          Member(value, "initial_orientation_error_set", field_path),
          std::string{field_path} +
              ".initial_orientation_error_set"),
      .initial_angular_velocity_error_set_radps = DecodeVectorSet3(
          Member(value,
                 "initial_angular_velocity_error_set_radps",
                 field_path),
          std::string{field_path} +
              ".initial_angular_velocity_error_set_radps"),
      .target_attitude_set = DecodeTargetAttitudeSet(
          Member(value, "target_attitude_set", field_path),
          std::string{field_path} + ".target_attitude_set"),
      .landing_angular_velocity_bounds_radps = DecodeVectorSet3(
          Member(value,
                 "landing_angular_velocity_bounds_radps",
                 field_path),
          std::string{field_path} +
              ".landing_angular_velocity_bounds_radps"),
      .settle_guard = DurationNanoseconds{ParseNanoseconds(
          Member(value, "settle_guard_ns", field_path),
          std::string{field_path} + ".settle_guard_ns",
          false)},
      .center_of_mass_translation_authority =
          AttitudeBoundary::TranslationAuthority::kNone,
      .certification_ref = DecodeContentRef(
          Member(value, "certification_ref", field_path),
          std::string{field_path} + ".certification_ref"),
  };
}

[[nodiscard]] Json EncodeAttitudeBoundary(
    const AttitudeBoundary& value) {
  return {
      {"initial_orientation_error_set",
       EncodeRotationVectorBall(value.initial_orientation_error_set)},
      {"initial_angular_velocity_error_set_radps",
       EncodeVectorSet3(
           value.initial_angular_velocity_error_set_radps)},
      {"target_attitude_set",
       EncodeTargetAttitudeSet(value.target_attitude_set)},
      {"landing_angular_velocity_bounds_radps",
       EncodeVectorSet3(
           value.landing_angular_velocity_bounds_radps)},
      {"settle_guard_ns",
       EncodeNonNegativeNanoseconds(
           value.settle_guard.value,
           "attitude_boundary.settle_guard_ns")},
      {"center_of_mass_translation_authority", "NONE"},
      {"certification_ref", EncodeContentRef(value.certification_ref)},
  };
}

[[nodiscard]] NominalAimPoint DecodeNominalAimPoint(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value, field_path, {"authority", "position_m"});
  RequireLiteral(Member(value, "authority", field_path),
                 "NON_AUTHORITATIVE_EXPLANATORY",
                 std::string{field_path} + ".authority");
  return {
      NominalAimPoint::Authority::kNonAuthoritativeExplanatory,
      DecodeVec3(Member(value, "position_m", field_path),
                 std::string{field_path} + ".position_m"),
  };
}

[[nodiscard]] Json EncodeNominalAimPoint(
    const NominalAimPoint& value) {
  return {
      {"authority", "NON_AUTHORITATIVE_EXPLANATORY"},
      {"position_m", EncodeVec3(value.position_m)},
  };
}

[[nodiscard]] FutureViability DecodeFutureViability(
    const Json& value,
    std::string_view field_path) {
  const std::string text = DecodeString(value, field_path);
  if (text == "VIABLE") {
    return FutureViability::kViable;
  }
  if (text == "UNKNOWN") {
    return FutureViability::kUnknown;
  }
  if (text == "NO_CERTIFIED_CONTINUATION") {
    return FutureViability::kNoCertifiedContinuation;
  }
  Fail(ErrorCode::kSchemaMismatch,
       field_path,
       "unknown future viability enum");
}

[[nodiscard]] std::string_view EncodeFutureViability(
    FutureViability value) {
  switch (value) {
    case FutureViability::kViable:
      return "VIABLE";
    case FutureViability::kUnknown:
      return "UNKNOWN";
    case FutureViability::kNoCertifiedContinuation:
      return "NO_CERTIFIED_CONTINUATION";
  }
  throw std::invalid_argument{"unknown FutureViability"};
}

[[nodiscard]] FutureRoutePreview DecodeFutureRoutePreview(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"authority",
               "future_viability",
               "reason_code",
               "candidate_region_ids"});
  RequireLiteral(Member(value, "authority", field_path),
                 "NON_AUTHORITATIVE_MISSION_PREVIEW",
                 std::string{field_path} + ".authority");
  const Json& candidates =
      Member(value, "candidate_region_ids", field_path);
  if (!candidates.is_array() || candidates.size() > 128U) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".candidate_region_ids",
         "future candidate count exceeds schema maximum");
  }
  FutureRoutePreview result{
      .authority =
          FutureRoutePreview::Authority::kNonAuthoritativeMissionPreview,
      .future_viability = DecodeFutureViability(
          Member(value, "future_viability", field_path),
          std::string{field_path} + ".future_viability"),
      .reason_code = DecodeReasonCode(
          Member(value, "reason_code", field_path),
          std::string{field_path} + ".reason_code"),
  };
  result.candidate_region_ids.reserve(candidates.size());
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    result.candidate_region_ids.push_back(DecodeIdentifier(
        candidates[index],
        std::string{field_path} + ".candidate_region_ids[" +
            std::to_string(index) + "]"));
  }
  return result;
}

[[nodiscard]] Json EncodeFutureRoutePreview(
    const FutureRoutePreview& value) {
  Json candidates = Json::array();
  for (const LandingRegionId& id : value.candidate_region_ids) {
    candidates.push_back(id);
  }
  return {
      {"authority", "NON_AUTHORITATIVE_MISSION_PREVIEW"},
      {"future_viability",
       EncodeFutureViability(value.future_viability)},
      {"reason_code", value.reason_code},
      {"candidate_region_ids", std::move(candidates)},
  };
}

[[nodiscard]] HopperReference DecodeHopperReference(
    const Json& value,
    std::string_view field_path) {
  CheckSchemaVersion(value, kHopperReferenceSchemaVersion, field_path);
  CheckObject(value,
              field_path,
              {"schema_version",
               "reference_id",
               "reference_hash",
               "platform_type",
               "reference_time_origin",
               "translation_model",
               "ground_hold_anchor",
               "next_landing_region",
               "jump_boundary",
               "predicted_landing_footprint",
               "certified_flight_tube",
               "attitude_boundary",
               "nominal_aim_point",
               "physical_certification_ref"},
              {"future_route_preview"});
  RequireLiteral(Member(value, "platform_type", field_path),
                 "HOPPER",
                 std::string{field_path} + ".platform_type");
  RequireLiteral(Member(value, "translation_model", field_path),
                 "PURE_BALLISTIC_NO_INFLIGHT_TRANSLATION_CONTROL",
                 std::string{field_path} + ".translation_model");
  HopperReference result{
      .reference_id = DecodeIdentifier(
          Member(value, "reference_id", field_path),
          std::string{field_path} + ".reference_id"),
      .reference_hash = DecodeHash(
          Member(value, "reference_hash", field_path),
          std::string{field_path} + ".reference_hash"),
      .reference_time_origin = DecodeClockStamp(
          Member(value, "reference_time_origin", field_path),
          std::string{field_path} + ".reference_time_origin"),
      .translation_model =
          HopperReference::TranslationModel::
              kPureBallisticNoInflightTranslationControl,
      .ground_hold_anchor = DecodeGroundHoldAnchor(
          Member(value, "ground_hold_anchor", field_path),
          std::string{field_path} + ".ground_hold_anchor"),
      .next_landing_region = DecodeNextLandingRegion(
          Member(value, "next_landing_region", field_path),
          std::string{field_path} + ".next_landing_region"),
      .jump_boundary = DecodeJumpBoundary(
          Member(value, "jump_boundary", field_path),
          std::string{field_path} + ".jump_boundary"),
      .predicted_landing_footprint = DecodePredictedLandingFootprint(
          Member(value, "predicted_landing_footprint", field_path),
          std::string{field_path} +
              ".predicted_landing_footprint"),
      .certified_flight_tube = DecodeCertifiedFlightTube(
          Member(value, "certified_flight_tube", field_path),
          std::string{field_path} + ".certified_flight_tube"),
      .attitude_boundary = DecodeAttitudeBoundary(
          Member(value, "attitude_boundary", field_path),
          std::string{field_path} + ".attitude_boundary"),
      .nominal_aim_point = DecodeNominalAimPoint(
          Member(value, "nominal_aim_point", field_path),
          std::string{field_path} + ".nominal_aim_point"),
      .physical_certification_ref = DecodeContentRef(
          Member(value, "physical_certification_ref", field_path),
          std::string{field_path} + ".physical_certification_ref"),
  };
  if (value.contains("future_route_preview")) {
    result.future_route_preview = DecodeFutureRoutePreview(
        value.at("future_route_preview"),
        std::string{field_path} + ".future_route_preview");
  }
  return result;
}

[[nodiscard]] Json EncodeHopperReference(
    const HopperReference& value) {
  Json result{
      {"schema_version", kHopperReferenceSchemaVersion},
      {"reference_id", value.reference_id},
      {"reference_hash", value.reference_hash},
      {"platform_type", "HOPPER"},
      {"reference_time_origin",
       EncodeClockStamp(value.reference_time_origin)},
      {"translation_model",
       "PURE_BALLISTIC_NO_INFLIGHT_TRANSLATION_CONTROL"},
      {"ground_hold_anchor",
       EncodeGroundHoldAnchor(value.ground_hold_anchor)},
      {"next_landing_region",
       EncodeNextLandingRegion(value.next_landing_region)},
      {"jump_boundary", EncodeJumpBoundary(value.jump_boundary)},
      {"predicted_landing_footprint",
       EncodePredictedLandingFootprint(
           value.predicted_landing_footprint)},
      {"certified_flight_tube",
       EncodeCertifiedFlightTube(value.certified_flight_tube)},
      {"attitude_boundary",
       EncodeAttitudeBoundary(value.attitude_boundary)},
      {"nominal_aim_point",
       EncodeNominalAimPoint(value.nominal_aim_point)},
      {"physical_certification_ref",
       EncodeContentRef(value.physical_certification_ref)},
  };
  if (value.future_route_preview.has_value()) {
    result["future_route_preview"] =
        EncodeFutureRoutePreview(*value.future_route_preview);
  }
  return result;
}

[[nodiscard]] PlatformReference DecodePlatformReference(
    const Json& value,
    PlatformType platform_type,
    std::string_view field_path) {
  switch (platform_type) {
    case PlatformType::kWheeled:
      return DecodeWheeledReference(value, field_path);
    case PlatformType::kLegged:
      return DecodeLeggedReference(value, field_path);
    case PlatformType::kHopper:
      return DecodeHopperReference(value, field_path);
  }
  Fail(ErrorCode::kSchemaMismatch,
       field_path,
       "unknown platform reference variant");
}

[[nodiscard]] Json EncodePlatformReference(
    const PlatformReference& value) {
  return std::visit(
      [](const auto& concrete) -> Json {
        using T = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<T, WheeledReference>) {
          return EncodeWheeledReference(concrete);
        } else if constexpr (std::is_same_v<T, LeggedBodyReference>) {
          return EncodeLeggedReference(concrete);
        } else {
          return EncodeHopperReference(concrete);
        }
      },
      value);
}

[[nodiscard]] RouteSkeletonContent DecodeRouteSkeletonContent(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"authority",
               "source_reference_id",
               "source_reference_hash",
               "waypoints"},
              {"unresolved_tail"});
  RequireLiteral(Member(value, "authority", field_path),
                 "NON_AUTHORITATIVE",
                 std::string{field_path} + ".authority");
  const Json& waypoints = Member(value, "waypoints", field_path);
  if (!waypoints.is_array() || waypoints.size() > 2048U) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".waypoints",
         "route waypoint count exceeds schema maximum");
  }
  RouteSkeletonContent result{
      .source_reference_id = DecodeIdentifier(
          Member(value, "source_reference_id", field_path),
          std::string{field_path} + ".source_reference_id"),
      .source_reference_hash = DecodeHash(
          Member(value, "source_reference_hash", field_path),
          std::string{field_path} + ".source_reference_hash"),
  };
  result.waypoints.reserve(waypoints.size());
  for (std::size_t index = 0; index < waypoints.size(); ++index) {
    result.waypoints.push_back(DecodePose(
        waypoints[index],
        std::string{field_path} + ".waypoints[" +
            std::to_string(index) + "]"));
  }
  if (value.contains("unresolved_tail")) {
    const Json& tail = value.at("unresolved_tail");
    if (!tail.is_array() || tail.size() > 512U) {
      Fail(ErrorCode::kSchemaMismatch,
           std::string{field_path} + ".unresolved_tail",
           "unresolved-tail count exceeds schema maximum");
    }
    result.unresolved_tail.reserve(tail.size());
    for (std::size_t index = 0; index < tail.size(); ++index) {
      result.unresolved_tail.push_back(DecodeVec3(
          tail[index],
          std::string{field_path} + ".unresolved_tail[" +
              std::to_string(index) + "]"));
    }
  }
  return result;
}

[[nodiscard]] Json EncodeRouteSkeletonContent(
    const RouteSkeletonContent& value) {
  Json waypoints = Json::array();
  for (const PoseXyzYaw& pose : value.waypoints) {
    waypoints.push_back(EncodePose(pose));
  }
  Json result{
      {"authority", "NON_AUTHORITATIVE"},
      {"source_reference_id", value.source_reference_id},
      {"source_reference_hash", value.source_reference_hash},
      {"waypoints", std::move(waypoints)},
  };
  if (!value.unresolved_tail.empty()) {
    Json tail = Json::array();
    for (const Vec3& point : value.unresolved_tail) {
      tail.push_back(EncodeVec3(point));
    }
    result["unresolved_tail"] = std::move(tail);
  }
  return result;
}

[[nodiscard]] InlineComponent<RouteSkeletonContent>
DecodeRouteSkeletonComponent(const Json& value,
                             std::string_view field_path) {
  CheckObject(
      value, field_path, {"component_id", "component_hash", "content"});
  return {
      DecodeIdentifier(Member(value, "component_id", field_path),
                       std::string{field_path} + ".component_id"),
      DecodeHash(Member(value, "component_hash", field_path),
                 std::string{field_path} + ".component_hash"),
      DecodeRouteSkeletonContent(
          Member(value, "content", field_path),
          std::string{field_path} + ".content"),
  };
}

[[nodiscard]] Json EncodeRouteSkeletonComponent(
    const InlineComponent<RouteSkeletonContent>& value) {
  return {
      {"component_id", value.component_id},
      {"component_hash", value.component_hash},
      {"content", EncodeRouteSkeletonContent(value.content)},
  };
}

[[nodiscard]] ReferenceViewSelector DecodeReferenceViewSelector(
    const Json& value,
    std::string_view field_path) {
  if (!value.is_object() || !value.contains("selector_type")) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "reference selector requires selector_type");
  }
  const std::string type = DecodeString(
      value.at("selector_type"),
      std::string{field_path} + ".selector_type");
  if (type == "TIME_INTERVAL") {
    CheckObject(value, field_path, {"selector_type", "time_interval"});
    return TimeViewSelector{DecodeTimeInterval(
        Member(value, "time_interval", field_path),
        std::string{field_path} + ".time_interval")};
  }
  if (type == "SEGMENT_RANGE") {
    CheckObject(value,
                field_path,
                {"selector_type",
                 "first_segment_index",
                 "past_last_segment_index"});
    return SegmentViewSelector{
        DecodeSize(Member(value, "first_segment_index", field_path),
                   std::string{field_path} + ".first_segment_index"),
        DecodeSize(Member(value, "past_last_segment_index", field_path),
                   std::string{field_path} +
                       ".past_last_segment_index"),
    };
  }
  if (type == "GROUND_HOLD") {
    CheckObject(value, field_path, {"selector_type", "anchor_id"});
    return GroundHoldViewSelector{DecodeIdentifier(
        Member(value, "anchor_id", field_path),
        std::string{field_path} + ".anchor_id")};
  }
  if (type == "JUMP_BOUNDARY") {
    CheckObject(
        value, field_path, {"selector_type", "boundary_id", "scope"});
    const std::string scope = DecodeString(
        Member(value, "scope", field_path),
        std::string{field_path} + ".scope");
    JumpViewSelector::Scope decoded_scope{};
    if (scope == "NEXT_HOP") {
      decoded_scope = JumpViewSelector::Scope::kNextHop;
    } else if (scope == "FUTURE_MISSION_PREVIEW") {
      decoded_scope = JumpViewSelector::Scope::kFutureMissionPreview;
    } else {
      Fail(ErrorCode::kSchemaMismatch,
           std::string{field_path} + ".scope",
           "unknown jump selector scope");
    }
    return JumpViewSelector{
        DecodeIdentifier(Member(value, "boundary_id", field_path),
                         std::string{field_path} + ".boundary_id"),
        decoded_scope,
    };
  }
  Fail(ErrorCode::kSchemaMismatch,
       std::string{field_path} + ".selector_type",
       "unknown reference selector type");
}

[[nodiscard]] Json EncodeReferenceViewSelector(
    const ReferenceViewSelector& value) {
  return std::visit(
      [](const auto& concrete) -> Json {
        using T = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<T, TimeViewSelector>) {
          return {
              {"selector_type", "TIME_INTERVAL"},
              {"time_interval",
               EncodeTimeInterval(concrete.time_interval)},
          };
        } else if constexpr (std::is_same_v<T, SegmentViewSelector>) {
          return {
              {"selector_type", "SEGMENT_RANGE"},
              {"first_segment_index", concrete.first_segment_index},
              {"past_last_segment_index",
               concrete.past_last_segment_index},
          };
        } else if constexpr (
            std::is_same_v<T, GroundHoldViewSelector>) {
          return {
              {"selector_type", "GROUND_HOLD"},
              {"anchor_id", concrete.anchor_id},
          };
        } else {
          return {
              {"selector_type", "JUMP_BOUNDARY"},
              {"boundary_id", concrete.boundary_id},
              {"scope",
               concrete.scope == JumpViewSelector::Scope::kNextHop
                   ? "NEXT_HOP"
                   : "FUTURE_MISSION_PREVIEW"},
          };
        }
      },
      value);
}

[[nodiscard]] ReferenceViewContent DecodeReferenceViewContent(
    const Json& value,
    ReferenceViewContent::Role expected_role,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"role",
               "source_reference_id",
               "source_reference_hash",
               "selector"});
  const std::string role = DecodeString(
      Member(value, "role", field_path),
      std::string{field_path} + ".role");
  const auto decoded_role =
      role == "COMMITTED_PREFIX"
          ? ReferenceViewContent::Role::kCommittedPrefix
          : role == "PREVIEW"
                ? ReferenceViewContent::Role::kPreview
                : (Fail(ErrorCode::kSchemaMismatch,
                        std::string{field_path} + ".role",
                        "unknown reference-view role"),
                   ReferenceViewContent::Role::kPreview);
  if (decoded_role != expected_role) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".role",
         "reference-view role does not match component");
  }
  return {
      decoded_role,
      DecodeIdentifier(
          Member(value, "source_reference_id", field_path),
          std::string{field_path} + ".source_reference_id"),
      DecodeHash(Member(value, "source_reference_hash", field_path),
                 std::string{field_path} + ".source_reference_hash"),
      DecodeReferenceViewSelector(
          Member(value, "selector", field_path),
          std::string{field_path} + ".selector"),
  };
}

[[nodiscard]] Json EncodeReferenceViewContent(
    const ReferenceViewContent& value) {
  return {
      {"role",
       value.role == ReferenceViewContent::Role::kCommittedPrefix
           ? "COMMITTED_PREFIX"
           : "PREVIEW"},
      {"source_reference_id", value.source_reference_id},
      {"source_reference_hash", value.source_reference_hash},
      {"selector", EncodeReferenceViewSelector(value.selector)},
  };
}

[[nodiscard]] InlineComponent<ReferenceViewContent>
DecodeReferenceViewComponent(
    const Json& value,
    ReferenceViewContent::Role expected_role,
    std::string_view field_path) {
  CheckObject(
      value, field_path, {"component_id", "component_hash", "content"});
  return {
      DecodeIdentifier(Member(value, "component_id", field_path),
                       std::string{field_path} + ".component_id"),
      DecodeHash(Member(value, "component_hash", field_path),
                 std::string{field_path} + ".component_hash"),
      DecodeReferenceViewContent(
          Member(value, "content", field_path),
          expected_role,
          std::string{field_path} + ".content"),
  };
}

[[nodiscard]] Json EncodeReferenceViewComponent(
    const InlineComponent<ReferenceViewContent>& value) {
  return {
      {"component_id", value.component_id},
      {"component_hash", value.component_hash},
      {"content", EncodeReferenceViewContent(value.content)},
  };
}

[[nodiscard]] InvalidationCondition DecodeInvalidationCondition(
    const Json& value,
    std::string_view field_path) {
  const std::string text = DecodeString(value, field_path);
  if (text == "MAP_SAFETY_REVISION_CHANGED") {
    return InvalidationCondition::kMapSafetyRevisionChanged;
  }
  if (text == "STATE_DEVIATION_EXCEEDED") {
    return InvalidationCondition::kStateDeviationExceeded;
  }
  if (text == "CAPABILITY_REVISION_CHANGED") {
    return InvalidationCondition::kCapabilityRevisionChanged;
  }
  if (text == "REFERENCE_HORIZON_EXHAUSTED") {
    return InvalidationCondition::kReferenceHorizonExhausted;
  }
  Fail(ErrorCode::kSchemaMismatch,
       field_path,
       "unknown invalidation condition");
}

[[nodiscard]] std::string_view EncodeInvalidationCondition(
    InvalidationCondition value) {
  switch (value) {
    case InvalidationCondition::kMapSafetyRevisionChanged:
      return "MAP_SAFETY_REVISION_CHANGED";
    case InvalidationCondition::kStateDeviationExceeded:
      return "STATE_DEVIATION_EXCEEDED";
    case InvalidationCondition::kCapabilityRevisionChanged:
      return "CAPABILITY_REVISION_CHANGED";
    case InvalidationCondition::kReferenceHorizonExhausted:
      return "REFERENCE_HORIZON_EXHAUSTED";
  }
  throw std::invalid_argument{"unknown InvalidationCondition"};
}

[[nodiscard]] ReferenceValidity DecodeReferenceValidity(
    const Json& value,
    PlatformType platform_type,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"valid_from",
               "required_map_snapshot_ref",
               "required_capability_ref",
               "allowed_state_deviation",
               "invalidation_conditions"},
              {"valid_until"});
  const Json& conditions =
      Member(value, "invalidation_conditions", field_path);
  if (!conditions.is_array() || conditions.empty() ||
      conditions.size() > 16U) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".invalidation_conditions",
         "invalidation-condition count is outside schema bounds");
  }
  ReferenceValidity result{
      .valid_from = DecodeClockStamp(
          Member(value, "valid_from", field_path),
          std::string{field_path} + ".valid_from"),
      .required_map_snapshot_ref = DecodeContentRef(
          Member(value, "required_map_snapshot_ref", field_path),
          std::string{field_path} + ".required_map_snapshot_ref"),
      .required_capability_ref = DecodeContentRef(
          Member(value, "required_capability_ref", field_path),
          std::string{field_path} + ".required_capability_ref"),
      .allowed_state_deviation =
          platform_type == PlatformType::kHopper
              ? std::variant<WheeledOrLeggedErrorBounds,
                             HopperErrorBounds>{
                    DecodeHopperErrorBounds(
                        Member(value,
                               "allowed_state_deviation",
                               field_path),
                        std::string{field_path} +
                            ".allowed_state_deviation")}
              : std::variant<WheeledOrLeggedErrorBounds,
                             HopperErrorBounds>{
                    DecodeWheeledOrLeggedErrorBounds(
                        Member(value,
                               "allowed_state_deviation",
                               field_path),
                        std::string{field_path} +
                            ".allowed_state_deviation")},
  };
  if (value.contains("valid_until")) {
    result.valid_until = DecodeClockStamp(
        value.at("valid_until"),
        std::string{field_path} + ".valid_until");
  }
  result.invalidation_conditions.reserve(conditions.size());
  for (std::size_t index = 0; index < conditions.size(); ++index) {
    result.invalidation_conditions.push_back(
        DecodeInvalidationCondition(
            conditions[index],
            std::string{field_path} + ".invalidation_conditions[" +
                std::to_string(index) + "]"));
  }
  return result;
}

[[nodiscard]] Json EncodeReferenceValidity(
    const ReferenceValidity& value) {
  Json conditions = Json::array();
  for (InvalidationCondition condition :
       value.invalidation_conditions) {
    conditions.push_back(EncodeInvalidationCondition(condition));
  }
  Json result{
      {"valid_from", EncodeClockStamp(value.valid_from)},
      {"required_map_snapshot_ref",
       EncodeContentRef(value.required_map_snapshot_ref)},
      {"required_capability_ref",
       EncodeContentRef(value.required_capability_ref)},
      {"allowed_state_deviation",
       std::visit(
           [](const auto& concrete) -> Json {
             using T = std::decay_t<decltype(concrete)>;
             if constexpr (
                 std::is_same_v<T, WheeledOrLeggedErrorBounds>) {
               return EncodeWheeledOrLeggedErrorBounds(concrete);
             } else {
               return EncodeHopperErrorBounds(concrete);
             }
           },
           value.allowed_state_deviation)},
      {"invalidation_conditions", std::move(conditions)},
  };
  if (value.valid_until.has_value()) {
    result["valid_until"] = EncodeClockStamp(*value.valid_until);
  }
  return result;
}

[[nodiscard]] std::vector<ContentRef> DecodeContentRefArray(
    const Json& value,
    std::string_view field_path,
    std::size_t minimum,
    std::size_t maximum) {
  if (!value.is_array() || value.size() < minimum ||
      value.size() > maximum) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "ContentRef array size is outside schema bounds");
  }
  std::vector<ContentRef> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    result.push_back(DecodeContentRef(
        value[index],
        std::string{field_path} + "[" + std::to_string(index) + "]"));
  }
  return result;
}

[[nodiscard]] Json EncodeContentRefArray(
    const std::vector<ContentRef>& values) {
  Json result = Json::array();
  for (const ContentRef& ref : values) {
    result.push_back(EncodeContentRef(ref));
  }
  return result;
}

[[nodiscard]] std::vector<std::string> DecodeReasonCodeArray(
    const Json& value,
    std::string_view field_path,
    std::size_t maximum) {
  if (!value.is_array() || value.size() > maximum) {
    Fail(ErrorCode::kSchemaMismatch,
         field_path,
         "reason-code array exceeds schema maximum");
  }
  std::vector<std::string> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    result.push_back(DecodeReasonCode(
        value[index],
        std::string{field_path} + "[" + std::to_string(index) + "]"));
  }
  return result;
}

[[nodiscard]] ValidationSummary DecodeValidationSummary(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"hard_constraints_passed",
               "continuous_validation_passed",
               "certificate_refs",
               "warning_codes"});
  RequireBooleanLiteral(
      Member(value, "hard_constraints_passed", field_path),
      true,
      std::string{field_path} + ".hard_constraints_passed");
  RequireBooleanLiteral(
      Member(value, "continuous_validation_passed", field_path),
      true,
      std::string{field_path} +
          ".continuous_validation_passed");
  return {
      true,
      true,
      DecodeContentRefArray(
          Member(value, "certificate_refs", field_path),
          std::string{field_path} + ".certificate_refs",
          1U,
          128U),
      DecodeReasonCodeArray(
          Member(value, "warning_codes", field_path),
          std::string{field_path} + ".warning_codes",
          64U),
  };
}

[[nodiscard]] Json EncodeValidationSummary(
    const ValidationSummary& value) {
  return {
      {"hard_constraints_passed", value.hard_constraints_passed},
      {"continuous_validation_passed",
       value.continuous_validation_passed},
      {"certificate_refs", EncodeContentRefArray(value.certificate_refs)},
      {"warning_codes", value.warning_codes},
  };
}

[[nodiscard]] GenerationMode DecodeGenerationMode(
    const Json& value,
    std::string_view field_path) {
  const std::string text = DecodeString(value, field_path);
  if (text == "SMOOTHED_SPLINE_REFERENCE") {
    return GenerationMode::kSmoothedSplineReference;
  }
  if (text == "VALIDATED_PRIMITIVE_CHAIN_REFERENCE") {
    return GenerationMode::kValidatedPrimitiveChainReference;
  }
  if (text == "CERTIFIED_BALLISTIC_REFERENCE") {
    return GenerationMode::kCertifiedBallisticReference;
  }
  Fail(ErrorCode::kSchemaMismatch,
       field_path,
       "unknown generation mode");
}

[[nodiscard]] std::string_view EncodeGenerationMode(
    GenerationMode value) {
  switch (value) {
    case GenerationMode::kSmoothedSplineReference:
      return "SMOOTHED_SPLINE_REFERENCE";
    case GenerationMode::kValidatedPrimitiveChainReference:
      return "VALIDATED_PRIMITIVE_CHAIN_REFERENCE";
    case GenerationMode::kCertifiedBallisticReference:
      return "CERTIFIED_BALLISTIC_REFERENCE";
  }
  throw std::invalid_argument{"unknown GenerationMode"};
}

[[nodiscard]] GenerationEvidence DecodeGenerationEvidence(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"selected_candidate_id",
               "generation_mode",
               "termination_reason",
               "evidence_refs"},
              {"learned_cost_snapshot_ref"});
  GenerationEvidence result{
      .selected_candidate_id = DecodeIdentifier(
          Member(value, "selected_candidate_id", field_path),
          std::string{field_path} + ".selected_candidate_id"),
      .generation_mode = DecodeGenerationMode(
          Member(value, "generation_mode", field_path),
          std::string{field_path} + ".generation_mode"),
      .termination_reason = DecodeReasonCode(
          Member(value, "termination_reason", field_path),
          std::string{field_path} + ".termination_reason"),
      .evidence_refs = DecodeContentRefArray(
          Member(value, "evidence_refs", field_path),
          std::string{field_path} + ".evidence_refs",
          1U,
          128U),
  };
  if (value.contains("learned_cost_snapshot_ref")) {
    result.learned_cost_snapshot_ref = DecodeContentRef(
        value.at("learned_cost_snapshot_ref"),
        std::string{field_path} + ".learned_cost_snapshot_ref");
  }
  return result;
}

[[nodiscard]] Json EncodeGenerationEvidence(
    const GenerationEvidence& value) {
  Json result{
      {"selected_candidate_id", value.selected_candidate_id},
      {"generation_mode", EncodeGenerationMode(value.generation_mode)},
      {"termination_reason", value.termination_reason},
      {"evidence_refs", EncodeContentRefArray(value.evidence_refs)},
  };
  if (value.learned_cost_snapshot_ref.has_value()) {
    result["learned_cost_snapshot_ref"] =
        EncodeContentRef(*value.learned_cost_snapshot_ref);
  }
  return result;
}

[[nodiscard]] ReferenceBundle DecodeReferenceBundle(
    const Json& value,
    std::string_view field_path) {
  CheckSchemaVersion(value, kReferenceBundleSchemaVersion, field_path);
  CheckObject(value,
              field_path,
              {"schema_version",
               "bundle_id",
               "bundle_revision",
               "bundle_hash",
               "source_request_id",
               "source_map_snapshot_ref",
               "source_safety_capability_ref",
               "source_algorithm_config_ref",
               "platform_type",
               "platform_reference",
               "route_skeleton",
               "committed_prefix",
               "preview",
               "validity",
               "validation_summary",
               "generation_evidence"},
              {"supersedes_bundle_id"});
  ReferenceBundle result{
      .bundle_id = DecodeIdentifier(
          Member(value, "bundle_id", field_path),
          std::string{field_path} + ".bundle_id"),
      .bundle_revision = DecodeRevision(
          Member(value, "bundle_revision", field_path),
          std::string{field_path} + ".bundle_revision"),
      .bundle_hash = DecodeHash(
          Member(value, "bundle_hash", field_path),
          std::string{field_path} + ".bundle_hash"),
      .source_request_id = DecodeIdentifier(
          Member(value, "source_request_id", field_path),
          std::string{field_path} + ".source_request_id"),
      .source_map_snapshot_ref = DecodeContentRef(
          Member(value, "source_map_snapshot_ref", field_path),
          std::string{field_path} + ".source_map_snapshot_ref"),
      .source_safety_capability_ref = DecodeContentRef(
          Member(value, "source_safety_capability_ref", field_path),
          std::string{field_path} +
              ".source_safety_capability_ref"),
      .source_algorithm_config_ref = DecodeContentRef(
          Member(value, "source_algorithm_config_ref", field_path),
          std::string{field_path} + ".source_algorithm_config_ref"),
      .platform_type = DecodePlatformType(
          Member(value, "platform_type", field_path),
          std::string{field_path} + ".platform_type"),
  };
  if (value.contains("supersedes_bundle_id")) {
    result.supersedes_bundle_id = DecodeIdentifier(
        value.at("supersedes_bundle_id"),
        std::string{field_path} + ".supersedes_bundle_id");
  }
  result.platform_reference = DecodePlatformReference(
      Member(value, "platform_reference", field_path),
      result.platform_type,
      std::string{field_path} + ".platform_reference");
  result.route_skeleton = DecodeRouteSkeletonComponent(
      Member(value, "route_skeleton", field_path),
      std::string{field_path} + ".route_skeleton");
  result.committed_prefix = DecodeReferenceViewComponent(
      Member(value, "committed_prefix", field_path),
      ReferenceViewContent::Role::kCommittedPrefix,
      std::string{field_path} + ".committed_prefix");
  result.preview = DecodeReferenceViewComponent(
      Member(value, "preview", field_path),
      ReferenceViewContent::Role::kPreview,
      std::string{field_path} + ".preview");
  result.validity = DecodeReferenceValidity(
      Member(value, "validity", field_path),
      result.platform_type,
      std::string{field_path} + ".validity");
  result.validation_summary = DecodeValidationSummary(
      Member(value, "validation_summary", field_path),
      std::string{field_path} + ".validation_summary");
  result.generation_evidence = DecodeGenerationEvidence(
      Member(value, "generation_evidence", field_path),
      std::string{field_path} + ".generation_evidence");
  return result;
}

[[nodiscard]] Json EncodeReferenceBundle(
    const ReferenceBundle& value) {
  Json result{
      {"schema_version", kReferenceBundleSchemaVersion},
      {"bundle_id", value.bundle_id},
      {"bundle_revision", value.bundle_revision},
      {"bundle_hash", value.bundle_hash},
      {"source_request_id", value.source_request_id},
      {"source_map_snapshot_ref",
       EncodeContentRef(value.source_map_snapshot_ref)},
      {"source_safety_capability_ref",
       EncodeContentRef(value.source_safety_capability_ref)},
      {"source_algorithm_config_ref",
       EncodeContentRef(value.source_algorithm_config_ref)},
      {"platform_type", EncodePlatformType(value.platform_type)},
      {"platform_reference",
       EncodePlatformReference(value.platform_reference)},
      {"route_skeleton",
       EncodeRouteSkeletonComponent(value.route_skeleton)},
      {"committed_prefix",
       EncodeReferenceViewComponent(value.committed_prefix)},
      {"preview", EncodeReferenceViewComponent(value.preview)},
      {"validity", EncodeReferenceValidity(value.validity)},
      {"validation_summary",
       EncodeValidationSummary(value.validation_summary)},
      {"generation_evidence",
       EncodeGenerationEvidence(value.generation_evidence)},
  };
  if (value.supersedes_bundle_id.has_value()) {
    result["supersedes_bundle_id"] = *value.supersedes_bundle_id;
  }
  return result;
}

[[nodiscard]] PlanningOutcome DecodePlanningOutcome(
    const Json& value,
    std::string_view field_path) {
  const std::string text = DecodeString(value, field_path);
  if (text == "NEW_REFERENCE_READY") {
    return PlanningOutcome::kNewReferenceReady;
  }
  if (text == "SAFE_FRONTIER_REFERENCE_READY") {
    return PlanningOutcome::kSafeFrontierReferenceReady;
  }
  if (text == "NO_KNOWN_SAFE_ROUTE") {
    return PlanningOutcome::kNoKnownSafeRoute;
  }
  if (text == "GOAL_INFEASIBLE") {
    return PlanningOutcome::kGoalInfeasible;
  }
  if (text == "INVALID_REQUEST") {
    return PlanningOutcome::kInvalidRequest;
  }
  if (text == "STALE_INPUT") {
    return PlanningOutcome::kStaleInput;
  }
  if (text == "NUMERICAL_FAILURE") {
    return PlanningOutcome::kNumericalFailure;
  }
  if (text == "RESOURCE_LIMIT") {
    return PlanningOutcome::kResourceLimit;
  }
  if (text == "ACTIVE_REFERENCE_INVALIDATED") {
    return PlanningOutcome::kActiveReferenceInvalidated;
  }
  Fail(ErrorCode::kSchemaMismatch,
       field_path,
       "unknown planning outcome");
}

[[nodiscard]] std::string_view EncodePlanningOutcome(
    PlanningOutcome value) {
  switch (value) {
    case PlanningOutcome::kNewReferenceReady:
      return "NEW_REFERENCE_READY";
    case PlanningOutcome::kSafeFrontierReferenceReady:
      return "SAFE_FRONTIER_REFERENCE_READY";
    case PlanningOutcome::kNoKnownSafeRoute:
      return "NO_KNOWN_SAFE_ROUTE";
    case PlanningOutcome::kGoalInfeasible:
      return "GOAL_INFEASIBLE";
    case PlanningOutcome::kInvalidRequest:
      return "INVALID_REQUEST";
    case PlanningOutcome::kStaleInput:
      return "STALE_INPUT";
    case PlanningOutcome::kNumericalFailure:
      return "NUMERICAL_FAILURE";
    case PlanningOutcome::kResourceLimit:
      return "RESOURCE_LIMIT";
    case PlanningOutcome::kActiveReferenceInvalidated:
      return "ACTIVE_REFERENCE_INVALIDATED";
  }
  throw std::invalid_argument{"unknown PlanningOutcome"};
}

[[nodiscard]] ExecutionDirective DecodeExecutionDirective(
    const Json& value,
    std::string_view field_path) {
  const std::string text = DecodeString(value, field_path);
  if (text == "ACTIVATE_NEW_BUNDLE") {
    return ExecutionDirective::kActivateNewBundle;
  }
  if (text == "CONTINUE_ACTIVE_BUNDLE") {
    return ExecutionDirective::kContinueActiveBundle;
  }
  if (text == "HOLD_STATIONARY") {
    return ExecutionDirective::kHoldStationary;
  }
  if (text == "CONTINUE_COMMITTED_JUMP") {
    return ExecutionDirective::kContinueCommittedJump;
  }
  if (text == "NO_SAFE_PLANNER_REFERENCE") {
    return ExecutionDirective::kNoSafePlannerReference;
  }
  Fail(ErrorCode::kSchemaMismatch,
       field_path,
       "unknown execution directive");
}

[[nodiscard]] std::string_view EncodeExecutionDirective(
    ExecutionDirective value) {
  switch (value) {
    case ExecutionDirective::kActivateNewBundle:
      return "ACTIVATE_NEW_BUNDLE";
    case ExecutionDirective::kContinueActiveBundle:
      return "CONTINUE_ACTIVE_BUNDLE";
    case ExecutionDirective::kHoldStationary:
      return "HOLD_STATIONARY";
    case ExecutionDirective::kContinueCommittedJump:
      return "CONTINUE_COMMITTED_JUMP";
    case ExecutionDirective::kNoSafePlannerReference:
      return "NO_SAFE_PLANNER_REFERENCE";
  }
  throw std::invalid_argument{"unknown ExecutionDirective"};
}

[[nodiscard]] LearnedCostUsage DecodeLearnedCostUsage(
    const Json& value,
    std::string_view field_path) {
  const std::string text = DecodeString(value, field_path);
  if (text == "DISABLED") {
    return LearnedCostUsage::kDisabled;
  }
  if (text == "USED_BOUNDED_SOFT_COST") {
    return LearnedCostUsage::kUsedBoundedSoftCost;
  }
  if (text == "FELL_BACK_TO_ANALYTIC") {
    return LearnedCostUsage::kFellBackToAnalytic;
  }
  Fail(ErrorCode::kSchemaMismatch,
       field_path,
       "unknown learned-cost usage");
}

[[nodiscard]] std::string_view EncodeLearnedCostUsage(
    LearnedCostUsage value) {
  switch (value) {
    case LearnedCostUsage::kDisabled:
      return "DISABLED";
    case LearnedCostUsage::kUsedBoundedSoftCost:
      return "USED_BOUNDED_SOFT_COST";
    case LearnedCostUsage::kFellBackToAnalytic:
      return "FELL_BACK_TO_ANALYTIC";
  }
  throw std::invalid_argument{"unknown LearnedCostUsage"};
}

[[nodiscard]] SecondaryCosts DecodeSecondaryCosts(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {},
              {"energy", "nonfatal_risk", "smoothness"});
  SecondaryCosts result;
  const auto decode_optional_cost =
      [&](std::string_view key) -> std::optional<double> {
    if (!value.contains(std::string{key})) {
      return std::nullopt;
    }
    const double decoded = DecodeNumber(
        value.at(std::string{key}),
        std::string{field_path} + "." + std::string{key});
    if (decoded < 0.0) {
      Fail(ErrorCode::kSchemaMismatch,
           std::string{field_path} + "." + std::string{key},
           "secondary cost must be non-negative");
    }
    return decoded;
  };
  result.energy = decode_optional_cost("energy");
  result.nonfatal_risk = decode_optional_cost("nonfatal_risk");
  result.smoothness = decode_optional_cost("smoothness");
  return result;
}

[[nodiscard]] Json EncodeSecondaryCosts(
    const SecondaryCosts& value) {
  Json result = Json::object();
  if (value.energy.has_value()) {
    result["energy"] = *value.energy;
  }
  if (value.nonfatal_risk.has_value()) {
    result["nonfatal_risk"] = *value.nonfatal_risk;
  }
  if (value.smoothness.has_value()) {
    result["smoothness"] = *value.smoothness;
  }
  return result;
}

[[nodiscard]] CallDiagnostics DecodeCallDiagnostics(
    const Json& value,
    std::string_view field_path) {
  CheckObject(value,
              field_path,
              {"api_latency_ns",
               "termination_reason",
               "expanded_state_count",
               "candidate_count",
               "resource_limit_hit",
               "learned_cost_usage"},
              {"final_epsilon",
               "expected_execution_time_ns",
               "secondary_costs",
               "reopened_state_count",
               "learned_cost_snapshot_ref",
               "message_codes"});
  CallDiagnostics result{
      .api_latency = DurationNanoseconds{ParseNanoseconds(
          Member(value, "api_latency_ns", field_path),
          std::string{field_path} + ".api_latency_ns",
          false)},
      .termination_reason = DecodeReasonCode(
          Member(value, "termination_reason", field_path),
          std::string{field_path} + ".termination_reason"),
      .expanded_state_count = DecodeUnsignedInteger(
          Member(value, "expanded_state_count", field_path),
          std::string{field_path} + ".expanded_state_count"),
      .candidate_count = DecodeUnsignedInteger(
          Member(value, "candidate_count", field_path),
          std::string{field_path} + ".candidate_count"),
      .resource_limit_hit = DecodeBoolean(
          Member(value, "resource_limit_hit", field_path),
          std::string{field_path} + ".resource_limit_hit"),
      .learned_cost_usage = DecodeLearnedCostUsage(
          Member(value, "learned_cost_usage", field_path),
          std::string{field_path} + ".learned_cost_usage"),
  };
  if (value.contains("final_epsilon")) {
    const double epsilon = DecodeNumber(
        value.at("final_epsilon"),
        std::string{field_path} + ".final_epsilon");
    if (epsilon < 1.0) {
      Fail(ErrorCode::kSchemaMismatch,
           std::string{field_path} + ".final_epsilon",
           "final epsilon must be at least one");
    }
    result.final_epsilon = epsilon;
  }
  if (value.contains("expected_execution_time_ns")) {
    result.expected_execution_time =
        DurationNanoseconds{ParseNanoseconds(
            value.at("expected_execution_time_ns"),
            std::string{field_path} +
                ".expected_execution_time_ns",
            false)};
  }
  if (value.contains("secondary_costs")) {
    result.secondary_costs = DecodeSecondaryCosts(
        value.at("secondary_costs"),
        std::string{field_path} + ".secondary_costs");
  }
  if (value.contains("reopened_state_count")) {
    result.reopened_state_count = DecodeUnsignedInteger(
        value.at("reopened_state_count"),
        std::string{field_path} + ".reopened_state_count");
  }
  if (value.contains("learned_cost_snapshot_ref")) {
    result.learned_cost_snapshot_ref = DecodeContentRef(
        value.at("learned_cost_snapshot_ref"),
        std::string{field_path} + ".learned_cost_snapshot_ref");
  }
  if (value.contains("message_codes")) {
    result.message_codes = DecodeReasonCodeArray(
        value.at("message_codes"),
        std::string{field_path} + ".message_codes",
        64U);
  }
  const bool has_snapshot =
      result.learned_cost_snapshot_ref.has_value();
  if (result.learned_cost_usage == LearnedCostUsage::kDisabled &&
      has_snapshot) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".learned_cost_snapshot_ref",
         "DISABLED learned-cost usage forbids a snapshot ref");
  }
  if (result.learned_cost_usage != LearnedCostUsage::kDisabled &&
      !has_snapshot) {
    Fail(ErrorCode::kSchemaMismatch,
         std::string{field_path} + ".learned_cost_snapshot_ref",
         "enabled or fallback learned-cost usage requires a snapshot ref");
  }
  return result;
}

[[nodiscard]] Json EncodeCallDiagnostics(
    const CallDiagnostics& value) {
  Json result{
      {"api_latency_ns",
       EncodeNonNegativeNanoseconds(
           value.api_latency.value,
           "call_diagnostics.api_latency_ns")},
      {"termination_reason", value.termination_reason},
      {"expanded_state_count", value.expanded_state_count},
      {"candidate_count", value.candidate_count},
      {"resource_limit_hit", value.resource_limit_hit},
      {"learned_cost_usage",
       EncodeLearnedCostUsage(value.learned_cost_usage)},
  };
  if (value.final_epsilon.has_value()) {
    result["final_epsilon"] = *value.final_epsilon;
  }
  if (value.expected_execution_time.has_value()) {
    result["expected_execution_time_ns"] =
        EncodeNonNegativeNanoseconds(
            value.expected_execution_time->value,
            "call_diagnostics.expected_execution_time_ns");
  }
  if (value.secondary_costs.has_value()) {
    result["secondary_costs"] =
        EncodeSecondaryCosts(*value.secondary_costs);
  }
  if (value.reopened_state_count != 0U) {
    result["reopened_state_count"] = value.reopened_state_count;
  }
  if (value.learned_cost_snapshot_ref.has_value()) {
    result["learned_cost_snapshot_ref"] =
        EncodeContentRef(*value.learned_cost_snapshot_ref);
  }
  if (!value.message_codes.empty()) {
    result["message_codes"] = value.message_codes;
  }
  return result;
}

[[nodiscard]] bool IsReadyOutcome(PlanningOutcome outcome) {
  return outcome == PlanningOutcome::kNewReferenceReady ||
         outcome == PlanningOutcome::kSafeFrontierReferenceReady;
}

void CheckResponseRelations(const PlanningResponse& response) {
  const bool activates =
      response.execution_directive ==
      ExecutionDirective::kActivateNewBundle;
  if (activates != response.new_reference_bundle.has_value()) {
    Fail(ErrorCode::kSchemaMismatch,
         "$.new_reference_bundle",
         activates
             ? "activation requires new_reference_bundle"
             : "non-activation directive forbids new_reference_bundle");
  }
  const bool continues =
      response.execution_directive ==
          ExecutionDirective::kContinueActiveBundle ||
      response.execution_directive ==
          ExecutionDirective::kContinueCommittedJump;
  if (continues != response.active_bundle_ref.has_value()) {
    Fail(ErrorCode::kSchemaMismatch,
         "$.active_bundle_ref",
         continues
             ? "continue directive requires active_bundle_ref"
             : "non-continue directive forbids active_bundle_ref");
  }
  if (IsReadyOutcome(response.planning_outcome) != activates) {
    Fail(ErrorCode::kSchemaMismatch,
         "$.planning_outcome",
         "READY outcome and activation directive must appear together");
  }
  if (response.execution_directive ==
          ExecutionDirective::kHoldStationary &&
      (response.new_reference_bundle.has_value() ||
       IsReadyOutcome(response.planning_outcome))) {
    Fail(ErrorCode::kSchemaMismatch,
         "$.execution_directive",
         "stationary hold cannot carry a READY outcome or new bundle");
  }
  if (response.planning_outcome ==
      PlanningOutcome::kActiveReferenceInvalidated) {
    const auto directive = response.execution_directive;
    if (directive != ExecutionDirective::kHoldStationary &&
        directive != ExecutionDirective::kContinueCommittedJump &&
        directive != ExecutionDirective::kNoSafePlannerReference) {
      Fail(ErrorCode::kSchemaMismatch,
           "$.execution_directive",
           "invalid directive for active-reference invalidation");
    }
  }
}

void RequireResolvedObject(
    const ContractObjectRegistry& registry,
    const ContentRef& ref,
    ContractObjectKind expected_kind,
    std::string_view field_path) {
  const auto resolved = registry.Resolve(ref, expected_kind);
  if (!IsOk(resolved)) {
    Error error = std::get<Error>(resolved);
    if (error.code != ErrorCode::kMissingRegistryObject) {
      error.code = ErrorCode::kMissingRegistryObject;
    }
    error.field_path = std::string{field_path};
    if (error.message.empty()) {
      error.message = "required contract object is absent from registry";
    }
    throw DecodeFailure{std::move(error)};
  }
  const auto& object =
      std::get<std::shared_ptr<const ImmutableContractObject>>(resolved);
  if (!object) {
    Fail(ErrorCode::kMissingRegistryObject,
         field_path,
         "registry returned a null contract object");
  }
  if (object->kind() != expected_kind || object->content_ref() != ref) {
    Fail(ErrorCode::kInconsistentSnapshot,
         field_path,
         "registry returned a mismatched contract object");
  }
}

void ResolveGeometricPathObjects(
    const GeometricPath& path,
    const ContractObjectRegistry& registry,
    std::string_view field_path) {
  if (const auto* chain =
          std::get_if<ValidatedPrimitiveChain>(&path)) {
    for (std::size_t index = 0; index < chain->primitives.size();
         ++index) {
      RequireResolvedObject(
          registry,
          chain->primitives[index].validation_ref,
          ContractObjectKind::kCertification,
          std::string{field_path} + ".primitives[" +
              std::to_string(index) + "].validation_ref");
    }
  }
}

void ResolvePlatformReferenceObjects(
    const PlatformReference& reference,
    const ContractObjectRegistry& registry,
    std::string_view field_path) {
  std::visit(
      [&](const auto& concrete) {
        using T = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<T, WheeledReference>) {
          for (std::size_t index = 0; index < concrete.segments.size();
               ++index) {
            if (const auto* drive =
                    std::get_if<DriveSegment>(
                        &concrete.segments[index])) {
              ResolveGeometricPathObjects(
                  drive->geometric_path,
                  registry,
                  std::string{field_path} + ".segments[" +
                      std::to_string(index) + "].geometric_path");
            }
          }
          RequireResolvedObject(
              registry,
              concrete.safe_stop_anchor.terrain_certification_ref,
              ContractObjectKind::kCertification,
              std::string{field_path} +
                  ".safe_stop_anchor.terrain_certification_ref");
        } else if constexpr (
            std::is_same_v<T, LeggedBodyReference>) {
          ResolveGeometricPathObjects(
              concrete.geometric_path,
              registry,
              std::string{field_path} + ".geometric_path");
          RequireResolvedObject(
              registry,
              concrete.terrain_normal_envelope
                  .source_terrain_certification_ref,
              ContractObjectKind::kCertification,
              std::string{field_path} +
                  ".terrain_normal_envelope."
                  "source_terrain_certification_ref");
          RequireResolvedObject(
              registry,
              concrete.safe_stop_anchor.terrain_certification_ref,
              ContractObjectKind::kCertification,
              std::string{field_path} +
                  ".safe_stop_anchor.terrain_certification_ref");
        } else {
          RequireResolvedObject(
              registry,
              concrete.ground_hold_anchor.terrain_certification_ref,
              ContractObjectKind::kCertification,
              std::string{field_path} +
                  ".ground_hold_anchor.terrain_certification_ref");
          RequireResolvedObject(
              registry,
              concrete.next_landing_region.terrain_certification_ref,
              ContractObjectKind::kCertification,
              std::string{field_path} +
                  ".next_landing_region.terrain_certification_ref");
          RequireResolvedObject(
              registry,
              concrete.jump_boundary.gravity_model_ref,
              ContractObjectKind::kGravityModel,
              std::string{field_path} +
                  ".jump_boundary.gravity_model_ref");
          RequireResolvedObject(
              registry,
              concrete.jump_boundary.actuator_or_impulse_profile_ref,
              ContractObjectKind::kActuatorOrImpulseProfile,
              std::string{field_path} +
                  ".jump_boundary.actuator_or_impulse_profile_ref");
          RequireResolvedObject(
              registry,
              concrete.predicted_landing_footprint
                  .source_error_model_ref,
              ContractObjectKind::kDeterministicErrorModel,
              std::string{field_path} +
                  ".predicted_landing_footprint."
                  "source_error_model_ref");
          RequireResolvedObject(
              registry,
              concrete.certified_flight_tube
                  .body_rotation_envelope_ref,
              ContractObjectKind::kBodyRotationEnvelope,
              std::string{field_path} +
                  ".certified_flight_tube."
                  "body_rotation_envelope_ref");
          RequireResolvedObject(
              registry,
              concrete.certified_flight_tube.error_model_ref,
              ContractObjectKind::kDeterministicErrorModel,
              std::string{field_path} +
                  ".certified_flight_tube.error_model_ref");
          RequireResolvedObject(
              registry,
              concrete.attitude_boundary.certification_ref,
              ContractObjectKind::kCertification,
              std::string{field_path} +
                  ".attitude_boundary.certification_ref");
          RequireResolvedObject(
              registry,
              concrete.physical_certification_ref,
              ContractObjectKind::kCertification,
              std::string{field_path} +
                  ".physical_certification_ref");
        }
      },
      reference);
}

void ResolveReferenceBundleObjects(
    const ReferenceBundle& bundle,
    const ContractObjectRegistry& registry,
    std::string_view field_path) {
  ResolvePlatformReferenceObjects(
      bundle.platform_reference,
      registry,
      std::string{field_path} + ".platform_reference");
  for (std::size_t index = 0;
       index < bundle.validation_summary.certificate_refs.size();
       ++index) {
    RequireResolvedObject(
        registry,
        bundle.validation_summary.certificate_refs[index],
        ContractObjectKind::kCertification,
        std::string{field_path} +
            ".validation_summary.certificate_refs[" +
            std::to_string(index) + "]");
  }
}

[[nodiscard]] PlanningResponse DecodePlanningResponseValue(
    const Json& root,
    const ReferenceActivationContext& activation_context) {
  CheckSchemaVersion(root, kPlanningResponseSchemaVersion, "$");
  CheckObject(root,
              "$",
              {"schema_version",
               "request_id",
               "response_time",
               "planning_outcome",
               "execution_directive",
               "reason_code",
               "call_diagnostics"},
              {"active_bundle_ref", "new_reference_bundle"});
  PlanningResponse result{
      .request_id =
          DecodeIdentifier(root.at("request_id"), "$.request_id"),
      .response_time =
          DecodeClockStamp(root.at("response_time"), "$.response_time"),
      .planning_outcome = DecodePlanningOutcome(
          root.at("planning_outcome"), "$.planning_outcome"),
      .execution_directive = DecodeExecutionDirective(
          root.at("execution_directive"), "$.execution_directive"),
      .reason_code =
          DecodeReasonCode(root.at("reason_code"), "$.reason_code"),
      .call_diagnostics = DecodeCallDiagnostics(
          root.at("call_diagnostics"), "$.call_diagnostics"),
  };
  if (root.contains("active_bundle_ref")) {
    result.active_bundle_ref = DecodeContentRef(
        root.at("active_bundle_ref"), "$.active_bundle_ref");
  }
  if (root.contains("new_reference_bundle")) {
    result.new_reference_bundle = DecodeReferenceBundle(
        root.at("new_reference_bundle"), "$.new_reference_bundle");
  }
  CheckResponseRelations(result);
  if (result.new_reference_bundle.has_value()) {
    ResolveReferenceBundleObjects(
        *result.new_reference_bundle,
        activation_context.registry,
        "$.new_reference_bundle");
  }
  return result;
}

[[nodiscard]] Json EncodePlanningResponseValue(
    const PlanningResponse& response) {
  Json result{
      {"schema_version", kPlanningResponseSchemaVersion},
      {"request_id", response.request_id},
      {"response_time", EncodeClockStamp(response.response_time)},
      {"planning_outcome",
       EncodePlanningOutcome(response.planning_outcome)},
      {"execution_directive",
       EncodeExecutionDirective(response.execution_directive)},
      {"reason_code", response.reason_code},
      {"call_diagnostics",
       EncodeCallDiagnostics(response.call_diagnostics)},
  };
  if (response.active_bundle_ref.has_value()) {
    result["active_bundle_ref"] =
        EncodeContentRef(*response.active_bundle_ref);
  }
  if (response.new_reference_bundle.has_value()) {
    result["new_reference_bundle"] =
        EncodeReferenceBundle(*response.new_reference_bundle);
  }
  return result;
}

}  // namespace

Result<PlanningResponse> JsonCodec::DecodePlanningResponse(
    std::string_view payload,
    const ReferenceActivationContext& activation_context) {
  try {
    return DecodePlanningResponseValue(
        ParseStrictJson(payload), activation_context);
  } catch (const DecodeFailure& failure) {
    return failure.error();
  } catch (const std::exception& exception) {
    return Error{ErrorCode::kInvalidArgument, "$", exception.what()};
  }
}

std::string JsonCodec::EncodePlanningResponse(
    const PlanningResponse& response) {
  try {
    CheckResponseRelations(response);
    const Json encoded = EncodePlanningResponseValue(response);
    CheckFiniteJsonTree(encoded, "$");
    return encoded.dump();
  } catch (const DecodeFailure& failure) {
    throw std::invalid_argument{
        failure.error().field_path + ": " + failure.error().message};
  }
}

namespace {

[[nodiscard]] Result<Sha256Digest> HashCanonicalJson(
    Json encoded,
    std::string_view field_path) {
  try {
    CheckFiniteJsonTree(encoded, "$");
    const auto canonical = JcsCanonicalizer::Canonicalize(encoded);
    if (!IsOk(canonical)) {
      return std::get<Error>(canonical);
    }
    return Sha256Hex(std::get<std::string>(canonical));
  } catch (const DecodeFailure& failure) {
    return failure.error();
  } catch (const std::exception& exception) {
    return Error{
        ErrorCode::kInvalidArgument,
        std::string{field_path},
        exception.what(),
    };
  }
}

}  // namespace

Result<Sha256Digest> CanonicalReferenceHash(
    const PlatformReference& reference) {
  try {
    Json encoded = EncodePlatformReference(reference);
    encoded.erase("reference_hash");
    return HashCanonicalJson(
        std::move(encoded), "platform_reference");
  } catch (const DecodeFailure& failure) {
    return failure.error();
  } catch (const std::exception& exception) {
    return Error{
        ErrorCode::kInvalidArgument,
        "platform_reference",
        exception.what(),
    };
  }
}

namespace {

[[nodiscard]] std::string_view EncodeLayerKind(LayerKind kind) {
  switch (kind) {
    case LayerKind::kKnownMask:
      return "KNOWN_MASK";
    case LayerKind::kElevation:
      return "ELEVATION";
    case LayerKind::kTerrainNormal:
      return "TERRAIN_NORMAL";
    case LayerKind::kRoughness:
      return "ROUGHNESS";
    case LayerKind::kHardObstacle:
      return "HARD_OBSTACLE";
    case LayerKind::kConfidence:
      return "CONFIDENCE";
    case LayerKind::kEsdf:
      return "ESDF";
    case LayerKind::kStaticSpeedLimit:
      return "STATIC_SPEED_LIMIT";
  }
  throw std::invalid_argument{"unknown LayerKind"};
}

[[nodiscard]] Json EncodeMapSnapshot(
    const ImmutableMapSnapshot& snapshot) {
  Json layers = Json::array();
  for (const LayerManifestEntry& entry : snapshot.layer_manifest()) {
    layers.push_back({
        {"layer_name", EncodeLayerKind(entry.layer_kind)},
        {"content_ref", EncodeContentRef(entry.content_ref)},
    });
  }
  return {
      {"snapshot_ref", EncodeContentRef(snapshot.snapshot_ref())},
      {"frame_id", snapshot.frame_id()},
      {"source_time", EncodeClockStamp(snapshot.source_time())},
      {"map_revision", snapshot.map_revision()},
      {"bounds",
       {{"minimum_m", EncodeVec3(snapshot.bounds().minimum_m)},
        {"maximum_m", EncodeVec3(snapshot.bounds().maximum_m)}}},
      {"resolution_m", snapshot.geometry().resolution_m},
      {"layer_manifest", std::move(layers)},
      {"immutable_data_handle", snapshot.immutable_data_handle()},
  };
}

[[nodiscard]] Json EncodePlatformState(
    const PlatformState& state,
    PlatformType platform_type) {
  if (platform_type == PlatformType::kHopper) {
    const auto* hopper = std::get_if<HopperState>(&state);
    if (hopper == nullptr) {
      throw std::invalid_argument{
          "HOPPER request requires HopperState"};
    }
    return EncodeHopperState(*hopper);
  }
  const auto* ground = std::get_if<WheeledOrLeggedState>(&state);
  if (ground == nullptr) {
    throw std::invalid_argument{
        "WHEELED/LEGGED request requires WheeledOrLeggedState"};
  }
  return EncodeWheeledOrLeggedState(*ground);
}

[[nodiscard]] Json EncodePlanningRequestValue(
    const PlanningRequest& request) {
  if (!request.map_snapshot) {
    throw std::invalid_argument{"request map_snapshot is null"};
  }
  if (!request.safety_capability) {
    throw std::invalid_argument{"request safety_capability is null"};
  }
  if (!request.algorithm_config) {
    throw std::invalid_argument{"request algorithm_config is null"};
  }
  Json result{
      {"schema_version", kPlanningRequestSchemaVersion},
      {"request_id", request.request_id},
      {"request_time", EncodeClockStamp(request.request_time)},
      {"state_time", EncodeClockStamp(request.state_time)},
      {"frame_id", request.frame_id},
      {"platform_type", EncodePlatformType(request.platform_type)},
      {"current_state",
       EncodePlatformState(request.current_state, request.platform_type)},
      {"goal", EncodeGoalRegion(request.goal)},
      {"map_snapshot", EncodeMapSnapshot(*request.map_snapshot)},
      {"safety_capability_ref",
       EncodeContentRef(request.safety_capability->content_ref)},
      {"algorithm_config_ref",
       EncodeContentRef(request.algorithm_config->content_ref)},
  };
  if (request.previous_execution_context.has_value()) {
    result["previous_execution_context"] =
        EncodePreviousExecutionContext(
            *request.previous_execution_context);
  }
  if (request.learned_cost_snapshot.has_value()) {
    result["learned_cost_snapshot"] = {
        {"snapshot_ref",
         EncodeContentRef(
             request.learned_cost_snapshot->snapshot_ref)},
        {"registry_handle",
         request.learned_cost_snapshot->registry_handle},
        {"ready_before_request", true},
        {"hard_feasibility_authority", "NONE"},
    };
  }
  return result;
}

}  // namespace

std::string JsonCodec::EncodePlanningRequest(
    const PlanningRequest& request) {
  try {
    const Json encoded = EncodePlanningRequestValue(request);
    CheckFiniteJsonTree(encoded, "$");
    return encoded.dump();
  } catch (const DecodeFailure& failure) {
    throw std::invalid_argument{
        failure.error().field_path + ": " + failure.error().message};
  }
}

Result<Sha256Digest> CanonicalComponentHash(
    const RouteSkeletonContent& content) {
  try {
    return HashCanonicalJson(
        EncodeRouteSkeletonContent(content),
        "route_skeleton.content");
  } catch (const DecodeFailure& failure) {
    return failure.error();
  } catch (const std::exception& exception) {
    return Error{
        ErrorCode::kInvalidArgument,
        "route_skeleton.content",
        exception.what(),
    };
  }
}

Result<Sha256Digest> CanonicalComponentHash(
    const ReferenceViewContent& content) {
  try {
    return HashCanonicalJson(
        EncodeReferenceViewContent(content),
        "reference_view.content");
  } catch (const DecodeFailure& failure) {
    return failure.error();
  } catch (const std::exception& exception) {
    return Error{
        ErrorCode::kInvalidArgument,
        "reference_view.content",
        exception.what(),
    };
  }
}

Result<Sha256Digest> CanonicalBundleHash(
    const ReferenceBundle& bundle) {
  try {
    Json encoded = EncodeReferenceBundle(bundle);
    encoded.erase("bundle_hash");
    return HashCanonicalJson(std::move(encoded), "bundle_hash");
  } catch (const DecodeFailure& failure) {
    return failure.error();
  } catch (const std::exception& exception) {
    return Error{
        ErrorCode::kInvalidArgument,
        "bundle_hash",
        exception.what(),
    };
  }
}

}  // namespace lunar::planning::v3
