#include "lunar_unreal_tcp_bridge/metadata_validation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace lunar::unreal_tcp {
namespace {

using Json = nlohmann::json;

[[nodiscard]] std::string Upper(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const char character : value) {
    if (character >= 'a' && character <= 'z') {
      result.push_back(static_cast<char>(character - 'a' + 'A'));
    } else if ((character >= 'A' && character <= 'Z') ||
               (character >= '0' && character <= '9')) {
      result.push_back(character);
    } else {
      result.push_back('_');
    }
  }
  return result;
}

[[nodiscard]] std::optional<MetadataError> Error(
    std::string reason_code) {
  return MetadataError{.reason_code = std::move(reason_code)};
}

[[nodiscard]] bool NonemptyString(const Json& value) {
  return value.is_string() && !value.get_ref<const std::string&>().empty();
}

[[nodiscard]] bool Integer(const Json& value) {
  return value.is_number_integer() || value.is_number_unsigned();
}

[[nodiscard]] bool UnsignedInteger(const Json& value) {
  if (!Integer(value)) {
    return false;
  }
  if (value.is_number_unsigned()) {
    return true;
  }
  return value.get<std::int64_t>() >= 0;
}

[[nodiscard]] bool FiniteNumber(const Json& value) {
  return value.is_number() && std::isfinite(value.get<double>());
}

[[nodiscard]] bool FiniteArray(const Json& value, const std::size_t size) {
  return value.is_array() && value.size() == size &&
      std::ranges::all_of(value, FiniteNumber);
}

[[nodiscard]] bool Uuid(const Json& value) {
  if (!NonemptyString(value)) {
    return false;
  }
  const std::string& text = value.get_ref<const std::string&>();
  if (text.size() != 36U) {
    return false;
  }
  constexpr std::array<std::size_t, 4> hyphens{8U, 13U, 18U, 23U};
  for (std::size_t index = 0U; index < text.size(); ++index) {
    const bool separator = std::ranges::find(hyphens, index) != hyphens.end();
    if (separator) {
      if (text[index] != '-') {
        return false;
      }
      continue;
    }
    const char character = text[index];
    const bool hexadecimal =
        (character >= '0' && character <= '9') ||
        (character >= 'a' && character <= 'f') ||
        (character >= 'A' && character <= 'F');
    if (!hexadecimal) {
      return false;
    }
  }
  return true;
}

template<typename Required, typename Optional>
[[nodiscard]] std::optional<MetadataError> ShapeImpl(
    const Json& metadata, const Required& required, const Optional& optional) {
  if (!metadata.is_object()) {
    return Error("METADATA_TOP_LEVEL_NOT_OBJECT");
  }
  for (const auto& raw_field : required) {
    const std::string_view field{raw_field};
    if (!metadata.contains(std::string{field})) {
      return Error("METADATA_REQUIRED_" + Upper(field));
    }
  }
  std::set<std::string_view, std::less<>> allowed;
  for (const auto& field : required) {
    allowed.emplace(field);
  }
  for (const auto& field : optional) {
    allowed.emplace(field);
  }
  for (auto iterator = metadata.begin(); iterator != metadata.end(); ++iterator) {
    if (!allowed.contains(iterator.key())) {
      return Error("METADATA_UNKNOWN_" + Upper(iterator.key()));
    }
  }
  return std::nullopt;
}

template<typename Required>
[[nodiscard]] std::optional<MetadataError> Shape(
    const Json& metadata, const Required& required) {
  constexpr std::array<std::string_view, 0> optional{};
  return ShapeImpl(metadata, required, optional);
}

template<typename Required, typename Optional>
[[nodiscard]] std::optional<MetadataError> Shape(
    const Json& metadata, const Required& required, const Optional& optional) {
  return ShapeImpl(metadata, required, optional);
}

[[nodiscard]] const Json& Field(
    const Json& metadata, const std::string_view field) {
  return metadata.at(std::string{field});
}

[[nodiscard]] std::optional<MetadataError> TypeError(
    const Json& metadata, const std::string_view field,
    const bool valid) {
  if (!valid) {
    return Error("METADATA_TYPE_" + Upper(field));
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<MetadataError> ValidateSessionId(
    const Json& metadata) {
  if (!Uuid(metadata.at("session_id"))) {
    return Error("METADATA_VALUE_SESSION_ID");
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<MetadataError> ValidatePose(
    const Json& pose, const std::string_view field) {
  constexpr std::array required{"position_cm", "quaternion_xyzw"};
  if (const auto shape = Shape(pose, required)) {
    return Error("METADATA_TYPE_" + Upper(field));
  }
  if (!FiniteArray(pose.at("position_cm"), 3U) ||
      !FiniteArray(pose.at("quaternion_xyzw"), 4U)) {
    return Error("METADATA_NONFINITE_" + Upper(field));
  }
  const auto& quaternion = pose.at("quaternion_xyzw");
  double norm_squared = 0.0;
  for (const auto& component : quaternion) {
    const double value = component.get<double>();
    norm_squared += value * value;
  }
  if (!std::isfinite(norm_squared) || norm_squared <= 1.0e-24) {
    return Error("METADATA_VALUE_" + Upper(field));
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<MetadataError> ValidateHello(
    const Json& metadata) {
  constexpr std::array required{
      "client_nonce", "heartbeat_rate_hz", "map_rate_hz",
      "maximum_body_bytes", "platform_type", "protocol", "robot_count",
      "role", "state_rate_hz"};
  if (const auto shape = Shape(metadata, required)) {
    return shape;
  }
  for (const std::string_view field :
       {"client_nonce", "platform_type", "protocol", "role"}) {
    if (const auto error = TypeError(metadata, field,
                                     NonemptyString(Field(metadata, field)))) {
      return error;
    }
  }
  for (const std::string_view field :
       {"heartbeat_rate_hz", "map_rate_hz", "maximum_body_bytes",
        "robot_count", "state_rate_hz"}) {
    if (const auto error = TypeError(metadata, field,
                                     UnsignedInteger(Field(metadata, field)))) {
      return error;
    }
  }
  if (metadata.at("protocol") != "lunar-unreal-tcp/v1") {
    return Error("METADATA_VALUE_PROTOCOL");
  }
  if (metadata.at("role") != "ROS_CLIENT") {
    return Error("METADATA_VALUE_ROLE");
  }
  if (metadata.at("platform_type") != "WHEELED") {
    return Error("METADATA_VALUE_PLATFORM_TYPE");
  }
  if (metadata.at("robot_count") != 1 ||
      metadata.at("state_rate_hz") != 20 || metadata.at("map_rate_hz") != 5 ||
      metadata.at("heartbeat_rate_hz") != 1 ||
      metadata.at("maximum_body_bytes") != kMaximumBodyBytes) {
    return Error("METADATA_VALUE_HELLO_LIMITS");
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<MetadataError> ValidateHelloAck(
    const Json& metadata) {
  constexpr std::array required{
      "session_id", "scene_id", "robot_id", "engine_version",
      "agx_plugin_version", "coordinate_convention", "length_unit_to_m",
      "handedness", "up_axis", "T_base_link_from_unreal_root",
      "T_base_footprint_from_base_link", "local_map_width",
      "local_map_height", "resolution_native_cm", "server_nonce",
      "maximum_body_bytes", "calibration_hash"};
  if (const auto shape = Shape(metadata, required)) {
    return shape;
  }
  if (const auto session = ValidateSessionId(metadata)) {
    return session;
  }
  for (const std::string_view field :
       {"scene_id", "robot_id", "engine_version", "agx_plugin_version",
        "coordinate_convention", "handedness", "up_axis", "server_nonce",
        "calibration_hash"}) {
    if (!NonemptyString(Field(metadata, field))) {
      return Error("METADATA_TYPE_" + Upper(field));
    }
  }
  for (const std::string_view field :
       {"T_base_link_from_unreal_root", "T_base_footprint_from_base_link"}) {
    if (!FiniteArray(Field(metadata, field), 16U)) {
      return Error("METADATA_TYPE_" + Upper(field));
    }
  }
  if (!FiniteNumber(metadata.at("length_unit_to_m")) ||
      metadata.at("length_unit_to_m").get<double>() <= 0.0 ||
      !FiniteNumber(metadata.at("resolution_native_cm")) ||
      metadata.at("resolution_native_cm").get<double>() != 20.0) {
    return Error("METADATA_VALUE_HELLO_ACK_GEOMETRY");
  }
  if (!UnsignedInteger(metadata.at("local_map_width")) ||
      !UnsignedInteger(metadata.at("local_map_height")) ||
      !UnsignedInteger(metadata.at("maximum_body_bytes")) ||
      metadata.at("local_map_width") != 320 ||
      metadata.at("local_map_height") != 320 ||
      metadata.at("maximum_body_bytes") != kMaximumBodyBytes) {
    return Error("METADATA_VALUE_HELLO_ACK_LIMITS");
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<MetadataError> ValidateControl(
    const Json& metadata) {
  constexpr std::array required{"session_id", "command", "command_id"};
  if (const auto shape = Shape(metadata, required)) {
    return shape;
  }
  if (const auto session = ValidateSessionId(metadata)) {
    return session;
  }
  if (!NonemptyString(metadata.at("command_id")) ||
      !NonemptyString(metadata.at("command"))) {
    return Error("METADATA_TYPE_CONTROL");
  }
  static const std::set<std::string, std::less<>> commands{
      "START", "HOLD", "RESUME", "RESET_SESSION"};
  if (!commands.contains(metadata.at("command").get<std::string>())) {
    return Error("METADATA_VALUE_COMMAND");
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<MetadataError> ValidateControlAck(
    const Json& metadata) {
  constexpr std::array required{
      "session_id", "command_id", "status", "reason_code"};
  if (const auto shape = Shape(metadata, required)) {
    return shape;
  }
  if (const auto session = ValidateSessionId(metadata)) {
    return session;
  }
  if (!NonemptyString(metadata.at("command_id")) ||
      !NonemptyString(metadata.at("status")) ||
      !metadata.at("reason_code").is_string()) {
    return Error("METADATA_TYPE_CONTROL_ACK");
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<MetadataError> ValidateRobotState(
    const Json& metadata) {
  constexpr std::array required{
      "session_id", "position_cm", "quaternion_xyzw",
      "linear_velocity_cmps", "angular_velocity_radps", "control_state"};
  constexpr std::array optional{
      "pose_covariance", "twist_covariance",
      "simulation_covariance_profile"};
  if (const auto shape = Shape(metadata, required, optional)) {
    return shape;
  }
  if (const auto session = ValidateSessionId(metadata)) {
    return session;
  }
  for (const auto [field, size] :
       {std::pair{"position_cm", 3U}, std::pair{"quaternion_xyzw", 4U},
        std::pair{"linear_velocity_cmps", 3U},
        std::pair{"angular_velocity_radps", 3U}}) {
    if (!FiniteArray(metadata.at(field), size)) {
      return Error("METADATA_NONFINITE_" + Upper(field));
    }
  }
  if (!NonemptyString(metadata.at("control_state"))) {
    return Error("METADATA_TYPE_CONTROL_STATE");
  }
  const bool profile = metadata.contains("simulation_covariance_profile") &&
      NonemptyString(metadata.at("simulation_covariance_profile"));
  const bool covariance = metadata.contains("pose_covariance") &&
      metadata.contains("twist_covariance") &&
      FiniteArray(metadata.at("pose_covariance"), 36U) &&
      FiniteArray(metadata.at("twist_covariance"), 36U);
  if (!profile && !covariance) {
    return Error("METADATA_COVARIANCE_REQUIRED");
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<MetadataError> ValidateLocalMap(
    const Json& metadata) {
  constexpr std::array required{
      "session_id", "width", "height", "resolution_cm",
      "cell_zero_center_world_cm", "u_axis_world", "v_axis_world",
      "sensor_pose_world", "robot_pose_world", "elevation_encoding",
      "valid_encoding"};
  if (const auto shape = Shape(metadata, required)) {
    return shape;
  }
  if (const auto session = ValidateSessionId(metadata)) {
    return session;
  }
  if (!UnsignedInteger(metadata.at("width")) ||
      !UnsignedInteger(metadata.at("height")) ||
      metadata.at("width") != 320 || metadata.at("height") != 320 ||
      !FiniteNumber(metadata.at("resolution_cm")) ||
      metadata.at("resolution_cm") != 20.0) {
    return Error("METADATA_VALUE_LOCAL_MAP_GEOMETRY");
  }
  for (const std::string_view field :
       {"cell_zero_center_world_cm", "u_axis_world", "v_axis_world"}) {
    if (!FiniteArray(Field(metadata, field), 3U)) {
      return Error("METADATA_NONFINITE_" + Upper(field));
    }
  }
  for (const std::string_view field :
       {"sensor_pose_world", "robot_pose_world"}) {
    if (const auto pose = ValidatePose(Field(metadata, field), field)) {
      return pose;
    }
  }
  if (metadata.at("elevation_encoding") != "float32_le" ||
      metadata.at("valid_encoding") != "bitset_lsb0") {
    return Error("METADATA_VALUE_LOCAL_MAP_ENCODING");
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<MetadataError> ValidateMotionReference(
    const Json& metadata) {
  constexpr std::array required{
      "session_id", "plan_id", "platform_type", "input_time_ns",
      "point_count", "source_frame", "target_unreal_reference",
      "execution_directive", "calibration_hash"};
  if (const auto shape = Shape(metadata, required)) {
    return shape;
  }
  if (const auto session = ValidateSessionId(metadata)) {
    return session;
  }
  for (const std::string_view field :
       {"plan_id", "platform_type", "source_frame",
        "target_unreal_reference", "execution_directive",
        "calibration_hash"}) {
    if (!NonemptyString(Field(metadata, field))) {
      return Error("METADATA_TYPE_" + Upper(field));
    }
  }
  if (metadata.at("platform_type") != "WHEELED" ||
      metadata.at("execution_directive") != "ACTIVATE_NEW_REFERENCE") {
    return Error("METADATA_VALUE_MOTION_REFERENCE");
  }
  if (!Integer(metadata.at("input_time_ns")) ||
      !UnsignedInteger(metadata.at("point_count")) ||
      metadata.at("point_count") == 0) {
    return Error("METADATA_TYPE_MOTION_REFERENCE_TIME_OR_COUNT");
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<MetadataError> ValidateExecutionFeedback(
    const Json& metadata) {
  constexpr std::array required{
      "session_id", "platform_type", "plan_id", "segment_id", "sequence",
      "state", "reason_code"};
  if (const auto shape = Shape(metadata, required)) {
    return shape;
  }
  if (const auto session = ValidateSessionId(metadata)) {
    return session;
  }
  for (const std::string_view field :
       {"platform_type", "plan_id", "segment_id", "state"}) {
    if (!NonemptyString(Field(metadata, field))) {
      return Error("METADATA_TYPE_" + Upper(field));
    }
  }
  if (!metadata.at("reason_code").is_string() ||
      !UnsignedInteger(metadata.at("sequence")) ||
      metadata.at("sequence") == 0 ||
      metadata.at("platform_type") != "WHEELED") {
    return Error("METADATA_VALUE_EXECUTION_FEEDBACK");
  }
  const std::string state = metadata.at("state").get<std::string>();
  static const std::set<std::string, std::less<>> states{
      "ACCEPTED", "EXECUTING", "SEGMENT_COMPLETE", "FAILED", "CANCELED"};
  if (!states.contains(state) ||
      ((state == "FAILED" || state == "CANCELED") &&
       metadata.at("reason_code").get_ref<const std::string&>().empty())) {
    return Error("METADATA_VALUE_EXECUTION_FEEDBACK_STATE");
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<MetadataError> ValidateHeartbeat(
    const Json& metadata) {
  constexpr std::array required{"session_id", "last_received_sequence"};
  if (const auto shape = Shape(metadata, required)) {
    return shape;
  }
  if (const auto session = ValidateSessionId(metadata)) {
    return session;
  }
  if (!UnsignedInteger(metadata.at("last_received_sequence"))) {
    return Error("METADATA_TYPE_LAST_RECEIVED_SEQUENCE");
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<MetadataError> ValidateError(
    const Json& metadata) {
  constexpr std::array required{"reason_code", "recoverable"};
  constexpr std::array optional{"session_id"};
  if (const auto shape = Shape(metadata, required, optional)) {
    return shape;
  }
  if (!NonemptyString(metadata.at("reason_code")) ||
      !metadata.at("recoverable").is_boolean()) {
    return Error("METADATA_TYPE_ERROR");
  }
  if (metadata.contains("session_id") && !Uuid(metadata.at("session_id"))) {
    return Error("METADATA_VALUE_SESSION_ID");
  }
  return std::nullopt;
}

}  // namespace

std::optional<MetadataError> ValidateMetadata(
    const MessageType type, const nlohmann::json& metadata) {
  switch (type) {
    case MessageType::kHello:
      return ValidateHello(metadata);
    case MessageType::kHelloAck:
      return ValidateHelloAck(metadata);
    case MessageType::kControl:
      return ValidateControl(metadata);
    case MessageType::kControlAck:
      return ValidateControlAck(metadata);
    case MessageType::kRobotState:
      return ValidateRobotState(metadata);
    case MessageType::kLocalElevationMap:
      return ValidateLocalMap(metadata);
    case MessageType::kMotionReference:
      return ValidateMotionReference(metadata);
    case MessageType::kExecutionFeedback:
      return ValidateExecutionFeedback(metadata);
    case MessageType::kHeartbeat:
      return ValidateHeartbeat(metadata);
    case MessageType::kError:
      return ValidateError(metadata);
  }
  return Error("METADATA_MESSAGE_TYPE_UNKNOWN");
}

}  // namespace lunar::unreal_tcp
