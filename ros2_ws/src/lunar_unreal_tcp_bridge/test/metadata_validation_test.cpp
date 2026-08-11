#include "lunar_unreal_tcp_bridge/metadata_validation.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace lunar::unreal_tcp {
namespace {

nlohmann::json Hello() {
  return {
      {"protocol", "lunar-unreal-tcp/v1"},
      {"role", "ROS_CLIENT"},
      {"client_nonce", "nonce"},
      {"platform_type", "WHEELED"},
      {"robot_count", 1},
      {"state_rate_hz", 20},
      {"map_rate_hz", 5},
      {"heartbeat_rate_hz", 1},
      {"maximum_body_bytes", 8'388'608},
  };
}

nlohmann::json Pose() {
  return {
      {"position_cm", {0.0, 0.0, 0.0}},
      {"quaternion_xyzw", {0.0, 0.0, 0.0, 1.0}},
  };
}

struct MetadataCase final {
  MessageType type;
  const char* required_field;
  nlohmann::json metadata;
};

std::vector<MetadataCase> InSessionSchemas() {
  const std::string session = "123e4567-e89b-12d3-a456-426614174000";
  const std::vector<double> identity{
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0};
  return {
      {
          MessageType::kHelloAck,
          "scene_id",
          {
              {"session_id", session}, {"scene_id", "scene"},
              {"robot_id", "rover"}, {"engine_version", "5.0.1"},
              {"agx_plugin_version", "UNKNOWN"},
              {"coordinate_convention", "UE_NATIVE"},
              {"length_unit_to_m", 0.01}, {"handedness", "LEFT"},
              {"up_axis", "Z"},
              {"T_base_link_from_unreal_root", identity},
              {"T_base_footprint_from_base_link", identity},
              {"local_map_width", 320}, {"local_map_height", 320},
              {"resolution_native_cm", 20.0}, {"server_nonce", "server"},
              {"maximum_body_bytes", 8'388'608},
              {"calibration_hash", "sha256:test"},
          },
      },
      {
          MessageType::kControl,
          "command_id",
          {{"session_id", session}, {"command", "START"},
           {"command_id", "control-1"}},
      },
      {
          MessageType::kControlAck,
          "status",
          {{"session_id", session}, {"command_id", "control-1"},
           {"status", "ACCEPTED"}, {"reason_code", ""}},
      },
      {
          MessageType::kLocalElevationMap,
          "cell_zero_center_world_cm",
          {
              {"session_id", session}, {"width", 320}, {"height", 320},
              {"resolution_cm", 20.0},
              {"cell_zero_center_world_cm", {0.0, 0.0, 0.0}},
              {"u_axis_world", {1.0, 0.0, 0.0}},
              {"v_axis_world", {0.0, 1.0, 0.0}},
              {"sensor_pose_world", Pose()}, {"robot_pose_world", Pose()},
              {"elevation_encoding", "float32_le"},
              {"valid_encoding", "bitset_lsb0"},
          },
      },
      {
          MessageType::kMotionReference,
          "plan_id",
          {
              {"session_id", session}, {"plan_id", "plan-1"},
              {"platform_type", "WHEELED"}, {"input_time_ns", 10},
              {"point_count", 2}, {"source_frame", "base_footprint"},
              {"target_unreal_reference", "robot_root"},
              {"execution_directive", "ACTIVATE_NEW_REFERENCE"},
              {"calibration_hash", "sha256:test"},
          },
      },
      {
          MessageType::kExecutionFeedback,
          "state",
          {
              {"session_id", session}, {"platform_type", "WHEELED"},
              {"plan_id", "plan-1"}, {"segment_id", "plan-1"},
              {"sequence", 1}, {"state", "ACCEPTED"},
              {"reason_code", ""},
          },
      },
      {
          MessageType::kHeartbeat,
          "last_received_sequence",
          {{"session_id", session}, {"last_received_sequence", 8}},
      },
      {
          MessageType::kError,
          "reason_code",
          {{"reason_code", "BUSY"}, {"recoverable", false}},
      },
  };
}

TEST(MetadataValidation, AcceptsExactHelloSchema) {
  EXPECT_FALSE(ValidateMetadata(MessageType::kHello, Hello()).has_value());
}

TEST(MetadataValidation, RejectsMissingAndUnknownFields) {
  auto missing = Hello();
  missing.erase("client_nonce");
  const auto missing_result = ValidateMetadata(MessageType::kHello, missing);
  ASSERT_TRUE(missing_result.has_value());
  EXPECT_EQ(missing_result->reason_code, "METADATA_REQUIRED_CLIENT_NONCE");

  auto unknown = Hello();
  unknown["silent_extension"] = true;
  const auto unknown_result = ValidateMetadata(MessageType::kHello, unknown);
  ASSERT_TRUE(unknown_result.has_value());
  EXPECT_EQ(unknown_result->reason_code, "METADATA_UNKNOWN_SILENT_EXTENSION");
}

TEST(MetadataValidation, RejectsWrongTypesRangesAndNonfiniteNumbers) {
  auto wrong_type = Hello();
  wrong_type["robot_count"] = "one";
  EXPECT_EQ(
      ValidateMetadata(MessageType::kHello, wrong_type)->reason_code,
      "METADATA_TYPE_ROBOT_COUNT");

  auto wrong_platform = Hello();
  wrong_platform["platform_type"] = "HOPPER";
  EXPECT_EQ(
      ValidateMetadata(MessageType::kHello, wrong_platform)->reason_code,
      "METADATA_VALUE_PLATFORM_TYPE");

  nlohmann::json state = {
      {"session_id", "123e4567-e89b-12d3-a456-426614174000"},
      {"position_cm", {0.0, 0.0, 0.0}},
      {"quaternion_xyzw", {0.0, 0.0, 0.0, 1.0}},
      {"linear_velocity_cmps", {0.0, 0.0, 0.0}},
      {"angular_velocity_radps", {0.0, 0.0, 0.0}},
      {"control_state", "READY"},
      {"simulation_covariance_profile", "sim-default"},
  };
  state["position_cm"][1] = std::numeric_limits<double>::infinity();
  EXPECT_EQ(
      ValidateMetadata(MessageType::kRobotState, state)->reason_code,
      "METADATA_NONFINITE_POSITION_CM");
}

TEST(MetadataValidation, RequiresSessionForInSessionMessages) {
  nlohmann::json heartbeat = {
      {"session_id", ""},
      {"last_received_sequence", 0},
  };
  EXPECT_EQ(
      ValidateMetadata(MessageType::kHeartbeat, heartbeat)->reason_code,
      "METADATA_VALUE_SESSION_ID");
}

TEST(MetadataValidation, AcceptsEveryFrozenMessageSchema) {
  for (const auto& test_case : InSessionSchemas()) {
    const auto result = ValidateMetadata(test_case.type, test_case.metadata);
    EXPECT_FALSE(result.has_value())
        << static_cast<std::uint16_t>(test_case.type) << ' '
        << (result ? result->reason_code : "");
  }
}

TEST(MetadataValidation, RejectsMissingFieldForEveryFrozenMessageSchema) {
  for (auto test_case : InSessionSchemas()) {
    test_case.metadata.erase(test_case.required_field);
    const auto result = ValidateMetadata(test_case.type, test_case.metadata);
    ASSERT_TRUE(result.has_value())
        << static_cast<std::uint16_t>(test_case.type);
    EXPECT_EQ(
        result->reason_code,
        std::string{"METADATA_REQUIRED_"} +
            [&test_case] {
              std::string value{test_case.required_field};
              std::transform(value.begin(), value.end(), value.begin(),
                             [](const char character) {
                               return character == '_' ? '_' :
                                   static_cast<char>(std::toupper(character));
                             });
              return value;
            }());
  }
}

}  // namespace
}  // namespace lunar::unreal_tcp
