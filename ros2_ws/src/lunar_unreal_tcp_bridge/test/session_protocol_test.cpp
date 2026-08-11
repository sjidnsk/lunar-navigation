#include "lunar_unreal_tcp_bridge/session_protocol.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace lunar::unreal_tcp {
namespace {

using namespace std::chrono_literals;

constexpr char kSession[] = "123e4567-e89b-12d3-a456-426614174000";

Frame HelloAck() {
  const std::vector<double> identity{
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0};
  Frame frame;
  frame.header.message_type = MessageType::kHelloAck;
  frame.header.sequence = 1U;
  frame.metadata = {
      {"session_id", kSession}, {"scene_id", "scene-1"},
      {"robot_id", "rover-1"}, {"engine_version", "5.0.1"},
      {"agx_plugin_version", "UNKNOWN"},
      {"coordinate_convention", "UE_NATIVE"},
      {"length_unit_to_m", 0.01}, {"handedness", "LEFT"},
      {"up_axis", "Z"}, {"T_base_link_from_unreal_root", identity},
      {"T_base_footprint_from_base_link", identity},
      {"local_map_width", 320}, {"local_map_height", 320},
      {"resolution_native_cm", 20.0}, {"server_nonce", "server"},
      {"maximum_body_bytes", 8'388'608},
      {"calibration_hash", "sha256:calibration"},
  };
  return frame;
}

Frame Heartbeat(const std::uint64_t sequence, const std::int64_t sim_time,
                std::string session = kSession) {
  Frame frame;
  frame.header.message_type = MessageType::kHeartbeat;
  frame.header.sequence = sequence;
  frame.header.simulation_time_ns = sim_time;
  frame.metadata = {
      {"session_id", std::move(session)},
      {"last_received_sequence", 1U},
  };
  return frame;
}

Frame RobotState(const std::uint64_t sequence) {
  Frame frame;
  frame.header.message_type = MessageType::kRobotState;
  frame.header.sequence = sequence;
  frame.header.simulation_time_ns = static_cast<std::int64_t>(sequence);
  frame.metadata = {
      {"session_id", kSession},
      {"position_cm", {0.0, 0.0, 0.0}},
      {"quaternion_xyzw", {0.0, 0.0, 0.0, 1.0}},
      {"linear_velocity_cmps", {0.0, 0.0, 0.0}},
      {"angular_velocity_radps", {0.0, 0.0, 0.0}},
      {"control_state", "IDLE"},
      {"simulation_covariance_profile", "deterministic"},
  };
  return frame;
}

Frame LocalMap(const std::uint64_t sequence) {
  const nlohmann::json pose{
      {"position_cm", {0.0, 0.0, 0.0}},
      {"quaternion_xyzw", {0.0, 0.0, 0.0, 1.0}},
  };
  Frame frame;
  frame.header.message_type = MessageType::kLocalElevationMap;
  frame.header.sequence = sequence;
  frame.header.simulation_time_ns = static_cast<std::int64_t>(sequence);
  frame.metadata = {
      {"session_id", kSession},
      {"width", 320},
      {"height", 320},
      {"resolution_cm", 20.0},
      {"cell_zero_center_world_cm", {0.0, 0.0, 0.0}},
      {"u_axis_world", {1.0, 0.0, 0.0}},
      {"v_axis_world", {0.0, 1.0, 0.0}},
      {"sensor_pose_world", pose},
      {"robot_pose_world", pose},
      {"elevation_encoding", "float32_le"},
      {"valid_encoding", "bitset_lsb0"},
  };
  return frame;
}

SessionProtocol Established(
    const SessionProtocol::SteadyTime now = SessionProtocol::SteadyTime{}) {
  SessionProtocol protocol{"client-nonce"};
  EXPECT_EQ(protocol.BuildHello().header.sequence, 1U);
  EXPECT_TRUE(protocol.AcceptHelloAck(HelloAck(), now).ok());
  return protocol;
}

TEST(SessionProtocol, HandshakeFreezesNewSessionAndCalibration) {
  SessionProtocol protocol{"client-nonce"};
  const Frame hello = protocol.BuildHello();
  EXPECT_EQ(protocol.state(), SessionState::kHandshaking);
  EXPECT_EQ(hello.header.message_type, MessageType::kHello);
  EXPECT_EQ(hello.header.sequence, 1U);
  EXPECT_EQ(hello.metadata.at("client_nonce"), "client-nonce");

  const auto result = protocol.AcceptHelloAck(
      HelloAck(), SessionProtocol::SteadyTime{});

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(protocol.state(), SessionState::kSyncing);
  ASSERT_TRUE(protocol.session().has_value());
  EXPECT_EQ(protocol.session()->session_id, kSession);
  EXPECT_EQ(protocol.session()->scene_id, "scene-1");
  EXPECT_EQ(protocol.session()->length_unit_to_m, 0.01);
  EXPECT_EQ(protocol.session()->calibration_hash, "sha256:calibration");
  EXPECT_EQ(protocol.last_received_sequence(), 1U);
}

TEST(SessionProtocol, RequiresSequenceOneThenStrictIncreaseAndAllowsForwardGap) {
  auto protocol = Established();
  EXPECT_TRUE(protocol.AcceptIncoming(
      Heartbeat(2U, 10), SessionProtocol::SteadyTime{}).ok());

  const auto duplicate = protocol.AcceptIncoming(
      Heartbeat(2U, 11), SessionProtocol::SteadyTime{});
  EXPECT_FALSE(duplicate.ok());
  EXPECT_TRUE(duplicate.force_hold);
  EXPECT_EQ(duplicate.reason_code, "SEQUENCE_NOT_INCREASING");

  auto gap_protocol = Established();
  EXPECT_TRUE(gap_protocol.AcceptIncoming(
      Heartbeat(4U, 10), SessionProtocol::SteadyTime{}).ok());
  EXPECT_EQ(gap_protocol.last_received_sequence(), 4U);
}

TEST(SessionProtocol, RejectsWrongSessionAndSimulationTimeRollback) {
  auto wrong_session = Established();
  const auto wrong = wrong_session.AcceptIncoming(
      Heartbeat(2U, 10, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
      SessionProtocol::SteadyTime{});
  EXPECT_FALSE(wrong.ok());
  EXPECT_EQ(wrong.reason_code, "SESSION_ID_MISMATCH");

  auto rollback = Established();
  EXPECT_TRUE(rollback.AcceptIncoming(
      Heartbeat(2U, 10), SessionProtocol::SteadyTime{}).ok());
  const auto backwards = rollback.AcceptIncoming(
      Heartbeat(3U, 9), SessionProtocol::SteadyTime{});
  EXPECT_FALSE(backwards.ok());
  EXPECT_EQ(backwards.reason_code, "SIMULATION_TIME_ROLLBACK");
}

TEST(SessionProtocol, ReconnectInvalidatesOldSession) {
  auto protocol = Established();
  protocol.Reset();
  EXPECT_EQ(protocol.state(), SessionState::kDisconnected);
  EXPECT_FALSE(protocol.session().has_value());
  EXPECT_EQ(protocol.BuildHello().header.sequence, 1U);

  const auto stale = protocol.AcceptIncoming(
      Heartbeat(2U, 10), SessionProtocol::SteadyTime{});
  EXPECT_FALSE(stale.ok());
  EXPECT_EQ(stale.reason_code, "SESSION_NOT_ESTABLISHED");
}

TEST(SessionProtocol, HeartbeatExpiryRequiresHold) {
  const SessionProtocol::SteadyTime start{};
  auto protocol = Established(start);
  EXPECT_TRUE(protocol.AcceptIncoming(Heartbeat(2U, 10), start).ok());

  EXPECT_FALSE(protocol.CheckHeartbeat(start + 3s).has_value());
  const auto expired = protocol.CheckHeartbeat(start + 3001ms);
  ASSERT_TRUE(expired.has_value());
  EXPECT_EQ(*expired, "HEARTBEAT_TIMEOUT");
  EXPECT_EQ(protocol.state(), SessionState::kHold);
}

TEST(SessionProtocol, BecomesReadyOnlyAfterFreshStateAndMap) {
  auto protocol = Established();

  EXPECT_TRUE(protocol.AcceptIncoming(
      RobotState(2U), SessionProtocol::SteadyTime{}).ok());
  EXPECT_EQ(protocol.state(), SessionState::kSyncing);

  EXPECT_TRUE(protocol.AcceptIncoming(
      LocalMap(3U), SessionProtocol::SteadyTime{}).ok());
  EXPECT_EQ(protocol.state(), SessionState::kReady);
}

}  // namespace
}  // namespace lunar::unreal_tcp
