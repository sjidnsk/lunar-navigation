#include "lunar_unreal_tcp_bridge/session_protocol.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "lunar_unreal_tcp_bridge/metadata_validation.hpp"

namespace lunar::unreal_tcp {
namespace {

[[nodiscard]] SessionAcceptResult Accept() {
  return SessionAcceptResult{
      .accepted = true,
      .force_hold = false,
      .reason_code = "ACCEPTED",
  };
}

[[nodiscard]] SessionAcceptResult Reject(
    SessionState& state, std::string reason_code,
    const bool force_hold = true) {
  if (force_hold) {
    state = SessionState::kHold;
  }
  return SessionAcceptResult{
      .accepted = false,
      .force_hold = force_hold,
      .reason_code = std::move(reason_code),
  };
}

template<std::size_t Size>
[[nodiscard]] std::array<double, Size> DoubleArray(
    const nlohmann::json& value) {
  std::array<double, Size> result{};
  for (std::size_t index = 0U; index < Size; ++index) {
    result[index] = value.at(index).get<double>();
  }
  return result;
}

}  // namespace

SessionProtocol::SessionProtocol(std::string client_nonce)
    : client_nonce_(std::move(client_nonce)) {}

Frame SessionProtocol::BuildHello() {
  session_.reset();
  last_received_sequence_ = 0U;
  next_outgoing_sequence_ = 1U;
  last_simulation_time_ns_.reset();
  last_heartbeat_.reset();
  has_state_ = false;
  has_map_ = false;
  state_ = SessionState::kHandshaking;

  Frame frame;
  frame.header.message_type = MessageType::kHello;
  frame.header.sequence = next_outgoing_sequence();
  frame.metadata = {
      {"protocol", "lunar-unreal-tcp/v1"},
      {"role", "ROS_CLIENT"},
      {"client_nonce", client_nonce_},
      {"platform_type", "WHEELED"},
      {"robot_count", 1},
      {"state_rate_hz", 20},
      {"map_rate_hz", 5},
      {"heartbeat_rate_hz", 1},
      {"maximum_body_bytes", kMaximumBodyBytes},
  };
  return frame;
}

SessionAcceptResult SessionProtocol::AcceptHelloAck(
    const Frame& frame, const SteadyTime received_at) {
  if (state_ != SessionState::kHandshaking) {
    return Reject(state_, "HELLO_ACK_UNEXPECTED", false);
  }
  if (frame.header.message_type != MessageType::kHelloAck) {
    return Reject(state_, "HELLO_ACK_TYPE_MISMATCH", false);
  }
  if (frame.header.sequence != 1U) {
    return Reject(state_, "HELLO_ACK_SEQUENCE_INVALID", false);
  }
  if (frame.header.simulation_time_ns < 0) {
    return Reject(state_, "SIMULATION_TIME_INVALID", false);
  }
  if (const auto metadata_error =
          ValidateMetadata(MessageType::kHelloAck, frame.metadata)) {
    return Reject(state_, metadata_error->reason_code, false);
  }
  session_ = FrozenSession{
      .session_id = frame.metadata.at("session_id").get<std::string>(),
      .scene_id = frame.metadata.at("scene_id").get<std::string>(),
      .robot_id = frame.metadata.at("robot_id").get<std::string>(),
      .engine_version = frame.metadata.at("engine_version").get<std::string>(),
      .agx_plugin_version =
          frame.metadata.at("agx_plugin_version").get<std::string>(),
      .coordinate_convention =
          frame.metadata.at("coordinate_convention").get<std::string>(),
      .handedness = frame.metadata.at("handedness").get<std::string>(),
      .up_axis = frame.metadata.at("up_axis").get<std::string>(),
      .length_unit_to_m = frame.metadata.at("length_unit_to_m").get<double>(),
      .base_link_from_unreal_root = DoubleArray<16>(
          frame.metadata.at("T_base_link_from_unreal_root")),
      .base_footprint_from_base_link = DoubleArray<16>(
          frame.metadata.at("T_base_footprint_from_base_link")),
      .calibration_hash =
          frame.metadata.at("calibration_hash").get<std::string>(),
  };
  last_received_sequence_ = 1U;
  last_simulation_time_ns_ = frame.header.simulation_time_ns;
  last_heartbeat_ = received_at;
  state_ = SessionState::kSyncing;
  return Accept();
}

SessionAcceptResult SessionProtocol::AcceptIncoming(
    const Frame& frame, const SteadyTime received_at) {
  if (!session_.has_value() || state_ == SessionState::kDisconnected ||
      state_ == SessionState::kHandshaking) {
    return Reject(state_, "SESSION_NOT_ESTABLISHED", false);
  }
  if (frame.header.message_type == MessageType::kHello ||
      frame.header.message_type == MessageType::kHelloAck) {
    return Reject(state_, "MESSAGE_TYPE_UNEXPECTED");
  }
  if (const auto metadata_error =
          ValidateMetadata(frame.header.message_type, frame.metadata)) {
    return Reject(state_, metadata_error->reason_code);
  }
  const auto session_id = frame.metadata.find("session_id");
  if (session_id == frame.metadata.end() || !session_id->is_string() ||
      session_id->get<std::string>() != session_->session_id) {
    return Reject(state_, "SESSION_ID_MISMATCH");
  }
  if (frame.header.sequence <= last_received_sequence_) {
    return Reject(state_, "SEQUENCE_NOT_INCREASING");
  }
  if (frame.header.simulation_time_ns < 0 ||
      (last_simulation_time_ns_.has_value() &&
       frame.header.simulation_time_ns < *last_simulation_time_ns_)) {
    return Reject(state_, "SIMULATION_TIME_ROLLBACK");
  }

  last_received_sequence_ = frame.header.sequence;
  last_simulation_time_ns_ = frame.header.simulation_time_ns;
  if (frame.header.message_type == MessageType::kHeartbeat) {
    last_heartbeat_ = received_at;
  } else if (frame.header.message_type == MessageType::kRobotState) {
    has_state_ = true;
  } else if (frame.header.message_type == MessageType::kLocalElevationMap) {
    has_map_ = true;
  } else if (frame.header.message_type == MessageType::kExecutionFeedback) {
    const std::string feedback_state =
        frame.metadata.at("state").get<std::string>();
    if (feedback_state == "ACCEPTED" || feedback_state == "EXECUTING") {
      state_ = SessionState::kExecuting;
    } else if (feedback_state == "FAILED" || feedback_state == "CANCELED") {
      state_ = SessionState::kHold;
    } else if (feedback_state == "SEGMENT_COMPLETE") {
      state_ = SessionState::kReady;
    }
  } else if (frame.header.message_type == MessageType::kError) {
    state_ = SessionState::kHold;
  }
  if (state_ == SessionState::kSyncing && has_state_ && has_map_) {
    state_ = SessionState::kReady;
  }
  return Accept();
}

std::optional<std::string> SessionProtocol::CheckHeartbeat(
    const SteadyTime now, const std::chrono::seconds timeout) {
  if (!session_.has_value() || !last_heartbeat_.has_value() ||
      timeout <= std::chrono::seconds::zero()) {
    return std::nullopt;
  }
  if (now - *last_heartbeat_ > timeout) {
    state_ = SessionState::kHold;
    return "HEARTBEAT_TIMEOUT";
  }
  return std::nullopt;
}

void SessionProtocol::ForceHold() noexcept {
  if (session_.has_value()) {
    state_ = SessionState::kHold;
  }
}

void SessionProtocol::Reset() noexcept {
  state_ = SessionState::kDisconnected;
  session_.reset();
  last_received_sequence_ = 0U;
  next_outgoing_sequence_ = 1U;
  last_simulation_time_ns_.reset();
  last_heartbeat_.reset();
  has_state_ = false;
  has_map_ = false;
}

SessionState SessionProtocol::state() const noexcept {
  return state_;
}

const std::optional<FrozenSession>& SessionProtocol::session() const noexcept {
  return session_;
}

std::uint64_t SessionProtocol::last_received_sequence() const noexcept {
  return last_received_sequence_;
}

std::uint64_t SessionProtocol::next_outgoing_sequence() noexcept {
  if (next_outgoing_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
    state_ = SessionState::kHold;
    return 0U;
  }
  return next_outgoing_sequence_++;
}

}  // namespace lunar::unreal_tcp
