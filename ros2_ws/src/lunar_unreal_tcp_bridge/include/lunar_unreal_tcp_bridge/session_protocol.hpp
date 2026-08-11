#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include "lunar_unreal_tcp_bridge/protocol.hpp"

namespace lunar::unreal_tcp {

enum class SessionState : std::uint8_t {
  kDisconnected,
  kHandshaking,
  kSyncing,
  kReady,
  kExecuting,
  kHold,
};

struct FrozenSession final {
  std::string session_id;
  std::string scene_id;
  std::string robot_id;
  std::string engine_version;
  std::string agx_plugin_version;
  std::string coordinate_convention;
  std::string handedness;
  std::string up_axis;
  double length_unit_to_m{};
  std::array<double, 16> base_link_from_unreal_root{};
  std::array<double, 16> base_footprint_from_base_link{};
  std::string calibration_hash;
};

struct SessionAcceptResult final {
  bool accepted{};
  bool force_hold{};
  std::string reason_code;
  [[nodiscard]] bool ok() const noexcept { return accepted; }
};

class SessionProtocol final {
 public:
  using SteadyTime = std::chrono::steady_clock::time_point;

  explicit SessionProtocol(std::string client_nonce = "ros-client");

  [[nodiscard]] Frame BuildHello();
  [[nodiscard]] SessionAcceptResult AcceptHelloAck(
      const Frame& frame, SteadyTime received_at);
  [[nodiscard]] SessionAcceptResult AcceptIncoming(
      const Frame& frame, SteadyTime received_at);
  [[nodiscard]] std::optional<std::string> CheckHeartbeat(
      SteadyTime now, std::chrono::seconds timeout = std::chrono::seconds{3});
  void Reset() noexcept;

  [[nodiscard]] SessionState state() const noexcept;
  [[nodiscard]] const std::optional<FrozenSession>& session() const noexcept;
  [[nodiscard]] std::uint64_t last_received_sequence() const noexcept;
  [[nodiscard]] std::uint64_t next_outgoing_sequence() noexcept;

 private:
  std::string client_nonce_;
  SessionState state_{SessionState::kDisconnected};
  std::optional<FrozenSession> session_;
  std::uint64_t last_received_sequence_{};
  std::uint64_t next_outgoing_sequence_{1U};
  std::optional<std::int64_t> last_simulation_time_ns_;
  std::optional<SteadyTime> last_heartbeat_;
  bool has_state_{};
  bool has_map_{};
};

}  // namespace lunar::unreal_tcp
