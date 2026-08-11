#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "lunar_unreal_tcp_bridge/outbound_queue.hpp"
#include "lunar_unreal_tcp_bridge/protocol.hpp"

namespace lunar::unreal_tcp {

struct TransportCallbacks final {
  std::function<void()> on_connected;
  std::function<void(Frame)> on_frame;
  std::function<void(std::string)> on_disconnected;
};

class SessionTransport {
 public:
  virtual ~SessionTransport() = default;
  virtual void SetCallbacks(TransportCallbacks callbacks) = 0;
  virtual void Start() = 0;
  virtual void Stop() noexcept = 0;
  [[nodiscard]] virtual QueuePushResult Send(
      Frame frame, std::optional<Frame> overflow_hold = std::nullopt) = 0;
  [[nodiscard]] virtual bool connected() const noexcept = 0;
};

struct TcpClientConfig final {
  std::string server_host{"127.0.0.1"};
  std::uint16_t server_port{47001U};
  std::size_t non_droppable_capacity{64U};
  std::chrono::milliseconds reconnect_initial{500};
  std::chrono::milliseconds reconnect_max{5000};
  std::chrono::milliseconds connect_timeout{2000};
};

class TcpClient final : public SessionTransport {
 public:
  explicit TcpClient(TcpClientConfig config);
  ~TcpClient() override;

  TcpClient(const TcpClient&) = delete;
  TcpClient& operator=(const TcpClient&) = delete;

  void SetCallbacks(TransportCallbacks callbacks) override;
  void Start() override;
  void Stop() noexcept override;
  [[nodiscard]] QueuePushResult Send(
      Frame frame, std::optional<Frame> overflow_hold = std::nullopt) override;
  [[nodiscard]] bool connected() const noexcept override;

  [[nodiscard]] bool socket_options_enabled() const noexcept;
  [[nodiscard]] static std::chrono::milliseconds ReconnectBackoff(
      std::size_t failed_attempt,
      std::chrono::milliseconds initial = std::chrono::milliseconds{500},
      std::chrono::milliseconds maximum = std::chrono::milliseconds{5000});

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::unreal_tcp
