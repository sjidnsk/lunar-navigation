#include "lunar_unreal_tcp_bridge/tcp_client.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_unreal_tcp_bridge/protocol.hpp"
#include "lunar_unreal_tcp_bridge/stream_decoder.hpp"

namespace lunar::unreal_tcp {
namespace {

using namespace std::chrono_literals;
constexpr char kSessionId[] = "123e4567-e89b-12d3-a456-426614174000";

class LoopbackListener final {
 public:
  LoopbackListener() {
    descriptor_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (descriptor_ < 0) {
      return;
    }
    int enabled = 1;
    (void)::setsockopt(
        descriptor_, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0U;
    if (::bind(
            descriptor_, reinterpret_cast<const sockaddr*>(&address),
            sizeof(address)) != 0 ||
        ::listen(descriptor_, 4) != 0) {
      ::close(descriptor_);
      descriptor_ = -1;
      return;
    }
    socklen_t length = sizeof(address);
    if (::getsockname(
            descriptor_, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
      ::close(descriptor_);
      descriptor_ = -1;
      return;
    }
    port_ = ntohs(address.sin_port);
  }

  ~LoopbackListener() {
    if (descriptor_ >= 0) {
      ::close(descriptor_);
    }
  }

  [[nodiscard]] bool ok() const noexcept { return descriptor_ >= 0; }
  [[nodiscard]] std::uint16_t port() const noexcept { return port_; }

  [[nodiscard]] int Accept(const std::chrono::milliseconds timeout = 2s) const {
    pollfd descriptor{.fd = descriptor_, .events = POLLIN, .revents = 0};
    if (::poll(&descriptor, 1U, static_cast<int>(timeout.count())) <= 0) {
      return -1;
    }
    return ::accept(descriptor_, nullptr, nullptr);
  }

 private:
  int descriptor_{-1};
  std::uint16_t port_{};
};

bool WaitUntil(
    const std::function<bool()>& predicate,
    const std::chrono::milliseconds timeout = 2s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(5ms);
  }
  return predicate();
}

bool SendAll(const int descriptor, const std::span<const std::byte> bytes) {
  std::size_t sent{};
  while (sent < bytes.size()) {
    const ssize_t result = ::send(
        descriptor, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
    if (result <= 0) {
      return false;
    }
    sent += static_cast<std::size_t>(result);
  }
  return true;
}

Frame Heartbeat(const std::uint64_t sequence) {
  Frame frame;
  frame.header.message_type = MessageType::kHeartbeat;
  frame.header.sequence = sequence;
  frame.header.simulation_time_ns = static_cast<std::int64_t>(sequence);
  frame.metadata = {
      {"session_id", kSessionId},
      {"last_received_sequence", sequence - 1U},
  };
  return frame;
}

Frame Reference(const std::uint64_t sequence) {
  Frame frame;
  frame.header.message_type = MessageType::kMotionReference;
  frame.header.sequence = sequence;
  return frame;
}

Frame Hold(const std::uint64_t sequence) {
  Frame frame;
  frame.header.message_type = MessageType::kControl;
  frame.header.sequence = sequence;
  frame.metadata = {
      {"session_id", kSessionId},
      {"command", "HOLD"},
      {"command_id", "overflow-hold"},
  };
  return frame;
}

TEST(TcpClient, ConnectsWithTcpNoDelayAndKeepalive) {
  LoopbackListener listener;
  ASSERT_TRUE(listener.ok());
  std::atomic<bool> accepted{false};
  std::jthread server([&](std::stop_token) {
    const int peer = listener.Accept();
    accepted = peer >= 0;
    if (peer >= 0) {
      std::this_thread::sleep_for(100ms);
      ::close(peer);
    }
  });

  TcpClient client(TcpClientConfig{.server_port = listener.port()});
  std::atomic<bool> connected{false};
  client.SetCallbacks(TransportCallbacks{
      .on_connected = [&] { connected = true; },
      .on_frame = {},
      .on_disconnected = {},
  });
  client.Start();

  ASSERT_TRUE(WaitUntil([&] { return connected.load() && accepted.load(); }));
  EXPECT_TRUE(client.connected());
  EXPECT_TRUE(client.socket_options_enabled());
  client.Stop();
}

TEST(TcpClient, HandlesFragmentedAndCoalescedFrames) {
  LoopbackListener listener;
  ASSERT_TRUE(listener.ok());
  std::jthread server([&](std::stop_token) {
    const int peer = listener.Accept();
    if (peer < 0) {
      return;
    }
    const auto first = EncodeFrame(Heartbeat(1U));
    const auto second = EncodeFrame(Heartbeat(2U));
    if (!first.ok() || !second.ok()) {
      ::close(peer);
      return;
    }
    (void)SendAll(peer, std::span{first.bytes->data(), 7U});
    std::vector<std::byte> remainder(
        first.bytes->begin() + 7, first.bytes->end());
    remainder.insert(remainder.end(), second.bytes->begin(), second.bytes->end());
    (void)SendAll(peer, remainder);
    std::this_thread::sleep_for(50ms);
    ::close(peer);
  });

  std::mutex mutex;
  std::vector<std::uint64_t> sequences;
  TcpClient client(TcpClientConfig{
      .server_port = listener.port(), .reconnect_initial = 5s,
      .reconnect_max = 5s});
  client.SetCallbacks(TransportCallbacks{
      .on_connected = {},
      .on_frame = [&](Frame frame) {
        std::scoped_lock lock{mutex};
        sequences.push_back(frame.header.sequence);
      },
      .on_disconnected = {},
  });
  client.Start();
  ASSERT_TRUE(WaitUntil([&] {
    std::scoped_lock lock{mutex};
    return sequences.size() == 2U;
  }));
  client.Stop();
  std::scoped_lock lock{mutex};
  EXPECT_EQ(sequences, (std::vector<std::uint64_t>{1U, 2U}));
}

TEST(TcpClient, ReconnectBackoffIsBoundedFromHalfToFiveSeconds) {
  EXPECT_EQ(TcpClient::ReconnectBackoff(0U), 500ms);
  EXPECT_EQ(TcpClient::ReconnectBackoff(1U), 1000ms);
  EXPECT_EQ(TcpClient::ReconnectBackoff(2U), 2000ms);
  EXPECT_EQ(TcpClient::ReconnectBackoff(3U), 4000ms);
  EXPECT_EQ(TcpClient::ReconnectBackoff(4U), 5000ms);
  EXPECT_EQ(TcpClient::ReconnectBackoff(50U), 5000ms);
}

TEST(TcpClient, DisconnectStopsWriterAndInvalidatesSession) {
  LoopbackListener listener;
  ASSERT_TRUE(listener.ok());
  std::jthread server([&](std::stop_token) {
    const int peer = listener.Accept();
    if (peer >= 0) {
      ::shutdown(peer, SHUT_RDWR);
      ::close(peer);
    }
  });
  std::atomic<std::size_t> disconnected{};
  TcpClient client(TcpClientConfig{
      .server_port = listener.port(), .reconnect_initial = 5s,
      .reconnect_max = 5s});
  client.SetCallbacks(TransportCallbacks{
      .on_connected = {},
      .on_frame = {},
      .on_disconnected = [&](std::string) { ++disconnected; },
  });
  client.Start();
  ASSERT_TRUE(WaitUntil([&] { return disconnected.load() == 1U; }));
  EXPECT_FALSE(client.connected());
  const auto send = client.Send(Heartbeat(3U));
  EXPECT_FALSE(send.accepted);
  EXPECT_EQ(send.reason_code, "TRANSPORT_NOT_CONNECTED");
  client.Stop();
}

TEST(TcpClient, HighPriorityOverflowSendsHoldThenDisconnects) {
  LoopbackListener listener;
  ASSERT_TRUE(listener.ok());
  std::mutex mutex;
  std::optional<Frame> received;
  std::jthread server([&](std::stop_token) {
    const int peer = listener.Accept();
    if (peer < 0) {
      return;
    }
    StreamDecoder decoder;
    std::array<std::byte, 2048> bytes{};
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (std::chrono::steady_clock::now() < deadline) {
      pollfd poll_descriptor{.fd = peer, .events = POLLIN, .revents = 0};
      if (::poll(&poll_descriptor, 1U, 100) <= 0) {
        continue;
      }
      const ssize_t count = ::recv(peer, bytes.data(), bytes.size(), 0);
      if (count <= 0) {
        break;
      }
      auto decoded = decoder.Push(std::span{
          bytes.data(), static_cast<std::size_t>(count)});
      if (!decoded.frames.empty()) {
        std::scoped_lock lock{mutex};
        received = std::move(decoded.frames.front());
        break;
      }
    }
    ::close(peer);
  });

  std::atomic<bool> connected{false};
  std::atomic<bool> disconnected{false};
  TcpClient client(TcpClientConfig{
      .server_port = listener.port(),
      .non_droppable_capacity = 0U,
      .reconnect_initial = 5s,
      .reconnect_max = 5s});
  client.SetCallbacks(TransportCallbacks{
      .on_connected = [&] { connected = true; },
      .on_frame = {},
      .on_disconnected = [&](std::string reason) {
        disconnected = reason == "OUTBOUND_RELIABLE_QUEUE_EXHAUSTED";
      },
  });
  client.Start();
  ASSERT_TRUE(WaitUntil([&] { return connected.load(); }));
  const auto result = client.Send(Reference(2U), Hold(3U));
  EXPECT_FALSE(result.accepted);
  EXPECT_TRUE(result.fatal);
  ASSERT_TRUE(WaitUntil([&] {
    std::scoped_lock lock{mutex};
    return received.has_value() && disconnected.load();
  }));
  client.Stop();
  std::scoped_lock lock{mutex};
  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->header.message_type, MessageType::kControl);
  EXPECT_EQ(received->metadata.at("command"), "HOLD");
}

}  // namespace
}  // namespace lunar::unreal_tcp
