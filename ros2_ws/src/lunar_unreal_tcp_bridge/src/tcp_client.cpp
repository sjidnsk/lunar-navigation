#include "lunar_unreal_tcp_bridge/tcp_client.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>

#include "lunar_unreal_tcp_bridge/stream_decoder.hpp"

namespace lunar::unreal_tcp {
namespace {

constexpr std::chrono::milliseconds kIoPollInterval{100};

[[nodiscard]] QueuePushResult NotConnected() {
  return QueuePushResult{
      .accepted = false,
      .replaced_latest = false,
      .fatal = false,
      .reason_code = "TRANSPORT_NOT_CONNECTED",
  };
}

}  // namespace

class TcpClient::Impl final {
 public:
  explicit Impl(TcpClientConfig config)
      : config_(std::move(config)),
        outbound_(config_.non_droppable_capacity) {
    if (config_.server_host.empty() || config_.server_port == 0U ||
        config_.reconnect_initial <= std::chrono::milliseconds::zero() ||
        config_.reconnect_max < config_.reconnect_initial ||
        config_.connect_timeout <= std::chrono::milliseconds::zero()) {
      throw std::invalid_argument("TCP_CLIENT_CONFIG_INVALID");
    }
  }

  ~Impl() { Stop(); }

  void SetCallbacks(TransportCallbacks callbacks) {
    std::scoped_lock lock{callbacks_mutex_};
    callbacks_ = std::move(callbacks);
  }

  void Start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
      return;
    }
    reader_ = std::jthread(
        [this](const std::stop_token stop_token) { ReaderLoop(stop_token); });
    writer_ = std::jthread(
        [this](const std::stop_token stop_token) { WriterLoop(stop_token); });
  }

  void Stop() noexcept {
    if (!running_.exchange(false)) {
      return;
    }
    reader_.request_stop();
    writer_.request_stop();
    CloseConnection("TRANSPORT_STOPPED", false);
    wake_.notify_all();
    if (reader_.joinable()) {
      reader_.join();
    }
    if (writer_.joinable()) {
      writer_.join();
    }
    outbound_.Clear();
    std::scoped_lock lock{emergency_mutex_};
    emergency_hold_.reset();
    emergency_reason_.clear();
  }

  [[nodiscard]] QueuePushResult Send(
      Frame frame, std::optional<Frame> overflow_hold) {
    if (!running_.load() || !connected_.load()) {
      return NotConnected();
    }
    QueuePushResult result = outbound_.Push(std::move(frame));
    if (result.fatal) {
      {
        std::scoped_lock lock{emergency_mutex_};
        emergency_hold_ = std::move(overflow_hold);
        emergency_reason_ = result.reason_code;
      }
      wake_.notify_all();
      return result;
    }
    if (result.accepted) {
      wake_.notify_all();
    }
    return result;
  }

  [[nodiscard]] bool connected() const noexcept {
    return connected_.load();
  }

  [[nodiscard]] bool socket_options_enabled() const noexcept {
    return socket_options_enabled_.load();
  }

 private:
  [[nodiscard]] TransportCallbacks Callbacks() const {
    std::scoped_lock lock{callbacks_mutex_};
    return callbacks_;
  }

  void NotifyConnected() {
    const auto callbacks = Callbacks();
    if (callbacks.on_connected) {
      callbacks.on_connected();
    }
  }

  void NotifyFrame(Frame frame) {
    const auto callbacks = Callbacks();
    if (callbacks.on_frame) {
      callbacks.on_frame(std::move(frame));
    }
  }

  void NotifyDisconnected(std::string reason) {
    const auto callbacks = Callbacks();
    if (callbacks.on_disconnected) {
      callbacks.on_disconnected(std::move(reason));
    }
  }

  void CloseConnection(std::string reason, const bool notify = true) noexcept {
    const int descriptor = descriptor_.exchange(-1);
    if (descriptor >= 0) {
      (void)::shutdown(descriptor, SHUT_RDWR);
      (void)::close(descriptor);
    }
    socket_options_enabled_ = false;
    const bool was_connected = connected_.exchange(false);
    outbound_.Clear();
    wake_.notify_all();
    if (notify && was_connected) {
      try {
        NotifyDisconnected(std::move(reason));
      } catch (...) {
      }
    }
  }

  [[nodiscard]] bool WaitFor(
      const std::stop_token stop_token,
      const std::chrono::milliseconds duration) {
    std::unique_lock lock{wake_mutex_};
    wake_.wait_for(lock, duration, [this, stop_token] {
      return stop_token.stop_requested() || !running_.load();
    });
    return !stop_token.stop_requested() && running_.load();
  }

  [[nodiscard]] int ConnectSocket(const std::stop_token stop_token) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    addrinfo* addresses{};
    const std::string service = std::to_string(config_.server_port);
    if (::getaddrinfo(
            config_.server_host.c_str(), service.c_str(), &hints,
            &addresses) != 0) {
      return -1;
    }
    const std::unique_ptr<addrinfo, decltype(&::freeaddrinfo)> address_guard{
        addresses, &::freeaddrinfo};

    for (const addrinfo* address = addresses; address != nullptr;
         address = address->ai_next) {
      if (stop_token.stop_requested() || !running_.load()) {
        return -1;
      }
      const int descriptor = ::socket(
          address->ai_family, address->ai_socktype, address->ai_protocol);
      if (descriptor < 0) {
        continue;
      }
      const int original_flags = ::fcntl(descriptor, F_GETFL, 0);
      if (original_flags < 0 ||
          ::fcntl(descriptor, F_SETFL, original_flags | O_NONBLOCK) != 0) {
        ::close(descriptor);
        continue;
      }
      int enabled = 1;
      if (::setsockopt(
              descriptor, IPPROTO_TCP, TCP_NODELAY, &enabled,
              sizeof(enabled)) != 0 ||
          ::setsockopt(
              descriptor, SOL_SOCKET, SO_KEEPALIVE, &enabled,
              sizeof(enabled)) != 0) {
        ::close(descriptor);
        continue;
      }

      bool connected = ::connect(
          descriptor, address->ai_addr, address->ai_addrlen) == 0;
      if (!connected && errno == EINPROGRESS) {
        const auto deadline =
            std::chrono::steady_clock::now() + config_.connect_timeout;
        while (!connected && std::chrono::steady_clock::now() < deadline &&
               !stop_token.stop_requested() && running_.load()) {
          const auto remaining = std::chrono::duration_cast<
              std::chrono::milliseconds>(
              deadline - std::chrono::steady_clock::now());
          pollfd poll_descriptor{
              .fd = descriptor, .events = POLLOUT, .revents = 0};
          const int poll_result = ::poll(
              &poll_descriptor, 1U,
              static_cast<int>(std::max(
                  std::chrono::milliseconds{1},
                  std::min(kIoPollInterval, remaining)).count()));
          if (poll_result < 0 && errno == EINTR) {
            continue;
          }
          if (poll_result <= 0) {
            continue;
          }
          int socket_error{};
          socklen_t error_length = sizeof(socket_error);
          connected = ::getsockopt(
              descriptor, SOL_SOCKET, SO_ERROR, &socket_error,
              &error_length) == 0 && socket_error == 0;
          if (!connected && socket_error != EINPROGRESS) {
            break;
          }
        }
      }
      if (!connected || stop_token.stop_requested() || !running_.load()) {
        ::close(descriptor);
        continue;
      }

      int no_delay{};
      int keepalive{};
      socklen_t option_length = sizeof(int);
      const bool options_ok = ::getsockopt(
          descriptor, IPPROTO_TCP, TCP_NODELAY, &no_delay,
          &option_length) == 0 && no_delay == 1;
      option_length = sizeof(int);
      const bool keepalive_ok = ::getsockopt(
          descriptor, SOL_SOCKET, SO_KEEPALIVE, &keepalive,
          &option_length) == 0 && keepalive == 1;
      if (!options_ok || !keepalive_ok) {
        ::close(descriptor);
        continue;
      }
      socket_options_enabled_ = true;
      return descriptor;
    }
    return -1;
  }

  [[nodiscard]] bool SendBytes(
      const int descriptor, const std::span<const std::byte> bytes,
      const std::stop_token stop_token) {
    std::size_t sent{};
    while (sent < bytes.size() && !stop_token.stop_requested() &&
           running_.load() && descriptor_.load() == descriptor) {
      const ssize_t result = ::send(
          descriptor, bytes.data() + sent, bytes.size() - sent,
          MSG_NOSIGNAL);
      if (result > 0) {
        sent += static_cast<std::size_t>(result);
        continue;
      }
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        pollfd poll_descriptor{
            .fd = descriptor, .events = POLLOUT, .revents = 0};
        if (::poll(
                &poll_descriptor, 1U,
                static_cast<int>(kIoPollInterval.count())) >= 0) {
          continue;
        }
      }
      return false;
    }
    return sent == bytes.size();
  }

  void ReaderLoop(const std::stop_token stop_token) {
    std::size_t failed_attempt{};
    while (!stop_token.stop_requested() && running_.load()) {
      if (!connected_.load()) {
        const int descriptor = ConnectSocket(stop_token);
        if (descriptor < 0) {
          const auto backoff = TcpClient::ReconnectBackoff(
              failed_attempt++, config_.reconnect_initial,
              config_.reconnect_max);
          if (!WaitFor(stop_token, backoff)) {
            break;
          }
          continue;
        }
        descriptor_ = descriptor;
        connected_ = true;
        failed_attempt = 0U;
        try {
          NotifyConnected();
        } catch (...) {
          CloseConnection("CONNECTED_CALLBACK_FAILED");
          continue;
        }
        wake_.notify_all();
      }

      StreamDecoder decoder;
      while (!stop_token.stop_requested() && running_.load() &&
             connected_.load()) {
        const int descriptor = descriptor_.load();
        if (descriptor < 0) {
          break;
        }
        pollfd poll_descriptor{
            .fd = descriptor,
            .events = static_cast<short>(POLLIN | POLLERR | POLLHUP),
            .revents = 0};
        const int poll_result = ::poll(
            &poll_descriptor, 1U,
            static_cast<int>(kIoPollInterval.count()));
        if (poll_result < 0 && errno == EINTR) {
          continue;
        }
        if (poll_result < 0) {
          CloseConnection("TCP_POLL_FAILED");
          break;
        }
        if (poll_result == 0) {
          continue;
        }
        if ((poll_descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
          CloseConnection("TCP_DISCONNECTED");
          break;
        }
        std::array<std::byte, 64U * 1024U> bytes{};
        const ssize_t count = ::recv(
            descriptor, bytes.data(), bytes.size(), 0);
        if (count == 0) {
          CloseConnection("TCP_DISCONNECTED");
          break;
        }
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK ||
                         errno == EINTR)) {
          continue;
        }
        if (count < 0) {
          CloseConnection("TCP_RECEIVE_FAILED");
          break;
        }
        StreamDecodeResult decoded = decoder.Push(std::span{
            bytes.data(), static_cast<std::size_t>(count)});
        if (decoded.error.has_value()) {
          CloseConnection(decoded.error->reason_code);
          break;
        }
        for (Frame& frame : decoded.frames) {
          try {
            NotifyFrame(std::move(frame));
          } catch (...) {
            CloseConnection("FRAME_CALLBACK_FAILED");
            break;
          }
        }
      }
      if (!stop_token.stop_requested() && running_.load() &&
          !connected_.load()) {
        const auto backoff = TcpClient::ReconnectBackoff(
            failed_attempt++, config_.reconnect_initial,
            config_.reconnect_max);
        if (!WaitFor(stop_token, backoff)) {
          break;
        }
      }
    }
  }

  void WriterLoop(const std::stop_token stop_token) {
    while (!stop_token.stop_requested() && running_.load()) {
      std::optional<Frame> emergency;
      std::string emergency_reason;
      {
        std::scoped_lock lock{emergency_mutex_};
        if (emergency_hold_.has_value()) {
          emergency = std::move(emergency_hold_);
          emergency_hold_.reset();
          emergency_reason = std::move(emergency_reason_);
          emergency_reason_.clear();
        }
      }
      if (emergency.has_value()) {
        const int descriptor = descriptor_.load();
        const EncodeResult encoded = EncodeFrame(*emergency);
        if (descriptor >= 0 && encoded.ok()) {
          (void)SendBytes(descriptor, *encoded.bytes, stop_token);
        }
        CloseConnection(
            emergency_reason.empty()
                ? "OUTBOUND_RELIABLE_QUEUE_EXHAUSTED"
                : std::move(emergency_reason));
        continue;
      }

      if (!connected_.load()) {
        std::unique_lock lock{wake_mutex_};
        wake_.wait_for(lock, kIoPollInterval);
        continue;
      }
      std::optional<Frame> frame = outbound_.Pop();
      if (!frame.has_value()) {
        std::unique_lock lock{wake_mutex_};
        wake_.wait_for(lock, kIoPollInterval);
        continue;
      }
      const EncodeResult encoded = EncodeFrame(*frame);
      if (!encoded.ok()) {
        CloseConnection(
            encoded.error.has_value()
                ? encoded.error->reason_code
                : "OUTBOUND_ENCODE_FAILED");
        continue;
      }
      const int descriptor = descriptor_.load();
      if (descriptor < 0 ||
          !SendBytes(descriptor, *encoded.bytes, stop_token)) {
        CloseConnection("TCP_SEND_FAILED");
      }
    }
  }

  TcpClientConfig config_;
  OutboundQueue outbound_;
  mutable std::mutex callbacks_mutex_;
  TransportCallbacks callbacks_;
  std::mutex wake_mutex_;
  std::condition_variable wake_;
  std::mutex emergency_mutex_;
  std::optional<Frame> emergency_hold_;
  std::string emergency_reason_;
  std::atomic<bool> running_{false};
  std::atomic<bool> connected_{false};
  std::atomic<bool> socket_options_enabled_{false};
  std::atomic<int> descriptor_{-1};
  std::jthread reader_;
  std::jthread writer_;
};

TcpClient::TcpClient(TcpClientConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

TcpClient::~TcpClient() = default;

void TcpClient::SetCallbacks(TransportCallbacks callbacks) {
  impl_->SetCallbacks(std::move(callbacks));
}

void TcpClient::Start() {
  impl_->Start();
}

void TcpClient::Stop() noexcept {
  impl_->Stop();
}

QueuePushResult TcpClient::Send(
    Frame frame, std::optional<Frame> overflow_hold) {
  return impl_->Send(std::move(frame), std::move(overflow_hold));
}

bool TcpClient::connected() const noexcept {
  return impl_->connected();
}

bool TcpClient::socket_options_enabled() const noexcept {
  return impl_->socket_options_enabled();
}

std::chrono::milliseconds TcpClient::ReconnectBackoff(
    const std::size_t failed_attempt,
    const std::chrono::milliseconds initial,
    const std::chrono::milliseconds maximum) {
  if (initial <= std::chrono::milliseconds::zero() || maximum < initial) {
    return maximum;
  }
  std::int64_t value = initial.count();
  const std::int64_t cap = maximum.count();
  for (std::size_t attempt = 0U;
       attempt < failed_attempt && value < cap; ++attempt) {
    value = value > cap / 2 ? cap : value * 2;
  }
  return std::chrono::milliseconds{std::min(value, cap)};
}

}  // namespace lunar::unreal_tcp
