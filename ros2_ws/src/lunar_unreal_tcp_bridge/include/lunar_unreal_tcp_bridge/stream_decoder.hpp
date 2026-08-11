#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "lunar_unreal_tcp_bridge/protocol.hpp"

namespace lunar::unreal_tcp {

struct StreamDecodeResult final {
  std::vector<Frame> frames;
  std::optional<ProtocolError> error;
  [[nodiscard]] bool ok() const noexcept { return !error.has_value(); }
};

class StreamDecoder final {
 public:
  [[nodiscard]] StreamDecodeResult Push(std::span<const std::byte> bytes);
  void Reset() noexcept;
  [[nodiscard]] std::size_t buffered_bytes() const noexcept;

 private:
  std::vector<std::byte> buffer_;
};

}  // namespace lunar::unreal_tcp
