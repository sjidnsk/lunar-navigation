#pragma once

#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <string>

#include "lunar_unreal_tcp_bridge/protocol.hpp"

namespace lunar::unreal_tcp {

struct QueuePushResult final {
  bool accepted{};
  bool replaced_latest{};
  bool fatal{};
  std::string reason_code;
};

class OutboundQueue final {
 public:
  explicit OutboundQueue(std::size_t non_droppable_capacity = 64U);

  [[nodiscard]] QueuePushResult Push(Frame frame);
  [[nodiscard]] std::optional<Frame> Pop();
  void Clear() noexcept;
  [[nodiscard]] std::size_t size() const noexcept;

 private:
  mutable std::mutex mutex_;
  std::size_t non_droppable_capacity_;
  std::deque<Frame> urgent_;
  std::deque<Frame> reliable_;
  std::optional<Frame> latest_state_;
  std::optional<Frame> latest_map_;
};

}  // namespace lunar::unreal_tcp
