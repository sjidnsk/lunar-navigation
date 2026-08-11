#include "lunar_unreal_tcp_bridge/outbound_queue.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace lunar::unreal_tcp {
namespace {

enum class Priority : std::uint8_t { kUrgent, kReliable, kState, kMap };

[[nodiscard]] Priority Classify(const Frame& frame) {
  if (frame.header.message_type == MessageType::kError) {
    return Priority::kUrgent;
  }
  if (frame.header.message_type == MessageType::kControl &&
      frame.metadata.is_object()) {
    const auto command = frame.metadata.find("command");
    if (command != frame.metadata.end() && command->is_string() &&
        (*command == "HOLD" || *command == "RESET_SESSION")) {
      return Priority::kUrgent;
    }
  }
  if (frame.header.message_type == MessageType::kRobotState) {
    return Priority::kState;
  }
  if (frame.header.message_type == MessageType::kLocalElevationMap) {
    return Priority::kMap;
  }
  return Priority::kReliable;
}

[[nodiscard]] QueuePushResult Accepted(const bool replaced = false) {
  return QueuePushResult{
      .accepted = true,
      .replaced_latest = replaced,
      .fatal = false,
      .reason_code = "ACCEPTED",
  };
}

}  // namespace

OutboundQueue::OutboundQueue(const std::size_t non_droppable_capacity)
    : non_droppable_capacity_(non_droppable_capacity) {}

QueuePushResult OutboundQueue::Push(Frame frame) {
  std::scoped_lock lock{mutex_};
  const Priority priority = Classify(frame);
  if (priority == Priority::kState) {
    const bool replaced = latest_state_.has_value();
    latest_state_ = std::move(frame);
    return Accepted(replaced);
  }
  if (priority == Priority::kMap) {
    const bool replaced = latest_map_.has_value();
    latest_map_ = std::move(frame);
    return Accepted(replaced);
  }
  if (urgent_.size() + reliable_.size() >= non_droppable_capacity_) {
    return QueuePushResult{
        .accepted = false,
        .replaced_latest = false,
        .fatal = true,
        .reason_code = "OUTBOUND_RELIABLE_QUEUE_EXHAUSTED",
    };
  }
  if (priority == Priority::kUrgent) {
    urgent_.push_back(std::move(frame));
  } else {
    reliable_.push_back(std::move(frame));
  }
  return Accepted();
}

std::optional<Frame> OutboundQueue::Pop() {
  std::scoped_lock lock{mutex_};
  const auto pop_front = [](std::deque<Frame>& queue) {
    Frame frame = std::move(queue.front());
    queue.pop_front();
    return std::optional<Frame>{std::move(frame)};
  };
  if (!urgent_.empty()) {
    return pop_front(urgent_);
  }
  if (!reliable_.empty()) {
    return pop_front(reliable_);
  }
  if (latest_state_.has_value()) {
    auto result = std::move(latest_state_);
    latest_state_.reset();
    return result;
  }
  if (latest_map_.has_value()) {
    auto result = std::move(latest_map_);
    latest_map_.reset();
    return result;
  }
  return std::nullopt;
}

void OutboundQueue::Clear() noexcept {
  std::scoped_lock lock{mutex_};
  urgent_.clear();
  reliable_.clear();
  latest_state_.reset();
  latest_map_.reset();
}

std::size_t OutboundQueue::size() const noexcept {
  std::scoped_lock lock{mutex_};
  return urgent_.size() + reliable_.size() +
      static_cast<std::size_t>(latest_state_.has_value()) +
      static_cast<std::size_t>(latest_map_.has_value());
}

}  // namespace lunar::unreal_tcp
