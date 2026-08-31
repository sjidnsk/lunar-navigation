#include "lunar_pure_planner_ros/traversability_input.hpp"

#include <utility>

#include "lunar_pure_planner_ros/state_adapter.hpp"

namespace lunar::pure_planner_ros {
namespace {

constexpr char kInvalidInput[] = "INVALID_INPUT";
constexpr char kTfUnavailable[] = "TF_UNAVAILABLE";

[[nodiscard]] std::string ReasonOrInvalid(const std::string& reason_code) {
  return reason_code.empty() ? kInvalidInput : reason_code;
}

[[nodiscard]] bool SameStamp(const builtin_interfaces::msg::Time& left,
                             const builtin_interfaces::msg::Time& right) {
  return left.sec == right.sec && left.nanosec == right.nanosec;
}

}  // namespace

TraversabilityInput::TraversabilityInput(
    lunar::pure_planning::TraversabilityProfile profile,
    const bool token_idempotent)
    : map_(std::move(profile)), token_idempotent_(token_idempotent) {}

void TraversabilityInput::UpdateGlobal(
    nav_msgs::msg::OccupancyGrid::ConstSharedPtr message) {
  std::scoped_lock lock{mutex_};
  if (!message) {
    direct_tf_error_ = false;
    reason_code_ = kInvalidInput;
    return;
  }
  auto adapted = AdaptGlobal(*message);
  if (!adapted.value.has_value()) {
    direct_tf_error_ = false;
    reason_code_ = ReasonOrInvalid(adapted.reason_code);
    return;
  }
  latest_global_ = std::move(*adapted.value);
  ++global_sequence_;
  const auto update = map_.UpdateGlobal(*latest_global_);
  reason_code_ = update.accepted ? std::string{} : ReasonOrInvalid(update.reason_code);
  if (update.accepted) {
    ApplyPendingLocalLocked();
  }
}

void TraversabilityInput::UpdateLocal(
    grid_map_msgs::msg::GridMap::ConstSharedPtr message) {
  std::scoped_lock lock{mutex_};
  if (!message) {
    direct_tf_error_ = false;
    reason_code_ = kInvalidInput;
    return;
  }
  auto adapted = AdaptLocal(*message);
  if (!adapted.value.has_value()) {
    direct_tf_error_ = false;
    reason_code_ = ReasonOrInvalid(adapted.reason_code);
    return;
  }
  const auto stamp = message->header.stamp;
  const bool duplicate_token =
      token_idempotent_ && latest_local_map_stamp_.has_value() &&
      SameStamp(stamp, *latest_local_map_stamp_);
  latest_local_ = std::move(*adapted.value);
  latest_local_map_stamp_ = stamp;
  if (!duplicate_token) {
    ++local_sequence_;
  }
  ApplyPendingLocalLocked();
}

void TraversabilityInput::UpdateTf(const tf2_msgs::msg::TFMessage& message) {
  std::scoped_lock lock{mutex_};
  bool received_direct_transform = false;
  for (const auto& transform : message.transforms) {
    if (transform.header.frame_id != "map" || transform.child_frame_id != "odom") {
      continue;
    }
    received_direct_transform = true;
    const auto adapted = AdaptDirectMapFromOdom(transform);
    if (!adapted.value.has_value()) {
      direct_tf_error_ = true;
      reason_code_ = ReasonOrInvalid(adapted.reason_code);
      continue;
    }
    const bool clear_direct_tf_error = direct_tf_error_;
    const bool local_was_pending =
        latest_local_.has_value() && applied_local_sequence_ != local_sequence_;
    direct_tf_error_ = false;
    latest_map_from_odom_ = *adapted.value;
    ++tf_sequence_;
    ApplyPendingLocalLocked();
    if (clear_direct_tf_error && !local_was_pending &&
        applied_local_sequence_ != 0U) {
      reason_code_.clear();
    }
  }
  if (!received_direct_transform && latest_local_.has_value() &&
      !latest_map_from_odom_.has_value()) {
    reason_code_ = kTfUnavailable;
  }
}

TraversabilityInputSnapshot TraversabilityInput::Capture() const {
  std::scoped_lock lock{mutex_};
  if (applied_local_sequence_ == 0U) {
    return {.snapshot = nullptr,
            .reason_code = reason_code_.empty() ? kTfUnavailable : reason_code_,
            .local_sequence = 0U,
            .local_map_stamp = builtin_interfaces::msg::Time{}};
  }
  return {.snapshot = map_.Capture(),
          .reason_code = reason_code_,
          .local_sequence = applied_local_sequence_,
          .local_map_stamp = applied_local_map_stamp_.value_or(
              builtin_interfaces::msg::Time{})};
}

void TraversabilityInput::ApplyPendingLocalLocked() {
  if (!latest_local_.has_value() || applied_local_sequence_ == local_sequence_) {
    return;
  }
  if (!latest_map_from_odom_.has_value()) {
    reason_code_ = kTfUnavailable;
    return;
  }
  const auto update = map_.UpdateLocal(*latest_local_, *latest_map_from_odom_,
                                        local_sequence_);
  if (update.accepted) {
    applied_local_sequence_ = local_sequence_;
    applied_local_map_stamp_ = latest_local_map_stamp_;
    direct_tf_error_ = false;
  }
  reason_code_ = update.accepted ? std::string{} : ReasonOrInvalid(update.reason_code);
}

}  // namespace lunar::pure_planner_ros
