#include "lunar_planner_ros/execution_feedback_tracker.hpp"

#include <cstdint>
#include <optional>
#include <utility>

namespace lunar::planning::ros {
namespace {

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;

[[nodiscard]] std::optional<std::int64_t> StampNanoseconds(
    const builtin_interfaces::msg::Time& stamp) noexcept {
  if (stamp.sec < 0 || stamp.nanosec >= kNanosecondsPerSecond ||
      (stamp.sec == 0 && stamp.nanosec == 0U)) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(stamp.sec) * kNanosecondsPerSecond +
      static_cast<std::int64_t>(stamp.nanosec);
}

[[nodiscard]] std::uint8_t PlatformMessageValue(
    const lunar::planning::PlatformType platform) noexcept {
  using Message = lunar_navigation_msgs::msg::MotionExecutionFeedback;
  switch (platform) {
    case lunar::planning::PlatformType::kWheeled:
      return Message::WHEELED;
    case lunar::planning::PlatformType::kLegged:
      return Message::LEGGED;
    case lunar::planning::PlatformType::kHopper:
      return Message::HOPPER;
  }
  return 0U;
}

[[nodiscard]] bool ValidExpected(const ExpectedExecution& expected) noexcept {
  return !expected.base_frame_id.empty() && !expected.plan_id.empty() &&
      !expected.segment_id.empty() && expected.maximum_age.count() > 0;
}

[[nodiscard]] FeedbackAcceptResult Reject(std::string reason_code) {
  return FeedbackAcceptResult{
      .accepted = false,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] std::optional<lunar::planning::ExecutionContext> MapContext(
    const lunar_navigation_msgs::msg::MotionExecutionFeedback& message,
    const ExpectedExecution& expected) {
  using Message = lunar_navigation_msgs::msg::MotionExecutionFeedback;
  if (expected.platform_type != lunar::planning::PlatformType::kHopper) {
    lunar::planning::GroundExecutionState state;
    switch (message.state) {
      case Message::IDLE:
        state = lunar::planning::GroundExecutionState::kIdle;
        break;
      case Message::ACCEPTED:
      case Message::EXECUTING:
        state = lunar::planning::GroundExecutionState::kExecuting;
        break;
      case Message::SEGMENT_COMPLETE:
        state = lunar::planning::GroundExecutionState::kHolding;
        break;
      case Message::FAILED:
      case Message::CANCELED:
        state = lunar::planning::GroundExecutionState::kFault;
        break;
      case Message::LANDED_HOLD:
      default:
        return std::nullopt;
    }
    return lunar::planning::GroundExecutionContext{
        .state = state,
        .active_plan_id = expected.plan_id,
        .active_segment_id = expected.segment_id,
    };
  }

  lunar::planning::HopperExecutionState state;
  switch (message.state) {
    case Message::IDLE:
      state = lunar::planning::HopperExecutionState::kGroundHold;
      break;
    case Message::ACCEPTED:
      state = lunar::planning::HopperExecutionState::kJumpReady;
      break;
    case Message::EXECUTING:
      state = lunar::planning::HopperExecutionState::kInFlight;
      break;
    case Message::LANDED_HOLD:
      state = lunar::planning::HopperExecutionState::kLandedHold;
      break;
    case Message::FAILED:
    case Message::CANCELED:
      state = lunar::planning::HopperExecutionState::kEmergencyDelegated;
      break;
    case Message::SEGMENT_COMPLETE:
    default:
      return std::nullopt;
  }
  return lunar::planning::HopperExecutionContext{
      .state = state,
      .active_plan_id = expected.plan_id,
      .active_segment_id = expected.segment_id,
  };
}

}  // namespace

void ExecutionFeedbackTracker::SetExpected(ExpectedExecution expected) {
  std::scoped_lock lock{mutex_};
  if (expected_.has_value() && *expected_ == expected) {
    return;
  }
  if (pending_expected_.has_value() && *pending_expected_ == expected) {
    return;
  }
  if (expected_.has_value() && context_.has_value()) {
    pending_expected_ = std::move(expected);
    return;
  }
  expected_ = std::move(expected);
  pending_expected_.reset();
  context_.reset();
  last_sequence_ = 0U;
  last_stamp_nanoseconds_ = 0;
}

void ExecutionFeedbackTracker::Clear() {
  std::scoped_lock lock{mutex_};
  expected_.reset();
  pending_expected_.reset();
  context_.reset();
  last_sequence_ = 0U;
  last_stamp_nanoseconds_ = 0;
}

FeedbackAcceptResult ExecutionFeedbackTracker::Accept(
    const lunar_navigation_msgs::msg::MotionExecutionFeedback& message,
    const ExpectedExecution& expected,
    const rclcpp::Time now) {
  SetExpected(expected);
  return Accept(message, now);
}

FeedbackAcceptResult ExecutionFeedbackTracker::Accept(
    const lunar_navigation_msgs::msg::MotionExecutionFeedback& message,
    const rclcpp::Time now) {
  std::scoped_lock lock{mutex_};
  const auto reject = [this](std::string reason_code) {
    context_.reset();
    return Reject(std::move(reason_code));
  };
  if (!expected_.has_value() || !ValidExpected(*expected_)) {
    return reject("EXECUTION_FEEDBACK_EXPECTATION_INVALID");
  }
  const bool pending_plan = pending_expected_.has_value() &&
      message.plan_id == pending_expected_->plan_id;
  const ExpectedExecution& selected =
      pending_plan ? *pending_expected_ : *expected_;
  if (message.header.frame_id != selected.base_frame_id) {
    return reject("EXECUTION_FEEDBACK_FRAME_MISMATCH");
  }
  const auto stamp = StampNanoseconds(message.header.stamp);
  if (!stamp.has_value()) {
    return reject("EXECUTION_FEEDBACK_STAMP_INVALID");
  }
  if (*stamp > now.nanoseconds() ||
      now.nanoseconds() - *stamp > selected.maximum_age.count()) {
    return reject("EXECUTION_FEEDBACK_STALE");
  }
  if (message.platform_type != PlatformMessageValue(selected.platform_type)) {
    return reject("EXECUTION_FEEDBACK_PLATFORM_MISMATCH");
  }
  if (message.plan_id.empty() || message.plan_id != selected.plan_id) {
    return reject("EXECUTION_FEEDBACK_PLAN_MISMATCH");
  }
  if (message.segment_id.empty() ||
      message.segment_id != selected.segment_id) {
    return reject("EXECUTION_FEEDBACK_SEGMENT_MISMATCH");
  }
  const std::uint64_t expected_sequence =
      pending_plan ? 1U : last_sequence_ + 1U;
  if (message.sequence == 0U || message.sequence != expected_sequence ||
      (!pending_plan && *stamp < last_stamp_nanoseconds_)) {
    return reject("EXECUTION_FEEDBACK_SEQUENCE_MISMATCH");
  }
  using Message = lunar_navigation_msgs::msg::MotionExecutionFeedback;
  if ((message.state == Message::FAILED ||
       message.state == Message::CANCELED) &&
      message.reason_code.empty()) {
    return reject("EXECUTION_FEEDBACK_REASON_REQUIRED");
  }
  auto context = MapContext(message, selected);
  if (!context.has_value()) {
    return reject("EXECUTION_FEEDBACK_STATE_INVALID");
  }

  if (pending_plan) {
    expected_ = std::move(pending_expected_);
    pending_expected_.reset();
  }
  last_sequence_ = message.sequence;
  last_stamp_nanoseconds_ = *stamp;
  context_ = std::move(context);
  return FeedbackAcceptResult{.accepted = true, .reason_code = "ACCEPTED"};
}

std::optional<lunar::planning::ExecutionContext>
ExecutionFeedbackTracker::context() const {
  std::scoped_lock lock{mutex_};
  return context_;
}

std::optional<lunar::planning::ExecutionContext>
ExecutionFeedbackTracker::context(const rclcpp::Time now) const {
  std::scoped_lock lock{mutex_};
  if (!context_.has_value() || !expected_.has_value() ||
      last_stamp_nanoseconds_ <= 0 ||
      last_stamp_nanoseconds_ > now.nanoseconds() ||
      now.nanoseconds() - last_stamp_nanoseconds_ >
          expected_->maximum_age.count()) {
    return std::nullopt;
  }
  return context_;
}

}  // namespace lunar::planning::ros
