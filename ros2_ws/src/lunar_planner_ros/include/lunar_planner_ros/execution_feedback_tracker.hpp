#pragma once

#include <chrono>
#include <compare>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

#include <lunar_navigation_msgs/msg/motion_execution_feedback.hpp>
#include <rclcpp/time.hpp>

#include "lunar_planner_core/types/execution_context.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"

namespace lunar::planning::ros {

struct ExpectedExecution final {
  lunar::planning::PlatformType platform_type{
      lunar::planning::PlatformType::kWheeled};
  std::string base_frame_id;
  std::string plan_id;
  std::string segment_id;
  std::chrono::nanoseconds maximum_age{std::chrono::seconds{1}};

  auto operator<=>(const ExpectedExecution&) const = default;
};

struct FeedbackAcceptResult final {
  bool accepted{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept { return accepted; }
};

class ExecutionFeedbackTracker final {
 public:
  void SetExpected(ExpectedExecution expected);
  void Clear();

  [[nodiscard]] FeedbackAcceptResult Accept(
      const lunar_navigation_msgs::msg::MotionExecutionFeedback& message,
      rclcpp::Time now);
  [[nodiscard]] FeedbackAcceptResult Accept(
      const lunar_navigation_msgs::msg::MotionExecutionFeedback& message,
      const ExpectedExecution& expected,
      rclcpp::Time now);

  [[nodiscard]] std::optional<lunar::planning::ExecutionContext> context()
      const;
  [[nodiscard]] std::optional<lunar::planning::ExecutionContext> context(
      rclcpp::Time now) const;

 private:
  mutable std::mutex mutex_;
  std::optional<ExpectedExecution> expected_;
  std::optional<ExpectedExecution> pending_expected_;
  std::optional<lunar::planning::ExecutionContext> context_;
  std::uint64_t last_sequence_{};
  std::int64_t last_stamp_nanoseconds_{};
};

}  // namespace lunar::planning::ros
