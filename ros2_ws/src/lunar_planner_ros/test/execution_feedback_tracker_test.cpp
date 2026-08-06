#include "lunar_planner_ros/execution_feedback_tracker.hpp"

#include <chrono>
#include <string>
#include <variant>

#include <gtest/gtest.h>

namespace lunar::planning::ros {
namespace {

using namespace std::chrono_literals;

ExpectedExecution GroundExpected(std::string plan_id = "wheel/request-2") {
  return ExpectedExecution{
      .platform_type = lunar::planning::PlatformType::kWheeled,
      .base_frame_id = "base_link",
      .plan_id = plan_id,
      .segment_id = plan_id,
      .maximum_age = 500ms,
  };
}

lunar_navigation_msgs::msg::MotionExecutionFeedback GroundFeedback(
    const std::uint64_t sequence,
    const std::uint8_t state =
        lunar_navigation_msgs::msg::MotionExecutionFeedback::EXECUTING) {
  lunar_navigation_msgs::msg::MotionExecutionFeedback message;
  message.header.frame_id = "base_link";
  message.header.stamp.sec = 10;
  message.sequence = sequence;
  message.platform_type = message.WHEELED;
  message.plan_id = "wheel/request-2";
  message.segment_id = "wheel/request-2";
  message.state = state;
  return message;
}

TEST(ExecutionFeedbackTracker, AcceptsStrictPerPlanSequenceAndMapsGroundIds) {
  ExecutionFeedbackTracker tracker;
  tracker.SetExpected(GroundExpected());

  EXPECT_TRUE(tracker.Accept(
      GroundFeedback(1U), rclcpp::Time{10'100'000'000LL}).ok());
  auto second = GroundFeedback(2U);
  second.state = second.SEGMENT_COMPLETE;
  EXPECT_TRUE(tracker.Accept(second, rclcpp::Time{10'200'000'000LL}).ok());

  const auto accepted_context = tracker.context();
  ASSERT_TRUE(accepted_context.has_value());
  const auto* context = std::get_if<lunar::planning::GroundExecutionContext>(
      &*accepted_context);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->state, lunar::planning::GroundExecutionState::kHolding);
  EXPECT_EQ(context->active_plan_id, "wheel/request-2");
  EXPECT_EQ(context->active_segment_id, "wheel/request-2");
}

TEST(ExecutionFeedbackTracker, RejectsStaleWrongPlatformAndWrongSegment) {
  ExecutionFeedbackTracker tracker;
  tracker.SetExpected(GroundExpected());

  auto wrong_segment = GroundFeedback(1U);
  wrong_segment.segment_id = "wheel/other";
  EXPECT_EQ(
      tracker.Accept(wrong_segment, rclcpp::Time{10'100'000'000LL})
          .reason_code,
      "EXECUTION_FEEDBACK_SEGMENT_MISMATCH");

  auto wrong_platform = GroundFeedback(1U);
  wrong_platform.platform_type = wrong_platform.LEGGED;
  EXPECT_EQ(
      tracker.Accept(wrong_platform, rclcpp::Time{10'100'000'000LL})
          .reason_code,
      "EXECUTION_FEEDBACK_PLATFORM_MISMATCH");

  EXPECT_EQ(
      tracker.Accept(GroundFeedback(1U), rclcpp::Time{10'600'000'001LL})
          .reason_code,
      "EXECUTION_FEEDBACK_STALE");
  EXPECT_FALSE(tracker.context().has_value());
}

TEST(ExecutionFeedbackTracker, RequiresSequenceOneThenContiguousUpdates) {
  ExecutionFeedbackTracker tracker;
  const auto expected = GroundExpected();

  EXPECT_EQ(
      tracker.Accept(
          GroundFeedback(2U), expected, rclcpp::Time{10'100'000'000LL})
          .reason_code,
      "EXECUTION_FEEDBACK_SEQUENCE_MISMATCH");
  EXPECT_TRUE(tracker.Accept(
      GroundFeedback(1U), expected, rclcpp::Time{10'100'000'000LL}).ok());
  EXPECT_EQ(
      tracker.Accept(
          GroundFeedback(3U), expected, rclcpp::Time{10'200'000'000LL})
          .reason_code,
      "EXECUTION_FEEDBACK_SEQUENCE_MISMATCH");
}

TEST(ExecutionFeedbackTracker, MapsHopperLandingAndRejectsGroundOnlyState) {
  ExecutionFeedbackTracker tracker;
  ExpectedExecution expected{
      .platform_type = lunar::planning::PlatformType::kHopper,
      .base_frame_id = "hopper_base",
      .plan_id = "hopper/request-1",
      .segment_id = "hop-1",
      .maximum_age = 500ms,
  };
  auto landed = GroundFeedback(1U);
  landed.header.frame_id = "hopper_base";
  landed.platform_type = landed.HOPPER;
  landed.plan_id = expected.plan_id;
  landed.segment_id = expected.segment_id;
  landed.state = landed.LANDED_HOLD;
  ASSERT_TRUE(tracker.Accept(
      landed, expected, rclcpp::Time{10'100'000'000LL}).ok());
  const auto landed_context = tracker.context();
  ASSERT_TRUE(landed_context.has_value());
  const auto* context = std::get_if<lunar::planning::HopperExecutionContext>(
      &*landed_context);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->state, lunar::planning::HopperExecutionState::kLandedHold);

  tracker.SetExpected(GroundExpected("wheel/request-3"));
  auto invalid = GroundFeedback(1U);
  invalid.plan_id = "wheel/request-3";
  invalid.segment_id = "wheel/request-3";
  invalid.state = invalid.LANDED_HOLD;
  EXPECT_EQ(
      tracker.Accept(invalid, rclcpp::Time{10'100'000'000LL}).reason_code,
      "EXECUTION_FEEDBACK_STATE_INVALID");
}

}  // namespace
}  // namespace lunar::planning::ros
