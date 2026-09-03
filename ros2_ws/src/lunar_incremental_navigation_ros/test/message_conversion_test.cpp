#include "lunar_incremental_navigation_ros/message_conversion.hpp"

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

namespace lunar::incremental_navigation_ros {
namespace {

using Action = lunar_planning_msgs::action::NavigateToPose;
using namespace lunar::incremental_navigation;

TEST(MessageConversion, ConvertsOnlyFiniteMinimalGoal) {
  Action::Goal goal;
  goal.target_x_m = 1.5;
  goal.target_y_m = -2.0;
  goal.has_target_yaw = false;
  goal.target_yaw_rad = std::numeric_limits<double>::quiet_NaN();
  const auto without_yaw = ConvertGoal(goal);
  ASSERT_TRUE(without_yaw.ok());
  EXPECT_DOUBLE_EQ(without_yaw.goal->target_x_m, 1.5);
  EXPECT_FALSE(without_yaw.goal->has_target_yaw);

  goal.has_target_yaw = true;
  EXPECT_FALSE(ConvertGoal(goal).ok());
  goal.target_yaw_rad = 0.4;
  ASSERT_TRUE(ConvertGoal(goal).ok());
  goal.target_x_m = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(ConvertGoal(goal).ok());
  EXPECT_EQ(ConvertGoal(goal).reason_code, "INVALID_GOAL");
}

TEST(MessageConversion, PreservesUuidAndBuildsZeroStampMapPathReference) {
  PathReference reference{
      .session_id = SessionId{{0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U,
                               8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U}},
      .segment_revision = 7U,
      .traversability_revision = 9U,
      .state = PathState::kActive,
      .reaches_final_goal = true,
  };
  reference.path.poses.push_back(Pose3{
      .position_m = {.x = 1.0, .y = 2.0, .z = 3.0},
      .orientation = {.w = 0.5, .x = 0.1, .y = 0.2, .z = 0.3},
  });

  const auto message = ConvertPathReference(reference);
  EXPECT_EQ(message.session_id.uuid, reference.session_id.bytes);
  EXPECT_EQ(message.segment_revision, 7U);
  EXPECT_EQ(message.traversability_revision, 9U);
  EXPECT_EQ(message.state, message.ACTIVE);
  EXPECT_TRUE(message.reaches_final_goal);
  ASSERT_EQ(message.path.poses.size(), 1U);
  EXPECT_EQ(message.path.header.frame_id, "map");
  EXPECT_EQ(message.path.header.stamp.sec, 0);
  EXPECT_EQ(message.path.header.stamp.nanosec, 0U);
  EXPECT_EQ(message.path.poses.front().header, message.path.header);
  EXPECT_DOUBLE_EQ(message.path.poses.front().pose.position.z, 3.0);
  EXPECT_DOUBLE_EQ(message.path.poses.front().pose.orientation.z, 0.3);
}

TEST(MessageConversion, InvalidatedReferenceAndClearedGlobalRouteAreEmpty) {
  PathReference invalid{
      .session_id = SessionId{{42U}},
      .segment_revision = 3U,
      .traversability_revision = 10U,
      .state = PathState::kInvalidated,
  };
  const auto message = ConvertPathReference(invalid);
  EXPECT_EQ(message.state, message.INVALIDATED);
  EXPECT_TRUE(message.path.poses.empty());

  const auto cleared = ConvertGlobalRoute(std::nullopt);
  EXPECT_EQ(cleared.header.frame_id, "map");
  EXPECT_EQ(cleared.header.stamp.sec, 0);
  EXPECT_EQ(cleared.header.stamp.nanosec, 0U);
  EXPECT_TRUE(cleared.poses.empty());
}

TEST(MessageConversion, ConvertsFeedbackAndAllTerminalOutcomesExactly) {
  const auto feedback = ConvertFeedback(NavigateToPoseFeedback{
      .session_state = SessionState::kReplanning,
      .planning_cycle = 5U,
      .active_segment_revision = 4U,
      .reason_code = "PATH_BLOCKED",
  });
  EXPECT_EQ(feedback.session_state, Action::Feedback::REPLANNING);
  EXPECT_EQ(feedback.planning_cycle, 5U);
  EXPECT_EQ(feedback.active_segment_revision, 4U);
  EXPECT_EQ(feedback.reason_code, "PATH_BLOCKED");

  for (const SessionOutcome outcome : {
           SessionOutcome::kGoalReached, SessionOutcome::kNoPath,
           SessionOutcome::kInvalidGoal, SessionOutcome::kMapUnavailable,
           SessionOutcome::kTimeout, SessionOutcome::kCanceled,
           SessionOutcome::kInternalError}) {
    const auto result = ConvertResult(NavigateToPoseResult{
        .outcome = outcome,
        .reason_code = "REASON",
        .last_segment_revision = 11U,
    });
    EXPECT_EQ(result.outcome, static_cast<std::uint8_t>(outcome));
    EXPECT_EQ(result.reason_code, "REASON");
    EXPECT_EQ(result.last_segment_revision, 11U);
  }
}

}  // namespace
}  // namespace lunar::incremental_navigation_ros
