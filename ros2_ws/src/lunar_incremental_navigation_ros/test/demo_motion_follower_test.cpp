#include <numbers>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_ros/demo_motion_follower.hpp"

namespace lunar::incremental_navigation_ros {
namespace {

TEST(DemoMotionFollower,
     DoesNotAdvanceBeyondTerminalPositionBeforeItsYawIsSatisfied) {
  // This catches the old emulator behavior that treated a duplicate terminal
  // position as an instantaneous yaw assignment and advanced immediately.
  DemoMotionFollower follower{1.0, 1.0};
  follower.SetPath({
      {.x_m = 0.0, .y_m = 0.0, .yaw_rad = 0.0},
      {.x_m = 1.0, .y_m = 0.0, .yaw_rad = 0.0},
      {.x_m = 1.0, .y_m = 0.0, .yaw_rad = std::numbers::pi / 2.0},
      {.x_m = 2.0, .y_m = 0.0, .yaw_rad = 0.0},
  });

  follower.Advance(1.0);

  const auto while_turning = follower.pose();
  EXPECT_DOUBLE_EQ(while_turning.x_m, 1.0);
  EXPECT_DOUBLE_EQ(while_turning.y_m, 0.0);
  EXPECT_NEAR(while_turning.yaw_rad, 1.0, 1.0e-9);

  follower.Advance(0.5);
  const auto still_turning = follower.pose();
  EXPECT_DOUBLE_EQ(still_turning.x_m, 1.0);
  EXPECT_DOUBLE_EQ(still_turning.y_m, 0.0);
  EXPECT_NEAR(still_turning.yaw_rad, 1.5, 1.0e-9);

  follower.Advance(0.5);
  const auto after_terminal_yaw = follower.pose();
  EXPECT_DOUBLE_EQ(after_terminal_yaw.x_m, 1.5);
  EXPECT_DOUBLE_EQ(after_terminal_yaw.y_m, 0.0);
  EXPECT_NEAR(after_terminal_yaw.yaw_rad, 0.0, 1.0e-9);
}

}  // namespace
}  // namespace lunar::incremental_navigation_ros
