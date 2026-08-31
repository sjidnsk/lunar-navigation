#include <gtest/gtest.h>

#include <numbers>

#include "lunar_pure_planner_ros/lunar_surface_demo_state.hpp"

namespace lunar::pure_planner_ros {
namespace {

nav_msgs::msg::Path StraightPath(const double finish_x_m) {
  nav_msgs::msg::Path path;
  path.header.frame_id = "odom";
  path.poses.resize(2U);
  path.poses[1].pose.position.x = finish_x_m;
  return path;
}

TEST(LunarSurfaceDemoState, PublishesFirstLocalMapThenEveryFourMetres) {
  LunarSurfaceDemoState state;
  state.Reset(0.0, 0.0);
  EXPECT_TRUE(state.LocalMapDue(4.0));
  state.MarkLocalMapPublished();
  EXPECT_FALSE(state.LocalMapDue(4.0));

  state.AcceptPath(StraightPath(10.0));
  for (int step = 0; step < 7; ++step) {
    state.Advance(0.5);
  }
  EXPECT_DOUBLE_EQ(state.x_m(), 3.5);
  EXPECT_FALSE(state.LocalMapDue(4.0));
  state.Advance(0.5);
  EXPECT_DOUBLE_EQ(state.x_m(), 4.0);
  EXPECT_TRUE(state.LocalMapDue(4.0));
}

TEST(LunarSurfaceDemoState, WaitsForBothConsumersBeforePublishing) {
  EXPECT_FALSE(LocalMapPublicationReady(true, false, 0U));
  EXPECT_FALSE(LocalMapPublicationReady(true, false, 1U));
  EXPECT_TRUE(LocalMapPublicationReady(true, false, 2U));
  EXPECT_FALSE(LocalMapPublicationReady(false, false, 2U));
  EXPECT_TRUE(LocalMapPublicationReady(false, true, 2U));
}

TEST(LunarSurfaceDemoState, ResetForcesLocalMapAndClearsMotion) {
  LunarSurfaceDemoState state;
  state.Reset(0.0, 0.0);
  state.MarkLocalMapPublished();
  state.AcceptPath(StraightPath(10.0));
  state.Advance(0.5);

  state.Reset(2.0, 3.0, 0.7);

  EXPECT_DOUBLE_EQ(state.x_m(), 2.0);
  EXPECT_DOUBLE_EQ(state.y_m(), 3.0);
  EXPECT_DOUBLE_EQ(state.yaw_rad(), 0.7);
  EXPECT_FALSE(state.has_active_path());
  EXPECT_TRUE(state.LocalMapDue(4.0));
  state.Advance(0.5);
  EXPECT_DOUBLE_EQ(state.x_m(), 2.0);
  EXPECT_DOUBLE_EQ(state.y_m(), 3.0);
}

TEST(LunarSurfaceDemoState, EmptyPathImmediatelyStopsExistingMotion) {
  LunarSurfaceDemoState state;
  state.Reset(0.0, 0.0);
  state.AcceptPath(StraightPath(10.0));
  ASSERT_TRUE(state.has_active_path());
  state.Advance(0.5);
  ASSERT_DOUBLE_EQ(state.x_m(), 0.5);

  state.AcceptPath(nav_msgs::msg::Path{});
  EXPECT_FALSE(state.has_active_path());
  state.Advance(0.5);
  EXPECT_DOUBLE_EQ(state.x_m(), 0.5);
}

TEST(LunarSurfaceDemoState, DenseWaypointsConsumeOnlyOneStepDistance) {
  LunarSurfaceDemoState state;
  state.Reset(0.0, 0.0);
  nav_msgs::msg::Path path;
  path.header.frame_id = "odom";
  path.poses.resize(4U);
  path.poses[1].pose.position.x = 0.2;
  path.poses[2].pose.position.x = 0.4;
  path.poses[3].pose.position.x = 1.0;
  state.AcceptPath(path);

  state.Advance(0.5);

  EXPECT_NEAR(state.x_m(), 0.5, 1.0e-12);
  EXPECT_DOUBLE_EQ(state.y_m(), 0.0);
}

TEST(LunarSurfaceDemoState, AdvanceUpdatesYawToActualMotionDirection) {
  LunarSurfaceDemoState state;
  state.Reset(0.0, 0.0);
  nav_msgs::msg::Path path;
  path.header.frame_id = "odom";
  path.poses.resize(2U);
  path.poses[1].pose.position.y = 1.0;
  state.AcceptPath(path);

  state.Advance(0.5);

  EXPECT_NEAR(state.yaw_rad(), std::numbers::pi / 2.0, 1.0e-12);
}

}  // namespace
}  // namespace lunar::pure_planner_ros
