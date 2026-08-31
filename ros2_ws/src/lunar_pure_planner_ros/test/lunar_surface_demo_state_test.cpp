#include <gtest/gtest.h>

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

}  // namespace
}  // namespace lunar::pure_planner_ros
