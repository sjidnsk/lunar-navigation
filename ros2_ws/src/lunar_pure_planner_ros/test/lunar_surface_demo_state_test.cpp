#include <gtest/gtest.h>

#include <numbers>

#include <lunar_planning_msgs/msg/demo_map_ack.hpp>
#include <lunar_planning_msgs/msg/demo_plan_segment.hpp>

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

builtin_interfaces::msg::Time Stamp(const int32_t seconds) {
  builtin_interfaces::msg::Time stamp;
  stamp.sec = seconds;
  return stamp;
}

lunar_planning_msgs::msg::DemoPlanSegment StopSegment(
    const std::string& request_id) {
  lunar_planning_msgs::msg::DemoPlanSegment segment;
  segment.request_id = request_id;
  segment.platform_type = lunar_planning_msgs::msg::DemoPlanSegment::WHEELED;
  segment.command = lunar_planning_msgs::msg::DemoPlanSegment::STOP;
  return segment;
}

lunar_planning_msgs::msg::DemoPlanSegment ExecuteSegment(
    const std::string& request_id, const builtin_interfaces::msg::Time& token,
    const uint32_t segment_index, const nav_msgs::msg::Path& path) {
  lunar_planning_msgs::msg::DemoPlanSegment segment;
  segment.request_id = request_id;
  segment.map_token = token;
  segment.segment_index = segment_index;
  segment.platform_type = lunar_planning_msgs::msg::DemoPlanSegment::WHEELED;
  segment.command = lunar_planning_msgs::msg::DemoPlanSegment::EXECUTE;
  segment.executable_path = path;
  return segment;
}

lunar_planning_msgs::msg::DemoMapAck Ack(
    const std::string& request_id, const builtin_interfaces::msg::Time& token) {
  lunar_planning_msgs::msg::DemoMapAck ack;
  ack.request_id = request_id;
  ack.map_token = token;
  ack.platform_type = lunar_planning_msgs::msg::DemoMapAck::WHEELED;
  return ack;
}

TEST(LunarSurfaceDemoState,
     DeliveryProtocolStopsUntilMatchingAckThenActivatesCachedSegment) {
  LunarSurfaceDemoState state;
  state.Reset(0.0, 0.0);
  state.EnableDeliveryProtocol(
      lunar_planning_msgs::msg::DemoPlanSegment::WHEELED);

  EXPECT_TRUE(state.HandleSegment(StopSegment("rviz-2")));
  EXPECT_FALSE(state.can_advance());
  state.BeginMapDelivery(Stamp(10));
  EXPECT_TRUE(state.map_delivery_pending());

  EXPECT_TRUE(state.HandleSegment(
      ExecuteSegment("rviz-2", Stamp(10), 1U, StraightPath(12.0))));
  state.Advance(0.5);
  EXPECT_DOUBLE_EQ(state.x_m(), 0.0);

  EXPECT_TRUE(state.HandleMapAck(Ack("rviz-2", Stamp(10))));
  EXPECT_FALSE(state.map_delivery_pending());
  EXPECT_TRUE(state.can_advance());
  state.Advance(0.5);
  EXPECT_DOUBLE_EQ(state.x_m(), 0.5);
}

TEST(LunarSurfaceDemoState,
     DeliveryProtocolRejectsExecuteSegmentFromOldRequestOrMapToken) {
  LunarSurfaceDemoState state;
  state.Reset(0.0, 0.0);
  state.EnableDeliveryProtocol(
      lunar_planning_msgs::msg::DemoPlanSegment::WHEELED);
  ASSERT_TRUE(state.HandleSegment(StopSegment("rviz-2")));
  state.BeginMapDelivery(Stamp(10));
  ASSERT_TRUE(state.HandleMapAck(Ack("rviz-2", Stamp(10))));

  EXPECT_FALSE(state.HandleSegment(
      ExecuteSegment("rviz-1", Stamp(10), 1U, StraightPath(12.0))));
  EXPECT_FALSE(state.HandleSegment(
      ExecuteSegment("rviz-2", Stamp(9), 1U, StraightPath(12.0))));
  EXPECT_FALSE(state.can_advance());
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

TEST(LunarSurfaceDemoState,
     DeliveryProtocolStopsAtFourMetresAndClearsTheExecutedSegment) {
  LunarSurfaceDemoState state;
  state.Reset(0.0, 0.0);
  state.EnableDeliveryProtocol(
      lunar_planning_msgs::msg::DemoPlanSegment::WHEELED);
  ASSERT_TRUE(state.HandleSegment(StopSegment("rviz-4m")));
  state.BeginMapDelivery(Stamp(10));
  ASSERT_TRUE(state.HandleMapAck(Ack("rviz-4m", Stamp(10))));
  ASSERT_TRUE(state.HandleSegment(
      ExecuteSegment("rviz-4m", Stamp(10), 1U, StraightPath(12.0))));

  for (int step = 0; step < 7; ++step) {
    state.Advance(0.5);
  }
  EXPECT_DOUBLE_EQ(state.x_m(), 3.5);
  EXPECT_FALSE(state.ShouldBeginMapDelivery(4.0));

  state.Advance(0.5);
  ASSERT_TRUE(state.ShouldBeginMapDelivery(4.0));
  state.BeginMapDelivery(Stamp(11));
  EXPECT_FALSE(state.can_advance());
  EXPECT_FALSE(state.has_active_path());
  state.Advance(0.5);
  EXPECT_DOUBLE_EQ(state.x_m(), 4.0);
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
