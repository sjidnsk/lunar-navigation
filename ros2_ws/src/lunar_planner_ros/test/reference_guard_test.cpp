#include "lunar_planner_ros/reference_guard.hpp"

#include <chrono>

#include <gtest/gtest.h>

namespace lunar::planning::ros {
namespace {

using namespace std::chrono_literals;

rclcpp::Time At(const std::int64_t nanoseconds) {
  return rclcpp::Time{nanoseconds, RCL_ROS_TIME};
}

lunar_planning_msgs::msg::MotionReference HopperReference() {
  lunar_planning_msgs::msg::MotionReference reference;
  reference.header.frame_id = "map";
  reference.plan_id = "hop-plan";
  reference.platform_type = reference.HOPPER;
  lunar_planning_msgs::msg::HopSegment hop;
  hop.header.frame_id = "odom";
  hop.header.stamp = At(10'000'000'000LL);
  hop.segment_id = "hop-1";
  hop.landing_region.points.resize(4U);
  hop.landing_region.points[0].x = 1.0F;
  hop.landing_region.points[0].y = -1.0F;
  hop.landing_region.points[1].x = 3.0F;
  hop.landing_region.points[1].y = -1.0F;
  hop.landing_region.points[2].x = 3.0F;
  hop.landing_region.points[2].y = 1.0F;
  hop.landing_region.points[3].x = 1.0F;
  hop.landing_region.points[3].y = 1.0F;
  hop.flight_time.sec = 2;
  hop.launch_pose.orientation.w = 1.0;
  hop.launch_velocity.x = 1.0;
  hop.launch_velocity.z = 2.0;
  hop.flight_tube_radius_m = 0.75;
  hop.nominal_landing_point.x = 2.0;
  hop.nominal_landing_point.z = 0.76;
  hop.required_delta_v_mps = 7.0;
  hop.available_delta_v_mps = 8.0;
  hop.capability_version = "hopper-capability-v1";
  hop.global_map_generation = 31U;
  hop.local_map_generation = 37U;
  reference.hops.push_back(hop);
  return reference;
}

nav_msgs::msg::Odometry LandedOdometry(const rclcpp::Time& stamp) {
  nav_msgs::msg::Odometry odometry;
  odometry.header.frame_id = "odom";
  odometry.child_frame_id = "base_link";
  odometry.header.stamp = stamp;
  odometry.pose.pose.position.x = 2.0;
  odometry.pose.pose.position.y = 0.0;
  odometry.pose.pose.orientation.w = 1.0;
  odometry.twist.twist.linear.x = 0.05;
  odometry.twist.twist.angular.z = 0.02;
  return odometry;
}

ReferenceGuard MakeGuard() {
  return ReferenceGuard{ReferenceGuardLimits{
      .minimum_settle_guard = 500ms,
      .maximum_landing_speed_mps = 0.2,
      .maximum_angular_speed_radps = 0.1,
  }};
}

TEST(ReferenceGuard, AllowsReplacementOnGroundAndRejectsInvalidCommit) {
  auto guard = MakeGuard();
  EXPECT_TRUE(guard.MayReplace(At(1), std::nullopt).may_replace);

  auto invalid = HopperReference();
  invalid.hops.clear();
  EXPECT_FALSE(guard.Commit(invalid, {0.0, 0.0, -1.62}));
  invalid = HopperReference();
  invalid.header.frame_id = "odom";
  EXPECT_FALSE(guard.Commit(invalid, {0.0, 0.0, -1.62}));
  invalid = HopperReference();
  invalid.hops.push_back(invalid.hops.front());
  EXPECT_FALSE(guard.Commit(invalid, {0.0, 0.0, -1.62}));
  invalid = HopperReference();
  invalid.hops.front().available_delta_v_mps = 6.9;
  EXPECT_FALSE(guard.Commit(invalid, {0.0, 0.0, -1.62}));
  invalid = HopperReference();
  invalid.hops.front().nominal_landing_point.x += 0.01;
  EXPECT_FALSE(guard.Commit(invalid, {0.0, 0.0, -1.62}));
  EXPECT_EQ(guard.state(), ReferenceGuardState::kGroundHold);
}

TEST(ReferenceGuard, LocksCommittedAndInFlightHopUntilStableLanding) {
  auto guard = MakeGuard();
  ASSERT_TRUE(guard.Commit(HopperReference(), {0.0, 0.0, -1.62}));

  auto decision = guard.MayReplace(At(9'000'000'000LL), std::nullopt);
  EXPECT_FALSE(decision.may_replace);
  EXPECT_EQ(decision.state, ReferenceGuardState::kJumpCommitted);

  decision = guard.MayReplace(At(11'000'000'000LL), std::nullopt);
  EXPECT_FALSE(decision.may_replace);
  EXPECT_EQ(decision.state, ReferenceGuardState::kInFlight);

  auto odometry = LandedOdometry(At(12'000'000'000LL));
  decision = guard.MayReplace(At(12'000'000'000LL), odometry);
  EXPECT_FALSE(decision.may_replace);
  EXPECT_EQ(decision.state, ReferenceGuardState::kLandedHold);

  odometry = LandedOdometry(At(12'600'000'000LL));
  decision = guard.MayReplace(At(12'600'000'000LL), odometry);
  EXPECT_TRUE(decision.may_replace);
  EXPECT_EQ(decision.state, ReferenceGuardState::kGroundHold);
}

TEST(ReferenceGuard, CommitsHopperUsingConfiguredExecutionFrameGravity) {
  auto reference = HopperReference();
  reference.hops.front().nominal_landing_point.z = 2.38;
  auto guard = MakeGuard();

  EXPECT_TRUE(guard.Commit(reference, {0.0, 0.0, -0.81}));
  EXPECT_EQ(guard.state(), ReferenceGuardState::kJumpCommitted);
}

TEST(ReferenceGuard, LatchesUnresolvedLandingUntilExplicitReset) {
  auto guard = MakeGuard();
  ASSERT_TRUE(guard.Commit(HopperReference(), {0.0, 0.0, -1.62}));

  auto invalid = LandedOdometry(At(12'100'000'000LL));
  invalid.pose.pose.position.x = 10.0;
  auto decision = guard.MayReplace(At(12'100'000'000LL), invalid);
  EXPECT_FALSE(decision.may_replace);
  EXPECT_EQ(decision.state, ReferenceGuardState::kUnresolved);
  EXPECT_EQ(decision.reason_code, "HOP_EXECUTION_STATE_UNRESOLVED");

  const auto valid = LandedOdometry(At(20'000'000'000LL));
  EXPECT_FALSE(guard.MayReplace(At(20'000'000'000LL), valid).may_replace);

  guard.Reset();
  EXPECT_TRUE(guard.MayReplace(At(20'000'000'001LL), valid).may_replace);
  EXPECT_EQ(guard.state(), ReferenceGuardState::kGroundHold);
}

TEST(ReferenceGuard, TreatsMissingPostFlightOdometryAsUnresolved) {
  auto guard = MakeGuard();
  ASSERT_TRUE(guard.Commit(HopperReference(), {0.0, 0.0, -1.62}));

  const auto decision = guard.MayReplace(At(13'000'000'000LL), std::nullopt);
  EXPECT_FALSE(decision.may_replace);
  EXPECT_EQ(decision.state, ReferenceGuardState::kUnresolved);
}

}  // namespace
}  // namespace lunar::planning::ros
