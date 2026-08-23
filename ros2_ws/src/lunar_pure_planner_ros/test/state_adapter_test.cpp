#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include "lunar_pure_planner_ros/state_adapter.hpp"

namespace lunar::pure_planner_ros {
namespace {

geometry_msgs::msg::TransformStamped MapFromOdom() {
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "map";
  transform.child_frame_id = "odom";
  transform.transform.translation.x = 10.0;
  transform.transform.translation.y = -2.0;
  transform.transform.rotation.w = 2.0;
  return transform;
}

nav_msgs::msg::Odometry OdomState() {
  nav_msgs::msg::Odometry odometry;
  odometry.header.stamp.sec = 0;
  odometry.header.frame_id = "odom";
  odometry.child_frame_id = "base_link";
  odometry.pose.pose.position.x = 2.0;
  odometry.pose.pose.position.y = 3.0;
  odometry.pose.pose.orientation.w = 3.0;
  odometry.twist.twist.linear.x = 1.5;
  odometry.twist.twist.angular.z = -0.25;
  odometry.pose.covariance[0] = 1000000.0;
  odometry.twist.covariance[0] = 1000000.0;
  return odometry;
}

TEST(StateAdapter, RejectsNonDirectTfAndNonOdomBaseOdometryFrames) {
  auto transform = MapFromOdom();
  transform.header.frame_id = "earth";
  EXPECT_FALSE(AdaptDirectMapFromOdom(transform).value.has_value());

  auto odometry = OdomState();
  odometry.child_frame_id = "wheel";
  EXPECT_FALSE(AdaptOdometry(odometry,
                             lunar::pure_planning::PlatformType::kWheeled)
                   .value.has_value());
}

TEST(StateAdapter, NormalizesDirectMapFromOdomWithoutReadingTfTime) {
  auto transform = MapFromOdom();
  transform.header.stamp.sec = 0;

  const auto adapted = AdaptDirectMapFromOdom(transform);

  ASSERT_TRUE(adapted.value.has_value()) << adapted.reason_code;
  EXPECT_DOUBLE_EQ(adapted.value->rotation.w, 1.0);
  EXPECT_EQ(adapted.value->stamp.nanoseconds_since_epoch, 0);
}

TEST(StateAdapter, ConvertsOdomPoseAndTwistDespiteLargeCovariance) {
  const auto adapted = AdaptOdometry(OdomState(),
                                     lunar::pure_planning::PlatformType::kWheeled);

  ASSERT_TRUE(adapted.value.has_value()) << adapted.reason_code;
  const auto* state = std::get_if<lunar::pure_planning::WheeledState>(&*adapted.value);
  ASSERT_NE(state, nullptr);
  EXPECT_DOUBLE_EQ(state->pose.position_m.x, 2.0);
  EXPECT_DOUBLE_EQ(state->pose.orientation.w, 1.0);
  EXPECT_DOUBLE_EQ(state->velocity.linear_mps.x, 1.5);
  EXPECT_DOUBLE_EQ(state->velocity.angular_radps.z, -0.25);
}

TEST(StateAdapter, ComposesMapFromOdomWithOdomFromBaseAndNeverUsesTfBase) {
  const auto transform = AdaptDirectMapFromOdom(MapFromOdom());
  ASSERT_TRUE(transform.value.has_value());
  const auto state = AdaptOdometry(OdomState(),
                                   lunar::pure_planning::PlatformType::kLegged);
  ASSERT_TRUE(state.value.has_value());

  const auto map_from_base = ComposeMapFromOdomAndOdometry(
      *transform.value, *state.value);

  ASSERT_TRUE(map_from_base.value.has_value()) << map_from_base.reason_code;
  EXPECT_DOUBLE_EQ(map_from_base.value->position_m.x, 12.0);
  EXPECT_DOUBLE_EQ(map_from_base.value->position_m.y, 1.0);
  EXPECT_DOUBLE_EQ(map_from_base.value->orientation.w, 1.0);
}

}  // namespace
}  // namespace lunar::pure_planner_ros
