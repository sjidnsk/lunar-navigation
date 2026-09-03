#include <limits>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_ros/state_adapter.hpp"

namespace lunar::incremental_navigation_ros {
namespace {

[[nodiscard]] geometry_msgs::msg::TransformStamped MapFromOdom() {
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "map";
  transform.child_frame_id = "odom";
  transform.transform.translation.x = 10.0;
  transform.transform.translation.y = -2.0;
  transform.transform.rotation.w = 2.0;
  return transform;
}

[[nodiscard]] nav_msgs::msg::Odometry OdomState(
    const std::string& child_frame = "base_link") {
  nav_msgs::msg::Odometry odometry;
  odometry.header.frame_id = "odom";
  odometry.child_frame_id = child_frame;
  odometry.pose.pose.position.x = 2.0;
  odometry.pose.pose.position.y = 3.0;
  odometry.pose.pose.orientation.w = 3.0;
  odometry.twist.twist.linear.x = 1.5;
  odometry.pose.covariance[0] = 1000000.0;
  odometry.twist.covariance[0] = 1000000.0;
  return odometry;
}

TEST(StateAdapterTest, RejectsNonDirectTransformAndUnsupportedFrames) {
  auto transform = MapFromOdom();
  transform.header.frame_id = "earth";
  EXPECT_FALSE(AdaptDirectMapFromOdom(transform).value);

  const auto direct = AdaptDirectMapFromOdom(MapFromOdom());
  ASSERT_TRUE(direct.value);
  auto odometry = OdomState("camera");
  EXPECT_FALSE(AdaptStateInput(*direct.value, odometry).value);
  odometry = OdomState();
  odometry.header.frame_id = "map";
  EXPECT_FALSE(AdaptStateInput(*direct.value, odometry).value);
}

TEST(StateAdapterTest, NormalizesDirectTransformWithoutFreshnessPolicy) {
  auto transform = MapFromOdom();
  transform.header.stamp.sec = 456;

  const auto adapted = AdaptDirectMapFromOdom(transform);

  ASSERT_TRUE(adapted.value) << adapted.reason_code;
  EXPECT_DOUBLE_EQ(adapted.value->rotation.w, 1.0);
  EXPECT_EQ(adapted.value->stamp.nanoseconds_since_epoch, 0);
}

TEST(StateAdapterTest, ConvertsOnlyTheSharedOdomToBaseLinkPoseContract) {
  const auto direct = AdaptDirectMapFromOdom(MapFromOdom());
  ASSERT_TRUE(direct.value);
  auto rejected_odometry = OdomState("base_footprint");
  EXPECT_FALSE(AdaptStateInput(*direct.value, rejected_odometry).value);

  auto odometry = OdomState("base_link");
  odometry.twist.twist.linear.x = std::numeric_limits<double>::quiet_NaN();

  const auto adapted = AdaptStateInput(*direct.value, odometry);

  ASSERT_TRUE(adapted.value) << adapted.reason_code;
  EXPECT_DOUBLE_EQ(adapted.value->base_link_pose.position_m.x, 12.0);
  EXPECT_DOUBLE_EQ(adapted.value->base_link_pose.position_m.y, 1.0);
  EXPECT_DOUBLE_EQ(adapted.value->base_link_pose.yaw_rad, 0.0);
}

}  // namespace
}  // namespace lunar::incremental_navigation_ros
