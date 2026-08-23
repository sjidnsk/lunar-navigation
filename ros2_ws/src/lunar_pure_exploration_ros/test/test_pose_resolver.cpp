#include "lunar_pure_exploration_ros/pose_resolver.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

#include "geometry_msgs/msg/transform_stamped.hpp"

namespace lunar::pure_exploration_ros {
namespace {

nav_msgs::msg::Odometry Odometry(double x, double y, double z, double yaw,
                                 double quaternion_scale = 1.0) {
  nav_msgs::msg::Odometry odometry;
  odometry.header.frame_id = "odom";
  odometry.child_frame_id = "base_link";
  odometry.pose.pose.position.x = x;
  odometry.pose.pose.position.y = y;
  odometry.pose.pose.position.z = z;
  odometry.pose.pose.orientation.z =
      quaternion_scale * std::sin(yaw / 2.0);
  odometry.pose.pose.orientation.w =
      quaternion_scale * std::cos(yaw / 2.0);
  return odometry;
}

geometry_msgs::msg::TransformStamped Transform(
    std::string parent, std::string child, double x, double y, double z,
    double yaw, double quaternion_scale = 1.0) {
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = std::move(parent);
  transform.child_frame_id = std::move(child);
  transform.transform.translation.x = x;
  transform.transform.translation.y = y;
  transform.transform.translation.z = z;
  transform.transform.rotation.z =
      quaternion_scale * std::sin(yaw / 2.0);
  transform.transform.rotation.w =
      quaternion_scale * std::cos(yaw / 2.0);
  return transform;
}

tf2_msgs::msg::TFMessage Transforms(
    std::initializer_list<geometry_msgs::msg::TransformStamped> values) {
  tf2_msgs::msg::TFMessage message;
  message.transforms.assign(values.begin(), values.end());
  return message;
}

void ExpectPose(const std::optional<lunar::pure_exploration::Pose2>& pose,
                double x, double y, double yaw) {
  ASSERT_TRUE(pose.has_value());
  EXPECT_NEAR(pose->x, x, 1e-12);
  EXPECT_NEAR(pose->y, y, 1e-12);
  EXPECT_NEAR(pose->yaw, yaw, 1e-12);
}

TEST(PoseResolver, WaitsUntilBothRequiredCachesExist) {
  PoseResolver resolver;
  EXPECT_FALSE(resolver.LatestPoseInMap().has_value());

  resolver.UpdateOdometry(Odometry(1.0, 2.0, 0.0, 0.0));
  EXPECT_FALSE(resolver.LatestPoseInMap().has_value());

  PoseResolver transform_only;
  transform_only.UpdateTransforms(
      Transforms({Transform("map", "odom", 3.0, 4.0, 0.0, 0.0)}));
  EXPECT_FALSE(transform_only.LatestPoseInMap().has_value());
}

TEST(PoseResolver, ComposesNormalizedPlanarPose) {
  PoseResolver resolver;
  resolver.UpdateOdometry(
      Odometry(2.0, 3.0, 9.0, -std::numbers::pi / 4.0, 7.0));
  resolver.UpdateTransforms(Transforms({Transform(
      "map", "odom", 10.0, -2.0, -4.0, std::numbers::pi / 2.0, 3.0)}));

  ExpectPose(resolver.LatestPoseInMap(), 7.0, 0.0,
             std::numbers::pi / 4.0);
}

TEST(PoseResolver, RejectsOverflowingOdometryUpdateAndPreservesOldPair) {
  PoseResolver resolver;
  const double maximum = std::numeric_limits<double>::max();
  resolver.UpdateTransforms(
      Transforms({Transform("map", "odom", maximum, 0.0, 0.0, 0.0)}));
  resolver.UpdateOdometry(Odometry(-maximum, 0.0, 0.0, 0.0));
  ExpectPose(resolver.LatestPoseInMap(), 0.0, 0.0, 0.0);

  EXPECT_THROW(resolver.UpdateOdometry(Odometry(maximum, 0.0, 0.0, 0.0)),
               std::invalid_argument);
  ExpectPose(resolver.LatestPoseInMap(), 0.0, 0.0, 0.0);
}

TEST(PoseResolver, RejectsOverflowingTransformUpdateAndPreservesOldPair) {
  PoseResolver resolver;
  const double maximum = std::numeric_limits<double>::max();
  resolver.UpdateOdometry(Odometry(maximum, 0.0, 0.0, 0.0));
  resolver.UpdateTransforms(
      Transforms({Transform("map", "odom", -maximum, 0.0, 0.0, 0.0)}));
  ExpectPose(resolver.LatestPoseInMap(), 0.0, 0.0, 0.0);

  EXPECT_THROW(
      resolver.UpdateTransforms(
          Transforms({Transform("map", "odom", maximum, 0.0, 0.0, 0.0)})),
      std::invalid_argument);
  ExpectPose(resolver.LatestPoseInMap(), 0.0, 0.0, 0.0);
}

TEST(PoseResolver, RejectsOverflowWhenOdometryIsSecondCache) {
  PoseResolver resolver;
  const double maximum = std::numeric_limits<double>::max();
  resolver.UpdateTransforms(
      Transforms({Transform("map", "odom", maximum, 0.0, 0.0, 0.0)}));
  EXPECT_FALSE(resolver.LatestPoseInMap().has_value());

  EXPECT_THROW(resolver.UpdateOdometry(Odometry(maximum, 0.0, 0.0, 0.0)),
               std::invalid_argument);
  EXPECT_FALSE(resolver.LatestPoseInMap().has_value());

  resolver.UpdateOdometry(Odometry(-maximum, 0.0, 0.0, 0.0));
  ExpectPose(resolver.LatestPoseInMap(), 0.0, 0.0, 0.0);
}

TEST(PoseResolver, RejectsOverflowWhenTransformIsSecondCache) {
  PoseResolver resolver;
  const double maximum = std::numeric_limits<double>::max();
  resolver.UpdateOdometry(Odometry(maximum, 0.0, 0.0, 0.0));
  EXPECT_FALSE(resolver.LatestPoseInMap().has_value());

  EXPECT_THROW(
      resolver.UpdateTransforms(
          Transforms({Transform("map", "odom", maximum, 0.0, 0.0, 0.0)})),
      std::invalid_argument);
  EXPECT_FALSE(resolver.LatestPoseInMap().has_value());

  resolver.UpdateTransforms(
      Transforms({Transform("map", "odom", -maximum, 0.0, 0.0, 0.0)}));
  ExpectPose(resolver.LatestPoseInMap(), 0.0, 0.0, 0.0);
}

TEST(PoseResolver, ReplacesByArrivalOrderIgnoringStampsAndCovariance) {
  PoseResolver resolver;
  auto first_odometry = Odometry(1.0, 0.0, 0.0, 0.0);
  first_odometry.header.stamp.sec = 100;
  resolver.UpdateOdometry(first_odometry);

  auto latest_odometry = Odometry(2.0, 0.0, 0.0, 0.0);
  latest_odometry.header.stamp.sec = 0;
  latest_odometry.pose.covariance.fill(
      std::numeric_limits<double>::quiet_NaN());
  latest_odometry.pose.covariance[0] = 1000000.0;
  latest_odometry.twist.covariance.fill(
      std::numeric_limits<double>::infinity());
  resolver.UpdateOdometry(latest_odometry);

  auto first_transform = Transform("map", "odom", 10.0, 0.0, 0.0, 0.0);
  first_transform.header.stamp.sec = 100;
  resolver.UpdateTransforms(Transforms({first_transform}));
  ExpectPose(resolver.LatestPoseInMap(), 12.0, 0.0, 0.0);

  auto latest_transform = Transform("map", "odom", 20.0, 0.0, 0.0, 0.0);
  latest_transform.header.stamp.sec = 0;
  resolver.UpdateTransforms(Transforms({latest_transform}));
  ExpectPose(resolver.LatestPoseInMap(), 22.0, 0.0, 0.0);
}

TEST(PoseResolver, TfOdomFromBaseNeverOverridesOdometry) {
  PoseResolver resolver;
  resolver.UpdateOdometry(Odometry(2.0, 3.0, 0.0, 0.0));
  resolver.UpdateTransforms(Transforms({
      Transform("map", "odom", 10.0, 20.0, 0.0, 0.0),
      Transform("odom", "base_link", 100.0, 200.0, 0.0, 0.0),
  }));
  ExpectPose(resolver.LatestPoseInMap(), 12.0, 23.0, 0.0);
}

TEST(PoseResolver, LastDirectTransformInMessageIsTheCandidate) {
  PoseResolver resolver;
  resolver.UpdateOdometry(Odometry(1.0, 0.0, 0.0, 0.0));

  auto malformed_first = Transform("map", "odom", 1.0, 0.0, 0.0, 0.0);
  malformed_first.transform.rotation.w = 0.0;
  resolver.UpdateTransforms(Transforms({
      malformed_first,
      Transform("map", "odom", 4.0, 0.0, 0.0, 0.0),
  }));
  ExpectPose(resolver.LatestPoseInMap(), 5.0, 0.0, 0.0);

  resolver.UpdateTransforms(Transforms({
      Transform("map", "odom", 6.0, 0.0, 0.0, 0.0),
      Transform("map", "odom", 8.0, 0.0, 0.0, 0.0),
  }));
  ExpectPose(resolver.LatestPoseInMap(), 9.0, 0.0, 0.0);
}

TEST(PoseResolver, InvalidSelectedDirectTransformPreservesOldCache) {
  PoseResolver resolver;
  resolver.UpdateOdometry(Odometry(1.0, 2.0, 0.0, 0.0));
  resolver.UpdateTransforms(
      Transforms({Transform("map", "odom", 3.0, 4.0, 0.0, 0.0)}));

  auto invalid_last = Transform("map", "odom", 8.0, 9.0, 0.0, 0.0);
  invalid_last.transform.rotation.w = 0.0;
  EXPECT_THROW(
      resolver.UpdateTransforms(Transforms({
          Transform("map", "odom", 20.0, 30.0, 0.0, 0.0), invalid_last})),
      std::invalid_argument);
  ExpectPose(resolver.LatestPoseInMap(), 4.0, 6.0, 0.0);
}

TEST(PoseResolver, UnrelatedChainInverseAndEmptyTfPreserveCache) {
  PoseResolver resolver;
  resolver.UpdateOdometry(Odometry(1.0, 2.0, 0.0, 0.0));
  resolver.UpdateTransforms(
      Transforms({Transform("map", "odom", 3.0, 4.0, 0.0, 0.0)}));

  auto invalid_unrelated = Transform("world", "sensor", 0.0, 0.0, 0.0, 0.0);
  invalid_unrelated.transform.translation.x =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_NO_THROW(resolver.UpdateTransforms(Transforms({
      Transform("odom", "map", 100.0, 0.0, 0.0, 0.0),
      Transform("map", "middle", 100.0, 0.0, 0.0, 0.0),
      Transform("middle", "odom", 100.0, 0.0, 0.0, 0.0),
      invalid_unrelated,
  })));
  EXPECT_NO_THROW(resolver.UpdateTransforms(Transforms({})));
  ExpectPose(resolver.LatestPoseInMap(), 4.0, 6.0, 0.0);
}

TEST(PoseResolver, RejectsWrongOdometryFramesWithStrongGuarantee) {
  PoseResolver resolver;
  resolver.UpdateOdometry(Odometry(1.0, 2.0, 0.0, 0.0));
  resolver.UpdateTransforms(
      Transforms({Transform("map", "odom", 3.0, 4.0, 0.0, 0.0)}));

  auto wrong_parent = Odometry(20.0, 30.0, 0.0, 0.0);
  wrong_parent.header.frame_id = "map";
  EXPECT_THROW(resolver.UpdateOdometry(wrong_parent), std::invalid_argument);

  auto wrong_child = Odometry(20.0, 30.0, 0.0, 0.0);
  wrong_child.child_frame_id = "base_footprint";
  EXPECT_THROW(resolver.UpdateOdometry(wrong_child), std::invalid_argument);
  ExpectPose(resolver.LatestPoseInMap(), 4.0, 6.0, 0.0);
}

TEST(PoseResolver, RejectsNonfiniteOdometryPositionWithStrongGuarantee) {
  PoseResolver resolver;
  resolver.UpdateOdometry(Odometry(1.0, 2.0, 3.0, 0.0));
  resolver.UpdateTransforms(
      Transforms({Transform("map", "odom", 3.0, 4.0, 0.0, 0.0)}));

  for (const double invalid_value : {
           std::numeric_limits<double>::quiet_NaN(),
           std::numeric_limits<double>::infinity(),
           -std::numeric_limits<double>::infinity()}) {
    for (const int coordinate : {0, 1, 2}) {
      auto invalid = Odometry(20.0, 30.0, 40.0, 0.0);
      double* const values[] = {&invalid.pose.pose.position.x,
                                &invalid.pose.pose.position.y,
                                &invalid.pose.pose.position.z};
      *values[coordinate] = invalid_value;
      EXPECT_THROW(resolver.UpdateOdometry(invalid), std::invalid_argument);
    }
  }
  ExpectPose(resolver.LatestPoseInMap(), 4.0, 6.0, 0.0);
}

TEST(PoseResolver, RejectsZeroOrNonfiniteOdometryQuaternion) {
  PoseResolver resolver;
  resolver.UpdateOdometry(Odometry(1.0, 2.0, 0.0, 0.0));
  resolver.UpdateTransforms(
      Transforms({Transform("map", "odom", 3.0, 4.0, 0.0, 0.0)}));

  auto zero = Odometry(20.0, 30.0, 0.0, 0.0);
  zero.pose.pose.orientation.w = 0.0;
  EXPECT_THROW(resolver.UpdateOdometry(zero), std::invalid_argument);

  for (const int component : {0, 1, 2, 3}) {
    auto nonfinite = Odometry(20.0, 30.0, 0.0, 0.0);
    double* const values[] = {&nonfinite.pose.pose.orientation.x,
                              &nonfinite.pose.pose.orientation.y,
                              &nonfinite.pose.pose.orientation.z,
                              &nonfinite.pose.pose.orientation.w};
    *values[component] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(resolver.UpdateOdometry(nonfinite), std::invalid_argument);
  }
  ExpectPose(resolver.LatestPoseInMap(), 4.0, 6.0, 0.0);
}

TEST(PoseResolver, RejectsNonfiniteDirectTranslationOrQuaternion) {
  PoseResolver resolver;
  resolver.UpdateOdometry(Odometry(1.0, 2.0, 0.0, 0.0));
  resolver.UpdateTransforms(
      Transforms({Transform("map", "odom", 3.0, 4.0, 5.0, 0.0)}));

  for (const double invalid_value : {
           std::numeric_limits<double>::quiet_NaN(),
           std::numeric_limits<double>::infinity(),
           -std::numeric_limits<double>::infinity()}) {
    for (const int coordinate : {0, 1, 2}) {
      auto invalid = Transform("map", "odom", 20.0, 30.0, 40.0, 0.0);
      double* const values[] = {&invalid.transform.translation.x,
                                &invalid.transform.translation.y,
                                &invalid.transform.translation.z};
      *values[coordinate] = invalid_value;
      EXPECT_THROW(resolver.UpdateTransforms(Transforms({invalid})),
                   std::invalid_argument);
    }
  }

  auto zero = Transform("map", "odom", 20.0, 30.0, 0.0, 0.0);
  zero.transform.rotation.w = 0.0;
  EXPECT_THROW(resolver.UpdateTransforms(Transforms({zero})),
               std::invalid_argument);

  for (const int component : {0, 1, 2, 3}) {
    auto nonfinite = Transform("map", "odom", 20.0, 30.0, 0.0, 0.0);
    double* const values[] = {&nonfinite.transform.rotation.x,
                              &nonfinite.transform.rotation.y,
                              &nonfinite.transform.rotation.z,
                              &nonfinite.transform.rotation.w};
    *values[component] = -std::numeric_limits<double>::infinity();
    EXPECT_THROW(resolver.UpdateTransforms(Transforms({nonfinite})),
                 std::invalid_argument);
  }
  ExpectPose(resolver.LatestPoseInMap(), 4.0, 6.0, 0.0);
}

TEST(PoseResolver, NormalizesComposedYaw) {
  PoseResolver resolver;
  resolver.UpdateOdometry(
      Odometry(0.0, 0.0, 0.0, 3.0 * std::numbers::pi / 4.0));
  resolver.UpdateTransforms(Transforms({Transform(
      "map", "odom", 0.0, 0.0, 0.0, 3.0 * std::numbers::pi / 4.0)}));
  ExpectPose(resolver.LatestPoseInMap(), 0.0, 0.0,
             -std::numbers::pi / 2.0);
}

TEST(PoseResolver, NormalizesFiniteExtremeQuaternionScales) {
  PoseResolver resolver;
  resolver.UpdateOdometry(Odometry(1.0, 2.0, 0.0,
                                    std::numbers::pi / 2.0,
                                    std::numeric_limits<double>::max()));
  resolver.UpdateTransforms(Transforms({Transform(
      "map", "odom", 3.0, 4.0, 0.0, 0.0,
      std::numeric_limits<double>::denorm_min())}));
  ExpectPose(resolver.LatestPoseInMap(), 4.0, 6.0,
             std::numbers::pi / 2.0);
}

}  // namespace
}  // namespace lunar::pure_exploration_ros
