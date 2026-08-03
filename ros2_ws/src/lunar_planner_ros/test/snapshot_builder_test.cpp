#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <variant>

#include <gtest/gtest.h>

#include "lunar_planner_ros/snapshot_builder.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning::ros {
namespace {

using namespace std::chrono_literals;

SnapshotPolicy ValidPolicy() {
  return SnapshotPolicy{
      .global_map_max_age = 500ms,
      .local_map_max_age = 500ms,
      .odometry_max_age = 500ms,
      .localization_status_max_age = 500ms,
      .tf_max_age = 500ms,
      .max_pairwise_skew = 200ms,
      .degraded_pose_covariance_limit = 0.5,
      .degraded_twist_covariance_limit = 0.5,
  };
}

GoalRequest ValidGoal() {
  return GoalRequest{
      .request_id = "request-1",
      .frame_id = "map",
      .stamp = rclcpp::Time{10'000'000'000LL},
      .goal =
          lunar::planning::GoalRegion{
              .goal_id = "goal-1",
              .target = lunar::planning::PointGoal{
                  .position_m = {12.0, 0.5, 0.0},
                  .tolerance_m = 0.2,
              },
              .yaw_rad = std::nullopt,
              .yaw_tolerance_rad = 0.0,
          },
      .previous_execution = std::nullopt,
      .stop_token = {},
  };
}

std::shared_ptr<SnapshotStore> ValidStore() {
  auto store = std::make_shared<SnapshotStore>();
  store->UpdateGlobalMap(test::MakeGridMap("map"));
  store->UpdateLocalMap(test::MakeGridMap("odom"));
  store->UpdateOdometry(test::MakeOdometry());
  store->UpdateLocalizationStatus(test::MakeLocalizationStatus());
  store->UpdateTransforms(test::MakeTransforms());
  return store;
}

SnapshotBuilder MakeBuilder(
    std::shared_ptr<const SnapshotStore> store,
    const SnapshotPolicy policy = ValidPolicy()) {
  return SnapshotBuilder{
      std::move(store), policy, test::MakeWheeledCapability()};
}

TEST(SnapshotBuilder, FreezesExactlyOneValidInputAndTransformsMapGoalToOdom) {
  const SnapshotBuildResult result = MakeBuilder(ValidStore()).Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});

  ASSERT_TRUE(result.ok());
  ASSERT_TRUE(result.input.has_value());
  EXPECT_FALSE(result.error.has_value());
  EXPECT_EQ(result.input->state_time.nanoseconds_since_epoch, 10'000'000'000LL);
  EXPECT_EQ(result.input->world.global_map.frame_id, "map");
  EXPECT_EQ(result.input->world.local_map.frame_id, "odom");
  EXPECT_EQ(result.input->world.map_from_odom.parent_frame, "map");
  EXPECT_EQ(result.input->world.map_from_odom.child_frame, "odom");
  const auto& goal = std::get<lunar::planning::PointGoal>(
      result.input->goal.target);
  EXPECT_NEAR(goal.position_m.x, 2.0, 1.0e-9);
  EXPECT_NEAR(goal.position_m.y, 0.5, 1.0e-9);
  ASSERT_TRUE(std::holds_alternative<lunar::planning::WheeledState>(
      result.input->current_state));
  EXPECT_EQ(
      result.input->config.maximum_input_skew,
      ValidPolicy().max_pairwise_skew);
}

TEST(SnapshotBuilder, RejectsMissingStaleAndSkewedInputs) {
  auto missing = ValidStore();
  missing->ClearTransforms();
  const SnapshotBuildResult missing_result = MakeBuilder(missing).Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(missing_result.error.has_value());
  EXPECT_EQ(missing_result.error->code, SnapshotErrorCode::kStaleTf);

  auto stale = ValidStore();
  auto stale_local = test::MakeGridMap("odom");
  stale_local.header.stamp = test::Stamp(9'000'000'000LL);
  stale->UpdateLocalMap(stale_local);
  const SnapshotBuildResult stale_result = MakeBuilder(stale).Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(stale_result.error.has_value());
  EXPECT_EQ(stale_result.error->code, SnapshotErrorCode::kStaleLocalMap);

  auto skewed = ValidStore();
  auto skewed_local = test::MakeGridMap("odom");
  skewed_local.header.stamp = test::Stamp(9'750'000'000LL);
  skewed->UpdateLocalMap(skewed_local);
  const SnapshotBuildResult skewed_result = MakeBuilder(skewed).Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(skewed_result.error.has_value());
  EXPECT_EQ(skewed_result.error->code, SnapshotErrorCode::kInputSkew);
}

TEST(SnapshotBuilder, RejectsWrongFramesInvalidCovarianceAndLocalizationState) {
  auto wrong_frame = ValidStore();
  auto odometry = test::MakeOdometry();
  odometry.header.frame_id = "map";
  wrong_frame->UpdateOdometry(odometry);
  const SnapshotBuildResult frame_result = MakeBuilder(wrong_frame).Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(frame_result.error.has_value());
  EXPECT_EQ(frame_result.error->code, SnapshotErrorCode::kInvalidOdometry);

  auto invalid_covariance = ValidStore();
  odometry = test::MakeOdometry();
  odometry.pose.covariance[0] =
      std::numeric_limits<double>::quiet_NaN();
  invalid_covariance->UpdateOdometry(odometry);
  const SnapshotBuildResult covariance_result =
      MakeBuilder(invalid_covariance).Freeze(
          ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(covariance_result.error.has_value());
  EXPECT_EQ(
      covariance_result.error->code,
      SnapshotErrorCode::kInvalidOdometry);

  auto invalid_status = ValidStore();
  invalid_status->UpdateLocalizationStatus(test::MakeLocalizationStatus(
      lunar_navigation_msgs::msg::LocalizationStatus::INVALID));
  const SnapshotBuildResult status_result = MakeBuilder(invalid_status).Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(status_result.error.has_value());
  EXPECT_EQ(
      status_result.error->code,
      SnapshotErrorCode::kInvalidLocalization);
}

TEST(SnapshotBuilder, AllowsOnlyBoundedDegradedCovariance) {
  auto bounded = ValidStore();
  bounded->UpdateLocalizationStatus(test::MakeLocalizationStatus(
      lunar_navigation_msgs::msg::LocalizationStatus::DEGRADED));
  EXPECT_TRUE(MakeBuilder(bounded)
                  .Freeze(ValidGoal(), rclcpp::Time{10'100'000'000LL})
                  .ok());

  auto excessive = ValidStore();
  excessive->UpdateLocalizationStatus(test::MakeLocalizationStatus(
      lunar_navigation_msgs::msg::LocalizationStatus::DEGRADED));
  auto odometry = test::MakeOdometry();
  odometry.pose.covariance[0] = 0.6;
  excessive->UpdateOdometry(odometry);
  const SnapshotBuildResult result = MakeBuilder(excessive).Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(result.error.has_value());
  EXPECT_EQ(result.error->code, SnapshotErrorCode::kCovarianceLimit);
}

TEST(SnapshotBuilder, RejectsTfExtrapolationAndIncompletePolicy) {
  auto extrapolated = ValidStore();
  extrapolated->ClearTransforms();
  extrapolated->UpdateTransforms(test::MakeTransforms(9'950'000'000LL));
  const SnapshotBuildResult tf_result = MakeBuilder(extrapolated).Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(tf_result.error.has_value());
  EXPECT_EQ(tf_result.error->code, SnapshotErrorCode::kStaleTf);

  SnapshotPolicy incomplete = ValidPolicy();
  incomplete.tf_max_age = 0ns;
  EXPECT_FALSE(ValidateSnapshotPolicy(incomplete));
  const SnapshotBuildResult policy_result =
      MakeBuilder(ValidStore(), incomplete)
          .Freeze(ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(policy_result.error.has_value());
  EXPECT_EQ(
      policy_result.error->code,
      SnapshotErrorCode::kConfigurationInvalid);
}

}  // namespace
}  // namespace lunar::planning::ros
