#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numbers>
#include <variant>

#include <gtest/gtest.h>
#include <lunar_navigation_msgs/msg/hopper_propellant_state.hpp>

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
      .propellant_state_max_age = 500ms,
      .tf_max_age = 500ms,
      .max_pairwise_skew = 200ms,
      .degraded_pose_covariance_limit = 0.5,
      .degraded_twist_covariance_limit = 0.5,
  };
}

lunar_navigation_msgs::msg::HopperPropellantState ValidPropellant(
    const std::int64_t nanoseconds = 10'000'000'000LL) {
  lunar_navigation_msgs::msg::HopperPropellantState message;
  message.header.stamp = test::Stamp(nanoseconds);
  message.header.frame_id = "base_link";
  message.platform_id = "test-hopper";
  message.capability_version = "test-v1";
  message.total_mass_kg = 20.0;
  message.remaining_usable_fuel_mass_kg = 0.2;
  return message;
}

lunar::planning::HopperCapability HopperCapability() {
  return lunar::planning::HopperCapability{
      .specific_impulse_s = 301.0,
      .landing_support_radius_m = 0.45,
      .flight_collision_radius_m = 0.55,
      .maximum_landing_plane_residual_m = 0.05,
      .landing_lateral_margin_m = 0.2,
      .flight_map_margin_m = 0.2,
      .reachability_delta_v_margin_ratio = 0.1,
      .standard_gravity_mps2 = 9.80665,
      .maximum_landing_slope_rad = 0.17453292519943295,
  };
}

GoalRequest ValidGoal() {
  return GoalRequest{
      .request_id = "request-1",
      .mission_id = "mission-1",
      .mission_revision = 7U,
      .platform_id = "test-rover",
      .capability_version = "test-v1",
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
      .continuation = nullptr,
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
    std::shared_ptr<SnapshotStore> store,
    const SnapshotPolicy policy = ValidPolicy(),
    lunar::planning::PlannerConfig config = {}) {
  config.global_map.base_resolution_m = 1.0;
  return SnapshotBuilder{
      std::move(store), policy, test::MakeWheeledCapability(),
      std::move(config)};
}

SnapshotBuilder MakeHopperBuilder(
    std::shared_ptr<SnapshotStore> store,
    const SnapshotPolicy policy = ValidPolicy()) {
  lunar::planning::PlannerConfig config;
  config.global_map.base_resolution_m = 1.0;
  return SnapshotBuilder{
      std::move(store), policy, HopperCapability(), std::move(config),
      "base_link"};
}

GoalRequest ValidHopperGoal() {
  GoalRequest request = ValidGoal();
  request.platform_id = "test-hopper";
  request.capability_version = "test-v1";
  request.goal.target = lunar::planning::PointGoal{
      .position_m = {12.0, 0.5, 0.0},
      .tolerance_m = 0.0,
  };
  request.goal.yaw_rad.reset();
  request.goal.yaw_tolerance_rad = 0.0;
  return request;
}

void SetResolution(
    grid_map_msgs::msg::GridMap& map, const double resolution_m) {
  map.info.resolution = resolution_m;
  map.info.length_x = static_cast<double>(test::kMapWidth) * resolution_m;
  map.info.length_y = static_cast<double>(test::kMapHeight) * resolution_m;
}

TEST(SnapshotBuilder, FreezesExactlyOneValidMapFrameInput) {
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
      result.input->goal_map.target);
  EXPECT_NEAR(goal.position_m.x, 12.0, 1.0e-9);
  EXPECT_NEAR(goal.position_m.y, 0.5, 1.0e-9);
  ASSERT_TRUE(std::holds_alternative<lunar::planning::WheeledState>(
      result.input->current_state));
  EXPECT_EQ(
      result.input->config.maximum_input_skew,
      ValidPolicy().max_pairwise_skew);
  EXPECT_EQ(result.input->mission_id, "mission-1");
  EXPECT_EQ(result.input->mission_revision, 7U);
  EXPECT_EQ(result.input->platform_id, "test-rover");
  EXPECT_EQ(result.input->capability_version, "test-v1");
  EXPECT_GT(result.input->global_map_generation, 0U);
  EXPECT_GT(result.input->local_map_generation, 0U);
  EXPECT_GT(result.input->map_from_odom_generation, 0U);
}

TEST(SnapshotBuilder, BindsValidHopperPropellantIntoImmutableInput) {
  auto store = ValidStore();
  store->UpdateHopperPropellantState(ValidPropellant());

  const auto result = MakeHopperBuilder(store).Freeze(
      ValidHopperGoal(), rclcpp::Time{10'100'000'000LL});

  ASSERT_TRUE(result.ok())
      << (result.error ? result.error->reason_code : "");
  ASSERT_TRUE(result.input->hopper_propellant.has_value());
  EXPECT_EQ(result.input->hopper_propellant->stamp.nanoseconds_since_epoch,
            10'000'000'000LL);
  EXPECT_EQ(result.input->hopper_propellant->platform_id, "test-hopper");
  EXPECT_EQ(result.input->hopper_propellant->capability_version, "test-v1");
  EXPECT_DOUBLE_EQ(result.input->hopper_propellant->total_mass_kg, 20.0);
  EXPECT_DOUBLE_EQ(
      result.input->hopper_propellant->remaining_usable_fuel_mass_kg, 0.2);
}

TEST(SnapshotBuilder, GroundPlatformsDoNotRequirePropellantState) {
  const auto result = MakeBuilder(ValidStore()).Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(result.ok());
  EXPECT_FALSE(result.input->hopper_propellant.has_value());
}

TEST(SnapshotBuilder, RejectsMissingStaleFutureAndSkewedHopperPropellant) {
  const auto missing = MakeHopperBuilder(ValidStore()).Freeze(
      ValidHopperGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(missing.error.has_value());
  EXPECT_EQ(missing.error->code,
            SnapshotErrorCode::kInvalidHopperPropellant);
  EXPECT_EQ(missing.error->reason_code,
            "HOPPER_PROPELLANT_STATE_INVALID");

  auto stale_store = ValidStore();
  stale_store->UpdateHopperPropellantState(
      ValidPropellant(9'500'000'000LL));
  const auto stale = MakeHopperBuilder(stale_store).Freeze(
      ValidHopperGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(stale.error.has_value());
  EXPECT_EQ(stale.error->code,
            SnapshotErrorCode::kStaleHopperPropellant);
  EXPECT_EQ(stale.error->reason_code,
            "HOPPER_PROPELLANT_STATE_STALE");

  auto future_store = ValidStore();
  future_store->UpdateHopperPropellantState(
      ValidPropellant(10'200'000'000LL));
  const auto future = MakeHopperBuilder(future_store).Freeze(
      ValidHopperGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(future.error.has_value());
  EXPECT_EQ(future.error->code,
            SnapshotErrorCode::kInvalidHopperPropellant);

  auto skewed_store = ValidStore();
  skewed_store->UpdateHopperPropellantState(
      ValidPropellant(9'750'000'000LL));
  const auto skewed = MakeHopperBuilder(skewed_store).Freeze(
      ValidHopperGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(skewed.error.has_value());
  EXPECT_EQ(skewed.error->code,
            SnapshotErrorCode::kStaleHopperPropellant);
}

TEST(SnapshotBuilder, RejectsInvalidHopperPropellantIdentityFrameAndMass) {
  const auto evaluate = [](auto mutate) {
    auto store = ValidStore();
    auto message = ValidPropellant();
    mutate(message);
    store->UpdateHopperPropellantState(message);
    return MakeHopperBuilder(store).Freeze(
        ValidHopperGoal(), rclcpp::Time{10'100'000'000LL});
  };
  for (const auto& result : {
           evaluate([](auto& value) {
             value.header.stamp.sec = 0;
             value.header.stamp.nanosec = 0U;
           }),
           evaluate([](auto& value) { value.header.frame_id = "odom"; }),
           evaluate([](auto& value) { value.platform_id = "other"; }),
           evaluate([](auto& value) { value.capability_version = "other"; }),
           evaluate([](auto& value) { value.total_mass_kg = 0.0; }),
           evaluate([](auto& value) {
             value.remaining_usable_fuel_mass_kg = value.total_mass_kg;
           }),
       }) {
    ASSERT_TRUE(result.error.has_value());
    EXPECT_EQ(result.error->code,
              SnapshotErrorCode::kInvalidHopperPropellant);
    EXPECT_EQ(result.error->reason_code,
              "HOPPER_PROPELLANT_STATE_INVALID");
  }
}

TEST(SnapshotBuilder, RequiresExactPointGoalWithoutYawForHopper) {
  const auto evaluate = [](GoalRequest request) {
    auto store = ValidStore();
    store->UpdateHopperPropellantState(ValidPropellant());
    return MakeHopperBuilder(store).Freeze(
        request, rclcpp::Time{10'100'000'000LL});
  };

  auto tolerance = ValidHopperGoal();
  std::get<lunar::planning::PointGoal>(tolerance.goal.target).tolerance_m =
      0.01;
  auto yaw = ValidHopperGoal();
  yaw.goal.yaw_rad = 0.0;
  auto region = ValidHopperGoal();
  region.goal.target = lunar::planning::PlanarRegionGoal{
      .boundary_m = {{1.0, 0.0, 0.0},
                     {2.0, 0.0, 0.0},
                     {1.0, 1.0, 0.0}},
      .normal_tolerance_m = 0.0,
  };
  for (const auto& result :
       {evaluate(tolerance), evaluate(yaw), evaluate(region)}) {
    ASSERT_TRUE(result.error.has_value());
    EXPECT_EQ(result.error->code, SnapshotErrorCode::kInvalidGoal);
    EXPECT_EQ(result.error->reason_code, "HOPPER_GOAL_INVALID");
  }
}

TEST(SnapshotBuilder, KeepsContentGenerationsStableAndPropagatesUncertainty) {
  auto store = ValidStore();
  auto odometry = test::MakeOdometry();
  odometry.pose.covariance[0] = 0.04;
  odometry.pose.covariance[7] = 0.09;
  odometry.pose.covariance[14] = 0.01;
  odometry.twist.covariance[0] = 0.01;
  odometry.twist.covariance[7] = 0.16;
  odometry.twist.covariance[14] = 0.04;
  store->UpdateOdometry(odometry);
  auto builder = MakeBuilder(store);

  const auto first = builder.Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(first.ok());
  EXPECT_NEAR(first.input->position_uncertainty_m, 0.9, 1.0e-12);
  EXPECT_NEAR(first.input->velocity_uncertainty_mps, 1.2, 1.0e-12);

  auto repeated_global = test::MakeGridMap("map");
  repeated_global.header.stamp = test::Stamp(10'050'000'000LL);
  auto repeated_local = test::MakeGridMap("odom");
  repeated_local.header.stamp = test::Stamp(10'050'000'000LL);
  store->UpdateGlobalMap(repeated_global);
  store->UpdateLocalMap(repeated_local);
  const auto repeated = builder.Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(repeated.ok());
  EXPECT_EQ(
      repeated.input->global_map_generation,
      first.input->global_map_generation);
  EXPECT_EQ(
      repeated.input->local_map_generation,
      first.input->local_map_generation);

  auto changed_global = repeated_global;
  const auto obstacle = test::LayerIndex(changed_global, "obstacle");
  ASSERT_LT(obstacle, changed_global.data.size());
  changed_global.data[obstacle].data[0] = 1.0F;
  store->UpdateGlobalMap(changed_global);
  const auto changed = builder.Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(changed.ok());
  EXPECT_GT(
      changed.input->global_map_generation,
      repeated.input->global_map_generation);
  EXPECT_EQ(
      changed.input->local_map_generation,
      repeated.input->local_map_generation);

  store->ClearTransforms();
  auto transforms = test::MakeTransforms();
  transforms.transforms.front().transform.translation.x += 0.01;
  store->UpdateTransforms(transforms);
  const auto moved_tf = builder.Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(moved_tf.ok());
  EXPECT_GT(
      moved_tf.input->map_from_odom_generation,
      changed.input->map_from_odom_generation);
}

TEST(SnapshotBuilder, TransformsOdomPointPolygonAndYawIntoMap) {
  auto store = ValidStore();
  auto transforms = test::MakeTransforms();
  transforms.transforms[0].transform.rotation.w = std::sqrt(0.5);
  transforms.transforms[0].transform.rotation.z = std::sqrt(0.5);
  store->UpdateTransforms(transforms);

  GoalRequest point = ValidGoal();
  point.frame_id = "odom";
  point.goal.target = lunar::planning::PointGoal{
      .position_m = {1.0, 0.0, 0.0},
      .tolerance_m = 0.2,
  };
  point.goal.yaw_rad = 0.0;
  const SnapshotBuildResult point_result = MakeBuilder(store).Freeze(
      point, rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(point_result.ok())
      << (point_result.error ? point_result.error->reason_code : "");
  const auto& map_point = std::get<lunar::planning::PointGoal>(
      point_result.input->goal_map.target);
  EXPECT_NEAR(map_point.position_m.x, 10.0, 1.0e-9);
  EXPECT_NEAR(map_point.position_m.y, 1.0, 1.0e-9);
  ASSERT_TRUE(point_result.input->goal_map.yaw_rad.has_value());
  EXPECT_NEAR(
      *point_result.input->goal_map.yaw_rad,
      std::numbers::pi / 2.0, 1.0e-9);

  GoalRequest polygon = point;
  polygon.goal.target = lunar::planning::PlanarRegionGoal{
      .boundary_m = {
          {0.0, 0.0, 0.0},
          {1.0, 0.0, 0.0},
          {0.0, 1.0, 0.0},
      },
      .normal_tolerance_m = 0.1,
  };
  const SnapshotBuildResult polygon_result = MakeBuilder(store).Freeze(
      polygon, rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(polygon_result.ok());
  const auto& map_polygon = std::get<lunar::planning::PlanarRegionGoal>(
      polygon_result.input->goal_map.target);
  ASSERT_EQ(map_polygon.boundary_m.size(), 3U);
  EXPECT_NEAR(map_polygon.boundary_m[0].x, 10.0, 1.0e-9);
  EXPECT_NEAR(map_polygon.boundary_m[0].y, 0.0, 1.0e-9);
  EXPECT_NEAR(map_polygon.boundary_m[1].x, 10.0, 1.0e-9);
  EXPECT_NEAR(map_polygon.boundary_m[1].y, 1.0, 1.0e-9);
  EXPECT_NEAR(map_polygon.boundary_m[2].x, 9.0, 1.0e-9);
  EXPECT_NEAR(map_polygon.boundary_m[2].y, 0.0, 1.0e-9);
}

TEST(SnapshotBuilder, RejectsInvalidLocalAndGlobalPyramidLevels) {
  auto local_wrong = ValidStore();
  auto local_map = test::MakeGridMap("odom");
  SetResolution(local_map, 2.0);
  local_wrong->UpdateLocalMap(local_map);
  const SnapshotBuildResult local_result = MakeBuilder(local_wrong).Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(local_result.error.has_value());
  EXPECT_EQ(local_result.error->code, SnapshotErrorCode::kInvalidLocalMap);
  EXPECT_EQ(local_result.error->reason_code, "LOCAL_MAP_LEVEL_INVALID");

  auto nondyadic = ValidStore();
  auto global_map = test::MakeGridMap("map");
  SetResolution(global_map, 1.5);
  nondyadic->UpdateGlobalMap(global_map);
  const SnapshotBuildResult nondyadic_result = MakeBuilder(nondyadic).Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(nondyadic_result.error.has_value());
  EXPECT_EQ(
      nondyadic_result.error->code, SnapshotErrorCode::kInvalidGlobalMap);
  EXPECT_EQ(
      nondyadic_result.error->reason_code, "GLOBAL_MAP_LEVEL_INVALID");

  auto overcoarse = ValidStore();
  global_map = test::MakeGridMap("map");
  SetResolution(global_map, 2.0);
  overcoarse->UpdateGlobalMap(global_map);
  const SnapshotBuildResult overcoarse_result = MakeBuilder(overcoarse).Freeze(
      ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(overcoarse_result.error.has_value());
  EXPECT_EQ(
      overcoarse_result.error->reason_code, "GLOBAL_MAP_LEVEL_INVALID");
}

TEST(SnapshotBuilder, EnforcesCellAxisAndLevelFourResourceBounds) {
  lunar::planning::PlannerConfig cell_limited;
  cell_limited.global_map.maximum_cells = 2U;
  const SnapshotBuildResult finer_than_required =
      MakeBuilder(ValidStore(), ValidPolicy(), cell_limited)
          .Freeze(ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(finer_than_required.error.has_value());
  EXPECT_EQ(
      finer_than_required.error->reason_code, "GLOBAL_MAP_LEVEL_INVALID");

  lunar::planning::PlannerConfig axis_limited;
  axis_limited.global_map.maximum_axis_cells = 2U;
  const SnapshotBuildResult axis_result =
      MakeBuilder(ValidStore(), ValidPolicy(), axis_limited)
          .Freeze(ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(axis_result.error.has_value());
  EXPECT_EQ(axis_result.error->reason_code, "GLOBAL_MAP_LEVEL_INVALID");

  auto too_large = ValidStore();
  auto global_map = test::MakeGridMap("map");
  SetResolution(global_map, 16.0);
  too_large->UpdateGlobalMap(global_map);
  lunar::planning::PlannerConfig exhausted;
  exhausted.global_map.maximum_cells = 1U;
  const SnapshotBuildResult exhausted_result =
      MakeBuilder(too_large, ValidPolicy(), exhausted)
          .Freeze(ValidGoal(), rclcpp::Time{10'100'000'000LL});
  ASSERT_TRUE(exhausted_result.error.has_value());
  EXPECT_EQ(
      exhausted_result.error->code, SnapshotErrorCode::kInvalidGlobalMap);
  EXPECT_EQ(
      exhausted_result.error->reason_code,
      "GLOBAL_MAP_SCALE_UNSUPPORTED");
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
