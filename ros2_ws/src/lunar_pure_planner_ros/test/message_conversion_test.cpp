#include "lunar_pure_planner_ros/message_conversion.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_planner_core/planner.hpp"

namespace lunar::pure_planner_ros {
namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using lunar::pure_planning::PlanningResult;
using lunar::pure_planning::PlanningStatus;
using namespace std::chrono_literals;

Action::Goal PointRequest(const std::string& frame = "map") {
  Action::Goal request;
  request.environment_mode = request.LUNAR_SURFACE;
  request.goal.header.frame_id = frame;
  request.goal.goal_id = "target";
  request.goal.goal_type = request.goal.POINT;
  request.goal.point.x = 3.0;
  request.goal.point.y = 4.0;
  request.goal.point.z = 100.0;
  request.goal.position_tolerance_m = 0.25;
  request.goal.has_yaw_constraint = true;
  request.goal.yaw_rad = 0.2;
  request.goal.yaw_tolerance_rad = 0.1;
  return request;
}

lunar::pure_planning::RigidTransform MapFromOdom(const double yaw = 0.0) {
  return {
      .parent_frame = "map",
      .child_frame = "odom",
      .translation_m = {10.0, 5.0, 7.0},
      .rotation = {.w = std::cos(yaw / 2.0), .x = 0.0, .y = 0.0,
                   .z = std::sin(yaw / 2.0)},
  };
}

lunar::pure_planning::MotionReference WheelReference(
    const lunar::pure_planning::PlatformType platform_type =
        lunar::pure_planning::PlatformType::kWheeled) {
  lunar::pure_planning::TrajectoryReference trajectory{
      .semantics = platform_type == lunar::pure_planning::PlatformType::kWheeled
          ? lunar::pure_planning::TrajectorySemantics::kWheeledBase
          : lunar::pure_planning::TrajectorySemantics::kLeggedBodyReference,
      .points = {{.time_from_start = 1500ms,
                  .pose = {.position_m = {1.0, 2.0, 3.0},
                           .orientation = {.w = 0.5, .x = 0.1, .y = 0.2, .z = 0.3}},
                  .velocity = {.linear_mps = {4.0, 5.0, 6.0},
                               .angular_radps = {0.4, 0.5, 0.6}}},
                 {.time_from_start = 3750ms,
                  .pose = {.position_m = {21.0, 22.0, 23.0},
                           .orientation = {.w = 0.91, .x = 0.81, .y = 0.71, .z = 0.61}},
                  .velocity = {.linear_mps = {24.0, 25.0, 26.0},
                               .angular_radps = {2.4, 2.5, 2.6}}}},
  };
  return {
      .plan_id = "plan-17",
      .platform_type = platform_type,
      .input_time = {2'250'000'000LL},
      .preview = {.poses_map = {
          {.position_m = {7.0, 8.0, 9.0},
           .orientation = {.w = 0.7, .x = 0.1, .y = 0.2, .z = 0.3}},
          {.position_m = {17.0, 18.0, 19.0},
           .orientation = {.w = 0.97, .x = 0.87, .y = 0.77, .z = 0.67}},
      }},
      .data = std::move(trajectory),
  };
}

lunar::pure_planning::MotionReference HopperReference() {
  lunar::pure_planning::HopReference hops{
      .segments = {{.segment_id = "hop-1",
                    .launch_pose = {.position_m = {1.0, 2.0, 3.0},
                                    .orientation = {.w = 0.9, .x = 0.1, .y = 0.2, .z = 0.3}},
                    .landing_region_boundary_m = {{4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}},
                    .flight_time = 2250ms,
                    .launch_velocity_mps = {10.0, 11.0, 12.0},
                    .flight_tube_radius_m = 0.7,
                    .nominal_landing_point_m = {13.0, 14.0, 15.0},
                    .required_delta_v_mps = 16.0,
                    .available_delta_v_mps = 17.0,
                    .capability_version = "hopper-v1",
                    .global_map_generation = 18U,
                    .local_map_generation = 19U},
                   {.segment_id = "hop-2",
                    .launch_pose = {.position_m = {31.0, 32.0, 33.0},
                                    .orientation = {.w = 0.19, .x = 0.29, .y = 0.39, .z = 0.49}},
                    .landing_region_boundary_m = {{34.0, 35.0, 36.0}, {37.0, 38.0, 39.0},
                                                  {40.0, 41.0, 42.0}},
                    .flight_time = 4250ms,
                    .launch_velocity_mps = {43.0, 44.0, 45.0},
                    .flight_tube_radius_m = 4.7,
                    .nominal_landing_point_m = {46.0, 47.0, 48.0},
                    .required_delta_v_mps = 49.0,
                    .available_delta_v_mps = 50.0,
                    .capability_version = "hopper-v2",
                    .global_map_generation = 51U,
                    .local_map_generation = 52U}},
  };
  return {
      .plan_id = "hop-plan",
      .platform_type = lunar::pure_planning::PlatformType::kHopper,
      .input_time = {3'000'000'000LL},
      .preview = {.poses_map = {
          {.position_m = {20.0, 21.0, 22.0},
           .orientation = {.w = 0.99, .x = 0.89, .y = 0.79, .z = 0.69}},
          {.position_m = {53.0, 54.0, 55.0},
           .orientation = {.w = 0.59, .x = 0.49, .y = 0.39, .z = 0.29}},
      }},
      .data = std::move(hops),
  };
}

PlanningResult Result(const PlanningStatus status,
                      std::optional<lunar::pure_planning::MotionReference> reference = std::nullopt) {
  return {.status = status,
          .reason_code = "IGNORED_BY_TYPED_STATUS",
          .reference = std::move(reference),
          .timing = {.total_elapsed = 2750ms},
          .expanded_states = 42U};
}

PlanningResult PlanWheelReferenceWithNonUnitMapFromOdom() {
  constexpr std::size_t kCellCount = 36U;
  lunar::pure_planning::PlanningRequest request{
      .request_id = "planner-to-ros",
      .environment_mode = lunar::pure_planning::EnvironmentMode::kLavaTube,
      .current_state = lunar::pure_planning::WheeledState{},
      .goal_map = lunar::pure_planning::GoalRegion{
          .goal_id = "goal",
          .target = lunar::pure_planning::PointGoal{
              .position_m = {9.0, -1.0, 0.0},
              .tolerance_m = 0.05,
          },
      },
      .world = lunar::pure_planning::MinimalWorldSnapshot{
          .global_map = std::nullopt,
          .local_map = lunar::pure_planning::GridMap{
              .frame_id = "odom",
              .width = 6U,
              .height = 6U,
              .resolution_m = 1.0,
              .layers = {
                  {"occupancy",
                   lunar::pure_planning::GridLayer{
                       .values = std::vector<float>(kCellCount, 0.0F)}},
                  {"elevation",
                   lunar::pure_planning::GridLayer{
                       .values = std::vector<float>(kCellCount, 0.0F)}},
              },
          },
          .map_from_odom = MapFromOdom(std::numbers::pi / 2.0),
      },
      .capability = lunar::pure_planning::WheeledCapability{},
  };
  request.world.map_from_odom.translation_m = {10.0, -3.0, 2.0};
  lunar::pure_planning::Planner planner(lunar::pure_planning::PlannerBackends{
      .global = {},
      .local = [](const lunar::pure_planning::PlanningRequest&,
                  const lunar::pure_planning::LocalGoalSet&,
                  lunar::pure_planning::SearchControl) {
        return lunar::pure_planning::LocalStageResult{
            .status = lunar::pure_planning::LocalPlanStatus::kSolved,
            .data = lunar::pure_planning::TrajectoryReference{
                .semantics = lunar::pure_planning::TrajectorySemantics::kWheeledBase,
                .points = {{
                    .time_from_start = 1250ms,
                    .pose = {.position_m = {1.0, 2.0, 3.0}},
                    .velocity = {
                        .linear_mps = {4.0, 5.0, 6.0},
                        .angular_radps = {0.4, 0.5, 0.6},
                    },
                }},
            },
        };
      },
  });
  return planner.Plan(request);
}

TEST(MessageConversion, UsesCoreRigidTransformForOdomPointAndYaw) {
  auto request = PointRequest("odom");
  request.goal.point.z = std::numeric_limits<double>::quiet_NaN();
  const auto converted = ConvertGoal(request, MapFromOdom(std::numbers::pi / 2.0));

  ASSERT_TRUE(converted.ok());
  const auto& point = std::get<lunar::pure_planning::PointGoal>(converted.goal->target);
  EXPECT_NEAR(point.position_m.x, 6.0, 1.0e-12);
  EXPECT_NEAR(point.position_m.y, 8.0, 1.0e-12);
  EXPECT_DOUBLE_EQ(point.position_m.z, 0.0);
  ASSERT_TRUE(converted.goal->yaw_rad.has_value());
  EXPECT_NEAR(*converted.goal->yaw_rad, 0.2 + std::numbers::pi / 2.0, 1.0e-12);
}

TEST(MessageConversion, IgnoresGoalZButRejectsNonfinitePlanarCoordinates) {
  for (const std::string frame : {"map", "odom"}) {
    for (const double z : {0.0, 100.0, std::numeric_limits<double>::quiet_NaN()}) {
      auto request = PointRequest(frame);
      request.goal.point.z = z;
      const auto converted = ConvertGoal(request, MapFromOdom());
      ASSERT_TRUE(converted.ok()) << frame;
      const auto& point = std::get<lunar::pure_planning::PointGoal>(converted.goal->target);
      EXPECT_DOUBLE_EQ(point.position_m.z, 0.0);
    }
  }
  auto request = PointRequest();
  request.goal.point.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(ConvertGoal(request, MapFromOdom()).reason_code, "INVALID_INPUT");
  request = PointRequest();
  request.goal.point.y = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(ConvertGoal(request, MapFromOdom()).reason_code, "INVALID_INPUT");
}

TEST(MessageConversion, RejectsUnsupportedGoalAndInvalidRigidTransform) {
  auto request = PointRequest();
  request.goal.goal_type = request.goal.PLANAR_REGION;
  EXPECT_EQ(ConvertGoal(request, MapFromOdom()).reason_code, "INVALID_INPUT");
  request = PointRequest("odom");
  auto transform = MapFromOdom();
  transform.rotation = {.w = 0.0, .x = 0.0, .y = 0.0, .z = 0.0};
  EXPECT_EQ(ConvertGoal(request, transform).reason_code, "INVALID_INPUT");
  request = PointRequest();
  request.environment_mode = 99U;
  EXPECT_EQ(ConvertGoal(request, MapFromOdom()).reason_code, "INVALID_INPUT");
}

TEST(MessageConversion, MapsTypedStatusExhaustivelyAndNeverPromotesFailureReference) {
  struct Case final {
    PlanningStatus status;
    std::uint8_t outcome;
    const char* reason;
  };
  const Case cases[] = {
      {PlanningStatus::kInvalidInput, Action::Result::INVALID_REQUEST, "INVALID_INPUT"},
      {PlanningStatus::kGoalOutsideLocalMap, Action::Result::GOAL_INFEASIBLE, "GOAL_OUTSIDE_LOCAL_MAP"},
      {PlanningStatus::kNoPath, Action::Result::GOAL_INFEASIBLE, "NO_PATH"},
      {PlanningStatus::kTimedOut, Action::Result::RESOURCE_EXHAUSTED, "TIMEOUT"},
      {PlanningStatus::kCanceled, Action::Result::CANCELED, "REQUEST_CANCELED"},
      {PlanningStatus::kPlannerError, Action::Result::NUMERICAL_FAILURE, "PLANNER_ERROR"},
  };
  for (const auto& expected : cases) {
    const auto result = ConvertResult(Result(expected.status, WheelReference()), 42U);
    EXPECT_EQ(result.planning_outcome, expected.outcome);
    EXPECT_EQ(result.execution_directive, Action::Result::NO_SAFE_REFERENCE);
    EXPECT_FALSE(result.has_reference);
    EXPECT_EQ(result.reason_code, expected.reason);
    EXPECT_EQ(result.mission_revision, 42U);
    EXPECT_NE(result.planning_outcome, Action::Result::STALE_INPUT);
    EXPECT_DOUBLE_EQ(result.diagnostics.elapsed_s, 2.75);
  }
}

TEST(MessageConversion, AddsWarningsAtExactTargetAndSlaMilestones) {
  struct Case final {
    std::chrono::milliseconds elapsed;
    std::vector<std::string> warnings;
  };
  for (const Case& test_case : {
           Case{999ms, {}},
           Case{1000ms, {"TARGET_MISSED"}},
           Case{1999ms, {"TARGET_MISSED"}},
           Case{2000ms, {"TARGET_MISSED", "PLANNING_SLA_MISSED"}},
           Case{2999ms, {"TARGET_MISSED", "PLANNING_SLA_MISSED"}},
           Case{3000ms, {"TARGET_MISSED", "PLANNING_SLA_MISSED"}},
         }) {
    auto source = Result(PlanningStatus::kSuccess, WheelReference());
    source.timing.total_elapsed = test_case.elapsed;

    const auto converted = ConvertResult(source, 1U);

    EXPECT_EQ(converted.diagnostics.warning_codes, test_case.warnings)
        << test_case.elapsed.count();
  }
}

TEST(MessageConversion, MapsAvailableSearchMetricsWithoutInventingACost) {
  auto with_cost = Result(PlanningStatus::kSuccess, WheelReference());
  with_cost.expanded_states = 37U;
  with_cost.best_cost = 23.5;
  const auto converted_with_cost = ConvertResult(with_cost, 1U);
  EXPECT_EQ(converted_with_cost.diagnostics.expanded_states, 37U);
  EXPECT_TRUE(converted_with_cost.diagnostics.has_best_cost);
  EXPECT_DOUBLE_EQ(converted_with_cost.diagnostics.best_cost, 23.5);

  auto without_cost = Result(PlanningStatus::kNoPath, std::nullopt);
  without_cost.expanded_states = 9U;
  const auto converted_without_cost = ConvertResult(without_cost, 1U);
  EXPECT_EQ(converted_without_cost.diagnostics.expanded_states, 9U);
  EXPECT_FALSE(converted_without_cost.diagnostics.has_best_cost);
  EXPECT_DOUBLE_EQ(converted_without_cost.diagnostics.best_cost, 0.0);
}

TEST(MessageConversion, ConvertsWheelAndLeggedReferencesLosslesslyInMapFrame) {
  for (const auto platform : {lunar::pure_planning::PlatformType::kWheeled,
                              lunar::pure_planning::PlatformType::kLegged}) {
    const auto result = ConvertResult(Result(PlanningStatus::kSuccess, WheelReference(platform)), 7U);
    ASSERT_TRUE(result.has_reference);
    EXPECT_EQ(result.planning_outcome, Action::Result::NEW_REFERENCE_AVAILABLE);
    EXPECT_EQ(result.execution_directive, Action::Result::ACTIVATE_NEW_REFERENCE);
    EXPECT_EQ(result.reference.plan_id, "plan-17");
    EXPECT_EQ(result.reference.platform_type,
              platform == lunar::pure_planning::PlatformType::kWheeled
                  ? result.reference.WHEELED : result.reference.LEGGED);
    EXPECT_EQ(result.reference.header.frame_id, "map");
    EXPECT_EQ(result.reference.header.stamp.sec, 2);
    EXPECT_EQ(result.reference.header.stamp.nanosec, 250'000'000U);
    EXPECT_EQ(result.reference.input_time.sec, 2);
    EXPECT_EQ(result.reference.input_time.nanosec, 250'000'000U);
    EXPECT_EQ(result.reference.path_preview.header.frame_id, "map");
    EXPECT_EQ(result.reference.path_preview.header.stamp.sec, 2);
    EXPECT_EQ(result.reference.path_preview.header.stamp.nanosec, 250'000'000U);
    ASSERT_EQ(result.reference.path_preview.poses.size(), 2U);
    const auto& preview_first = result.reference.path_preview.poses[0];
    EXPECT_EQ(preview_first.header.frame_id, "map");
    EXPECT_EQ(preview_first.header.stamp.sec, 2);
    EXPECT_EQ(preview_first.header.stamp.nanosec, 250'000'000U);
    EXPECT_DOUBLE_EQ(preview_first.pose.position.x, 7.0);
    EXPECT_DOUBLE_EQ(preview_first.pose.position.y, 8.0);
    EXPECT_DOUBLE_EQ(preview_first.pose.position.z, 9.0);
    EXPECT_DOUBLE_EQ(preview_first.pose.orientation.w, 0.7);
    EXPECT_DOUBLE_EQ(preview_first.pose.orientation.x, 0.1);
    EXPECT_DOUBLE_EQ(preview_first.pose.orientation.y, 0.2);
    EXPECT_DOUBLE_EQ(preview_first.pose.orientation.z, 0.3);
    const auto& preview_second = result.reference.path_preview.poses[1];
    EXPECT_EQ(preview_second.header.frame_id, "map");
    EXPECT_EQ(preview_second.header.stamp.sec, 2);
    EXPECT_EQ(preview_second.header.stamp.nanosec, 250'000'000U);
    EXPECT_DOUBLE_EQ(preview_second.pose.position.x, 17.0);
    EXPECT_DOUBLE_EQ(preview_second.pose.position.y, 18.0);
    EXPECT_DOUBLE_EQ(preview_second.pose.position.z, 19.0);
    EXPECT_DOUBLE_EQ(preview_second.pose.orientation.w, 0.97);
    EXPECT_DOUBLE_EQ(preview_second.pose.orientation.x, 0.87);
    EXPECT_DOUBLE_EQ(preview_second.pose.orientation.y, 0.77);
    EXPECT_DOUBLE_EQ(preview_second.pose.orientation.z, 0.67);

    EXPECT_EQ(result.reference.trajectory.header.frame_id, "map");
    EXPECT_EQ(result.reference.trajectory.header.stamp.sec, 2);
    EXPECT_EQ(result.reference.trajectory.header.stamp.nanosec, 250'000'000U);
    EXPECT_EQ(result.reference.trajectory.joint_names, (std::vector<std::string>{"base_link"}));
    ASSERT_EQ(result.reference.trajectory.points.size(), 2U);
    const auto& first = result.reference.trajectory.points[0];
    EXPECT_EQ(first.time_from_start.sec, 1);
    EXPECT_EQ(first.time_from_start.nanosec, 500'000'000U);
    ASSERT_EQ(first.transforms.size(), 1U);
    ASSERT_EQ(first.velocities.size(), 1U);
    EXPECT_TRUE(first.accelerations.empty());
    EXPECT_DOUBLE_EQ(first.transforms[0].translation.x, 1.0);
    EXPECT_DOUBLE_EQ(first.transforms[0].translation.y, 2.0);
    EXPECT_DOUBLE_EQ(first.transforms[0].translation.z, 3.0);
    EXPECT_DOUBLE_EQ(first.transforms[0].rotation.w, 0.5);
    EXPECT_DOUBLE_EQ(first.transforms[0].rotation.x, 0.1);
    EXPECT_DOUBLE_EQ(first.transforms[0].rotation.y, 0.2);
    EXPECT_DOUBLE_EQ(first.transforms[0].rotation.z, 0.3);
    EXPECT_DOUBLE_EQ(first.velocities[0].linear.x, 4.0);
    EXPECT_DOUBLE_EQ(first.velocities[0].linear.y, 5.0);
    EXPECT_DOUBLE_EQ(first.velocities[0].linear.z, 6.0);
    EXPECT_DOUBLE_EQ(first.velocities[0].angular.x, 0.4);
    EXPECT_DOUBLE_EQ(first.velocities[0].angular.y, 0.5);
    EXPECT_DOUBLE_EQ(first.velocities[0].angular.z, 0.6);
    const auto& second = result.reference.trajectory.points[1];
    EXPECT_EQ(second.time_from_start.sec, 3);
    EXPECT_EQ(second.time_from_start.nanosec, 750'000'000U);
    ASSERT_EQ(second.transforms.size(), 1U);
    ASSERT_EQ(second.velocities.size(), 1U);
    EXPECT_TRUE(second.accelerations.empty());
    EXPECT_DOUBLE_EQ(second.transforms[0].translation.x, 21.0);
    EXPECT_DOUBLE_EQ(second.transforms[0].translation.y, 22.0);
    EXPECT_DOUBLE_EQ(second.transforms[0].translation.z, 23.0);
    EXPECT_DOUBLE_EQ(second.transforms[0].rotation.w, 0.91);
    EXPECT_DOUBLE_EQ(second.transforms[0].rotation.x, 0.81);
    EXPECT_DOUBLE_EQ(second.transforms[0].rotation.y, 0.71);
    EXPECT_DOUBLE_EQ(second.transforms[0].rotation.z, 0.61);
    EXPECT_DOUBLE_EQ(second.velocities[0].linear.x, 24.0);
    EXPECT_DOUBLE_EQ(second.velocities[0].linear.y, 25.0);
    EXPECT_DOUBLE_EQ(second.velocities[0].linear.z, 26.0);
    EXPECT_DOUBLE_EQ(second.velocities[0].angular.x, 2.4);
    EXPECT_DOUBLE_EQ(second.velocities[0].angular.y, 2.5);
    EXPECT_DOUBLE_EQ(second.velocities[0].angular.z, 2.6);
    EXPECT_TRUE(result.reference.hops.empty());
  }
}

TEST(MessageConversion,
     PlannerCompositionAndRosConversionAgreeOnMapFrameAndNumbers) {
  const PlanningResult planned = PlanWheelReferenceWithNonUnitMapFromOdom();
  ASSERT_EQ(planned.status, PlanningStatus::kSuccess) << planned.reason_code;

  const auto result = ConvertResult(planned, 23U);

  ASSERT_TRUE(result.has_reference);
  EXPECT_EQ(result.reference.header.frame_id, "map");
  EXPECT_EQ(result.reference.trajectory.header.frame_id, "map");
  ASSERT_EQ(result.reference.trajectory.points.size(), 1U);
  const auto& point = result.reference.trajectory.points.front();
  ASSERT_EQ(point.transforms.size(), 1U);
  ASSERT_EQ(point.velocities.size(), 1U);
  EXPECT_NEAR(point.transforms.front().translation.x, 8.0, 1.0e-12);
  EXPECT_NEAR(point.transforms.front().translation.y, -2.0, 1.0e-12);
  EXPECT_NEAR(point.transforms.front().translation.z, 5.0, 1.0e-12);
  EXPECT_NEAR(point.transforms.front().rotation.w,
              std::sqrt(0.5), 1.0e-12);
  EXPECT_NEAR(point.transforms.front().rotation.z,
              std::sqrt(0.5), 1.0e-12);
  EXPECT_NEAR(point.velocities.front().linear.x, -5.0, 1.0e-12);
  EXPECT_NEAR(point.velocities.front().linear.y, 4.0, 1.0e-12);
  EXPECT_NEAR(point.velocities.front().linear.z, 6.0, 1.0e-12);
  EXPECT_NEAR(point.velocities.front().angular.x, -0.5, 1.0e-12);
  EXPECT_NEAR(point.velocities.front().angular.y, 0.4, 1.0e-12);
  EXPECT_NEAR(point.velocities.front().angular.z, 0.6, 1.0e-12);
  EXPECT_EQ(point.time_from_start.sec, 1);
  EXPECT_EQ(point.time_from_start.nanosec, 250'000'000U);
}

TEST(MessageConversion, ConvertsHopperReferenceLosslesslyInMapFrame) {
  const auto result = ConvertResult(Result(PlanningStatus::kSuccess, HopperReference()), 9U);
  ASSERT_TRUE(result.has_reference);
  EXPECT_EQ(result.reference.plan_id, "hop-plan");
  EXPECT_EQ(result.reference.platform_type, result.reference.HOPPER);
  EXPECT_EQ(result.reference.header.frame_id, "map");
  EXPECT_EQ(result.reference.header.stamp.sec, 3);
  EXPECT_EQ(result.reference.header.stamp.nanosec, 0U);
  EXPECT_EQ(result.reference.input_time.sec, 3);
  EXPECT_EQ(result.reference.input_time.nanosec, 0U);
  EXPECT_EQ(result.reference.path_preview.header.frame_id, "map");
  EXPECT_EQ(result.reference.path_preview.header.stamp.sec, 3);
  EXPECT_EQ(result.reference.path_preview.header.stamp.nanosec, 0U);
  ASSERT_EQ(result.reference.path_preview.poses.size(), 2U);
  const auto& preview_first = result.reference.path_preview.poses[0];
  EXPECT_EQ(preview_first.header.frame_id, "map");
  EXPECT_EQ(preview_first.header.stamp.sec, 3);
  EXPECT_EQ(preview_first.header.stamp.nanosec, 0U);
  EXPECT_DOUBLE_EQ(preview_first.pose.position.x, 20.0);
  EXPECT_DOUBLE_EQ(preview_first.pose.position.y, 21.0);
  EXPECT_DOUBLE_EQ(preview_first.pose.position.z, 22.0);
  EXPECT_DOUBLE_EQ(preview_first.pose.orientation.w, 0.99);
  EXPECT_DOUBLE_EQ(preview_first.pose.orientation.x, 0.89);
  EXPECT_DOUBLE_EQ(preview_first.pose.orientation.y, 0.79);
  EXPECT_DOUBLE_EQ(preview_first.pose.orientation.z, 0.69);
  const auto& preview_second = result.reference.path_preview.poses[1];
  EXPECT_EQ(preview_second.header.frame_id, "map");
  EXPECT_EQ(preview_second.header.stamp.sec, 3);
  EXPECT_EQ(preview_second.header.stamp.nanosec, 0U);
  EXPECT_DOUBLE_EQ(preview_second.pose.position.x, 53.0);
  EXPECT_DOUBLE_EQ(preview_second.pose.position.y, 54.0);
  EXPECT_DOUBLE_EQ(preview_second.pose.position.z, 55.0);
  EXPECT_DOUBLE_EQ(preview_second.pose.orientation.w, 0.59);
  EXPECT_DOUBLE_EQ(preview_second.pose.orientation.x, 0.49);
  EXPECT_DOUBLE_EQ(preview_second.pose.orientation.y, 0.39);
  EXPECT_DOUBLE_EQ(preview_second.pose.orientation.z, 0.29);
  EXPECT_TRUE(result.reference.trajectory.points.empty());
  EXPECT_TRUE(result.reference.trajectory.joint_names.empty());

  ASSERT_EQ(result.reference.hops.size(), 2U);
  const auto& first = result.reference.hops[0];
  EXPECT_EQ(first.header.frame_id, "map");
  EXPECT_EQ(first.header.stamp.sec, 3);
  EXPECT_EQ(first.header.stamp.nanosec, 0U);
  EXPECT_EQ(first.segment_id, "hop-1");
  EXPECT_DOUBLE_EQ(first.launch_pose.position.x, 1.0);
  EXPECT_DOUBLE_EQ(first.launch_pose.position.y, 2.0);
  EXPECT_DOUBLE_EQ(first.launch_pose.position.z, 3.0);
  EXPECT_DOUBLE_EQ(first.launch_pose.orientation.w, 0.9);
  EXPECT_DOUBLE_EQ(first.launch_pose.orientation.x, 0.1);
  EXPECT_DOUBLE_EQ(first.launch_pose.orientation.y, 0.2);
  EXPECT_DOUBLE_EQ(first.launch_pose.orientation.z, 0.3);
  EXPECT_EQ(first.flight_time.sec, 2);
  EXPECT_EQ(first.flight_time.nanosec, 250'000'000U);
  EXPECT_DOUBLE_EQ(first.launch_velocity.x, 10.0);
  EXPECT_DOUBLE_EQ(first.launch_velocity.y, 11.0);
  EXPECT_DOUBLE_EQ(first.launch_velocity.z, 12.0);
  EXPECT_DOUBLE_EQ(first.flight_tube_radius_m, 0.7);
  EXPECT_DOUBLE_EQ(first.nominal_landing_point.x, 13.0);
  EXPECT_DOUBLE_EQ(first.nominal_landing_point.y, 14.0);
  EXPECT_DOUBLE_EQ(first.nominal_landing_point.z, 15.0);
  EXPECT_DOUBLE_EQ(first.required_delta_v_mps, 16.0);
  EXPECT_DOUBLE_EQ(first.available_delta_v_mps, 17.0);
  EXPECT_EQ(first.capability_version, "hopper-v1");
  EXPECT_EQ(first.global_map_generation, 18U);
  EXPECT_EQ(first.local_map_generation, 19U);
  ASSERT_EQ(first.landing_region.points.size(), 2U);
  EXPECT_FLOAT_EQ(first.landing_region.points[0].x, 4.0F);
  EXPECT_FLOAT_EQ(first.landing_region.points[0].y, 5.0F);
  EXPECT_FLOAT_EQ(first.landing_region.points[0].z, 6.0F);
  EXPECT_FLOAT_EQ(first.landing_region.points[1].x, 7.0F);
  EXPECT_FLOAT_EQ(first.landing_region.points[1].y, 8.0F);
  EXPECT_FLOAT_EQ(first.landing_region.points[1].z, 9.0F);
  const auto& second = result.reference.hops[1];
  EXPECT_EQ(second.header.frame_id, "map");
  EXPECT_EQ(second.header.stamp.sec, 3);
  EXPECT_EQ(second.header.stamp.nanosec, 0U);
  EXPECT_EQ(second.segment_id, "hop-2");
  EXPECT_DOUBLE_EQ(second.launch_pose.position.x, 31.0);
  EXPECT_DOUBLE_EQ(second.launch_pose.position.y, 32.0);
  EXPECT_DOUBLE_EQ(second.launch_pose.position.z, 33.0);
  EXPECT_DOUBLE_EQ(second.launch_pose.orientation.w, 0.19);
  EXPECT_DOUBLE_EQ(second.launch_pose.orientation.x, 0.29);
  EXPECT_DOUBLE_EQ(second.launch_pose.orientation.y, 0.39);
  EXPECT_DOUBLE_EQ(second.launch_pose.orientation.z, 0.49);
  EXPECT_EQ(second.flight_time.sec, 4);
  EXPECT_EQ(second.flight_time.nanosec, 250'000'000U);
  EXPECT_DOUBLE_EQ(second.launch_velocity.x, 43.0);
  EXPECT_DOUBLE_EQ(second.launch_velocity.y, 44.0);
  EXPECT_DOUBLE_EQ(second.launch_velocity.z, 45.0);
  EXPECT_DOUBLE_EQ(second.flight_tube_radius_m, 4.7);
  EXPECT_DOUBLE_EQ(second.nominal_landing_point.x, 46.0);
  EXPECT_DOUBLE_EQ(second.nominal_landing_point.y, 47.0);
  EXPECT_DOUBLE_EQ(second.nominal_landing_point.z, 48.0);
  EXPECT_DOUBLE_EQ(second.required_delta_v_mps, 49.0);
  EXPECT_DOUBLE_EQ(second.available_delta_v_mps, 50.0);
  EXPECT_EQ(second.capability_version, "hopper-v2");
  EXPECT_EQ(second.global_map_generation, 51U);
  EXPECT_EQ(second.local_map_generation, 52U);
  ASSERT_EQ(second.landing_region.points.size(), 3U);
  EXPECT_FLOAT_EQ(second.landing_region.points[0].x, 34.0F);
  EXPECT_FLOAT_EQ(second.landing_region.points[0].y, 35.0F);
  EXPECT_FLOAT_EQ(second.landing_region.points[0].z, 36.0F);
  EXPECT_FLOAT_EQ(second.landing_region.points[1].x, 37.0F);
  EXPECT_FLOAT_EQ(second.landing_region.points[1].y, 38.0F);
  EXPECT_FLOAT_EQ(second.landing_region.points[1].z, 39.0F);
  EXPECT_FLOAT_EQ(second.landing_region.points[2].x, 40.0F);
  EXPECT_FLOAT_EQ(second.landing_region.points[2].y, 41.0F);
  EXPECT_FLOAT_EQ(second.landing_region.points[2].z, 42.0F);
}

}  // namespace
}  // namespace lunar::pure_planner_ros
