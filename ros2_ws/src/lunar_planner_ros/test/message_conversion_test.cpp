#include "lunar_planner_ros/message_conversion.hpp"

#include <chrono>
#include <limits>
#include <utility>
#include <variant>

#include <gtest/gtest.h>

namespace lunar::planning::ros {
namespace {

using namespace std::chrono_literals;

lunar_planning_msgs::msg::GoalRegion PointGoalMessage() {
  lunar_planning_msgs::msg::GoalRegion message;
  message.header.frame_id = "map";
  message.header.stamp.sec = 10;
  message.goal_id = "site-a";
  message.goal_type = message.POINT;
  message.point.x = 4.0;
  message.point.y = -2.0;
  message.position_tolerance_m = 0.25;
  message.has_yaw_constraint = true;
  message.yaw_rad = 0.5;
  message.yaw_tolerance_rad = 0.1;
  return message;
}

PlannerResultContext Context() {
  return PlannerResultContext{
      .global_map_stamp = {10'000'000'000LL},
      .local_map_stamp = {10'100'000'000LL},
      .state_stamp = {10'200'000'000LL},
      .mission_revision = 7U,
      .capability_version = "hopper-capability-v1",
      .global_map_generation = 31U,
      .local_map_generation = 37U,
      .preview_frame = "map",
      .execution_frame = "odom",
      .execution_gravity_mps2 = {0.0, 0.0, -1.62},
  };
}

lunar::planning::PlannerOutput WheelOutput() {
  lunar::planning::TrajectoryReference trajectory{
      .semantics = lunar::planning::TrajectorySemantics::kWheeledBase,
      .points = {
          lunar::planning::TrajectoryPoint{
              .time_from_start = 100ms,
              .pose = {
                  .position_m = {1.0, 2.0, 0.0},
                  .orientation = {},
              },
              .velocity = {
                  .linear_mps = {0.5, 0.0, 0.0},
                  .angular_radps = {0.0, 0.0, 0.1},
              },
          },
      },
  };
  return lunar::planning::PlannerOutput{
      .outcome = lunar::planning::PlanningOutcome::kNewReferenceAvailable,
      .directive = lunar::planning::ExecutionDirective::kActivateNewReference,
      .reason_code = "PLAN_FOUND",
      .reference = lunar::planning::MotionReference{
          .plan_id = "wheel-plan",
          .platform_type = lunar::planning::PlatformType::kWheeled,
          .input_time = {10'200'000'000LL},
          .preview = lunar::planning::GlobalRoutePreview{
              .poses_map = {
                  lunar::planning::Pose3{
                      .position_m = {10.0, 20.0, 0.0},
                      .orientation = {},
                  },
                  lunar::planning::Pose3{
                      .position_m = {30.0, 40.0, 0.0},
                      .orientation = {},
                  },
              },
          },
          .data = std::move(trajectory),
      },
      .diagnostics = {
          .planner_name = "cpp_v3",
          .elapsed = 20ms,
          .expanded_states = 42U,
          .best_cost = 3.5,
          .warning_codes = {"LOW_MARGIN"},
          .local_trajectory =
              lunar::planning::LocalTrajectoryDiagnostics{
                  .trajectory_mode =
                      lunar::planning::TrajectoryMode::kOptimized,
                  .start_anchor_error_m = 0.0,
                  .endpoint_error_m = 0.05,
                  .maximum_curvature_per_m = 0.25,
                  .collision_validation =
                      lunar::planning::CollisionValidation::kCertified,
                  .smoothing_elapsed_s = 0.002,
                  .landing_field_elapsed_s = 0.0,
              },
      },
  };
}

TEST(MessageConversion, ConvertsPointAndPlanarGoalsWithoutInventingData) {
  const auto point = ConvertGoalMessage(PointGoalMessage());
  ASSERT_TRUE(point.ok()) << point.reason_code;
  ASSERT_TRUE(std::holds_alternative<lunar::planning::PointGoal>(
      point.goal->target));
  const auto& target = std::get<lunar::planning::PointGoal>(point.goal->target);
  EXPECT_DOUBLE_EQ(target.position_m.x, 4.0);
  EXPECT_DOUBLE_EQ(target.tolerance_m, 0.25);
  EXPECT_EQ(point.goal->yaw_rad, 0.5);

  auto polygon_message = PointGoalMessage();
  polygon_message.goal_type = polygon_message.PLANAR_REGION;
  polygon_message.planar_region.points.resize(3U);
  polygon_message.planar_region.points[1].x = 1.0F;
  polygon_message.planar_region.points[2].y = 1.0F;
  const auto polygon = ConvertGoalMessage(polygon_message);
  ASSERT_TRUE(polygon.ok()) << polygon.reason_code;
  EXPECT_EQ(
      std::get<lunar::planning::PlanarRegionGoal>(polygon.goal->target)
          .boundary_m.size(),
      3U);
}

TEST(MessageConversion, RejectsMalformedGoalMessages) {
  auto message = PointGoalMessage();
  message.goal_id.clear();
  EXPECT_EQ(ConvertGoalMessage(message).reason_code, "GOAL_ID_EMPTY");
  message = PointGoalMessage();
  message.goal_type = 99U;
  EXPECT_EQ(ConvertGoalMessage(message).reason_code, "GOAL_TYPE_INVALID");
  message = PointGoalMessage();
  message.position_tolerance_m = 0.0;
  EXPECT_TRUE(ConvertGoalMessage(message).ok());
}

TEST(MessageConversion, EmitsDefaultReferenceWhenPlannerHasNone) {
  lunar::planning::PlannerOutput output{
      .outcome = lunar::planning::PlanningOutcome::kNoKnownSafeRoute,
      .directive = lunar::planning::ExecutionDirective::kHoldPosition,
      .reason_code = "NO_ROUTE",
      .reference = std::nullopt,
      .diagnostics = {},
  };
  const auto converted = ConvertPlannerOutput(output, Context());
  ASSERT_TRUE(converted.ok()) << converted.reason_code;
  EXPECT_FALSE(converted.result->has_reference);
  EXPECT_EQ(
      converted.result->reference,
      lunar_planning_msgs::msg::MotionReference{});
  EXPECT_EQ(converted.result->mission_revision, 7U);
}

TEST(MessageConversion, ConvertsMapPreviewAndOdomWheelExecutionSeparately) {
  const auto converted = ConvertPlannerOutput(WheelOutput(), Context());
  ASSERT_TRUE(converted.ok()) << converted.reason_code;
  ASSERT_TRUE(converted.result->has_reference);
  EXPECT_EQ(converted.result->reference.plan_id, "wheel-plan");
  EXPECT_EQ(converted.result->reference.header.frame_id, "map");
  EXPECT_EQ(converted.result->reference.path_preview.header.frame_id, "map");
  EXPECT_EQ(converted.result->reference.trajectory.header.frame_id, "odom");
  EXPECT_EQ(converted.result->reference.input_time.sec, 10);
  EXPECT_EQ(converted.result->reference.input_time.nanosec, 200'000'000U);
  EXPECT_EQ(
      converted.result->reference.platform_type,
      lunar_planning_msgs::msg::MotionReference::WHEELED);
  ASSERT_EQ(converted.result->reference.path_preview.poses.size(), 2U);
  EXPECT_DOUBLE_EQ(
      converted.result->reference.path_preview.poses[0].pose.position.x,
      10.0);
  EXPECT_DOUBLE_EQ(
      converted.result->reference.path_preview.poses[1].pose.position.x,
      30.0);
  for (const auto& pose : converted.result->reference.path_preview.poses) {
    EXPECT_EQ(pose.header.frame_id, "map");
  }
  ASSERT_EQ(converted.result->reference.trajectory.points.size(), 1U);
  EXPECT_DOUBLE_EQ(
      converted.result->reference.trajectory.points[0]
          .transforms[0].translation.x,
      1.0);
  EXPECT_EQ(converted.result->diagnostics.expanded_states, 42U);
  EXPECT_TRUE(converted.result->diagnostics.has_best_cost);
  EXPECT_DOUBLE_EQ(converted.result->diagnostics.best_cost, 3.5);
}

TEST(MessageConversion, ConvertsHopperSegmentsWithExecutableTiming) {
  lunar::planning::HopReference hops{
      .segments = {
          lunar::planning::HopSegment{
              .segment_id = "hop-a",
              .launch_pose = {},
              .landing_region_boundary_m = {
                  {1.0, -1.0, 0.0}, {3.0, -1.0, 0.0},
                  {3.0, 1.0, 0.0}, {1.0, 1.0, 0.0}},
              .flight_time = 2s,
              .launch_velocity_mps = {1.0, 0.0, 2.0},
              .flight_tube_radius_m = 0.2,
              .nominal_landing_point_m = {2.0, 0.0, 0.76},
              .required_delta_v_mps = 7.0,
              .available_delta_v_mps = 8.0,
              .capability_version = "hopper-capability-v1",
              .global_map_generation = 31U,
              .local_map_generation = 37U,
          },
      },
  };
  auto output = WheelOutput();
  output.reference = lunar::planning::MotionReference{
      .plan_id = "hop-plan",
      .platform_type = lunar::planning::PlatformType::kHopper,
      .input_time = {10'200'000'000LL},
      .preview = lunar::planning::GlobalRoutePreview{
          .poses_map = {
              lunar::planning::Pose3{
                  .position_m = {0.0, 0.0, 0.0},
                  .orientation = {},
              },
              lunar::planning::Pose3{
                  .position_m = {2.0, 0.0, 0.0},
                  .orientation = {},
              },
              lunar::planning::Pose3{
                  .position_m = {4.0, 0.0, 0.0},
                  .orientation = {},
              },
          },
      },
      .data = std::move(hops),
  };
  const auto converted = ConvertPlannerOutput(output, Context());
  ASSERT_TRUE(converted.ok()) << converted.reason_code;
  ASSERT_EQ(converted.result->reference.hops.size(), 1U);
  EXPECT_EQ(converted.result->reference.hops.front().segment_id, "hop-a");
  EXPECT_EQ(converted.result->reference.hops.front().header.frame_id, "odom");
  EXPECT_EQ(converted.result->reference.hops.front().flight_time.sec, 2);
  EXPECT_DOUBLE_EQ(
      converted.result->reference.hops.front().nominal_landing_point.x, 2.0);
  EXPECT_DOUBLE_EQ(
      converted.result->reference.hops.front().nominal_landing_point.z, 0.76);
  EXPECT_DOUBLE_EQ(
      converted.result->reference.hops.front().required_delta_v_mps, 7.0);
  EXPECT_DOUBLE_EQ(
      converted.result->reference.hops.front().available_delta_v_mps, 8.0);
  EXPECT_EQ(
      converted.result->reference.hops.front().capability_version,
      "hopper-capability-v1");
  EXPECT_EQ(
      converted.result->reference.hops.front().global_map_generation, 31U);
  EXPECT_EQ(
      converted.result->reference.hops.front().local_map_generation, 37U);
  EXPECT_EQ(converted.result->reference.header.frame_id, "map");
  EXPECT_EQ(converted.result->reference.path_preview.header.frame_id, "map");
  ASSERT_EQ(converted.result->reference.path_preview.poses.size(), 3U);
  EXPECT_DOUBLE_EQ(
      converted.result->reference.path_preview.poses.back().pose.position.x,
      4.0);

  auto multiple_hops = output;
  auto& segments = std::get<lunar::planning::HopReference>(
      multiple_hops.reference->data).segments;
  segments.push_back(segments.front());
  segments.back().segment_id = "hop-b";
  EXPECT_EQ(
      ConvertPlannerOutput(multiple_hops, Context()).reason_code,
      "REFERENCE_HOP_AUTHORIZATION_INVALID");
}

TEST(
    MessageConversion,
    AcceptsHopperBallisticsUsingConfiguredExecutionFrameGravity) {
  lunar::planning::HopReference hops{
      .segments = {
          lunar::planning::HopSegment{
              .segment_id = "hop-low-gravity",
              .launch_pose = {},
              .landing_region_boundary_m = {
                  {1.0, -1.0, 0.0}, {3.0, -1.0, 0.0},
                  {3.0, 1.0, 0.0}, {1.0, 1.0, 0.0}},
              .flight_time = 2s,
              .launch_velocity_mps = {1.0, 0.0, 2.0},
              .flight_tube_radius_m = 0.2,
              .nominal_landing_point_m = {2.0, 0.0, 2.38},
              .required_delta_v_mps = 7.0,
              .available_delta_v_mps = 8.0,
              .capability_version = "hopper-capability-v1",
              .global_map_generation = 31U,
              .local_map_generation = 37U,
          },
      },
  };
  auto output = WheelOutput();
  output.reference = lunar::planning::MotionReference{
      .plan_id = "low-gravity-hop-plan",
      .platform_type = lunar::planning::PlatformType::kHopper,
      .input_time = {10'200'000'000LL},
      .preview = lunar::planning::GlobalRoutePreview{
          .poses_map = {
              lunar::planning::Pose3{},
              lunar::planning::Pose3{.position_m = {2.0, 0.0, 2.38}},
          },
      },
      .data = std::move(hops),
  };
  auto context = Context();
  context.execution_gravity_mps2 = {0.0, 0.0, -0.81};

  const auto converted = ConvertPlannerOutput(output, context);

  ASSERT_TRUE(converted.ok()) << converted.reason_code;
  ASSERT_EQ(converted.result->reference.hops.size(), 1U);
  EXPECT_DOUBLE_EQ(
      converted.result->reference.hops.front().nominal_landing_point.z, 2.38);
}

TEST(MessageConversion, RejectsEmptySingleHopAuthorization) {
  auto output = WheelOutput();
  output.reference = lunar::planning::MotionReference{
      .plan_id = "hop-plan",
      .platform_type = lunar::planning::PlatformType::kHopper,
      .input_time = {10'200'000'000LL},
      .preview = lunar::planning::GlobalRoutePreview{
          .poses_map = {
              lunar::planning::Pose3{},
              lunar::planning::Pose3{.position_m = {2.0, 0.0, 0.76}},
          },
      },
      .data = lunar::planning::HopReference{},
  };
  EXPECT_EQ(
      ConvertPlannerOutput(output, Context()).reason_code,
      "REFERENCE_HOPS_EMPTY");
}

TEST(MessageConversion, RejectsHopEvidenceAndSnapshotDrift) {
  lunar::planning::HopSegment valid{
      .segment_id = "hop-a",
      .launch_pose = {},
      .landing_region_boundary_m = {
          {1.0, -1.0, 0.0}, {3.0, -1.0, 0.0},
          {3.0, 1.0, 0.0}, {1.0, 1.0, 0.0}},
      .flight_time = 2s,
      .launch_velocity_mps = {1.0, 0.0, 2.0},
      .flight_tube_radius_m = 0.2,
      .nominal_landing_point_m = {2.0, 0.0, 0.76},
      .required_delta_v_mps = 7.0,
      .available_delta_v_mps = 8.0,
      .capability_version = "hopper-capability-v1",
      .global_map_generation = 31U,
      .local_map_generation = 37U,
  };
  const auto make_output = [&](lunar::planning::HopSegment segment) {
    auto output = WheelOutput();
    output.reference = lunar::planning::MotionReference{
        .plan_id = "hop-plan",
        .platform_type = lunar::planning::PlatformType::kHopper,
        .input_time = {10'200'000'000LL},
        .preview = lunar::planning::GlobalRoutePreview{
            .poses_map = {
                lunar::planning::Pose3{},
                lunar::planning::Pose3{.position_m = {2.0, 0.0, 0.76}},
            },
        },
        .data = lunar::planning::HopReference{
            .segments = {std::move(segment)},
        },
    };
    return output;
  };

  auto invalid = valid;
  invalid.available_delta_v_mps = 6.9;
  EXPECT_EQ(
      ConvertPlannerOutput(make_output(invalid), Context()).reason_code,
      "REFERENCE_HOP_EVIDENCE_INVALID");

  invalid = valid;
  invalid.capability_version = "stale";
  EXPECT_EQ(
      ConvertPlannerOutput(make_output(invalid), Context()).reason_code,
      "REFERENCE_HOP_SNAPSHOT_MISMATCH");

  invalid = valid;
  invalid.global_map_generation = 30U;
  EXPECT_EQ(
      ConvertPlannerOutput(make_output(invalid), Context()).reason_code,
      "REFERENCE_HOP_SNAPSHOT_MISMATCH");

  invalid = valid;
  invalid.nominal_landing_point_m.x += 0.01;
  EXPECT_EQ(
      ConvertPlannerOutput(make_output(invalid), Context()).reason_code,
      "REFERENCE_HOP_BALLISTIC_INCONSISTENT");
}

TEST(MessageConversion, RejectsResultInvariantViolations) {
  auto output = WheelOutput();
  output.reference->plan_id.clear();
  EXPECT_EQ(
      ConvertPlannerOutput(output, Context()).reason_code,
      "REFERENCE_PLAN_ID_EMPTY");

  output = WheelOutput();
  output.reference->platform_type = lunar::planning::PlatformType::kHopper;
  EXPECT_EQ(
      ConvertPlannerOutput(output, Context()).reason_code,
      "REFERENCE_PLATFORM_DATA_MISMATCH");

  output = WheelOutput();
  output.reference.reset();
  EXPECT_EQ(
      ConvertPlannerOutput(output, Context()).reason_code,
      "REFERENCE_REQUIRED_BY_OUTCOME");

  output = WheelOutput();
  auto invalid_context = Context();
  invalid_context.state_stamp = {};
  EXPECT_EQ(
      ConvertPlannerOutput(output, invalid_context).reason_code,
      "RESULT_CONTEXT_INVALID");

  invalid_context = Context();
  invalid_context.preview_frame = "odom";
  EXPECT_EQ(
      ConvertPlannerOutput(output, invalid_context).reason_code,
      "RESULT_CONTEXT_INVALID");

  invalid_context = Context();
  invalid_context.execution_frame = "map";
  EXPECT_EQ(
      ConvertPlannerOutput(output, invalid_context).reason_code,
      "RESULT_CONTEXT_INVALID");

  output = WheelOutput();
  output.reference->preview.poses_map.clear();
  EXPECT_EQ(
      ConvertPlannerOutput(output, Context()).reason_code,
      "REFERENCE_GLOBAL_PREVIEW_EMPTY");

  output = WheelOutput();
  output.reference->preview.poses_map.front().position_m.x =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(
      ConvertPlannerOutput(output, Context()).reason_code,
      "REFERENCE_GLOBAL_PREVIEW_INVALID");

  output = WheelOutput();
  output.diagnostics.local_trajectory->endpoint_error_m =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(
      ConvertPlannerOutput(output, Context()).reason_code,
      "RESULT_DIAGNOSTICS_INVALID");

  output = WheelOutput();
  output.outcome = static_cast<lunar::planning::PlanningOutcome>(255U);
  EXPECT_EQ(
      ConvertPlannerOutput(output, Context()).reason_code,
      "RESULT_ENUM_INVALID");
}

}  // namespace
}  // namespace lunar::planning::ros
