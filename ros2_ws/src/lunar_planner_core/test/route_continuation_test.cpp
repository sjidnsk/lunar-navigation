#include "hierarchical/route_continuation.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include "hierarchical/frame_transform.hpp"
#include "lunar_planner_core/planner.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning {
namespace {

void SetRollingIdentity(PlannerInput &input, const std::string &platform_id,
                        const std::string &capability_version) {
  input.mission_id = "mission-lunar-route";
  input.mission_revision = 7U;
  input.platform_id = platform_id;
  input.capability_version = capability_version;
  input.global_map_generation = 11U;
  input.local_map_generation = 19U;
  input.map_from_odom_generation = 23U;
}

[[nodiscard]] PlannerInput DistantWheelInput() {
  PlannerInput input = test::MakeValidWheelInput();
  input.request_id = "rolling-wheel-0";
  input.world.global_map = test::MakeFlatMap("map", 24U, 8U, 1.0);
  input.world.local_map = test::MakeFlatMap("odom", 12U, 8U, 1.0);
  input.config.global_map.base_resolution_m = 1.0;
  input.goal_map = GoalRegion{
      .goal_id = "rolling-wheel-goal",
      .target =
          PointGoal{
              .position_m = {18.5, 3.5, 0.0},
              .tolerance_m = 0.2,
          },
  };
  SetRollingIdentity(input, "wheel-alpha", "wheel-test-v1");
  return input;
}

[[nodiscard]] PlannerInput AdvancedWheelInput(const PlannerInput &initial) {
  PlannerInput next = initial;
  next.request_id = "rolling-wheel-1";
  next.current_state = WheeledState{
      .pose = Pose3{.position_m = {4.5, 3.6, 0.0}},
  };
  next.local_map_generation += 1U;
  return next;
}

[[nodiscard]] PlannerInput ThreeHopInput() {
  PlannerInput input = test::MakeValidHopperInput();
  input.request_id = "rolling-hopper-0";
  input.world.global_map = test::MakeFlatMap("map", 20U, 7U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 20U, 7U, 0.5);
  input.world.map_from_odom = RigidTransform{
      .parent_frame = "map",
      .child_frame = "odom",
      .stamp = input.state_time,
  };
  input.config.global_map.base_resolution_m = 0.5;
  input.config.global_map.maximum_cells = 1'024U;
  input.config.global_map.maximum_axis_cells = 1'024U;
  input.current_state = HopperState{
      .pose = Pose3{.position_m = {1.5, 1.75, 0.5}},
  };
  input.goal_map = GoalRegion{
      .goal_id = "rolling-hopper-goal",
      .target =
          PointGoal{
              .position_m = {7.25, 1.75, 0.0},
              .tolerance_m = 0.1,
          },
  };
  auto &capability = std::get<HopperCapability>(input.capability);
  capability.body_half_extent_m.x = 0.1;
  capability.body_half_extent_m.y = 0.1;
  capability.minimum_landing_region_area_m2 = 0.1;
  capability.maximum_launch_speed_mps = 2.0;
  capability.maximum_launch_impulse_newton_seconds = 100.0;
  capability.maximum_landing_speed_mps = 2.0;
  capability.minimum_flight_time = std::chrono::milliseconds{500};
  capability.maximum_flight_time = std::chrono::seconds{3};
  SetRollingIdentity(input, "hopper-alpha", "hopper-test-v1");
  return input;
}

[[nodiscard]] PlannerInput LandedAfterAuthorizedHop(
    const PlannerInput &initial, const PlannerOutput &first) {
  PlannerInput next = initial;
  next.request_id = "rolling-hopper-1";
  next.local_map_generation += 1U;
  next.continuation = first.continuation;
  const CertifiedHopPreview &completed = first.certified_hops.front();
  const auto landed_odom = hierarchical::TransformPose(
      completed.landing_pose_map, next.world.map_from_odom,
      hierarchical::TransformDirection::kParentToChild);
  EXPECT_TRUE(landed_odom.has_value());
  next.current_state = HopperState{
      .pose = landed_odom.value_or(Pose3{}),
      .velocity = {},
  };
  next.previous_execution = ExecutionContext{HopperExecutionContext{
      .state = HopperExecutionState::kLandedHold,
      .active_plan_id = first.reference->plan_id,
      .active_segment_id = completed.segment_id,
  }};
  return next;
}

TEST(RouteContinuation, ReusesGroundRouteForSmallRealPoseDeviation) {
  Planner planner;
  const PlannerInput initial = DistantWheelInput();
  const PlannerOutput first = planner.Plan(initial);
  ASSERT_TRUE(first.reference.has_value()) << first.reason_code;
  ASSERT_NE(first.continuation, nullptr);
  PlannerInput next = AdvancedWheelInput(initial);
  next.continuation = first.continuation;

  const PlannerOutput second = planner.Plan(next);

  ASSERT_TRUE(second.reference.has_value()) << second.reason_code;
  ASSERT_TRUE(second.diagnostics.hierarchical.has_value());
  EXPECT_TRUE(second.diagnostics.hierarchical->route_reused);
  EXPECT_EQ(second.diagnostics.hierarchical->global_expanded_states, 0U);
  ASSERT_NE(second.continuation, nullptr);
  EXPECT_EQ(second.continuation->route_id(), first.continuation->route_id());
  const auto actual_map = hierarchical::TransformPose(
      std::get<WheeledState>(next.current_state).pose, next.world.map_from_odom,
      hierarchical::TransformDirection::kChildToParent);
  ASSERT_TRUE(actual_map.has_value());
  EXPECT_EQ(second.reference->preview.poses_map.front(), *actual_map);
}

TEST(RouteContinuation, ReusesTheSameGroundContractForLeggedPlatform) {
  Planner planner;
  PlannerInput initial = test::MakeValidLeggedInput();
  initial.request_id = "rolling-legged-0";
  initial.world.global_map = test::MakeFlatMap("map", 24U, 8U, 1.0);
  initial.world.local_map = test::MakeFlatMap("odom", 12U, 8U, 1.0);
  initial.goal_map = GoalRegion{
      .goal_id = "rolling-legged-goal",
      .target =
          PointGoal{
              .position_m = {18.5, 3.5, 0.0},
              .tolerance_m = 0.2,
          },
  };
  SetRollingIdentity(initial, "legged-alpha", "legged-test-v1");
  const PlannerOutput first = planner.Plan(initial);
  ASSERT_NE(first.continuation, nullptr) << first.reason_code;
  PlannerInput next = initial;
  next.request_id = "rolling-legged-1";
  next.local_map_generation += 1U;
  next.current_state = LeggedState{
      .body_pose = Pose3{.position_m = {4.5, 3.6, 0.5}},
  };
  next.continuation = first.continuation;

  const PlannerOutput second = planner.Plan(next);

  ASSERT_TRUE(second.reference.has_value()) << second.reason_code;
  ASSERT_TRUE(second.diagnostics.hierarchical.has_value());
  EXPECT_TRUE(second.diagnostics.hierarchical->route_reused);
  EXPECT_EQ(second.diagnostics.hierarchical->global_expanded_states, 0U);
}

TEST(RouteContinuation, ChecksEveryStableGroundRouteIdentity) {
  Planner planner;
  const PlannerInput initial = DistantWheelInput();
  const PlannerOutput first = planner.Plan(initial);
  ASSERT_NE(first.continuation, nullptr) << first.reason_code;
  const PlannerInput matching = AdvancedWheelInput(initial);

  EXPECT_TRUE(
      hierarchical::TryReuseGroundRoute(matching, *first.continuation).ok());

  PlannerInput small_tf_update = matching;
  small_tf_update.map_from_odom_generation += 1U;
  small_tf_update.world.map_from_odom.translation_m.y = 0.05;
  EXPECT_TRUE(
      hierarchical::TryReuseGroundRoute(small_tf_update, *first.continuation)
          .ok());

  PlannerInput changed = matching;
  changed.mission_id = "different-mission";
  EXPECT_EQ(hierarchical::TryReuseGroundRoute(changed, *first.continuation)
                .reason_code,
            "ROUTE_CONTINUATION_MISSION_MISMATCH");

  changed = matching;
  changed.mission_revision += 1U;
  EXPECT_EQ(hierarchical::TryReuseGroundRoute(changed, *first.continuation)
                .reason_code,
            "ROUTE_CONTINUATION_MISSION_MISMATCH");

  changed = matching;
  changed.goal_map.goal_id = "different-goal";
  EXPECT_EQ(hierarchical::TryReuseGroundRoute(changed, *first.continuation)
                .reason_code,
            "ROUTE_CONTINUATION_GOAL_MISMATCH");

  changed = matching;
  changed.platform_id = "wheel-beta";
  EXPECT_EQ(hierarchical::TryReuseGroundRoute(changed, *first.continuation)
                .reason_code,
            "ROUTE_CONTINUATION_PLATFORM_MISMATCH");

  changed = matching;
  changed.capability_version = "wheel-test-v2";
  EXPECT_EQ(hierarchical::TryReuseGroundRoute(changed, *first.continuation)
                .reason_code,
            "ROUTE_CONTINUATION_CAPABILITY_MISMATCH");

  changed = matching;
  changed.global_map_generation += 1U;
  EXPECT_EQ(hierarchical::TryReuseGroundRoute(changed, *first.continuation)
                .reason_code,
            "ROUTE_CONTINUATION_GLOBAL_MAP_MISMATCH");

  changed = matching;
  changed.map_from_odom_generation += 1U;
  changed.world.map_from_odom.translation_m.y = 4.0;
  EXPECT_EQ(hierarchical::TryReuseGroundRoute(changed, *first.continuation)
                .reason_code,
            "ROUTE_TF_DELTA_REPLAN_REQUIRED");
}

TEST(RouteContinuation, FallsBackToFreshGroundRouteOnIdentityMismatch) {
  Planner planner;
  const PlannerInput initial = DistantWheelInput();
  const PlannerOutput first = planner.Plan(initial);
  ASSERT_NE(first.continuation, nullptr) << first.reason_code;
  PlannerInput next = AdvancedWheelInput(initial);
  next.continuation = first.continuation;
  next.global_map_generation += 1U;

  const PlannerOutput replanned = planner.Plan(next);

  ASSERT_TRUE(replanned.reference.has_value()) << replanned.reason_code;
  ASSERT_TRUE(replanned.diagnostics.hierarchical.has_value());
  EXPECT_FALSE(replanned.diagnostics.hierarchical->route_reused);
  EXPECT_GT(replanned.diagnostics.hierarchical->global_expanded_states, 0U);
  ASSERT_NE(replanned.continuation, nullptr);
  EXPECT_NE(replanned.continuation->route_id(), first.continuation->route_id());
}

TEST(RouteContinuation, PromotesOnlyStableLandingInsidePromotionRegion) {
  Planner planner;
  const PlannerInput initial = ThreeHopInput();
  const PlannerOutput first = planner.Plan(initial);
  ASSERT_TRUE(first.reference.has_value()) << first.reason_code;
  ASSERT_NE(first.continuation, nullptr);
  ASSERT_GT(first.certified_hops.size(), 1U);
  PlannerInput landed = LandedAfterAuthorizedHop(initial, first);

  const hierarchical::HopperHopPromotionResult promoted =
      hierarchical::TryPromoteHopperHop(landed, *first.continuation);

  ASSERT_TRUE(promoted.ok()) << promoted.reason_code;
  ASSERT_TRUE(promoted.hop.has_value());
  EXPECT_EQ(promoted.hop->segment_id, first.certified_hops[1U].segment_id);
  EXPECT_EQ(promoted.route_cursor, 1U);

  PlannerInput outside = landed;
  auto &outside_state = std::get<HopperState>(outside.current_state);
  outside_state.pose.position_m.y += 2.0;
  EXPECT_EQ(hierarchical::TryPromoteHopperHop(outside, *first.continuation)
                .reason_code,
            "HOP_LANDING_DEVIATION_REPLAN_REQUIRED");

  PlannerInput moving = landed;
  std::get<HopperState>(moving.current_state).velocity.linear_mps.x = 0.5;
  EXPECT_EQ(hierarchical::TryPromoteHopperHop(moving, *first.continuation)
                .reason_code,
            "HOP_LANDING_NOT_STABLE");

  PlannerInput stale = landed;
  stale.state_time.nanoseconds_since_epoch +=
      std::chrono::seconds{2}.count() * 1'000'000'000LL;
  EXPECT_EQ(
      hierarchical::TryPromoteHopperHop(stale, *first.continuation).reason_code,
      "HOP_LANDING_STATE_STALE");
}

TEST(RouteContinuation, PromotesExactlyOneNextCertifiedHopThroughPlanner) {
  Planner planner;
  const PlannerInput initial = ThreeHopInput();
  const PlannerOutput first = planner.Plan(initial);
  ASSERT_NE(first.continuation, nullptr) << first.reason_code;
  ASSERT_GT(first.certified_hops.size(), 1U);
  PlannerInput landed = LandedAfterAuthorizedHop(initial, first);

  const PlannerOutput second = planner.Plan(landed);

  ASSERT_TRUE(second.reference.has_value()) << second.reason_code;
  ASSERT_TRUE(second.diagnostics.hierarchical.has_value());
  EXPECT_TRUE(second.diagnostics.hierarchical->route_reused);
  EXPECT_EQ(second.diagnostics.hierarchical->route_cursor, 1U);
  const auto *hops = std::get_if<HopReference>(&second.reference->data);
  ASSERT_NE(hops, nullptr);
  ASSERT_EQ(hops->segments.size(), 1U);
  EXPECT_EQ(hops->segments.front().segment_id,
            first.certified_hops[1U].segment_id);
  EXPECT_EQ(hops->segments.front().launch_pose,
            std::get<HopperState>(landed.current_state).pose);
  ASSERT_NE(second.continuation, nullptr);
  EXPECT_EQ(second.continuation->route_id(), first.continuation->route_id());
}

}  // namespace
}  // namespace lunar::planning
