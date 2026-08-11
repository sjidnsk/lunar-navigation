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
  input.world.local_map = test::MakeFlatMap("odom", 24U, 16U, 1.0);
  input.world.local_map.origin_m.x = -4.0;
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
  initial.world.local_map = test::MakeFlatMap("odom", 24U, 16U, 1.0);
  initial.world.local_map.origin_m.x = -4.0;
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

TEST(RouteContinuation, HopperSingleHopDoesNotCreateOrPromoteAContinuation) {
  Planner planner;
  const PlannerOutput output = planner.Plan(test::MakeValidHopperInput());

  ASSERT_TRUE(output.reference.has_value()) << output.reason_code;
  const auto *hops = std::get_if<HopReference>(&output.reference->data);
  ASSERT_NE(hops, nullptr);
  ASSERT_EQ(hops->segments.size(), 1U);
  EXPECT_EQ(output.continuation, nullptr);
  ASSERT_EQ(output.certified_hops.size(), 1U);
  EXPECT_TRUE(output.certified_hops.front().promotion_region_map.empty());
}

}  // namespace
}  // namespace lunar::planning
