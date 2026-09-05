#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_core/global_route_planner.hpp"
#include "lunar_incremental_navigation_core/legged_local_planner.hpp"
#include "lunar_incremental_navigation_core/wheel_local_planner.hpp"
#include "support/complex_terrain_scenarios.hpp"

namespace lunar::incremental_navigation::test_support {
namespace {

TEST(ComplexTerrainScenarios,
     AlternatingWallMazeIsDeterministicAndKeepsEndpointsFree) {
  const ComplexTerrainSpec spec{
      .pattern = ComplexTerrainPattern::kAlternatingWallMaze,
      .width = 64U,
      .height = 64U,
      .resolution_m = 0.2,
      .seed = 0x5EEDU,
  };

  const ComplexTerrainScenario first = MakeComplexTerrainScenario(spec);
  const ComplexTerrainScenario second = MakeComplexTerrainScenario(spec);

  ASSERT_EQ(first.fine_states.size(), 64U * 64U);
  EXPECT_EQ(first.fine_states, second.fine_states);
  EXPECT_EQ(first.start_cell, second.start_cell);
  EXPECT_EQ(first.goal_cell, second.goal_cell);
  EXPECT_EQ(first.State(first.start_cell), FineCellState::kFree);
  EXPECT_EQ(first.State(first.goal_cell), FineCellState::kFree);
  EXPECT_TRUE(first.expected_reachable);
  EXPECT_GT(std::count(first.fine_states.begin(), first.fine_states.end(),
                       FineCellState::kBlocked),
            64U);
}

TEST(ComplexTerrainScenarios,
     DenseRockFieldHasDeterministicClutterAndACarvedRoute) {
  const ComplexTerrainSpec spec{
      .pattern = ComplexTerrainPattern::kDenseRockField,
      .width = 96U,
      .height = 96U,
      .resolution_m = 0.2,
      .seed = 0x1234ABCDU,
  };

  const ComplexTerrainScenario first = MakeComplexTerrainScenario(spec);
  const ComplexTerrainScenario second = MakeComplexTerrainScenario(spec);

  EXPECT_EQ(first.fine_states, second.fine_states);
  EXPECT_TRUE(first.expected_reachable);
  EXPECT_EQ(first.State(first.start_cell), FineCellState::kFree);
  EXPECT_EQ(first.State(first.goal_cell), FineCellState::kFree);
  const auto blocked = std::count(first.fine_states.begin(),
                                  first.fine_states.end(),
                                  FineCellState::kBlocked);
  EXPECT_GT(blocked, 96U * 96U / 4U);
  EXPECT_LT(blocked, 96U * 96U / 2U);
}

TEST(ComplexTerrainScenarios,
     NarrowPassagesContainDeadEndsButKeepTheMainRouteConnected) {
  const ComplexTerrainScenario scenario = MakeComplexTerrainScenario({
      .pattern = ComplexTerrainPattern::kNarrowPassagesAndDeadEnds,
      .width = 96U,
      .height = 96U,
      .resolution_m = 0.2,
      .seed = 7U,
  });

  EXPECT_TRUE(scenario.expected_reachable);
  EXPECT_EQ(scenario.State(scenario.start_cell), FineCellState::kFree);
  EXPECT_EQ(scenario.State(scenario.goal_cell), FineCellState::kFree);
  EXPECT_GT(std::count(scenario.fine_states.begin(),
                       scenario.fine_states.end(), FineCellState::kBlocked),
            96U * 96U / 3U);
}

TEST(ComplexTerrainScenarios,
     RiskAndUnknownBandsContainBothSoftAndUnobservedTerrain) {
  const ComplexTerrainScenario scenario = MakeComplexTerrainScenario({
      .pattern = ComplexTerrainPattern::kRiskAndUnknownBands,
      .width = 96U,
      .height = 96U,
      .resolution_m = 0.2,
      .seed = 9U,
  });

  EXPECT_TRUE(scenario.expected_reachable);
  EXPECT_TRUE(std::any_of(scenario.traversal_costs.begin(),
                          scenario.traversal_costs.end(),
                          [](const double value) { return value >= 20.0; }));
  EXPECT_NE(std::find(scenario.fine_states.begin(),
                      scenario.fine_states.end(), FineCellState::kUnknown),
            scenario.fine_states.end());
  EXPECT_EQ(scenario.State(scenario.start_cell), FineCellState::kFree);
  EXPECT_EQ(scenario.State(scenario.goal_cell), FineCellState::kFree);
}

TEST(ComplexTerrainScenarios, EnclosedGoalIsAnIntentionalNoPathCase) {
  const ComplexTerrainScenario scenario = MakeComplexTerrainScenario({
      .pattern = ComplexTerrainPattern::kEnclosedGoal,
      .width = 64U,
      .height = 64U,
      .resolution_m = 0.2,
      .seed = 11U,
  });

  EXPECT_FALSE(scenario.expected_reachable);
  EXPECT_EQ(scenario.State(scenario.goal_cell), FineCellState::kFree);
  for (std::int64_t dy = -2; dy <= 2; ++dy) {
    for (std::int64_t dx = -2; dx <= 2; ++dx) {
      if (std::max(std::abs(dx), std::abs(dy)) != 2) {
        continue;
      }
      EXPECT_EQ(scenario.State({.x = scenario.goal_cell.x + dx,
                                .y = scenario.goal_cell.y + dy}),
                FineCellState::kBlocked);
    }
  }
}

TEST(ComplexTerrainScenarios,
     LeggedStepGapFieldLeavesCellsFreeForDirectedEdgeCertification) {
  const ComplexTerrainScenario scenario = MakeComplexTerrainScenario({
      .pattern = ComplexTerrainPattern::kLeggedStepGapField,
      .width = 96U,
      .height = 96U,
      .resolution_m = 0.2,
      .seed = 13U,
  });

  EXPECT_TRUE(scenario.expected_reachable);
  EXPECT_EQ(std::count(scenario.fine_states.begin(),
                       scenario.fine_states.end(), FineCellState::kBlocked),
            2U * 96U + 2U * (96U - 2U));
  EXPECT_TRUE(std::any_of(scenario.elevation_m.begin(),
                          scenario.elevation_m.end(),
                          [](const float value) { return value >= 0.8F; }));
  EXPECT_EQ(scenario.State(scenario.start_cell), FineCellState::kFree);
  EXPECT_EQ(scenario.State(scenario.goal_cell), FineCellState::kFree);
}

TEST(ComplexTerrainPlanners,
     GlobalAndWheelPlannersFindTheAlternatingMazeRoute) {
  const ComplexTerrainScenario scenario = MakeComplexTerrainScenario({
      .pattern = ComplexTerrainPattern::kAlternatingWallMaze,
      .width = 64U,
      .height = 64U,
      .resolution_m = 0.2,
      .seed = 17U,
  });
  const auto fine = MakeFineSnapshot(scenario, "wheel-benchmark");
  const auto guidance = MakeGuidanceSnapshot(scenario, "wheel-benchmark");
  const Pose2 start = StartPose(scenario);
  const LocalTarget target = LocalTargetForGoal(scenario);

  const RequestLocalPlanningView view(fine, start, 0.0, {});
  const LocalPlanResult local = WheelLocalPlanner(BenchmarkWheelCapability())
                                    .Plan(view, start, target,
                                          SteadyClock::now() +
                                              std::chrono::seconds(5),
                                          {});
  ASSERT_EQ(local.status, LocalPlanResult::Status::kPlanFound);
  EXPECT_GT(local.statistics.expanded_states, 0U);

  GlobalRoutePlanner global(GlobalRoutePlannerConfig{
      .global_detour_margin_m =
          static_cast<double>(scenario.geometry.height) *
          scenario.geometry.resolution_m,
      .unknown_step_risk = 5.0,
  });
  const GlobalRouteResult route = global.Plan(
      *guidance, start.position_m, target.center,
      SteadyClock::now() + std::chrono::seconds(5), {});
  EXPECT_EQ(route.status, GuidanceStatus::kAvailable);
  EXPECT_TRUE(route.route.has_value());
}

TEST(ComplexTerrainPlanners,
     GlobalAndWheelPlannersExhaustTheEnclosedGoalWithoutPublishingAPath) {
  const ComplexTerrainScenario scenario = MakeComplexTerrainScenario({
      .pattern = ComplexTerrainPattern::kEnclosedGoal,
      .width = 64U,
      .height = 64U,
      .resolution_m = 0.2,
      .seed = 19U,
  });
  const auto fine = MakeFineSnapshot(scenario, "wheel-benchmark");
  const auto guidance = MakeGuidanceSnapshot(scenario, "wheel-benchmark");
  const Pose2 start = StartPose(scenario);
  const LocalTarget target = LocalTargetForGoal(scenario);

  const RequestLocalPlanningView view(fine, start, 0.0, {});
  const LocalPlanResult local = WheelLocalPlanner(BenchmarkWheelCapability())
                                    .Plan(view, start, target,
                                          SteadyClock::now() +
                                              std::chrono::seconds(5),
                                          {});
  EXPECT_EQ(local.status, LocalPlanResult::Status::kNoPath);
  EXPECT_TRUE(local.path.empty());

  GlobalRoutePlanner global(GlobalRoutePlannerConfig{
      .global_detour_margin_m =
          static_cast<double>(scenario.geometry.height) *
          scenario.geometry.resolution_m,
      .unknown_step_risk = 5.0,
  });
  const GlobalRouteResult route = global.Plan(
      *guidance, start.position_m, target.center,
      SteadyClock::now() + std::chrono::seconds(5), {});
  EXPECT_EQ(route.status, GuidanceStatus::kNoRoute);
  EXPECT_FALSE(route.route.has_value());
}

TEST(ComplexTerrainPlanners,
     LeggedPlannerUsesElevationCertificationInTheStepGapField) {
  const ComplexTerrainScenario scenario = MakeComplexTerrainScenario({
      .pattern = ComplexTerrainPattern::kLeggedStepGapField,
      .width = 64U,
      .height = 64U,
      .resolution_m = 0.2,
      .seed = 23U,
  });
  const auto fine = MakeFineSnapshot(scenario, "legged-benchmark");
  const Pose2 start = StartPose(scenario);
  const RequestLocalPlanningView view(fine, start, 0.0, {});

  const LocalPlanResult result = LeggedLocalPlanner(BenchmarkLeggedCapability())
                                     .Plan(view, start,
                                           LocalTargetForGoal(scenario),
                                           SteadyClock::now() +
                                               std::chrono::seconds(5),
                                           {});

  EXPECT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  EXPECT_GT(result.statistics.evaluated_transitions,
            result.statistics.generated_states);
}

}  // namespace
}  // namespace lunar::incremental_navigation::test_support
