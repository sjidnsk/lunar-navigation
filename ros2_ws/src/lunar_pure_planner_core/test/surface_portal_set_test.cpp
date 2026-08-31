#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "hierarchical/global_route_planner.hpp"
#include "hierarchical/surface_portal_set.hpp"

namespace lunar::pure_planning::hierarchical {
namespace {

[[nodiscard]] GridMap GlobalMap(const std::size_t width = 50U,
                                const std::size_t height = 21U,
                                const double origin_y = 0.0) {
  return GridMap{
      .frame_id = "map",
      .width = width,
      .height = height,
      .resolution_m = 1.0,
      .origin_m = {.x = 0.0, .y = origin_y, .z = 0.0},
      .layers = {{"occupancy",
                  GridLayer{.values =
                                std::vector<std::int8_t>(width * height, 0)}}},
  };
}

[[nodiscard]] GridMap LocalMap(const std::size_t width = 50U,
                               const std::size_t height = 21U,
                               const double origin_y = 0.0) {
  return GridMap{
      .frame_id = "odom",
      .width = width,
      .height = height,
      .resolution_m = 1.0,
      .origin_m = {.x = 0.0, .y = origin_y, .z = 0.0},
      .layers =
          {{"occupancy",
            GridLayer{.values = std::vector<float>(width * height, 0.0F)}},
           {"elevation",
            GridLayer{.values = std::vector<float>(width * height, 0.0F)}}},
  };
}

[[nodiscard]] GridMap FineLocalMap() {
  GridMap map = LocalMap(250U, 105U);
  map.resolution_m = 0.2;
  return map;
}

void SetGlobalHazard(GridMap& map, const std::size_t x,
                     const std::size_t y) {
  auto& occupancy =
      std::get<std::vector<std::int8_t>>(map.layers.at("occupancy").values);
  occupancy.at(y * map.width + x) = 100;
}

void SetLocalHazard(GridMap& map, const std::size_t x, const std::size_t y) {
  auto& occupancy =
      std::get<std::vector<float>>(map.layers.at("occupancy").values);
  occupancy.at(y * map.width + x) = 1.0F;
}

void SetLocalHazardAt(GridMap& map, const double x_m, const double y_m) {
  const auto x = static_cast<std::size_t>(
      std::floor((x_m - map.origin_m.x) / map.resolution_m));
  const auto y = static_cast<std::size_t>(
      std::floor((y_m - map.origin_m.y) / map.resolution_m));
  SetLocalHazard(map, x, y);
}

[[nodiscard]] LeggedCapability LeggedPortalCapability() {
  return LeggedCapability{
      .body_extent_m = {.x = 0.68, .y = 0.33, .z = 0.35},
      .maximum_slope_rad = 0.5235987755982988,
      .maximum_step_height_m = 0.5,
      .minimum_body_clearance_m = 0.3,
  };
}

[[nodiscard]] GoalRegion FinalGoal(const double x = 40.25,
                                   const double y = 10.5) {
  return GoalRegion{
      .goal_id = "mission-final",
      .target = PointGoal{
          .position_m = {.x = x, .y = y, .z = 0.0},
          .tolerance_m = 0.35,
      },
      .yaw_rad = 0.7,
      .yaw_tolerance_rad = 0.12,
  };
}

[[nodiscard]] GlobalRoute StraightRoute(const double finish_x = 40.25,
                                        const double y = 10.5) {
  return GlobalRoute{.poses_map = {
                         {.position_m = {.x = 1.5, .y = y, .z = 0.0}},
                         {.position_m = {.x = finish_x, .y = y, .z = 0.0}},
                     }};
}

[[nodiscard]] PlanningRequest Request() {
  return PlanningRequest{
      .request_id = "portal-request",
      .environment_mode = EnvironmentMode::kLunarSurface,
      .current_state = WheeledState{
          .pose = {.position_m = {.x = 1.5, .y = 10.5, .z = 0.0}}},
      .goal_map = FinalGoal(),
      .world = MinimalWorldSnapshot{
          .global_map = GlobalMap(),
          .local_map = LocalMap(),
          .map_from_odom = RigidTransform{
              .parent_frame = "map",
              .child_frame = "odom",
          },
      },
      .capability = WheeledCapability{
          .footprint_xy_m = {{.x = 0.6, .y = 0.0}},
          .minimum_clearance_m = 0.0,
      },
  };
}

[[nodiscard]] SurfaceRollingDecision RollingDecision(
    const GlobalRoute& route, const GoalRegion& goal,
    const double pose_x = 1.5, const double horizon_m = 8.0) {
  return SurfaceRollingSession(
             route, goal,
             {.horizon_m = horizon_m, .max_deviation_m = 2.0})
      .Decide(Pose3{.position_m = {.x = pose_x,
                                   .y = route.poses_map.front().position_m.y,
                                   .z = 0.0}});
}

[[nodiscard]] auto CandidateIdentity(const SurfacePortalCandidate& candidate) {
  const auto* point = std::get_if<PointGoal>(&candidate.goal_odom.target);
  return std::tuple{
      candidate.route_progress_m, candidate.global_cell.x,
      candidate.global_cell.y, candidate.local_cell.x, candidate.local_cell.y,
      candidate.global_clearance_m, candidate.local_clearance_m,
      candidate.lateral_offset_cells, candidate.stable_rank,
      point == nullptr ? 0.0 : point->position_m.x,
      point == nullptr ? 0.0 : point->position_m.y,
  };
}

TEST(SurfacePortalSet, UsesSafeLateralCellsWhenNominalHorizonIsBlocked) {
  PlanningRequest input = Request();
  SetGlobalHazard(*input.world.global_map, 9U, 10U);
  const GlobalRoute route = StraightRoute();
  const auto decision = RollingDecision(route, input.goal_map);

  const auto result = BuildSurfacePortalSet(input, route, decision, 32U, {});

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_TRUE(std::ranges::any_of(result.candidates, [&](const auto& candidate) {
    return candidate.route_progress_m == decision.desired_horizon_progress_m &&
           std::abs(candidate.global_cell.y - 10) >= 2;
  }));
  for (const auto& candidate : result.candidates) {
    EXPECT_GE(candidate.global_clearance_m, 0.6F);
    EXPECT_FALSE(candidate.goal_odom.yaw_rad.has_value());
    const auto* point = std::get_if<PointGoal>(&candidate.goal_odom.target);
    ASSERT_NE(point, nullptr);
    EXPECT_DOUBLE_EQ(point->tolerance_m, 0.0);
  }
}

TEST(SurfacePortalSet,
     LeggedPrefersTheRouteCenterlineOverHigherClearanceLateralPortals) {
  PlanningRequest input = Request();
  input.current_state = LeggedState{
      .body_pose = {.position_m = {.x = 1.5, .y = 10.5, .z = 0.0}},
  };
  input.capability = LeggedPortalCapability();
  SetGlobalHazard(*input.world.global_map, 9U, 11U);
  const GlobalRoute route = StraightRoute();
  const auto decision = RollingDecision(route, input.goal_map);

  const auto result = BuildSurfacePortalSet(input, route, decision, 32U, {});

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_DOUBLE_EQ(result.candidates.front().route_progress_m,
                   decision.desired_horizon_progress_m);
  EXPECT_EQ(result.candidates.front().global_cell,
            (shared::GridCell{.x = 9, .y = 10}));
}

TEST(SurfacePortalSet,
     LeggedFiltersBodyInfeasiblePortalsBeforeTheThirtyTwoCandidateCap) {
  PlanningRequest input = Request();
  input.current_state = LeggedState{
      .body_pose = {.position_m = {.x = 1.5, .y = 10.5, .z = 0.33}},
  };
  input.capability = LeggedPortalCapability();
  input.world.local_map = FineLocalMap();
  for (const double portal_x : {9.5, 8.5, 7.5, 6.5}) {
    for (std::int32_t lateral = -4; lateral <= 4; ++lateral) {
      SetLocalHazardAt(input.world.local_map, portal_x + 0.2,
                       10.5 + static_cast<double>(lateral));
    }
  }
  const GlobalRoute route = StraightRoute();
  const auto decision = RollingDecision(route, input.goal_map);

  const auto result = BuildSurfacePortalSet(input, route, decision, 32U, {});

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_LE(result.candidates.front().route_progress_m, 4.0);
  EXPECT_TRUE(std::ranges::all_of(result.candidates, [](const auto& candidate) {
    return candidate.route_progress_m <= 4.0;
  }));
}

TEST(SurfacePortalSet, BacksOffLongitudinallyWhenHorizonColumnIsBlocked) {
  PlanningRequest input = Request();
  std::get<WheeledCapability>(input.capability).footprint_xy_m.clear();
  for (std::size_t y = 0U; y < input.world.global_map->height; ++y) {
    SetGlobalHazard(*input.world.global_map, 9U, y);
  }
  const GlobalRoute route = StraightRoute();
  const auto decision = RollingDecision(route, input.goal_map);

  const auto result = BuildSurfacePortalSet(input, route, decision, 32U, {});

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_LT(result.candidates.front().route_progress_m,
            decision.desired_horizon_progress_m);
  EXPECT_TRUE(std::ranges::all_of(result.candidates, [](const auto& candidate) {
    return candidate.global_cell.x != 9;
  }));
}

TEST(SurfacePortalSet, ClipsCandidatesToTheObservedLocalMap) {
  PlanningRequest clipped = Request();
  clipped.world.local_map = LocalMap(50U, 3U, 9.0);
  const GlobalRoute route = StraightRoute();
  const auto decision = RollingDecision(route, clipped.goal_map);
  const auto clipped_result =
      BuildSurfacePortalSet(clipped, route, decision, 32U, {});

  PlanningRequest wide = Request();
  const auto wide_result =
      BuildSurfacePortalSet(wide, route, decision, 32U, {});

  ASSERT_TRUE(clipped_result.ok()) << clipped_result.reason_code;
  ASSERT_TRUE(wide_result.ok()) << wide_result.reason_code;
  EXPECT_LT(clipped_result.candidates.size(), wide_result.candidates.size());
  for (const auto& candidate : clipped_result.candidates) {
    EXPECT_GE(candidate.local_cell.y, 0);
    EXPECT_LT(candidate.local_cell.y, 3);
    EXPECT_GE(candidate.global_cell.y, 9);
    EXPECT_LE(candidate.global_cell.y, 11);
  }
}

TEST(SurfacePortalSet, RemovesRepeatedGlobalAndLocalCellPairs) {
  PlanningRequest input = Request();
  const GlobalRoute route = StraightRoute();
  const auto decision = RollingDecision(route, input.goal_map, 1.5, 0.2);

  const auto result = BuildSurfacePortalSet(input, route, decision, 32U, {});

  ASSERT_TRUE(result.ok()) << result.reason_code;
  std::set<std::tuple<std::int32_t, std::int32_t, std::int32_t, std::int32_t>>
      unique_cells;
  for (const auto& candidate : result.candidates) {
    EXPECT_TRUE(unique_cells
                    .emplace(candidate.global_cell.x, candidate.global_cell.y,
                             candidate.local_cell.x, candidate.local_cell.y)
                    .second);
  }
}

TEST(SurfacePortalSet, RepeatsTheSameDocumentedSafetyOrder) {
  PlanningRequest input = Request();
  SetGlobalHazard(*input.world.global_map, 12U, 6U);
  SetLocalHazard(input.world.local_map, 11U, 10U);
  const GlobalRoute route = StraightRoute();
  const auto decision = RollingDecision(route, input.goal_map, 4.5, 12.0);

  const auto first = BuildSurfacePortalSet(input, route, decision, 32U, {});
  const auto repeated = BuildSurfacePortalSet(input, route, decision, 32U, {});

  ASSERT_TRUE(first.ok()) << first.reason_code;
  ASSERT_TRUE(repeated.ok()) << repeated.reason_code;
  ASSERT_EQ(first.candidates.size(), repeated.candidates.size());
  for (std::size_t index = 0U; index < first.candidates.size(); ++index) {
    EXPECT_EQ(CandidateIdentity(first.candidates[index]),
              CandidateIdentity(repeated.candidates[index]));
  }
  EXPECT_TRUE(std::is_sorted(
      first.candidates.begin(), first.candidates.end(),
      [](const auto& left, const auto& right) {
        return std::tuple{-left.route_progress_m, -left.global_clearance_m,
                          -left.local_clearance_m, left.global_cell.y,
                          left.global_cell.x, left.local_cell.y,
                          left.local_cell.x, left.stable_rank} <
               std::tuple{-right.route_progress_m, -right.global_clearance_m,
                          -right.local_clearance_m, right.global_cell.y,
                          right.global_cell.x, right.local_cell.y,
                          right.local_cell.x, right.stable_rank};
      }));
}

TEST(SurfacePortalSet, EnforcesTheHardThirtyTwoCandidateCap) {
  PlanningRequest input = Request();
  input.world.global_map = GlobalMap(100U, 31U);
  input.world.local_map = LocalMap(100U, 31U);
  input.goal_map = FinalGoal(90.5, 15.5);
  const GlobalRoute route = StraightRoute(90.5, 15.5);
  const auto decision = RollingDecision(route, input.goal_map, 1.5, 40.0);

  const auto hard_cap = BuildSurfacePortalSet(input, route, decision, 100U, {});
  const auto caller_cap = BuildSurfacePortalSet(input, route, decision, 5U, {});

  ASSERT_TRUE(hard_cap.ok()) << hard_cap.reason_code;
  ASSERT_TRUE(caller_cap.ok()) << caller_cap.reason_code;
  EXPECT_EQ(hard_cap.candidates.size(), 32U);
  EXPECT_EQ(caller_cap.candidates.size(), 5U);
}

TEST(SurfacePortalSet, EmitsOnlyTheUnmodifiedMissionGoalAtFinalHorizon) {
  PlanningRequest input = Request();
  const GlobalRoute route = StraightRoute();
  const auto decision = RollingDecision(route, input.goal_map, 39.5, 8.0);
  ASSERT_TRUE(decision.targets_final_goal);

  const auto result = BuildSurfacePortalSet(input, route, decision, 32U, {});

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_EQ(result.candidates.size(), 1U);
  const auto& final = result.candidates.front();
  EXPECT_EQ(final.goal_odom.goal_id, input.goal_map.goal_id);
  const auto* expected = std::get_if<PointGoal>(&input.goal_map.target);
  const auto* actual = std::get_if<PointGoal>(&final.goal_odom.target);
  ASSERT_NE(expected, nullptr);
  ASSERT_NE(actual, nullptr);
  EXPECT_EQ(actual->position_m, expected->position_m);
  EXPECT_DOUBLE_EQ(actual->tolerance_m, expected->tolerance_m);
  EXPECT_EQ(final.goal_odom.yaw_rad, input.goal_map.yaw_rad);
  EXPECT_DOUBLE_EQ(final.goal_odom.yaw_tolerance_rad,
                   input.goal_map.yaw_tolerance_rad);
  EXPECT_DOUBLE_EQ(final.route_progress_m,
                   decision.desired_horizon_progress_m);
}

TEST(SurfacePortalSet, ReturnsNoPathWhenEveryLocalCandidateIsHazardous) {
  PlanningRequest input = Request();
  auto& occupancy = std::get<std::vector<float>>(
      input.world.local_map.layers.at("occupancy").values);
  std::ranges::fill(occupancy, 1.0F);
  const GlobalRoute route = StraightRoute();
  const auto decision = RollingDecision(route, input.goal_map);

  const auto result = BuildSurfacePortalSet(input, route, decision, 32U, {});

  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.candidates.empty());
  EXPECT_EQ(result.reason_code, "NO_PATH");
}

TEST(SurfacePortalSet, SnapsIntermediatePortalToItsLocalSafeCellCenter) {
  GridMap global_map = GlobalMap(4U, 4U);
  GridMap local_map = LocalMap(4U, 4U);
  local_map.origin_m = {.x = 0.5, .y = 0.5, .z = 0.0};
  const GoalRegion portal_goal{
      .goal_id = "portal",
      .target = PointGoal{
          .position_m = {.x = 1.5, .y = 1.5, .z = 0.25},
          .tolerance_m = 0.0,
      },
      .yaw_rad = 0.75,
      .yaw_tolerance_rad = 0.1,
  };
  const SurfacePortalSetResult portals{
      .candidates = {SurfacePortalCandidate{
          .goal_odom = portal_goal,
          .route_progress_m = 2.0,
          .global_cell = {.x = 1, .y = 1},
          .local_cell = {.x = 1, .y = 1},
      }},
  };
  const SurfaceRollingDecision decision{
      .kind = SurfaceRollingDecision::Kind::kNextPortalSet,
      .projected_route_progress_m = 0.0,
      .desired_horizon_progress_m = 2.0,
      .targets_final_goal = false,
  };

  const auto converted = ConvertSurfacePortalsToLocalGoals(
      portals, decision,
      Pose3{.position_m = {.x = 0.5, .y = 0.5, .z = 0.0}}, global_map,
      local_map, true);

  ASSERT_TRUE(converted.ok()) << converted.reason_code;
  ASSERT_EQ(converted.goals->goals_odom.size(), 1U);
  const auto* point = std::get_if<PointGoal>(
      &converted.goals->goals_odom.front().target);
  ASSERT_NE(point, nullptr);
  EXPECT_DOUBLE_EQ(point->position_m.x, 2.0);
  EXPECT_DOUBLE_EQ(point->position_m.y, 2.0);
  EXPECT_DOUBLE_EQ(point->position_m.z, 0.25);
  EXPECT_NEAR(point->tolerance_m, 0.5 - 1.0e-9, 1.0e-12);
  EXPECT_FALSE(converted.goals->goals_odom.front().yaw_rad.has_value());
  EXPECT_DOUBLE_EQ(converted.goals->goals_odom.front().yaw_tolerance_rad, 0.0);
}

TEST(SurfacePortalSet, PreservesTheExactFinalLocalGoal) {
  const GridMap global_map = GlobalMap(4U, 4U);
  GridMap local_map = LocalMap(4U, 4U);
  local_map.resolution_m = 0.5;
  const GoalRegion exact_goal{
      .goal_id = "final",
      .target = PointGoal{
          .position_m = {.x = 1.37, .y = 1.63, .z = 0.4},
          .tolerance_m = 0.07,
      },
      .yaw_rad = 0.75,
      .yaw_tolerance_rad = 0.1,
  };
  const SurfacePortalSetResult portals{
      .candidates = {SurfacePortalCandidate{
          .goal_odom = exact_goal,
          .route_progress_m = 2.0,
          .global_cell = {.x = 1, .y = 1},
          .local_cell = {.x = 2, .y = 3},
      }},
  };
  const SurfaceRollingDecision decision{
      .kind = SurfaceRollingDecision::Kind::kNextPortalSet,
      .projected_route_progress_m = 0.0,
      .desired_horizon_progress_m = 2.0,
      .targets_final_goal = true,
  };

  const auto converted = ConvertSurfacePortalsToLocalGoals(
      portals, decision, Pose3{}, global_map, local_map);

  ASSERT_TRUE(converted.ok()) << converted.reason_code;
  ASSERT_TRUE(converted.goals->exact_final_goal);
  ASSERT_EQ(converted.goals->goals_odom.size(), 1U);
  const GoalRegion& retained = converted.goals->goals_odom.front();
  const auto* retained_point = std::get_if<PointGoal>(&retained.target);
  ASSERT_NE(retained_point, nullptr);
  EXPECT_EQ(retained.goal_id, exact_goal.goal_id);
  EXPECT_EQ(retained_point->position_m,
            std::get<PointGoal>(exact_goal.target).position_m);
  EXPECT_DOUBLE_EQ(retained_point->tolerance_m,
                   std::get<PointGoal>(exact_goal.target).tolerance_m);
  EXPECT_EQ(retained.yaw_rad, exact_goal.yaw_rad);
  EXPECT_DOUBLE_EQ(retained.yaw_tolerance_rad,
                   exact_goal.yaw_tolerance_rad);
}

}  // namespace
}  // namespace lunar::pure_planning::hierarchical
