#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

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
      candidate.stable_rank, point == nullptr ? 0.0 : point->position_m.x,
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

}  // namespace
}  // namespace lunar::pure_planning::hierarchical
