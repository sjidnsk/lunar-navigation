#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include <gtest/gtest.h>

#include "hierarchical/surface_rolling_session.hpp"

namespace lunar::pure_planning::hierarchical {
namespace {

[[nodiscard]] GoalRegion FinalGoal() {
  return GoalRegion{
      .goal_id = "final",
      .target = PointGoal{
          .position_m = {.x = 20.0, .y = 20.0, .z = 0.0},
          .tolerance_m = 0.0,
      },
      .yaw_rad = std::nullopt,
      .yaw_tolerance_rad = 0.0,
  };
}

[[nodiscard]] GlobalRoute Route() {
  return GlobalRoute{
      .poses_map = {
          {.position_m = {.x = 0.0, .y = 0.0, .z = 0.0},
           .orientation = {}},
          {.position_m = {.x = 20.0, .y = 0.0, .z = 0.0},
           .orientation = {}},
          {.position_m = {.x = 20.0, .y = 20.0, .z = 0.0},
           .orientation = {}},
      },
  };
}

[[nodiscard]] Pose3 Pose(const double x, const double y) {
  return Pose3{.position_m = {.x = x, .y = y, .z = 0.0},
               .orientation = {}};
}

TEST(SurfaceRollingSession, ReportsProjectedAndDesiredProgressOnCurrentSegment) {
  const SurfaceRollingSession session(
      Route(), FinalGoal(), {.horizon_m = 8.0, .max_deviation_m = 2.0});

  const auto decision = session.Decide(Pose(3.0, 0.0));

  ASSERT_EQ(decision.kind, SurfaceRollingDecision::Kind::kNextPortalSet);
  EXPECT_DOUBLE_EQ(decision.projected_route_progress_m, 3.0);
  EXPECT_DOUBLE_EQ(decision.desired_horizon_progress_m, 11.0);
  EXPECT_FALSE(decision.targets_final_goal);
  EXPECT_DOUBLE_EQ(decision.lateral_deviation_m, 0.0);
}

TEST(SurfaceRollingSession, CarriesHorizonAcrossPolylineCorner) {
  const SurfaceRollingSession session(
      Route(), FinalGoal(), {.horizon_m = 8.0, .max_deviation_m = 2.0});

  const auto decision = session.Decide(Pose(19.0, 1.0));

  ASSERT_EQ(decision.kind, SurfaceRollingDecision::Kind::kNextPortalSet);
  EXPECT_DOUBLE_EQ(decision.projected_route_progress_m, 21.0);
  EXPECT_DOUBLE_EQ(decision.desired_horizon_progress_m, 29.0);
  EXPECT_FALSE(decision.targets_final_goal);
}

TEST(SurfaceRollingSession, RecognizesTheFinalGoal) {
  const SurfaceRollingSession session(
      Route(), FinalGoal(), {.horizon_m = 8.0, .max_deviation_m = 2.0});

  const auto decision = session.Decide(Pose(20.0, 20.0));

  EXPECT_EQ(decision.kind, SurfaceRollingDecision::Kind::kFinalGoalReached);
  EXPECT_FALSE(decision.targets_final_goal);
}

TEST(SurfaceRollingSession, ReportsLateralDeviationFromThePolyline) {
  const SurfaceRollingSession session(
      Route(), FinalGoal(), {.horizon_m = 8.0, .max_deviation_m = 2.0});

  const auto decision = session.Decide(Pose(3.0, 3.0));

  EXPECT_EQ(decision.kind, SurfaceRollingDecision::Kind::kNextPortalSet);
  EXPECT_DOUBLE_EQ(decision.lateral_deviation_m, 3.0);
}

TEST(SurfaceRollingSession, UsesTheFinalGoalWhenTheRouteIsShorterThanHorizon) {
  const SurfaceRollingSession session(
      Route(), FinalGoal(), {.horizon_m = 100.0, .max_deviation_m = 2.0});

  const auto decision = session.Decide(Pose(3.0, 0.0));

  ASSERT_EQ(decision.kind, SurfaceRollingDecision::Kind::kNextPortalSet);
  EXPECT_DOUBLE_EQ(decision.projected_route_progress_m, 3.0);
  EXPECT_DOUBLE_EQ(decision.desired_horizon_progress_m, 40.0);
  EXPECT_TRUE(decision.targets_final_goal);
}

TEST(SurfaceRollingSession, RejectsInvalidRouteConfigurationAndInputs) {
  const auto invalid_route = SurfaceRollingSession(
      {}, FinalGoal(), {.horizon_m = 8.0, .max_deviation_m = 2.0})
                                 .Decide(Pose(0.0, 0.0));
  EXPECT_EQ(invalid_route.kind, SurfaceRollingDecision::Kind::kInvalidRoute);

  GoalRegion planar_goal = FinalGoal();
  planar_goal.target = PlanarRegionGoal{};
  const auto non_point_goal = SurfaceRollingSession(
      Route(), std::move(planar_goal), {.horizon_m = 8.0, .max_deviation_m = 2.0})
                                  .Decide(Pose(0.0, 0.0));
  EXPECT_EQ(non_point_goal.kind,
            SurfaceRollingDecision::Kind::kInvalidRoute);

  const auto invalid_horizon = SurfaceRollingSession(
      Route(), FinalGoal(), {.horizon_m = 0.0, .max_deviation_m = 2.0})
                                   .Decide(Pose(0.0, 0.0));
  EXPECT_EQ(invalid_horizon.kind,
            SurfaceRollingDecision::Kind::kInvalidRoute);

  const auto non_finite_pose = SurfaceRollingSession(
      Route(), FinalGoal(), {.horizon_m = 8.0, .max_deviation_m = 2.0})
                                   .Decide(Pose(std::numeric_limits<double>::quiet_NaN(),
                                                 0.0));
  EXPECT_EQ(non_finite_pose.kind,
            SurfaceRollingDecision::Kind::kInvalidRoute);
}

}  // namespace
}  // namespace lunar::pure_planning::hierarchical
