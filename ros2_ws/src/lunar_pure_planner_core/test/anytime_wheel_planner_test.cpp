#include "wheel/anytime_wheel_planner.hpp"

#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <iostream>
#include <random>
#include <stop_token>
#include <vector>

#include <gtest/gtest.h>

#include "hierarchical/global_route_planner.hpp"
#include "lunar_pure_planner_core/planner.hpp"
#include "shared/local_terrain_projection.hpp"
#include "shared/map_snapshot.hpp"
#include "wheel/wheel_types.hpp"
#include "wheel_long_range_fixture.hpp"

namespace lunar::pure_planning::wheel {
namespace {

using namespace std::chrono_literals;

template <class Request>
concept HasRequestCostConfig = requires(Request request) { request.cost; };

static_assert(!HasRequestCostConfig<WheelPlanRequest>);

struct TerrainFixture final {
  std::shared_ptr<const shared::MapSnapshot> map;
  shared::LocalTerrainProjection terrain;
};

[[nodiscard]] Pose3 Pose(const double x, const double y,
                         const double yaw = 0.0) {
  return Pose3{
      .position_m = Vec3{.x = x, .y = y, .z = 0.0},
      .orientation = QuaternionFromYaw(yaw),
  };
}

[[nodiscard]] WheelMotionPrimitive Primitive(
    const char* id, const WheelPrimitiveKind kind, const double x,
    const double y = 0.0, const double yaw = 0.0) {
  return WheelMotionPrimitive{
      .primitive_id = id,
      .kind = kind,
      .relative_end_pose = Pose3{
          .position_m = Vec3{.x = x, .y = y, .z = 0.0},
          .orientation = QuaternionFromYaw(yaw),
      },
  };
}

[[nodiscard]] WheelMotionPrimitive ArcPrimitive(
    const char* id, const WheelPrimitiveKind kind, const double radius_m,
    const double yaw_rad) {
  return Primitive(id, kind, radius_m * std::sin(yaw_rad),
                   radius_m * (1.0 - std::cos(yaw_rad)), yaw_rad);
}

[[nodiscard]] WheeledCapability Capability(
    const double footprint_length = 1.0,
    const double footprint_width = 0.6) {
  const double half_length = footprint_length / 2.0;
  const double half_width = footprint_width / 2.0;
  return WheeledCapability{
      .footprint_xy_m = {
          Vec2{.x = -half_length, .y = -half_width},
          Vec2{.x = half_length, .y = -half_width},
          Vec2{.x = half_length, .y = half_width},
          Vec2{.x = -half_length, .y = half_width},
      },
      .body_extent_m = Vec3{.x = footprint_length, .y = footprint_width,
                            .z = 0.5},
      .wheel_diameter_m = 0.4,
      .wheel_width_m = 0.1,
      .wheelbase_m = 0.7,
      .track_width_m = 0.5,
      .minimum_underbody_clearance_m = 0.15,
      .maximum_local_obstacle_relief_m = 0.12,
      .allow_unsupported_gap = false,
      .minimum_body_z_m = 0.0,
      .maximum_body_z_m = 2.0,
      .maximum_forward_speed_mps = 1.0,
      .maximum_reverse_speed_mps = 0.5,
      .maximum_spin_rate_radps = 1.0,
      .maximum_acceleration_mps2 = 1.0,
      .maximum_braking_deceleration_mps2 = 1.0,
      .maximum_yaw_acceleration_radps2 = 1.0,
      .maximum_lateral_acceleration_mps2 = 1.0,
      .maximum_curvature_per_m = 2.0,
      .maximum_slope_rad = 0.5,
      .minimum_clearance_m = 0.0,
      .motion_primitives = {
          Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
          Primitive("left", WheelPrimitiveKind::kForwardArc, 0.2, 0.01,
                    std::numbers::pi / 32.0),
          Primitive("right", WheelPrimitiveKind::kForwardArc, 0.2, -0.01,
                    -std::numbers::pi / 32.0),
          Primitive("reverse", WheelPrimitiveKind::kReverse, -0.2),
      },
  };
}

[[nodiscard]] WheeledCapability ProjectWheelCapability() {
  return WheeledCapability{
      .footprint_xy_m = {
          Vec2{.x = 0.591, .y = 0.409},
          Vec2{.x = 0.591, .y = -0.409},
          Vec2{.x = -0.591, .y = -0.409},
          Vec2{.x = -0.591, .y = 0.409},
      },
      .body_extent_m = Vec3{.x = 1.182, .y = 0.818, .z = 1.29996},
      .wheel_diameter_m = 0.319,
      .wheel_width_m = 0.148,
      .wheelbase_m = 0.8175,
      .track_width_m = 0.67,
      .minimum_underbody_clearance_m = 0.21,
      .maximum_local_obstacle_relief_m = 0.2,
      .allow_unsupported_gap = false,
      .minimum_body_z_m = 0.0,
      .maximum_body_z_m = 1.29996,
      .maximum_forward_speed_mps = 1.5,
      .maximum_reverse_speed_mps = 1.5,
      .maximum_spin_rate_radps = 1.0,
      .maximum_acceleration_mps2 = 0.5,
      .maximum_braking_deceleration_mps2 = 0.5,
      .maximum_yaw_acceleration_radps2 = 0.5,
      .maximum_lateral_acceleration_mps2 = 0.5,
      .maximum_curvature_per_m = 1.0,
      .maximum_slope_rad = 0.3490658503988659,
      .minimum_clearance_m = 0.2,
      .motion_primitives = {
          Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
          Primitive("forward-arc-left", WheelPrimitiveKind::kForwardArc,
                    0.19509032201612825, 0.01921471959676957,
                    std::numbers::pi / 16.0),
          Primitive("forward-arc-right", WheelPrimitiveKind::kForwardArc,
                    0.19509032201612825, -0.01921471959676957,
                    -std::numbers::pi / 16.0),
          Primitive("reverse", WheelPrimitiveKind::kReverse, -0.2),
          Primitive("reverse-arc-left", WheelPrimitiveKind::kReverseArc,
                    -0.19509032201612825, 0.01921471959676957,
                    -std::numbers::pi / 16.0),
          Primitive("reverse-arc-right", WheelPrimitiveKind::kReverseArc,
                    -0.19509032201612825, -0.01921471959676957,
                    std::numbers::pi / 16.0),
          Primitive("spin-left", WheelPrimitiveKind::kSpinCounterclockwise,
                    0.0, 0.0, std::numbers::pi / 16.0),
          Primitive("spin-right", WheelPrimitiveKind::kSpinClockwise, 0.0,
                    0.0, -std::numbers::pi / 16.0),
          Primitive("stop-switch", WheelPrimitiveKind::kStopAndSwitch, 0.0),
      },
  };
}

[[nodiscard]] TerrainFixture MakeTerrain(
    const std::size_t width, const std::size_t height,
    std::vector<float> occupancy, const double resolution = 0.2,
    std::vector<float> elevation = {}, const Vec3 origin = {}) {
  if (elevation.empty()) {
    elevation.assign(width * height, 0.0F);
  }
  GridMap grid{
      .frame_id = "odom",
      .width = width,
      .height = height,
      .resolution_m = resolution,
      .origin_m = origin,
      .layers = {
          {"occupancy", GridLayer{.values = std::move(occupancy)}},
          {"elevation", GridLayer{.values = std::move(elevation)}},
      },
  };
  auto snapshot = shared::MapSnapshot::Create(
      std::move(grid), shared::MapContract::kLocalElevation);
  EXPECT_TRUE(snapshot.ok()) << snapshot.reason_code;
  auto projection = shared::BuildLocalTerrainProjection(snapshot.snapshot);
  EXPECT_TRUE(projection.ok()) << projection.reason_code;
  return TerrainFixture{
      .map = std::move(snapshot.snapshot),
      .terrain = std::move(*projection.value),
  };
}

[[nodiscard]] TerrainFixture FlatTerrain(
    const std::size_t width = 40U, const std::size_t height = 20U) {
  return MakeTerrain(width, height,
                     std::vector<float>(width * height, 0.0F));
}

[[nodiscard]] WheelPlanRequest RequestTo(
    const TerrainFixture& fixture, const WheeledCapability& capability,
    const double x, const double y, const double yaw,
    const Pose3& start = Pose(1.0, 1.0)) {
  return WheelPlanRequest{
      .start = WheeledState{.pose = start},
      .goals_odom = LocalGoalSet{
          .goals_odom = {GoalRegion{
              .goal_id = "point",
              .target = PointGoal{
                  .position_m = Vec3{.x = x, .y = y, .z = 0.0},
                  .tolerance_m = 1.0e-6,
              },
              .yaw_rad = yaw,
              .yaw_tolerance_rad = 1.0e-6,
          }},
          .exact_final_goal = true,
      },
      .terrain = &fixture.terrain,
      .capability = &capability,
      .control = SearchControl{
          .deadline = SteadyClock::now() + 2s,
      },
  };
}

[[nodiscard]] double TrajectoryYaw(const TrajectoryPoint& point) {
  const auto yaw = YawFromQuaternion(point.pose.orientation);
  EXPECT_TRUE(yaw.has_value());
  return yaw.value_or(std::numeric_limits<double>::quiet_NaN());
}

[[nodiscard]] Pose3 ComposePlanar(const Pose3& source,
                                  const Pose3& relative) {
  const double yaw = YawFromQuaternion(source.orientation).value();
  const double relative_yaw =
      YawFromQuaternion(relative.orientation).value();
  return Pose(
      source.position_m.x + std::cos(yaw) * relative.position_m.x -
          std::sin(yaw) * relative.position_m.y,
      source.position_m.y + std::sin(yaw) * relative.position_m.x +
          std::cos(yaw) * relative.position_m.y,
      NormalizeYaw(yaw + relative_yaw));
}

[[nodiscard]] bool MatchesAnyFullPrimitive(
    const Pose3& source, const Pose3& target,
    const WheeledCapability& capability) {
  const double target_yaw = YawFromQuaternion(target.orientation).value();
  return std::ranges::any_of(
      capability.motion_primitives,
      [&](const WheelMotionPrimitive& primitive) {
        const Pose3 expected =
            ComposePlanar(source, primitive.relative_end_pose);
        const double expected_yaw =
            YawFromQuaternion(expected.orientation).value();
        return std::hypot(expected.position_m.x - target.position_m.x,
                          expected.position_m.y - target.position_m.y) <=
                   1.0e-8 &&
               std::abs(ShortestYawDelta(expected_yaw, target_yaw)) <=
                   1.0e-8;
      });
}

TEST(WheelPlanner, UsesOneSearchForRankedPortalFallback) {
  constexpr std::size_t kWidth = 40U;
  constexpr std::size_t kHeight = 20U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  occupancy[7U * kWidth + 7U] = 1.0F;
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  WheeledCapability capability = Capability();
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2)};
  const auto portal = [](const char* id, const double x, const double y) {
    return GoalRegion{
        .goal_id = id,
        .target = PointGoal{
            .position_m = Vec3{.x = x, .y = y, .z = 0.0},
            .tolerance_m = 1.0e-6,
        },
    };
  };
  const WheelPlanRequest request{
      .start = WheeledState{.pose = Pose(1.0, 1.0)},
      .goals_odom = LocalGoalSet{
          .goals_odom = {portal("rank-0-blocked", 1.4, 1.4),
                         portal("rank-1-reachable", 1.4, 1.0)},
          .exact_final_goal = false,
      },
      .terrain = &fixture.terrain,
      .capability = &capability,
      .control = SearchControl{.deadline = SteadyClock::now() + 2s},
      .search = AnytimeSearchConfig{.stop_after_first_solution = true},
  };

  const WheelPlanResult result = PlanWheel(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.preferred_builder_invocations, 1U);
  EXPECT_EQ(result.ara_search_invocations, 1U);
  EXPECT_GT(result.metrics.expanded_states, 0U);
  ASSERT_TRUE(result.selected_goal_index.has_value());
  EXPECT_EQ(*result.selected_goal_index, 1U);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x, 1.4, 1.0e-9);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.y, 1.0, 1.0e-9);
}

TEST(WheelPlanner, BoundsTerminalGoalSignaturesAtThirtyTwoPortals) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability();
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2)};
  const auto portal = [](const std::size_t index) {
    return GoalRegion{
        .goal_id = "portal-" + std::to_string(index),
        .target = PointGoal{
            .position_m = Vec3{.x = 1.2, .y = 1.0, .z = 0.0},
            .tolerance_m = 1.0e-6,
        },
    };
  };
  WheelPlanRequest request{
      .start = WheeledState{.pose = Pose(1.0, 1.0)},
      .goals_odom = LocalGoalSet{.exact_final_goal = false},
      .terrain = &fixture.terrain,
      .capability = &capability,
      .control = SearchControl{.deadline = SteadyClock::now() + 2s},
  };
  for (std::size_t index = 0U; index < 32U; ++index) {
    request.goals_odom.goals_odom.push_back(portal(index));
  }

  const WheelPlanResult at_limit = PlanWheel(request);
  ASSERT_TRUE(at_limit.ok()) << at_limit.reason_code;
  ASSERT_TRUE(at_limit.selected_goal_index.has_value());
  EXPECT_EQ(*at_limit.selected_goal_index, 0U);

  request.goals_odom.goals_odom.push_back(portal(32U));
  const WheelPlanResult above_limit = PlanWheel(request);
  EXPECT_EQ(above_limit.status, LocalPlanStatus::kInvalidInput);
  EXPECT_EQ(above_limit.reason_code, "WHEEL_INPUT_INVALID");
}

TEST(WheelPlanner, ExactFinalGoalSetContainsOnlyUntouchedMissionGoal) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability();
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2)};
  const GoalRegion mission_goal{
      .goal_id = "mission-final",
      .target = PointGoal{
          .position_m = Vec3{.x = 1.2, .y = 1.0, .z = 0.0},
          .tolerance_m = 1.0e-6,
      },
      .yaw_rad = 0.0,
      .yaw_tolerance_rad = 1.0e-6,
  };
  WheelPlanRequest request{
      .start = WheeledState{.pose = Pose(1.0, 1.0)},
      .goals_odom = LocalGoalSet{
          .goals_odom = {mission_goal},
          .exact_final_goal = true,
      },
      .terrain = &fixture.terrain,
      .capability = &capability,
      .control = SearchControl{.deadline = SteadyClock::now() + 2s},
  };

  const WheelPlanResult exact = PlanWheel(request);
  ASSERT_TRUE(exact.ok()) << exact.reason_code;
  ASSERT_TRUE(exact.selected_goal_index.has_value());
  EXPECT_EQ(*exact.selected_goal_index, 0U);
  EXPECT_NEAR(TrajectoryYaw(exact.trajectory.back()), 0.0, 1.0e-9);

  request.goals_odom.goals_odom.push_back(mission_goal);
  EXPECT_EQ(PlanWheel(request).status, LocalPlanStatus::kInvalidInput);
  request.goals_odom.exact_final_goal = false;
  request.goals_odom.goals_odom.resize(1U);
  EXPECT_EQ(PlanWheel(request).status, LocalPlanStatus::kInvalidInput);
}

TEST(WheelPlanner, ConsecutiveProjectArcsRemainPhysicalInteriorEdges) {
  const TerrainFixture fixture = FlatTerrain(80U, 80U);
  WheeledCapability capability = ProjectWheelCapability();
  const WheelMotionPrimitive arc = capability.motion_primitives[1U];
  capability.motion_primitives = {arc};
  const Pose3 start = Pose(3.0, 3.0, 0.31);
  const Pose3 first = ComposePlanar(start, arc.relative_end_pose);
  const Pose3 second = ComposePlanar(first, arc.relative_end_pose);
  const Pose3 half_arc = Pose(
      std::sin(std::numbers::pi / 32.0),
      1.0 - std::cos(std::numbers::pi / 32.0),
      std::numbers::pi / 32.0);
  const Pose3 goal = ComposePlanar(second, half_arc);

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, goal.position_m.x, goal.position_m.y,
      TrajectoryYaw(TrajectoryPoint{.pose = goal}), start));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_GE(result.trajectory.size(), 25U);
  EXPECT_NEAR(result.trajectory[8U].pose.position_m.x, first.position_m.x,
              1.0e-9);
  EXPECT_NEAR(result.trajectory[8U].pose.position_m.y, first.position_m.y,
              1.0e-9);
  EXPECT_NEAR(TrajectoryYaw(result.trajectory[8U]),
              TrajectoryYaw(TrajectoryPoint{.pose = first}), 1.0e-9);
  EXPECT_NEAR(result.trajectory[16U].pose.position_m.x, second.position_m.x,
              1.0e-9);
  EXPECT_NEAR(result.trajectory[16U].pose.position_m.y, second.position_m.y,
              1.0e-9);
  EXPECT_NEAR(TrajectoryYaw(result.trajectory[16U]),
              TrajectoryYaw(TrajectoryPoint{.pose = second}), 1.0e-9);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x, goal.position_m.x,
              1.0e-9);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.y, goal.position_m.y,
              1.0e-9);
}

TEST(WheelPlanner, PlansCertifiedSTurnWithContinuousPrimitiveEndpoints) {
  const TerrainFixture fixture = FlatTerrain(80U, 80U);
  WheeledCapability capability = ProjectWheelCapability();
  const WheelMotionPrimitive left = capability.motion_primitives[1U];
  const WheelMotionPrimitive right = capability.motion_primitives[2U];
  capability.motion_primitives = {left, right};
  const Pose3 start = Pose(3.0, 3.0, -0.27);
  const Pose3 first = ComposePlanar(start, left.relative_end_pose);
  const Pose3 goal = ComposePlanar(first, right.relative_end_pose);

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, goal.position_m.x, goal.position_m.y,
      TrajectoryYaw(TrajectoryPoint{.pose = goal}), start));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_GE(result.trajectory.size(), 17U);
  EXPECT_NEAR(result.trajectory[8U].pose.position_m.x, first.position_m.x,
              1.0e-9);
  EXPECT_NEAR(result.trajectory[8U].pose.position_m.y, first.position_m.y,
              1.0e-9);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x, goal.position_m.x,
              1.0e-9);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.y, goal.position_m.y,
              1.0e-9);
  EXPECT_NEAR(TrajectoryYaw(result.trajectory.back()),
              TrajectoryYaw(TrajectoryPoint{.pose = goal}), 1.0e-9);
}

TEST(WheelPlanner, PlansConsecutiveReverseArcsAtPhysicalEndpoints) {
  const TerrainFixture fixture = FlatTerrain(80U, 80U);
  WheeledCapability capability = ProjectWheelCapability();
  const WheelMotionPrimitive reverse_arc = capability.motion_primitives[4U];
  capability.motion_primitives = {reverse_arc};
  const Pose3 start = Pose(5.0, 5.0, 0.23);
  const Pose3 first = ComposePlanar(start, reverse_arc.relative_end_pose);
  const Pose3 goal = ComposePlanar(first, reverse_arc.relative_end_pose);

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, goal.position_m.x, goal.position_m.y,
      TrajectoryYaw(TrajectoryPoint{.pose = goal}), start));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_GE(result.trajectory.size(), 17U);
  EXPECT_NEAR(result.trajectory[8U].pose.position_m.x, first.position_m.x,
              1.0e-9);
  EXPECT_NEAR(result.trajectory[8U].pose.position_m.y, first.position_m.y,
              1.0e-9);
  EXPECT_EQ(result.reverse_edge_count, 2U);
}

TEST(WheelPlanner, SelectsYawBinsFromPrimitiveIncrements) {
  const TerrainFixture fixture = FlatTerrain();
  const std::array<std::pair<double, std::size_t>, 6U> cases{{
      {std::numbers::pi / 8.0, 16U},
      {std::numbers::pi / 16.0, 32U},
      {std::numbers::pi / 32.0, 64U},
      {std::numbers::pi / 64.0, 128U},
      {std::numbers::pi / 128.0, 256U},
      {0.1, 256U},
  }};
  for (const auto& [yaw_delta, expected_bins] : cases) {
    WheeledCapability capability = Capability(0.2, 0.2);
    capability.motion_primitives = {
        ArcPrimitive("arc", WheelPrimitiveKind::kForwardArc, 1.0,
                     yaw_delta),
    };
    const WheelPlanResult result = PlanWheel(RequestTo(
        fixture, capability, 1.0, 1.0, 0.0, Pose(1.0, 1.0)));
    ASSERT_TRUE(result.ok()) << result.reason_code;
    EXPECT_EQ(result.maximum_yaw_bins, expected_bins)
        << "yaw_delta=" << yaw_delta;
  }
}

TEST(WheelPlanner, LongRangeHeuristicIsStrongAndBelowCertifiedCost) {
  const TerrainFixture fixture = FlatTerrain(1800U, 30U);
  WheeledCapability capability = ProjectWheelCapability();
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };
  WheelPlanRequest stopped = RequestTo(
      fixture, capability, 341.0, 2.0, 0.0, Pose(1.0, 2.0));
  stopped.control.deadline = SteadyClock::now() + 5s;
  WheelPlanRequest moving = stopped;
  moving.start.velocity.linear_mps.x = 0.2;

  const WheelPlanResult stopped_result = PlanWheel(stopped);
  const WheelPlanResult moving_result = PlanWheel(moving);

  ASSERT_TRUE(stopped_result.ok()) << stopped_result.reason_code;
  ASSERT_TRUE(moving_result.ok()) << moving_result.reason_code;
  const double distance_only = 340.0 / stopped_result.cost_scales[0U];
  EXPECT_GT(stopped_result.start_heuristic_lower_bound,
            5.0 * distance_only);
  EXPECT_LE(stopped_result.start_heuristic_lower_bound,
            stopped_result.cost + 1.0e-9);
  EXPECT_LE(moving_result.start_heuristic_lower_bound,
            moving_result.cost + 1.0e-9);
}

TEST(WheelPlanner, HeuristicDiscountsInitialAndScaledTerminalEdges) {
  const TerrainFixture fixture = FlatTerrain(30U, 20U);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };
  for (const double distance : {0.1, 0.3999999, 0.4, 0.4000001}) {
    const WheelPlanResult result = PlanWheel(RequestTo(
        fixture, capability, 1.0 + distance, 2.0, 0.0, Pose(1.0, 2.0)));
    ASSERT_TRUE(result.ok()) << "distance=" << distance << ' '
                             << result.reason_code;
    EXPECT_LE(result.start_heuristic_lower_bound, result.cost + 1.0e-9)
        << "distance=" << distance;
  }
}

TEST(WheelPlanner, HeuristicDoesNotOverestimateLongObliquePrimitiveChain) {
  const TerrainFixture fixture = FlatTerrain(600U, 350U);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };
  const double yaw = std::atan(0.5);
  const Pose3 start = Pose(5.0, 5.0, yaw);
  constexpr std::size_t kEdges = 500U;
  const double travel = 0.2 * static_cast<double>(kEdges);
  const Pose3 goal = Pose(5.0 + travel * std::cos(yaw),
                          5.0 + travel * std::sin(yaw), yaw);

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, goal.position_m.x, goal.position_m.y, yaw, start));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_LE(result.start_heuristic_lower_bound, result.cost + 1.0e-9);
}

TEST(WheelPlanner, DisablesBarrierLowerBoundForConcaveFootprint) {
  constexpr std::size_t kWidth = 60U;
  constexpr std::size_t kHeight = 30U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    occupancy[y * kWidth + 25U] = 1.0F;
  }
  const TerrainFixture blocked =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  const TerrainFixture flat = FlatTerrain(kWidth, kHeight);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.footprint_xy_m = {
      Vec2{.x = -0.1, .y = -0.1}, Vec2{.x = 0.1, .y = -0.1},
      Vec2{.x = 0.04, .y = 0.0}, Vec2{.x = 0.1, .y = 0.1},
      Vec2{.x = -0.1, .y = 0.1},
  };
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };

  const WheelPlanResult blocked_result = PlanWheel(RequestTo(
      blocked, capability, 8.0, 3.0, 0.0, Pose(2.0, 3.0)));
  const WheelPlanResult flat_result = PlanWheel(RequestTo(
      flat, capability, 8.0, 3.0, 0.0, Pose(2.0, 3.0)));

  EXPECT_NEAR(blocked_result.start_heuristic_lower_bound,
              flat_result.start_heuristic_lower_bound, 1.0e-9);
}

TEST(WheelPlanner, SkipsBarrierWhoseInteriorMeetsTheGoalDisk) {
  constexpr std::size_t kWidth = 40U;
  constexpr std::size_t kHeight = 30U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    occupancy[y * kWidth + 20U] = 1.0F;
  }
  const TerrainFixture blocked =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  const TerrainFixture flat = FlatTerrain(kWidth, kHeight);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };
  WheelPlanRequest blocked_request = RequestTo(
      blocked, capability, 4.1, 3.0, 0.0, Pose(2.0, 3.0));
  WheelPlanRequest flat_request = RequestTo(
      flat, capability, 4.1, 3.0, 0.0, Pose(2.0, 3.0));
  std::get<PointGoal>(blocked_request.goals_odom.goals_odom.front().target)
      .tolerance_m = 0.3;
  std::get<PointGoal>(flat_request.goals_odom.goals_odom.front().target)
      .tolerance_m = 0.3;

  const WheelPlanResult blocked_result = PlanWheel(blocked_request);
  const WheelPlanResult flat_result = PlanWheel(flat_request);

  EXPECT_NEAR(blocked_result.start_heuristic_lower_bound,
              flat_result.start_heuristic_lower_bound, 1.0e-9);
}

TEST(WheelPlanner, KeepsBlockingBarrierAheadOfThirtyTwoLongIrrelevantRuns) {
  constexpr std::size_t kWidth = 150U;
  constexpr std::size_t kHeight = 150U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  for (std::size_t run = 0U; run < 32U; ++run) {
    const std::size_t y = 2U * run;
    for (std::size_t x = 0U; x < 50U; ++x) {
      occupancy[y * kWidth + x] = 1.0F;
    }
  }
  for (std::size_t y = 95U; y <= 105U; ++y) {
    occupancy[y * kWidth + 75U] = 1.0F;
  }
  const TerrainFixture blocked =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  const TerrainFixture flat = FlatTerrain(kWidth, kHeight);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };

  const WheelPlanResult blocked_result = PlanWheel(RequestTo(
      blocked, capability, 25.0, 20.0, 0.0, Pose(5.0, 20.0)));
  const WheelPlanResult flat_result = PlanWheel(RequestTo(
      flat, capability, 25.0, 20.0, 0.0, Pose(5.0, 20.0)));

  EXPECT_GT(blocked_result.start_heuristic_lower_bound,
            flat_result.start_heuristic_lower_bound);
}

TEST(WheelPlanner, HeuristicUsesArcChordAndPureYawLowerBounds) {
  const TerrainFixture fixture = FlatTerrain(50U, 50U);
  WheeledCapability arc_capability = Capability(0.2, 0.2);
  arc_capability.motion_primitives = {
      ArcPrimitive("arc", WheelPrimitiveKind::kForwardArc, 1.0,
                   std::numbers::pi / 2.0),
  };
  const WheelPlanResult arc_result = PlanWheel(RequestTo(
      fixture, arc_capability, 3.0, 3.0, std::numbers::pi / 2.0,
      Pose(2.0, 2.0)));
  ASSERT_TRUE(arc_result.ok()) << arc_result.reason_code;
  EXPECT_LE(arc_result.start_heuristic_lower_bound,
            arc_result.cost + 1.0e-9);

  WheeledCapability spin_capability = Capability(0.2, 0.2);
  spin_capability.motion_primitives = {
      Primitive("spin", WheelPrimitiveKind::kSpinCounterclockwise, 0.0, 0.0,
                std::numbers::pi / 2.0),
  };
  const WheelPlanResult spin_result = PlanWheel(RequestTo(
      fixture, spin_capability, 2.0, 2.0, std::numbers::pi / 2.0,
      Pose(2.0, 2.0)));
  ASSERT_TRUE(spin_result.ok()) << spin_result.reason_code;
  EXPECT_GT(spin_result.start_heuristic_lower_bound, 0.0);
  EXPECT_LE(spin_result.start_heuristic_lower_bound,
            spin_result.cost + 1.0e-9);
}

TEST(WheelPlanner, RetainsObstacleDistinctLabelsInOneKey) {
  constexpr std::size_t kWidth = 80U;
  constexpr std::size_t kHeight = 60U;
  constexpr double kResolution = 0.2;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  occupancy[11U * kWidth + 10U] = 1.0F;
  const TerrainFixture blocked = MakeTerrain(
      kWidth, kHeight, std::move(occupancy), kResolution);
  const TerrainFixture clear = MakeTerrain(
      kWidth, kHeight, std::vector<float>(kWidth * kHeight, 0.0F),
      kResolution);
  WheeledCapability capability = Capability(0.02, 0.02);
  const WheelMotionPrimitive short_arc =
      ArcPrimitive("a-short", WheelPrimitiveKind::kForwardArc, 1.0,
                   std::numbers::pi / 16.0);
  const WheelMotionPrimitive long_arc =
      ArcPrimitive("b-long", WheelPrimitiveKind::kForwardArc, 1.2,
                   std::numbers::pi / 16.0);
  capability.motion_primitives = {short_arc, long_arc};
  const Pose3 start = Pose(2.0, 2.16);
  const Pose3 short_first =
      ComposePlanar(start, short_arc.relative_end_pose);
  const Pose3 short_goal =
      ComposePlanar(short_first, short_arc.relative_end_pose);
  const Pose3 long_first = ComposePlanar(start, long_arc.relative_end_pose);
  const Pose3 long_goal = ComposePlanar(long_first, long_arc.relative_end_pose);
  const Pose3 goal = Pose(
      0.5 * (short_goal.position_m.x + long_goal.position_m.x),
      0.5 * (short_goal.position_m.y + long_goal.position_m.y),
      TrajectoryYaw(TrajectoryPoint{.pose = long_goal}));
  const auto make_request = [&](const TerrainFixture& fixture) {
    WheelPlanRequest request = RequestTo(
        fixture, capability, goal.position_m.x, goal.position_m.y,
        TrajectoryYaw(TrajectoryPoint{.pose = goal}), start);
    std::get<PointGoal>(request.goals_odom.goals_odom.front().target)
        .tolerance_m = 0.04;
    return request;
  };

  const WheelPlanResult clear_result = PlanWheel(make_request(clear));
  const WheelPlanResult blocked_result = PlanWheel(make_request(blocked));
  const WheelPlanResult short_edge_result = PlanWheel(RequestTo(
      blocked, capability, short_first.position_m.x, short_first.position_m.y,
      TrajectoryYaw(TrajectoryPoint{.pose = short_first}), start));

  ASSERT_TRUE(clear_result.ok()) << clear_result.reason_code;
  ASSERT_TRUE(blocked_result.ok()) << blocked_result.reason_code;
  ASSERT_TRUE(short_edge_result.ok()) << short_edge_result.reason_code;
  ASSERT_GE(clear_result.trajectory.size(), 17U);
  ASSERT_GE(blocked_result.trajectory.size(), 17U);
  EXPECT_NEAR(clear_result.trajectory[8U].pose.position_m.x,
              short_first.position_m.x, 1.0e-9);
  EXPECT_NEAR(blocked_result.trajectory[8U].pose.position_m.x,
              long_first.position_m.x, 1.0e-9);
  EXPECT_GT(blocked_result.quantized_endpoint_aliases, 0U);
}

TEST(WheelPlanner, RetainsHigherCostNearLabelWithCertifiedGoalConnector) {
  const TerrainFixture fixture = FlatTerrain(60U, 40U);
  WheeledCapability capability = Capability(0.2, 0.2);
  const WheelMotionPrimitive short_arc = ArcPrimitive(
      "a-short", WheelPrimitiveKind::kForwardArc, 1.0,
      std::numbers::pi / 16.0);
  const WheelMotionPrimitive terminal_arc = ArcPrimitive(
      "b-terminal", WheelPrimitiveKind::kForwardArc, 1.2,
      std::numbers::pi / 16.0);
  const WheelMotionPrimitive connector =
      Primitive("c-connector", WheelPrimitiveKind::kForward, 0.2);
  capability.motion_primitives = {short_arc, terminal_arc, connector};
  const Pose3 start = Pose(3.0, 3.0);
  const Pose3 terminal_source =
      ComposePlanar(start, terminal_arc.relative_end_pose);
  const Pose3 goal =
      ComposePlanar(terminal_source, Pose(0.1, 0.0, 0.0));

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, goal.position_m.x, goal.position_m.y,
      TrajectoryYaw(TrajectoryPoint{.pose = goal}), start));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_GE(result.trajectory.size(), 17U);
  EXPECT_NEAR(result.trajectory[8U].pose.position_m.x,
              terminal_source.position_m.x, 1.0e-9);
  EXPECT_NEAR(result.trajectory[8U].pose.position_m.y,
              terminal_source.position_m.y, 1.0e-9);
}

TEST(WheelPlanner, CertifiedGoalLabelReplacesOneOfFourNonterminalLabels) {
  const TerrainFixture fixture = FlatTerrain(60U, 40U);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      ArcPrimitive("a", WheelPrimitiveKind::kForwardArc, 0.8,
                   std::numbers::pi / 16.0),
      ArcPrimitive("b", WheelPrimitiveKind::kForwardArc, 0.9,
                   std::numbers::pi / 16.0),
      ArcPrimitive("c", WheelPrimitiveKind::kForwardArc, 1.0,
                   std::numbers::pi / 16.0),
      ArcPrimitive("d", WheelPrimitiveKind::kForwardArc, 1.1,
                   std::numbers::pi / 16.0),
      ArcPrimitive("e-terminal", WheelPrimitiveKind::kForwardArc, 1.2,
                   std::numbers::pi / 16.0),
      Primitive("f-connector", WheelPrimitiveKind::kForward, 0.2),
  };
  const Pose3 start = Pose(3.0, 3.0);
  const Pose3 terminal_source =
      ComposePlanar(start, capability.motion_primitives[4U].relative_end_pose);
  const Pose3 goal =
      ComposePlanar(terminal_source, Pose(0.1, 0.0, 0.0));

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, goal.position_m.x, goal.position_m.y,
      TrajectoryYaw(TrajectoryPoint{.pose = goal}), start));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_LE(result.maximum_active_labels_per_key, 4U);
  EXPECT_NEAR(result.trajectory[8U].pose.position_m.x,
              terminal_source.position_m.x, 1.0e-9);
  EXPECT_NEAR(result.trajectory[8U].pose.position_m.y,
              terminal_source.position_m.y, 1.0e-9);
}

TEST(WheelPlanner, ReachesOffGridPoseOnFlatFreeMap) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.6, 0.4);
  capability.motion_primitives = {
      ArcPrimitive("left", WheelPrimitiveKind::kForwardArc, 1.0, 0.6),
  };
  constexpr double kGoalYaw = 0.3;
  const double goal_x = 1.0 + std::sin(kGoalYaw);
  const double goal_y = 1.0 + 1.0 - std::cos(kGoalYaw);

  const WheelPlanResult result =
      PlanWheel(RequestTo(fixture, capability, goal_x, goal_y, kGoalYaw));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.trajectory.empty());
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x, goal_x, 1.0e-6);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.y, goal_y, 1.0e-6);
  EXPECT_NEAR(TrajectoryYaw(result.trajectory.back()), kGoalYaw, 1.0e-6);
}

TEST(WheelPlanner, PlansMultiplePrimitivesFromArbitraryTranslatedStart) {
  const TerrainFixture fixture = FlatTerrain(80U, 80U);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };
  const Pose3 start = Pose(1.13, 1.07);

  const WheelPlanResult result = PlanWheel(
      RequestTo(fixture, capability, 2.13, 1.07, 0.0, start));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_GT(result.trajectory.size(), 2U);
  EXPECT_NEAR(result.trajectory.front().pose.position_m.x, 1.13, 1.0e-12);
  EXPECT_NEAR(result.trajectory.front().pose.position_m.y, 1.07, 1.0e-12);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x, 2.13, 1.0e-9);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.y, 1.07, 1.0e-9);
}

TEST(WheelPlanner, PlansMultiplePrimitivesFromArbitrarySE2Start) {
  const TerrainFixture fixture = FlatTerrain(80U, 80U);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };
  constexpr double kStartYaw = 0.37;
  const Pose3 start = Pose(2.13, 2.17, kStartYaw);
  const double goal_x = 2.13 + std::cos(kStartYaw);
  const double goal_y = 2.17 + std::sin(kStartYaw);

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, goal_x, goal_y, kStartYaw, start));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_GT(result.trajectory.size(), 2U);
  EXPECT_NEAR(result.trajectory.front().pose.position_m.x, 2.13, 1.0e-12);
  EXPECT_NEAR(result.trajectory.front().pose.position_m.y, 2.17, 1.0e-12);
  EXPECT_NEAR(TrajectoryYaw(result.trajectory.front()), kStartYaw, 1.0e-12);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x, goal_x, 1.0e-9);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.y, goal_y, 1.0e-9);
  EXPECT_NEAR(TrajectoryYaw(result.trajectory.back()), kStartYaw, 1.0e-9);
}

TEST(WheelPlanner, RigidTransformPreservesRequestRelativeTrajectory) {
  const TerrainFixture fixture = FlatTerrain(100U, 100U);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };
  const WheelPlanResult baseline = PlanWheel(RequestTo(
      fixture, capability, 3.0, 2.0, 0.0, Pose(2.0, 2.0)));
  constexpr double kYaw = 0.63;
  const double transformed_goal_x = 4.17 + std::cos(kYaw);
  const double transformed_goal_y = 3.11 + std::sin(kYaw);
  const WheelPlanResult transformed = PlanWheel(RequestTo(
      fixture, capability, transformed_goal_x, transformed_goal_y, kYaw,
      Pose(4.17, 3.11, kYaw)));

  ASSERT_TRUE(baseline.ok()) << baseline.reason_code;
  ASSERT_TRUE(transformed.ok()) << transformed.reason_code;
  ASSERT_EQ(transformed.trajectory.size(), baseline.trajectory.size());
  EXPECT_NEAR(transformed.cost, baseline.cost, 1.0e-9);
  for (std::size_t index = 0U; index < baseline.trajectory.size(); ++index) {
    const double world_dx =
        transformed.trajectory[index].pose.position_m.x - 4.17;
    const double world_dy =
        transformed.trajectory[index].pose.position_m.y - 3.11;
    const double local_x =
        std::cos(kYaw) * world_dx + std::sin(kYaw) * world_dy;
    const double local_y =
        -std::sin(kYaw) * world_dx + std::cos(kYaw) * world_dy;
    EXPECT_NEAR(local_x,
                baseline.trajectory[index].pose.position_m.x - 2.0,
                1.0e-8);
    EXPECT_NEAR(local_y,
                baseline.trajectory[index].pose.position_m.y - 2.0,
                1.0e-8);
    EXPECT_NEAR(ShortestYawDelta(
                    kYaw, TrajectoryYaw(transformed.trajectory[index])),
                TrajectoryYaw(baseline.trajectory[index]), 1.0e-8);
  }
}

TEST(WheelPlanner, StopsWhenControlTriggersOnlyDuringTrajectoryReconstruction) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.6, 0.4);
  capability.motion_primitives = {
      ArcPrimitive("left", WheelPrimitiveKind::kForwardArc, 1.0, 0.6),
  };
  constexpr double kGoalYaw = 0.3;
  const double goal_x = 1.0 + std::sin(kGoalYaw);
  const double goal_y = 2.0 - std::cos(kGoalYaw);

  std::size_t baseline_reads = 0U;
  WheelPlanRequest baseline =
      RequestTo(fixture, capability, goal_x, goal_y, kGoalYaw);
  baseline.control.deadline = SteadyClock::time_point::max();
  baseline.control.now = [&] {
    ++baseline_reads;
    return SteadyClock::time_point{};
  };
  const WheelPlanResult baseline_result = PlanWheel(baseline);
  ASSERT_TRUE(baseline_result.ok()) << baseline_result.reason_code;
  ASSERT_GT(baseline_result.trajectory.size(), 1U);
  ASSERT_EQ((baseline_result.trajectory.size() - 1U) % 8U, 0U);
  const std::size_t reconstructed_edges =
      (baseline_result.trajectory.size() - 1U) / 8U;
  // Reconstruction performs two entry checkpoints, one checkpoint plus
  // eight timed samples per edge, and one final checkpoint. Derive the phase
  // boundary from observable output instead of dense-search initialization.
  const std::size_t reconstruction_reads = 3U + 9U * reconstructed_edges;
  ASSERT_GT(baseline_reads, reconstruction_reads);
  const std::size_t first_reconstruction_read =
      baseline_reads - reconstruction_reads;

  std::stop_source stop;
  std::size_t cancel_reads = 0U;
  WheelPlanRequest canceled =
      RequestTo(fixture, capability, goal_x, goal_y, kGoalYaw);
  canceled.control.stop_token = stop.get_token();
  canceled.control.deadline = SteadyClock::time_point::max();
  canceled.control.now = [&] {
    if (cancel_reads++ == first_reconstruction_read) {
      stop.request_stop();
    }
    return SteadyClock::time_point{};
  };
  const WheelPlanResult canceled_result = PlanWheel(canceled);
  EXPECT_EQ(canceled_result.status, LocalPlanStatus::kCanceled)
      << "clock_reads=" << cancel_reads;

  std::size_t timeout_reads = 0U;
  WheelPlanRequest timed_out =
      RequestTo(fixture, capability, goal_x, goal_y, kGoalYaw);
  timed_out.control.deadline = SteadyClock::time_point{1ms};
  timed_out.control.now = [&] {
    return timeout_reads++ < first_reconstruction_read
               ? SteadyClock::time_point{}
               : SteadyClock::time_point{2ms};
  };
  const WheelPlanResult timeout_result = PlanWheel(timed_out);
  EXPECT_EQ(timeout_result.status, LocalPlanStatus::kTimedOut)
      << "clock_reads=" << timeout_reads;
}

TEST(WheelPlanner, DoesNotInventLateralMotionForAForwardOnlyCapability) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.4),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 1.0, 1.15, 0.0, Pose(1.0, 1.0)));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
}

TEST(WheelPlanner, DoesNotInventReverseForAForwardOnlyCapability) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.4),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 0.85, 1.0, 0.0, Pose(1.0, 1.0)));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
}

TEST(WheelPlanner, DoesNotAllocateAStateForAnInvalidSweepEdge) {
  constexpr std::size_t kWidth = 20U;
  constexpr std::size_t kHeight = 20U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  occupancy[5U * kWidth + 6U] = 1.0F;
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 1.4, 1.0, 0.0, Pose(1.0, 1.0)));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
  EXPECT_EQ(result.quantized_state_count, 1U);
  EXPECT_EQ(result.metrics.edge_validation_evaluations, 2U);
  EXPECT_GE(result.broad_phase_rejects, 1U);
}

TEST(WheelPlanner, DoesNotInventTranslationForASpinOnlyCapability) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("spin", WheelPrimitiveKind::kSpinCounterclockwise, 0.0, 0.0,
                std::numbers::pi / 2.0),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 1.05, 1.0, 0.0, Pose(1.0, 1.0)));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
}

TEST(WheelPlanner, ShortensARealSpinPrimitiveToTheExactYaw) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("spin", WheelPrimitiveKind::kSpinCounterclockwise, 0.0, 0.0,
                std::numbers::pi / 2.0),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 1.0, 1.0, 0.37, Pose(1.0, 1.0)));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x, 1.0, 1.0e-9);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.y, 1.0, 1.0e-9);
  EXPECT_NEAR(TrajectoryYaw(result.trajectory.back()), 0.37, 1.0e-9);
}

TEST(WheelPlanner, ShortensARealReversePrimitiveToTheExactGoal) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("reverse", WheelPrimitiveKind::kReverse, -0.4),
  };

  WheelPlanRequest request = RequestTo(
      fixture, capability, 0.83, 1.0, 0.0, Pose(1.0, 1.0));
  std::get<PointGoal>(request.goals_odom.goals_odom.front().target)
      .tolerance_m = 0.0;

  const WheelPlanResult result = PlanWheel(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x, 0.83, 1.0e-6);
}

TEST(WheelPlanner, OptionalYawKeepsThePhysicallyReachedArcYaw) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      ArcPrimitive("left", WheelPrimitiveKind::kForwardArc, 1.0, 0.6),
  };
  constexpr double kReachedYaw = 0.3;
  WheelPlanRequest request = RequestTo(
      fixture, capability, 1.0 + std::sin(kReachedYaw),
      2.0 - std::cos(kReachedYaw), 0.0);
  request.goals_odom.goals_odom.front().yaw_rad.reset();
  request.goals_odom.goals_odom.front().yaw_tolerance_rad = 0.4;

  const WheelPlanResult result = PlanWheel(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_NEAR(TrajectoryYaw(result.trajectory.back()), kReachedYaw, 1.0e-6);
}

TEST(WheelPlanner, IgnoresPointGoalZAndUsesTerrainSupportedEndpointZ) {
  constexpr std::size_t kWidth = 40U;
  constexpr std::size_t kHeight = 20U;
  constexpr double kGroundZ = 2.5;
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::vector<float>(kWidth * kHeight, 0.0F), 0.2,
      std::vector<float>(kWidth * kHeight, static_cast<float>(kGroundZ)));
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.4),
  };
  const Pose3 supported_start = Pose3{
      .position_m = Vec3{.x = 1.0, .y = 1.0, .z = kGroundZ},
      .orientation = QuaternionFromYaw(0.0),
  };
  WheelPlanRequest zero_z = RequestTo(
      fixture, capability, 1.4, 1.0, 0.0, supported_start);
  WheelPlanRequest high_z = zero_z;
  WheelPlanRequest nan_z = zero_z;
  std::get<PointGoal>(high_z.goals_odom.goals_odom.front().target)
      .position_m.z = 100.0;
  std::get<PointGoal>(nan_z.goals_odom.goals_odom.front().target).position_m.z =
      std::numeric_limits<double>::quiet_NaN();

  const WheelPlanResult zero_result = PlanWheel(zero_z);
  const WheelPlanResult high_result = PlanWheel(high_z);
  const WheelPlanResult nan_result = PlanWheel(nan_z);

  ASSERT_TRUE(zero_result.ok()) << zero_result.reason_code;
  ASSERT_TRUE(high_result.ok()) << high_result.reason_code;
  ASSERT_TRUE(nan_result.ok()) << nan_result.reason_code;
  for (const WheelPlanResult* result :
       {&zero_result, &high_result, &nan_result}) {
    ASSERT_FALSE(result->trajectory.empty());
    EXPECT_NEAR(result->trajectory.back().pose.position_m.x, 1.4, 1.0e-9);
    EXPECT_NEAR(result->trajectory.back().pose.position_m.y, 1.0, 1.0e-9);
    EXPECT_NEAR(result->trajectory.back().pose.position_m.z, kGroundZ,
                1.0e-9);
    EXPECT_NEAR(TrajectoryYaw(result->trajectory.back()), 0.0, 1.0e-9);
    EXPECT_NEAR(result->cost, zero_result.cost, 1.0e-12);
    EXPECT_EQ(result->trajectory.back().time_from_start,
              zero_result.trajectory.back().time_from_start);
  }
}

TEST(WheelPlanner, UsesOrientedRectangleAndNarrowResolutionInTightCorridor) {
  constexpr std::size_t kWidth = 40U;
  constexpr std::size_t kHeight = 20U;
  std::vector<float> occupancy(kWidth * kHeight, 1.0F);
  for (std::size_t y = 8U; y <= 11U; ++y) {
    for (std::size_t x = 0U; x < kWidth; ++x) {
      occupancy[y * kWidth + x] = 0.0F;
    }
  }
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  WheeledCapability capability = Capability(1.0, 0.6);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 5.0, 2.0, 0.0, Pose(1.0, 2.0)));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_TRUE(result.metrics.used_narrow_resolution);
  EXPECT_NEAR(result.finest_xy_key_resolution_m, 0.1, 1.0e-12);
  EXPECT_EQ(result.maximum_yaw_bins, 16U);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x, 5.0, 1.0e-6);
  EXPECT_GT(result.full_certifications, 0U);
  ASSERT_EQ((result.trajectory.size() - 1U) % 8U, 0U);
  EXPECT_EQ(result.returned_edge_certificate_confirmations,
            (result.trajectory.size() - 1U) / 8U);
}

TEST(WheelPlanner, EnforcesMinimumOccupancyClearanceOutsideTheRealFootprint) {
  constexpr std::size_t kWidth = 40U;
  constexpr std::size_t kHeight = 20U;
  std::vector<float> occupancy(kWidth * kHeight, 1.0F);
  for (std::size_t y = 8U; y <= 11U; ++y) {
    for (std::size_t x = 0U; x < kWidth; ++x) {
      occupancy[y * kWidth + x] = 0.0F;
    }
  }
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  WheeledCapability capability = Capability(0.4, 0.4);
  capability.minimum_clearance_m = 0.21;
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.4),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 4.0, 2.0, 0.0, Pose(1.0, 2.0)));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
}

TEST(WheelPlanner, AcceptsCorridorWhenMinimumOccupancyClearanceIsMet) {
  constexpr std::size_t kWidth = 40U;
  constexpr std::size_t kHeight = 20U;
  std::vector<float> occupancy(kWidth * kHeight, 1.0F);
  for (std::size_t y = 8U; y <= 11U; ++y) {
    for (std::size_t x = 0U; x < kWidth; ++x) {
      occupancy[y * kWidth + x] = 0.0F;
    }
  }
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  WheeledCapability capability = Capability(0.4, 0.4);
  capability.minimum_clearance_m = 0.19;
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.4),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 4.0, 2.0, 0.0, Pose(1.0, 2.0)));

  ASSERT_TRUE(result.ok()) << result.reason_code;
}

TEST(WheelPlanner, LowClearanceCostUsesDistanceFromTheRealPolygon) {
  constexpr std::size_t kWidth = 40U;
  constexpr std::size_t kHeight = 20U;
  std::vector<float> near_occupancy(kWidth * kHeight, 0.0F);
  std::vector<float> far_occupancy(kWidth * kHeight, 0.0F);
  near_occupancy[6U * kWidth + 6U] = 1.0F;
  far_occupancy[8U * kWidth + 6U] = 1.0F;
  const TerrainFixture near_fixture =
      MakeTerrain(kWidth, kHeight, std::move(near_occupancy));
  const TerrainFixture far_fixture =
      MakeTerrain(kWidth, kHeight, std::move(far_occupancy));
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.4),
  };
  WheelPlanRequest near_request = RequestTo(
      near_fixture, capability, 1.4, 1.0, 0.0, Pose(1.0, 1.0));
  WheelPlanRequest far_request = RequestTo(
      far_fixture, capability, 1.4, 1.0, 0.0, Pose(1.0, 1.0));

  const WheelPlanResult near_result = PlanWheel(near_request);
  const WheelPlanResult far_result = PlanWheel(far_request);

  ASSERT_TRUE(near_result.ok()) << near_result.reason_code;
  ASSERT_TRUE(far_result.ok()) << far_result.reason_code;
  EXPECT_GT(near_result.cost_components[3], far_result.cost_components[3]);
  EXPECT_GT(near_result.cost, far_result.cost);
}

TEST(WheelPlanner, ReportsNoPathWhenChokeIsOneCellNarrowerThanFootprint) {
  constexpr std::size_t kWidth = 40U;
  constexpr std::size_t kHeight = 20U;
  std::vector<float> occupancy(kWidth * kHeight, 1.0F);
  for (std::size_t x = 0U; x < kWidth; ++x) {
    const bool inside_choke = x >= 14U && x <= 20U;
    const std::size_t first_row = inside_choke ? 9U : 8U;
    const std::size_t last_row = inside_choke ? 10U : 11U;
    for (std::size_t y = first_row; y <= last_row; ++y) {
      occupancy[y * kWidth + x] = 0.0F;
    }
  }
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  const WheeledCapability capability = Capability(1.0, 0.6);

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 5.0, 2.0, 0.0, Pose(1.0, 2.0)));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
}

TEST(WheelPlanner, ContinuousSweepDoesNotStepOverAnOccupiedCell) {
  constexpr std::size_t kWidth = 30U;
  constexpr std::size_t kHeight = 20U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    occupancy[y * kWidth + 10U] = 1.0F;
  }
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("long-forward", WheelPrimitiveKind::kForward, 0.4),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 3.0, 2.0, 0.0, Pose(1.0, 2.0)));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
  EXPECT_GT(result.metrics.edge_validation_evaluations, 0U);
}

TEST(WheelPlanner, EvaluatesEachExpandedStatePrimitiveOnlyOnceAcrossAraRounds) {
  const TerrainFixture fixture = FlatTerrain(20U, 20U);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.4),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 1.8, 2.0, 0.0, Pose(1.0, 2.0)));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  // One exact start-pose validation plus two unique (state, primitive) edges.
  EXPECT_EQ(result.metrics.edge_validation_evaluations, 3U);
  EXPECT_GE(result.edge_validation_cache_hits, 1U);
  EXPECT_GE(result.metrics.expanded_states, 2U);
  EXPECT_EQ(result.returned_edge_certificate_confirmations, 2U);
}

TEST(WheelPlanner, RetainsContinuousStatesInsideOneQuantizedBucket) {
  const TerrainFixture fixture = FlatTerrain(20U, 20U);
  WheeledCapability short_first = Capability(0.2, 0.2);
  short_first.motion_primitives = {
      Primitive("a-short", WheelPrimitiveKind::kForward, 0.21),
      Primitive("b-long", WheelPrimitiveKind::kForward, 0.29),
  };
  WheeledCapability long_first = short_first;
  long_first.motion_primitives = {
      Primitive("a-long", WheelPrimitiveKind::kForward, 0.29),
      Primitive("b-short", WheelPrimitiveKind::kForward, 0.21),
  };

  WheelPlanRequest first_request = RequestTo(
      fixture, short_first, 1.38, 2.0, 0.0, Pose(1.0, 2.0));
  first_request.search.stop_after_first_solution = false;
  WheelPlanRequest second_request = RequestTo(
      fixture, long_first, 1.38, 2.0, 0.0, Pose(1.0, 2.0));
  second_request.search.stop_after_first_solution = false;

  const WheelPlanResult first = PlanWheel(first_request);
  const WheelPlanResult second = PlanWheel(second_request);

  ASSERT_TRUE(first.ok()) << first.reason_code;
  ASSERT_TRUE(second.ok()) << second.reason_code;
  EXPECT_GT(first.quantized_state_reuses, 0U);
  EXPECT_GT(second.quantized_state_reuses, 0U);
  EXPECT_GE(first.quantized_state_count, 8U);
  EXPECT_GE(second.quantized_state_count, 8U);
  EXPECT_NEAR(first.trajectory.back().pose.position_m.x, 1.38, 1.0e-9);
  EXPECT_NEAR(second.trajectory.back().pose.position_m.x, 1.38, 1.0e-9);
  for (const WheelPlanResult* result : {&first, &second}) {
    for (std::size_t index = 1U; index < result->trajectory.size(); ++index) {
      const Pose3& source = result->trajectory[index - 1U].pose;
      const Pose3& target = result->trajectory[index].pose;
      const double dx = target.position_m.x - source.position_m.x;
      EXPECT_GT(dx, 0.0);
      EXPECT_LE(dx, 0.20 + 1.0e-9);
      EXPECT_NEAR(target.position_m.y, source.position_m.y, 1.0e-9);
      EXPECT_NEAR(TrajectoryYaw(result->trajectory[index]), 0.0, 1.0e-9);
    }
  }
}

TEST(WheelPlanner, DoesNotStretchAShortPrimitiveToACanonicalBucketCenter) {
  const TerrainFixture fixture = FlatTerrain(20U, 20U);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("short", WheelPrimitiveKind::kForward, 0.11),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 1.2, 2.0, 0.0, Pose(1.0, 2.0)));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_GE(result.trajectory.size(), 17U);
  EXPECT_NEAR(result.trajectory[8U].pose.position_m.x, 1.11, 1.0e-9);
  EXPECT_NEAR(result.trajectory[8U].pose.position_m.y, 2.0, 1.0e-9);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x, 1.2, 1.0e-9);
}

TEST(WheelPlanner, AcceptsImmutablePrimitiveEndpointInsideXYGoalRegion) {
  const TerrainFixture fixture = FlatTerrain(20U, 20U);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("short", WheelPrimitiveKind::kForward, 0.11),
  };
  WheelPlanRequest request = RequestTo(
      fixture, capability, 1.15, 2.0, 0.0, Pose(1.0, 2.0));
  std::get<PointGoal>(request.goals_odom.goals_odom.front().target)
      .tolerance_m = 0.05;

  const WheelPlanResult result = PlanWheel(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x, 1.11, 1.0e-9);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.y, 2.0, 1.0e-9);
  EXPECT_LE(std::hypot(result.trajectory.back().pose.position_m.x - 1.15,
                       result.trajectory.back().pose.position_m.y - 2.0),
            0.05 + 1.0e-12);
}

TEST(WheelPlanner, RetainsPhysicalPoseWhenItsQuantizedBandDiffers) {
  constexpr std::size_t kWidth = 20U;
  constexpr std::size_t kHeight = 20U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  occupancy[5U * kWidth + 3U] = 1.0F;
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.19),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 1.2, 1.0, 0.0, Pose(1.0, 1.0)));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_NEAR(result.trajectory[8U].pose.position_m.x, 1.19, 1.0e-9);
}

TEST(WheelPlanner, KeepsRequestStartAsOnlyStateForSubResolutionReturn) {
  const TerrainFixture fixture = FlatTerrain(20U, 20U);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("reverse-to-key", WheelPrimitiveKind::kReverse, -0.04),
      Primitive("switch", WheelPrimitiveKind::kStopAndSwitch, 0.0),
  };
  WheelPlanRequest request = RequestTo(
      fixture, capability, 2.0, 2.0, 0.0, Pose(1.04, 2.0));
  const WheelPlanResult result = PlanWheel(request);

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
  EXPECT_LE(result.quantized_state_count, 2U);
}

TEST(WheelPlanner, ContinuousLabelsReachGoalWithoutEndpointReplacement) {
  const TerrainFixture fixture = FlatTerrain(30U, 20U);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("short", WheelPrimitiveKind::kForward, 0.21),
      Primitive("long", WheelPrimitiveKind::kForward, 0.29),
  };
  WheelPlanRequest request = RequestTo(
      fixture, capability, 2.0, 2.0, 0.0, Pose(1.0, 2.0));
  const WheelPlanResult result = PlanWheel(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x, 2.0, 1.0e-9);
}

TEST(WheelPlanner, NarrowKeysReduceCoarseEndpointReuse) {
  constexpr std::size_t kWidth = 20U;
  constexpr std::size_t kHeight = 20U;
  const TerrainFixture wide = FlatTerrain(kWidth, kHeight);
  std::vector<float> occupancy(kWidth * kHeight, 1.0F);
  for (std::size_t y = 8U; y <= 11U; ++y) {
    for (std::size_t x = 0U; x < kWidth; ++x) {
      occupancy[y * kWidth + x] = 0.0F;
    }
  }
  const TerrainFixture narrow =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("short", WheelPrimitiveKind::kForward, 0.21),
      Primitive("long", WheelPrimitiveKind::kForward, 0.29),
  };

  const WheelPlanResult coarse_result = PlanWheel(RequestTo(
      wide, capability, 1.38, 2.0, 0.0, Pose(1.0, 2.0)));
  const WheelPlanResult narrow_result = PlanWheel(RequestTo(
      narrow, capability, 1.38, 2.0, 0.0, Pose(1.0, 2.0)));

  ASSERT_TRUE(coarse_result.ok()) << coarse_result.reason_code;
  ASSERT_TRUE(narrow_result.ok()) << narrow_result.reason_code;
  EXPECT_GT(coarse_result.quantized_endpoint_aliases,
            narrow_result.quantized_endpoint_aliases);
  EXPECT_EQ(narrow_result.maximum_yaw_bins, 16U);
}

TEST(WheelPlanner, RejectsAHeightDiscontinuityAboveTheSlopeDerivedStepLimit) {
  constexpr std::size_t kWidth = 30U;
  constexpr std::size_t kHeight = 20U;
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    for (std::size_t x = 10U; x < kWidth; ++x) {
      elevation[y * kWidth + x] = 0.12F;
    }
  }
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::vector<float>(kWidth * kHeight, 0.0F), 0.2,
      std::move(elevation));
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.maximum_slope_rad = 0.5;
  capability.maximum_local_obstacle_relief_m = 1.0;
  capability.minimum_underbody_clearance_m = 1.0;
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 3.0, 2.0, 0.0, Pose(1.0, 2.0)));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
}

TEST(WheelPlanner, AcceptsAContinuousSlopeDespiteLargeTotalHeightChange) {
  constexpr std::size_t kWidth = 40U;
  constexpr std::size_t kHeight = 20U;
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    for (std::size_t x = 0U; x < kWidth; ++x) {
      elevation[y * kWidth + x] = static_cast<float>(x) * 0.04F;
    }
  }
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::vector<float>(kWidth * kHeight, 0.0F), 0.2,
      std::move(elevation));
  WheeledCapability capability = Capability(1.0, 0.4);
  capability.maximum_slope_rad = 0.3;
  capability.maximum_local_obstacle_relief_m = 0.03;
  capability.minimum_underbody_clearance_m = 0.03;
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };

  WheelPlanRequest request = RequestTo(
      fixture, capability, 3.0, 2.0, 0.0, Pose(1.0, 2.0));
  std::get<PointGoal>(request.goals_odom.goals_odom.front().target).position_m.z =
      fixture.map->SampleElevationBilinear(Vec2{.x = 3.0, .y = 2.0})
          .value();
  request.start.pose.position_m.z =
      fixture.map->SampleElevationBilinear(Vec2{.x = 1.0, .y = 2.0})
          .value();
  const WheelPlanResult result = PlanWheel(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
}

TEST(WheelPlanner, RejectsALocalBumpAboveUnderbodyClearance) {
  constexpr std::size_t kWidth = 30U;
  constexpr std::size_t kHeight = 20U;
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  elevation[10U * kWidth + 10U] = 0.20F;
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::vector<float>(kWidth * kHeight, 0.0F), 0.2,
      std::move(elevation));
  WheeledCapability capability = Capability(0.8, 0.4);
  capability.maximum_slope_rad = 1.4;
  capability.maximum_local_obstacle_relief_m = 1.0;
  capability.minimum_underbody_clearance_m = 0.10;
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 3.0, 2.1, 0.0, Pose(1.0, 2.1)));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
}

TEST(WheelPlanner, SpinSweepChecksIntermediateRectangleCorners) {
  constexpr std::size_t kWidth = 50U;
  constexpr std::size_t kHeight = 50U;
  constexpr double kResolution = 0.1;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  occupancy[23U * kWidth + 22U] = 1.0F;
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::move(occupancy), kResolution);
  WheeledCapability capability = Capability(0.8, 0.2);
  capability.motion_primitives = {
      Primitive("spin", WheelPrimitiveKind::kSpinCounterclockwise, 0.0, 0.0,
                std::numbers::pi / 2.0),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 2.0, 2.0, std::numbers::pi / 2.0,
      Pose(2.0, 2.0)));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
}

TEST(WheelPlanner, ArcSweepFollowsTheCurvedCenterlineAndChecksCorners) {
  constexpr std::size_t kWidth = 50U;
  constexpr std::size_t kHeight = 50U;
  constexpr double kResolution = 0.1;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  occupancy[23U * kWidth + 27U] = 1.0F;
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::move(occupancy), kResolution);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      ArcPrimitive("left", WheelPrimitiveKind::kForwardArc, 1.0,
                   std::numbers::pi / 2.0),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 3.0, 3.0, std::numbers::pi / 2.0,
      Pose(2.0, 2.0)));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
}

TEST(WheelPlanner, ReverseArcSweepFollowsTheCurvedCenterline) {
  constexpr std::size_t kWidth = 60U;
  constexpr std::size_t kHeight = 60U;
  constexpr double kResolution = 0.1;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  occupancy[27U * kWidth + 22U] = 1.0F;
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::move(occupancy), kResolution);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      ArcPrimitive("reverse-left", WheelPrimitiveKind::kReverseArc, -1.0,
                   std::numbers::pi / 2.0),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 2.0, 2.0, std::numbers::pi / 2.0,
      Pose(3.0, 3.0)));

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
}

TEST(WheelPlanner, StopsLargeInnerSweepBatchWhenClockCrossesDeadline) {
  constexpr std::size_t kWidth = 120U;
  constexpr std::size_t kHeight = 120U;
  constexpr double kResolution = 0.05;
  const TerrainFixture fixture = MakeTerrain(
      kWidth, kHeight, std::vector<float>(kWidth * kHeight, 0.0F),
      kResolution);
  WheeledCapability capability = Capability(3.0, 3.0);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };
  WheelPlanRequest request = RequestTo(
      fixture, capability, 4.0, 3.0, 0.0, Pose(3.0, 3.0));
  const auto deadline = SteadyClock::time_point{} + 1s;
  std::size_t clock_calls = 0U;
  request.control.deadline = deadline;
  request.control.now = [&] {
    ++clock_calls;
    return clock_calls < 5U ? deadline - 1ms : deadline + 1ms;
  };

  const WheelPlanResult result = PlanWheel(request);

  EXPECT_EQ(result.status, LocalPlanStatus::kTimedOut) << result.reason_code;
  EXPECT_LT(result.sweep_cell_checks, 1024U);
  EXPECT_GE(clock_calls, 5U);
}

TEST(WheelPlanner, ParameterizesExecutableVelocityAndAccelerationTiming) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.maximum_acceleration_mps2 = 0.5;
  capability.maximum_braking_deceleration_mps2 = 0.5;
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.4),
  };

  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 1.4, 1.0, 0.0, Pose(1.0, 1.0)));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_GT(result.trajectory.size(), 2U);
  EXPECT_EQ(result.trajectory.front().time_from_start,
            std::chrono::nanoseconds{0});
  bool saw_motion = false;
  for (std::size_t index = 1U; index < result.trajectory.size(); ++index) {
    EXPECT_GT(result.trajectory[index].time_from_start,
              result.trajectory[index - 1U].time_from_start);
    saw_motion = saw_motion ||
                 std::hypot(result.trajectory[index].velocity.linear_mps.x,
                            result.trajectory[index].velocity.linear_mps.y) >
                     1.0e-6;
  }
  EXPECT_TRUE(saw_motion);
  EXPECT_NEAR(result.trajectory.back().velocity.linear_mps.x, 0.0, 1.0e-9);
  EXPECT_NEAR(result.trajectory.back().velocity.angular_radps.z, 0.0,
              1.0e-9);
  EXPECT_GT(result.trajectory.back().time_from_start, 1s);
}

TEST(WheelPlanner, CertifiesPreferredWallChainInOnePlanningRequest) {
  constexpr std::size_t kWidth = 70U;
  constexpr std::size_t kHeight = 50U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  for (std::size_t y = 14U; y <= 25U; ++y) {
    occupancy[y * kWidth + 30U] = 1.0F;
  }
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  const WheeledCapability capability = ProjectWheelCapability();
  WheelPlanRequest request = RequestTo(
      fixture, capability, 10.0, 4.0, 0.0, Pose(2.0, 4.0));
  request.control.deadline = SteadyClock::now() + 1s;
  request.search.stop_after_first_solution = true;

  const WheelPlanResult result = PlanWheel(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_TRUE(result.has_certified_preferred_candidate);
  EXPECT_EQ(result.metrics.expanded_states, 0U);
  EXPECT_EQ(result.quantized_state_count, 1U);
  ASSERT_EQ(result.trajectory.size(),
            8U * result.preferred_candidate_certified_edge_count + 1U);
  for (std::size_t edge = 0U;
       edge < result.preferred_candidate_full_primitive_edge_count; ++edge) {
    EXPECT_TRUE(MatchesAnyFullPrimitive(result.trajectory[8U * edge].pose,
                                        result.trajectory[8U * (edge + 1U)].pose,
                                        capability));
  }
}

TEST(WheelPlanner, BuildsCertifiedPreferredCandidateAroundOccupiedWall) {
  constexpr std::size_t kWidth = 70U;
  constexpr std::size_t kHeight = 50U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  for (std::size_t y = 14U; y <= 25U; ++y) {
    occupancy[y * kWidth + 30U] = 1.0F;
  }
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  const WheeledCapability capability = ProjectWheelCapability();
  WheelPlanRequest request = RequestTo(
      fixture, capability, 10.0, 4.0, 0.0, Pose(2.0, 4.0));
  request.control.deadline = SteadyClock::now() + 100ms;
  request.search.stop_after_first_solution = true;

  const WheelPlanResult result = PlanWheel(request);

  EXPECT_TRUE(result.has_certified_preferred_candidate);
  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.metrics.expanded_states, 0U);
  EXPECT_EQ(result.quantized_state_count, 1U);
  EXPECT_GT(result.preferred_candidate_full_primitive_edge_count, 0U);
  EXPECT_LE(result.preferred_candidate_terminal_connector_edge_count, 1U);
  EXPECT_EQ(result.preferred_candidate_certified_edge_count,
            result.preferred_candidate_full_primitive_edge_count +
                result.preferred_candidate_terminal_connector_edge_count);
  EXPECT_TRUE(std::isfinite(result.preferred_candidate_cost));
  EXPECT_GT(result.preferred_candidate_cost, 0.0);
  ASSERT_EQ(result.trajectory.size(),
            8U * result.preferred_candidate_certified_edge_count + 1U);
  for (std::size_t edge = 0U;
       edge < result.preferred_candidate_full_primitive_edge_count; ++edge) {
    EXPECT_TRUE(MatchesAnyFullPrimitive(result.trajectory[8U * edge].pose,
                                        result.trajectory[8U * (edge + 1U)].pose,
                                        capability));
  }
}

TEST(WheelPlanner, BuildsCertifiedPreferredCandidateForMoreThan300Meters) {
  constexpr std::size_t kWidth = 2500U;
  constexpr std::size_t kHeight = 1000U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  for (std::size_t y = 350U; y <= 650U; ++y) {
    occupancy[y * kWidth + 950U] = 1.0F;
  }
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  const WheeledCapability capability = ProjectWheelCapability();
  WheelPlanRequest request = RequestTo(
      fixture, capability, 360.0, 100.0, 0.0, Pose(20.0, 100.0));
  request.control.deadline = SteadyClock::now() + 5s;
  request.search.stop_after_first_solution = true;

  const WheelPlanResult result = PlanWheel(request);

  EXPECT_TRUE(result.has_certified_preferred_candidate);
  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.metrics.expanded_states, 0U);
  EXPECT_EQ(result.quantized_state_count, 1U);
  EXPECT_GT(result.preferred_candidate_full_primitive_edge_count, 1500U);
  EXPECT_LE(result.preferred_candidate_terminal_connector_edge_count, 1U);
  EXPECT_EQ(result.preferred_candidate_certified_edge_count,
            result.preferred_candidate_full_primitive_edge_count +
                result.preferred_candidate_terminal_connector_edge_count);
  EXPECT_TRUE(std::isfinite(result.preferred_candidate_cost));
  EXPECT_GT(result.preferred_candidate_cost, 0.0);
  ASSERT_EQ(result.trajectory.size(),
            8U * result.preferred_candidate_certified_edge_count + 1U);
  for (std::size_t edge = 0U;
       edge < result.preferred_candidate_full_primitive_edge_count; ++edge) {
    EXPECT_TRUE(MatchesAnyFullPrimitive(result.trajectory[8U * edge].pose,
                                        result.trajectory[8U * (edge + 1U)].pose,
                                        capability));
  }
}

TEST(WheelPlanner, ProjectWheelCapabilityDetoursAroundAnOccupiedWall) {
  constexpr std::size_t kWidth = 70U;
  constexpr std::size_t kHeight = 50U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  for (std::size_t y = 14U; y <= 25U; ++y) {
    occupancy[y * kWidth + 30U] = 1.0F;
  }
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  const WheeledCapability capability = ProjectWheelCapability();
  WheelPlanRequest request = RequestTo(
      fixture, capability, 10.0, 4.0, 0.0, Pose(2.0, 4.0));
  request.control.deadline = SteadyClock::now() + 1s;
  request.search.stop_after_first_solution = true;

  const WheelPlanResult result = PlanWheel(request);

  ASSERT_TRUE(result.ok()) << result.reason_code
                           << " expanded=" << result.metrics.expanded_states
                           << " validations="
                           << result.metrics.edge_validation_evaluations
                           << " states=" << result.quantized_state_count
                           << " sweep_cells=" << result.sweep_cell_checks;
  EXPECT_TRUE(result.has_certified_preferred_candidate);
  EXPECT_EQ(result.metrics.expanded_states, 0U);
  EXPECT_EQ(result.quantized_state_count, 1U);
  EXPECT_LE(result.start_heuristic_lower_bound, result.cost + 1.0e-9);
  ASSERT_FALSE(result.trajectory.empty());
  bool leaves_direct_corridor = false;
  for (const TrajectoryPoint& point : result.trajectory) {
    leaves_direct_corridor =
        leaves_direct_corridor || std::abs(point.pose.position_m.y - 4.0) > 1.3;
  }
  EXPECT_TRUE(leaves_direct_corridor);
}

TEST(WheelPlanner, ProjectWheelCapabilityDetoursForMoreThan300Meters) {
  constexpr std::size_t kWidth = 2500U;   // 500 m at 0.2 m resolution.
  constexpr std::size_t kHeight = 1000U;  // 200 m at 0.2 m resolution.
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  for (std::size_t y = 350U; y <= 650U; ++y) {
    occupancy[y * kWidth + 950U] = 1.0F;  // x=190 m, y=[70, 130] m.
  }
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  const WheeledCapability capability = ProjectWheelCapability();
  WheelPlanRequest request = RequestTo(
      fixture, capability, 360.0, 100.0, 0.0, Pose(20.0, 100.0));
  request.control.deadline = SteadyClock::now() + 30s;
  request.search.stop_after_first_solution = true;

  const SteadyClock::time_point planning_started = SteadyClock::now();
  const WheelPlanResult result = PlanWheel(request);
  const auto planning_elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          SteadyClock::now() - planning_started);
  RecordProperty("plan_wheel_elapsed_ms", planning_elapsed.count());

  ASSERT_TRUE(result.ok()) << result.reason_code
                           << " expanded=" << result.metrics.expanded_states
                           << " validations="
                           << result.metrics.edge_validation_evaluations
                           << " states=" << result.quantized_state_count
                           << " sweep_cells=" << result.sweep_cell_checks;
  EXPECT_TRUE(result.has_certified_preferred_candidate);
  EXPECT_EQ(result.metrics.expanded_states, 0U);
  EXPECT_EQ(result.quantized_state_count, 1U);
  EXPECT_LT(planning_elapsed, 1s);
  ASSERT_FALSE(result.trajectory.empty());
  EXPECT_GT(result.trajectory.back().pose.position_m.x -
                result.trajectory.front().pose.position_m.x,
            300.0);
  bool leaves_direct_corridor = false;
  for (const TrajectoryPoint& point : result.trajectory) {
    leaves_direct_corridor =
        leaves_direct_corridor || std::abs(point.pose.position_m.y - 100.0) >
                                     30.0;
  }
  EXPECT_TRUE(leaves_direct_corridor);
}

TEST(WheelPlanner,
     GlobalRouteWithEightMeterRollingHorizonReaches750MeterGoalThroughRandomObstacles) {
  const WheelLongRangeFixture fixture;
  const WheelLongRangeFixture::RunRecord record =
      fixture.Run750MeterRollingScenario();

  ASSERT_TRUE(record.success)
      << "failed_segment=" << record.failed_segment
      << " failure_stage=" << record.failure_stage
      << " reason_code=" << record.failure_reason_code
      << " max_cycle_elapsed_ms=" << record.max_cycle_elapsed_ms;
  EXPECT_LT(record.rolling_segments, 320U);
  EXPECT_EQ(record.cycles_at_least_three_seconds, 0U);
  EXPECT_EQ(record.cycles_under_one_second +
                record.cycles_one_to_two_seconds +
                record.cycles_two_to_three_seconds +
                record.cycles_at_least_three_seconds,
            record.rolling_segments);
  EXPECT_NEAR(record.final_pose.position_m.x, 800.0, 0.2);
  EXPECT_NEAR(record.final_pose.position_m.y, 500.0, 0.2);
}

TEST(WheelPlanner, InitialVelocityShortensTheFirstExecutableProfileAndCost) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.maximum_acceleration_mps2 = 0.5;
  capability.maximum_braking_deceleration_mps2 = 0.5;
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.4),
  };
  WheelPlanRequest stopped = RequestTo(
      fixture, capability, 1.4, 1.0, 0.0, Pose(1.0, 1.0));
  WheelPlanRequest moving = stopped;
  moving.start.velocity.linear_mps.x = 0.2;

  const WheelPlanResult stopped_result = PlanWheel(stopped);
  const WheelPlanResult moving_result = PlanWheel(moving);

  ASSERT_TRUE(stopped_result.ok()) << stopped_result.reason_code;
  ASSERT_TRUE(moving_result.ok()) << moving_result.reason_code;
  EXPECT_LT(moving_result.trajectory.back().time_from_start,
            stopped_result.trajectory.back().time_from_start);
  EXPECT_LT(moving_result.cost, stopped_result.cost);
  EXPECT_NEAR(moving_result.trajectory.front().velocity.linear_mps.x, 0.2,
              1.0e-9);
}

TEST(WheelPlanner, ArcProfileRespectsYawAccelerationFromInitialVelocity) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.maximum_forward_speed_mps = 10.0;
  capability.maximum_spin_rate_radps = 10.0;
  capability.maximum_acceleration_mps2 = 10.0;
  capability.maximum_braking_deceleration_mps2 = 10.0;
  capability.maximum_lateral_acceleration_mps2 = 100.0;
  capability.maximum_yaw_acceleration_radps2 = 0.1;
  capability.maximum_curvature_per_m = 3.0;
  capability.motion_primitives = {
      ArcPrimitive("left", WheelPrimitiveKind::kForwardArc, 0.5, 0.8),
  };
  constexpr double kGoalYaw = 0.4;
  WheelPlanRequest request = RequestTo(
      fixture, capability, 1.0 + 0.5 * std::sin(kGoalYaw),
      1.0 + 0.5 * (1.0 - std::cos(kGoalYaw)), kGoalYaw);
  request.start.velocity.linear_mps.x = 0.1;
  request.start.velocity.angular_radps.z = 0.2;

  const WheelPlanResult result = PlanWheel(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  for (std::size_t index = 1U; index < result.trajectory.size(); ++index) {
    const double delta_time = std::chrono::duration<double>(
                                  result.trajectory[index].time_from_start -
                                  result.trajectory[index - 1U].time_from_start)
                                  .count();
    ASSERT_GT(delta_time, 0.0);
    const double yaw_acceleration =
        std::abs(result.trajectory[index].velocity.angular_radps.z -
                 result.trajectory[index - 1U].velocity.angular_radps.z) /
        delta_time;
    EXPECT_LE(yaw_acceleration,
              capability.maximum_yaw_acceleration_radps2 + 1.0e-6);
  }
}

TEST(WheelPlanner, DoesNotInstantlyFlipAnOpposingInitialVelocity) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.4),
  };
  WheelPlanRequest request = RequestTo(
      fixture, capability, 1.4, 1.0, 0.0, Pose(1.0, 1.0));
  request.start.velocity.linear_mps.x = -0.2;

  const WheelPlanResult result = PlanWheel(request);

  EXPECT_EQ(result.status, LocalPlanStatus::kNoPath) << result.reason_code;
  EXPECT_GT(result.full_certifications, 0U);
  EXPECT_GT(result.full_invalidations, 0U);
}

TEST(WheelPlanner, RejectsReturnedCertificateWhenRequestIdentityChanges) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.4),
  };

  std::size_t baseline_reads = 0U;
  WheelPlanRequest baseline = RequestTo(
      fixture, capability, 1.4, 1.0, 0.0, Pose(1.0, 1.0));
  baseline.local_source_sequence = 7U;
  baseline.local_terrain_semantics_id = 13U;
  baseline.capability_fingerprint = 11U;
  baseline.control.deadline = SteadyClock::time_point::max();
  baseline.control.now = [&] {
    ++baseline_reads;
    return SteadyClock::time_point{};
  };
  const WheelPlanResult baseline_result = PlanWheel(baseline);
  ASSERT_TRUE(baseline_result.ok()) << baseline_result.reason_code;
  ASSERT_EQ(baseline_result.returned_edge_certificate_confirmations, 1U);
  ASSERT_GT(baseline_result.trajectory.size(), 1U);
  ASSERT_EQ((baseline_result.trajectory.size() - 1U) % 8U, 0U);
  const std::size_t reconstructed_edges =
      (baseline_result.trajectory.size() - 1U) / 8U;
  const std::size_t reconstruction_reads = 3U + 9U * reconstructed_edges;
  ASSERT_GT(baseline_reads, reconstruction_reads);
  const std::size_t first_reconstruction_read =
      baseline_reads - reconstruction_reads;

  WheelPlanRequest changed = RequestTo(
      fixture, capability, 1.4, 1.0, 0.0, Pose(1.0, 1.0));
  changed.local_source_sequence = 7U;
  changed.local_terrain_semantics_id = 13U;
  changed.capability_fingerprint = 11U;
  changed.control.deadline = SteadyClock::time_point::max();
  std::size_t changed_reads = 0U;
  changed.control.now = [&] {
    if (changed_reads++ == first_reconstruction_read) {
      changed.local_source_sequence = 8U;
    }
    return SteadyClock::time_point{};
  };

  const WheelPlanResult result = PlanWheel(changed);

  EXPECT_EQ(result.status, LocalPlanStatus::kPlannerError);
  EXPECT_EQ(result.reason_code, "WHEEL_CERTIFIED_EDGE_MISSING");
  EXPECT_EQ(result.returned_edge_certificate_confirmations, 0U);

  WheelPlanRequest changed_semantics = RequestTo(
      fixture, capability, 1.4, 1.0, 0.0, Pose(1.0, 1.0));
  changed_semantics.local_source_sequence = 7U;
  changed_semantics.local_terrain_semantics_id = 13U;
  changed_semantics.capability_fingerprint = 11U;
  changed_semantics.control.deadline = SteadyClock::time_point::max();
  std::size_t semantics_reads = 0U;
  changed_semantics.control.now = [&] {
    if (semantics_reads++ == first_reconstruction_read) {
      changed_semantics.local_terrain_semantics_id = 14U;
    }
    return SteadyClock::time_point{};
  };

  const WheelPlanResult semantics_result = PlanWheel(changed_semantics);

  EXPECT_EQ(semantics_result.status, LocalPlanStatus::kPlannerError);
  EXPECT_EQ(semantics_result.reason_code, "WHEEL_CERTIFIED_EDGE_MISSING");
  EXPECT_EQ(semantics_result.returned_edge_certificate_confirmations, 0U);
}

TEST(WheelPlanner, SpinProfileContinuesInitialYawVelocityWithinAcceleration) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.maximum_yaw_acceleration_radps2 = 0.1;
  capability.motion_primitives = {
      Primitive("spin", WheelPrimitiveKind::kSpinCounterclockwise, 0.0, 0.0,
                0.8),
  };
  WheelPlanRequest request = RequestTo(
      fixture, capability, 1.0, 1.0, 0.4, Pose(1.0, 1.0));
  request.start.velocity.angular_radps.z = 0.2;

  const WheelPlanResult result = PlanWheel(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  for (std::size_t index = 1U; index < result.trajectory.size(); ++index) {
    const double delta_time = std::chrono::duration<double>(
                                  result.trajectory[index].time_from_start -
                                  result.trajectory[index - 1U].time_from_start)
                                  .count();
    ASSERT_GT(delta_time, 0.0);
    EXPECT_LE(std::abs(
                  result.trajectory[index].velocity.angular_radps.z -
                  result.trajectory[index - 1U].velocity.angular_radps.z) /
                  delta_time,
              capability.maximum_yaw_acceleration_radps2 + 1.0e-6);
  }
}

TEST(WheelPlanner, RejectsNonFiniteInitialVelocityAsStructuralInput) {
  const TerrainFixture fixture = FlatTerrain();
  const WheeledCapability capability = Capability(0.2, 0.2);
  WheelPlanRequest request = RequestTo(
      fixture, capability, 1.4, 1.0, 0.0, Pose(1.0, 1.0));
  request.start.velocity.angular_radps.z =
      std::numeric_limits<double>::quiet_NaN();

  const WheelPlanResult result = PlanWheel(request);

  EXPECT_EQ(result.status, LocalPlanStatus::kInvalidInput);
}

TEST(WheelPlanner, ExecutionTimeCostMatchesExecutableReferenceDuration) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      ArcPrimitive("left", WheelPrimitiveKind::kForwardArc, 0.5, 0.6),
  };
  WheelPlanRequest request = RequestTo(
      fixture, capability, 1.0 + 0.5 * std::sin(0.3),
      1.0 + 0.5 * (1.0 - std::cos(0.3)), 0.3);

  const WheelPlanResult result = PlanWheel(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const double duration_s = std::chrono::duration<double>(
                                result.trajectory.back().time_from_start)
                                .count();
  EXPECT_NEAR(result.cost_components[1], duration_s, 1.0e-6);
  double expected_cost = 0.0;
  for (std::size_t component = 0U; component < result.cost_components.size();
       ++component) {
    expected_cost +=
        result.cost_components[component] / result.cost_scales[component];
  }
  EXPECT_NEAR(result.cost, expected_cost, 1.0e-6);
}

TEST(WheelPlanner, CostScalesAreFixedByPlatformNotMapResolution) {
  const TerrainFixture coarse = FlatTerrain(40U, 20U);
  constexpr std::size_t kFineWidth = 80U;
  constexpr std::size_t kFineHeight = 40U;
  const TerrainFixture fine = MakeTerrain(
      kFineWidth, kFineHeight,
      std::vector<float>(kFineWidth * kFineHeight, 0.0F), 0.1);
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.4),
  };

  const WheelPlanResult coarse_result = PlanWheel(RequestTo(
      coarse, capability, 1.4, 1.0, 0.0, Pose(1.0, 1.0)));
  const WheelPlanResult fine_result = PlanWheel(RequestTo(
      fine, capability, 1.4, 1.0, 0.0, Pose(1.0, 1.0)));

  ASSERT_TRUE(coarse_result.ok()) << coarse_result.reason_code;
  ASSERT_TRUE(fine_result.ok()) << fine_result.reason_code;
  EXPECT_EQ(coarse_result.cost_scales, fine_result.cost_scales);
}

TEST(WheelPlanner, UsesStopSwitchBeforeAReverseArcWhenModeRequiresIt) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.4),
      Primitive("switch", WheelPrimitiveKind::kStopAndSwitch, 0.0),
      ArcPrimitive("reverse-left", WheelPrimitiveKind::kReverseArc, -1.0,
                   0.6),
  };
  constexpr double kYaw = 0.3;
  const WheelPlanResult result = PlanWheel(RequestTo(
      fixture, capability, 1.4 - std::sin(kYaw),
      1.0 - (1.0 - std::cos(kYaw)), kYaw));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_GE(result.mode_switch_edge_count, 1U);
  EXPECT_GE(result.reverse_edge_count, 1U);
}

TEST(WheelPlanner, DuplicatePrimitiveIdsRemainOrderIndependent) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability first = Capability(0.2, 0.2);
  first.motion_primitives = {
      Primitive("duplicate", WheelPrimitiveKind::kForward, 0.21),
      Primitive("duplicate", WheelPrimitiveKind::kForward, 0.29),
  };
  WheeledCapability second = first;
  std::reverse(second.motion_primitives.begin(), second.motion_primitives.end());

  const WheelPlanResult first_result = PlanWheel(RequestTo(
      fixture, first, 1.6, 1.0, 0.0, Pose(1.0, 1.0)));
  const WheelPlanResult second_result = PlanWheel(RequestTo(
      fixture, second, 1.6, 1.0, 0.0, Pose(1.0, 1.0)));

  ASSERT_TRUE(first_result.ok()) << first_result.reason_code;
  ASSERT_TRUE(second_result.ok()) << second_result.reason_code;
  EXPECT_NEAR(first_result.cost, second_result.cost, 1.0e-9);
  EXPECT_EQ(first_result.quantized_state_count,
            second_result.quantized_state_count);
}

TEST(WheelPlanner, UsesGoalTolerancesButStillReturnsTheExactPointAndYaw) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      ArcPrimitive("left", WheelPrimitiveKind::kForwardArc, 1.0, 0.6),
  };
  constexpr double kGoalYaw = 0.3;
  WheelPlanRequest request = RequestTo(
      fixture, capability, 1.0 + std::sin(kGoalYaw),
      2.0 - std::cos(kGoalYaw), kGoalYaw);
  std::get<PointGoal>(request.goals_odom.goals_odom.front().target)
      .tolerance_m = 0.05;
  request.goals_odom.goals_odom.front().yaw_tolerance_rad = 0.1;

  const WheelPlanResult result = PlanWheel(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x,
              1.0 + std::sin(kGoalYaw), 1.0e-9);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.y,
              2.0 - std::cos(kGoalYaw), 1.0e-9);
  EXPECT_NEAR(TrajectoryYaw(result.trajectory.back()), kGoalYaw, 1.0e-9);
}

TEST(WheelPlanner, ConnectsInsideBothPositionAndYawToleranceIntervals) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      ArcPrimitive("left", WheelPrimitiveKind::kForwardArc, 1.0, 0.4),
  };
  constexpr double kPositionCenterYaw = 0.18;
  WheelPlanRequest request = RequestTo(
      fixture, capability, 1.0 + std::sin(kPositionCenterYaw),
      2.0 - std::cos(kPositionCenterYaw), 0.2);
  std::get<PointGoal>(request.goals_odom.goals_odom.front().target)
      .tolerance_m = 0.03;
  request.goals_odom.goals_odom.front().yaw_tolerance_rad = 0.03;

  const WheelPlanResult result = PlanWheel(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const Pose3& endpoint = result.trajectory.back().pose;
  EXPECT_LE(std::hypot(endpoint.position_m.x -
                           (1.0 + std::sin(kPositionCenterYaw)),
                       endpoint.position_m.y -
                           (2.0 - std::cos(kPositionCenterYaw))),
            0.03 + 1.0e-9);
  EXPECT_LE(std::abs(ShortestYawDelta(TrajectoryYaw(result.trajectory.back()),
                                     0.2)),
            0.03 + 1.0e-9);
  EXPECT_GT(std::abs(ShortestYawDelta(TrajectoryYaw(result.trajectory.back()),
                                     0.2)),
            1.0e-6);
}

TEST(WheelPlanner, ConnectsAtANonCenterXYPointInsidePositionTolerance) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.4),
  };
  WheelPlanRequest request = RequestTo(
      fixture, capability, 1.25, 1.03, 0.0, Pose(1.0, 1.0));
  std::get<PointGoal>(request.goals_odom.goals_odom.front().target)
      .tolerance_m = 0.05;

  const WheelPlanResult result = PlanWheel(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const Pose3& endpoint = result.trajectory.back().pose;
  EXPECT_NEAR(endpoint.position_m.x, 1.25, 1.0e-9);
  EXPECT_NEAR(endpoint.position_m.y, 1.0, 1.0e-9);
  EXPECT_LE(std::hypot(endpoint.position_m.x - 1.25,
                       endpoint.position_m.y - 1.03),
            0.05 + 1.0e-9);
  EXPECT_GT(std::hypot(endpoint.position_m.x - 1.25,
                       endpoint.position_m.y - 1.03),
            1.0e-6);
}

TEST(WheelPlanner, AcceptsACertifiedNonCenterStateInsidePointGoalRegion) {
  constexpr std::size_t kWidth = 20U;
  constexpr std::size_t kHeight = 20U;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  occupancy[5U * kWidth + 7U] = 1.0F;
  const TerrainFixture fixture =
      MakeTerrain(kWidth, kHeight, std::move(occupancy));
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };
  WheelPlanRequest request = RequestTo(
      fixture, capability, 1.35, 1.0, 0.0, Pose(1.0, 1.0));
  std::get<PointGoal>(request.goals_odom.goals_odom.front().target)
      .tolerance_m = 0.15;

  const WheelPlanResult result = PlanWheel(request);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_NEAR(result.trajectory.back().pose.position_m.x, 1.2, 1.0e-9);
  EXPECT_NEAR(result.trajectory.back().pose.position_m.y, 1.0, 1.0e-9);
}

TEST(WheelPlanner, MapsImmediateCancellationAndExpiredDeadline) {
  const TerrainFixture fixture = FlatTerrain();
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.2),
  };
  std::stop_source stop;
  stop.request_stop();
  WheelPlanRequest canceled =
      RequestTo(fixture, capability, 2.0, 1.0, 0.0);
  canceled.control.stop_token = stop.get_token();
  const WheelPlanResult canceled_result = PlanWheel(canceled);
  EXPECT_EQ(canceled_result.status, LocalPlanStatus::kCanceled);
  EXPECT_EQ(canceled_result.reason_code, "REQUEST_CANCELED");

  WheelPlanRequest expired =
      RequestTo(fixture, capability, 2.0, 1.0, 0.0);
  expired.control.deadline = SteadyClock::now() - 1ms;
  const WheelPlanResult expired_result = PlanWheel(expired);
  EXPECT_EQ(expired_result.status, LocalPlanStatus::kTimedOut);
  EXPECT_EQ(expired_result.reason_code, "TIMEOUT");
}

TEST(WheelPlanner, SolvesAFullSizeHighBranchingSearchWithoutAFixedStateCap) {
  constexpr std::size_t kSide = 320U;
  const TerrainFixture fixture = MakeTerrain(
      kSide, kSide, std::vector<float>(kSide * kSide, 0.0F));
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward-04", WheelPrimitiveKind::kForward, 0.4),
      Primitive("forward-06", WheelPrimitiveKind::kForward, 0.6),
      Primitive("forward-08", WheelPrimitiveKind::kForward, 0.8),
      Primitive("forward-10", WheelPrimitiveKind::kForward, 1.0),
  };
  WheelPlanRequest request = RequestTo(
      fixture, capability, 62.0, 32.0, 0.0, Pose(2.0, 32.0));
  request.control.deadline = SteadyClock::now() + 950ms;

  const auto begin = SteadyClock::now();
  const WheelPlanResult result = PlanWheel(request);
  const auto elapsed = SteadyClock::now() - begin;

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_LT(elapsed, 1s);
  EXPECT_GT(result.quantized_state_count, 192U);
}

TEST(WheelPlanner, SolvesAFullSizeNarrowCorridorUnderOneSecond) {
  constexpr std::size_t kSide = 320U;
  std::vector<float> occupancy(kSide * kSide, 1.0F);
  for (std::size_t y = 158U; y <= 161U; ++y) {
    for (std::size_t x = 0U; x < kSide; ++x) {
      occupancy[y * kSide + x] = 0.0F;
    }
  }
  const TerrainFixture fixture =
      MakeTerrain(kSide, kSide, std::move(occupancy));
  WheeledCapability capability = Capability(0.2, 0.2);
  capability.motion_primitives = {
      Primitive("forward", WheelPrimitiveKind::kForward, 0.4),
  };
  WheelPlanRequest request = RequestTo(
      fixture, capability, 62.0, 32.0, 0.0, Pose(2.0, 32.0));
  request.control.deadline = SteadyClock::now() + 950ms;

  const auto begin = SteadyClock::now();
  const WheelPlanResult result = PlanWheel(request);
  const auto elapsed = SteadyClock::now() - begin;

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_LT(elapsed, 1s);
  EXPECT_TRUE(result.metrics.used_narrow_resolution);
}

TEST(WheelPlanner, RejectsPlanarRegionWithoutReadingLegacyAdmissionData) {
  const TerrainFixture fixture = FlatTerrain();
  const WheeledCapability capability = Capability();
  WheelPlanRequest request = RequestTo(fixture, capability, 2.0, 1.0, 0.0);
  request.goals_odom.goals_odom.front().target = PlanarRegionGoal{};

  const WheelPlanResult result = PlanWheel(request);

  EXPECT_EQ(result.status, LocalPlanStatus::kInvalidInput);
}

}  // namespace
}  // namespace lunar::pure_planning::wheel
