#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stop_token>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_planner_core/reachability_projection.hpp"
#include "lunar_planner_core/traversability_projection.hpp"
#include "hopper/hop_certifier.hpp"
#include "hopper/landing_region.hpp"
#include "legged/legged_terrain.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning {
namespace {

void SetObstacle(GridMap& map, const std::size_t row,
                 const std::size_t column, const float height_m = 0.1F) {
  const std::size_t index = row * map.width + column;
  std::get<std::vector<std::uint8_t>>(
      map.layers.at("obstacle").values).at(index) = 1U;
  std::get<std::vector<float>>(
      map.layers.at("obstacle_height").values).at(index) = height_m;
}

void SetElevation(GridMap& map, const std::size_t row,
                  const std::size_t column, const float elevation_m) {
  std::get<std::vector<float>>(
      map.layers.at("elevation").values).at(row * map.width + column) =
      elevation_m;
}

void SetKnown(GridMap& map, const std::size_t row,
              const std::size_t column, const bool known) {
  std::get<std::vector<std::uint8_t>>(
      map.layers.at("valid_mask").values).at(row * map.width + column) =
      static_cast<std::uint8_t>(known);
}

std::size_t Index(const GridMap& map, const std::size_t row,
                  const std::size_t column) {
  return row * map.width + column;
}

bool IsKnown(const GridMap& map, const std::size_t row,
             const std::size_t column) {
  return std::get<std::vector<std::uint8_t>>(
             map.layers.at("valid_mask").values)
             .at(Index(map, row, column)) != 0U;
}

void KeepOnlyLandingEvidencePatches(
    GridMap& map,
    const std::vector<std::pair<std::size_t, std::size_t>>& centers) {
  auto& valid = std::get<std::vector<std::uint8_t>>(
      map.layers.at("valid_mask").values);
  std::fill(valid.begin(), valid.end(), 0U);
  for (const auto [center_row, center_column] : centers) {
    for (std::size_t row = center_row - 2U; row <= center_row + 2U; ++row) {
      for (std::size_t column = center_column - 2U;
           column <= center_column + 2U; ++column) {
        valid.at(Index(map, row, column)) = 1U;
      }
    }
  }
}

TEST(ReachabilityProjection, GroundMaskIsTheCppStartConnectedComponent) {
  PlannerInput input = test::MakeValidWheelInput();
  input.world.global_map = test::MakeFlatMap("map", 12U, 8U, 1.0);
  input.world.local_map = test::MakeFlatMap("odom", 12U, 8U, 1.0);
  for (std::size_t row = 0U; row < input.world.local_map.height; ++row) {
    SetObstacle(input.world.local_map, row, 6U);
    SetObstacle(input.world.global_map, row, 6U);
  }
  auto& state = std::get<WheeledState>(input.current_state);
  state.pose.position_m = {2.5, 3.5, 0.0};

  const auto traversability = ProjectTraversability(
      input.world, input.capability, input.config.map_safety, {});
  const auto reachability = ProjectReachability(input, 30.0);

  ASSERT_TRUE(traversability.ok()) << traversability.reason_code;
  ASSERT_TRUE(reachability.ok()) << reachability.reason_code;
  const std::size_t start = Index(input.world.local_map, 3U, 2U);
  const std::int32_t component =
      traversability.projection->connected_component.at(start);
  ASSERT_GE(component, 0);
  ASSERT_EQ(reachability.projection->reachable.size(),
            traversability.projection->connected_component.size());
  for (std::size_t index = 0U;
       index < reachability.projection->reachable.size(); ++index) {
    const bool expected =
        traversability.projection->hard_feasible[index] != 0U &&
        traversability.projection->connected_component[index] == component;
    EXPECT_EQ(reachability.projection->reachable[index] != 0U, expected)
        << "index=" << index;
  }
  EXPECT_EQ(reachability.projection->algorithm_id,
            "cpp-ground-start-connected-component/v1");
  EXPECT_EQ(reachability.projection->candidate_edges_evaluated, 0U);
}

TEST(ReachabilityProjection,
     GroundPhysicalMaskIgnoresPlannerPrimitiveIdentityAndOrder) {
  PlannerInput first = test::MakeValidWheelInput();
  PlannerInput second = first;
  auto& primitives =
      std::get<WheeledCapability>(second.capability).motion_primitives;
  std::reverse(primitives.begin(), primitives.end());
  for (std::size_t index = 0U; index < primitives.size(); ++index) {
    primitives[index].primitive_id =
        "physically-irrelevant-primitive-" + std::to_string(index);
  }

  const auto first_projection = ProjectReachability(first, 30.0);
  const auto second_projection = ProjectReachability(second, 30.0);

  ASSERT_TRUE(first_projection.ok()) << first_projection.reason_code;
  ASSERT_TRUE(second_projection.ok()) << second_projection.reason_code;
  EXPECT_EQ(first_projection.projection->reachable,
            second_projection.projection->reachable);
  EXPECT_EQ(first_projection.projection->algorithm_id,
            second_projection.projection->algorithm_id);
}

TEST(ReachabilityProjection,
     GroundPhysicalMaskRejectsSlopeWheelSupportAndClearance) {
  PlannerInput slope = test::MakeValidWheelInput();
  auto& slope_capability = std::get<WheeledCapability>(slope.capability);
  slope_capability.maximum_slope_rad = 5.0 * std::numbers::pi / 180.0;
  for (std::size_t row = 0U; row < slope.world.global_map.height; ++row) {
    for (std::size_t column = 0U;
         column < slope.world.global_map.width; ++column) {
      SetElevation(slope.world.global_map, row, column,
                   static_cast<float>(0.25 * column));
    }
  }
  const auto slope_projection = ProjectReachability(slope, 30.0);
  ASSERT_TRUE(slope_projection.ok()) << slope_projection.reason_code;
  EXPECT_EQ(slope_projection.projection->reachable[Index(
                slope.world.global_map, 3U, 2U)],
            0U);

  PlannerInput support = test::MakeValidWheelInput();
  support.world.global_map = test::MakeFlatMap("map", 80U, 60U, 0.1);
  support.world.local_map = test::MakeFlatMap("odom", 80U, 60U, 0.1);
  support.config.global_map.base_resolution_m = 0.1;
  support.config.wheel.xy_resolution_m = 0.1;
  support.config.legged.xy_resolution_m = 0.1;
  constexpr std::size_t kWheelTargetRow = 35U;
  constexpr std::size_t kWheelTargetColumn = 45U;
  const auto support_baseline = ProjectReachability(support, 30.0);
  ASSERT_TRUE(support_baseline.ok()) << support_baseline.reason_code;
  ASSERT_NE(support_baseline.projection->reachable[Index(
                support.world.global_map, kWheelTargetRow,
                kWheelTargetColumn)],
            0U);
  // The missing patch lies under the rear-left wheel footprint.  The rover's
  // body-center cell stays known, so this is support evidence rather than a
  // renamed center-cell unknownness check.
  for (std::size_t row = 31U; row <= 32U; ++row) {
    for (std::size_t column = 41U; column <= 42U; ++column) {
      SetKnown(support.world.global_map, row, column, false);
    }
  }
  ASSERT_TRUE(IsKnown(support.world.global_map, kWheelTargetRow,
                      kWheelTargetColumn));
  const auto support_projection = ProjectReachability(support, 30.0);
  ASSERT_TRUE(support_projection.ok()) << support_projection.reason_code;
  EXPECT_EQ(support_projection.projection->reachable[Index(
                support.world.global_map, kWheelTargetRow,
                kWheelTargetColumn)],
            0U);

  PlannerInput clearance = test::MakeValidWheelInput();
  std::get<WheeledCapability>(clearance.capability).minimum_clearance_m = 1.1;
  SetObstacle(clearance.world.global_map, 3U, 4U);
  const auto clearance_projection = ProjectReachability(clearance, 30.0);
  ASSERT_TRUE(clearance_projection.ok()) << clearance_projection.reason_code;
  EXPECT_EQ(clearance_projection.projection->reachable[Index(
                clearance.world.global_map, 3U, 3U)],
            0U);
}

TEST(ReachabilityProjection, LeggedPhysicalMaskRejectsMissingFoothold) {
  PlannerInput foothold = test::MakeValidLeggedInput();
  foothold.world.global_map = test::MakeFlatMap("map", 80U, 60U, 0.1);
  foothold.world.local_map = test::MakeFlatMap("odom", 80U, 60U, 0.1);
  foothold.config.global_map.base_resolution_m = 0.1;
  foothold.config.wheel.xy_resolution_m = 0.1;
  foothold.config.legged.xy_resolution_m = 0.1;
  constexpr std::size_t kLeggedTargetRow = 35U;
  constexpr std::size_t kLeggedTargetColumn = 45U;
  const auto foothold_baseline = ProjectReachability(foothold, 30.0);
  ASSERT_TRUE(foothold_baseline.ok()) << foothold_baseline.reason_code;
  ASSERT_NE(foothold_baseline.projection->reachable[Index(
                foothold.world.global_map, kLeggedTargetRow,
                kLeggedTargetColumn)],
            0U);
  // Remove only the front-right foothold patch at the target pose.  The body
  // center remains observed and physically distinct from the missing support.
  for (std::size_t row = 36U; row <= 37U; ++row) {
    for (std::size_t column = 48U; column <= 49U; ++column) {
      SetKnown(foothold.world.global_map, row, column, false);
    }
  }
  ASSERT_TRUE(IsKnown(foothold.world.global_map, kLeggedTargetRow,
                      kLeggedTargetColumn));
  const auto foothold_projection = ProjectReachability(foothold, 30.0);
  ASSERT_TRUE(foothold_projection.ok()) << foothold_projection.reason_code;
  EXPECT_EQ(foothold_projection.projection->reachable[Index(
                foothold.world.global_map, kLeggedTargetRow,
                kLeggedTargetColumn)],
            0U);
}

TEST(ReachabilityProjection,
     LeggedBodyHeightIntervalRejectsRaisedTerrainEnvelope) {
  PlannerInput input = test::MakeValidLeggedInput();
  auto& capability = std::get<LeggedCapability>(input.capability);
  ASSERT_DOUBLE_EQ(capability.minimum_body_clearance_m, 0.3);
  ASSERT_DOUBLE_EQ(capability.body_height_m.lower, 0.4);
  ASSERT_DOUBLE_EQ(capability.body_height_m.upper, 0.6);
  SetElevation(input.world.global_map, 3U, 4U, 0.4F);
  ASSERT_TRUE(IsKnown(input.world.global_map, 3U, 4U));

  const auto map = shared::MapSnapshot::Create(input.world.global_map);
  ASSERT_TRUE(map.ok()) << map.reason_code;
  const auto safe = shared::BuildSafeProjection(
      map.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(safe.ok()) << safe.reason_code;
  const hierarchical::LocalSearchDomain full_domain{
      input.world.global_map.width, input.world.global_map.height,
      std::vector<std::uint8_t>(
          input.world.global_map.width * input.world.global_map.height, 1U)};
  const legged::LeggedPose pose{
      .position_m = {4.5, 3.5, 0.5},
      .yaw_rad = 0.0,
  };
  const legged::LeggedSweepResult body_height_projection =
      legged::ValidateLeggedBodySweep(
          pose, pose, Interval{.lower = 0.5, .upper = 0.5},
          *safe.projection, capability, full_domain, {});

  EXPECT_FALSE(body_height_projection.valid);
  EXPECT_EQ(body_height_projection.reason_code,
            "LEGGED_START_HEIGHT_INTERVAL_EMPTY");
}

TEST(ReachabilityProjection, HopperCertifiedHopCrossesGroundDisconnectedGap) {
  PlannerInput input = test::MakeValidHopperInput();
  input.world.global_map = test::MakeFlatMap("map", 80U, 16U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 80U, 16U, 0.5);
  input.config.global_map.base_resolution_m = 0.5;
  auto& state = std::get<HopperState>(input.current_state);
  state.pose.position_m = {3.25, 4.25, 0.0};
  for (std::size_t row = 0U; row < 16U; ++row) {
    SetObstacle(input.world.global_map, row, 20U);
  }
  KeepOnlyLandingEvidencePatches(
      input.world.local_map, {{8U, 6U}, {8U, 40U}});

  const auto reachability = ProjectReachability(input, 30.0);

  ASSERT_TRUE(reachability.ok()) << reachability.reason_code;
  EXPECT_EQ(reachability.projection->algorithm_id,
            "cpp-hopper-certified-bidirectional-bfs/v3");
  EXPECT_NE(reachability.projection->reachable[
                Index(input.world.global_map, 8U, 40U)],
            0U);
  EXPECT_EQ(reachability.projection->candidate_edges_evaluated % 2U, 0U);
  EXPECT_EQ(reachability.projection->certified_edges % 2U, 0U);
}

TEST(ReachabilityProjection, HopperFixtureHasCertifiedStartLandingAndLocalEdge) {
  PlannerInput input = test::MakeValidHopperInput();
  input.world.global_map = test::MakeFlatMap("map", 80U, 16U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 80U, 16U, 0.5);
  input.config.global_map.base_resolution_m = 0.5;
  auto& state = std::get<HopperState>(input.current_state);
  state.pose.position_m = {3.25, 4.25, 0.0};
  for (std::size_t row = 0U; row < 16U; ++row) {
    SetObstacle(input.world.global_map, row, 20U);
    SetObstacle(input.world.local_map, row, 20U);
  }
  const auto global = shared::MapSnapshot::Create(input.world.global_map);
  const auto local = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(global.ok()) << global.reason_code;
  ASSERT_TRUE(local.ok()) << local.reason_code;
  const auto safe = shared::BuildSafeProjection(
      global.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(safe.ok()) << safe.reason_code;
  const shared::GridCell start_cell{.x = 6, .y = 8};
  const shared::GridCell target_cell{.x = 40, .y = 8};
  EXPECT_TRUE(safe.projection->HardFeasible(start_cell));
  EXPECT_TRUE(safe.projection->HardFeasible(target_cell));
  const auto capability = std::get<HopperCapability>(input.capability);
  const auto landing = [&](const shared::GridCell cell) {
    const Vec3 center = local.snapshot->CellCenter(cell);
    return hopper::CertifyExactLandingRegion(
        *local.snapshot,
        GoalRegion{
            .goal_id = "diagnostic",
            .target = PointGoal{.position_m = center, .tolerance_m = 0.0},
        },
        capability, input.config.map_safety, {});
  };
  const auto source = landing(start_cell);
  const auto target = landing(target_cell);
  ASSERT_TRUE(source.ok()) << source.reason_code;
  ASSERT_TRUE(target.ok()) << target.reason_code;
  const auto hop = hopper::CertifySingleHop(
      hopper::SingleHopCertificationProblem{
          .launch_position_m = source.region->aim_position_on_surface_m,
          .landing_position_m = target.region->aim_position_on_surface_m,
          .gravity_mps2 = {0.0, 0.0, -1.62},
          .flight_map = global.snapshot.get(),
          .capability = &capability,
          .map_safety = &input.config.map_safety,
      });
  ASSERT_TRUE(hop.ok()) << hop.reason_code;
}

TEST(ReachabilityProjection, HopperCannotUseAnEdgeBeyondTheRequestedDistance) {
  PlannerInput input = test::MakeValidHopperInput();
  input.world.global_map = test::MakeFlatMap("map", 80U, 16U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 80U, 16U, 0.5);
  input.config.global_map.base_resolution_m = 0.5;
  auto& state = std::get<HopperState>(input.current_state);
  state.pose.position_m = {3.25, 4.25, 0.0};
  KeepOnlyLandingEvidencePatches(
      input.world.local_map, {{8U, 6U}, {8U, 74U}});

  const auto reachability = ProjectReachability(input, 30.0);

  ASSERT_TRUE(reachability.ok()) << reachability.reason_code;
  EXPECT_EQ(reachability.projection->reachable[
                Index(input.world.local_map, 8U, 74U)],
            0U);
  EXPECT_LE(reachability.projection->maximum_certified_edge_distance_m, 30.0);
}

TEST(ReachabilityProjection, HopperRejectsInsufficientDeltaVWithoutAborting) {
  PlannerInput input = test::MakeValidHopperInput();
  input.world.global_map = test::MakeFlatMap("map", 80U, 16U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 80U, 16U, 0.5);
  input.config.global_map.base_resolution_m = 0.5;
  std::get<HopperState>(input.current_state).pose.position_m =
      {3.25, 4.25, 0.0};
  KeepOnlyLandingEvidencePatches(
      input.world.local_map, {{8U, 6U}, {8U, 40U}});
  std::get<HopperCapability>(input.capability)
      .reference_propellant_mass_kg = 0.001;

  const auto reachability = ProjectReachability(input, 30.0);

  ASSERT_TRUE(reachability.ok()) << reachability.reason_code;
  EXPECT_EQ(reachability.projection->reachable[
                Index(input.world.global_map, 8U, 40U)],
            0U);
  EXPECT_GT(reachability.projection->candidate_edges_evaluated, 0U);
  EXPECT_GT(reachability.projection->rejected_edges, 0U);
}

TEST(ReachabilityProjection, HopperRejectsEveryBlockedFlightTube) {
  PlannerInput input = test::MakeValidHopperInput();
  input.world.global_map = test::MakeFlatMap("map", 80U, 16U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 80U, 16U, 0.5);
  input.config.global_map.base_resolution_m = 0.5;
  std::get<HopperState>(input.current_state).pose.position_m =
      {3.25, 4.25, 0.0};
  KeepOnlyLandingEvidencePatches(
      input.world.local_map, {{8U, 6U}, {8U, 40U}});
  for (std::size_t row = 0U; row < 16U; ++row) {
    SetObstacle(input.world.global_map, row, 20U, 100.0F);
  }

  const auto reachability = ProjectReachability(input, 30.0);

  ASSERT_TRUE(reachability.ok()) << reachability.reason_code;
  EXPECT_EQ(reachability.projection->reachable[
                Index(input.world.global_map, 8U, 40U)],
            0U);
  EXPECT_GT(reachability.projection->candidate_edges_evaluated, 0U);
  EXPECT_GT(reachability.projection->rejected_edges, 0U);
}

TEST(ReachabilityProjection, HopperRejectsUncertifiableLandingRegionTube) {
  PlannerInput input = test::MakeValidHopperInput();
  input.world.global_map = test::MakeFlatMap("map", 80U, 16U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 80U, 16U, 0.5);
  input.config.global_map.base_resolution_m = 0.5;
  std::get<HopperState>(input.current_state).pose.position_m =
      {3.25, 4.25, 0.0};
  auto& capability = std::get<HopperCapability>(input.capability);
  capability.flight_collision_radius_m = 0.55 - 2.0e-9;
  KeepOnlyLandingEvidencePatches(
      input.world.local_map, {{8U, 6U}, {8U, 40U}});
  SetObstacle(input.world.global_map, 6U, 20U, 100.0F);
  SetObstacle(input.world.global_map, 10U, 20U, 100.0F);

  const auto global = shared::MapSnapshot::Create(input.world.global_map);
  const auto local = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(global.ok()) << global.reason_code;
  ASSERT_TRUE(local.ok()) << local.reason_code;
  const auto source = hopper::CertifyExactLandingRegion(
      *local.snapshot,
      GoalRegion{
          .goal_id = "source",
          .target = PointGoal{
              .position_m = {3.25, 4.25, 0.0}, .tolerance_m = 0.0},
      },
      capability, input.config.map_safety, {});
  const auto target = hopper::CertifyExactLandingRegion(
      *local.snapshot,
      GoalRegion{
          .goal_id = "target",
          .target = PointGoal{
              .position_m = {20.25, 4.25, 0.0}, .tolerance_m = 0.0},
      },
      capability, input.config.map_safety, {});
  ASSERT_TRUE(source.ok()) << source.reason_code;
  ASSERT_TRUE(target.ok()) << target.reason_code;
  const auto center_hop = hopper::CertifySingleHop(
      hopper::SingleHopCertificationProblem{
          .launch_position_m = source.region->aim_position_on_surface_m,
          .landing_position_m = target.region->aim_position_on_surface_m,
          .gravity_mps2 = {0.0, 0.0, -1.62},
          .flight_map = global.snapshot.get(),
          .capability = &capability,
          .map_safety = &input.config.map_safety,
      });
  ASSERT_TRUE(center_hop.ok()) << center_hop.reason_code;

  const auto reachability = ProjectReachability(input, 30.0);

  ASSERT_TRUE(reachability.ok()) << reachability.reason_code;
  EXPECT_EQ(reachability.projection->reachable[
                Index(input.world.global_map, 8U, 40U)],
            0U);
  EXPECT_GT(reachability.projection->rejected_edges, 0U);
}

TEST(ReachabilityProjection, HopperRejectsUncertifiedLandingCell) {
  PlannerInput input = test::MakeValidHopperInput();
  input.world.global_map = test::MakeFlatMap("map", 80U, 16U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 80U, 16U, 0.5);
  input.config.global_map.base_resolution_m = 0.5;
  std::get<HopperState>(input.current_state).pose.position_m =
      {3.25, 4.25, 0.0};
  KeepOnlyLandingEvidencePatches(input.world.local_map, {{8U, 6U}});

  const auto reachability = ProjectReachability(input, 30.0);

  ASSERT_TRUE(reachability.ok()) << reachability.reason_code;
  EXPECT_EQ(reachability.projection->reachable[
                Index(input.world.global_map, 8U, 40U)],
            0U);
}

TEST(ReachabilityProjection, RepeatedInputIsBitAndDiagnosticIdentical) {
  const PlannerInput input = test::MakeValidHopperInput();

  const auto first = ProjectReachability(input, 2.0);
  const auto second = ProjectReachability(input, 2.0);

  ASSERT_TRUE(first.ok()) << first.reason_code;
  ASSERT_TRUE(second.ok()) << second.reason_code;
  EXPECT_EQ(first.projection->reachable, second.projection->reachable);
  EXPECT_EQ(first.projection->candidate_edges_evaluated,
            second.projection->candidate_edges_evaluated);
  EXPECT_EQ(first.projection->certified_edges,
            second.projection->certified_edges);
  EXPECT_EQ(first.projection->rejected_edges,
            second.projection->rejected_edges);
  EXPECT_DOUBLE_EQ(first.projection->maximum_certified_edge_distance_m,
                   second.projection->maximum_certified_edge_distance_m);
}

TEST(ReachabilityProjection, CancellationAndInvalidDistanceFailClosed) {
  PlannerInput canceled = test::MakeValidHopperInput();
  std::stop_source source;
  source.request_stop();
  canceled.stop_token = source.get_token();

  const auto canceled_result = ProjectReachability(canceled, 30.0);
  const auto invalid_result = ProjectReachability(
      test::MakeValidHopperInput(), std::nan(""));

  EXPECT_FALSE(canceled_result.ok());
  EXPECT_FALSE(canceled_result.projection.has_value());
  EXPECT_EQ(canceled_result.reason_code, "REQUEST_CANCELED");
  EXPECT_FALSE(invalid_result.ok());
  EXPECT_FALSE(invalid_result.projection.has_value());
  EXPECT_EQ(invalid_result.reason_code,
            "REACHABILITY_MAXIMUM_EDGE_DISTANCE_INVALID");
}

TEST(ReachabilityProjection, DetailLandingEvidenceFeedsTheSameHopperGraph) {
  PlannerInput input = test::MakeValidHopperInput();
  input.world.global_map = test::MakeFlatMap("map", 80U, 16U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 80U, 16U, 0.5);
  input.config.global_map.base_resolution_m = 0.5;
  std::get<HopperState>(input.current_state).pose.position_m =
      {3.25, 4.25, 0.0};
  KeepOnlyLandingEvidencePatches(
      input.world.local_map, {{8U, 6U}, {8U, 40U}});
  const auto global = shared::MapSnapshot::Create(input.world.global_map);
  ASSERT_TRUE(global.ok()) << global.reason_code;
  std::vector<Vec3> targets;
  targets.reserve(global.snapshot->cell_count());
  for (std::size_t index = 0U; index < global.snapshot->cell_count(); ++index) {
    targets.push_back(global.snapshot->CellCenter(shared::GridCell{
        .x = static_cast<std::int32_t>(index % global.snapshot->width()),
        .y = static_cast<std::int32_t>(index / global.snapshot->width()),
    }));
  }

  const auto landing = ProjectHopperLandingEvidence(input, targets);
  ASSERT_TRUE(landing.ok()) << landing.reason_code;
  HopperLandingEvidenceGrid evidence{
      .width = global.snapshot->width(),
      .height = global.snapshot->height(),
      .landings = landing.projection->landings,
      .algorithm_id = landing.projection->algorithm_id,
  };
  const auto internal = ProjectReachability(input, 30.0);
  const auto external = ProjectReachability(input, 30.0, evidence);

  ASSERT_TRUE(internal.ok()) << internal.reason_code;
  ASSERT_TRUE(external.ok()) << external.reason_code;
  EXPECT_EQ(external.projection->reachable, internal.projection->reachable);
  EXPECT_EQ(external.projection->certified_edges,
            internal.projection->certified_edges);
  EXPECT_EQ(external.projection->rejected_edges,
            internal.projection->rejected_edges);

  evidence.width = 79U;
  const auto invalid = ProjectReachability(input, 30.0, evidence);
  EXPECT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.reason_code, "HOPPER_LANDING_EVIDENCE_GEOMETRY_INVALID");
}

TEST(ReachabilityProjection,
     DirectHopperProjectionDoesNotTraverseIntermediateLandingNodes) {
  PlannerInput input = test::MakeValidHopperInput();
  input.world.global_map = test::MakeFlatMap("map", 24U, 16U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 24U, 16U, 0.5);
  input.config.global_map.base_resolution_m = 0.5;
  std::get<HopperState>(input.current_state).pose.position_m =
      {3.25, 4.25, 0.0};
  const auto global = shared::MapSnapshot::Create(input.world.global_map);
  ASSERT_TRUE(global.ok()) << global.reason_code;
  std::vector<Vec3> targets;
  targets.reserve(global.snapshot->cell_count());
  for (std::size_t index = 0U; index < global.snapshot->cell_count(); ++index) {
    targets.push_back(global.snapshot->CellCenter(shared::GridCell{
        .x = static_cast<std::int32_t>(index % global.snapshot->width()),
        .y = static_cast<std::int32_t>(index / global.snapshot->width()),
    }));
  }
  const auto landing = ProjectHopperLandingEvidence(input, targets);
  ASSERT_TRUE(landing.ok()) << landing.reason_code;
  const HopperLandingEvidenceGrid evidence{
      .width = global.snapshot->width(),
      .height = global.snapshot->height(),
      .landings = landing.projection->landings,
      .algorithm_id = landing.projection->algorithm_id,
  };

  const auto graph = ProjectReachability(input, 2.0, evidence);
  const auto direct = ProjectDirectHopperReachability(input, 2.0, evidence);

  ASSERT_TRUE(graph.ok()) << graph.reason_code;
  ASSERT_TRUE(direct.ok()) << direct.reason_code;
  EXPECT_EQ(direct.projection->algorithm_id,
            "cpp-hopper-certified-bidirectional-direct/v1");
  EXPECT_NE(direct.projection->reachable[
                Index(input.world.global_map, 8U, 8U)],
            0U);
  EXPECT_NE(graph.projection->reachable[
                Index(input.world.global_map, 8U, 14U)],
            0U);
  EXPECT_EQ(direct.projection->reachable[
                Index(input.world.global_map, 8U, 14U)],
            0U);
}

TEST(ReachabilityProjection,
     ExactDetailLandingEvidenceOverridesItsCoarseAggregateCell) {
  PlannerInput input = test::MakeValidHopperInput();
  input.world.global_map = test::MakeFlatMap("map", 80U, 16U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 80U, 16U, 0.5);
  input.config.global_map.base_resolution_m = 0.5;
  std::get<HopperState>(input.current_state).pose.position_m =
      {3.25, 4.25, 0.0};
  const auto global = shared::MapSnapshot::Create(input.world.global_map);
  ASSERT_TRUE(global.ok()) << global.reason_code;
  std::vector<Vec3> targets;
  targets.reserve(global.snapshot->cell_count());
  for (std::size_t index = 0U; index < global.snapshot->cell_count(); ++index) {
    targets.push_back(global.snapshot->CellCenter(shared::GridCell{
        .x = static_cast<std::int32_t>(index % global.snapshot->width()),
        .y = static_cast<std::int32_t>(index / global.snapshot->width()),
    }));
  }
  const auto landing = ProjectHopperLandingEvidence(input, targets);
  ASSERT_TRUE(landing.ok()) << landing.reason_code;
  HopperLandingEvidenceGrid evidence{
      .width = global.snapshot->width(),
      .height = global.snapshot->height(),
      .landings = landing.projection->landings,
      .algorithm_id = landing.projection->algorithm_id,
  };
  SetObstacle(input.world.global_map, 8U, 6U);

  const auto internal = ProjectReachability(input, 30.0);
  const auto external = ProjectReachability(input, 30.0, evidence);

  ASSERT_TRUE(internal.ok()) << internal.reason_code;
  ASSERT_TRUE(external.ok()) << external.reason_code;
  EXPECT_EQ(internal.projection->reachable[
                Index(input.world.global_map, 8U, 6U)],
            0U);
  EXPECT_NE(external.projection->reachable[
                Index(input.world.global_map, 8U, 6U)],
            0U);
}

}  // namespace
}  // namespace lunar::planning
