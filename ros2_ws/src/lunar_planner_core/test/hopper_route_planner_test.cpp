#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <stop_token>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "hierarchical/hopper_route_planner.hpp"
#include "hierarchical/route_continuation.hpp"
#include "hopper/flight_tube_certifier.hpp"
#include "lunar_planner_core/planner.hpp"
#include "shared/map_snapshot.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning::hierarchical {
namespace {

template <typename Config>
concept HasTruncatingHopperGraphLimits = requires(Config config) {
  config.maximum_landing_regions;
  config.maximum_graph_nodes;
  config.maximum_graph_out_degree;
};

[[nodiscard]] PlannerInput ThreeHopInput() {
  PlannerInput input = test::MakeValidHopperInput();
  input.request_id = "three-hop-route";
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
      .goal_id = "far-hopper-goal",
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
  return input;
}

[[nodiscard]] PlannerInput LargeLandingAreaInput() {
  PlannerInput input = test::MakeValidHopperInput();
  input.request_id = "large-landing-support";
  input.world.global_map = test::MakeFlatMap("map", 40U, 40U, 0.2);
  input.world.local_map = test::MakeFlatMap("odom", 40U, 40U, 0.2);
  input.config.global_map.base_resolution_m = 0.2;
  input.current_state = HopperState{
      .pose = Pose3{.position_m = {3.1, 3.1, 0.5}},
  };
  input.goal_map.target = PointGoal{
      .position_m = {4.1, 3.1, 0.0},
      .tolerance_m = 0.05,
  };
  auto &capability = std::get<HopperCapability>(input.capability);
  capability.body_half_extent_m.x = 0.1;
  capability.body_half_extent_m.y = 0.1;
  capability.minimum_landing_region_area_m2 = 1.327322;
  return input;
}

[[nodiscard]] PlannerInput ReachableFrontierInput() {
  PlannerInput input = ThreeHopInput();
  input.request_id = "reachable-frontier-route";
  input.world.global_map = test::MakeFlatMap("map", 40U, 20U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 40U, 20U, 0.5);
  input.current_state = HopperState{
      .pose = Pose3{.position_m = {1.25, 7.25, 0.5}},
  };
  input.goal_map.target = PointGoal{
      .position_m = {17.25, 7.25, 0.0},
      .tolerance_m = 0.1,
  };
  return input;
}

[[nodiscard]] PlannerInput LongChainBeyondLegacyNodeCap() {
  PlannerInput input = ThreeHopInput();
  input.request_id = "complete-long-hopper-chain";
  input.world.global_map = test::MakeFlatMap("map", 720U, 7U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 720U, 7U, 0.5);
  input.config.global_map.maximum_cells = 8'192U;
  input.config.global_map.maximum_axis_cells = 2'048U;
  input.goal_map.target = PointGoal{
      .position_m = {350.25, 1.75, 0.0},
      .tolerance_m = 0.1,
  };
  return input;
}

[[nodiscard]] PlannerInput CulDeSacWithCompleteDetour() {
  PlannerInput input = ThreeHopInput();
  input.request_id = "hopper-complete-detour";
  input.world.global_map = test::MakeFlatMap("map", 52U, 32U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 52U, 32U, 0.5);
  input.config.global_map.maximum_cells = 4'096U;
  input.config.global_map.maximum_axis_cells = 4'096U;
  input.current_state = HopperState{
      .pose = Pose3{.position_m = {2.25, 5.25, 0.5}},
  };
  input.goal_map.target = PointGoal{
      .position_m = {23.25, 5.25, 0.0},
      .tolerance_m = 0.1,
  };
  auto &valid = std::get<std::vector<std::uint8_t>>(
      input.world.global_map.layers.at("valid_mask").values);
  std::fill(valid.begin(), valid.end(), 0U);
  const auto mark_rectangle =
      [&](const std::size_t minimum_x, const std::size_t maximum_x,
          const std::size_t minimum_y, const std::size_t maximum_y) {
        for (std::size_t y = minimum_y; y <= maximum_y; ++y) {
          for (std::size_t x = minimum_x; x <= maximum_x; ++x) {
            valid[y * input.world.global_map.width + x] = 1U;
          }
        }
      };
  mark_rectangle(0U, 30U, 6U, 14U);
  mark_rectangle(0U, 12U, 6U, 28U);
  mark_rectangle(0U, 50U, 20U, 28U);
  mark_rectangle(42U, 50U, 6U, 28U);
  return input;
}

void SetByte(GridMap &map, const std::string &layer, const std::size_t x,
             const std::size_t y, const std::uint8_t value) {
  std::get<std::vector<std::uint8_t>>(map.layers.at(layer).values)
      .at(y * map.width + x) = value;
}

void SetFloat(GridMap &map, const std::string &layer, const std::size_t x,
              const std::size_t y, const float value) {
  std::get<std::vector<float>>(map.layers.at(layer).values)
      .at(y * map.width + x) = value;
}

void AddTallObstacle(GridMap &map, const std::size_t x, const std::size_t y) {
  SetByte(map, "obstacle", x, y, 1U);
  SetFloat(map, "obstacle_height", x, y, 4.0F);
}

[[nodiscard]] bool HasWarning(const PlannerOutput &output,
                              const std::string &warning) {
  return std::ranges::find(output.diagnostics.warning_codes, warning) !=
         output.diagnostics.warning_codes.end();
}

TEST(HopperRoutePlanner, BuildsADeterministicThreeHopLandingChain) {
  const PlannerInput input = ThreeHopInput();

  const HopperRoutePlanResult first = PlanHopperGlobalRoute(input);
  const HopperRoutePlanResult second = PlanHopperGlobalRoute(input);

  ASSERT_TRUE(first.ok()) << first.reason_code;
  ASSERT_TRUE(second.ok()) << second.reason_code;
  ASSERT_TRUE(first.route.has_value());
  EXPECT_EQ(first.reason_code, "HOPPER_GLOBAL_ROUTE_AVAILABLE");
  EXPECT_EQ(first.route_hops, 3U);
  EXPECT_EQ(first.route->poses_map.size(), 4U);
  EXPECT_EQ(first.route->raw_cells, second.route->raw_cells);
  EXPECT_DOUBLE_EQ(first.route->cost, second.route->cost);
  EXPECT_GT(first.maximum_horizontal_reach_m, 2.0);
  EXPECT_LT(first.maximum_horizontal_reach_m, 2.5);
  EXPECT_EQ(first.safe_landing_nodes, second.safe_landing_nodes);
  EXPECT_EQ(first.evaluated_edge_pairs, second.evaluated_edge_pairs);
  EXPECT_EQ(first.coarse_edges_rejected, second.coarse_edges_rejected);
  EXPECT_EQ(first.full_edges_certified, second.full_edges_certified);
  EXPECT_EQ(first.full_edges_invalidated, second.full_edges_invalidated);
  EXPECT_EQ(
      first.edge_certificate_cache_hits, second.edge_certificate_cache_hits);
  EXPECT_EQ(first.expanded_nodes, second.expanded_nodes);
  EXPECT_EQ(first.open_peak, second.open_peak);
  EXPECT_GT(first.safe_landing_nodes, 0U);
  EXPECT_GT(first.spatial_index_elapsed.count(), 0);
  EXPECT_GT(first.ballistic_solve_elapsed.count(), 0);
  EXPECT_GT(first.flight_tube_certification_elapsed.count(), 0);
}

TEST(HopperRoutePlanner,
     AcceptsLandingRegionLargerThanOneCellWhenSupportAreaIsSafe) {
  const PlannerInput input = LargeLandingAreaInput();

  const HopperRoutePlanResult result = PlanHopperGlobalRoute(input);

  ASSERT_TRUE(result.ok()) << result.reason_code
                           << " nodes=" << result.graph_nodes
                           << " edges=" << result.graph_edges
                           << " expanded=" << result.expanded_nodes
                           << " evaluated=" << result.evaluated_edge_pairs;
  ASSERT_TRUE(result.route.has_value());
  EXPECT_EQ(result.reason_code, "HOPPER_GLOBAL_ROUTE_AVAILABLE");
  EXPECT_EQ(result.route_hops, 1U);
  EXPECT_GE(result.route->estimated_work_memory_bytes,
            input.world.global_map.CellCount() *
                (2U * sizeof(std::uint8_t) + sizeof(double)));
  EXPECT_GT(result.landing_field_elapsed.count(), 0);
}

TEST(HopperRoutePlanner,
     DiscoversLandingNodesFromTheReachableFrontierInsteadOfMapPrefix) {
  const PlannerInput input = ReachableFrontierInput();

  const HopperRoutePlanResult result = PlanHopperGlobalRoute(input);

  ASSERT_TRUE(result.ok()) << result.reason_code
                           << " nodes=" << result.graph_nodes
                           << " edges=" << result.graph_edges
                           << " expanded=" << result.expanded_nodes
                           << " evaluated=" << result.evaluated_edge_pairs;
  EXPECT_GT(result.route_hops, 1U);
  EXPECT_GT(result.graph_nodes, 128U);
}

TEST(HopperRoutePlanner, SolvesAChainRequiringMoreThan128LandingNodes) {
  const HopperRoutePlanResult result =
      PlanHopperGlobalRoute(LongChainBeyondLegacyNodeCap());

  ASSERT_TRUE(result.ok()) << result.reason_code
                           << " nodes=" << result.graph_nodes
                           << " edges=" << result.graph_edges
                           << " expanded=" << result.expanded_nodes;
  EXPECT_GT(result.route_hops, 128U);
  EXPECT_GT(result.graph_nodes, 128U);
  ASSERT_TRUE(result.route.has_value());
  EXPECT_EQ(result.route->poses_map.size(), result.route_hops + 1U);
}

TEST(HopperRoutePlanner, ConnectsAlreadyIndexedNonGoalLandingCenters) {
  const PlannerInput input = ReachableFrontierInput();
  const HopperRoutePlanResult result = PlanHopperGlobalRoute(input);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_GT(result.nominal_hops.size(), 1U);
  EXPECT_EQ(result.nominal_hops.size(), result.route_hops);
  for (std::size_t index = 1U; index < result.nominal_hops.size(); ++index) {
    EXPECT_EQ(result.nominal_hops[index].source_id,
              result.nominal_hops[index - 1U].target_id);
    EXPECT_LT(result.nominal_hops[index].source_id,
              input.world.global_map.CellCount());
  }
}

TEST(HopperRoutePlanner, FallsBackToCompleteSearchAfterAGreedyCulDeSac) {
  const HopperRoutePlanResult result =
      PlanHopperGlobalRoute(CulDeSacWithCompleteDetour());

  ASSERT_TRUE(result.ok()) << result.reason_code
                           << " nodes=" << result.graph_nodes
                           << " edges=" << result.graph_edges
                           << " expanded=" << result.expanded_nodes;
  ASSERT_TRUE(result.route.has_value());
  const auto maximum_y =
      std::ranges::max(result.route->poses_map, {},
                       [](const Pose3 &pose) { return pose.position_m.y; });
  EXPECT_GT(maximum_y.position_m.y, 10.0);
}

TEST(HopperRoutePlanner, InvalidatesBlockedCandidateEdgeAndFindsAlternative) {
  PlannerInput input = ThreeHopInput();
  input.world.global_map = test::MakeFlatMap("map", 20U, 15U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 20U, 15U, 0.5);
  input.current_state = HopperState{
      .pose = Pose3{.position_m = {1.5, 3.75, 0.5}},
  };
  input.goal_map.target = PointGoal{
      .position_m = {7.25, 3.75, 0.0},
      .tolerance_m = 0.1,
  };
  AddTallObstacle(input.world.global_map, 9U, 7U);

  const HopperRoutePlanResult result = PlanHopperGlobalRoute(input);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_GE(result.full_edges_invalidated, 1U);
  EXPECT_EQ(result.certified_hops.size(), result.route_hops);
  EXPECT_TRUE(std::ranges::all_of(result.certified_hops,
                                  [](const CertifiedHopPreview &hop) {
                                    return hop.flight_tube_radius_m > 0.0 &&
                                           !hop.landing_region_map.empty() &&
                                           !hop.promotion_region_map.empty();
                                  }));
}

TEST(HopperRoutePlanner, ContractsPromotionRegionByRuntimeUncertaintyAndCell) {
  const PlannerInput input = ThreeHopInput();

  const HopperRoutePlanResult result = PlanHopperGlobalRoute(input);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.certified_hops.empty());
  const CertifiedHopPreview &hop = result.certified_hops.front();
  const auto x_extent = [](const std::vector<Vec3> &points) {
    const auto [minimum, maximum] = std::ranges::minmax(
        points, {}, [](const Vec3 point) { return point.x; });
    return maximum.x - minimum.x;
  };
  const double expected_contraction =
      input.position_uncertainty_m + input.world.global_map.resolution_m;
  EXPECT_NEAR(0.5 * (x_extent(hop.landing_region_map) -
                     x_extent(hop.promotion_region_map)),
              expected_contraction, 1.0e-12);
}

TEST(HopperRoutePlanner, PreservesNumericalIndeterminacyFromFlightTube) {
  const shared::MapSnapshotBuildResult map =
      shared::MapSnapshot::Create(test::MakeFlatMap("map", 8U, 8U, 1.0));
  ASSERT_TRUE(map.ok()) << map.reason_code;
  const PlannerInput input = test::MakeValidHopperInput();
  const HopperCapability capability =
      std::get<HopperCapability>(input.capability);
  const hopper::CertifiedLandingRegion region{
      .seed_cell = {.x = 3, .y = 3},
      .aim_position_on_surface_m = {3.5, 3.5, 0.0},
      .plane_normal = {0.0, 0.0, 1.0},
      .boundary_m = {{3.0, 3.0, 0.0},
                     {4.0, 3.0, 0.0},
                     {4.0, 4.0, 0.0},
                     {3.0, 4.0, 0.0}},
      .area_m2 = 1.0,
  };
  const double huge = std::numeric_limits<double>::max() / 2.0;
  const hopper::BallisticArc nonrepresentable_sweep{
      .launch_position_m = {huge, 3.5, 0.5},
      .landing_position_m = {huge, 3.5, 0.5},
      .gravity_mps2 = {0.0, 0.0, -1.62},
      .launch_velocity_mps = {huge, 0.0, 1.62},
      .landing_velocity_mps = {huge, 0.0, -1.62},
      .flight_time_s = 128.0,
  };

  const hopper::FlightTubeCertificationResult certified =
      hopper::CertifyFlightTube(nonrepresentable_sweep, *map.snapshot, region,
                                region, capability, input.config, {});

  EXPECT_FALSE(certified.certified);
  EXPECT_FALSE(certified.canceled);
  EXPECT_EQ(certified.reason_code,
            "HOPPER_FLIGHT_TUBE_NUMERICAL_INDETERMINATE");
}

TEST(HopperRoutePlanner, RejectsAMidArcRockAcrossTheOnlyPassage) {
  PlannerInput input = ThreeHopInput();
  for (std::size_t y = 0U; y < input.world.global_map.height; ++y) {
    AddTallObstacle(input.world.global_map, 8U, y);
  }

  const HopperRoutePlanResult result = PlanHopperGlobalRoute(input);

  EXPECT_EQ(result.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(result.reason_code, "GLOBAL_NO_KNOWN_SAFE_ROUTE");
  EXPECT_GE(result.full_edges_invalidated, 1U);
  EXPECT_TRUE(result.certified_hops.empty());
}

TEST(HopperRoutePlanner, ExhaustsACompleteBrokenChainBeforeReportingNoRoute) {
  PlannerInput broken = ThreeHopInput();
  for (std::size_t x = 7U; x <= 11U; ++x) {
    for (std::size_t y = 0U; y < broken.world.global_map.height; ++y) {
      SetByte(broken.world.global_map, "valid_mask", x, y, 0U);
    }
  }

  const HopperRoutePlanResult no_path = PlanHopperGlobalRoute(broken);

  EXPECT_EQ(no_path.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(no_path.reason_code, "GLOBAL_NO_KNOWN_SAFE_ROUTE");
  EXPECT_GT(no_path.expanded_nodes, 0U);
}

TEST(HopperRoutePlanner, RejectsAGlobalCellWiderThanHalfTheHopReach) {
  PlannerInput input = ThreeHopInput();
  input.world.global_map = test::MakeFlatMap("map", 5U, 1U, 2.0);
  input.world.local_map = test::MakeFlatMap("odom", 5U, 1U, 2.0);
  input.config.global_map.base_resolution_m = 2.0;

  const HopperRoutePlanResult result = PlanHopperGlobalRoute(input);

  EXPECT_EQ(result.outcome, PlanningOutcome::kInvalidRequest);
  EXPECT_EQ(result.reason_code, "HOPPER_GLOBAL_RESOLUTION_INSUFFICIENT");
  EXPECT_GT(input.world.global_map.resolution_m,
            result.maximum_horizontal_reach_m / 2.0);
}

TEST(HopperRoutePlanner, RejectsANonPlanarGlobalLandingCandidate) {
  PlannerInput input = test::MakeValidHopperInput();
  input.goal_map.target = PointGoal{
      .position_m = {4.25, 3.25, 0.0},
      .tolerance_m = 0.05,
  };
  SetFloat(input.world.global_map, "elevation", 8U, 6U, 0.08F);
  SetFloat(input.world.global_map, "elevation", 7U, 6U, 0.0F);
  SetFloat(input.world.global_map, "elevation", 9U, 6U, 0.0F);

  const HopperRoutePlanResult result = PlanHopperGlobalRoute(input);

  EXPECT_EQ(result.outcome, PlanningOutcome::kGoalInfeasible);
  EXPECT_EQ(result.reason_code, "HOPPER_GLOBAL_GOAL_INFEASIBLE");
}

TEST(HopperRoutePlanner, HonorsCancellationBeforeBuildingTheGraph) {
  PlannerInput input = ThreeHopInput();
  std::stop_source stop;
  stop.request_stop();
  input.stop_token = stop.get_token();

  const HopperRoutePlanResult result = PlanHopperGlobalRoute(input);

  EXPECT_EQ(result.outcome, PlanningOutcome::kCanceled);
  EXPECT_EQ(result.reason_code, "REQUEST_CANCELED");
  EXPECT_EQ(result.graph_nodes, 0U);
}

TEST(HopperRoutePlanner, BoundsEdgeEvaluationToExpandedReachableFrontier) {
  PlannerInput input = ThreeHopInput();
  input.goal_map.target = PointGoal{
      .position_m = {3.75, 1.75, 0.0},
      .tolerance_m = 0.05,
  };

  const HopperRoutePlanResult result = PlanHopperGlobalRoute(input);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_GT(result.evaluated_edge_pairs, 0U);
  EXPECT_LE(result.evaluated_edge_pairs,
            result.expanded_nodes * result.graph_nodes);
  EXPECT_LE(result.expanded_nodes, result.graph_nodes);
}

TEST(HopperRoutePlanner, PublishesTheFullPreviewButAuthorizesOnlyOneHop) {
  Planner planner;
  PlannerInput input = ThreeHopInput();
  input.config.hopper.maximum_authorized_hops = 4U;

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  ASSERT_TRUE(output.reference.has_value());
  const auto *hops = std::get_if<HopReference>(&output.reference->data);
  ASSERT_NE(hops, nullptr);
  EXPECT_EQ(hops->segments.size(), 1U);
  EXPECT_EQ(output.reference->preview.poses_map.size(), 4U);
  EXPECT_TRUE(HasWarning(output, "HOPPER_REMAINING_HOPS_PREVIEW_ONLY"));
  EXPECT_TRUE(HasWarning(output, "HOPPER_AUTHORIZATION_CLAMPED_TO_ONE"));
}

TEST(HopperRoutePlanner, ThinsPreviewWithoutTruncatingTheCertifiedHopChain) {
  Planner planner;
  PlannerInput input = ThreeHopInput();
  input.config.global_search.maximum_preview_points = 2U;

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  ASSERT_TRUE(output.reference.has_value());
  ASSERT_NE(output.continuation, nullptr);
  ASSERT_EQ(output.reference->preview.poses_map.size(), 2U);
  EXPECT_EQ(output.reference->preview.poses_map.front(),
            output.continuation->global_route().poses_map.front());
  EXPECT_EQ(output.reference->preview.poses_map.back(),
            output.continuation->global_route().poses_map.back());
  EXPECT_EQ(output.continuation->certified_hops().size(), 3U);
}

TEST(HopperRoutePlanner, RejectsWhenTheFirstLocalFlightTubeIsBlocked) {
  Planner planner;
  PlannerInput input = ThreeHopInput();
  input.goal_map.target = PointGoal{
      .position_m = {3.75, 1.75, 0.0},
      .tolerance_m = 0.05,
  };
  SetByte(input.world.local_map, "obstacle", 5U, 3U, 1U);
  SetFloat(input.world.local_map, "obstacle_height", 5U, 3U, 4.0F);

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.reason_code, "HOPPER_FLIGHT_TUBE_OBSTACLE_COLLISION");
  EXPECT_FALSE(output.reference.has_value());
}

static_assert(!HasTruncatingHopperGraphLimits<HopperPlannerConfig>);

} // namespace
} // namespace lunar::planning::hierarchical
