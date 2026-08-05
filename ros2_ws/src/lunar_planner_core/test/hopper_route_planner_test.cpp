#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <stop_token>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "hierarchical/hopper_route_planner.hpp"
#include "lunar_planner_core/planner.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning::hierarchical {
namespace {

[[nodiscard]] PlannerInput ThreeHopInput() {
  PlannerInput input = test::MakeValidHopperInput();
  input.request_id = "three-hop-route";
  input.world.global_map = test::MakeFlatMap("map", 20U, 1U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 20U, 1U, 0.5);
  input.world.map_from_odom = RigidTransform{
      .parent_frame = "map",
      .child_frame = "odom",
      .stamp = input.state_time,
  };
  input.config.global_map.base_resolution_m = 0.5;
  input.config.global_map.maximum_cells = 1'024U;
  input.config.global_map.maximum_axis_cells = 1'024U;
  input.current_state = HopperState{
      .pose = Pose3{.position_m = {1.5, 0.25, 0.5}},
  };
  input.goal_map = GoalRegion{
      .goal_id = "far-hopper-goal",
      .target =
          PointGoal{
              .position_m = {7.5, 0.25, 0.0},
              .tolerance_m = 0.1,
          },
  };
  auto &capability = std::get<HopperCapability>(input.capability);
  capability.body_half_extent_m.x = 0.1;
  capability.body_half_extent_m.y = 0.1;
  capability.maximum_launch_speed_mps = 2.0;
  capability.maximum_launch_impulse_newton_seconds = 100.0;
  capability.maximum_landing_speed_mps = 2.0;
  capability.minimum_flight_time = std::chrono::milliseconds{500};
  capability.maximum_flight_time = std::chrono::seconds{3};
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
}

TEST(HopperRoutePlanner, DistinguishesACompleteBrokenChainFromResourceLimits) {
  PlannerInput broken = ThreeHopInput();
  for (std::size_t x = 7U; x <= 11U; ++x) {
    SetByte(broken.world.global_map, "valid_mask", x, 0U, 0U);
  }

  const HopperRoutePlanResult no_path = PlanHopperGlobalRoute(broken);

  EXPECT_EQ(no_path.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(no_path.reason_code, "GLOBAL_NO_KNOWN_SAFE_ROUTE");
  EXPECT_FALSE(no_path.graph_truncated);

  PlannerInput nodes = ThreeHopInput();
  nodes.config.hopper.maximum_graph_nodes = 2U;
  const HopperRoutePlanResult node_limit = PlanHopperGlobalRoute(nodes);
  EXPECT_EQ(node_limit.outcome, PlanningOutcome::kResourceExhausted);
  EXPECT_EQ(node_limit.reason_code, "HOPPER_GLOBAL_ROUTE_RESOURCE_LIMIT");
  EXPECT_TRUE(node_limit.graph_truncated);

  PlannerInput degree = ThreeHopInput();
  degree.config.hopper.maximum_graph_out_degree = 1U;
  const HopperRoutePlanResult degree_limit = PlanHopperGlobalRoute(degree);
  EXPECT_EQ(degree_limit.outcome, PlanningOutcome::kResourceExhausted);
  EXPECT_EQ(degree_limit.reason_code, "HOPPER_GLOBAL_ROUTE_RESOURCE_LIMIT");
  EXPECT_TRUE(degree_limit.graph_truncated);
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

TEST(HopperRoutePlanner, GeneratesOutgoingEdgesOnlyForExpandedNodes) {
  PlannerInput input = ThreeHopInput();
  input.goal_map.target = PointGoal{
      .position_m = {3.75, 0.25, 0.0},
      .tolerance_m = 0.05,
  };

  const HopperRoutePlanResult result = PlanHopperGlobalRoute(input);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.evaluated_edge_pairs,
            result.expanded_nodes * (result.graph_nodes - 1U));
  EXPECT_LT(result.expanded_nodes, result.graph_nodes);
  EXPECT_LT(result.evaluated_edge_pairs,
            result.graph_nodes * (result.graph_nodes - 1U));
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

TEST(HopperRoutePlanner, RejectsWhenTheFirstLocalFlightTubeIsBlocked) {
  Planner planner;
  PlannerInput input = ThreeHopInput();
  input.goal_map.target = PointGoal{
      .position_m = {3.75, 0.25, 0.0},
      .tolerance_m = 0.05,
  };
  SetByte(input.world.local_map, "obstacle", 5U, 0U, 1U);
  SetFloat(input.world.local_map, "obstacle_height", 5U, 0U, 4.0F);

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.reason_code, "HOPPER_FLIGHT_TUBE_OBSTACLE_COLLISION");
  EXPECT_FALSE(output.reference.has_value());
}

} // namespace
} // namespace lunar::planning::hierarchical
