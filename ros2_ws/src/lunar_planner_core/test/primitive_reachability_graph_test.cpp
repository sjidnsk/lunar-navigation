#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "legged/legged_lattice.hpp"
#include "lunar_planner_core/primitive_reachability.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/primitive_reachability_graph.hpp"
#include "shared/safe_projection.hpp"
#include "shared/sha256.hpp"
#include "test_fixtures.hpp"
#include "wheel/wheel_primitive_expansion.hpp"

namespace lunar::planning::shared {
namespace {

PrimitiveReachabilityState MakeState(
    const std::int32_t cell_x, const std::int32_t yaw_bin,
    const std::int32_t motion_mode, const double x,
    const double path_cost = 0.0) {
  return PrimitiveReachabilityState{
      .position_m = Vec3{x, 0.5, 0.0},
      .yaw_rad = static_cast<double>(yaw_bin) * 0.25,
      .cell_x = cell_x,
      .cell_y = 0,
      .yaw_bin = yaw_bin,
      .motion_mode = motion_mode,
      .body_z_m = Interval{0.0, 0.0},
      .path_cost = path_cost,
      .observation_state = 1U,
  };
}

PrimitiveGraphPotentialEdge MakeEdge(
    const std::size_t source, const std::size_t target,
    const std::uint32_t primitive_index, std::string primitive_id,
    const double cost = 1.0) {
  return PrimitiveGraphPotentialEdge{
      .source_state_index = source,
      .target_state_index = target,
      .primitive_index = primitive_index,
      .primitive_id = std::move(primitive_id),
      .cost = cost,
      .certified = true,
  };
}

PrimitiveGraphBuildResult MakeDirectedFixture() {
  PrimitiveGraphBuildResult graph{
      .platform_type = PlatformType::kWheeled,
      .width = 4U,
      .height = 1U,
      .anchor_state_index = 0U,
      .algorithm_id = "test-primitive-graph/v1",
      .state_schema = "test-state/v1",
      .primitive_set_canonical_bytes = {0x01U, 0x02U, 0x03U},
      .revision = 7U,
      .invalidated_edge_count = 2U,
      .revalidated_edge_count = 3U,
  };
  graph.states = {
      MakeState(0, 0, 0, 0.5, 0.0),
      MakeState(1, 0, 1, 1.5, 1.0),
      MakeState(2, 0, 1, 2.5, 1.0),
      MakeState(1, 1, 2, 1.5, 1.5),
  };
  graph.potential_edges = {
      MakeEdge(0U, 1U, 0U, "forward"),
      MakeEdge(1U, 0U, 1U, "reverse"),
      MakeEdge(0U, 2U, 0U, "forward"),
      MakeEdge(0U, 3U, 2U, "turn"),
      MakeEdge(3U, 0U, 3U, "turn-back"),
      PrimitiveGraphPotentialEdge{
          .source_state_index = 2U,
          .target_state_index = 0U,
          .primitive_index = 4U,
          .primitive_id = "rejected-return",
          .cost = 1.0,
          .certified = false,
          .rejection_reason = "BLOCKED",
      },
  };
  return graph;
}

const PrimitiveReachabilityState* FindState(
    const PrimitiveReachabilitySnapshot& snapshot,
    const std::int32_t cell_x, const std::int32_t yaw_bin,
    const std::int32_t motion_mode) {
  const auto iterator = std::find_if(
      snapshot.states.begin(), snapshot.states.end(),
      [&](const PrimitiveReachabilityState& state) {
        return state.cell_x == cell_x && state.yaw_bin == yaw_bin &&
            state.motion_mode == motion_mode;
      });
  return iterator == snapshot.states.end() ? nullptr : &*iterator;
}

TEST(PrimitiveReachabilityGraph, Sha256MatchesStandardVectors) {
  EXPECT_EQ(Sha256Hex({}),
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855");
  const std::vector<std::uint8_t> abc{'a', 'b', 'c'};
  EXPECT_EQ(Sha256Hex(abc),
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad");
}

TEST(PrimitiveReachabilityGraph,
     LabelsForwardAndReturnPathsWithoutCollapsingState) {
  const PrimitiveReachabilityResult result =
      FinalizePrimitiveGraph(MakeDirectedFixture(), {});

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const PrimitiveReachabilitySnapshot& snapshot = *result.snapshot;
  ASSERT_EQ(snapshot.states.size(), 4U);
  ASSERT_EQ(snapshot.edges.size(), 5U);
  ASSERT_EQ(snapshot.reachable.size(), 4U);

  const auto* anchor = FindState(snapshot, 0, 0, 0);
  const auto* reversible = FindState(snapshot, 1, 0, 1);
  const auto* outbound_only = FindState(snapshot, 2, 0, 1);
  const auto* distinct_mode = FindState(snapshot, 1, 1, 2);
  ASSERT_NE(anchor, nullptr);
  ASSERT_NE(reversible, nullptr);
  ASSERT_NE(outbound_only, nullptr);
  ASSERT_NE(distinct_mode, nullptr);

  EXPECT_EQ(anchor->forward_reachable, 1U);
  EXPECT_EQ(anchor->returnable, 1U);
  EXPECT_EQ(reversible->forward_reachable, 1U);
  EXPECT_EQ(reversible->returnable, 1U);
  EXPECT_EQ(outbound_only->forward_reachable, 1U);
  EXPECT_EQ(outbound_only->returnable, 0U);
  EXPECT_EQ(distinct_mode->forward_reachable, 1U);
  EXPECT_EQ(distinct_mode->returnable, 1U);
  EXPECT_EQ(reversible->direct_successor, 1U);
  EXPECT_EQ(outbound_only->direct_successor, 1U);
  EXPECT_EQ(distinct_mode->direct_successor, 1U);

  EXPECT_EQ(snapshot.reachable[0U], 1U);
  EXPECT_EQ(snapshot.reachable[1U], 1U);
  EXPECT_EQ(snapshot.reachable[2U], 0U);
  EXPECT_EQ(snapshot.reachable[3U], 0U);
  EXPECT_EQ(snapshot.algorithm_id, "test-primitive-graph/v1");
  EXPECT_EQ(snapshot.state_schema, "test-state/v1");
  EXPECT_EQ(snapshot.revision, 7U);
  EXPECT_EQ(snapshot.invalidated_edge_count, 2U);
  EXPECT_EQ(snapshot.revalidated_edge_count, 3U);
  EXPECT_EQ(snapshot.primitive_set_sha256.size(), 64U);
  EXPECT_EQ(snapshot.graph_sha256.size(), 64U);
}

TEST(PrimitiveReachabilityGraph, IdentityIsIndependentOfInputOrder) {
  PrimitiveGraphBuildResult first = MakeDirectedFixture();
  PrimitiveGraphBuildResult second = MakeDirectedFixture();
  const std::vector<std::size_t> permutation{3U, 1U, 0U, 2U};
  std::vector<std::size_t> new_index(second.states.size());
  std::vector<PrimitiveReachabilityState> reordered;
  for (std::size_t index = 0U; index < permutation.size(); ++index) {
    new_index[permutation[index]] = index;
    reordered.push_back(second.states[permutation[index]]);
  }
  second.states = std::move(reordered);
  second.anchor_state_index = new_index[second.anchor_state_index];
  for (auto& edge : second.potential_edges) {
    edge.source_state_index = new_index[edge.source_state_index];
    edge.target_state_index = new_index[edge.target_state_index];
  }
  std::reverse(second.potential_edges.begin(), second.potential_edges.end());

  const PrimitiveReachabilityResult first_result =
      FinalizePrimitiveGraph(std::move(first), {});
  const PrimitiveReachabilityResult second_result =
      FinalizePrimitiveGraph(std::move(second), {});

  ASSERT_TRUE(first_result.ok()) << first_result.reason_code;
  ASSERT_TRUE(second_result.ok()) << second_result.reason_code;
  EXPECT_EQ(first_result.snapshot->primitive_set_sha256,
            second_result.snapshot->primitive_set_sha256);
  EXPECT_EQ(first_result.snapshot->graph_sha256,
            second_result.snapshot->graph_sha256);
  EXPECT_EQ(first_result.snapshot->states, second_result.snapshot->states);
  EXPECT_EQ(first_result.snapshot->edges, second_result.snapshot->edges);
}

TEST(PrimitiveReachabilityGraph, NormalizesNegativeZeroInIdentity) {
  PrimitiveGraphBuildResult negative = MakeDirectedFixture();
  PrimitiveGraphBuildResult positive = MakeDirectedFixture();
  negative.states[0U].position_m.z = -0.0;
  positive.states[0U].position_m.z = +0.0;

  const auto negative_result = FinalizePrimitiveGraph(std::move(negative), {});
  const auto positive_result = FinalizePrimitiveGraph(std::move(positive), {});

  ASSERT_TRUE(negative_result.ok()) << negative_result.reason_code;
  ASSERT_TRUE(positive_result.ok()) << positive_result.reason_code;
  EXPECT_EQ(negative_result.snapshot->graph_sha256,
            positive_result.snapshot->graph_sha256);
}

TEST(PrimitiveReachabilityGraph, RejectsMalformedOrCanceledGraphs) {
  PrimitiveGraphBuildResult nonfinite = MakeDirectedFixture();
  nonfinite.states[1U].position_m.x =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(FinalizePrimitiveGraph(std::move(nonfinite), {}).reason_code,
            "PRIMITIVE_GRAPH_STATE_NONFINITE");

  PrimitiveGraphBuildResult negative_cost = MakeDirectedFixture();
  negative_cost.potential_edges[0U].cost = -1.0;
  EXPECT_EQ(FinalizePrimitiveGraph(std::move(negative_cost), {}).reason_code,
            "PRIMITIVE_GRAPH_EDGE_COST_INVALID");

  PrimitiveGraphBuildResult invalid_index = MakeDirectedFixture();
  invalid_index.potential_edges[0U].target_state_index = 99U;
  EXPECT_EQ(FinalizePrimitiveGraph(std::move(invalid_index), {}).reason_code,
            "PRIMITIVE_GRAPH_EDGE_INDEX_INVALID");

  PrimitiveGraphBuildResult duplicate = MakeDirectedFixture();
  duplicate.states.push_back(duplicate.states.front());
  EXPECT_EQ(FinalizePrimitiveGraph(std::move(duplicate), {}).reason_code,
            "PRIMITIVE_GRAPH_STATE_KEY_DUPLICATE");

  std::stop_source stop_source;
  stop_source.request_stop();
  EXPECT_EQ(FinalizePrimitiveGraph(MakeDirectedFixture(),
                                   stop_source.get_token()).reason_code,
            "REQUEST_CANCELED");
}

PrimitiveReachabilityResult BuildWheelFixture(PlannerInput input) {
  const auto map = MapSnapshot::Create(input.world.global_map);
  if (!map.ok()) {
    return PrimitiveReachabilityResult{.reason_code = map.reason_code};
  }
  const auto safe = BuildSafeProjection(
      map.snapshot, input.capability, input.config.map_safety,
      input.stop_token);
  if (!safe.ok()) {
    return PrimitiveReachabilityResult{.reason_code = safe.reason_code};
  }
  return FinalizePrimitiveGraph(
      wheel::BuildWheelPrimitiveGraph(
          std::get<WheeledState>(input.current_state), *safe.projection,
          std::get<WheeledCapability>(input.capability), input.config,
          input.stop_token),
      input.stop_token);
}

TEST(PrimitiveReachabilityGraph,
     WheelProjectionDoesNotConfuseConnectedGroundWithPrimitiveReachability) {
  PlannerInput input = test::MakeValidWheelInput();
  input.world.global_map = test::MakeFlatMap("map", 8U, 5U, 1.0);
  input.world.local_map = input.world.global_map;
  auto capability = std::get<WheeledCapability>(input.capability);
  capability.motion_primitives.erase(
      std::remove_if(
          capability.motion_primitives.begin(),
          capability.motion_primitives.end(),
          [](const WheelMotionPrimitive& primitive) {
            return primitive.kind != WheelPrimitiveKind::kForward &&
                primitive.kind != WheelPrimitiveKind::kReverse &&
                primitive.kind != WheelPrimitiveKind::kStopAndSwitch;
          }),
      capability.motion_primitives.end());
  input.capability = capability;
  std::get<WheeledState>(input.current_state).pose.position_m =
      {2.5, 2.5, 0.0};

  const auto map = MapSnapshot::Create(input.world.global_map);
  ASSERT_TRUE(map.ok()) << map.reason_code;
  const auto safe = BuildSafeProjection(
      map.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(safe.ok()) << safe.reason_code;
  const GridCell start{.x = 2, .y = 2};
  const GridCell side{.x = 2, .y = 3};
  ASSERT_TRUE(safe.projection->HardFeasible(side));
  ASSERT_EQ(safe.projection->ConnectedComponent(start),
            safe.projection->ConnectedComponent(side));

  const PrimitiveReachabilityResult result = BuildWheelFixture(input);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.snapshot->algorithm_id,
            "cpp-wheel-motion-primitive-recoverable-graph/v1");
  EXPECT_EQ(result.snapshot->reachable[3U * 8U + 2U], 0U);
  EXPECT_EQ(result.snapshot->reachable[2U * 8U + 3U], 1U);
  const auto same_cell_states = std::count_if(
      result.snapshot->states.begin(), result.snapshot->states.end(),
      [](const PrimitiveReachabilityState& state) {
        return state.cell_x == 3 && state.cell_y == 2 &&
            state.yaw_bin == 0;
      });
  EXPECT_GE(same_cell_states, 2);
}

TEST(PrimitiveReachabilityGraph, WheelExcludesOutboundOnlyState) {
  PlannerInput input = test::MakeValidWheelInput();
  input.world.global_map = test::MakeFlatMap("map", 7U, 5U, 1.0);
  input.world.local_map = input.world.global_map;
  auto capability = std::get<WheeledCapability>(input.capability);
  capability.motion_primitives.erase(
      std::remove_if(
          capability.motion_primitives.begin(),
          capability.motion_primitives.end(),
          [](const WheelMotionPrimitive& primitive) {
            return primitive.kind != WheelPrimitiveKind::kForward;
          }),
      capability.motion_primitives.end());
  input.capability = capability;
  std::get<WheeledState>(input.current_state).pose.position_m =
      {2.5, 2.5, 0.0};

  const PrimitiveReachabilityResult result = BuildWheelFixture(input);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const auto* outbound = FindState(*result.snapshot, 3, 0, 1);
  ASSERT_NE(outbound, nullptr);
  EXPECT_EQ(outbound->forward_reachable, 1U);
  EXPECT_EQ(outbound->returnable, 0U);
  EXPECT_EQ(result.snapshot->reachable[2U * 7U + 3U], 0U);
}

TEST(PrimitiveReachabilityGraph, StatefulEnginePublishesWheelGraphSnapshots) {
  PlannerInput input = test::MakeValidWheelInput();
  PrimitiveReachabilityEngine engine;

  const PrimitiveReachabilityResult first = engine.Update(input, std::nullopt);
  const PrimitiveReachabilityResult second = engine.Update(input, 30.0);

  ASSERT_TRUE(first.ok()) << first.reason_code;
  ASSERT_TRUE(second.ok()) << second.reason_code;
  EXPECT_EQ(first.snapshot->algorithm_id,
            "cpp-wheel-motion-primitive-recoverable-graph/v1");
  EXPECT_EQ(first.snapshot->revision, 1U);
  EXPECT_EQ(second.snapshot->revision, 2U);
  EXPECT_EQ(first.snapshot->graph_sha256, second.snapshot->graph_sha256);
}

PrimitiveReachabilityResult BuildLeggedFixture(PlannerInput input) {
  const auto map = MapSnapshot::Create(input.world.global_map);
  if (!map.ok()) {
    return PrimitiveReachabilityResult{.reason_code = map.reason_code};
  }
  const auto safe = BuildSafeProjection(
      map.snapshot, input.capability, input.config.map_safety,
      input.stop_token);
  if (!safe.ok()) {
    return PrimitiveReachabilityResult{.reason_code = safe.reason_code};
  }
  return FinalizePrimitiveGraph(
      legged::BuildLeggedPrimitiveGraph(
          std::get<LeggedState>(input.current_state), *safe.projection,
          std::get<LeggedCapability>(input.capability), input.config,
          input.stop_token),
      input.stop_token);
}

TEST(PrimitiveReachabilityGraph,
     LeggedProjectionUsesPrimitiveDirectionsAndReturnability) {
  PlannerInput input = test::MakeValidLeggedInput();
  input.world.global_map = test::MakeFlatMap("map", 8U, 5U, 1.0);
  input.world.local_map = input.world.global_map;
  auto capability = std::get<LeggedCapability>(input.capability);
  capability.motion_primitives.erase(
      std::remove_if(
          capability.motion_primitives.begin(),
          capability.motion_primitives.end(),
          [](const LeggedBodyPrimitive& primitive) {
            return primitive.kind != LeggedPrimitiveKind::kForward &&
                primitive.kind != LeggedPrimitiveKind::kBackward;
          }),
      capability.motion_primitives.end());
  input.capability = capability;
  std::get<LeggedState>(input.current_state).body_pose.position_m =
      {2.5, 2.5, 0.5};

  const auto map = MapSnapshot::Create(input.world.global_map);
  ASSERT_TRUE(map.ok()) << map.reason_code;
  const auto safe = BuildSafeProjection(
      map.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(safe.ok()) << safe.reason_code;
  const GridCell start{.x = 2, .y = 2};
  const GridCell side{.x = 2, .y = 3};
  ASSERT_TRUE(safe.projection->HardFeasible(side));
  ASSERT_EQ(safe.projection->ConnectedComponent(start),
            safe.projection->ConnectedComponent(side));

  const PrimitiveReachabilityResult result = BuildLeggedFixture(input);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.snapshot->algorithm_id,
            "cpp-legged-motion-primitive-recoverable-graph/v1");
  EXPECT_EQ(result.snapshot->reachable[3U * 8U + 2U], 0U);
  EXPECT_EQ(result.snapshot->reachable[2U * 8U + 3U], 1U);
}

TEST(PrimitiveReachabilityGraph, LeggedExcludesOutboundOnlyState) {
  PlannerInput input = test::MakeValidLeggedInput();
  input.world.global_map = test::MakeFlatMap("map", 7U, 5U, 1.0);
  input.world.local_map = input.world.global_map;
  auto capability = std::get<LeggedCapability>(input.capability);
  capability.motion_primitives.erase(
      std::remove_if(
          capability.motion_primitives.begin(),
          capability.motion_primitives.end(),
          [](const LeggedBodyPrimitive& primitive) {
            return primitive.kind != LeggedPrimitiveKind::kForward;
          }),
      capability.motion_primitives.end());
  input.capability = capability;
  std::get<LeggedState>(input.current_state).body_pose.position_m =
      {2.5, 2.5, 0.5};

  const PrimitiveReachabilityResult result = BuildLeggedFixture(input);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const auto iterator = std::find_if(
      result.snapshot->states.begin(), result.snapshot->states.end(),
      [](const PrimitiveReachabilityState& state) {
        return state.cell_x == 3 && state.cell_y == 2;
      });
  ASSERT_NE(iterator, result.snapshot->states.end());
  EXPECT_EQ(iterator->forward_reachable, 1U);
  EXPECT_EQ(iterator->returnable, 0U);
  EXPECT_EQ(result.snapshot->reachable[2U * 7U + 3U], 0U);
}

TEST(PrimitiveReachabilityGraph, LeggedIntervalsRemainDistinctStateKeys) {
  PrimitiveGraphBuildResult graph = MakeDirectedFixture();
  graph.states[1U].body_z_m = Interval{0.4, 0.6};
  graph.states[3U].cell_x = graph.states[1U].cell_x;
  graph.states[3U].cell_y = graph.states[1U].cell_y;
  graph.states[3U].yaw_bin = graph.states[1U].yaw_bin;
  graph.states[3U].motion_mode = graph.states[1U].motion_mode;
  graph.states[3U].position_m = graph.states[1U].position_m;
  graph.states[3U].yaw_rad = graph.states[1U].yaw_rad;
  graph.states[3U].body_z_m = Interval{0.5, 0.6};

  const PrimitiveReachabilityResult result =
      FinalizePrimitiveGraph(std::move(graph), {});

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.snapshot->states.size(), 4U);
}

TEST(PrimitiveReachabilityGraph, StatefulEnginePublishesLeggedGraphSnapshot) {
  PrimitiveReachabilityEngine engine;
  const PrimitiveReachabilityResult result =
      engine.Update(test::MakeValidLeggedInput(), std::nullopt);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.snapshot->algorithm_id,
            "cpp-legged-motion-primitive-recoverable-graph/v1");
}

void KeepOnlyHopperLandingPatches(
    GridMap& map,
    const std::vector<std::pair<std::size_t, std::size_t>>& centers) {
  auto& valid = std::get<std::vector<std::uint8_t>>(
      map.layers.at("valid_mask").values);
  std::fill(valid.begin(), valid.end(), 0U);
  for (const auto [row_center, column_center] : centers) {
    for (std::size_t row = row_center - 2U; row <= row_center + 2U; ++row) {
      for (std::size_t column = column_center - 2U;
           column <= column_center + 2U; ++column) {
        valid.at(row * map.width + column) = 1U;
      }
    }
  }
}

TEST(PrimitiveReachabilityGraph,
     HopperPublishesMultiHopReachabilityButOnlyDirectSuccessors) {
  PlannerInput input = test::MakeValidHopperInput();
  input.world.global_map = test::MakeFlatMap("map", 80U, 16U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 80U, 16U, 0.5);
  input.config.global_map.base_resolution_m = 0.5;
  std::get<HopperState>(input.current_state).pose.position_m =
      {3.25, 4.25, 0.0};
  KeepOnlyHopperLandingPatches(
      input.world.local_map, {{8U, 6U}, {8U, 40U}, {8U, 74U}});
  PrimitiveReachabilityEngine engine;

  const PrimitiveReachabilityResult result = engine.Update(input, 30.0);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.snapshot->algorithm_id,
            "cpp-hopper-certified-recoverable-state-graph/v4");
  const auto state_at = [&](const std::int32_t cell_x) {
    return std::find_if(
        result.snapshot->states.begin(), result.snapshot->states.end(),
        [cell_x](const PrimitiveReachabilityState& state) {
          return state.cell_x == cell_x && state.cell_y == 8;
        });
  };
  const auto middle = state_at(40);
  const auto remote = state_at(74);
  ASSERT_NE(middle, result.snapshot->states.end());
  ASSERT_NE(remote, result.snapshot->states.end());
  EXPECT_EQ(middle->forward_reachable, 1U);
  EXPECT_EQ(middle->returnable, 1U);
  EXPECT_EQ(middle->direct_successor, 1U);
  EXPECT_EQ(remote->forward_reachable, 1U);
  EXPECT_EQ(remote->returnable, 1U);
  EXPECT_EQ(remote->direct_successor, 0U);
  EXPECT_EQ(result.snapshot->reachable[8U * 80U + 74U], 1U);
}

TEST(PrimitiveReachabilityGraph,
     HopperRecordsCertifiedEdgesBetweenAlreadyDiscoveredLandingStates) {
  PlannerInput input = test::MakeValidHopperInput();
  input.world.global_map = test::MakeFlatMap("map", 64U, 16U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 64U, 16U, 0.5);
  input.config.global_map.base_resolution_m = 0.5;
  std::get<HopperState>(input.current_state).pose.position_m =
      {3.25, 4.25, 0.0};
  KeepOnlyHopperLandingPatches(
      input.world.local_map, {{8U, 6U}, {8U, 30U}, {8U, 54U}});
  PrimitiveReachabilityEngine engine;

  const PrimitiveReachabilityResult result = engine.Update(input, 30.0);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const auto state_at = [&](const std::int32_t cell_x)
      -> const PrimitiveReachabilityState* {
    const auto iterator = std::find_if(
        result.snapshot->states.begin(), result.snapshot->states.end(),
        [cell_x](const PrimitiveReachabilityState& state) {
          return state.cell_x == cell_x && state.cell_y == 8;
        });
    return iterator == result.snapshot->states.end() ? nullptr : &*iterator;
  };
  const PrimitiveReachabilityState* middle = state_at(30);
  const PrimitiveReachabilityState* remote = state_at(54);
  ASSERT_NE(middle, nullptr);
  ASSERT_NE(remote, nullptr);
  const auto has_edge = [&](const std::uint64_t source,
                            const std::uint64_t target) {
    return std::ranges::any_of(
        result.snapshot->edges,
        [source, target](const PrimitiveReachabilityEdge& edge) {
          return edge.source_state_id == source &&
              edge.target_state_id == target;
        });
  };
  EXPECT_TRUE(has_edge(middle->state_id, remote->state_id));
  EXPECT_TRUE(has_edge(remote->state_id, middle->state_id));
}

TEST(PrimitiveReachabilityGraph,
     HopperSkipsANumericallyIndeterminateNonAnchorLandingCell) {
  PlannerInput input = test::MakeValidHopperInput();
  input.world.global_map = test::MakeFlatMap("map", 8U, 5U, 1.0);
  input.world.local_map = test::MakeFlatMap("odom", 20U, 20U, 0.6);
  input.world.local_map.origin_m = {0.1, 0.1, 0.0};
  input.config.global_map.base_resolution_m = 1.0;
  std::get<HopperState>(input.current_state).pose.position_m =
      {2.5, 2.5, 0.0};
  auto capability = std::get<HopperCapability>(input.capability);
  capability.landing_support_radius_m = 0.01;
  capability.landing_lateral_margin_m = 0.0;
  input.capability = capability;
  PrimitiveReachabilityEngine engine;

  const PrimitiveReachabilityResult result = engine.Update(input, 30.0);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const auto has_cell = [&](const std::int32_t x, const std::int32_t y) {
    return std::ranges::any_of(
        result.snapshot->states,
        [x, y](const PrimitiveReachabilityState& state) {
          return state.cell_x == x && state.cell_y == y;
        });
  };
  EXPECT_TRUE(has_cell(5, 2));
  EXPECT_FALSE(has_cell(3, 2));
}

}  // namespace
}  // namespace lunar::planning::shared
