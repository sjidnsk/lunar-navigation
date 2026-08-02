#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/wheel/wheel_search.hpp"

namespace lunar::planning::v3 {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] ContentRef Ref(std::string id, char digit) {
  return {std::move(id), 1U, std::string(64U, digit)};
}

[[nodiscard]] WheelMotionPrimitive Primitive(
    std::string id,
    WheelMotionPrimitive::Kind kind,
    PoseXyzYaw relative_end,
    std::chrono::nanoseconds duration) {
  return {
      .primitive_id = std::move(id),
      .kind = kind,
      .relative_end_pose = relative_end,
      .nominal_duration = DurationNanoseconds{duration},
      .swept_geometry_ref = Ref("sweep", 'b'),
  };
}

struct SearchFixture final {
  std::shared_ptr<const ImmutableMapSnapshot> map;
  SafetyCapabilityProfile profile;
  PlannerAlgorithmConfig algorithm;
  SafeProjection projection;
  WheeledOrLeggedState current_state;
  ResolvedTerminalSet terminals;
  WheelPrimitiveCatalog catalog;
};

[[nodiscard]] MapSnapshotInput MapInput() {
  constexpr std::size_t kWidth = 12U;
  constexpr std::size_t kHeight = 12U;
  constexpr std::size_t kCount = kWidth * kHeight;
  return {
      .snapshot_ref = Ref("map-snapshot", '1'),
      .map_revision = 1U,
      .immutable_data_handle = "wheel-search-map",
      .source_time = {.clock_id = "mission", .tick = 1ns},
      .bounds =
          {
              .minimum_m = {0.0, 0.0, -1.0},
              .maximum_m = {12.0, 12.0, 1.0},
          },
      .geometry =
          {
              .width = kWidth,
              .height = kHeight,
              .resolution_m = 1.0,
              .origin_m = {0.0, 0.0},
              .frame_id = "map",
          },
      .layer_manifest =
          {
              {LayerKind::kKnownMask, Ref("known", '2')},
              {LayerKind::kElevation, Ref("elevation", '3')},
              {LayerKind::kTerrainNormal, Ref("normal", '4')},
              {LayerKind::kRoughness, Ref("roughness", '5')},
              {LayerKind::kHardObstacle, Ref("obstacle", '6')},
              {LayerKind::kConfidence, Ref("confidence", '7')},
          },
      .known_mask = std::vector<std::uint8_t>(kCount, 1U),
      .elevation_m = std::vector<float>(kCount, 0.0F),
      .normal_x = std::vector<float>(kCount, 0.0F),
      .normal_y = std::vector<float>(kCount, 0.0F),
      .normal_z = std::vector<float>(kCount, 1.0F),
      .roughness_m = std::vector<float>(kCount, 0.0F),
      .hard_obstacle_mask =
          std::vector<std::uint8_t>(kCount, 0U),
      .confidence = std::vector<float>(kCount, 1.0F),
  };
}

[[nodiscard]] SearchFixture MakeTwoRouteProblem(
    std::chrono::nanoseconds fast_duration,
    std::chrono::nanoseconds slow_duration,
    std::chrono::nanoseconds time_equivalence) {
  SearchFixture fixture;
  auto map_result = ImmutableMapSnapshot::Create(MapInput());
  EXPECT_TRUE(IsOk(map_result));
  fixture.map =
      std::get<std::shared_ptr<const ImmutableMapSnapshot>>(
          std::move(map_result));

  fixture.profile = {
      .content_ref = Ref("wheel-capability", 'a'),
      .content =
          WheeledCapability{
              .frame_id = "map",
              .collision_envelope =
                  {
                      .vertices_xy_m =
                          {{-0.2, -0.2}, {0.2, -0.2},
                           {0.2, 0.2}, {-0.2, 0.2}},
                      .minimum_z_m = -0.1,
                      .maximum_z_m = 0.5,
                  },
              .motion_model_ref = Ref("motion", 'c'),
              .analytic_cost_model_ref = Ref("cost", 'd'),
              .hard_limits =
                  {
                      .maximum_forward_speed_mps = 1.0,
                      .maximum_reverse_speed_mps = 0.8,
                      .maximum_spin_rate_radps = 1.0,
                      .maximum_forward_acceleration_mps2 = 1.0,
                      .maximum_braking_deceleration_mps2 = 1.0,
                      .maximum_yaw_acceleration_radps2 = 1.0,
                      .maximum_lateral_acceleration_mps2 = 1.0,
                      .maximum_drive_curvature_per_m = 1.0,
                      .maximum_slope_rad = 0.5,
                      .minimum_clearance_m = 0.0,
                  },
              .motion_primitives =
                  {
                      Primitive(
                          "fast-forward",
                          WheelMotionPrimitive::Kind::kDriveForwardLine,
                          {{1.0, 0.0, 0.0}, 0.0},
                          fast_duration),
                      Primitive(
                          "slow-forward",
                          WheelMotionPrimitive::Kind::kDriveForwardArc,
                          {{0.0, 1.0, 0.0}, 0.0},
                          slow_duration),
                      Primitive(
                          "reverse",
                          WheelMotionPrimitive::Kind::kDriveReverseLine,
                          {{-1.0, 0.0, 0.0}, 0.0}, 100s),
                      Primitive("spin-cw",
                                WheelMotionPrimitive::Kind::kSpinCw,
                                {{0.0, 0.0, 0.0}, -0.5}, 100s),
                      Primitive("spin-ccw",
                                WheelMotionPrimitive::Kind::kSpinCcw,
                                {{0.0, 0.0, 0.0}, 0.5}, 100s),
                      Primitive(
                          "switch",
                          WheelMotionPrimitive::Kind::kStopAndSwitch,
                          {{0.0, 0.0, 0.0}, 0.0}, 500ms),
                  },
          },
  };

  fixture.algorithm.content_ref = Ref("algorithm", 'e');
  fixture.algorithm.time_equivalence_tolerance =
      DurationNanoseconds{time_equivalence};
  fixture.algorithm.wheeled.state_lattice = {
      .xy_resolution_m = 1.0,
      .yaw_bin_count = 16U,
      .maximum_terminal_candidates = 8U,
  };
  fixture.algorithm.ara_star = {
      .initial_epsilon = 1.0,
      .epsilon_decrement = 0.1,
      .target_epsilon = 1.0,
      .resource_caps =
          {
              .maximum_expanded_states = 128U,
              .maximum_reopened_states = 128U,
              .maximum_generated_candidates = 8U,
              .maximum_open_states = 128U,
              .maximum_memory_bytes = 1U << 20U,
          },
  };

  const auto projection_result = BuildSafeProjection(
      {
          .map = fixture.map,
          .capability =
              std::make_shared<const SafetyCapabilityProfile>(
                  fixture.profile),
          .algorithm_config =
              std::make_shared<const PlannerAlgorithmConfig>(
                  fixture.algorithm),
          .learned_cost = nullptr,
      });
  EXPECT_TRUE(IsOk(projection_result));
  fixture.projection =
      std::get<SafeProjection>(projection_result);
  const std::size_t fast_index = 4U * 12U + 5U;
  const std::size_t slow_index = 5U * 12U + 4U;
  fixture.projection.resolved_energy[fast_index] = 20.0F;
  fixture.projection.resolved_energy[slow_index] = 1.0F;
  fixture.projection.resolved_nonfatal_risk[fast_index] = 0.0F;
  fixture.projection.resolved_nonfatal_risk[slow_index] = 0.0F;

  fixture.current_state = {
      .position_m = {4.5, 4.5, 0.0},
      .yaw_rad = 0.0,
      .linear_velocity_mps = {0.0, 0.0, 0.0},
      .yaw_rate_radps = 0.0,
  };
  fixture.terminals = {
      .kind = TerminalKind::kGoal,
      .candidates =
          {
              {
                  .stable_id = "fast-goal",
                  .cell = {5, 4},
                  .position_m = {5.5, 4.5, 0.0},
                  .requires_zero_speed = true,
                  .clearance_m = 5.0,
              },
              {
                  .stable_id = "slow-goal",
                  .cell = {4, 5},
                  .position_m = {4.5, 5.5, 0.0},
                  .requires_zero_speed = true,
                  .clearance_m = 5.0,
              },
          },
      .reason_code = "GOAL_REACHABLE",
  };
  fixture.catalog = std::get<WheelPrimitiveCatalog>(
      WheelPrimitiveCatalog::Create(fixture.profile));
  return fixture;
}

[[nodiscard]] Result<WheelDiscretePlan> Plan(
    SearchFixture& fixture) {
  return PlanWheelDiscrete(
      WheelPlanningProblem{
          fixture.projection,
          fixture.current_state,
          fixture.terminals,
          fixture.profile,
          fixture.algorithm,
      },
      fixture.catalog, fixture.algorithm.ara_star);
}

TEST(WheelSearchTest, TimeDominatesEnergyOutsideEquivalencePool) {
  auto fixture = MakeTwoRouteProblem(10s, 10600ms, 500ms);

  const auto result = Plan(fixture);

  ASSERT_TRUE(IsOk(result));
  const auto& plan = std::get<WheelDiscretePlan>(result);
  EXPECT_EQ(plan.expected_time.value, 11s);
  EXPECT_DOUBLE_EQ(plan.total_secondary_costs.energy, 20.0);
}

TEST(WheelSearchTest, EnergyBreaksTieInsideTimeEquivalencePool) {
  auto fixture = MakeTwoRouteProblem(10s, 10400ms, 500ms);

  const auto result = Plan(fixture);

  ASSERT_TRUE(IsOk(result));
  const auto& plan = std::get<WheelDiscretePlan>(result);
  EXPECT_EQ(plan.expected_time.value, 11400ms);
  EXPECT_DOUBLE_EQ(plan.total_secondary_costs.energy, 1.0);
}

TEST(WheelSearchTest,
     HubEdgesRemainInsideOneForwardExecutionSegment) {
  auto fixture = MakeTwoRouteProblem(10s, 10600ms, 500ms);

  const auto result = Plan(fixture);

  ASSERT_TRUE(IsOk(result));
  const auto& plan = std::get<WheelDiscretePlan>(result);
  ASSERT_EQ(plan.segments.size(), 1U);
  EXPECT_EQ(plan.segments.front().mode,
            WheelMotionMode::kForward);
  ASSERT_EQ(plan.segments.front().edges.size(), 3U);
  EXPECT_EQ(plan.segments.front().edges.front().source.motion_mode,
            WheelMotionMode::kStart);
  EXPECT_EQ(plan.segments.front().edges.back().target.motion_mode,
            WheelMotionMode::kStart);
  EXPECT_EQ(plan.safe_stop_anchor.pose.position_m.x, 5.5);
  EXPECT_EQ(plan.safe_stop_anchor.target_linear_velocity_mps, 0.0);
  EXPECT_EQ(plan.safe_stop_anchor.target_yaw_rate_radps, 0.0);
}

}  // namespace
}  // namespace lunar::planning::v3
