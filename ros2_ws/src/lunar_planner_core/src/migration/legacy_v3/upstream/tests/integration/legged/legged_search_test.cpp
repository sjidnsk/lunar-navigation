#include <chrono>
#include <cstdint>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/legged/legged_search.hpp"

namespace lunar::planning::v3 {
namespace {

ContentRef Ref(std::string id, char digit) {
  return ContentRef{
      .id = std::move(id),
      .revision = 1U,
      .content_hash = std::string(64U, digit),
  };
}

MapSnapshotInput MapInput() {
  constexpr std::size_t kWidth = 5U;
  constexpr std::size_t kHeight = 5U;
  constexpr std::size_t kCount = kWidth * kHeight;
  return MapSnapshotInput{
      .snapshot_ref = Ref("map", 'a'),
      .map_revision = 1U,
      .immutable_data_handle = "legged-search-map",
      .source_time =
          ClockStamp{"mission", std::chrono::nanoseconds{1}},
      .bounds =
          MapBounds{
              .minimum_m = {0.0, 0.0, -1.0},
              .maximum_m = {5.0, 5.0, 2.0},
          },
      .geometry =
          GridGeometry{
              .width = kWidth,
              .height = kHeight,
              .resolution_m = 1.0,
              .origin_m = {0.0, 0.0},
              .frame_id = "map",
          },
      .layer_manifest =
          {
              {LayerKind::kKnownMask, Ref("known", '1')},
              {LayerKind::kElevation, Ref("elevation", '2')},
              {LayerKind::kTerrainNormal, Ref("normal", '3')},
              {LayerKind::kRoughness, Ref("roughness", '4')},
              {LayerKind::kHardObstacle, Ref("obstacle", '5')},
              {LayerKind::kConfidence, Ref("confidence", '6')},
          },
      .known_mask = std::vector<std::uint8_t>(kCount, 1U),
      .elevation_m = std::vector<float>(kCount, 0.0F),
      .normal_x = std::vector<float>(kCount, 0.0F),
      .normal_y = std::vector<float>(kCount, 0.0F),
      .normal_z = std::vector<float>(kCount, 1.0F),
      .roughness_m = std::vector<float>(kCount, 0.0F),
      .hard_obstacle_mask = std::vector<std::uint8_t>(kCount, 0U),
      .confidence = std::vector<float>(kCount, 1.0F),
  };
}

BodyConvexPolytope BodyEnvelope() {
  return BodyConvexPolytope{
      .body_frame_halfspaces =
          ConvexPolytope3{
              .halfspaces =
                  {
                      {{1.0, 0.0, 0.0}, 0.2},
                      {{-1.0, 0.0, 0.0}, 0.2},
                      {{0.0, 1.0, 0.0}, 0.2},
                      {{0.0, -1.0, 0.0}, 0.2},
                      {{0.0, 0.0, 1.0}, 0.3},
                      {{0.0, 0.0, -1.0}, 0.3},
                  },
          },
  };
}

LeggedBodyPrimitive Primitive(std::string id,
                              LeggedBodyPrimitive::Kind kind,
                              Vec3 displacement,
                              double yaw_change_rad) {
  return LeggedBodyPrimitive{
      .primitive_id = std::move(id),
      .kind = kind,
      .body_frame_displacement_m = displacement,
      .yaw_change_rad = yaw_change_rad,
      .nominal_duration =
          DurationNanoseconds{std::chrono::seconds{2}},
      .sampled_body_sweep_ref = Ref("sweep", 'c'),
  };
}

SafetyCapabilityProfile CapabilityProfile() {
  LeggedCapability capability{
      .frame_id = "map",
      .reference_point_id = "base_link",
      .collision_envelope = BodyEnvelope(),
      .motion_model_ref = Ref("motion", 'd'),
      .analytic_cost_model_ref = Ref("cost", 'e'),
      .terrain_thresholds =
          {
              .maximum_slope_rad = 0.6,
              .maximum_roughness_m = 0.2,
              .maximum_step_height_m = 0.3,
              .maximum_gap_width_m = 0.4,
              .minimum_confidence = 0.8,
              .minimum_body_clearance_m = 0.0,
              .minimum_body_height_m = 0.4,
              .maximum_body_height_m = 0.6,
          },
      .body_velocity_limits =
          {
              .forward_mps = {-0.5, 0.5},
              .lateral_mps = {-0.5, 0.5},
              .vertical_mps = {-0.1, 0.1},
              .yaw_rate_radps = {-1.0, 1.0},
              .linear_acceleration_mps2 = 0.5,
              .yaw_acceleration_radps2 = 1.0,
          },
  };
  capability.motion_primitives = {
      Primitive("forward", LeggedBodyPrimitive::Kind::kForward,
                {1.0, 0.0, 0.0}, 0.0),
      Primitive("backward", LeggedBodyPrimitive::Kind::kBackward,
                {-1.0, 0.0, 0.0}, 0.0),
      Primitive("left", LeggedBodyPrimitive::Kind::kLateral,
                {0.0, 1.0, 0.0}, 0.0),
      Primitive("right", LeggedBodyPrimitive::Kind::kLateral,
                {0.0, -1.0, 0.0}, 0.0),
      Primitive("spin", LeggedBodyPrimitive::Kind::kSpin,
                {0.0, 0.0, 0.0}, std::numbers::pi / 2.0),
  };
  return SafetyCapabilityProfile{
      .content_ref = Ref("capability", 'f'),
      .content = std::move(capability),
  };
}

PlannerAlgorithmConfig AlgorithmConfig() {
  PlannerAlgorithmConfig config{};
  config.content_ref = Ref("algorithm", 'b');
  config.time_equivalence_tolerance =
      DurationNanoseconds{std::chrono::milliseconds{100}};
  config.ara_star =
      AraStarConfig{
          .initial_epsilon = 2.0,
          .epsilon_decrement = 0.5,
          .target_epsilon = 1.0,
          .resource_caps =
              ResourceCaps{
                  .maximum_expanded_states = 200U,
                  .maximum_reopened_states = 200U,
                  .maximum_generated_candidates = 16U,
                  .maximum_open_states = 200U,
                  .maximum_memory_bytes = 1U << 20U,
              },
      };
  config.legged.pose_lattice =
      GridConfig{
          .xy_resolution_m = 1.0,
          .yaw_bin_count = 4U,
          .maximum_terminal_candidates = 8U,
      };
  return config;
}

struct Scenario final {
  std::shared_ptr<const ImmutableMapSnapshot> map;
  SafetyCapabilityProfile capability;
  PlannerAlgorithmConfig algorithm;
  SafeProjection projection;
  BodyMotionPrimitiveCatalog catalog;
};

Scenario MakeScenario() {
  auto map_result = ImmutableMapSnapshot::Create(MapInput());
  EXPECT_TRUE(IsOk(map_result));
  auto map =
      std::get<std::shared_ptr<const ImmutableMapSnapshot>>(
          std::move(map_result));
  auto capability = CapabilityProfile();
  auto algorithm = AlgorithmConfig();
  auto projection_result = BuildSafeProjection(
      SafeProjectionRequest{
          .map = map,
          .capability =
              std::make_shared<const SafetyCapabilityProfile>(
                  capability),
          .algorithm_config =
              std::make_shared<const PlannerAlgorithmConfig>(algorithm),
          .learned_cost = nullptr,
      });
  EXPECT_TRUE(IsOk(projection_result));
  auto catalog_result = BodyMotionPrimitiveCatalog::Create(capability);
  EXPECT_TRUE(IsOk(catalog_result));
  return Scenario{
      .map = std::move(map),
      .capability = std::move(capability),
      .algorithm = std::move(algorithm),
      .projection = std::get<SafeProjection>(
          std::move(projection_result)),
      .catalog = std::get<BodyMotionPrimitiveCatalog>(
          std::move(catalog_result)),
  };
}

TEST(LeggedSearchTest, PlansDeterministicBodyPathWithSafeStopPose) {
  const auto scenario = MakeScenario();
  const std::vector<TerminalCandidate> terminals{
      TerminalCandidate{
          .stable_id = "goal",
          .cell = {3, 2},
          .position_m = {3.5, 2.5, 0.0},
          .yaw_interval = std::nullopt,
          .requires_zero_speed = true,
          .clearance_m = 1.0,
      },
  };
  const LeggedPlanningProblem problem{
      .projection = scenario.projection,
      .start =
          LeggedLatticeState{
              .ix = 1,
              .iy = 2,
              .iyaw = 0,
              .reachable_z = {0.4, 0.6},
          },
      .terminals = terminals,
      .grid = scenario.algorithm.legged.pose_lattice,
      .time_equivalence_tolerance =
          scenario.algorithm.time_equivalence_tolerance,
  };

  const auto first = PlanLeggedDiscrete(
      problem, scenario.catalog, scenario.algorithm.ara_star);
  const auto second = PlanLeggedDiscrete(
      problem, scenario.catalog, scenario.algorithm.ara_star);

  ASSERT_TRUE(IsOk(first));
  ASSERT_TRUE(IsOk(second));
  const auto& first_plan = std::get<LeggedDiscretePlan>(first);
  const auto& second_plan = std::get<LeggedDiscretePlan>(second);
  EXPECT_EQ(first_plan.stable_candidate_id,
            second_plan.stable_candidate_id);
  EXPECT_EQ(first_plan.states.back().ix, 3);
  EXPECT_EQ(first_plan.states.back().iy, 2);
  EXPECT_DOUBLE_EQ(first_plan.terminal_pose.position_m.z, 0.5);
  EXPECT_DOUBLE_EQ(first_plan.target_linear_velocity_mps, 0.0);
  EXPECT_DOUBLE_EQ(first_plan.target_yaw_rate_radps, 0.0);
}

}  // namespace
}  // namespace lunar::planning::v3
