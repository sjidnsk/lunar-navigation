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

#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/legged/legged_planner.hpp"

namespace lunar::planning::v3 {
namespace {

ContentRef Ref(std::string id, char digit) {
  return ContentRef{
      .id = std::move(id),
      .revision = 1U,
      .content_hash = std::string(64U, digit),
  };
}

template <class T>
std::shared_ptr<const T> OpaqueResolvedObject() {
  auto owner = std::make_shared<int>(1);
  const T* opaque =
      reinterpret_cast<const T*>(owner.get());
  return std::shared_ptr<const T>(
      std::move(owner), opaque);
}

MapSnapshotInput MapInput() {
  constexpr std::size_t kWidth = 7U;
  constexpr std::size_t kHeight = 5U;
  constexpr std::size_t kCount = kWidth * kHeight;
  return MapSnapshotInput{
      .snapshot_ref = Ref("planner-map", 'a'),
      .map_revision = 1U,
      .immutable_data_handle = "legged-planner-map",
      .source_time =
          ClockStamp{"mission", std::chrono::nanoseconds{1}},
      .bounds =
          {
              .minimum_m = {0.0, 0.0, -1.0},
              .maximum_m = {7.0, 5.0, 2.0},
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

LeggedBodyPrimitive Primitive(
    std::string id, LeggedBodyPrimitive::Kind kind,
    Vec3 displacement, double yaw_change_rad) {
  return LeggedBodyPrimitive{
      .primitive_id = std::move(id),
      .kind = kind,
      .body_frame_displacement_m = displacement,
      .yaw_change_rad = yaw_change_rad,
      .nominal_duration =
          DurationNanoseconds{std::chrono::seconds{2}},
      .sampled_body_sweep_ref = Ref("body-sweep", 'c'),
  };
}

SafetyCapabilityProfile Capability() {
  LeggedCapability legged{
      .frame_id = "map",
      .reference_point_id = "base_link",
      .collision_envelope = BodyEnvelope(),
      .motion_model_ref = Ref("motion", 'd'),
      .analytic_cost_model_ref = Ref("analytic-cost", 'e'),
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
  legged.motion_primitives = {
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
      .content_ref = Ref("legged-capability", 'f'),
      .content = std::move(legged),
  };
}

PlannerAlgorithmConfig Algorithm() {
  PlannerAlgorithmConfig config{};
  config.content_ref = Ref("legged-algorithm", 'b');
  config.time_equivalence_tolerance =
      DurationNanoseconds{std::chrono::milliseconds{100}};
  config.ara_star =
      AraStarConfig{
          .initial_epsilon = 2.0,
          .epsilon_decrement = 0.5,
          .target_epsilon = 1.0,
          .resource_caps =
              {
                  .maximum_expanded_states = 200U,
                  .maximum_reopened_states = 200U,
                  .maximum_generated_candidates = 16U,
                  .maximum_open_states = 200U,
                  .maximum_memory_bytes = 1U << 20U,
              },
      };
  config.legged.pose_lattice =
      {
          .xy_resolution_m = 1.0,
          .yaw_bin_count = 4U,
          .maximum_terminal_candidates = 8U,
      };
  config.legged.maximum_height_interval_splits = 8U;
  config.legged.corridor =
      {
          .maximum_regions = 8U,
          .maximum_inflation_iterations = 256U,
          .maximum_halfplanes_per_region = 32U,
          .maximum_split_depth = 2U,
          .minimum_overlap_m = 0.05,
          .sampling_spacing_m = 0.5,
      };
  config.legged.smoothing =
      {
          .maximum_scp_iterations = 2U,
          .maximum_trust_region_reductions = 1U,
          .initial_trust_region_m = 0.1,
          .minimum_trust_region_m = 0.01,
          .constraint_tolerance = 1.0e-8,
          .maximum_time_increase =
              DurationNanoseconds{std::chrono::seconds{1}},
      };
  config.legged.time_scaling =
      {
          .maximum_adaptive_samples = 33U,
          .minimum_parameter_step = 1.0e-4,
          .maximum_forward_passes = 1U,
          .maximum_backward_passes = 1U,
          .enable_jerk_smoothing = false,
          .maximum_jerk_smoothing_iterations = 0U,
      };
  config.legged.continuous_validation_maximum_subdivisions =
      33U;
  config.learned_cost_policy.mode =
      LearnedCostPolicy::Mode::kDisabled;
  return config;
}

struct Fixture final {
  PlanningRequest request;
  ResolvedTerminalSet terminal;
};

Fixture MakeFixture() {
  auto map_result = ImmutableMapSnapshot::Create(MapInput());
  EXPECT_TRUE(IsOk(map_result));
  auto map =
      std::get<std::shared_ptr<const ImmutableMapSnapshot>>(
          std::move(map_result));
  auto capability =
      std::make_shared<const SafetyCapabilityProfile>(Capability());
  auto algorithm =
      std::make_shared<const PlannerAlgorithmConfig>(Algorithm());
  const auto& legged = std::get<LeggedCapability>(
      capability->content);

  PlanningRequest request{
      .request_id = "legged-request",
      .request_time =
          ClockStamp{"mission", std::chrono::nanoseconds{10}},
      .state_time =
          ClockStamp{"mission", std::chrono::nanoseconds{10}},
      .frame_id = "map",
      .platform_type = PlatformType::kLegged,
      .current_state =
          WheeledOrLeggedState{
              .position_m = {1.5, 2.5, 0.5},
              .yaw_rad = 0.0,
              .linear_velocity_mps = {},
              .yaw_rate_radps = 0.0,
          },
      .goal =
          GoalRegion{
              .goal_id = "goal",
              .target =
                  PointGoal{
                      .position_m = {3.5, 2.5, 0.0},
                      .position_tolerance_m = 0.25,
                  },
          },
      .map_snapshot = std::move(map),
      .safety_capability = capability,
      .algorithm_config = algorithm,
      .capability_bindings =
          ResolvedCapabilityBindings{
              .motion_model =
                  {
                      .content_ref = legged.motion_model_ref,
                      .object =
                          OpaqueResolvedObject<MotionModel>(),
                  },
              .analytic_cost_model =
                  {
                      .content_ref =
                          legged.analytic_cost_model_ref,
                      .object =
                          OpaqueResolvedObject<AnalyticCostModel>(),
                  },
          },
  };
  ResolvedTerminalSet terminal{
      .kind = TerminalKind::kGoal,
      .candidates =
          {
              TerminalCandidate{
                  .stable_id = "goal-cell",
                  .cell = {3, 2},
                  .position_m = {3.5, 2.5, 0.0},
                  .yaw_interval = std::nullopt,
                  .requires_zero_speed = true,
                  .clearance_m = 1.0,
              },
          },
  };
  return {
      .request = std::move(request),
      .terminal = std::move(terminal),
  };
}

class FailingQpSolver final : public BoundedQpSolver {
 public:
  [[nodiscard]] std::size_t call_count() const noexcept {
    return call_count_;
  }

 private:
  [[nodiscard]] Result<QpSolution> DoSolve(
      const SparseQpProblem&,
      const BoundedQpSettings&) const override {
    ++call_count_;
    return Error{
        .code = ErrorCode::kNumericalFailure,
        .field_path = "/test_qp",
        .message = "INTENTIONAL_QP_FAILURE",
    };
  }

  mutable std::size_t call_count_{};
};

TEST(LeggedPlannerTest, AlwaysDeclaresLimitedFeasibilityScope) {
  const auto fixture = MakeFixture();
  const auto result =
      LeggedPlanner{}.Plan(fixture.request, fixture.terminal);

  ASSERT_TRUE(IsOk(result));
  const auto& reference = std::get<LeggedBodyReference>(result);
  EXPECT_EQ(reference.feasibility_scope,
            "body_geometry_and_terrain_thresholds_only");
  EXPECT_FALSE(reference.footstep_feasibility_guaranteed);
  EXPECT_EQ(reference.reference_point_id, "base_link");
}

TEST(LeggedPlannerTest, SplineFailureReturnsPrimitiveChainOnly) {
  const auto fixture = MakeFixture();
  FailingQpSolver failing_qp;
  const LeggedPlanner planner{
      LeggedPlannerDependencies{
          .qp_solver = &failing_qp,
      }};

  const auto result =
      planner.Plan(fixture.request, fixture.terminal);

  ASSERT_TRUE(IsOk(result));
  EXPECT_GT(failing_qp.call_count(), 0U);
  EXPECT_TRUE(std::holds_alternative<ValidatedPrimitiveChain>(
      std::get<LeggedBodyReference>(result).geometric_path));
}

TEST(LeggedPlannerTest,
     FinalReferenceHasStableIdentityOriginAndJcsHash) {
  const auto fixture = MakeFixture();
  const LeggedPlanner planner;

  const auto first =
      planner.Plan(fixture.request, fixture.terminal);
  const auto second =
      planner.Plan(fixture.request, fixture.terminal);

  ASSERT_TRUE(IsOk(first));
  ASSERT_TRUE(IsOk(second));
  const auto& a = std::get<LeggedBodyReference>(first);
  const auto& b = std::get<LeggedBodyReference>(second);
  ASSERT_FALSE(a.reference_id.empty());
  EXPECT_EQ(a.reference_id, b.reference_id);
  EXPECT_EQ(a.reference_time_origin.clock_id,
            fixture.request.request_time.clock_id);
  EXPECT_EQ(a.reference_time_origin.tick,
            fixture.request.request_time.tick);
  const auto canonical =
      CanonicalReferenceHash(PlatformReference{a});
  ASSERT_TRUE(IsOk(canonical));
  EXPECT_EQ(a.reference_hash,
            std::get<Sha256Digest>(canonical));
  EXPECT_EQ(a.reference_hash, b.reference_hash);
}

}  // namespace
}  // namespace lunar::planning::v3
