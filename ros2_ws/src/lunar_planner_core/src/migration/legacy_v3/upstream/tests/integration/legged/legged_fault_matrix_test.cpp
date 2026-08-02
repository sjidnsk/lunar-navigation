#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "legged_test_fixture.hpp"
#include "lunar_path_planner/v3/legged/legged_terrain.hpp"

namespace lunar::planning::v3 {
namespace {

class RejectEveryProduct final
    : public LeggedCartesianProductCertifier {
 public:
  [[nodiscard]] bool Certify(
      const LeggedProductBox&) const noexcept override {
    return false;
  }
};

class FailingQpSolver final : public BoundedQpSolver {
 public:
  [[nodiscard]] std::size_t call_count() const noexcept {
    return calls_;
  }

 private:
  [[nodiscard]] Result<QpSolution> DoSolve(
      const SparseQpProblem&,
      const BoundedQpSettings&) const override {
    ++calls_;
    return Error{
        .code = ErrorCode::kNumericalFailure,
        .field_path = "/fault_matrix/qp",
        .message = "QP_INFEASIBLE",
    };
  }

  mutable std::size_t calls_{};
};

struct DiscreteSetup final {
  legged_test::Fixture fixture;
  SafeProjection projection;
  LeggedCapabilityView capability;
  LeggedDiscretePlan plan;
};

DiscreteSetup MakeDiscreteSetup() {
  auto fixture = legged_test::MakeFixture();
  const auto projection_result =
      BuildSafeProjection(
          {
              .map = fixture.request.map_snapshot,
              .capability =
                  fixture.request.safety_capability,
              .algorithm_config =
                  fixture.request.algorithm_config,
              .learned_cost = nullptr,
          });
  if (!IsOk(projection_result)) {
    throw std::runtime_error("projection setup failed");
  }
  const auto capability_result =
      LeggedCapabilityView::Create(
          *fixture.request.safety_capability);
  const auto catalog_result =
      BodyMotionPrimitiveCatalog::Create(
          *fixture.request.safety_capability);
  if (!IsOk(capability_result) || !IsOk(catalog_result)) {
    throw std::runtime_error("capability setup failed");
  }
  auto projection =
      std::get<SafeProjection>(projection_result);
  auto capability =
      std::get<LeggedCapabilityView>(capability_result);
  const auto& algorithm =
      *fixture.request.algorithm_config;
  const auto plan_result =
      PlanLeggedDiscrete(
          LeggedPlanningProblem{
              .projection = projection,
              .start =
                  {
                      .ix = 1,
                      .iy = 2,
                      .iyaw = 0,
                      .reachable_z = {0.4, 0.6},
                  },
              .terminals = fixture.terminal.candidates,
              .grid = algorithm.legged.pose_lattice,
              .time_equivalence_tolerance =
                  algorithm.time_equivalence_tolerance,
          },
          std::get<BodyMotionPrimitiveCatalog>(
              catalog_result),
          algorithm.ara_star);
  if (!IsOk(plan_result)) {
    throw std::runtime_error("discrete setup failed");
  }
  return {
      .fixture = std::move(fixture),
      .projection = std::move(projection),
      .capability = std::move(capability),
      .plan = std::get<LeggedDiscretePlan>(
          plan_result),
  };
}

TEST(LeggedFaultMatrixTest,
     FrozenTerrainHardFailureCausesAreRejected) {
  auto profile = legged_test::Capability();
  auto& source = std::get<LeggedCapability>(profile.content);
  source.terrain_thresholds.minimum_body_clearance_m = 0.2;
  const auto capability_result =
      LeggedCapabilityView::Create(profile);
  ASSERT_TRUE(IsOk(capability_result));
  const auto& capability =
      std::get<LeggedCapabilityView>(capability_result);

  const LeggedTerrainCellSample baseline{
      .known = true,
      .hard_obstacle = false,
      .confidence = 1.0,
      .elevation_m = 0.0,
      .normal = {0.0, 0.0, 1.0},
      .roughness_m = 0.0,
      .maximum_neighbor_step_m = 0.0,
      .unsupported_gap_width_m = 0.0,
      .body_clearance_m = 0.5,
      .analytic_time_cost_s = 1.0,
      .resolved_energy = 1.0,
      .resolved_nonfatal_risk = 0.1,
  };
  struct FaultCase final {
    const char* name;
    const char* rejection_reason;
    std::function<void(LeggedTerrainCellSample&)> inject;
  };
  const std::vector<FaultCase> cases{
      {
          "unknown_or_low_confidence",
          "known_and_confident",
          [](auto& sample) { sample.confidence = 0.1; },
      },
      {
          "slope_limit",
          "slope",
          [](auto& sample) {
            sample.normal = {std::sin(0.7), 0.0,
                             std::cos(0.7)};
          },
      },
      {
          "roughness_limit",
          "roughness_and_plane_residual",
          [](auto& sample) { sample.roughness_m = 0.3; },
      },
      {
          "step_limit",
          "step_height",
          [](auto& sample) {
            sample.maximum_neighbor_step_m = 0.4;
          },
      },
      {
          "gap_limit",
          "gap_width",
          [](auto& sample) {
            sample.unsupported_gap_width_m = 0.5;
          },
      },
      {
          "top_clearance",
          "body_collision_and_clearance",
          [](auto& sample) { sample.body_clearance_m = 0.1; },
      },
  };

  for (const auto& fault : cases) {
    SCOPED_TRACE(fault.name);
    auto sample = baseline;
    fault.inject(sample);
    const auto evaluation =
        EvaluateLeggedTerrainSample(sample, capability);
    EXPECT_FALSE(evaluation.hard_feasible);
    EXPECT_NE(
        std::find(evaluation.rejection_reasons.begin(),
                  evaluation.rejection_reasons.end(),
                  fault.rejection_reason),
        evaluation.rejection_reasons.end());
    EXPECT_DOUBLE_EQ(evaluation.secondary_costs.energy, 0.0);
    EXPECT_DOUBLE_EQ(evaluation.secondary_costs.risk, 0.0);
  }
}

TEST(LeggedFaultMatrixTest, HeightIntervalDisconnectRejectsEdge) {
  auto profile = legged_test::Capability();
  const auto catalog_result =
      BodyMotionPrimitiveCatalog::Create(profile);
  ASSERT_TRUE(IsOk(catalog_result));
  const auto& primitive =
      std::get<BodyMotionPrimitiveCatalog>(catalog_result)
          .ordered_primitives()
          .front();
  const std::vector<LeggedTerrainEvaluation> samples{
      LeggedTerrainEvaluation{
          .hard_feasible = true,
          .body_height_interval = {0.4, 0.6},
      },
      LeggedTerrainEvaluation{
          .hard_feasible = true,
          .body_height_interval = {1.0, 1.2},
      },
  };

  EXPECT_FALSE(
      PropagateLeggedEdgeHeightInterval(
          HeightInterval{0.4, 0.6}, primitive, samples)
          .has_value());
}

TEST(LeggedFaultMatrixTest,
     CartesianProductCornerUnsafeRejectsCorridor) {
  auto setup = MakeDiscreteSetup();
  auto config =
      setup.fixture.request.algorithm_config->legged.corridor;
  config.maximum_split_depth = 0U;
  const RejectEveryProduct reject;

  const auto result =
      BuildLeggedCorridor(
          {
              .projection = setup.projection,
              .discrete_plan = setup.plan,
              .capability = setup.capability,
              .config = config,
              .product_certifier = &reject,
          });

  EXPECT_FALSE(IsOk(result));
}

TEST(LeggedFaultMatrixTest,
     QpInfeasibleFallsBackToWholePrimitiveChain) {
  const auto fixture = legged_test::MakeFixture();
  FailingQpSolver solver;
  const LeggedPlanner planner{
      {
          .qp_solver = &solver,
      }};

  const auto result =
      planner.Plan(fixture.request, fixture.terminal);

  ASSERT_TRUE(IsOk(result));
  EXPECT_GT(solver.call_count(), 0U);
  EXPECT_TRUE(std::holds_alternative<ValidatedPrimitiveChain>(
      std::get<LeggedBodyReference>(result).geometric_path));
}

TEST(LeggedFaultMatrixTest,
     TimingResourceLimitIsExplicit) {
  const auto fixture = legged_test::MakeFixture();
  const auto& legged = std::get<LeggedCapability>(
      fixture.request.safety_capability->content);
  const GeometricPath path{
      ValidatedPrimitiveChain{
          .primitives =
              {
                  ValidatedPrimitive{
                      .primitive_id = "fault-edge",
                      .capability_primitive_id = "forward",
                      .primitive_kind =
                          PrimitiveKind::kBodyTranslation,
                      .start_pose =
                          {
                              .position_m = {1.5, 2.5, 0.5},
                              .yaw_rad = 0.0,
                          },
                      .end_pose =
                          {
                              .position_m = {2.5, 2.5, 0.5},
                              .yaw_rad = 0.0,
                          },
                      .nominal_duration =
                          DurationNanoseconds{
                              std::chrono::seconds{2}},
                      .validation_ref =
                          legged.motion_primitives.front()
                              .sampled_body_sweep_ref,
                  },
              },
      }};
  auto config =
      fixture.request.algorithm_config->legged.time_scaling;
  config.maximum_adaptive_samples = 2U;

  const auto result =
      ParameterizeLeggedBodyPath(
          path,
          BodyFrameVelocityEnvelope{
              .forward_mps = legged.body_velocity_limits.forward_mps,
              .lateral_mps = legged.body_velocity_limits.lateral_mps,
              .vertical_mps = legged.body_velocity_limits.vertical_mps,
              .yaw_rate_radps =
                  legged.body_velocity_limits.yaw_rate_radps,
          },
          {
              .sampling = config,
              .maximum_linear_acceleration_mps2 =
                  legged.body_velocity_limits
                      .linear_acceleration_mps2,
              .maximum_yaw_acceleration_radps2 =
                  legged.body_velocity_limits
                      .yaw_acceleration_radps2,
          });

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).code,
            ErrorCode::kResourceLimit);
}

TEST(LeggedFaultMatrixTest,
     ContinuousValidationInconclusivePublishesNoSpline) {
  auto setup = MakeDiscreteSetup();
  const auto corridor_result =
      BuildLeggedCorridor(
          {
              .projection = setup.projection,
              .discrete_plan = setup.plan,
              .capability = setup.capability,
              .config =
                  setup.fixture.request.algorithm_config
                      ->legged.corridor,
          });
  ASSERT_TRUE(IsOk(corridor_result));
  FailingQpSolver unused_solver;

  const auto result =
      OptimizeLeggedBodySpline(
          {
              .discrete_plan = setup.plan,
              .corridor =
                  std::get<LeggedCorridor>(
                      corridor_result),
              .config =
                  {
                      .smoothing =
                          setup.fixture.request
                              .algorithm_config->legged
                              .smoothing,
                      .qp_settings = {},
                      .validation_sample_count = 1U,
                  },
              .preferred_body_height_m =
                  setup.capability.preferred_body_height_m,
              .committed_points = {},
          },
          unused_solver);

  EXPECT_FALSE(result.body_spline.has_value());
  EXPECT_EQ(result.termination,
            LeggedOptimizationTermination::kInvalidRequest);
}

}  // namespace
}  // namespace lunar::planning::v3
