#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "integration/wheel/wheel_planner_test_fixture.hpp"
#include "lunar_path_planner/v3/wheel/wheel_planner.hpp"

namespace lunar::planning::v3 {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] SafeProjection Projection() {
  SafeProjection projection;
  projection.geometry = {
      .width = 6U,
      .height = 6U,
      .resolution_m = 1.0,
      .origin_m = {0.0, 0.0},
      .frame_id = "map",
  };
  const std::size_t count = projection.geometry.CellCount();
  projection.known_mask.assign(count, 1U);
  projection.hard_feasible_mask.assign(count, 1U);
  projection.esdf_clearance_m.assign(count, 10.0F);
  projection.conservative_speed_limit_mps.assign(count, 1.0F);
  return projection;
}

[[nodiscard]] WheelCollisionEnvelope Envelope() {
  return {
      .footprint =
          {
              .vertices_xy_m =
                  {{-0.9, -0.2}, {0.9, -0.2},
                   {0.9, 0.2}, {-0.9, 0.2}},
              .minimum_z_m = -0.1,
              .maximum_z_m = 0.5,
          },
  };
}

[[nodiscard]] ValidatedPrimitiveChain Chain() {
  return {
      .primitives =
          {
              {
                  .primitive_id = "drive-instance",
                  .capability_primitive_id = "forward",
                  .primitive_kind =
                      PrimitiveKind::kDriveForward,
                  .start_pose = {{1.5, 2.5, 0.0}, 0.0},
                  .end_pose = {{3.5, 2.5, 0.0}, 0.0},
                  .nominal_duration =
                      DurationNanoseconds{2s},
                  .validation_ref =
                      test::Ref("sweep", 'a'),
              },
          },
  };
}

[[nodiscard]] WheelLatticeEdge CorridorEdge() {
  return {
      .source =
          {1, 2, 0, WheelMotionMode::kForward},
      .target =
          {2, 2, 0, WheelMotionMode::kForward},
      .primitive_id = "forward",
      .capability_primitive_id = "forward",
      .primitive_kind = WheelPrimitiveKind::kDriveLine,
      .source_pose = {{1.5, 2.5, 0.0}, 0.0},
      .target_pose = {{2.5, 2.5, 0.0}, 0.0},
      .transition_time = DurationNanoseconds{1s},
      .validation_ref = test::Ref("sweep", 'b'),
  };
}

[[nodiscard]] WheelDiscreteSegment OptimizationSegment() {
  auto edge = CorridorEdge();
  edge.target = {4, 2, 0, WheelMotionMode::kForward};
  edge.target_pose = {{4.5, 2.5, 0.0}, 0.0};
  edge.transition_time = DurationNanoseconds{5s};
  return {
      .mode = WheelMotionMode::kForward,
      .edges = {edge},
      .expected_time = DurationNanoseconds{5s},
  };
}

[[nodiscard]] CorridorResult CertifiedCorridor() {
  return {
      .status = CorridorStatus::kCertified,
      .fallback = CorridorFallback::kNone,
      .cells =
          {
              {
                  .stable_cell_id = "cell",
                  .centerline_s_begin = 0.0,
                  .centerline_s_end = 3.0,
                  .half_planes =
                      {
                          {{1.0, 0.0}, 10.0},
                          {{-1.0, 0.0}, 10.0},
                          {{0.0, 1.0}, 10.0},
                          {{0.0, -1.0}, 10.0},
                      },
              },
          },
      .reason_code = "certified",
  };
}

[[nodiscard]] WheelSplineConfig SplineConfig() {
  return {
      .maximum_control_points = 16U,
      .maximum_scp_iterations = 2U,
      .maximum_qp_iterations = 10U,
      .path_deviation_weight = 1.0,
      .second_difference_weight = 0.1,
      .initial_trust_region_m = 1.0,
      .minimum_trust_region_m = 0.01,
      .constraint_tolerance = 1.0e-7,
      .absolute_qp_tolerance = 1.0e-7,
      .relative_qp_tolerance = 1.0e-7,
      .maximum_curvature_per_m = 1.0,
      .time_equivalence_tolerance =
          DurationNanoseconds{1s},
  };
}

class TerminatingSolver final : public BoundedQpSolver {
 public:
  explicit TerminatingSolver(QpTermination termination)
      : termination_(termination) {}

 private:
  [[nodiscard]] Result<QpSolution> DoSolve(
      const SparseQpProblem& problem,
      const BoundedQpSettings&) const override {
    return QpSolution{
        .termination = termination_,
        .primal =
            Eigen::VectorXd::Zero(problem.gradient.size()),
    };
  }

  QpTermination termination_;
};

TEST(WheelFaultMatrixTest, UnknownCellFailsClosed) {
  auto projection = Projection();
  projection.known_mask[2U * 6U + 3U] = 0U;
  WheelSweepValidator validator{
      projection, Envelope(),
      {
          .maximum_subdivisions = 8U,
          .maximum_footprint_cells_per_sample = 64U,
      }};

  const auto report =
      validator.ValidatePrimitiveChain(Chain());

  ASSERT_FALSE(report.ok());
  EXPECT_EQ(report.issues.front().reason_code,
            "UNKNOWN_OR_UNSAFE_CELL");
}

TEST(WheelFaultMatrixTest,
     NonCircularSpinCollisionFailsClosed) {
  auto projection = Projection();
  projection.hard_feasible_mask[3U * 6U + 3U] = 0U;
  WheelSweepValidator validator{
      projection, Envelope(),
      {
          .maximum_subdivisions = 8U,
          .maximum_footprint_cells_per_sample = 64U,
      }};
  const auto yaw = ParameterizeWheelSpin(
      0.0, std::numbers::pi / 2.0,
      {
          .maximum_yaw_rate_radps = 1.0,
          .maximum_yaw_acceleration_radps2 = 1.0,
      });
  ASSERT_TRUE(IsOk(yaw));

  const auto report = validator.ValidateSpin(
      {2.5, 2.5, 0.0},
      std::get<PiecewiseCubicScalarTrajectory>(yaw));

  ASSERT_FALSE(report.ok());
  EXPECT_EQ(report.issues.front().reason_code,
            "SWEPT_COLLISION");
}

TEST(WheelFaultMatrixTest, CorridorDisconnectedFallsBack) {
  auto projection = Projection();
  projection.esdf_clearance_m.assign(
      projection.geometry.CellCount(), 0.5F);
  const std::vector edges{CorridorEdge()};

  const auto result = BuildWheelCorridor(
      {
          .edges = edges,
          .projection = projection,
          .envelope =
              {
                  .footprint =
                      {
                          .vertices_xy_m =
                              {{-0.1, -0.1},
                               {0.1, -0.1},
                               {0.1, 0.1},
                               {-0.1, 0.1}},
                      },
              },
          .config =
              {
                  .maximum_regions = 8U,
                  .maximum_inflation_iterations = 128U,
                  .maximum_halfplanes_per_region = 16U,
                  .maximum_centerline_samples = 8U,
                  .sampling_spacing_m = 10.0,
                  .maximum_curvature_per_m = 2.0,
              },
      });

  EXPECT_EQ(result.status,
            CorridorStatus::kFallbackRequired);
  EXPECT_EQ(result.fallback,
            CorridorFallback::kUseDiscreteValidatedPrimitives);
}

TEST(WheelFaultMatrixTest, QpInfeasibleRejectsSpline) {
  const auto segment = OptimizationSegment();
  const auto corridor = CertifiedCorridor();
  const auto capability = std::get<WheelCapabilityView>(
      WheelCapabilityView::Create(test::Capability()));
  TerminatingSolver solver{QpTermination::kPrimalInfeasible};

  const auto result = OptimizeWheelSpline(
      {
          .discrete_segment = segment,
          .corridor = corridor,
          .capability = capability,
          .config = SplineConfig(),
          .committed_points = {},
      },
      solver);

  EXPECT_FALSE(result.spline.has_value());
  EXPECT_EQ(result.termination,
            WheelOptimizationTermination::kQpInfeasible);
}

TEST(WheelFaultMatrixTest, QpIterationLimitRejectsSpline) {
  const auto segment = OptimizationSegment();
  const auto corridor = CertifiedCorridor();
  const auto capability = std::get<WheelCapabilityView>(
      WheelCapabilityView::Create(test::Capability()));
  TerminatingSolver solver{QpTermination::kMaxIterations};

  const auto result = OptimizeWheelSpline(
      {
          .discrete_segment = segment,
          .corridor = corridor,
          .capability = capability,
          .config = SplineConfig(),
          .committed_points = {},
      },
      solver);

  EXPECT_FALSE(result.spline.has_value());
  EXPECT_EQ(result.termination,
            WheelOptimizationTermination::kQpIterationLimit);
}

TEST(WheelFaultMatrixTest, TimingSampleLimitFailsClosed) {
  const auto result = ParameterizeWheelDrive(
      Chain(), DriveDirection::kForward,
      {
          .maximum_forward_speed_mps = 1.0,
          .maximum_reverse_speed_mps = 0.8,
          .maximum_linear_acceleration_mps2 = 1.0,
          .maximum_braking_deceleration_mps2 = 1.0,
          .maximum_yaw_rate_radps = 1.0,
          .maximum_lateral_acceleration_mps2 = 1.0,
          .terrain_speed_limit_mps = 1.0,
          .clearance_speed_limit_mps = 1.0,
          .traction_speed_limit_mps = 1.0,
      },
      {
          .maximum_adaptive_samples = 2U,
          .minimum_parameter_step = 0.1,
          .curvature_refinement_threshold_per_m = 0.1,
          .maximum_forward_passes = 1U,
          .maximum_backward_passes = 1U,
      });

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).code,
            ErrorCode::kResourceLimit);
}

TEST(WheelFaultMatrixTest,
     ContinuousValidationInconclusiveFailsClosed) {
  const auto projection = Projection();
  WheelSweepValidator validator{
      projection, Envelope(),
      {
          .maximum_subdivisions = 0U,
          .maximum_footprint_cells_per_sample = 64U,
      }};

  const auto report =
      validator.ValidatePrimitiveChain(Chain());

  ASSERT_FALSE(report.ok());
  EXPECT_EQ(report.issues.front().reason_code,
            "CONTINUOUS_VALIDATION_INCONCLUSIVE");
}

TEST(WheelFaultMatrixTest,
     SearchResourceLimitWithIncumbentPublishesCompleteReference) {
  auto fixture = test::MakeFixture();
  auto config = *fixture.request.algorithm_config;
  config.ara_star.resource_caps
      .maximum_generated_candidates = 1U;
  fixture.request.algorithm_config =
      std::make_shared<const PlannerAlgorithmConfig>(
          std::move(config));
  WheelPlanner planner;

  const auto result =
      planner.Plan(fixture.request, fixture.terminal);

  ASSERT_TRUE(IsOk(result));
  const auto& reference = std::get<WheeledReference>(result);
  EXPECT_FALSE(reference.segments.empty());
  EXPECT_EQ(reference.reference_hash.size(), 64U);
}

TEST(WheelFaultMatrixTest,
     SearchResourceLimitWithoutIncumbentPublishesNothing) {
  auto fixture = test::MakeFixture();
  auto config = *fixture.request.algorithm_config;
  config.ara_star.resource_caps
      .maximum_generated_candidates = 0U;
  fixture.request.algorithm_config =
      std::make_shared<const PlannerAlgorithmConfig>(
          std::move(config));
  WheelPlanner planner;

  const auto result =
      planner.Plan(fixture.request, fixture.terminal);

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).code,
            ErrorCode::kResourceLimit);
}

}  // namespace
}  // namespace lunar::planning::v3
