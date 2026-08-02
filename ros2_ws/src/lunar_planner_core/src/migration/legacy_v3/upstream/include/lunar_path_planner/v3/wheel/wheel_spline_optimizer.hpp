#pragma once

#include <cstddef>
#include <optional>
#include <span>

#include "lunar_path_planner/v3/optimization/bounded_qp_solver.hpp"
#include "lunar_path_planner/v3/wheel/wheel_corridor.hpp"
#include "lunar_path_planner/v3/wheel/wheel_search.hpp"

namespace lunar::planning::v3 {

struct WheelFrozenControlPoint final {
  std::size_t index{};
  PoseXyzYaw value;
};

struct WheelSplineConfig final {
  std::size_t maximum_control_points{};
  std::size_t maximum_scp_iterations{};
  std::size_t maximum_qp_iterations{};
  double path_deviation_weight{};
  double second_difference_weight{};
  double initial_trust_region_m{};
  double minimum_trust_region_m{};
  double constraint_tolerance{};
  double absolute_qp_tolerance{};
  double relative_qp_tolerance{};
  double maximum_curvature_per_m{};
  DurationNanoseconds time_equivalence_tolerance;
};

enum class WheelOptimizationTermination {
  kConverged,
  kInvalidRequest,
  kQpInfeasible,
  kQpIterationLimit,
  kNumericalFailure,
  kConstraintViolation,
  kTimeToleranceExceeded,
};

struct WheelSplineOptimizationRequest final {
  const WheelDiscreteSegment& discrete_segment;
  const CorridorResult& corridor;
  const WheelCapabilityView& capability;
  WheelSplineConfig config;
  std::span<const WheelFrozenControlPoint> committed_points;
};

struct WheelSplineOptimizationResult final {
  std::optional<ClampedCubicBSplinePath> spline;
  WheelOptimizationTermination termination{
      WheelOptimizationTermination::kInvalidRequest};
  DurationNanoseconds estimated_execution_time;
  std::size_t scp_iterations{};
};

[[nodiscard]] WheelSplineOptimizationResult OptimizeWheelSpline(
    const WheelSplineOptimizationRequest& request,
    BoundedQpSolver& solver);

}  // namespace lunar::planning::v3
