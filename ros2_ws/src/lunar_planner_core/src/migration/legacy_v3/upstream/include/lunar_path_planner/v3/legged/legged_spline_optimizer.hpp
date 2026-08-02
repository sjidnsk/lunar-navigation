#pragma once

#include <cstddef>
#include <optional>
#include <span>

#include "lunar_path_planner/v3/legged/legged_corridor.hpp"
#include "lunar_path_planner/v3/optimization/bounded_qp_solver.hpp"

namespace lunar::planning::v3 {

struct LeggedFrozenControlPoint final {
  std::size_t control_point_index{};
  PoseXyzYaw pose;
};

struct LeggedSplineConfig final {
  SmoothingConfig smoothing;
  BoundedQpSettings qp_settings;
  std::size_t validation_sample_count{};
};

struct LeggedSplineRequest final {
  const LeggedDiscretePlan& discrete_plan;
  const LeggedCorridor& corridor;
  LeggedSplineConfig config;
  double preferred_body_height_m{};
  std::span<const LeggedFrozenControlPoint> committed_points;
};

enum class LeggedOptimizationTermination {
  kSolved,
  kInvalidRequest,
  kQpFailure,
  kContinuousValidationFailure,
  kTimeEquivalentViolation,
};

struct LeggedSplineResult final {
  std::optional<ClampedCubicBSplinePath> body_spline;
  DurationNanoseconds estimated_execution_time;
  LeggedOptimizationTermination termination{
      LeggedOptimizationTermination::kInvalidRequest};
};

[[nodiscard]] LeggedSplineResult OptimizeLeggedBodySpline(
    const LeggedSplineRequest& request,
    BoundedQpSolver& solver);

[[nodiscard]] PoseXyzYaw EvaluateLeggedSpline(
    const ClampedCubicBSplinePath& spline,
    double parameter) noexcept;

[[nodiscard]] double EvaluateLeggedSplineTangentYaw(
    const ClampedCubicBSplinePath& spline,
    double parameter) noexcept;

}  // namespace lunar::planning::v3
