#include "wheel/wheel_spline_optimizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <ranges>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

#include "shared/bounded_qp_solver.hpp"

namespace lunar::planning::wheel {
namespace {

constexpr double kComparisonTolerance = 1.0e-9;
constexpr double kDeviationWeight = 1.0;
constexpr double kSmoothnessWeight = 4.0;

struct AxisBounds final {
  double minimum_x{-std::numeric_limits<double>::infinity()};
  double maximum_x{std::numeric_limits<double>::infinity()};
  double minimum_y{-std::numeric_limits<double>::infinity()};
  double maximum_y{std::numeric_limits<double>::infinity()};
};

[[nodiscard]] WheelOptimizationResult Result(
    std::vector<WheelTransition> transitions, const bool optimized,
    const bool canceled, std::string reason_code) {
  return WheelOptimizationResult{
      .transitions = std::move(transitions),
      .optimized = optimized,
      .canceled = canceled,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] std::optional<AxisBounds> BoundsOf(
    const shared::ConvexCorridorCell& cell) noexcept {
  AxisBounds bounds;
  for (const shared::HalfPlane2& plane : cell.half_planes) {
    if (!std::isfinite(plane.outward_unit_normal.x) ||
        !std::isfinite(plane.outward_unit_normal.y) ||
        !std::isfinite(plane.upper_offset_m)) {
      return std::nullopt;
    }
    if (std::abs(plane.outward_unit_normal.x - 1.0) <=
            kComparisonTolerance &&
        std::abs(plane.outward_unit_normal.y) <= kComparisonTolerance) {
      bounds.maximum_x = std::min(bounds.maximum_x, plane.upper_offset_m);
    } else if (std::abs(plane.outward_unit_normal.x + 1.0) <=
                   kComparisonTolerance &&
               std::abs(plane.outward_unit_normal.y) <=
                   kComparisonTolerance) {
      bounds.minimum_x = std::max(bounds.minimum_x, -plane.upper_offset_m);
    } else if (std::abs(plane.outward_unit_normal.y - 1.0) <=
                   kComparisonTolerance &&
               std::abs(plane.outward_unit_normal.x) <=
                   kComparisonTolerance) {
      bounds.maximum_y = std::min(bounds.maximum_y, plane.upper_offset_m);
    } else if (std::abs(plane.outward_unit_normal.y + 1.0) <=
                   kComparisonTolerance &&
               std::abs(plane.outward_unit_normal.x) <=
                   kComparisonTolerance) {
      bounds.minimum_y = std::max(bounds.minimum_y, -plane.upper_offset_m);
    } else {
      return std::nullopt;
    }
  }
  if (!std::isfinite(bounds.minimum_x) ||
      !std::isfinite(bounds.maximum_x) ||
      !std::isfinite(bounds.minimum_y) ||
      !std::isfinite(bounds.maximum_y) ||
      bounds.minimum_x > bounds.maximum_x ||
      bounds.minimum_y > bounds.maximum_y) {
    return std::nullopt;
  }
  return bounds;
}

[[nodiscard]] bool IsSpinOrStop(const WheelTransition& transition) noexcept {
  return transition.primitive_kind == WheelPrimitiveKind::kSpinClockwise ||
         transition.primitive_kind ==
             WheelPrimitiveKind::kSpinCounterclockwise ||
         transition.primitive_kind == WheelPrimitiveKind::kStopAndSwitch;
}

void AddQuadratic(
    shared::BoundedQpProblem& problem,
    const std::array<std::size_t, 3>& indices,
    const std::array<double, 3>& coefficients,
    const double base, const double weight) {
  for (std::size_t row = 0U; row < indices.size(); ++row) {
    problem.gradient[indices[row]] +=
        weight * base * coefficients[row];
    for (std::size_t column = 0U; column < indices.size(); ++column) {
      problem.hessian[
          indices[row] * problem.dimension + indices[column]] +=
          weight * coefficients[row] * coefficients[column];
    }
  }
}

[[nodiscard]] std::vector<double> ArcLengths(
    const std::vector<Vec2>& points) {
  std::vector<double> lengths(points.size(), 0.0);
  for (std::size_t index = 1U; index < points.size(); ++index) {
    lengths[index] = lengths[index - 1U] + std::hypot(
        points[index].x - points[index - 1U].x,
        points[index].y - points[index - 1U].y);
  }
  return lengths;
}

[[nodiscard]] const shared::ConvexCorridorCell* CellAt(
    const shared::CorridorResult& corridor, const double arc_length_m) {
  const auto found = std::ranges::find_if(
      corridor.cells, [&](const shared::ConvexCorridorCell& cell) {
        return arc_length_m + kComparisonTolerance >=
                   cell.centerline_s_begin_m &&
               arc_length_m <=
                   cell.centerline_s_end_m + kComparisonTolerance;
      });
  if (found != corridor.cells.end()) {
    return &*found;
  }
  return corridor.cells.empty() ? nullptr : &corridor.cells.back();
}

}  // namespace

WheelOptimizationResult OptimizeWheelSpline(
    const std::vector<WheelTransition>& discrete_transitions,
    const shared::CorridorResult& corridor,
    const OptimizationConfig& config,
    const std::stop_token stop_token) {
  if (stop_token.stop_requested()) {
    return Result({}, false, true, "REQUEST_CANCELED");
  }
  if (discrete_transitions.empty()) {
    return Result({}, false, false, "WHEEL_OPTIMIZATION_PATH_EMPTY");
  }
  if (corridor.status != shared::CorridorStatus::kCertified ||
      corridor.cells.empty()) {
    return Result(
        discrete_transitions, false, false,
        "WHEEL_OPTIMIZATION_DISCRETE_FALLBACK");
  }
  if (config.maximum_iterations == 0U ||
      config.maximum_trust_region_reductions == 0U ||
      !std::isfinite(config.initial_trust_region_m) ||
      !std::isfinite(config.minimum_trust_region_m) ||
      !std::isfinite(config.constraint_tolerance) ||
      config.initial_trust_region_m <= 0.0 ||
      config.minimum_trust_region_m <= 0.0 ||
      config.minimum_trust_region_m > config.initial_trust_region_m ||
      config.constraint_tolerance <= 0.0) {
    return Result(
        discrete_transitions, false, false,
        "WHEEL_OPTIMIZATION_CONFIG_INVALID");
  }

  std::vector<Vec2> points;
  points.reserve(discrete_transitions.size() + 1U);
  points.push_back(Vec2{
      .x = discrete_transitions.front().source_pose.position_m.x,
      .y = discrete_transitions.front().source_pose.position_m.y,
  });
  for (const WheelTransition& transition : discrete_transitions) {
    points.push_back(Vec2{
        .x = transition.target_pose.position_m.x,
        .y = transition.target_pose.position_m.y,
    });
  }
  if (points.size() < 3U) {
    return Result(
        discrete_transitions, false, false,
        "WHEEL_OPTIMIZATION_NOT_NEEDED");
  }
  const std::vector<double> arc_lengths = ArcLengths(points);
  const std::size_t dimension = points.size() * 2U;
  double trust_region = config.initial_trust_region_m;

  for (std::size_t reduction = 0U;
       reduction < config.maximum_trust_region_reductions; ++reduction) {
    if (stop_token.stop_requested()) {
      return Result({}, false, true, "REQUEST_CANCELED");
    }
    shared::BoundedQpProblem problem{
        .dimension = dimension,
        .hessian = std::vector<double>(dimension * dimension, 0.0),
        .gradient = std::vector<double>(dimension, 0.0),
        .lower_bounds = std::vector<double>(dimension, -trust_region),
        .upper_bounds = std::vector<double>(dimension, trust_region),
    };
    for (std::size_t point = 0U; point < points.size(); ++point) {
      const shared::ConvexCorridorCell* cell =
          CellAt(corridor, arc_lengths[point]);
      const auto bounds = cell == nullptr ? std::nullopt : BoundsOf(*cell);
      if (!bounds.has_value()) {
        return Result(
            discrete_transitions, false, false,
            "WHEEL_OPTIMIZATION_CORRIDOR_INVALID");
      }
      problem.lower_bounds[2U * point] = std::max(
          problem.lower_bounds[2U * point],
          bounds->minimum_x - points[point].x);
      problem.upper_bounds[2U * point] = std::min(
          problem.upper_bounds[2U * point],
          bounds->maximum_x - points[point].x);
      problem.lower_bounds[2U * point + 1U] = std::max(
          problem.lower_bounds[2U * point + 1U],
          bounds->minimum_y - points[point].y);
      problem.upper_bounds[2U * point + 1U] = std::min(
          problem.upper_bounds[2U * point + 1U],
          bounds->maximum_y - points[point].y);
      const bool fixed = point == 0U || point + 1U == points.size() ||
          (point > 0U && IsSpinOrStop(discrete_transitions[point - 1U])) ||
          (point < discrete_transitions.size() &&
           IsSpinOrStop(discrete_transitions[point]));
      if (fixed) {
        problem.lower_bounds[2U * point] = 0.0;
        problem.upper_bounds[2U * point] = 0.0;
        problem.lower_bounds[2U * point + 1U] = 0.0;
        problem.upper_bounds[2U * point + 1U] = 0.0;
      }
      if (problem.lower_bounds[2U * point] >
              problem.upper_bounds[2U * point] ||
          problem.lower_bounds[2U * point + 1U] >
              problem.upper_bounds[2U * point + 1U]) {
        return Result(
            discrete_transitions, false, false,
            "WHEEL_OPTIMIZATION_CORRIDOR_EMPTY");
      }
      problem.hessian[(2U * point) * dimension + 2U * point] +=
          kDeviationWeight;
      problem.hessian[(2U * point + 1U) * dimension + 2U * point + 1U] +=
          kDeviationWeight;
    }
    for (std::size_t point = 1U; point + 1U < points.size(); ++point) {
      const std::array<double, 3> coefficients{1.0, -2.0, 1.0};
      AddQuadratic(
          problem,
          {2U * (point - 1U), 2U * point, 2U * (point + 1U)},
          coefficients,
          points[point - 1U].x - 2.0 * points[point].x +
              points[point + 1U].x,
          kSmoothnessWeight);
      AddQuadratic(
          problem,
          {2U * (point - 1U) + 1U, 2U * point + 1U,
           2U * (point + 1U) + 1U},
          coefficients,
          points[point - 1U].y - 2.0 * points[point].y +
              points[point + 1U].y,
          kSmoothnessWeight);
    }

    const shared::BoundedQpSolution solution = shared::SolveBoundedQp(
        problem,
        shared::BoundedQpSettings{
            .maximum_iterations = config.maximum_iterations,
            .absolute_tolerance = config.constraint_tolerance,
        },
        stop_token);
    if (solution.termination == shared::QpTermination::kCanceled) {
      return Result({}, false, true, "REQUEST_CANCELED");
    }
    if (solution.termination != shared::QpTermination::kSolved ||
        solution.primal.size() != dimension) {
      trust_region *= 0.5;
      if (trust_region < config.minimum_trust_region_m) {
        break;
      }
      continue;
    }

    std::vector<Vec2> optimized_points = points;
    bool moved = false;
    for (std::size_t point = 0U; point < points.size(); ++point) {
      optimized_points[point].x += solution.primal[2U * point];
      optimized_points[point].y += solution.primal[2U * point + 1U];
      moved = moved || std::abs(solution.primal[2U * point]) >
              config.constraint_tolerance ||
          std::abs(solution.primal[2U * point + 1U]) >
              config.constraint_tolerance;
    }
    std::vector<WheelTransition> optimized = discrete_transitions;
    for (std::size_t index = 0U; index < optimized.size(); ++index) {
      optimized[index].source_pose.position_m.x = optimized_points[index].x;
      optimized[index].source_pose.position_m.y = optimized_points[index].y;
      optimized[index].target_pose.position_m.x = optimized_points[index + 1U].x;
      optimized[index].target_pose.position_m.y = optimized_points[index + 1U].y;
      optimized[index].path_length_m = std::hypot(
          optimized[index].target_pose.position_m.x -
              optimized[index].source_pose.position_m.x,
          optimized[index].target_pose.position_m.y -
              optimized[index].source_pose.position_m.y);
      optimized[index].curvature_per_m =
          optimized[index].path_length_m > kComparisonTolerance &&
              !IsSpinOrStop(optimized[index])
          ? ShortestYawDelta(
                optimized[index].source_pose.yaw_rad,
                optimized[index].target_pose.yaw_rad) /
                optimized[index].path_length_m
          : 0.0;
    }
    return Result(
        std::move(optimized), moved, false,
        moved ? "WHEEL_OPTIMIZATION_SOLVED" :
                "WHEEL_OPTIMIZATION_NOT_NEEDED");
  }
  return Result(
      discrete_transitions, false, false,
      "WHEEL_OPTIMIZATION_DISCRETE_FALLBACK");
}

}  // namespace lunar::planning::wheel
