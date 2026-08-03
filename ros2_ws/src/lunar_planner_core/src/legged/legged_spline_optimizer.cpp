#include "legged/legged_spline_optimizer.hpp"

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

namespace lunar::planning::legged {
namespace {

constexpr double kTolerance = 1.0e-9;
constexpr std::size_t kAxes = 3U;

struct Bounds2 final {
  double minimum_x{-std::numeric_limits<double>::infinity()};
  double maximum_x{std::numeric_limits<double>::infinity()};
  double minimum_y{-std::numeric_limits<double>::infinity()};
  double maximum_y{std::numeric_limits<double>::infinity()};
};

[[nodiscard]] LeggedOptimizationResult Result(
    std::vector<LeggedTransition> transitions, const bool optimized,
    const bool canceled, std::string reason_code) {
  return LeggedOptimizationResult{
      .transitions = std::move(transitions),
      .optimized = optimized,
      .canceled = canceled,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] std::optional<Bounds2> BoundsOf(
    const shared::ConvexCorridorCell& cell) noexcept {
  Bounds2 result;
  for (const shared::HalfPlane2& plane : cell.half_planes) {
    if (std::abs(plane.outward_unit_normal.x - 1.0) <= kTolerance &&
        std::abs(plane.outward_unit_normal.y) <= kTolerance) {
      result.maximum_x = std::min(result.maximum_x, plane.upper_offset_m);
    } else if (
        std::abs(plane.outward_unit_normal.x + 1.0) <= kTolerance &&
        std::abs(plane.outward_unit_normal.y) <= kTolerance) {
      result.minimum_x = std::max(result.minimum_x, -plane.upper_offset_m);
    } else if (
        std::abs(plane.outward_unit_normal.y - 1.0) <= kTolerance &&
        std::abs(plane.outward_unit_normal.x) <= kTolerance) {
      result.maximum_y = std::min(result.maximum_y, plane.upper_offset_m);
    } else if (
        std::abs(plane.outward_unit_normal.y + 1.0) <= kTolerance &&
        std::abs(plane.outward_unit_normal.x) <= kTolerance) {
      result.minimum_y = std::max(result.minimum_y, -plane.upper_offset_m);
    } else {
      return std::nullopt;
    }
  }
  return std::isfinite(result.minimum_x) &&
          std::isfinite(result.maximum_x) &&
          std::isfinite(result.minimum_y) &&
          std::isfinite(result.maximum_y) &&
          result.minimum_x <= result.maximum_x &&
          result.minimum_y <= result.maximum_y
      ? std::optional<Bounds2>{result}
      : std::nullopt;
}

[[nodiscard]] std::vector<double> ArcLengths(
    const std::vector<LeggedPose>& points) {
  std::vector<double> result(points.size(), 0.0);
  for (std::size_t index = 1U; index < points.size(); ++index) {
    result[index] = result[index - 1U] + std::hypot(
        points[index].position_m.x - points[index - 1U].position_m.x,
        points[index].position_m.y - points[index - 1U].position_m.y);
  }
  return result;
}

[[nodiscard]] const shared::ConvexCorridorCell* CorridorCellAt(
    const shared::CorridorResult& corridor, const double arc_length) {
  const auto found = std::ranges::find_if(
      corridor.cells, [&](const shared::ConvexCorridorCell& cell) {
        return arc_length + kTolerance >= cell.centerline_s_begin_m &&
            arc_length <= cell.centerline_s_end_m + kTolerance;
      });
  return found == corridor.cells.end() ? nullptr : &*found;
}

void AddSmoothness(
    shared::BoundedQpProblem& problem,
    const std::array<std::size_t, 3>& variables,
    const double base) {
  constexpr std::array<double, 3> kCoefficient{1.0, -2.0, 1.0};
  constexpr double kWeight = 4.0;
  for (std::size_t row = 0U; row < variables.size(); ++row) {
    problem.gradient[variables[row]] +=
        kWeight * base * kCoefficient[row];
    for (std::size_t column = 0U; column < variables.size(); ++column) {
      problem.hessian[
          variables[row] * problem.dimension + variables[column]] +=
          kWeight * kCoefficient[row] * kCoefficient[column];
    }
  }
}

[[nodiscard]] double Coordinate(
    const LeggedPose& pose, const std::size_t axis) noexcept {
  if (axis == 0U) {
    return pose.position_m.x;
  }
  return axis == 1U ? pose.position_m.y : pose.position_m.z;
}

}  // namespace

LeggedOptimizationResult OptimizeLeggedBodySpline(
    const std::vector<LeggedTransition>& discrete_transitions,
    const shared::CorridorResult& corridor,
    const OptimizationConfig& config,
    const std::stop_token stop_token) {
  if (stop_token.stop_requested()) {
    return Result({}, false, true, "REQUEST_CANCELED");
  }
  if (discrete_transitions.empty()) {
    return Result({}, false, false, "LEGGED_OPTIMIZATION_PATH_EMPTY");
  }
  if (corridor.status != shared::CorridorStatus::kCertified ||
      corridor.cells.empty() || discrete_transitions.size() < 2U) {
    return Result(
        discrete_transitions, false, false,
        "LEGGED_OPTIMIZATION_DISCRETE_FALLBACK");
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
        "LEGGED_OPTIMIZATION_CONFIG_INVALID");
  }

  std::vector<LeggedPose> points;
  points.reserve(discrete_transitions.size() + 1U);
  points.push_back(discrete_transitions.front().source_pose);
  for (const LeggedTransition& transition : discrete_transitions) {
    points.push_back(transition.target_pose);
  }
  const std::vector<double> arc_lengths = ArcLengths(points);
  const std::size_t dimension = points.size() * kAxes;
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
      const auto* cell = CorridorCellAt(corridor, arc_lengths[point]);
      const auto bounds = cell == nullptr ? std::nullopt : BoundsOf(*cell);
      if (!bounds.has_value()) {
        return Result(
            discrete_transitions, false, false,
            "LEGGED_OPTIMIZATION_CORRIDOR_INVALID");
      }
      const std::size_t base = point * kAxes;
      problem.lower_bounds[base] = std::max(
          problem.lower_bounds[base],
          bounds->minimum_x - points[point].position_m.x);
      problem.upper_bounds[base] = std::min(
          problem.upper_bounds[base],
          bounds->maximum_x - points[point].position_m.x);
      problem.lower_bounds[base + 1U] = std::max(
          problem.lower_bounds[base + 1U],
          bounds->minimum_y - points[point].position_m.y);
      problem.upper_bounds[base + 1U] = std::min(
          problem.upper_bounds[base + 1U],
          bounds->maximum_y - points[point].position_m.y);
      if (point > 0U) {
        const Interval& z = discrete_transitions[point - 1U].target_body_z_m;
        problem.lower_bounds[base + 2U] = std::max(
            problem.lower_bounds[base + 2U],
            z.lower - points[point].position_m.z);
        problem.upper_bounds[base + 2U] = std::min(
            problem.upper_bounds[base + 2U],
            z.upper - points[point].position_m.z);
      }
      for (std::size_t axis = 0U; axis < kAxes; ++axis) {
        problem.hessian[(base + axis) * dimension + base + axis] += 1.0;
      }
      if (point == 0U || point + 1U == points.size()) {
        for (std::size_t axis = 0U; axis < kAxes; ++axis) {
          problem.lower_bounds[base + axis] = 0.0;
          problem.upper_bounds[base + axis] = 0.0;
        }
      }
      if (std::ranges::any_of(
              std::views::iota(std::size_t{0}, kAxes),
              [&](const std::size_t axis) {
                return problem.lower_bounds[base + axis] >
                    problem.upper_bounds[base + axis];
              })) {
        return Result(
            discrete_transitions, false, false,
            "LEGGED_OPTIMIZATION_BOUNDS_EMPTY");
      }
    }
    for (std::size_t point = 1U; point + 1U < points.size(); ++point) {
      for (std::size_t axis = 0U; axis < kAxes; ++axis) {
        AddSmoothness(
            problem,
            {kAxes * (point - 1U) + axis,
             kAxes * point + axis,
             kAxes * (point + 1U) + axis},
            Coordinate(points[point - 1U], axis) -
                2.0 * Coordinate(points[point], axis) +
                Coordinate(points[point + 1U], axis));
      }
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
    bool moved = false;
    for (std::size_t point = 0U; point < points.size(); ++point) {
      points[point].position_m.x += solution.primal[kAxes * point];
      points[point].position_m.y += solution.primal[kAxes * point + 1U];
      points[point].position_m.z += solution.primal[kAxes * point + 2U];
      moved = moved ||
          std::abs(solution.primal[kAxes * point]) >
              config.constraint_tolerance ||
          std::abs(solution.primal[kAxes * point + 1U]) >
              config.constraint_tolerance ||
          std::abs(solution.primal[kAxes * point + 2U]) >
              config.constraint_tolerance;
    }
    std::vector<LeggedTransition> optimized = discrete_transitions;
    for (std::size_t index = 0U; index < optimized.size(); ++index) {
      optimized[index].source_pose = points[index];
      optimized[index].target_pose = points[index + 1U];
      optimized[index].path_length_m = std::hypot(
          std::hypot(
              points[index + 1U].position_m.x - points[index].position_m.x,
              points[index + 1U].position_m.y - points[index].position_m.y),
          points[index + 1U].position_m.z - points[index].position_m.z);
    }
    return Result(
        std::move(optimized), moved, false,
        moved ? "LEGGED_OPTIMIZATION_SOLVED" :
                "LEGGED_OPTIMIZATION_NOT_NEEDED");
  }
  return Result(
      discrete_transitions, false, false,
      "LEGGED_OPTIMIZATION_DISCRETE_FALLBACK");
}

}  // namespace lunar::planning::legged
