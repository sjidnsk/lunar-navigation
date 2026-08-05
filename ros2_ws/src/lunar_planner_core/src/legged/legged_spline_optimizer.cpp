#include "legged/legged_spline_optimizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
constexpr std::size_t kHardMaximumControlPoints = 64U;
constexpr std::size_t kHardMaximumSamples = 512U;
constexpr std::size_t kHardMaximumIterations = 128U;
constexpr std::size_t kHardMaximumTrustReductions = 8U;
constexpr std::size_t kTimingSamplesPerTransition = 8U;
constexpr std::size_t kPreferredCurveSubdivisions = 4U;

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

[[nodiscard]] bool CorridorContains(
    const shared::CorridorResult& corridor, const Vec2 point) noexcept {
  return std::ranges::any_of(
      corridor.cells, [&](const shared::ConvexCorridorCell& cell) {
        return std::ranges::all_of(
            cell.half_planes, [&](const shared::HalfPlane2& plane) {
              return std::isfinite(plane.outward_unit_normal.x) &&
                  std::isfinite(plane.outward_unit_normal.y) &&
                  std::isfinite(plane.upper_offset_m) &&
                  plane.outward_unit_normal.x * point.x +
                          plane.outward_unit_normal.y * point.y <=
                      plane.upper_offset_m + kTolerance;
            });
      });
}

[[nodiscard]] Vec3 PositionTangent(
    const std::vector<LeggedPose>& points,
    const std::size_t point) noexcept {
  if (point == 0U) {
    return Vec3{
        .x = points[1U].position_m.x - points[0U].position_m.x,
        .y = points[1U].position_m.y - points[0U].position_m.y,
        .z = points[1U].position_m.z - points[0U].position_m.z,
    };
  }
  if (point + 1U == points.size()) {
    return Vec3{
        .x = points[point].position_m.x -
            points[point - 1U].position_m.x,
        .y = points[point].position_m.y -
            points[point - 1U].position_m.y,
        .z = points[point].position_m.z -
            points[point - 1U].position_m.z,
    };
  }
  return Vec3{
      .x = 0.5 * (points[point + 1U].position_m.x -
                  points[point - 1U].position_m.x),
      .y = 0.5 * (points[point + 1U].position_m.y -
                  points[point - 1U].position_m.y),
      .z = 0.5 * (points[point + 1U].position_m.z -
                  points[point - 1U].position_m.z),
  };
}

[[nodiscard]] LeggedPose HermiteSample(
    const std::vector<LeggedPose>& points,
    const std::size_t segment,
    const double parameter) noexcept {
  const LeggedPose& source = points[segment];
  const LeggedPose& target = points[segment + 1U];
  const Vec3 source_tangent = PositionTangent(points, segment);
  const Vec3 target_tangent = PositionTangent(points, segment + 1U);
  const double t2 = parameter * parameter;
  const double t3 = t2 * parameter;
  const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
  const double h10 = t3 - 2.0 * t2 + parameter;
  const double h01 = -2.0 * t3 + 3.0 * t2;
  const double h11 = t3 - t2;
  const double yaw_progress = t2 * (3.0 - 2.0 * parameter);
  LeggedPose sample{
      .position_m = Vec3{
          .x = h00 * source.position_m.x + h10 * source_tangent.x +
              h01 * target.position_m.x + h11 * target_tangent.x,
          .y = h00 * source.position_m.y + h10 * source_tangent.y +
              h01 * target.position_m.y + h11 * target_tangent.y,
          .z = h00 * source.position_m.z + h10 * source_tangent.z +
              h01 * target.position_m.z + h11 * target_tangent.z,
      },
      .yaw_rad = NormalizeYaw(
          source.yaw_rad + yaw_progress *
              ShortestYawDelta(source.yaw_rad, target.yaw_rad)),
  };
  if (parameter <= kTolerance) {
    return source;
  }
  if (parameter >= 1.0 - kTolerance) {
    return target;
  }
  return sample;
}

[[nodiscard]] LeggedOptimizationResult BuildCubicTransitions(
    const std::vector<LeggedTransition>& controls,
    const std::vector<LeggedPose>& points,
    const shared::CorridorResult& corridor,
    const OptimizationConfig& config) {
  if (points.size() != controls.size() + 1U ||
      points.size() > config.maximum_smoothing_control_points) {
    return Result(controls, false, false,
                  "LEGGED_OPTIMIZATION_CONTROL_POINT_LIMIT");
  }
  const std::size_t maximum_chords =
      (config.maximum_smoothing_samples - 1U) /
      kTimingSamplesPerTransition;
  if (maximum_chords < controls.size()) {
    return Result(controls, false, false,
                  "LEGGED_OPTIMIZATION_SAMPLE_LIMIT");
  }
  std::vector<std::size_t> subdivisions(controls.size(), 1U);
  std::size_t chord_count = controls.size();
  for (std::size_t level = 2U; level <= kPreferredCurveSubdivisions;
       ++level) {
    for (std::size_t index = 0U;
         index < controls.size() && chord_count < maximum_chords; ++index) {
      subdivisions[index] = level;
      ++chord_count;
    }
  }

  std::vector<LeggedTransition> result;
  result.reserve(chord_count);
  for (std::size_t index = 0U; index < controls.size(); ++index) {
    const LeggedTransition& control = controls[index];
    const std::size_t count = subdivisions[index];
    LeggedPose source = HermiteSample(points, index, 0.0);
    if (!CorridorContains(
            corridor, Vec2{source.position_m.x, source.position_m.y})) {
      return Result(controls, false, false,
                    "LEGGED_OPTIMIZATION_CURVE_OUTSIDE_CORRIDOR");
    }
    for (std::size_t sample = 1U; sample <= count; ++sample) {
      const double parameter =
          static_cast<double>(sample) / static_cast<double>(count);
      const LeggedPose target = HermiteSample(points, index, parameter);
      if (!CorridorContains(
              corridor, Vec2{target.position_m.x, target.position_m.y})) {
        return Result(controls, false, false,
                      "LEGGED_OPTIMIZATION_CURVE_OUTSIDE_CORRIDOR");
      }
      const std::int64_t duration_begin =
          control.nominal_duration.count() *
          static_cast<std::int64_t>(sample - 1U) /
          static_cast<std::int64_t>(count);
      const std::int64_t duration_end =
          control.nominal_duration.count() *
          static_cast<std::int64_t>(sample) /
          static_cast<std::int64_t>(count);
      result.push_back(LeggedTransition{
          .source_pose = source,
          .target_pose = target,
          .target_body_z_m = control.target_body_z_m,
          .primitive_index = control.primitive_index,
          .primitive_kind = control.primitive_kind,
          .nominal_duration =
              std::chrono::nanoseconds{duration_end - duration_begin},
          .path_length_m = std::hypot(
              std::hypot(target.position_m.x - source.position_m.x,
                         target.position_m.y - source.position_m.y),
              target.position_m.z - source.position_m.z),
          .stable_index = control.stable_index,
      });
      source = target;
    }
  }
  return Result(std::move(result), true, false,
                "LEGGED_OPTIMIZATION_SOLVED");
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
      corridor.cells.empty()) {
    return Result(
        discrete_transitions, false, false,
        "LEGGED_OPTIMIZATION_DISCRETE_FALLBACK");
  }
  if (config.maximum_smoothing_control_points < 2U ||
      config.maximum_smoothing_control_points > kHardMaximumControlPoints ||
      config.maximum_smoothing_samples <= kTimingSamplesPerTransition ||
      config.maximum_smoothing_samples > kHardMaximumSamples ||
      config.maximum_iterations == 0U ||
      config.maximum_iterations > kHardMaximumIterations ||
      config.maximum_trust_region_reductions == 0U ||
      config.maximum_trust_region_reductions >
          kHardMaximumTrustReductions ||
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
  if (points.size() > config.maximum_smoothing_control_points) {
    return Result(
        discrete_transitions, false, false,
        "LEGGED_OPTIMIZATION_CONTROL_POINT_LIMIT");
  }
  if (points.size() < 3U) {
    return BuildCubicTransitions(
        discrete_transitions, points, corridor, config);
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
    static_cast<void>(moved);
    return BuildCubicTransitions(optimized, points, corridor, config);
  }
  return Result(
      discrete_transitions, false, false,
      "LEGGED_OPTIMIZATION_DISCRETE_FALLBACK");
}

}  // namespace lunar::planning::legged
