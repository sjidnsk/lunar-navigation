#include "wheel/wheel_spline_optimizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
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
constexpr std::size_t kHardMaximumControlPoints = 64U;
constexpr std::size_t kHardMaximumSamples = 512U;
constexpr std::size_t kHardMaximumIterations = 128U;
constexpr std::size_t kHardMaximumTrustReductions = 8U;
constexpr std::size_t kTimingSamplesPerTransition = 8U;
constexpr std::size_t kPreferredCurveSubdivisions = 4U;

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

[[nodiscard]] bool CorridorContains(
    const shared::CorridorResult& corridor, const Vec2 point) noexcept {
  return std::ranges::any_of(
      corridor.cells, [&](const shared::ConvexCorridorCell& cell) {
        return std::ranges::all_of(
            cell.half_planes, [&](const shared::HalfPlane2& plane) {
              return plane.outward_unit_normal.x * point.x +
                         plane.outward_unit_normal.y * point.y <=
                     plane.upper_offset_m + kComparisonTolerance;
            });
      });
}

struct CubicSample final {
  Vec3 position_m;
  double yaw_rad{};
};

[[nodiscard]] CubicSample HermiteSample(
    const WheelTransition& transition, const double parameter) noexcept {
  const double t2 = parameter * parameter;
  const double t3 = t2 * parameter;
  const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
  const double h10 = t3 - 2.0 * t2 + parameter;
  const double h01 = -2.0 * t3 + 3.0 * t2;
  const double h11 = t3 - t2;
  const double dh00 = 6.0 * t2 - 6.0 * parameter;
  const double dh10 = 3.0 * t2 - 4.0 * parameter + 1.0;
  const double dh01 = -6.0 * t2 + 6.0 * parameter;
  const double dh11 = 3.0 * t2 - 2.0 * parameter;
  const double chord = std::hypot(
      transition.target_pose.position_m.x -
          transition.source_pose.position_m.x,
      transition.target_pose.position_m.y -
          transition.source_pose.position_m.y);
  const double direction = transition.reverse ? -1.0 : 1.0;
  const Vec2 source_tangent{
      .x = direction * chord * std::cos(transition.source_pose.yaw_rad),
      .y = direction * chord * std::sin(transition.source_pose.yaw_rad),
  };
  const Vec2 target_tangent{
      .x = direction * chord * std::cos(transition.target_pose.yaw_rad),
      .y = direction * chord * std::sin(transition.target_pose.yaw_rad),
  };
  const Vec2 derivative{
      .x = dh00 * transition.source_pose.position_m.x +
           dh10 * source_tangent.x +
           dh01 * transition.target_pose.position_m.x +
           dh11 * target_tangent.x,
      .y = dh00 * transition.source_pose.position_m.y +
           dh10 * source_tangent.y +
           dh01 * transition.target_pose.position_m.y +
           dh11 * target_tangent.y,
  };
  double yaw = transition.source_pose.yaw_rad +
               parameter * ShortestYawDelta(
                               transition.source_pose.yaw_rad,
                               transition.target_pose.yaw_rad);
  if (std::hypot(derivative.x, derivative.y) > kComparisonTolerance) {
    yaw = std::atan2(derivative.y, derivative.x) +
          (transition.reverse ? std::numbers::pi : 0.0);
  }
  if (parameter <= kComparisonTolerance) {
    yaw = transition.source_pose.yaw_rad;
  } else if (parameter >= 1.0 - kComparisonTolerance) {
    yaw = transition.target_pose.yaw_rad;
  }
  return CubicSample{
      .position_m =
          Vec3{
              .x = h00 * transition.source_pose.position_m.x +
                   h10 * source_tangent.x +
                   h01 * transition.target_pose.position_m.x +
                   h11 * target_tangent.x,
              .y = h00 * transition.source_pose.position_m.y +
                   h10 * source_tangent.y +
                   h01 * transition.target_pose.position_m.y +
                   h11 * target_tangent.y,
              .z = transition.source_pose.position_m.z +
                   parameter * (transition.target_pose.position_m.z -
                                transition.source_pose.position_m.z),
          },
      .yaw_rad = NormalizeYaw(yaw),
  };
}

[[nodiscard]] WheelOptimizationResult BuildCubicTransitions(
    const std::vector<WheelTransition>& controls,
    const shared::CorridorResult& corridor,
    const OptimizationConfig& config) {
  if (controls.size() + 1U > config.maximum_smoothing_control_points) {
    return Result(controls, false, false,
                  "WHEEL_OPTIMIZATION_CONTROL_POINT_LIMIT");
  }
  const std::size_t maximum_chords =
      (config.maximum_smoothing_samples - 1U) /
      kTimingSamplesPerTransition;
  if (maximum_chords < controls.size()) {
    return Result(controls, false, false,
                  "WHEEL_OPTIMIZATION_SAMPLE_LIMIT");
  }
  std::vector<std::size_t> subdivisions(controls.size(), 1U);
  std::size_t chord_count = controls.size();
  for (std::size_t level = 2U; level <= kPreferredCurveSubdivisions;
       ++level) {
    for (std::size_t index = 0U;
         index < controls.size() && chord_count < maximum_chords; ++index) {
      if (IsSpinOrStop(controls[index]) ||
          controls[index].path_length_m <= kComparisonTolerance) {
        continue;
      }
      subdivisions[index] = level;
      ++chord_count;
    }
  }

  std::vector<WheelTransition> result;
  result.reserve(chord_count);
  for (std::size_t index = 0U; index < controls.size(); ++index) {
    const WheelTransition& control = controls[index];
    const std::size_t count = subdivisions[index];
    if (count == 1U) {
      result.push_back(control);
      continue;
    }
    CubicSample source = HermiteSample(control, 0.0);
    if (!CorridorContains(corridor,
                          Vec2{source.position_m.x, source.position_m.y})) {
      return Result(controls, false, false,
                    "WHEEL_OPTIMIZATION_CURVE_OUTSIDE_CORRIDOR");
    }
    for (std::size_t sample = 1U; sample <= count; ++sample) {
      const double parameter =
          static_cast<double>(sample) / static_cast<double>(count);
      const CubicSample target = HermiteSample(control, parameter);
      if (!CorridorContains(corridor,
                            Vec2{target.position_m.x, target.position_m.y})) {
        return Result(controls, false, false,
                      "WHEEL_OPTIMIZATION_CURVE_OUTSIDE_CORRIDOR");
      }
      const double path_length = std::hypot(
          target.position_m.x - source.position_m.x,
          target.position_m.y - source.position_m.y);
      result.push_back(WheelTransition{
          .source_pose = WheelPose{.position_m = source.position_m,
                                   .yaw_rad = source.yaw_rad},
          .target_pose = WheelPose{.position_m = target.position_m,
                                   .yaw_rad = target.yaw_rad},
          .curvature_per_m =
              path_length > kComparisonTolerance
                  ? ShortestYawDelta(source.yaw_rad, target.yaw_rad) /
                        path_length
                  : 0.0,
          .primitive_index = control.primitive_index,
          .primitive_kind = control.primitive_kind,
          .source_mode = control.source_mode,
          .target_mode = control.target_mode,
          .path_length_m = path_length,
          .surface_slope_rad = control.surface_slope_rad,
          .roughness_m = control.roughness_m,
          .reverse = control.reverse,
          .stable_index = control.stable_index,
      });
      source = target;
    }
  }
  return Result(std::move(result), true, false,
                "WHEEL_OPTIMIZATION_SOLVED");
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
  if (config.maximum_smoothing_control_points < 2U ||
      config.maximum_smoothing_control_points > kHardMaximumControlPoints ||
      config.maximum_smoothing_samples <= kTimingSamplesPerTransition ||
      config.maximum_smoothing_samples > kHardMaximumSamples ||
      config.maximum_iterations == 0U ||
      config.maximum_iterations > kHardMaximumIterations ||
      config.maximum_trust_region_reductions == 0U ||
      config.maximum_trust_region_reductions > kHardMaximumTrustReductions ||
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
  if (corridor.status != shared::CorridorStatus::kCertified ||
      corridor.cells.empty()) {
    return Result(
        discrete_transitions, false, false,
        "WHEEL_OPTIMIZATION_DISCRETE_FALLBACK");
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
  if (points.size() > config.maximum_smoothing_control_points) {
    return Result(
        discrete_transitions, false, false,
        "WHEEL_OPTIMIZATION_CONTROL_POINT_LIMIT");
  }
  if (points.size() < 3U) {
    return BuildCubicTransitions(discrete_transitions, corridor, config);
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
    static_cast<void>(moved);
    return BuildCubicTransitions(optimized, corridor, config);
  }
  return Result(
      discrete_transitions, false, false,
      "WHEEL_OPTIMIZATION_DISCRETE_FALLBACK");
}

}  // namespace lunar::planning::wheel
