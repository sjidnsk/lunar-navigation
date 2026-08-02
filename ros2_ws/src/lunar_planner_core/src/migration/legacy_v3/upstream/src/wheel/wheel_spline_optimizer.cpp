#include "lunar_path_planner/v3/wheel/wheel_spline_optimizer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/SparseCore>

namespace lunar::planning::v3 {
namespace {

constexpr double kGeometryTolerance = 1.0e-9;
constexpr double kFiniteBound = 1.0e12;

[[nodiscard]] WheelSplineOptimizationResult Failure(
    WheelOptimizationTermination termination,
    std::size_t iterations = 0U) {
  return {
      .spline = std::nullopt,
      .termination = termination,
      .estimated_execution_time = {},
      .scp_iterations = iterations,
  };
}

[[nodiscard]] bool FinitePositive(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool FiniteNonnegative(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] bool FinitePose(const PoseXyzYaw& pose) noexcept {
  return std::isfinite(pose.position_m.x) &&
         std::isfinite(pose.position_m.y) &&
         std::isfinite(pose.position_m.z) &&
         std::isfinite(pose.yaw_rad);
}

[[nodiscard]] double Distance(
    const PoseXyzYaw& lhs,
    const PoseXyzYaw& rhs) noexcept {
  return std::hypot(lhs.position_m.x - rhs.position_m.x,
                    lhs.position_m.y - rhs.position_m.y);
}

[[nodiscard]] std::vector<PoseXyzYaw> InitialControlPoints(
    const WheelDiscreteSegment& segment,
    std::size_t maximum_control_points) {
  std::vector<PoseXyzYaw> waypoints;
  for (const WheelLatticeEdge& edge : segment.edges) {
    if (edge.primitive_kind == WheelPrimitiveKind::kModeSwitch) {
      continue;
    }
    if (waypoints.empty()) {
      waypoints.push_back(edge.source_pose);
    }
    if (waypoints.empty() ||
        Distance(waypoints.back(), edge.target_pose) >
            kGeometryTolerance) {
      waypoints.push_back(edge.target_pose);
    }
  }
  if (waypoints.size() < 2U) {
    return {};
  }
  if (waypoints.size() > maximum_control_points) {
    return {};
  }
  if (waypoints.size() >= 4U) {
    return waypoints;
  }

  const PoseXyzYaw start = waypoints.front();
  const PoseXyzYaw finish = waypoints.back();
  std::vector<PoseXyzYaw> controls;
  controls.reserve(4U);
  for (std::size_t index = 0U; index < 4U; ++index) {
    const double fraction = static_cast<double>(index) / 3.0;
    controls.push_back(
        {
            .position_m =
                {
                    start.position_m.x +
                        fraction * (finish.position_m.x -
                                    start.position_m.x),
                    start.position_m.y +
                        fraction * (finish.position_m.y -
                                    start.position_m.y),
                    start.position_m.z +
                        fraction * (finish.position_m.z -
                                    start.position_m.z),
                },
            .yaw_rad =
                start.yaw_rad +
                fraction * (finish.yaw_rad - start.yaw_rad),
        });
  }
  return controls;
}

[[nodiscard]] std::vector<double> ClampedKnots(
    std::size_t control_count) {
  constexpr std::size_t kDegree = 3U;
  std::vector<double> knots(control_count + kDegree + 1U, 0.0);
  for (std::size_t index = control_count;
       index < knots.size(); ++index) {
    knots[index] = 1.0;
  }
  if (control_count > 4U) {
    const double denominator =
        static_cast<double>(control_count - kDegree);
    for (std::size_t internal = 1U;
         internal < control_count - kDegree; ++internal) {
      knots[kDegree + internal] =
          static_cast<double>(internal) / denominator;
    }
  }
  return knots;
}

[[nodiscard]] std::optional<std::size_t> CorridorCellFor(
    const CorridorResult& corridor,
    double parameter) noexcept {
  if (corridor.cells.empty()) {
    return std::nullopt;
  }
  const double final_s =
      corridor.cells.back().centerline_s_end;
  const double target_s = parameter * final_s;
  for (std::size_t index = 0U; index < corridor.cells.size();
       ++index) {
    const auto& cell = corridor.cells[index];
    if (target_s + kGeometryTolerance >=
            cell.centerline_s_begin &&
        target_s <= cell.centerline_s_end +
                        kGeometryTolerance) {
      return index;
    }
  }
  return corridor.cells.size() - 1U;
}

struct QpRows final {
  std::vector<Eigen::Triplet<double, int>> entries;
  std::vector<double> lower;
  std::vector<double> upper;

  void Add(std::span<const std::pair<int, double>> coefficients,
           double lower_bound,
           double upper_bound) {
    const int row = static_cast<int>(lower.size());
    for (const auto& [column, value] : coefficients) {
      entries.emplace_back(row, column, value);
    }
    lower.push_back(lower_bound);
    upper.push_back(upper_bound);
  }
};

[[nodiscard]] SparseQpProblem BuildQp(
    std::span<const PoseXyzYaw> controls,
    const CorridorResult& corridor,
    const WheelSplineConfig& config,
    std::span<const WheelFrozenControlPoint> frozen) {
  const auto count = static_cast<int>(controls.size());
  const int variables = 2 * count;
  Eigen::MatrixXd dense =
      Eigen::MatrixXd::Zero(variables, variables);
  Eigen::VectorXd gradient =
      Eigen::VectorXd::Zero(variables);
  for (int point = 0; point < count; ++point) {
    for (int axis = 0; axis < 2; ++axis) {
      const int variable = 2 * point + axis;
      const double initial =
          axis == 0 ? controls[static_cast<std::size_t>(point)]
                          .position_m.x
                    : controls[static_cast<std::size_t>(point)]
                          .position_m.y;
      dense(variable, variable) +=
          2.0 * config.path_deviation_weight;
      gradient[variable] -=
          2.0 * config.path_deviation_weight * initial;
    }
  }
  for (int point = 1; point + 1 < count; ++point) {
    constexpr std::array<double, 3U> kDifference{
        1.0, -2.0, 1.0};
    for (int axis = 0; axis < 2; ++axis) {
      for (int first = 0; first < 3; ++first) {
        for (int second = 0; second < 3; ++second) {
          const int row_variable =
              2 * (point + first - 1) + axis;
          const int column_variable =
              2 * (point + second - 1) + axis;
          dense(row_variable, column_variable) +=
              2.0 * config.second_difference_weight *
              kDifference[static_cast<std::size_t>(first)] *
              kDifference[static_cast<std::size_t>(second)];
        }
      }
    }
  }
  std::vector<Eigen::Triplet<double, int>> hessian_entries;
  for (int row = 0; row < variables; ++row) {
    for (int column = row; column < variables; ++column) {
      if (dense(row, column) != 0.0) {
        hessian_entries.emplace_back(
            row, column, dense(row, column));
      }
    }
  }
  SparseQpMatrix hessian(variables, variables);
  hessian.setFromTriplets(hessian_entries.begin(),
                          hessian_entries.end());

  QpRows rows;
  for (int point = 0; point < count; ++point) {
    const double parameter =
        count == 1
            ? 0.0
            : static_cast<double>(point) /
                  static_cast<double>(count - 1);
    const auto cell_index =
        CorridorCellFor(corridor, parameter);
    if (cell_index.has_value()) {
      for (const HalfPlane2& plane :
           corridor.cells[*cell_index].half_planes) {
        const std::array coefficients{
            std::pair{2 * point,
                      plane.outward_unit_normal.x},
            std::pair{2 * point + 1,
                      plane.outward_unit_normal.y},
        };
        rows.Add(coefficients, -kFiniteBound,
                 plane.upper_offset_m);
      }
    }
    const std::array x_coefficient{
        std::pair{2 * point, 1.0}};
    const std::array y_coefficient{
        std::pair{2 * point + 1, 1.0}};
    rows.Add(
        x_coefficient,
        controls[static_cast<std::size_t>(point)]
                .position_m.x -
            config.initial_trust_region_m,
        controls[static_cast<std::size_t>(point)]
                .position_m.x +
            config.initial_trust_region_m);
    rows.Add(
        y_coefficient,
        controls[static_cast<std::size_t>(point)]
                .position_m.y -
            config.initial_trust_region_m,
        controls[static_cast<std::size_t>(point)]
                .position_m.y +
            config.initial_trust_region_m);
  }

  for (const int point : {0, count - 1}) {
    const auto& pose =
        controls[static_cast<std::size_t>(point)];
    const std::array x_coefficient{
        std::pair{2 * point, 1.0}};
    const std::array y_coefficient{
        std::pair{2 * point + 1, 1.0}};
    rows.Add(x_coefficient, pose.position_m.x,
             pose.position_m.x);
    rows.Add(y_coefficient, pose.position_m.y,
             pose.position_m.y);
  }
  for (const WheelFrozenControlPoint& point : frozen) {
    const int index = static_cast<int>(point.index);
    const std::array x_coefficient{
        std::pair{2 * index, 1.0}};
    const std::array y_coefficient{
        std::pair{2 * index + 1, 1.0}};
    rows.Add(x_coefficient, point.value.position_m.x,
             point.value.position_m.x);
    rows.Add(y_coefficient, point.value.position_m.y,
             point.value.position_m.y);
  }
  const double nominal_spacing =
      controls.size() > 1U
          ? Distance(controls.front(), controls.back()) /
                static_cast<double>(controls.size() - 1U)
          : 0.0;
  const double difference_limit =
      config.maximum_curvature_per_m *
      nominal_spacing * nominal_spacing;
  for (int point = 1; point + 1 < count; ++point) {
    for (int axis = 0; axis < 2; ++axis) {
      const std::array coefficients{
          std::pair{2 * (point - 1) + axis, 1.0},
          std::pair{2 * point + axis, -2.0},
          std::pair{2 * (point + 1) + axis, 1.0},
      };
      rows.Add(coefficients, -difference_limit,
               difference_limit);
    }
  }

  SparseQpMatrix constraints(
      static_cast<int>(rows.lower.size()), variables);
  constraints.setFromTriplets(rows.entries.begin(),
                              rows.entries.end());
  Eigen::VectorXd lower(
      static_cast<Eigen::Index>(rows.lower.size()));
  Eigen::VectorXd upper(
      static_cast<Eigen::Index>(rows.upper.size()));
  for (std::size_t index = 0U; index < rows.lower.size();
       ++index) {
    lower[static_cast<Eigen::Index>(index)] =
        rows.lower[index];
    upper[static_cast<Eigen::Index>(index)] =
        rows.upper[index];
  }
  return {
      .hessian_upper_triangle = std::move(hessian),
      .gradient = std::move(gradient),
      .constraints = std::move(constraints),
      .lower_bounds = std::move(lower),
      .upper_bounds = std::move(upper),
  };
}

[[nodiscard]] bool InsideCorridor(
    std::span<const PoseXyzYaw> controls,
    const CorridorResult& corridor,
    double tolerance) noexcept {
  for (std::size_t index = 0U; index < controls.size(); ++index) {
    const double parameter =
        controls.size() == 1U
            ? 0.0
            : static_cast<double>(index) /
                  static_cast<double>(controls.size() - 1U);
    const auto cell_index =
        CorridorCellFor(corridor, parameter);
    if (!cell_index.has_value()) {
      return false;
    }
    for (const HalfPlane2& plane :
         corridor.cells[*cell_index].half_planes) {
      const double value =
          plane.outward_unit_normal.x *
              controls[index].position_m.x +
          plane.outward_unit_normal.y *
              controls[index].position_m.y;
      if (value > plane.upper_offset_m + tolerance) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] double MaximumPolylineCurvature(
    std::span<const PoseXyzYaw> controls) noexcept {
  double maximum = 0.0;
  for (std::size_t index = 1U;
       index + 1U < controls.size(); ++index) {
    const auto& a = controls[index - 1U].position_m;
    const auto& b = controls[index].position_m;
    const auto& c = controls[index + 1U].position_m;
    const double ab = std::hypot(b.x - a.x, b.y - a.y);
    const double bc = std::hypot(c.x - b.x, c.y - b.y);
    const double ac = std::hypot(c.x - a.x, c.y - a.y);
    const double denominator = ab * bc * ac;
    if (denominator <= kGeometryTolerance) {
      continue;
    }
    const double twice_area =
        std::abs((b.x - a.x) * (c.y - a.y) -
                 (b.y - a.y) * (c.x - a.x));
    maximum = std::max(maximum,
                       2.0 * twice_area / denominator);
  }
  return maximum;
}

void AssignYaw(std::vector<PoseXyzYaw>& controls,
               WheelMotionMode mode) {
  std::optional<double> previous;
  for (std::size_t index = 0U; index < controls.size(); ++index) {
    const std::size_t next =
        std::min(index + 1U, controls.size() - 1U);
    const std::size_t prior = index == next ? index - 1U : index;
    const double dx =
        controls[next].position_m.x -
        controls[prior].position_m.x;
    const double dy =
        controls[next].position_m.y -
        controls[prior].position_m.y;
    double yaw = std::atan2(dy, dx);
    if (mode == WheelMotionMode::kReverse) {
      yaw += std::numbers::pi;
    }
    if (previous.has_value()) {
      yaw = *previous +
            std::remainder(yaw - *previous,
                           2.0 * std::numbers::pi);
    }
    controls[index].yaw_rad = yaw;
    previous = yaw;
  }
}

[[nodiscard]] DurationNanoseconds EstimateExecutionTime(
    std::span<const PoseXyzYaw> controls,
    const WheelDiscreteSegment& segment,
    const WheelCapabilityView& capability) noexcept {
  double length_m = 0.0;
  for (std::size_t index = 1U; index < controls.size(); ++index) {
    length_m += Distance(controls[index - 1U], controls[index]);
  }
  const double speed =
      segment.mode == WheelMotionMode::kReverse
          ? capability.hard_limits().maximum_reverse_speed_mps
          : capability.hard_limits().maximum_forward_speed_mps;
  double seconds = speed > 0.0
                       ? length_m / speed
                       : std::numeric_limits<double>::infinity();
  for (const WheelLatticeEdge& edge : segment.edges) {
    if (edge.primitive_kind == WheelPrimitiveKind::kModeSwitch) {
      seconds +=
          std::chrono::duration<double>(
              edge.transition_time.value)
              .count();
    }
  }
  if (!std::isfinite(seconds) || seconds < 0.0) {
    return DurationNanoseconds{
        std::chrono::nanoseconds{
            std::numeric_limits<std::int64_t>::max()}};
  }
  const long double nanoseconds =
      static_cast<long double>(seconds) * 1.0e9L;
  if (nanoseconds >= static_cast<long double>(
                         std::numeric_limits<std::int64_t>::max())) {
    return DurationNanoseconds{
        std::chrono::nanoseconds{
            std::numeric_limits<std::int64_t>::max()}};
  }
  return DurationNanoseconds{
      std::chrono::nanoseconds{
          static_cast<std::int64_t>(std::ceil(nanoseconds))}};
}

[[nodiscard]] bool TimeWithinTolerance(
    DurationNanoseconds estimate,
    DurationNanoseconds baseline,
    DurationNanoseconds tolerance) noexcept {
  if (estimate.value.count() < 0 ||
      baseline.value.count() < 0 ||
      tolerance.value.count() < 0 ||
      baseline.value.count() >
          std::numeric_limits<std::int64_t>::max() -
              tolerance.value.count()) {
    return false;
  }
  return estimate.value <= baseline.value + tolerance.value;
}

}  // namespace

WheelSplineOptimizationResult OptimizeWheelSpline(
    const WheelSplineOptimizationRequest& request,
    BoundedQpSolver& solver) {
  const auto& config = request.config;
  if (request.corridor.status != CorridorStatus::kCertified ||
      request.corridor.cells.empty() ||
      request.discrete_segment.edges.empty() ||
      (request.discrete_segment.mode !=
           WheelMotionMode::kForward &&
       request.discrete_segment.mode !=
           WheelMotionMode::kReverse) ||
      config.maximum_control_points < 4U ||
      config.maximum_scp_iterations == 0U ||
      config.maximum_qp_iterations == 0U ||
      !FinitePositive(config.path_deviation_weight) ||
      !FiniteNonnegative(config.second_difference_weight) ||
      !FinitePositive(config.initial_trust_region_m) ||
      !FinitePositive(config.minimum_trust_region_m) ||
      config.minimum_trust_region_m >
          config.initial_trust_region_m ||
      !FiniteNonnegative(config.constraint_tolerance) ||
      !FinitePositive(config.absolute_qp_tolerance) ||
      !FinitePositive(config.relative_qp_tolerance) ||
      !FinitePositive(config.maximum_curvature_per_m) ||
      config.time_equivalence_tolerance.value.count() < 0) {
    return Failure(WheelOptimizationTermination::kInvalidRequest);
  }
  auto controls = InitialControlPoints(
      request.discrete_segment, config.maximum_control_points);
  if (controls.size() < 4U ||
      !std::ranges::all_of(controls, FinitePose)) {
    return Failure(WheelOptimizationTermination::kInvalidRequest);
  }
  for (const WheelFrozenControlPoint& point :
       request.committed_points) {
    if (point.index >= controls.size() ||
        !FinitePose(point.value)) {
      return Failure(WheelOptimizationTermination::kInvalidRequest);
    }
    controls[point.index] = point.value;
  }
  const PoseXyzYaw fixed_start = controls.front();
  const PoseXyzYaw fixed_finish = controls.back();

  std::size_t iterations = 0U;
  for (; iterations < config.maximum_scp_iterations;
       ++iterations) {
    auto problem = BuildQp(
        controls, request.corridor, config,
        request.committed_points);
    const auto solution_result = solver.Solve(
        problem,
        {
            .max_iterations = config.maximum_qp_iterations,
            .absolute_tolerance = config.absolute_qp_tolerance,
            .relative_tolerance = config.relative_qp_tolerance,
            .polish = false,
        });
    if (!IsOk(solution_result)) {
      return Failure(WheelOptimizationTermination::kNumericalFailure,
                     iterations + 1U);
    }
    const auto& solution = std::get<QpSolution>(solution_result);
    if (solution.termination == QpTermination::kPrimalInfeasible ||
        solution.termination == QpTermination::kDualInfeasible) {
      return Failure(WheelOptimizationTermination::kQpInfeasible,
                     iterations + 1U);
    }
    if (solution.termination == QpTermination::kMaxIterations) {
      return Failure(
          WheelOptimizationTermination::kQpIterationLimit,
          iterations + 1U);
    }
    if (solution.termination != QpTermination::kSolved ||
        solution.primal.size() !=
            static_cast<Eigen::Index>(2U * controls.size()) ||
        !solution.primal.allFinite()) {
      return Failure(WheelOptimizationTermination::kNumericalFailure,
                     iterations + 1U);
    }
    for (std::size_t point = 0U; point < controls.size();
         ++point) {
      controls[point].position_m.x =
          solution.primal[
              static_cast<Eigen::Index>(2U * point)];
      controls[point].position_m.y =
          solution.primal[
              static_cast<Eigen::Index>(2U * point + 1U)];
    }
    controls.front() = fixed_start;
    controls.back() = fixed_finish;
    for (const WheelFrozenControlPoint& point :
         request.committed_points) {
      controls[point.index] = point.value;
    }
  }
  AssignYaw(controls, request.discrete_segment.mode);
  const auto estimated_time = EstimateExecutionTime(
      controls, request.discrete_segment, request.capability);
  if (!TimeWithinTolerance(
          estimated_time,
          request.discrete_segment.expected_time,
          config.time_equivalence_tolerance)) {
    return {
        .spline = std::nullopt,
        .termination =
            WheelOptimizationTermination::kTimeToleranceExceeded,
        .estimated_execution_time = estimated_time,
        .scp_iterations = iterations,
    };
  }
  if (!InsideCorridor(controls, request.corridor,
                      config.constraint_tolerance) ||
      MaximumPolylineCurvature(controls) >
          std::min(
              config.maximum_curvature_per_m,
              request.capability.hard_limits()
                  .maximum_drive_curvature_per_m) +
              config.constraint_tolerance) {
    return {
        .spline = std::nullopt,
        .termination =
            WheelOptimizationTermination::kConstraintViolation,
        .estimated_execution_time = estimated_time,
        .scp_iterations = iterations,
    };
  }

  return {
      .spline =
          ClampedCubicBSplinePath{
              .knots = ClampedKnots(controls.size()),
              .control_points = std::move(controls),
          },
      .termination = WheelOptimizationTermination::kConverged,
      .estimated_execution_time = estimated_time,
      .scp_iterations = iterations,
  };
}

}  // namespace lunar::planning::v3
