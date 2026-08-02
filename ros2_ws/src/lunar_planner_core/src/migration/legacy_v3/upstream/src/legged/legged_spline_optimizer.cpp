#include "lunar_path_planner/v3/legged/legged_spline_optimizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <set>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace lunar::planning::v3 {
namespace {

constexpr std::size_t kDimension = 4U;
constexpr std::size_t kMinimumControlPoints = 4U;
constexpr double kFiniteInfinity = 1.0e12;

[[nodiscard]] bool IsFinite(const PoseXyzYaw& pose) noexcept {
  return std::isfinite(pose.position_m.x) &&
         std::isfinite(pose.position_m.y) &&
         std::isfinite(pose.position_m.z) &&
         std::isfinite(pose.yaw_rad);
}

[[nodiscard]] double Component(const PoseXyzYaw& pose,
                               std::size_t component) noexcept {
  switch (component) {
    case 0U:
      return pose.position_m.x;
    case 1U:
      return pose.position_m.y;
    case 2U:
      return pose.position_m.z;
    default:
      return pose.yaw_rad;
  }
}

void SetComponent(PoseXyzYaw& pose, std::size_t component,
                  double value) noexcept {
  switch (component) {
    case 0U:
      pose.position_m.x = value;
      break;
    case 1U:
      pose.position_m.y = value;
      break;
    case 2U:
      pose.position_m.z = value;
      break;
    default:
      pose.yaw_rad = value;
      break;
  }
}

[[nodiscard]] double UnwrapNear(double yaw,
                                double previous) noexcept {
  const double period = 2.0 * std::numbers::pi;
  while (yaw - previous > std::numbers::pi) {
    yaw -= period;
  }
  while (yaw - previous < -std::numbers::pi) {
    yaw += period;
  }
  return yaw;
}

[[nodiscard]] PoseXyzYaw StatePose(
    const LeggedDiscretePlan& plan, std::size_t index,
    double preferred_height) noexcept {
  const auto& state = plan.states[index];
  const double raw_yaw =
      static_cast<double>(state.iyaw) *
      (2.0 * std::numbers::pi /
       static_cast<double>(plan.grid.yaw_bin_count));
  return PoseXyzYaw{
      .position_m =
          {
              plan.grid_origin_m.x +
                  (static_cast<double>(state.ix) + 0.5) *
                      plan.grid.xy_resolution_m,
              plan.grid_origin_m.y +
                  (static_cast<double>(state.iy) + 0.5) *
                      plan.grid.xy_resolution_m,
              std::clamp(preferred_height,
                         state.reachable_z.min_m,
                         state.reachable_z.max_m),
          },
      .yaw_rad = raw_yaw,
  };
}

[[nodiscard]] std::vector<PoseXyzYaw> InitialControlPoints(
    const LeggedSplineRequest& request) {
  std::vector<PoseXyzYaw> state_poses;
  state_poses.reserve(request.discrete_plan.states.size());
  for (std::size_t index = 0U;
       index < request.discrete_plan.states.size(); ++index) {
    auto pose = StatePose(request.discrete_plan, index,
                          request.preferred_body_height_m);
    if (!state_poses.empty()) {
      pose.yaw_rad =
          UnwrapNear(pose.yaw_rad, state_poses.back().yaw_rad);
    }
    state_poses.push_back(pose);
  }

  const std::size_t control_count =
      std::max(kMinimumControlPoints, state_poses.size());
  std::vector<PoseXyzYaw> controls;
  controls.reserve(control_count);
  for (std::size_t control = 0U; control < control_count;
       ++control) {
    const double scaled =
        control_count == 1U
            ? 0.0
            : static_cast<double>(control) /
                  static_cast<double>(control_count - 1U) *
                  static_cast<double>(state_poses.size() - 1U);
    const std::size_t lower =
        static_cast<std::size_t>(std::floor(scaled));
    const std::size_t upper =
        std::min(lower + 1U, state_poses.size() - 1U);
    const double alpha = scaled - static_cast<double>(lower);
    PoseXyzYaw pose;
    for (std::size_t dimension = 0U; dimension < kDimension;
         ++dimension) {
      SetComponent(
          pose, dimension,
          (1.0 - alpha) * Component(state_poses[lower], dimension) +
              alpha * Component(state_poses[upper], dimension));
    }
    controls.push_back(pose);
  }
  return controls;
}

[[nodiscard]] std::vector<double> ClampedUniformKnots(
    std::size_t control_count) {
  std::vector<double> knots(control_count + 4U, 0.0);
  for (std::size_t index = control_count;
       index < knots.size(); ++index) {
    knots[index] = 1.0;
  }
  if (control_count > 4U) {
    const double denominator =
        static_cast<double>(control_count - 3U);
    for (std::size_t internal = 1U;
         internal <= control_count - 4U; ++internal) {
      knots[internal + 3U] =
          static_cast<double>(internal) / denominator;
    }
  }
  return knots;
}

[[nodiscard]] const LeggedCorridorSection* SectionAt(
    const LeggedCorridor& corridor, double parameter) noexcept {
  if (corridor.sections.empty() || !std::isfinite(parameter)) {
    return nullptr;
  }
  const double bounded = std::clamp(parameter, 0.0, 1.0);
  const std::size_t index =
      std::min(
          static_cast<std::size_t>(
              std::floor(
                  bounded *
                  static_cast<double>(corridor.sections.size()))),
          corridor.sections.size() - 1U);
  return &corridor.sections[index];
}

[[nodiscard]] bool InsideSection(
    const LeggedCorridorSection& section,
    const PoseXyzYaw& pose, double tolerance) noexcept {
  if (!section.cartesian_product_certified ||
      !IsFinite(pose) ||
      pose.position_m.z < section.z.min_m - tolerance ||
      pose.position_m.z > section.z.max_m + tolerance ||
      pose.yaw_rad < section.yaw.min_unwrapped_rad - tolerance ||
      pose.yaw_rad > section.yaw.max_unwrapped_rad + tolerance) {
    return false;
  }
  return std::ranges::all_of(
      section.xy.half_planes, [&](const HalfPlane2& plane) {
        return plane.outward_unit_normal.x * pose.position_m.x +
                   plane.outward_unit_normal.y * pose.position_m.y <=
               plane.upper_offset_m + tolerance;
      });
}

struct QpBuild final {
  SparseQpProblem problem;
  std::vector<PoseXyzYaw> nominal;
};

[[nodiscard]] std::optional<QpBuild> BuildQp(
    const LeggedSplineRequest& request,
    const std::vector<PoseXyzYaw>& nominal) {
  using Triplet = Eigen::Triplet<double, int>;
  const std::size_t variable_count = nominal.size() * kDimension;
  if (variable_count >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return std::nullopt;
  }
  const auto variable = [](std::size_t point,
                           std::size_t dimension) {
    return static_cast<int>(point * kDimension + dimension);
  };

  std::vector<Triplet> hessian;
  Eigen::VectorXd gradient(
      static_cast<Eigen::Index>(variable_count));
  gradient.setZero();
  constexpr double kFidelityWeight = 2.0;
  for (std::size_t point = 0U; point < nominal.size(); ++point) {
    for (std::size_t dimension = 0U; dimension < kDimension;
         ++dimension) {
      const int index = variable(point, dimension);
      hessian.emplace_back(index, index, kFidelityWeight);
      gradient[index] =
          -kFidelityWeight * Component(nominal[point], dimension);
    }
  }
  for (std::size_t point = 1U; point + 1U < nominal.size();
       ++point) {
    for (std::size_t dimension = 0U; dimension < kDimension;
         ++dimension) {
      const double weight = dimension == 3U ? 0.1 : 0.2;
      const std::array<int, 3> indices{
          variable(point - 1U, dimension),
          variable(point, dimension),
          variable(point + 1U, dimension),
      };
      constexpr std::array<double, 3> coefficients{1.0, -2.0, 1.0};
      for (std::size_t row = 0U; row < 3U; ++row) {
        for (std::size_t column = row; column < 3U; ++column) {
          hessian.emplace_back(
              indices[row], indices[column],
              weight * coefficients[row] * coefficients[column]);
        }
      }
    }
  }

  std::vector<Triplet> constraints;
  std::vector<double> lower;
  std::vector<double> upper;
  const auto add_row =
      [&](std::span<const std::pair<int, double>> entries,
          double lower_bound, double upper_bound) {
        const int row = static_cast<int>(lower.size());
        for (const auto& [column, value] : entries) {
          constraints.emplace_back(row, column, value);
        }
        lower.push_back(lower_bound);
        upper.push_back(upper_bound);
      };
  const double trust =
      request.config.smoothing.initial_trust_region_m;
  if (!std::isfinite(trust) || trust <= 0.0) {
    return std::nullopt;
  }
  for (std::size_t point = 0U; point < nominal.size(); ++point) {
    const double parameter =
        static_cast<double>(point) /
        static_cast<double>(nominal.size() - 1U);
    const auto* section = SectionAt(request.corridor, parameter);
    if (section == nullptr || !section->cartesian_product_certified) {
      return std::nullopt;
    }
    for (const auto& plane : section->xy.half_planes) {
      const std::array entries{
          std::pair{variable(point, 0U),
                    plane.outward_unit_normal.x},
          std::pair{variable(point, 1U),
                    plane.outward_unit_normal.y},
      };
      add_row(entries, -kFiniteInfinity,
              plane.upper_offset_m);
    }
    const std::array z_entry{
        std::pair{variable(point, 2U), 1.0}};
    add_row(z_entry, section->z.min_m, section->z.max_m);
    const std::array yaw_entry{
        std::pair{variable(point, 3U), 1.0}};
    add_row(yaw_entry, section->yaw.min_unwrapped_rad,
            section->yaw.max_unwrapped_rad);
    for (std::size_t dimension = 0U; dimension < kDimension;
         ++dimension) {
      const std::array entry{
          std::pair{variable(point, dimension), 1.0}};
      const double value = Component(nominal[point], dimension);
      add_row(entry, value - trust, value + trust);
    }
  }

  std::set<std::size_t> frozen_indices;
  for (const auto& frozen : request.committed_points) {
    if (frozen.control_point_index >= nominal.size() ||
        !IsFinite(frozen.pose) ||
        !frozen_indices.insert(frozen.control_point_index).second) {
      return std::nullopt;
    }
    for (std::size_t dimension = 0U; dimension < kDimension;
         ++dimension) {
      const std::array entry{
          std::pair{variable(frozen.control_point_index, dimension),
                    1.0}};
      const double value = Component(frozen.pose, dimension);
      add_row(entry, value, value);
    }
  }

  SparseQpProblem problem;
  problem.hessian_upper_triangle.resize(
      static_cast<int>(variable_count),
      static_cast<int>(variable_count));
  problem.hessian_upper_triangle.setFromTriplets(
      hessian.begin(), hessian.end());
  problem.gradient = std::move(gradient);
  problem.constraints.resize(
      static_cast<int>(lower.size()),
      static_cast<int>(variable_count));
  problem.constraints.setFromTriplets(
      constraints.begin(), constraints.end());
  problem.lower_bounds =
      Eigen::Map<Eigen::VectorXd>(lower.data(),
                                  static_cast<Eigen::Index>(lower.size()));
  problem.upper_bounds =
      Eigen::Map<Eigen::VectorXd>(upper.data(),
                                  static_cast<Eigen::Index>(upper.size()));
  return QpBuild{
      .problem = std::move(problem),
      .nominal = nominal,
  };
}

[[nodiscard]] bool QpSolutionSatisfies(
    const SparseQpProblem& problem,
    const QpSolution& solution, double tolerance) {
  if (solution.primal.size() != problem.gradient.size() ||
      !solution.primal.allFinite() || !std::isfinite(tolerance) ||
      tolerance < 0.0) {
    return false;
  }
  const Eigen::VectorXd values =
      problem.constraints * solution.primal;
  for (Eigen::Index row = 0; row < values.size(); ++row) {
    if (values[row] < problem.lower_bounds[row] - tolerance ||
        values[row] > problem.upper_bounds[row] + tolerance) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::vector<PoseXyzYaw> Unpack(
    const Eigen::VectorXd& primal) {
  const std::size_t count =
      static_cast<std::size_t>(primal.size()) / kDimension;
  std::vector<PoseXyzYaw> controls(count);
  for (std::size_t point = 0U; point < count; ++point) {
    for (std::size_t dimension = 0U; dimension < kDimension;
         ++dimension) {
      SetComponent(
          controls[point], dimension,
          primal[static_cast<Eigen::Index>(
              point * kDimension + dimension)]);
    }
  }
  return controls;
}

[[nodiscard]] bool ValidateSpline(
    const ClampedCubicBSplinePath& spline,
    const LeggedSplineRequest& request) noexcept {
  if (request.config.validation_sample_count < 2U) {
    return false;
  }
  for (std::size_t sample = 0U;
       sample < request.config.validation_sample_count; ++sample) {
    const double parameter =
        static_cast<double>(sample) /
        static_cast<double>(
            request.config.validation_sample_count - 1U);
    const auto* section = SectionAt(request.corridor, parameter);
    if (section == nullptr ||
        !InsideSection(
            *section, EvaluateLeggedSpline(spline, parameter),
            request.config.smoothing.constraint_tolerance)) {
      return false;
    }
  }
  return true;
}

}  // namespace

PoseXyzYaw EvaluateLeggedSpline(
    const ClampedCubicBSplinePath& spline,
    double parameter) noexcept {
  constexpr std::size_t kDegree = 3U;
  if (spline.control_points.size() < kMinimumControlPoints ||
      spline.knots.size() != spline.control_points.size() + 4U ||
      !std::isfinite(parameter)) {
    return {};
  }
  const double u = std::clamp(parameter, 0.0, 1.0);
  const std::size_t n = spline.control_points.size() - 1U;
  std::size_t span = n;
  if (u < 1.0) {
    for (std::size_t candidate = kDegree; candidate <= n;
         ++candidate) {
      if (u >= spline.knots[candidate] &&
          u < spline.knots[candidate + 1U]) {
        span = candidate;
        break;
      }
    }
  }
  std::array<PoseXyzYaw, 4U> working{};
  for (std::size_t index = 0U; index <= kDegree; ++index) {
    working[index] =
        spline.control_points[span - kDegree + index];
  }
  for (std::size_t level = 1U; level <= kDegree; ++level) {
    for (std::size_t index = kDegree; index >= level; --index) {
      const std::size_t knot_index =
          span - kDegree + index;
      const double denominator =
          spline.knots[knot_index + kDegree - level + 1U] -
          spline.knots[knot_index];
      const double alpha =
          denominator > 0.0
              ? (u - spline.knots[knot_index]) / denominator
              : 0.0;
      for (std::size_t dimension = 0U; dimension < kDimension;
           ++dimension) {
        SetComponent(
            working[index], dimension,
            (1.0 - alpha) *
                    Component(working[index - 1U], dimension) +
                alpha * Component(working[index], dimension));
      }
      if (index == level) {
        break;
      }
    }
  }
  return working[kDegree];
}

double EvaluateLeggedSplineTangentYaw(
    const ClampedCubicBSplinePath& spline,
    double parameter) noexcept {
  constexpr double kDelta = 1.0e-5;
  const auto before =
      EvaluateLeggedSpline(spline, parameter - kDelta);
  const auto after =
      EvaluateLeggedSpline(spline, parameter + kDelta);
  const double dx = after.position_m.x - before.position_m.x;
  const double dy = after.position_m.y - before.position_m.y;
  if (std::hypot(dx, dy) <=
      std::numeric_limits<double>::epsilon()) {
    return EvaluateLeggedSpline(spline, parameter).yaw_rad;
  }
  return std::atan2(dy, dx);
}

LeggedSplineResult OptimizeLeggedBodySpline(
    const LeggedSplineRequest& request,
    BoundedQpSolver& solver) {
  LeggedSplineResult result{
      .body_spline = std::nullopt,
      .estimated_execution_time =
          request.discrete_plan.estimated_execution_time,
      .termination = LeggedOptimizationTermination::kInvalidRequest,
  };
  if (request.discrete_plan.states.empty() ||
      request.discrete_plan.grid.yaw_bin_count == 0U ||
      request.corridor.sections.empty() ||
      !std::isfinite(request.preferred_body_height_m) ||
      request.config.smoothing.maximum_scp_iterations == 0U ||
      request.config.validation_sample_count < 2U ||
      request.discrete_plan.estimated_execution_time.value.count() < 0 ||
      request.config.smoothing.maximum_time_increase.value.count() < 0) {
    return result;
  }

  auto controls = InitialControlPoints(request);
  if (controls.size() < kMinimumControlPoints ||
      !std::ranges::all_of(controls, IsFinite)) {
    return result;
  }

  bool solved = false;
  for (std::size_t iteration = 0U;
       iteration < request.config.smoothing.maximum_scp_iterations;
       ++iteration) {
    const auto qp = BuildQp(request, controls);
    if (!qp.has_value()) {
      return result;
    }
    const auto solution =
        solver.Solve(qp->problem, request.config.qp_settings);
    if (!IsOk(solution)) {
      result.termination = LeggedOptimizationTermination::kQpFailure;
      return result;
    }
    const auto& solved_qp = std::get<QpSolution>(solution);
    if (solved_qp.termination != QpTermination::kSolved ||
        !QpSolutionSatisfies(
            qp->problem, solved_qp,
            request.config.smoothing.constraint_tolerance)) {
      result.termination = LeggedOptimizationTermination::kQpFailure;
      return result;
    }
    auto next_controls = Unpack(solved_qp.primal);
    for (const auto& frozen : request.committed_points) {
      next_controls[frozen.control_point_index] = frozen.pose;
    }
    double maximum_change = 0.0;
    for (std::size_t point = 0U; point < controls.size(); ++point) {
      for (std::size_t dimension = 0U; dimension < kDimension;
           ++dimension) {
        maximum_change = std::max(
            maximum_change,
            std::abs(Component(next_controls[point], dimension) -
                     Component(controls[point], dimension)));
      }
    }
    controls = std::move(next_controls);
    if (maximum_change >
            request.config.smoothing.constraint_tolerance &&
        iteration + 1U <
            request.config.smoothing.maximum_scp_iterations) {
      continue;
    }
    solved = true;
    break;
  }
  if (!solved) {
    result.termination = LeggedOptimizationTermination::kQpFailure;
    return result;
  }

  ClampedCubicBSplinePath spline{
      .knots = ClampedUniformKnots(controls.size()),
      .control_points = std::move(controls),
  };
  if (!ValidateSpline(spline, request)) {
    result.termination =
        LeggedOptimizationTermination::kContinuousValidationFailure;
    return result;
  }
  if (result.estimated_execution_time.value >
      request.discrete_plan.estimated_execution_time.value +
          request.config.smoothing.maximum_time_increase.value) {
    result.termination =
        LeggedOptimizationTermination::kTimeEquivalentViolation;
    return result;
  }
  result.body_spline = std::move(spline);
  result.termination = LeggedOptimizationTermination::kSolved;
  return result;
}

}  // namespace lunar::planning::v3
