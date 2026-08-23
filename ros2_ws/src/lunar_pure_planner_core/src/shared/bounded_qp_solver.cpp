#include "shared/bounded_qp_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace lunar::pure_planning::shared {
namespace {

constexpr double kSymmetryTolerance = 1.0e-10;
constexpr double kConvexityTolerance = 1.0e-12;

[[nodiscard]] BoundedQpSolution Failure(
    const QpTermination termination, std::string reason_code,
    const std::size_t iterations = 0U,
    std::vector<double> primal = {}) {
  return BoundedQpSolution{
      .termination = termination,
      .primal = std::move(primal),
      .iterations = iterations,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool AllFinite(
    const std::vector<double>& values) noexcept {
  return std::ranges::all_of(values, [](const double value) {
    return std::isfinite(value);
  });
}

[[nodiscard]] std::string Validate(
    const BoundedQpProblem& problem,
    const BoundedQpSettings& settings) {
  if (problem.dimension == 0U ||
      problem.dimension >
          std::numeric_limits<std::size_t>::max() / problem.dimension ||
      problem.hessian.size() != problem.dimension * problem.dimension ||
      problem.gradient.size() != problem.dimension ||
      problem.lower_bounds.size() != problem.dimension ||
      problem.upper_bounds.size() != problem.dimension) {
    return "QP_SHAPE_INVALID";
  }
  if (settings.maximum_iterations == 0U ||
      !std::isfinite(settings.absolute_tolerance) ||
      settings.absolute_tolerance <= 0.0) {
    return "QP_SETTINGS_INVALID";
  }
  if (!AllFinite(problem.hessian)) {
    return "QP_NONFINITE_HESSIAN";
  }
  if (!AllFinite(problem.gradient)) {
    return "QP_NONFINITE_GRADIENT";
  }
  if (!AllFinite(problem.lower_bounds) ||
      !AllFinite(problem.upper_bounds)) {
    return "QP_NONFINITE_BOUNDS";
  }
  for (std::size_t index = 0U; index < problem.dimension; ++index) {
    if (problem.lower_bounds[index] > problem.upper_bounds[index]) {
      return "QP_BOUNDS_INVALID";
    }
    for (std::size_t column = 0U; column < index; ++column) {
      const double lower =
          problem.hessian[index * problem.dimension + column];
      const double upper =
          problem.hessian[column * problem.dimension + index];
      if (std::abs(lower - upper) > kSymmetryTolerance) {
        return "QP_HESSIAN_NOT_SYMMETRIC";
      }
    }
  }

  std::vector<double> cholesky(
      problem.dimension * problem.dimension, 0.0);
  for (std::size_t row = 0U; row < problem.dimension; ++row) {
    for (std::size_t column = 0U; column <= row; ++column) {
      double residual =
          problem.hessian[row * problem.dimension + column];
      for (std::size_t inner = 0U; inner < column; ++inner) {
        residual -= cholesky[row * problem.dimension + inner] *
                    cholesky[column * problem.dimension + inner];
      }
      if (row == column) {
        if (residual < -kConvexityTolerance) {
          return "QP_HESSIAN_NOT_CONVEX";
        }
        cholesky[row * problem.dimension + column] =
            std::sqrt(std::max(0.0, residual));
      } else {
        const double diagonal =
            cholesky[column * problem.dimension + column];
        if (diagonal > kConvexityTolerance) {
          cholesky[row * problem.dimension + column] = residual / diagonal;
        } else if (std::abs(residual) > kConvexityTolerance) {
          return "QP_HESSIAN_NOT_CONVEX";
        }
      }
    }
  }
  return {};
}

[[nodiscard]] std::vector<double> Gradient(
    const BoundedQpProblem& problem,
    const std::vector<double>& primal) {
  std::vector<double> gradient = problem.gradient;
  for (std::size_t row = 0U; row < problem.dimension; ++row) {
    for (std::size_t column = 0U; column < problem.dimension; ++column) {
      gradient[row] +=
          problem.hessian[row * problem.dimension + column] * primal[column];
    }
  }
  return gradient;
}

[[nodiscard]] double ProjectedGradientNorm(
    const BoundedQpProblem& problem,
    const std::vector<double>& primal,
    const std::vector<double>& gradient) noexcept {
  double norm = 0.0;
  for (std::size_t index = 0U; index < problem.dimension; ++index) {
    const double projected = std::clamp(
        primal[index] - gradient[index],
        problem.lower_bounds[index], problem.upper_bounds[index]);
    norm = std::max(norm, std::abs(primal[index] - projected));
  }
  return norm;
}

[[nodiscard]] double LipschitzUpperBound(
    const BoundedQpProblem& problem) noexcept {
  double maximum_row_sum = 0.0;
  for (std::size_t row = 0U; row < problem.dimension; ++row) {
    double row_sum = 0.0;
    for (std::size_t column = 0U; column < problem.dimension; ++column) {
      row_sum +=
          std::abs(problem.hessian[row * problem.dimension + column]);
    }
    maximum_row_sum = std::max(maximum_row_sum, row_sum);
  }
  return std::max(maximum_row_sum, 1.0);
}

[[nodiscard]] double Objective(
    const BoundedQpProblem& problem,
    const std::vector<double>& primal) noexcept {
  double objective = 0.0;
  for (std::size_t row = 0U; row < problem.dimension; ++row) {
    objective += problem.gradient[row] * primal[row];
    for (std::size_t column = 0U; column < problem.dimension; ++column) {
      objective += 0.5 * primal[row] *
          problem.hessian[row * problem.dimension + column] * primal[column];
    }
  }
  return objective;
}

}  // namespace

BoundedQpSolution SolveBoundedQp(
    const BoundedQpProblem& problem,
    const BoundedQpSettings& settings,
    const std::stop_token stop_token) {
  if (const std::string reason = Validate(problem, settings); !reason.empty()) {
    return Failure(QpTermination::kInvalidProblem, reason);
  }
  std::vector<double> primal(problem.dimension, 0.0);
  for (std::size_t index = 0U; index < problem.dimension; ++index) {
    primal[index] = std::clamp(
        0.0, problem.lower_bounds[index], problem.upper_bounds[index]);
  }
  const double step_size = 1.0 / LipschitzUpperBound(problem);
  for (std::size_t iteration = 0U;
       iteration < settings.maximum_iterations; ++iteration) {
    if (stop_token.stop_requested()) {
      return Failure(
          QpTermination::kCanceled, "REQUEST_CANCELED", iteration,
          std::move(primal));
    }
    const std::vector<double> gradient = Gradient(problem, primal);
    if (!AllFinite(gradient)) {
      return Failure(
          QpTermination::kNumericalFailure, "QP_NUMERICAL_FAILURE",
          iteration, std::move(primal));
    }
    const double projected_norm =
        ProjectedGradientNorm(problem, primal, gradient);
    if (projected_norm <= settings.absolute_tolerance) {
      const double objective = Objective(problem, primal);
      return BoundedQpSolution{
          .termination = QpTermination::kSolved,
          .primal = std::move(primal),
          .objective = objective,
          .projected_gradient_norm = projected_norm,
          .iterations = iteration,
          .reason_code = "QP_SOLVED",
      };
    }
    for (std::size_t index = 0U; index < problem.dimension; ++index) {
      primal[index] = std::clamp(
          primal[index] - step_size * gradient[index],
          problem.lower_bounds[index], problem.upper_bounds[index]);
    }
    if (!AllFinite(primal)) {
      return Failure(
          QpTermination::kNumericalFailure, "QP_NUMERICAL_FAILURE",
          iteration + 1U, std::move(primal));
    }
  }

  const std::vector<double> gradient = Gradient(problem, primal);
  const double objective = Objective(problem, primal);
  const double projected_norm =
      ProjectedGradientNorm(problem, primal, gradient);
  return BoundedQpSolution{
      .termination = QpTermination::kMaximumIterations,
      .primal = std::move(primal),
      .objective = objective,
      .projected_gradient_norm = projected_norm,
      .iterations = settings.maximum_iterations,
      .reason_code = "QP_MAXIMUM_ITERATIONS",
  };
}

}  // namespace lunar::pure_planning::shared
