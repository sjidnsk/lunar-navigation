#include "lunar_path_planner/v3/optimization/bounded_qp_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include <Eigen/Eigenvalues>

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] Error InvalidArgument(
    std::string field_path,
    std::string message) {
  return Error{
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool SparseValuesAreFinite(
    const SparseQpMatrix& matrix) {
  for (int outer = 0; outer < matrix.outerSize(); ++outer) {
    for (SparseQpMatrix::InnerIterator entry(matrix, outer); entry; ++entry) {
      if (!std::isfinite(entry.value())) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool UsesUpperTriangleConvention(
    const SparseQpMatrix& hessian_upper_triangle) {
  for (int outer = 0;
       outer < hessian_upper_triangle.outerSize();
       ++outer) {
    for (SparseQpMatrix::InnerIterator entry(
             hessian_upper_triangle, outer);
         entry;
         ++entry) {
      if (entry.row() > entry.col()) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] Result<std::monostate> ValidatePositiveSemidefinite(
    const SparseQpMatrix& hessian_upper_triangle) {
  const Eigen::Index decision_count = hessian_upper_triangle.rows();
  Eigen::MatrixXd symmetric_hessian =
      Eigen::MatrixXd::Zero(decision_count, decision_count);

  for (int outer = 0;
       outer < hessian_upper_triangle.outerSize();
       ++outer) {
    for (SparseQpMatrix::InnerIterator entry(
             hessian_upper_triangle, outer);
         entry;
         ++entry) {
      symmetric_hessian(entry.row(), entry.col()) = entry.value();
      if (entry.row() != entry.col()) {
        symmetric_hessian(entry.col(), entry.row()) = entry.value();
      }
    }
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> factorization(
      symmetric_hessian, Eigen::EigenvaluesOnly);
  if (factorization.info() != Eigen::Success) {
    return Error{
        .code = ErrorCode::kNumericalFailure,
        .field_path = "problem.hessian_upper_triangle",
        .message =
            "positive-semidefinite factorization did not converge",
    };
  }

  double maximum_absolute_row_sum = 0.0;
  for (Eigen::Index row = 0; row < decision_count; ++row) {
    double absolute_row_sum = 0.0;
    for (Eigen::Index column = 0; column < decision_count; ++column) {
      absolute_row_sum += std::abs(symmetric_hessian(row, column));
    }
    maximum_absolute_row_sum =
        std::max(maximum_absolute_row_sum, absolute_row_sum);
  }
  const double scale = std::max(1.0, maximum_absolute_row_sum);
  const double tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * scale *
      static_cast<double>(std::max<Eigen::Index>(decision_count, 1));
  if (factorization.eigenvalues().minCoeff() < -tolerance) {
    return InvalidArgument(
        "problem.hessian_upper_triangle",
        "hessian must be positive semidefinite");
  }

  return std::monostate{};
}

}  // namespace

Result<std::monostate> ValidateBoundedQpInputs(
    const SparseQpProblem& problem,
    const BoundedQpSettings& settings) {
  if (problem.hessian_upper_triangle.rows() !=
      problem.hessian_upper_triangle.cols()) {
    return InvalidArgument(
        "problem.hessian_upper_triangle",
        "hessian must be square");
  }

  const Eigen::Index decision_count =
      problem.hessian_upper_triangle.rows();
  if (decision_count <= 0) {
    return InvalidArgument(
        "problem.gradient",
        "at least one decision variable is required");
  }
  if (problem.gradient.size() != decision_count) {
    return InvalidArgument(
        "problem.gradient",
        "gradient size must equal the hessian dimension");
  }

  if (problem.constraints.cols() != decision_count) {
    return InvalidArgument(
        "problem.constraints",
        "constraint columns must equal the decision dimension");
  }

  const Eigen::Index constraint_count = problem.constraints.rows();
  if (problem.lower_bounds.size() != constraint_count) {
    return InvalidArgument(
        "problem.lower_bounds",
        "lower-bound size must equal the constraint row count");
  }
  if (problem.upper_bounds.size() != constraint_count) {
    return InvalidArgument(
        "problem.upper_bounds",
        "upper-bound size must equal the constraint row count");
  }

  if (!SparseValuesAreFinite(problem.hessian_upper_triangle)) {
    return InvalidArgument(
        "problem.hessian_upper_triangle",
        "hessian coefficients must be finite");
  }
  if (!problem.gradient.allFinite()) {
    return InvalidArgument(
        "problem.gradient",
        "gradient coefficients must be finite");
  }
  if (!SparseValuesAreFinite(problem.constraints)) {
    return InvalidArgument(
        "problem.constraints",
        "constraint coefficients must be finite");
  }
  if (!problem.lower_bounds.allFinite()) {
    return InvalidArgument(
        "problem.lower_bounds",
        "lower bounds must be finite");
  }
  if (!problem.upper_bounds.allFinite()) {
    return InvalidArgument(
        "problem.upper_bounds",
        "upper bounds must be finite");
  }

  for (Eigen::Index row = 0; row < constraint_count; ++row) {
    if (problem.lower_bounds[row] > problem.upper_bounds[row]) {
      return InvalidArgument(
          "problem.bounds[" + std::to_string(row) + "]",
          "lower bound must not exceed upper bound");
    }
  }

  if (!UsesUpperTriangleConvention(problem.hessian_upper_triangle)) {
    return InvalidArgument(
        "problem.hessian_upper_triangle",
        "hessian must store only diagonal and upper-triangular entries");
  }

  if (settings.max_iterations == 0U) {
    return InvalidArgument(
        "settings.max_iterations",
        "maximum iterations must be positive");
  }
  if (settings.max_iterations >
      static_cast<std::size_t>(
          std::numeric_limits<std::int64_t>::max())) {
    return InvalidArgument(
        "settings.max_iterations",
        "maximum iterations exceed the supported signed range");
  }
  if (!(settings.absolute_tolerance > 0.0) ||
      !std::isfinite(settings.absolute_tolerance)) {
    return InvalidArgument(
        "settings.absolute_tolerance",
        "absolute tolerance must be positive and finite");
  }
  if (!(settings.relative_tolerance > 0.0) ||
      !std::isfinite(settings.relative_tolerance)) {
    return InvalidArgument(
        "settings.relative_tolerance",
        "relative tolerance must be positive and finite");
  }

  return ValidatePositiveSemidefinite(problem.hessian_upper_triangle);
}

Result<QpSolution> BoundedQpSolver::Solve(
    const SparseQpProblem& problem,
    const BoundedQpSettings& settings) const {
  const auto validation = ValidateBoundedQpInputs(problem, settings);
  if (!IsOk(validation)) {
    return std::get<Error>(validation);
  }
  return DoSolve(problem, settings);
}

}  // namespace lunar::planning::v3
