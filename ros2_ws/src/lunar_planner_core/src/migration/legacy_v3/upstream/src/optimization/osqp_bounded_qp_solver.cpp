#include "lunar_path_planner/v3/optimization/bounded_qp_solver.hpp"

#if defined(LPP_V3_HAS_OSQP)

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <osqp.h>

namespace lunar::planning::v3 {
namespace {

static_assert(
    std::is_same_v<OSQPFloat, double>,
    "The pinned OSQP backend must use double-precision scalars");

[[nodiscard]] Error NumericalError(
    std::string field_path,
    std::string message) {
  return Error{
      .code = ErrorCode::kNumericalFailure,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] Error InvalidArgument(
    std::string field_path,
    std::string message) {
  return Error{
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] std::string OsqpErrorMessage(OSQPInt error_flag) {
  const char* const native_message = osqp_error_message(error_flag);
  if (native_message == nullptr) {
    return "OSQP returned an unknown native error";
  }
  return std::string(native_message);
}

struct OsqpSolverDeleter final {
  void operator()(OSQPSolver* solver) const noexcept {
    if (solver != nullptr) {
      static_cast<void>(osqp_cleanup(solver));
    }
  }
};

using UniqueOsqpSolver =
    std::unique_ptr<OSQPSolver, OsqpSolverDeleter>;

[[nodiscard]] QpTermination MapTermination(OSQPInt status) noexcept {
  switch (status) {
    case OSQP_SOLVED:
    case OSQP_SOLVED_INACCURATE:
      return QpTermination::kSolved;
    case OSQP_PRIMAL_INFEASIBLE:
    case OSQP_PRIMAL_INFEASIBLE_INACCURATE:
      return QpTermination::kPrimalInfeasible;
    case OSQP_DUAL_INFEASIBLE:
    case OSQP_DUAL_INFEASIBLE_INACCURATE:
      return QpTermination::kDualInfeasible;
    case OSQP_MAX_ITER_REACHED:
      return QpTermination::kMaxIterations;
    case OSQP_TIME_LIMIT_REACHED:
    case OSQP_NON_CVX:
    case OSQP_SIGINT:
    case OSQP_UNSOLVED:
      return QpTermination::kNumericalFailure;
    default:
      return QpTermination::kNumericalFailure;
  }
}

struct OsqpCscStorage final {
  explicit OsqpCscStorage(const SparseQpMatrix& source)
      : compressed(source) {
    compressed.makeCompressed();
    values.reserve(static_cast<std::size_t>(compressed.nonZeros()));
    row_indices.reserve(
        static_cast<std::size_t>(compressed.nonZeros()));
    column_pointers.reserve(
        static_cast<std::size_t>(compressed.cols() + 1));

    for (Eigen::Index entry = 0; entry < compressed.nonZeros(); ++entry) {
      values.push_back(
          static_cast<OSQPFloat>(compressed.valuePtr()[entry]));
      row_indices.push_back(
          static_cast<OSQPInt>(compressed.innerIndexPtr()[entry]));
    }
    for (Eigen::Index column = 0;
         column <= compressed.cols();
         ++column) {
      column_pointers.push_back(
          static_cast<OSQPInt>(compressed.outerIndexPtr()[column]));
    }

    OSQPCscMatrix_set_data(
        &matrix,
        static_cast<OSQPInt>(compressed.rows()),
        static_cast<OSQPInt>(compressed.cols()),
        static_cast<OSQPInt>(compressed.nonZeros()),
        values.data(),
        row_indices.data(),
        column_pointers.data());
  }

  OsqpCscStorage(const OsqpCscStorage&) = delete;
  OsqpCscStorage& operator=(const OsqpCscStorage&) = delete;
  OsqpCscStorage(OsqpCscStorage&&) = delete;
  OsqpCscStorage& operator=(OsqpCscStorage&&) = delete;

  SparseQpMatrix compressed;
  std::vector<OSQPFloat> values;
  std::vector<OSQPInt> row_indices;
  std::vector<OSQPInt> column_pointers;
  OSQPCscMatrix matrix{};
};

[[nodiscard]] Result<QpSolution> CopySolution(
    const OSQPSolver& solver,
    Eigen::Index decision_count) {
  if (solver.info == nullptr) {
    return NumericalError(
        "osqp.info", "OSQP did not return solver information");
  }
  if (solver.solution == nullptr || solver.solution->x == nullptr) {
    return NumericalError(
        "osqp.solution.primal",
        "OSQP did not return a primal vector");
  }
  if (solver.info->iter < 0) {
    return NumericalError(
        "osqp.info.iter",
        "OSQP returned a negative iteration count");
  }

  QpSolution solution{
      .termination = MapTermination(solver.info->status_val),
      .primal = Eigen::VectorXd(decision_count),
      .objective = static_cast<double>(solver.info->obj_val),
      .primal_residual = static_cast<double>(solver.info->prim_res),
      .dual_residual = static_cast<double>(solver.info->dual_res),
      .iterations = static_cast<std::size_t>(solver.info->iter),
  };
  for (Eigen::Index index = 0; index < decision_count; ++index) {
    solution.primal[index] =
        static_cast<double>(solver.solution->x[index]);
  }

  if (!solution.primal.allFinite()) {
    return NumericalError(
        "osqp.solution.primal",
        "OSQP returned a non-finite primal vector");
  }
  if (!std::isfinite(solution.objective)) {
    return NumericalError(
        "osqp.info.obj_val",
        "OSQP returned a non-finite objective");
  }
  if (!std::isfinite(solution.primal_residual) ||
      solution.primal_residual < 0.0) {
    return NumericalError(
        "osqp.info.prim_res",
        "OSQP returned an invalid primal residual");
  }
  if (!std::isfinite(solution.dual_residual) ||
      solution.dual_residual < 0.0) {
    return NumericalError(
        "osqp.info.dual_res",
        "OSQP returned an invalid dual residual");
  }
  return solution;
}

}  // namespace

Result<QpSolution> OsqpBoundedQpSolver::DoSolve(
    const SparseQpProblem& problem,
    const BoundedQpSettings& settings) const {
  if (settings.max_iterations >
      static_cast<std::size_t>(
          std::numeric_limits<OSQPInt>::max())) {
    return InvalidArgument(
        "settings.max_iterations",
        "maximum iterations exceed the OSQP integer range");
  }

  OsqpCscStorage hessian(problem.hessian_upper_triangle);
  OsqpCscStorage constraints(problem.constraints);

  OSQPSettings native_settings{};
  osqp_set_default_settings(&native_settings);
  native_settings.max_iter =
      static_cast<OSQPInt>(settings.max_iterations);
  native_settings.eps_abs =
      static_cast<OSQPFloat>(settings.absolute_tolerance);
  native_settings.eps_rel =
      static_cast<OSQPFloat>(settings.relative_tolerance);
  native_settings.polishing = settings.polish ? 1 : 0;
  native_settings.allocate_solution = 1;
  native_settings.verbose = 0;
  native_settings.profiler_level = 0;
  native_settings.warm_starting = 0;
  native_settings.adaptive_rho =
      OSQP_ADAPTIVE_RHO_UPDATE_DISABLED;
  native_settings.time_limit = OSQP_TIME_LIMIT;

  OSQPSolver* raw_solver = nullptr;
  const OSQPInt setup_result = osqp_setup(
      &raw_solver,
      &hessian.matrix,
      problem.gradient.data(),
      &constraints.matrix,
      problem.lower_bounds.data(),
      problem.upper_bounds.data(),
      static_cast<OSQPInt>(problem.constraints.rows()),
      static_cast<OSQPInt>(problem.gradient.size()),
      &native_settings);
  UniqueOsqpSolver solver(raw_solver);
  if (setup_result != 0) {
    return NumericalError(
        "osqp.setup", OsqpErrorMessage(setup_result));
  }
  if (solver == nullptr) {
    return NumericalError(
        "osqp.setup", "OSQP setup returned a null solver");
  }

  const OSQPInt solve_result = osqp_solve(solver.get());
  if (solve_result != 0) {
    return NumericalError(
        "osqp.solve", OsqpErrorMessage(solve_result));
  }

  return CopySolution(*solver, problem.gradient.size());
}

}  // namespace lunar::planning::v3

#endif
