#pragma once

#include <cstddef>
#include <variant>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "lunar_path_planner/v3/contracts/status.hpp"

namespace lunar::planning::v3 {

using SparseQpMatrix =
    Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

struct SparseQpProblem final {
  SparseQpMatrix hessian_upper_triangle;
  Eigen::VectorXd gradient;
  SparseQpMatrix constraints;
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;
};

struct BoundedQpSettings final {
  std::size_t max_iterations{};
  double absolute_tolerance{};
  double relative_tolerance{};
  bool polish{};
};

enum class QpTermination {
  kSolved,
  kMaxIterations,
  kPrimalInfeasible,
  kDualInfeasible,
  kNumericalFailure,
};

struct QpSolution final {
  QpTermination termination{QpTermination::kNumericalFailure};
  Eigen::VectorXd primal;
  double objective{};
  double primal_residual{};
  double dual_residual{};
  std::size_t iterations{};
};

[[nodiscard]] Result<std::monostate> ValidateBoundedQpInputs(
    const SparseQpProblem& problem,
    const BoundedQpSettings& settings);

class BoundedQpSolver {
 public:
  virtual ~BoundedQpSolver() = default;

  [[nodiscard]] Result<QpSolution> Solve(
      const SparseQpProblem& problem,
      const BoundedQpSettings& settings) const;

 protected:
  [[nodiscard]] virtual Result<QpSolution> DoSolve(
      const SparseQpProblem& problem,
      const BoundedQpSettings& settings) const = 0;
};

#if defined(LPP_V3_HAS_OSQP)
class OsqpBoundedQpSolver final : public BoundedQpSolver {
 private:
  [[nodiscard]] Result<QpSolution> DoSolve(
      const SparseQpProblem& problem,
      const BoundedQpSettings& settings) const override;
};
#endif

}  // namespace lunar::planning::v3
