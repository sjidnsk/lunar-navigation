#pragma once

#include <cstddef>
#include <stop_token>
#include <string>
#include <vector>

namespace lunar::pure_planning::shared {

struct BoundedQpProblem final {
  std::size_t dimension{};
  std::vector<double> hessian;
  std::vector<double> gradient;
  std::vector<double> lower_bounds;
  std::vector<double> upper_bounds;
};

struct BoundedQpSettings final {
  std::size_t maximum_iterations{};
  double absolute_tolerance{};
};

enum class QpTermination {
  kSolved,
  kMaximumIterations,
  kCanceled,
  kInvalidProblem,
  kNumericalFailure,
};

struct BoundedQpSolution final {
  QpTermination termination{QpTermination::kInvalidProblem};
  std::vector<double> primal;
  double objective{};
  double projected_gradient_norm{};
  std::size_t iterations{};
  std::string reason_code;
};

[[nodiscard]] BoundedQpSolution SolveBoundedQp(
    const BoundedQpProblem& problem,
    const BoundedQpSettings& settings,
    std::stop_token stop_token);

}  // namespace lunar::pure_planning::shared
