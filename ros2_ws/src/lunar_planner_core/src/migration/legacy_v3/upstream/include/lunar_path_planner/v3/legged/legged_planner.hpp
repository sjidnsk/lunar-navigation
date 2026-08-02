#pragma once

#include "lunar_path_planner/v3/goal/terminal_resolver.hpp"
#include "lunar_path_planner/v3/legged/legged_spline_optimizer.hpp"
#include "lunar_path_planner/v3/legged/legged_timing.hpp"

namespace lunar::planning::v3 {

struct LeggedPlannerDependencies final {
  BoundedQpSolver* qp_solver{};
  const LeggedCartesianProductCertifier* product_certifier{};
  BoundedQpSettings qp_settings{
      .max_iterations = 256U,
      .absolute_tolerance = 1.0e-8,
      .relative_tolerance = 1.0e-8,
      .polish = false,
  };
};

class LeggedPlanner final {
 public:
  explicit LeggedPlanner(
      LeggedPlannerDependencies dependencies = {}) noexcept;

  // Produces only an authoritative fixed body-reference xyz+yaw path and
  // timing. Footsteps, gait, contacts, and contact forces are intentionally
  // outside this interface.
  [[nodiscard]] Result<LeggedBodyReference> Plan(
      const PlanningRequest& request,
      const ResolvedTerminalSet& terminal_set) const noexcept;

 private:
  LeggedPlannerDependencies dependencies_;
};

}  // namespace lunar::planning::v3
