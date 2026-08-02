#pragma once

#include "lunar_path_planner/v3/optimization/bounded_qp_solver.hpp"
#include "lunar_path_planner/v3/wheel/wheel_search.hpp"
#include "lunar_path_planner/v3/wheel/wheel_spline_optimizer.hpp"
#include "lunar_path_planner/v3/wheel/wheel_sweep_validator.hpp"
#include "lunar_path_planner/v3/wheel/wheel_timing.hpp"

namespace lunar::planning::v3 {

using WheelCorridorBuilder =
    CorridorResult (*)(const WheelCorridorRequest&);

class WheelPlanner final {
 public:
  WheelPlanner() noexcept;

  explicit WheelPlanner(
      BoundedQpSolver* qp_solver,
      WheelCorridorBuilder corridor_builder =
          &BuildWheelCorridor) noexcept;

  [[nodiscard]] Result<WheeledReference> Plan(
      const PlanningRequest& request,
      const ResolvedTerminalSet& terminal_set) noexcept;

 private:
  BoundedQpSolver* qp_solver_{};
  WheelCorridorBuilder corridor_builder_{};
};

}  // namespace lunar::planning::v3
