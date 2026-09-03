#pragma once

#include <optional>

#include "lunar_incremental_navigation_core/traversability_snapshot.hpp"
#include "lunar_incremental_navigation_core/types/global_route.hpp"
#include "lunar_incremental_navigation_core/types/planning_cycle.hpp"
#include "lunar_incremental_navigation_core/types/platform_capability.hpp"

namespace lunar::incremental_navigation {

struct LocalTarget final {
  Point2 center;
  double position_tolerance_m{};
  bool is_final_goal{};
  std::optional<double> terminal_yaw_rad;
};

class LocalTargetSelector final {
 public:
  [[nodiscard]] std::optional<LocalTarget> Select(
      const FineTraversabilitySnapshot& fine,
      const SparseGridGeometry& local_window, Point2 start,
      const FinalGoal& final_goal,
      const std::optional<GlobalRoute>& guidance) const;

  // Compatibility entry point for callers that intentionally operate on the
  // whole snapshot.  Session planning always supplies a bounded window.
  [[nodiscard]] std::optional<LocalTarget> Select(
      const FineTraversabilitySnapshot& fine, Point2 start,
      const FinalGoal& final_goal,
      const std::optional<GlobalRoute>& guidance) const;
};

}  // namespace lunar::incremental_navigation
