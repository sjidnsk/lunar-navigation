#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "hierarchical/surface_rolling_session.hpp"
#include "lunar_pure_planner_core/planner.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::hierarchical {

struct SurfacePortalCandidate final {
  GoalRegion goal_odom;
  double route_progress_m{};
  shared::GridCell global_cell;
  shared::GridCell local_cell;
  float global_clearance_m{};
  float local_clearance_m{};
  std::size_t stable_rank{};
};

struct SurfacePortalSetResult final {
  std::vector<SurfacePortalCandidate> candidates;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return !candidates.empty() && reason_code.empty();
  }
};

[[nodiscard]] SurfacePortalSetResult BuildSurfacePortalSet(
    const PlanningRequest& input, const GlobalRoute& route,
    const SurfaceRollingDecision& decision, std::size_t max_candidates,
    SearchControl control);

}  // namespace lunar::pure_planning::hierarchical
