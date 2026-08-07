#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <span>
#include <string>

#include "hierarchical/global_route.hpp"
#include "lunar_planner_core/types/planner_io.hpp"
#include "shared/projection_cache.hpp"

namespace lunar::planning::hierarchical {

struct GlobalRoutePlanResult final {
  PlanningOutcome outcome{PlanningOutcome::kInvalidRequest};
  std::string reason_code;
  std::optional<GlobalRoute> route;
  std::optional<std::size_t> global_level;
  std::chrono::nanoseconds elapsed{};
  bool projection_cache_hit{};

  [[nodiscard]] bool ok() const noexcept {
    return route.has_value() && global_level.has_value() &&
           outcome == PlanningOutcome::kNewReferenceAvailable &&
           reason_code == "GLOBAL_ROUTE_AVAILABLE";
  }
};

[[nodiscard]] GlobalRoutePlanResult
PlanGroundGlobalRoute(const PlannerInput &input,
                      std::span<const shared::GridCell> excluded_cells = {},
                      shared::ProjectionCache *projection_cache = nullptr);

} // namespace lunar::planning::hierarchical
