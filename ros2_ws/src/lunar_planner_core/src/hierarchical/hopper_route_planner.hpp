#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>

#include "hierarchical/global_route.hpp"
#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning::hierarchical {

struct HopperRoutePlanResult final {
  PlanningOutcome outcome{PlanningOutcome::kInvalidRequest};
  std::optional<GlobalRoute> route;
  std::optional<std::size_t> global_level;
  double maximum_horizontal_reach_m{};
  std::size_t graph_nodes{};
  std::size_t graph_edges{};
  std::size_t evaluated_edge_pairs{};
  std::size_t route_hops{};
  std::uint64_t expanded_nodes{};
  bool graph_truncated{};
  std::chrono::nanoseconds elapsed{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return outcome == PlanningOutcome::kNewReferenceAvailable &&
           route.has_value() && route_hops > 0U &&
           reason_code == "HOPPER_GLOBAL_ROUTE_AVAILABLE";
  }
};

[[nodiscard]] double
ConservativeMaximumHorizontalReach(const HopperCapability &capability) noexcept;

[[nodiscard]] HopperRoutePlanResult
PlanHopperGlobalRoute(const PlannerInput &input);

} // namespace lunar::planning::hierarchical
