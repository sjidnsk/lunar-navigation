#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "hierarchical/global_route.hpp"
#include "hierarchical/landing_support_field.hpp"
#include "hopper/hopper_types.hpp"
#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning::hierarchical {

struct NominalHopEdge final {
  LandingNodeId source_id{};
  LandingNodeId target_id{};
  double cost{};
  hopper::BallisticArc arc;
};

struct HopperRoutePlanResult final {
  PlanningOutcome outcome{PlanningOutcome::kInvalidRequest};
  std::optional<GlobalRoute> route;
  std::vector<NominalHopEdge> nominal_hops;
  std::optional<std::size_t> global_level;
  double maximum_horizontal_reach_m{};
  std::size_t graph_nodes{};
  std::size_t graph_edges{};
  std::size_t evaluated_edge_pairs{};
  std::size_t route_hops{};
  std::uint64_t expanded_nodes{};
  std::chrono::nanoseconds landing_field_elapsed{};
  std::chrono::nanoseconds elapsed{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return outcome == PlanningOutcome::kNewReferenceAvailable &&
           route.has_value() && route_hops > 0U &&
           nominal_hops.size() == route_hops &&
           reason_code == "HOPPER_GLOBAL_ROUTE_AVAILABLE";
  }
};

[[nodiscard]] double
ConservativeMaximumHorizontalReach(const HopperCapability &capability) noexcept;

[[nodiscard]] HopperRoutePlanResult
PlanHopperGlobalRoute(const PlannerInput &input);

} // namespace lunar::planning::hierarchical
