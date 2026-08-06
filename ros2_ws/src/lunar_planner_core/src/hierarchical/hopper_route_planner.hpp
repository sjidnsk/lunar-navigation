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
  hopper::CertifiedLandingRegion landing_region;
  std::vector<Vec3> promotion_region_map;
  double flight_tube_radius_m{};
  double tube_expansion_margin_m{};
};

struct HopperRoutePlanResult final {
  PlanningOutcome outcome{PlanningOutcome::kInvalidRequest};
  std::optional<GlobalRoute> route;
  std::vector<NominalHopEdge> nominal_hops;
  std::vector<CertifiedHopPreview> certified_hops;
  std::optional<std::size_t> global_level;
  double maximum_horizontal_reach_m{};
  std::size_t graph_nodes{};
  std::size_t graph_edges{};
  std::size_t evaluated_edge_pairs{};
  std::size_t coarse_edges_rejected{};
  std::size_t full_edges_certified{};
  std::size_t full_edges_invalidated{};
  std::size_t edge_certificate_cache_hits{};
  std::size_t route_hops{};
  std::uint64_t expanded_nodes{};
  std::size_t open_peak{};
  std::size_t safe_landing_nodes{};
  std::chrono::nanoseconds landing_field_elapsed{};
  std::chrono::nanoseconds spatial_index_elapsed{};
  std::chrono::nanoseconds ballistic_solve_elapsed{};
  std::chrono::nanoseconds flight_tube_certification_elapsed{};
  std::chrono::nanoseconds elapsed{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return outcome == PlanningOutcome::kNewReferenceAvailable &&
           route.has_value() && route_hops > 0U &&
           nominal_hops.size() == route_hops &&
           certified_hops.size() == route_hops &&
           reason_code == "HOPPER_GLOBAL_ROUTE_AVAILABLE";
  }
};

[[nodiscard]] double
ConservativeMaximumHorizontalReach(const HopperCapability &capability) noexcept;

[[nodiscard]] HopperRoutePlanResult
PlanHopperGlobalRoute(const PlannerInput &input);

} // namespace lunar::planning::hierarchical
