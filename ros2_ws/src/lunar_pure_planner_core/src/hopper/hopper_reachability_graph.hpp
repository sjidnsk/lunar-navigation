#pragma once

#include <memory>
#include <optional>

#include "lunar_pure_planner_core/reachability_projection.hpp"
#include "lunar_pure_planner_core/types/planner_io.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/primitive_reachability_graph.hpp"
#include "shared/safe_projection.hpp"

namespace lunar::pure_planning::hopper {

[[nodiscard]] shared::PrimitiveGraphBuildResult BuildHopperPrimitiveGraph(
    const PlannerInput& input,
    const std::shared_ptr<const shared::MapSnapshot>& global_map,
    const std::shared_ptr<const shared::MapSnapshot>& local_map,
    const shared::SafeProjection& safe,
    std::optional<double> maximum_edge_distance_m,
    const HopperLandingEvidenceGrid* landing_evidence = nullptr);

}  // namespace lunar::pure_planning::hopper
