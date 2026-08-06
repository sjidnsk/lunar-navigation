#pragma once

#include <stop_token>

#include "hopper/hopper_types.hpp"
#include "lunar_planner_core/types/planner_config.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::planning::hopper {

[[nodiscard]] FlightTubeCertificationResult CertifyFlightTube(
    const BallisticArc& arc,
    const shared::MapSnapshot& map,
    const HopperCapability& capability,
    const MapSafetyConfig& map_safety,
    std::stop_token stop_token,
    double additional_radius_m = 0.0);

[[nodiscard]] FlightTubeCertificationResult CertifyFlightTube(
    const BallisticArc& arc,
    const shared::MapSnapshot& map,
    const CertifiedLandingRegion& source_region,
    const CertifiedLandingRegion& target_region,
    const HopperCapability& capability,
    const PlannerConfig& config,
    std::stop_token stop_token,
    double additional_radius_m = 0.0);

}  // namespace lunar::planning::hopper
