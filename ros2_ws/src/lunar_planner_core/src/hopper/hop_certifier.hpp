#pragma once

#include "hopper/hopper_types.hpp"
#include "lunar_planner_core/types/planner_io.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::planning::hopper {

[[nodiscard]] HopCertificationResult CertifyFirstHop(
    const PlannerInput& input,
    const HopperState& state,
    const HopperCapability& capability,
    const shared::MapSnapshot& map,
    const CertifiedLandingRegion& source_region,
    const CertifiedLandingRegion& target_region);

}  // namespace lunar::planning::hopper
