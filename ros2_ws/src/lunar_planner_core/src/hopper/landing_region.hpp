#pragma once

#include <stop_token>

#include "hopper/hopper_types.hpp"
#include "lunar_planner_core/types/goal.hpp"
#include "lunar_planner_core/types/planner_config.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"

namespace lunar::planning::hopper {

[[nodiscard]] LandingRegionResult CertifyExactLandingRegion(
    const shared::MapSnapshot& map,
    const GoalRegion& goal,
    const HopperCapability& capability,
    const MapSafetyConfig& map_safety,
    std::stop_token stop_token);

[[nodiscard]] LandingRegionResult CertifyLandingRegion(
    const shared::SafeProjection& projection,
    const GoalRegion& goal,
    const HopperCapability& capability,
    const MapSafetyConfig& map_safety,
    std::stop_token stop_token);

[[nodiscard]] LandingRegionResult CertifyHoldingRegion(
    const shared::SafeProjection& projection,
    Vec3 holding_position_m,
    const HopperCapability& capability,
    const MapSafetyConfig& map_safety,
    std::stop_token stop_token);

}  // namespace lunar::planning::hopper
