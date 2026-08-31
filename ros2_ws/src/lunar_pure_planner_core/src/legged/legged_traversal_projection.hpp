#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lunar_pure_planner_core/search_control.hpp"
#include "lunar_pure_planner_core/types/platform_capability.hpp"
#include "shared/local_terrain_projection.hpp"

namespace lunar::pure_planning::legged {

struct LeggedTraversalProjection final {
  std::shared_ptr<const shared::LocalTerrainProjection> terrain;
  std::vector<std::uint8_t> hard_feasible;
  std::vector<std::uint8_t> step_feasible;
  std::vector<float> slope_rad;
  std::vector<float> roughness_m;
  std::vector<float> clearance_m;
  std::vector<std::size_t> hard_infeasible_prefix_sum;

  [[nodiscard]] bool AllCellsTraversable(
      shared::GridCell minimum,
      shared::GridCell maximum) const noexcept;
};

struct LeggedTraversalProjectionBuildResult final {
  std::shared_ptr<const LeggedTraversalProjection> value;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return value != nullptr && reason_code.empty();
  }
};

[[nodiscard]] LeggedTraversalProjectionBuildResult
BuildLeggedTraversalProjection(
    std::shared_ptr<const shared::LocalTerrainProjection> terrain,
    const LeggedCapability& capability,
    const SearchControl& control = {});

}  // namespace lunar::pure_planning::legged
