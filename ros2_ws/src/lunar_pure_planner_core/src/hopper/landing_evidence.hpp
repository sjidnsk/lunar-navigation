#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "hopper/landing_region.hpp"
#include "lunar_pure_planner_core/reachability_projection.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::hopper {

struct ExternalLandingBuildResult final {
  std::optional<std::vector<std::optional<CertifiedLandingRegion>>> landings;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return landings.has_value() && reason_code.empty();
  }
};

[[nodiscard]] ExternalLandingBuildResult BuildExternalLandings(
    const shared::MapSnapshot& global_map,
    const HopperLandingEvidenceGrid& evidence);

}  // namespace lunar::pure_planning::hopper
