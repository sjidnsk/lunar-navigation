#include "hopper/landing_evidence.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <string_view>
#include <utility>
#include <vector>

namespace lunar::pure_planning::hopper {
namespace {

constexpr std::string_view kHopperLandingEvidenceAlgorithm =
    "cpp-hopper-detail-landing-regions/v1";

[[nodiscard]] bool Finite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
      std::isfinite(value.z);
}

}  // namespace

ExternalLandingBuildResult BuildExternalLandings(
    const shared::MapSnapshot& global_map,
    const HopperLandingEvidenceGrid& evidence) {
  if (evidence.width != global_map.width() ||
      evidence.height != global_map.height() ||
      evidence.landings.size() != global_map.cell_count()) {
    return {.reason_code = "HOPPER_LANDING_EVIDENCE_GEOMETRY_INVALID"};
  }
  if (evidence.algorithm_id != kHopperLandingEvidenceAlgorithm) {
    return {.reason_code = "HOPPER_LANDING_EVIDENCE_ALGORITHM_INVALID"};
  }
  std::vector<std::optional<CertifiedLandingRegion>> output(
      evidence.landings.size());
  for (std::size_t index = 0U; index < evidence.landings.size(); ++index) {
    const HopperLandingEvidence& landing = evidence.landings[index];
    if (landing.certified > 1U) {
      return {.reason_code = "HOPPER_LANDING_EVIDENCE_VALUE_INVALID"};
    }
    if (landing.certified == 0U) {
      continue;
    }
    if (!Finite(landing.aim_position_on_surface_m) ||
        !std::isfinite(landing.area_m2) || landing.area_m2 <= 0.0 ||
        !std::ranges::all_of(landing.boundary_m, Finite)) {
      return {.reason_code = "HOPPER_LANDING_EVIDENCE_VALUE_INVALID"};
    }
    const auto cell = global_map.PositionToCell(Vec2{
        landing.aim_position_on_surface_m.x,
        landing.aim_position_on_surface_m.y,
    });
    const shared::GridCell expected{
        .x = static_cast<std::int32_t>(index % global_map.width()),
        .y = static_cast<std::int32_t>(index / global_map.width()),
    };
    if (!cell.has_value() || cell->x != expected.x ||
        cell->y != expected.y) {
      return {.reason_code = "HOPPER_LANDING_EVIDENCE_CELL_INVALID"};
    }
    output[index] = CertifiedLandingRegion{
        .seed_cell = expected,
        .aim_position_on_surface_m = landing.aim_position_on_surface_m,
        .boundary_m = std::vector<Vec3>(
            landing.boundary_m.begin(), landing.boundary_m.end()),
        .area_m2 = landing.area_m2,
    };
  }
  return {.landings = std::move(output)};
}

}  // namespace lunar::pure_planning::hopper
