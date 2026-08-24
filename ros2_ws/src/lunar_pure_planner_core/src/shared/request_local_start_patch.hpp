#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "lunar_pure_planner_core/search_control.hpp"
#include "lunar_pure_planner_core/types/planning_request.hpp"

namespace lunar::pure_planning::shared {

// Describes a request-owned overlay. Offsets may name NaN cells only; applying
// the overlay never mutates the map supplied by Task 3 or any shared cache.
struct RequestLocalStartPatch final {
  std::uint64_t identity{};
  std::vector<std::size_t> occupancy_offsets;
  std::vector<std::size_t> elevation_offsets;
  double elevation_x_slope{};
  double elevation_y_slope{};
  double elevation_offset{};

  [[nodiscard]] bool required() const noexcept {
    return !occupancy_offsets.empty() || !elevation_offsets.empty();
  }
};

struct RequestLocalStartPatchResult final {
  std::optional<RequestLocalStartPatch> patch;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return patch.has_value() && reason_code.empty();
  }
};

struct RequestLocalPatchedMapResult final {
  std::optional<GridMap> map;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return map.has_value() && reason_code.empty();
  }
};

[[nodiscard]] RequestLocalStartPatchResult AnalyzeWheelStartPatch(
    const GridMap& map, const WheeledState& state,
    const WheeledCapability& capability, double occupancy_threshold,
    const SearchControl& control);

// Applies the already analyzed overlay to a private map copy. Any non-NaN
// occupancy value, including an obstacle or infinity, remains authoritative.
[[nodiscard]] std::string ApplyWheelStartPatch(
    GridMap& map, const RequestLocalStartPatch& patch,
    const SearchControl& control);

[[nodiscard]] RequestLocalPatchedMapResult BuildWheelStartPatchedMap(
    const GridMap& map, const RequestLocalStartPatch& patch,
    const SearchControl& control);

}  // namespace lunar::pure_planning::shared
