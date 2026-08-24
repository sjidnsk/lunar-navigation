#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "lunar_pure_planner_core/search_control.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::shared {

struct LocalTerrainProjection final {
  std::shared_ptr<const MapSnapshot> map;
  std::vector<std::uint8_t> free_with_height;
  std::vector<float> clearance_m;
  std::vector<float> slope_rad;
  std::vector<float> roughness_m;
};

struct LocalTerrainProjectionResult final {
  std::optional<LocalTerrainProjection> value;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && reason_code.empty();
  }
};

[[nodiscard]] LocalTerrainProjectionResult BuildLocalTerrainProjection(
    std::shared_ptr<const MapSnapshot> map, float occupancy_threshold = 0.5F,
    SearchControl control = {});

}  // namespace lunar::pure_planning::shared
