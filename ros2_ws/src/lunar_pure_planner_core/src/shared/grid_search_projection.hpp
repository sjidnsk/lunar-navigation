#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::shared {

struct GridSearchProjectionView final {
  const MapSnapshot* map{};
  std::span<const std::uint8_t> hard_feasible;
  std::span<const float> clearance_m;
  std::span<const float> slope_rad;
  std::span<const float> roughness_m;
  std::span<const float> traversal_cost;

  [[nodiscard]] bool Valid() const noexcept {
    return map != nullptr && hard_feasible.size() == map->cell_count() &&
           clearance_m.size() == map->cell_count() &&
           slope_rad.size() == map->cell_count() &&
           roughness_m.size() == map->cell_count() &&
           traversal_cost.size() == map->cell_count();
  }

  [[nodiscard]] bool HardFeasible(const GridCell cell) const noexcept {
    return ValueAt(cell, hard_feasible, std::uint8_t{0}) != 0U;
  }

  [[nodiscard]] float ClearanceMeters(const GridCell cell) const noexcept {
    return ValueAt(cell, clearance_m, 0.0F);
  }

  [[nodiscard]] float SlopeRadians(const GridCell cell) const noexcept {
    return ValueAt(cell, slope_rad, std::numeric_limits<float>::infinity());
  }

  [[nodiscard]] float RoughnessMeters(const GridCell cell) const noexcept {
    return ValueAt(cell, roughness_m,
                   std::numeric_limits<float>::infinity());
  }

  [[nodiscard]] float TraversalCost(const GridCell cell) const noexcept {
    return ValueAt(cell, traversal_cost,
                   std::numeric_limits<float>::infinity());
  }

 private:
  template <class T>
  [[nodiscard]] T ValueAt(const GridCell cell, const std::span<const T> values,
                          const T fallback) const noexcept {
    if (map == nullptr || !map->InBounds(cell)) {
      return fallback;
    }
    const std::size_t index = map->Index(cell);
    return index < values.size() ? values[index] : fallback;
  }
};

}  // namespace lunar::pure_planning::shared
