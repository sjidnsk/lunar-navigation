#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "lunar_planner_core/types/planner_config.hpp"
#include "lunar_planner_core/types/world_snapshot.hpp"

namespace lunar::planning::hierarchical {

struct ExpectedMapLevelResult final {
  std::optional<std::size_t> level;
  std::size_t width{};
  std::size_t height{};
  double resolution_m{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return level.has_value() && reason_code.empty();
  }
};

struct MapLevelValidationResult final {
  std::optional<std::size_t> global_level;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return global_level.has_value() && reason_code.empty();
  }
};

[[nodiscard]] ExpectedMapLevelResult
ExpectedGlobalMapLevel(double size_x_m, double size_y_m,
                       const GlobalMapConfig &config) noexcept;

[[nodiscard]] MapLevelValidationResult
ValidateMapLevels(const WorldSnapshot &world,
                  const GlobalMapConfig &config) noexcept;

} // namespace lunar::planning::hierarchical
