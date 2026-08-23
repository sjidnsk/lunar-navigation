#include "hierarchical/map_level.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>

namespace lunar::pure_planning::hierarchical {
namespace {

constexpr double kRelativeTolerance = 1.0e-9;
constexpr std::array<std::size_t, 6U> kScaleFactors{1U, 2U, 4U, 8U, 16U,
                                                    20U};
constexpr std::size_t kSupportedMaximumLevel = kScaleFactors.size() - 1U;

[[nodiscard]] bool ValidConfig(const GlobalMapConfig &config) noexcept {
  return std::isfinite(config.base_resolution_m) &&
         config.base_resolution_m > 0.0 &&
         config.maximum_level == kSupportedMaximumLevel &&
         config.maximum_cells > 0U && config.maximum_axis_cells > 0U &&
         config.target_axis_cells > 0U;
}

[[nodiscard]] std::optional<std::size_t>
CeilCells(const double size_m, const double resolution_m) noexcept {
  if (!std::isfinite(size_m) || size_m <= 0.0 || !std::isfinite(resolution_m) ||
      resolution_m <= 0.0) {
    return std::nullopt;
  }
  const double quotient = size_m / resolution_m;
  const double adjusted =
      quotient - kRelativeTolerance * std::max(1.0, std::abs(quotient));
  const double cells = std::ceil(adjusted);
  if (!std::isfinite(cells) || cells < 1.0 ||
      cells > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(cells);
}

[[nodiscard]] bool ProductWithin(const std::size_t width,
                                 const std::size_t height,
                                 const std::size_t maximum) noexcept {
  return width > 0U && height > 0U && width <= maximum / height;
}

[[nodiscard]] bool Close(const double left, const double right) noexcept {
  return std::isfinite(left) && std::isfinite(right) &&
         std::abs(left - right) <=
             kRelativeTolerance *
                 std::max({1.0, std::abs(left), std::abs(right)});
}

} // namespace

ExpectedMapLevelResult
ExpectedGlobalMapLevel(const double size_x_m, const double size_y_m,
                       const GlobalMapConfig &config) noexcept {
  if (!ValidConfig(config) || !std::isfinite(size_x_m) ||
      !std::isfinite(size_y_m) || size_x_m <= 0.0 || size_y_m <= 0.0) {
    return ExpectedMapLevelResult{
        .reason_code = "GLOBAL_MAP_CONFIGURATION_INVALID",
    };
  }
  for (std::size_t level = 0U; level <= config.maximum_level; ++level) {
    const double resolution = config.base_resolution_m *
                              static_cast<double>(kScaleFactors[level]);
    const auto width = CeilCells(size_x_m, resolution);
    const auto height = CeilCells(size_y_m, resolution);
    if (!width || !height) {
      return ExpectedMapLevelResult{
          .reason_code = "GLOBAL_MAP_CONFIGURATION_INVALID",
      };
    }
    const std::size_t target_axis_cells =
        std::min(config.target_axis_cells, config.maximum_axis_cells);
    if (*width <= target_axis_cells &&
        *height <= target_axis_cells &&
        *width <= config.maximum_axis_cells &&
        *height <= config.maximum_axis_cells &&
        ProductWithin(*width, *height, config.maximum_cells)) {
      return ExpectedMapLevelResult{
          .level = level,
          .width = *width,
          .height = *height,
          .resolution_m = resolution,
          .reason_code = {},
      };
    }
  }
  return ExpectedMapLevelResult{
      .reason_code = "GLOBAL_MAP_SCALE_UNSUPPORTED",
  };
}

MapLevelValidationResult
ValidateMapLevels(const WorldSnapshot &world,
                  const GlobalMapConfig &config) noexcept {
  if (!ValidConfig(config)) {
    return MapLevelValidationResult{
        .reason_code = "GLOBAL_MAP_CONFIGURATION_INVALID",
    };
  }
  if (world.local_map.width == 0U || world.local_map.height == 0U ||
      !Close(world.local_map.resolution_m, config.base_resolution_m)) {
    return MapLevelValidationResult{
        .reason_code = "LOCAL_MAP_LEVEL_INVALID",
    };
  }
  if (world.global_map.width == 0U || world.global_map.height == 0U ||
      !std::isfinite(world.global_map.resolution_m) ||
      world.global_map.resolution_m <= 0.0) {
    return MapLevelValidationResult{
        .reason_code = "GLOBAL_MAP_LEVEL_INVALID",
    };
  }
  const double size_x = static_cast<double>(world.global_map.width) *
                        world.global_map.resolution_m;
  const double size_y = static_cast<double>(world.global_map.height) *
                        world.global_map.resolution_m;
  const ExpectedMapLevelResult expected =
      ExpectedGlobalMapLevel(size_x, size_y, config);
  if (!expected.ok()) {
    return MapLevelValidationResult{.reason_code = expected.reason_code};
  }
  if (!Close(world.global_map.resolution_m, expected.resolution_m) ||
      world.global_map.width != expected.width ||
      world.global_map.height != expected.height) {
    return MapLevelValidationResult{
        .reason_code = "GLOBAL_MAP_LEVEL_INVALID",
    };
  }
  return MapLevelValidationResult{
      .global_level = expected.level,
      .reason_code = {},
  };
}

} // namespace lunar::pure_planning::hierarchical
