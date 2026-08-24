#include "shared/local_terrain_projection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "shared/cell_area_distance_transform.hpp"
#include "shared/controlled_work.hpp"

namespace lunar::pure_planning::shared {
namespace {

[[nodiscard]] bool IsFreeOccupancy(const float value,
                                   const float threshold) noexcept {
  return std::isfinite(value) && value >= 0.0F && value < threshold;
}

[[nodiscard]] bool IsOccupiedInflationSource(
    const float value, const float threshold) noexcept {
  return std::isfinite(value) && value >= 0.0F && value <= 1.0F &&
         value >= threshold;
}

[[nodiscard]] bool IsNarrowBandSource(
    const float value, const float threshold) noexcept {
  return !IsFreeOccupancy(value, threshold);
}

[[nodiscard]] GridCell CellFromIndex(const MapSnapshot& map,
                                     const std::size_t index) noexcept {
  return GridCell{.x = static_cast<std::int32_t>(index % map.width()),
                  .y = static_cast<std::int32_t>(index / map.width())};
}

[[nodiscard]] bool SolvePlane(
    const std::array<std::array<double, 3>, 9>& rows,
    const std::array<double, 9>& heights, const std::size_t sample_count,
    std::array<double, 3>* coefficients) noexcept {
  if (sample_count < 3U || sample_count > rows.size() ||
      coefficients == nullptr) {
    return false;
  }
  double mean_height = 0.0;
  for (std::size_t sample = 0U; sample < sample_count; ++sample) {
    mean_height += heights[sample];
  }
  mean_height /= static_cast<double>(sample_count);
  if (!std::isfinite(mean_height)) {
    return false;
  }
  std::array<std::array<double, 4>, 3> normal{};
  for (std::size_t sample = 0U; sample < sample_count; ++sample) {
    for (std::size_t row = 0U; row < 3U; ++row) {
      normal[row][3] += rows[sample][row] *
                        (heights[sample] - mean_height);
      for (std::size_t column = 0U; column < 3U; ++column) {
        normal[row][column] += rows[sample][row] * rows[sample][column];
      }
    }
  }
  double largest = 0.0;
  for (const auto& row : normal) {
    for (std::size_t column = 0U; column < 3U; ++column) {
      largest = std::max(largest, std::abs(row[column]));
    }
  }
  if (!std::isfinite(largest) || largest == 0.0) {
    return false;
  }
  const double tolerance = largest * 1.0e-12;
  for (std::size_t pivot = 0U; pivot < 3U; ++pivot) {
    std::size_t selected = pivot;
    for (std::size_t row = pivot + 1U; row < 3U; ++row) {
      if (std::abs(normal[row][pivot]) > std::abs(normal[selected][pivot])) {
        selected = row;
      }
    }
    if (!std::isfinite(normal[selected][pivot]) ||
        std::abs(normal[selected][pivot]) <= tolerance) {
      return false;
    }
    std::swap(normal[pivot], normal[selected]);
    const double scale = normal[pivot][pivot];
    for (std::size_t column = pivot; column < 4U; ++column) {
      normal[pivot][column] /= scale;
    }
    for (std::size_t row = 0U; row < 3U; ++row) {
      if (row == pivot) {
        continue;
      }
      const double factor = normal[row][pivot];
      for (std::size_t column = pivot; column < 4U; ++column) {
        normal[row][column] -= factor * normal[pivot][column];
      }
    }
  }
  for (std::size_t row = 0U; row < 3U; ++row) {
    (*coefficients)[row] = normal[row][3];
    if (!std::isfinite((*coefficients)[row])) {
      return false;
    }
  }
  (*coefficients)[2] += mean_height;
  if (!std::isfinite((*coefficients)[2])) {
    return false;
  }
  return true;
}

[[nodiscard]] float EstimateSlope(const MapSnapshot& map,
                                  const std::span<const float> elevation,
                                  const GridCell cell) noexcept {
  const std::size_t index = map.Index(cell);
  if (index == map.cell_count() || !std::isfinite(elevation[index])) {
    return std::numeric_limits<float>::infinity();
  }
  const auto derivative = [&](const std::int32_t dx, const std::int32_t dy) {
    const GridCell minus{.x = cell.x - dx, .y = cell.y - dy};
    const GridCell plus{.x = cell.x + dx, .y = cell.y + dy};
    const bool has_minus = map.InBounds(minus) &&
                           std::isfinite(elevation[map.Index(minus)]);
    const bool has_plus = map.InBounds(plus) &&
                          std::isfinite(elevation[map.Index(plus)]);
    const double current = elevation[index];
    if (has_minus && has_plus) {
      return (static_cast<double>(elevation[map.Index(plus)]) -
              static_cast<double>(elevation[map.Index(minus)])) /
             (2.0 * map.resolution_m());
    }
    if (has_plus) {
      return (static_cast<double>(elevation[map.Index(plus)]) - current) /
             map.resolution_m();
    }
    if (has_minus) {
      return (current - static_cast<double>(elevation[map.Index(minus)])) /
             map.resolution_m();
    }
    return std::numeric_limits<double>::quiet_NaN();
  };
  const double dx = derivative(1, 0);
  const double dy = derivative(0, 1);
  if (!std::isfinite(dx) || !std::isfinite(dy)) {
    return std::numeric_limits<float>::infinity();
  }
  return static_cast<float>(std::atan(std::hypot(dx, dy)));
}

[[nodiscard]] float EstimateRoughness(const MapSnapshot& map,
                                      const std::span<const float> elevation,
                                      const GridCell center) noexcept {
  std::array<std::array<double, 3>, 9> rows{};
  std::array<double, 9> heights{};
  std::size_t sample_count = 0U;
  for (std::int32_t dy = -1; dy <= 1; ++dy) {
    for (std::int32_t dx = -1; dx <= 1; ++dx) {
      const GridCell cell{.x = center.x + dx, .y = center.y + dy};
      if (!map.InBounds(cell)) {
        continue;
      }
      const float height = elevation[map.Index(cell)];
      if (!std::isfinite(height)) {
        continue;
      }
      rows[sample_count] = {static_cast<double>(dx) * map.resolution_m(),
                            static_cast<double>(dy) * map.resolution_m(), 1.0};
      heights[sample_count] = height;
      ++sample_count;
    }
  }
  std::array<double, 3> coefficients{};
  if (!SolvePlane(rows, heights, sample_count, &coefficients)) {
    return std::numeric_limits<float>::infinity();
  }
  double squared_error = 0.0;
  for (std::size_t sample = 0U; sample < sample_count; ++sample) {
    const double residual = heights[sample] -
                            (coefficients[0] * rows[sample][0] +
                             coefficients[1] * rows[sample][1] +
                             coefficients[2]);
    squared_error += residual * residual;
  }
  const double roughness =
      std::sqrt(squared_error / static_cast<double>(sample_count));
  return std::isfinite(roughness)
             ? static_cast<float>(roughness)
             : std::numeric_limits<float>::infinity();
}

}  // namespace

LocalTerrainProjectionResult BuildLocalTerrainProjection(
    std::shared_ptr<const MapSnapshot> map, const float occupancy_threshold,
    SearchControl control) {
  if (map == nullptr || !std::isfinite(occupancy_threshold) ||
      occupancy_threshold <= 0.0F || occupancy_threshold > 1.0F) {
    return {.reason_code = "LOCAL_TERRAIN_PROJECTION_INVALID"};
  }
  const auto occupancy = map->FloatLayer("occupancy");
  const auto elevation = map->FloatLayer("elevation");
  if (occupancy.size() != map->cell_count() ||
      elevation.size() != map->cell_count()) {
    return {.reason_code = "LOCAL_TERRAIN_PROJECTION_INVALID"};
  }
  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }

  LocalTerrainProjection projection;
  projection.map = std::move(map);
  const std::size_t count = projection.map->cell_count();
  const float float_infinity = std::numeric_limits<float>::infinity();
  if (const auto stopped = ControlledFill(
          &projection.free_with_height, count, std::uint8_t{0U}, control);
      stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  if (const auto stopped = ControlledFill(
          &projection.occupied, count, std::uint8_t{0U}, control);
      stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  if (const auto stopped = ControlledFill(
          &projection.slope_rad, count, float_infinity, control);
      stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  if (const auto stopped = ControlledFill(
          &projection.roughness_m, count, float_infinity, control);
      stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }

  std::vector<std::uint8_t> narrow_band_mask(count, 0U);
  for (std::size_t index = 0U; index < count; ++index) {
    if (ControlCheckDue(index)) {
      if (const auto stopped = StopReason(control); stopped.has_value()) {
        return {.reason_code = std::string{*stopped}};
      }
    }
    const bool free = IsFreeOccupancy(occupancy[index], occupancy_threshold);
    projection.free_with_height[index] = static_cast<std::uint8_t>(
        free && std::isfinite(elevation[index]));
    projection.occupied[index] = static_cast<std::uint8_t>(
        IsOccupiedInflationSource(occupancy[index], occupancy_threshold));
    narrow_band_mask[index] = static_cast<std::uint8_t>(
        IsNarrowBandSource(occupancy[index], occupancy_threshold));
  }

  auto clearance = BuildCellAreaClearance(
      projection.map->width(), projection.map->height(),
      projection.map->resolution_m(), projection.occupied, control);
  if (!clearance.ok()) {
    return {.reason_code = std::move(clearance.reason_code)};
  }
  projection.clearance_m = std::move(clearance.clearance_m);

  auto narrow_band_distance = BuildCellAreaClearance(
      projection.map->width(), projection.map->height(),
      projection.map->resolution_m(), narrow_band_mask, control);
  if (!narrow_band_distance.ok()) {
    return {.reason_code = std::move(narrow_band_distance.reason_code)};
  }
  projection.narrow_band_distance_m =
      std::move(narrow_band_distance.clearance_m);

  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  for (std::size_t index = 0U; index < count; ++index) {
    if (ControlCheckDue(index)) {
      if (const auto stopped = StopReason(control); stopped.has_value()) {
        return {.reason_code = std::string{*stopped}};
      }
    }
    const GridCell cell = CellFromIndex(*projection.map, index);
    projection.slope_rad[index] = EstimateSlope(*projection.map, elevation, cell);
    projection.roughness_m[index] =
        EstimateRoughness(*projection.map, elevation, cell);
  }
  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  return {.value = std::move(projection), .reason_code = {}};
}

}  // namespace lunar::pure_planning::shared
