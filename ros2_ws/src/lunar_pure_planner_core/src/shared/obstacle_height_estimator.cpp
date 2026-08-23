#include "shared/obstacle_height_estimator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace lunar::pure_planning::shared {
namespace {

[[nodiscard]] bool SolveCenteredPlane(
    const std::vector<std::array<double, 3>>& rows,
    const std::vector<double>& heights,
    std::array<double, 3>* coefficients) noexcept {
  if (rows.size() != heights.size() || rows.size() < 6U ||
      coefficients == nullptr) {
    return false;
  }
  double mean_height = 0.0;
  for (const double height : heights) {
    mean_height += height;
  }
  mean_height /= static_cast<double>(heights.size());
  if (!std::isfinite(mean_height)) {
    return false;
  }
  std::array<std::array<double, 4>, 3> normal{};
  for (std::size_t sample = 0U; sample < rows.size(); ++sample) {
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

}  // namespace

ObstacleHeight EstimateObstacleHeight(const MapSnapshot& map,
                                      const GridCell occupied) {
  if (!map.InBounds(occupied)) {
    return {};
  }
  const std::span<const float> occupancy = map.FloatLayer("occupancy");
  const std::span<const float> elevation = map.FloatLayer("elevation");
  if (occupancy.size() != map.cell_count() || elevation.size() != map.cell_count()) {
    return {};
  }
  const std::size_t occupied_index = map.Index(occupied);
  const float occupied_occupancy = occupancy[occupied_index];
  const float occupied_elevation = elevation[occupied_index];
  if (!std::isfinite(occupied_occupancy) || occupied_occupancy < 0.5F ||
      occupied_occupancy > 1.0F || !std::isfinite(occupied_elevation)) {
    return {};
  }

  std::vector<std::array<double, 3>> rows;
  std::vector<double> heights;
  rows.reserve(48U);
  heights.reserve(48U);
  for (std::int32_t dy = -3; dy <= 3; ++dy) {
    for (std::int32_t dx = -3; dx <= 3; ++dx) {
      const GridCell sample{.x = occupied.x + dx, .y = occupied.y + dy};
      if (!map.InBounds(sample)) {
        continue;
      }
      const std::size_t sample_index = map.Index(sample);
      if (!std::isfinite(occupancy[sample_index]) ||
          occupancy[sample_index] < 0.0F || occupancy[sample_index] >= 0.5F ||
          !std::isfinite(elevation[sample_index])) {
        continue;
      }
      rows.push_back({static_cast<double>(dx) * map.resolution_m(),
                      static_cast<double>(dy) * map.resolution_m(), 1.0});
      heights.push_back(elevation[sample_index]);
    }
  }
  std::array<double, 3> plane{};
  if (!SolveCenteredPlane(rows, heights, &plane)) {
    return {};
  }
  const double height = std::max(
      0.0, static_cast<double>(occupied_elevation) - plane[2]);
  if (!std::isfinite(height) || height <= 0.0) {
    return {};
  }
  return {.flyover_allowed = true, .height_m = height};
}

}  // namespace lunar::pure_planning::shared
