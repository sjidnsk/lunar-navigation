#include "luna_t3_map_adapter/conservative_aggregation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace luna::task3 {
namespace {

constexpr double kResolutionTolerance = 1.0e-9;

[[nodiscard]] bool IsFiniteCell(const FineCell& cell) noexcept {
  return std::isfinite(cell.elevation_m) &&
         std::isfinite(cell.obstacle_height_m) &&
         std::isfinite(cell.observation_age_s) &&
         std::isfinite(cell.observation_quality) &&
         std::isfinite(cell.elevation_variance_m2) &&
         std::isfinite(cell.obstacle_variance_m2) &&
         cell.obstacle_height_m >= 0.0 && cell.observation_age_s >= 0.0 &&
         cell.observation_quality >= 0.0 && cell.observation_quality <= 1.0 &&
         cell.elevation_variance_m2 >= 0.0 && cell.obstacle_variance_m2 >= 0.0;
}

[[nodiscard]] bool ValidSourceShape(const FineGrid& source) noexcept {
  if (!std::isfinite(source.resolution_m) || source.resolution_m <= 0.0 ||
      source.width == 0U || source.height == 0U ||
      source.width > std::numeric_limits<std::size_t>::max() / source.height ||
      source.cells.size() != source.width * source.height) {
    return false;
  }
  return std::ranges::all_of(source.cells, IsFiniteCell);
}

[[nodiscard]] std::size_t CeilRatio(const std::size_t source_cells,
                                    const std::size_t factor) noexcept {
  return source_cells / factor + (source_cells % factor == 0U ? 0U : 1U);
}

[[nodiscard]] FineCell AggregateParent(const FineGrid& source,
                                        const std::size_t start_x,
                                        const std::size_t start_y,
                                        const std::size_t factor) noexcept {
  bool complete = true;
  bool all_valid = true;
  bool obstacle = false;
  bool forbidden = false;
  double max_obstacle_height = 0.0;
  double max_observation_age = 0.0;
  double min_observation_quality = 1.0;
  double max_elevation_variance = 0.0;
  double max_obstacle_variance = 0.0;
  std::size_t min_observation_count = std::numeric_limits<std::size_t>::max();
  std::vector<double> valid_elevations;
  valid_elevations.reserve(factor * factor);

  for (std::size_t child_y = 0U; child_y < factor; ++child_y) {
    for (std::size_t child_x = 0U; child_x < factor; ++child_x) {
      const std::size_t x = start_x + child_x;
      const std::size_t y = start_y + child_y;
      if (x >= source.width || y >= source.height) {
        complete = false;
        continue;
      }
      const FineCell& child = source.At(x, y);
      all_valid = all_valid && child.valid_mask;
      obstacle = obstacle || child.obstacle;
      forbidden = forbidden || child.forbidden;
      max_obstacle_height = std::max(max_obstacle_height, child.obstacle_height_m);
      max_observation_age = std::max(max_observation_age, child.observation_age_s);
      min_observation_quality =
          std::min(min_observation_quality, child.observation_quality);
      max_elevation_variance =
          std::max(max_elevation_variance, child.elevation_variance_m2);
      max_obstacle_variance =
          std::max(max_obstacle_variance, child.obstacle_variance_m2);
      min_observation_count =
          std::min(min_observation_count, child.observation_count);
      if (child.valid_mask) {
        valid_elevations.push_back(child.elevation_m);
      }
    }
  }

  const bool valid = complete && all_valid;
  double elevation = 0.0;
  double elevation_population_variance = 0.0;
  if (valid) {
    for (const double value : valid_elevations) {
      elevation += value;
    }
    elevation /= static_cast<double>(valid_elevations.size());
    for (const double value : valid_elevations) {
      const double delta = value - elevation;
      elevation_population_variance += delta * delta;
    }
    elevation_population_variance /= static_cast<double>(valid_elevations.size());
  }
  return FineCell{
      .elevation_m = elevation,
      .valid_mask = valid,
      .obstacle = obstacle,
      .obstacle_height_m = max_obstacle_height,
      .observation_age_s = max_observation_age,
      .observation_quality = min_observation_quality,
      .elevation_variance_m2 = max_elevation_variance + elevation_population_variance,
      .obstacle_variance_m2 = max_obstacle_variance,
      .observation_count =
          min_observation_count == std::numeric_limits<std::size_t>::max()
              ? 0U
              : min_observation_count,
      .forbidden = forbidden || !valid,
  };
}

}  // namespace

Result<FineGrid> AggregateConservatively(const FineGrid& source,
                                         const SelectedGlobalLevel& target) {
  if (!ValidSourceShape(source)) {
    return Result<FineGrid>{.value = std::nullopt,
                            .reason_code = "TASK3_FINE_GRID_SHAPE_INVALID"};
  }
  if (!std::isfinite(target.resolution_m) || target.resolution_m <= 0.0 ||
      target.width == 0U || target.height == 0U ||
      target.resolution_m + kResolutionTolerance < source.resolution_m) {
    return Result<FineGrid>{.value = std::nullopt,
                            .reason_code = "TASK3_AGGREGATION_TARGET_INVALID"};
  }
  const double ratio = target.resolution_m / source.resolution_m;
  const std::size_t factor = static_cast<std::size_t>(std::llround(ratio));
  if (factor == 0U || std::abs(ratio - static_cast<double>(factor)) >
                         kResolutionTolerance * std::max(1.0, ratio) ||
      CeilRatio(source.width, factor) != target.width ||
      CeilRatio(source.height, factor) != target.height) {
    return Result<FineGrid>{.value = std::nullopt,
                            .reason_code = "TASK3_AGGREGATION_TARGET_INVALID"};
  }

  FineGrid aggregated{
      .resolution_m = target.resolution_m,
      .width = target.width,
      .height = target.height,
      .cells = {},
  };
  aggregated.cells.reserve(target.width * target.height);
  for (std::size_t y = 0U; y < target.height; ++y) {
    for (std::size_t x = 0U; x < target.width; ++x) {
      aggregated.cells.push_back(AggregateParent(
          source, x * factor, y * factor, factor));
    }
  }
  return Result<FineGrid>{.value = std::move(aggregated), .reason_code = {}};
}

}  // namespace luna::task3
