#include "lunar_observed_map/obstacle_classifier.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "lunar_observed_map/sparse_observed_map.hpp"

namespace lunar::observed_map {
namespace {

double Median(std::vector<double> values) {
  const std::size_t middle = values.size() / 2U;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  if (values.size() % 2U != 0U) {
    return values[middle];
  }
  const double upper = values[middle];
  const double lower = *std::max_element(
      values.begin(), values.begin() + middle);
  return (lower + upper) * 0.5;
}

}  // namespace

ObstacleClassifier::ObstacleClassifier(ObstacleClassifierConfig config)
    : config_(config) {
  if (!std::isfinite(config_.maximum_local_obstacle_relief_m) ||
      config_.maximum_local_obstacle_relief_m <= 0.0 ||
      !std::isfinite(config_.resolution_m) || config_.resolution_m <= 0.0) {
    throw std::invalid_argument("OBSTACLE_CLASSIFIER_CONFIG_INVALID");
  }
}

CellClassification ObstacleClassifier::Classify(
    const CellNeighborhood& neighborhood) const {
  const std::optional<CellEvidence>& center =
      neighborhood[kNeighborhoodCenter];
  CellClassification result;
  if (center.has_value()) {
    result.forbidden = center->forbidden;
  }
  if (!center.has_value() || !center->valid ||
      !std::isfinite(center->elevation_m)) {
    return result;
  }

  std::vector<double> support_samples;
  support_samples.reserve(neighborhood.size());
  double maximum_discontinuity = 0.0;
  double maximum_variance = center->elevation_variance();
  for (const auto& evidence : neighborhood) {
    if (!evidence.has_value() || !evidence->valid ||
        !std::isfinite(evidence->elevation_m)) {
      continue;
    }
    support_samples.push_back(evidence->elevation_m);
    maximum_discontinuity = std::max(
        maximum_discontinuity,
        std::abs(center->elevation_m - evidence->elevation_m));
    maximum_variance = std::max(
        maximum_variance, evidence->elevation_variance());
  }

  result.support_elevation_m = Median(std::move(support_samples));
  const double positive_residual =
      center->elevation_m - result.support_elevation_m;
  result.obstacle =
      positive_residual > config_.maximum_local_obstacle_relief_m ||
      maximum_discontinuity > config_.maximum_local_obstacle_relief_m;
  result.occupancy =
      result.obstacle ? Occupancy::kOccupied : Occupancy::kFree;
  if (result.obstacle) {
    result.obstacle_height_m = std::max(
        {config_.resolution_m, positive_residual, maximum_discontinuity});
    result.obstacle_variance = maximum_variance;
  }
  return result;
}

CellClassification ObstacleClassifier::Classify(
    const SparseObservedMap& map, const GridCellIndex cell) const {
  CellNeighborhood neighborhood;
  std::size_t offset = 0U;
  for (std::int64_t dy = -1; dy <= 1; ++dy) {
    for (std::int64_t dx = -1; dx <= 1; ++dx) {
      if ((dx < 0 && cell.x == std::numeric_limits<std::int64_t>::min()) ||
          (dx > 0 && cell.x == std::numeric_limits<std::int64_t>::max()) ||
          (dy < 0 && cell.y == std::numeric_limits<std::int64_t>::min()) ||
          (dy > 0 && cell.y == std::numeric_limits<std::int64_t>::max())) {
        ++offset;
        continue;
      }
      const CellEvidence* evidence = map.Find({cell.x + dx, cell.y + dy});
      if (evidence != nullptr) {
        neighborhood[offset] = *evidence;
      }
      ++offset;
    }
  }
  return Classify(neighborhood);
}

}  // namespace lunar::observed_map
