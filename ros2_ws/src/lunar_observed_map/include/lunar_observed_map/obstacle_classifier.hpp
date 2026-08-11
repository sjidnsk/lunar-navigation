#pragma once

#include "lunar_observed_map/map_types.hpp"

namespace lunar::observed_map {

class SparseObservedMap;

struct ObstacleClassifierConfig final {
  double maximum_local_obstacle_relief_m{0.20};
  double resolution_m{kL0ResolutionM};
};

class ObstacleClassifier final {
 public:
  explicit ObstacleClassifier(ObstacleClassifierConfig config = {});

  [[nodiscard]] CellClassification Classify(
      const CellNeighborhood& neighborhood) const;
  [[nodiscard]] CellClassification Classify(
      const SparseObservedMap& map, GridCellIndex cell) const;

 private:
  ObstacleClassifierConfig config_;
};

}  // namespace lunar::observed_map
