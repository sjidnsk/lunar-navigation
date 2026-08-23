#pragma once

#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::shared {

struct ObstacleHeight final {
  bool flyover_allowed{};
  double height_m{};
};

[[nodiscard]] ObstacleHeight EstimateObstacleHeight(const MapSnapshot& map,
                                                     GridCell occupied);

}  // namespace lunar::pure_planning::shared
