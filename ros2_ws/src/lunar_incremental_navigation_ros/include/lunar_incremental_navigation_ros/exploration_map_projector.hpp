#pragma once

#include <cstdint>
#include <vector>

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/traversability_snapshot.hpp"

namespace lunar::incremental_navigation_ros {

struct ExplorationMapProjection final {
  lunar::incremental_navigation::SparseGridGeometry geometry;
  std::uint64_t source_fine_traversability_revision{};
  std::vector<std::int8_t> data;
};

class ExplorationMapProjector final {
 public:
  [[nodiscard]] ExplorationMapProjection Project(
      const lunar::incremental_navigation::FineTraversabilitySnapshot& fine,
      const lunar::incremental_navigation::SparseGridGeometry& coarse_geometry)
      const;
};

}  // namespace lunar::incremental_navigation_ros
