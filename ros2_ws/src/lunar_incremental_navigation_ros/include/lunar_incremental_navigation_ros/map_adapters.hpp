#pragma once

#include <grid_map_msgs/msg/grid_map.hpp>
#include <vector>

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_ros/input_store.hpp"

namespace lunar::incremental_navigation_ros {

struct OwnedElevationEvidence final {
  lunar::incremental_navigation::GridGeometry geometry;
  std::vector<float> elevation_m;
  lunar::incremental_navigation::RigidTransform map_from_source;

  [[nodiscard]] lunar::incremental_navigation::ElevationEvidence View() const noexcept;
};

[[nodiscard]] AdapterResult<OwnedElevationEvidence> AdaptLocalElevation(
    const InputSnapshot& input);

}  // namespace lunar::incremental_navigation_ros
