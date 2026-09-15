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
  // Optional atomic terrain_slope/terrain_relief/terrain_positive_rise/
  // terrain_complete layers; same cell ordering as elevation_m. Upright
  // quarter-turn transforms preserve the native 3x3 scalar semantics. Four
  // NaNs mean absent evidence at a cell; otherwise all scalars must be finite
  // and nonnegative, with terrain_complete exactly 0 or 1 and finite elevation.
  std::vector<lunar::incremental_navigation::LocalTerrainMeasurements> terrain_measurements;

  [[nodiscard]] lunar::incremental_navigation::ElevationEvidence View() const noexcept;
};

[[nodiscard]] AdapterResult<OwnedElevationEvidence> AdaptLocalElevation(
    const InputSnapshot& input);

}  // namespace lunar::incremental_navigation_ros
