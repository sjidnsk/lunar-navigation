#include "lunar_pure_planner_ros/lunar_surface_local_map.hpp"

#include <limits>

namespace lunar::pure_planner_ros {

LunarSurfaceLocalRaster BuildLunarSurfaceLocalRaster(
    const LunarSurfaceScenario& scenario, const double center_x_m,
    const double center_y_m) {
  LunarSurfaceLocalRaster raster;
  raster.center_x_m = center_x_m;
  raster.center_y_m = center_y_m;
  const std::size_t cell_count = raster.width * raster.height;
  const float unknown = std::numeric_limits<float>::quiet_NaN();
  raster.occupancy.assign(cell_count, unknown);
  raster.elevation_m.assign(cell_count, unknown);
  const double origin_x_m = center_x_m - 0.5 * raster.length_x_m();
  const double origin_y_m = center_y_m - 0.5 * raster.length_y_m();
  for (std::size_t y = 0U; y < raster.height; ++y) {
    for (std::size_t x = 0U; x < raster.width; ++x) {
      const double world_x_m =
          origin_x_m + (static_cast<double>(x) + 0.5) * raster.resolution_m;
      const double world_y_m =
          origin_y_m + (static_cast<double>(y) + 0.5) * raster.resolution_m;
      const auto sample = scenario.Sample(world_x_m, world_y_m);
      if (!sample.has_value()) {
        continue;
      }
      const std::size_t index = y * raster.width + x;
      raster.occupancy[index] = sample->occupied ? 1.0F : 0.0F;
      raster.elevation_m[index] = sample->elevation_m;
    }
  }
  return raster;
}

}  // namespace lunar::pure_planner_ros
