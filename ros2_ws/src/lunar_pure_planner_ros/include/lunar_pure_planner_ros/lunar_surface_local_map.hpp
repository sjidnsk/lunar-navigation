#pragma once

#include <cstddef>
#include <vector>

#include "lunar_pure_planner_ros/lunar_surface_scenario.hpp"

namespace lunar::pure_planner_ros {

struct LunarSurfaceLocalRaster final {
  std::size_t width{320U};
  std::size_t height{320U};
  double resolution_m{0.2};
  double center_x_m{};
  double center_y_m{};
  std::vector<float> occupancy;
  std::vector<float> elevation_m;

  [[nodiscard]] double length_x_m() const noexcept {
    return static_cast<double>(width) * resolution_m;
  }
  [[nodiscard]] double length_y_m() const noexcept {
    return static_cast<double>(height) * resolution_m;
  }
};

[[nodiscard]] LunarSurfaceLocalRaster BuildLunarSurfaceLocalRaster(
    const LunarSurfaceScenario& scenario, double center_x_m,
    double center_y_m);

}  // namespace lunar::pure_planner_ros
