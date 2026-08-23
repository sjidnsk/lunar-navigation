#pragma once

#include "lunar_pure_exploration_core/task_raster.hpp"

namespace lunar::pure_exploration {

struct CoverageStats {
  double polygon_area_m2;
  double task_raster_area_m2;
  double known_free_area_m2;
  double known_occupied_area_m2;
  double unknown_area_m2;
  double outside_map_area_m2;
  double coverage_ratio;
};

CoverageStats CalculateCoverage(const TaskRaster& raster);

}  // namespace lunar::pure_exploration
