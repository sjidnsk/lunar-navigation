#pragma once

#include <cstddef>

#include "lunar_pure_exploration_core/platform_geometry.hpp"
#include "lunar_pure_exploration_core/task_raster.hpp"

namespace lunar::pure_exploration {

class SafePoseValidator final {
 public:
  SafePoseValidator(PlatformGeometry platform,
                    std::size_t maximum_collision_work_units);
  bool IsMapFree(const OccupancyGridView& map, Pose2 pose,
                 std::size_t& consumed_work) const;
  bool IsTaskFree(const TaskRaster& raster, Pose2 pose,
                  std::size_t& consumed_work) const;
  double platform_length_m() const;
  double platform_width_m() const;
  double footprint_circumscribed_radius_m() const;
  double minimum_standoff_m() const;

 private:
  PlatformGeometry platform_;
  std::size_t maximum_collision_work_units_;
  double platform_length_m_;
  double platform_width_m_;
  double footprint_circumscribed_radius_m_;
};

}  // namespace lunar::pure_exploration
