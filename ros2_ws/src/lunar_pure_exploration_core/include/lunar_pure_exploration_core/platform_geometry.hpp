#pragma once

#include <string>
#include <vector>

#include "lunar_pure_exploration_core/types.hpp"

namespace lunar::pure_exploration {

struct PlatformGeometry {
  std::string platform_id;
  std::string platform_type;
  std::string base_frame_id;
  std::vector<Vec2> footprint_vertices;
  double minimum_clearance_m;
};

}  // namespace lunar::pure_exploration
