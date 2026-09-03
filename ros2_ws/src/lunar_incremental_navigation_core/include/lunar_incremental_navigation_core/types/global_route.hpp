#pragma once

#include <cstdint>
#include <vector>

#include "lunar_incremental_navigation_core/types/geometry.hpp"

namespace lunar::incremental_navigation {

struct GlobalRoute final {
  std::vector<Pose3> poses_map;
  std::uint64_t expanded_states{};
};

}  // namespace lunar::incremental_navigation
