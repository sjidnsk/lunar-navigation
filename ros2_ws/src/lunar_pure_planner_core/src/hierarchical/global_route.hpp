#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "lunar_pure_planner_core/types/geometry.hpp"
#include "shared/grid_search_projection.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::hierarchical {

struct GlobalRoute final {
  std::vector<shared::GridCell> raw_cells;
  std::vector<shared::GridCell> conditional_cells;
  std::vector<shared::GridCell> simplified_cells;
  std::vector<Pose3> poses_map;
  double cost{};
  std::uint64_t expanded_states{};
  std::size_t open_peak{};
  std::size_t estimated_work_memory_bytes{};
};

[[nodiscard]] std::vector<shared::GridCell>
SimplifyRouteSupercover(shared::GridSearchProjectionView projection,
                        std::span<const shared::GridCell> route,
                        std::size_t maximum_points,
                        std::span<const std::uint8_t> excluded_mask = {});

[[nodiscard]] std::vector<Pose3> ThinRoutePreview(std::span<const Pose3> route,
                                                  std::size_t maximum_points);

} // namespace lunar::pure_planning::hierarchical
