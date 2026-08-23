#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lunar_pure_planner_core/search_control.hpp"
#include "lunar_pure_planner_core/types/motion_reference.hpp"
#include "lunar_pure_planner_core/types/planning_request.hpp"
#include "shared/global_occupancy_projection.hpp"

namespace lunar::pure_planning::hierarchical {

struct SurfaceGlobalSearchProblem final {
  shared::GlobalOccupancyProjectionView projection;
  shared::GridCell start;
  shared::GridCell goal;
  Pose3 start_pose_map;
  Pose3 goal_pose_map;
  SearchControl control;
  AnytimeSearchConfig search;
};

enum class SurfaceGlobalSearchStatus : std::uint8_t {
  kSolved,
  kNoPath,
  kCanceled,
  kTimedOut,
  kResourceExhausted,
  kInvalidProblem,
};

struct SurfaceGlobalSearchResult final {
  SurfaceGlobalSearchStatus status{SurfaceGlobalSearchStatus::kInvalidProblem};
  std::vector<shared::GridCell> raw_cells;
  std::vector<shared::GridCell> simplified_cells;
  GlobalRoutePreview preview;
  double cost{};
  std::size_t expanded_states{};
  bool deadline_reached{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return status == SurfaceGlobalSearchStatus::kSolved &&
           !raw_cells.empty() && !preview.poses_map.empty() &&
           reason_code == "SEARCH_SOLVED";
  }
};

[[nodiscard]] SurfaceGlobalSearchResult SearchSurfaceGlobal(
    const SurfaceGlobalSearchProblem& problem);

}  // namespace lunar::pure_planning::hierarchical
