#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

#include "lunar_planner_core/types/planner_config.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"

namespace lunar::planning::hierarchical {

enum class GlobalSearchStatus : std::uint8_t {
  kSolved,
  kNoPath,
  kCanceled,
  kAllocationFailed,
  kInvalidProblem,
};

struct GlobalGridSearchProblem final {
  const shared::SafeProjection &projection;
  shared::GridCell start;
  std::span<const std::uint8_t> goal_mask;
  double maximum_speed_mps{1.0};
  GlobalSearchConfig config;
  std::stop_token stop_token;
};

struct GlobalGridSearchResult final {
  GlobalSearchStatus status{GlobalSearchStatus::kInvalidProblem};
  std::vector<shared::GridCell> path_cells;
  double cost{};
  std::uint64_t expanded_states{};
  std::size_t open_peak{};
  std::size_t estimated_work_memory_bytes{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return status == GlobalSearchStatus::kSolved && reason_code.empty() &&
           !path_cells.empty();
  }
};

[[nodiscard]] GlobalGridSearchResult
SearchGlobalGrid(const GlobalGridSearchProblem &problem);

} // namespace lunar::planning::hierarchical
