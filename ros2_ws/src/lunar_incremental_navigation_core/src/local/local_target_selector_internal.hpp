#pragma once

#include <cstddef>
#include <optional>

#include "lunar_incremental_navigation_core/local_target_selector.hpp"

namespace lunar::incremental_navigation::local_target_selector_internal {

struct SelectionMetrics final {
  std::size_t candidate_tile_lookups{};
  std::size_t candidate_cells_examined{};
  std::size_t guidance_cells_examined{};
};

[[nodiscard]] std::optional<LocalTarget> Select(
    const FineTraversabilitySnapshot& fine,
    const SparseGridGeometry& local_window, Point2 start,
    const FinalGoal& final_goal, const std::optional<GlobalRoute>& guidance,
    SelectionMetrics* metrics);

[[nodiscard]] std::optional<LocalTarget> Select(
    const FineTraversabilitySnapshot& fine, Point2 start,
    const FinalGoal& final_goal, const std::optional<GlobalRoute>& guidance,
    SelectionMetrics* metrics);

}  // namespace lunar::incremental_navigation::local_target_selector_internal
