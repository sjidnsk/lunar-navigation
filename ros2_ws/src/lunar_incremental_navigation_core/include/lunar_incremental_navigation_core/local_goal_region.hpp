#pragma once

#include <cstdint>
#include <vector>
#include "lunar_incremental_navigation_core/local_target_selector.hpp"

namespace lunar::incremental_navigation {

struct LocalGoalCandidate final {
  LocalTarget target;
  double remaining_cost{};
};

// Immutable per-request metadata. FREE is only an eligibility condition;
// the platform search must certify every edge before accepting a candidate.
class LocalGoalRegion final {
 public:
  SparseGridGeometry geometry;
  Point2 final_point;
  // UNKNOWN adjacent to evidence inside fine extent, or a final point beyond
  // that extent. A fully observed map edge alone is not missing evidence.
  // Window-wide summary for diagnostics only; it is not platform reachability.
  bool has_unknown_boundary{};
  std::vector<LocalGoalCandidate> candidates;
  std::vector<std::int32_t> candidate_by_cell;
  // Diagnostic potential only until a platform certifies reaching the cell.
  // Includes cells excluded as new targets because the robot already arrived.
  std::vector<std::uint8_t> forward_unknown_by_cell;

  [[nodiscard]] const LocalGoalCandidate* At(GridIndex cell) const noexcept;
  [[nodiscard]] bool HasForwardUnknownBoundaryAt(GridIndex cell) const noexcept;
  [[nodiscard]] double LowerBound(Point2 point) const noexcept;
};

}  // namespace lunar::incremental_navigation
