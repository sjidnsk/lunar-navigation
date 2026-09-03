#pragma once

#include <cstddef>
#include <optional>

#include "lunar_incremental_navigation_core/traversability_snapshot.hpp"

namespace lunar::incremental_navigation {

// The local solvers use fixed-size dense workspaces.  This is an algorithmic
// bound in cells, not a map-resolution requirement.
inline constexpr std::size_t kMaximumLocalPlanningWindowAxisCells{320U};

[[nodiscard]] bool IsLocalPlanningWindowFor(
    const SparseGridGeometry& window,
    const SparseGridGeometry& base) noexcept;

// Returns a lattice-aligned subset of `fine.geometry()` that contains `start`.
// The persistent fine snapshot remains immutable and may grow without bound.
[[nodiscard]] std::optional<SparseGridGeometry> BuildLocalPlanningWindow(
    const FineTraversabilitySnapshot& fine, Vec2 start) noexcept;

}  // namespace lunar::incremental_navigation
