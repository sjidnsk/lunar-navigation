#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "lunar_incremental_navigation_core/traversability_snapshot.hpp"

namespace lunar::incremental_navigation {

// The local solvers use fixed-size dense workspaces. This is a capacity bound
// only: a configured meter window is never silently shortened to fit it.
// 640 cells covers the default 64 m window at the supported 0.1 m fine-map
// resolution.
inline constexpr std::size_t kMaximumLocalPlanningWindowAxisCells{640U};

enum class LocalPlanningWindowStatus : std::uint8_t {
  kReady,
  kStartOutsideFineMap,
  kCapacityExceeded,
};

struct LocalPlanningWindowResult final {
  LocalPlanningWindowStatus status{LocalPlanningWindowStatus::kStartOutsideFineMap};
  std::optional<SparseGridGeometry> geometry;
};

[[nodiscard]] bool IsLocalPlanningWindowFor(
    const SparseGridGeometry& window,
    const SparseGridGeometry& base) noexcept;

// Returns a lattice-aligned subset of `fine.geometry()` that contains `start`.
// The persistent fine snapshot remains immutable and may grow without bound.
// If `local_window_size_m` needs more than the fixed solver workspace, returns
// kCapacityExceeded instead of truncating the requested physical window.
[[nodiscard]] LocalPlanningWindowResult BuildLocalPlanningWindow(
    const FineTraversabilitySnapshot& fine, Vec2 start,
    double local_window_size_m) noexcept;

}  // namespace lunar::incremental_navigation
