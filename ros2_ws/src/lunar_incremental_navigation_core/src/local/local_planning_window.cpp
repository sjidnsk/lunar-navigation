#include "lunar_incremental_navigation_core/local_planning_window.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace lunar::incremental_navigation {
namespace {

[[nodiscard]] std::optional<GridIndex> WorldToCell(
    const SparseGridGeometry& geometry, const Vec2 point) noexcept {
  if (!geometry.valid() || !std::isfinite(point.x) ||
      !std::isfinite(point.y)) {
    return std::nullopt;
  }
  const Vec3 origin = geometry.origin_m();
  const double resolution_m = geometry.resolution_m();
  const double x = std::floor((point.x - origin.x) / resolution_m);
  const double y = std::floor((point.y - origin.y) / resolution_m);
  if (!std::isfinite(x) || !std::isfinite(y) ||
      x < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      x >= static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
      y < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      y >= static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    return std::nullopt;
  }
  const GridIndex index{.x = static_cast<std::int64_t>(x),
                        .y = static_cast<std::int64_t>(y)};
  return geometry.Contains(index) ? std::optional<GridIndex>{index}
                                  : std::nullopt;
}

[[nodiscard]] std::int64_t WindowMinimum(const std::int64_t base_min,
                                          const std::int64_t base_max,
                                          const std::int64_t anchor,
                                          const std::int64_t span) noexcept {
  const std::int64_t half_span = span / 2;
  const std::int64_t largest_minimum = base_max - span;
  if (anchor <= base_min + half_span) {
    return base_min;
  }
  if (anchor >= largest_minimum + half_span) {
    return largest_minimum;
  }
  return anchor - half_span;
}

}  // namespace

bool IsLocalPlanningWindowFor(const SparseGridGeometry& window,
                              const SparseGridGeometry& base) noexcept {
  if (!window.valid() || !base.valid() ||
      window.frame_id() != base.frame_id() ||
      window.resolution_m() != base.resolution_m() ||
      window.origin_m() != base.origin_m()) {
    return false;
  }
  const GridIndex window_minimum = window.min_inclusive();
  const GridIndex window_maximum = window.max_exclusive();
  const GridIndex base_minimum = base.min_inclusive();
  const GridIndex base_maximum = base.max_exclusive();
  return window_minimum.x >= base_minimum.x &&
         window_minimum.y >= base_minimum.y &&
         window_maximum.x <= base_maximum.x &&
         window_maximum.y <= base_maximum.y;
}

LocalPlanningWindowResult BuildLocalPlanningWindow(
    const FineTraversabilitySnapshot& fine, const Vec2 start,
    const double local_window_size_m) noexcept {
  const SparseGridGeometry& base = fine.geometry();
  const std::optional<GridIndex> anchor = WorldToCell(base, start);
  if (!anchor) {
    return {.status = LocalPlanningWindowStatus::kStartOutsideFineMap};
  }
  if (!std::isfinite(local_window_size_m) || local_window_size_m <= 0.0) {
    return {.status = LocalPlanningWindowStatus::kCapacityExceeded};
  }
  const double requested_axis_cells =
      std::ceil(local_window_size_m / base.resolution_m());
  if (!std::isfinite(requested_axis_cells) || requested_axis_cells <= 0.0 ||
      requested_axis_cells >
          static_cast<double>(kMaximumLocalPlanningWindowAxisCells)) {
    return {.status = LocalPlanningWindowStatus::kCapacityExceeded};
  }
  const std::size_t requested_axis =
      static_cast<std::size_t>(requested_axis_cells);

  const std::size_t width = std::min(base.width(), requested_axis);
  const std::size_t height = std::min(base.height(), requested_axis);
  if (width == 0U || height == 0U) {
    return {.status = LocalPlanningWindowStatus::kStartOutsideFineMap};
  }
  const auto span = [](const std::size_t value) {
    return static_cast<std::int64_t>(value);
  };
  const GridIndex base_minimum = base.min_inclusive();
  const GridIndex base_maximum = base.max_exclusive();
  const GridIndex window_minimum{
      .x = WindowMinimum(base_minimum.x, base_maximum.x, anchor->x,
                         span(width)),
      .y = WindowMinimum(base_minimum.y, base_maximum.y, anchor->y,
                         span(height)),
  };
  const GridIndex window_maximum{
      .x = window_minimum.x + span(width),
      .y = window_minimum.y + span(height),
  };
  SparseGridGeometry window(base.frame_id(), base.resolution_m(),
                            base.origin_m(), window_minimum, window_maximum);
  if (!IsLocalPlanningWindowFor(window, base) || !window.Contains(*anchor)) {
    return {.status = LocalPlanningWindowStatus::kStartOutsideFineMap};
  }
  return {.status = LocalPlanningWindowStatus::kReady,
          .geometry = std::move(window)};
}

}  // namespace lunar::incremental_navigation
