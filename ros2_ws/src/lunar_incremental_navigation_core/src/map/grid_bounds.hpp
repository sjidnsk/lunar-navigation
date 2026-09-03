#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "lunar_incremental_navigation_core/elevation_map.hpp"

namespace lunar::incremental_navigation::detail {

[[nodiscard]] inline std::optional<std::int64_t> CheckedAdd(
    const std::int64_t value, const std::int64_t delta) noexcept {
  std::int64_t result{};
  return __builtin_add_overflow(value, delta, &result)
             ? std::nullopt
             : std::optional<std::int64_t>(result);
}

[[nodiscard]] inline std::optional<std::int64_t> CheckedSubtract(
    const std::int64_t value, const std::int64_t delta) noexcept {
  std::int64_t result{};
  return __builtin_sub_overflow(value, delta, &result)
             ? std::nullopt
             : std::optional<std::int64_t>(result);
}

[[nodiscard]] inline std::optional<std::int64_t> CheckedMultiply(
    const std::int64_t value, const std::int64_t factor) noexcept {
  std::int64_t result{};
  return __builtin_mul_overflow(value, factor, &result)
             ? std::nullopt
             : std::optional<std::int64_t>(result);
}

[[nodiscard]] inline std::int64_t SaturatingCellExtent(
    const double radius_m, const double resolution_m,
    const std::int64_t padding_cells) noexcept {
  if (!std::isfinite(radius_m) || radius_m < 0.0 ||
      !std::isfinite(resolution_m) || resolution_m <= 0.0 ||
      padding_cells < 0) {
    return 0;
  }
  const double cells = std::ceil(radius_m / resolution_m);
  constexpr std::int64_t kMaximum =
      std::numeric_limits<std::int64_t>::max();
  if (!std::isfinite(cells) ||
      cells >= static_cast<double>(kMaximum - padding_cells)) {
    return kMaximum;
  }
  return static_cast<std::int64_t>(cells) + padding_cells;
}

[[nodiscard]] inline std::int64_t ClampSubtract(
    const std::int64_t value, const std::int64_t delta,
    const std::int64_t lower_bound) noexcept {
  const std::optional<std::int64_t> result = CheckedSubtract(value, delta);
  return result ? std::max(lower_bound, *result) : lower_bound;
}

[[nodiscard]] inline std::int64_t ClampAdd(
    const std::int64_t value, const std::int64_t delta,
    const std::int64_t upper_bound) noexcept {
  const std::optional<std::int64_t> result = CheckedAdd(value, delta);
  return result ? std::min(upper_bound, *result) : upper_bound;
}

[[nodiscard]] inline std::pair<GridIndex, GridIndex> CellBoundsForRadius(
    const SparseGridGeometry& geometry, const GridIndex center,
    const double radius_m) noexcept {
  const std::int64_t extent =
      SaturatingCellExtent(radius_m, geometry.resolution_m(), 1);
  const GridIndex minimum{
      .x = ClampSubtract(center.x, extent, geometry.min_inclusive().x),
      .y = ClampSubtract(center.y, extent, geometry.min_inclusive().y),
  };
  const std::int64_t upper_x =
      ClampAdd(center.x, extent, geometry.max_exclusive().x);
  const std::int64_t upper_y =
      ClampAdd(center.y, extent, geometry.max_exclusive().y);
  return {
      minimum,
      GridIndex{
          .x = ClampAdd(upper_x, 1, geometry.max_exclusive().x),
          .y = ClampAdd(upper_y, 1, geometry.max_exclusive().y),
      },
  };
}

[[nodiscard]] inline std::optional<GridIndex> OffsetWithinGeometry(
    const SparseGridGeometry& geometry, const GridIndex index,
    const std::int64_t dx, const std::int64_t dy) noexcept {
  const std::optional<std::int64_t> x = CheckedAdd(index.x, dx);
  const std::optional<std::int64_t> y = CheckedAdd(index.y, dy);
  if (!x || !y) {
    return std::nullopt;
  }
  const GridIndex result{.x = *x, .y = *y};
  return geometry.Contains(result) ? std::optional<GridIndex>(result)
                                   : std::nullopt;
}

}  // namespace lunar::incremental_navigation::detail
