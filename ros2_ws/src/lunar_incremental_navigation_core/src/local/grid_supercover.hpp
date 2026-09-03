#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

#include "lunar_incremental_navigation_core/elevation_map.hpp"

namespace lunar::incremental_navigation::local {

[[nodiscard]] inline std::optional<GridIndex> SupercoverWorldToCell(
    const SparseGridGeometry& geometry, const Vec3 point) noexcept {
  if (!geometry.valid() || !std::isfinite(point.x) ||
      !std::isfinite(point.y)) {
    return std::nullopt;
  }
  const Vec3 origin = geometry.origin_m();
  const double x = std::floor((point.x - origin.x) / geometry.resolution_m());
  const double y = std::floor((point.y - origin.y) / geometry.resolution_m());
  if (!std::isfinite(x) || !std::isfinite(y) ||
      x < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      x >= static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
      y < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      y >= static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    return std::nullopt;
  }
  const GridIndex index{.x = static_cast<std::int64_t>(x),
                        .y = static_cast<std::int64_t>(y)};
  return geometry.Contains(index) ? std::optional<GridIndex>(index)
                                  : std::nullopt;
}

template <typename Visitor>
[[nodiscard]] bool VisitSupercoverCells(
    const SparseGridGeometry& geometry, const Vec3 from, const Vec3 to,
    Visitor&& visitor) {
  auto current = SupercoverWorldToCell(geometry, from);
  const auto finish = SupercoverWorldToCell(geometry, to);
  if (!current || !finish) {
    return false;
  }

  const Vec3 origin = geometry.origin_m();
  const double resolution = geometry.resolution_m();
  const double x0 = (from.x - origin.x) / resolution;
  const double y0 = (from.y - origin.y) / resolution;
  const double x1 = (to.x - origin.x) / resolution;
  const double y1 = (to.y - origin.y) / resolution;
  const double dx = x1 - x0;
  const double dy = y1 - y0;
  const std::int64_t step_x = dx > 0.0 ? 1 : (dx < 0.0 ? -1 : 0);
  const std::int64_t step_y = dy > 0.0 ? 1 : (dy < 0.0 ? -1 : 0);
  const double infinity = std::numeric_limits<double>::infinity();
  const double t_delta_x = step_x == 0 ? infinity : std::abs(1.0 / dx);
  const double t_delta_y = step_y == 0 ? infinity : std::abs(1.0 / dy);
  const double next_x = step_x > 0 ? std::floor(x0) + 1.0 : std::floor(x0);
  const double next_y = step_y > 0 ? std::floor(y0) + 1.0 : std::floor(y0);
  double t_max_x = step_x == 0 ? infinity : (next_x - x0) / dx;
  double t_max_y = step_y == 0 ? infinity : (next_y - y0) / dy;

  const auto near_integer = [](const double value) {
    const double nearest = std::round(value);
    const double tolerance =
        64.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::abs(value));
    return std::abs(value - nearest) <= tolerance;
  };
  const bool horizontal_boundary =
      dy == 0.0 && near_integer(y0) && near_integer(y1);
  const bool vertical_boundary =
      dx == 0.0 && near_integer(x0) && near_integer(x1);
  const auto visit_coverage = [&](const GridIndex index) {
    if (!geometry.Contains(index) || !visitor(index)) {
      return false;
    }
    if (horizontal_boundary) {
      if (index.y == std::numeric_limits<std::int64_t>::min()) {
        return false;
      }
      const GridIndex opposite{.x = index.x, .y = index.y - 1};
      if (!geometry.Contains(opposite) || !visitor(opposite)) {
        return false;
      }
    }
    if (vertical_boundary) {
      if (index.x == std::numeric_limits<std::int64_t>::min()) {
        return false;
      }
      const GridIndex opposite{.x = index.x - 1, .y = index.y};
      if (!geometry.Contains(opposite) || !visitor(opposite)) {
        return false;
      }
      if (horizontal_boundary) {
        const GridIndex diagonal{.x = index.x - 1, .y = index.y - 1};
        if (!geometry.Contains(diagonal) || !visitor(diagonal)) {
          return false;
        }
      }
    }
    return true;
  };
  if (!visit_coverage(*current) || !visit_coverage(*finish)) {
    return false;
  }

  while (*current != *finish) {
    const double corner_tolerance =
        64.0 * std::numeric_limits<double>::epsilon() *
        std::max({1.0, std::abs(t_max_x), std::abs(t_max_y)});
    const bool mathematical_corner =
        std::isfinite(t_max_x) && std::isfinite(t_max_y) &&
        std::abs(t_max_x - t_max_y) <= corner_tolerance;
    if (!mathematical_corner && t_max_x < t_max_y) {
      current->x += step_x;
      t_max_x += t_delta_x;
      if (!visit_coverage(*current)) {
        return false;
      }
      continue;
    }
    if (!mathematical_corner && t_max_y < t_max_x) {
      current->y += step_y;
      t_max_y += t_delta_y;
      if (!visit_coverage(*current)) {
        return false;
      }
      continue;
    }

    const GridIndex side_x{.x = current->x + step_x, .y = current->y};
    const GridIndex side_y{.x = current->x, .y = current->y + step_y};
    if (!visit_coverage(side_x) || !visit_coverage(side_y)) {
      return false;
    }
    current->x += step_x;
    current->y += step_y;
    t_max_x += t_delta_x;
    t_max_y += t_delta_y;
    if (!visit_coverage(*current)) {
      return false;
    }
  }
  return true;
}

}  // namespace lunar::incremental_navigation::local
