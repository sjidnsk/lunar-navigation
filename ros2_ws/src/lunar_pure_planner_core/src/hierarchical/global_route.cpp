#include "hierarchical/global_route.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace lunar::pure_planning::hierarchical {
namespace {

[[nodiscard]] int Sign(const std::int32_t value) noexcept {
  return (value > 0) - (value < 0);
}

[[nodiscard]] bool
CellSafe(const shared::GridSearchProjectionView projection,
         const shared::GridCell cell,
         const std::span<const std::uint8_t> excluded_mask) noexcept {
  if (!projection.HardFeasible(cell)) {
    return false;
  }
  const auto* map = projection.map;
  return excluded_mask.empty() ||
         (map != nullptr && excluded_mask.size() == map->cell_count() &&
          excluded_mask[map->Index(cell)] == 0U);
}

[[nodiscard]] bool
SegmentSafe(const shared::GridSearchProjectionView projection,
            const shared::GridCell start, const shared::GridCell end,
            const std::span<const std::uint8_t> excluded_mask) noexcept {
  if (!CellSafe(projection, start, excluded_mask) ||
      !CellSafe(projection, end, excluded_mask)) {
    return false;
  }
  if (start == end) {
    return true;
  }

  std::int32_t x = start.x;
  std::int32_t y = start.y;
  const std::int32_t delta_x = end.x - start.x;
  const std::int32_t delta_y = end.y - start.y;
  const int step_x = Sign(delta_x);
  const int step_y = Sign(delta_y);
  const double infinity = std::numeric_limits<double>::infinity();
  const double absolute_x = std::abs(static_cast<double>(delta_x));
  const double absolute_y = std::abs(static_cast<double>(delta_y));
  const double delta_t_x = absolute_x > 0.0 ? 1.0 / absolute_x : infinity;
  const double delta_t_y = absolute_y > 0.0 ? 1.0 / absolute_y : infinity;
  double maximum_t_x = absolute_x > 0.0 ? 0.5 / absolute_x : infinity;
  double maximum_t_y = absolute_y > 0.0 ? 0.5 / absolute_y : infinity;

  while (x != end.x || y != end.y) {
    if (step_y == 0) {
      x += step_x;
      maximum_t_x += delta_t_x;
    } else if (step_x == 0) {
      y += step_y;
      maximum_t_y += delta_t_y;
    } else {
      const double comparison_tolerance =
          1.0e-12 *
          std::max({1.0, std::abs(maximum_t_x), std::abs(maximum_t_y)});
      if (maximum_t_x + comparison_tolerance < maximum_t_y) {
        x += step_x;
        maximum_t_x += delta_t_x;
      } else if (maximum_t_y + comparison_tolerance < maximum_t_x) {
        y += step_y;
        maximum_t_y += delta_t_y;
      } else {
        const shared::GridCell side_x{x + step_x, y};
        const shared::GridCell side_y{x, y + step_y};
        if (!CellSafe(projection, side_x, excluded_mask) ||
            !CellSafe(projection, side_y, excluded_mask)) {
          return false;
        }
        x += step_x;
        y += step_y;
        maximum_t_x += delta_t_x;
        maximum_t_y += delta_t_y;
      }
    }
    if (!CellSafe(projection, shared::GridCell{x, y}, excluded_mask)) {
      return false;
    }
  }
  return true;
}

} // namespace

std::vector<shared::GridCell>
SimplifyRouteSupercover(const shared::GridSearchProjectionView projection,
                        const std::span<const shared::GridCell> route,
                        const std::size_t maximum_points,
                        const std::span<const std::uint8_t> excluded_mask) {
  if (route.empty() || maximum_points == 0U) {
    return {};
  }
  const auto* map = projection.map;
  if (!excluded_mask.empty() &&
      (map == nullptr || excluded_mask.size() != map->cell_count())) {
    return {};
  }
  for (std::size_t index = 0U; index < route.size(); ++index) {
    if (!CellSafe(projection, route[index], excluded_mask) ||
        (index > 0U && !SegmentSafe(projection, route[index - 1U], route[index],
                                    excluded_mask))) {
      return {};
    }
  }
  if (route.size() == 1U) {
    return {route.front()};
  }

  std::vector<shared::GridCell> simplified;
  simplified.reserve(std::min(route.size(), maximum_points));
  simplified.push_back(route.front());
  std::size_t current = 0U;
  while (current + 1U < route.size()) {
    std::size_t selected = current + 1U;
    for (std::size_t candidate = route.size() - 1U; candidate > current;
         --candidate) {
      if (SegmentSafe(projection, route[current], route[candidate],
                      excluded_mask)) {
        selected = candidate;
        break;
      }
    }
    simplified.push_back(route[selected]);
    if (simplified.size() > maximum_points) {
      return {};
    }
    current = selected;
  }
  return simplified;
}

std::vector<Pose3> ThinRoutePreview(const std::span<const Pose3> route,
                                    const std::size_t maximum_points) {
  if (route.empty() || maximum_points == 0U) {
    return {};
  }
  if (route.size() <= maximum_points) {
    return {route.begin(), route.end()};
  }
  if (maximum_points == 1U) {
    return {route.back()};
  }

  std::vector<Pose3> preview;
  preview.reserve(maximum_points);
  const std::size_t last = route.size() - 1U;
  const std::size_t intervals = maximum_points - 1U;
  for (std::size_t index = 0U; index < maximum_points; ++index) {
    preview.push_back(route[index * last / intervals]);
  }
  return preview;
}

} // namespace lunar::pure_planning::hierarchical
