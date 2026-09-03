#include "lunar_incremental_navigation_core/local_target_selector.hpp"

#include "lunar_incremental_navigation_core/local_planning_window.hpp"

#include "local/local_target_selector_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace lunar::incremental_navigation {
namespace {

using local_target_selector_internal::SelectionMetrics;

[[nodiscard]] bool Finite(const Point2 point) noexcept {
  return std::isfinite(point.x) && std::isfinite(point.y);
}

[[nodiscard]] std::optional<GridIndex> CellAt(
    const SparseGridGeometry& geometry, const Point2 point) noexcept {
  if (!Finite(point)) {
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
  return GridIndex{.x = static_cast<std::int64_t>(x),
                   .y = static_cast<std::int64_t>(y)};
}

[[nodiscard]] Point2 CellCenter(const SparseGridGeometry& geometry,
                                const GridIndex index) noexcept {
  const Vec3 origin = geometry.origin_m();
  const double resolution_m = geometry.resolution_m();
  return Point2{
      .x = std::fma(static_cast<double>(index.x) + 0.5, resolution_m,
                    origin.x),
      .y = std::fma(static_cast<double>(index.y) + 0.5, resolution_m,
                    origin.y),
  };
}

[[nodiscard]] double SquaredDistance(const Point2 left,
                                     const Point2 right) noexcept {
  const double dx = left.x - right.x;
  const double dy = left.y - right.y;
  return dx * dx + dy * dy;
}

void Increment(std::size_t* value) noexcept {
  if (value && *value != std::numeric_limits<std::size_t>::max()) {
    ++*value;
  }
}

[[nodiscard]] bool IsKnown(const FineTraversabilitySnapshot& fine,
                           const SparseGridGeometry& geometry,
                           const Point2 point) noexcept {
  const auto cell = CellAt(geometry, point);
  return cell && geometry.Contains(*cell) &&
         fine.State(*cell) != FineCellState::kUnknown;
}

struct WindowBounds final {
  double min_x{};
  double min_y{};
  double max_x{};
  double max_y{};
};

struct ClippedSegment final {
  Point2 begin;
  Point2 end;
  long double enter{};
  long double leave{};
};

[[nodiscard]] WindowBounds Bounds(
    const SparseGridGeometry& geometry) noexcept {
  const Vec3 origin = geometry.origin_m();
  const double resolution_m = geometry.resolution_m();
  const GridIndex min = geometry.min_inclusive();
  const GridIndex max = geometry.max_exclusive();
  return WindowBounds{
      .min_x = std::fma(static_cast<double>(min.x), resolution_m, origin.x),
      .min_y = std::fma(static_cast<double>(min.y), resolution_m, origin.y),
      .max_x = std::fma(static_cast<double>(max.x), resolution_m, origin.x),
      .max_y = std::fma(static_cast<double>(max.y), resolution_m, origin.y),
  };
}

[[nodiscard]] bool ClipAxis(const long double begin,
                            const long double end,
                            const long double minimum,
                            const long double maximum,
                            long double& enter,
                            long double& leave) noexcept {
  const long double delta = end - begin;
  if (delta == 0.0L) {
    return begin >= minimum && begin <= maximum;
  }
  long double first = (minimum - begin) / delta;
  long double second = (maximum - begin) / delta;
  if (first > second) {
    std::swap(first, second);
  }
  enter = std::max(enter, first);
  leave = std::min(leave, second);
  return enter <= leave;
}

[[nodiscard]] double InterpolateAndClamp(const double begin,
                                         const double end,
                                         const long double ratio,
                                         const double minimum,
                                         const double maximum) noexcept {
  const double interpolated =
      std::lerp(begin, end, static_cast<double>(ratio));
  if (!std::isfinite(interpolated)) {
    return ratio <= 0.5L ? std::clamp(begin, minimum, maximum)
                         : std::clamp(end, minimum, maximum);
  }
  return std::clamp(interpolated, minimum, maximum);
}

[[nodiscard]] std::optional<ClippedSegment> ClipToWindow(
    const SparseGridGeometry& geometry, const Point2 begin,
    const Point2 end) noexcept {
  if (!Finite(begin) || !Finite(end)) {
    return std::nullopt;
  }
  const WindowBounds bounds = Bounds(geometry);
  const long double scale = std::max(
      {1.0L, std::abs(static_cast<long double>(begin.x)),
       std::abs(static_cast<long double>(begin.y)),
       std::abs(static_cast<long double>(end.x)),
       std::abs(static_cast<long double>(end.y)),
       std::abs(static_cast<long double>(bounds.min_x)),
       std::abs(static_cast<long double>(bounds.min_y)),
       std::abs(static_cast<long double>(bounds.max_x)),
       std::abs(static_cast<long double>(bounds.max_y))});
  long double enter = 0.0L;
  long double leave = 1.0L;
  if (!ClipAxis(static_cast<long double>(begin.x) / scale,
                static_cast<long double>(end.x) / scale,
                static_cast<long double>(bounds.min_x) / scale,
                static_cast<long double>(bounds.max_x) / scale, enter,
                leave) ||
      !ClipAxis(static_cast<long double>(begin.y) / scale,
                static_cast<long double>(end.y) / scale,
                static_cast<long double>(bounds.min_y) / scale,
                static_cast<long double>(bounds.max_y) / scale, enter,
                leave)) {
    return std::nullopt;
  }
  const double inside_max_x = std::nextafter(bounds.max_x, bounds.min_x);
  const double inside_max_y = std::nextafter(bounds.max_y, bounds.min_y);
  return ClippedSegment{
      .begin = {.x = InterpolateAndClamp(begin.x, end.x, enter,
                                        bounds.min_x, inside_max_x),
                .y = InterpolateAndClamp(begin.y, end.y, enter,
                                        bounds.min_y, inside_max_y)},
      .end = {.x = InterpolateAndClamp(begin.x, end.x, leave, bounds.min_x,
                                      inside_max_x),
              .y = InterpolateAndClamp(begin.y, end.y, leave, bounds.min_y,
                                      inside_max_y)},
      .enter = enter,
      .leave = leave,
  };
}

[[nodiscard]] std::uint64_t OrderedCoordinate(
    const std::int64_t value) noexcept {
  constexpr std::uint64_t kSignBit = std::uint64_t{1} << 63U;
  return static_cast<std::uint64_t>(value) ^ kSignBit;
}

[[nodiscard]] std::uint64_t CoordinateDistance(
    const std::int64_t left, const std::int64_t right) noexcept {
  const std::uint64_t ordered_left = OrderedCoordinate(left);
  const std::uint64_t ordered_right = OrderedCoordinate(right);
  return ordered_left < ordered_right ? ordered_right - ordered_left
                                      : ordered_left - ordered_right;
}

[[nodiscard]] std::optional<Point2> TraverseClippedSegment(
    const FineTraversabilitySnapshot& fine,
    const SparseGridGeometry& geometry, const ClippedSegment& segment,
    bool& saw_known, SelectionMetrics* metrics) noexcept {
  const auto begin_cell = CellAt(geometry, segment.begin);
  const auto end_cell = CellAt(geometry, segment.end);
  if (!begin_cell || !end_cell || !geometry.Contains(*begin_cell) ||
      !geometry.Contains(*end_cell)) {
    return std::nullopt;
  }
  const std::uint64_t delta_x =
      CoordinateDistance(begin_cell->x, end_cell->x);
  const std::uint64_t delta_y =
      CoordinateDistance(begin_cell->y, end_cell->y);
  if (delta_x >= geometry.width() || delta_y >= geometry.height()) {
    return std::nullopt;
  }
  const std::uint64_t steps = std::max(delta_x, delta_y);
  std::optional<GridIndex> previous;
  for (std::uint64_t step = 0U;; ++step) {
    const long double ratio =
        steps == 0U ? 0.0L
                    : static_cast<long double>(step) /
                          static_cast<long double>(steps);
    const long double x =
        std::lerp(static_cast<long double>(begin_cell->x),
                  static_cast<long double>(end_cell->x), ratio);
    const long double y =
        std::lerp(static_cast<long double>(begin_cell->y),
                  static_cast<long double>(end_cell->y), ratio);
    GridIndex cell{.x = static_cast<std::int64_t>(std::llround(x)),
                   .y = static_cast<std::int64_t>(std::llround(y))};
    if (!previous || cell != *previous) {
      previous = cell;
      Increment(metrics ? &metrics->guidance_cells_examined : nullptr);
      if (fine.State(cell) == FineCellState::kUnknown) {
        if (saw_known) {
          return CellCenter(geometry, cell);
        }
      } else {
        saw_known = true;
      }
    }
    if (step == steps) {
      break;
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<Point2> GuidanceExit(
    const FineTraversabilitySnapshot& fine,
    const SparseGridGeometry& geometry, const Point2 start,
    const GlobalRoute& guidance,
    SelectionMetrics* metrics) noexcept {
  if (guidance.poses_map.empty()) {
    return std::nullopt;
  }

  std::size_t route_start = 0U;
  double nearest_start_distance = std::numeric_limits<double>::infinity();
  bool found_finite_point = false;
  for (std::size_t index = 0U; index < guidance.poses_map.size(); ++index) {
    const Point2 point{.x = guidance.poses_map[index].position_m.x,
                       .y = guidance.poses_map[index].position_m.y};
    if (!Finite(point)) {
      continue;
    }
    const double distance = std::hypot(point.x - start.x, point.y - start.y);
    if (!found_finite_point || distance < nearest_start_distance) {
      found_finite_point = true;
      nearest_start_distance = distance;
      route_start = index;
    }
  }
  if (!found_finite_point) {
    return std::nullopt;
  }

  bool saw_known = IsKnown(fine, geometry, start);
  Point2 begin = start;
  std::optional<Point2> last_inside;
  for (std::size_t index = route_start; index < guidance.poses_map.size();
       ++index) {
    const Point2 end{.x = guidance.poses_map[index].position_m.x,
                     .y = guidance.poses_map[index].position_m.y};
    if (!Finite(end)) {
      continue;
    }
    const auto clipped = ClipToWindow(geometry, begin, end);
    if (!clipped) {
      if (saw_known && last_inside) {
        return last_inside;
      }
      begin = end;
      continue;
    }
    if (saw_known && clipped->enter > 0.0L) {
      return clipped->begin;
    }
    if (const auto exit =
            TraverseClippedSegment(fine, geometry, *clipped, saw_known,
                                   metrics)) {
      return exit;
    }
    last_inside = clipped->end;
    if (saw_known && clipped->leave < 1.0L) {
      return clipped->end;
    }
    begin = end;
  }
  return saw_known ? last_inside : std::nullopt;
}

struct Candidate final {
  GridIndex index;
  Point2 center;
  double primary{};
  double secondary{};
  double traversal_cost{};
};

[[nodiscard]] bool BetterGuidanceCandidate(const Candidate& candidate,
                                           const Candidate& best) noexcept {
  if (candidate.primary != best.primary) {
    return candidate.primary < best.primary;
  }
  if (candidate.traversal_cost != best.traversal_cost) {
    return candidate.traversal_cost < best.traversal_cost;
  }
  return candidate.index < best.index;
}

[[nodiscard]] bool BetterDirectionalCandidate(
    const Candidate& candidate, const Candidate& best) noexcept {
  if (candidate.primary != best.primary) {
    return candidate.primary > best.primary;
  }
  if (candidate.secondary != best.secondary) {
    return candidate.secondary < best.secondary;
  }
  if (candidate.traversal_cost != best.traversal_cost) {
    return candidate.traversal_cost < best.traversal_cost;
  }
  return candidate.index < best.index;
}

struct AllocatedFineTile final {
  TileIndex index;
  std::shared_ptr<const FineTraversabilityTile> tile;
};

[[nodiscard]] FineCellState StateFromAllocatedTiles(
    const std::vector<AllocatedFineTile>& tiles,
    const GridIndex index) noexcept {
  const TileIndex tile_index = TileForCell(index);
  const auto found = std::lower_bound(
      tiles.begin(), tiles.end(), tile_index,
      [](const AllocatedFineTile& entry, const TileIndex value) {
        return entry.index < value;
      });
  return found != tiles.end() && found->index == tile_index
             ? found->tile->State(TileCellOffset(index))
             : FineCellState::kUnknown;
}

[[nodiscard]] bool IsFrontier(
    const SparseGridGeometry& geometry,
    const std::vector<AllocatedFineTile>& tiles,
    const GridIndex index) noexcept {
  const GridIndex min = geometry.min_inclusive();
  const GridIndex max = geometry.max_exclusive();
  if (index.x == min.x || index.y == min.y || index.x == max.x - 1 ||
      index.y == max.y - 1) {
    return true;
  }
  for (std::int64_t dy = -1; dy <= 1; ++dy) {
    for (std::int64_t dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      const GridIndex neighbor{.x = index.x + dx, .y = index.y + dy};
      if (!geometry.Contains(neighbor) ||
          StateFromAllocatedTiles(tiles, neighbor) == FineCellState::kUnknown) {
        return true;
      }
    }
  }
  return false;
}

template <typename MakeCandidate, typename IsBetter>
[[nodiscard]] std::optional<Candidate> BestFreeCandidate(
    const FineTraversabilitySnapshot& fine,
    const SparseGridGeometry& geometry, MakeCandidate make_candidate,
    IsBetter is_better, const bool frontier_only,
    SelectionMetrics* metrics) {
  std::optional<Candidate> best;
  const GridIndex min = geometry.min_inclusive();
  const GridIndex max = geometry.max_exclusive();
  const TileIndex first_tile = TileForCell(min);
  const TileIndex last_tile = TileForCell(
      {.x = max.x - 1, .y = max.y - 1});
  std::vector<AllocatedFineTile> allocated_tiles;
  const auto append_tile = [&](const TileIndex tile_index) {
    Increment(metrics ? &metrics->candidate_tile_lookups : nullptr);
    const auto tile = fine.FindTile(tile_index);
    if (tile) {
      allocated_tiles.push_back(
          AllocatedFineTile{.index = tile_index, .tile = tile});
    }
  };
  if (geometry.width() <= kMaximumLocalPlanningWindowAxisCells &&
      geometry.height() <= kMaximumLocalPlanningWindowAxisCells) {
    // A session window intersects at most a few 256-cell tiles. Query those
    // tiles directly instead of walking all history allocated by the
    // persistent map.
    for (std::int64_t x = first_tile.x; x <= last_tile.x; ++x) {
      for (std::int64_t y = first_tile.y; y <= last_tile.y; ++y) {
        append_tile({.x = x, .y = y});
      }
    }
  } else {
    // Retain the sparse whole-snapshot compatibility path without scanning
    // empty holes in an unbounded historical geometry.
    for (const TileIndex tile_index : fine.tile_indices()) {
      if (tile_index.x < first_tile.x || tile_index.x > last_tile.x ||
          tile_index.y < first_tile.y || tile_index.y > last_tile.y) {
        continue;
      }
      append_tile(tile_index);
    }
  }
  for (const AllocatedFineTile& entry : allocated_tiles) {
    const TileIndex tile_index = entry.index;
    const auto& tile = entry.tile;
    const std::int64_t tile_min_x = tile_index.x * kGridTileWidthCells;
    const std::int64_t tile_min_y = tile_index.y * kGridTileWidthCells;
    const std::int64_t begin_x = std::max(min.x, tile_min_x);
    const std::int64_t begin_y = std::max(min.y, tile_min_y);
    const std::int64_t end_x =
        std::min(max.x, tile_min_x + kGridTileWidthCells);
    const std::int64_t end_y =
        std::min(max.y, tile_min_y + kGridTileWidthCells);
    for (std::int64_t y = begin_y; y < end_y; ++y) {
      for (std::int64_t x = begin_x; x < end_x; ++x) {
        Increment(metrics ? &metrics->candidate_cells_examined : nullptr);
        const GridIndex index{.x = x, .y = y};
        const std::size_t offset = TileCellOffset(index);
        if (tile->State(offset) != FineCellState::kFree ||
            (frontier_only &&
             !IsFrontier(geometry, allocated_tiles, index))) {
          continue;
        }
        const Candidate candidate = make_candidate(index, *tile, offset);
        if (!best || is_better(candidate, *best)) {
          best = candidate;
        }
      }
    }
  }
  return best;
}

[[nodiscard]] double PositionTolerance(
    const FineTraversabilitySnapshot& fine) noexcept {
  // Inflation makes cells non-traversable; it is not an arrival tolerance.
  // The selector owns only the unavoidable grid quantization component.  The
  // platform profile component is applied by the session coordinator, where
  // the profile and the selected local target meet.
  return 0.5 * fine.geometry().resolution_m();
}

}  // namespace

std::optional<LocalTarget> local_target_selector_internal::Select(
    const FineTraversabilitySnapshot& fine,
    const SparseGridGeometry& local_window, const Point2 start,
    const FinalGoal& final_goal,
    const std::optional<GlobalRoute>& guidance,
    local_target_selector_internal::SelectionMetrics* metrics) {
  if (metrics) {
    *metrics = local_target_selector_internal::SelectionMetrics{};
  }
  if (!IsLocalPlanningWindowFor(local_window, fine.geometry()) ||
      !Finite(start) ||
      !IsValidFinalGoal(final_goal)) {
    return std::nullopt;
  }

  const Point2 final_point{.x = final_goal.target_x_m,
                           .y = final_goal.target_y_m};
  const auto final_cell = CellAt(local_window, final_point);
  const double tolerance_m = PositionTolerance(fine);
  if (final_cell && local_window.Contains(*final_cell) &&
      fine.State(*final_cell) == FineCellState::kFree) {
    return LocalTarget{
        .center = final_point,
        .position_tolerance_m = tolerance_m,
        .is_final_goal = true,
        .terminal_yaw_rad = final_goal.has_target_yaw
                                ? std::optional<double>(
                                      final_goal.target_yaw_rad)
                                : std::nullopt,
    };
  }

  if (guidance) {
    const auto exit = GuidanceExit(fine, local_window, start, *guidance,
                                   metrics);
    if (exit) {
      const auto selected = BestFreeCandidate(
          fine, local_window,
          [&](const GridIndex index, const FineTraversabilityTile& tile,
              const std::size_t offset) {
            const Point2 center = CellCenter(local_window, index);
            return Candidate{.index = index,
                             .center = center,
                             .primary = SquaredDistance(center, *exit),
                             .traversal_cost = tile.TraversalCost(offset)};
          },
          BetterGuidanceCandidate, false, metrics);
      if (selected) {
        return LocalTarget{.center = selected->center,
                           .position_tolerance_m = tolerance_m};
      }
    }
  }

  const double direction_x = final_point.x - start.x;
  const double direction_y = final_point.y - start.y;
  const double direction_length = std::hypot(direction_x, direction_y);
  if (!std::isfinite(direction_length) || direction_length == 0.0) {
    return std::nullopt;
  }
  const Point2 direction{.x = direction_x / direction_length,
                         .y = direction_y / direction_length};
  const auto selected = BestFreeCandidate(
      fine, local_window,
      [&](const GridIndex index, const FineTraversabilityTile& tile,
          const std::size_t offset) {
        const Point2 center = CellCenter(local_window, index);
        const double offset_x = center.x - start.x;
        const double offset_y = center.y - start.y;
        return Candidate{
            .index = index,
            .center = center,
            .primary = offset_x * direction.x + offset_y * direction.y,
            .secondary = std::abs(offset_x * direction.y -
                                  offset_y * direction.x),
            .traversal_cost = tile.TraversalCost(offset),
        };
      },
      BetterDirectionalCandidate, true, metrics);
  if (!selected) {
    return std::nullopt;
  }
  return LocalTarget{.center = selected->center,
                     .position_tolerance_m = tolerance_m};
}

std::optional<LocalTarget> LocalTargetSelector::Select(
    const FineTraversabilitySnapshot& fine,
    const SparseGridGeometry& local_window, const Point2 start,
    const FinalGoal& final_goal,
    const std::optional<GlobalRoute>& guidance) const {
  return local_target_selector_internal::Select(
      fine, local_window, start, final_goal, guidance, nullptr);
}

std::optional<LocalTarget> LocalTargetSelector::Select(
    const FineTraversabilitySnapshot& fine, const Point2 start,
    const FinalGoal& final_goal,
    const std::optional<GlobalRoute>& guidance) const {
  return Select(fine, fine.geometry(), start, final_goal, guidance);
}

std::optional<LocalTarget> local_target_selector_internal::Select(
    const FineTraversabilitySnapshot& fine, const Point2 start,
    const FinalGoal& final_goal,
    const std::optional<GlobalRoute>& guidance, SelectionMetrics* metrics) {
  return Select(fine, fine.geometry(), start, final_goal, guidance, metrics);
}

}  // namespace lunar::incremental_navigation
