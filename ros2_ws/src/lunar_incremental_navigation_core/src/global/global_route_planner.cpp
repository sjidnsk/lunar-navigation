#include "lunar_incremental_navigation_core/global_route_planner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lunar::incremental_navigation {
namespace {

constexpr double kHeuristicWeight = 1.5;
constexpr double kDiagonalDistance = 1.4142135623730950488;
using SearchCost = long double;
constexpr SearchCost kMaximumSearchCost =
    std::numeric_limits<SearchCost>::max();

struct SearchBounds final {
  GridIndex min_inclusive;
  GridIndex max_inclusive;

  [[nodiscard]] bool Contains(const GridIndex index) const noexcept {
    return index.x >= min_inclusive.x && index.x <= max_inclusive.x &&
           index.y >= min_inclusive.y && index.y <= max_inclusive.y;
  }
};

struct Record final {
  SearchCost cost{std::numeric_limits<SearchCost>::infinity()};
  GridIndex parent;
  bool has_parent{};
  bool closed{};
};

struct OpenEntry final {
  GridIndex index;
  SearchCost cost{};
  SearchCost heuristic{};
  SearchCost priority{};
  std::uint64_t insertion_order{};
};

struct OpenEntryLater final {
  [[nodiscard]] bool operator()(const OpenEntry& left,
                                const OpenEntry& right) const noexcept {
    if (left.priority != right.priority) {
      return left.priority > right.priority;
    }
    if (left.heuristic != right.heuristic) {
      return left.heuristic > right.heuristic;
    }
    if (left.cost != right.cost) {
      return left.cost > right.cost;
    }
    if (left.index.y != right.index.y) {
      return left.index.y > right.index.y;
    }
    if (left.index.x != right.index.x) {
      return left.index.x > right.index.x;
    }
    return left.insertion_order > right.insertion_order;
  }
};

[[nodiscard]] bool SameGeometry(const SparseGridGeometry& left,
                                const SparseGridGeometry& right) noexcept {
  return left.frame_id() == right.frame_id() &&
         left.resolution_m() == right.resolution_m() &&
         left.origin_m() == right.origin_m() &&
         left.min_inclusive() == right.min_inclusive() &&
         left.max_exclusive() == right.max_exclusive();
}

[[nodiscard]] std::optional<GridIndex> WorldToCell(
    const SparseGridGeometry& geometry, const Point2 point) noexcept {
  if (!geometry.valid() || !std::isfinite(point.x) ||
      !std::isfinite(point.y)) {
    return std::nullopt;
  }
  const Vec3 origin = geometry.origin_m();
  const double grid_x = (point.x - origin.x) / geometry.resolution_m();
  const double grid_y = (point.y - origin.y) / geometry.resolution_m();
  if (!std::isfinite(grid_x) || !std::isfinite(grid_y)) {
    return std::nullopt;
  }
  const double cell_x = std::floor(grid_x);
  const double cell_y = std::floor(grid_y);
  if (cell_x < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      cell_x >= static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
      cell_y < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      cell_y >= static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    return std::nullopt;
  }
  const GridIndex index{.x = static_cast<std::int64_t>(cell_x),
                        .y = static_cast<std::int64_t>(cell_y)};
  return geometry.Contains(index) ? std::optional<GridIndex>(index)
                                  : std::nullopt;
}

[[nodiscard]] std::optional<std::int64_t> CheckedOffset(
    const std::int64_t value, const std::int64_t offset) noexcept {
  if ((offset < 0 &&
       value < std::numeric_limits<std::int64_t>::min() - offset) ||
      (offset > 0 &&
       value > std::numeric_limits<std::int64_t>::max() - offset)) {
    return std::nullopt;
  }
  return value + offset;
}

[[nodiscard]] std::optional<GridIndex> OffsetCell(
    const GridIndex index, const std::int64_t dx,
    const std::int64_t dy) noexcept {
  const std::optional<std::int64_t> x = CheckedOffset(index.x, dx);
  const std::optional<std::int64_t> y = CheckedOffset(index.y, dy);
  if (!x || !y) {
    return std::nullopt;
  }
  return GridIndex{.x = *x, .y = *y};
}

[[nodiscard]] Point2 CellCenter(const SparseGridGeometry& geometry,
                                const GridIndex index) noexcept {
  const Vec3 origin = geometry.origin_m();
  const double resolution = geometry.resolution_m();
  return Point2{
      .x = std::fma(static_cast<double>(index.x) + 0.5, resolution, origin.x),
      .y = std::fma(static_cast<double>(index.y) + 0.5, resolution, origin.y),
  };
}

[[nodiscard]] SearchCost SaturatingAdd(const SearchCost left,
                                       const SearchCost right) noexcept {
  if (left >= kMaximumSearchCost - right) {
    return kMaximumSearchCost;
  }
  return left + right;
}

[[nodiscard]] SearchCost SaturatingMultiply(
    const SearchCost left, const SearchCost right) noexcept {
  if (left == 0.0L || right == 0.0L) {
    return 0.0L;
  }
  if (left >= kMaximumSearchCost / right) {
    return kMaximumSearchCost;
  }
  return left * right;
}

[[nodiscard]] SearchCost OctileHeuristic(
    const GridIndex from, const GridIndex goal,
    const double resolution_m) noexcept {
  const auto absolute_difference = [](const std::int64_t left,
                                      const std::int64_t right) {
    const long double delta = static_cast<long double>(left) -
                              static_cast<long double>(right);
    return std::abs(delta);
  };
  const SearchCost dx = absolute_difference(from.x, goal.x);
  const SearchCost dy = absolute_difference(from.y, goal.y);
  const SearchCost diagonal = std::min(dx, dy);
  const SearchCost straight = std::max(dx, dy) - diagonal;
  const SearchCost grid_distance = SaturatingAdd(
      SaturatingMultiply(static_cast<SearchCost>(kDiagonalDistance),
                         diagonal),
      straight);
  return SaturatingMultiply(grid_distance,
                            static_cast<SearchCost>(resolution_m));
}

[[nodiscard]] SearchBounds MakeSearchBounds(
    const SparseGridGeometry& geometry, const GridIndex start,
    const GridIndex goal, const double detour_margin_m) {
  const long double margin_cells_double =
      std::ceil(static_cast<long double>(detour_margin_m) /
                static_cast<long double>(geometry.resolution_m()));
  const long double int64_exclusive_upper = std::ldexp(1.0L, 63);
  if (!std::isfinite(margin_cells_double) ||
      margin_cells_double >= int64_exclusive_upper) {
    throw std::invalid_argument("global detour margin exceeds grid range");
  }
  const std::int64_t margin_cells =
      static_cast<std::int64_t>(margin_cells_double);
  const auto saturating_subtract = [](const std::int64_t value,
                                      const std::int64_t amount) {
    return value < std::numeric_limits<std::int64_t>::min() + amount
               ? std::numeric_limits<std::int64_t>::min()
               : value - amount;
  };
  const auto saturating_add = [](const std::int64_t value,
                                 const std::int64_t amount) {
    return value > std::numeric_limits<std::int64_t>::max() - amount
               ? std::numeric_limits<std::int64_t>::max()
               : value + amount;
  };
  const GridIndex geometry_max{
      .x = geometry.max_exclusive().x - 1,
      .y = geometry.max_exclusive().y - 1,
  };
  return SearchBounds{
      .min_inclusive =
          {.x = std::max(geometry.min_inclusive().x,
                         saturating_subtract(std::min(start.x, goal.x),
                                             margin_cells)),
           .y = std::max(geometry.min_inclusive().y,
                         saturating_subtract(std::min(start.y, goal.y),
                                             margin_cells))},
      .max_inclusive =
          {.x = std::min(geometry_max.x,
                         saturating_add(std::max(start.x, goal.x),
                                        margin_cells)),
           .y = std::min(geometry_max.y,
                         saturating_add(std::max(start.y, goal.y),
                                        margin_cells))},
  };
}

[[nodiscard]] bool TimedOut(const SearchDeadline deadline,
                            const StopToken& stop,
                            const NowFn& now) {
  return stop.stop_requested() || now() >= deadline;
}

[[nodiscard]] bool IsBlocked(const GlobalGuidanceSnapshot& snapshot,
                             const GridIndex index) noexcept {
  return snapshot.State(index) == GuidanceCellState::kProvenBlocked;
}

[[nodiscard]] bool DiagonalCrossesBlockedCorner(
    const GlobalGuidanceSnapshot& snapshot, const GridIndex from,
    const std::int64_t dx, const std::int64_t dy) noexcept {
  if (dx == 0 || dy == 0) {
    return false;
  }
  const std::optional<GridIndex> horizontal = OffsetCell(from, dx, 0);
  const std::optional<GridIndex> vertical = OffsetCell(from, 0, dy);
  return !horizontal || !vertical || IsBlocked(snapshot, *horizontal) ||
         IsBlocked(snapshot, *vertical);
}

[[nodiscard]] std::optional<GlobalRoute> BuildRoute(
    const SparseGridGeometry& geometry, const std::span<const GridIndex> cells,
    const Point2 start, const Point2 goal,
    const std::uint64_t expanded_states, const SearchDeadline deadline,
    const StopToken& stop, const NowFn& now) {
  if (TimedOut(deadline, stop, now)) {
    return std::nullopt;
  }
  GlobalRoute route;
  route.expanded_states = expanded_states;
  route.poses_map.reserve(std::max<std::size_t>(2U, cells.size()));
  route.poses_map.push_back(
      Pose3{.position_m = Vec3{.x = start.x, .y = start.y, .z = 0.0}});
  for (std::size_t index = 1U; index + 1U < cells.size(); ++index) {
    if (TimedOut(deadline, stop, now)) {
      return std::nullopt;
    }
    const Point2 center = CellCenter(geometry, cells[index]);
    route.poses_map.push_back(
        Pose3{.position_m = Vec3{.x = center.x, .y = center.y, .z = 0.0}});
  }
  if (start != goal) {
    if (TimedOut(deadline, stop, now)) {
      return std::nullopt;
    }
    route.poses_map.push_back(
        Pose3{.position_m = Vec3{.x = goal.x, .y = goal.y, .z = 0.0}});
  }
  if (TimedOut(deadline, stop, now)) {
    return std::nullopt;
  }
  return route;
}

struct CellFingerprint final {
  GridIndex index;
  GuidanceCellState state{GuidanceCellState::kUnknown};
  double terrain_risk{};
};

[[nodiscard]] std::vector<CellFingerprint> RouteInfluenceFingerprint(
    const GlobalGuidanceSnapshot& snapshot,
    const std::vector<GridIndex>& cells, const SearchDeadline deadline,
    const StopToken& stop, const NowFn& now, bool& timed_out) {
  std::set<GridIndex> influence_cells;
  for (const GridIndex cell : cells) {
    if (TimedOut(deadline, stop, now)) {
      timed_out = true;
      return {};
    }
    for (std::int64_t dy = -1; dy <= 1; ++dy) {
      for (std::int64_t dx = -1; dx <= 1; ++dx) {
        const std::optional<GridIndex> neighbor = OffsetCell(cell, dx, dy);
        if (neighbor && snapshot.geometry().Contains(*neighbor)) {
          influence_cells.insert(*neighbor);
        }
      }
    }
  }
  std::vector<CellFingerprint> fingerprint;
  fingerprint.reserve(influence_cells.size());
  for (const GridIndex index : influence_cells) {
    if (TimedOut(deadline, stop, now)) {
      timed_out = true;
      return {};
    }
    fingerprint.push_back(CellFingerprint{
        .index = index,
        .state = snapshot.State(index),
        .terrain_risk = snapshot.TerrainRisk(index),
    });
  }
  if (TimedOut(deadline, stop, now)) {
    timed_out = true;
    return {};
  }
  return fingerprint;
}

enum class InfluenceMatch : std::uint8_t {
  kMatch,
  kMismatch,
  kTimeout,
};

[[nodiscard]] InfluenceMatch MatchesRouteInfluence(
    const GlobalGuidanceSnapshot& snapshot,
    const std::vector<CellFingerprint>& fingerprint,
    const SearchDeadline deadline, const StopToken& stop,
    const NowFn& now) {
  for (const CellFingerprint& cell : fingerprint) {
    if (TimedOut(deadline, stop, now)) {
      return InfluenceMatch::kTimeout;
    }
    if (snapshot.State(cell.index) != cell.state ||
        snapshot.TerrainRisk(cell.index) != cell.terrain_risk) {
      return InfluenceMatch::kMismatch;
    }
  }
  return InfluenceMatch::kMatch;
}

}  // namespace

struct GlobalRoutePlanner::Impl final {
  struct CacheEntry final {
    std::uint64_t guidance_revision{};
    GridIndex goal_cell;
    std::string platform_profile_hash;
    SparseGridGeometry geometry;
    GlobalGuidanceTileDirectory tile_directory;
    std::vector<GridIndex> cells;
    std::vector<CellFingerprint> influence_fingerprint;
  };

  explicit Impl(const GlobalRoutePlannerConfig planner_config)
      : config(planner_config) {
    if (!std::isfinite(config.global_detour_margin_m) ||
        config.global_detour_margin_m < 0.0 ||
        !std::isfinite(config.unknown_step_risk) ||
        config.unknown_step_risk <= 0.0 || !config.now) {
      throw std::invalid_argument(
          "global route configuration requires finite nonnegative detour "
          "margin, positive UNKNOWN risk and a monotonic clock");
    }
  }

  GlobalRoutePlannerConfig config;
  std::optional<CacheEntry> cache;
};

GlobalRoutePlanner::GlobalRoutePlanner(const GlobalRoutePlannerConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

GlobalRoutePlanner::~GlobalRoutePlanner() = default;
GlobalRoutePlanner::GlobalRoutePlanner(GlobalRoutePlanner&&) noexcept =
    default;
GlobalRoutePlanner& GlobalRoutePlanner::operator=(
    GlobalRoutePlanner&&) noexcept = default;

GlobalRouteResult GlobalRoutePlanner::Plan(
    const GlobalGuidanceSnapshot& snapshot, const Point2 start,
    const Point2 goal, const SearchDeadline deadline,
    const StopToken& stop) {
  const std::optional<GridIndex> start_cell =
      WorldToCell(snapshot.geometry(), start);
  const std::optional<GridIndex> goal_cell =
      WorldToCell(snapshot.geometry(), goal);
  if (!start_cell || !goal_cell) {
    return GlobalRouteResult{.status = GuidanceStatus::kUnavailable};
  }
  if (TimedOut(deadline, stop, impl_->config.now)) {
    return GlobalRouteResult{.status = GuidanceStatus::kTimeout};
  }
  if (IsBlocked(snapshot, *start_cell) || IsBlocked(snapshot, *goal_cell)) {
    return GlobalRouteResult{.status = GuidanceStatus::kNoRoute};
  }

  if (impl_->cache && impl_->cache->goal_cell == *goal_cell &&
      impl_->cache->platform_profile_hash == snapshot.platform_profile_hash() &&
      SameGeometry(impl_->cache->geometry, snapshot.geometry())) {
    std::optional<std::size_t> start_offset;
    for (std::size_t index = 0U; index < impl_->cache->cells.size(); ++index) {
      if (TimedOut(deadline, stop, impl_->config.now)) {
        return GlobalRouteResult{.status = GuidanceStatus::kTimeout};
      }
      if (impl_->cache->cells[index] == *start_cell) {
        start_offset = index;
        break;
      }
    }
    if (start_offset.has_value()) {
      const bool same_revision =
          impl_->cache->guidance_revision ==
          snapshot.global_guidance_revision();
      const bool successor_revision =
          impl_->cache->guidance_revision <
              std::numeric_limits<std::uint64_t>::max() &&
          snapshot.global_guidance_revision() ==
              impl_->cache->guidance_revision + 1U;
      bool reusable =
          same_revision && impl_->cache->tile_directory.shares_root_with(
                               snapshot.tile_directory());
      if (!reusable && (same_revision || successor_revision)) {
        const InfluenceMatch match = MatchesRouteInfluence(
            snapshot, impl_->cache->influence_fingerprint, deadline, stop,
            impl_->config.now);
        if (match == InfluenceMatch::kTimeout) {
          return GlobalRouteResult{.status = GuidanceStatus::kTimeout};
        }
        reusable = match == InfluenceMatch::kMatch;
      }
      if (reusable) {
        const std::span<const GridIndex> cached_cells(impl_->cache->cells);
        std::optional<GlobalRoute> route = BuildRoute(
            snapshot.geometry(), cached_cells.subspan(*start_offset), start,
            goal, 0U, deadline, stop, impl_->config.now);
        if (!route) {
          return GlobalRouteResult{.status = GuidanceStatus::kTimeout};
        }
        impl_->cache->guidance_revision =
            snapshot.global_guidance_revision();
        impl_->cache->tile_directory = snapshot.tile_directory();
        return GlobalRouteResult{
            .status = GuidanceStatus::kAvailable,
            .route = std::move(route),
            .reused_cache = true,
        };
      }
    }
  }

  const SearchBounds bounds = MakeSearchBounds(
      snapshot.geometry(), *start_cell, *goal_cell,
      impl_->config.global_detour_margin_m);
  std::map<GridIndex, Record> records;
  std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenEntryLater> open;
  SearchStatistics statistics;
  std::uint64_t insertion_order = 0U;
  Record& start_record = records[*start_cell];
  start_record.cost = 0.0;
  const SearchCost start_heuristic = OctileHeuristic(
      *start_cell, *goal_cell, snapshot.geometry().resolution_m());
  open.push(OpenEntry{.index = *start_cell,
                      .cost = 0.0,
                      .heuristic = start_heuristic,
                      .priority = SaturatingMultiply(
                          static_cast<SearchCost>(kHeuristicWeight),
                          start_heuristic),
                      .insertion_order = insertion_order++});
  statistics.open_peak = open.size();

  constexpr std::array<GridIndex, 8U> kNeighborOffsets{{
      {.x = -1, .y = -1},
      {.x = 0, .y = -1},
      {.x = 1, .y = -1},
      {.x = -1, .y = 0},
      {.x = 1, .y = 0},
      {.x = -1, .y = 1},
      {.x = 0, .y = 1},
      {.x = 1, .y = 1},
  }};

  bool found = false;
  while (!open.empty()) {
    if (TimedOut(deadline, stop, impl_->config.now)) {
      return GlobalRouteResult{.status = GuidanceStatus::kTimeout,
                               .statistics = statistics};
    }
    const OpenEntry current = open.top();
    open.pop();
    Record& current_record = records[current.index];
    if (current_record.closed || current.cost != current_record.cost) {
      continue;
    }
    current_record.closed = true;
    ++statistics.expanded_states;
    if (current.index == *goal_cell) {
      found = true;
      break;
    }

    for (const GridIndex offset : kNeighborOffsets) {
      const std::optional<GridIndex> neighbor =
          OffsetCell(current.index, offset.x, offset.y);
      if (!neighbor || !bounds.Contains(*neighbor) ||
          !snapshot.geometry().Contains(*neighbor) ||
          IsBlocked(snapshot, *neighbor) ||
          DiagonalCrossesBlockedCorner(snapshot, current.index, offset.x,
                                       offset.y)) {
        continue;
      }
      const GuidanceCellState state = snapshot.State(*neighbor);
      const SearchCost geometric_step =
          (offset.x != 0 && offset.y != 0 ? kDiagonalDistance : 1.0) *
          snapshot.geometry().resolution_m();
      const SearchCost unknown_risk =
          state == GuidanceCellState::kUnknown
              ? static_cast<SearchCost>(impl_->config.unknown_step_risk)
              : 0.0L;
      const SearchCost tentative_cost = SaturatingAdd(
          SaturatingAdd(
              SaturatingAdd(current_record.cost, geometric_step),
              static_cast<SearchCost>(snapshot.TerrainRisk(*neighbor))),
          unknown_risk);
      Record& neighbor_record = records[*neighbor];
      if (neighbor_record.closed || tentative_cost >= neighbor_record.cost) {
        continue;
      }
      neighbor_record.cost = tentative_cost;
      neighbor_record.parent = current.index;
      neighbor_record.has_parent = true;
      const SearchCost heuristic = OctileHeuristic(
          *neighbor, *goal_cell, snapshot.geometry().resolution_m());
      open.push(OpenEntry{
          .index = *neighbor,
          .cost = tentative_cost,
          .heuristic = heuristic,
          .priority = SaturatingAdd(
              tentative_cost,
              SaturatingMultiply(
                  static_cast<SearchCost>(kHeuristicWeight), heuristic)),
          .insertion_order = insertion_order++,
      });
      statistics.open_peak = std::max(statistics.open_peak, open.size());
      ++statistics.generated_states;
    }
  }

  if (!found) {
    return GlobalRouteResult{.status = GuidanceStatus::kNoRoute,
                             .statistics = statistics};
  }

  std::vector<GridIndex> cells;
  for (GridIndex current = *goal_cell;;) {
    if (TimedOut(deadline, stop, impl_->config.now)) {
      return GlobalRouteResult{.status = GuidanceStatus::kTimeout,
                               .statistics = statistics};
    }
    cells.push_back(current);
    if (current == *start_cell) {
      break;
    }
    const auto found_record = records.find(current);
    if (found_record == records.end() || !found_record->second.has_parent) {
      return GlobalRouteResult{.status = GuidanceStatus::kNoRoute,
                               .statistics = statistics};
    }
    current = found_record->second.parent;
  }
  if (!cells.empty()) {
    std::size_t left = 0U;
    std::size_t right = cells.size() - 1U;
    while (left < right) {
      if (TimedOut(deadline, stop, impl_->config.now)) {
        return GlobalRouteResult{.status = GuidanceStatus::kTimeout,
                                 .statistics = statistics};
      }
      std::swap(cells[left], cells[right]);
      ++left;
      --right;
    }
  }

  bool fingerprint_timed_out = false;
  std::vector<CellFingerprint> influence_fingerprint =
      RouteInfluenceFingerprint(snapshot, cells, deadline, stop,
                                impl_->config.now, fingerprint_timed_out);
  if (fingerprint_timed_out) {
    return GlobalRouteResult{.status = GuidanceStatus::kTimeout,
                             .statistics = statistics};
  }
  std::optional<GlobalRoute> route = BuildRoute(
      snapshot.geometry(), cells, start, goal, statistics.expanded_states,
      deadline, stop, impl_->config.now);
  if (!route) {
    return GlobalRouteResult{.status = GuidanceStatus::kTimeout,
                             .statistics = statistics};
  }
  Impl::CacheEntry next_cache{
      .guidance_revision = snapshot.global_guidance_revision(),
      .goal_cell = *goal_cell,
      .platform_profile_hash = snapshot.platform_profile_hash(),
      .geometry = snapshot.geometry(),
      .tile_directory = snapshot.tile_directory(),
      .cells = std::move(cells),
      .influence_fingerprint = std::move(influence_fingerprint),
  };
  impl_->cache = std::move(next_cache);
  return GlobalRouteResult{
      .status = GuidanceStatus::kAvailable,
      .route = std::move(route),
      .reused_cache = false,
      .statistics = statistics,
  };
}

}  // namespace lunar::incremental_navigation
