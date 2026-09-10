#include "lunar_incremental_navigation_core/fine_traversability_builder.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "map/grid_bounds.hpp"

namespace lunar::incremental_navigation {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] bool FiniteNonnegative(const double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] Point2 CellCenter(const SparseGridGeometry& geometry,
                                const GridIndex index) noexcept {
  const Vec3 origin = geometry.origin_m();
  const double resolution_m = geometry.resolution_m();
  return Point2{
      .x = origin.x + (static_cast<double>(index.x) + 0.5) * resolution_m,
      .y = origin.y + (static_cast<double>(index.y) + 0.5) * resolution_m,
  };
}

[[nodiscard]] bool PointOnSegment(const Point2 point, const Point2 start,
                                  const Point2 end) noexcept {
  const double cross = (point.x - start.x) * (end.y - start.y) -
                       (point.y - start.y) * (end.x - start.x);
  if (std::abs(cross) > 1.0e-12) {
    return false;
  }
  const double dot = (point.x - start.x) * (point.x - end.x) +
                     (point.y - start.y) * (point.y - end.y);
  return dot <= 0.0;
}

[[nodiscard]] bool EnvelopeContainsOrigin(
    const std::span<const Point2> envelope) noexcept {
  const Point2 origin{};
  bool inside = false;
  for (std::size_t current = 0U, previous = envelope.size() - 1U;
       current < envelope.size(); previous = current++) {
    const Point2 a = envelope[previous];
    const Point2 b = envelope[current];
    if (PointOnSegment(origin, a, b)) {
      return true;
    }
    const bool crosses = (a.y > 0.0) != (b.y > 0.0);
    if (crosses) {
      const double crossing_x =
          a.x + (b.x - a.x) * (-a.y) / (b.y - a.y);
      if (crossing_x > 0.0) {
        inside = !inside;
      }
    }
  }
  return inside;
}

void ValidateProfile(const TraversabilityProfile& profile) {
  if (profile.planar_envelope_xy_m.size() < 3U ||
      !FiniteNonnegative(profile.preferred_clearance_m) ||
      !FiniteNonnegative(profile.slope_weight) ||
      !FiniteNonnegative(profile.relief_weight) ||
      !FiniteNonnegative(profile.clearance_weight)) {
    throw std::invalid_argument(
        "traversability profile parameters must be finite and nonnegative");
  }
  double twice_area = 0.0;
  for (std::size_t index = 0U; index < profile.planar_envelope_xy_m.size();
       ++index) {
    const Point2 current = profile.planar_envelope_xy_m[index];
    const Point2 next = profile.planar_envelope_xy_m[
        (index + 1U) % profile.planar_envelope_xy_m.size()];
    if (!std::isfinite(current.x) || !std::isfinite(current.y)) {
      throw std::invalid_argument("planar envelope vertices must be finite");
    }
    twice_area += current.x * next.y - current.y * next.x;
  }
  if (!std::isfinite(twice_area) || std::abs(twice_area) <= 1.0e-12 ||
      !EnvelopeContainsOrigin(profile.planar_envelope_xy_m)) {
    throw std::invalid_argument(
        "planar envelope must be nondegenerate and contain the origin");
  }
}

class ProfileHasher final {
 public:
  void Add(const std::uint64_t value) noexcept {
    for (unsigned int shift = 0U; shift < 64U; shift += 8U) {
      hash_ ^= (value >> shift) & 0xffU;
      hash_ *= 1099511628211ULL;
    }
  }

  void Add(const double value) noexcept { Add(std::bit_cast<std::uint64_t>(value)); }

  [[nodiscard]] std::string Finish() const {
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << hash_;
    return stream.str();
  }

 private:
  std::uint64_t hash_{1469598103934665603ULL};
};

[[nodiscard]] std::string PlatformProfileHash(
    const PlatformCapability& capability,
    const TraversabilityProfile& profile) {
  ProfileHasher hasher;
  hasher.Add(static_cast<std::uint64_t>(CapabilityPlatform(capability)));
  for (const Point2 vertex : profile.planar_envelope_xy_m) {
    hasher.Add(vertex.x);
    hasher.Add(vertex.y);
  }
  hasher.Add(profile.preferred_clearance_m);
  hasher.Add(profile.slope_weight);
  hasher.Add(profile.relief_weight);
  hasher.Add(profile.clearance_weight);
  std::visit(
      [&hasher](const auto& typed_capability) {
        using Capability = std::decay_t<decltype(typed_capability)>;
        if constexpr (std::is_same_v<Capability, WheeledCapability>) {
          hasher.Add(typed_capability.maximum_slope_rad);
          hasher.Add(typed_capability.maximum_local_obstacle_relief_m);
          hasher.Add(typed_capability.minimum_underbody_clearance_m);
          hasher.Add(static_cast<std::uint64_t>(
              typed_capability.allow_unsupported_gap));
        } else if constexpr (std::is_same_v<Capability, LeggedCapability>) {
          hasher.Add(typed_capability.maximum_slope_rad);
          hasher.Add(typed_capability.maximum_step_height_m);
          hasher.Add(typed_capability.maximum_gap_width_m);
        }
      },
      capability);
  return hasher.Finish();
}

[[nodiscard]] std::set<TileIndex> CandidateTiles(
    const ElevationSnapshot& raw, const double hard_radius_m) {
  std::set<TileIndex> result;
  const SparseGridGeometry& geometry = raw.geometry();
  const std::int64_t cell_halo =
      detail::SaturatingCellExtent(hard_radius_m, geometry.resolution_m(), 1);
  for (const TileIndex raw_tile : raw.tile_indices()) {
    const std::optional<std::int64_t> raw_min_x =
        detail::CheckedMultiply(raw_tile.x, kGridTileWidthCells);
    const std::optional<std::int64_t> raw_min_y =
        detail::CheckedMultiply(raw_tile.y, kGridTileWidthCells);
    if (!raw_min_x || !raw_min_y) {
      continue;
    }
    const GridIndex raw_min{
        .x = *raw_min_x,
        .y = *raw_min_y,
    };
    const GridIndex raw_max{
        .x = detail::CheckedAdd(raw_min.x, kGridTileWidthCells)
                 .value_or(std::numeric_limits<std::int64_t>::max()),
        .y = detail::CheckedAdd(raw_min.y, kGridTileWidthCells)
                 .value_or(std::numeric_limits<std::int64_t>::max()),
    };
    const GridIndex expanded_min{
        .x = detail::ClampSubtract(raw_min.x, cell_halo,
                                   geometry.min_inclusive().x),
        .y = detail::ClampSubtract(raw_min.y, cell_halo,
                                   geometry.min_inclusive().y),
    };
    const GridIndex expanded_max{
        .x = detail::ClampAdd(raw_max.x, cell_halo,
                              geometry.max_exclusive().x),
        .y = detail::ClampAdd(raw_max.y, cell_halo,
                              geometry.max_exclusive().y),
    };
    if (expanded_min.x >= expanded_max.x ||
        expanded_min.y >= expanded_max.y) {
      continue;
    }
    const TileIndex tile_min = TileForCell(expanded_min);
    const TileIndex tile_max = TileForCell(
        GridIndex{.x = expanded_max.x - 1, .y = expanded_max.y - 1});
    for (std::int64_t tile_y = tile_min.y;; ++tile_y) {
      for (std::int64_t tile_x = tile_min.x;; ++tile_x) {
        result.insert(TileIndex{.x = tile_x, .y = tile_y});
        if (tile_x == tile_max.x) {
          break;
        }
      }
      if (tile_y == tile_max.y) {
        break;
      }
    }
  }
  return result;
}

[[nodiscard]] bool SameCanonicalLattice(
    const SparseGridGeometry& previous,
    const SparseGridGeometry& current) noexcept {
  return previous.frame_id() == current.frame_id() &&
         previous.resolution_m() == current.resolution_m() &&
         previous.origin_m() == current.origin_m() &&
         current.min_inclusive().x <= previous.min_inclusive().x &&
         current.min_inclusive().y <= previous.min_inclusive().y &&
         current.max_exclusive().x >= previous.max_exclusive().x &&
         current.max_exclusive().y >= previous.max_exclusive().y;
}

void ValidateMergedChanges(
    const ElevationSnapshot& raw,
    const FineTraversabilitySnapshot& previous,
    const FineElevationChangeSet& changes,
    const std::string& profile_hash) {
  if (previous.platform_profile_hash() != profile_hash ||
      changes.base_raw_elevation_revision !=
          previous.raw_elevation_revision() ||
      raw.raw_elevation_revision() <=
          changes.base_raw_elevation_revision ||
      changes.changed_cells.empty() ||
      !raw.SharesLineageWith(*previous.elevation()) ||
      !SameCanonicalLattice(previous.elevation()->geometry(),
                            raw.geometry())) {
    throw std::invalid_argument(
        "merged fine changes do not continue the previous raw lineage");
  }
  for (const GridIndex index : changes.changed_cells) {
    if (!raw.geometry().Contains(index)) {
      throw std::invalid_argument(
          "merged fine changes contain a cell outside the latest raw map");
    }
  }
}

[[nodiscard]] std::set<GridIndex> InfluenceCells(
    const SparseGridGeometry& geometry,
    const std::set<GridIndex>& changed_raw_cells,
    const double influence_m) {
  std::set<GridIndex> result;
  for (const GridIndex changed : changed_raw_cells) {
    const auto [minimum, maximum] =
        detail::CellBoundsForRadius(geometry, changed, influence_m);
    for (std::int64_t y = minimum.y; y < maximum.y; ++y) {
      for (std::int64_t x = minimum.x; x < maximum.x; ++x) {
        const GridIndex candidate{.x = x, .y = y};
        if (CircleIntersectsCellArea(CellCenter(geometry, candidate),
                                     influence_m, geometry, changed)) {
          result.insert(candidate);
        }
      }
    }
  }
  return result;
}

void IncludeNewGeometryCells(const ElevationSnapshot& raw,
                             const SparseGridGeometry& previous,
                             const double hard_radius_m,
                             std::set<GridIndex>& recompute_cells) {
  const auto& current = raw.geometry();
  const auto old_minimum = previous.min_inclusive();
  const auto old_maximum = previous.max_exclusive();
  if (current.min_inclusive() == old_minimum &&
      current.max_exclusive() == old_maximum) {
    return;
  }
  const auto include_rectangle = [&recompute_cells](const GridIndex minimum,
                                                    const GridIndex maximum) {
    for (auto y = minimum.y; y < maximum.y; ++y) {
      for (auto x = minimum.x; x < maximum.x; ++x) {
        recompute_cells.insert(GridIndex{.x = x, .y = y});
      }
    }
  };
  // Growth can admit cells in an unchanged obstacle's inflation halo. They
  // were never derived under the old bounds and are absent from changed_raw.
  // Visit only sparse candidate tiles, then subtract the old bounds; a distant
  // observation must not make us walk the entire newly enclosed rectangle.
  for (const auto tile : CandidateTiles(raw, hard_radius_m)) {
    const auto tile_x = detail::CheckedMultiply(tile.x, kGridTileWidthCells);
    const auto tile_y = detail::CheckedMultiply(tile.y, kGridTileWidthCells);
    if (!tile_x || !tile_y) {
      throw std::overflow_error("fine tile origin exceeds int64 grid");
    }
    const GridIndex minimum{
        .x = std::max(*tile_x, current.min_inclusive().x),
        .y = std::max(*tile_y, current.min_inclusive().y)};
    const GridIndex maximum{
        .x = std::min(detail::CheckedAdd(*tile_x, kGridTileWidthCells)
                          .value_or(std::numeric_limits<std::int64_t>::max()),
                      current.max_exclusive().x),
        .y = std::min(detail::CheckedAdd(*tile_y, kGridTileWidthCells)
                          .value_or(std::numeric_limits<std::int64_t>::max()),
                      current.max_exclusive().y)};
    if (minimum.x >= maximum.x || minimum.y >= maximum.y) {
      continue;
    }
    const GridIndex overlap_minimum{.x = std::max(minimum.x, old_minimum.x),
                                    .y = std::max(minimum.y, old_minimum.y)};
    const GridIndex overlap_maximum{.x = std::min(maximum.x, old_maximum.x),
                                    .y = std::min(maximum.y, old_maximum.y)};
    if (overlap_minimum.x >= overlap_maximum.x ||
        overlap_minimum.y >= overlap_maximum.y) {
      include_rectangle(minimum, maximum);
      continue;
    }
    include_rectangle(minimum, {maximum.x, overlap_minimum.y});
    include_rectangle({minimum.x, overlap_maximum.y}, maximum);
    include_rectangle({minimum.x, overlap_minimum.y},
                      {overlap_minimum.x, overlap_maximum.y});
    include_rectangle({overlap_maximum.x, overlap_minimum.y},
                      {maximum.x, overlap_maximum.y});
  }
}

[[nodiscard]] bool TileIsUnobserved(
    const FineTraversabilityTile::StateArray& states,
    const FineTraversabilityTile::CostArray& costs) {
  for (std::size_t offset = 0U; offset < kGridTileCellCount; ++offset) {
    if (states[offset] != FineCellState::kUnknown || costs[offset] != 0.0) {
      return false;
    }
  }
  return true;
}

}  // namespace

double CircumscribedRadius(const std::span<const Point2> envelope) {
  if (envelope.empty()) {
    throw std::invalid_argument("planar envelope must not be empty");
  }
  double radius_m = 0.0;
  for (const Point2 point : envelope) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
      throw std::invalid_argument("planar envelope vertices must be finite");
    }
    radius_m = std::max(radius_m, std::hypot(point.x, point.y));
  }
  return radius_m;
}

bool CircleIntersectsCellArea(const Point2 circle_center_m,
                              const double radius_m,
                              const SparseGridGeometry& geometry,
                              const GridIndex cell) noexcept {
  if (!geometry.valid() || !geometry.Contains(cell) ||
      !std::isfinite(circle_center_m.x) ||
      !std::isfinite(circle_center_m.y) || !FiniteNonnegative(radius_m)) {
    return false;
  }
  const Vec3 origin = geometry.origin_m();
  const double resolution_m = geometry.resolution_m();
  const double center_x_cells =
      (circle_center_m.x - origin.x) / resolution_m;
  const double center_y_cells =
      (circle_center_m.y - origin.y) / resolution_m;
  const double minimum_x = static_cast<double>(cell.x);
  const double minimum_y = static_cast<double>(cell.y);
  const double dx = std::max(
      {minimum_x - center_x_cells, 0.0, center_x_cells - minimum_x - 1.0});
  const double dy = std::max(
      {minimum_y - center_y_cells, 0.0, center_y_cells - minimum_y - 1.0});
  return std::hypot(dx, dy) <= radius_m / resolution_m;
}

std::shared_ptr<const FineTraversabilitySnapshot>
FineTraversabilityBuilder::Derive(
    std::shared_ptr<const ElevationSnapshot> raw,
    const PlatformCapability& capability,
    const TraversabilityProfile& profile,
    std::shared_ptr<const FineTraversabilitySnapshot> previous,
    std::optional<FineElevationChangeSet> merged_changes) const {
  const Clock::time_point started = Clock::now();
  if (!raw || !raw->valid()) {
    throw std::invalid_argument(
        "fine traversability requires a valid elevation snapshot");
  }
  ValidateProfile(profile);
  FineCellEvaluator cell_evaluator(*raw, capability, profile);
  const double hard_radius_m = cell_evaluator.hard_inflation_radius_m();
  const std::string profile_hash = PlatformProfileHash(capability, profile);
  if (merged_changes) {
    if (!previous) {
      throw std::invalid_argument(
          "merged fine changes require a previous fine snapshot");
    }
    ValidateMergedChanges(*raw, *previous, *merged_changes, profile_hash);
  }
  if (previous && previous->elevation() == raw &&
      previous->platform_profile_hash() == profile_hash) {
    return previous;
  }
  const bool incremental = previous &&
      previous->platform_profile_hash() == profile_hash &&
      (raw->IsDirectSuccessorOf(*previous->elevation()) ||
       merged_changes.has_value());
  std::set<GridIndex> recompute_cells;
  if (incremental) {
    const std::span<const GridIndex> changed =
        merged_changes ? merged_changes->changed_cells : raw->changed_cells();
    const std::set<GridIndex> changed_raw(changed.begin(), changed.end());
    const double influence_m =
        hard_radius_m + profile.preferred_clearance_m +
        std::sqrt(2.0) * raw->geometry().resolution_m();
    recompute_cells =
        InfluenceCells(raw->geometry(), changed_raw, influence_m);
    IncludeNewGeometryCells(*raw, previous->geometry(), hard_radius_m,
                             recompute_cells);
  }

  std::set<TileIndex> halo_tiles;
  std::map<TileIndex, std::vector<GridIndex>> recompute_by_tile;
  for (const GridIndex cell : recompute_cells) {
    const TileIndex tile = TileForCell(cell);
    halo_tiles.insert(tile);
    recompute_by_tile[tile].push_back(cell);
  }
  std::set<TileIndex> tiles_to_process;
  if (!incremental) {
    tiles_to_process = CandidateTiles(*raw, hard_radius_m);
    halo_tiles = tiles_to_process;
  } else {
    tiles_to_process = halo_tiles;
  }

  FineTraversabilityTileDirectory directory =
      incremental
          ? previous->tile_directory().WithGeometry(raw->geometry())
          : FineTraversabilityTileDirectory(raw->geometry());
  std::vector<TileIndex> changed_tiles;
  std::size_t updated_cells = 0U;
  for (const TileIndex tile_index : tiles_to_process) {
    const std::shared_ptr<const FineTraversabilityTile> old_tile =
        incremental ? previous->FindTile(tile_index) : nullptr;

    FineTraversabilityTile::StateArray states;
    FineTraversabilityTile::CostArray costs;
    if (old_tile) {
      states = old_tile->states();
      costs = old_tile->traversal_costs();
    } else {
      states.fill(FineCellState::kUnknown);
      costs.fill(0.0);
    }
    const auto old_states = states;
    const auto old_costs = costs;
    const auto update_cell = [&](const GridIndex cell) {
      const FineCellEvaluation evaluation = cell_evaluator.Evaluate(cell);
      const std::size_t offset = TileCellOffset(cell);
      states[offset] = evaluation.state;
      costs[offset] = evaluation.traversal_cost;
      ++updated_cells;
    };
    if (incremental) {
      for (const GridIndex cell : recompute_by_tile.at(tile_index)) {
        update_cell(cell);
      }
    } else {
      const std::optional<std::int64_t> tile_min_x =
          detail::CheckedMultiply(tile_index.x, kGridTileWidthCells);
      const std::optional<std::int64_t> tile_min_y =
          detail::CheckedMultiply(tile_index.y, kGridTileWidthCells);
      if (!tile_min_x || !tile_min_y) {
        throw std::overflow_error("fine tile origin exceeds int64 grid");
      }
      const GridIndex tile_min{.x = *tile_min_x, .y = *tile_min_y};
      for (std::int64_t local_y = 0; local_y < kGridTileWidthCells;
           ++local_y) {
        for (std::int64_t local_x = 0; local_x < kGridTileWidthCells;
             ++local_x) {
          const std::optional<std::int64_t> cell_x =
              detail::CheckedAdd(tile_min.x, local_x);
          const std::optional<std::int64_t> cell_y =
              detail::CheckedAdd(tile_min.y, local_y);
          if (!cell_x || !cell_y) {
            continue;
          }
          const GridIndex cell{.x = *cell_x, .y = *cell_y};
          if (raw->geometry().Contains(cell)) {
            update_cell(cell);
          }
        }
      }
    }

    const bool unobserved = TileIsUnobserved(states, costs);
    const bool changed = old_tile ? states != old_states || costs != old_costs
                                  : !unobserved;
    if (!changed) {
      continue;
    }
    changed_tiles.push_back(tile_index);
    directory = directory.WithTile(
        tile_index, std::make_shared<const FineTraversabilityTile>(
                        std::move(states), std::move(costs)));
  }

  std::vector<TileIndex> changed_halo_tiles(halo_tiles.begin(),
                                             halo_tiles.end());
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - started)
          .count();
  const FineSnapshotMetrics metrics{
      .updated_cells = updated_cells,
      .elevation_cells_examined =
          cell_evaluator.evaluated_elevation_cells(),
      .elevation_cache_tiles = cell_evaluator.cached_elevation_tiles(),
      .allocated_cells = directory.tile_count() * kGridTileCellCount,
      .allocated_bytes = directory.tile_count() *
                         (sizeof(FineCellState) + sizeof(double)) *
                         kGridTileCellCount,
      .derivation_ms = elapsed_ms,
  };
  const std::uint64_t fine_revision =
      previous ? previous->fine_traversability_revision() + 1U : 1U;
  return std::make_shared<const FineTraversabilitySnapshot>(
      raw->geometry(), raw->raw_elevation_revision(), fine_revision,
      profile_hash, hard_radius_m, profile.preferred_clearance_m,
      TraversalCostWeights{.slope = profile.slope_weight,
                           .relief = profile.relief_weight,
                           .clearance = profile.clearance_weight},
      std::move(raw), std::move(directory), std::move(changed_tiles),
      std::move(changed_halo_tiles), metrics);
}

}  // namespace lunar::incremental_navigation
