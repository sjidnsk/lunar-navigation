#include "lunar_incremental_navigation_core/global_guidance_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lunar::incremental_navigation {
namespace {

struct WorldBounds final {
  double min_x_m{};
  double min_y_m{};
  double max_x_m{};
  double max_y_m{};
};

struct CellEvidence final {
  double fine_coverage_area_m2{};
  double fine_blocked_area_m2{};
  double minimum_free_risk{std::numeric_limits<double>::infinity()};
};

struct CellValue final {
  GuidanceCellState state{GuidanceCellState::kUnknown};
  double terrain_risk{};
};

struct MutableGuidanceTile final {
  GlobalGuidanceTile::StateArray states;
  GlobalGuidanceTile::RiskArray risks;

  explicit MutableGuidanceTile(
      const std::shared_ptr<const GlobalGuidanceTile>& previous) {
    if (previous) {
      states = previous->states();
      risks = previous->terrain_risks();
    } else {
      states.fill(GuidanceCellState::kUnknown);
      risks.fill(0.0);
    }
  }
};

[[nodiscard]] double CheckedCellArea(const double resolution_m,
                                     const char* const label) {
  const double area = resolution_m * resolution_m;
  if (!std::isfinite(resolution_m) || resolution_m <= 0.0 ||
      !std::isfinite(area) || area <= 0.0) {
    throw std::invalid_argument(std::string(label) +
                                " resolution has no finite positive area");
  }
  return area;
}

[[nodiscard]] double SnapNearInteger(const double value) noexcept {
  if (!std::isfinite(value)) {
    return value;
  }
  const double nearest = std::nearbyint(value);
  const double tolerance =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, std::abs(value));
  return std::abs(value - nearest) <= tolerance ? nearest : value;
}

[[nodiscard]] std::int64_t CheckedFloor(const double value) {
  const double floored = std::floor(SnapNearInteger(value));
  if (!std::isfinite(floored) ||
      floored < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      floored >=
          static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("guidance geometry exceeds grid index range");
  }
  return static_cast<std::int64_t>(floored);
}

[[nodiscard]] std::int64_t CheckedCeil(const double value) {
  const double ceiled = std::ceil(SnapNearInteger(value));
  if (!std::isfinite(ceiled) ||
      ceiled <=
          static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      ceiled > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("guidance geometry exceeds grid index range");
  }
  return static_cast<std::int64_t>(ceiled);
}

[[nodiscard]] WorldBounds BoundsOf(const SparseGridGeometry& geometry) {
  static_cast<void>(
      CheckedCellArea(geometry.resolution_m(), "source grid"));
  const Vec3 origin = geometry.origin_m();
  const GridIndex min = geometry.min_inclusive();
  const GridIndex max = geometry.max_exclusive();
  const WorldBounds bounds{
      .min_x_m = std::fma(static_cast<double>(min.x),
                          geometry.resolution_m(), origin.x),
      .min_y_m = std::fma(static_cast<double>(min.y),
                          geometry.resolution_m(), origin.y),
      .max_x_m = std::fma(static_cast<double>(max.x),
                          geometry.resolution_m(), origin.x),
      .max_y_m = std::fma(static_cast<double>(max.y),
                          geometry.resolution_m(), origin.y),
  };
  if (!std::isfinite(bounds.min_x_m) || !std::isfinite(bounds.min_y_m) ||
      !std::isfinite(bounds.max_x_m) || !std::isfinite(bounds.max_y_m) ||
      bounds.max_x_m <= bounds.min_x_m ||
      bounds.max_y_m <= bounds.min_y_m) {
    throw std::invalid_argument("source grid has invalid finite bounds");
  }
  return bounds;
}

[[nodiscard]] WorldBounds CellBounds(const SparseGridGeometry& geometry,
                                     const GridIndex index) {
  const Vec3 origin = geometry.origin_m();
  const double resolution = geometry.resolution_m();
  const WorldBounds bounds{
      .min_x_m = std::fma(static_cast<double>(index.x), resolution, origin.x),
      .min_y_m = std::fma(static_cast<double>(index.y), resolution, origin.y),
      .max_x_m = std::fma(static_cast<double>(index.x) + 1.0, resolution,
                          origin.x),
      .max_y_m = std::fma(static_cast<double>(index.y) + 1.0, resolution,
                          origin.y),
  };
  if (!std::isfinite(bounds.min_x_m) || !std::isfinite(bounds.min_y_m) ||
      !std::isfinite(bounds.max_x_m) || !std::isfinite(bounds.max_y_m) ||
      bounds.max_x_m <= bounds.min_x_m ||
      bounds.max_y_m <= bounds.min_y_m) {
    throw std::invalid_argument("grid cell has invalid finite bounds");
  }
  return bounds;
}

[[nodiscard]] SparseGridGeometry GuidanceGeometry(
    const SparseGridGeometry& fine_geometry,
    const std::shared_ptr<const ElevationSnapshot>& prior,
    const std::shared_ptr<const GlobalGuidanceSnapshot>& previous,
    const double resolution_m) {
  const Vec3 origin = previous ? previous->geometry().origin_m()
                               : fine_geometry.origin_m();
  WorldBounds bounds = BoundsOf(fine_geometry);
  const auto include = [&bounds](const WorldBounds candidate) {
    bounds.min_x_m = std::min(bounds.min_x_m, candidate.min_x_m);
    bounds.min_y_m = std::min(bounds.min_y_m, candidate.min_y_m);
    bounds.max_x_m = std::max(bounds.max_x_m, candidate.max_x_m);
    bounds.max_y_m = std::max(bounds.max_y_m, candidate.max_y_m);
  };
  if (prior) {
    include(BoundsOf(prior->geometry()));
  }
  if (previous) {
    include(BoundsOf(previous->geometry()));
  }
  const GridIndex min{
      .x = CheckedFloor((bounds.min_x_m - origin.x) / resolution_m),
      .y = CheckedFloor((bounds.min_y_m - origin.y) / resolution_m),
  };
  const GridIndex max{
      .x = CheckedCeil((bounds.max_x_m - origin.x) / resolution_m),
      .y = CheckedCeil((bounds.max_y_m - origin.y) / resolution_m),
  };
  SparseGridGeometry geometry(fine_geometry.frame_id(), resolution_m, origin, min, max);
  if (!geometry.valid()) {
    throw std::invalid_argument("guidance geometry is invalid");
  }
  return geometry;
}

[[nodiscard]] WorldBounds TileBounds(
    const SparseGridGeometry& geometry, const TileIndex tile) {
  const std::int64_t tile_min_x = tile.x * kGridTileWidthCells;
  const std::int64_t tile_min_y = tile.y * kGridTileWidthCells;
  const GridIndex min{
      .x = std::max(tile_min_x, geometry.min_inclusive().x),
      .y = std::max(tile_min_y, geometry.min_inclusive().y),
  };
  const GridIndex max{
      .x = std::min(tile_min_x + kGridTileWidthCells,
                    geometry.max_exclusive().x),
      .y = std::min(tile_min_y + kGridTileWidthCells,
                    geometry.max_exclusive().y),
  };
  const Vec3 origin = geometry.origin_m();
  const double resolution = geometry.resolution_m();
  const WorldBounds bounds{
      .min_x_m = std::fma(static_cast<double>(min.x), resolution, origin.x),
      .min_y_m = std::fma(static_cast<double>(min.y), resolution, origin.y),
      .max_x_m = std::fma(static_cast<double>(max.x), resolution, origin.x),
      .max_y_m = std::fma(static_cast<double>(max.y), resolution, origin.y),
  };
  if (max.x <= min.x || max.y <= min.y || !std::isfinite(bounds.min_x_m) ||
      !std::isfinite(bounds.min_y_m) || !std::isfinite(bounds.max_x_m) ||
      !std::isfinite(bounds.max_y_m)) {
    throw std::invalid_argument("changed tile has invalid finite bounds");
  }
  return bounds;
}

template <typename Visitor>
void VisitTileCells(const SparseGridGeometry& geometry,
                    const std::vector<TileIndex>& tile_indices,
                    Visitor&& visitor) {
  for (const TileIndex tile : tile_indices) {
    const std::int64_t tile_min_x = tile.x * kGridTileWidthCells;
    const std::int64_t tile_min_y = tile.y * kGridTileWidthCells;
    for (std::int64_t local_y = 0; local_y < kGridTileWidthCells; ++local_y) {
      for (std::int64_t local_x = 0; local_x < kGridTileWidthCells; ++local_x) {
        const GridIndex index{.x = tile_min_x + local_x,
                              .y = tile_min_y + local_y};
        if (geometry.Contains(index)) {
          visitor(index);
        }
      }
    }
  }
}

template <typename Visitor>
void VisitOverlappingCells(const WorldBounds& source,
                           const SparseGridGeometry& target,
                           Visitor&& visitor) {
  const Vec3 origin = target.origin_m();
  const double resolution = target.resolution_m();
  const GridIndex min{
      .x = std::max(CheckedFloor((source.min_x_m - origin.x) / resolution),
                    target.min_inclusive().x),
      .y = std::max(CheckedFloor((source.min_y_m - origin.y) / resolution),
                    target.min_inclusive().y),
  };
  const GridIndex max{
      .x = std::min(CheckedCeil((source.max_x_m - origin.x) / resolution),
                    target.max_exclusive().x),
      .y = std::min(CheckedCeil((source.max_y_m - origin.y) / resolution),
                    target.max_exclusive().y),
  };
  const double area_tolerance =
      std::max(1.0, CheckedCellArea(resolution, "target grid")) * 64.0 *
      std::numeric_limits<double>::epsilon();
  for (std::int64_t y = min.y; y < max.y; ++y) {
    for (std::int64_t x = min.x; x < max.x; ++x) {
      const GridIndex index{.x = x, .y = y};
      const WorldBounds cell = CellBounds(target, index);
      const double overlap_x =
          std::max(0.0, std::min(source.max_x_m, cell.max_x_m) -
                            std::max(source.min_x_m, cell.min_x_m));
      const double overlap_y =
          std::max(0.0, std::min(source.max_y_m, cell.max_y_m) -
                            std::max(source.min_y_m, cell.min_y_m));
      const double overlap_area = overlap_x * overlap_y;
      if (!std::isfinite(overlap_x) || !std::isfinite(overlap_y) ||
          !std::isfinite(overlap_area)) {
        throw std::invalid_argument("grid overlap area is not finite");
      }
      if (overlap_area > area_tolerance) {
        visitor(index, overlap_area);
      }
    }
  }
}

void AddOverlappingGuidanceCells(const WorldBounds& source,
                                 const SparseGridGeometry& guidance,
                                 std::set<GridIndex>& cells) {
  VisitOverlappingCells(source, guidance,
                        [&cells](const GridIndex index, const double) {
                          cells.insert(index);
                        });
}

[[nodiscard]] std::set<GridIndex> PriorCoverage(
    const std::shared_ptr<const ElevationSnapshot>& prior,
    const SparseGridGeometry& guidance) {
  std::set<GridIndex> cells;
  if (!prior) {
    return cells;
  }
  VisitTileCells(prior->geometry(), prior->tile_indices(),
                 [&](const GridIndex index) {
    if (prior->ElevationRangeAt(index)) {
      AddOverlappingGuidanceCells(CellBounds(prior->geometry(), index),
                                  guidance, cells);
    }
  });
  return cells;
}

[[nodiscard]] CellValue AggregateCell(
    const GridIndex coarse_index, const SparseGridGeometry& guidance,
    const FineTraversabilitySnapshot& fine,
    const std::shared_ptr<const ElevationSnapshot>& prior) {
  const WorldBounds coarse_bounds = CellBounds(guidance, coarse_index);
  CellEvidence evidence;
  VisitOverlappingCells(
      coarse_bounds, fine.geometry(),
      [&](const GridIndex fine_index, const double overlap_area) {
        const FineCellState state = fine.State(fine_index);
        const bool has_raw_elevation =
            fine.elevation()->ElevationRangeAt(fine_index).has_value();
        if (state == FineCellState::kUnknown && !has_raw_elevation) {
          return;
        }
        evidence.fine_coverage_area_m2 += overlap_area;
        if (state == FineCellState::kBlocked) {
          evidence.fine_blocked_area_m2 += overlap_area;
        } else if (state == FineCellState::kFree) {
          evidence.minimum_free_risk =
              std::min(evidence.minimum_free_risk,
                       fine.TraversalCost(fine_index));
        }
      });

  const double coarse_area =
      CheckedCellArea(guidance.resolution_m(), "guidance grid");
  const double area_tolerance =
      std::max(1.0, coarse_area) * 64.0 *
      std::numeric_limits<double>::epsilon();
  if (evidence.fine_coverage_area_m2 > area_tolerance) {
    return CellValue{
        .state = evidence.fine_blocked_area_m2 + area_tolerance >= coarse_area
                     ? GuidanceCellState::kProvenBlocked
                     : GuidanceCellState::kCandidate,
        .terrain_risk = std::isfinite(evidence.minimum_free_risk)
                            ? evidence.minimum_free_risk
                            : 0.0,
    };
  }

  bool has_prior = false;
  if (prior) {
    VisitOverlappingCells(
        coarse_bounds, prior->geometry(),
        [&](const GridIndex prior_index, const double) {
          has_prior =
              has_prior || prior->ElevationRangeAt(prior_index).has_value();
        });
  }
  return CellValue{.state = has_prior ? GuidanceCellState::kCandidate
                                      : GuidanceCellState::kUnknown,
                   .terrain_risk = 0.0};
}

[[nodiscard]] bool SameGeometry(const SparseGridGeometry& left,
                                const SparseGridGeometry& right) noexcept {
  return left.frame_id() == right.frame_id() &&
         left.resolution_m() == right.resolution_m() &&
         left.origin_m() == right.origin_m() &&
         left.min_inclusive() == right.min_inclusive() &&
         left.max_exclusive() == right.max_exclusive();
}

[[nodiscard]] bool SameCell(const CellValue left,
                            const CellValue right) noexcept {
  return left.state == right.state && left.terrain_risk == right.terrain_risk;
}

[[nodiscard]] CellValue ValueAt(
    const std::shared_ptr<const GlobalGuidanceTile>& tile,
    const GridIndex index) noexcept {
  if (!tile) {
    return CellValue{};
  }
  const std::size_t offset = TileCellOffset(index);
  return CellValue{.state = tile->State(offset),
                   .terrain_risk = tile->TerrainRisk(offset)};
}

[[nodiscard]] bool IsUnobserved(const MutableGuidanceTile& tile) noexcept {
  for (std::size_t offset = 0U; offset < kGridTileCellCount; ++offset) {
    if (tile.states[offset] != GuidanceCellState::kUnknown ||
        tile.risks[offset] != 0.0) {
      return false;
    }
  }
  return true;
}

}  // namespace

GlobalGuidanceBuilder::GlobalGuidanceBuilder(
    const double coarse_resolution_m)
    : coarse_resolution_m_(coarse_resolution_m) {
  static_cast<void>(CheckedCellArea(coarse_resolution_m_,
                                    "global guidance coarse"));
}

double GlobalGuidanceBuilder::coarse_resolution_m() const noexcept {
  return coarse_resolution_m_;
}

std::shared_ptr<const GlobalGuidanceSnapshot> GlobalGuidanceBuilder::Derive(
    std::shared_ptr<const FineTraversabilitySnapshot> fine,
    std::optional<GlobalElevationPrior> external_prior,
    std::shared_ptr<const GlobalGuidanceSnapshot> previous) const {
  if (!fine || !fine->elevation() || !fine->elevation()->valid()) {
    throw std::invalid_argument(
        "global guidance requires a valid fine traversability snapshot");
  }
  static_cast<void>(
      CheckedCellArea(fine->geometry().resolution_m(), "fine grid"));
  if (external_prior &&
      (!external_prior->elevation || !external_prior->elevation->valid())) {
    throw std::invalid_argument(
        "global guidance rejects an invalid external elevation prior");
  }
  if (external_prior) {
    static_cast<void>(CheckedCellArea(
        external_prior->elevation->geometry().resolution_m(),
        "external elevation prior"));
  }
  if (previous &&
      (previous->geometry().resolution_m() != coarse_resolution_m_ ||
       previous->geometry().origin_m() != fine->geometry().origin_m())) {
    throw std::invalid_argument(
        "previous guidance does not use this builder's frozen lattice");
  }
  if (previous &&
      previous->platform_profile_hash() != fine->platform_profile_hash()) {
    previous.reset();
  }

  const std::shared_ptr<const ElevationSnapshot> previous_prior =
      previous ? previous->external_elevation_prior() : nullptr;
  const std::shared_ptr<const ElevationSnapshot> effective_prior =
      external_prior ? external_prior->elevation : previous_prior;
  const SparseGridGeometry geometry = GuidanceGeometry(
      fine->geometry(), effective_prior, previous, coarse_resolution_m_);

  std::set<GridIndex> affected_cells;
  std::vector<TileIndex> affected_fine_tiles;
  if (!previous) {
    affected_fine_tiles = fine->tile_indices();
  } else if (fine->fine_traversability_revision() !=
             previous->source_fine_traversability_revision()) {
    affected_fine_tiles.assign(fine->changed_halo_tiles().begin(),
                               fine->changed_halo_tiles().end());
    if (affected_fine_tiles.empty()) {
      affected_fine_tiles = fine->tile_indices();
    }
  }
  for (const TileIndex tile : affected_fine_tiles) {
    AddOverlappingGuidanceCells(TileBounds(fine->geometry(), tile), geometry,
                                affected_cells);
  }

  if (!previous) {
    const std::set<GridIndex> prior_cells =
        PriorCoverage(effective_prior, geometry);
    affected_cells.insert(prior_cells.begin(), prior_cells.end());
  } else if (external_prior &&
             external_prior->elevation != previous_prior) {
    const std::set<GridIndex> old_prior_cells =
        PriorCoverage(previous_prior, geometry);
    const std::set<GridIndex> new_prior_cells =
        PriorCoverage(external_prior->elevation, geometry);
    std::set_symmetric_difference(
        old_prior_cells.begin(), old_prior_cells.end(), new_prior_cells.begin(),
        new_prior_cells.end(),
        std::inserter(affected_cells, affected_cells.end()));
  }

  GlobalGuidanceTileDirectory directory =
      previous ? previous->tile_directory().WithGeometry(geometry)
               : GlobalGuidanceTileDirectory(geometry);
  std::map<TileIndex, MutableGuidanceTile> mutable_tiles;
  std::vector<GridIndex> changed_cells;
  for (const GridIndex index : affected_cells) {
    const TileIndex tile_index = TileForCell(index);
    const auto old_tile = directory.Find(tile_index);
    auto [found, inserted] = mutable_tiles.try_emplace(tile_index, old_tile);
    static_cast<void>(inserted);
    MutableGuidanceTile& tile = found->second;
    const CellValue old_value = ValueAt(old_tile, index);
    const CellValue new_value =
        AggregateCell(index, geometry, *fine, effective_prior);
    if (!SameCell(old_value, new_value)) {
      changed_cells.push_back(index);
    }
    const std::size_t offset = TileCellOffset(index);
    tile.states[offset] = new_value.state;
    tile.risks[offset] = new_value.terrain_risk;
  }

  for (auto& [index, tile] : mutable_tiles) {
    const auto old_tile = directory.Find(index);
    if (old_tile && old_tile->states() == tile.states &&
        old_tile->terrain_risks() == tile.risks) {
      continue;
    }
    if (!old_tile && IsUnobserved(tile)) {
      continue;
    }
    directory = directory.WithTile(
        index, std::make_shared<const GlobalGuidanceTile>(
                   std::move(tile.states), std::move(tile.risks)));
  }

  const bool same_geometry =
      previous && SameGeometry(geometry, previous->geometry());
  if (previous && same_geometry && changed_cells.empty()) {
    return previous;
  }

  std::vector<TileIndex> changed_tiles;
  std::vector<TileIndex> changed_halo_tiles;
  for (const GridIndex index : changed_cells) {
    changed_tiles.push_back(TileForCell(index));
    for (std::int64_t dy = -1; dy <= 1; ++dy) {
      for (std::int64_t dx = -1; dx <= 1; ++dx) {
        if ((dx < 0 && index.x == std::numeric_limits<std::int64_t>::min()) ||
            (dx > 0 && index.x == std::numeric_limits<std::int64_t>::max()) ||
            (dy < 0 && index.y == std::numeric_limits<std::int64_t>::min()) ||
            (dy > 0 && index.y == std::numeric_limits<std::int64_t>::max())) {
          continue;
        }
        const GridIndex neighbor{.x = index.x + dx, .y = index.y + dy};
        if (geometry.Contains(neighbor)) {
          changed_halo_tiles.push_back(TileForCell(neighbor));
        }
      }
    }
  }

  const std::uint64_t revision =
      previous ? previous->global_guidance_revision() + 1U : 1U;
  if (revision == 0U) {
    throw std::overflow_error("global guidance revision overflow");
  }
  return std::make_shared<const GlobalGuidanceSnapshot>(
      geometry, fine->raw_elevation_revision(),
      fine->fine_traversability_revision(), revision,
      fine->platform_profile_hash(), std::move(directory),
      std::move(changed_tiles), std::move(changed_halo_tiles), effective_prior);
}

}  // namespace lunar::incremental_navigation
