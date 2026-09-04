#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/types/persistent_tile_directory.hpp"

namespace lunar::incremental_navigation {

enum class FineCellState : std::uint8_t {
  kUnknown,
  kFree,
  kBlocked,
};

struct TraversalCostWeights final {
  double slope{};
  double relief{};
  double clearance{};
};

struct FineSnapshotMetrics final {
  std::size_t updated_cells{};
  std::size_t elevation_cells_examined{};
  std::size_t elevation_cache_tiles{};
  std::size_t allocated_cells{};
  std::size_t allocated_bytes{};
  double derivation_ms{};
};

class FineTraversabilityTile final {
 public:
  using StateArray = std::array<FineCellState, kGridTileCellCount>;
  using CostArray = std::array<double, kGridTileCellCount>;

  FineTraversabilityTile(const FineTraversabilityTile&) = default;
  FineTraversabilityTile(FineTraversabilityTile&&) noexcept = default;
  FineTraversabilityTile& operator=(const FineTraversabilityTile&) = delete;
  FineTraversabilityTile& operator=(FineTraversabilityTile&&) = delete;

  FineTraversabilityTile(StateArray states, CostArray traversal_costs)
      : states_(std::move(states)),
        traversal_costs_(std::move(traversal_costs)) {
    for (std::size_t offset = 0U; offset < kGridTileCellCount; ++offset) {
      if (!ValidState(states_[offset]) ||
          !std::isfinite(traversal_costs_[offset]) ||
          traversal_costs_[offset] < 0.0) {
        throw std::invalid_argument(
            "fine tile states and traversal costs must be valid");
      }
    }
  }

  [[nodiscard]] const StateArray& states() const noexcept { return states_; }

  [[nodiscard]] const CostArray& traversal_costs() const noexcept {
    return traversal_costs_;
  }

  [[nodiscard]] FineCellState State(const std::size_t offset) const noexcept {
    return states_[offset];
  }

  [[nodiscard]] double TraversalCost(
      const std::size_t offset) const noexcept {
    return traversal_costs_[offset];
  }

  [[nodiscard]] bool IsUnobserved(const std::size_t offset) const noexcept {
    return offset < kGridTileCellCount &&
           states_[offset] == FineCellState::kUnknown &&
           traversal_costs_[offset] == 0.0;
  }

 private:
  [[nodiscard]] static bool ValidState(const FineCellState state) noexcept {
    switch (state) {
      case FineCellState::kUnknown:
      case FineCellState::kFree:
      case FineCellState::kBlocked:
        return true;
    }
    return false;
  }

  StateArray states_;
  CostArray traversal_costs_;
};

using FineTraversabilityTileDirectory =
    PersistentTileDirectory<FineTraversabilityTile>;

class FineTraversabilitySnapshot final {
 public:
  FineTraversabilitySnapshot(
      SparseGridGeometry geometry, const std::uint64_t raw_elevation_revision,
      const std::uint64_t fine_traversability_revision,
      std::string platform_profile_hash,
      const double hard_inflation_radius_m,
      const double preferred_clearance_m,
      const TraversalCostWeights cost_weights,
      std::shared_ptr<const ElevationSnapshot> elevation,
      FineTraversabilityTileDirectory tiles,
      std::vector<TileIndex> changed_tiles,
      std::vector<TileIndex> changed_halo_tiles,
      const FineSnapshotMetrics metrics)
      : geometry_(std::move(geometry)),
        raw_elevation_revision_(raw_elevation_revision),
        fine_traversability_revision_(fine_traversability_revision),
        platform_profile_hash_(std::move(platform_profile_hash)),
        hard_inflation_radius_m_(hard_inflation_radius_m),
        preferred_clearance_m_(preferred_clearance_m),
        cost_weights_(cost_weights),
        elevation_(std::move(elevation)),
        tiles_(std::move(tiles)),
        changed_tiles_(std::move(changed_tiles)),
        changed_halo_tiles_(std::move(changed_halo_tiles)),
        metrics_(metrics) {
    if (!geometry_.valid()) {
      throw std::invalid_argument(
          "fine snapshot requires valid sparse map geometry");
    }
    if (!SameGeometry(geometry_, tiles_.geometry())) {
      throw std::invalid_argument(
          "fine snapshot tile directory geometry does not match");
    }
    if (!elevation_ || !elevation_->valid() || raw_elevation_revision_ == 0U ||
        fine_traversability_revision_ == 0U || platform_profile_hash_.empty() ||
        elevation_->raw_elevation_revision() != raw_elevation_revision_) {
      throw std::invalid_argument(
          "fine snapshot elevation lineage does not match its raw revision");
    }
    if (!MatchesElevationLattice()) {
      throw std::invalid_argument(
          "fine snapshot geometry must use the elevation canonical lattice");
    }
    if (!IsFiniteNonnegative(hard_inflation_radius_m_) ||
        !IsFiniteNonnegative(preferred_clearance_m_) ||
        !IsFiniteNonnegative(cost_weights_.slope) ||
        !IsFiniteNonnegative(cost_weights_.relief) ||
        !IsFiniteNonnegative(cost_weights_.clearance) ||
        !IsFiniteNonnegative(metrics_.derivation_ms)) {
      throw std::invalid_argument(
          "fine snapshot parameters must be finite and nonnegative");
    }
    NormalizeTiles(changed_tiles_);
    NormalizeTiles(changed_halo_tiles_);
  }

  [[nodiscard]] const SparseGridGeometry& geometry() const noexcept {
    return geometry_;
  }

  [[nodiscard]] std::uint64_t raw_elevation_revision() const noexcept {
    return raw_elevation_revision_;
  }

  [[nodiscard]] std::uint64_t fine_traversability_revision() const noexcept {
    return fine_traversability_revision_;
  }

  [[nodiscard]] const std::string& platform_profile_hash() const noexcept {
    return platform_profile_hash_;
  }

  [[nodiscard]] double hard_inflation_radius_m() const noexcept {
    return hard_inflation_radius_m_;
  }

  [[nodiscard]] double preferred_clearance_m() const noexcept {
    return preferred_clearance_m_;
  }

  [[nodiscard]] const TraversalCostWeights& cost_weights() const noexcept {
    return cost_weights_;
  }

  [[nodiscard]] std::shared_ptr<const ElevationSnapshot> elevation()
      const noexcept {
    return elevation_;
  }

  [[nodiscard]] std::shared_ptr<const FineTraversabilityTile> FindTile(
      const TileIndex index) const noexcept {
    return tiles_.Find(index);
  }

  [[nodiscard]] std::vector<TileIndex> tile_indices() const {
    return tiles_.tile_indices();
  }

  [[nodiscard]] const FineTraversabilityTileDirectory& tile_directory()
      const noexcept {
    return tiles_;
  }

  [[nodiscard]] FineCellState State(const GridIndex index) const noexcept {
    if (!geometry_.Contains(index)) {
      return FineCellState::kUnknown;
    }
    const auto tile = FindTile(TileForCell(index));
    return tile ? tile->State(TileCellOffset(index))
                : FineCellState::kUnknown;
  }

  [[nodiscard]] double TraversalCost(const GridIndex index) const noexcept {
    if (!geometry_.Contains(index)) {
      return 0.0;
    }
    const auto tile = FindTile(TileForCell(index));
    return tile ? tile->TraversalCost(TileCellOffset(index)) : 0.0;
  }

  [[nodiscard]] std::span<const TileIndex> changed_tiles() const noexcept {
    return changed_tiles_;
  }

  [[nodiscard]] std::span<const TileIndex> changed_halo_tiles()
      const noexcept {
    return changed_halo_tiles_;
  }

  [[nodiscard]] const FineSnapshotMetrics& metrics() const noexcept {
    return metrics_;
  }

 private:
  [[nodiscard]] static bool IsFiniteNonnegative(const double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
  }

  [[nodiscard]] static bool SameGeometry(
      const SparseGridGeometry& left,
      const SparseGridGeometry& right) noexcept {
    return left.frame_id() == right.frame_id() &&
           left.resolution_m() == right.resolution_m() &&
           left.origin_m() == right.origin_m() &&
           left.min_inclusive() == right.min_inclusive() &&
           left.max_exclusive() == right.max_exclusive();
  }

  [[nodiscard]] bool MatchesElevationLattice() const noexcept {
    const SparseGridGeometry& raw = elevation_->geometry();
    const Vec3 origin = geometry_.origin_m();
    const Vec3 raw_origin = raw.origin_m();
    const GridIndex min = geometry_.min_inclusive();
    const GridIndex max = geometry_.max_exclusive();
    const GridIndex raw_min = raw.min_inclusive();
    const GridIndex raw_max = raw.max_exclusive();
    return geometry_.frame_id() == raw.frame_id() &&
           geometry_.resolution_m() == raw.resolution_m() &&
           origin == raw_origin && min.x >= raw_min.x && min.y >= raw_min.y &&
           max.x <= raw_max.x && max.y <= raw_max.y;
  }

  void NormalizeTiles(std::vector<TileIndex>& tiles) const {
    for (const TileIndex tile : tiles) {
      if (!geometry_.Contains(tile)) {
        throw std::invalid_argument(
            "fine snapshot changed tile lies outside its geometry");
      }
    }
    std::sort(tiles.begin(), tiles.end());
    tiles.erase(std::unique(tiles.begin(), tiles.end()), tiles.end());
  }

  SparseGridGeometry geometry_;
  std::uint64_t raw_elevation_revision_{};
  std::uint64_t fine_traversability_revision_{};
  std::string platform_profile_hash_;
  double hard_inflation_radius_m_{};
  double preferred_clearance_m_{};
  TraversalCostWeights cost_weights_;
  std::shared_ptr<const ElevationSnapshot> elevation_;
  FineTraversabilityTileDirectory tiles_;
  std::vector<TileIndex> changed_tiles_;
  std::vector<TileIndex> changed_halo_tiles_;
  FineSnapshotMetrics metrics_;
};

}  // namespace lunar::incremental_navigation
