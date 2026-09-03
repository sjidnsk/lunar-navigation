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

enum class GuidanceCellState : std::uint8_t {
  kUnknown,
  kCandidate,
  kProvenBlocked,
};

class GlobalGuidanceTile final {
 public:
  using StateArray = std::array<GuidanceCellState, kGridTileCellCount>;
  using RiskArray = std::array<double, kGridTileCellCount>;

  GlobalGuidanceTile(const GlobalGuidanceTile&) = default;
  GlobalGuidanceTile(GlobalGuidanceTile&&) noexcept = default;
  GlobalGuidanceTile& operator=(const GlobalGuidanceTile&) = delete;
  GlobalGuidanceTile& operator=(GlobalGuidanceTile&&) = delete;

  GlobalGuidanceTile(StateArray states, RiskArray terrain_risks)
      : states_(std::move(states)), terrain_risks_(std::move(terrain_risks)) {
    for (std::size_t offset = 0U; offset < kGridTileCellCount; ++offset) {
      if (!ValidState(states_[offset]) ||
          !std::isfinite(terrain_risks_[offset]) ||
          terrain_risks_[offset] < 0.0) {
        throw std::invalid_argument(
            "guidance tile states and terrain risks must be valid");
      }
    }
  }

  [[nodiscard]] const StateArray& states() const noexcept { return states_; }

  [[nodiscard]] const RiskArray& terrain_risks() const noexcept {
    return terrain_risks_;
  }

  [[nodiscard]] GuidanceCellState State(
      const std::size_t offset) const noexcept {
    return states_[offset];
  }

  [[nodiscard]] double TerrainRisk(const std::size_t offset) const noexcept {
    return terrain_risks_[offset];
  }

  [[nodiscard]] bool IsUnobserved(const std::size_t offset) const noexcept {
    return offset < kGridTileCellCount &&
           states_[offset] == GuidanceCellState::kUnknown &&
           terrain_risks_[offset] == 0.0;
  }

 private:
  [[nodiscard]] static bool ValidState(
      const GuidanceCellState state) noexcept {
    switch (state) {
      case GuidanceCellState::kUnknown:
      case GuidanceCellState::kCandidate:
      case GuidanceCellState::kProvenBlocked:
        return true;
    }
    return false;
  }

  StateArray states_;
  RiskArray terrain_risks_;
};

using GlobalGuidanceTileDirectory =
    PersistentTileDirectory<GlobalGuidanceTile>;

class GlobalGuidanceSnapshot final {
 public:
  GlobalGuidanceSnapshot(
      SparseGridGeometry geometry,
      const std::uint64_t source_raw_elevation_revision,
      const std::uint64_t source_fine_traversability_revision,
      const std::uint64_t global_guidance_revision,
      std::string platform_profile_hash, GlobalGuidanceTileDirectory tiles,
      std::vector<TileIndex> changed_tiles,
      std::vector<TileIndex> changed_halo_tiles,
      std::shared_ptr<const ElevationSnapshot> external_elevation_prior =
          nullptr)
      : geometry_(std::move(geometry)),
        source_raw_elevation_revision_(source_raw_elevation_revision),
        source_fine_traversability_revision_(
            source_fine_traversability_revision),
        global_guidance_revision_(global_guidance_revision),
        platform_profile_hash_(std::move(platform_profile_hash)),
        tiles_(std::move(tiles)),
        changed_tiles_(std::move(changed_tiles)),
        changed_halo_tiles_(std::move(changed_halo_tiles)),
        external_elevation_prior_(std::move(external_elevation_prior)) {
    if (!geometry_.valid()) {
      throw std::invalid_argument(
          "guidance snapshot requires valid sparse map geometry");
    }
    if (!SameGeometry(geometry_, tiles_.geometry())) {
      throw std::invalid_argument(
          "guidance snapshot tile directory geometry does not match");
    }
    if (source_raw_elevation_revision_ == 0U ||
        source_fine_traversability_revision_ == 0U ||
        global_guidance_revision_ == 0U || platform_profile_hash_.empty()) {
      throw std::invalid_argument(
          "guidance snapshot requires published source and guidance lineage");
    }
    if (external_elevation_prior_ && !external_elevation_prior_->valid()) {
      throw std::invalid_argument(
          "guidance snapshot rejects invalid external elevation prior");
    }
    NormalizeTiles(changed_tiles_);
    NormalizeTiles(changed_halo_tiles_);
  }

  [[nodiscard]] const SparseGridGeometry& geometry() const noexcept {
    return geometry_;
  }

  [[nodiscard]] std::uint64_t source_raw_elevation_revision() const noexcept {
    return source_raw_elevation_revision_;
  }

  [[nodiscard]] std::uint64_t source_fine_traversability_revision()
      const noexcept {
    return source_fine_traversability_revision_;
  }

  [[nodiscard]] std::uint64_t global_guidance_revision() const noexcept {
    return global_guidance_revision_;
  }

  [[nodiscard]] const std::string& platform_profile_hash() const noexcept {
    return platform_profile_hash_;
  }

  [[nodiscard]] std::shared_ptr<const GlobalGuidanceTile> FindTile(
      const TileIndex index) const noexcept {
    return tiles_.Find(index);
  }

  [[nodiscard]] std::vector<TileIndex> tile_indices() const {
    return tiles_.tile_indices();
  }

  [[nodiscard]] const GlobalGuidanceTileDirectory& tile_directory()
      const noexcept {
    return tiles_;
  }

  [[nodiscard]] std::shared_ptr<const ElevationSnapshot>
  external_elevation_prior() const noexcept {
    return external_elevation_prior_;
  }

  [[nodiscard]] GuidanceCellState State(const GridIndex index) const noexcept {
    if (!geometry_.Contains(index)) {
      return GuidanceCellState::kUnknown;
    }
    const auto tile = FindTile(TileForCell(index));
    return tile ? tile->State(TileCellOffset(index))
                : GuidanceCellState::kUnknown;
  }

  [[nodiscard]] double TerrainRisk(const GridIndex index) const noexcept {
    if (!geometry_.Contains(index)) {
      return 0.0;
    }
    const auto tile = FindTile(TileForCell(index));
    return tile ? tile->TerrainRisk(TileCellOffset(index)) : 0.0;
  }

  [[nodiscard]] std::span<const TileIndex> changed_tiles() const noexcept {
    return changed_tiles_;
  }

  [[nodiscard]] std::span<const TileIndex> changed_halo_tiles()
      const noexcept {
    return changed_halo_tiles_;
  }

 private:
  [[nodiscard]] static bool SameGeometry(
      const SparseGridGeometry& left,
      const SparseGridGeometry& right) noexcept {
    return left.frame_id() == right.frame_id() &&
           left.resolution_m() == right.resolution_m() &&
           left.origin_m() == right.origin_m() &&
           left.min_inclusive() == right.min_inclusive() &&
           left.max_exclusive() == right.max_exclusive();
  }

  void NormalizeTiles(std::vector<TileIndex>& tiles) const {
    for (const TileIndex tile : tiles) {
      if (!geometry_.Contains(tile)) {
        throw std::invalid_argument(
            "guidance snapshot changed tile lies outside its geometry");
      }
    }
    std::sort(tiles.begin(), tiles.end());
    tiles.erase(std::unique(tiles.begin(), tiles.end()), tiles.end());
  }

  SparseGridGeometry geometry_;
  std::uint64_t source_raw_elevation_revision_{};
  std::uint64_t source_fine_traversability_revision_{};
  std::uint64_t global_guidance_revision_{};
  std::string platform_profile_hash_;
  GlobalGuidanceTileDirectory tiles_;
  std::vector<TileIndex> changed_tiles_;
  std::vector<TileIndex> changed_halo_tiles_;
  std::shared_ptr<const ElevationSnapshot> external_elevation_prior_;
};

}  // namespace lunar::incremental_navigation
