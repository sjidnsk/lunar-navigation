#include "lunar_observed_map/sparse_observed_map.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lunar::observed_map {
namespace {

struct FloorDivision final {
  std::int64_t quotient{};
  std::size_t remainder{};
};

FloorDivision DivideCell(const std::int64_t value) noexcept {
  constexpr std::int64_t divisor =
      static_cast<std::int64_t>(kTileSideCells);
  std::int64_t quotient = value / divisor;
  std::int64_t remainder = value % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }
  return FloorDivision{
      .quotient = quotient,
      .remainder = static_cast<std::size_t>(remainder),
  };
}

bool CheckedAdd(
    const std::int64_t base, const std::size_t offset,
    std::int64_t& result) noexcept {
  if (offset > static_cast<std::size_t>(
                   std::numeric_limits<std::int64_t>::max())) {
    return false;
  }
  const std::int64_t signed_offset = static_cast<std::int64_t>(offset);
  if (base > std::numeric_limits<std::int64_t>::max() - signed_offset) {
    return false;
  }
  result = base + signed_offset;
  return true;
}

std::size_t CellOffset(
    const std::size_t local_x, const std::size_t local_y) noexcept {
  return local_y * kTileSideCells + local_x;
}

}  // namespace

std::size_t TileKeyHash::operator()(const TileKey& key) const noexcept {
  const std::size_t x = std::hash<std::int64_t>{}(key.x);
  const std::size_t y = std::hash<std::int64_t>{}(key.y);
  return x ^ (y + 0x9e3779b97f4a7c15ULL + (x << 6U) + (x >> 2U));
}

struct SparseObservedMap::Tile final {
  Tile() : cells(kCellsPerTile) {}
  std::vector<CellEvidence> cells;
};

SparseObservedMap::SparseObservedMap(const std::size_t maximum_tiles)
    : maximum_tiles_(maximum_tiles) {
  if (maximum_tiles_ == 0U ||
      maximum_tiles_ > kMaximumTilesPerSession) {
    throw std::invalid_argument("MAXIMUM_TILES_OUT_OF_RANGE");
  }
}

SparseObservedMap::~SparseObservedMap() = default;
SparseObservedMap::SparseObservedMap(SparseObservedMap&&) noexcept = default;
SparseObservedMap& SparseObservedMap::operator=(
    SparseObservedMap&&) noexcept = default;

void SparseObservedMap::ResetSession(std::string session_id) {
  tiles_.clear();
  session_id_ = std::move(session_id);
  observed_cell_count_ = 0U;
  observed_bounds_.reset();
}

FuseResult SparseObservedMap::Fuse(const ObservedPatch& patch) {
  if (session_id_.empty() || patch.session_id != session_id_) {
    return FuseResult{
        .accepted = false,
        .reason_code = "MAP_SESSION_MISMATCH",
    };
  }
  if (patch.width == 0U || patch.height == 0U ||
      patch.width > std::numeric_limits<std::size_t>::max() / patch.height) {
    return FuseResult{
        .accepted = false,
        .reason_code = "MAP_PATCH_DIMENSIONS_INVALID",
    };
  }
  const std::size_t cell_count = patch.width * patch.height;
  if (patch.elevation_m.size() != cell_count ||
      patch.valid.size() != cell_count) {
    return FuseResult{
        .accepted = false,
        .reason_code = "MAP_PATCH_LAYOUT_INVALID",
    };
  }

  std::unordered_set<TileKey, TileKeyHash> required_tiles;
  required_tiles.reserve(std::min(cell_count, maximum_tiles_ + 1U));
  for (std::size_t row = 0U; row < patch.height; ++row) {
    std::int64_t cell_y{};
    if (!CheckedAdd(patch.cell_zero.y, row, cell_y)) {
      return FuseResult{
          .accepted = false,
          .reason_code = "MAP_CELL_INDEX_OVERFLOW",
      };
    }
    for (std::size_t column = 0U; column < patch.width; ++column) {
      const std::size_t offset = row * patch.width + column;
      if (patch.valid[offset] > 1U) {
        return FuseResult{
            .accepted = false,
            .reason_code = "MAP_VALID_MASK_INVALID",
        };
      }
      if (patch.valid[offset] == 0U) {
        continue;
      }
      if (!std::isfinite(patch.elevation_m[offset])) {
        return FuseResult{
            .accepted = false,
            .reason_code = "MAP_VALID_ELEVATION_NONFINITE",
        };
      }
      std::int64_t cell_x{};
      if (!CheckedAdd(patch.cell_zero.x, column, cell_x)) {
        return FuseResult{
            .accepted = false,
            .reason_code = "MAP_CELL_INDEX_OVERFLOW",
        };
      }
      required_tiles.insert(Locate({cell_x, cell_y}).tile);
    }
  }

  std::vector<TileKey> new_keys;
  new_keys.reserve(required_tiles.size());
  for (const TileKey& key : required_tiles) {
    if (!tiles_.contains(key)) {
      new_keys.push_back(key);
    }
  }
  if (new_keys.size() > maximum_tiles_ - tiles_.size()) {
    return FuseResult{
        .accepted = false,
        .reason_code = "MAP_TILE_BUDGET_EXCEEDED",
    };
  }

  std::vector<std::pair<TileKey, std::unique_ptr<Tile>>> allocations;
  allocations.reserve(new_keys.size());
  try {
    for (const TileKey& key : new_keys) {
      allocations.emplace_back(key, std::make_unique<Tile>());
    }
  } catch (const std::bad_alloc&) {
    return FuseResult{
        .accepted = false,
        .reason_code = "MAP_ALLOCATION_FAILED",
    };
  }
  for (auto& [key, tile] : allocations) {
    tiles_.emplace(key, std::move(tile));
  }

  std::size_t updated_cells = 0U;
  for (std::size_t row = 0U; row < patch.height; ++row) {
    const std::int64_t cell_y =
        patch.cell_zero.y + static_cast<std::int64_t>(row);
    for (std::size_t column = 0U; column < patch.width; ++column) {
      const std::size_t offset = row * patch.width + column;
      if (patch.valid[offset] == 0U) {
        continue;
      }
      const std::int64_t cell_x =
          patch.cell_zero.x + static_cast<std::int64_t>(column);
      const GridCellIndex index{cell_x, cell_y};
      const CellLocation location = Locate(index);
      CellEvidence& evidence = tiles_.at(location.tile)
                                   ->cells[CellOffset(
                                       location.local_x, location.local_y)];
      const double sample = patch.elevation_m[offset];
      if (!evidence.valid) {
        evidence.valid = true;
        evidence.elevation_m = sample;
        evidence.elevation_m2 = 0.0;
        evidence.observation_count = 1U;
        ++observed_cell_count_;
        if (!observed_bounds_.has_value()) {
          observed_bounds_ = ObservedBounds{index, index};
        } else {
          observed_bounds_->minimum.x =
              std::min(observed_bounds_->minimum.x, index.x);
          observed_bounds_->minimum.y =
              std::min(observed_bounds_->minimum.y, index.y);
          observed_bounds_->maximum.x =
              std::max(observed_bounds_->maximum.x, index.x);
          observed_bounds_->maximum.y =
              std::max(observed_bounds_->maximum.y, index.y);
        }
      } else {
        ++evidence.observation_count;
        const double delta = sample - evidence.elevation_m;
        evidence.elevation_m +=
            delta / static_cast<double>(evidence.observation_count);
        const double second_delta = sample - evidence.elevation_m;
        evidence.elevation_m2 += delta * second_delta;
      }
      evidence.last_observed_time_ns = patch.simulation_time_ns;
      ++updated_cells;
    }
  }

  return FuseResult{
      .accepted = true,
      .updated_cells = updated_cells,
      .allocated_tiles = new_keys.size(),
      .reason_code = "ACCEPTED",
  };
}

const CellEvidence* SparseObservedMap::Find(
    const GridCellIndex cell) const noexcept {
  const CellLocation location = Locate(cell);
  const auto tile = tiles_.find(location.tile);
  if (tile == tiles_.end()) {
    return nullptr;
  }
  return &tile->second->cells[CellOffset(
      location.local_x, location.local_y)];
}

CellEvidence* SparseObservedMap::FindMutable(
    const GridCellIndex cell) noexcept {
  return const_cast<CellEvidence*>(
      static_cast<const SparseObservedMap&>(*this).Find(cell));
}

bool SparseObservedMap::SetForbidden(
    const GridCellIndex cell, const bool forbidden) noexcept {
  CellEvidence* evidence = FindMutable(cell);
  if (evidence == nullptr) {
    return false;
  }
  evidence->forbidden = forbidden;
  return true;
}

void SparseObservedMap::ForEachObserved(
    const std::function<void(GridCellIndex, const CellEvidence&)>& visitor)
    const {
  for (const auto& [key, tile] : tiles_) {
    for (std::size_t local_y = 0U; local_y < kTileSideCells; ++local_y) {
      for (std::size_t local_x = 0U; local_x < kTileSideCells; ++local_x) {
        const CellEvidence& evidence =
            tile->cells[CellOffset(local_x, local_y)];
        if (!evidence.valid) {
          continue;
        }
        visitor(
            GridCellIndex{
                key.x * static_cast<std::int64_t>(kTileSideCells) +
                    static_cast<std::int64_t>(local_x),
                key.y * static_cast<std::int64_t>(kTileSideCells) +
                    static_cast<std::int64_t>(local_y),
            },
            evidence);
      }
    }
  }
}

std::optional<ObservedBounds> SparseObservedMap::observed_bounds()
    const noexcept {
  return observed_bounds_;
}

const std::string& SparseObservedMap::session_id() const noexcept {
  return session_id_;
}

std::size_t SparseObservedMap::tile_count() const noexcept {
  return tiles_.size();
}

std::size_t SparseObservedMap::observed_cell_count() const noexcept {
  return observed_cell_count_;
}

std::size_t SparseObservedMap::maximum_tiles() const noexcept {
  return maximum_tiles_;
}

CellLocation SparseObservedMap::Locate(const GridCellIndex cell) noexcept {
  const FloorDivision x = DivideCell(cell.x);
  const FloorDivision y = DivideCell(cell.y);
  return CellLocation{
      .tile = {x.quotient, y.quotient},
      .local_x = x.remainder,
      .local_y = y.remainder,
  };
}

}  // namespace lunar::observed_map
