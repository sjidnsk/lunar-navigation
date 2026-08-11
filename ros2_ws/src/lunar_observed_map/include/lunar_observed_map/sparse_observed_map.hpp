#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "lunar_observed_map/map_types.hpp"

namespace lunar::observed_map {

class SparseObservedMap final {
 public:
  explicit SparseObservedMap(
      std::size_t maximum_tiles = kMaximumTilesPerSession);
  ~SparseObservedMap();

  SparseObservedMap(const SparseObservedMap&) = delete;
  SparseObservedMap& operator=(const SparseObservedMap&) = delete;
  SparseObservedMap(SparseObservedMap&&) noexcept;
  SparseObservedMap& operator=(SparseObservedMap&&) noexcept;

  void ResetSession(std::string session_id);
  [[nodiscard]] FuseResult Fuse(const ObservedPatch& patch);

  [[nodiscard]] const CellEvidence* Find(GridCellIndex cell) const noexcept;
  [[nodiscard]] CellEvidence* FindMutable(GridCellIndex cell) noexcept;
  [[nodiscard]] bool SetForbidden(GridCellIndex cell, bool forbidden) noexcept;
  void ForEachObserved(
      const std::function<void(GridCellIndex, const CellEvidence&)>& visitor)
      const;

  [[nodiscard]] std::optional<ObservedBounds> observed_bounds() const noexcept;
  [[nodiscard]] const std::string& session_id() const noexcept;
  [[nodiscard]] std::size_t tile_count() const noexcept;
  [[nodiscard]] std::size_t observed_cell_count() const noexcept;
  [[nodiscard]] std::size_t maximum_tiles() const noexcept;

  [[nodiscard]] static CellLocation Locate(GridCellIndex cell) noexcept;

 private:
  struct Tile;

  std::unordered_map<TileKey, std::unique_ptr<Tile>, TileKeyHash> tiles_;
  std::string session_id_;
  std::size_t observed_cell_count_{};
  std::size_t maximum_tiles_;
  std::optional<ObservedBounds> observed_bounds_;
};

}  // namespace lunar::observed_map
