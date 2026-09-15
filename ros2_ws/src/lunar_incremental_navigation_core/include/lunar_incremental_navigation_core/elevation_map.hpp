#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "lunar_incremental_navigation_core/types/grid_geometry.hpp"
#include "lunar_incremental_navigation_core/types/world_snapshot.hpp"

namespace lunar::incremental_navigation {

// Platform-neutral native 3x3 measurements, never a traversability label.
struct LocalTerrainMeasurements final {
  bool center_known{};
  bool neighborhood_complete{};
  double slope_rad{};
  double relief_m{};
  double positive_rise_m{};

  auto operator<=>(const LocalTerrainMeasurements&) const = default;
};

struct ElevationEvidence final {
  GridGeometry geometry;
  std::span<const float> elevation_m;
  RigidTransform map_from_source;
  // Empty preserves height-only input. Otherwise one entry per source cell;
  // center_known=false means absent. Scalars require an upright quarter-turn
  // transform and the same resolution and aligned lattice as the canonical map.
  std::span<const LocalTerrainMeasurements> terrain_measurements{};
};

struct ElevationRange final {
  float min_m{};
  float max_m{};

  auto operator<=>(const ElevationRange&) const = default;
};

struct ElevationUpdateResult final {
  enum class Status : std::uint8_t {
    kApplied,
    kDuplicate,
    kRejected,
  };

  Status status{Status::kRejected};
  std::uint64_t raw_elevation_revision{};
  std::size_t updated_cells{};
  std::vector<TileIndex> dirty_tiles;
};

struct ElevationMapCounters final {
  std::uint64_t applied_updates{};
  std::uint64_t duplicate_updates{};
  std::uint64_t rejected_updates{};
  std::size_t allocated_tiles{};
  std::uint64_t tile_allocations{};
  std::uint64_t tile_copies{};
  std::uint64_t updated_cells{};
  std::size_t estimated_bytes{};
};

class SparseGridGeometry final {
 public:
  SparseGridGeometry() noexcept = default;
  SparseGridGeometry(std::string frame_id, double resolution_m, Vec3 origin_m,
                     GridIndex min_inclusive,
                     GridIndex max_exclusive) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const std::string& frame_id() const noexcept;
  [[nodiscard]] double resolution_m() const noexcept;
  [[nodiscard]] Vec3 origin_m() const noexcept;
  [[nodiscard]] GridIndex min_inclusive() const noexcept;
  [[nodiscard]] GridIndex max_exclusive() const noexcept;
  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] std::size_t width() const noexcept;
  [[nodiscard]] std::size_t height() const noexcept;
  [[nodiscard]] std::size_t CellCount() const noexcept;
  [[nodiscard]] bool Contains(GridIndex index) const noexcept;
  [[nodiscard]] bool Contains(TileIndex index) const noexcept;

 private:
  std::string frame_id_;
  double resolution_m_{};
  Vec3 origin_m_;
  GridIndex min_inclusive_;
  GridIndex max_exclusive_;
};

class ElevationRangeView {
 public:
  virtual ~ElevationRangeView() = default;

  [[nodiscard]] virtual const SparseGridGeometry& geometry() const noexcept = 0;
  [[nodiscard]] virtual std::optional<ElevationRange> ElevationRangeAt(
      GridIndex index) const noexcept = 0;
  [[nodiscard]] virtual std::optional<LocalTerrainMeasurements>
  TerrainMeasurementsAt(GridIndex) const noexcept { return std::nullopt; }
};

class ElevationSnapshot final : public ElevationRangeView {
 public:
  ElevationSnapshot() noexcept = default;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const SparseGridGeometry& geometry() const noexcept override;
  [[nodiscard]] std::uint64_t raw_elevation_revision() const noexcept;
  [[nodiscard]] std::optional<ElevationRange> ElevationRangeAt(
      GridIndex index) const noexcept override;
  [[nodiscard]] std::optional<LocalTerrainMeasurements> TerrainMeasurementsAt(
      GridIndex index) const noexcept override;
  [[nodiscard]] std::optional<ElevationRange> ElevationRangeAtWorld(
      double x_m, double y_m) const noexcept;
  [[nodiscard]] std::optional<float> ElevationAt(GridIndex index) const noexcept;
  [[nodiscard]] std::optional<float> ElevationAtWorld(double x_m,
                                                      double y_m) const noexcept;
  [[nodiscard]] std::span<const GridIndex> changed_cells() const noexcept;
  [[nodiscard]] std::span<const TileIndex> changed_tiles() const noexcept;
  [[nodiscard]] bool IsDirectSuccessorOf(
      const ElevationSnapshot& previous) const noexcept;
  [[nodiscard]] bool SharesLineageWith(
      const ElevationSnapshot& other) const noexcept;
  [[nodiscard]] std::vector<TileIndex> tile_indices() const;
  [[nodiscard]] std::size_t allocated_tiles() const noexcept;

 private:
  struct Impl;
  explicit ElevationSnapshot(std::shared_ptr<const Impl> impl) noexcept;

  std::shared_ptr<const Impl> impl_;

  friend class PersistentElevationMap;
};

class PersistentElevationMap final {
 public:
  PersistentElevationMap();
  ~PersistentElevationMap();

  PersistentElevationMap(const PersistentElevationMap&) = delete;
  PersistentElevationMap& operator=(const PersistentElevationMap&) = delete;
  PersistentElevationMap(PersistentElevationMap&&) noexcept;
  PersistentElevationMap& operator=(PersistentElevationMap&&) noexcept;

  [[nodiscard]] ElevationUpdateResult Apply(const ElevationEvidence& evidence);
  [[nodiscard]] std::shared_ptr<const ElevationSnapshot> Snapshot() const;
  [[nodiscard]] ElevationMapCounters Counters() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::incremental_navigation
