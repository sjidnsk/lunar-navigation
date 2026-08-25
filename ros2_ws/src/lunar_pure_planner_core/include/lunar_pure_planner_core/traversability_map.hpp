#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "lunar_pure_planner_core/types/geometry.hpp"
#include "lunar_pure_planner_core/types/world_snapshot.hpp"

namespace lunar::pure_planning {

enum class TraversabilityState : std::uint8_t {
  kUnknown,
  kFree,
  kBlocked,
};

enum class WheelPlannerMode : std::uint8_t {
  kLegacyCertified,
  kGridTraversabilityV1,
};

[[nodiscard]] constexpr std::string_view WheelPlannerModeName(
    const WheelPlannerMode mode) noexcept {
  switch (mode) {
    case WheelPlannerMode::kLegacyCertified:
      return "legacy_certified";
    case WheelPlannerMode::kGridTraversabilityV1:
      return "grid_traversability_v1";
  }
  return "legacy_certified";
}

struct TraversabilityProfile final {
  std::int32_t global_occupancy_threshold{50};
  double local_occupancy_threshold{0.5};
  double maximum_slope_rad{};
  double inflation_radius_m{};
};

struct TraversabilityMetrics final {
  std::uint64_t revision{};
  std::uint64_t profile_hash{};
  std::size_t allocated_tiles{};
  std::size_t estimated_bytes{};
  std::size_t updated_cells{};
  std::size_t dirty_tiles{};
  std::size_t halo_recomputed_cells{};
  std::size_t free_cells{};
  std::size_t blocked_cells{};
  std::size_t unknown_cells{};
  std::size_t prior_conflicts{};
};

class TraversabilitySnapshot final {
 public:
  TraversabilitySnapshot() noexcept = default;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] double resolution_m() const noexcept;
  [[nodiscard]] Vec3 origin_m() const noexcept;
  [[nodiscard]] std::uint64_t revision() const noexcept;
  [[nodiscard]] std::uint64_t profile_hash() const noexcept;
  [[nodiscard]] TraversabilityState StateAtWorld(double x_m,
                                                 double y_m) const;
  [[nodiscard]] TraversabilityMetrics metrics() const noexcept;

 private:
  struct Impl;
  explicit TraversabilitySnapshot(std::shared_ptr<const Impl> impl) noexcept;

  std::shared_ptr<const Impl> impl_;

  friend class PersistentTraversabilityMap;
};

struct TraversabilityUpdateResult final {
  bool accepted{};
  bool changed{};
  std::string reason_code;
  TraversabilityMetrics metrics;
};

class PersistentTraversabilityMap final {
 public:
  explicit PersistentTraversabilityMap(TraversabilityProfile profile);
  ~PersistentTraversabilityMap();

  PersistentTraversabilityMap(const PersistentTraversabilityMap&) = delete;
  PersistentTraversabilityMap& operator=(const PersistentTraversabilityMap&) =
      delete;
  PersistentTraversabilityMap(PersistentTraversabilityMap&&) noexcept;
  PersistentTraversabilityMap& operator=(PersistentTraversabilityMap&&) noexcept;

  [[nodiscard]] TraversabilityUpdateResult UpdateGlobal(const GridMap& global);
  [[nodiscard]] TraversabilityUpdateResult UpdateLocal(
      const GridMap& local, const RigidTransform& map_from_odom,
      std::uint64_t local_sequence);
  [[nodiscard]] std::shared_ptr<const TraversabilitySnapshot> Capture() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::pure_planning
