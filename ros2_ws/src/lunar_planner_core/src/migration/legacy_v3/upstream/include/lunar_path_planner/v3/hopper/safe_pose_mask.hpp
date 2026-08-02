#pragma once

#include <cstdint>
#include <vector>

#include "lunar_path_planner/v3/hopper/hopper_config.hpp"
#include "lunar_path_planner/v3/hopper/swept_footprint.hpp"
#include "lunar_path_planner/v3/map/immutable_snapshot.hpp"
#include "lunar_path_planner/v3/map/safe_projection.hpp"

namespace lunar::planning::v3 {

enum class LandingCellState : std::uint8_t {
  kUnsafe = 0,
  kSafe = 1,
};

struct SafePoseMask final {
  GridGeometry geometry;
  std::vector<LandingCellState> cells;
  std::vector<double> clearance_m;
  CircularYawInterval certified_yaw_interval;
  ContentRef source_snapshot_ref;
};

struct LandingSeed final {
  Cell cell;
  double clearance_m{};
  std::uint64_t stable_id{};
};

[[nodiscard]] Result<SafePoseMask> build_safe_pose_mask(
    const ImmutableMapSnapshot& map,
    const HopperCapabilityView& capability,
    const SweptFootprintEnvelope& footprint_envelope);

[[nodiscard]] std::vector<LandingSeed> select_landing_seeds(
    const SafePoseMask& mask, std::size_t maximum_seed_count);

[[nodiscard]] bool is_landing_cell_safe(
    const SafePoseMask& mask, Cell cell) noexcept;
[[nodiscard]] std::size_t landing_cell_index(
    const GridGeometry& geometry, Cell cell) noexcept;

}  // namespace lunar::planning::v3
