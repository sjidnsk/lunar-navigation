#include "lunar_planner_core/traversability_projection.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"

namespace lunar::planning {
namespace {

[[nodiscard]] TraversabilityProjectionResult Failure(
    std::string reason_code) {
  return TraversabilityProjectionResult{
      .projection = std::nullopt,
      .reason_code = std::move(reason_code),
  };
}

}  // namespace

TraversabilityProjectionResult ProjectTraversability(
    const WorldSnapshot& world,
    const PlatformCapability& capability,
    const MapSafetyConfig& config,
    const std::stop_token stop_token) {
  const shared::MapSnapshotBuildResult map =
      shared::MapSnapshot::Create(world.local_map);
  if (!map.ok()) {
    return Failure(map.reason_code);
  }
  const shared::SafeProjectionBuildResult built =
      shared::BuildSafeProjection(map.snapshot, capability, config, stop_token);
  if (!built.ok()) {
    return Failure(built.reason_code);
  }

  const shared::SafeProjection& source = *built.projection;
  TraversabilityProjection projection{
      .platform_type = source.platform_type(),
      .width = map.snapshot->width(),
      .height = map.snapshot->height(),
  };
  const std::size_t count = map.snapshot->cell_count();
  projection.known.reserve(count);
  projection.hard_feasible.reserve(count);
  projection.clearance_m.reserve(count);
  projection.slope_rad.reserve(count);
  projection.roughness_m.reserve(count);
  projection.traversal_cost.reserve(count);
  projection.connected_component.reserve(count);
  for (std::size_t index = 0U; index < count; ++index) {
    const shared::GridCell cell{
        .x = static_cast<std::int32_t>(index % projection.width),
        .y = static_cast<std::int32_t>(index / projection.width),
    };
    projection.known.push_back(static_cast<std::uint8_t>(source.Known(cell)));
    projection.hard_feasible.push_back(
        static_cast<std::uint8_t>(source.HardFeasible(cell)));
    projection.clearance_m.push_back(source.ClearanceMeters(cell));
    projection.slope_rad.push_back(source.SlopeRadians(cell));
    projection.roughness_m.push_back(source.RoughnessMeters(cell));
    projection.traversal_cost.push_back(source.TraversalCost(cell));
    projection.connected_component.push_back(source.ConnectedComponent(cell));
  }
  return TraversabilityProjectionResult{
      .projection = std::move(projection),
      .reason_code = {},
  };
}

}  // namespace lunar::planning
