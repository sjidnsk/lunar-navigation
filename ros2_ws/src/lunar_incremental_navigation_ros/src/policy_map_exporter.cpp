#include "lunar_incremental_navigation_ros/policy_map_exporter.hpp"

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <lunar_planning_msgs/msg/policy_map_tile.hpp>

#include "lunar_incremental_navigation_core/local_planning_window.hpp"

namespace lunar::incremental_navigation_ros {
namespace core = lunar::incremental_navigation;

namespace {

std::uint8_t WireState(const core::FineCellState state) noexcept {
  switch (state) {
    case core::FineCellState::kFree: return kPolicyMapFree;
    case core::FineCellState::kBlocked: return kPolicyMapBlocked;
    case core::FineCellState::kUnknown: return kPolicyMapUnknown;
  }
  return kPolicyMapUnknown;
}

void FillTime(const std::int64_t nanoseconds, builtin_interfaces::msg::Time& output) {
  const std::int64_t seconds = nanoseconds / 1000000000LL;
  output.sec = static_cast<std::int32_t>(seconds);
  output.nanosec = static_cast<std::uint32_t>(nanoseconds - seconds * 1000000000LL);
}

}  // namespace

std::uint8_t EffectiveObservedState(const core::IntrinsicCellState intrinsic,
                                    const float center_elevation_m) noexcept {
  if (!std::isfinite(center_elevation_m)) return kPolicyMapUnknown;
  switch (intrinsic) {
    case core::IntrinsicCellState::kFree: return kPolicyMapFree;
    case core::IntrinsicCellState::kBlocked: return kPolicyMapBlocked;
    case core::IntrinsicCellState::kUnknown: return kPolicyMapUnknown;
  }
  return kPolicyMapUnknown;
}

PolicyMapExporter::PolicyMapExporter(core::PlatformCapability capability,
                                     core::TraversabilityProfile profile)
    : capability_(std::move(capability)), profile_(std::move(profile)) {}

lunar_planning_msgs::srv::GetPolicyMap::Response PolicyMapExporter::Export(
    const PolicyMapExportInput& input, const std::uint64_t since_revision,
    const std::int64_t minimum_map_stamp_ns) const {
  lunar_planning_msgs::srv::GetPolicyMap::Response response;
  if (!input.fine) {
    response.ready = false;
    response.reason_code = "INPUT_UNAVAILABLE";
    return response;
  }
  if (input.processed_stamp_ns < minimum_map_stamp_ns) {
    response.ready = false;
    response.reason_code = "MAP_STAMP_TOO_OLD";
    return response;
  }
  const auto& fine = *input.fine;
  response.ready = true;
  response.reason_code = "READY";
  response.epoch = input.epoch;
  FillTime(input.processed_stamp_ns, response.processed_stamp);
  response.raw_elevation_revision = fine.raw_elevation_revision();
  response.fine_revision = fine.fine_traversability_revision();
  response.full_snapshot = since_revision + 1U != response.fine_revision;
  const auto& geometry = fine.geometry();
  response.frame_id = geometry.frame_id();
  response.resolution_m = geometry.resolution_m();
  response.origin.x = geometry.origin_m().x;
  response.origin.y = geometry.origin_m().y;
  response.origin.z = geometry.origin_m().z;
  response.profile_hash = fine.platform_profile_hash();
  response.goal_position_tolerance_m = profile_.goal_position_tolerance_m;
  response.goal_yaw_tolerance_rad = profile_.goal_yaw_tolerance_rad;
  response.local_bounds = {geometry.min_inclusive().x, geometry.min_inclusive().y,
                           geometry.max_exclusive().x, geometry.max_exclusive().y};

  if (input.anchor) {
    response.anchor_pose.position.x = input.anchor->position_m.x;
    response.anchor_pose.position.y = input.anchor->position_m.y;
    response.anchor_pose.orientation.w = std::cos(input.anchor->yaw_rad / 2.0);
    response.anchor_pose.orientation.z = std::sin(input.anchor->yaw_rad / 2.0);
    const auto local = core::BuildLocalPlanningWindow(
        fine, input.anchor->position_m, input.local_window_size_m);
    if (local.geometry) {
      const auto connections = core::RequestLocalStartPatchBuilder{}.BuildStartConnections(
          input.fine, *local.geometry, *input.anchor, capability_, profile_);
      response.start_connection_status = static_cast<std::uint8_t>(connections.status);
      for (const auto& connection : connections.connections) {
        response.start_connection_x.push_back(connection.index.x);
        response.start_connection_y.push_back(connection.index.y);
      }
      response.local_bounds = {local.geometry->min_inclusive().x, local.geometry->min_inclusive().y,
                               local.geometry->max_exclusive().x, local.geometry->max_exclusive().y};
    } else {
      response.start_connection_status = static_cast<std::uint8_t>(core::StartPatchResult::Status::kUnresolved);
    }
  } else {
    response.start_connection_status = static_cast<std::uint8_t>(core::StartPatchResult::Status::kUnresolved);
  }

  const std::vector<core::TileIndex> indices = response.full_snapshot
      ? fine.tile_indices()
      : std::vector<core::TileIndex>(fine.changed_tiles().begin(), fine.changed_tiles().end());
  const auto intrinsic_evaluator = core::MakePlatformElevationEvaluator(capability_);
  for (const core::TileIndex index : indices) {
    const auto tile = fine.FindTile(index);
    if (!tile) continue;
    lunar_planning_msgs::msg::PolicyMapTile output;
    output.tile_x = index.x;
    output.tile_y = index.y;
    output.states.reserve(core::kGridTileCellCount);
    output.intrinsic_states.reserve(core::kGridTileCellCount);
    output.observed.reserve(core::kGridTileCellCount);
    output.costs.reserve(core::kGridTileCellCount);
    output.elevation_m.reserve(core::kGridTileCellCount);
    for (std::size_t offset = 0U; offset < core::kGridTileCellCount; ++offset) {
      const core::GridIndex cell{.x = index.x * core::kGridTileWidthCells +
                                       static_cast<std::int64_t>(offset % core::kGridTileWidthCells),
                                 .y = index.y * core::kGridTileWidthCells +
                                       static_cast<std::int64_t>(offset / core::kGridTileWidthCells)};
      const auto center = fine.elevation()->ElevationAt(cell);
      const auto classification = intrinsic_evaluator->Evaluate(*fine.elevation(), cell);
      output.states.push_back(WireState(tile->State(offset)));
      output.intrinsic_states.push_back(static_cast<std::uint8_t>(classification.state));
      output.observed.push_back(EffectiveObservedState(classification.state,
          center.value_or(std::numeric_limits<float>::quiet_NaN())));
      output.costs.push_back(static_cast<float>(tile->TraversalCost(offset)));
      output.elevation_m.push_back(center.value_or(std::numeric_limits<float>::quiet_NaN()));
    }
    response.tiles.push_back(std::move(output));
  }
  return response;
}

}  // namespace lunar::incremental_navigation_ros
