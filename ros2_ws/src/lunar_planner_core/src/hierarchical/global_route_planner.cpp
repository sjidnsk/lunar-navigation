#include "hierarchical/global_route_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <stop_token>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "hierarchical/frame_transform.hpp"
#include "hierarchical/grid_search.hpp"
#include "hierarchical/map_level.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/projection_cache.hpp"
#include "shared/safe_projection.hpp"
#include "shared/terrain_checks.hpp"

namespace lunar::planning::hierarchical {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] GlobalRoutePlanResult
Failure(const PlanningOutcome outcome, std::string reason_code,
        const Clock::time_point started,
        const std::optional<std::size_t> level = std::nullopt) {
  return GlobalRoutePlanResult{
      .outcome = outcome,
      .reason_code = std::move(reason_code),
      .route = std::nullopt,
      .global_level = level,
      .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - started),
  };
}

[[nodiscard]] bool Finite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] std::optional<Pose3>
CurrentPose(const PlannerInput &input) noexcept {
  if (const auto *state = std::get_if<WheeledState>(&input.current_state)) {
    return state->pose;
  }
  if (const auto *state = std::get_if<LeggedState>(&input.current_state)) {
    return state->body_pose;
  }
  return std::nullopt;
}

[[nodiscard]] bool PointOnSegment(const Vec2 point, const Vec2 start,
                                  const Vec2 end) noexcept {
  constexpr double kTolerance = 1.0e-9;
  const double cross = (point.x - start.x) * (end.y - start.y) -
                       (point.y - start.y) * (end.x - start.x);
  if (std::abs(cross) > kTolerance) {
    return false;
  }
  const double dot = (point.x - start.x) * (point.x - end.x) +
                     (point.y - start.y) * (point.y - end.y);
  return dot <= kTolerance;
}

[[nodiscard]] bool PointInPolygon(const Vec2 point,
                                  const std::vector<Vec3> &boundary) noexcept {
  bool inside = false;
  for (std::size_t current = 0U, previous = boundary.size() - 1U;
       current < boundary.size(); previous = current++) {
    const Vec2 start{boundary[previous].x, boundary[previous].y};
    const Vec2 end{boundary[current].x, boundary[current].y};
    if (PointOnSegment(point, start, end)) {
      return true;
    }
    const bool straddles = (start.y > point.y) != (end.y > point.y);
    if (!straddles) {
      continue;
    }
    const double intersection_x =
        start.x + (point.y - start.y) * (end.x - start.x) / (end.y - start.y);
    if (point.x < intersection_x) {
      inside = !inside;
    }
  }
  return inside;
}

[[nodiscard]] bool ValidGoal(const GoalRegion &goal) noexcept {
  if (goal.goal_id.empty() || !std::isfinite(goal.yaw_tolerance_rad) ||
      goal.yaw_tolerance_rad < 0.0 ||
      (goal.yaw_rad && !std::isfinite(*goal.yaw_rad))) {
    return false;
  }
  if (const auto *point = std::get_if<PointGoal>(&goal.target)) {
    return Finite(point->position_m) && std::isfinite(point->tolerance_m) &&
           point->tolerance_m >= 0.0;
  }
  const auto *region = std::get_if<PlanarRegionGoal>(&goal.target);
  return region != nullptr && region->boundary_m.size() >= 3U &&
         std::ranges::all_of(region->boundary_m, Finite) &&
         std::isfinite(region->normal_tolerance_m) &&
         region->normal_tolerance_m >= 0.0;
}

[[nodiscard]] bool GoalContains(const GoalRegion &goal, const Vec3 cell_center,
                                const double cell_resolution_m) noexcept {
  if (const auto *point = std::get_if<PointGoal>(&goal.target)) {
    const double half_cell = cell_resolution_m / 2.0;
    const double delta_x = std::max(
        std::abs(cell_center.x - point->position_m.x) - half_cell, 0.0);
    const double delta_y = std::max(
        std::abs(cell_center.y - point->position_m.y) - half_cell, 0.0);
    return std::hypot(delta_x, delta_y) <= point->tolerance_m + 1.0e-9;
  }
  const auto &region = std::get<PlanarRegionGoal>(goal.target);
  return PointInPolygon(Vec2{cell_center.x, cell_center.y}, region.boundary_m);
}

[[nodiscard]] std::vector<std::uint8_t>
BuildGoalMask(const GoalRegion &goal, const shared::SafeProjection &projection,
              const std::stop_token stop_token) {
  const auto &map = projection.source_map();
  std::vector<std::uint8_t> mask(map->cell_count(), 0U);
  for (std::size_t index = 0U; index < map->cell_count(); ++index) {
    if (stop_token.stop_requested()) {
      return {};
    }
    const shared::GridCell cell{
        .x = static_cast<std::int32_t>(index % map->width()),
        .y = static_cast<std::int32_t>(index / map->width()),
    };
    if (projection.HardFeasible(cell) &&
        GoalContains(goal, map->CellCenter(cell), map->resolution_m())) {
      mask[index] = 1U;
    }
  }
  return mask;
}

[[nodiscard]] bool
GoalHasUnknownGlobalSupport(const GoalRegion &goal,
                            const shared::SafeProjection &projection) {
  const auto &map = projection.source_map();
  for (std::size_t index = 0U; index < map->cell_count(); ++index) {
    const shared::GridCell cell{
        .x = static_cast<std::int32_t>(index % map->width()),
        .y = static_cast<std::int32_t>(index / map->width()),
    };
    if (!projection.Known(cell) &&
        GoalContains(goal, map->CellCenter(cell), map->resolution_m())) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] Quaternion YawQuaternion(const double yaw_rad) noexcept {
  return Quaternion{
      .w = std::cos(yaw_rad / 2.0),
      .x = 0.0,
      .y = 0.0,
      .z = std::sin(yaw_rad / 2.0),
  };
}

[[nodiscard]] std::vector<Pose3>
BuildPoses(const shared::MapSnapshot &map,
           const std::vector<shared::GridCell> &cells, const GoalRegion &goal,
           const Pose3 &start_pose_map) {
  std::vector<Pose3> poses;
  poses.reserve(cells.size());
  for (std::size_t index = 0U; index < cells.size(); ++index) {
    const Vec3 position =
        index == 0U ? start_pose_map.position_m : map.CellCenter(cells[index]);
    Quaternion orientation = start_pose_map.orientation;
    if (index > 0U && index + 1U < cells.size()) {
      const Vec3 next = map.CellCenter(cells[index + 1U]);
      orientation =
          YawQuaternion(std::atan2(next.y - position.y, next.x - position.x));
    } else if (goal.yaw_rad) {
      orientation = YawQuaternion(*goal.yaw_rad);
    } else if (index > 0U) {
      const Vec3 previous = map.CellCenter(cells[index - 1U]);
      orientation = YawQuaternion(
          std::atan2(position.y - previous.y, position.x - previous.x));
    }
    poses.push_back(Pose3{
        .position_m = position,
        .orientation = orientation,
    });
  }
  return poses;
}

[[nodiscard]] GlobalRoutePlanResult
SearchFailure(const GlobalGridSearchResult &search,
              const Clock::time_point started, const std::size_t level) {
  switch (search.status) {
  case GlobalSearchStatus::kCanceled:
    return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED", started,
                   level);
  case GlobalSearchStatus::kAllocationFailed:
    return Failure(PlanningOutcome::kResourceExhausted,
                   "GLOBAL_SEARCH_ALLOCATION_FAILED", started, level);
  case GlobalSearchStatus::kNoPath:
    return Failure(PlanningOutcome::kNoKnownSafeRoute,
                   "GLOBAL_NO_KNOWN_SAFE_ROUTE", started, level);
  case GlobalSearchStatus::kInvalidProblem:
    return Failure(PlanningOutcome::kNumericalFailure, search.reason_code,
                   started, level);
  case GlobalSearchStatus::kSolved:
    break;
  }
  return Failure(PlanningOutcome::kNumericalFailure,
                 "GLOBAL_SEARCH_RESULT_INVALID", started, level);
}

struct LocalStartPortal final {
  shared::GridCell global_cell;
  GlobalGridSearchResult connector;
  Pose3 portal_pose_map;
};

[[nodiscard]] std::vector<Pose3>
BuildConnectorPoses(const shared::MapSnapshot &local_map,
                    const shared::SafeProjection &local_projection,
                    const GlobalGridSearchResult &connector,
                    const Pose3 &start_pose_map,
                    const RigidTransform &map_from_odom) {
  std::vector<shared::GridCell> simplified = SimplifyRouteSupercover(
      local_projection, connector.path_cells, connector.path_cells.size());
  if (simplified.empty()) {
    return {};
  }
  std::vector<Pose3> poses{start_pose_map};
  poses.reserve(simplified.size() + 1U);
  for (auto cell = std::next(simplified.begin()); cell != simplified.end();
       ++cell) {
    const auto pose_map = TransformPose(
        Pose3{
            .position_m = local_map.CellCenter(*cell),
            .orientation = start_pose_map.orientation,
        },
        map_from_odom, TransformDirection::kChildToParent);
    if (!pose_map.has_value()) {
      return {};
    }
    poses.push_back(*pose_map);
  }
  return poses;
}

[[nodiscard]] std::optional<GlobalRoutePlanResult>
PlanObservedLocalGoal(const PlannerInput &input, const Pose3 &start_pose_map,
                      const Clock::time_point started,
                      const std::size_t level) {
  const shared::MapSnapshotBuildResult local_map =
      shared::MapSnapshot::Create(input.world.local_map);
  if (!local_map.ok()) {
    return Failure(PlanningOutcome::kInvalidRequest, local_map.reason_code,
                   started, level);
  }
  const shared::SafeProjectionBuildResult local_projection =
      shared::BuildSafeProjection(local_map.snapshot, input.capability,
                                  input.config.map_safety,
                                  input.stop_token);
  if (!local_projection.ok()) {
    const PlanningOutcome outcome =
        local_projection.reason_code == "REQUEST_CANCELED"
            ? PlanningOutcome::kCanceled
            : PlanningOutcome::kInvalidRequest;
    return Failure(outcome, local_projection.reason_code, started, level);
  }
  const auto goal_odom =
      TransformGoal(input.goal_map, input.world.map_from_odom,
                    TransformDirection::kParentToChild);
  if (!goal_odom.has_value()) {
    return Failure(PlanningOutcome::kInvalidRequest,
                   "FRAME_TRANSFORM_INVALID", started, level);
  }
  std::vector<std::uint8_t> goal_mask =
      BuildGoalMask(*goal_odom, *local_projection.projection,
                    input.stop_token);
  if (input.stop_token.stop_requested()) {
    return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED", started,
                   level);
  }
  if (goal_mask.empty() ||
      std::ranges::none_of(
          goal_mask, [](const std::uint8_t value) { return value != 0U; })) {
    return std::nullopt;
  }
  const auto current_pose_odom = CurrentPose(input);
  const auto start_cell =
      current_pose_odom
          ? local_map.snapshot->PositionToCell(
                Vec2{.x = current_pose_odom->position_m.x,
                     .y = current_pose_odom->position_m.y})
          : std::nullopt;
  if (!start_cell.has_value() ||
      !local_projection.projection->HardFeasible(*start_cell)) {
    return Failure(PlanningOutcome::kNoKnownSafeRoute,
                   "GLOBAL_NO_KNOWN_SAFE_ROUTE", started, level);
  }
  GlobalGridSearchResult search =
      SearchGlobalGrid(GlobalGridSearchProblem{
          .projection = *local_projection.projection,
          .start = *start_cell,
          .goal_mask = goal_mask,
          .excluded_mask = {},
          .maximum_speed_mps = 1.0,
          .config = input.config.global_search,
          .stop_token = input.stop_token,
      });
  if (!search.ok()) {
    return SearchFailure(search, started, level);
  }
  std::vector<Pose3> poses = BuildConnectorPoses(
      *local_map.snapshot, *local_projection.projection, search,
      start_pose_map, input.world.map_from_odom);
  if (poses.size() < 2U) {
    return Failure(PlanningOutcome::kNumericalFailure,
                   "LOCAL_DETAIL_ROUTE_RESULT_INVALID", started, level);
  }
  const double cost = search.cost;
  const std::uint64_t expanded_states = search.expanded_states;
  const std::size_t open_peak = search.open_peak;
  const std::size_t estimated_work_memory_bytes =
      search.estimated_work_memory_bytes;
  return GlobalRoutePlanResult{
      .outcome = PlanningOutcome::kNewReferenceAvailable,
      .reason_code = "GLOBAL_ROUTE_AVAILABLE",
      .route =
          GlobalRoute{
              // These cells belong to the odom-local raster, not the global
              // raster, so only the transformed pose chain crosses this API.
              .raw_cells = {},
              .conditional_cells = {},
              .simplified_cells = {},
              .poses_map = std::move(poses),
              .cost = cost,
              .expanded_states = expanded_states,
              .open_peak = open_peak,
              .estimated_work_memory_bytes = estimated_work_memory_bytes,
          },
      .global_level = level,
      .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - started),
      .projection_cache_hit = false,
  };
}

} // namespace

GlobalRoutePlanResult
PlanGroundGlobalRoute(const PlannerInput &input,
                      const std::span<const shared::GridCell> excluded_cells,
                      shared::ProjectionCache *const projection_cache) {
  const Clock::time_point started = Clock::now();
  if (input.stop_token.stop_requested()) {
    return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED", started);
  }
  const PlatformType platform = CapabilityPlatform(input.capability);
  if ((platform != PlatformType::kWheeled &&
       platform != PlatformType::kLegged) ||
      (platform == PlatformType::kWheeled &&
       !std::holds_alternative<WheeledState>(input.current_state)) ||
      (platform == PlatformType::kLegged &&
       !std::holds_alternative<LeggedState>(input.current_state)) ||
      input.world.global_map.frame_id != "map" ||
      input.world.local_map.frame_id != "odom" || !ValidGoal(input.goal_map)) {
    return Failure(PlanningOutcome::kInvalidRequest, "GLOBAL_REQUEST_INVALID",
                   started);
  }

  const MapLevelValidationResult levels =
      ValidateMapLevels(input.world, input.config.global_map);
  if (!levels.ok()) {
    const PlanningOutcome outcome =
        levels.reason_code == "GLOBAL_MAP_SCALE_UNSUPPORTED"
            ? PlanningOutcome::kResourceExhausted
            : PlanningOutcome::kInvalidRequest;
    return Failure(outcome, levels.reason_code, started);
  }
  const std::size_t level = *levels.global_level;

  shared::ProjectionCache local_cache;
  shared::ProjectionCache &cache =
      projection_cache == nullptr ? local_cache : *projection_cache;
  const shared::ProjectionContextResult projection = cache.GetOrBuild(
      shared::MakeProjectionCacheKey(
          input.global_map_generation, input.platform_id,
          input.capability_version, input.world.global_map, input.capability,
          input.config.map_safety),
      input.world.global_map, input.capability, input.config.map_safety,
      input.stop_token);
  if (!projection.ok()) {
    const PlanningOutcome outcome = projection.reason_code == "REQUEST_CANCELED"
                                        ? PlanningOutcome::kCanceled
                                        : PlanningOutcome::kInvalidRequest;
    return Failure(outcome, projection.reason_code, started, level);
  }
  const auto &map = projection.context->map;
  const auto &safe_projection = projection.context->projection;

  const auto current_pose = CurrentPose(input);
  const auto start_pose_map =
      current_pose ? TransformPose(*current_pose, input.world.map_from_odom,
                                   TransformDirection::kChildToParent)
                   : std::nullopt;
  if (!start_pose_map) {
    return Failure(PlanningOutcome::kInvalidRequest,
                   "GLOBAL_START_TRANSFORM_INVALID", started, level);
  }
  const auto start_cell = map->PositionToCell(
      Vec2{start_pose_map->position_m.x, start_pose_map->position_m.y});
  if (!start_cell) {
    return Failure(PlanningOutcome::kNoKnownSafeRoute,
                   "GLOBAL_NO_KNOWN_SAFE_ROUTE", started, level);
  }

  std::vector<std::uint8_t> goal_mask =
      BuildGoalMask(input.goal_map, *safe_projection, input.stop_token);
  if (input.stop_token.stop_requested()) {
    return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED", started,
                   level);
  }
  if (goal_mask.empty() ||
      std::ranges::none_of(
          goal_mask, [](const std::uint8_t value) { return value != 0U; })) {
    if (GoalHasUnknownGlobalSupport(input.goal_map, *safe_projection)) {
      if (auto local =
              PlanObservedLocalGoal(input, *start_pose_map, started, level);
          local.has_value()) {
        return std::move(*local);
      }
    }
    return Failure(PlanningOutcome::kGoalInfeasible, "GLOBAL_GOAL_INFEASIBLE",
                   started, level);
  }

  std::vector<std::uint8_t> excluded_mask(map->cell_count(), 0U);
  for (const shared::GridCell cell : excluded_cells) {
    if (!map->InBounds(cell)) {
      return Failure(PlanningOutcome::kNumericalFailure,
                     "GLOBAL_EXCLUDED_CORRIDOR_INVALID", started, level);
    }
    excluded_mask[map->Index(cell)] = 1U;
  }

  std::optional<LocalStartPortal> selected_portal;
  GlobalGridSearchResult search;
  if (safe_projection->HardFeasible(*start_cell) &&
      excluded_mask[map->Index(*start_cell)] == 0U) {
    search = SearchGlobalGrid(GlobalGridSearchProblem{
        .projection = *safe_projection,
        .start = *start_cell,
        .goal_mask = goal_mask,
        .excluded_mask = excluded_mask,
        .maximum_speed_mps =
            projection.context->terrain_limits.maximum_speed_mps,
        .config = input.config.global_search,
        .stop_token = input.stop_token,
    });
  } else {
    const shared::MapSnapshotBuildResult local_map =
        shared::MapSnapshot::Create(input.world.local_map);
    if (!local_map.ok()) {
      return Failure(PlanningOutcome::kInvalidRequest, local_map.reason_code,
                     started, level);
    }
    const shared::SafeProjectionBuildResult local_projection =
        shared::BuildSafeProjection(local_map.snapshot, input.capability,
                                    input.config.map_safety, input.stop_token);
    if (!local_projection.ok()) {
      const PlanningOutcome outcome =
          local_projection.reason_code == "REQUEST_CANCELED"
              ? PlanningOutcome::kCanceled
              : PlanningOutcome::kInvalidRequest;
      return Failure(outcome, local_projection.reason_code, started, level);
    }
    const auto current_pose_odom = CurrentPose(input);
    const auto local_start = current_pose_odom
                                 ? local_map.snapshot->PositionToCell(Vec2{
                                       current_pose_odom->position_m.x,
                                       current_pose_odom->position_m.y,
                                   })
                                 : std::nullopt;
    if (!local_start.has_value() ||
        !local_projection.projection->HardFeasible(*local_start)) {
      return Failure(PlanningOutcome::kNoKnownSafeRoute,
                     "GLOBAL_NO_KNOWN_SAFE_ROUTE", started, level);
    }
    const std::int32_t local_component =
        local_projection.projection->ConnectedComponent(*local_start);
    std::vector<LocalStartPortal> portals;
    for (std::size_t index = 0U; index < map->cell_count(); ++index) {
      if (input.stop_token.stop_requested()) {
        return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED", started,
                       level);
      }
      const shared::GridCell global_cell{
          .x = static_cast<std::int32_t>(index % map->width()),
          .y = static_cast<std::int32_t>(index / map->width()),
      };
      if (excluded_mask[index] != 0U ||
          !safe_projection->HardFeasible(global_cell)) {
        continue;
      }
      const auto portal_pose_odom = TransformPose(
          Pose3{
              .position_m = map->CellCenter(global_cell),
              .orientation = current_pose_odom->orientation,
          },
          input.world.map_from_odom, TransformDirection::kParentToChild);
      if (!portal_pose_odom.has_value()) {
        return Failure(PlanningOutcome::kInvalidRequest,
                       "GLOBAL_START_TRANSFORM_INVALID", started, level);
      }
      const auto local_cell = local_map.snapshot->PositionToCell(Vec2{
          portal_pose_odom->position_m.x,
          portal_pose_odom->position_m.y,
      });
      if (!local_cell.has_value() ||
          !local_projection.projection->HardFeasible(*local_cell) ||
          local_projection.projection->ConnectedComponent(*local_cell) !=
              local_component) {
        continue;
      }
      std::vector<std::uint8_t> portal_mask(local_map.snapshot->cell_count(),
                                            0U);
      portal_mask[local_map.snapshot->Index(*local_cell)] = 1U;
      GlobalGridSearchResult connector =
          SearchGlobalGrid(GlobalGridSearchProblem{
              .projection = *local_projection.projection,
              .start = *local_start,
              .goal_mask = portal_mask,
              .excluded_mask = {},
              .maximum_speed_mps =
                  projection.context->terrain_limits.maximum_speed_mps,
              .config = input.config.global_search,
              .stop_token = input.stop_token,
          });
      if (!connector.ok()) {
        if (connector.status == GlobalSearchStatus::kNoPath) {
          continue;
        }
        return SearchFailure(connector, started, level);
      }
      const auto portal_pose_map = TransformPose(
          Pose3{
              .position_m = local_map.snapshot->CellCenter(*local_cell),
              .orientation = current_pose_odom->orientation,
          },
          input.world.map_from_odom, TransformDirection::kChildToParent);
      if (!portal_pose_map.has_value()) {
        return Failure(PlanningOutcome::kInvalidRequest,
                       "GLOBAL_START_TRANSFORM_INVALID", started, level);
      }
      portals.push_back(LocalStartPortal{
          .global_cell = global_cell,
          .connector = std::move(connector),
          .portal_pose_map = *portal_pose_map,
      });
    }
    std::ranges::sort(portals, {}, [](const LocalStartPortal &portal) {
      return std::tuple{portal.connector.cost, portal.global_cell.y,
                        portal.global_cell.x};
    });
    for (LocalStartPortal &portal : portals) {
      GlobalGridSearchResult candidate =
          SearchGlobalGrid(GlobalGridSearchProblem{
              .projection = *safe_projection,
              .start = portal.global_cell,
              .goal_mask = goal_mask,
              .excluded_mask = excluded_mask,
              .maximum_speed_mps =
                  projection.context->terrain_limits.maximum_speed_mps,
              .config = input.config.global_search,
              .stop_token = input.stop_token,
          });
      if (candidate.ok()) {
        search = std::move(candidate);
        selected_portal = std::move(portal);
        break;
      }
      if (candidate.status != GlobalSearchStatus::kNoPath) {
        return SearchFailure(candidate, started, level);
      }
    }
    if (!selected_portal.has_value()) {
      return Failure(PlanningOutcome::kNoKnownSafeRoute,
                     "GLOBAL_NO_KNOWN_SAFE_ROUTE", started, level);
    }
  }
  if (!search.ok()) {
    return SearchFailure(search, started, level);
  }
  std::vector<shared::GridCell> simplified =
      SimplifyRouteSupercover(*safe_projection, search.path_cells,
                              search.path_cells.size(), excluded_mask);
  if (simplified.empty()) {
    return Failure(PlanningOutcome::kNumericalFailure,
                   "GLOBAL_ROUTE_SIMPLIFICATION_FAILED", started, level);
  }
  std::vector<Pose3> poses;
  if (selected_portal.has_value()) {
    const shared::MapSnapshotBuildResult local_map =
        shared::MapSnapshot::Create(input.world.local_map);
    const shared::SafeProjectionBuildResult local_projection =
        local_map.ok()
            ? shared::BuildSafeProjection(local_map.snapshot, input.capability,
                                          input.config.map_safety,
                                          input.stop_token)
            : shared::SafeProjectionBuildResult{};
    if (!local_map.ok() || !local_projection.ok()) {
      return Failure(PlanningOutcome::kNumericalFailure,
                     "GLOBAL_START_CONNECTOR_INVALID", started, level);
    }
    poses = BuildConnectorPoses(
        *local_map.snapshot, *local_projection.projection,
        selected_portal->connector, *start_pose_map, input.world.map_from_odom);
    std::vector<Pose3> global_poses = BuildPoses(
        *map, simplified, input.goal_map, selected_portal->portal_pose_map);
    if (poses.empty() || global_poses.empty()) {
      return Failure(PlanningOutcome::kNumericalFailure,
                     "GLOBAL_START_CONNECTOR_INVALID", started, level);
    }
    poses.insert(poses.end(), std::next(global_poses.begin()),
                 global_poses.end());
  } else {
    poses = BuildPoses(*map, simplified, input.goal_map, *start_pose_map);
  }
  if (poses.empty()) {
    return Failure(PlanningOutcome::kNumericalFailure,
                   "GLOBAL_ROUTE_RESULT_INVALID", started, level);
  }
  std::vector<shared::GridCell> conditional_cells;
  std::ranges::copy_if(search.path_cells, std::back_inserter(conditional_cells),
                       [&](const shared::GridCell cell) {
                         return safe_projection->ClearanceClassification(
                                    cell) ==
                                shared::ClearanceClass::kConditional;
                       });

  return GlobalRoutePlanResult{
      .outcome = PlanningOutcome::kNewReferenceAvailable,
      .reason_code = "GLOBAL_ROUTE_AVAILABLE",
      .route =
          GlobalRoute{
              .raw_cells = std::move(search.path_cells),
              .conditional_cells = std::move(conditional_cells),
              .simplified_cells = std::move(simplified),
              .poses_map = std::move(poses),
              .cost = search.cost + (selected_portal.has_value()
                                         ? selected_portal->connector.cost
                                         : 0.0),
              .expanded_states =
                  search.expanded_states +
                  (selected_portal.has_value()
                       ? selected_portal->connector.expanded_states
                       : 0U),
              .open_peak = std::max(search.open_peak,
                                    selected_portal.has_value()
                                        ? selected_portal->connector.open_peak
                                        : 0U),
              .estimated_work_memory_bytes =
                  search.estimated_work_memory_bytes +
                  (selected_portal.has_value()
                       ? selected_portal->connector.estimated_work_memory_bytes
                       : 0U),
          },
      .global_level = level,
      .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - started),
      .projection_cache_hit = projection.cache_hit,
  };
}

} // namespace lunar::planning::hierarchical
