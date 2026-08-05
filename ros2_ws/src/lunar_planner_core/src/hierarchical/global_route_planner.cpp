#include "hierarchical/global_route_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <stop_token>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "hierarchical/frame_transform.hpp"
#include "hierarchical/grid_search.hpp"
#include "hierarchical/map_level.hpp"
#include "shared/map_snapshot.hpp"
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
  case GlobalSearchStatus::kResourceExhausted:
    return Failure(PlanningOutcome::kResourceExhausted,
                   "GLOBAL_SEARCH_RESOURCE_LIMIT", started, level);
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

} // namespace

GlobalRoutePlanResult PlanGroundGlobalRoute(const PlannerInput &input) {
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

  const auto map = shared::MapSnapshot::Create(input.world.global_map);
  if (!map.ok()) {
    return Failure(PlanningOutcome::kInvalidRequest, map.reason_code, started,
                   level);
  }
  const auto projection =
      shared::BuildSafeProjection(map.snapshot, input.capability,
                                  input.config.map_safety, input.stop_token);
  if (!projection.ok()) {
    const PlanningOutcome outcome = projection.reason_code == "REQUEST_CANCELED"
                                        ? PlanningOutcome::kCanceled
                                        : PlanningOutcome::kInvalidRequest;
    return Failure(outcome, projection.reason_code, started, level);
  }
  const auto terrain_limits =
      shared::ResolveTerrainLimits(input.capability, input.config.map_safety);
  if (!terrain_limits.ok()) {
    return Failure(PlanningOutcome::kInvalidRequest, terrain_limits.reason_code,
                   started, level);
  }

  const auto current_pose = CurrentPose(input);
  const auto start_pose_map =
      current_pose ? TransformPose(*current_pose, input.world.map_from_odom,
                                   TransformDirection::kChildToParent)
                   : std::nullopt;
  if (!start_pose_map) {
    return Failure(PlanningOutcome::kInvalidRequest,
                   "GLOBAL_START_TRANSFORM_INVALID", started, level);
  }
  const auto start_cell = map.snapshot->PositionToCell(
      Vec2{start_pose_map->position_m.x, start_pose_map->position_m.y});
  if (!start_cell || !projection.projection->HardFeasible(*start_cell)) {
    return Failure(PlanningOutcome::kNoKnownSafeRoute,
                   "GLOBAL_NO_KNOWN_SAFE_ROUTE", started, level);
  }

  std::vector<std::uint8_t> goal_mask =
      BuildGoalMask(input.goal_map, *projection.projection, input.stop_token);
  if (input.stop_token.stop_requested()) {
    return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED", started,
                   level);
  }
  if (goal_mask.empty() ||
      std::ranges::none_of(
          goal_mask, [](const std::uint8_t value) { return value != 0U; })) {
    return Failure(PlanningOutcome::kGoalInfeasible, "GLOBAL_GOAL_INFEASIBLE",
                   started, level);
  }

  GlobalGridSearchResult search = SearchGlobalGrid(GlobalGridSearchProblem{
      .projection = *projection.projection,
      .start = *start_cell,
      .goal_mask = goal_mask,
      .maximum_speed_mps = terrain_limits.limits->maximum_speed_mps,
      .config = input.config.global_search,
      .stop_token = input.stop_token,
  });
  if (!search.ok()) {
    return SearchFailure(search, started, level);
  }
  std::vector<shared::GridCell> simplified = SimplifyRouteSupercover(
      *projection.projection, search.path_cells,
      input.config.global_search.maximum_preview_points);
  if (simplified.empty()) {
    return Failure(PlanningOutcome::kResourceExhausted,
                   "GLOBAL_SEARCH_RESOURCE_LIMIT", started, level);
  }
  std::vector<Pose3> poses =
      BuildPoses(*map.snapshot, simplified, input.goal_map, *start_pose_map);
  if (poses.empty()) {
    return Failure(PlanningOutcome::kNumericalFailure,
                   "GLOBAL_ROUTE_RESULT_INVALID", started, level);
  }

  return GlobalRoutePlanResult{
      .outcome = PlanningOutcome::kNewReferenceAvailable,
      .reason_code = "GLOBAL_ROUTE_AVAILABLE",
      .route =
          GlobalRoute{
              .raw_cells = std::move(search.path_cells),
              .simplified_cells = std::move(simplified),
              .poses_map = std::move(poses),
              .cost = search.cost,
              .expanded_states = search.expanded_states,
              .open_peak = search.open_peak,
              .estimated_work_memory_bytes = search.estimated_work_memory_bytes,
          },
      .global_level = level,
      .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - started),
  };
}

} // namespace lunar::planning::hierarchical
