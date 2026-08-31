#include "hierarchical/global_route_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#include "hierarchical/frame_transform.hpp"
#include "hierarchical/surface_portal_set.hpp"
#include "hierarchical/surface_rolling_session.hpp"
#include "hierarchical/surface_global_search.hpp"
#include "shared/controlled_work.hpp"
#include "shared/active_planner_cache.hpp"
#include "shared/global_occupancy_projection.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::hierarchical {
namespace {

[[nodiscard]] Quaternion QuaternionFromYaw(const double yaw) noexcept {
  return Quaternion{.w = std::cos(yaw / 2.0), .z = std::sin(yaw / 2.0)};
}

[[nodiscard]] std::optional<double> YawFromQuaternion(
    const Quaternion& quaternion) noexcept {
  const double squared_norm = quaternion.w * quaternion.w +
                              quaternion.x * quaternion.x +
                              quaternion.y * quaternion.y +
                              quaternion.z * quaternion.z;
  if (!std::isfinite(squared_norm) || squared_norm <= 1.0e-24) {
    return std::nullopt;
  }
  const double inverse_norm = 1.0 / std::sqrt(squared_norm);
  const double w = quaternion.w * inverse_norm;
  const double x = quaternion.x * inverse_norm;
  const double y = quaternion.y * inverse_norm;
  const double z = quaternion.z * inverse_norm;
  const double yaw = std::atan2(2.0 * (w * z + x * y),
                                1.0 - 2.0 * (y * y + z * z));
  return std::isfinite(yaw) ? std::optional<double>{yaw} : std::nullopt;
}

[[nodiscard]] const Pose3* CurrentPose(const PlanningRequest& input) noexcept {
  return std::visit(
      [](const auto& state) -> const Pose3* {
        using State = std::decay_t<decltype(state)>;
        if constexpr (std::is_same_v<State, LeggedState>) {
          return &state.body_pose;
        } else {
          return &state.pose;
        }
      },
      input.current_state);
}

[[nodiscard]] bool PositionInsideGridMap(const GridMap& map,
                                         const Vec2 position) noexcept {
  if (map.CellCount() == 0U || !std::isfinite(map.resolution_m) ||
      map.resolution_m <= 0.0 || !std::isfinite(map.origin_m.x) ||
      !std::isfinite(map.origin_m.y) || !std::isfinite(position.x) ||
      !std::isfinite(position.y)) {
    return false;
  }
  const double relative_x =
      (position.x - map.origin_m.x) / map.resolution_m;
  const double relative_y =
      (position.y - map.origin_m.y) / map.resolution_m;
  return std::isfinite(relative_x) && std::isfinite(relative_y) &&
         relative_x >= 0.0 && relative_y >= 0.0 &&
         relative_x < static_cast<double>(map.width) &&
         relative_y < static_cast<double>(map.height);
}

[[nodiscard]] GlobalStageResult Failure(std::string reason_code) {
  return {.route = std::nullopt, .reason_code = std::move(reason_code)};
}

[[nodiscard]] std::string SurfaceFailureReason(
    const SurfaceGlobalSearchStatus status) {
  switch (status) {
    case SurfaceGlobalSearchStatus::kNoPath:
      return "NO_PATH";
    case SurfaceGlobalSearchStatus::kCanceled:
      return "REQUEST_CANCELED";
    case SurfaceGlobalSearchStatus::kTimedOut:
      return "TIMEOUT";
    case SurfaceGlobalSearchStatus::kInvalidProblem:
      return "INVALID_INPUT";
    case SurfaceGlobalSearchStatus::kResourceExhausted:
      return "PLANNER_ERROR";
    case SurfaceGlobalSearchStatus::kSolved:
      return "PLANNER_ERROR";
  }
  return "PLANNER_ERROR";
}

[[nodiscard]] double WheelGlobalInflation(const PlanningRequest& input) {
  const auto* wheel = std::get_if<WheeledCapability>(&input.capability);
  if (wheel == nullptr) {
    return 0.0;
  }
  if (!std::isfinite(wheel->minimum_clearance_m) ||
      wheel->minimum_clearance_m < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double footprint_radius_m{};
  for (const Vec2 vertex : wheel->footprint_xy_m) {
    if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    footprint_radius_m =
        std::max(footprint_radius_m, std::hypot(vertex.x, vertex.y));
  }
  return footprint_radius_m + wheel->minimum_clearance_m;
}

}  // namespace

GlobalStageResult PlanSurfaceGlobal(const PlanningRequest& input,
                                    SearchControl control) {
  shared::ActivePlannerCache cache;
  return PlanSurfaceGlobal(input, std::move(control), cache);
}

GlobalStageResult PlanSurfaceGlobal(const PlanningRequest& input,
                                    SearchControl control,
                                    const double inflation_m) {
  shared::ActivePlannerCache cache;
  return PlanSurfaceGlobal(input, std::move(control), cache, inflation_m);
}

GlobalStageResult PlanSurfaceGlobal(const PlanningRequest& input,
                                    SearchControl control,
                                    shared::ActivePlannerCache& cache) {
  return PlanSurfaceGlobal(input, std::move(control), cache,
                           WheelGlobalInflation(input));
}

GlobalStageResult PlanSurfaceGlobal(const PlanningRequest& input,
                                    SearchControl control,
                                    shared::ActivePlannerCache& cache,
                                    const double inflation_m) {
  if (!input.world.global_map.has_value()) {
    return Failure("INVALID_INPUT");
  }
  const bool use_legged_grid_v1 =
      std::holds_alternative<LeggedCapability>(input.capability) &&
      input.config.legged_global_mode ==
          LeggedGlobalMode::kGridTraversabilityV1;
  shared::ImmutableCacheResult<shared::MapSnapshot> map;
  if (!use_legged_grid_v1) {
    map = cache.global_snapshot().GetOrBuild(
        shared::MakeGlobalSnapshotCacheKey(input.world.global_map_sequence),
        control, [&](const SearchControl& build_control) {
          auto built = shared::MapSnapshot::Create(
              *input.world.global_map, shared::MapContract::kGlobalOccupancy,
              build_control);
          return shared::ImmutableCacheBuildResult<shared::MapSnapshot>{
              .value = std::move(built.snapshot),
              .reason_code = std::move(built.reason_code),
          };
        });
    if (!map.ok()) {
      return Failure(map.reason_code == "TIMEOUT" ||
                             map.reason_code == "REQUEST_CANCELED"
                         ? map.reason_code
                         : "INVALID_INPUT");
    }
  } else if (!input.world.traversability_snapshot ||
             !input.world.traversability_snapshot->valid()) {
    return Failure("INVALID_INPUT");
  }
  const std::uint64_t capability_fingerprint =
      shared::StableCapabilityFingerprint(input.capability);
  const auto projection_key = use_legged_grid_v1
      ? shared::MakeLeggedTraversabilityProjectionCacheKey(
            input.world.traversability_snapshot->revision(),
            input.world.traversability_snapshot->profile_hash(),
            capability_fingerprint)
      : shared::MakeGlobalProjectionCacheKey(
            input.world.global_map_sequence,
            input.config.global_occupancy_threshold, inflation_m,
            capability_fingerprint);
  const auto projection = cache.global_projection().GetOrBuild(
      projection_key,
      control, [&](const SearchControl& build_control) {
        auto built = use_legged_grid_v1
            ? shared::BuildLeggedTraversabilityProjection(
                  *input.world.traversability_snapshot, build_control)
            : shared::BuildInflatedGlobalOccupancyProjection(
                  map.value, input.config.global_occupancy_threshold,
                  inflation_m, build_control);
        return shared::ImmutableCacheBuildResult<
            shared::GlobalOccupancyProjection>{
            .value = !built.projection.has_value()
                         ? nullptr
                         : std::make_shared<
                               const shared::GlobalOccupancyProjection>(
                               std::move(*built.projection)),
            .reason_code = std::move(built.reason_code),
        };
      });
  if (!projection.ok()) {
    GlobalStageResult failure =
        Failure(projection.reason_code == "TIMEOUT" ||
                        projection.reason_code == "REQUEST_CANCELED"
                    ? projection.reason_code
                    : "INVALID_INPUT");
    failure.snapshot_cache_hit = !use_legged_grid_v1 && map.cache_hit;
    return failure;
  }
  const shared::MapSnapshot* projection_map = projection.value->View().map;
  if (projection_map == nullptr) {
    return Failure("INVALID_INPUT");
  }
  const auto failure_after_projection = [&](std::string reason_code) {
    GlobalStageResult failure = Failure(std::move(reason_code));
    failure.snapshot_cache_hit = !use_legged_grid_v1 && map.cache_hit;
    failure.projection_cache_hit = projection.cache_hit;
    return failure;
  };
  const auto* point = std::get_if<PointGoal>(&input.goal_map.target);
  const Pose3* current = CurrentPose(input);
  if (point == nullptr || current == nullptr) {
    return failure_after_projection("INVALID_INPUT");
  }
  const auto start_map = TransformPose(*current, input.world.map_from_odom,
                                       TransformDirection::kChildToParent);
  if (!start_map.has_value()) {
    return failure_after_projection("INVALID_INPUT");
  }
  Pose3 goal_pose{
      .position_m = {.x = point->position_m.x,
                     .y = point->position_m.y,
                     .z = projection_map->origin_m().z},
      .orientation = input.goal_map.yaw_rad.has_value()
                         ? QuaternionFromYaw(*input.goal_map.yaw_rad)
                         : start_map->orientation,
  };
  const auto start_cell = projection_map->PositionToCell(
      {.x = start_map->position_m.x, .y = start_map->position_m.y});
  const auto goal_cell = projection_map->PositionToCell(
      {.x = goal_pose.position_m.x, .y = goal_pose.position_m.y});
  if (!start_cell.has_value() || !goal_cell.has_value()) {
    return failure_after_projection("NO_PATH");
  }
  const auto route_key = use_legged_grid_v1
      ? shared::MakeLeggedTraversabilityRouteCacheKey(
            input, input.world.traversability_snapshot->revision(),
            input.world.traversability_snapshot->profile_hash(),
            capability_fingerprint)
      : shared::MakeGlobalRouteCacheKey(input, inflation_m,
                                        capability_fingerprint);
  const auto route = cache.global_route().GetOrBuild(
      route_key,
      control, [&](const SearchControl& build_control) {
        SurfaceGlobalSearchResult result = SearchSurfaceGlobal({
            .projection = projection.value->View(),
            .start = *start_cell,
            .goal = *goal_cell,
            .start_pose_map = *start_map,
            .goal_pose_map = goal_pose,
            .control = build_control,
            .search = input.config.search,
        });
        if (!result.ok()) {
          return shared::ImmutableCacheBuildResult<GlobalRoute>{
              .reason_code = SurfaceFailureReason(result.status)};
        }
        return shared::ImmutableCacheBuildResult<GlobalRoute>{
            .value = std::make_shared<const GlobalRoute>(GlobalRoute{
                .poses_map = std::move(result.preview.poses_map),
                .expanded_states = result.expanded_states,
            })};
      });
  if (!route.ok()) {
    return failure_after_projection(route.reason_code);
  }
  return GlobalStageResult{
      .route = *route.value,
      .reason_code = {},
      .snapshot_cache_hit = !use_legged_grid_v1 && map.cache_hit,
      .projection_cache_hit = projection.cache_hit,
      .route_cache_hit = route.cache_hit,
  };
}

std::optional<GoalRegion> GoalMapToOdomPlanar(
    const PlanningRequest& input, const GoalRegion& goal_map) {
  const auto* point = std::get_if<PointGoal>(&goal_map.target);
  if (point == nullptr) {
    return std::nullopt;
  }
  const auto position = TransformPoint(
      {.x = point->position_m.x, .y = point->position_m.y, .z = 0.0},
      input.world.map_from_odom, TransformDirection::kParentToChild);
  if (!position.has_value()) {
    return std::nullopt;
  }
  std::optional<double> yaw;
  if (goal_map.yaw_rad.has_value()) {
    const auto orientation = TransformOrientation(
        QuaternionFromYaw(*goal_map.yaw_rad), input.world.map_from_odom,
        TransformDirection::kParentToChild);
    if (!orientation.has_value()) {
      return std::nullopt;
    }
    yaw = YawFromQuaternion(*orientation);
    if (!yaw.has_value()) {
      return std::nullopt;
    }
  }
  return GoalRegion{
      .goal_id = goal_map.goal_id,
      .target = PointGoal{
          .position_m = {.x = position->x, .y = position->y, .z = 0.0},
          .tolerance_m = point->tolerance_m,
      },
      .yaw_rad = yaw,
      .yaw_tolerance_rad = goal_map.yaw_tolerance_rad,
  };
}

bool GoalInsideLocalMap(const PlanningRequest& input,
                        const GoalRegion& goal_map) {
  const auto goal = GoalMapToOdomPlanar(input, goal_map);
  if (!goal.has_value()) {
    return false;
  }
  const auto* point = std::get_if<PointGoal>(&goal->target);
  return point != nullptr &&
         PositionInsideGridMap(input.world.local_map,
                               {.x = point->position_m.x,
                                .y = point->position_m.y});
}

LocalGoalSetResult ConvertSurfacePortalsToLocalGoals(
    const SurfacePortalSetResult& portals,
    const SurfaceRollingDecision& decision, const Pose3& current_pose_odom,
    const GridMap& global_map, const GridMap& local_map,
    const bool snap_intermediate_to_local_cell) {
  constexpr double kNumericalEpsilon = 1.0e-9;
  if (!portals.ok() ||
      decision.kind != SurfaceRollingDecision::Kind::kNextPortalSet ||
      !std::isfinite(current_pose_odom.position_m.x) ||
      !std::isfinite(current_pose_odom.position_m.y) ||
      !std::isfinite(global_map.resolution_m) ||
      global_map.resolution_m <= 0.0 ||
      !std::isfinite(local_map.resolution_m) ||
      local_map.resolution_m <= 0.0) {
    return {.reason_code = "INVALID_INPUT"};
  }
  if (decision.targets_final_goal) {
    if (portals.candidates.size() != 1U) {
      return {.reason_code = "INVALID_INPUT"};
    }
    return {.goals = LocalGoalSet{
                .goals_odom = {portals.candidates.front().goal_odom},
                .exact_final_goal = true,
            }};
  }

  LocalGoalSet goals{.exact_final_goal = false};
  goals.goals_odom.reserve(portals.candidates.size());
  const auto in_bounds = [](const GridMap& map,
                            const shared::GridCell cell) {
    return cell.x >= 0 && cell.y >= 0 &&
           static_cast<std::size_t>(cell.x) < map.width &&
           static_cast<std::size_t>(cell.y) < map.height;
  };
  for (const SurfacePortalCandidate& portal : portals.candidates) {
    if (portal.route_progress_m <=
        decision.projected_route_progress_m + kNumericalEpsilon) {
      continue;
    }
    const auto* point = std::get_if<PointGoal>(&portal.goal_odom.target);
    if (point == nullptr || !in_bounds(global_map, portal.global_cell) ||
        !in_bounds(local_map, portal.local_cell)) {
      return {.reason_code = "INVALID_INPUT"};
    }
    const double minimum_x = local_map.origin_m.x +
        static_cast<double>(portal.local_cell.x) * local_map.resolution_m;
    const double minimum_y = local_map.origin_m.y +
        static_cast<double>(portal.local_cell.y) * local_map.resolution_m;
    const double maximum_x = minimum_x + local_map.resolution_m;
    const double maximum_y = minimum_y + local_map.resolution_m;
    const double local_boundary_distance = std::min(
        {point->position_m.x - minimum_x, maximum_x - point->position_m.x,
         point->position_m.y - minimum_y, maximum_y - point->position_m.y,
         0.5 * local_map.resolution_m});
    const double tolerance = snap_intermediate_to_local_cell
        ? std::max(0.0,
                   0.5 * local_map.resolution_m - kNumericalEpsilon)
        : std::max(0.0,
                   std::min(local_boundary_distance,
                            0.5 * global_map.resolution_m) -
                       kNumericalEpsilon);
    if (!std::isfinite(tolerance)) {
      return {.reason_code = "INVALID_INPUT"};
    }
    GoalRegion goal = portal.goal_odom;
    auto& snapped = std::get<PointGoal>(goal.target);
    if (snap_intermediate_to_local_cell) {
      snapped.position_m.x = minimum_x + 0.5 * local_map.resolution_m;
      snapped.position_m.y = minimum_y + 0.5 * local_map.resolution_m;
    }
    snapped.tolerance_m = tolerance;
    if (std::hypot(snapped.position_m.x - current_pose_odom.position_m.x,
                   snapped.position_m.y - current_pose_odom.position_m.y) <=
        tolerance + kNumericalEpsilon) {
      continue;
    }
    goal.yaw_rad.reset();
    goal.yaw_tolerance_rad = 0.0;
    goals.goals_odom.push_back(std::move(goal));
  }
  if (goals.goals_odom.empty()) {
    return {.reason_code = "NO_PATH"};
  }
  return {.goals = std::move(goals)};
}

LocalGoalSetResult SelectSurfaceLocalGoals(
    const PlanningRequest& input, const GlobalRoute& route,
    SearchControl control) {
  if (const auto stopped = shared::StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  if (route.poses_map.empty()) {
    return {.reason_code = "NO_PATH"};
  }
  if (GoalInsideLocalMap(input, input.goal_map)) {
    auto goal = GoalMapToOdomPlanar(input, input.goal_map);
    return goal.has_value()
               ? LocalGoalSetResult{.goals = LocalGoalSet{
                                        .goals_odom = {*std::move(goal)},
                                        .exact_final_goal = true,
                                    }}
               : LocalGoalSetResult{.reason_code = "INVALID_INPUT"};
  }
  const Pose3* start = CurrentPose(input);
  if (start == nullptr) {
    return {.reason_code = "INVALID_INPUT"};
  }
  const auto start_map = TransformPose(
      *start, input.world.map_from_odom,
      TransformDirection::kChildToParent);
  if (!start_map.has_value()) {
    return {.reason_code = "INVALID_INPUT"};
  }
  const double horizon_m =
      std::holds_alternative<LeggedCapability>(input.capability) ? 3.0 : 8.0;
  const SurfaceRollingSession session(
      route, input.goal_map,
      SurfaceRollingConfig{.horizon_m = horizon_m, .max_deviation_m = 2.0});
  SurfaceRollingDecision decision = session.Decide(*start_map);
  if (decision.kind != SurfaceRollingDecision::Kind::kNextPortalSet) {
    return {.reason_code = decision.kind ==
                                    SurfaceRollingDecision::Kind::kInvalidRoute
                                ? "INVALID_INPUT"
                                : "NO_PATH"};
  }
  // Reaching the route horizon is not sufficient to request the exact final
  // goal when that goal is outside the currently certified local map.
  decision.targets_final_goal = false;
  SurfacePortalSetResult portals =
      BuildSurfacePortalSet(input, route, decision, 32U, control);
  if (!portals.ok()) {
    return {.reason_code = std::move(portals.reason_code)};
  }
  return ConvertSurfacePortalsToLocalGoals(
      portals, decision, *start, *input.world.global_map,
      input.world.local_map,
      std::holds_alternative<LeggedCapability>(input.capability));
}

}  // namespace lunar::pure_planning::hierarchical
