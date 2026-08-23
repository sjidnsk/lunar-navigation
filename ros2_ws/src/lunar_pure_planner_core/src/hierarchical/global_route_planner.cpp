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
#include "hierarchical/surface_global_search.hpp"
#include "shared/controlled_work.hpp"
#include "shared/global_occupancy_projection.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::hierarchical {
namespace {

using namespace std::chrono_literals;

// Leave the outer stage enough time to unwind, finish its real timing sample,
// and map the controlled failure before the hard 150 ms stage boundary.
constexpr auto kGlobalBackendReturnReserve = 10ms;

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
  return PlanSurfaceGlobal(input, std::move(control),
                           WheelGlobalInflation(input));
}

GlobalStageResult PlanSurfaceGlobal(const PlanningRequest& input,
                                    SearchControl control,
                                    const double inflation_m) {
  if (control.deadline != SteadyClock::time_point::max()) {
    control.deadline -= kGlobalBackendReturnReserve;
  }
  if (!input.world.global_map.has_value()) {
    return Failure("INVALID_INPUT");
  }
  auto map = shared::MapSnapshot::Create(
      *input.world.global_map, shared::MapContract::kGlobalOccupancy, control);
  if (!map.ok()) {
    return Failure(map.reason_code == "TIMEOUT" ||
                           map.reason_code == "REQUEST_CANCELED"
                       ? map.reason_code
                       : "INVALID_INPUT");
  }
  auto projection = shared::BuildInflatedGlobalOccupancyProjection(
      map.snapshot, input.config.global_occupancy_threshold, inflation_m,
      control);
  if (!projection.ok()) {
    return Failure(projection.reason_code == "TIMEOUT" ||
                           projection.reason_code == "REQUEST_CANCELED"
                       ? projection.reason_code
                       : "INVALID_INPUT");
  }
  const auto* point = std::get_if<PointGoal>(&input.goal_map.target);
  const Pose3* current = CurrentPose(input);
  if (point == nullptr || current == nullptr) {
    return Failure("INVALID_INPUT");
  }
  const auto start_map = TransformPose(*current, input.world.map_from_odom,
                                       TransformDirection::kChildToParent);
  if (!start_map.has_value()) {
    return Failure("INVALID_INPUT");
  }
  Pose3 goal_pose{
      .position_m = {.x = point->position_m.x,
                     .y = point->position_m.y,
                     .z = map.snapshot->origin_m().z},
      .orientation = input.goal_map.yaw_rad.has_value()
                         ? QuaternionFromYaw(*input.goal_map.yaw_rad)
                         : start_map->orientation,
  };
  const auto start_cell = map.snapshot->PositionToCell(
      {.x = start_map->position_m.x, .y = start_map->position_m.y});
  const auto goal_cell = map.snapshot->PositionToCell(
      {.x = goal_pose.position_m.x, .y = goal_pose.position_m.y});
  if (!start_cell.has_value() || !goal_cell.has_value()) {
    return Failure("NO_PATH");
  }
  SurfaceGlobalSearchResult result = SearchSurfaceGlobal({
      .projection = projection.projection->View(),
      .start = *start_cell,
      .goal = *goal_cell,
      .start_pose_map = *start_map,
      .goal_pose_map = goal_pose,
      .control = std::move(control),
      .search = input.config.search,
  });
  if (!result.ok()) {
    return Failure(SurfaceFailureReason(result.status));
  }
  return GlobalStageResult{
      .route = GlobalRoute{
          .poses_map = std::move(result.preview.poses_map),
          .expanded_states = result.expanded_states,
      },
      .reason_code = {},
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

LocalGoalSelectionResult SelectSurfaceLocalGoal(
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
               ? LocalGoalSelectionResult{.goal = std::move(goal)}
               : LocalGoalSelectionResult{.reason_code = "INVALID_INPUT"};
  }
  const Pose3* start = CurrentPose(input);
  if (start == nullptr) {
    return {.reason_code = "INVALID_INPUT"};
  }
  std::optional<Vec3> best;
  bool entered{};
  bool exited{};
  const double sample_step = input.world.local_map.resolution_m * 0.5;
  std::size_t work_index{};
  for (std::size_t segment = 1U;
       segment < route.poses_map.size() && !exited; ++segment) {
    if (shared::ControlCheckDue(work_index++)) {
      if (const auto stopped = shared::StopReason(control);
          stopped.has_value()) {
        return {.reason_code = std::string{*stopped}};
      }
    }
    const Pose3& from = route.poses_map[segment - 1U];
    const Pose3& to = route.poses_map[segment];
    const double distance = std::hypot(to.position_m.x - from.position_m.x,
                                       to.position_m.y - from.position_m.y);
    const std::size_t steps = std::max<std::size_t>(
        1U, static_cast<std::size_t>(std::ceil(distance / sample_step)));
    for (std::size_t index = segment == 1U ? 0U : 1U; index <= steps;
         ++index) {
      if (shared::ControlCheckDue(work_index++)) {
        if (const auto stopped = shared::StopReason(control);
            stopped.has_value()) {
          return {.reason_code = std::string{*stopped}};
        }
      }
      const double ratio = static_cast<double>(index) /
                           static_cast<double>(steps);
      const Vec3 point_map{
          .x = from.position_m.x + ratio *
                                        (to.position_m.x - from.position_m.x),
          .y = from.position_m.y + ratio *
                                        (to.position_m.y - from.position_m.y),
          .z = 0.0,
      };
      const auto point_odom = TransformPoint(
          point_map, input.world.map_from_odom,
          TransformDirection::kParentToChild);
      const bool inside =
          point_odom.has_value() &&
          PositionInsideGridMap(input.world.local_map,
                                {.x = point_odom->x, .y = point_odom->y});
      if (inside) {
        entered = true;
        best = point_odom;
      } else if (entered) {
        exited = true;
        break;
      }
    }
  }
  if (!best.has_value() ||
      std::hypot(best->x - start->position_m.x,
                 best->y - start->position_m.y) <= 1.0e-9) {
    return {.reason_code = "NO_PATH"};
  }
  const auto* final_goal = std::get_if<PointGoal>(&input.goal_map.target);
  if (const auto stopped = shared::StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  return {
      .goal = GoalRegion{
          .goal_id = input.goal_map.goal_id + "/local",
          .target = PointGoal{
              .position_m = {.x = best->x, .y = best->y, .z = 0.0},
              .tolerance_m = final_goal == nullptr ? 0.0
                                                   : final_goal->tolerance_m,
          },
          .yaw_rad = std::nullopt,
          .yaw_tolerance_rad = input.goal_map.yaw_tolerance_rad,
      },
  };
}

}  // namespace lunar::pure_planning::hierarchical
