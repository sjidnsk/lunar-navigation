#include "hierarchical/local_frontier.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <set>
#include <stop_token>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "hierarchical/frame_transform.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"

namespace lunar::planning::hierarchical {
namespace {

constexpr double kTolerance = 1.0e-9;

struct PlatformGeometry final {
  Vec3 current_position_odom;
  double horizon_m{};
  double support_radius_m{};
  double minimum_clearance_m{};
};

struct FrontierSample final {
  shared::GridCell cell;
  Vec3 position_odom;
  double route_distance_m{};
  double tangent_yaw_rad{};
  bool is_route_end{};
};

enum class LocalGoalFeasibility : std::uint8_t {
  kNotCovered,
  kFeasible,
  kInfeasible,
};

[[nodiscard]] LocalFrontierResult
Failure(const LocalFrontierStatus status, std::string reason_code,
        const double corridor_half_width_m = 0.0) {
  return LocalFrontierResult{
      .status = status,
      .corridor_half_width_m = corridor_half_width_m,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool IsFinite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool Close(const double lhs, const double rhs) noexcept {
  return std::isfinite(lhs) && std::isfinite(rhs) &&
         std::abs(lhs - rhs) <=
             kTolerance * std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

[[nodiscard]] bool PointOnSegment(const Vec2 point, const Vec2 start,
                                  const Vec2 end) noexcept {
  const double cross = (point.x - start.x) * (end.y - start.y) -
                       (point.y - start.y) * (end.x - start.x);
  if (std::abs(cross) > kTolerance) {
    return false;
  }
  return (point.x - start.x) * (point.x - end.x) +
             (point.y - start.y) * (point.y - end.y) <=
         kTolerance;
}

[[nodiscard]] bool PointInPolygon(const Vec2 point,
                                  const std::vector<Vec3> &boundary) noexcept {
  if (boundary.size() < 3U) {
    return false;
  }
  bool inside = false;
  for (std::size_t current = 0U, previous = boundary.size() - 1U;
       current < boundary.size(); previous = current++) {
    const Vec2 start{boundary[previous].x, boundary[previous].y};
    const Vec2 end{boundary[current].x, boundary[current].y};
    if (PointOnSegment(point, start, end)) {
      return true;
    }
    if ((start.y > point.y) == (end.y > point.y)) {
      continue;
    }
    const double crossing =
        start.x + (point.y - start.y) * (end.x - start.x) / (end.y - start.y);
    if (point.x < crossing) {
      inside = !inside;
    }
  }
  return inside;
}

[[nodiscard]] bool GoalIntersectsCell(const GoalRegion &goal,
                                      const Vec3 center,
                                      const double resolution_m) noexcept {
  if (const auto *point = std::get_if<PointGoal>(&goal.target)) {
    const double half = resolution_m / 2.0;
    const double dx =
        std::max(std::abs(center.x - point->position_m.x) - half, 0.0);
    const double dy =
        std::max(std::abs(center.y - point->position_m.y) - half, 0.0);
    return std::hypot(dx, dy) <= point->tolerance_m + kTolerance;
  }
  const auto *region = std::get_if<PlanarRegionGoal>(&goal.target);
  return region != nullptr &&
         PointInPolygon(Vec2{center.x, center.y}, region->boundary_m);
}

[[nodiscard]] LocalGoalFeasibility EvaluateLocalGoal(
    const shared::MapSnapshot &map,
    const shared::SafeProjection &projection,
    const GoalRegion &goal) noexcept {
  bool covered = false;
  for (std::size_t index = 0U; index < map.cell_count(); ++index) {
    const shared::GridCell cell{
        .x = static_cast<std::int32_t>(index % map.width()),
        .y = static_cast<std::int32_t>(index / map.width()),
    };
    if (!GoalIntersectsCell(goal, map.CellCenter(cell), map.resolution_m())) {
      continue;
    }
    covered = true;
    if (projection.HardFeasible(cell)) {
      return LocalGoalFeasibility::kFeasible;
    }
  }
  return covered ? LocalGoalFeasibility::kInfeasible
                 : LocalGoalFeasibility::kNotCovered;
}

[[nodiscard]] std::optional<PlatformGeometry>
ResolvePlatformGeometry(const PlannerInput &input) noexcept {
  if (const auto *capability =
          std::get_if<WheeledCapability>(&input.capability)) {
    const auto *state = std::get_if<WheeledState>(&input.current_state);
    if (state == nullptr || capability->footprint_xy_m.empty()) {
      return std::nullopt;
    }
    double radius = 0.0;
    for (const Vec2 vertex : capability->footprint_xy_m) {
      if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y)) {
        return std::nullopt;
      }
      radius = std::max(radius, std::hypot(vertex.x, vertex.y));
    }
    return PlatformGeometry{
        .current_position_odom = state->pose.position_m,
        .horizon_m = input.config.local_frontier.wheel_horizon_m,
        .support_radius_m = radius,
        .minimum_clearance_m = capability->minimum_clearance_m,
    };
  }
  if (const auto *capability =
          std::get_if<LeggedCapability>(&input.capability)) {
    const auto *state = std::get_if<LeggedState>(&input.current_state);
    if (state == nullptr) {
      return std::nullopt;
    }
    return PlatformGeometry{
        .current_position_odom = state->body_pose.position_m,
        .horizon_m = input.config.local_frontier.legged_horizon_m,
        .support_radius_m = std::hypot(capability->body_half_extent_m.x,
                                       capability->body_half_extent_m.y),
        .minimum_clearance_m = capability->minimum_body_clearance_m,
    };
  }
  return std::nullopt;
}

[[nodiscard]] bool
ValidPlatformGeometry(const PlatformGeometry &geometry,
                      const LocalFrontierConfig &config) noexcept {
  return IsFinite(geometry.current_position_odom) &&
         std::isfinite(geometry.horizon_m) && geometry.horizon_m > 0.0 &&
         std::isfinite(geometry.support_radius_m) &&
         geometry.support_radius_m >= 0.0 &&
         std::isfinite(geometry.minimum_clearance_m) &&
         geometry.minimum_clearance_m >= 0.0 &&
         std::isfinite(config.additional_corridor_margin_m) &&
         config.additional_corridor_margin_m >= 0.0 &&
         config.maximum_attempts > 0U;
}

[[nodiscard]] bool HasEdgeMargin(const shared::MapSnapshot &map,
                                 const Vec3 point,
                                 const double margin_m) noexcept {
  const double maximum_x =
      map.origin_m().x + static_cast<double>(map.width()) * map.resolution_m();
  const double maximum_y =
      map.origin_m().y + static_cast<double>(map.height()) * map.resolution_m();
  return point.x - map.origin_m().x + kTolerance >= margin_m &&
         point.y - map.origin_m().y + kTolerance >= margin_m &&
         maximum_x - point.x + kTolerance >= margin_m &&
         maximum_y - point.y + kTolerance >= margin_m;
}

[[nodiscard]] double SegmentLength(const Vec3 lhs, const Vec3 rhs) noexcept {
  return std::hypot(rhs.x - lhs.x, rhs.y - lhs.y);
}

[[nodiscard]] double DistanceToSegment(const Vec3 point, const Vec3 start,
                                       const Vec3 finish) noexcept {
  const double dx = finish.x - start.x;
  const double dy = finish.y - start.y;
  const double squared_length = dx * dx + dy * dy;
  if (squared_length <= kTolerance * kTolerance) {
    return std::hypot(point.x - start.x, point.y - start.y);
  }
  const double parameter = std::clamp(
      ((point.x - start.x) * dx + (point.y - start.y) * dy) / squared_length,
      0.0, 1.0);
  return std::hypot(point.x - (start.x + parameter * dx),
                    point.y - (start.y + parameter * dy));
}

[[nodiscard]] double
DistanceToPolyline(const Vec3 point,
                   const std::vector<Vec3> &polyline) noexcept {
  if (polyline.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  if (polyline.size() == 1U) {
    return std::hypot(point.x - polyline.front().x,
                      point.y - polyline.front().y);
  }
  double distance = std::numeric_limits<double>::infinity();
  for (std::size_t index = 1U; index < polyline.size(); ++index) {
    distance = std::min(distance, DistanceToSegment(point, polyline[index - 1U],
                                                    polyline[index]));
  }
  return distance;
}

[[nodiscard]] std::vector<Vec3>
PrefixPolyline(const std::vector<Vec3> &route_odom, const double distance_m,
               const Vec3 frontier) {
  std::vector<Vec3> prefix;
  if (route_odom.empty()) {
    return prefix;
  }
  prefix.push_back(route_odom.front());
  double consumed = 0.0;
  for (std::size_t index = 1U; index < route_odom.size(); ++index) {
    const Vec3 start = route_odom[index - 1U];
    const Vec3 finish = route_odom[index];
    const double length = SegmentLength(start, finish);
    if (length <= kTolerance) {
      continue;
    }
    if (consumed + length >= distance_m - kTolerance) {
      const double fraction =
          std::clamp((distance_m - consumed) / length, 0.0, 1.0);
      prefix.push_back(Vec3{
          .x = start.x + fraction * (finish.x - start.x),
          .y = start.y + fraction * (finish.y - start.y),
          .z = start.z + fraction * (finish.z - start.z),
      });
      break;
    }
    prefix.push_back(finish);
    consumed += length;
  }
  if (SegmentLength(prefix.back(), frontier) > kTolerance) {
    prefix.push_back(frontier);
  }
  return prefix;
}

[[nodiscard]] GridMap BuildLocalView(const GridMap &source, const Vec3 current,
                                     const std::vector<Vec3> &route_prefix,
                                     const double horizon_m,
                                     const double corridor_half_width_m) {
  GridMap view = source;
  auto &valid =
      std::get<std::vector<std::uint8_t>>(view.layers.at("valid_mask").values);
  auto &forbidden =
      std::get<std::vector<std::uint8_t>>(view.layers.at("forbidden").values);
  for (std::size_t y = 0U; y < view.height; ++y) {
    for (std::size_t x = 0U; x < view.width; ++x) {
      const std::size_t index = y * view.width + x;
      const Vec3 center{
          .x = view.origin_m.x +
               (static_cast<double>(x) + 0.5) * view.resolution_m,
          .y = view.origin_m.y +
               (static_cast<double>(y) + 0.5) * view.resolution_m,
      };
      const bool inside_horizon =
          std::hypot(center.x - current.x, center.y - current.y) <=
          horizon_m + kTolerance;
      const bool inside_corridor = DistanceToPolyline(center, route_prefix) <=
                                   corridor_half_width_m + kTolerance;
      if (!inside_horizon || !inside_corridor) {
        valid[index] = 0U;
        forbidden[index] = 1U;
      }
    }
  }
  return view;
}

[[nodiscard]] GoalRegion FrontierGoal(const PlannerInput &input,
                                      const FrontierSample &sample,
                                      const double resolution_m,
                                      const std::size_t attempt_index) {
  GoalRegion goal{
      .goal_id = input.goal_map.goal_id + "/local-frontier-" +
                 std::to_string(attempt_index),
      .target =
          PointGoal{
              .position_m = sample.position_odom,
              .tolerance_m = std::max(0.25 * resolution_m, 0.1),
          },
      .yaw_rad = sample.tangent_yaw_rad,
      .yaw_tolerance_rad = std::numbers::pi / 4.0,
  };
  if (sample.is_route_end && input.goal_map.yaw_rad.has_value()) {
    const auto transformed =
        TransformGoal(input.goal_map, input.world.map_from_odom,
                      TransformDirection::kParentToChild);
    if (transformed.has_value() && transformed->yaw_rad.has_value()) {
      goal.yaw_rad = transformed->yaw_rad;
      goal.yaw_tolerance_rad = transformed->yaw_tolerance_rad;
    }
  }
  return goal;
}

} // namespace

LocalFrontierResult BuildLocalFrontiers(const PlannerInput &input,
                                        const GlobalRoute &route) {
  if (input.stop_token.stop_requested()) {
    return Failure(LocalFrontierStatus::kCanceled, "REQUEST_CANCELED");
  }
  const auto geometry = ResolvePlatformGeometry(input);
  if (!geometry.has_value() ||
      !ValidPlatformGeometry(*geometry, input.config.local_frontier)) {
    return Failure(LocalFrontierStatus::kInvalidRequest,
                   "LOCAL_FRONTIER_CONFIGURATION_INVALID");
  }
  const double corridor_half_width =
      geometry->support_radius_m + geometry->minimum_clearance_m +
      input.config.local_frontier.additional_corridor_margin_m;
  if (route.poses_map.empty()) {
    return Failure(LocalFrontierStatus::kInvalidRequest, "GLOBAL_ROUTE_INVALID",
                   corridor_half_width);
  }
  const GridMap &local_map = input.world.local_map;
  if (!Close(local_map.resolution_m,
             input.config.global_map.base_resolution_m)) {
    return Failure(LocalFrontierStatus::kInvalidRequest,
                   "LOCAL_MAP_LEVEL_INVALID", corridor_half_width);
  }
  if (local_map.frame_id.empty() ||
      local_map.frame_id != input.world.map_from_odom.child_frame ||
      input.world.global_map.frame_id.empty() ||
      input.world.global_map.frame_id !=
          input.world.map_from_odom.parent_frame) {
    return Failure(LocalFrontierStatus::kInvalidRequest,
                   "FRAME_CONTRACT_INVALID", corridor_half_width);
  }

  const shared::MapSnapshotBuildResult snapshot =
      shared::MapSnapshot::Create(local_map);
  if (!snapshot.ok()) {
    return Failure(LocalFrontierStatus::kInvalidRequest, snapshot.reason_code,
                   corridor_half_width);
  }
  const shared::SafeProjectionBuildResult projection =
      shared::BuildSafeProjection(snapshot.snapshot, input.capability,
                                  input.config.map_safety, input.stop_token);
  if (!projection.ok()) {
    return Failure(projection.reason_code == "REQUEST_CANCELED"
                       ? LocalFrontierStatus::kCanceled
                       : LocalFrontierStatus::kInvalidRequest,
                   projection.reason_code, corridor_half_width);
  }

  const auto goal_odom =
      TransformGoal(input.goal_map, input.world.map_from_odom,
                    TransformDirection::kParentToChild);
  if (!goal_odom.has_value()) {
    return Failure(LocalFrontierStatus::kInvalidRequest,
                   "FRAME_TRANSFORM_INVALID", corridor_half_width);
  }
  if (EvaluateLocalGoal(*snapshot.snapshot, *projection.projection,
                        *goal_odom) ==
      LocalGoalFeasibility::kInfeasible) {
    return Failure(LocalFrontierStatus::kGoalInfeasible,
                   "GLOBAL_GOAL_INFEASIBLE", corridor_half_width);
  }

  const auto start_cell = snapshot.snapshot->PositionToCell(Vec2{
      .x = geometry->current_position_odom.x,
      .y = geometry->current_position_odom.y,
  });
  if (!start_cell.has_value() ||
      !projection.projection->HardFeasible(*start_cell) ||
      !HasEdgeMargin(*snapshot.snapshot,
                     snapshot.snapshot->CellCenter(*start_cell),
                     corridor_half_width)) {
    return Failure(LocalFrontierStatus::kCoverageInsufficient,
                   "LOCAL_MAP_COVERAGE_INSUFFICIENT", corridor_half_width);
  }
  const std::int32_t start_component =
      projection.projection->ConnectedComponent(*start_cell);

  std::vector<Vec3> route_odom;
  route_odom.reserve(route.poses_map.size() + 1U);
  route_odom.push_back(geometry->current_position_odom);
  for (const Pose3 &pose_map : route.poses_map) {
    if (input.stop_token.stop_requested()) {
      return Failure(LocalFrontierStatus::kCanceled, "REQUEST_CANCELED",
                     corridor_half_width);
    }
    const auto pose_odom = TransformPose(pose_map, input.world.map_from_odom,
                                         TransformDirection::kParentToChild);
    if (!pose_odom.has_value() || !IsFinite(pose_odom->position_m)) {
      return Failure(LocalFrontierStatus::kInvalidRequest,
                     "FRAME_TRANSFORM_INVALID", corridor_half_width);
    }
    if (SegmentLength(route_odom.back(), pose_odom->position_m) > kTolerance) {
      route_odom.push_back(pose_odom->position_m);
    }
  }
  if (route_odom.size() < 2U) {
    LocalFrontierResult stationary{
        .status = LocalFrontierStatus::kReady,
        .corridor_half_width_m = corridor_half_width,
        .reason_code = "LOCAL_FRONTIERS_AVAILABLE",
    };
    stationary.problems.push_back(LocalPlanningProblem{
        .request_id = input.request_id + "/local-0",
        .state_time = input.state_time,
        .current_state = input.current_state,
        .goal_odom = *goal_odom,
        .local_map_view = BuildLocalView(
            local_map, geometry->current_position_odom,
            {geometry->current_position_odom}, geometry->horizon_m,
            corridor_half_width),
        .capability = input.capability,
        .config = input.config,
        .previous_execution = input.previous_execution,
        .stop_token = input.stop_token,
    });
    stationary.frontier_distances_m.push_back(0.0);
    return stationary;
  }

  std::vector<FrontierSample> covered;
  std::set<shared::GridCell> seen_cells{*start_cell};
  double route_distance = 0.0;
  bool stop_scanning = false;
  const double maximum_sample_step = 0.5 * snapshot.snapshot->resolution_m();
  for (std::size_t segment_index = 1U;
       segment_index < route_odom.size() && !stop_scanning; ++segment_index) {
    const Vec3 start = route_odom[segment_index - 1U];
    const Vec3 finish = route_odom[segment_index];
    const double length = SegmentLength(start, finish);
    if (!std::isfinite(length) || length <= kTolerance) {
      continue;
    }
    const std::size_t steps =
        static_cast<std::size_t>(std::ceil(length / maximum_sample_step));
    const double step_length = length / static_cast<double>(steps);
    const double tangent_yaw =
        std::atan2(finish.y - start.y, finish.x - start.x);
    for (std::size_t step = 1U; step <= steps; ++step) {
      if (input.stop_token.stop_requested()) {
        return Failure(LocalFrontierStatus::kCanceled, "REQUEST_CANCELED",
                       corridor_half_width);
      }
      const double next_distance =
          route_distance + static_cast<double>(step) * step_length;
      if (next_distance > geometry->horizon_m + kTolerance) {
        stop_scanning = true;
        break;
      }
      const double fraction =
          static_cast<double>(step) / static_cast<double>(steps);
      const Vec3 sample{
          .x = start.x + fraction * (finish.x - start.x),
          .y = start.y + fraction * (finish.y - start.y),
          .z = start.z + fraction * (finish.z - start.z),
      };
      const auto cell =
          snapshot.snapshot->PositionToCell(Vec2{.x = sample.x, .y = sample.y});
      if (!cell.has_value()) {
        stop_scanning = true;
        break;
      }
      const Vec3 center = snapshot.snapshot->CellCenter(*cell);
      if (!HasEdgeMargin(*snapshot.snapshot, center, corridor_half_width) ||
          !projection.projection->HardFeasible(*cell) ||
          projection.projection->ConnectedComponent(*cell) != start_component) {
        stop_scanning = true;
        break;
      }
      const bool inside_horizon =
          std::hypot(center.x - geometry->current_position_odom.x,
                     center.y - geometry->current_position_odom.y) <=
          geometry->horizon_m + kTolerance;
      if (!inside_horizon || *cell == *start_cell) {
        continue;
      }
      const bool route_end =
          segment_index + 1U == route_odom.size() && step == steps;
      if (!covered.empty() && covered.back().cell == *cell) {
        covered.back().route_distance_m = next_distance;
        covered.back().tangent_yaw_rad = tangent_yaw;
        covered.back().is_route_end = route_end;
      } else if (seen_cells.insert(*cell).second) {
        covered.push_back(FrontierSample{
            .cell = *cell,
            .position_odom = center,
            .route_distance_m = next_distance,
            .tangent_yaw_rad = tangent_yaw,
            .is_route_end = route_end,
        });
      }
    }
    route_distance += length;
  }
  if (covered.empty()) {
    return Failure(LocalFrontierStatus::kCoverageInsufficient,
                   "LOCAL_MAP_COVERAGE_INSUFFICIENT", corridor_half_width);
  }

  const std::size_t attempt_count =
      std::min(input.config.local_frontier.maximum_attempts, covered.size());
  LocalFrontierResult result{
      .status = LocalFrontierStatus::kReady,
      .corridor_half_width_m = corridor_half_width,
      .reason_code = "LOCAL_FRONTIERS_AVAILABLE",
  };
  result.problems.reserve(attempt_count);
  result.frontier_distances_m.reserve(attempt_count);
  std::size_t previous_index = covered.size();
  for (std::size_t attempt = 0U; attempt < attempt_count; ++attempt) {
    const std::size_t remaining = attempt_count - attempt;
    std::size_t candidate_index =
        (covered.size() - 1U) * remaining / attempt_count;
    if (candidate_index >= previous_index) {
      candidate_index = previous_index - 1U;
    }
    previous_index = candidate_index;
    const FrontierSample &candidate = covered[candidate_index];
    const std::vector<Vec3> prefix = PrefixPolyline(
        route_odom, candidate.route_distance_m, candidate.position_odom);
    result.problems.push_back(LocalPlanningProblem{
        .request_id = input.request_id + "/local-" + std::to_string(attempt),
        .state_time = input.state_time,
        .current_state = input.current_state,
        .goal_odom = FrontierGoal(input, candidate,
                                  snapshot.snapshot->resolution_m(), attempt),
        .local_map_view =
            BuildLocalView(local_map, geometry->current_position_odom, prefix,
                           geometry->horizon_m, corridor_half_width),
        .capability = input.capability,
        .config = input.config,
        .previous_execution = input.previous_execution,
        .stop_token = input.stop_token,
    });
    result.frontier_distances_m.push_back(candidate.route_distance_m);
  }
  return result;
}

} // namespace lunar::planning::hierarchical
