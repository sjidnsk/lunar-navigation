#include "hierarchical/hopper_route_planner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <new>
#include <numbers>
#include <optional>
#include <queue>
#include <ranges>
#include <stop_token>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "hierarchical/frame_transform.hpp"
#include "hierarchical/landing_spatial_index.hpp"
#include "hierarchical/landing_support_field.hpp"
#include "hierarchical/map_level.hpp"
#include "hopper/ballistic_envelope.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"

namespace lunar::planning::hierarchical {
namespace {

using Clock = std::chrono::steady_clock;
constexpr double kTolerance = 1.0e-9;

enum class EdgeBuildStatus : std::uint8_t {
  kValid,
  kInfeasible,
  kInvalid,
  kCanceled,
  kNumericalIndeterminate,
};

struct EdgeBuildResult final {
  EdgeBuildStatus status{EdgeBuildStatus::kInvalid};
  std::optional<NominalHopEdge> edge;
  std::string reason_code;
};

using EdgeKey = std::pair<LandingNodeId, LandingNodeId>;

[[nodiscard]] HopperRoutePlanResult
Failure(const PlanningOutcome outcome, std::string reason_code,
        const Clock::time_point started, const double reach_m = 0.0,
        const std::optional<std::size_t> level = std::nullopt,
        const std::size_t graph_nodes = 0U, const std::size_t graph_edges = 0U,
        const std::uint64_t expanded = 0U,
        const std::size_t evaluated_edge_pairs = 0U,
        const std::chrono::nanoseconds landing_field_elapsed = {}) {
  return HopperRoutePlanResult{
      .outcome = outcome,
      .route = std::nullopt,
      .nominal_hops = {},
      .global_level = level,
      .maximum_horizontal_reach_m = reach_m,
      .graph_nodes = graph_nodes,
      .graph_edges = graph_edges,
      .evaluated_edge_pairs = evaluated_edge_pairs,
      .route_hops = 0U,
      .expanded_nodes = expanded,
      .landing_field_elapsed = landing_field_elapsed,
      .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - started),
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool Finite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] double Norm(const Vec3 value) noexcept {
  return std::hypot(std::hypot(value.x, value.y), value.z);
}

[[nodiscard]] bool
ValidReachCapability(const HopperCapability &capability) noexcept {
  const double minimum_time =
      std::chrono::duration<double>(capability.minimum_flight_time).count();
  const double maximum_time =
      std::chrono::duration<double>(capability.maximum_flight_time).count();
  return Finite(capability.gravity_mps2) &&
         Norm(capability.gravity_mps2) > kTolerance &&
         std::isfinite(capability.platform_mass_kg) &&
         capability.platform_mass_kg > 0.0 &&
         std::isfinite(capability.maximum_launch_speed_mps) &&
         capability.maximum_launch_speed_mps > 0.0 &&
         std::isfinite(capability.maximum_launch_impulse_newton_seconds) &&
         capability.maximum_launch_impulse_newton_seconds > 0.0 &&
         std::isfinite(capability.maximum_landing_speed_mps) &&
         capability.maximum_landing_speed_mps > 0.0 &&
         std::isfinite(capability.minimum_downward_impact_speed_mps) &&
         capability.minimum_downward_impact_speed_mps >= 0.0 &&
         std::isfinite(minimum_time) && minimum_time > 0.0 &&
         std::isfinite(maximum_time) && maximum_time >= minimum_time;
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

[[nodiscard]] bool GoalIntersectsCell(const GoalRegion &goal, const Vec3 center,
                                      const double resolution_m) noexcept {
  if (const auto *point = std::get_if<PointGoal>(&goal.target)) {
    if (!Finite(point->position_m) || !std::isfinite(point->tolerance_m) ||
        point->tolerance_m < 0.0) {
      return false;
    }
    const double half = resolution_m / 2.0;
    const double dx =
        std::max(std::abs(center.x - point->position_m.x) - half, 0.0);
    const double dy =
        std::max(std::abs(center.y - point->position_m.y) - half, 0.0);
    return std::hypot(dx, dy) <= point->tolerance_m + kTolerance;
  }
  const auto *region = std::get_if<PlanarRegionGoal>(&goal.target);
  return region != nullptr && region->boundary_m.size() >= 3U &&
         PointInPolygon(Vec2{center.x, center.y}, region->boundary_m);
}

[[nodiscard]] double DistanceToSegment(const Vec2 point, const Vec2 start,
                                       const Vec2 end) noexcept {
  const double dx = end.x - start.x;
  const double dy = end.y - start.y;
  const double squared_length = dx * dx + dy * dy;
  if (squared_length <= kTolerance * kTolerance) {
    return std::hypot(point.x - start.x, point.y - start.y);
  }
  const double ratio = std::clamp(
      ((point.x - start.x) * dx + (point.y - start.y) * dy) / squared_length,
      0.0, 1.0);
  return std::hypot(point.x - (start.x + ratio * dx),
                    point.y - (start.y + ratio * dy));
}

[[nodiscard]] double
GoalDistanceLowerBound(const Vec2 point, const GoalRegion &goal,
                       const double map_resolution_m) noexcept {
  const double cell_half_diagonal =
      map_resolution_m * std::numbers::sqrt2 / 2.0;
  if (const auto *target = std::get_if<PointGoal>(&goal.target)) {
    const double center_distance = std::hypot(point.x - target->position_m.x,
                                              point.y - target->position_m.y);
    return std::max(0.0,
                    center_distance - target->tolerance_m - cell_half_diagonal);
  }
  const auto *region = std::get_if<PlanarRegionGoal>(&goal.target);
  if (region == nullptr || region->boundary_m.size() < 3U) {
    return 0.0;
  }
  if (PointInPolygon(point, region->boundary_m)) {
    return 0.0;
  }
  double distance = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0U; index < region->boundary_m.size(); ++index) {
    const Vec3 start = region->boundary_m[index];
    const Vec3 end =
        region->boundary_m[(index + 1U) % region->boundary_m.size()];
    distance = std::min(
        distance, DistanceToSegment(point, {start.x, start.y}, {end.x, end.y}));
  }
  return std::isfinite(distance) ? std::max(0.0, distance - cell_half_diagonal)
                                 : 0.0;
}

[[nodiscard]] shared::GridCell
CellForNode(const LandingNodeId id, const LandingNodeId start_id,
            const shared::GridCell start_cell,
            const shared::MapSnapshot &map) noexcept {
  if (id == start_id) {
    return start_cell;
  }
  return shared::GridCell{
      .x = static_cast<std::int32_t>(id % map.width()),
      .y = static_cast<std::int32_t>(id / map.width()),
  };
}

[[nodiscard]] Vec3
BodyPositionForNode(const LandingNodeId id, const LandingNodeId start_id,
                    const Pose3 &start_pose_map, const shared::MapSnapshot &map,
                    const HopperCapability &capability) noexcept {
  if (id == start_id) {
    return start_pose_map.position_m;
  }
  const shared::GridCell cell = CellForNode(id, start_id, {}, map);
  const Vec3 surface = map.CellCenter(cell);
  return Vec3{
      .x = surface.x,
      .y = surface.y,
      .z = surface.z + capability.body_half_extent_m.z,
  };
}

[[nodiscard]] Quaternion YawQuaternion(const double yaw_rad) noexcept {
  return Quaternion{
      .w = std::cos(yaw_rad / 2.0),
      .z = std::sin(yaw_rad / 2.0),
  };
}

[[nodiscard]] EdgeBuildResult
BuildNominalEdge(const LandingNodeId source_id, const LandingNodeId target_id,
                 const LandingNodeId start_id, const Pose3 &start_pose_map,
                 const Vec3 start_velocity_map, const shared::MapSnapshot &map,
                 const HopperCapability &capability,
                 const std::stop_token stop_token) {
  if (stop_token.stop_requested()) {
    return EdgeBuildResult{
        .status = EdgeBuildStatus::kCanceled,
        .edge = std::nullopt,
        .reason_code = "REQUEST_CANCELED",
    };
  }
  const Vec3 launch =
      BodyPositionForNode(source_id, start_id, start_pose_map, map, capability);
  const Vec3 landing =
      BodyPositionForNode(target_id, start_id, start_pose_map, map, capability);
  const double horizontal =
      std::hypot(landing.x - launch.x, landing.y - launch.y);
  if (!std::isfinite(horizontal) || horizontal <= kTolerance) {
    return EdgeBuildResult{
        .status = EdgeBuildStatus::kInfeasible,
        .edge = std::nullopt,
        .reason_code = "HOPPER_BALLISTIC_INFEASIBLE",
    };
  }
  const Vec3 initial_velocity =
      source_id == start_id ? start_velocity_map : Vec3{};
  const hopper::BallisticEnvelopeResult solved = hopper::SolveBallisticEnvelope(
      launch, landing, initial_velocity, capability, 0.0, stop_token);
  if (!solved.ok()) {
    EdgeBuildStatus status = EdgeBuildStatus::kNumericalIndeterminate;
    switch (solved.status) {
    case hopper::BallisticEnvelopeStatus::kInfeasible:
      status = EdgeBuildStatus::kInfeasible;
      break;
    case hopper::BallisticEnvelopeStatus::kInvalid:
      status = EdgeBuildStatus::kInvalid;
      break;
    case hopper::BallisticEnvelopeStatus::kCanceled:
      status = EdgeBuildStatus::kCanceled;
      break;
    case hopper::BallisticEnvelopeStatus::kNumericalIndeterminate:
    case hopper::BallisticEnvelopeStatus::kSolved:
      status = EdgeBuildStatus::kNumericalIndeterminate;
      break;
    }
    return EdgeBuildResult{
        .status = status,
        .edge = std::nullopt,
        .reason_code = solved.reason_code,
    };
  }
  if (!solved.arc.has_value()) {
    return EdgeBuildResult{
        .status = EdgeBuildStatus::kNumericalIndeterminate,
        .edge = std::nullopt,
        .reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE",
    };
  }
  const Vec3 velocity_change{
      .x = solved.arc->launch_velocity_mps.x - initial_velocity.x,
      .y = solved.arc->launch_velocity_mps.y - initial_velocity.y,
      .z = solved.arc->launch_velocity_mps.z - initial_velocity.z,
  };
  const double impulse = capability.platform_mass_kg * Norm(velocity_change);
  const double maximum_time =
      std::chrono::duration<double>(capability.maximum_flight_time).count();
  if (!std::isfinite(impulse) || !std::isfinite(maximum_time) ||
      maximum_time <= 0.0) {
    return EdgeBuildResult{
        .status = EdgeBuildStatus::kNumericalIndeterminate,
        .edge = std::nullopt,
        .reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE",
    };
  }
  const double cost =
      1.0 + solved.arc->flight_time_s / maximum_time +
      impulse / capability.maximum_launch_impulse_newton_seconds;
  if (!std::isfinite(cost) || cost <= 0.0) {
    return EdgeBuildResult{
        .status = EdgeBuildStatus::kNumericalIndeterminate,
        .edge = std::nullopt,
        .reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE",
    };
  }
  return EdgeBuildResult{
      .status = EdgeBuildStatus::kValid,
      .edge =
          NominalHopEdge{
              .source_id = source_id,
              .target_id = target_id,
              .cost = cost,
              .arc = *solved.arc,
          },
      .reason_code = {},
  };
}

[[nodiscard]] std::optional<PlanningOutcome>
FatalOutcome(const EdgeBuildStatus status) noexcept {
  switch (status) {
  case EdgeBuildStatus::kInvalid:
    return PlanningOutcome::kInvalidRequest;
  case EdgeBuildStatus::kCanceled:
    return PlanningOutcome::kCanceled;
  case EdgeBuildStatus::kNumericalIndeterminate:
    return PlanningOutcome::kNumericalFailure;
  case EdgeBuildStatus::kValid:
  case EdgeBuildStatus::kInfeasible:
    return std::nullopt;
  }
  return PlanningOutcome::kNumericalFailure;
}

} // namespace

double ConservativeMaximumHorizontalReach(
    const HopperCapability &capability) noexcept {
  if (!ValidReachCapability(capability)) {
    return 0.0;
  }
  const double minimum_time =
      std::chrono::duration<double>(capability.minimum_flight_time).count();
  const double maximum_time =
      std::chrono::duration<double>(capability.maximum_flight_time).count();
  const double gravity = Norm(capability.gravity_mps2);
  const double horizontal_gravity =
      std::hypot(capability.gravity_mps2.x, capability.gravity_mps2.y);
  if (horizontal_gravity > kTolerance) {
    const double upper = capability.maximum_launch_speed_mps * maximum_time +
                         0.5 * horizontal_gravity * maximum_time * maximum_time;
    return std::isfinite(upper) ? upper : 0.0;
  }

  const double launch_limit = capability.maximum_launch_speed_mps;
  const double landing_limit = capability.maximum_landing_speed_mps;
  const double smaller_limit = std::min(launch_limit, landing_limit);
  const double larger_limit = std::max(launch_limit, landing_limit);
  const double time_limit =
      std::min(maximum_time, (launch_limit + landing_limit) / gravity);
  if (!std::isfinite(time_limit) || time_limit < minimum_time) {
    return 0.0;
  }
  const double equal_bound_time =
      2.0 *
      std::sqrt(std::max(0.0, larger_limit * larger_limit -
                                  smaller_limit * smaller_limit)) /
      gravity;
  const double stationary_time = std::numbers::sqrt2 * larger_limit / gravity;
  const std::array<double, 4U> candidates{
      minimum_time,
      time_limit,
      std::clamp(equal_bound_time, minimum_time, time_limit),
      std::clamp(stationary_time, minimum_time, time_limit),
  };
  double maximum_reach = 0.0;
  for (const double time_s : candidates) {
    const double forced_vertical = 0.5 * gravity * time_s;
    const double speed_bound = std::sqrt(std::max(
        0.0, larger_limit * larger_limit - forced_vertical * forced_vertical));
    const double horizontal_speed = std::min(smaller_limit, speed_bound);
    maximum_reach = std::max(maximum_reach, time_s * horizontal_speed);
  }
  return std::isfinite(maximum_reach) ? maximum_reach : 0.0;
}

HopperRoutePlanResult PlanHopperGlobalRoute(const PlannerInput &input) {
  const Clock::time_point started = Clock::now();
  try {
    if (input.stop_token.stop_requested()) {
      return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED", started);
    }
    const auto *state = std::get_if<HopperState>(&input.current_state);
    const auto *capability = std::get_if<HopperCapability>(&input.capability);
    if (state == nullptr || capability == nullptr ||
        !ValidReachCapability(*capability)) {
      return Failure(PlanningOutcome::kInvalidRequest,
                     "HOPPER_GLOBAL_CONFIGURATION_INVALID", started);
    }
    const double reach = ConservativeMaximumHorizontalReach(*capability);
    if (!(reach > 0.0)) {
      return Failure(PlanningOutcome::kInvalidRequest,
                     "HOPPER_GLOBAL_REACH_INVALID", started, reach);
    }
    const MapLevelValidationResult levels =
        ValidateMapLevels(input.world, input.config.global_map);
    if (!levels.ok()) {
      return Failure(PlanningOutcome::kInvalidRequest, levels.reason_code,
                     started, reach);
    }
    if (input.world.global_map.resolution_m > reach / 2.0 + kTolerance) {
      return Failure(PlanningOutcome::kInvalidRequest,
                     "HOPPER_GLOBAL_RESOLUTION_INSUFFICIENT", started, reach,
                     levels.global_level);
    }
    if (input.world.global_map.frame_id !=
            input.world.map_from_odom.parent_frame ||
        input.world.local_map.frame_id !=
            input.world.map_from_odom.child_frame) {
      return Failure(PlanningOutcome::kInvalidRequest, "FRAME_CONTRACT_INVALID",
                     started, reach, levels.global_level);
    }

    const shared::MapSnapshotBuildResult snapshot =
        shared::MapSnapshot::Create(input.world.global_map);
    if (!snapshot.ok()) {
      return Failure(PlanningOutcome::kInvalidRequest, snapshot.reason_code,
                     started, reach, levels.global_level);
    }
    const shared::SafeProjectionBuildResult projection =
        shared::BuildSafeProjection(snapshot.snapshot, input.capability,
                                    input.config.map_safety, input.stop_token);
    if (!projection.ok()) {
      return Failure(projection.reason_code == "REQUEST_CANCELED"
                         ? PlanningOutcome::kCanceled
                         : PlanningOutcome::kInvalidRequest,
                     projection.reason_code, started, reach,
                     levels.global_level);
    }
    LandingSupportFieldBuildResult landing_field = BuildLandingSupportField(
        *projection.projection, *capability, input.stop_token);
    if (!landing_field.ok()) {
      return Failure(landing_field.reason_code == "REQUEST_CANCELED"
                         ? PlanningOutcome::kCanceled
                         : PlanningOutcome::kInvalidRequest,
                     landing_field.reason_code, started, reach,
                     levels.global_level);
    }
    LandingSpatialIndex spatial_index(*landing_field.field, reach,
                                      input.stop_token);
    if (!spatial_index.ok()) {
      return Failure(spatial_index.reason_code() == "REQUEST_CANCELED"
                         ? PlanningOutcome::kCanceled
                         : PlanningOutcome::kInvalidRequest,
                     std::string{spatial_index.reason_code()}, started, reach,
                     levels.global_level, 0U, 0U, 0U, 0U,
                     landing_field.elapsed);
    }

    const auto start_pose_map =
        TransformPose(state->pose, input.world.map_from_odom,
                      TransformDirection::kChildToParent);
    RigidTransform velocity_rotation = input.world.map_from_odom;
    velocity_rotation.translation_m = {};
    const auto start_velocity_map =
        TransformPoint(state->velocity.linear_mps, velocity_rotation,
                       TransformDirection::kChildToParent);
    if (!start_pose_map.has_value() || !start_velocity_map.has_value()) {
      return Failure(
          PlanningOutcome::kInvalidRequest, "FRAME_TRANSFORM_INVALID", started,
          reach, levels.global_level, 0U, 0U, 0U, 0U, landing_field.elapsed);
    }
    const auto start_cell = snapshot.snapshot->PositionToCell(Vec2{
        .x = start_pose_map->position_m.x,
        .y = start_pose_map->position_m.y,
    });
    if (!start_cell.has_value() ||
        !landing_field.field->BaseSafe(*start_cell)) {
      return Failure(
          PlanningOutcome::kNoKnownSafeRoute, "HOPPER_START_NOT_SAFE", started,
          reach, levels.global_level, 0U, 0U, 0U, 0U, landing_field.elapsed);
    }
    if (!landing_field.field->CenterSafe(*start_cell)) {
      return Failure(PlanningOutcome::kNoKnownSafeRoute,
                     "HOPPER_START_REGION_AREA_INSUFFICIENT", started, reach,
                     levels.global_level, 0U, 0U, 0U, 0U,
                     landing_field.elapsed);
    }

    std::vector<std::uint8_t> goal_mask(snapshot.snapshot->cell_count(), 0U);
    std::size_t goal_count = 0U;
    for (const LandingNodeId id : landing_field.field->SafeCenterIds()) {
      if (input.stop_token.stop_requested()) {
        return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED", started,
                       reach, levels.global_level, 0U, 0U, 0U, 0U,
                       landing_field.elapsed);
      }
      const shared::GridCell cell{
          .x = static_cast<std::int32_t>(id % snapshot.snapshot->width()),
          .y = static_cast<std::int32_t>(id / snapshot.snapshot->width()),
      };
      if (GoalIntersectsCell(input.goal_map,
                             snapshot.snapshot->CellCenter(cell),
                             snapshot.snapshot->resolution_m())) {
        goal_mask[id] = 1U;
        ++goal_count;
      }
    }
    if (goal_count == 0U) {
      return Failure(PlanningOutcome::kGoalInfeasible,
                     "HOPPER_GLOBAL_GOAL_INFEASIBLE", started, reach,
                     levels.global_level, 0U, 0U, 0U, 0U,
                     landing_field.elapsed);
    }

    const LandingNodeId start_id = snapshot.snapshot->cell_count();
    const std::size_t graph_node_count =
        landing_field.field->SafeCenterIds().size() + 1U;
    std::map<EdgeKey, EdgeBuildResult> edge_cache;
    std::size_t graph_edge_count = 0U;
    std::size_t evaluated_edge_pairs = 0U;
    const auto get_edge =
        [&](const LandingNodeId source_id,
            const LandingNodeId target_id) -> const EdgeBuildResult & {
      const EdgeKey key{source_id, target_id};
      if (const auto found = edge_cache.find(key); found != edge_cache.end()) {
        return found->second;
      }
      ++evaluated_edge_pairs;
      EdgeBuildResult built = BuildNominalEdge(
          source_id, target_id, start_id, *start_pose_map, *start_velocity_map,
          *snapshot.snapshot, *capability, input.stop_token);
      if (built.status == EdgeBuildStatus::kValid) {
        ++graph_edge_count;
      }
      return edge_cache.emplace(key, std::move(built)).first->second;
    };

    const auto heuristic_cost = [&](const LandingNodeId id) {
      const Vec3 position = BodyPositionForNode(
          id, start_id, *start_pose_map, *snapshot.snapshot, *capability);
      const double distance =
          GoalDistanceLowerBound({position.x, position.y}, input.goal_map,
                                 snapshot.snapshot->resolution_m());
      if (!std::isfinite(distance) || distance <= kTolerance) {
        return 0.0;
      }
      const double minimum_hops =
          std::ceil(std::max(0.0, distance / reach - 16.0 * kTolerance));
      double lower_bound = minimum_hops;
      const double maximum_time =
          std::chrono::duration<double>(capability->maximum_flight_time)
              .count();
      if (std::hypot(capability->gravity_mps2.x, capability->gravity_mps2.y) <=
          kTolerance) {
        if (id == start_id) {
          lower_bound +=
              distance / (capability->maximum_launch_speed_mps * maximum_time);
        } else {
          const double minimum_time =
              std::chrono::duration<double>(capability->minimum_flight_time)
                  .count();
          const double time_coefficient = 1.0 / (maximum_time * reach);
          const double impulse_coefficient =
              capability->platform_mass_kg /
              capability->maximum_launch_impulse_newton_seconds;
          const double minimizing_time =
              std::clamp(std::sqrt(impulse_coefficient / time_coefficient),
                         minimum_time, maximum_time);
          const double variable_cost_per_m =
              time_coefficient * minimizing_time +
              impulse_coefficient / minimizing_time;
          lower_bound += variable_cost_per_m * distance;
        }
      }
      return lower_bound;
    };

    const auto optimistic_edge_cost = [&](const LandingNodeId source_id,
                                          const LandingNodeId target_id) {
      const Vec3 source =
          BodyPositionForNode(source_id, start_id, *start_pose_map,
                              *snapshot.snapshot, *capability);
      const Vec3 target =
          BodyPositionForNode(target_id, start_id, *start_pose_map,
                              *snapshot.snapshot, *capability);
      const double distance =
          std::hypot(target.x - source.x, target.y - source.y);
      if (!std::isfinite(distance) || distance <= kTolerance) {
        return std::numeric_limits<double>::infinity();
      }
      if (std::hypot(capability->gravity_mps2.x, capability->gravity_mps2.y) >
          kTolerance) {
        return 1.0;
      }
      const double minimum_time =
          std::chrono::duration<double>(capability->minimum_flight_time)
              .count();
      const double maximum_time =
          std::chrono::duration<double>(capability->maximum_flight_time)
              .count();
      const double time_lower_bound = std::max(
          minimum_time, distance / capability->maximum_launch_speed_mps);
      if (time_lower_bound > maximum_time + kTolerance) {
        return std::numeric_limits<double>::infinity();
      }
      if (source_id == start_id) {
        return 1.0 + time_lower_bound / maximum_time;
      }
      const double impulse_coefficient =
          capability->platform_mass_kg * distance /
          capability->maximum_launch_impulse_newton_seconds;
      const double minimizing_time =
          std::clamp(std::sqrt(impulse_coefficient * maximum_time),
                     time_lower_bound, maximum_time);
      return 1.0 + minimizing_time / maximum_time +
             impulse_coefficient / minimizing_time;
    };

    const auto success =
        [&](std::vector<LandingNodeId> node_path,
            std::vector<NominalHopEdge> nominal_hops,
            const std::uint64_t expanded, const std::size_t open_peak,
            const std::size_t working_memory_bytes) -> HopperRoutePlanResult {
      GlobalRoute route;
      route.raw_cells.reserve(node_path.size());
      route.simplified_cells.reserve(node_path.size());
      route.poses_map.reserve(node_path.size());
      for (std::size_t index = 0U; index < node_path.size(); ++index) {
        const LandingNodeId id = node_path[index];
        const shared::GridCell cell =
            CellForNode(id, start_id, *start_cell, *snapshot.snapshot);
        const Vec3 body_position = BodyPositionForNode(
            id, start_id, *start_pose_map, *snapshot.snapshot, *capability);
        route.raw_cells.push_back(cell);
        route.simplified_cells.push_back(cell);
        double yaw = 0.0;
        if (index + 1U < node_path.size()) {
          const Vec3 next = BodyPositionForNode(
              node_path[index + 1U], start_id, *start_pose_map,
              *snapshot.snapshot, *capability);
          yaw = std::atan2(next.y - body_position.y, next.x - body_position.x);
        } else if (input.goal_map.yaw_rad.has_value()) {
          yaw = *input.goal_map.yaw_rad;
        } else if (index > 0U) {
          const Vec3 previous = BodyPositionForNode(
              node_path[index - 1U], start_id, *start_pose_map,
              *snapshot.snapshot, *capability);
          yaw = std::atan2(body_position.y - previous.y,
                           body_position.x - previous.x);
        }
        route.poses_map.push_back(Pose3{
            .position_m = body_position,
            .orientation = YawQuaternion(yaw),
        });
      }
      double total_cost = 0.0;
      for (const NominalHopEdge &edge : nominal_hops) {
        total_cost += edge.cost;
      }
      route.cost = total_cost;
      route.expanded_states = expanded;
      route.open_peak = open_peak;
      route.estimated_work_memory_bytes =
          landing_field.field->EstimatedWorkMemoryBytes() +
          spatial_index.EstimatedWorkMemoryBytes() + working_memory_bytes +
          edge_cache.size() * (sizeof(EdgeKey) + sizeof(EdgeBuildResult));
      return HopperRoutePlanResult{
          .outcome = PlanningOutcome::kNewReferenceAvailable,
          .route = std::move(route),
          .nominal_hops = std::move(nominal_hops),
          .global_level = levels.global_level,
          .maximum_horizontal_reach_m = reach,
          .graph_nodes = graph_node_count,
          .graph_edges = graph_edge_count,
          .evaluated_edge_pairs = evaluated_edge_pairs,
          .route_hops = node_path.size() - 1U,
          .expanded_nodes = expanded,
          .landing_field_elapsed = landing_field.elapsed,
          .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
              Clock::now() - started),
          .reason_code = "HOPPER_GLOBAL_ROUTE_AVAILABLE",
      };
    };

    const std::vector<LandingNodeId> start_neighbors = spatial_index.Query(
        {start_pose_map->position_m.x, start_pose_map->position_m.y},
        input.stop_token);
    if (input.stop_token.stop_requested()) {
      return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED", started,
                     reach, levels.global_level, graph_node_count,
                     graph_edge_count, 0U, evaluated_edge_pairs,
                     landing_field.elapsed);
    }
    std::optional<NominalHopEdge> selected_direct;
    for (const LandingNodeId target_id : start_neighbors) {
      if (goal_mask[target_id] == 0U) {
        continue;
      }
      const EdgeBuildResult &edge = get_edge(start_id, target_id);
      if (const auto fatal = FatalOutcome(edge.status); fatal.has_value()) {
        return Failure(*fatal, edge.reason_code, started, reach,
                       levels.global_level, graph_node_count, graph_edge_count,
                       0U, evaluated_edge_pairs, landing_field.elapsed);
      }
      if (!edge.edge.has_value()) {
        continue;
      }
      if (!selected_direct.has_value() ||
          std::tie(edge.edge->cost, edge.edge->target_id) <
              std::tie(selected_direct->cost, selected_direct->target_id)) {
        selected_direct = edge.edge;
      }
    }
    if (selected_direct.has_value()) {
      return success({start_id, selected_direct->target_id}, {*selected_direct},
                     1U, 1U, 0U);
    }

    const std::size_t state_count = snapshot.snapshot->cell_count() + 1U;
    std::vector<std::uint8_t> greedy_visited(state_count, 0U);
    std::vector<LandingNodeId> greedy_path{start_id};
    std::vector<NominalHopEdge> greedy_hops;
    LandingNodeId greedy_source = start_id;
    greedy_visited[start_id] = 1U;
    std::uint64_t greedy_expanded = 0U;
    while (true) {
      if (input.stop_token.stop_requested()) {
        return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED", started,
                       reach, levels.global_level, graph_node_count,
                       graph_edge_count, greedy_expanded, evaluated_edge_pairs,
                       landing_field.elapsed);
      }
      ++greedy_expanded;
      const Vec3 source_position =
          BodyPositionForNode(greedy_source, start_id, *start_pose_map,
                              *snapshot.snapshot, *capability);
      const double source_goal_distance = GoalDistanceLowerBound(
          {source_position.x, source_position.y}, input.goal_map,
          snapshot.snapshot->resolution_m());
      const std::vector<LandingNodeId> neighbors = spatial_index.Query(
          {source_position.x, source_position.y}, input.stop_token);
      if (input.stop_token.stop_requested()) {
        return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED", started,
                       reach, levels.global_level, graph_node_count,
                       graph_edge_count, greedy_expanded, evaluated_edge_pairs,
                       landing_field.elapsed);
      }

      std::optional<NominalHopEdge> direct_goal;
      for (const LandingNodeId target_id : neighbors) {
        if (goal_mask[target_id] == 0U) {
          continue;
        }
        const EdgeBuildResult &edge = get_edge(greedy_source, target_id);
        if (const auto fatal = FatalOutcome(edge.status); fatal.has_value()) {
          return Failure(*fatal, edge.reason_code, started, reach,
                         levels.global_level, graph_node_count,
                         graph_edge_count, greedy_expanded,
                         evaluated_edge_pairs, landing_field.elapsed);
        }
        if (!edge.edge.has_value()) {
          continue;
        }
        if (!direct_goal.has_value() ||
            std::tie(edge.edge->cost, edge.edge->target_id) <
                std::tie(direct_goal->cost, direct_goal->target_id)) {
          direct_goal = edge.edge;
        }
      }
      if (direct_goal.has_value()) {
        greedy_path.push_back(direct_goal->target_id);
        greedy_hops.push_back(*direct_goal);
        return success(std::move(greedy_path), std::move(greedy_hops),
                       greedy_expanded, 1U,
                       greedy_visited.capacity() * sizeof(std::uint8_t));
      }

      std::vector<std::tuple<double, double, LandingNodeId>> ranked;
      ranked.reserve(neighbors.size());
      for (const LandingNodeId target_id : neighbors) {
        if (greedy_visited[target_id] != 0U || goal_mask[target_id] != 0U) {
          continue;
        }
        const Vec3 target_position =
            BodyPositionForNode(target_id, start_id, *start_pose_map,
                                *snapshot.snapshot, *capability);
        const double target_goal_distance = GoalDistanceLowerBound(
            {target_position.x, target_position.y}, input.goal_map,
            snapshot.snapshot->resolution_m());
        if (!std::isfinite(target_goal_distance) ||
            target_goal_distance + kTolerance >= source_goal_distance) {
          continue;
        }
        const double source_distance =
            std::hypot(target_position.x - source_position.x,
                       target_position.y - source_position.y);
        ranked.emplace_back(target_goal_distance, -source_distance, target_id);
      }
      std::ranges::sort(ranked);

      bool advanced = false;
      for (const auto &[goal_distance, negative_distance, target_id] : ranked) {
        static_cast<void>(goal_distance);
        static_cast<void>(negative_distance);
        const EdgeBuildResult &edge = get_edge(greedy_source, target_id);
        if (const auto fatal = FatalOutcome(edge.status); fatal.has_value()) {
          return Failure(*fatal, edge.reason_code, started, reach,
                         levels.global_level, graph_node_count,
                         graph_edge_count, greedy_expanded,
                         evaluated_edge_pairs, landing_field.elapsed);
        }
        if (!edge.edge.has_value()) {
          continue;
        }
        greedy_source = target_id;
        greedy_visited[target_id] = 1U;
        greedy_path.push_back(target_id);
        greedy_hops.push_back(*edge.edge);
        advanced = true;
        break;
      }
      if (!advanced) {
        break;
      }
    }

    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<double> distances(state_count, infinity);
    std::vector<double> expanded_costs(state_count, infinity);
    std::vector<std::size_t> hop_counts(
        state_count, std::numeric_limits<std::size_t>::max());
    std::vector<std::optional<LandingNodeId>> parents(state_count);
    using QueueEntry =
        std::tuple<double, double, double, std::size_t, LandingNodeId>;
    std::uint64_t total_expanded = greedy_expanded;
    std::size_t overall_open_peak = 1U;

    while (true) {
      std::ranges::fill(distances, infinity);
      std::ranges::fill(expanded_costs, infinity);
      std::ranges::fill(hop_counts, std::numeric_limits<std::size_t>::max());
      std::ranges::fill(parents, std::nullopt);
      std::priority_queue<QueueEntry, std::vector<QueueEntry>,
                          std::greater<QueueEntry>>
          open;
      distances[start_id] = 0.0;
      hop_counts[start_id] = 0U;
      const double start_heuristic = heuristic_cost(start_id);
      open.emplace(start_heuristic, start_heuristic, 0.0, 0U, start_id);
      std::optional<LandingNodeId> reached_goal;

      while (!open.empty()) {
        if (input.stop_token.stop_requested()) {
          return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED",
                         started, reach, levels.global_level, graph_node_count,
                         graph_edge_count, total_expanded, evaluated_edge_pairs,
                         landing_field.elapsed);
        }
        const auto [estimated_total, heuristic, cost, hop_count, source_id] =
            open.top();
        open.pop();
        static_cast<void>(estimated_total);
        static_cast<void>(heuristic);
        if (cost > distances[source_id] + kTolerance ||
            hop_count != hop_counts[source_id] ||
            cost + kTolerance >= expanded_costs[source_id]) {
          continue;
        }
        expanded_costs[source_id] = cost;
        if (source_id != start_id && goal_mask[source_id] != 0U) {
          reached_goal = source_id;
          break;
        }
        ++total_expanded;
        const Vec3 source_position =
            BodyPositionForNode(source_id, start_id, *start_pose_map,
                                *snapshot.snapshot, *capability);
        const std::vector<LandingNodeId> neighbors = spatial_index.Query(
            {source_position.x, source_position.y}, input.stop_token);
        if (input.stop_token.stop_requested()) {
          return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED",
                         started, reach, levels.global_level, graph_node_count,
                         graph_edge_count, total_expanded, evaluated_edge_pairs,
                         landing_field.elapsed);
        }
        for (const LandingNodeId target_id : neighbors) {
          if (target_id == source_id) {
            continue;
          }
          const EdgeKey key{source_id, target_id};
          const auto cached = edge_cache.find(key);
          if (cached != edge_cache.end()) {
            if (const auto fatal = FatalOutcome(cached->second.status);
                fatal.has_value()) {
              return Failure(*fatal, cached->second.reason_code, started, reach,
                             levels.global_level, graph_node_count,
                             graph_edge_count, total_expanded,
                             evaluated_edge_pairs, landing_field.elapsed);
            }
            if (!cached->second.edge.has_value()) {
              continue;
            }
          }
          const double edge_cost =
              cached == edge_cache.end()
                  ? optimistic_edge_cost(source_id, target_id)
                  : cached->second.edge->cost;
          if (!std::isfinite(edge_cost) || edge_cost <= 0.0) {
            continue;
          }
          const double candidate_cost = cost + edge_cost;
          const std::size_t candidate_hops = hop_count + 1U;
          const bool better =
              candidate_cost + kTolerance < distances[target_id];
          const bool equal_cost =
              std::abs(candidate_cost - distances[target_id]) <= kTolerance;
          const bool fewer_hops =
              equal_cost && candidate_hops < hop_counts[target_id];
          const bool stable_parent = equal_cost &&
                                     candidate_hops == hop_counts[target_id] &&
                                     (!parents[target_id].has_value() ||
                                      source_id < *parents[target_id]);
          if (!better && !fewer_hops && !stable_parent) {
            continue;
          }
          distances[target_id] = candidate_cost;
          hop_counts[target_id] = candidate_hops;
          parents[target_id] = source_id;
          const double target_heuristic = heuristic_cost(target_id);
          open.emplace(candidate_cost + target_heuristic, target_heuristic,
                       candidate_cost, candidate_hops, target_id);
          overall_open_peak = std::max(overall_open_peak, open.size());
        }
      }

      if (!reached_goal.has_value()) {
        return Failure(PlanningOutcome::kNoKnownSafeRoute,
                       "GLOBAL_NO_KNOWN_SAFE_ROUTE", started, reach,
                       levels.global_level, graph_node_count, graph_edge_count,
                       total_expanded, evaluated_edge_pairs,
                       landing_field.elapsed);
      }

      std::vector<LandingNodeId> node_path;
      LandingNodeId current = *reached_goal;
      while (true) {
        node_path.push_back(current);
        if (current == start_id) {
          break;
        }
        if (!parents[current].has_value()) {
          return Failure(PlanningOutcome::kNumericalFailure,
                         "HOPPER_GLOBAL_ROUTE_RESULT_INVALID", started, reach,
                         levels.global_level, graph_node_count,
                         graph_edge_count, total_expanded, evaluated_edge_pairs,
                         landing_field.elapsed);
        }
        current = *parents[current];
      }
      std::ranges::reverse(node_path);
      if (node_path.size() < 2U) {
        return Failure(PlanningOutcome::kNumericalFailure,
                       "HOPPER_GLOBAL_ROUTE_RESULT_INVALID", started, reach,
                       levels.global_level, graph_node_count, graph_edge_count,
                       total_expanded, evaluated_edge_pairs,
                       landing_field.elapsed);
      }

      bool certificate_changed = false;
      bool candidate_invalidated = false;
      std::vector<NominalHopEdge> nominal_hops;
      nominal_hops.reserve(node_path.size() - 1U);
      for (std::size_t index = 1U; index < node_path.size(); ++index) {
        const EdgeKey key{node_path[index - 1U], node_path[index]};
        auto cached = edge_cache.find(key);
        if (cached == edge_cache.end()) {
          static_cast<void>(get_edge(key.first, key.second));
          cached = edge_cache.find(key);
          certificate_changed = true;
        }
        if (cached == edge_cache.end()) {
          return Failure(PlanningOutcome::kNumericalFailure,
                         "HOPPER_GLOBAL_ROUTE_RESULT_INVALID", started, reach,
                         levels.global_level, graph_node_count,
                         graph_edge_count, total_expanded, evaluated_edge_pairs,
                         landing_field.elapsed);
        }
        if (const auto fatal = FatalOutcome(cached->second.status);
            fatal.has_value()) {
          return Failure(*fatal, cached->second.reason_code, started, reach,
                         levels.global_level, graph_node_count,
                         graph_edge_count, total_expanded, evaluated_edge_pairs,
                         landing_field.elapsed);
        }
        if (!cached->second.edge.has_value()) {
          candidate_invalidated = true;
          break;
        }
        nominal_hops.push_back(*cached->second.edge);
      }
      if (candidate_invalidated || certificate_changed) {
        continue;
      }

      const std::size_t working_memory_bytes =
          distances.capacity() * sizeof(double) +
          expanded_costs.capacity() * sizeof(double) +
          hop_counts.capacity() * sizeof(std::size_t) +
          parents.capacity() * sizeof(std::optional<LandingNodeId>);
      return success(std::move(node_path), std::move(nominal_hops),
                     total_expanded, overall_open_peak, working_memory_bytes);
    }
  } catch (const std::bad_alloc &) {
    return Failure(PlanningOutcome::kResourceExhausted,
                   "GLOBAL_SEARCH_ALLOCATION_FAILED", started);
  }
}

} // namespace lunar::planning::hierarchical
