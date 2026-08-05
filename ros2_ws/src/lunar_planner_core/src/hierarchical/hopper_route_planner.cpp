#include "hierarchical/hopper_route_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <ranges>
#include <set>
#include <stop_token>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "hierarchical/frame_transform.hpp"
#include "hierarchical/landing_support_field.hpp"
#include "hierarchical/map_level.hpp"
#include "hopper/ballistic_kinematics.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"

namespace lunar::planning::hierarchical {
namespace {

using Clock = std::chrono::steady_clock;
constexpr double kTolerance = 1.0e-9;
constexpr std::size_t kReachSamples = 512U;
// The global graph is only a conservative topology preview.  Keep this
// deliberately coarse; the selected first edge is re-sampled and certified
// against the complete L0 flight tube before it can be authorized.
constexpr std::size_t kEdgeTimeSamples = 16U;
constexpr std::size_t kTubeSamples = 8U;

struct LandingNode final {
  shared::GridCell cell;
  Vec3 surface_position_map;
  bool goal{};
};

struct NominalEdge final {
  std::size_t target{};
  double cost{};
  double flight_time_s{};
};

struct OutgoingEdgesResult final {
  std::vector<NominalEdge> edges;
  std::size_t evaluated_pairs{};
  bool truncated{};
  bool canceled{};
};

[[nodiscard]] HopperRoutePlanResult
Failure(const PlanningOutcome outcome, std::string reason_code,
        const Clock::time_point started, const double reach_m = 0.0,
        const std::optional<std::size_t> level = std::nullopt,
        const std::size_t graph_nodes = 0U, const std::size_t graph_edges = 0U,
        const std::uint64_t expanded = 0U, const bool truncated = false,
        const std::size_t evaluated_edge_pairs = 0U) {
  return HopperRoutePlanResult{
      .outcome = outcome,
      .route = std::nullopt,
      .global_level = level,
      .maximum_horizontal_reach_m = reach_m,
      .graph_nodes = graph_nodes,
      .graph_edges = graph_edges,
      .evaluated_edge_pairs = evaluated_edge_pairs,
      .route_hops = 0U,
      .expanded_nodes = expanded,
      .graph_truncated = truncated,
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

[[nodiscard]] double Dot(const Vec3 lhs, const Vec3 rhs) noexcept {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
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

[[nodiscard]] bool KnownAt(const shared::MapSnapshot &map,
                           const std::size_t index,
                           const MapSafetyConfig &config) noexcept {
  return map.ByteLayer("valid_mask")[index] != 0U &&
         static_cast<double>(map.FloatLayer("elevation_variance")[index]) <=
             config.maximum_elevation_variance_m2 + kTolerance &&
         static_cast<double>(map.FloatLayer("obstacle_variance")[index]) <=
             config.maximum_obstacle_variance_m2 + kTolerance &&
         static_cast<double>(map.FloatLayer("observation_age_s")[index]) <=
             config.maximum_observation_age_s + kTolerance &&
         static_cast<double>(map.FloatLayer("observation_quality")[index]) +
                 kTolerance >=
             config.minimum_observation_quality &&
         map.CountLayer("observation_count")[index] >=
             config.minimum_observation_count;
}

[[nodiscard]] bool LowCostTubeSafe(const hopper::BallisticArc &arc,
                                   const shared::MapSnapshot &map,
                                   const HopperCapability &capability,
                                   const PlannerConfig &config) {
  const auto obstacles = map.ByteLayer("obstacle");
  const auto forbidden = map.ByteLayer("forbidden");
  const auto elevation = map.FloatLayer("elevation");
  const auto obstacle_height = map.FloatLayer("obstacle_height");
  const double half_x =
      capability.body_half_extent_m.x + capability.minimum_lateral_clearance_m;
  const double half_y =
      capability.body_half_extent_m.y + capability.minimum_lateral_clearance_m;
  for (std::size_t sample = 1U; sample < kTubeSamples; ++sample) {
    const double time_s = arc.flight_time_s * static_cast<double>(sample) /
                          static_cast<double>(kTubeSamples);
    const hopper::BallisticState state =
        hopper::EvaluateBallisticState(arc, time_s);
    if (!Finite(state.position_m)) {
      return false;
    }
    const auto index_of = [&](const double coordinate, const double origin) {
      return static_cast<long long>(
          std::floor((coordinate - origin) / map.resolution_m()));
    };
    const long long minimum_x =
        index_of(state.position_m.x - half_x, map.origin_m().x);
    const long long maximum_x =
        index_of(state.position_m.x + half_x, map.origin_m().x);
    const long long minimum_y =
        index_of(state.position_m.y - half_y, map.origin_m().y);
    const long long maximum_y =
        index_of(state.position_m.y + half_y, map.origin_m().y);
    if (minimum_x < 0 || minimum_y < 0 ||
        maximum_x >= static_cast<long long>(map.width()) ||
        maximum_y >= static_cast<long long>(map.height())) {
      return false;
    }
    const double body_bottom =
        state.position_m.z - capability.body_half_extent_m.z;
    for (long long y = minimum_y; y <= maximum_y; ++y) {
      for (long long x = minimum_x; x <= maximum_x; ++x) {
        const shared::GridCell cell{
            .x = static_cast<std::int32_t>(x),
            .y = static_cast<std::int32_t>(y),
        };
        const std::size_t index = map.Index(cell);
        if (!KnownAt(map, index, config.map_safety) || forbidden[index] != 0U) {
          return false;
        }
        const double terrain = static_cast<double>(elevation[index]);
        if (body_bottom + kTolerance < terrain) {
          return false;
        }
        if (obstacles[index] != 0U) {
          const double top =
              terrain + std::max(static_cast<double>(obstacle_height[index]),
                                 map.resolution_m());
          if (top + capability.minimum_overhead_clearance_m >=
              body_bottom - kTolerance) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

[[nodiscard]] std::optional<NominalEdge>
BuildNominalEdge(const LandingNode &source, const LandingNode &target,
                 const shared::MapSnapshot &map,
                 const HopperCapability &capability,
                 const PlannerConfig &config) {
  const Vec3 launch{
      .x = source.surface_position_map.x,
      .y = source.surface_position_map.y,
      .z = source.surface_position_map.z + capability.body_half_extent_m.z,
  };
  const Vec3 landing{
      .x = target.surface_position_map.x,
      .y = target.surface_position_map.y,
      .z = target.surface_position_map.z + capability.body_half_extent_m.z,
  };
  const double horizontal =
      std::hypot(landing.x - launch.x, landing.y - launch.y);
  if (!std::isfinite(horizontal) || horizontal <= kTolerance) {
    return std::nullopt;
  }
  const double minimum_time =
      std::chrono::duration<double>(capability.minimum_flight_time).count();
  const double maximum_time =
      std::chrono::duration<double>(capability.maximum_flight_time).count();
  const double gravity_norm = Norm(capability.gravity_mps2);
  std::optional<NominalEdge> selected;
  for (std::size_t sample = 0U; sample <= kEdgeTimeSamples; ++sample) {
    const double ratio =
        static_cast<double>(sample) / static_cast<double>(kEdgeTimeSamples);
    const double time_s = minimum_time + ratio * (maximum_time - minimum_time);
    const hopper::BallisticSolveResult solved = hopper::SolveBallisticArc(
        launch, landing, capability.gravity_mps2, time_s);
    if (!solved.ok()) {
      continue;
    }
    const double launch_speed = Norm(solved.arc->launch_velocity_mps);
    const double landing_speed = Norm(solved.arc->landing_velocity_mps);
    const double impulse = capability.platform_mass_kg * launch_speed;
    const double downward_speed =
        Dot(solved.arc->landing_velocity_mps, capability.gravity_mps2) /
        gravity_norm;
    if (!std::isfinite(launch_speed) ||
        launch_speed > capability.maximum_launch_speed_mps + kTolerance ||
        !std::isfinite(landing_speed) ||
        landing_speed > capability.maximum_landing_speed_mps + kTolerance ||
        !std::isfinite(impulse) ||
        impulse >
            capability.maximum_launch_impulse_newton_seconds + kTolerance ||
        !std::isfinite(downward_speed) ||
        downward_speed + kTolerance <
            capability.minimum_downward_impact_speed_mps ||
        !LowCostTubeSafe(*solved.arc, map, capability, config)) {
      continue;
    }
    const double normalized_time = time_s / maximum_time;
    const double normalized_impulse =
        impulse / capability.maximum_launch_impulse_newton_seconds;
    const double cost = 1.0 + normalized_time + normalized_impulse;
    const NominalEdge edge{
        .cost = cost,
        .flight_time_s = time_s,
    };
    if (!selected.has_value() ||
        std::tie(edge.cost, edge.flight_time_s) <
            std::tie(selected->cost, selected->flight_time_s)) {
      selected = edge;
    }
  }
  return selected;
}

[[nodiscard]] OutgoingEdgesResult BuildOutgoingEdges(
    const std::size_t source, const std::vector<LandingNode> &nodes,
    const shared::MapSnapshot &map, const HopperCapability &capability,
    const PlannerConfig &config, const double maximum_reach_m,
    const std::stop_token stop_token) {
  OutgoingEdgesResult result;
  for (std::size_t target = 0U; target < nodes.size(); ++target) {
    if (stop_token.stop_requested()) {
      result.canceled = true;
      return result;
    }
    if (source == target) {
      continue;
    }
    ++result.evaluated_pairs;
    const double horizontal_distance =
        std::hypot(nodes[target].surface_position_map.x -
                       nodes[source].surface_position_map.x,
                   nodes[target].surface_position_map.y -
                       nodes[source].surface_position_map.y);
    if (!std::isfinite(horizontal_distance) ||
        horizontal_distance > maximum_reach_m + kTolerance) {
      continue;
    }
    auto edge =
        BuildNominalEdge(nodes[source], nodes[target], map, capability, config);
    if (!edge.has_value()) {
      continue;
    }
    edge->target = target;
    result.edges.push_back(*edge);
  }
  std::ranges::sort(
      result.edges, [](const NominalEdge &lhs, const NominalEdge &rhs) {
        return std::tie(lhs.cost, lhs.target) < std::tie(rhs.cost, rhs.target);
      });
  if (result.edges.size() > config.hopper.maximum_graph_out_degree) {
    result.truncated = true;
    result.edges.resize(config.hopper.maximum_graph_out_degree);
  }
  return result;
}

[[nodiscard]] Quaternion YawQuaternion(const double yaw_rad) noexcept {
  return Quaternion{
      .w = std::cos(yaw_rad / 2.0),
      .z = std::sin(yaw_rad / 2.0),
  };
}

} // namespace

double ConservativeMaximumHorizontalReach(
    const HopperCapability &capability) noexcept {
  if (!ValidReachCapability(capability)) {
    return 0.0;
  }
  const double gravity = Norm(capability.gravity_mps2);
  const double launch_limit =
      std::min(capability.maximum_launch_speed_mps,
               capability.maximum_launch_impulse_newton_seconds /
                   capability.platform_mass_kg);
  const double landing_limit = capability.maximum_landing_speed_mps;
  const double minimum_time =
      std::chrono::duration<double>(capability.minimum_flight_time).count();
  const double maximum_time =
      std::chrono::duration<double>(capability.maximum_flight_time).count();
  double maximum_reach = 0.0;
  for (std::size_t sample = 0U; sample <= kReachSamples; ++sample) {
    const double ratio =
        static_cast<double>(sample) / static_cast<double>(kReachSamples);
    const double time_s = minimum_time + ratio * (maximum_time - minimum_time);
    const double vertical_speed = 0.5 * gravity * time_s;
    if (vertical_speed + kTolerance <
            capability.minimum_downward_impact_speed_mps ||
        vertical_speed > launch_limit + kTolerance ||
        vertical_speed > landing_limit + kTolerance) {
      continue;
    }
    const double launch_horizontal = std::sqrt(std::max(
        0.0, launch_limit * launch_limit - vertical_speed * vertical_speed));
    const double landing_horizontal = std::sqrt(std::max(
        0.0, landing_limit * landing_limit - vertical_speed * vertical_speed));
    maximum_reach =
        std::max(maximum_reach,
                 time_s * std::min(launch_horizontal, landing_horizontal));
  }
  return std::isfinite(maximum_reach) ? maximum_reach : 0.0;
}

HopperRoutePlanResult PlanHopperGlobalRoute(const PlannerInput &input) {
  const Clock::time_point started = Clock::now();
  if (input.stop_token.stop_requested()) {
    return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED", started);
  }
  const auto *state = std::get_if<HopperState>(&input.current_state);
  const auto *capability = std::get_if<HopperCapability>(&input.capability);
  if (state == nullptr || capability == nullptr ||
      !ValidReachCapability(*capability) ||
      input.config.hopper.maximum_landing_regions == 0U ||
      input.config.hopper.maximum_graph_nodes < 2U ||
      input.config.hopper.maximum_graph_out_degree == 0U ||
      input.config.global_search.resources.maximum_expanded_states == 0U) {
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
      input.world.local_map.frame_id != input.world.map_from_odom.child_frame) {
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
                   projection.reason_code, started, reach, levels.global_level);
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
  const auto start_pose_map =
      TransformPose(state->pose, input.world.map_from_odom,
                    TransformDirection::kChildToParent);
  if (!start_pose_map.has_value()) {
    return Failure(PlanningOutcome::kInvalidRequest, "FRAME_TRANSFORM_INVALID",
                   started, reach, levels.global_level);
  }
  const auto start_cell = snapshot.snapshot->PositionToCell(Vec2{
      .x = start_pose_map->position_m.x,
      .y = start_pose_map->position_m.y,
  });
  if (!start_cell.has_value() || !landing_field.field->BaseSafe(*start_cell)) {
    return Failure(PlanningOutcome::kNoKnownSafeRoute, "HOPPER_START_NOT_SAFE",
                   started, reach, levels.global_level);
  }
  if (!landing_field.field->CenterSafe(*start_cell)) {
    return Failure(PlanningOutcome::kNoKnownSafeRoute,
                   "HOPPER_START_REGION_AREA_INSUFFICIENT", started, reach,
                   levels.global_level);
  }

  std::vector<shared::GridCell> goal_cells;
  for (std::size_t index = 0U; index < snapshot.snapshot->cell_count();
       ++index) {
    if (input.stop_token.stop_requested()) {
      return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED", started,
                     reach, levels.global_level);
    }
    const shared::GridCell cell{
        .x = static_cast<std::int32_t>(index % snapshot.snapshot->width()),
        .y = static_cast<std::int32_t>(index / snapshot.snapshot->width()),
    };
    const Vec3 center = snapshot.snapshot->CellCenter(cell);
    if (GoalIntersectsCell(input.goal_map, center,
                           snapshot.snapshot->resolution_m()) &&
        landing_field.field->CenterSafe(cell)) {
      goal_cells.push_back(cell);
    }
  }
  if (goal_cells.empty()) {
    return Failure(PlanningOutcome::kGoalInfeasible,
                   "HOPPER_GLOBAL_GOAL_INFEASIBLE", started, reach,
                   levels.global_level);
  }

  const std::size_t node_capacity =
      std::min(input.config.hopper.maximum_graph_nodes,
               input.config.hopper.maximum_landing_regions >=
                       input.config.hopper.maximum_graph_nodes
                   ? input.config.hopper.maximum_graph_nodes
                   : input.config.hopper.maximum_landing_regions + 1U);
  std::vector<LandingNode> nodes;
  nodes.reserve(node_capacity);
  std::set<shared::GridCell> inserted;
  nodes.push_back(LandingNode{
      .cell = *start_cell,
      .surface_position_map =
          Vec3{
              .x = start_pose_map->position_m.x,
              .y = start_pose_map->position_m.y,
              .z = snapshot.snapshot->CellCenter(*start_cell).z,
          },
      .goal = false,
  });
  inserted.insert(*start_cell);
  bool node_truncated = false;
  for (const shared::GridCell cell : goal_cells) {
    if (inserted.contains(cell)) {
      continue;
    }
    if (nodes.size() >= node_capacity) {
      return Failure(PlanningOutcome::kResourceExhausted,
                     "HOPPER_GLOBAL_ROUTE_RESOURCE_LIMIT", started, reach,
                     levels.global_level, nodes.size(), 0U, 0U, true);
    }
    nodes.push_back(LandingNode{
        .cell = cell,
        .surface_position_map = snapshot.snapshot->CellCenter(cell),
        .goal = true,
    });
    inserted.insert(cell);
  }
  for (std::size_t index = 0U; index < snapshot.snapshot->cell_count();
       ++index) {
    const shared::GridCell cell{
        .x = static_cast<std::int32_t>(index % snapshot.snapshot->width()),
        .y = static_cast<std::int32_t>(index / snapshot.snapshot->width()),
    };
    if (inserted.contains(cell) || !landing_field.field->CenterSafe(cell)) {
      continue;
    }
    if (nodes.size() >= node_capacity) {
      node_truncated = true;
      break;
    }
    nodes.push_back(LandingNode{
        .cell = cell,
        .surface_position_map = snapshot.snapshot->CellCenter(cell),
        .goal = false,
    });
    inserted.insert(cell);
  }

  std::vector<std::optional<std::vector<NominalEdge>>> adjacency(nodes.size());
  std::size_t graph_edge_count = 0U;
  std::size_t evaluated_edge_pairs = 0U;
  bool graph_truncated = node_truncated;

  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> distances(nodes.size(), infinity);
  std::vector<std::size_t> hops(nodes.size(),
                                std::numeric_limits<std::size_t>::max());
  std::vector<std::optional<std::size_t>> parent(nodes.size());
  using QueueEntry = std::tuple<double, std::size_t, std::size_t>;
  std::priority_queue<QueueEntry, std::vector<QueueEntry>,
                      std::greater<QueueEntry>>
      open;
  distances[0] = 0.0;
  hops[0] = 0U;
  open.emplace(0.0, 0U, 0U);
  std::uint64_t expanded = 0U;
  std::size_t open_peak = 1U;
  std::optional<std::size_t> reached_goal;
  while (!open.empty()) {
    if (input.stop_token.stop_requested()) {
      return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED", started,
                     reach, levels.global_level, nodes.size(), graph_edge_count,
                     expanded, graph_truncated, evaluated_edge_pairs);
    }
    const auto [cost, hop_count, node] = open.top();
    open.pop();
    if (cost > distances[node] + kTolerance || hop_count != hops[node]) {
      continue;
    }
    if (nodes[node].goal) {
      reached_goal = node;
      break;
    }
    if (expanded >=
        input.config.global_search.resources.maximum_expanded_states) {
      return Failure(PlanningOutcome::kResourceExhausted,
                     "HOPPER_GLOBAL_ROUTE_RESOURCE_LIMIT", started, reach,
                     levels.global_level, nodes.size(), graph_edge_count,
                     expanded, true, evaluated_edge_pairs);
    }
    ++expanded;
    if (!adjacency[node].has_value()) {
      OutgoingEdgesResult outgoing =
          BuildOutgoingEdges(node, nodes, *snapshot.snapshot, *capability,
                             input.config, reach, input.stop_token);
      evaluated_edge_pairs += outgoing.evaluated_pairs;
      if (outgoing.canceled) {
        return Failure(PlanningOutcome::kCanceled, "REQUEST_CANCELED", started,
                       reach, levels.global_level, nodes.size(),
                       graph_edge_count, expanded, graph_truncated,
                       evaluated_edge_pairs);
      }
      graph_truncated = graph_truncated || outgoing.truncated;
      graph_edge_count += outgoing.edges.size();
      adjacency[node] = std::move(outgoing.edges);
    }
    for (const NominalEdge &edge : *adjacency[node]) {
      const double candidate_cost = cost + edge.cost;
      const std::size_t candidate_hops = hop_count + 1U;
      const bool better = candidate_cost + kTolerance < distances[edge.target];
      const bool stable_tie =
          std::abs(candidate_cost - distances[edge.target]) <= kTolerance &&
          candidate_hops < hops[edge.target];
      if (!better && !stable_tie) {
        continue;
      }
      distances[edge.target] = candidate_cost;
      hops[edge.target] = candidate_hops;
      parent[edge.target] = node;
      open.emplace(candidate_cost, candidate_hops, edge.target);
      open_peak = std::max(open_peak, open.size());
    }
  }
  if (!reached_goal.has_value()) {
    return Failure(graph_truncated ? PlanningOutcome::kResourceExhausted
                                   : PlanningOutcome::kNoKnownSafeRoute,
                   graph_truncated ? "HOPPER_GLOBAL_ROUTE_RESOURCE_LIMIT"
                                   : "GLOBAL_NO_KNOWN_SAFE_ROUTE",
                   started, reach, levels.global_level, nodes.size(),
                   graph_edge_count, expanded, graph_truncated,
                   evaluated_edge_pairs);
  }

  std::vector<std::size_t> node_path;
  for (std::optional<std::size_t> current = reached_goal; current.has_value();
       current = parent[*current]) {
    node_path.push_back(*current);
  }
  std::ranges::reverse(node_path);
  if (node_path.size() < 2U || node_path.front() != 0U) {
    return Failure(PlanningOutcome::kNumericalFailure,
                   "HOPPER_GLOBAL_ROUTE_RESULT_INVALID", started, reach,
                   levels.global_level, nodes.size(), graph_edge_count,
                   expanded, graph_truncated, evaluated_edge_pairs);
  }

  GlobalRoute route;
  route.raw_cells.reserve(node_path.size());
  route.simplified_cells.reserve(node_path.size());
  route.poses_map.reserve(node_path.size());
  for (std::size_t index = 0U; index < node_path.size(); ++index) {
    const LandingNode &node = nodes[node_path[index]];
    route.raw_cells.push_back(node.cell);
    route.simplified_cells.push_back(node.cell);
    double yaw = 0.0;
    if (index + 1U < node_path.size()) {
      const Vec3 next = nodes[node_path[index + 1U]].surface_position_map;
      yaw = std::atan2(next.y - node.surface_position_map.y,
                       next.x - node.surface_position_map.x);
    } else if (input.goal_map.yaw_rad.has_value()) {
      yaw = *input.goal_map.yaw_rad;
    } else if (index > 0U) {
      const Vec3 previous = nodes[node_path[index - 1U]].surface_position_map;
      yaw = std::atan2(node.surface_position_map.y - previous.y,
                       node.surface_position_map.x - previous.x);
    }
    route.poses_map.push_back(Pose3{
        .position_m =
            Vec3{
                .x = node.surface_position_map.x,
                .y = node.surface_position_map.y,
                .z = node.surface_position_map.z +
                     capability->body_half_extent_m.z,
            },
        .orientation = YawQuaternion(yaw),
    });
  }
  route.cost = distances[*reached_goal];
  route.expanded_states = expanded;
  route.open_peak = open_peak;
  route.estimated_work_memory_bytes =
      landing_field.field->EstimatedWorkMemoryBytes() +
      nodes.size() * sizeof(LandingNode) +
      graph_edge_count * sizeof(NominalEdge) +
      nodes.size() * (sizeof(double) + sizeof(std::size_t) +
                      sizeof(std::optional<std::size_t>));
  return HopperRoutePlanResult{
      .outcome = PlanningOutcome::kNewReferenceAvailable,
      .route = std::move(route),
      .global_level = levels.global_level,
      .maximum_horizontal_reach_m = reach,
      .graph_nodes = nodes.size(),
      .graph_edges = graph_edge_count,
      .evaluated_edge_pairs = evaluated_edge_pairs,
      .route_hops = node_path.size() - 1U,
      .expanded_nodes = expanded,
      .graph_truncated = graph_truncated,
      .landing_field_elapsed = landing_field.elapsed,
      .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - started),
      .reason_code = "HOPPER_GLOBAL_ROUTE_AVAILABLE",
  };
}

} // namespace lunar::planning::hierarchical
