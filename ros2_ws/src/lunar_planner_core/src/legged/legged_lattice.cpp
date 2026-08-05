#include "legged/legged_lattice.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <queue>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "legged/legged_terrain.hpp"

namespace lunar::planning::legged {
namespace {

constexpr double kComparisonTolerance = 1.0e-9;

struct LeggedStateKey final {
  std::int32_t cell_x{};
  std::int32_t cell_y{};
  std::int32_t yaw_bin{};

  auto operator<=>(const LeggedStateKey&) const = default;
};

struct OrderedPrimitive final {
  std::size_t original_index{};
  const LeggedBodyPrimitive* primitive{};
};

[[nodiscard]] LeggedLatticeBuildResult Failure(
    const LeggedLatticeStatus status, std::string reason_code) {
  return LeggedLatticeBuildResult{
      .status = status,
      .graph = std::nullopt,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool IsFinite(const Vec3& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool IsFinite(const Pose3& pose) noexcept {
  return IsFinite(pose.position_m) &&
      std::isfinite(pose.orientation.w) &&
      std::isfinite(pose.orientation.x) &&
      std::isfinite(pose.orientation.y) &&
      std::isfinite(pose.orientation.z);
}

[[nodiscard]] bool PointInsidePolygon(
    const std::vector<Vec3>& vertices, const Vec3& point) noexcept {
  if (vertices.size() < 3U) {
    return false;
  }
  bool inside = false;
  for (std::size_t current = 0U, previous = vertices.size() - 1U;
       current < vertices.size(); previous = current++) {
    const Vec3& a = vertices[current];
    const Vec3& b = vertices[previous];
    if ((a.y > point.y) == (b.y > point.y)) {
      continue;
    }
    const double crossing_x =
        (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x;
    if (point.x < crossing_x) {
      inside = !inside;
    }
  }
  return inside;
}

[[nodiscard]] std::int32_t YawToBin(
    const double yaw_rad, const std::size_t bin_count) noexcept {
  const double width = 2.0 * std::numbers::pi /
      static_cast<double>(bin_count);
  const double normalized = NormalizeYaw(yaw_rad);
  const double positive = normalized < 0.0
      ? normalized + 2.0 * std::numbers::pi
      : normalized;
  const auto raw = static_cast<std::int64_t>(std::llround(positive / width));
  return static_cast<std::int32_t>(
      raw % static_cast<std::int64_t>(bin_count));
}

[[nodiscard]] double BinToYaw(
    const std::int32_t bin, const std::size_t bin_count) noexcept {
  return NormalizeYaw(
      static_cast<double>(bin) * 2.0 * std::numbers::pi /
      static_cast<double>(bin_count));
}

[[nodiscard]] LeggedStateKey KeyOf(
    const LeggedLatticeState& state) noexcept {
  return LeggedStateKey{
      .cell_x = state.cell_x,
      .cell_y = state.cell_y,
      .yaw_bin = state.yaw_bin,
  };
}

[[nodiscard]] LeggedPose StatePose(
    const LeggedLatticeState& state,
    const shared::MapSnapshot& map,
    const std::size_t yaw_bin_count) noexcept {
  Vec3 position = map.CellCenter(shared::GridCell{
      .x = state.cell_x,
      .y = state.cell_y,
  });
  position.z = ValidInterval(state.reachable_body_z_m)
      ? 0.5 * (state.reachable_body_z_m.lower +
               state.reachable_body_z_m.upper)
      : position.z;
  return LeggedPose{
      .position_m = position,
      .yaw_rad = BinToYaw(state.yaw_bin, yaw_bin_count),
  };
}

[[nodiscard]] bool ValidCapability(
    const LeggedCapability& capability,
    const PlannerConfig& config) noexcept {
  const auto valid_interval = [](const Interval& interval) {
    return ValidInterval(interval) && interval.lower <= 0.0 &&
           interval.upper >= 0.0 && interval.lower < interval.upper;
  };
  if (!IsFinite(capability.body_half_extent_m) ||
      capability.body_half_extent_m.x <= 0.0 ||
      capability.body_half_extent_m.y <= 0.0 ||
      capability.body_half_extent_m.z <= 0.0 ||
      !ValidInterval(capability.body_height_m) ||
      !valid_interval(capability.forward_speed_mps) ||
      !valid_interval(capability.lateral_speed_mps) ||
      !valid_interval(capability.vertical_speed_mps) ||
      !valid_interval(capability.yaw_rate_radps) ||
      capability.motion_primitives.empty() ||
      !std::isfinite(config.legged.xy_resolution_m) ||
      config.legged.xy_resolution_m <= 0.0 ||
      config.legged.yaw_bin_count < 4U ||
      config.legged.maximum_terminal_candidates == 0U ||
      config.legged.maximum_height_interval_splits == 0U ||
      config.legged.continuous_validation_maximum_subdivisions == 0U ||
      config.legged.continuous_validation_maximum_subdivisions > 32U) {
    return false;
  }
  return std::ranges::all_of(
      capability.motion_primitives,
      [](const LeggedBodyPrimitive& primitive) {
        return !primitive.primitive_id.empty() &&
            IsFinite(primitive.body_frame_displacement_m) &&
            std::isfinite(primitive.yaw_change_rad) &&
            primitive.nominal_duration.count() > 0;
      });
}

[[nodiscard]] double MaximumPlanarSpeed(
    const LeggedCapability& capability) noexcept {
  const double forward = std::max(
      std::abs(capability.forward_speed_mps.lower),
      std::abs(capability.forward_speed_mps.upper));
  const double lateral = std::max(
      std::abs(capability.lateral_speed_mps.lower),
      std::abs(capability.lateral_speed_mps.upper));
  return std::hypot(forward, lateral);
}

[[nodiscard]] double GoalDistance(
    const GoalRegion& goal, const LeggedPose& pose) noexcept {
  if (const auto* point = std::get_if<PointGoal>(&goal.target)) {
    return std::hypot(
        point->position_m.x - pose.position_m.x,
        point->position_m.y - pose.position_m.y);
  }
  const auto* region = std::get_if<PlanarRegionGoal>(&goal.target);
  if (region == nullptr || region->boundary_m.empty() ||
      PointInsidePolygon(region->boundary_m, pose.position_m)) {
    return 0.0;
  }
  double best = std::numeric_limits<double>::infinity();
  for (const Vec3& vertex : region->boundary_m) {
    best = std::min(best, std::hypot(
        vertex.x - pose.position_m.x,
        vertex.y - pose.position_m.y));
  }
  return best;
}

[[nodiscard]] std::optional<LeggedTransition> ApplyPrimitive(
    const LeggedPose& source, const Interval& source_body_z_m,
    const LeggedBodyPrimitive& primitive,
    const std::size_t primitive_index,
    const shared::SafeProjection& projection,
    const LeggedCapability& capability,
    const PlannerConfig& config,
    const std::stop_token stop_token,
    bool& canceled) {
  const shared::MapSnapshot& map = *projection.source_map();
  const double cosine = std::cos(source.yaw_rad);
  const double sine = std::sin(source.yaw_rad);
  const Vec3& relative = primitive.body_frame_displacement_m;
  const Vec3 raw_target{
      .x = source.position_m.x + cosine * relative.x - sine * relative.y,
      .y = source.position_m.y + sine * relative.x + cosine * relative.y,
      .z = source.position_m.z + relative.z,
  };
  const auto target_cell = map.PositionToCell(
      Vec2{.x = raw_target.x, .y = raw_target.y});
  if (!target_cell.has_value()) {
    return std::nullopt;
  }
  LeggedPose target{
      .position_m = map.CellCenter(*target_cell),
      .yaw_rad = BinToYaw(
          YawToBin(
              source.yaw_rad + primitive.yaw_change_rad,
              config.legged.yaw_bin_count),
          config.legged.yaw_bin_count),
  };
  target.position_m.z = raw_target.z;
  const LeggedSweepResult sweep = ValidateLeggedBodySweep(
      source, target, source_body_z_m,
      primitive.nominal_duration, projection, capability,
      std::min(
          config.legged.continuous_validation_maximum_subdivisions,
          config.legged.maximum_height_interval_splits),
      stop_token);
  if (sweep.canceled) {
    canceled = true;
    return std::nullopt;
  }
  if (!sweep.valid) {
    return std::nullopt;
  }
  target.position_m.z = 0.5 * (
      sweep.reachable_body_z_m.lower +
      sweep.reachable_body_z_m.upper);
  return LeggedTransition{
      .source_pose = source,
      .target_pose = target,
      .target_body_z_m = sweep.reachable_body_z_m,
      .primitive_index = primitive_index,
      .primitive_kind = primitive.kind,
      .nominal_duration = primitive.nominal_duration,
      .path_length_m = std::hypot(
          std::hypot(
              target.position_m.x - source.position_m.x,
              target.position_m.y - source.position_m.y),
          target.position_m.z - source.position_m.z),
  };
}

}  // namespace

bool GoalContainsBodyPose(
    const GoalRegion& goal, const LeggedPose& pose) noexcept {
  bool position_matches = false;
  if (const auto* point = std::get_if<PointGoal>(&goal.target)) {
    position_matches = std::isfinite(point->tolerance_m) &&
        point->tolerance_m >= 0.0 &&
        std::hypot(
            pose.position_m.x - point->position_m.x,
            pose.position_m.y - point->position_m.y) <=
            point->tolerance_m + kComparisonTolerance;
  } else if (const auto* region =
                 std::get_if<PlanarRegionGoal>(&goal.target)) {
    position_matches = region->boundary_m.size() >= 3U &&
        std::ranges::all_of(
            region->boundary_m,
            [](const Vec3& value) { return IsFinite(value); }) &&
        PointInsidePolygon(region->boundary_m, pose.position_m);
  }
  if (!position_matches) {
    return false;
  }
  if (!goal.yaw_rad.has_value()) {
    return true;
  }
  return std::isfinite(*goal.yaw_rad) &&
      std::isfinite(goal.yaw_tolerance_rad) &&
      goal.yaw_tolerance_rad >= 0.0 &&
      std::abs(ShortestYawDelta(pose.yaw_rad, *goal.yaw_rad)) <=
          goal.yaw_tolerance_rad + kComparisonTolerance;
}

LeggedLatticeBuildResult BuildLeggedLattice(
    const LeggedState& current_state,
    const GoalRegion& goal,
    const shared::SafeProjection& projection,
    const LeggedCapability& capability,
    const PlannerConfig& config,
    const std::stop_token stop_token) {
  if (stop_token.stop_requested()) {
    return Failure(LeggedLatticeStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (projection.source_map() == nullptr ||
      !ValidCapability(capability, config) ||
      !IsFinite(current_state.body_pose)) {
    return Failure(
        LeggedLatticeStatus::kInvalidRequest,
        "LEGGED_LATTICE_REQUEST_INVALID");
  }
  const auto current_yaw =
      YawFromQuaternion(current_state.body_pose.orientation);
  const auto start_cell = projection.source_map()->PositionToCell(Vec2{
      .x = current_state.body_pose.position_m.x,
      .y = current_state.body_pose.position_m.y,
  });
  if (!current_yaw.has_value() || !start_cell.has_value()) {
    return Failure(
        LeggedLatticeStatus::kInvalidRequest, "LEGGED_START_NOT_SAFE");
  }
  const LeggedTerrainEvaluation start_terrain =
      EvaluateLeggedTerrainCell(
          projection, capability, *start_cell, stop_token);
  if (start_terrain.canceled) {
    return Failure(LeggedLatticeStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (!start_terrain.hard_feasible ||
      current_state.body_pose.position_m.z <
          start_terrain.body_height_m.lower - kComparisonTolerance ||
      current_state.body_pose.position_m.z >
          start_terrain.body_height_m.upper + kComparisonTolerance) {
    return Failure(
        LeggedLatticeStatus::kInvalidRequest, "LEGGED_START_NOT_SAFE");
  }

  std::vector<OrderedPrimitive> ordered_primitives;
  ordered_primitives.reserve(capability.motion_primitives.size());
  for (std::size_t index = 0U;
       index < capability.motion_primitives.size(); ++index) {
    ordered_primitives.push_back(OrderedPrimitive{
        .original_index = index,
        .primitive = &capability.motion_primitives[index],
    });
  }
  std::stable_sort(
      ordered_primitives.begin(), ordered_primitives.end(),
      [](const OrderedPrimitive& lhs, const OrderedPrimitive& rhs) {
        return std::tie(lhs.primitive->primitive_id, lhs.original_index) <
            std::tie(rhs.primitive->primitive_id, rhs.original_index);
      });

  LeggedLatticeGraph graph;
  graph.true_start_pose = LeggedPose{
      .position_m = current_state.body_pose.position_m,
      .yaw_rad = *current_yaw,
  };
  graph.true_start_body_z_m = Interval{
      .lower = current_state.body_pose.position_m.z,
      .upper = current_state.body_pose.position_m.z,
  };
  const LeggedLatticeState start{
      .cell_x = start_cell->x,
      .cell_y = start_cell->y,
      .yaw_bin = YawToBin(*current_yaw, config.legged.yaw_bin_count),
      .reachable_body_z_m = start_terrain.body_height_m,
  };
  graph.states.push_back(start);
  graph.search_problem.outgoing_edges.emplace_back();
  std::map<LeggedStateKey, std::size_t> state_indices;
  std::queue<std::size_t> pending;
  pending.push(0U);
  const std::size_t map_cells = projection.source_map()->cell_count();
  const std::size_t upper_bound = map_cells >
          std::numeric_limits<std::size_t>::max() /
              config.legged.yaw_bin_count
      ? std::numeric_limits<std::size_t>::max()
      : map_cells * config.legged.yaw_bin_count;
  const std::size_t maximum_states = std::min(
      upper_bound, config.search.resources.maximum_generated_candidates);

  while (!pending.empty()) {
    if (stop_token.stop_requested()) {
      return Failure(LeggedLatticeStatus::kCanceled, "REQUEST_CANCELED");
    }
    const std::size_t source_index = pending.front();
    pending.pop();
    const LeggedLatticeState source_state = graph.states[source_index];
    const LeggedPose source_pose = source_index == 0U
        ? graph.true_start_pose
        : StatePose(source_state, *projection.source_map(),
                    config.legged.yaw_bin_count);
    const Interval source_body_z_m = source_index == 0U
        ? graph.true_start_body_z_m
        : source_state.reachable_body_z_m;
    for (const OrderedPrimitive& ordered : ordered_primitives) {
      bool canceled = false;
      auto transition = ApplyPrimitive(
          source_pose, source_body_z_m, *ordered.primitive,
          ordered.original_index, projection, capability, config, stop_token,
          canceled);
      if (canceled) {
        return Failure(LeggedLatticeStatus::kCanceled, "REQUEST_CANCELED");
      }
      if (!transition.has_value()) {
        continue;
      }
      const auto target_cell = projection.source_map()->PositionToCell(Vec2{
          .x = transition->target_pose.position_m.x,
          .y = transition->target_pose.position_m.y,
      });
      if (!target_cell.has_value()) {
        continue;
      }
      const LeggedLatticeState target_state{
          .cell_x = target_cell->x,
          .cell_y = target_cell->y,
          .yaw_bin = YawToBin(
              transition->target_pose.yaw_rad,
              config.legged.yaw_bin_count),
          .reachable_body_z_m = transition->target_body_z_m,
      };
      std::size_t target_index{};
      const auto found = state_indices.find(KeyOf(target_state));
      if (found == state_indices.end()) {
        if (graph.states.size() >= maximum_states) {
          return Failure(
              LeggedLatticeStatus::kResourceExhausted,
              "LEGGED_LATTICE_STATE_LIMIT");
        }
        target_index = graph.states.size();
        graph.states.push_back(target_state);
        graph.search_problem.outgoing_edges.emplace_back();
        state_indices.emplace(KeyOf(target_state), target_index);
        pending.push(target_index);
      } else {
        target_index = found->second;
      }
      transition->stable_index = graph.transitions.size();
      const double edge_cost =
          std::chrono::duration<double>(transition->nominal_duration).count() +
          0.5 * transition->path_length_m +
          0.25 * std::abs(ShortestYawDelta(
              transition->source_pose.yaw_rad,
              transition->target_pose.yaw_rad));
      if (!std::isfinite(edge_cost) || edge_cost <= 0.0) {
        return Failure(
            LeggedLatticeStatus::kInvalidRequest,
            "LEGGED_LATTICE_EDGE_COST_INVALID");
      }
      graph.transitions.push_back(*transition);
      graph.search_problem.outgoing_edges[source_index].push_back(
          shared::GraphEdge{
              .target_state = target_index,
              .cost = edge_cost,
              .stable_index = transition->stable_index,
          });
    }
  }

  graph.search_problem.state_count = graph.states.size();
  graph.search_problem.start_state = 0U;
  graph.search_problem.heuristic.reserve(graph.states.size());
  graph.search_problem.goal_mask.assign(graph.states.size(), 0U);
  const double maximum_speed = MaximumPlanarSpeed(capability);
  std::size_t terminal_count = 0U;
  for (std::size_t index = 0U; index < graph.states.size(); ++index) {
    const LeggedPose pose = index == 0U
        ? graph.true_start_pose
        : StatePose(graph.states[index], *projection.source_map(),
                    config.legged.yaw_bin_count);
    double heuristic = GoalDistance(goal, pose) / maximum_speed;
    if (goal.yaw_rad.has_value()) {
      const double yaw_speed = std::max(
          std::abs(capability.yaw_rate_radps.lower),
          std::abs(capability.yaw_rate_radps.upper));
      heuristic = std::max(
          heuristic,
          std::abs(ShortestYawDelta(pose.yaw_rad, *goal.yaw_rad)) /
              yaw_speed);
    }
    graph.search_problem.heuristic.push_back(
        std::isfinite(heuristic) ? heuristic : 0.0);
    if (terminal_count < config.legged.maximum_terminal_candidates &&
        GoalContainsBodyPose(goal, pose)) {
      graph.search_problem.goal_mask[index] = 1U;
      ++terminal_count;
    }
  }
  if (graph.search_problem.goal_mask.front() == 0U &&
      graph.search_problem.outgoing_edges.front().empty()) {
    return Failure(LeggedLatticeStatus::kInvalidRequest,
                   "LEGGED_START_CONNECTOR_INFEASIBLE");
  }
  graph.search_problem.config = config.search;
  return LeggedLatticeBuildResult{
      .status = LeggedLatticeStatus::kReady,
      .graph = std::move(graph),
      .reason_code = {},
  };
}

std::optional<LeggedDiscretePlan> ResolveLeggedPlan(
    const LeggedLatticeGraph& graph,
    const shared::AraStarResult& search_result) {
  if (search_result.status != shared::AraStarStatus::kSolved ||
      search_result.candidates.empty()) {
    return std::nullopt;
  }
  const shared::SearchCandidate& candidate = search_result.candidates.front();
  LeggedDiscretePlan plan{
      .transitions = {},
      .cost = candidate.cost,
      .expanded_states = search_result.expanded_states,
  };
  plan.transitions.reserve(candidate.stable_edge_indices.size());
  for (const std::size_t edge : candidate.stable_edge_indices) {
    if (edge >= graph.transitions.size()) {
      return std::nullopt;
    }
    plan.transitions.push_back(graph.transitions[edge]);
  }
  return plan.transitions.size() + 1U == candidate.states.size()
      ? std::optional<LeggedDiscretePlan>{std::move(plan)}
      : std::nullopt;
}

}  // namespace lunar::planning::legged
