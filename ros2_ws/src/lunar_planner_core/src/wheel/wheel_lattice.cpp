#include "wheel/wheel_lattice.hpp"

#include <algorithm>
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

#include "wheel/wheel_sweep_validator.hpp"

namespace lunar::planning::wheel {
namespace {

constexpr double kComparisonTolerance = 1.0e-9;

struct OrderedPrimitive final {
  std::size_t original_index{};
  const WheelMotionPrimitive* primitive{};
};

[[nodiscard]] WheelLatticeBuildResult Failure(
    const WheelLatticeStatus status, std::string reason_code) {
  return WheelLatticeBuildResult{
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

[[nodiscard]] bool IsSpin(const WheelPrimitiveKind kind) noexcept {
  return kind == WheelPrimitiveKind::kSpinClockwise ||
         kind == WheelPrimitiveKind::kSpinCounterclockwise;
}

[[nodiscard]] bool IsForward(const WheelPrimitiveKind kind) noexcept {
  return kind == WheelPrimitiveKind::kForward ||
         kind == WheelPrimitiveKind::kForwardArc;
}

[[nodiscard]] bool IsReverse(const WheelPrimitiveKind kind) noexcept {
  return kind == WheelPrimitiveKind::kReverse ||
         kind == WheelPrimitiveKind::kReverseArc;
}

[[nodiscard]] bool ModeAllows(
    const WheelMotionMode mode, const WheelPrimitiveKind kind) noexcept {
  if (IsForward(kind)) {
    return mode == WheelMotionMode::kStart ||
           mode == WheelMotionMode::kForward;
  }
  if (IsReverse(kind)) {
    return mode == WheelMotionMode::kStart ||
           mode == WheelMotionMode::kReverse;
  }
  if (kind == WheelPrimitiveKind::kStopAndSwitch) {
    return mode != WheelMotionMode::kStart;
  }
  return IsSpin(kind);
}

[[nodiscard]] WheelMotionMode TargetMode(
    const WheelPrimitiveKind kind) noexcept {
  if (IsForward(kind)) {
    return WheelMotionMode::kForward;
  }
  if (IsReverse(kind)) {
    return WheelMotionMode::kReverse;
  }
  return WheelMotionMode::kStart;
}

[[nodiscard]] std::int32_t YawToBin(
    const double yaw_rad, const std::size_t bin_count) noexcept {
  const double bin_width = 2.0 * std::numbers::pi /
      static_cast<double>(bin_count);
  const double positive = NormalizeYaw(yaw_rad) < 0.0
      ? NormalizeYaw(yaw_rad) + 2.0 * std::numbers::pi
      : NormalizeYaw(yaw_rad);
  const auto rounded = static_cast<std::int64_t>(
      std::llround(positive / bin_width));
  return static_cast<std::int32_t>(
      rounded % static_cast<std::int64_t>(bin_count));
}

[[nodiscard]] double BinToYaw(
    const std::int32_t yaw_bin, const std::size_t bin_count) noexcept {
  return NormalizeYaw(
      static_cast<double>(yaw_bin) * 2.0 * std::numbers::pi /
      static_cast<double>(bin_count));
}

[[nodiscard]] WheelPose StatePose(
    const WheelLatticeState& state, const shared::MapSnapshot& map,
    const std::size_t yaw_bin_count) noexcept {
  return WheelPose{
      .position_m = map.CellCenter(shared::GridCell{
          .x = state.cell_x,
          .y = state.cell_y,
      }),
      .yaw_rad = BinToYaw(state.yaw_bin, yaw_bin_count),
  };
}

[[nodiscard]] std::optional<WheelTransition> ApplyPrimitive(
    const WheelLatticeState& source_state, const WheelPose& source,
    const WheelMotionPrimitive& primitive,
    const std::size_t primitive_index,
    const shared::MapSnapshot& map,
    const std::size_t yaw_bin_count) {
  if (!ModeAllows(source_state.motion_mode, primitive.kind)) {
    return std::nullopt;
  }
  const auto relative_yaw = YawFromQuaternion(
      primitive.relative_end_pose.orientation);
  if (!relative_yaw.has_value()) {
    return std::nullopt;
  }

  const double cosine = std::cos(source.yaw_rad);
  const double sine = std::sin(source.yaw_rad);
  const Vec3& relative = primitive.relative_end_pose.position_m;
  Vec3 target_position{
      .x = source.position_m.x + cosine * relative.x - sine * relative.y,
      .y = source.position_m.y + sine * relative.x + cosine * relative.y,
      .z = source.position_m.z + relative.z,
  };
  if (primitive.kind == WheelPrimitiveKind::kStopAndSwitch) {
    target_position = source.position_m;
  }
  const auto target_cell = map.PositionToCell(
      Vec2{.x = target_position.x, .y = target_position.y});
  if (!target_cell.has_value()) {
    return std::nullopt;
  }
  const double target_yaw = primitive.kind == WheelPrimitiveKind::kStopAndSwitch
      ? source.yaw_rad
      : NormalizeYaw(source.yaw_rad + *relative_yaw);
  const WheelLatticeState target_state{
      .cell_x = target_cell->x,
      .cell_y = target_cell->y,
      .yaw_bin = YawToBin(target_yaw, yaw_bin_count),
      .motion_mode = TargetMode(primitive.kind),
  };
  WheelPose target = StatePose(target_state, map, yaw_bin_count);
  const double path_length_m = std::hypot(
      target.position_m.x - source.position_m.x,
      target.position_m.y - source.position_m.y);
  const double yaw_delta = ShortestYawDelta(source.yaw_rad, target.yaw_rad);
  const double curvature_per_m = path_length_m > kComparisonTolerance &&
          !IsSpin(primitive.kind)
      ? yaw_delta / path_length_m
      : 0.0;
  return WheelTransition{
      .source_pose = source,
      .target_pose = target,
      .curvature_per_m = curvature_per_m,
      .primitive_index = primitive_index,
      .primitive_kind = primitive.kind,
      .source_mode = source_state.motion_mode,
      .target_mode = target_state.motion_mode,
      .nominal_duration = primitive.nominal_duration,
      .path_length_m = path_length_m,
      .reverse = IsReverse(primitive.kind),
  };
}

[[nodiscard]] WheelLatticeState TransitionTarget(
    const WheelTransition& transition,
    const shared::MapSnapshot& map,
    const std::size_t yaw_bin_count) {
  const auto cell = map.PositionToCell(Vec2{
      .x = transition.target_pose.position_m.x,
      .y = transition.target_pose.position_m.y,
  });
  return WheelLatticeState{
      .cell_x = cell.has_value() ? cell->x : -1,
      .cell_y = cell.has_value() ? cell->y : -1,
      .yaw_bin = YawToBin(transition.target_pose.yaw_rad, yaw_bin_count),
      .motion_mode = transition.target_mode,
  };
}

[[nodiscard]] double Seconds(
    const std::chrono::nanoseconds duration) noexcept {
  return std::chrono::duration<double>(duration).count();
}

[[nodiscard]] double GoalDistance(
    const GoalRegion& goal, const WheelPose& pose) noexcept {
  if (const auto* point = std::get_if<PointGoal>(&goal.target)) {
    return std::hypot(
        point->position_m.x - pose.position_m.x,
        point->position_m.y - pose.position_m.y);
  }
  const auto* region = std::get_if<PlanarRegionGoal>(&goal.target);
  if (region == nullptr || region->boundary_m.empty()) {
    return 0.0;
  }
  double best = std::numeric_limits<double>::infinity();
  for (const Vec3& vertex : region->boundary_m) {
    best = std::min(best, std::hypot(
        vertex.x - pose.position_m.x, vertex.y - pose.position_m.y));
  }
  return best;
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
    const bool crosses = (a.y > point.y) != (b.y > point.y);
    if (!crosses) {
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

[[nodiscard]] bool ValidateCapability(
    const WheeledCapability& capability,
    const PlannerConfig& config) noexcept {
  if (capability.footprint_xy_m.size() < 3U ||
      capability.motion_primitives.empty() ||
      !std::isfinite(capability.maximum_forward_speed_mps) ||
      !std::isfinite(capability.maximum_reverse_speed_mps) ||
      !std::isfinite(capability.maximum_spin_rate_radps) ||
      !std::isfinite(capability.maximum_acceleration_mps2) ||
      !std::isfinite(capability.maximum_braking_deceleration_mps2) ||
      !std::isfinite(capability.maximum_yaw_acceleration_radps2) ||
      !std::isfinite(capability.maximum_lateral_acceleration_mps2) ||
      !std::isfinite(capability.maximum_curvature_per_m) ||
      capability.maximum_forward_speed_mps <= 0.0 ||
      capability.maximum_reverse_speed_mps <= 0.0 ||
      capability.maximum_spin_rate_radps <= 0.0 ||
      capability.maximum_acceleration_mps2 <= 0.0 ||
      capability.maximum_braking_deceleration_mps2 <= 0.0 ||
      capability.maximum_yaw_acceleration_radps2 <= 0.0 ||
      capability.maximum_lateral_acceleration_mps2 <= 0.0 ||
      capability.maximum_curvature_per_m <= 0.0 ||
      !std::isfinite(config.wheel.xy_resolution_m) ||
      config.wheel.xy_resolution_m <= 0.0 ||
      config.wheel.yaw_bin_count < 4U ||
      config.wheel.maximum_terminal_candidates == 0U ||
      config.wheel.continuous_validation_maximum_subdivisions == 0U) {
    return false;
  }
  if (!std::ranges::all_of(capability.footprint_xy_m, [](const Vec2& value) {
        return std::isfinite(value.x) && std::isfinite(value.y);
      })) {
    return false;
  }
  return std::ranges::all_of(
      capability.motion_primitives, [](const WheelMotionPrimitive& primitive) {
        return !primitive.primitive_id.empty() &&
               primitive.nominal_duration.count() > 0 &&
               IsFinite(primitive.relative_end_pose) &&
               YawFromQuaternion(
                   primitive.relative_end_pose.orientation).has_value();
      });
}

}  // namespace

bool GoalContainsPose(
    const GoalRegion& goal, const WheelPose& pose) noexcept {
  bool position_matches = false;
  if (const auto* point = std::get_if<PointGoal>(&goal.target)) {
    const double dx = pose.position_m.x - point->position_m.x;
    const double dy = pose.position_m.y - point->position_m.y;
    const double dz = pose.position_m.z - point->position_m.z;
    position_matches = std::isfinite(point->tolerance_m) &&
        point->tolerance_m >= 0.0 &&
        std::hypot(dx, dy, dz) <= point->tolerance_m + kComparisonTolerance;
  } else if (const auto* region =
                 std::get_if<PlanarRegionGoal>(&goal.target)) {
    if (!std::isfinite(region->normal_tolerance_m) ||
        region->normal_tolerance_m < 0.0 || region->boundary_m.size() < 3U ||
        !std::ranges::all_of(
            region->boundary_m,
            [](const Vec3& vertex) { return IsFinite(vertex); })) {
      return false;
    }
    double mean_z = 0.0;
    for (const Vec3& vertex : region->boundary_m) {
      mean_z += vertex.z;
    }
    mean_z /= static_cast<double>(region->boundary_m.size());
    position_matches =
        std::abs(pose.position_m.z - mean_z) <=
            region->normal_tolerance_m + kComparisonTolerance &&
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

WheelLatticeBuildResult BuildWheelLattice(
    const WheeledState& current_state,
    const GoalRegion& goal,
    const shared::SafeProjection& projection,
    const WheeledCapability& capability,
    const PlannerConfig& config,
    const std::stop_token stop_token) {
  if (stop_token.stop_requested()) {
    return Failure(WheelLatticeStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (projection.source_map() == nullptr ||
      !ValidateCapability(capability, config) ||
      !IsFinite(current_state.pose)) {
    return Failure(
        WheelLatticeStatus::kInvalidRequest, "WHEEL_LATTICE_REQUEST_INVALID");
  }
  const auto current_yaw = YawFromQuaternion(current_state.pose.orientation);
  const auto current_cell = projection.source_map()->PositionToCell(Vec2{
      .x = current_state.pose.position_m.x,
      .y = current_state.pose.position_m.y,
  });
  if (!current_yaw.has_value() || !current_cell.has_value() ||
      !projection.HardFeasible(*current_cell)) {
    return Failure(
        WheelLatticeStatus::kInvalidRequest, "WHEEL_START_NOT_SAFE");
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

  WheelLatticeGraph graph;
  graph.true_start_pose = WheelPose{
      .position_m = current_state.pose.position_m,
      .yaw_rad = *current_yaw,
  };
  const WheelLatticeState start{
      .cell_x = current_cell->x,
      .cell_y = current_cell->y,
      .yaw_bin = YawToBin(*current_yaw, config.wheel.yaw_bin_count),
      .motion_mode = WheelMotionMode::kStart,
  };
  graph.states.push_back(start);
  graph.search_problem.outgoing_edges.emplace_back();
  std::map<WheelLatticeState, std::size_t> state_indices;
  std::queue<std::size_t> pending;
  pending.push(0U);
  const std::size_t per_map_upper_bound = projection.source_map()->cell_count() >
          std::numeric_limits<std::size_t>::max() /
              config.wheel.yaw_bin_count / 3U
      ? std::numeric_limits<std::size_t>::max()
      : projection.source_map()->cell_count() *
            config.wheel.yaw_bin_count * 3U;
  const std::size_t maximum_states = std::min(
      per_map_upper_bound,
      config.search.resources.maximum_generated_candidates);
  WheelSweepValidator validator{
      projection, capability,
      config.wheel.continuous_validation_maximum_subdivisions};

  while (!pending.empty()) {
    if (stop_token.stop_requested()) {
      return Failure(WheelLatticeStatus::kCanceled, "REQUEST_CANCELED");
    }
    const std::size_t source_index = pending.front();
    pending.pop();
    const WheelLatticeState source_state = graph.states[source_index];
    const WheelPose source_pose = source_index == 0U
        ? graph.true_start_pose
        : StatePose(source_state, *projection.source_map(),
                    config.wheel.yaw_bin_count);
    for (const OrderedPrimitive& ordered : ordered_primitives) {
      auto transition = ApplyPrimitive(
          source_state, source_pose, *ordered.primitive, ordered.original_index,
          *projection.source_map(), config.wheel.yaw_bin_count);
      if (!transition.has_value()) {
        continue;
      }
      const WheelLatticeState target_state = TransitionTarget(
          *transition, *projection.source_map(), config.wheel.yaw_bin_count);
      const shared::GridCell target_cell{
          .x = target_state.cell_x,
          .y = target_state.cell_y,
      };
      if (!projection.HardFeasible(target_cell)) {
        continue;
      }
      const WheelSweepValidation sweep =
          validator.Validate(*transition, stop_token);
      if (sweep.canceled) {
        return Failure(WheelLatticeStatus::kCanceled, "REQUEST_CANCELED");
      }
      if (!sweep.valid) {
        continue;
      }

      std::size_t target_index{};
      const auto found = state_indices.find(target_state);
      if (found == state_indices.end()) {
        if (graph.states.size() >= maximum_states) {
          return Failure(
              WheelLatticeStatus::kResourceExhausted,
              "WHEEL_LATTICE_STATE_LIMIT");
        }
        target_index = graph.states.size();
        graph.states.push_back(target_state);
        graph.search_problem.outgoing_edges.emplace_back();
        state_indices.emplace(target_state, target_index);
        pending.push(target_index);
      } else {
        target_index = found->second;
      }

      transition->stable_index = graph.transitions.size();
      const double traversal_cost = transition->path_length_m >
              kComparisonTolerance
          ? static_cast<double>(projection.TraversalCost(target_cell))
          : 0.0;
      const double edge_cost = Seconds(transition->nominal_duration) +
          0.6 * traversal_cost;
      if (!std::isfinite(edge_cost) || edge_cost <= 0.0) {
        return Failure(
            WheelLatticeStatus::kInvalidRequest,
            "WHEEL_LATTICE_EDGE_COST_INVALID");
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
  graph.search_problem.goal_mask.reserve(graph.states.size());
  const double maximum_translation_speed = std::max(
      capability.maximum_forward_speed_mps,
      capability.maximum_reverse_speed_mps);
  for (std::size_t index = 0U; index < graph.states.size(); ++index) {
    const WheelPose pose = index == 0U
        ? graph.true_start_pose
        : StatePose(graph.states[index], *projection.source_map(),
                    config.wheel.yaw_bin_count);
    double heuristic = GoalDistance(goal, pose) / maximum_translation_speed;
    if (goal.yaw_rad.has_value()) {
      heuristic += std::abs(ShortestYawDelta(pose.yaw_rad, *goal.yaw_rad)) /
          capability.maximum_spin_rate_radps;
    }
    graph.search_problem.heuristic.push_back(
        std::isfinite(heuristic) ? heuristic : 0.0);
    graph.search_problem.goal_mask.push_back(
        GoalContainsPose(goal, pose) ? 1U : 0U);
  }
  if (graph.search_problem.goal_mask.front() == 0U &&
      graph.search_problem.outgoing_edges.front().empty()) {
    return Failure(WheelLatticeStatus::kInvalidRequest,
                   "WHEEL_START_CONNECTOR_INFEASIBLE");
  }
  graph.search_problem.config = config.search;
  return WheelLatticeBuildResult{
      .status = WheelLatticeStatus::kReady,
      .graph = std::move(graph),
      .reason_code = {},
  };
}

std::optional<WheelDiscretePlan> ResolveWheelPlan(
    const WheelLatticeGraph& graph,
    const shared::AraStarResult& search_result) {
  if (search_result.status != shared::AraStarStatus::kSolved ||
      search_result.candidates.empty()) {
    return std::nullopt;
  }
  const shared::SearchCandidate& candidate = search_result.candidates.front();
  WheelDiscretePlan plan{
      .transitions = {},
      .cost = candidate.cost,
      .expanded_states = search_result.expanded_states,
  };
  plan.transitions.reserve(candidate.stable_edge_indices.size());
  for (const std::size_t stable_edge : candidate.stable_edge_indices) {
    if (stable_edge >= graph.transitions.size()) {
      return std::nullopt;
    }
    plan.transitions.push_back(graph.transitions[stable_edge]);
  }
  if (plan.transitions.size() + 1U != candidate.states.size()) {
    return std::nullopt;
  }
  return plan;
}

}  // namespace lunar::planning::wheel
