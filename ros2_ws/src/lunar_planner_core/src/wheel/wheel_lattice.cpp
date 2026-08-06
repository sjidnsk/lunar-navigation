#include "wheel/wheel_lattice.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <new>
#include <numbers>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "wheel/wheel_sweep_validator.hpp"

namespace lunar::planning::wheel {
namespace {

constexpr double kComparisonTolerance = 1.0e-9;
constexpr double kWheelRoughnessReferenceM = 0.1595;
constexpr std::size_t kNoParent = std::numeric_limits<std::size_t>::max();

struct OrderedPrimitive final {
  std::size_t original_index{};
  const WheelMotionPrimitive* primitive{};
};

struct SearchNode final {
  WheelLatticeState key;
  WheelPose pose;
  double path_cost{std::numeric_limits<double>::infinity()};
  std::size_t parent{kNoParent};
  std::optional<WheelTransition> incoming;
  std::size_t open_sequence{};
  bool closed{};
};

struct OpenEntry final {
  double estimated_total_cost{};
  double path_cost{};
  WheelLatticeState key;
  std::size_t node_index{};
  std::size_t sequence{};
};

struct OpenLess final {
  [[nodiscard]] bool operator()(
      const OpenEntry& lhs, const OpenEntry& rhs) const noexcept {
    return std::tie(
               lhs.estimated_total_cost, lhs.path_cost, lhs.key,
               lhs.sequence, lhs.node_index) <
        std::tie(
               rhs.estimated_total_cost, rhs.path_cost, rhs.key,
               rhs.sequence, rhs.node_index);
  }
};

[[nodiscard]] WheelLatticeSearchResult Failure(
    const WheelLatticeStatus status, std::string reason_code,
    const std::size_t expanded_states = 0U) {
  return WheelLatticeSearchResult{
      .status = status,
      .plan = status == WheelLatticeStatus::kNoPath
          ? std::optional<WheelDiscretePlan>{WheelDiscretePlan{
                .transitions = {},
                .cost = std::numeric_limits<double>::infinity(),
                .expanded_states = expanded_states,
            }}
          : std::nullopt,
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
  const double normalized = NormalizeYaw(yaw_rad);
  const double positive = normalized < 0.0
      ? normalized + 2.0 * std::numbers::pi
      : normalized;
  const auto rounded = static_cast<std::int64_t>(
      std::llround(positive / bin_width));
  return static_cast<std::int32_t>(
      rounded % static_cast<std::int64_t>(bin_count));
}

[[nodiscard]] WheelLatticeState PoseKey(
    const WheelPose& pose, const WheelMotionMode mode,
    const shared::MapSnapshot& map,
    const std::size_t yaw_bin_count) noexcept {
  const auto cell = map.PositionToCell(Vec2{
      .x = pose.position_m.x,
      .y = pose.position_m.y,
  });
  return WheelLatticeState{
      .cell_x = cell.has_value() ? cell->x : -1,
      .cell_y = cell.has_value() ? cell->y : -1,
      .yaw_bin = YawToBin(pose.yaw_rad, yaw_bin_count),
      .motion_mode = mode,
  };
}

[[nodiscard]] double ArcLength(
    const WheelPrimitiveKind kind, const double chord_m,
    const double yaw_delta) noexcept {
  if ((kind != WheelPrimitiveKind::kForwardArc &&
       kind != WheelPrimitiveKind::kReverseArc) ||
      chord_m <= kComparisonTolerance ||
      std::abs(yaw_delta) <= kComparisonTolerance) {
    return chord_m;
  }
  const double sine = std::abs(std::sin(0.5 * yaw_delta));
  if (sine <= kComparisonTolerance) {
    return chord_m;
  }
  return std::abs(yaw_delta) * chord_m / (2.0 * sine);
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
  const auto target_cell = map.PositionToCell(Vec2{
      .x = target_position.x,
      .y = target_position.y,
  });
  if (!target_cell.has_value()) {
    return std::nullopt;
  }
  const double target_yaw =
      primitive.kind == WheelPrimitiveKind::kStopAndSwitch
      ? source.yaw_rad
      : NormalizeYaw(source.yaw_rad + *relative_yaw);
  const WheelMotionMode target_mode = TargetMode(primitive.kind);
  const WheelLatticeState target_state{
      .cell_x = target_cell->x,
      .cell_y = target_cell->y,
      .yaw_bin = YawToBin(target_yaw, yaw_bin_count),
      .motion_mode = target_mode,
  };
  if (target_state == source_state) {
    return std::nullopt;
  }
  const double chord_m = std::hypot(
      target_position.x - source.position_m.x,
      target_position.y - source.position_m.y);
  const double yaw_delta = ShortestYawDelta(source.yaw_rad, target_yaw);
  const double path_length_m = ArcLength(
      primitive.kind, chord_m, yaw_delta);
  const double curvature_per_m =
      path_length_m > kComparisonTolerance && !IsSpin(primitive.kind)
      ? yaw_delta / path_length_m
      : 0.0;
  return WheelTransition{
      .source_pose = source,
      .target_pose = WheelPose{
          .position_m = target_position,
          .yaw_rad = target_yaw,
      },
      .curvature_per_m = curvature_per_m,
      .primitive_index = primitive_index,
      .primitive_kind = primitive.kind,
      .source_mode = source_state.motion_mode,
      .target_mode = target_mode,
      .path_length_m = path_length_m,
      .reverse = IsReverse(primitive.kind),
  };
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
  if (GoalContainsPose(goal, pose)) {
    return 0.0;
  }
  double best = std::numeric_limits<double>::infinity();
  for (const Vec3& vertex : region->boundary_m) {
    best = std::min(
        best, std::hypot(
                  vertex.x - pose.position_m.x,
                  vertex.y - pose.position_m.y));
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
      !std::isfinite(capability.wheel_diameter_m) ||
      !std::isfinite(capability.wheelbase_m) ||
      !std::isfinite(capability.track_width_m) ||
      !std::isfinite(capability.minimum_underbody_clearance_m) ||
      !std::isfinite(capability.maximum_local_obstacle_relief_m) ||
      !std::isfinite(capability.maximum_forward_speed_mps) ||
      !std::isfinite(capability.maximum_reverse_speed_mps) ||
      !std::isfinite(capability.maximum_spin_rate_radps) ||
      !std::isfinite(capability.maximum_acceleration_mps2) ||
      !std::isfinite(capability.maximum_braking_deceleration_mps2) ||
      !std::isfinite(capability.maximum_yaw_acceleration_radps2) ||
      !std::isfinite(capability.maximum_lateral_acceleration_mps2) ||
      !std::isfinite(capability.maximum_curvature_per_m) ||
      capability.wheel_diameter_m <= 0.0 ||
      capability.wheelbase_m <= 0.0 ||
      capability.track_width_m <= 0.0 ||
      capability.minimum_underbody_clearance_m <= 0.0 ||
      capability.maximum_local_obstacle_relief_m < 0.0 ||
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
      config.wheel.yaw_bin_count < 4U) {
    return false;
  }
  if (!std::ranges::all_of(
          capability.footprint_xy_m, [](const Vec2& value) {
            return std::isfinite(value.x) && std::isfinite(value.y);
          })) {
    return false;
  }
  return std::ranges::all_of(
      capability.motion_primitives,
      [](const WheelMotionPrimitive& primitive) {
        return !primitive.primitive_id.empty() &&
            IsFinite(primitive.relative_end_pose) &&
            YawFromQuaternion(
                primitive.relative_end_pose.orientation).has_value();
      });
}

[[nodiscard]] double EdgeCost(
    const WheelTransition& transition,
    const shared::SafeProjection& projection,
    const WheeledCapability& capability) noexcept {
  const double yaw_distance = std::abs(ShortestYawDelta(
      transition.source_pose.yaw_rad, transition.target_pose.yaw_rad));
  double cost = 0.0;
  if (transition.path_length_m > kComparisonTolerance) {
    double speed_limit = transition.reverse
        ? capability.maximum_reverse_speed_mps
        : capability.maximum_forward_speed_mps;
    const double curvature = std::abs(transition.curvature_per_m);
    if (curvature > kComparisonTolerance) {
      speed_limit = std::min(
          {speed_limit,
           capability.maximum_spin_rate_radps / curvature,
           std::sqrt(
               capability.maximum_lateral_acceleration_mps2 / curvature)});
    }
    const double ratio =
        transition.roughness_m / kWheelRoughnessReferenceM;
    const double roughness_scale = 1.0 / (1.0 + ratio * ratio);
    speed_limit *=
        std::max(0.0, std::cos(transition.surface_slope_rad)) *
        roughness_scale;
    cost = transition.path_length_m / speed_limit;
  } else if (yaw_distance > kComparisonTolerance) {
    cost = yaw_distance / capability.maximum_spin_rate_radps;
  } else {
    cost = 1.0e-3;
  }
  const auto target_cell = projection.source_map()->PositionToCell(Vec2{
      .x = transition.target_pose.position_m.x,
      .y = transition.target_pose.position_m.y,
  });
  if (target_cell.has_value() &&
      transition.path_length_m > kComparisonTolerance) {
    cost += 0.6 * static_cast<double>(
        projection.TraversalCost(*target_cell));
  }
  return cost;
}

[[nodiscard]] std::optional<WheelTransition> ExactGoalConnector(
    const SearchNode& source, const GoalRegion& goal,
    const double maximum_connector_length_m,
    const shared::SafeProjection& projection,
    const WheeledCapability& capability,
    const PlannerConfig& config,
    const WheelSweepValidator& validator,
    const std::stop_token stop_token) {
  const auto* point = std::get_if<PointGoal>(&goal.target);
  if (point == nullptr || !IsFinite(point->position_m)) {
    return std::nullopt;
  }
  const double dx = point->position_m.x - source.pose.position_m.x;
  const double dy = point->position_m.y - source.pose.position_m.y;
  const double distance = std::hypot(dx, dy);
  if (distance > maximum_connector_length_m + kComparisonTolerance) {
    return std::nullopt;
  }
  bool reverse = false;
  WheelPrimitiveKind kind = WheelPrimitiveKind::kForward;
  double target_yaw = goal.yaw_rad.value_or(source.pose.yaw_rad);
  double connector_curvature = 0.0;
  if (distance > kComparisonTolerance) {
    const double direction = std::atan2(dy, dx);
    const double forward_delta = ShortestYawDelta(
        source.pose.yaw_rad, direction);
    const double reverse_direction =
        NormalizeYaw(direction + std::numbers::pi);
    const double reverse_delta = ShortestYawDelta(
        source.pose.yaw_rad, reverse_direction);
    const double forward_error = std::abs(forward_delta);
    const double reverse_error = std::abs(reverse_delta);
    const double heading_tolerance = std::numbers::pi /
        static_cast<double>(config.wheel.yaw_bin_count) +
        kComparisonTolerance;
    const double reachable_heading_change =
        capability.maximum_curvature_per_m * distance + heading_tolerance;
    if (forward_error <= reachable_heading_change &&
        (source.key.motion_mode == WheelMotionMode::kStart ||
         source.key.motion_mode == WheelMotionMode::kForward)) {
      if (!goal.yaw_rad.has_value()) {
        target_yaw = direction;
      }
      kind = goal.yaw_rad.has_value() &&
              std::abs(ShortestYawDelta(source.pose.yaw_rad, target_yaw)) >
                  kComparisonTolerance
          ? WheelPrimitiveKind::kForwardArc
          : WheelPrimitiveKind::kForward;
      const double terminal_heading_error = std::abs(ShortestYawDelta(
          target_yaw, direction));
      connector_curvature = std::copysign(
          std::max({
              std::abs(ShortestYawDelta(
                  source.pose.yaw_rad, target_yaw)) / distance,
              2.0 * forward_error / distance,
              2.0 * terminal_heading_error / distance}),
          std::abs(forward_delta) > kComparisonTolerance
              ? forward_delta
              : ShortestYawDelta(source.pose.yaw_rad, target_yaw));
    } else if (reverse_error <= reachable_heading_change &&
               (source.key.motion_mode == WheelMotionMode::kStart ||
                source.key.motion_mode == WheelMotionMode::kReverse)) {
      reverse = true;
      if (!goal.yaw_rad.has_value()) {
        target_yaw = reverse_direction;
      }
      kind = goal.yaw_rad.has_value() &&
              std::abs(ShortestYawDelta(source.pose.yaw_rad, target_yaw)) >
                  kComparisonTolerance
          ? WheelPrimitiveKind::kReverseArc
          : WheelPrimitiveKind::kReverse;
      const double terminal_heading_error = std::abs(ShortestYawDelta(
          target_yaw, reverse_direction));
      connector_curvature = std::copysign(
          std::max({
              std::abs(ShortestYawDelta(
                  source.pose.yaw_rad, target_yaw)) / distance,
              2.0 * reverse_error / distance,
              2.0 * terminal_heading_error / distance}),
          std::abs(reverse_delta) > kComparisonTolerance
              ? reverse_delta
              : ShortestYawDelta(source.pose.yaw_rad, target_yaw));
    } else {
      return std::nullopt;
    }
  } else if (goal.yaw_rad.has_value()) {
    kind = ShortestYawDelta(source.pose.yaw_rad, target_yaw) >= 0.0
        ? WheelPrimitiveKind::kSpinCounterclockwise
        : WheelPrimitiveKind::kSpinClockwise;
  } else {
    return std::nullopt;
  }
  WheelTransition transition{
      .source_pose = source.pose,
      .target_pose = WheelPose{
          .position_m = point->position_m,
          .yaw_rad = NormalizeYaw(target_yaw),
      },
      .curvature_per_m = connector_curvature,
      .primitive_index = capability.motion_primitives.size(),
      .primitive_kind = kind,
      .source_mode = source.key.motion_mode,
      .target_mode = distance <= kComparisonTolerance
          ? WheelMotionMode::kStart
          : (reverse ? WheelMotionMode::kReverse
                     : WheelMotionMode::kForward),
      .path_length_m = distance,
      .reverse = reverse,
  };
  const auto target_cell = projection.source_map()->PositionToCell(Vec2{
      .x = point->position_m.x,
      .y = point->position_m.y,
  });
  if (!target_cell.has_value() ||
      !projection.HardFeasible(*target_cell)) {
    return std::nullopt;
  }
  const WheelSweepValidation sweep = validator.Validate(transition, stop_token);
  if (!sweep.valid) {
    return std::nullopt;
  }
  transition.surface_slope_rad = sweep.maximum_surface_slope_rad;
  transition.roughness_m = sweep.maximum_roughness_m;
  return transition;
}

[[nodiscard]] std::optional<WheelDiscretePlan> Reconstruct(
    const std::vector<SearchNode>& nodes, std::size_t terminal_parent,
    const std::optional<WheelTransition>& terminal_transition,
    const double cost, const std::size_t expanded_states) {
  std::vector<WheelTransition> reversed;
  if (terminal_transition.has_value()) {
    reversed.push_back(*terminal_transition);
  }
  std::size_t current = terminal_parent;
  while (current != 0U) {
    if (current >= nodes.size() || nodes[current].parent == kNoParent ||
        !nodes[current].incoming.has_value()) {
      return std::nullopt;
    }
    reversed.push_back(*nodes[current].incoming);
    current = nodes[current].parent;
    if (reversed.size() > nodes.size() + 1U) {
      return std::nullopt;
    }
  }
  std::reverse(reversed.begin(), reversed.end());
  for (std::size_t index = 0U; index < reversed.size(); ++index) {
    reversed[index].stable_index = index;
  }
  return WheelDiscretePlan{
      .transitions = std::move(reversed),
      .cost = cost,
      .expanded_states = expanded_states,
  };
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
        std::hypot(std::hypot(dx, dy), dz) <=
            point->tolerance_m + kComparisonTolerance;
  } else if (const auto* region =
                 std::get_if<PlanarRegionGoal>(&goal.target)) {
    if (!std::isfinite(region->normal_tolerance_m) ||
        region->normal_tolerance_m < 0.0 ||
        region->boundary_m.size() < 3U ||
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

WheelLatticeSearchResult SearchWheelLattice(
    const WheeledState& current_state, const GoalRegion& goal,
    const shared::SafeProjection& projection,
    const WheeledCapability& capability, const PlannerConfig& config,
    const std::stop_token stop_token) try {
  if (stop_token.stop_requested()) {
    return Failure(WheelLatticeStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (projection.source_map() == nullptr ||
      !ValidateCapability(capability, config) ||
      !IsFinite(current_state.pose)) {
    return Failure(
        WheelLatticeStatus::kInvalidRequest,
        "WHEEL_LATTICE_REQUEST_INVALID");
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

  const WheelPose true_start{
      .position_m = current_state.pose.position_m,
      .yaw_rad = *current_yaw,
  };
  if (GoalContainsPose(goal, true_start)) {
    return WheelLatticeSearchResult{
        .status = WheelLatticeStatus::kSolved,
        .plan = WheelDiscretePlan{
            .transitions = {}, .cost = 0.0, .expanded_states = 0U},
        .reason_code = {},
    };
  }

  std::vector<OrderedPrimitive> ordered_primitives;
  ordered_primitives.reserve(capability.motion_primitives.size());
  double maximum_connector_length_m = 0.0;
  for (std::size_t index = 0U;
       index < capability.motion_primitives.size(); ++index) {
    const WheelMotionPrimitive& primitive = capability.motion_primitives[index];
    ordered_primitives.push_back(OrderedPrimitive{
        .original_index = index,
        .primitive = &primitive,
    });
    maximum_connector_length_m = std::max(
        maximum_connector_length_m,
        std::hypot(
            primitive.relative_end_pose.position_m.x,
            primitive.relative_end_pose.position_m.y));
  }
  maximum_connector_length_m +=
      0.51 * projection.source_map()->resolution_m();
  std::stable_sort(
      ordered_primitives.begin(), ordered_primitives.end(),
      [](const OrderedPrimitive& lhs, const OrderedPrimitive& rhs) {
        return std::tie(
                   lhs.primitive->primitive_id, lhs.original_index) <
            std::tie(rhs.primitive->primitive_id, rhs.original_index);
      });

  const double maximum_translation_speed = std::max(
      capability.maximum_forward_speed_mps,
      capability.maximum_reverse_speed_mps);
  const auto heuristic = [&](const WheelPose& pose) {
    const double distance_cost =
        GoalDistance(goal, pose) / maximum_translation_speed;
    const double yaw_cost = goal.yaw_rad.has_value()
        ? std::abs(ShortestYawDelta(pose.yaw_rad, *goal.yaw_rad)) /
              capability.maximum_spin_rate_radps
        : 0.0;
    return std::max(distance_cost, yaw_cost);
  };

  const WheelLatticeState start_key{
      .cell_x = current_cell->x,
      .cell_y = current_cell->y,
      .yaw_bin = YawToBin(*current_yaw, config.wheel.yaw_bin_count),
      .motion_mode = WheelMotionMode::kStart,
  };
  std::vector<SearchNode> nodes;
  nodes.push_back(SearchNode{
      .key = start_key,
      .pose = true_start,
      .path_cost = 0.0,
      .parent = kNoParent,
      .incoming = std::nullopt,
      .open_sequence = 0U,
      .closed = false,
  });
  std::map<WheelLatticeState, std::size_t> node_by_key;
  node_by_key.emplace(start_key, 0U);
  std::set<OpenEntry, OpenLess> open;
  std::vector<std::optional<OpenEntry>> open_by_node(1U);
  const OpenEntry start_open{
      .estimated_total_cost = heuristic(true_start),
      .path_cost = 0.0,
      .key = start_key,
      .node_index = 0U,
      .sequence = 0U,
  };
  open.insert(start_open);
  open_by_node[0U] = start_open;
  std::size_t next_sequence = 1U;
  std::size_t expanded_states = 0U;
  std::optional<double> best_goal_cost;
  std::size_t best_goal_parent = kNoParent;
  std::optional<WheelTransition> best_terminal_transition;
  bool start_has_valid_edge = false;
  WheelSweepValidator validator{projection, capability};

  const auto update_goal = [&](
                               const std::size_t parent,
                               std::optional<WheelTransition> terminal,
                               const double cost) {
    if (!std::isfinite(cost)) {
      return;
    }
    if (!best_goal_cost.has_value() ||
        cost + kComparisonTolerance < *best_goal_cost) {
      best_goal_cost = cost;
      best_goal_parent = parent;
      best_terminal_transition = std::move(terminal);
    }
  };

  while (!open.empty()) {
    if (stop_token.stop_requested()) {
      return Failure(
          WheelLatticeStatus::kCanceled, "REQUEST_CANCELED",
          expanded_states);
    }
    if (best_goal_cost.has_value() &&
        open.begin()->estimated_total_cost + kComparisonTolerance >=
            *best_goal_cost) {
      break;
    }
    const OpenEntry current_entry = *open.begin();
    open.erase(open.begin());
    open_by_node[current_entry.node_index].reset();
    SearchNode& current = nodes[current_entry.node_index];
    if (current.closed ||
        std::abs(current.path_cost - current_entry.path_cost) >
            kComparisonTolerance) {
      continue;
    }
    current.closed = true;
    ++expanded_states;
    const SearchNode current_snapshot = current;

    if (std::holds_alternative<PlanarRegionGoal>(goal.target) &&
        GoalContainsPose(goal, current_snapshot.pose)) {
      update_goal(
          current_entry.node_index, std::nullopt,
          current_snapshot.path_cost);
    }
    if (auto terminal = ExactGoalConnector(
            current_snapshot, goal, maximum_connector_length_m, projection,
            capability, config, validator, stop_token);
        terminal.has_value()) {
      if (current_entry.node_index == 0U) {
        start_has_valid_edge = true;
      }
      const double terminal_cost = EdgeCost(
          *terminal, projection, capability);
      update_goal(
          current_entry.node_index, std::move(terminal),
          current_snapshot.path_cost + terminal_cost);
    }

    for (const OrderedPrimitive& ordered : ordered_primitives) {
      auto transition = ApplyPrimitive(
          current_snapshot.key, current_snapshot.pose, *ordered.primitive,
          ordered.original_index, *projection.source_map(),
          config.wheel.yaw_bin_count);
      if (!transition.has_value()) {
        continue;
      }
      const WheelLatticeState target_key = PoseKey(
          transition->target_pose, transition->target_mode,
          *projection.source_map(), config.wheel.yaw_bin_count);
      const shared::GridCell target_cell{
          .x = target_key.cell_x,
          .y = target_key.cell_y,
      };
      if (!projection.HardFeasible(target_cell)) {
        continue;
      }
      const WheelSweepValidation sweep =
          validator.Validate(*transition, stop_token);
      if (sweep.canceled) {
        return Failure(
            WheelLatticeStatus::kCanceled, "REQUEST_CANCELED",
            expanded_states);
      }
      if (!sweep.valid) {
        continue;
      }
      if (current_entry.node_index == 0U) {
        start_has_valid_edge = true;
      }
      transition->surface_slope_rad = sweep.maximum_surface_slope_rad;
      transition->roughness_m = sweep.maximum_roughness_m;
      const double edge_cost = EdgeCost(*transition, projection, capability);
      if (!std::isfinite(edge_cost) || edge_cost <= 0.0) {
        return Failure(
            WheelLatticeStatus::kInvalidRequest,
            "WHEEL_LATTICE_EDGE_COST_INVALID", expanded_states);
      }
      const double candidate_cost = current_snapshot.path_cost + edge_cost;
      std::size_t target_index{};
      const auto found = node_by_key.find(target_key);
      if (found == node_by_key.end()) {
        target_index = nodes.size();
        nodes.push_back(SearchNode{
            .key = target_key,
            .pose = transition->target_pose,
        });
        node_by_key.emplace(target_key, target_index);
        open_by_node.emplace_back();
      } else {
        target_index = found->second;
      }
      SearchNode& target = nodes[target_index];
      if (candidate_cost + kComparisonTolerance >= target.path_cost) {
        continue;
      }
      if (open_by_node[target_index].has_value()) {
        open.erase(*open_by_node[target_index]);
        open_by_node[target_index].reset();
      }
      target.pose = transition->target_pose;
      target.path_cost = candidate_cost;
      target.parent = current_entry.node_index;
      target.incoming = std::move(transition);
      target.open_sequence = next_sequence++;
      target.closed = false;
      const OpenEntry entry{
          .estimated_total_cost = candidate_cost + heuristic(target.pose),
          .path_cost = candidate_cost,
          .key = target.key,
          .node_index = target_index,
          .sequence = target.open_sequence,
      };
      open.insert(entry);
      open_by_node[target_index] = entry;
    }
  }

  if (!best_goal_cost.has_value() || best_goal_parent == kNoParent) {
    if (!start_has_valid_edge) {
      return Failure(
          WheelLatticeStatus::kInvalidRequest,
          "WHEEL_START_CONNECTOR_INFEASIBLE", expanded_states);
    }
    return Failure(
        WheelLatticeStatus::kNoPath,
        "WHEEL_NO_KNOWN_SAFE_ROUTE", expanded_states);
  }
  const auto plan = Reconstruct(
      nodes, best_goal_parent, best_terminal_transition,
      *best_goal_cost, expanded_states);
  if (!plan.has_value()) {
    return Failure(
        WheelLatticeStatus::kInvalidRequest,
        "WHEEL_SEARCH_RESULT_INVALID", expanded_states);
  }
  return WheelLatticeSearchResult{
      .status = WheelLatticeStatus::kSolved,
      .plan = std::move(plan),
      .reason_code = {},
  };
} catch (const std::bad_alloc&) {
  return Failure(
      WheelLatticeStatus::kResourceExhausted,
      "WHEEL_SEARCH_ALLOCATION_FAILURE");
}

}  // namespace lunar::planning::wheel
