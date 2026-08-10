#include "wheel/wheel_lattice.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numbers>
#include <optional>
#include <queue>
#include <ranges>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "wheel/wheel_sweep_validator.hpp"
#include "wheel/wheel_primitive_expansion.hpp"

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

struct ExactConnector final {
  std::vector<WheelTransition> transitions;
  double cost{};
};

struct OpenGreater final {
  [[nodiscard]] bool operator()(
      const OpenEntry& lhs, const OpenEntry& rhs) const noexcept {
    return std::tie(
               lhs.estimated_total_cost, lhs.path_cost, lhs.key,
               lhs.sequence, lhs.node_index) >
        std::tie(
               rhs.estimated_total_cost, rhs.path_cost, rhs.key,
               rhs.sequence, rhs.node_index);
  }
};

[[nodiscard]] std::optional<std::size_t> DenseStateIndex(
    const WheelLatticeState& state, const std::size_t width,
    const std::size_t height, const std::size_t yaw_bin_count) noexcept {
  if (state.cell_x < 0 || state.cell_y < 0 || state.yaw_bin < 0 ||
      static_cast<std::size_t>(state.cell_x) >= width ||
      static_cast<std::size_t>(state.cell_y) >= height ||
      static_cast<std::size_t>(state.yaw_bin) >= yaw_bin_count) {
    return std::nullopt;
  }
  constexpr std::size_t kMotionModeCount = 3U;
  const auto mode = static_cast<std::size_t>(state.motion_mode);
  if (mode >= kMotionModeCount) {
    return std::nullopt;
  }
  return (((static_cast<std::size_t>(state.cell_y) * width +
            static_cast<std::size_t>(state.cell_x)) *
               yaw_bin_count +
           static_cast<std::size_t>(state.yaw_bin)) *
              kMotionModeCount +
          mode);
}

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
  return WheelPrimitivePoseKey(pose, mode, map, yaw_bin_count);
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
  return ApplyWheelPrimitiveKinematics(
      source_state, source, primitive, primitive_index, map, yaw_bin_count);
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
  return ValidateWheelPrimitiveCapability(capability, config);
}

[[nodiscard]] double EdgeCost(
    const WheelTransition& transition,
    const shared::SafeProjection& projection,
    const WheeledCapability& capability) noexcept {
  return WheelPrimitiveEdgeCost(transition, projection, capability);
}

[[nodiscard]] std::optional<ExactConnector> ExactGoalConnector(
    const SearchNode& source, const GoalRegion& goal,
    const double maximum_connector_length_m,
    const shared::SafeProjection& projection,
    const WheeledCapability& capability,
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
  const auto target_cell = projection.source_map()->PositionToCell(Vec2{
      .x = point->position_m.x,
      .y = point->position_m.y,
  });
  if (!target_cell.has_value() ||
      !projection.HardFeasible(*target_cell)) {
    return std::nullopt;
  }

  const auto supports = [&](const WheelPrimitiveKind kind) {
    return std::ranges::any_of(
        capability.motion_primitives,
        [kind](const WheelMotionPrimitive& primitive) {
          return primitive.kind == kind;
        });
  };
  const auto build = [&](const bool reverse)
      -> std::optional<ExactConnector> {
    ExactConnector connector;
    WheelPose pose = source.pose;
    WheelMotionMode mode = source.key.motion_mode;
    auto append = [&](WheelTransition transition) {
      const WheelSweepValidation sweep =
          validator.Validate(transition, stop_token);
      if (!sweep.valid) {
        return false;
      }
      transition.surface_slope_rad = sweep.maximum_surface_slope_rad;
      transition.roughness_m = sweep.maximum_roughness_m;
      const double edge_cost = EdgeCost(transition, projection, capability);
      if (!std::isfinite(edge_cost) || edge_cost <= 0.0) {
        return false;
      }
      connector.cost += edge_cost;
      pose = transition.target_pose;
      mode = transition.target_mode;
      connector.transitions.push_back(std::move(transition));
      return true;
    };
    auto append_spin = [&](const double requested_yaw) {
      const double target_yaw = NormalizeYaw(requested_yaw);
      const double delta = ShortestYawDelta(pose.yaw_rad, target_yaw);
      if (std::abs(delta) <= kComparisonTolerance) {
        return true;
      }
      const WheelPrimitiveKind kind = delta >= 0.0
          ? WheelPrimitiveKind::kSpinCounterclockwise
          : WheelPrimitiveKind::kSpinClockwise;
      return supports(kind) && append(WheelTransition{
          .source_pose = pose,
          .target_pose = WheelPose{
              .position_m = pose.position_m,
              .yaw_rad = target_yaw,
          },
          .curvature_per_m = 0.0,
          .primitive_index = capability.motion_primitives.size(),
          .primitive_kind = kind,
          .source_mode = mode,
          .target_mode = WheelMotionMode::kStart,
          .path_length_m = 0.0,
          .reverse = false,
      });
    };
    auto append_stop = [&]() {
      if (mode == WheelMotionMode::kStart) {
        return true;
      }
      return supports(WheelPrimitiveKind::kStopAndSwitch) &&
          append(WheelTransition{
              .source_pose = pose,
              .target_pose = pose,
              .curvature_per_m = 0.0,
              .primitive_index = capability.motion_primitives.size(),
              .primitive_kind = WheelPrimitiveKind::kStopAndSwitch,
              .source_mode = mode,
              .target_mode = WheelMotionMode::kStart,
              .path_length_m = 0.0,
              .reverse = false,
          });
    };

    if (distance <= kComparisonTolerance) {
      if (!goal.yaw_rad.has_value() || !append_spin(*goal.yaw_rad)) {
        return std::nullopt;
      }
      return connector.transitions.empty()
          ? std::nullopt
          : std::optional<ExactConnector>{std::move(connector)};
    }

    const double travel_heading = std::atan2(dy, dx);
    const double body_heading = reverse
        ? NormalizeYaw(travel_heading + std::numbers::pi)
        : travel_heading;
    const WheelPrimitiveKind translation_kind = reverse
        ? WheelPrimitiveKind::kReverse
        : WheelPrimitiveKind::kForward;
    const WheelMotionMode translation_mode = reverse
        ? WheelMotionMode::kReverse
        : WheelMotionMode::kForward;
    if (!supports(translation_kind) || !append_spin(body_heading)) {
      return std::nullopt;
    }
    if (mode != WheelMotionMode::kStart && mode != translation_mode &&
        !append_stop()) {
      return std::nullopt;
    }
    if (!append(WheelTransition{
            .source_pose = pose,
            .target_pose = WheelPose{
                .position_m = point->position_m,
                .yaw_rad = NormalizeYaw(body_heading),
            },
            .curvature_per_m = 0.0,
            .primitive_index = capability.motion_primitives.size(),
            .primitive_kind = translation_kind,
            .source_mode = mode,
            .target_mode = translation_mode,
            .path_length_m = distance,
            .reverse = reverse,
        })) {
      return std::nullopt;
    }
    if (goal.yaw_rad.has_value() && !append_spin(*goal.yaw_rad)) {
      return std::nullopt;
    }
    return connector;
  };

  std::optional<ExactConnector> best;
  for (const bool reverse : {false, true}) {
    auto candidate = build(reverse);
    if (candidate.has_value() &&
        (!best.has_value() ||
         candidate->cost + kComparisonTolerance < best->cost)) {
      best = std::move(candidate);
    }
  }
  return best;
}

[[nodiscard]] std::optional<WheelDiscretePlan> Reconstruct(
    const std::vector<SearchNode>& nodes, std::size_t terminal_parent,
    const std::vector<WheelTransition>& terminal_transitions,
    const double cost, const std::size_t expanded_states) {
  std::vector<WheelTransition> reversed;
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
  reversed.insert(
      reversed.end(), terminal_transitions.begin(), terminal_transitions.end());
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

WheelLatticeSearchResult SearchWheelLatticeRanked(
    const WheeledState& current_state,
    const std::span<const GoalRegion> ranked_goals,
    const shared::SafeProjection& projection,
    const WheeledCapability& capability, const PlannerConfig& config,
    const std::stop_token stop_token) try {
  if (stop_token.stop_requested()) {
    return Failure(WheelLatticeStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (ranked_goals.empty() || projection.source_map() == nullptr ||
      !ValidateCapability(capability, config) ||
      !IsFinite(current_state.pose)) {
    return Failure(
        WheelLatticeStatus::kInvalidRequest,
        "WHEEL_LATTICE_REQUEST_INVALID");
  }
  const GoalRegion& goal = ranked_goals.front();
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
        .selected_goal_index = 0U,
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
  // Keep the exact connector long enough to bridge the terminal lattice gap.
  // A corridor-masked local view can legitimately leave only its centerline
  // feasible, so requiring an extra lattice state before connecting creates a
  // false no-route result near an otherwise safe point goal.  Two primitive
  // lengths plus half a cell diagonal cover that discretization gap.  The
  // synthetic connector is decomposed into supported in-place spins and one
  // body-aligned translation, with every component sweep-validated below.
  maximum_connector_length_m =
      2.0 * maximum_connector_length_m +
      std::numbers::sqrt2 * 0.5 * projection.source_map()->resolution_m();
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
  constexpr std::size_t kMotionModeCount = 3U;
  const std::size_t width = projection.source_map()->width();
  const std::size_t height = projection.source_map()->height();
  if (height != 0U && width > std::numeric_limits<std::size_t>::max() / height) {
    return Failure(
        WheelLatticeStatus::kResourceExhausted,
        "WHEEL_SEARCH_STATE_SPACE_OVERFLOW");
  }
  const std::size_t cell_count = width * height;
  if (config.wheel.yaw_bin_count != 0U &&
      cell_count > std::numeric_limits<std::size_t>::max() /
                       config.wheel.yaw_bin_count) {
    return Failure(
        WheelLatticeStatus::kResourceExhausted,
        "WHEEL_SEARCH_STATE_SPACE_OVERFLOW");
  }
  const std::size_t oriented_cell_count =
      cell_count * config.wheel.yaw_bin_count;
  if (oriented_cell_count >
      std::numeric_limits<std::size_t>::max() / kMotionModeCount) {
    return Failure(
        WheelLatticeStatus::kResourceExhausted,
        "WHEEL_SEARCH_STATE_SPACE_OVERFLOW");
  }
  std::vector<std::size_t> node_by_state(
      oriented_cell_count * kMotionModeCount, kNoParent);
  const auto start_state_index = DenseStateIndex(
      start_key, width, height, config.wheel.yaw_bin_count);
  if (!start_state_index.has_value()) {
    return Failure(
        WheelLatticeStatus::kInvalidRequest,
        "WHEEL_START_STATE_INVALID");
  }
  node_by_state[*start_state_index] = 0U;
  std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenGreater> open;
  const OpenEntry start_open{
      .estimated_total_cost = heuristic(true_start),
      .path_cost = 0.0,
      .key = start_key,
      .node_index = 0U,
      .sequence = 0U,
  };
  open.push(start_open);
  std::size_t next_sequence = 1U;
  std::size_t expanded_states = 0U;
  std::optional<double> best_goal_cost;
  std::size_t best_goal_parent = kNoParent;
  std::vector<WheelTransition> best_terminal_transitions;
  bool start_has_valid_edge = false;
  WheelSweepValidator validator{projection, capability};

  const auto update_goal = [&](
                               const std::size_t parent,
                               std::vector<WheelTransition> terminal,
                               const double cost) {
    if (!std::isfinite(cost)) {
      return;
    }
    if (!best_goal_cost.has_value() ||
        cost + kComparisonTolerance < *best_goal_cost) {
      best_goal_cost = cost;
      best_goal_parent = parent;
      best_terminal_transitions = std::move(terminal);
    }
  };

  while (!open.empty()) {
    if (stop_token.stop_requested()) {
      return Failure(
          WheelLatticeStatus::kCanceled, "REQUEST_CANCELED",
          expanded_states);
    }
    if (best_goal_cost.has_value() &&
        open.top().estimated_total_cost + kComparisonTolerance >=
            *best_goal_cost) {
      break;
    }
    const OpenEntry current_entry = open.top();
    open.pop();
    SearchNode& current = nodes[current_entry.node_index];
    if (current.closed ||
        std::abs(current.path_cost - current_entry.path_cost) >
            kComparisonTolerance ||
        current.open_sequence != current_entry.sequence) {
      continue;
    }
    current.closed = true;
    ++expanded_states;
    const SearchNode current_snapshot = current;

    if (std::holds_alternative<PlanarRegionGoal>(goal.target) &&
        GoalContainsPose(goal, current_snapshot.pose)) {
      update_goal(
          current_entry.node_index, {},
          current_snapshot.path_cost);
    }
    if (auto terminal = ExactGoalConnector(
            current_snapshot, goal, maximum_connector_length_m, projection,
            capability, validator, stop_token);
        terminal.has_value()) {
      if (current_entry.node_index == 0U) {
        start_has_valid_edge = true;
      }
      update_goal(
          current_entry.node_index, std::move(terminal->transitions),
          current_snapshot.path_cost + terminal->cost);
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
      const auto dense_target_index = DenseStateIndex(
          target_key, width, height, config.wheel.yaw_bin_count);
      if (!dense_target_index.has_value()) {
        return Failure(
            WheelLatticeStatus::kInvalidRequest,
            "WHEEL_LATTICE_EDGE_STATE_INVALID", expanded_states);
      }
      const std::size_t found = node_by_state[*dense_target_index];
      if (found == kNoParent) {
        target_index = nodes.size();
        nodes.push_back(SearchNode{
            .key = target_key,
            .pose = transition->target_pose,
        });
        node_by_state[*dense_target_index] = target_index;
      } else {
        target_index = found;
      }
      SearchNode& target = nodes[target_index];
      if (candidate_cost + kComparisonTolerance >= target.path_cost) {
        continue;
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
      open.push(entry);
    }
  }

  std::size_t selected_goal_index = 0U;
  if (!best_goal_cost.has_value() && start_has_valid_edge) {
    for (std::size_t goal_index = 1U;
         goal_index < ranked_goals.size() && !best_goal_cost.has_value();
         ++goal_index) {
      const GoalRegion& fallback_goal = ranked_goals[goal_index];
      for (std::size_t node_index = 0U; node_index < nodes.size();
           ++node_index) {
        if (stop_token.stop_requested()) {
          return Failure(
              WheelLatticeStatus::kCanceled, "REQUEST_CANCELED",
              expanded_states);
        }
        const SearchNode& node = nodes[node_index];
        if (!std::isfinite(node.path_cost)) {
          continue;
        }
        if (std::holds_alternative<PlanarRegionGoal>(fallback_goal.target) &&
            GoalContainsPose(fallback_goal, node.pose)) {
          update_goal(node_index, {}, node.path_cost);
          continue;
        }
        if (auto terminal = ExactGoalConnector(
                node, fallback_goal, maximum_connector_length_m, projection,
                capability, validator, stop_token);
            terminal.has_value()) {
          update_goal(
              node_index, std::move(terminal->transitions),
              node.path_cost + terminal->cost);
        }
      }
      if (best_goal_cost.has_value()) {
        selected_goal_index = goal_index;
      }
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
      nodes, best_goal_parent, best_terminal_transitions,
      *best_goal_cost, expanded_states);
  if (!plan.has_value()) {
    return Failure(
        WheelLatticeStatus::kInvalidRequest,
        "WHEEL_SEARCH_RESULT_INVALID", expanded_states);
  }
  return WheelLatticeSearchResult{
      .status = WheelLatticeStatus::kSolved,
      .plan = std::move(plan),
      .selected_goal_index = selected_goal_index,
      .reason_code = {},
  };
} catch (const std::bad_alloc&) {
  return Failure(
      WheelLatticeStatus::kResourceExhausted,
      "WHEEL_SEARCH_ALLOCATION_FAILURE");
}

WheelLatticeSearchResult SearchWheelLattice(
    const WheeledState& current_state, const GoalRegion& goal,
    const shared::SafeProjection& projection,
    const WheeledCapability& capability, const PlannerConfig& config,
    const std::stop_token stop_token) {
  return SearchWheelLatticeRanked(
      current_state, std::span<const GoalRegion>{&goal, 1U}, projection,
      capability, config, stop_token);
}

}  // namespace lunar::planning::wheel
