#include "wheel/wheel_primitive_expansion.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numbers>
#include <optional>
#include <queue>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "wheel/wheel_sweep_validator.hpp"

namespace lunar::planning::wheel {
namespace {

constexpr double kComparisonTolerance = 1.0e-9;
constexpr double kWheelRoughnessReferenceM = 0.1595;
constexpr std::size_t kNoNode = std::numeric_limits<std::size_t>::max();

[[nodiscard]] bool Finite(const Vec3& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
      std::isfinite(value.z);
}

[[nodiscard]] bool Finite(const Pose3& pose) noexcept {
  return Finite(pose.position_m) &&
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

[[nodiscard]] std::optional<std::size_t> DenseStateIndex(
    const WheelLatticeState& state, const std::size_t width,
    const std::size_t height, const std::size_t yaw_bin_count) noexcept {
  if (state.cell_x < 0 || state.cell_y < 0 || state.yaw_bin < 0 ||
      static_cast<std::size_t>(state.cell_x) >= width ||
      static_cast<std::size_t>(state.cell_y) >= height ||
      static_cast<std::size_t>(state.yaw_bin) >= yaw_bin_count) {
    return std::nullopt;
  }
  constexpr std::size_t kModeCount = 3U;
  const auto mode = static_cast<std::size_t>(state.motion_mode);
  if (mode >= kModeCount) {
    return std::nullopt;
  }
  return (((static_cast<std::size_t>(state.cell_y) * width +
            static_cast<std::size_t>(state.cell_x)) * yaw_bin_count +
           static_cast<std::size_t>(state.yaw_bin)) * kModeCount + mode);
}

void AppendUint32(
    std::vector<std::uint8_t>& output, const std::uint32_t value) {
  for (std::size_t index = 0U; index < 4U; ++index) {
    output.push_back(static_cast<std::uint8_t>(value >> (index * 8U)));
  }
}

void AppendUint64(
    std::vector<std::uint8_t>& output, const std::uint64_t value) {
  for (std::size_t index = 0U; index < 8U; ++index) {
    output.push_back(static_cast<std::uint8_t>(value >> (index * 8U)));
  }
}

void AppendDouble(std::vector<std::uint8_t>& output, double value) {
  if (value == 0.0) {
    value = 0.0;
  }
  AppendUint64(output, std::bit_cast<std::uint64_t>(value));
}

void AppendString(
    std::vector<std::uint8_t>& output, const std::string& value) {
  AppendUint64(output, static_cast<std::uint64_t>(value.size()));
  output.insert(output.end(), value.begin(), value.end());
}

[[nodiscard]] std::vector<std::uint8_t> PrimitiveSetBytes(
    const WheeledCapability& capability) {
  std::vector<std::uint8_t> output;
  AppendString(output, "wheel-motion-primitives/v1");
  AppendUint64(output,
               static_cast<std::uint64_t>(capability.motion_primitives.size()));
  for (const WheelMotionPrimitive& primitive : capability.motion_primitives) {
    AppendString(output, primitive.primitive_id);
    output.push_back(static_cast<std::uint8_t>(primitive.kind));
    AppendDouble(output, primitive.relative_end_pose.position_m.x);
    AppendDouble(output, primitive.relative_end_pose.position_m.y);
    AppendDouble(output, primitive.relative_end_pose.position_m.z);
    AppendDouble(output, primitive.relative_end_pose.orientation.w);
    AppendDouble(output, primitive.relative_end_pose.orientation.x);
    AppendDouble(output, primitive.relative_end_pose.orientation.y);
    AppendDouble(output, primitive.relative_end_pose.orientation.z);
  }
  return output;
}

struct OrderedPrimitive final {
  std::size_t original_index{};
  const WheelMotionPrimitive* primitive{};
};

struct GraphNode final {
  WheelLatticeState key;
  WheelPose pose;
  double path_cost{std::numeric_limits<double>::infinity()};
  std::size_t sequence{};
  bool closed{};
};

struct OpenEntry final {
  double path_cost{};
  WheelLatticeState key;
  std::size_t node_index{};
  std::size_t sequence{};
};

struct OpenGreater final {
  bool operator()(const OpenEntry& lhs, const OpenEntry& rhs) const noexcept {
    return std::tie(lhs.path_cost, lhs.key, lhs.sequence, lhs.node_index) >
        std::tie(rhs.path_cost, rhs.key, rhs.sequence, rhs.node_index);
  }
};

struct PendingEdge final {
  std::size_t source{};
  std::size_t target{};
  std::size_t source_sequence{};
  std::size_t primitive_index{};
  std::string primitive_id;
  double cost{};
};

[[nodiscard]] shared::PrimitiveGraphBuildResult Failure(
    std::string reason_code) {
  return shared::PrimitiveGraphBuildResult{
      .reason_code = std::move(reason_code),
  };
}

}  // namespace

bool ValidateWheelPrimitiveCapability(
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
        return !primitive.primitive_id.empty() && Finite(primitive.relative_end_pose) &&
            YawFromQuaternion(primitive.relative_end_pose.orientation).has_value();
      });
}

WheelLatticeState WheelPrimitivePoseKey(
    const WheelPose& pose, const WheelMotionMode mode,
    const shared::MapSnapshot& map,
    const std::size_t yaw_bin_count) noexcept {
  const auto cell = map.PositionToCell(
      Vec2{.x = pose.position_m.x, .y = pose.position_m.y});
  return WheelLatticeState{
      .cell_x = cell.has_value() ? cell->x : -1,
      .cell_y = cell.has_value() ? cell->y : -1,
      .yaw_bin = YawToBin(pose.yaw_rad, yaw_bin_count),
      .motion_mode = mode,
  };
}

std::optional<WheelTransition> ApplyWheelPrimitiveKinematics(
    const WheelLatticeState& source_state, const WheelPose& source,
    const WheelMotionPrimitive& primitive,
    const std::size_t primitive_index, const shared::MapSnapshot& map,
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
  const double path_length_m = ArcLength(primitive.kind, chord_m, yaw_delta);
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

double WheelPrimitiveEdgeCost(
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
           std::sqrt(capability.maximum_lateral_acceleration_mps2 / curvature)});
    }
    const double ratio = transition.roughness_m / kWheelRoughnessReferenceM;
    speed_limit *= std::max(0.0, std::cos(transition.surface_slope_rad)) /
        (1.0 + ratio * ratio);
    cost = transition.path_length_m / speed_limit;
  } else if (yaw_distance > kComparisonTolerance) {
    cost = yaw_distance / capability.maximum_spin_rate_radps;
  } else {
    cost = 1.0e-3;
  }
  const auto target_cell = projection.source_map()->PositionToCell(
      Vec2{.x = transition.target_pose.position_m.x,
           .y = transition.target_pose.position_m.y});
  if (target_cell.has_value() &&
      transition.path_length_m > kComparisonTolerance) {
    cost += 0.6 * static_cast<double>(projection.TraversalCost(*target_cell));
  }
  return cost;
}

double WheelPrimitiveCostLowerBound(
    const WheelTransition& transition,
    const WheeledCapability& capability) noexcept {
  const double yaw_distance = std::abs(ShortestYawDelta(
      transition.source_pose.yaw_rad, transition.target_pose.yaw_rad));
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
    return transition.path_length_m / speed_limit;
  }
  if (yaw_distance > kComparisonTolerance) {
    return yaw_distance / capability.maximum_spin_rate_radps;
  }
  return 1.0e-3;
}

bool ExistingTargetDominates(
    const double source_cost, const double edge_cost_lower_bound,
    const double existing_target_cost) noexcept {
  return std::isfinite(source_cost) &&
      std::isfinite(edge_cost_lower_bound) && edge_cost_lower_bound > 0.0 &&
      std::isfinite(existing_target_cost) &&
      source_cost + edge_cost_lower_bound + kComparisonTolerance >=
          existing_target_cost;
}

shared::PrimitiveGraphBuildResult BuildWheelPrimitiveGraph(
    const WheeledState& current_state,
    const shared::SafeProjection& projection,
    const WheeledCapability& capability, const PlannerConfig& config,
    const std::stop_token stop_token) try {
  if (stop_token.stop_requested()) {
    return Failure("REQUEST_CANCELED");
  }
  if (projection.source_map() == nullptr ||
      !ValidateWheelPrimitiveCapability(capability, config) ||
      !Finite(current_state.pose)) {
    return Failure("WHEEL_PRIMITIVE_GRAPH_REQUEST_INVALID");
  }
  const auto start_yaw = YawFromQuaternion(current_state.pose.orientation);
  const auto start_cell = projection.source_map()->PositionToCell(
      Vec2{.x = current_state.pose.position_m.x,
           .y = current_state.pose.position_m.y});
  if (!start_yaw.has_value() || !start_cell.has_value() ||
      !projection.HardFeasible(*start_cell)) {
    return Failure("WHEEL_START_NOT_SAFE");
  }

  const std::size_t width = projection.source_map()->width();
  const std::size_t height = projection.source_map()->height();
  constexpr std::size_t kModeCount = 3U;
  if (height != 0U && width > std::numeric_limits<std::size_t>::max() / height) {
    return Failure("WHEEL_SEARCH_STATE_SPACE_OVERFLOW");
  }
  const std::size_t cell_count = width * height;
  if (config.wheel.yaw_bin_count != 0U &&
      cell_count > std::numeric_limits<std::size_t>::max() /
                       config.wheel.yaw_bin_count) {
    return Failure("WHEEL_SEARCH_STATE_SPACE_OVERFLOW");
  }
  const std::size_t oriented_count =
      cell_count * config.wheel.yaw_bin_count;
  if (oriented_count > std::numeric_limits<std::size_t>::max() / kModeCount) {
    return Failure("WHEEL_SEARCH_STATE_SPACE_OVERFLOW");
  }

  std::vector<OrderedPrimitive> primitives;
  primitives.reserve(capability.motion_primitives.size());
  for (std::size_t index = 0U; index < capability.motion_primitives.size();
       ++index) {
    primitives.push_back(
        OrderedPrimitive{index, &capability.motion_primitives[index]});
  }
  std::stable_sort(
      primitives.begin(), primitives.end(),
      [](const OrderedPrimitive& lhs, const OrderedPrimitive& rhs) {
        return std::tie(lhs.primitive->primitive_id, lhs.original_index) <
            std::tie(rhs.primitive->primitive_id, rhs.original_index);
      });

  const WheelPose start_pose{
      .position_m = current_state.pose.position_m,
      .yaw_rad = *start_yaw,
  };
  const WheelLatticeState start_key{
      .cell_x = start_cell->x,
      .cell_y = start_cell->y,
      .yaw_bin = YawToBin(*start_yaw, config.wheel.yaw_bin_count),
      .motion_mode = WheelMotionMode::kStart,
  };
  std::vector<GraphNode> nodes{
      GraphNode{.key = start_key, .pose = start_pose, .path_cost = 0.0}};
  std::vector<std::size_t> node_by_state(
      oriented_count * kModeCount, kNoNode);
  const auto dense_start = DenseStateIndex(
      start_key, width, height, config.wheel.yaw_bin_count);
  if (!dense_start.has_value()) {
    return Failure("WHEEL_START_STATE_INVALID");
  }
  node_by_state[*dense_start] = 0U;
  std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenGreater> open;
  open.push(OpenEntry{.path_cost = 0.0, .key = start_key});
  std::size_t next_sequence = 1U;
  std::vector<PendingEdge> pending_edges;
  WheelSweepValidator validator{projection, capability};

  while (!open.empty()) {
    if (stop_token.stop_requested()) {
      return Failure("REQUEST_CANCELED");
    }
    const OpenEntry entry = open.top();
    open.pop();
    GraphNode& current = nodes[entry.node_index];
    if (current.closed || current.sequence != entry.sequence ||
        std::abs(current.path_cost - entry.path_cost) >
            kComparisonTolerance) {
      continue;
    }
    current.closed = true;
    const WheelLatticeState source_key = current.key;
    const WheelPose source_pose = current.pose;
    const double source_cost = current.path_cost;
    for (const OrderedPrimitive& ordered : primitives) {
      auto transition = ApplyWheelPrimitiveKinematics(
          source_key, source_pose, *ordered.primitive,
          ordered.original_index, *projection.source_map(),
          config.wheel.yaw_bin_count);
      if (!transition.has_value()) {
        continue;
      }
      const WheelLatticeState target_key = WheelPrimitivePoseKey(
          transition->target_pose, transition->target_mode,
          *projection.source_map(), config.wheel.yaw_bin_count);
      const shared::GridCell target_cell{
          .x = target_key.cell_x, .y = target_key.cell_y};
      if (!projection.HardFeasible(target_cell)) {
        continue;
      }
      const WheelSweepValidation sweep = validator.Validate(*transition, stop_token);
      if (sweep.canceled) {
        return Failure("REQUEST_CANCELED");
      }
      if (!sweep.valid) {
        continue;
      }
      transition->surface_slope_rad = sweep.maximum_surface_slope_rad;
      transition->roughness_m = sweep.maximum_roughness_m;
      const double edge_cost =
          WheelPrimitiveEdgeCost(*transition, projection, capability);
      if (!std::isfinite(edge_cost) || edge_cost <= 0.0) {
        return Failure("WHEEL_LATTICE_EDGE_COST_INVALID");
      }
      const auto dense_target = DenseStateIndex(
          target_key, width, height, config.wheel.yaw_bin_count);
      if (!dense_target.has_value()) {
        return Failure("WHEEL_LATTICE_EDGE_STATE_INVALID");
      }
      std::size_t target_index = node_by_state[*dense_target];
      if (target_index == kNoNode) {
        target_index = nodes.size();
        nodes.push_back(GraphNode{.key = target_key, .pose = transition->target_pose});
        node_by_state[*dense_target] = target_index;
      }
      pending_edges.push_back(PendingEdge{
          .source = entry.node_index,
          .target = target_index,
          .source_sequence = entry.sequence,
          .primitive_index = ordered.original_index,
          .primitive_id = ordered.primitive->primitive_id,
          .cost = edge_cost,
      });
      const double candidate_cost = source_cost + edge_cost;
      GraphNode& target = nodes[target_index];
      if (candidate_cost + kComparisonTolerance < target.path_cost) {
        target.pose = transition->target_pose;
        target.path_cost = candidate_cost;
        target.closed = false;
        target.sequence = next_sequence++;
        open.push(OpenEntry{
            .path_cost = candidate_cost,
            .key = target.key,
            .node_index = target_index,
            .sequence = target.sequence,
        });
      }
    }
  }

  shared::PrimitiveGraphBuildResult graph{
      .platform_type = PlatformType::kWheeled,
      .width = width,
      .height = height,
      .anchor_state_index = 0U,
      .algorithm_id = "cpp-wheel-motion-primitive-recoverable-graph/v1",
      .state_schema = "wheel-lattice-state/v1",
      .primitive_set_canonical_bytes = PrimitiveSetBytes(capability),
  };
  graph.states.reserve(nodes.size());
  for (const GraphNode& node : nodes) {
    graph.states.push_back(PrimitiveReachabilityState{
        .position_m = node.pose.position_m,
        .yaw_rad = node.pose.yaw_rad,
        .cell_x = node.key.cell_x,
        .cell_y = node.key.cell_y,
        .yaw_bin = node.key.yaw_bin,
        .motion_mode = static_cast<std::int32_t>(node.key.motion_mode),
        .body_z_m = Interval{},
        .path_cost = node.path_cost,
        .observation_state = 1U,
    });
  }
  for (const PendingEdge& edge : pending_edges) {
    if (edge.source_sequence == nodes[edge.source].sequence) {
      graph.potential_edges.push_back(shared::PrimitiveGraphPotentialEdge{
          .source_state_index = edge.source,
          .target_state_index = edge.target,
          .primitive_index = static_cast<std::uint32_t>(edge.primitive_index),
          .primitive_id = edge.primitive_id,
          .cost = edge.cost,
          .certified = true,
      });
    }
  }
  return graph;
} catch (const std::bad_alloc&) {
  return Failure("WHEEL_SEARCH_ALLOCATION_FAILURE");
}

}  // namespace lunar::planning::wheel
