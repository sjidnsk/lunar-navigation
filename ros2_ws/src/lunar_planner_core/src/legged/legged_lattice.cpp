#include "legged/legged_lattice.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <new>
#include <functional>
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

constexpr std::size_t kNoParent = std::numeric_limits<std::size_t>::max();

struct LazySearchNode final {
  LeggedLatticeState state;
  LeggedPose pose;
  double path_cost{std::numeric_limits<double>::infinity()};
  std::size_t parent{kNoParent};
  std::optional<LeggedTransition> incoming;
  std::size_t open_sequence{};
  bool closed{};
};

struct LazyOpenEntry final {
  double estimated_total_cost{};
  double path_cost{};
  LeggedStateKey key;
  std::size_t node_index{};
  std::size_t sequence{};
};

struct LazyOpenGreater final {
  [[nodiscard]] bool operator()(
      const LazyOpenEntry& lhs, const LazyOpenEntry& rhs) const noexcept {
    return std::tie(lhs.estimated_total_cost, lhs.path_cost, lhs.key,
                    lhs.sequence, lhs.node_index) >
        std::tie(rhs.estimated_total_cost, rhs.path_cost, rhs.key,
                 rhs.sequence, rhs.node_index);
  }
};

[[nodiscard]] LeggedLatticeSearchResult SearchFailure(
    const LeggedLatticeStatus status, std::string reason_code,
    const std::size_t expanded_states = 0U) {
  return LeggedLatticeSearchResult{
      .status = status,
      .plan = status == LeggedLatticeStatus::kNoPath
          ? std::optional<LeggedDiscretePlan>{LeggedDiscretePlan{
                .transitions = {},
                .cost = std::numeric_limits<double>::infinity(),
                .expanded_states = expanded_states,
            }}
          : std::nullopt,
      .reason_code = std::move(reason_code),
  };
}

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

[[nodiscard]] bool ValidCapability(
    const LeggedCapability& capability,
    const PlannerConfig& config) noexcept {
  const auto valid_interval = [](const Interval& interval) {
    return ValidInterval(interval) && interval.lower <= 0.0 &&
           interval.upper >= 0.0 && interval.lower < interval.upper;
  };
  if (!IsFinite(capability.body_extent_m) ||
      capability.body_extent_m.x <= 0.0 ||
      capability.body_extent_m.y <= 0.0 ||
      capability.body_extent_m.z <= 0.0 ||
      !ValidInterval(capability.body_height_m) ||
      !valid_interval(capability.forward_speed_mps) ||
      !valid_interval(capability.lateral_speed_mps) ||
      !valid_interval(capability.yaw_rate_radps) ||
      !std::isfinite(capability.step_vertical_rate_mps) ||
      capability.step_vertical_rate_mps <= 0.0 ||
      capability.motion_primitives.empty() ||
      !std::isfinite(config.legged.xy_resolution_m) ||
      config.legged.xy_resolution_m <= 0.0 ||
      config.legged.yaw_bin_count < 4U) {
    return false;
  }
  return std::ranges::all_of(
      capability.motion_primitives,
      [](const LeggedBodyPrimitive& primitive) {
        return !primitive.primitive_id.empty() &&
            IsFinite(primitive.body_frame_displacement_m) &&
            std::isfinite(primitive.yaw_change_rad);
      });
}

[[nodiscard]] double TransitionDurationLowerBound(
    const LeggedTransition& transition,
    const LeggedCapability& capability) noexcept {
  const double cosine = std::cos(transition.source_pose.yaw_rad);
  const double sine = std::sin(transition.source_pose.yaw_rad);
  const double dx = transition.target_pose.position_m.x -
      transition.source_pose.position_m.x;
  const double dy = transition.target_pose.position_m.y -
      transition.source_pose.position_m.y;
  const double dz = transition.target_pose.position_m.z -
      transition.source_pose.position_m.z;
  const double forward = cosine * dx + sine * dy;
  const double lateral = -sine * dx + cosine * dy;
  const double yaw = ShortestYawDelta(
      transition.source_pose.yaw_rad, transition.target_pose.yaw_rad);
  const auto duration = [](const double displacement, const Interval limits) {
    if (std::abs(displacement) <= kComparisonTolerance) {
      return 0.0;
    }
    const double speed = displacement > 0.0 ? limits.upper : -limits.lower;
    return speed > 0.0 ? std::abs(displacement) / speed
                       : std::numeric_limits<double>::infinity();
  };
  double seconds = std::max(
      {duration(forward, capability.forward_speed_mps),
       duration(lateral, capability.lateral_speed_mps),
       std::abs(dz) / capability.step_vertical_rate_mps,
       duration(yaw, capability.yaw_rate_radps),
       std::sqrt(6.0 * transition.path_length_m /
                 capability.maximum_linear_acceleration_mps2),
       std::sqrt(6.0 * std::abs(yaw) /
                 capability.maximum_yaw_acceleration_radps2)});
  return seconds;
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

[[nodiscard]] double MaximumPrimitiveTranslation(
    const LeggedCapability& capability) noexcept {
  double maximum = 0.0;
  for (const LeggedBodyPrimitive& primitive : capability.motion_primitives) {
    maximum = std::max(
        maximum,
        std::hypot(primitive.body_frame_displacement_m.x,
                   primitive.body_frame_displacement_m.y));
  }
  return maximum;
}

[[nodiscard]] double MaximumPrimitiveYaw(
    const LeggedCapability& capability,
    const PlannerConfig& config) noexcept {
  double maximum = 2.0 * std::numbers::pi /
      static_cast<double>(config.legged.yaw_bin_count);
  for (const LeggedBodyPrimitive& primitive : capability.motion_primitives) {
    maximum = std::max(maximum, std::abs(primitive.yaw_change_rad));
  }
  return maximum;
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
    const hierarchical::LocalSearchDomain& search_domain,
    const LeggedCapability& capability,
    const PlannerConfig& config,
    const std::stop_token stop_token,
    bool& canceled, bool& search_domain_rejected,
    const LeggedTerrainGrid* const terrain_grid) {
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
      source, target, source_body_z_m, projection, capability, search_domain,
      stop_token, terrain_grid);
  if (sweep.canceled) {
    canceled = true;
    return std::nullopt;
  }
  if (!sweep.valid) {
    search_domain_rejected = search_domain_rejected ||
        sweep.reason_code == "LEGGED_BODY_SWEEP_OUTSIDE_SEARCH_DOMAIN";
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
      .path_length_m = std::hypot(
          std::hypot(
              target.position_m.x - source.position_m.x,
              target.position_m.y - source.position_m.y),
          target.position_m.z - source.position_m.z),
  };
}

[[nodiscard]] std::optional<LeggedTransition> ApplyPointGoalConnector(
    const LeggedPose& source, const Interval& source_body_z_m,
    const PointGoal& point_goal, const std::optional<double> goal_yaw,
    const double maximum_translation, const double maximum_yaw,
    const shared::SafeProjection& projection,
    const hierarchical::LocalSearchDomain& search_domain,
    const LeggedCapability& capability, const std::stop_token stop_token,
    bool& canceled, bool& search_domain_rejected,
    const LeggedTerrainGrid* const terrain_grid) {
  const double distance = std::hypot(
      point_goal.position_m.x - source.position_m.x,
      point_goal.position_m.y - source.position_m.y);
  const double target_yaw = goal_yaw.value_or(source.yaw_rad);
  const double yaw_delta =
      std::abs(ShortestYawDelta(source.yaw_rad, target_yaw));
  if (distance > maximum_translation + kComparisonTolerance ||
      yaw_delta > maximum_yaw + kComparisonTolerance ||
      (distance <= kComparisonTolerance &&
       yaw_delta <= kComparisonTolerance)) {
    return std::nullopt;
  }
  const shared::MapSnapshot& map = *projection.source_map();
  const auto target_cell = map.PositionToCell(Vec2{
      .x = point_goal.position_m.x,
      .y = point_goal.position_m.y,
  });
  if (!target_cell.has_value()) {
    return std::nullopt;
  }
  std::optional<LeggedTerrainEvaluation> direct_terrain;
  const LeggedTerrainEvaluation* terrain{};
  if (terrain_grid != nullptr && !stop_token.stop_requested()) {
    terrain = terrain_grid->Find(*target_cell);
  } else {
    direct_terrain = EvaluateLeggedTerrainCell(
        projection, capability, *target_cell, stop_token);
    terrain = &*direct_terrain;
  }
  if (terrain == nullptr || terrain->canceled) {
    canceled = true;
    return std::nullopt;
  }
  if (!terrain->hard_feasible) {
    return std::nullopt;
  }
  Vec3 certified_target = point_goal.position_m;
  const Vec3 cell_center = map.CellCenter(*target_cell);
  const double center_goal_error = std::hypot(
      cell_center.x - point_goal.position_m.x,
      cell_center.y - point_goal.position_m.y);
  const double center_translation = std::hypot(
      cell_center.x - source.position_m.x,
      cell_center.y - source.position_m.y);
  if (center_goal_error <= point_goal.tolerance_m + kComparisonTolerance &&
      center_translation <= maximum_translation + kComparisonTolerance) {
    certified_target.x = cell_center.x;
    certified_target.y = cell_center.y;
  }
  LeggedPose target{
      .position_m = certified_target,
      .yaw_rad = target_yaw,
  };
  target.position_m.z = 0.5 *
      (terrain->body_height_m.lower + terrain->body_height_m.upper);
  const LeggedSweepResult sweep = ValidateLeggedBodySweep(
      source, target, source_body_z_m, projection, capability, search_domain,
      stop_token, terrain_grid);
  if (sweep.canceled) {
    canceled = true;
    return std::nullopt;
  }
  if (!sweep.valid) {
    search_domain_rejected = search_domain_rejected ||
        sweep.reason_code == "LEGGED_BODY_SWEEP_OUTSIDE_SEARCH_DOMAIN";
    return std::nullopt;
  }
  target.position_m.z = 0.5 *
      (sweep.reachable_body_z_m.lower + sweep.reachable_body_z_m.upper);
  return LeggedTransition{
      .source_pose = source,
      .target_pose = target,
      .target_body_z_m = sweep.reachable_body_z_m,
      .primitive_index = std::numeric_limits<std::size_t>::max(),
      .primitive_kind = LeggedPrimitiveKind::kCoupled,
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
    const hierarchical::LocalSearchDomain& search_domain,
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
  if (search_domain.width() != projection.source_map()->width() ||
      search_domain.height() != projection.source_map()->height()) {
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
  if (!search_domain.Contains(*start_cell)) {
    return Failure(
        LeggedLatticeStatus::kNoPath, "LOCAL_SEARCH_DOMAIN_EXHAUSTED");
  }
  const LeggedTerrainGrid terrain_grid{projection, capability, stop_token};
  if (terrain_grid.canceled()) {
    return Failure(LeggedLatticeStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (!terrain_grid.ok()) {
    return Failure(
        LeggedLatticeStatus::kInvalidRequest,
        "LEGGED_LATTICE_REQUEST_INVALID");
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
  graph.state_poses.push_back(graph.true_start_pose);
  graph.search_problem.outgoing_edges.emplace_back();
  std::map<LeggedStateKey, std::size_t> state_indices;
  state_indices.emplace(KeyOf(start), 0U);
  std::queue<std::size_t> pending;
  pending.push(0U);
  const auto* point_goal = std::get_if<PointGoal>(&goal.target);
  const double maximum_goal_translation =
      MaximumPrimitiveTranslation(capability) +
      projection.source_map()->resolution_m() * std::numbers::sqrt2 / 2.0;
  const double maximum_goal_yaw = MaximumPrimitiveYaw(capability, config);
  bool search_domain_rejected = false;
  while (!pending.empty()) {
    if (stop_token.stop_requested()) {
      return Failure(LeggedLatticeStatus::kCanceled, "REQUEST_CANCELED");
    }
    const std::size_t source_index = pending.front();
    pending.pop();
    const LeggedLatticeState source_state = graph.states[source_index];
    const LeggedPose source_pose = graph.state_poses[source_index];
    const Interval source_body_z_m = source_index == 0U
        ? graph.true_start_body_z_m
        : source_state.reachable_body_z_m;
    if (point_goal != nullptr && !GoalContainsBodyPose(goal, source_pose)) {
      bool canceled = false;
      auto connector = ApplyPointGoalConnector(
          source_pose, source_body_z_m, *point_goal, goal.yaw_rad,
          maximum_goal_translation, maximum_goal_yaw, projection,
          search_domain, capability, stop_token, canceled,
          search_domain_rejected, &terrain_grid);
      if (canceled) {
        return Failure(LeggedLatticeStatus::kCanceled, "REQUEST_CANCELED");
      }
      if (connector.has_value()) {
        const auto target_cell = projection.source_map()->PositionToCell(Vec2{
            .x = connector->target_pose.position_m.x,
            .y = connector->target_pose.position_m.y,
        });
        if (target_cell.has_value()) {
          const std::size_t target_index = graph.states.size();
          graph.states.push_back(LeggedLatticeState{
              .cell_x = target_cell->x,
              .cell_y = target_cell->y,
              .yaw_bin = YawToBin(
                  connector->target_pose.yaw_rad,
                  config.legged.yaw_bin_count),
              .reachable_body_z_m = connector->target_body_z_m,
          });
          graph.state_poses.push_back(connector->target_pose);
          graph.search_problem.outgoing_edges.emplace_back();
          connector->stable_index = graph.transitions.size();
          const double edge_cost =
              TransitionDurationLowerBound(*connector, capability) +
              0.5 * connector->path_length_m +
              0.25 * std::abs(ShortestYawDelta(
                  connector->source_pose.yaw_rad,
                  connector->target_pose.yaw_rad));
          graph.transitions.push_back(*connector);
          graph.search_problem.outgoing_edges[source_index].push_back(
              shared::GraphEdge{
                  .target_state = target_index,
                  .cost = edge_cost,
                  .stable_index = connector->stable_index,
              });
        }
      }
    }
    for (const OrderedPrimitive& ordered : ordered_primitives) {
      bool canceled = false;
      auto transition = ApplyPrimitive(
          source_pose, source_body_z_m, *ordered.primitive,
          ordered.original_index, projection, search_domain, capability,
          config, stop_token, canceled, search_domain_rejected,
          &terrain_grid);
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
        target_index = graph.states.size();
        graph.states.push_back(target_state);
        graph.state_poses.push_back(transition->target_pose);
        graph.search_problem.outgoing_edges.emplace_back();
        state_indices.emplace(KeyOf(target_state), target_index);
        pending.push(target_index);
      } else {
        target_index = found->second;
      }
      transition->stable_index = graph.transitions.size();
      const double edge_cost =
          TransitionDurationLowerBound(*transition, capability) +
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
  for (std::size_t index = 0U; index < graph.states.size(); ++index) {
    const LeggedPose pose = graph.state_poses[index];
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
    if (GoalContainsBodyPose(goal, pose)) {
      graph.search_problem.goal_mask[index] = 1U;
    }
  }
  if (graph.search_problem.goal_mask.front() == 0U &&
      graph.search_problem.outgoing_edges.front().empty()) {
    if (search_domain_rejected) {
      return Failure(
          LeggedLatticeStatus::kNoPath, "LOCAL_SEARCH_DOMAIN_EXHAUSTED");
    }
    return Failure(LeggedLatticeStatus::kInvalidRequest,
                   "LEGGED_START_CONNECTOR_INFEASIBLE");
  }
  if (std::ranges::none_of(
          graph.search_problem.goal_mask,
          [](const std::uint8_t value) { return value != 0U; }) &&
      search_domain_rejected) {
    return Failure(
        LeggedLatticeStatus::kNoPath, "LOCAL_SEARCH_DOMAIN_EXHAUSTED");
  }
  graph.search_problem.config = config.search;
  return LeggedLatticeBuildResult{
      .status = LeggedLatticeStatus::kReady,
      .graph = std::move(graph),
      .reason_code = {},
  };
}

LeggedLatticeSearchResult SearchLeggedLattice(
    const LeggedState& current_state, const GoalRegion& goal,
    const shared::SafeProjection& projection,
    const hierarchical::LocalSearchDomain& search_domain,
    const LeggedCapability& capability, const PlannerConfig& config,
    const std::stop_token stop_token) try {
  if (stop_token.stop_requested()) {
    return SearchFailure(LeggedLatticeStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (projection.source_map() == nullptr ||
      !ValidCapability(capability, config) ||
      !IsFinite(current_state.body_pose) ||
      search_domain.width() != projection.source_map()->width() ||
      search_domain.height() != projection.source_map()->height()) {
    return SearchFailure(
        LeggedLatticeStatus::kInvalidRequest,
        "LEGGED_LATTICE_REQUEST_INVALID");
  }
  const auto current_yaw = YawFromQuaternion(current_state.body_pose.orientation);
  const auto start_cell = projection.source_map()->PositionToCell(Vec2{
      .x = current_state.body_pose.position_m.x,
      .y = current_state.body_pose.position_m.y,
  });
  if (!current_yaw.has_value() || !start_cell.has_value()) {
    return SearchFailure(
        LeggedLatticeStatus::kInvalidRequest, "LEGGED_START_NOT_SAFE");
  }
  const LeggedTerrainEvaluation start_terrain = EvaluateLeggedTerrainCell(
      projection, capability, *start_cell, stop_token);
  if (start_terrain.canceled) {
    return SearchFailure(LeggedLatticeStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (!start_terrain.hard_feasible ||
      current_state.body_pose.position_m.z <
          start_terrain.body_height_m.lower - kComparisonTolerance ||
      current_state.body_pose.position_m.z >
          start_terrain.body_height_m.upper + kComparisonTolerance) {
    return SearchFailure(
        LeggedLatticeStatus::kInvalidRequest, "LEGGED_START_NOT_SAFE");
  }
  if (!search_domain.Contains(*start_cell)) {
    return SearchFailure(
        LeggedLatticeStatus::kNoPath, "LOCAL_SEARCH_DOMAIN_EXHAUSTED");
  }
  if (GoalContainsBodyPose(
          goal, LeggedPose{.position_m = current_state.body_pose.position_m,
                           .yaw_rad = *current_yaw})) {
    return LeggedLatticeSearchResult{
        .status = LeggedLatticeStatus::kReady,
        .plan = LeggedDiscretePlan{
            .transitions = {}, .cost = 0.0, .expanded_states = 0U},
        .reason_code = {},
    };
  }

  const LeggedTerrainGrid terrain_grid{projection, capability, stop_token};
  if (terrain_grid.canceled()) {
    return SearchFailure(LeggedLatticeStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (!terrain_grid.ok()) {
    return SearchFailure(
        LeggedLatticeStatus::kInvalidRequest,
        "LEGGED_LATTICE_REQUEST_INVALID");
  }
  std::vector<OrderedPrimitive> ordered_primitives;
  ordered_primitives.reserve(capability.motion_primitives.size());
  for (std::size_t index = 0U; index < capability.motion_primitives.size();
       ++index) {
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

  const double maximum_speed = MaximumPlanarSpeed(capability);
  const auto heuristic = [&](const LeggedPose& pose) {
    double estimate = GoalDistance(goal, pose) / maximum_speed;
    if (goal.yaw_rad.has_value()) {
      const double yaw_speed = std::max(
          std::abs(capability.yaw_rate_radps.lower),
          std::abs(capability.yaw_rate_radps.upper));
      estimate = std::max(
          estimate,
          std::abs(ShortestYawDelta(pose.yaw_rad, *goal.yaw_rad)) /
              yaw_speed);
    }
    return estimate;
  };
  const LeggedPose true_start{
      .position_m = current_state.body_pose.position_m,
      .yaw_rad = *current_yaw,
  };
  const LeggedLatticeState start{
      .cell_x = start_cell->x,
      .cell_y = start_cell->y,
      .yaw_bin = YawToBin(*current_yaw, config.legged.yaw_bin_count),
      .reachable_body_z_m = start_terrain.body_height_m,
  };
  std::vector<LazySearchNode> nodes;
  nodes.push_back(LazySearchNode{
      .state = start,
      .pose = true_start,
      .path_cost = 0.0,
      .parent = kNoParent,
      .incoming = std::nullopt,
      .open_sequence = 0U,
      .closed = false,
  });
  std::map<LeggedStateKey, std::size_t> node_by_state;
  node_by_state.emplace(KeyOf(start), 0U);
  std::priority_queue<
      LazyOpenEntry, std::vector<LazyOpenEntry>, LazyOpenGreater> open;
  open.push(LazyOpenEntry{
      .estimated_total_cost = heuristic(true_start),
      .path_cost = 0.0,
      .key = KeyOf(start),
      .node_index = 0U,
      .sequence = 0U,
  });
  std::size_t next_sequence = 1U;
  std::size_t expanded_states = 0U;
  std::optional<double> best_goal_cost;
  std::size_t best_goal_parent = kNoParent;
  std::optional<LeggedTransition> best_terminal;
  bool start_has_valid_edge = false;
  bool start_has_domain_rejected_edge = false;
  bool search_domain_rejected = false;
  const auto* point_goal = std::get_if<PointGoal>(&goal.target);
  const double maximum_goal_translation = MaximumPrimitiveTranslation(capability) +
      projection.source_map()->resolution_m() * std::numbers::sqrt2 / 2.0;
  const double maximum_goal_yaw = MaximumPrimitiveYaw(capability, config);
  const auto update_goal = [&](const std::size_t parent,
                               std::optional<LeggedTransition> terminal,
                               const double cost) {
    if (std::isfinite(cost) &&
        (!best_goal_cost.has_value() ||
         cost + kComparisonTolerance < *best_goal_cost)) {
      best_goal_cost = cost;
      best_goal_parent = parent;
      best_terminal = std::move(terminal);
    }
  };

  while (!open.empty()) {
    if (stop_token.stop_requested()) {
      return SearchFailure(
          LeggedLatticeStatus::kCanceled, "REQUEST_CANCELED",
          expanded_states);
    }
    if (best_goal_cost.has_value() &&
        open.top().estimated_total_cost + kComparisonTolerance >=
            *best_goal_cost) {
      break;
    }
    const LazyOpenEntry current_entry = open.top();
    open.pop();
    LazySearchNode& current = nodes[current_entry.node_index];
    if (current.closed || current.open_sequence != current_entry.sequence ||
        std::abs(current.path_cost - current_entry.path_cost) >
            kComparisonTolerance) {
      continue;
    }
    current.closed = true;
    ++expanded_states;
    const LazySearchNode current_snapshot = current;
    if (std::holds_alternative<PlanarRegionGoal>(goal.target) &&
        GoalContainsBodyPose(goal, current_snapshot.pose)) {
      update_goal(current_entry.node_index, std::nullopt,
                  current_snapshot.path_cost);
    }
    if (point_goal != nullptr &&
        !GoalContainsBodyPose(goal, current_snapshot.pose)) {
      bool canceled = false;
      bool domain_rejected = false;
      auto terminal = ApplyPointGoalConnector(
          current_snapshot.pose,
          current_entry.node_index == 0U
              ? Interval{.lower = current_state.body_pose.position_m.z,
                         .upper = current_state.body_pose.position_m.z}
              : current_snapshot.state.reachable_body_z_m,
          *point_goal, goal.yaw_rad, maximum_goal_translation,
          maximum_goal_yaw, projection, search_domain, capability,
          stop_token, canceled, domain_rejected, &terrain_grid);
      if (canceled) {
        return SearchFailure(
            LeggedLatticeStatus::kCanceled, "REQUEST_CANCELED",
            expanded_states);
      }
      if (terminal.has_value()) {
        if (current_entry.node_index == 0U) {
          start_has_valid_edge = true;
        }
        const double edge_cost =
            TransitionDurationLowerBound(*terminal, capability) +
            0.5 * terminal->path_length_m +
            0.25 * std::abs(ShortestYawDelta(
                       terminal->source_pose.yaw_rad,
                       terminal->target_pose.yaw_rad));
        update_goal(current_entry.node_index, std::move(terminal),
                    current_snapshot.path_cost + edge_cost);
      }
      if (current_entry.node_index == 0U && domain_rejected) {
        start_has_domain_rejected_edge = true;
      }
      search_domain_rejected = search_domain_rejected || domain_rejected;
    }

    for (const OrderedPrimitive& ordered : ordered_primitives) {
      bool canceled = false;
      bool domain_rejected = false;
      auto transition = ApplyPrimitive(
          current_snapshot.pose,
          current_entry.node_index == 0U
              ? Interval{.lower = current_state.body_pose.position_m.z,
                         .upper = current_state.body_pose.position_m.z}
              : current_snapshot.state.reachable_body_z_m,
          *ordered.primitive, ordered.original_index, projection,
          search_domain, capability, config, stop_token, canceled,
          domain_rejected, &terrain_grid);
      if (canceled) {
        return SearchFailure(
            LeggedLatticeStatus::kCanceled, "REQUEST_CANCELED",
            expanded_states);
      }
      if (!transition.has_value()) {
        if (current_entry.node_index == 0U && domain_rejected) {
          start_has_domain_rejected_edge = true;
        }
        search_domain_rejected = search_domain_rejected || domain_rejected;
        continue;
      }
      if (current_entry.node_index == 0U) {
        start_has_valid_edge = true;
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
      const double edge_cost =
          TransitionDurationLowerBound(*transition, capability) +
          0.5 * transition->path_length_m +
          0.25 * std::abs(ShortestYawDelta(
                     transition->source_pose.yaw_rad,
                     transition->target_pose.yaw_rad));
      if (!std::isfinite(edge_cost) || edge_cost <= 0.0) {
        return SearchFailure(
            LeggedLatticeStatus::kInvalidRequest,
            "LEGGED_LATTICE_EDGE_COST_INVALID", expanded_states);
      }
      const double candidate_cost = current_snapshot.path_cost + edge_cost;
      std::size_t target_index{};
      const auto found = node_by_state.find(KeyOf(target_state));
      if (found == node_by_state.end()) {
        target_index = nodes.size();
        nodes.push_back(LazySearchNode{
            .state = target_state,
            .pose = transition->target_pose,
        });
        node_by_state.emplace(KeyOf(target_state), target_index);
      } else {
        target_index = found->second;
      }
      LazySearchNode& target = nodes[target_index];
      if (candidate_cost + kComparisonTolerance >= target.path_cost) {
        continue;
      }
      target.pose = transition->target_pose;
      target.state = target_state;
      target.path_cost = candidate_cost;
      target.parent = current_entry.node_index;
      target.incoming = std::move(transition);
      target.open_sequence = next_sequence++;
      target.closed = false;
      open.push(LazyOpenEntry{
          .estimated_total_cost = candidate_cost + heuristic(target.pose),
          .path_cost = candidate_cost,
          .key = KeyOf(target.state),
          .node_index = target_index,
          .sequence = target.open_sequence,
      });
    }
  }

  if (!best_goal_cost.has_value() || best_goal_parent == kNoParent) {
    if (!start_has_valid_edge) {
      return SearchFailure(
          start_has_domain_rejected_edge ? LeggedLatticeStatus::kNoPath
                                         : LeggedLatticeStatus::kInvalidRequest,
          start_has_domain_rejected_edge
              ? "LOCAL_SEARCH_DOMAIN_EXHAUSTED"
              : "LEGGED_START_CONNECTOR_INFEASIBLE",
          expanded_states);
    }
    return SearchFailure(
        LeggedLatticeStatus::kNoPath,
        search_domain_rejected ? "LOCAL_SEARCH_DOMAIN_EXHAUSTED"
                               : "LEGGED_NO_KNOWN_SAFE_ROUTE",
        expanded_states);
  }
  std::vector<LeggedTransition> transitions;
  for (std::size_t current = best_goal_parent; current != 0U;) {
    if (current >= nodes.size() || nodes[current].parent == kNoParent ||
        !nodes[current].incoming.has_value() ||
        transitions.size() >= nodes.size()) {
      return SearchFailure(
          LeggedLatticeStatus::kInvalidRequest,
          "LEGGED_SEARCH_RESULT_INVALID", expanded_states);
    }
    transitions.push_back(*nodes[current].incoming);
    current = nodes[current].parent;
  }
  std::reverse(transitions.begin(), transitions.end());
  if (best_terminal.has_value()) {
    transitions.push_back(*best_terminal);
  }
  return LeggedLatticeSearchResult{
      .status = LeggedLatticeStatus::kReady,
      .plan = LeggedDiscretePlan{
          .transitions = std::move(transitions),
          .cost = *best_goal_cost,
          .expanded_states = expanded_states,
      },
      .reason_code = {},
  };
} catch (const std::bad_alloc&) {
  return SearchFailure(
      LeggedLatticeStatus::kResourceExhausted,
      "LEGGED_SEARCH_ALLOCATION_FAILURE");
}

shared::PrimitiveGraphBuildResult BuildLeggedPrimitiveGraph(
    const LeggedState& current_state,
    const shared::SafeProjection& projection,
    const LeggedCapability& capability,
    const PlannerConfig& config,
    const std::stop_token stop_token) try {
  const auto fail = [](std::string reason_code) {
    return shared::PrimitiveGraphBuildResult{
        .reason_code = std::move(reason_code),
    };
  };
  if (stop_token.stop_requested()) {
    return fail("REQUEST_CANCELED");
  }
  if (projection.source_map() == nullptr ||
      !ValidCapability(capability, config) ||
      !IsFinite(current_state.body_pose)) {
    return fail("LEGGED_PRIMITIVE_GRAPH_REQUEST_INVALID");
  }
  const auto current_yaw =
      YawFromQuaternion(current_state.body_pose.orientation);
  const auto start_cell = projection.source_map()->PositionToCell(Vec2{
      .x = current_state.body_pose.position_m.x,
      .y = current_state.body_pose.position_m.y,
  });
  if (!current_yaw.has_value() || !start_cell.has_value()) {
    return fail("LEGGED_START_NOT_SAFE");
  }
  const LeggedTerrainEvaluation start_terrain = EvaluateLeggedTerrainCell(
      projection, capability, *start_cell, stop_token);
  if (start_terrain.canceled) {
    return fail("REQUEST_CANCELED");
  }
  if (!start_terrain.hard_feasible ||
      current_state.body_pose.position_m.z <
          start_terrain.body_height_m.lower - kComparisonTolerance ||
      current_state.body_pose.position_m.z >
          start_terrain.body_height_m.upper + kComparisonTolerance) {
    return fail("LEGGED_START_NOT_SAFE");
  }
  hierarchical::LocalSearchDomain full_search_domain{
      projection.source_map()->width(), projection.source_map()->height(),
      std::vector<std::uint8_t>(projection.source_map()->cell_count(), 1U)};

  std::vector<OrderedPrimitive> ordered_primitives;
  ordered_primitives.reserve(capability.motion_primitives.size());
  for (std::size_t index = 0U;
       index < capability.motion_primitives.size(); ++index) {
    if (index > std::numeric_limits<std::uint32_t>::max()) {
      return fail("LEGGED_PRIMITIVE_INDEX_OVERFLOW");
    }
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

  const auto normalized_bits = [](double value) {
    if (value == 0.0) {
      value = 0.0;
    }
    return std::bit_cast<std::uint64_t>(value);
  };
  using StateKey = std::tuple<
      std::int32_t, std::int32_t, std::int32_t,
      std::uint64_t, std::uint64_t>;
  const auto state_key = [&](const LeggedLatticeState& state) {
    return StateKey{
        state.cell_x,
        state.cell_y,
        state.yaw_bin,
        normalized_bits(state.reachable_body_z_m.lower),
        normalized_bits(state.reachable_body_z_m.upper),
    };
  };
  struct GraphNode final {
    LeggedLatticeState state;
    LeggedPose pose;
    double path_cost{std::numeric_limits<double>::infinity()};
    bool closed{};
  };
  struct PendingEdge final {
    std::size_t source{};
    std::size_t target{};
    std::uint32_t primitive_index{};
    std::string primitive_id;
    double cost{};
  };

  const Interval start_interval = start_terrain.body_height_m;
  LeggedPose start_pose{
      .position_m = current_state.body_pose.position_m,
      .yaw_rad = *current_yaw,
  };
  start_pose.position_m.z =
      0.5 * (start_interval.lower + start_interval.upper);
  const LeggedLatticeState start_state{
      .cell_x = start_cell->x,
      .cell_y = start_cell->y,
      .yaw_bin = YawToBin(*current_yaw, config.legged.yaw_bin_count),
      .reachable_body_z_m = start_interval,
  };
  std::vector<GraphNode> nodes{
      GraphNode{
          .state = start_state,
          .pose = start_pose,
          .path_cost = 0.0,
      },
  };
  std::map<StateKey, std::size_t> state_indices;
  state_indices.emplace(state_key(start_state), 0U);
  using OpenEntry = std::pair<double, std::size_t>;
  std::priority_queue<
      OpenEntry, std::vector<OpenEntry>, std::greater<OpenEntry>> open;
  open.emplace(0.0, 0U);
  std::vector<PendingEdge> pending_edges;

  while (!open.empty()) {
    if (stop_token.stop_requested()) {
      return fail("REQUEST_CANCELED");
    }
    const auto [queued_cost, source_index] = open.top();
    open.pop();
    GraphNode& source = nodes[source_index];
    if (source.closed ||
        std::abs(queued_cost - source.path_cost) > kComparisonTolerance) {
      continue;
    }
    source.closed = true;
    const LeggedPose source_pose = source.pose;
    const Interval source_interval = source.state.reachable_body_z_m;
    const double source_cost = source.path_cost;
    for (const OrderedPrimitive& ordered : ordered_primitives) {
      bool canceled = false;
      bool search_domain_rejected = false;
      auto transition = ApplyPrimitive(
          source_pose, source_interval, *ordered.primitive,
          ordered.original_index, projection, full_search_domain, capability,
          config, stop_token, canceled, search_domain_rejected, nullptr);
      if (canceled) {
        return fail("REQUEST_CANCELED");
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
      const StateKey key = state_key(target_state);
      std::size_t target_index{};
      const auto found = state_indices.find(key);
      if (found == state_indices.end()) {
        target_index = nodes.size();
        nodes.push_back(GraphNode{
            .state = target_state,
            .pose = transition->target_pose,
        });
        state_indices.emplace(key, target_index);
      } else {
        target_index = found->second;
      }
      const double edge_cost =
          TransitionDurationLowerBound(*transition, capability) +
          0.5 * transition->path_length_m +
          0.25 * std::abs(ShortestYawDelta(
              transition->source_pose.yaw_rad,
              transition->target_pose.yaw_rad));
      if (!std::isfinite(edge_cost) || edge_cost <= 0.0) {
        return fail("LEGGED_LATTICE_EDGE_COST_INVALID");
      }
      pending_edges.push_back(PendingEdge{
          .source = source_index,
          .target = target_index,
          .primitive_index =
              static_cast<std::uint32_t>(ordered.original_index),
          .primitive_id = ordered.primitive->primitive_id,
          .cost = edge_cost,
      });
      const double candidate_cost = source_cost + edge_cost;
      GraphNode& target = nodes[target_index];
      if (candidate_cost + kComparisonTolerance < target.path_cost) {
        target.path_cost = candidate_cost;
        target.closed = false;
        open.emplace(candidate_cost, target_index);
      }
    }
  }

  std::vector<std::uint8_t> primitive_bytes;
  const auto append_uint64 = [&](const std::uint64_t value) {
    for (std::size_t index = 0U; index < 8U; ++index) {
      primitive_bytes.push_back(
          static_cast<std::uint8_t>(value >> (index * 8U)));
    }
  };
  const auto append_double = [&](double value) {
    if (value == 0.0) {
      value = 0.0;
    }
    append_uint64(std::bit_cast<std::uint64_t>(value));
  };
  const auto append_string = [&](const std::string& value) {
    append_uint64(static_cast<std::uint64_t>(value.size()));
    primitive_bytes.insert(
        primitive_bytes.end(), value.begin(), value.end());
  };
  append_string("legged-body-primitives/v1");
  append_uint64(
      static_cast<std::uint64_t>(capability.motion_primitives.size()));
  for (const LeggedBodyPrimitive& primitive : capability.motion_primitives) {
    append_string(primitive.primitive_id);
    primitive_bytes.push_back(static_cast<std::uint8_t>(primitive.kind));
    append_double(primitive.body_frame_displacement_m.x);
    append_double(primitive.body_frame_displacement_m.y);
    append_double(primitive.body_frame_displacement_m.z);
    append_double(primitive.yaw_change_rad);
  }

  shared::PrimitiveGraphBuildResult graph{
      .platform_type = PlatformType::kLegged,
      .width = projection.source_map()->width(),
      .height = projection.source_map()->height(),
      .anchor_state_index = 0U,
      .algorithm_id =
          "cpp-legged-motion-primitive-recoverable-graph/v1",
      .state_schema = "legged-lattice-state/v1",
      .primitive_set_canonical_bytes = std::move(primitive_bytes),
  };
  graph.states.reserve(nodes.size());
  for (const GraphNode& node : nodes) {
    graph.states.push_back(PrimitiveReachabilityState{
        .position_m = node.pose.position_m,
        .yaw_rad = node.pose.yaw_rad,
        .cell_x = node.state.cell_x,
        .cell_y = node.state.cell_y,
        .yaw_bin = node.state.yaw_bin,
        .motion_mode = 1,
        .body_z_m = node.state.reachable_body_z_m,
        .path_cost = node.path_cost,
        .observation_state = 1U,
    });
  }
  graph.potential_edges.reserve(pending_edges.size());
  for (const PendingEdge& edge : pending_edges) {
    graph.potential_edges.push_back(shared::PrimitiveGraphPotentialEdge{
        .source_state_index = edge.source,
        .target_state_index = edge.target,
        .primitive_index = edge.primitive_index,
        .primitive_id = edge.primitive_id,
        .cost = edge.cost,
        .certified = true,
    });
  }
  return graph;
} catch (const std::bad_alloc&) {
  return shared::PrimitiveGraphBuildResult{
      .reason_code = "LEGGED_SEARCH_ALLOCATION_FAILURE",
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
