#include "grid_v1/grid_v1_planner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lunar_pure_planner_core/planning_timing.hpp"

namespace lunar::pure_planning::grid_v1 {
namespace {

constexpr double kEpsilon = 1.0e-9;

struct Cell final {
  std::int32_t x{};
  std::int32_t y{};

  [[nodiscard]] bool operator==(const Cell&) const = default;
};

struct CellHash final {
  [[nodiscard]] std::size_t operator()(const Cell cell) const noexcept {
    const auto x = static_cast<std::uint32_t>(cell.x);
    const auto y = static_cast<std::uint32_t>(cell.y);
    return (static_cast<std::size_t>(x) << 32U) ^ y;
  }
};

[[nodiscard]] bool Finite(const double value) noexcept {
  return std::isfinite(value);
}

[[nodiscard]] bool FinitePoint(const Vec3 point) noexcept {
  return Finite(point.x) && Finite(point.y) && Finite(point.z);
}

[[nodiscard]] double NormalizeAngle(double angle) noexcept {
  while (angle > std::numbers::pi) {
    angle -= 2.0 * std::numbers::pi;
  }
  while (angle <= -std::numbers::pi) {
    angle += 2.0 * std::numbers::pi;
  }
  return angle;
}

[[nodiscard]] double Yaw(const Quaternion& orientation) noexcept {
  return std::atan2(
      2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
      1.0 - 2.0 * (orientation.y * orientation.y +
                   orientation.z * orientation.z));
}

[[nodiscard]] Quaternion YawQuaternion(const double yaw) noexcept {
  return Quaternion{.w = std::cos(yaw / 2.0), .z = std::sin(yaw / 2.0)};
}

[[nodiscard]] std::optional<Quaternion> NormalizeQuaternion(
    const Quaternion& value) noexcept {
  if (!Finite(value.w) || !Finite(value.x) || !Finite(value.y) ||
      !Finite(value.z)) {
    return std::nullopt;
  }
  const double magnitude_squared = value.w * value.w + value.x * value.x +
                                   value.y * value.y + value.z * value.z;
  if (!Finite(magnitude_squared) || magnitude_squared <= kEpsilon) {
    return std::nullopt;
  }
  const double inverse = 1.0 / std::sqrt(magnitude_squared);
  return Quaternion{.w = value.w * inverse, .x = value.x * inverse,
                    .y = value.y * inverse, .z = value.z * inverse};
}

[[nodiscard]] Quaternion Multiply(const Quaternion& left,
                                  const Quaternion& right) noexcept {
  return Quaternion{
      .w = left.w * right.w - left.x * right.x - left.y * right.y -
           left.z * right.z,
      .x = left.w * right.x + left.x * right.w + left.y * right.z -
           left.z * right.y,
      .y = left.w * right.y - left.x * right.z + left.y * right.w +
           left.z * right.x,
      .z = left.w * right.z + left.x * right.y - left.y * right.x +
           left.z * right.w,
  };
}

[[nodiscard]] Vec3 Rotate(const Quaternion& rotation, const Vec3& point) noexcept {
  const Quaternion point_quaternion{.x = point.x, .y = point.y, .z = point.z};
  const Quaternion inverse{.w = rotation.w, .x = -rotation.x,
                           .y = -rotation.y, .z = -rotation.z};
  const Quaternion rotated = Multiply(Multiply(rotation, point_quaternion), inverse);
  return Vec3{.x = rotated.x, .y = rotated.y, .z = rotated.z};
}

[[nodiscard]] std::optional<WheeledState> StateInMap(
    const WheeledState& odom_state, const RigidTransform& map_from_odom) noexcept {
  const auto transform_rotation = NormalizeQuaternion(map_from_odom.rotation);
  const auto pose_rotation = NormalizeQuaternion(odom_state.pose.orientation);
  if (!transform_rotation || !pose_rotation || !FinitePoint(odom_state.pose.position_m) ||
      !FinitePoint(odom_state.velocity.linear_mps) ||
      !FinitePoint(odom_state.velocity.angular_radps) ||
      !FinitePoint(map_from_odom.translation_m) ||
      map_from_odom.parent_frame != "map" || map_from_odom.child_frame != "odom") {
    return std::nullopt;
  }
  const Vec3 rotated_position = Rotate(*transform_rotation, odom_state.pose.position_m);
  const auto orientation = NormalizeQuaternion(
      Multiply(*transform_rotation, *pose_rotation));
  if (!orientation) {
    return std::nullopt;
  }
  WheeledState result = odom_state;
  result.pose.position_m = Vec3{.x = rotated_position.x + map_from_odom.translation_m.x,
                                .y = rotated_position.y + map_from_odom.translation_m.y,
                                .z = rotated_position.z + map_from_odom.translation_m.z};
  result.pose.orientation = *orientation;
  result.velocity.linear_mps = Rotate(*transform_rotation, odom_state.velocity.linear_mps);
  result.velocity.angular_radps = Rotate(*transform_rotation, odom_state.velocity.angular_radps);
  return result;
}

[[nodiscard]] std::optional<Cell> ToCell(
    const TraversabilitySnapshot& snapshot, const Vec3 point) noexcept {
  const double resolution = snapshot.resolution_m();
  const Vec3 origin = snapshot.origin_m();
  if (!FinitePoint(point) || !Finite(resolution) || resolution <= 0.0 ||
      !FinitePoint(origin)) {
    return std::nullopt;
  }
  const double x = std::floor((point.x - origin.x) / resolution);
  const double y = std::floor((point.y - origin.y) / resolution);
  if (!Finite(x) || !Finite(y) ||
      x < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
      x > static_cast<double>(std::numeric_limits<std::int32_t>::max()) ||
      y < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
      y > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
    return std::nullopt;
  }
  return Cell{.x = static_cast<std::int32_t>(x),
              .y = static_cast<std::int32_t>(y)};
}

[[nodiscard]] Vec3 CellCenter(const TraversabilitySnapshot& snapshot,
                               const Cell cell) noexcept {
  const double resolution = snapshot.resolution_m();
  const Vec3 origin = snapshot.origin_m();
  return Vec3{.x = origin.x + (static_cast<double>(cell.x) + 0.5) * resolution,
              .y = origin.y + (static_cast<double>(cell.y) + 0.5) * resolution,
              .z = origin.z};
}

[[nodiscard]] TraversabilityState StateAt(
    const TraversabilitySnapshot& snapshot, const Cell cell) {
  const Vec3 center = CellCenter(snapshot, cell);
  return snapshot.StateAtWorld(center.x, center.y);
}

void AppendUnique(std::vector<Cell>* cells, const Cell cell) {
  if (cells->empty() || cells->back() != cell) {
    cells->push_back(cell);
  }
}

[[nodiscard]] std::vector<Cell> Supercover(const Cell start,
                                            const Cell end) {
  std::vector<Cell> cells;
  std::int32_t x = start.x;
  std::int32_t y = start.y;
  const std::int32_t dx = end.x - start.x;
  const std::int32_t dy = end.y - start.y;
  const std::int32_t sign_x = dx < 0 ? -1 : 1;
  const std::int32_t sign_y = dy < 0 ? -1 : 1;
  const std::int64_t nx = std::llabs(static_cast<std::int64_t>(dx));
  const std::int64_t ny = std::llabs(static_cast<std::int64_t>(dy));
  std::int64_t ix{};
  std::int64_t iy{};
  AppendUnique(&cells, Cell{.x = x, .y = y});
  while (ix < nx || iy < ny) {
    const std::int64_t decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx;
    if (decision == 0) {
      AppendUnique(&cells, Cell{.x = x + sign_x, .y = y});
      AppendUnique(&cells, Cell{.x = x, .y = y + sign_y});
      x += sign_x;
      y += sign_y;
      ++ix;
      ++iy;
    } else if (decision < 0) {
      x += sign_x;
      ++ix;
    } else {
      y += sign_y;
      ++iy;
    }
    AppendUnique(&cells, Cell{.x = x, .y = y});
  }
  return cells;
}

[[nodiscard]] bool CellsFree(const TraversabilitySnapshot& snapshot,
                             const std::vector<Cell>& cells,
                             std::size_t* checked_cells) {
  for (const Cell cell : cells) {
    if (checked_cells != nullptr) {
      ++*checked_cells;
    }
    if (StateAt(snapshot, cell) != TraversabilityState::kFree) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool SegmentFree(const TraversabilitySnapshot& snapshot,
                               const Cell start, const Cell end) {
  return CellsFree(snapshot, Supercover(start, end), nullptr);
}

enum class SearchStatus : std::uint8_t {
  kSolved,
  kNoPath,
  kTimedOut,
  kCanceled,
};

struct SearchResult final {
  SearchStatus status{SearchStatus::kNoPath};
  std::vector<Cell> path;
  std::uint64_t expanded{};
  std::size_t open_peak{};
};

[[nodiscard]] double Heuristic(const Cell from, const Cell to) noexcept {
  return std::hypot(static_cast<double>(to.x - from.x),
                    static_cast<double>(to.y - from.y));
}

struct OpenNode final {
  Cell cell;
  double f{};
  double h{};
  std::uint64_t order{};
};

struct LaterNode final {
  [[nodiscard]] bool operator()(const OpenNode& lhs,
                                const OpenNode& rhs) const noexcept {
    if (lhs.f != rhs.f) {
      return lhs.f > rhs.f;
    }
    if (lhs.h != rhs.h) {
      return lhs.h > rhs.h;
    }
    if (lhs.cell.y != rhs.cell.y) {
      return lhs.cell.y > rhs.cell.y;
    }
    if (lhs.cell.x != rhs.cell.x) {
      return lhs.cell.x > rhs.cell.x;
    }
    return lhs.order > rhs.order;
  }
};

[[nodiscard]] SearchResult Search(const TraversabilitySnapshot& snapshot,
                                  const Cell start, const Cell goal,
                                  const SearchControl& control,
                                  const std::optional<double> local_horizon_m =
                                      std::nullopt) {
  try {
    if (start == goal) {
      return {.status = SearchStatus::kSolved, .path = {start}};
    }
    std::priority_queue<OpenNode, std::vector<OpenNode>, LaterNode> open;
    std::unordered_map<Cell, double, CellHash> g_score;
    std::unordered_map<Cell, Cell, CellHash> predecessor;
    std::uint64_t order{};
    const double initial_h = Heuristic(start, goal);
    open.push(OpenNode{.cell = start, .f = initial_h, .h = initial_h,
                       .order = order++});
    g_score.emplace(start, 0.0);
    std::size_t open_peak = 1U;
    std::uint64_t expanded{};
    constexpr std::array<std::pair<std::int32_t, std::int32_t>, 8U> kNeighbors{
        {{1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}}};
    while (!open.empty()) {
      if (control.canceled()) {
        return {.status = SearchStatus::kCanceled, .expanded = expanded,
                .open_peak = open_peak};
      }
      if (control.expired()) {
        return {.status = SearchStatus::kTimedOut, .expanded = expanded,
                .open_peak = open_peak};
      }
      const OpenNode current = open.top();
      open.pop();
      const auto current_score = g_score.find(current.cell);
      if (current_score == g_score.end() ||
          current.f > current_score->second + current.h + kEpsilon) {
        continue;
      }
      ++expanded;
      if (current.cell == goal) {
        std::vector<Cell> path;
        for (Cell at = goal;; at = predecessor.at(at)) {
          path.push_back(at);
          if (at == start) {
            break;
          }
        }
        std::reverse(path.begin(), path.end());
        return {.status = SearchStatus::kSolved, .path = std::move(path),
                .expanded = expanded, .open_peak = open_peak};
      }
      for (const auto [dx, dy] : kNeighbors) {
        const Cell next{.x = current.cell.x + dx, .y = current.cell.y + dy};
        if (local_horizon_m.has_value() &&
            Heuristic(start, next) * snapshot.resolution_m() >
                *local_horizon_m + kEpsilon) {
          continue;
        }
        if (StateAt(snapshot, next) != TraversabilityState::kFree) {
          continue;
        }
        if (dx != 0 && dy != 0 &&
            (StateAt(snapshot, Cell{.x = current.cell.x + dx,
                                    .y = current.cell.y}) !=
                 TraversabilityState::kFree ||
             StateAt(snapshot, Cell{.x = current.cell.x,
                                    .y = current.cell.y + dy}) !=
                 TraversabilityState::kFree)) {
          continue;
        }
        const double next_g = current_score->second +
                              (dx == 0 || dy == 0 ? 1.0 : std::sqrt(2.0));
        const auto known = g_score.find(next);
        if (known != g_score.end() && next_g >= known->second - kEpsilon) {
          continue;
        }
        const double h = Heuristic(next, goal);
        g_score.insert_or_assign(next, next_g);
        predecessor.insert_or_assign(next, current.cell);
        open.push(OpenNode{.cell = next, .f = next_g + h, .h = h,
                           .order = order++});
        open_peak = std::max(open_peak, open.size());
      }
    }
    return {.status = SearchStatus::kNoPath, .expanded = expanded,
            .open_peak = open_peak};
  } catch (...) {
    return {.status = SearchStatus::kTimedOut};
  }
}

[[nodiscard]] std::vector<Cell> Shortcut(const TraversabilitySnapshot& snapshot,
                                         const std::vector<Cell>& raw) {
  if (raw.empty()) {
    return {};
  }
  std::vector<Cell> shortcut{raw.front()};
  std::size_t current{};
  while (current + 1U < raw.size()) {
    std::size_t selected = current + 1U;
    for (std::size_t candidate = raw.size() - 1U; candidate > current;
         --candidate) {
      if (SegmentFree(snapshot, raw[current], raw[candidate])) {
        selected = candidate;
        break;
      }
    }
    shortcut.push_back(raw[selected]);
    current = selected;
  }
  return shortcut;
}

[[nodiscard]] std::vector<Vec3> Resample(const TraversabilitySnapshot& snapshot,
                                          const std::vector<Cell>& cells) {
  std::vector<Vec3> points;
  if (cells.empty()) {
    return points;
  }
  points.push_back(CellCenter(snapshot, cells.front()));
  const double resolution = snapshot.resolution_m();
  for (std::size_t index = 1U; index < cells.size(); ++index) {
    const Vec3 from = CellCenter(snapshot, cells[index - 1U]);
    const Vec3 to = CellCenter(snapshot, cells[index]);
    const double distance = std::hypot(to.x - from.x, to.y - from.y);
    const std::size_t steps = std::max<std::size_t>(
        1U, static_cast<std::size_t>(std::ceil(distance / resolution)));
    for (std::size_t step = 1U; step <= steps; ++step) {
      double ratio = static_cast<double>(step) / static_cast<double>(steps);
      // A diagonal midpoint can land exactly on a cell boundary.  Keep an
      // intermediate sample in its preceding cell so world-to-cell conversion
      // preserves the supercover that certified the segment.
      if (step < steps) {
        ratio = std::max(0.0, ratio - 1.0e-10);
      }
      points.push_back(Vec3{.x = from.x + ratio * (to.x - from.x),
                            .y = from.y + ratio * (to.y - from.y),
                            .z = from.z + ratio * (to.z - from.z)});
    }
  }
  return points;
}

[[nodiscard]] std::chrono::nanoseconds PositiveDuration(const double seconds) {
  const double nanoseconds = std::ceil(std::max(seconds, 0.0) * 1.0e9);
  if (!Finite(nanoseconds) ||
      nanoseconds >= static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    return std::chrono::nanoseconds{1};
  }
  return std::chrono::nanoseconds{
      std::max<std::int64_t>(1, static_cast<std::int64_t>(nanoseconds))};
}

[[nodiscard]] PlanningResult Failure(const PlanningStatus status,
                                     std::string reason,
                                     GridV1Diagnostics diagnostics = {},
                                     const PlannerCallTiming& timing = {}) {
  diagnostics.active = true;
  return PlanningResult{.status = status, .reason_code = std::move(reason),
                        .timing = timing,
                        .grid_v1 = std::move(diagnostics)};
}

[[nodiscard]] std::optional<Vec3> PointGoalPosition(
    const GoalRegion& goal) noexcept {
  const auto* point = std::get_if<PointGoal>(&goal.target);
  if (point == nullptr || !FinitePoint(point->position_m) ||
      !Finite(point->tolerance_m) || point->tolerance_m < 0.0) {
    return std::nullopt;
  }
  if ((goal.yaw_rad.has_value() && !Finite(*goal.yaw_rad)) ||
      !Finite(goal.yaw_tolerance_rad) || goal.yaw_tolerance_rad < 0.0) {
    return std::nullopt;
  }
  return point->position_m;
}

[[nodiscard]] bool IsWheeledRequest(const PlanningRequest& request) noexcept {
  return std::holds_alternative<WheeledState>(request.current_state) &&
         std::holds_alternative<WheeledCapability>(request.capability);
}

[[nodiscard]] double DirectionCost(const std::vector<Vec3>& positions,
                                   const double initial_yaw,
                                   const std::optional<double> goal_yaw,
                                   const double speed,
                                   const double spin_rate,
                                   const bool forward) noexcept {
  double cost{};
  double orientation = initial_yaw;
  for (std::size_t index = 1U; index < positions.size(); ++index) {
    const double dx = positions[index].x - positions[index - 1U].x;
    const double dy = positions[index].y - positions[index - 1U].y;
    const double distance = std::hypot(dx, dy);
    if (distance <= kEpsilon) {
      continue;
    }
    const double tangent = std::atan2(dy, dx);
    const double desired = forward ? tangent : tangent + std::numbers::pi;
    cost += std::abs(NormalizeAngle(desired - orientation)) / spin_rate;
    cost += distance / speed;
    orientation = desired;
  }
  if (goal_yaw.has_value()) {
    cost += std::abs(NormalizeAngle(*goal_yaw - orientation)) / spin_rate;
  }
  return cost;
}

}  // namespace

bool PathIsFree(const TraversabilitySnapshot& snapshot,
                const std::vector<Pose3>& path,
                std::size_t* checked_cells) noexcept {
  if (checked_cells != nullptr) {
    *checked_cells = 0U;
  }
  try {
    if (!snapshot.valid() || path.empty()) {
      return false;
    }
    std::optional<Cell> previous;
    for (const Pose3& pose : path) {
      const auto cell = ToCell(snapshot, pose.position_m);
      if (!cell.has_value()) {
        return false;
      }
      const std::vector<Cell> cells = previous.has_value()
                                          ? Supercover(*previous, *cell)
                                          : std::vector<Cell>{*cell};
      if (!CellsFree(snapshot, cells, checked_cells)) {
        return false;
      }
      previous = cell;
    }
    return true;
  } catch (...) {
    return false;
  }
}

PlanningResult Plan(const PlanningRequest& request) noexcept {
  PlannerCallTiming timing;
  try {
    if (!IsWheeledRequest(request) ||
        request.environment_mode != EnvironmentMode::kLunarSurface ||
        request.request_id.empty() ||
        !request.world.traversability_snapshot ||
        !request.world.traversability_snapshot->valid()) {
      return Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT");
    }
    const TraversabilitySnapshot& snapshot = *request.world.traversability_snapshot;
    const auto goal_position = PointGoalPosition(request.goal_map);
    const auto state = StateInMap(std::get<WheeledState>(request.current_state),
                                  request.world.map_from_odom);
    const auto& capability = std::get<WheeledCapability>(request.capability);
    if (!goal_position.has_value() || !state.has_value() ||
        !Finite(capability.maximum_forward_speed_mps) ||
        !Finite(capability.maximum_reverse_speed_mps) ||
        !Finite(capability.maximum_spin_rate_radps) ||
        capability.maximum_forward_speed_mps <= 0.0 ||
        capability.maximum_reverse_speed_mps <= 0.0 ||
        capability.maximum_spin_rate_radps <= 0.0 ||
        !Finite(request.config.grid_v1_local_horizon_m) ||
        request.config.grid_v1_local_horizon_m <= 0.0) {
      return Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT");
    }
    GridV1Diagnostics diagnostics;
    diagnostics.active = true;
    const TraversabilityMetrics metrics = snapshot.metrics();
    diagnostics.traversability_revision = snapshot.revision();
    diagnostics.profile_hash = snapshot.profile_hash();
    diagnostics.canonical_resolution_m = snapshot.resolution_m();
    diagnostics.map_origin_m = snapshot.origin_m();
    diagnostics.allocated_tiles = metrics.allocated_tiles;
    diagnostics.estimated_map_bytes = metrics.estimated_bytes;
    diagnostics.updated_cells = metrics.updated_cells;
    diagnostics.dirty_tiles = metrics.dirty_tiles;
    diagnostics.halo_recomputed_cells = metrics.halo_recomputed_cells;
    diagnostics.free_cells = metrics.free_cells;
    diagnostics.blocked_cells = metrics.blocked_cells;
    diagnostics.unknown_cells = metrics.unknown_cells;
    diagnostics.prior_conflicts = metrics.prior_conflicts;
    if (request.control.canceled()) {
      return Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED",
                     std::move(diagnostics));
    }
    if (request.control.expired()) {
      return Failure(PlanningStatus::kTimedOut, "TIMEOUT", std::move(diagnostics));
    }
    const auto start = ToCell(snapshot, state->pose.position_m);
    const auto goal = ToCell(snapshot, *goal_position);
    if (!start.has_value() || !goal.has_value()) {
      return Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT",
                     std::move(diagnostics));
    }
    if (StateAt(snapshot, *start) != TraversabilityState::kFree) {
      return Failure(PlanningStatus::kNoPath, "START_NOT_FREE",
                     std::move(diagnostics));
    }
    if (StateAt(snapshot, *goal) != TraversabilityState::kFree) {
      return Failure(PlanningStatus::kNoPath, "GOAL_NOT_FREE",
                     std::move(diagnostics));
    }
    SearchResult global;
    {
      ScopedPlannerCall call(PlannerStage::kGlobal, timing,
                             request.control.now);
      global = Search(snapshot, *start, *goal, request.control);
      if (!call.Finish()) {
        return Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR",
                       std::move(diagnostics), timing);
      }
    }
    diagnostics.global_expanded_states = global.expanded;
    diagnostics.global_open_peak = global.open_peak;
    if (global.status == SearchStatus::kCanceled) {
      return Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED",
                     std::move(diagnostics), timing);
    }
    if (global.status == SearchStatus::kTimedOut) {
      return Failure(PlanningStatus::kTimedOut, "TIMEOUT",
                     std::move(diagnostics), timing);
    }
    if (global.status != SearchStatus::kSolved) {
      return Failure(PlanningStatus::kNoPath, "GLOBAL_NO_PATH",
                     std::move(diagnostics), timing);
    }
    GlobalRoutePreview global_route_preview;
    global_route_preview.poses_map.reserve(global.path.size());
    for (const Cell cell : global.path) {
      global_route_preview.poses_map.push_back(
          Pose3{.position_m = CellCenter(snapshot, cell)});
    }
    std::vector<Cell> candidates;
    constexpr std::array<double, 4U> kCandidateDistances{8.0, 6.0, 4.0, 2.0};
    for (const double threshold : kCandidateDistances) {
      double cumulative_distance{};
      std::size_t route_index{};
      while (route_index + 1U < global.path.size()) {
        const double next_distance = cumulative_distance +
            Heuristic(global.path[route_index], global.path[route_index + 1U]) *
                snapshot.resolution_m();
        if (next_distance > threshold + kEpsilon) {
          break;
        }
        cumulative_distance = next_distance;
        ++route_index;
      }
      const Cell candidate = global.path[route_index];
      if (candidate != *start &&
          std::find(candidates.begin(), candidates.end(), candidate) == candidates.end()) {
        candidates.push_back(candidate);
      }
    }
    if (global.path.size() == 1U) {
      candidates.push_back(*goal);
    }
    diagnostics.local_candidate_count = candidates.size();
    if (candidates.empty()) {
      return Failure(PlanningStatus::kNoPath, "LOCAL_NO_CANDIDATE",
                     std::move(diagnostics), timing);
    }
    std::vector<Cell> raw_path;
    for (std::size_t index = 0U; index < candidates.size(); ++index) {
      ++diagnostics.local_attempt_count;
      SearchResult search;
      {
        ScopedPlannerCall call(PlannerStage::kLocal, timing,
                               request.control.now);
        search = Search(snapshot, *start, candidates[index], request.control,
                        request.config.grid_v1_local_horizon_m);
        if (!call.Finish()) {
          return Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR",
                         std::move(diagnostics), timing);
        }
      }
      diagnostics.local_expanded_states += search.expanded;
      diagnostics.local_open_peak = std::max(diagnostics.local_open_peak,
                                             search.open_peak);
      if (search.status == SearchStatus::kCanceled) {
        return Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED",
                       std::move(diagnostics), timing);
      }
      if (search.status == SearchStatus::kTimedOut) {
        return Failure(PlanningStatus::kTimedOut, "TIMEOUT",
                       std::move(diagnostics), timing);
      }
      if (search.status == SearchStatus::kSolved) {
        diagnostics.selected_candidate_index = index;
        raw_path = search.path;
        break;
      }
    }
    if (raw_path.empty()) {
      return Failure(PlanningStatus::kNoPath, "LOCAL_NO_PATH",
                     std::move(diagnostics), timing);
    }
    if (request.control.canceled()) {
      return Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED",
                     std::move(diagnostics), timing);
    }
    if (request.control.expired()) {
      return Failure(PlanningStatus::kTimedOut, "TIMEOUT",
                     std::move(diagnostics), timing);
    }
    diagnostics.raw_path_points = raw_path.size();
    std::vector<Cell> postprocessed;
    std::vector<Pose3> shortcut_poses;
    std::size_t final_supercover_cells{};
    bool shortcut_safe{};
    try {
      postprocessed = Shortcut(snapshot, raw_path);
      shortcut_poses.reserve(postprocessed.size());
      for (const Cell cell : postprocessed) {
        shortcut_poses.push_back(Pose3{.position_m = CellCenter(snapshot, cell)});
      }
      shortcut_safe = !postprocessed.empty() &&
                      PathIsFree(snapshot, shortcut_poses, &final_supercover_cells);
    } catch (...) {
      shortcut_safe = false;
    }
    if (request.control.canceled()) {
      return Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED",
                     std::move(diagnostics), timing);
    }
    if (request.control.expired()) {
      return Failure(PlanningStatus::kTimedOut, "TIMEOUT",
                     std::move(diagnostics), timing);
    }
    if (!shortcut_safe) {
      postprocessed = raw_path;
      diagnostics.postprocess_mode = "RAW_GRID_FALLBACK";
      shortcut_poses.clear();
      for (const Cell cell : postprocessed) {
        shortcut_poses.push_back(Pose3{.position_m = CellCenter(snapshot, cell)});
      }
      bool raw_safe{};
      try {
        raw_safe = PathIsFree(snapshot, shortcut_poses, &final_supercover_cells);
      } catch (...) {
        raw_safe = false;
      }
      if (!raw_safe) {
        return Failure(PlanningStatus::kPlannerError, "POSTCHECK_FAILED",
                       std::move(diagnostics), timing);
      }
    } else {
      diagnostics.postprocess_mode = "SHORTCUT";
    }
    diagnostics.shortcut_path_points = postprocessed.size();
    diagnostics.final_supercover_cells = final_supercover_cells;
    if (request.control.canceled()) {
      return Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED",
                     std::move(diagnostics), timing);
    }
    if (request.control.expired()) {
      return Failure(PlanningStatus::kTimedOut, "TIMEOUT",
                     std::move(diagnostics), timing);
    }
    const std::vector<Vec3> positions = Resample(snapshot, postprocessed);
    diagnostics.resampled_path_points = positions.size();
    if (positions.empty()) {
      return Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR",
                     std::move(diagnostics), timing);
    }
    const double initial_yaw = Yaw(state->pose.orientation);
    diagnostics.forward_cost = DirectionCost(
        positions, initial_yaw, request.goal_map.yaw_rad,
        capability.maximum_forward_speed_mps, capability.maximum_spin_rate_radps,
        true);
    diagnostics.reverse_cost = DirectionCost(
        positions, initial_yaw, request.goal_map.yaw_rad,
        capability.maximum_reverse_speed_mps, capability.maximum_spin_rate_radps,
        false);
    const bool forward = diagnostics.forward_cost <= diagnostics.reverse_cost;
    diagnostics.direction = forward ? "FORWARD" : "REVERSE";
    const double speed = forward ? capability.maximum_forward_speed_mps
                                 : capability.maximum_reverse_speed_mps;
    std::vector<TrajectoryPoint> points;
    points.reserve(positions.size() * 2U + 1U);
    std::chrono::nanoseconds time{};
    double orientation_yaw = initial_yaw;
    points.push_back(TrajectoryPoint{
        .time_from_start = time,
        .pose = Pose3{.position_m = positions.front(),
                      .orientation = YawQuaternion(orientation_yaw)},
    });
    for (std::size_t index = 1U; index < positions.size(); ++index) {
      if (request.control.canceled()) {
        return Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED",
                       std::move(diagnostics), timing);
      }
      if (request.control.expired()) {
        return Failure(PlanningStatus::kTimedOut, "TIMEOUT",
                       std::move(diagnostics), timing);
      }
      const Vec3 from = positions[index - 1U];
      const Vec3 to = positions[index];
      const double dx = to.x - from.x;
      const double dy = to.y - from.y;
      const double distance = std::hypot(dx, dy);
      if (distance <= kEpsilon) {
        continue;
      }
      const double tangent = std::atan2(dy, dx);
      const double desired_yaw = forward ? tangent : tangent + std::numbers::pi;
      const double turn = NormalizeAngle(desired_yaw - orientation_yaw);
      if (std::abs(turn) > kEpsilon) {
        time += PositiveDuration(std::abs(turn) /
                                 capability.maximum_spin_rate_radps);
        orientation_yaw = desired_yaw;
        points.push_back(TrajectoryPoint{
            .time_from_start = time,
            .pose = Pose3{.position_m = from,
                          .orientation = YawQuaternion(orientation_yaw)},
            .velocity = Twist3{.angular_radps = Vec3{
                .z = std::copysign(capability.maximum_spin_rate_radps, turn)}},
        });
        time += std::chrono::nanoseconds{1};
        points.push_back(TrajectoryPoint{
            .time_from_start = time,
            .pose = Pose3{.position_m = from,
                          .orientation = YawQuaternion(orientation_yaw)},
        });
      }
      time += PositiveDuration(distance / speed);
      const bool terminal = index + 1U == positions.size();
      points.push_back(TrajectoryPoint{
          .time_from_start = time,
          .pose = Pose3{.position_m = to,
                        .orientation = YawQuaternion(orientation_yaw)},
          .velocity = terminal ? Twist3{}
                               : Twist3{.linear_mps = Vec3{
                                     .x = speed * dx / distance,
                                     .y = speed * dy / distance}},
      });
    }
    if (request.goal_map.yaw_rad.has_value() &&
        positions.size() > 1U) {
      if (request.control.canceled()) {
        return Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED",
                       std::move(diagnostics), timing);
      }
      if (request.control.expired()) {
        return Failure(PlanningStatus::kTimedOut, "TIMEOUT",
                       std::move(diagnostics), timing);
      }
      const double turn = NormalizeAngle(*request.goal_map.yaw_rad - orientation_yaw);
      if (std::abs(turn) > kEpsilon) {
        time += PositiveDuration(std::abs(turn) /
                                 capability.maximum_spin_rate_radps);
        orientation_yaw = *request.goal_map.yaw_rad;
        points.push_back(TrajectoryPoint{
            .time_from_start = time,
            .pose = Pose3{.position_m = positions.back(),
                          .orientation = YawQuaternion(orientation_yaw)},
            .velocity = Twist3{.angular_radps = Vec3{
                .z = std::copysign(capability.maximum_spin_rate_radps, turn)}},
        });
        time += std::chrono::nanoseconds{1};
        points.push_back(TrajectoryPoint{
            .time_from_start = time,
            .pose = Pose3{.position_m = positions.back(),
                          .orientation = YawQuaternion(orientation_yaw)},
        });
      }
    }
    diagnostics.final_trajectory_points = points.size();
    GlobalRoutePreview preview;
    preview.poses_map.reserve(points.size());
    for (const TrajectoryPoint& point : points) {
      preview.poses_map.push_back(point.pose);
    }
    return PlanningResult{
        .status = PlanningStatus::kSuccess,
        .reference = MotionReference{
            .plan_id = request.request_id,
            .platform_type = PlatformType::kWheeled,
            .input_time = request.world.local_map.stamp,
            .preview = std::move(preview),
            .data = TrajectoryReference{.semantics = TrajectorySemantics::kWheeledBase,
                                        .points = std::move(points)},
        },
        .global_route_preview = std::move(global_route_preview),
        .timing = timing,
        .expanded_states = diagnostics.local_expanded_states,
        .selected_goal_index = diagnostics.selected_candidate_index,
        .grid_v1 = std::move(diagnostics),
    };
  } catch (...) {
    return Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR", {}, timing);
  }
}

}  // namespace lunar::pure_planning::grid_v1
