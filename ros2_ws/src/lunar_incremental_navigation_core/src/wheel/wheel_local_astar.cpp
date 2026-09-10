#include "lunar_incremental_navigation_core/wheel_local_planner.hpp"
#include "lunar_incremental_navigation_core/local_goal_region.hpp"
#include "lunar_incremental_navigation_core/arrival_tolerance.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lunar::incremental_navigation {
namespace {

constexpr double kDiagonalDistance = 1.4142135623730950488;
constexpr double kGeometryEpsilon = 1.0e-12;
constexpr std::size_t kMaximumEnvelopeWidthCells =
    kMaximumLocalPlanningWindowAxisCells;
constexpr std::size_t kMaximumEnvelopeHeightCells =
    kMaximumLocalPlanningWindowAxisCells;
constexpr std::size_t kInvalidParent =
    std::numeric_limits<std::size_t>::max();

struct OpenEntry final {
  std::size_t offset{};
  long double g{};
  double h{};
  long double f{};
  std::uint64_t insertion_order{};
};

struct OpenEntryLater final {
  [[nodiscard]] bool operator()(const OpenEntry& left,
                                const OpenEntry& right) const noexcept {
    if (left.f != right.f) {
      return left.f > right.f;
    }
    if (left.h != right.h) {
      return left.h > right.h;
    }
    if (left.g != right.g) {
      return left.g > right.g;
    }
    if (left.offset != right.offset) {
      return left.offset > right.offset;
    }
    return left.insertion_order > right.insertion_order;
  }
};

[[nodiscard]] double NormalizeAngle(double angle) noexcept {
  return std::remainder(angle, 2.0 * 3.14159265358979323846);
}

[[nodiscard]] Quaternion YawQuaternion(const double yaw) noexcept {
  return Quaternion{.w = std::cos(0.5 * yaw), .z = std::sin(0.5 * yaw)};
}

[[nodiscard]] std::optional<GridIndex> WorldToCell(
    const SparseGridGeometry& geometry, const Point2 point) noexcept {
  if (!geometry.valid() || !std::isfinite(point.x) ||
      !std::isfinite(point.y)) {
    return std::nullopt;
  }
  const Vec3 origin = geometry.origin_m();
  const double x = std::floor((point.x - origin.x) / geometry.resolution_m());
  const double y = std::floor((point.y - origin.y) / geometry.resolution_m());
  if (!std::isfinite(x) || !std::isfinite(y) ||
      x < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      x >= static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
      y < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      y >= static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    return std::nullopt;
  }
  const GridIndex index{.x = static_cast<std::int64_t>(x),
                        .y = static_cast<std::int64_t>(y)};
  return geometry.Contains(index) ? std::optional<GridIndex>(index)
                                  : std::nullopt;
}

[[nodiscard]] Point2 CellCenter(const SparseGridGeometry& geometry,
                                const GridIndex index) noexcept {
  const Vec3 origin = geometry.origin_m();
  return Point2{
      .x = std::fma(static_cast<double>(index.x) + 0.5,
                    geometry.resolution_m(), origin.x),
      .y = std::fma(static_cast<double>(index.y) + 0.5,
                    geometry.resolution_m(), origin.y),
  };
}

[[nodiscard]] std::optional<std::int64_t> CheckedAdd(
    const std::int64_t value, const std::int64_t delta) noexcept {
  if ((delta < 0 &&
       value < std::numeric_limits<std::int64_t>::min() - delta) ||
      (delta > 0 &&
       value > std::numeric_limits<std::int64_t>::max() - delta)) {
    return std::nullopt;
  }
  return value + delta;
}

[[nodiscard]] bool SupportsSpin(const WheeledCapability& capability) noexcept {
  return std::any_of(
      capability.motion_primitives.begin(), capability.motion_primitives.end(),
      [](const WheelMotionPrimitive& primitive) {
        return primitive.kind == WheelPrimitiveKind::kSpinClockwise ||
               primitive.kind == WheelPrimitiveKind::kSpinCounterclockwise;
      });
}

[[nodiscard]] double Tangent(const Vec3& from, const Vec3& to,
                             const double fallback) noexcept {
  const double dx = to.x - from.x;
  const double dy = to.y - from.y;
  return std::hypot(dx, dy) > kGeometryEpsilon ? std::atan2(dy, dx)
                                                : fallback;
}

[[nodiscard]] bool ApplyPathOrientations(
    std::vector<PathPoint>& path, const double start_yaw,
    const std::optional<double> final_yaw, const SearchControl& control) {
  if (control.canceled() || control.expired()) {
    return false;
  }
  if (path.empty()) {
    return true;
  }
  path.front().pose.orientation = YawQuaternion(start_yaw);
  for (std::size_t i = 1U; i + 1U < path.size(); ++i) {
    if (control.canceled() || control.expired()) {
      return false;
    }
    path[i].pose.orientation = YawQuaternion(
        Tangent(path[i].pose.position_m, path[i + 1U].pose.position_m,
                start_yaw));
  }
  if (path.size() > 1U) {
    const double arrival = Tangent(path[path.size() - 2U].pose.position_m,
                                   path.back().pose.position_m, start_yaw);
    path.back().pose.orientation =
        YawQuaternion(final_yaw.value_or(arrival));
  } else {
    path.back().pose.orientation =
        YawQuaternion(final_yaw.value_or(start_yaw));
  }
  return !(control.canceled() || control.expired());
}

[[nodiscard]] std::optional<LocalPlanResult::Status> Interrupted(
    const SearchControl& control) {
  if (control.canceled()) {
    return LocalPlanResult::Status::kCanceled;
  }
  if (control.expired()) {
    return LocalPlanResult::Status::kTimeout;
  }
  return std::nullopt;
}

[[nodiscard]] bool ArrivalYawIsFeasible(
    const Vec3& from, const Point2 target, const double fallback_yaw,
    const double requested_yaw, const double tolerance_rad) noexcept {
  const double dx = target.x - from.x;
  const double dy = target.y - from.y;
  const double arrival = std::hypot(dx, dy) > kGeometryEpsilon
                             ? std::atan2(dy, dx)
                             : fallback_yaw;
  return std::abs(NormalizeAngle(arrival - requested_yaw)) <= tolerance_rad;
}

}  // namespace

WheelLocalPlanner::WheelLocalPlanner() = default;

WheelLocalPlanner::WheelLocalPlanner(WheeledCapability capability,
                                     WheelLocalPlannerConfig config)
    : capability_(std::move(capability)), config_(config) {
  if (config_.maximum_width_cells == 0U ||
      config_.maximum_height_cells == 0U ||
      config_.maximum_width_cells > kMaximumEnvelopeWidthCells ||
      config_.maximum_height_cells > kMaximumEnvelopeHeightCells ||
      !std::isfinite(config_.terminal_yaw_tolerance_rad) ||
      config_.terminal_yaw_tolerance_rad < 0.0) {
    throw std::invalid_argument("wheel local planner config is invalid");
  }
}

std::size_t WheelLocalPlanner::buffer_capacity_cells() const noexcept {
  return buffer_capacity_cells_;
}

std::size_t WheelLocalPlanner::buffer_allocation_count() const noexcept {
  return buffer_allocation_count_;
}

LocalPlanResult WheelLocalPlanner::Plan(
    const RequestLocalPlanningView& view, const Pose2& start,
    const LocalTarget& requested_target, const SearchDeadline deadline,
    const StopToken& stop) {
  LocalTarget target = requested_target;
  const auto region = target.region;
  LocalPlanResult result;
  if (region) result.reason_code = "LOCAL_NO_PROGRESS";
  const SearchControl control{
      .deadline = deadline, .stop_token = stop, .now = config_.now};
  if (const auto interrupted = Interrupted(control)) {
    result.status = *interrupted;
    return result;
  }
  if (!std::isfinite(start.position_m.x) ||
      !std::isfinite(start.position_m.y) || !std::isfinite(start.yaw_rad) ||
      !std::isfinite(target.center.x) || !std::isfinite(target.center.y) ||
      !std::isfinite(target.position_tolerance_m) ||
      target.position_tolerance_m < 0.0 ||
      (target.terminal_yaw_rad &&
       !std::isfinite(*target.terminal_yaw_rad))) {
    return result;
  }

  const SparseGridGeometry& geometry = view.geometry();
  const std::size_t width = geometry.width();
  const std::size_t height = geometry.height();
  if (!geometry.valid() || width == 0U || height == 0U ||
      width > config_.maximum_width_cells ||
      height > config_.maximum_height_cells ||
      width > std::numeric_limits<std::size_t>::max() / height) {
    return result;
  }
  const std::size_t cell_count = width * height;
  if (buffer_capacity_cells_ < cell_count) {
    g_cost_.resize(cell_count);
    parent_.resize(cell_count);
    state_.resize(cell_count);
    phase_.resize(cell_count);
    generation_.resize(cell_count, 0U);
    buffer_capacity_cells_ = cell_count;
    ++buffer_allocation_count_;
  }
  if (++current_generation_ == 0U) {
    std::fill(generation_.begin(), generation_.end(), 0U);
    current_generation_ = 1U;
  }

  const auto start_index = WorldToCell(geometry, start.position_m);
  auto goal_index = WorldToCell(geometry, target.center);
  if (!start_index || (!region && (!goal_index || !view.CanBeEndpoint(*goal_index)))) {
    return result;
  }
  const LocalCellSource start_source = view.Source(*start_index);
  if (start_source != LocalCellSource::kEvidenceFree &&
      start_source != LocalCellSource::kStartAssumedFree) {
    return result;
  }
  const StartPhase start_phase =
      start_source == LocalCellSource::kStartAssumedFree
          ? StartPhase::kStartPrefix
          : StartPhase::kNormal;
  const GridIndex minimum = geometry.min_inclusive();
  const auto offset_of = [&](const GridIndex index) {
    return static_cast<std::size_t>(index.y - minimum.y) * width +
           static_cast<std::size_t>(index.x - minimum.x);
  };
  const auto index_of = [&](const std::size_t offset) {
    return GridIndex{.x = minimum.x +
                          static_cast<std::int64_t>(offset % width),
                     .y = minimum.y +
                          static_cast<std::int64_t>(offset / width)};
  };
  const auto initialize = [&](const std::size_t offset) {
    if (generation_[offset] != current_generation_) {
      generation_[offset] = current_generation_;
      g_cost_[offset] = std::numeric_limits<long double>::infinity();
      parent_[offset] = kInvalidParent;
      state_[offset] = 0U;
      phase_[offset] = StartPhase::kNormal;
    }
  };
  const auto heuristic = [&](const GridIndex index) {
    if (region) return region->LowerBound(CellCenter(geometry, index));
    const double dx = static_cast<double>(index.x - goal_index->x);
    const double dy = static_cast<double>(index.y - goal_index->y);
    return std::hypot(dx, dy) * geometry.resolution_m();
  };
  const std::size_t start_offset = offset_of(*start_index);
  const bool supports_spin = SupportsSpin(capability_);
  const auto terminal_feasible = [&](const std::size_t predecessor_offset,
                                     const LocalTarget& terminal_target) {
    if (!terminal_target.is_final_goal || !terminal_target.terminal_yaw_rad) {
      return true;
    }
    if (supports_spin) {
      return true;
    }
    const Point2 from = predecessor_offset == start_offset
                            ? start.position_m
                            : CellCenter(geometry, index_of(predecessor_offset));
    return ArrivalYawIsFeasible(
        {.x = from.x, .y = from.y}, terminal_target.center, start.yaw_rad,
        *terminal_target.terminal_yaw_rad, config_.terminal_yaw_tolerance_rad);
  };

  initialize(start_offset);
  g_cost_[start_offset] = 0.0;
  phase_[start_offset] = start_phase;
  std::vector<OpenEntry> open_storage;
  open_storage.reserve(cell_count);
  std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenEntryLater> open(
      OpenEntryLater{}, std::move(open_storage));
  std::uint64_t insertion_order = 0U;
  open.push({.offset = start_offset,
             .g = 0.0,
             .h = heuristic(*start_index),
             .f = heuristic(*start_index),
             .insertion_order = insertion_order++});
  ++result.statistics.generated_states;
  result.statistics.open_peak = open.size();

  constexpr std::array<std::pair<std::int64_t, std::int64_t>, 8U> kNeighbors{{
      {-1, 0}, {0, -1}, {1, 0}, {0, 1},
      {-1, -1}, {1, -1}, {-1, 1}, {1, 1},
  }};
  std::optional<std::size_t> found;
  bool reached_forward_unknown_boundary = false;
  long double best_goal_cost = std::numeric_limits<long double>::infinity();
  while (!open.empty()) {
    if (region && found && open.top().f >= best_goal_cost) break;
    if (const auto interrupted = Interrupted(control)) {
      result.status = *interrupted;
      return result;
    }
    const OpenEntry current = open.top();
    open.pop();
    initialize(current.offset);
    if (state_[current.offset] == 2U ||
        current.g > g_cost_[current.offset]) {
      continue;
    }
    state_[current.offset] = 2U;
    ++result.statistics.expanded_states;
    const GridIndex current_index = index_of(current.offset);
    const std::size_t predecessor_offset =
        parent_[current.offset] == kInvalidParent
            ? current.offset
            : parent_[current.offset];
    if (region) {
      // Only this platform's certified expanded states establish reachable
      // missing evidence. The window-wide metadata does not establish it.
      reached_forward_unknown_boundary |=
          view.CanBeEndpoint(current_index) &&
          region->HasForwardUnknownBoundaryAt(current_index);
      const auto* candidate = region->At(current_index);
      if (candidate &&
          (candidate->target.is_final_goal || !WithinArrivalTolerance(
              std::hypot(candidate->target.center.x-start.position_m.x,
                         candidate->target.center.y-start.position_m.y),
              requested_target.position_tolerance_m)) &&
          view.CanBeEndpoint(current_index) &&
          terminal_feasible(predecessor_offset, candidate->target)) {
        const long double cost = current.g + candidate->remaining_cost;
        if (cost < best_goal_cost) {
          best_goal_cost = cost;
          found = current.offset;
          target = candidate->target;
        }
      }
    } else if (current_index == *goal_index &&
               terminal_feasible(predecessor_offset, target)) {
      found = current.offset;
      break;
    }

    for (const auto [dx, dy] : kNeighbors) {
      const auto next_x = CheckedAdd(current_index.x, dx);
      const auto next_y = CheckedAdd(current_index.y, dy);
      if (!next_x || !next_y) {
        continue;
      }
      const GridIndex next{.x = *next_x, .y = *next_y};
      if (!geometry.Contains(next)) {
        continue;
      }
      const auto next_phase = view.AdvancePhase(phase_[current.offset], next);
      if (!next_phase) {
        continue;
      }
      if (dx != 0 && dy != 0) {
        const GridIndex side_x{.x = *next_x, .y = current_index.y};
        const GridIndex side_y{.x = current_index.x, .y = *next_y};
        if (!view.AdvancePhase(phase_[current.offset], side_x) ||
            !view.AdvancePhase(phase_[current.offset], side_y)) {
          continue;
        }
      }
      // Invalid terminal edges are not inserted, so the search continues to
      // other legal entering edges instead of accepting a position-only goal.
      const auto* next_candidate = region ? region->At(next) : nullptr;
      if ((!region && next == *goal_index && !terminal_feasible(current.offset, target)) ||
          (next_candidate && next_candidate->target.is_final_goal &&
           !terminal_feasible(current.offset, next_candidate->target))) {
        continue;
      }
      const std::size_t next_offset = offset_of(next);
      initialize(next_offset);
      if (state_[next_offset] == 2U) {
        continue;
      }
      const double distance =
          (dx != 0 && dy != 0 ? kDiagonalDistance : 1.0) *
          geometry.resolution_m();
      const long double step_cost =
          static_cast<long double>(distance) *
          (1.0L + static_cast<long double>(view.TraversalCost(next)));
      const long double candidate_g = g_cost_[current.offset] + step_cost;
      if (!std::isfinite(candidate_g) || candidate_g >= g_cost_[next_offset]) {
        continue;
      }
      g_cost_[next_offset] = candidate_g;
      parent_[next_offset] = current.offset;
      phase_[next_offset] = *next_phase;
      state_[next_offset] = 1U;
      const double h = heuristic(next);
      open.push({.offset = next_offset,
                 .g = candidate_g,
                 .h = h,
                 .f = candidate_g + h,
                 .insertion_order = insertion_order++});
      result.statistics.open_peak =
          std::max(result.statistics.open_peak, open.size());
      ++result.statistics.generated_states;
    }
  }
  if (!found) {
    if (region && reached_forward_unknown_boundary) {
      result.reason_code = "LOCAL_WAITING_FOR_MAP";
    }
    return result;
  }

  std::vector<std::size_t> offsets;
  for (std::size_t cursor = *found; cursor != kInvalidParent;
       cursor = parent_[cursor]) {
    if (const auto interrupted = Interrupted(control)) {
      result.status = *interrupted;
      result.raw_path.clear();
      result.path.clear();
      return result;
    }
    offsets.push_back(cursor);
  }
  std::reverse(offsets.begin(), offsets.end());
  if (offsets.empty()) {
    return result;
  }
  result.raw_path.reserve(offsets.size() + 1U);
  for (std::size_t i = 0U; i < offsets.size(); ++i) {
    if (const auto interrupted = Interrupted(control)) {
      result.status = *interrupted;
      result.raw_path.clear();
      result.path.clear();
      return result;
    }
    const bool first = i == 0U;
    const bool last = i + 1U == offsets.size();
    const Point2 point = first ? start.position_m
                              : (last ? target.center
                                      : CellCenter(geometry, index_of(offsets[i])));
    result.raw_path.push_back(PathPoint{
        .pose = {.position_m = {.x = point.x, .y = point.y, .z = 0.0}},
        .phase = phase_[offsets[i]],
    });
  }
  if (result.raw_path.size() == 1U && start.position_m != target.center) {
    result.raw_path.push_back(PathPoint{
        .pose = {.position_m = {
            .x = target.center.x, .y = target.center.y, .z = 0.0}},
        .phase = phase_[*found],
    });
  }
  if (result.raw_path.empty()) {
    return result;
  }
  const std::optional<double> requested_yaw =
      target.is_final_goal ? target.terminal_yaw_rad : std::nullopt;
  const auto postprocess_started = SteadyClock::now();
  if (target.is_final_goal && target.terminal_yaw_rad && !supports_spin) {
    const Vec3& from = result.raw_path.size() > 1U
                           ? result.raw_path[result.raw_path.size() - 2U]
                                 .pose.position_m
                           : result.raw_path.front().pose.position_m;
    if (!ArrivalYawIsFeasible(from, target.center, start.yaw_rad,
                              *target.terminal_yaw_rad,
                              config_.terminal_yaw_tolerance_rad)) {
      // Every invalid goal edge was filtered before insertion. Reaching this
      // branch means no certifiable raw terminal segment can be published.
      result.raw_path.clear();
      return result;
    }
  }
  if (!ApplyPathOrientations(result.raw_path, start.yaw_rad, requested_yaw,
                             control)) {
    result.status = Interrupted(control).value_or(
        LocalPlanResult::Status::kNoPath);
    if (result.status == LocalPlanResult::Status::kNoPath) {
      result.reason_code = "PATH_POSTPROCESS_FAILED";
    }
    result.raw_path.clear();
    return result;
  }
  result.path = SimplifyPhaseAwarePath(view, result.raw_path, control);
  if (const auto interrupted = Interrupted(control)) {
    result.status = *interrupted;
    result.raw_path.clear();
    result.path.clear();
    return result;
  }
  if (result.path.empty() ||
      !ApplyPathOrientations(result.path, start.yaw_rad, requested_yaw,
                             control)) {
    result.status = Interrupted(control).value_or(
        LocalPlanResult::Status::kNoPath);
    if (result.status == LocalPlanResult::Status::kNoPath) {
      result.reason_code = "PATH_POSTPROCESS_FAILED";
    }
    result.raw_path.clear();
    result.path.clear();
    return result;
  }

  if (target.is_final_goal && target.terminal_yaw_rad && !supports_spin &&
      result.path.size() > 1U) {
    const double simplified_arrival =
        Tangent(result.path[result.path.size() - 2U].pose.position_m,
                result.path.back().pose.position_m, start.yaw_rad);
    if (std::abs(NormalizeAngle(simplified_arrival -
                                *target.terminal_yaw_rad)) >
        config_.terminal_yaw_tolerance_rad) {
      result.path = result.raw_path;
    }
  }

  if (target.is_final_goal && target.terminal_yaw_rad && supports_spin) {
    if (const auto interrupted = Interrupted(control)) {
      result.status = *interrupted;
      result.raw_path.clear();
      result.path.clear();
      return result;
    }
    const double arrival = result.path.size() > 1U
                               ? Tangent(result.path[result.path.size() - 2U]
                                             .pose.position_m,
                                         result.path.back().pose.position_m,
                                         start.yaw_rad)
                               : start.yaw_rad;
    if (std::abs(NormalizeAngle(arrival - *target.terminal_yaw_rad)) >
        config_.terminal_yaw_tolerance_rad) {
      result.path.back().pose.orientation = YawQuaternion(arrival);
      PathPoint spin = result.path.back();
      spin.pose.orientation = YawQuaternion(*target.terminal_yaw_rad);
      result.path.push_back(spin);
    }
  }
  if (const auto interrupted = Interrupted(control)) {
    result.status = *interrupted;
    result.raw_path.clear();
    result.path.clear();
    return result;
  }
  result.reason_code.clear();
  result.status = LocalPlanResult::Status::kPlanFound;
  result.reaches_final_goal = target.is_final_goal;
  result.postprocess_elapsed = SteadyClock::now() - postprocess_started;
  return result;
}

}  // namespace lunar::incremental_navigation
