#include "lunar_incremental_navigation_core/legged_local_planner.hpp"

#include "lunar_incremental_navigation_core/local_planning_window.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lunar::incremental_navigation {
namespace {

constexpr std::size_t kStatesPerCell = 4U;
constexpr std::uint32_t kInvalidState =
    std::numeric_limits<std::uint32_t>::max();
constexpr std::size_t kMaximumLocalCells =
    kMaximumLocalPlanningWindowAxisCells *
    kMaximumLocalPlanningWindowAxisCells;
constexpr double kTwoPi = 2.0 * std::numbers::pi;
constexpr double kComparisonTolerance = 1.0e-9;
constexpr double kGeometryTolerance = 1.0e-12;
constexpr std::size_t kMaximumTerminalSpinStates = 128U;
constexpr std::size_t kMaximumTerminalSpinDepth = 32U;

enum class WorkStatus : std::uint8_t {
  kReady,
  kTimeout,
  kCanceled,
  kInvalid,
};

struct OpenEntry final {
  double f{};
  double g{};
  std::uint32_t state{};
  std::uint64_t sequence{};
};

struct OpenLater final {
  [[nodiscard]] bool operator()(const OpenEntry& left,
                                const OpenEntry& right) const noexcept {
    if (left.f != right.f) {
      return left.f > right.f;
    }
    if (left.g != right.g) {
      return left.g > right.g;
    }
    if (left.state != right.state) {
      return left.state > right.state;
    }
    return left.sequence > right.sequence;
  }
};

struct DirectedEdgeCertificate final {
  bool feasible{};
  bool uses_assumed_support{};
  double cost{};
  double arrival_yaw_rad{};
};

struct EdgeLookup final {
  WorkStatus status{WorkStatus::kReady};
  DirectedEdgeCertificate certificate;
};

struct TerminalYawCertificate final {
  bool feasible{};
  bool requires_spin{};
  double yaw_rad{};
  double cost{};
};

struct TerminalCertificate final {
  bool feasible{};
  double cost{};
  bool uses_assumed_support{};
  bool connects_exact_target{};
  bool requires_terminal_spin{};
  double arrival_yaw_rad{};
  double terminal_yaw_rad{};
  bool yaw_rejected{};
};

struct TerminalCacheEntry final {
  bool evaluated{};
  TerminalCertificate certificate;
};

struct RawVertex final {
  PathPoint point;
  std::uint32_t state{};
  bool lattice_position{};
};

struct SimplificationResult final {
  WorkStatus status{WorkStatus::kReady};
  std::vector<RawVertex> vertices;
};

[[nodiscard]] bool Finite(const Pose2& pose) noexcept {
  return std::isfinite(pose.position_m.x) &&
         std::isfinite(pose.position_m.y) && std::isfinite(pose.yaw_rad);
}

[[nodiscard]] bool Finite(const LocalTarget& target) noexcept {
  return std::isfinite(target.center.x) && std::isfinite(target.center.y) &&
         std::isfinite(target.position_tolerance_m) &&
         target.position_tolerance_m >= 0.0 &&
         (!target.terminal_yaw_rad.has_value() ||
          std::isfinite(*target.terminal_yaw_rad));
}

[[nodiscard]] double NormalizeYaw(const double yaw) noexcept {
  return std::remainder(yaw, kTwoPi);
}

[[nodiscard]] double AngleError(const double left,
                                const double right) noexcept {
  return std::abs(NormalizeYaw(left - right));
}

[[nodiscard]] Quaternion YawQuaternion(const double yaw) noexcept {
  return Quaternion{.w = std::cos(yaw / 2.0),
                    .z = std::sin(yaw / 2.0)};
}

[[nodiscard]] std::optional<GridIndex> WorldToCell(
    const SparseGridGeometry& geometry, const Point2 point) noexcept {
  if (!geometry.valid() || !std::isfinite(point.x) ||
      !std::isfinite(point.y)) {
    return std::nullopt;
  }
  const Vec3 origin = geometry.origin_m();
  const double resolution = geometry.resolution_m();
  const double x = std::floor((point.x - origin.x) / resolution);
  const double y = std::floor((point.y - origin.y) / resolution);
  if (x < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      x > static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
      y < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      y > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    return std::nullopt;
  }
  const GridIndex cell{.x = static_cast<std::int64_t>(x),
                       .y = static_cast<std::int64_t>(y)};
  return geometry.Contains(cell) ? std::optional<GridIndex>{cell}
                                 : std::nullopt;
}

[[nodiscard]] Point2 CellCenter(const SparseGridGeometry& geometry,
                                const GridIndex cell) noexcept {
  const Vec3 origin = geometry.origin_m();
  return Point2{
      .x = std::fma(static_cast<double>(cell.x) + 0.5,
                    geometry.resolution_m(), origin.x),
      .y = std::fma(static_cast<double>(cell.y) + 0.5,
                    geometry.resolution_m(), origin.y),
  };
}

[[nodiscard]] bool IsCellCenter(const SparseGridGeometry& geometry,
                                const GridIndex cell,
                                const Point2 point) noexcept {
  const Point2 center = CellCenter(geometry, cell);
  return std::abs(point.x - center.x) <= kGeometryTolerance &&
         std::abs(point.y - center.y) <= kGeometryTolerance;
}

[[nodiscard]] std::size_t CellOffset(const SparseGridGeometry& geometry,
                                     const GridIndex cell) noexcept {
  const GridIndex minimum = geometry.min_inclusive();
  return static_cast<std::size_t>(cell.y - minimum.y) * geometry.width() +
         static_cast<std::size_t>(cell.x - minimum.x);
}

[[nodiscard]] GridIndex CellFromOffset(const SparseGridGeometry& geometry,
                                       const std::size_t offset) noexcept {
  const GridIndex minimum = geometry.min_inclusive();
  return GridIndex{
      .x = minimum.x +
           static_cast<std::int64_t>(offset % geometry.width()),
      .y = minimum.y +
           static_cast<std::int64_t>(offset / geometry.width()),
  };
}

[[nodiscard]] std::uint32_t SearchState(const std::size_t cell_offset,
                                        const StartPhase phase,
                                        const bool at_start_anchor) noexcept {
  return static_cast<std::uint32_t>(
      cell_offset * kStatesPerCell +
      (phase == StartPhase::kNormal ? std::size_t{2U} : std::size_t{0U}) +
      (at_start_anchor ? std::size_t{1U} : std::size_t{0U}));
}

[[nodiscard]] std::size_t SearchCellOffset(
    const std::uint32_t state) noexcept {
  return static_cast<std::size_t>(state) / kStatesPerCell;
}

[[nodiscard]] StartPhase SearchPhase(const std::uint32_t state) noexcept {
  return state % kStatesPerCell < 2U ? StartPhase::kStartPrefix
                                     : StartPhase::kNormal;
}

[[nodiscard]] bool SearchAtStartAnchor(
    const std::uint32_t state) noexcept {
  return state % 2U != 0U;
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

[[nodiscard]] WorkStatus CheckWork(const SearchDeadline deadline,
                                   const StopToken& stop) noexcept {
  if (stop.stop_requested()) {
    return WorkStatus::kCanceled;
  }
  return SteadyClock::now() >= deadline ? WorkStatus::kTimeout
                                        : WorkStatus::kReady;
}

[[nodiscard]] LocalPlanResult StoppedResult(const WorkStatus status,
                                            SearchStatistics statistics = {}) {
  LocalPlanResult result;
  result.statistics = statistics;
  result.status = status == WorkStatus::kCanceled
                      ? LocalPlanResult::Status::kCanceled
                      : status == WorkStatus::kTimeout
                            ? LocalPlanResult::Status::kTimeout
                            : LocalPlanResult::Status::kNoPath;
  return result;
}

[[nodiscard]] std::vector<GridIndex> SegmentCells(
    const SparseGridGeometry& geometry, const Point2 source,
    const Point2 target) {
  const auto source_cell = WorldToCell(geometry, source);
  const auto target_cell = WorldToCell(geometry, target);
  if (!source_cell.has_value() || !target_cell.has_value()) {
    return {};
  }
  std::vector<GridIndex> cells;
  cells.reserve(static_cast<std::size_t>(
                    std::abs(target_cell->x - source_cell->x) +
                    std::abs(target_cell->y - source_cell->y)) *
                    3U +
                1U);
  const auto append = [&](const GridIndex cell) {
    if (!geometry.Contains(cell)) {
      return false;
    }
    if (std::find(cells.begin(), cells.end(), cell) == cells.end()) {
      cells.push_back(cell);
    }
    return true;
  };
  GridIndex current = *source_cell;
  if (!append(current)) {
    return {};
  }
  const double dx = target.x - source.x;
  const double dy = target.y - source.y;
  const std::int64_t step_x = dx > 0.0 ? 1 : dx < 0.0 ? -1 : 0;
  const std::int64_t step_y = dy > 0.0 ? 1 : dy < 0.0 ? -1 : 0;
  const Vec3 origin = geometry.origin_m();
  const double resolution = geometry.resolution_m();
  const auto first_boundary = [&](const std::int64_t cell,
                                  const std::int64_t step,
                                  const double axis_origin) {
    const std::int64_t boundary_cell = step > 0 ? cell + 1 : cell;
    return std::fma(static_cast<double>(boundary_cell), resolution,
                    axis_origin);
  };
  double t_max_x = step_x == 0
                       ? std::numeric_limits<double>::infinity()
                       : (first_boundary(current.x, step_x, origin.x) -
                          source.x) /
                             dx;
  double t_max_y = step_y == 0
                       ? std::numeric_limits<double>::infinity()
                       : (first_boundary(current.y, step_y, origin.y) -
                          source.y) /
                             dy;
  const double t_delta_x =
      step_x == 0 ? std::numeric_limits<double>::infinity()
                  : resolution / std::abs(dx);
  const double t_delta_y =
      step_y == 0 ? std::numeric_limits<double>::infinity()
                  : resolution / std::abs(dy);
  while (current != *target_cell) {
    if (std::abs(t_max_x - t_max_y) <= kComparisonTolerance) {
      const GridIndex side_x{.x = current.x + step_x, .y = current.y};
      const GridIndex side_y{.x = current.x, .y = current.y + step_y};
      if (!append(side_x) || !append(side_y)) {
        return {};
      }
      current.x += step_x;
      current.y += step_y;
      t_max_x += t_delta_x;
      t_max_y += t_delta_y;
    } else if (t_max_x < t_max_y) {
      current.x += step_x;
      t_max_x += t_delta_x;
    } else {
      current.y += step_y;
      t_max_y += t_delta_y;
    }
    if (!append(current)) {
      return {};
    }
  }
  return cells;
}

[[nodiscard]] bool ElevationStepAndGapFeasible(
    const ElevationSnapshot& elevation, const std::vector<GridIndex>& cells,
    const double maximum_step_m, const double maximum_gap_m,
    const bool initial_support_assumed) noexcept {
  std::optional<ElevationRange> previous;
  std::size_t missing_cells = 0U;
  for (std::size_t index = 0U; index < cells.size(); ++index) {
    const std::optional<ElevationRange> current =
        elevation.ElevationRangeAt(cells[index]);
    if (!current.has_value()) {
      if (initial_support_assumed && index == 0U) {
        continue;
      }
      ++missing_cells;
      continue;
    }
    if (static_cast<double>(missing_cells) *
            elevation.geometry().resolution_m() >
        maximum_gap_m + kComparisonTolerance) {
      return false;
    }
    missing_cells = 0U;
    if (previous.has_value()) {
      const double separated = std::max(
          {0.0,
           static_cast<double>(current->min_m) - previous->max_m,
           static_cast<double>(previous->min_m) - current->max_m});
      if (separated > maximum_step_m + kComparisonTolerance) {
        return false;
      }
    }
    previous = current;
  }
  return previous.has_value() && missing_cells == 0U;
}

[[nodiscard]] bool PrimitiveValid(
    const LeggedBodyPrimitive& primitive) noexcept {
  return !primitive.primitive_id.empty() &&
         std::isfinite(primitive.body_frame_displacement_m.x) &&
         std::isfinite(primitive.body_frame_displacement_m.y) &&
         std::isfinite(primitive.body_frame_displacement_m.z) &&
         std::isfinite(primitive.yaw_change_rad);
}

[[nodiscard]] bool CapabilityValid(
    const LeggedCapability& capability) noexcept {
  return std::isfinite(capability.maximum_step_height_m) &&
         capability.maximum_step_height_m >= 0.0 &&
         std::isfinite(capability.maximum_gap_width_m) &&
         capability.maximum_gap_width_m >= 0.0 &&
         !capability.motion_primitives.empty() &&
         std::all_of(capability.motion_primitives.begin(),
                     capability.motion_primitives.end(), PrimitiveValid);
}

[[nodiscard]] double SaturatingAdd(const double left,
                                   const double right) noexcept {
  constexpr double largest = std::numeric_limits<double>::max();
  if (left >= largest - right) {
    return largest;
  }
  const double sum = left + right;
  return std::isfinite(sum) ? sum : largest;
}

[[nodiscard]] double SaturatingTerrainEdgeCost(
    const double geometric_length, const double traversal_cost,
    const double rotation_cost = 0.0) noexcept {
  constexpr long double largest =
      static_cast<long double>(std::numeric_limits<double>::max());
  const long double weighted =
      static_cast<long double>(geometric_length) *
          (1.0L + static_cast<long double>(traversal_cost)) +
      static_cast<long double>(rotation_cost);
  return weighted >= largest ? std::numeric_limits<double>::max()
                             : static_cast<double>(weighted);
}

[[nodiscard]] Point2 PositionForState(
    const RequestLocalPlanningView& view, const std::uint32_t state,
    const Pose2& start) noexcept {
  return SearchAtStartAnchor(state)
             ? start.position_m
             : CellCenter(view.geometry(),
                          CellFromOffset(view.geometry(),
                                         SearchCellOffset(state)));
}

[[nodiscard]] double Tangent(const Point2 source, const Point2 target,
                             const double fallback) noexcept {
  const double dx = target.x - source.x;
  const double dy = target.y - source.y;
  return std::hypot(dx, dy) > kGeometryTolerance ? std::atan2(dy, dx)
                                                  : fallback;
}

[[nodiscard]] WorkStatus SupportsTranslation(
    const LeggedCapability& capability, const double requested_length,
    const double endpoint_quantization_allowance, const SearchDeadline deadline,
    const StopToken& stop, SearchStatistics& statistics, bool& supported) {
  supported = false;
  for (const LeggedBodyPrimitive& primitive : capability.motion_primitives) {
    ++statistics.evaluated_transitions;
    if (const WorkStatus status = CheckWork(deadline, stop);
        status != WorkStatus::kReady) {
      return status;
    }
    const double available_length = std::hypot(
        primitive.body_frame_displacement_m.x,
        primitive.body_frame_displacement_m.y);
    if (available_length <= kGeometryTolerance ||
        std::abs(primitive.body_frame_displacement_m.z) >
            capability.maximum_step_height_m + kComparisonTolerance ||
        available_length + endpoint_quantization_allowance +
                kComparisonTolerance <
            requested_length) {
      continue;
    }
    supported = true;
    return WorkStatus::kReady;
  }
  return WorkStatus::kReady;
}

class LeggedDirectedEdgeCache final {
 public:
  [[nodiscard]] EdgeLookup Certify(
      const RequestLocalPlanningView& view, const LeggedCapability& capability,
      const std::uint32_t source_state, const Point2 source_point,
      const GridIndex target_cell, const Point2 target_point,
      const bool target_is_lattice_point, const SearchDeadline deadline,
      const StopToken& stop, SearchStatistics& statistics) {
    if (const WorkStatus status = CheckWork(deadline, stop);
        status != WorkStatus::kReady) {
      return {.status = status};
    }
    const std::uint64_t key =
        (static_cast<std::uint64_t>(source_state) << 32U) |
        (static_cast<std::uint64_t>(
             CellOffset(view.geometry(), target_cell))
         << 1U) |
        static_cast<std::uint64_t>(target_is_lattice_point);
    if (const auto found = entries_.find(key); found != entries_.end()) {
      return {.certificate = found->second};
    }

    DirectedEdgeCertificate certificate;
    const GridIndex source_cell =
        CellFromOffset(view.geometry(), SearchCellOffset(source_state));
    const LocalCellSource source_kind = view.Source(source_cell);
    const bool source_is_start_prefix =
        SearchPhase(source_state) == StartPhase::kStartPrefix;
    const bool source_assumed =
        source_kind == LocalCellSource::kStartAssumedFree;
    const LocalCellSource target_kind = view.Source(target_cell);
    if ((source_kind != LocalCellSource::kEvidenceFree &&
         !(source_is_start_prefix && source_assumed)) ||
        (target_kind != LocalCellSource::kEvidenceFree &&
         !(source_is_start_prefix &&
           target_kind == LocalCellSource::kStartAssumedFree))) {
      entries_.emplace(key, certificate);
      return {.certificate = certificate};
    }

    const double dx = target_point.x - source_point.x;
    const double dy = target_point.y - source_point.y;
    const double distance = std::hypot(dx, dy);
    if (distance <= kGeometryTolerance) {
      entries_.emplace(key, certificate);
      return {.certificate = certificate};
    }
    const double endpoint_quantization_allowance =
        target_is_lattice_point
            ? 0.5 * std::numbers::sqrt2 *
                  view.geometry().resolution_m()
            : 0.0;
    bool primitive_supports_edge = false;
    if (const WorkStatus status = SupportsTranslation(
            capability, distance, endpoint_quantization_allowance, deadline,
            stop, statistics, primitive_supports_edge);
        status != WorkStatus::kReady) {
      return {.status = status};
    }
    if (!primitive_supports_edge) {
      entries_.emplace(key, certificate);
      return {.certificate = certificate};
    }

    const std::vector<GridIndex> cells =
        SegmentCells(view.geometry(), source_point, target_point);
    if (cells.empty()) {
      entries_.emplace(key, certificate);
      return {.certificate = certificate};
    }
    bool uses_assumed_support = false;
    for (const GridIndex cell : cells) {
      switch (view.Source(cell)) {
        case LocalCellSource::kEvidenceFree:
          break;
        case LocalCellSource::kStartAssumedFree:
          if (!source_is_start_prefix) {
            entries_.emplace(key, certificate);
            return {.certificate = certificate};
          }
          uses_assumed_support = true;
          break;
        case LocalCellSource::kUnknown:
        case LocalCellSource::kBlocked:
          entries_.emplace(key, certificate);
          return {.certificate = certificate};
      }
    }
    if (!uses_assumed_support && !ElevationStepAndGapFeasible(
                                     *view.base()->elevation(), cells,
                                     capability.maximum_step_height_m,
                                     capability.maximum_gap_width_m, false)) {
      entries_.emplace(key, certificate);
      return {.certificate = certificate};
    }

    certificate.feasible = true;
    certificate.uses_assumed_support = uses_assumed_support;
    certificate.cost = SaturatingTerrainEdgeCost(
        distance, view.TraversalCost(target_cell));
    certificate.arrival_yaw_rad = std::atan2(dy, dx);
    entries_.emplace(key, certificate);
    return {.certificate = certificate};
  }

 private:
  std::unordered_map<std::uint64_t, DirectedEdgeCertificate> entries_;
};

[[nodiscard]] TerminalYawCertificate CertifyTerminalYaw(
    const LeggedCapability& capability, const LeggedLocalPlannerConfig& config,
    const bool is_final_goal, const std::optional<double>& requested_yaw,
    const double arrival_yaw, const SearchDeadline deadline,
    const StopToken& stop, SearchStatistics& statistics, WorkStatus& status) {
  if (!is_final_goal || !requested_yaw.has_value()) {
    return {.feasible = true, .yaw_rad = arrival_yaw};
  }
  if (AngleError(arrival_yaw, *requested_yaw) <=
      config.terminal_yaw_tolerance_rad) {
    return {.feasible = true, .yaw_rad = arrival_yaw};
  }

  std::vector<double> spin_deltas;
  spin_deltas.reserve(capability.motion_primitives.size());
  for (const LeggedBodyPrimitive& primitive : capability.motion_primitives) {
    ++statistics.evaluated_transitions;
    status = CheckWork(deadline, stop);
    if (status != WorkStatus::kReady) {
      return {};
    }
    if (std::hypot(primitive.body_frame_displacement_m.x,
                   primitive.body_frame_displacement_m.y) >
            kGeometryTolerance ||
        std::abs(primitive.body_frame_displacement_m.z) >
            capability.maximum_step_height_m + kComparisonTolerance) {
      continue;
    }
    const double delta = NormalizeYaw(primitive.yaw_change_rad);
    if (std::abs(delta) <= kGeometryTolerance ||
        std::any_of(spin_deltas.begin(), spin_deltas.end(),
                    [&](const double known) {
                      return AngleError(known, delta) <=
                             kGeometryTolerance;
                    })) {
      continue;
    }
    spin_deltas.push_back(delta);
  }
  if (spin_deltas.empty()) {
    return {};
  }

  struct SpinState final {
    double yaw_rad{};
    double cost{};
    std::size_t depth{};
  };
  std::vector<SpinState> queue;
  std::vector<double> visited;
  queue.reserve(kMaximumTerminalSpinStates);
  visited.reserve(kMaximumTerminalSpinStates);
  queue.push_back({.yaw_rad = arrival_yaw});
  visited.push_back(arrival_yaw);
  for (std::size_t cursor = 0U;
       cursor < queue.size() && queue.size() < kMaximumTerminalSpinStates;
       ++cursor) {
    const SpinState current = queue[cursor];
    if (current.depth >= kMaximumTerminalSpinDepth) {
      continue;
    }
    for (const double delta : spin_deltas) {
      ++statistics.evaluated_transitions;
      status = CheckWork(deadline, stop);
      if (status != WorkStatus::kReady) {
        return {};
      }
      const double next_yaw = NormalizeYaw(current.yaw_rad + delta);
      const double next_cost = SaturatingAdd(current.cost, std::abs(delta));
      if (AngleError(next_yaw, *requested_yaw) <=
          config.terminal_yaw_tolerance_rad) {
        return {.feasible = true,
                .requires_spin = true,
                .yaw_rad = next_yaw,
                .cost = next_cost};
      }
      if (std::any_of(visited.begin(), visited.end(),
                      [&](const double known) {
                        return AngleError(known, next_yaw) <=
                               kGeometryTolerance;
                      })) {
        continue;
      }
      visited.push_back(next_yaw);
      queue.push_back({.yaw_rad = next_yaw,
                       .cost = next_cost,
                       .depth = current.depth + 1U});
      if (queue.size() >= kMaximumTerminalSpinStates) {
        break;
      }
    }
  }
  return {};
}

[[nodiscard]] TerminalCertificate CertifyTerminal(
    const RequestLocalPlanningView& view, const LeggedCapability& capability,
    const LeggedLocalPlannerConfig& config,
    LeggedDirectedEdgeCache& edge_cache, const std::uint32_t source_state,
    const Point2 source_point, const double source_arrival_yaw,
    const LocalTarget& target, const GridIndex target_cell,
    const SearchDeadline deadline, const StopToken& stop,
    SearchStatistics& statistics, WorkStatus& status) {
  if (!view.CanBeEndpoint(target_cell)) {
    return {};
  }
  const GridIndex source_cell =
      CellFromOffset(view.geometry(), SearchCellOffset(source_state));
  const bool source_is_start_prefix =
      SearchPhase(source_state) == StartPhase::kStartPrefix;
  const bool source_assumed =
      view.Source(source_cell) == LocalCellSource::kStartAssumedFree;
  if (view.Source(source_cell) != LocalCellSource::kEvidenceFree &&
      !(source_is_start_prefix && source_assumed)) {
    return {};
  }
  const double distance = std::hypot(target.center.x - source_point.x,
                                     target.center.y - source_point.y);
  if (distance <= target.position_tolerance_m + kComparisonTolerance) {
    const TerminalYawCertificate yaw = CertifyTerminalYaw(
        capability, config, target.is_final_goal, target.terminal_yaw_rad,
        source_arrival_yaw, deadline, stop, statistics, status);
    if (status != WorkStatus::kReady) {
      return {};
    }
    if (yaw.feasible) {
      return {.feasible = true,
              .cost = yaw.cost,
              .uses_assumed_support = source_assumed,
              .requires_terminal_spin = yaw.requires_spin,
              .arrival_yaw_rad = source_arrival_yaw,
              .terminal_yaw_rad = yaw.yaw_rad};
    }
    // A positional hit with an unsatisfied requested yaw may still have a
    // certified short connector to the continuous final point.
  }

  const EdgeLookup edge = edge_cache.Certify(
      view, capability, source_state, source_point, target_cell, target.center,
      IsCellCenter(view.geometry(), target_cell, target.center), deadline, stop,
      statistics);
  if (edge.status != WorkStatus::kReady) {
    status = edge.status;
    return {};
  }
  if (!edge.certificate.feasible) {
    return {};
  }
  const TerminalYawCertificate yaw = CertifyTerminalYaw(
      capability, config, target.is_final_goal, target.terminal_yaw_rad,
      edge.certificate.arrival_yaw_rad, deadline, stop, statistics, status);
  if (status != WorkStatus::kReady) {
    return {};
  }
  if (!yaw.feasible) {
    return {.yaw_rejected = target.is_final_goal &&
                            target.terminal_yaw_rad.has_value()};
  }
  return {.feasible = true,
          .cost = SaturatingAdd(edge.certificate.cost, yaw.cost),
          .uses_assumed_support = edge.certificate.uses_assumed_support,
          .connects_exact_target = true,
          .requires_terminal_spin = yaw.requires_spin,
          .arrival_yaw_rad = edge.certificate.arrival_yaw_rad,
          .terminal_yaw_rad = yaw.yaw_rad};
}

[[nodiscard]] double Heuristic(const Point2 point,
                               const LocalTarget& target) noexcept {
  return std::max(0.0, std::hypot(point.x - target.center.x,
                                  point.y - target.center.y) -
                           target.position_tolerance_m);
}

[[nodiscard]] WorkStatus ApplyPathOrientations(
    std::vector<RawVertex>& vertices, const double start_yaw,
    const std::optional<double> final_movement_yaw,
    const SearchDeadline deadline, const StopToken& stop) {
  if (const WorkStatus status = CheckWork(deadline, stop);
      status != WorkStatus::kReady) {
    return status;
  }
  if (vertices.empty()) {
    return WorkStatus::kReady;
  }
  vertices.front().point.pose.orientation = YawQuaternion(start_yaw);
  for (std::size_t index = 1U; index + 1U < vertices.size(); ++index) {
    if (const WorkStatus status = CheckWork(deadline, stop);
        status != WorkStatus::kReady) {
      return status;
    }
    const Point2 source{
        .x = vertices[index].point.pose.position_m.x,
        .y = vertices[index].point.pose.position_m.y,
    };
    const Point2 target{
        .x = vertices[index + 1U].point.pose.position_m.x,
        .y = vertices[index + 1U].point.pose.position_m.y,
    };
    vertices[index].point.pose.orientation =
        YawQuaternion(Tangent(source, target, start_yaw));
  }
  if (vertices.size() == 1U) {
    vertices.front().point.pose.orientation =
        YawQuaternion(final_movement_yaw.value_or(start_yaw));
    return WorkStatus::kReady;
  }
  const Point2 penultimate{
      .x = vertices[vertices.size() - 2U].point.pose.position_m.x,
      .y = vertices[vertices.size() - 2U].point.pose.position_m.y,
  };
  const Point2 terminal{
      .x = vertices.back().point.pose.position_m.x,
      .y = vertices.back().point.pose.position_m.y,
  };
  vertices.back().point.pose.orientation = YawQuaternion(
      final_movement_yaw.value_or(Tangent(penultimate, terminal, start_yaw)));
  return CheckWork(deadline, stop);
}

[[nodiscard]] SimplificationResult SimplifyLeggedPhaseAwarePath(
    const RequestLocalPlanningView& view,
    const std::vector<RawVertex>& raw_vertices,
    const std::size_t protected_tail_vertices,
    LeggedDirectedEdgeCache& edge_cache, const LeggedCapability& capability,
    const SearchDeadline deadline, const StopToken& stop,
    SearchStatistics& statistics) {
  SimplificationResult result;
  if (raw_vertices.empty() ||
      protected_tail_vertices > raw_vertices.size()) {
    result.status = WorkStatus::kInvalid;
    return result;
  }
  const std::size_t simplifiable_size =
      raw_vertices.size() - protected_tail_vertices;
  result.vertices.reserve(raw_vertices.size());
  if (simplifiable_size == 0U) {
    result.vertices = raw_vertices;
    return result;
  }
  result.vertices.push_back(raw_vertices.front());
  for (std::size_t index = 1U; index < simplifiable_size; ++index) {
    if (const WorkStatus status = CheckWork(deadline, stop);
        status != WorkStatus::kReady) {
      result.status = status;
      return result;
    }
    const RawVertex& candidate = raw_vertices[index];
    if (result.vertices.size() < 2U) {
      result.vertices.push_back(candidate);
      continue;
    }
    RawVertex& previous = result.vertices.back();
    const RawVertex& source = result.vertices[result.vertices.size() - 2U];
    const Point2 source_position{
        .x = source.point.pose.position_m.x,
        .y = source.point.pose.position_m.y,
    };
    const Point2 candidate_position{
        .x = candidate.point.pose.position_m.x,
        .y = candidate.point.pose.position_m.y,
    };
    const auto candidate_cell = WorldToCell(view.geometry(), candidate_position);
    if (!candidate_cell.has_value()) {
      result.status = WorkStatus::kInvalid;
      return result;
    }
    const EdgeLookup shortcut = edge_cache.Certify(
        view, capability, source.state, source_position, *candidate_cell,
        candidate_position, candidate.lattice_position, deadline, stop,
        statistics);
    if (shortcut.status != WorkStatus::kReady) {
      result.status = shortcut.status;
      return result;
    }
    const auto next_phase =
        view.AdvancePhase(source.point.phase, *candidate_cell);
    if (shortcut.certificate.feasible && next_phase.has_value() &&
        *next_phase == candidate.point.phase &&
        (!shortcut.certificate.uses_assumed_support ||
         source.point.phase == StartPhase::kStartPrefix)) {
      previous = candidate;
    } else {
      result.vertices.push_back(candidate);
    }
  }
  result.vertices.insert(
      result.vertices.end(),
      raw_vertices.begin() + static_cast<std::ptrdiff_t>(simplifiable_size),
      raw_vertices.end());
  return result;
}

[[nodiscard]] std::vector<PathPoint> PathPoints(
    const std::vector<RawVertex>& vertices) {
  std::vector<PathPoint> path;
  path.reserve(vertices.size());
  for (const RawVertex& vertex : vertices) {
    path.push_back(vertex.point);
  }
  return path;
}

void AppendTerminalSpin(std::vector<PathPoint>& path,
                        const double terminal_yaw) {
  if (path.empty()) {
    return;
  }
  PathPoint spin = path.back();
  spin.phase = StartPhase::kNormal;
  spin.pose.orientation = YawQuaternion(terminal_yaw);
  path.push_back(spin);
}

}  // namespace

LeggedLocalPlanner::LeggedLocalPlanner(LeggedCapability capability,
                                       LeggedLocalPlannerConfig config)
    : capability_(std::move(capability)), config_(std::move(config)) {
  if (!CapabilityValid(capability_) ||
      !std::isfinite(config_.terminal_yaw_tolerance_rad) ||
      config_.terminal_yaw_tolerance_rad < 0.0) {
    throw std::invalid_argument("legged local planner configuration is invalid");
  }
}

LocalPlanResult LeggedLocalPlanner::Plan(
    const RequestLocalPlanningView& view, const Pose2& start,
    const LocalTarget& target, const SearchDeadline deadline,
    const StopToken& stop) {
  if (const WorkStatus stopped = CheckWork(deadline, stop);
      stopped != WorkStatus::kReady) {
    return StoppedResult(stopped);
  }
  const SparseGridGeometry& geometry = view.geometry();
  const std::size_t cell_count = geometry.CellCount();
  if (!Finite(start) || !Finite(target) || !geometry.valid() ||
      cell_count == 0U || cell_count > kMaximumLocalCells ||
      cell_count > std::numeric_limits<std::size_t>::max() / kStatesPerCell) {
    return {};
  }
  const std::size_t state_count = cell_count * kStatesPerCell;
  if (buffer_capacity_states_ < state_count) {
    g_cost_.resize(state_count);
    parent_.resize(state_count);
    state_.resize(state_count);
    generation_.resize(state_count, 0U);
    buffer_capacity_states_ = state_count;
  }
  if (++current_generation_ == 0U) {
    std::fill(generation_.begin(), generation_.end(), 0U);
    current_generation_ = 1U;
  }

  const auto start_cell = WorldToCell(geometry, start.position_m);
  const auto target_cell = WorldToCell(geometry, target.center);
  if (!start_cell.has_value() || !target_cell.has_value() ||
      !view.CanCertifyLeggedSupport(*start_cell, true) ||
      !view.CanBeEndpoint(*target_cell)) {
    return {};
  }
  const StartPhase initial_phase =
      view.Source(*start_cell) == LocalCellSource::kStartAssumedFree
          ? StartPhase::kStartPrefix
          : StartPhase::kNormal;
  const auto initialize = [&](const std::uint32_t state) {
    const std::size_t offset = static_cast<std::size_t>(state);
    if (generation_[offset] != current_generation_) {
      generation_[offset] = current_generation_;
      g_cost_[offset] = std::numeric_limits<double>::infinity();
      parent_[offset] = kInvalidState;
      state_[offset] = 0U;
    }
  };
  const std::uint32_t start_state =
      SearchState(CellOffset(geometry, *start_cell), initial_phase, true);
  initialize(start_state);
  g_cost_[start_state] = 0.0;

  std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenLater> open;
  std::uint64_t sequence = 0U;
  open.push(OpenEntry{.f = Heuristic(start.position_m, target),
                      .g = 0.0,
                      .state = start_state,
                      .sequence = sequence++});
  SearchStatistics statistics{.generated_states = 1U, .open_peak = open.size()};
  LeggedDirectedEdgeCache edge_cache;
  std::unordered_map<std::uint32_t, TerminalCacheEntry> terminal_cache;
  std::optional<std::uint32_t> goal_state;
  TerminalCertificate goal_terminal;
  double best_goal_cost = std::numeric_limits<double>::infinity();
  bool saw_terminal_yaw_rejection = false;

  constexpr std::array<std::pair<std::int64_t, std::int64_t>, 8U> kNeighbors{{
      {-1, 0}, {0, -1}, {1, 0}, {0, 1},
      {-1, -1}, {1, -1}, {-1, 1}, {1, 1},
  }};
  while (!open.empty()) {
    if (goal_state.has_value() && open.top().f >= best_goal_cost) {
      break;
    }
    if (const WorkStatus stopped = CheckWork(deadline, stop);
        stopped != WorkStatus::kReady) {
      return StoppedResult(stopped, statistics);
    }
    const OpenEntry current = open.top();
    open.pop();
    initialize(current.state);
    if (state_[current.state] == 2U ||
        current.g != g_cost_[current.state]) {
      continue;
    }
    state_[current.state] = 2U;
    ++statistics.expanded_states;
    const GridIndex current_cell =
        CellFromOffset(geometry, SearchCellOffset(current.state));
    const Point2 current_point = PositionForState(view, current.state, start);
    const std::uint32_t parent_state = parent_[current.state];
    const double current_arrival_yaw =
        parent_state == kInvalidState
            ? start.yaw_rad
            : Tangent(PositionForState(view, parent_state, start),
                      current_point, start.yaw_rad);

    TerminalCacheEntry& terminal_entry = terminal_cache[current.state];
    if (!terminal_entry.evaluated) {
      WorkStatus terminal_status = WorkStatus::kReady;
      terminal_entry.certificate = CertifyTerminal(
          view, capability_, config_, edge_cache, current.state, current_point,
          current_arrival_yaw, target, *target_cell, deadline, stop, statistics,
          terminal_status);
      if (terminal_status != WorkStatus::kReady) {
        return StoppedResult(terminal_status, statistics);
      }
      terminal_entry.evaluated = true;
    }
    const TerminalCertificate terminal = terminal_entry.certificate;
    saw_terminal_yaw_rejection = saw_terminal_yaw_rejection ||
                                  terminal.yaw_rejected;
    if (terminal.feasible &&
        (!terminal.uses_assumed_support ||
         SearchPhase(current.state) == StartPhase::kStartPrefix)) {
      const double terminal_total = SaturatingAdd(current.g, terminal.cost);
      if (!goal_state.has_value() || terminal_total < best_goal_cost) {
        goal_state = current.state;
        goal_terminal = terminal;
        best_goal_cost = terminal_total;
      }
    }

    for (const auto [dx, dy] : kNeighbors) {
      const auto next_x = CheckedAdd(current_cell.x, dx);
      const auto next_y = CheckedAdd(current_cell.y, dy);
      if (!next_x.has_value() || !next_y.has_value()) {
        continue;
      }
      const GridIndex next_cell{.x = *next_x, .y = *next_y};
      if (!geometry.Contains(next_cell)) {
        continue;
      }
      const auto next_phase =
          view.AdvancePhase(SearchPhase(current.state), next_cell);
      if (!next_phase.has_value()) {
        continue;
      }
      const Point2 next_point = CellCenter(geometry, next_cell);
      const EdgeLookup edge = edge_cache.Certify(
          view, capability_, current.state, current_point, next_cell, next_point,
          true, deadline, stop, statistics);
      if (edge.status != WorkStatus::kReady) {
        return StoppedResult(edge.status, statistics);
      }
      if (!edge.certificate.feasible ||
          (edge.certificate.uses_assumed_support &&
           SearchPhase(current.state) != StartPhase::kStartPrefix)) {
        continue;
      }
      const std::uint32_t next_state =
          SearchState(CellOffset(geometry, next_cell), *next_phase, false);
      initialize(next_state);
      if (state_[next_state] == 2U) {
        continue;
      }
      const double candidate = SaturatingAdd(current.g, edge.certificate.cost);
      if (candidate + kComparisonTolerance >= g_cost_[next_state]) {
        continue;
      }
      g_cost_[next_state] = candidate;
      parent_[next_state] = current.state;
      state_[next_state] = 1U;
      open.push(OpenEntry{
          .f = SaturatingAdd(candidate, Heuristic(next_point, target)),
          .g = candidate,
          .state = next_state,
          .sequence = sequence++,
      });
      statistics.open_peak = std::max(statistics.open_peak, open.size());
      ++statistics.generated_states;
    }
  }
  if (!goal_state.has_value()) {
    LocalPlanResult result = StoppedResult(WorkStatus::kInvalid, statistics);
    if (saw_terminal_yaw_rejection) {
      result.reason_code = "TERMINAL_YAW_UNREACHABLE";
    }
    return result;
  }

  const auto postprocess_started = SteadyClock::now();
  std::vector<std::uint32_t> states;
  for (std::uint32_t cursor = *goal_state;;) {
    if (const WorkStatus stopped = CheckWork(deadline, stop);
        stopped != WorkStatus::kReady) {
      return StoppedResult(stopped, statistics);
    }
    states.push_back(cursor);
    if (cursor == start_state) {
      break;
    }
    if (parent_[cursor] == kInvalidState) {
      return StoppedResult(WorkStatus::kInvalid, statistics);
    }
    cursor = parent_[cursor];
  }
  std::reverse(states.begin(), states.end());
  std::vector<RawVertex> raw_vertices;
  raw_vertices.reserve(states.size() +
                       (goal_terminal.connects_exact_target ? 1U : 0U));
  for (const std::uint32_t state : states) {
    const Point2 position = PositionForState(view, state, start);
    raw_vertices.push_back({
        .point = PathPoint{
            .pose = Pose3{.position_m = {.x = position.x,
                                          .y = position.y,
                                          .z = 0.0}},
            .phase = SearchPhase(state),
        },
        .state = state,
        .lattice_position = !SearchAtStartAnchor(state),
    });
  }
  if (raw_vertices.empty()) {
    return StoppedResult(WorkStatus::kInvalid, statistics);
  }
  if (goal_terminal.connects_exact_target) {
    raw_vertices.push_back({
        .point = PathPoint{
            .pose = Pose3{.position_m = {.x = target.center.x,
                                          .y = target.center.y,
                                          .z = 0.0}},
            .phase = StartPhase::kNormal,
        },
        .state = *goal_state,
        .lattice_position = IsCellCenter(geometry, *target_cell, target.center),
    });
  }

  const bool protects_final_arrival =
      target.is_final_goal && target.terminal_yaw_rad.has_value() &&
      !goal_terminal.requires_terminal_spin;
  const std::optional<double> raw_final_yaw =
      protects_final_arrival
          ? std::optional<double>(goal_terminal.arrival_yaw_rad)
          : std::nullopt;
  if (const WorkStatus status = ApplyPathOrientations(
          raw_vertices, start.yaw_rad, raw_final_yaw, deadline, stop);
      status != WorkStatus::kReady) {
    return StoppedResult(status, statistics);
  }

  const std::size_t protected_tail_vertices =
      protects_final_arrival && raw_vertices.size() > 1U ? 1U : 0U;
  SimplificationResult simplified = SimplifyLeggedPhaseAwarePath(
      view, raw_vertices, protected_tail_vertices, edge_cache, capability_,
      deadline, stop, statistics);
  if (simplified.status == WorkStatus::kCanceled) {
    return StoppedResult(simplified.status, statistics);
  }
  if (simplified.status == WorkStatus::kTimeout) {
    // The raw path was certified by the same edge cache before simplification.
    // Preserve it instead of turning an already safe route into NO_PATH.
    simplified.vertices = raw_vertices;
  } else if (simplified.status != WorkStatus::kReady ||
             simplified.vertices.empty()) {
    return StoppedResult(simplified.status, statistics);
  }
  if (simplified.status == WorkStatus::kReady) {
    if (const WorkStatus status = ApplyPathOrientations(
            simplified.vertices, start.yaw_rad, raw_final_yaw, deadline, stop);
        status != WorkStatus::kReady) {
      return StoppedResult(status, statistics);
    }
  }

  LocalPlanResult result;
  result.status = LocalPlanResult::Status::kPlanFound;
  result.reaches_final_goal = target.is_final_goal;
  result.statistics = statistics;
  result.raw_path = PathPoints(raw_vertices);
  result.path = PathPoints(simplified.vertices);
  if (goal_terminal.requires_terminal_spin) {
    AppendTerminalSpin(result.raw_path, goal_terminal.terminal_yaw_rad);
    AppendTerminalSpin(result.path, goal_terminal.terminal_yaw_rad);
  }
  result.postprocess_elapsed = SteadyClock::now() - postprocess_started;
  return result;
}

}  // namespace lunar::incremental_navigation
