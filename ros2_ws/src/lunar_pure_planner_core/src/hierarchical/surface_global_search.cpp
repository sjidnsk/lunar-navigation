#include "hierarchical/surface_global_search.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "shared/anytime_ara_star.hpp"

namespace lunar::pure_planning::hierarchical {
namespace {

using shared::GlobalOccupancyProjectionView;
using shared::GridCell;

constexpr std::array<GridCell, 8> kEightNeighbors{{
    {.x = -1, .y = -1}, {.x = 0, .y = -1}, {.x = 1, .y = -1},
    {.x = -1, .y = 0},                         {.x = 1, .y = 0},
    {.x = -1, .y = 1},  {.x = 0, .y = 1},  {.x = 1, .y = 1},
}};

[[nodiscard]] SurfaceGlobalSearchResult Failure(
    const SurfaceGlobalSearchStatus status, std::string reason_code,
    const std::size_t expanded_states = 0U,
    const bool deadline_reached = false) {
  return {.status = status,
          .raw_cells = {},
          .simplified_cells = {},
          .preview = {},
          .cost = 0.0,
          .expanded_states = expanded_states,
          .deadline_reached = deadline_reached,
          .reason_code = std::move(reason_code)};
}

[[nodiscard]] GridCell CellFromIndex(const shared::MapSnapshot& map,
                                      const std::size_t index) noexcept {
  return {.x = static_cast<std::int32_t>(index % map.width()),
          .y = static_cast<std::int32_t>(index / map.width())};
}

[[nodiscard]] bool IsDiagonal(const GridCell from,
                              const GridCell to) noexcept {
  return from.x != to.x && from.y != to.y;
}

[[nodiscard]] bool CornerBlocked(const GridCell from, const GridCell to,
                                 const GlobalOccupancyProjectionView view) noexcept {
  return !view.HardFeasible({.x = to.x, .y = from.y}) ||
         !view.HardFeasible({.x = from.x, .y = to.y});
}

[[nodiscard]] double StepLength(const GridCell from, const GridCell to,
                                const double resolution_m) noexcept {
  return resolution_m * (IsDiagonal(from, to) ? std::numbers::sqrt2 : 1.0);
}

[[nodiscard]] double ClearanceCost(const GlobalOccupancyProjectionView view,
                                   const GridCell cell) noexcept {
  const double clearance = static_cast<double>(view.ClearanceMeters(cell));
  if (!std::isfinite(clearance)) {
    return clearance > 0.0 ? 0.0 : 1.0;
  }
  return 1.0 / (1.0 + std::max(0.0, clearance));
}

[[nodiscard]] int Sign(const std::int32_t value) noexcept {
  return (value > 0) - (value < 0);
}

enum class ControlState : std::uint8_t { kRunning, kCanceled, kDeadline };

[[nodiscard]] ControlState CheckControl(const SearchControl& control) {
  if (control.canceled()) {
    return ControlState::kCanceled;
  }
  const bool expired = control.expired();
  return control.canceled() ? ControlState::kCanceled
                            : (expired ? ControlState::kDeadline
                                       : ControlState::kRunning);
}

struct SafetyResult final {
  bool safe{};
  ControlState control{ControlState::kRunning};
};

struct PreviewResult final {
  GlobalRoutePreview preview;
  ControlState control{ControlState::kRunning};
};

[[nodiscard]] bool Finite(const Pose3& pose) noexcept {
  return std::isfinite(pose.position_m.x) && std::isfinite(pose.position_m.y) &&
         std::isfinite(pose.position_m.z) && std::isfinite(pose.orientation.w) &&
         std::isfinite(pose.orientation.x) && std::isfinite(pose.orientation.y) &&
         std::isfinite(pose.orientation.z);
}

[[nodiscard]] bool PoseMatchesCell(const shared::MapSnapshot& map,
                                    const Pose3& pose,
                                    const GridCell cell) noexcept {
  const auto mapped = map.PositionToCell(
      {.x = pose.position_m.x, .y = pose.position_m.y});
  return Finite(pose) && mapped.has_value() && *mapped == cell;
}

[[nodiscard]] SafetyResult SupercoverSafe(
    const GlobalOccupancyProjectionView view, const GridCell start,
    const GridCell end, const SearchControl& control) {
  if (!view.HardFeasible(start) || !view.HardFeasible(end)) {
    return {.safe = false};
  }
  std::int32_t x = start.x;
  std::int32_t y = start.y;
  const std::int64_t dx = static_cast<std::int64_t>(end.x) - start.x;
  const std::int64_t dy = static_cast<std::int64_t>(end.y) - start.y;
  const std::int64_t nx = std::abs(dx);
  const std::int64_t ny = std::abs(dy);
  const int sx = Sign(static_cast<std::int32_t>(dx));
  const int sy = Sign(static_cast<std::int32_t>(dy));
  std::int64_t ix{};
  std::int64_t iy{};

  while (ix < nx || iy < ny) {
    const ControlState control_state = CheckControl(control);
    if (control_state != ControlState::kRunning) {
      return {.safe = false, .control = control_state};
    }
    if (ix == nx) {
      y += sy;
      ++iy;
    } else if (iy == ny) {
      x += sx;
      ++ix;
    } else {
      const std::int64_t x_cross = (1 + 2 * ix) * ny;
      const std::int64_t y_cross = (1 + 2 * iy) * nx;
      if (x_cross < y_cross) {
        x += sx;
        ++ix;
      } else if (y_cross < x_cross) {
        y += sy;
        ++iy;
      } else {
        if (!view.HardFeasible({.x = x + sx, .y = y}) ||
            !view.HardFeasible({.x = x, .y = y + sy})) {
          return {.safe = false};
        }
        x += sx;
        y += sy;
        ++ix;
        ++iy;
      }
    }
    if (!view.HardFeasible({.x = x, .y = y})) {
      return {.safe = false};
    }
  }
  return {.safe = true};
}

[[nodiscard]] SafetyResult WorldSegmentSafe(
    const GlobalOccupancyProjectionView view, const Vec2 start,
    const Vec2 end, const SearchControl& control) {
  const auto* map = view.map;
  if (map == nullptr) {
    return {.safe = false};
  }
  const auto start_cell = map->PositionToCell(start);
  const auto end_cell = map->PositionToCell(end);
  if (!start_cell.has_value() || !end_cell.has_value() ||
      !view.HardFeasible(*start_cell) || !view.HardFeasible(*end_cell)) {
    return {.safe = false};
  }
  const double resolution_m = map->resolution_m();
  const double x0 = (start.x - map->origin_m().x) / resolution_m;
  const double y0 = (start.y - map->origin_m().y) / resolution_m;
  const double x1 = (end.x - map->origin_m().x) / resolution_m;
  const double y1 = (end.y - map->origin_m().y) / resolution_m;
  const double dx = x1 - x0;
  const double dy = y1 - y0;
  const int sx = dx > 0.0 ? 1 : (dx < 0.0 ? -1 : 0);
  const int sy = dy > 0.0 ? 1 : (dy < 0.0 ? -1 : 0);
  std::int32_t x = start_cell->x;
  std::int32_t y = start_cell->y;
  const double infinity = std::numeric_limits<double>::infinity();
  const double delta_tx = sx == 0 ? infinity : 1.0 / std::abs(dx);
  const double delta_ty = sy == 0 ? infinity : 1.0 / std::abs(dy);
  const double next_x = sx > 0 ? std::floor(x0) + 1.0 : std::floor(x0);
  const double next_y = sy > 0 ? std::floor(y0) + 1.0 : std::floor(y0);
  double maximum_tx = sx == 0 ? infinity : std::abs((next_x - x0) / dx);
  double maximum_ty = sy == 0 ? infinity : std::abs((next_y - y0) / dy);
  constexpr double kTolerance = 64.0 * std::numeric_limits<double>::epsilon();

  while (x != end_cell->x || y != end_cell->y) {
    const ControlState control_state = CheckControl(control);
    if (control_state != ControlState::kRunning) {
      return {.safe = false, .control = control_state};
    }
    const double scale = std::max({1.0, std::abs(maximum_tx),
                                   std::abs(maximum_ty)});
    if (maximum_tx + kTolerance * scale < maximum_ty) {
      x += sx;
      maximum_tx += delta_tx;
    } else if (maximum_ty + kTolerance * scale < maximum_tx) {
      y += sy;
      maximum_ty += delta_ty;
    } else {
      if (!view.HardFeasible({.x = x + sx, .y = y}) ||
          !view.HardFeasible({.x = x, .y = y + sy})) {
        return {.safe = false};
      }
      x += sx;
      y += sy;
      maximum_tx += delta_tx;
      maximum_ty += delta_ty;
    }
    if (!view.HardFeasible({.x = x, .y = y})) {
      return {.safe = false};
    }
  }
  return {.safe = true};
}

struct SimplifyResult final {
  std::vector<GridCell> cells;
  ControlState control{ControlState::kRunning};
};

[[nodiscard]] SimplifyResult SimplifySupercover(
    const GlobalOccupancyProjectionView view,
    const std::span<const GridCell> route, const SearchControl& control) {
  if (route.empty()) {
    return {};
  }
  std::vector<GridCell> simplified{route.front()};
  std::size_t current = 0U;
  while (current + 1U < route.size()) {
    const ControlState control_state = CheckControl(control);
    if (control_state != ControlState::kRunning) {
      return {.cells = {}, .control = control_state};
    }
    std::size_t selected = current + 1U;
    for (std::size_t candidate = route.size() - 1U; candidate > current;
         --candidate) {
      const SafetyResult safe =
          SupercoverSafe(view, route[current], route[candidate], control);
      if (safe.control != ControlState::kRunning) {
        return {.cells = {}, .control = safe.control};
      }
      if (safe.safe) {
        selected = candidate;
        break;
      }
    }
    simplified.push_back(route[selected]);
    current = selected;
  }
  return {.cells = std::move(simplified)};
}

[[nodiscard]] PreviewResult BuildPreview(
    const shared::MapSnapshot& map, const std::span<const GridCell> cells,
    const Pose3& start_pose, const Pose3& goal_pose,
    const SearchControl& control) {
  GlobalRoutePreview preview;
  if (cells.empty()) {
    return {.preview = std::move(preview)};
  }
  preview.poses_map.reserve(cells.size());
  preview.poses_map.push_back(start_pose);
  for (std::size_t index = 1U; index + 1U < cells.size(); ++index) {
    const ControlState control_state = CheckControl(control);
    if (control_state != ControlState::kRunning) {
      return {.preview = {}, .control = control_state};
    }
    preview.poses_map.push_back(
        {.position_m = map.CellCenter(cells[index]),
         .orientation = start_pose.orientation});
  }
  if (cells.size() > 1U || goal_pose != start_pose) {
    const ControlState control_state = CheckControl(control);
    if (control_state != ControlState::kRunning) {
      return {.preview = {}, .control = control_state};
    }
    preview.poses_map.push_back(goal_pose);
  }
  return {.preview = std::move(preview)};
}

[[nodiscard]] SafetyResult PreviewSafe(
    const GlobalOccupancyProjectionView view,
    const std::span<const Pose3> poses, const SearchControl& control) {
  if (poses.empty()) {
    return {.safe = false};
  }
  for (std::size_t index = 1U; index < poses.size(); ++index) {
    const SafetyResult safe = WorldSegmentSafe(
        view, {.x = poses[index - 1U].position_m.x,
               .y = poses[index - 1U].position_m.y},
        {.x = poses[index].position_m.x, .y = poses[index].position_m.y},
        control);
    if (!safe.safe || safe.control != ControlState::kRunning) {
      return safe;
    }
  }
  return {.safe = true};
}

[[nodiscard]] SurfaceGlobalSearchResult FromAraResult(
    const shared::anytime::AraStarResult& result) {
  using AraStatus = shared::anytime::AraStarStatus;
  switch (result.status) {
  case AraStatus::kNoPath:
    return Failure(SurfaceGlobalSearchStatus::kNoPath, result.reason_code,
                   result.expanded_states, result.deadline_reached);
  case AraStatus::kCanceled:
    return Failure(SurfaceGlobalSearchStatus::kCanceled, result.reason_code,
                   result.expanded_states, result.deadline_reached);
  case AraStatus::kTimedOut:
    return Failure(SurfaceGlobalSearchStatus::kTimedOut, result.reason_code,
                   result.expanded_states, result.deadline_reached);
  case AraStatus::kResourceExhausted:
    return Failure(SurfaceGlobalSearchStatus::kResourceExhausted,
                   result.reason_code, result.expanded_states,
                   result.deadline_reached);
  case AraStatus::kInvalidProblem:
  case AraStatus::kSolved:
    return Failure(SurfaceGlobalSearchStatus::kInvalidProblem,
                   result.reason_code, result.expanded_states,
                   result.deadline_reached);
  }
  return Failure(SurfaceGlobalSearchStatus::kInvalidProblem,
                 "SEARCH_RESULT_INVALID", result.expanded_states,
                 result.deadline_reached);
}

}  // namespace

SurfaceGlobalSearchResult SearchSurfaceGlobal(
    const SurfaceGlobalSearchProblem& problem) {
  const GlobalOccupancyProjectionView view = problem.projection;
  if (!view.Valid() || !view.map->InBounds(problem.start) ||
      !view.map->InBounds(problem.goal)) {
    return Failure(SurfaceGlobalSearchStatus::kInvalidProblem,
                   "GLOBAL_SEARCH_INPUT_INVALID");
  }
  const shared::MapSnapshot& map = *view.map;
  if (!PoseMatchesCell(map, problem.start_pose_map, problem.start) ||
      !PoseMatchesCell(map, problem.goal_pose_map, problem.goal)) {
    return Failure(SurfaceGlobalSearchStatus::kInvalidProblem,
                   "GLOBAL_SEARCH_POSE_CELL_MISMATCH");
  }
  if (!view.HardFeasible(problem.start)) {
    return Failure(SurfaceGlobalSearchStatus::kNoPath,
                   "GLOBAL_START_INFEASIBLE");
  }
  if (!view.HardFeasible(problem.goal)) {
    return Failure(SurfaceGlobalSearchStatus::kNoPath,
                   "GLOBAL_GOAL_INFEASIBLE");
  }
  const double resolution_m = map.resolution_m();
  const std::size_t goal_index = map.Index(problem.goal);
  const auto index_of = [&map](const GridCell cell) { return map.Index(cell); };
  const shared::anytime::AraStarResult search =
      shared::anytime::SearchAnytimeAraStar({
          .state_count = map.cell_count(),
          .start_state = index_of(problem.start),
          .expand = [view, &map, resolution_m, index_of](
                        const std::size_t id,
                        const double,
                        std::vector<shared::GraphEdge>& edges) {
            const GridCell current = CellFromIndex(map, id);
            for (std::size_t offset = 0U; offset < kEightNeighbors.size(); ++offset) {
              const GridCell next{.x = current.x + kEightNeighbors[offset].x,
                                  .y = current.y + kEightNeighbors[offset].y};
              if (!view.HardFeasible(next) ||
                  (IsDiagonal(current, next) && CornerBlocked(current, next, view))) {
                continue;
              }
              edges.push_back({.target_state = index_of(next),
                               .cost = StepLength(current, next, resolution_m) +
                                       ClearanceCost(view, next),
                               .stable_index = id * kEightNeighbors.size() + offset});
            }
          },
          .heuristic = [&map, goal = problem.goal, resolution_m](const std::size_t id) {
            const GridCell cell = CellFromIndex(map, id);
            return resolution_m * std::hypot(
                                      static_cast<double>(goal.x - cell.x),
                                      static_cast<double>(goal.y - cell.y));
          },
          .is_goal = [goal_index](const std::size_t id) { return id == goal_index; },
          .edges_are_stably_sorted = true,
          .config = problem.search,
          .control = problem.control,
      });
  if (search.status != shared::anytime::AraStarStatus::kSolved ||
      search.candidates.empty()) {
    return FromAraResult(search);
  }
  const shared::SearchCandidate& candidate = search.candidates.back();
  std::vector<GridCell> raw_cells;
  raw_cells.reserve(candidate.states.size());
  if (search.deadline_reached) {
    return Failure(SurfaceGlobalSearchStatus::kTimedOut, "TIMEOUT",
                   search.expanded_states, true);
  }
  bool deadline_reached{};
  for (const std::size_t id : candidate.states) {
    const ControlState control_state = CheckControl(problem.control);
    if (control_state == ControlState::kCanceled) {
      return Failure(SurfaceGlobalSearchStatus::kCanceled, "REQUEST_CANCELED",
                     search.expanded_states, deadline_reached);
    }
    if (control_state == ControlState::kDeadline) {
      return Failure(SurfaceGlobalSearchStatus::kTimedOut, "TIMEOUT",
                     search.expanded_states, true);
    }
    raw_cells.push_back(CellFromIndex(map, id));
  }
  const auto raw_preview = BuildPreview(map, raw_cells, problem.start_pose_map,
                                        problem.goal_pose_map, problem.control);
  if (raw_preview.control == ControlState::kCanceled) {
    return Failure(SurfaceGlobalSearchStatus::kCanceled, "REQUEST_CANCELED",
                   search.expanded_states, deadline_reached);
  }
  if (raw_preview.control == ControlState::kDeadline) {
    return Failure(SurfaceGlobalSearchStatus::kTimedOut, "TIMEOUT",
                   search.expanded_states, true);
  }
  const SafetyResult raw_safe = PreviewSafe(view, raw_preview.preview.poses_map,
                                            problem.control);
  if (raw_safe.control == ControlState::kCanceled) {
    return Failure(SurfaceGlobalSearchStatus::kCanceled, "REQUEST_CANCELED",
                   search.expanded_states, deadline_reached);
  }
  if (raw_safe.control == ControlState::kDeadline) {
    return Failure(SurfaceGlobalSearchStatus::kTimedOut, "TIMEOUT",
                   search.expanded_states, true);
  }
  if (!raw_safe.safe) {
    return Failure(SurfaceGlobalSearchStatus::kNoPath,
                   "GLOBAL_EXACT_CONNECTION_INFEASIBLE", search.expanded_states,
                   deadline_reached);
  }
  const auto raw_result = [&] {
    return SurfaceGlobalSearchResult{
        .status = SurfaceGlobalSearchStatus::kSolved,
        .raw_cells = raw_cells,
        .simplified_cells = raw_cells,
        .preview = raw_preview.preview,
        .cost = candidate.cost,
        .expanded_states = search.expanded_states,
        .deadline_reached = deadline_reached,
        .reason_code = "SEARCH_SOLVED",
    };
  };
  const SimplifyResult simplified =
      SimplifySupercover(view, raw_cells, problem.control);
  if (simplified.control == ControlState::kCanceled) {
    return Failure(SurfaceGlobalSearchStatus::kCanceled, "REQUEST_CANCELED",
                   search.expanded_states);
  }
  if (simplified.control == ControlState::kDeadline) {
    deadline_reached = true;
    return raw_result();
  }
  if (simplified.cells.empty()) {
    return Failure(SurfaceGlobalSearchStatus::kInvalidProblem,
                   "GLOBAL_ROUTE_SIMPLIFICATION_INVALID", search.expanded_states);
  }
  const PreviewResult preview = BuildPreview(map, simplified.cells,
                                             problem.start_pose_map,
                                             problem.goal_pose_map,
                                             problem.control);
  if (preview.control == ControlState::kCanceled) {
    return Failure(SurfaceGlobalSearchStatus::kCanceled, "REQUEST_CANCELED",
                   search.expanded_states);
  }
  if (preview.control == ControlState::kDeadline) {
    deadline_reached = true;
    return raw_result();
  }
  const SafetyResult preview_safe =
      PreviewSafe(view, preview.preview.poses_map, problem.control);
  if (preview_safe.control == ControlState::kCanceled) {
    return Failure(SurfaceGlobalSearchStatus::kCanceled, "REQUEST_CANCELED",
                   search.expanded_states);
  }
  if (preview_safe.control == ControlState::kDeadline || !preview_safe.safe) {
    deadline_reached = preview_safe.control == ControlState::kDeadline;
    return raw_result();
  }
  return {.status = SurfaceGlobalSearchStatus::kSolved,
          .raw_cells = std::move(raw_cells),
          .simplified_cells = std::move(simplified.cells),
          .preview = std::move(preview.preview),
          .cost = candidate.cost,
          .expanded_states = search.expanded_states,
          .deadline_reached = false,
          .reason_code = "SEARCH_SOLVED"};
}

}  // namespace lunar::pure_planning::hierarchical
