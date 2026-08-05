#include "hierarchical/grid_search.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace lunar::planning::hierarchical {
namespace {

constexpr std::array<std::int32_t, 8> kNeighborX{0, 1, 1, 1, 0, -1, -1, -1};
constexpr std::array<std::int32_t, 8> kNeighborY{-1, -1, 0, 1, 1, 1, 0, -1};

struct OpenEntry final {
  double f{};
  double h{};
  double g{};
  std::size_t state{};
  std::uint64_t serial{};
};

struct WorseOpenEntry final {
  [[nodiscard]] bool operator()(const OpenEntry &left,
                                const OpenEntry &right) const noexcept {
    return std::tie(left.f, left.h, left.g, left.state, left.serial) >
           std::tie(right.f, right.h, right.g, right.state, right.serial);
  }
};

[[nodiscard]] GlobalGridSearchResult Failure(const GlobalSearchStatus status,
                                             std::string reason,
                                             const std::uint64_t expanded = 0U,
                                             const std::size_t open_peak = 0U,
                                             const std::size_t memory = 0U) {
  return GlobalGridSearchResult{
      .status = status,
      .path_cells = {},
      .cost = 0.0,
      .expanded_states = expanded,
      .open_peak = open_peak,
      .estimated_work_memory_bytes = memory,
      .reason_code = std::move(reason),
  };
}

[[nodiscard]] shared::GridCell CellFromIndex(const shared::MapSnapshot &map,
                                             const std::size_t index) noexcept {
  return shared::GridCell{
      .x = static_cast<std::int32_t>(index % map.width()),
      .y = static_cast<std::int32_t>(index / map.width()),
  };
}

[[nodiscard]] bool
ValidConfig(const GlobalGridSearchProblem &problem) noexcept {
  const auto &resources = problem.config.resources;
  return std::isfinite(problem.maximum_speed_mps) &&
         problem.maximum_speed_mps > 0.0 &&
         std::isfinite(problem.config.slope_weight) &&
         problem.config.slope_weight >= 0.0 &&
         std::isfinite(problem.config.roughness_weight) &&
         problem.config.roughness_weight >= 0.0 &&
         std::isfinite(problem.config.clearance_weight) &&
         problem.config.clearance_weight >= 0.0 &&
         resources.maximum_expanded_states > 0U &&
         resources.maximum_reopened_states > 0U &&
         resources.maximum_generated_candidates > 0U &&
         resources.maximum_open_states > 0U &&
         resources.maximum_memory_bytes > 0U;
}

[[nodiscard]] std::optional<std::size_t>
FixedMemoryBytes(const std::size_t cells) noexcept {
  constexpr std::size_t kBytesPerCell =
      sizeof(double) + sizeof(std::size_t) + sizeof(std::uint8_t);
  if (cells > std::numeric_limits<std::size_t>::max() / kBytesPerCell) {
    return std::nullopt;
  }
  return cells * kBytesPerCell;
}

[[nodiscard]] double Heuristic(const shared::MapSnapshot &map,
                               const shared::GridCell cell,
                               const std::span<const shared::GridCell> goals,
                               const double maximum_speed_mps) noexcept {
  double minimum = std::numeric_limits<double>::infinity();
  for (const shared::GridCell goal : goals) {
    const double delta_x = static_cast<double>(goal.x - cell.x);
    const double delta_y = static_cast<double>(goal.y - cell.y);
    minimum = std::min(minimum, std::hypot(delta_x, delta_y));
  }
  return minimum * map.resolution_m() / maximum_speed_mps;
}

[[nodiscard]] bool DiagonalAllowed(const shared::SafeProjection &projection,
                                   const shared::GridCell current,
                                   const std::int32_t delta_x,
                                   const std::int32_t delta_y) noexcept {
  if (delta_x == 0 || delta_y == 0) {
    return true;
  }
  return projection.HardFeasible(
             shared::GridCell{current.x + delta_x, current.y}) &&
         projection.HardFeasible(
             shared::GridCell{current.x, current.y + delta_y});
}

[[nodiscard]] double EdgeCost(const GlobalGridSearchProblem &problem,
                              const shared::GridCell next,
                              const bool diagonal) noexcept {
  const double resolution = problem.projection.source_map()->resolution_m();
  const double distance = resolution * (diagonal ? std::numbers::sqrt2 : 1.0);
  const double clearance =
      std::max<double>(problem.projection.ClearanceMeters(next), resolution);
  const double slope = problem.projection.SlopeRadians(next);
  const double risk = problem.config.slope_weight * slope * slope +
                      problem.config.roughness_weight *
                          problem.projection.RoughnessMeters(next) +
                      problem.config.clearance_weight / clearance;
  return distance + problem.projection.TraversalCost(next) + distance * risk;
}

[[nodiscard]] std::vector<shared::GridCell>
Reconstruct(const shared::MapSnapshot &map,
            const std::span<const std::size_t> parent, const std::size_t start,
            const std::size_t goal) {
  std::vector<shared::GridCell> path;
  path.reserve(std::min<std::size_t>(parent.size(), 4'096U));
  std::size_t current = goal;
  for (std::size_t count = 0U; count < parent.size(); ++count) {
    path.push_back(CellFromIndex(map, current));
    if (current == start) {
      std::ranges::reverse(path);
      return path;
    }
    current = parent[current];
    if (current >= parent.size()) {
      return {};
    }
  }
  return {};
}

} // namespace

GlobalGridSearchResult
SearchGlobalGrid(const GlobalGridSearchProblem &problem) {
  if (problem.stop_token.stop_requested()) {
    return Failure(GlobalSearchStatus::kCanceled, "REQUEST_CANCELED");
  }
  const auto &map = problem.projection.source_map();
  if (map == nullptr || !ValidConfig(problem) ||
      problem.goal_mask.size() != map->cell_count() ||
      !map->InBounds(problem.start) ||
      !problem.projection.HardFeasible(problem.start)) {
    return Failure(GlobalSearchStatus::kInvalidProblem,
                   "GLOBAL_SEARCH_PROBLEM_INVALID");
  }

  std::vector<shared::GridCell> goals;
  for (std::size_t index = 0U; index < problem.goal_mask.size(); ++index) {
    if (problem.goal_mask[index] != 0U) {
      const shared::GridCell goal = CellFromIndex(*map, index);
      if (problem.projection.HardFeasible(goal)) {
        goals.push_back(goal);
      }
    }
  }
  if (goals.empty()) {
    return Failure(GlobalSearchStatus::kInvalidProblem,
                   "GLOBAL_SEARCH_PROBLEM_INVALID");
  }
  const std::int32_t start_component =
      problem.projection.ConnectedComponent(problem.start);
  if (std::ranges::none_of(goals, [&](const shared::GridCell goal) {
        return problem.projection.ConnectedComponent(goal) == start_component;
      })) {
    return Failure(GlobalSearchStatus::kNoPath, "GLOBAL_NO_KNOWN_SAFE_ROUTE");
  }

  const auto fixed_memory = FixedMemoryBytes(map->cell_count());
  if (!fixed_memory ||
      *fixed_memory > problem.config.resources.maximum_memory_bytes) {
    return Failure(
        GlobalSearchStatus::kResourceExhausted, "GLOBAL_SEARCH_RESOURCE_LIMIT",
        0U, 0U, fixed_memory.value_or(std::numeric_limits<std::size_t>::max()));
  }

  const double infinity = std::numeric_limits<double>::infinity();
  const std::size_t no_parent = std::numeric_limits<std::size_t>::max();
  std::vector<double> best_g(map->cell_count(), infinity);
  std::vector<std::size_t> parent(map->cell_count(), no_parent);
  std::vector<std::uint8_t> state(map->cell_count(), 0U);
  std::priority_queue<OpenEntry, std::vector<OpenEntry>, WorseOpenEntry> open;

  const std::size_t start_index = map->Index(problem.start);
  const double start_h =
      Heuristic(*map, problem.start, goals, problem.maximum_speed_mps);
  best_g[start_index] = 0.0;
  open.push(OpenEntry{
      .f = start_h,
      .h = start_h,
      .g = 0.0,
      .state = start_index,
      .serial = 0U,
  });
  state[start_index] = 1U;
  std::uint64_t serial = 1U;
  std::uint64_t expanded = 0U;
  std::size_t reopened = 0U;
  std::size_t generated = 0U;
  std::size_t open_peak = 1U;
  std::size_t estimated_memory =
      *fixed_memory + open.size() * sizeof(OpenEntry);

  while (!open.empty()) {
    if (problem.stop_token.stop_requested()) {
      return Failure(GlobalSearchStatus::kCanceled, "REQUEST_CANCELED",
                     expanded, open_peak, estimated_memory);
    }
    const OpenEntry current_entry = open.top();
    open.pop();
    if (current_entry.g > best_g[current_entry.state] ||
        state[current_entry.state] == 2U) {
      continue;
    }
    if (expanded >= problem.config.resources.maximum_expanded_states) {
      return Failure(GlobalSearchStatus::kResourceExhausted,
                     "GLOBAL_SEARCH_RESOURCE_LIMIT", expanded, open_peak,
                     estimated_memory);
    }
    const shared::GridCell current = CellFromIndex(*map, current_entry.state);
    if (problem.goal_mask[current_entry.state] != 0U) {
      std::vector<shared::GridCell> path =
          Reconstruct(*map, parent, start_index, current_entry.state);
      if (path.empty()) {
        return Failure(GlobalSearchStatus::kInvalidProblem,
                       "GLOBAL_SEARCH_RESULT_INVALID", expanded, open_peak,
                       estimated_memory);
      }
      return GlobalGridSearchResult{
          .status = GlobalSearchStatus::kSolved,
          .path_cells = std::move(path),
          .cost = current_entry.g,
          .expanded_states = expanded,
          .open_peak = open_peak,
          .estimated_work_memory_bytes = estimated_memory,
          .reason_code = {},
      };
    }
    state[current_entry.state] = 2U;
    ++expanded;

    for (std::size_t neighbor_index = 0U; neighbor_index < kNeighborX.size();
         ++neighbor_index) {
      if (problem.stop_token.stop_requested()) {
        return Failure(GlobalSearchStatus::kCanceled, "REQUEST_CANCELED",
                       expanded, open_peak, estimated_memory);
      }
      const std::int32_t delta_x = kNeighborX[neighbor_index];
      const std::int32_t delta_y = kNeighborY[neighbor_index];
      const shared::GridCell next{
          .x = current.x + delta_x,
          .y = current.y + delta_y,
      };
      if (!problem.projection.HardFeasible(next) ||
          !DiagonalAllowed(problem.projection, current, delta_x, delta_y)) {
        continue;
      }
      if (generated >= problem.config.resources.maximum_generated_candidates) {
        return Failure(GlobalSearchStatus::kResourceExhausted,
                       "GLOBAL_SEARCH_RESOURCE_LIMIT", expanded, open_peak,
                       estimated_memory);
      }
      ++generated;
      const std::size_t next_state = map->Index(next);
      const double candidate_g =
          current_entry.g +
          EdgeCost(problem, next, delta_x != 0 && delta_y != 0);
      if (!std::isfinite(candidate_g) || candidate_g >= best_g[next_state]) {
        continue;
      }
      if (state[next_state] == 2U) {
        if (reopened >= problem.config.resources.maximum_reopened_states) {
          return Failure(GlobalSearchStatus::kResourceExhausted,
                         "GLOBAL_SEARCH_RESOURCE_LIMIT", expanded, open_peak,
                         estimated_memory);
        }
        ++reopened;
      }
      if (open.size() >= problem.config.resources.maximum_open_states) {
        return Failure(GlobalSearchStatus::kResourceExhausted,
                       "GLOBAL_SEARCH_RESOURCE_LIMIT", expanded, open_peak,
                       estimated_memory);
      }
      const double h = Heuristic(*map, next, goals, problem.maximum_speed_mps);
      best_g[next_state] = candidate_g;
      parent[next_state] = current_entry.state;
      state[next_state] = 1U;
      open.push(OpenEntry{
          .f = candidate_g + h,
          .h = h,
          .g = candidate_g,
          .state = next_state,
          .serial = serial++,
      });
      open_peak = std::max(open_peak, open.size());
      estimated_memory = std::max(
          estimated_memory, *fixed_memory + open.size() * sizeof(OpenEntry));
      if (estimated_memory > problem.config.resources.maximum_memory_bytes) {
        return Failure(GlobalSearchStatus::kResourceExhausted,
                       "GLOBAL_SEARCH_RESOURCE_LIMIT", expanded, open_peak,
                       estimated_memory);
      }
    }
  }
  return Failure(GlobalSearchStatus::kNoPath, "GLOBAL_NO_KNOWN_SAFE_ROUTE",
                 expanded, open_peak, estimated_memory);
}

} // namespace lunar::planning::hierarchical
