#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "lunar_pure_planner_core/search_control.hpp"
#include "lunar_pure_planner_core/types/planning_request.hpp"
#include "shared/ara_star.hpp"

namespace lunar::pure_planning::shared::anytime {

using ExpandFn =
    std::function<void(std::size_t, double, std::vector<GraphEdge>&)>;
using HeuristicFn = std::function<double(std::size_t)>;
using GuidanceFn = std::function<double(std::size_t)>;
using GoalFn = std::function<bool(std::size_t)>;
using StateExpandableFn = std::function<bool(std::size_t)>;
using RelaxedFn = std::function<void(std::size_t, double)>;

struct AraStarProblem final {
  std::size_t state_count{};
  std::size_t start_state{};
  ExpandFn expand;
  HeuristicFn heuristic;
  GuidanceFn guidance;
  GoalFn is_goal;
  StateExpandableFn state_expandable;
  RelaxedFn on_relaxed;
  bool edges_are_stably_sorted{};
  AnytimeSearchConfig config;
  SearchControl control;
};

enum class AraStarStatus {
  kSolved,
  kNoPath,
  kCanceled,
  kTimedOut,
  kResourceExhausted,
  kInvalidProblem,
};

struct AraStarResult final {
  AraStarStatus status{AraStarStatus::kInvalidProblem};
  std::vector<SearchCandidate> candidates;
  std::vector<double> epsilon_history;
  std::size_t expanded_states{};
  std::size_t generated_states{};
  std::size_t reopened_states{};
  std::size_t open_peak{};
  bool deadline_reached{};
  std::string reason_code;
};

[[nodiscard]] AraStarResult SearchAnytimeAraStar(const AraStarProblem& problem);

}  // namespace lunar::pure_planning::shared::anytime
