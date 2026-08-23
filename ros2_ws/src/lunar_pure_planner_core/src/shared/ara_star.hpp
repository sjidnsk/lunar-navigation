#pragma once

#include <cstddef>
#include <cstdint>
#include <stop_token>
#include <string>
#include <vector>

#include "lunar_pure_planner_core/types/planner_config.hpp"

namespace lunar::pure_planning::shared {

struct GraphEdge final {
  std::size_t target_state{};
  double cost{};
  std::size_t stable_index{};

  bool operator==(const GraphEdge&) const = default;
};

struct AraStarProblem final {
  std::size_t state_count{};
  std::size_t start_state{};
  std::vector<std::vector<GraphEdge>> outgoing_edges;
  std::vector<double> heuristic;
  std::vector<std::uint8_t> goal_mask;
  AraStarConfig config;
};

struct SearchCandidate final {
  std::vector<std::size_t> states;
  std::vector<std::size_t> stable_edge_indices;
  double cost{};
  std::size_t stable_index{};

  bool operator==(const SearchCandidate&) const = default;
};

enum class AraStarStatus {
  kSolved,
  kNoPath,
  kCanceled,
  kResourceExhausted,
  kInvalidProblem,
};

struct AraStarResult final {
  AraStarStatus status{AraStarStatus::kInvalidProblem};
  std::vector<SearchCandidate> candidates;
  std::size_t expanded_states{};
  std::size_t reopened_states{};
  std::string reason_code;
};

[[nodiscard]] AraStarResult SearchAraStar(
    const AraStarProblem& problem, std::stop_token stop_token);

}  // namespace lunar::pure_planning::shared
