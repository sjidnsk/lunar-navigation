#include "shared/ara_star.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "shared/candidate_ranker.hpp"

namespace lunar::planning::shared {
namespace {

constexpr double kCostTolerance = 1.0e-12;
constexpr std::size_t kInvalidState =
    std::numeric_limits<std::size_t>::max();

struct SearchNode final {
  double path_cost{std::numeric_limits<double>::infinity()};
  std::size_t parent_state{kInvalidState};
  std::size_t incoming_stable_edge{kInvalidState};
  std::size_t open_sequence{};
  bool has_open_sequence{};
};

struct OpenEntry final {
  double weighted_cost{};
  double path_cost{};
  std::size_t state{};
  std::size_t sequence{};
};

struct OpenEntryLess final {
  [[nodiscard]] bool operator()(
      const OpenEntry& lhs, const OpenEntry& rhs) const noexcept {
    return std::tie(
               lhs.weighted_cost, lhs.path_cost, lhs.state, lhs.sequence) <
           std::tie(
               rhs.weighted_cost, rhs.path_cost, rhs.state, rhs.sequence);
  }
};

[[nodiscard]] AraStarResult Failure(
    const AraStarStatus status, std::string reason_code,
    const std::size_t expanded_states = 0U,
    const std::size_t reopened_states = 0U) {
  return AraStarResult{
      .status = status,
      .candidates = {},
      .expanded_states = expanded_states,
      .reopened_states = reopened_states,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] std::string ValidateProblem(const AraStarProblem& problem) {
  if (problem.state_count == 0U ||
      problem.start_state >= problem.state_count ||
      problem.outgoing_edges.size() != problem.state_count ||
      problem.heuristic.size() != problem.state_count ||
      problem.goal_mask.size() != problem.state_count) {
    return "SEARCH_SHAPE_INVALID";
  }
  if (!std::isfinite(problem.config.initial_epsilon) ||
      !std::isfinite(problem.config.epsilon_decrement) ||
      !std::isfinite(problem.config.target_epsilon) ||
      problem.config.initial_epsilon < problem.config.target_epsilon ||
      problem.config.target_epsilon < 1.0 ||
      (problem.config.initial_epsilon > problem.config.target_epsilon &&
       problem.config.epsilon_decrement <= 0.0)) {
    return "SEARCH_EPSILON_INVALID";
  }
  if (std::ranges::none_of(problem.goal_mask, [](const std::uint8_t value) {
        return value != 0U;
      })) {
    return "SEARCH_GOAL_EMPTY";
  }
  for (std::size_t state = 0U; state < problem.state_count; ++state) {
    if (!std::isfinite(problem.heuristic[state]) ||
        problem.heuristic[state] < 0.0 || problem.goal_mask[state] > 1U) {
      return "SEARCH_STATE_VALUE_INVALID";
    }
    for (const GraphEdge& edge : problem.outgoing_edges[state]) {
      if (edge.target_state >= problem.state_count ||
          !std::isfinite(edge.cost) || edge.cost <= 0.0) {
        return "SEARCH_EDGE_INVALID";
      }
    }
  }

  std::size_t edge_count = 0U;
  for (const auto& edges : problem.outgoing_edges) {
    if (edge_count > std::numeric_limits<std::size_t>::max() - edges.size()) {
      return "SEARCH_MEMORY_LIMIT";
    }
    edge_count += edges.size();
  }
  return {};
}

[[nodiscard]] std::size_t StablePathIndex(
    const std::vector<std::size_t>& stable_edges) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const std::size_t edge : stable_edges) {
    std::uint64_t value = static_cast<std::uint64_t>(edge);
    for (std::size_t byte = 0U; byte < sizeof(value); ++byte) {
      hash ^= value & 0xffU;
      hash *= 1099511628211ULL;
      value >>= 8U;
    }
  }
  return static_cast<std::size_t>(hash);
}

[[nodiscard]] std::optional<SearchCandidate> ReconstructCandidate(
    const std::size_t start, const std::size_t goal,
    const std::vector<SearchNode>& nodes) {
  SearchCandidate candidate{
      .states = {},
      .stable_edge_indices = {},
      .cost = nodes[goal].path_cost,
      .stable_index = 0U,
  };
  std::size_t current = goal;
  candidate.states.push_back(current);
  while (current != start) {
    if (current >= nodes.size() || nodes[current].parent_state == kInvalidState ||
        nodes[current].incoming_stable_edge == kInvalidState) {
      return std::nullopt;
    }
    candidate.stable_edge_indices.push_back(
        nodes[current].incoming_stable_edge);
    current = nodes[current].parent_state;
    candidate.states.push_back(current);
    if (candidate.states.size() > nodes.size()) {
      return std::nullopt;
    }
  }
  std::reverse(candidate.states.begin(), candidate.states.end());
  std::reverse(
      candidate.stable_edge_indices.begin(),
      candidate.stable_edge_indices.end());
  candidate.stable_index = StablePathIndex(candidate.stable_edge_indices);
  return candidate;
}

[[nodiscard]] std::vector<SearchCandidate> RankSearchCandidates(
    std::vector<SearchCandidate> candidates) {
  std::vector<CandidateScore> scores;
  scores.reserve(candidates.size());
  for (const SearchCandidate& candidate : candidates) {
    scores.push_back(CandidateScore{
        .cost = candidate.cost,
        .stable_index = candidate.stable_index,
        .fully_hard_validated = true,
    });
  }
  const std::vector<CandidateScore> ranked = RankCandidates(scores);
  std::vector<SearchCandidate> result;
  result.reserve(ranked.size());
  for (const CandidateScore& score : ranked) {
    const auto found = std::find_if(
        candidates.begin(), candidates.end(),
        [&](const SearchCandidate& candidate) {
          return candidate.stable_index == score.stable_index &&
                 candidate.cost == score.cost;
        });
    if (found != candidates.end()) {
      result.push_back(std::move(*found));
    }
  }
  return result;
}

}  // namespace

AraStarResult SearchAraStar(
    const AraStarProblem& problem, const std::stop_token stop_token) try {
  if (stop_token.stop_requested()) {
    return Failure(AraStarStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (const std::string reason = ValidateProblem(problem); !reason.empty()) {
    return Failure(AraStarStatus::kInvalidProblem, reason);
  }

  std::vector<SearchNode> nodes(problem.state_count);
  std::vector<std::uint8_t> closed(problem.state_count, 0U);
  std::set<OpenEntry, OpenEntryLess> open;
  std::vector<std::optional<OpenEntry>> open_by_state(problem.state_count);
  std::set<std::size_t> incons;
  std::vector<SearchCandidate> candidates;
  std::optional<double> best_goal_cost;
  std::size_t expansions = 0U;
  std::size_t reopens = 0U;
  std::size_t next_sequence = 0U;
  double epsilon = problem.config.initial_epsilon;

  const auto finish = [&](const AraStarStatus status,
                          std::string reason_code) {
    return AraStarResult{
        .status = status,
        .candidates = RankSearchCandidates(candidates),
        .expanded_states = expansions,
        .reopened_states = reopens,
        .reason_code = std::move(reason_code),
    };
  };
  const auto make_open_entry = [&](const std::size_t state) {
    return OpenEntry{
        .weighted_cost = nodes[state].path_cost +
            epsilon * problem.heuristic[state],
        .path_cost = nodes[state].path_cost,
        .state = state,
        .sequence = nodes[state].open_sequence,
    };
  };
  const auto put_open = [&](const std::size_t state,
                            const bool preserve_sequence) {
    if (open_by_state[state].has_value()) {
      open.erase(*open_by_state[state]);
      open_by_state[state].reset();
    }
    if (!preserve_sequence || !nodes[state].has_open_sequence) {
      nodes[state].open_sequence = next_sequence++;
      nodes[state].has_open_sequence = true;
    }
    const OpenEntry entry = make_open_entry(state);
    open.insert(entry);
    open_by_state[state] = entry;
  };
  const auto capture_candidate = [&](const std::size_t goal)
      -> std::optional<AraStarResult> {
    auto candidate =
        ReconstructCandidate(problem.start_state, goal, nodes);
    if (!candidate.has_value()) {
      return finish(
          AraStarStatus::kInvalidProblem,
          "SEARCH_PARENT_CHAIN_INVALID");
    }
    const auto existing = std::find_if(
        candidates.begin(), candidates.end(),
        [&](const SearchCandidate& value) {
          return value.stable_edge_indices ==
                 candidate->stable_edge_indices;
        });
    if (existing == candidates.end()) {
      candidates.push_back(*candidate);
    } else if (candidate->cost + kCostTolerance < existing->cost) {
      *existing = *candidate;
    }
    best_goal_cost = best_goal_cost.has_value()
        ? std::min(*best_goal_cost, candidate->cost)
        : candidate->cost;
    return std::nullopt;
  };

  nodes[problem.start_state].path_cost = 0.0;
  if (problem.goal_mask[problem.start_state] != 0U) {
    if (const auto captured = capture_candidate(problem.start_state);
        captured.has_value()) {
      return *captured;
    }
    return finish(AraStarStatus::kSolved, "SEARCH_SOLVED");
  }
  put_open(problem.start_state, false);

  for (;;) {
    while (!open.empty()) {
      if (stop_token.stop_requested()) {
        return finish(AraStarStatus::kCanceled, "REQUEST_CANCELED");
      }
      if (best_goal_cost.has_value() &&
          *best_goal_cost <=
              open.begin()->weighted_cost + kCostTolerance) {
        break;
      }
      const OpenEntry current = *open.begin();
      open.erase(open.begin());
      open_by_state[current.state].reset();
      nodes[current.state].has_open_sequence = false;
      closed[current.state] = 1U;
      ++expansions;

      std::vector<GraphEdge> ordered_edges =
          problem.outgoing_edges[current.state];
      std::stable_sort(
          ordered_edges.begin(), ordered_edges.end(),
          [](const GraphEdge& lhs, const GraphEdge& rhs) {
            return std::tie(lhs.stable_index, lhs.target_state, lhs.cost) <
                   std::tie(rhs.stable_index, rhs.target_state, rhs.cost);
          });
      for (const GraphEdge& edge : ordered_edges) {
        const double tentative = nodes[current.state].path_cost + edge.cost;
        if (!std::isfinite(tentative)) {
          return finish(
              AraStarStatus::kInvalidProblem,
              "SEARCH_COST_OVERFLOW");
        }
        if (tentative + kCostTolerance >=
            nodes[edge.target_state].path_cost) {
          continue;
        }
        nodes[edge.target_state].path_cost = tentative;
        nodes[edge.target_state].parent_state = current.state;
        nodes[edge.target_state].incoming_stable_edge = edge.stable_index;

        if (closed[edge.target_state] != 0U) {
          if (incons.insert(edge.target_state).second) {
            ++reopens;
          }
        } else {
          put_open(edge.target_state, false);
        }

        if (problem.goal_mask[edge.target_state] != 0U) {
          if (const auto captured = capture_candidate(edge.target_state);
              captured.has_value()) {
            return *captured;
          }
        }
      }
    }

    if (epsilon <=
        problem.config.target_epsilon + kCostTolerance) {
      return candidates.empty()
          ? finish(AraStarStatus::kNoPath, "SEARCH_NO_PATH")
          : finish(AraStarStatus::kSolved, "SEARCH_SOLVED");
    }
    if (open.empty() && incons.empty()) {
      return candidates.empty()
          ? finish(AraStarStatus::kNoPath, "SEARCH_NO_PATH")
          : finish(AraStarStatus::kSolved, "SEARCH_SOLVED");
    }

    const double next_epsilon = std::max(
        problem.config.target_epsilon,
        epsilon - problem.config.epsilon_decrement);
    if (next_epsilon >= epsilon) {
      return finish(
          AraStarStatus::kInvalidProblem,
          "SEARCH_EPSILON_NO_PROGRESS");
    }
    epsilon = next_epsilon;

    std::set<std::size_t> next_open_states;
    for (std::size_t state = 0U; state < open_by_state.size(); ++state) {
      if (open_by_state[state].has_value()) {
        next_open_states.insert(state);
      }
    }
    next_open_states.insert(incons.begin(), incons.end());
    open.clear();
    std::fill(open_by_state.begin(), open_by_state.end(), std::nullopt);
    std::fill(closed.begin(), closed.end(), 0U);
    incons.clear();
    for (const std::size_t state : next_open_states) {
      put_open(state, true);
    }
  }
} catch (const std::bad_alloc&) {
  return Failure(
      AraStarStatus::kResourceExhausted,
      "SEARCH_ALLOCATION_FAILED");
}

}  // namespace lunar::planning::shared
