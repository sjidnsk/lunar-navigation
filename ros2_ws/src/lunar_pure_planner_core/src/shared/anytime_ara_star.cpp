#include "shared/anytime_ara_star.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace lunar::pure_planning::shared::anytime {
namespace {

constexpr double kCostTolerance = 1.0e-12;
constexpr std::size_t kInvalidState = std::numeric_limits<std::size_t>::max();
constexpr std::array<double, 4> kEpsilonSchedule{2.5, 2.0, 1.5, 1.0};

struct SearchNode final {
  double path_cost{std::numeric_limits<double>::infinity()};
  std::size_t parent_state{kInvalidState};
  std::optional<std::size_t> incoming_stable_edge;
  std::vector<GraphEdge> successors;
  std::size_t open_generation{};
  bool successors_cached{};
  bool closed{};
  bool in_incons{};
  bool in_open{};
};

struct OpenEntry final {
  double anchor_key{};
  double path_cost{};
  double guidance_cost{};
  std::size_t state{};
  std::size_t sequence{};
  std::size_t generation{};
};

struct OpenEntryGreater final {
  [[nodiscard]] bool operator()(const OpenEntry& lhs,
                                const OpenEntry& rhs) const noexcept {
    return std::tie(lhs.anchor_key, lhs.path_cost, lhs.guidance_cost,
                    lhs.state, lhs.sequence) >
           std::tie(rhs.anchor_key, rhs.path_cost, rhs.guidance_cost,
                    rhs.state, rhs.sequence);
  }
};

[[nodiscard]] AraStarResult Failure(
    const AraStarStatus status, std::string reason_code,
    const std::size_t expanded_states = 0U,
    const std::size_t generated_states = 0U,
    const std::size_t reopened_states = 0U,
    const std::size_t open_peak = 0U,
    std::vector<double> epsilon_history = {},
    const bool deadline_reached = false,
    std::vector<SearchCandidate> candidates = {}) {
  return AraStarResult{
      .status = status,
      .candidates = std::move(candidates),
      .epsilon_history = std::move(epsilon_history),
      .expanded_states = expanded_states,
      .generated_states = generated_states,
      .reopened_states = reopened_states,
      .open_peak = open_peak,
      .deadline_reached = deadline_reached,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] std::string ValidateStructure(const AraStarProblem& problem) {
  if (problem.state_count == 0U || problem.start_state >= problem.state_count ||
      !problem.expand || !problem.heuristic || !problem.is_goal) {
    return "SEARCH_SHAPE_INVALID";
  }
  if (problem.config.epsilon_schedule != kEpsilonSchedule) {
    return "SEARCH_EPSILON_SCHEDULE_INVALID";
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
  if (goal >= nodes.size() || !std::isfinite(nodes[goal].path_cost)) {
    return std::nullopt;
  }

  SearchCandidate candidate{
      .states = {goal},
      .stable_edge_indices = {},
      .cost = nodes[goal].path_cost,
      .stable_index = 0U,
  };
  std::size_t current = goal;
  while (current != start) {
    if (current >= nodes.size() || nodes[current].parent_state == kInvalidState ||
        !nodes[current].incoming_stable_edge.has_value()) {
      return std::nullopt;
    }
    candidate.stable_edge_indices.push_back(*nodes[current].incoming_stable_edge);
    current = nodes[current].parent_state;
    candidate.states.push_back(current);
    if (candidate.states.size() > nodes.size()) {
      return std::nullopt;
    }
  }
  std::reverse(candidate.states.begin(), candidate.states.end());
  std::reverse(candidate.stable_edge_indices.begin(),
               candidate.stable_edge_indices.end());
  candidate.stable_index = StablePathIndex(candidate.stable_edge_indices);
  return candidate;
}

}  // namespace

AraStarResult SearchAnytimeAraStar(const AraStarProblem& problem) try {
  if (problem.control.canceled()) {
    return Failure(AraStarStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (const std::string reason = ValidateStructure(problem); !reason.empty()) {
    return Failure(AraStarStatus::kInvalidProblem, reason);
  }
  if (problem.control.expired()) {
    if (problem.control.canceled()) {
      return Failure(AraStarStatus::kCanceled, "REQUEST_CANCELED");
    }
    return Failure(AraStarStatus::kTimedOut, "TIMEOUT", 0U, 0U, 0U, 0U,
                   {}, true);
  }

  std::vector<SearchNode> nodes(problem.start_state + 1U);
  std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenEntryGreater> open;
  std::vector<std::size_t> open_states;
  std::vector<std::size_t> incons_states;
  std::vector<std::size_t> closed_states;
  std::vector<SearchCandidate> candidates;
  std::optional<std::size_t> incumbent_goal;
  std::optional<double> incumbent_cost;
  std::vector<double> epsilon_history;
  std::size_t expanded_states = 0U;
  std::size_t generated_states = 0U;
  std::size_t reopened_states = 0U;
  std::size_t open_peak = 0U;
  std::size_t live_open_count = 0U;
  std::size_t next_sequence = 0U;

  const auto finish = [&](const AraStarStatus status, std::string reason_code,
                          const bool deadline_reached = false) {
    return Failure(status, std::move(reason_code), expanded_states,
                   generated_states, reopened_states, open_peak,
                   epsilon_history, deadline_reached, candidates);
  };
  const auto resource_exhausted = [&]() {
    return Failure(AraStarStatus::kResourceExhausted, {}, expanded_states,
                   generated_states, reopened_states, open_peak,
                   std::move(epsilon_history), false, std::move(candidates));
  };
  const auto check_stop_or_deadline = [&]() -> std::optional<AraStarResult> {
    if (problem.control.canceled()) {
      return finish(AraStarStatus::kCanceled, "REQUEST_CANCELED");
    }
    if (problem.control.expired()) {
      if (problem.control.canceled()) {
        return finish(AraStarStatus::kCanceled, "REQUEST_CANCELED");
      }
      return incumbent_goal.has_value()
                 ? finish(AraStarStatus::kSolved, "SEARCH_SOLVED", true)
                 : finish(AraStarStatus::kTimedOut, "TIMEOUT", true);
    }
    return std::nullopt;
  };
  const auto read_heuristic = [&](const std::size_t state)
      -> std::optional<double> {
    const double value = problem.heuristic(state);
    if (!std::isfinite(value) || value < 0.0) {
      return std::nullopt;
    }
    return value;
  };
  const auto read_guidance = [&](const std::size_t state)
      -> std::optional<double> {
    if (!problem.guidance) {
      return 0.0;
    }
    const double value = problem.guidance(state);
    if (!std::isfinite(value) || value < 0.0) {
      return std::nullopt;
    }
    return value;
  };

  double epsilon = kEpsilonSchedule.front();
  const auto put_open = [&](const std::size_t state)
      -> std::optional<AraStarResult> {
    const auto heuristic = read_heuristic(state);
    if (!heuristic.has_value()) {
      return finish(AraStarStatus::kInvalidProblem,
                    "SEARCH_HEURISTIC_INVALID");
    }
    const auto guidance = read_guidance(state);
    if (!guidance.has_value()) {
      return finish(AraStarStatus::kInvalidProblem,
                    "SEARCH_GUIDANCE_INVALID");
    }
    const double anchor_key =
        nodes[state].path_cost + epsilon * *heuristic;
    if (!std::isfinite(anchor_key)) {
      return finish(AraStarStatus::kInvalidProblem, "SEARCH_COST_OVERFLOW");
    }

    SearchNode& node = nodes[state];
    if (!node.in_open) {
      node.in_open = true;
      open_states.push_back(state);
      ++live_open_count;
      open_peak = std::max(open_peak, live_open_count);
    }
    ++node.open_generation;
    open.push(OpenEntry{
        .anchor_key = anchor_key,
        .path_cost = node.path_cost,
        .guidance_cost = *guidance,
        .state = state,
        .sequence = next_sequence++,
        .generation = node.open_generation,
    });
    return std::nullopt;
  };
  const auto prune_invalid_open = [&]() -> std::optional<AraStarResult> {
    while (!open.empty()) {
      const OpenEntry& entry = open.top();
      const SearchNode& node = nodes[entry.state];
      if (node.in_open && node.open_generation == entry.generation &&
          node.path_cost == entry.path_cost) {
        break;
      }
      open.pop();
      if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
        return stopped;
      }
    }
    return std::nullopt;
  };
  const auto update_incumbent = [&](const std::size_t goal)
      -> std::optional<AraStarResult> {
    if (!incumbent_cost.has_value() ||
        nodes[goal].path_cost + kCostTolerance < *incumbent_cost) {
      const auto candidate =
          ReconstructCandidate(problem.start_state, goal, nodes);
      if (!candidate.has_value()) {
        return finish(AraStarStatus::kInvalidProblem,
                      "SEARCH_PARENT_CHAIN_INVALID");
      }
      incumbent_goal = goal;
      incumbent_cost = nodes[goal].path_cost;
      candidates.push_back(*candidate);
    }
    return std::nullopt;
  };
  const auto rekey = [&]() -> std::optional<AraStarResult> {
    if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
      return stopped;
    }
    std::vector<std::size_t> states;
    states.reserve(open_states.size() + incons_states.size());
    for (const std::size_t state : open_states) {
      if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
        return stopped;
      }
      if (nodes[state].in_open) {
        nodes[state].in_open = false;
        states.push_back(state);
      }
    }
    for (const std::size_t state : incons_states) {
      if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
        return stopped;
      }
      if (nodes[state].in_incons) {
        nodes[state].in_incons = false;
        states.push_back(state);
      }
    }
    for (const std::size_t state : closed_states) {
      if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
        return stopped;
      }
      nodes[state].closed = false;
    }

    open = decltype(open){};
    open_states.clear();
    incons_states.clear();
    closed_states.clear();
    live_open_count = 0U;
    for (const std::size_t state : states) {
      if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
        return stopped;
      }
      if (const auto failure = put_open(state); failure.has_value()) {
        return failure;
      }
    }
    return check_stop_or_deadline();
  };

  try {
    nodes[problem.start_state].path_cost = 0.0;
    generated_states = 1U;
    if (!read_heuristic(problem.start_state).has_value()) {
      return finish(AraStarStatus::kInvalidProblem,
                    "SEARCH_HEURISTIC_INVALID");
    }
    if (problem.is_goal(problem.start_state)) {
      epsilon_history.push_back(kEpsilonSchedule.front());
      if (const auto failure = update_incumbent(problem.start_state);
          failure.has_value()) {
        return *failure;
      }
      return finish(AraStarStatus::kSolved, "SEARCH_SOLVED");
    }
    if (const auto failure = put_open(problem.start_state);
        failure.has_value()) {
      return *failure;
    }

    for (std::size_t epsilon_index = 0U;
         epsilon_index < kEpsilonSchedule.size(); ++epsilon_index) {
      epsilon = kEpsilonSchedule[epsilon_index];
      epsilon_history.push_back(epsilon);

      const auto improve_path = [&]() -> std::optional<AraStarResult> {
        for (;;) {
          if (const auto stopped = check_stop_or_deadline();
              stopped.has_value()) {
            return stopped;
          }
          if (const auto failure = prune_invalid_open(); failure.has_value()) {
            return failure;
          }
          if (open.empty()) {
            break;
          }
          if (incumbent_cost.has_value() &&
              *incumbent_cost <= open.top().anchor_key + kCostTolerance) {
            break;
          }

          const OpenEntry current = open.top();
          open.pop();
          nodes[current.state].in_open = false;
          --live_open_count;

          if (problem.state_expandable &&
              !problem.state_expandable(current.state)) {
            if (const auto stopped = check_stop_or_deadline();
                stopped.has_value()) {
              return stopped;
            }
            continue;
          }

          nodes[current.state].closed = true;
          closed_states.push_back(current.state);
          ++expanded_states;

          if (!nodes[current.state].successors_cached) {
            problem.expand(current.state, nodes[current.state].path_cost,
                           nodes[current.state].successors);
            nodes[current.state].successors_cached = true;
            if (const auto stopped = check_stop_or_deadline();
                stopped.has_value()) {
              return stopped;
            }

            if (!nodes[current.state].successors.empty()) {
              const auto largest = std::max_element(
                  nodes[current.state].successors.begin(),
                  nodes[current.state].successors.end(),
                  [](const GraphEdge& lhs, const GraphEdge& rhs) {
                    return lhs.target_state < rhs.target_state;
                  });
              if (largest->target_state ==
                  std::numeric_limits<std::size_t>::max()) {
                return finish(AraStarStatus::kInvalidProblem,
                              "SEARCH_EDGE_INVALID");
              }
              if (largest->target_state >= nodes.size()) {
                nodes.resize(largest->target_state + 1U);
              }
            }

            for (const GraphEdge& edge : nodes[current.state].successors) {
              if (const auto stopped = check_stop_or_deadline();
                  stopped.has_value()) {
                return stopped;
              }
              if (edge.target_state >= problem.state_count ||
                  !std::isfinite(edge.cost) || edge.cost <= 0.0) {
                return finish(AraStarStatus::kInvalidProblem,
                              "SEARCH_EDGE_INVALID");
              }
            }
            if (!problem.edges_are_stably_sorted) {
              std::stable_sort(
                  nodes[current.state].successors.begin(),
                  nodes[current.state].successors.end(),
                  [](const GraphEdge& lhs, const GraphEdge& rhs) {
                    return std::tie(lhs.stable_index, lhs.target_state,
                                    lhs.cost) <
                           std::tie(rhs.stable_index, rhs.target_state,
                                    rhs.cost);
                  });
            }
          }

          for (const GraphEdge& edge : nodes[current.state].successors) {
            if (const auto stopped = check_stop_or_deadline();
                stopped.has_value()) {
              return stopped;
            }
            const double tentative =
                nodes[current.state].path_cost + edge.cost;
            if (!std::isfinite(tentative)) {
              return finish(AraStarStatus::kInvalidProblem,
                            "SEARCH_COST_OVERFLOW");
            }
            SearchNode& target = nodes[edge.target_state];
            if (tentative + kCostTolerance >= target.path_cost) {
              continue;
            }

            const bool first_generation = !std::isfinite(target.path_cost);
            target.path_cost = tentative;
            target.parent_state = current.state;
            target.incoming_stable_edge = edge.stable_index;
            if (first_generation) {
              ++generated_states;
            }
            if (problem.on_relaxed) {
              problem.on_relaxed(edge.target_state, tentative);
            }

            const bool target_is_goal = problem.is_goal(edge.target_state);
            if (target_is_goal) {
              if (const auto failure = update_incumbent(edge.target_state);
                  failure.has_value()) {
                return failure;
              }
            }
            if (const auto stopped = check_stop_or_deadline();
                stopped.has_value()) {
              return stopped;
            }
            if (target_is_goal &&
                problem.config.stop_after_first_solution) {
              return finish(AraStarStatus::kSolved, "SEARCH_SOLVED");
            }

            if (target.closed) {
              if (!target.in_incons) {
                target.in_incons = true;
                incons_states.push_back(edge.target_state);
                ++reopened_states;
              }
            } else if (const auto failure = put_open(edge.target_state);
                       failure.has_value()) {
              return failure;
            }
          }
          if (const auto stopped = check_stop_or_deadline();
              stopped.has_value()) {
            return stopped;
          }
        }
        return std::nullopt;
      };

      for (;;) {
        if (const auto result = improve_path(); result.has_value()) {
          return *result;
        }
        if (const auto stopped = check_stop_or_deadline();
            stopped.has_value()) {
          return *stopped;
        }
        if (epsilon_index + 1U < kEpsilonSchedule.size() ||
            incons_states.empty()) {
          break;
        }
        if (const auto failure = rekey(); failure.has_value()) {
          return *failure;
        }
      }

      if (const auto failure = prune_invalid_open(); failure.has_value()) {
        return *failure;
      }
      if (open.empty() && incons_states.empty()) {
        return incumbent_goal.has_value()
                   ? finish(AraStarStatus::kSolved, "SEARCH_SOLVED")
                   : finish(AraStarStatus::kNoPath, "SEARCH_NO_PATH");
      }
      if (epsilon_index + 1U < kEpsilonSchedule.size()) {
        epsilon = kEpsilonSchedule[epsilon_index + 1U];
        if (const auto failure = rekey(); failure.has_value()) {
          return *failure;
        }
      }
    }

    return incumbent_goal.has_value()
               ? finish(AraStarStatus::kSolved, "SEARCH_SOLVED")
               : finish(AraStarStatus::kNoPath, "SEARCH_NO_PATH");
  } catch (const std::bad_alloc&) {
    return resource_exhausted();
  } catch (const std::length_error&) {
    return resource_exhausted();
  }
} catch (const std::bad_alloc&) {
  return Failure(AraStarStatus::kResourceExhausted, {});
} catch (const std::length_error&) {
  return Failure(AraStarStatus::kResourceExhausted, {});
}

}  // namespace lunar::pure_planning::shared::anytime
