#include "shared/anytime_ara_star.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "shared/controlled_work.hpp"

namespace lunar::pure_planning::shared::anytime {
namespace {

constexpr double kCostTolerance = 1.0e-12;
constexpr std::size_t kInvalidState = std::numeric_limits<std::size_t>::max();
constexpr std::array<double, 4> kEpsilonSchedule{2.5, 2.0, 1.5, 1.0};

struct SearchNode final {
  double path_cost{std::numeric_limits<double>::infinity()};
  std::size_t parent_state{kInvalidState};
  std::optional<std::size_t> incoming_stable_edge;
  std::size_t open_sequence{};
  bool has_open_sequence{};
};

struct OpenEntry final {
  double weighted_cost{};
  double path_cost{};
  double guidance_cost{};
  std::size_t state{};
  std::size_t sequence{};
};

struct OpenEntryLess final {
  [[nodiscard]] bool operator()(const OpenEntry& lhs,
                                const OpenEntry& rhs) const noexcept {
    return std::tie(lhs.guidance_cost, lhs.weighted_cost, lhs.path_cost,
                    lhs.state, lhs.sequence) <
           std::tie(rhs.guidance_cost, rhs.weighted_cost, rhs.path_cost,
                    rhs.state, rhs.sequence);
  }
};

[[nodiscard]] AraStarResult Failure(
    const AraStarStatus status, std::string reason_code,
    const std::size_t expanded_states = 0U,
    std::vector<double> epsilon_history = {}, const bool deadline_reached = false,
    std::vector<SearchCandidate> candidates = {}) {
  return AraStarResult{
      .status = status,
      .candidates = std::move(candidates),
      .epsilon_history = std::move(epsilon_history),
      .expanded_states = expanded_states,
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
    return Failure(AraStarStatus::kTimedOut, "TIMEOUT", 0U, {}, true);
  }

  const auto initialization_failure = [&](const std::string_view reason) {
    const bool canceled = reason == "REQUEST_CANCELED";
    return Failure(canceled ? AraStarStatus::kCanceled
                            : AraStarStatus::kTimedOut,
                   std::string{reason}, 0U, {}, !canceled);
  };
  std::vector<SearchNode> nodes;
  if (const auto stopped = ControlledFill(
          &nodes, problem.state_count, SearchNode{}, problem.control);
      stopped.has_value()) {
    return initialization_failure(*stopped);
  }
  std::vector<std::uint8_t> closed;
  if (const auto stopped = ControlledFill(
          &closed, problem.state_count, std::uint8_t{0U}, problem.control);
      stopped.has_value()) {
    return initialization_failure(*stopped);
  }
  std::vector<std::uint8_t> expanded;
  if (const auto stopped = ControlledFill(
          &expanded, problem.state_count, std::uint8_t{0U}, problem.control);
      stopped.has_value()) {
    return initialization_failure(*stopped);
  }
  std::vector<std::vector<GraphEdge>> cached_edges;
  if (const auto stopped = ControlledFill(
          &cached_edges, problem.state_count, std::vector<GraphEdge>{},
          problem.control);
      stopped.has_value()) {
    return initialization_failure(*stopped);
  }
  std::set<OpenEntry, OpenEntryLess> open;
  std::vector<std::optional<OpenEntry>> open_by_state;
  if (const auto stopped = ControlledFill(
          &open_by_state, problem.state_count, std::optional<OpenEntry>{},
          problem.control);
      stopped.has_value()) {
    return initialization_failure(*stopped);
  }
  std::set<std::size_t> incons;
  std::vector<SearchCandidate> candidates;
  std::optional<std::size_t> incumbent_goal;
  std::optional<double> incumbent_cost;
  std::vector<double> epsilon_history;
  std::size_t expanded_states = 0U;
  std::size_t next_sequence = 0U;

  const auto finish = [&](const AraStarStatus status, std::string reason_code,
                          const bool deadline_reached = false) {
    return Failure(status, std::move(reason_code), expanded_states,
                   epsilon_history, deadline_reached, candidates);
  };
  const auto resource_exhausted = [&]() {
    return AraStarResult{
        .status = AraStarStatus::kResourceExhausted,
        .candidates = std::move(candidates),
        .epsilon_history = std::move(epsilon_history),
        .expanded_states = expanded_states,
        .deadline_reached = false,
        .reason_code = {},
    };
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
  const auto make_open_entry = [&](const std::size_t state,
                                   const double heuristic,
                                   const double guidance) {
    return OpenEntry{
        .weighted_cost = nodes[state].path_cost + epsilon * heuristic,
        .path_cost = nodes[state].path_cost,
        .guidance_cost = guidance,
        .state = state,
        .sequence = nodes[state].open_sequence,
    };
  };
  const auto put_open = [&](const std::size_t state,
                            const bool preserve_sequence)
      -> std::optional<AraStarResult> {
    const auto heuristic = read_heuristic(state);
    if (!heuristic.has_value()) {
      return finish(AraStarStatus::kInvalidProblem, "SEARCH_HEURISTIC_INVALID");
    }
    const auto guidance = read_guidance(state);
    if (!guidance.has_value()) {
      return finish(AraStarStatus::kInvalidProblem, "SEARCH_GUIDANCE_INVALID");
    }
    if (open_by_state[state].has_value()) {
      open.erase(*open_by_state[state]);
      open_by_state[state].reset();
    }
    if (!preserve_sequence || !nodes[state].has_open_sequence) {
      nodes[state].open_sequence = next_sequence++;
      nodes[state].has_open_sequence = true;
    }
    const OpenEntry entry = make_open_entry(state, *heuristic, *guidance);
    if (!std::isfinite(entry.weighted_cost)) {
      return finish(AraStarStatus::kInvalidProblem, "SEARCH_COST_OVERFLOW");
    }
    open.insert(entry);
    open_by_state[state] = entry;
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
    for (std::size_t state = 0U; state < open_by_state.size(); ++state) {
      if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
        return stopped;
      }
      if (open_by_state[state].has_value()) {
        states.push_back(state);
      }
    }
    for (auto it = incons.begin(); it != incons.end();) {
      if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
        return stopped;
      }
      states.push_back(*it);
      it = incons.erase(it);
    }
    for (auto it = open.begin(); it != open.end();) {
      if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
        return stopped;
      }
      it = open.erase(it);
    }
    for (auto& entry : open_by_state) {
      if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
        return stopped;
      }
      entry.reset();
    }
    for (std::uint8_t& value : closed) {
      if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
        return stopped;
      }
      value = 0U;
    }
    for (const std::size_t state : states) {
      if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
        return stopped;
      }
      if (const auto failure = put_open(state, true); failure.has_value()) {
        return failure;
      }
    }
    if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
      return stopped;
    }
    return std::nullopt;
  };

  try {
  nodes[problem.start_state].path_cost = 0.0;
  if (!read_heuristic(problem.start_state).has_value()) {
    return finish(AraStarStatus::kInvalidProblem, "SEARCH_HEURISTIC_INVALID");
  }
  if (problem.is_goal(problem.start_state)) {
    epsilon_history.push_back(kEpsilonSchedule.front());
    if (const auto failure = update_incumbent(problem.start_state);
        failure.has_value()) {
      return *failure;
    }
    return finish(AraStarStatus::kSolved, "SEARCH_SOLVED");
  }
  if (const auto failure = put_open(problem.start_state, false);
      failure.has_value()) {
    return *failure;
  }

  for (std::size_t epsilon_index = 0U;
       epsilon_index < kEpsilonSchedule.size(); ++epsilon_index) {
    epsilon = kEpsilonSchedule[epsilon_index];
    epsilon_history.push_back(epsilon);

    const auto improve_path = [&]() -> std::optional<AraStarResult> {
      while (!open.empty()) {
        if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
          return stopped;
        }
        if (incumbent_cost.has_value() &&
            *incumbent_cost <= open.begin()->weighted_cost + kCostTolerance) {
          break;
        }

        const OpenEntry current = *open.begin();
        open.erase(open.begin());
        open_by_state[current.state].reset();
        nodes[current.state].has_open_sequence = false;
        closed[current.state] = 1U;
        ++expanded_states;

        if (expanded[current.state] == 0U) {
          problem.expand(current.state, cached_edges[current.state]);
          expanded[current.state] = 1U;
          for (const GraphEdge& edge : cached_edges[current.state]) {
            if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
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
                cached_edges[current.state].begin(), cached_edges[current.state].end(),
                [](const GraphEdge& lhs, const GraphEdge& rhs) {
                  return std::tie(lhs.stable_index, lhs.target_state, lhs.cost) <
                         std::tie(rhs.stable_index, rhs.target_state, rhs.cost);
                });
          }
        }

        for (const GraphEdge& edge : cached_edges[current.state]) {
          if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
            return stopped;
          }
          const auto target_heuristic = read_heuristic(edge.target_state);
          if (!target_heuristic.has_value()) {
            return finish(AraStarStatus::kInvalidProblem,
                          "SEARCH_HEURISTIC_INVALID");
          }
          const double tentative = nodes[current.state].path_cost + edge.cost;
          if (!std::isfinite(tentative)) {
            return finish(AraStarStatus::kInvalidProblem, "SEARCH_COST_OVERFLOW");
          }
          if (tentative + kCostTolerance >= nodes[edge.target_state].path_cost) {
            continue;
          }

          nodes[edge.target_state].path_cost = tentative;
          nodes[edge.target_state].parent_state = current.state;
          nodes[edge.target_state].incoming_stable_edge = edge.stable_index;

          if (problem.is_goal(edge.target_state)) {
            if (const auto failure = update_incumbent(edge.target_state);
                failure.has_value()) {
              return failure;
            }
            if (problem.config.stop_after_first_solution) {
              return finish(AraStarStatus::kSolved, "SEARCH_SOLVED");
            }
          }
          if (closed[edge.target_state] != 0U) {
            incons.insert(edge.target_state);
          } else if (const auto failure = put_open(edge.target_state, false);
                     failure.has_value()) {
            return failure;
          }
        }
      }
      return std::nullopt;
    };

    for (;;) {
      if (const auto result = improve_path(); result.has_value()) {
        return *result;
      }
      if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
        return *stopped;
      }
      if (epsilon_index + 1U < kEpsilonSchedule.size() || incons.empty()) {
        break;
      }
      if (const auto failure = rekey(); failure.has_value()) {
        return *failure;
      }
      if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
        return *stopped;
      }
    }

    if (open.empty() && incons.empty()) {
      return incumbent_goal.has_value()
          ? finish(AraStarStatus::kSolved, "SEARCH_SOLVED")
          : finish(AraStarStatus::kNoPath, "SEARCH_NO_PATH");
    }
    if (epsilon_index + 1U < kEpsilonSchedule.size()) {
      epsilon = kEpsilonSchedule[epsilon_index + 1U];
      if (const auto failure = rekey(); failure.has_value()) {
        return *failure;
      }
      if (const auto stopped = check_stop_or_deadline(); stopped.has_value()) {
        return *stopped;
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
  return AraStarResult{
      .status = AraStarStatus::kResourceExhausted,
      .candidates = {},
      .epsilon_history = {},
      .expanded_states = 0U,
      .deadline_reached = false,
      .reason_code = {},
  };
} catch (const std::length_error&) {
  return AraStarResult{
      .status = AraStarStatus::kResourceExhausted,
      .candidates = {},
      .epsilon_history = {},
      .expanded_states = 0U,
      .deadline_reached = false,
      .reason_code = {},
  };
}

}  // namespace lunar::pure_planning::shared::anytime
