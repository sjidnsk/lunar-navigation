#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "lunar_path_planner/v3/search/platform_adapter.hpp"

namespace lunar::planning::v3 {

enum class SearchStatus {
  kSolved,
  kNoPath,
  kResourceLimit,
  kInvalidAdapterCost,
};

template <class State, class Edge>
struct SearchPath final {
  std::string stable_path_id;
  std::vector<State> states;
  std::vector<Edge> edges;
  std::vector<std::string> edge_ids;
  DurationNanoseconds total_time;
  SecondaryCostVector secondary_costs;
  bool fully_hard_validated{};
};

template <class State, class Edge>
struct SearchResult final {
  SearchStatus status{SearchStatus::kNoPath};
  std::vector<SearchPath<State, Edge>> candidates;
  double achieved_epsilon{};
  std::size_t expansions{};
  std::size_t generated_states{};
  std::string reason_code;
};

namespace search_detail {

inline constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;

[[nodiscard]] inline long double ToPrioritySeconds(
    const std::int64_t nanoseconds) {
  return static_cast<long double>(nanoseconds) /
         static_cast<long double>(kNanosecondsPerSecond);
}

[[nodiscard]] inline bool IsValidDuration(
    const DurationNanoseconds& duration) {
  return duration.value.count() >= 0;
}

[[nodiscard]] inline bool IsValidSecondaryCosts(
    const SecondaryCostVector& costs) {
  return std::isfinite(costs.energy) && costs.energy >= 0.0 &&
         std::isfinite(costs.risk) && costs.risk >= 0.0 &&
         std::isfinite(costs.smoothness) && costs.smoothness >= 0.0;
}

[[nodiscard]] inline std::optional<std::int64_t> CheckedTimeSum(
    const std::int64_t lhs,
    const std::int64_t rhs) {
  if (rhs < 0 || lhs > std::numeric_limits<std::int64_t>::max() - rhs) {
    return std::nullopt;
  }
  return lhs + rhs;
}

[[nodiscard]] inline std::optional<SecondaryCostVector>
CheckedSecondaryCostSum(const SecondaryCostVector& lhs,
                        const SecondaryCostVector& rhs) {
  if (!IsValidSecondaryCosts(lhs) || !IsValidSecondaryCosts(rhs)) {
    return std::nullopt;
  }
  const SecondaryCostVector sum{
      .energy = lhs.energy + rhs.energy,
      .risk = lhs.risk + rhs.risk,
      .smoothness = lhs.smoothness + rhs.smoothness,
  };
  if (!IsValidSecondaryCosts(sum)) {
    return std::nullopt;
  }
  return sum;
}

[[nodiscard]] inline std::string StableEdgeIdHash(
    const std::vector<std::string>& edge_ids) {
  constexpr std::uint64_t kOffsetBasis = 14695981039346656037ULL;
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  std::uint64_t hash = kOffsetBasis;
  const auto add_byte = [&hash](const std::uint8_t byte) {
    hash ^= static_cast<std::uint64_t>(byte);
    hash *= kPrime;
  };
  for (const auto& edge_id : edge_ids) {
    const auto size = static_cast<std::uint64_t>(edge_id.size());
    for (std::size_t index = 0U; index < sizeof(size); ++index) {
      add_byte(static_cast<std::uint8_t>(
          (size >> (index * 8U)) & static_cast<std::uint64_t>(0xffU)));
    }
    for (const unsigned char character : edge_id) {
      add_byte(static_cast<std::uint8_t>(character));
    }
  }

  constexpr char kHexDigits[] = "0123456789abcdef";
  std::string result = "fnv1a64:";
  result.resize(result.size() + 16U);
  const std::size_t offset = result.size() - 16U;
  for (std::size_t index = 0U; index < 16U; ++index) {
    const auto shift = static_cast<unsigned>((15U - index) * 4U);
    result[offset + index] =
        kHexDigits[static_cast<std::size_t>((hash >> shift) & 0x0fU)];
  }
  return result;
}

template <class State, class Edge>
struct SearchNode final {
  State state;
  std::int64_t g_time_ns{};
  std::int64_t heuristic_time_ns{};
  SecondaryCostVector secondary_costs{};
  std::optional<StateKey> predecessor_key;
  std::optional<Edge> incoming_edge;
  std::string incoming_edge_id;
  std::uint64_t open_sequence{};
  bool has_g_time{false};
  bool has_open_sequence{false};
};

struct OpenEntry final {
  long double priority_seconds{};
  std::int64_t g_time_ns{};
  StateKey state_key{};
  std::uint64_t insertion_sequence{};
};

struct OpenEntryLess final {
  [[nodiscard]] bool operator()(const OpenEntry& lhs,
                                const OpenEntry& rhs) const {
    return std::tie(lhs.priority_seconds,
                    lhs.g_time_ns,
                    lhs.state_key,
                    lhs.insertion_sequence) <
           std::tie(rhs.priority_seconds,
                    rhs.g_time_ns,
                    rhs.state_key,
                    rhs.insertion_sequence);
  }
};

template <class State, class Edge>
[[nodiscard]] std::optional<SearchPath<State, Edge>> ReconstructPath(
    const std::map<StateKey, SearchNode<State, Edge>>& nodes,
    const StateKey terminal_key) {
  SearchPath<State, Edge> path;
  path.fully_hard_validated = true;

  StateKey key = terminal_key;
  for (std::size_t count = 0U; count <= nodes.size(); ++count) {
    const auto node = nodes.find(key);
    if (node == nodes.end()) {
      return std::nullopt;
    }
    path.states.push_back(node->second.state);
    if (!node->second.predecessor_key.has_value()) {
      std::reverse(path.states.begin(), path.states.end());
      std::reverse(path.edges.begin(), path.edges.end());
      std::reverse(path.edge_ids.begin(), path.edge_ids.end());
      path.total_time =
          DurationNanoseconds{std::chrono::nanoseconds{
              nodes.at(terminal_key).g_time_ns}};
      path.secondary_costs = nodes.at(terminal_key).secondary_costs;
      path.stable_path_id = StableEdgeIdHash(path.edge_ids);
      return path;
    }
    if (!node->second.incoming_edge.has_value() ||
        node->second.incoming_edge_id.empty()) {
      return std::nullopt;
    }
    path.edges.push_back(*node->second.incoming_edge);
    path.edge_ids.push_back(node->second.incoming_edge_id);
    key = *node->second.predecessor_key;
  }
  return std::nullopt;
}

template <class State, class Edge>
void SortCandidates(std::vector<SearchPath<State, Edge>>& candidates) {
  std::sort(candidates.begin(),
            candidates.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tie(lhs.total_time.value, lhs.stable_path_id) <
                     std::tie(rhs.total_time.value, rhs.stable_path_id);
            });
}

}  // namespace search_detail

template <class State, class Edge, class Adapter>
  requires PlatformSearchAdapter<Adapter, State, Edge>
[[nodiscard]] SearchResult<State, Edge> RunAraStar(
    const Adapter& adapter,
    const SearchProblem<State>& problem,
    const AraStarConfig& config) {
  using Node = search_detail::SearchNode<State, Edge>;
  using Path = SearchPath<State, Edge>;

  SearchResult<State, Edge> result;
  result.achieved_epsilon = config.initial_epsilon;

  std::map<StateKey, Node> nodes;
  std::set<search_detail::OpenEntry, search_detail::OpenEntryLess> open;
  std::map<StateKey, search_detail::OpenEntry> open_by_key;
  std::set<StateKey> closed;
  std::set<StateKey> incons;
  std::map<std::string, Path> distinct_candidates;
  std::optional<std::int64_t> best_terminal_time_ns;
  std::uint64_t next_insertion_sequence = 0U;
  double epsilon = config.initial_epsilon;

  const auto finish = [&](const SearchStatus status,
                          const std::string& reason_code) {
    result.status = status;
    result.reason_code = reason_code;
    result.achieved_epsilon = epsilon;
    result.candidates.clear();
    result.candidates.reserve(distinct_candidates.size());
    for (const auto& [path_id, path] : distinct_candidates) {
      static_cast<void>(path_id);
      result.candidates.push_back(path);
    }
    search_detail::SortCandidates(result.candidates);
    return result;
  };

  if (!std::isfinite(config.initial_epsilon) ||
      !std::isfinite(config.epsilon_decrement) ||
      !std::isfinite(config.target_epsilon) ||
      config.target_epsilon < 1.0 ||
      config.initial_epsilon < config.target_epsilon ||
      (config.initial_epsilon > config.target_epsilon &&
       config.epsilon_decrement <= 0.0)) {
    return finish(SearchStatus::kInvalidAdapterCost,
                  "search.invalid_configuration");
  }

  const auto make_open_entry = [&](const StateKey key) {
    const auto& node = nodes.at(key);
    return search_detail::OpenEntry{
        .priority_seconds =
            search_detail::ToPrioritySeconds(node.g_time_ns) +
            static_cast<long double>(epsilon) *
                search_detail::ToPrioritySeconds(node.heuristic_time_ns),
        .g_time_ns = node.g_time_ns,
        .state_key = key,
        .insertion_sequence = node.open_sequence,
    };
  };

  const auto put_open = [&](const StateKey key,
                            const bool preserve_sequence) {
    if (const auto existing = open_by_key.find(key);
        existing != open_by_key.end()) {
      open.erase(existing->second);
      open_by_key.erase(existing);
    }
    auto& node = nodes.at(key);
    if (!preserve_sequence || !node.has_open_sequence) {
      node.open_sequence = next_insertion_sequence++;
      node.has_open_sequence = true;
    }
    const auto entry = make_open_entry(key);
    open.insert(entry);
    open_by_key.emplace(key, entry);
  };

  const auto capture_candidate =
      [&](const StateKey terminal_key) -> std::optional<SearchResult<State, Edge>> {
    const auto reconstructed =
        search_detail::ReconstructPath<State, Edge>(nodes, terminal_key);
    if (!reconstructed.has_value()) {
      return finish(SearchStatus::kInvalidAdapterCost,
                    "search.invalid_predecessor_chain");
    }

    const auto existing =
        distinct_candidates.find(reconstructed->stable_path_id);
    if (existing == distinct_candidates.end()) {
      if (problem.limits.maximum_generated_candidates == 0U) {
        return finish(
            SearchStatus::kResourceLimit,
            "search.resource_limit.maximum_generated_candidates");
      }
      distinct_candidates.emplace(reconstructed->stable_path_id,
                                  *reconstructed);
      if (distinct_candidates.size() >=
          problem.limits.maximum_generated_candidates) {
        best_terminal_time_ns =
            best_terminal_time_ns.has_value()
                ? std::min(*best_terminal_time_ns,
                           reconstructed->total_time.value.count())
                : reconstructed->total_time.value.count();
        return finish(
            SearchStatus::kResourceLimit,
            "search.resource_limit.maximum_generated_candidates");
      }
    } else if (reconstructed->total_time.value <
               existing->second.total_time.value) {
      existing->second = *reconstructed;
    }

    best_terminal_time_ns =
        best_terminal_time_ns.has_value()
            ? std::min(*best_terminal_time_ns,
                       reconstructed->total_time.value.count())
            : reconstructed->total_time.value.count();
    return std::nullopt;
  };

  const StateKey start_key = adapter.Key(problem.start);
  const auto start_heuristic =
      adapter.AdmissibleTimeHeuristic(problem.start, problem);
  if (!search_detail::IsValidDuration(start_heuristic)) {
    return finish(SearchStatus::kInvalidAdapterCost,
                  "search.invalid_adapter_cost.invalid_time_heuristic");
  }
  nodes.emplace(
      start_key,
      Node{
          .state = problem.start,
          .g_time_ns = 0LL,
          .heuristic_time_ns = start_heuristic.value.count(),
          .has_g_time = true,
      });
  result.generated_states = 1U;

  if (adapter.IsTerminal(problem.start, problem)) {
    if (const auto terminal_result = capture_candidate(start_key);
        terminal_result.has_value()) {
      return *terminal_result;
    }
  }

  put_open(start_key, false);
  if (open.size() > problem.limits.maximum_open_states) {
    return finish(SearchStatus::kResourceLimit,
                  "search.resource_limit.maximum_open_states");
  }

  while (true) {
    while (true) {
      if (open.empty()) {
        break;
      }
      if (best_terminal_time_ns.has_value() &&
          search_detail::ToPrioritySeconds(*best_terminal_time_ns) <=
              open.begin()->priority_seconds) {
        break;
      }
      if (result.expansions >=
          problem.limits.maximum_expanded_states) {
        return finish(SearchStatus::kResourceLimit,
                      "search.resource_limit.maximum_expanded_states");
      }

      const auto current_entry = *open.begin();
      open.erase(open.begin());
      open_by_key.erase(current_entry.state_key);
      const StateKey current_key = current_entry.state_key;
      auto& current_node = nodes.at(current_key);
      current_node.has_open_sequence = false;
      closed.insert(current_key);
      ++result.expansions;

      auto transitions = adapter.Expand(current_node.state);
      std::sort(
          transitions.begin(),
          transitions.end(),
          [&adapter](const auto& lhs, const auto& rhs) {
            return std::tuple{adapter.Key(lhs.successor),
                              lhs.stable_edge_id} <
                   std::tuple{adapter.Key(rhs.successor),
                              rhs.stable_edge_id};
          });

      for (const auto& transition : transitions) {
        if (!adapter.HardFeasible(transition)) {
          continue;
        }

        const auto transition_time = adapter.TransitionTime(transition);
        if (!search_detail::IsValidDuration(transition_time)) {
          return finish(
              SearchStatus::kInvalidAdapterCost,
              "search.invalid_adapter_cost.negative_transition_time");
        }
        const auto transition_costs = adapter.SecondaryCosts(transition);
        if (!search_detail::IsValidSecondaryCosts(transition_costs)) {
          return finish(
              SearchStatus::kInvalidAdapterCost,
              "search.invalid_adapter_cost.invalid_secondary_cost");
        }

        const auto tentative_time = search_detail::CheckedTimeSum(
            current_node.g_time_ns, transition_time.value.count());
        if (!tentative_time.has_value()) {
          return finish(
              SearchStatus::kInvalidAdapterCost,
              "search.invalid_adapter_cost.transition_time_overflow");
        }
        const auto tentative_secondary =
            search_detail::CheckedSecondaryCostSum(
                current_node.secondary_costs, transition_costs);
        if (!tentative_secondary.has_value()) {
          return finish(
              SearchStatus::kInvalidAdapterCost,
              "search.invalid_adapter_cost.secondary_cost_overflow");
        }
        ++result.generated_states;

        const StateKey successor_key = adapter.Key(transition.successor);
        auto successor_node = nodes.find(successor_key);
        if (successor_node == nodes.end()) {
          const auto heuristic =
              adapter.AdmissibleTimeHeuristic(transition.successor, problem);
          if (!search_detail::IsValidDuration(heuristic)) {
            return finish(
                SearchStatus::kInvalidAdapterCost,
                "search.invalid_adapter_cost.invalid_time_heuristic");
          }
          successor_node =
              nodes
                  .emplace(
                      successor_key,
                      Node{
                          .state = transition.successor,
                          .heuristic_time_ns = heuristic.value.count(),
                      })
                  .first;
        }

        if (successor_node->second.has_g_time &&
            *tentative_time >= successor_node->second.g_time_ns) {
          continue;
        }

        successor_node->second.state = transition.successor;
        successor_node->second.g_time_ns = *tentative_time;
        successor_node->second.has_g_time = true;
        successor_node->second.secondary_costs = *tentative_secondary;
        successor_node->second.predecessor_key = current_key;
        successor_node->second.incoming_edge = transition.edge;
        successor_node->second.incoming_edge_id =
            transition.stable_edge_id;

        if (closed.contains(successor_key)) {
          incons.insert(successor_key);
        } else {
          put_open(successor_key, false);
          if (open.size() > problem.limits.maximum_open_states) {
            return finish(SearchStatus::kResourceLimit,
                          "search.resource_limit.maximum_open_states");
          }
        }

        if (adapter.IsTerminal(transition.successor, problem)) {
          if (const auto terminal_result =
                  capture_candidate(successor_key);
              terminal_result.has_value()) {
            return *terminal_result;
          }
        }
      }
    }

    if (epsilon <= config.target_epsilon) {
      return finish(distinct_candidates.empty() ? SearchStatus::kNoPath
                                                : SearchStatus::kSolved,
                    distinct_candidates.empty() ? "search.no_path"
                                                : "search.solved");
    }

    if (open.empty() && incons.empty()) {
      epsilon = config.target_epsilon;
      return finish(distinct_candidates.empty() ? SearchStatus::kNoPath
                                                : SearchStatus::kSolved,
                    distinct_candidates.empty() ? "search.no_path"
                                                : "search.solved");
    }

    epsilon =
        std::max(config.target_epsilon, epsilon - config.epsilon_decrement);

    std::set<StateKey> next_open_keys;
    for (const auto& [key, entry] : open_by_key) {
      static_cast<void>(entry);
      next_open_keys.insert(key);
    }
    next_open_keys.insert(incons.begin(), incons.end());

    open.clear();
    open_by_key.clear();
    closed.clear();
    incons.clear();
    for (const StateKey key : next_open_keys) {
      put_open(key, true);
      if (open.size() > problem.limits.maximum_open_states) {
        return finish(SearchStatus::kResourceLimit,
                      "search.resource_limit.maximum_open_states");
      }
    }
  }
}

}  // namespace lunar::planning::v3
