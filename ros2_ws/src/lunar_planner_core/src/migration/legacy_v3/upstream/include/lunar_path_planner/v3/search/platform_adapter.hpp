#pragma once

#include <concepts>
#include <cstdint>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/contracts/base_types.hpp"
#include "lunar_path_planner/v3/contracts/profiles.hpp"

namespace lunar::planning::v3 {

using StateKey = std::uint64_t;

struct SecondaryCostVector final {
  double energy{};
  double risk{};
  double smoothness{};

  SecondaryCostVector& operator+=(const SecondaryCostVector& rhs) {
    energy += rhs.energy;
    risk += rhs.risk;
    smoothness += rhs.smoothness;
    return *this;
  }
};

template <class State, class Edge>
struct SearchTransition final {
  std::string stable_edge_id;
  State successor;
  Edge edge;
};

template <class State>
struct SearchProblem final {
  State start;
  ResourceCaps limits;
};

template <class Adapter, class State, class Edge>
concept PlatformSearchAdapter =
    requires(const Adapter& adapter,
             const State& state,
             const SearchProblem<State>& problem,
             const SearchTransition<State, Edge>& transition) {
      { adapter.Key(state) } -> std::same_as<StateKey>;
      { adapter.Expand(state) }
          -> std::same_as<std::vector<SearchTransition<State, Edge>>>;
      { adapter.HardFeasible(transition) } -> std::same_as<bool>;
      { adapter.TransitionTime(transition) }
          -> std::same_as<DurationNanoseconds>;
      { adapter.AdmissibleTimeHeuristic(state, problem) }
          -> std::same_as<DurationNanoseconds>;
      { adapter.SecondaryCosts(transition) }
          -> std::same_as<SecondaryCostVector>;
      { adapter.IsTerminal(state, problem) } -> std::same_as<bool>;
    };

}  // namespace lunar::planning::v3
