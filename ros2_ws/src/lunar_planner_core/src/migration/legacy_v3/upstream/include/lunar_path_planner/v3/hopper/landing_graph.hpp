#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/hopper/aim_point_generator.hpp"
#include "lunar_path_planner/v3/search/ara_star.hpp"

namespace lunar::planning::v3 {

enum class LandingEdgeState : std::uint8_t {
  kCheapPossible,
  kCertifiedNextHop,
  kRejected,
};

struct LandingGraphNode final {
  std::string node_id;
  TerrainCertifiedLandingRegion region;
  std::vector<AimPointCandidate> aim_points;
};

struct LandingGraphEdge final {
  std::string edge_id;
  std::size_t from_node{};
  std::size_t to_node{};
  LandingEdgeState state{LandingEdgeState::kCheapPossible};
  DurationNanoseconds lower_bound_execution_time;
  double estimated_energy_lower_bound_j{};
};

struct LandingGraphSearchState final {
  std::size_t node_index{};
};

struct LandingGraphSearchEdge final {
  std::size_t edge_index{};
};

class LandingGraphAdapter final {
 public:
  LandingGraphAdapter(
      std::span<const LandingGraphNode> nodes,
      std::span<const LandingGraphEdge> edges,
      std::span<const std::size_t> terminal_nodes);

  [[nodiscard]] StateKey Key(
      const LandingGraphSearchState& state) const;
  [[nodiscard]] std::vector<SearchTransition<
      LandingGraphSearchState, LandingGraphSearchEdge>>
  Expand(const LandingGraphSearchState& state) const;
  [[nodiscard]] bool HardFeasible(
      const SearchTransition<
          LandingGraphSearchState,
          LandingGraphSearchEdge>& transition) const;
  [[nodiscard]] DurationNanoseconds TransitionTime(
      const SearchTransition<
          LandingGraphSearchState,
          LandingGraphSearchEdge>& transition) const;
  [[nodiscard]] DurationNanoseconds AdmissibleTimeHeuristic(
      const LandingGraphSearchState& state,
      const SearchProblem<LandingGraphSearchState>& problem) const;
  [[nodiscard]] SecondaryCostVector SecondaryCosts(
      const SearchTransition<
          LandingGraphSearchState,
          LandingGraphSearchEdge>& transition) const;
  [[nodiscard]] bool IsTerminal(
      const LandingGraphSearchState& state,
      const SearchProblem<LandingGraphSearchState>& problem) const;

 private:
  std::span<const LandingGraphNode> nodes_;
  std::span<const LandingGraphEdge> edges_;
  std::span<const std::size_t> terminal_nodes_;
};

struct LazyNextHopSelection final {
  std::size_t edge_index{};
  std::string selected_aim_point_id;
  HopperReference certified_next_hop;
  FutureRoutePreview future_preview;
  std::size_t full_certification_attempt_count{};
  enum class IncumbentStatus {
    kAllCompetitiveFirstEdgesRuledOut,
    kResourceLimitedCompetitiveEdgesRemain,
  } incumbent_status{
      IncumbentStatus::kAllCompetitiveFirstEdgesRuledOut};
  std::size_t remaining_competitive_first_edge_count{};
};

struct FirstEdgeCertificationResult final {
  struct CertifiedCandidate final {
    HopperReference reference;
    DurationNanoseconds expected_execution_time;
    SecondaryCostVector secondary_costs;
  };
  std::optional<CertifiedCandidate> certified_candidate;
  std::string rejection_reason;
};

class NextHopCertificationOracle {
 public:
  virtual ~NextHopCertificationOracle() = default;
  virtual FirstEdgeCertificationResult certify_first_edge(
      const LandingGraphNode& from,
      const LandingGraphNode& to) = 0;
};

class LazyLandingGraphPlanner final {
 public:
  [[nodiscard]] Result<LazyNextHopSelection> select(
      std::vector<LandingGraphNode> nodes,
      std::vector<LandingGraphEdge> edges,
      std::size_t start_node,
      std::span<const std::size_t> terminal_nodes,
      NextHopCertificationOracle& oracle,
      const HopperPlannerLimits& limits) const;
};

}  // namespace lunar::planning::v3
