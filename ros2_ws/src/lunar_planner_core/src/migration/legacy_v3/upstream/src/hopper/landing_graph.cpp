#include "lunar_path_planner/v3/hopper/landing_graph.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <set>
#include <tuple>
#include <utility>

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] Error Failure(
    const ErrorCode code, std::string message) {
  return {
      .code = code,
      .field_path = "landing_graph",
      .message = std::move(message),
  };
}

[[nodiscard]] bool IsTerminalIndex(
    const std::size_t index,
    const std::span<const std::size_t> terminals) {
  return std::find(terminals.begin(), terminals.end(), index) !=
         terminals.end();
}

[[nodiscard]] bool BetterCertified(
    const FirstEdgeCertificationResult::CertifiedCandidate& lhs,
    const FirstEdgeCertificationResult::CertifiedCandidate& rhs,
    const DurationNanoseconds tolerance) {
  const auto lhs_time = lhs.expected_execution_time.value;
  const auto rhs_time = rhs.expected_execution_time.value;
  const auto tolerance_count = tolerance.value.count();
  const auto lhs_count = lhs_time.count();
  const auto rhs_count = rhs_time.count();
  if (lhs_count < rhs_count &&
      rhs_count - lhs_count > tolerance_count) {
    return true;
  }
  if (rhs_count < lhs_count &&
      lhs_count - rhs_count > tolerance_count) {
    return false;
  }
  return std::tuple{
             lhs.secondary_costs.energy,
             lhs.secondary_costs.risk,
             lhs.secondary_costs.smoothness,
             lhs.reference.reference_id} <
         std::tuple{
             rhs.secondary_costs.energy,
             rhs.secondary_costs.risk,
             rhs.secondary_costs.smoothness,
             rhs.reference.reference_id};
}

[[nodiscard]] std::chrono::nanoseconds SaturatingAdd(
    const std::chrono::nanoseconds lhs,
    const std::chrono::nanoseconds rhs) {
  const auto maximum =
      std::numeric_limits<std::int64_t>::max();
  if (lhs.count() < 0 || rhs.count() < 0 ||
      lhs.count() > maximum - rhs.count()) {
    return std::chrono::nanoseconds{maximum};
  }
  return lhs + rhs;
}

}  // namespace

LandingGraphAdapter::LandingGraphAdapter(
    const std::span<const LandingGraphNode> nodes,
    const std::span<const LandingGraphEdge> edges,
    const std::span<const std::size_t> terminal_nodes)
    : nodes_(nodes),
      edges_(edges),
      terminal_nodes_(terminal_nodes) {}

StateKey LandingGraphAdapter::Key(
    const LandingGraphSearchState& state) const {
  return static_cast<StateKey>(state.node_index);
}

std::vector<SearchTransition<
    LandingGraphSearchState, LandingGraphSearchEdge>>
LandingGraphAdapter::Expand(
    const LandingGraphSearchState& state) const {
  std::vector<SearchTransition<
      LandingGraphSearchState, LandingGraphSearchEdge>>
      transitions;
  if (state.node_index >= nodes_.size()) {
    return transitions;
  }
  for (std::size_t index = 0U; index < edges_.size(); ++index) {
    const LandingGraphEdge& edge = edges_[index];
    if (edge.from_node == state.node_index &&
        edge.to_node < nodes_.size() &&
        edge.state != LandingEdgeState::kRejected) {
      transitions.push_back(
          {
              .stable_edge_id = edge.edge_id,
              .successor =
                  LandingGraphSearchState{edge.to_node},
              .edge = LandingGraphSearchEdge{index},
          });
    }
  }
  std::sort(
      transitions.begin(), transitions.end(),
      [this](const auto& lhs, const auto& rhs) {
        const auto& left = edges_[lhs.edge.edge_index];
        const auto& right = edges_[rhs.edge.edge_index];
        return std::tie(
                   left.lower_bound_execution_time.value,
                   left.estimated_energy_lower_bound_j,
                   left.edge_id) <
               std::tie(
                   right.lower_bound_execution_time.value,
                   right.estimated_energy_lower_bound_j,
                   right.edge_id);
      });
  return transitions;
}

bool LandingGraphAdapter::HardFeasible(
    const SearchTransition<
        LandingGraphSearchState,
        LandingGraphSearchEdge>& transition) const {
  return transition.edge.edge_index < edges_.size() &&
         edges_[transition.edge.edge_index].state !=
             LandingEdgeState::kRejected;
}

DurationNanoseconds LandingGraphAdapter::TransitionTime(
    const SearchTransition<
        LandingGraphSearchState,
        LandingGraphSearchEdge>& transition) const {
  if (transition.edge.edge_index >= edges_.size()) {
    return DurationNanoseconds{
        std::chrono::nanoseconds{-1}};
  }
  return edges_[transition.edge.edge_index]
      .lower_bound_execution_time;
}

DurationNanoseconds
LandingGraphAdapter::AdmissibleTimeHeuristic(
    const LandingGraphSearchState& state,
    const SearchProblem<LandingGraphSearchState>& problem) const {
  static_cast<void>(state);
  static_cast<void>(problem);
  return DurationNanoseconds{std::chrono::nanoseconds{0}};
}

SecondaryCostVector LandingGraphAdapter::SecondaryCosts(
    const SearchTransition<
        LandingGraphSearchState,
        LandingGraphSearchEdge>& transition) const {
  if (transition.edge.edge_index >= edges_.size()) {
    return {
        .energy = std::numeric_limits<double>::infinity(),
    };
  }
  return {
      .energy =
          edges_[transition.edge.edge_index]
              .estimated_energy_lower_bound_j,
  };
}

bool LandingGraphAdapter::IsTerminal(
    const LandingGraphSearchState& state,
    const SearchProblem<LandingGraphSearchState>& problem) const {
  static_cast<void>(problem);
  return IsTerminalIndex(state.node_index, terminal_nodes_);
}

Result<LazyNextHopSelection>
LazyLandingGraphPlanner::select(
    std::vector<LandingGraphNode> nodes,
    std::vector<LandingGraphEdge> edges,
    const std::size_t start_node,
    const std::span<const std::size_t> terminal_nodes,
    NextHopCertificationOracle& oracle,
    const HopperPlannerLimits& limits) const {
  if (nodes.empty() || start_node >= nodes.size() ||
      terminal_nodes.empty() ||
      limits.maximum_full_certification_attempts == 0U) {
    return Failure(
        ErrorCode::kInvalidArgument,
        "invalid landing graph selection input");
  }
  for (const std::size_t terminal : terminal_nodes) {
    if (terminal >= nodes.size()) {
      return Failure(
          ErrorCode::kInvalidArgument,
          "terminal node is outside graph");
    }
  }

  struct Incumbent final {
    std::size_t edge_index{};
    FirstEdgeCertificationResult::CertifiedCandidate candidate;
    std::vector<LandingRegionId> future_region_ids;
    bool reaches_terminal{};
  };
  std::optional<Incumbent> incumbent;
  std::set<std::size_t> attempted_first_edges;
  std::size_t attempts = 0U;
  std::string last_rejection;

  while (attempts <
         limits.maximum_full_certification_attempts) {
    const LandingGraphAdapter adapter{
        nodes, edges, terminal_nodes};
    const SearchProblem<LandingGraphSearchState> problem{
        .start = LandingGraphSearchState{start_node},
        .limits = limits.ara_star.resource_caps,
    };
    const auto search =
        RunAraStar<
            LandingGraphSearchState,
            LandingGraphSearchEdge>(
            adapter, problem, limits.ara_star);
    if (search.candidates.empty()) {
      break;
    }
    const auto& path = search.candidates.front();
    if (path.edges.empty()) {
      break;
    }
    const std::size_t first_edge_index =
        path.edges.front().edge_index;
    if (first_edge_index >= edges.size()) {
      return Failure(
          ErrorCode::kNumericalFailure,
          "search returned invalid edge index");
    }
    if (attempted_first_edges.contains(first_edge_index)) {
      break;
    }
    attempted_first_edges.insert(first_edge_index);
    ++attempts;
    const LandingGraphEdge& first_edge =
        edges[first_edge_index];
    FirstEdgeCertificationResult certified =
        oracle.certify_first_edge(
            nodes[first_edge.from_node],
            nodes[first_edge.to_node]);
    if (!certified.certified_candidate.has_value()) {
      last_rejection = certified.rejection_reason;
      edges[first_edge_index].state =
          LandingEdgeState::kRejected;
      continue;
    }
    edges[first_edge_index].state =
        LandingEdgeState::kCertifiedNextHop;
    std::vector<LandingRegionId> future_ids;
    for (std::size_t state_index = 2U;
         state_index < path.states.size(); ++state_index) {
      const auto node_index =
          path.states[state_index].node_index;
      if (node_index < nodes.size() &&
          !nodes[node_index].region.region_id.empty()) {
        future_ids.push_back(
            nodes[node_index].region.region_id);
      }
    }
    Incumbent candidate{
        .edge_index = first_edge_index,
        .candidate =
            std::move(*certified.certified_candidate),
        .future_region_ids = std::move(future_ids),
        .reaches_terminal =
            IsTerminalIndex(first_edge.to_node, terminal_nodes),
    };
    if (!incumbent.has_value() ||
        BetterCertified(
            candidate.candidate, incumbent->candidate,
            limits.time_equivalence_tolerance)) {
      incumbent = std::move(candidate);
    }

    bool has_competitive_unattempted = false;
    const auto competitive_until = SaturatingAdd(
        incumbent->candidate.expected_execution_time.value,
        limits.time_equivalence_tolerance.value);
    for (std::size_t edge_index = 0U;
         edge_index < edges.size(); ++edge_index) {
      if (edges[edge_index].from_node == start_node &&
          edges[edge_index].state ==
              LandingEdgeState::kCheapPossible &&
          !attempted_first_edges.contains(edge_index) &&
          edges[edge_index].lower_bound_execution_time.value <=
              competitive_until) {
        has_competitive_unattempted = true;
        break;
      }
    }
    if (!has_competitive_unattempted) {
      break;
    }
    edges[first_edge_index].state =
        LandingEdgeState::kRejected;
  }

  if (!incumbent.has_value()) {
    return Failure(
        attempts >= limits.maximum_full_certification_attempts
            ? ErrorCode::kResourceLimit
            : ErrorCode::kNoKnownSafeRoute,
        last_rejection.empty()
            ? "no fully certified next hop"
            : "no fully certified next hop: " +
                  last_rejection);
  }
  std::size_t remaining = 0U;
  const auto competitive_until = SaturatingAdd(
      incumbent->candidate.expected_execution_time.value,
      limits.time_equivalence_tolerance.value);
  for (std::size_t edge_index = 0U;
       edge_index < edges.size(); ++edge_index) {
    if (edges[edge_index].from_node == start_node &&
        !attempted_first_edges.contains(edge_index) &&
        edges[edge_index].state != LandingEdgeState::kRejected &&
        edges[edge_index].lower_bound_execution_time.value <=
            competitive_until) {
      ++remaining;
    }
  }
  FutureRoutePreview preview{
      .future_viability =
          incumbent->reaches_terminal
              ? FutureViability::kViable
              : (incumbent->future_region_ids.empty()
                     ? FutureViability::kNoCertifiedContinuation
                     : FutureViability::kUnknown),
      .reason_code =
          !incumbent->reaches_terminal &&
                  incumbent->future_region_ids.empty()
              ? "SAFE_DEAD_END"
              : "FUTURE_ROUTE_NON_AUTHORITATIVE",
      .candidate_region_ids =
          incumbent->future_region_ids,
  };
  return LazyNextHopSelection{
      .edge_index = incumbent->edge_index,
      .selected_aim_point_id = "",
      .certified_next_hop =
          std::move(incumbent->candidate.reference),
      .future_preview = std::move(preview),
      .full_certification_attempt_count = attempts,
      .incumbent_status =
          remaining == 0U
              ? LazyNextHopSelection::IncumbentStatus::
                    kAllCompetitiveFirstEdgesRuledOut
              : LazyNextHopSelection::IncumbentStatus::
                    kResourceLimitedCompetitiveEdgesRemain,
      .remaining_competitive_first_edge_count = remaining,
  };
}

}  // namespace lunar::planning::v3
