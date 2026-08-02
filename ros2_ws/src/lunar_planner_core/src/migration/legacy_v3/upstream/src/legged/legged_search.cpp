#include "lunar_path_planner/v3/legged/legged_search.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace lunar::planning::v3 {
namespace {

using HeightTuple =
    std::tuple<std::int32_t, std::int32_t, std::int32_t,
               std::uint64_t, std::uint64_t>;

[[nodiscard]] Error ErrorResult(ErrorCode code, std::string path,
                                std::string message) {
  return Error{
      .code = code,
      .field_path = std::move(path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool HasNegativeZero(double value) noexcept {
  return value == 0.0 && std::signbit(value);
}

[[nodiscard]] std::optional<HeightTuple> MakeHeightTuple(
    const LeggedLatticeState& state) noexcept {
  if (!IsValidHeightInterval(state.reachable_z) ||
      HasNegativeZero(state.reachable_z.min_m) ||
      HasNegativeZero(state.reachable_z.max_m)) {
    return std::nullopt;
  }
  return HeightTuple{
      state.ix,
      state.iy,
      state.iyaw,
      std::bit_cast<std::uint64_t>(state.reachable_z.min_m),
      std::bit_cast<std::uint64_t>(state.reachable_z.max_m),
  };
}

class HeightLabelInterner final {
 public:
  explicit HeightLabelInterner(std::size_t capacity) noexcept
      : capacity_(capacity) {}

  [[nodiscard]] std::optional<StateKey> Intern(
      const LeggedLatticeState& state) {
    const auto tuple = MakeHeightTuple(state);
    if (!tuple.has_value()) {
      invalid_interval_ = true;
      return std::nullopt;
    }
    if (const auto existing = labels_.find(*tuple);
        existing != labels_.end()) {
      return existing->second;
    }
    if (labels_.size() >= capacity_ ||
        next_id_ == std::numeric_limits<StateKey>::max()) {
      resource_limit_hit_ = true;
      return std::nullopt;
    }
    const StateKey id = next_id_++;
    labels_.emplace(*tuple, id);
    return id;
  }

  [[nodiscard]] bool resource_limit_hit() const noexcept {
    return resource_limit_hit_;
  }

  [[nodiscard]] bool invalid_interval() const noexcept {
    return invalid_interval_;
  }

 private:
  std::size_t capacity_{};
  StateKey next_id_{1U};
  std::map<HeightTuple, StateKey> labels_;
  bool resource_limit_hit_{};
  bool invalid_interval_{};
};

struct SearchHeightLabel final {
  LeggedLatticeState lattice_state;
  StateKey unique_label_key{};
};

class MultiLabelSearchAdapter final {
 public:
  MultiLabelSearchAdapter(const LeggedLatticeAdapter& base,
                          std::size_t maximum_labels,
                          const LeggedLatticeState& start)
      : base_(&base), interner_(maximum_labels) {
    const auto key = interner_.Intern(start);
    if (key.has_value()) {
      start_ = SearchHeightLabel{
          .lattice_state = start,
          .unique_label_key = *key,
      };
    }
  }

  [[nodiscard]] const std::optional<SearchHeightLabel>& start()
      const noexcept {
    return start_;
  }

  [[nodiscard]] StateKey Key(
      const SearchHeightLabel& state) const noexcept {
    return state.unique_label_key;
  }

  [[nodiscard]] std::vector<
      SearchTransition<SearchHeightLabel, LeggedLatticeEdge>>
  Expand(const SearchHeightLabel& state) const {
    std::vector<
        SearchTransition<SearchHeightLabel, LeggedLatticeEdge>>
        result;
    const auto base_transitions =
        base_->Expand(state.lattice_state);
    result.reserve(base_transitions.size());
    for (const auto& transition : base_transitions) {
      const auto key = interner_.Intern(transition.successor);
      if (!key.has_value()) {
        continue;
      }
      result.push_back(
          SearchTransition<SearchHeightLabel, LeggedLatticeEdge>{
              .stable_edge_id = transition.stable_edge_id,
              .successor =
                  SearchHeightLabel{
                      .lattice_state = transition.successor,
                      .unique_label_key = *key,
                  },
              .edge = transition.edge,
          });
    }
    return result;
  }

  [[nodiscard]] bool HardFeasible(
      const SearchTransition<SearchHeightLabel,
                             LeggedLatticeEdge>& transition) const {
    return base_->HardFeasible(
        SearchTransition<LeggedLatticeState, LeggedLatticeEdge>{
            .stable_edge_id = transition.stable_edge_id,
            .successor = transition.successor.lattice_state,
            .edge = transition.edge,
        });
  }

  [[nodiscard]] DurationNanoseconds TransitionTime(
      const SearchTransition<SearchHeightLabel,
                             LeggedLatticeEdge>& transition) const noexcept {
    return transition.edge.transition_time;
  }

  [[nodiscard]] DurationNanoseconds AdmissibleTimeHeuristic(
      const SearchHeightLabel& state,
      const SearchProblem<SearchHeightLabel>& problem) const noexcept {
    return base_->AdmissibleTimeHeuristic(
        state.lattice_state,
        SearchProblem<LeggedLatticeState>{
            .start = state.lattice_state,
            .limits = problem.limits,
        });
  }

  [[nodiscard]] SecondaryCostVector SecondaryCosts(
      const SearchTransition<SearchHeightLabel,
                             LeggedLatticeEdge>& transition) const noexcept {
    return transition.edge.secondary_costs;
  }

  [[nodiscard]] bool IsTerminal(
      const SearchHeightLabel& state,
      const SearchProblem<SearchHeightLabel>& problem) const noexcept {
    return base_->IsTerminal(
        state.lattice_state,
        SearchProblem<LeggedLatticeState>{
            .start = state.lattice_state,
            .limits = problem.limits,
        });
  }

  [[nodiscard]] bool resource_limit_hit() const noexcept {
    return interner_.resource_limit_hit();
  }

  [[nodiscard]] bool invalid_interval() const noexcept {
    return interner_.invalid_interval();
  }

 private:
  const LeggedLatticeAdapter* base_;
  mutable HeightLabelInterner interner_;
  std::optional<SearchHeightLabel> start_;
};

static_assert(
    PlatformSearchAdapter<MultiLabelSearchAdapter, SearchHeightLabel,
                          LeggedLatticeEdge>);

[[nodiscard]] std::optional<std::size_t> MaximumLabelCount(
    const ResourceCaps& limits) noexcept {
  if (limits.maximum_open_states == 0U ||
      limits.maximum_expanded_states == 0U) {
    return std::nullopt;
  }
  if (limits.maximum_open_states >
      std::numeric_limits<std::size_t>::max() -
          limits.maximum_expanded_states) {
    return std::nullopt;
  }
  return limits.maximum_open_states +
         limits.maximum_expanded_states;
}

}  // namespace

Result<LeggedDiscretePlan> PlanLeggedDiscrete(
    const LeggedPlanningProblem& problem,
    const BodyMotionPrimitiveCatalog& catalog,
    const AraStarConfig& config) {
  if (problem.projection.platform_type() != PlatformType::kLegged ||
      !std::isfinite(problem.grid.xy_resolution_m) ||
      problem.grid.xy_resolution_m <= 0.0 ||
      problem.grid.yaw_bin_count == 0U ||
      problem.terminals.empty() ||
      !problem.projection.InBounds(
          {problem.start.ix, problem.start.iy}) ||
      !problem.projection.HardFeasible(
          {problem.start.ix, problem.start.iy}) ||
      !IsValidHeightInterval(problem.start.reachable_z) ||
      HasNegativeZero(problem.start.reachable_z.min_m) ||
      HasNegativeZero(problem.start.reachable_z.max_m) ||
      problem.time_equivalence_tolerance.value.count() < 0) {
    return ErrorResult(ErrorCode::kInvalidArgument,
                       "legged_planning_problem",
                       "legged discrete problem is invalid");
  }
  const double resolution_difference =
      std::abs(problem.grid.xy_resolution_m -
               problem.projection.geometry.resolution_m);
  if (resolution_difference >
      32.0 * std::numeric_limits<double>::epsilon() *
          std::max(problem.grid.xy_resolution_m,
                   problem.projection.geometry.resolution_m)) {
    return ErrorResult(ErrorCode::kInvalidArgument,
                       "legged_planning_problem.grid",
                       "lattice and projection resolutions must match");
  }
  const auto maximum_labels =
      MaximumLabelCount(config.resource_caps);
  if (!maximum_labels.has_value()) {
    return ErrorResult(ErrorCode::kInvalidArgument,
                       "ara_star.resource_caps",
                       "multi-label search requires positive bounded pools");
  }

  const LeggedLatticeAdapter base_adapter(
      problem.projection, problem.terminals, problem.grid, catalog);
  MultiLabelSearchAdapter adapter(
      base_adapter, *maximum_labels, problem.start);
  if (!adapter.start().has_value()) {
    return ErrorResult(
        adapter.resource_limit_hit() ? ErrorCode::kResourceLimit
                                     : ErrorCode::kInvalidArgument,
        "legged_planning_problem.start.reachable_z",
        "start height interval could not be interned");
  }
  const SearchProblem<SearchHeightLabel> search_problem{
      .start = *adapter.start(),
      .limits = config.resource_caps,
  };
  const auto search =
      RunAraStar<SearchHeightLabel, LeggedLatticeEdge>(
          adapter, search_problem, config);

  if (adapter.invalid_interval()) {
    return ErrorResult(
        ErrorCode::kInvalidArgument,
        "legged_search.height_label",
        "search produced a non-finite or non-canonical height label");
  }
  if (search.candidates.empty()) {
    if (adapter.resource_limit_hit() ||
        search.status == SearchStatus::kResourceLimit) {
      return ErrorResult(
          ErrorCode::kResourceLimit, "legged_search",
          "multi-label search exhausted a configured resource bound");
    }
    return ErrorResult(ErrorCode::kNoKnownSafeRoute, "legged_search",
                       "no hard-feasible terminal label was found");
  }

  std::vector<HeightLabelRecord> terminal_labels;
  terminal_labels.reserve(search.candidates.size());
  for (const auto& candidate : search.candidates) {
    const auto& terminal = candidate.states.back().lattice_state;
    terminal_labels.push_back(
        HeightLabelRecord{
            .stable_label_id = candidate.stable_path_id,
            .pose_key = base_adapter.Key(terminal),
            .arrival_time = candidate.total_time,
            .reachable_z = terminal.reachable_z,
        });
  }
  const auto non_dominated =
      FilterNonDominatedHeightLabels(terminal_labels);
  std::set<std::string> retained_ids;
  for (const auto& label : non_dominated) {
    retained_ids.insert(label.stable_label_id);
  }

  std::vector<CandidateScore> scores;
  for (const auto& candidate : search.candidates) {
    if (!retained_ids.contains(candidate.stable_path_id)) {
      continue;
    }
    scores.push_back(
        CandidateScore{
            .stable_candidate_id = candidate.stable_path_id,
            .total_time = candidate.total_time,
            .energy = candidate.secondary_costs.energy,
            .risk = candidate.secondary_costs.risk,
            .smoothness = candidate.secondary_costs.smoothness,
            .fully_hard_validated =
                candidate.fully_hard_validated,
        });
  }
  const auto pool = CandidateRanker::BuildTimeEquivalentPool(
      scores, problem.time_equivalence_tolerance);
  if (!IsOk(pool)) {
    return std::get<Error>(pool);
  }
  const auto& selected_score =
      std::get<TimeEquivalentPool>(pool).ordered_candidates.front();
  const auto selected = std::ranges::find(
      search.candidates, selected_score.stable_candidate_id,
      &SearchPath<SearchHeightLabel,
                  LeggedLatticeEdge>::stable_path_id);
  if (selected == search.candidates.end()) {
    return ErrorResult(ErrorCode::kNumericalFailure, "legged_search",
                       "ranked candidate was not recoverable");
  }

  LeggedDiscretePlan plan;
  plan.stable_candidate_id = selected->stable_path_id;
  plan.grid = problem.grid;
  plan.grid_origin_m = problem.projection.geometry.origin_m;
  plan.edges = selected->edges;
  plan.estimated_execution_time = selected->total_time;
  plan.secondary_costs = selected->secondary_costs;
  plan.achieved_epsilon = search.achieved_epsilon;
  plan.resource_limit_hit =
      adapter.resource_limit_hit() ||
      search.status == SearchStatus::kResourceLimit;
  plan.states.reserve(selected->states.size());
  for (const auto& state : selected->states) {
    plan.states.push_back(state.lattice_state);
  }
  const auto& terminal = plan.states.back();
  plan.terminal_pose =
      PoseXyzYaw{
          .position_m =
              {
                  problem.projection.geometry.origin_m.x +
                      (static_cast<double>(terminal.ix) + 0.5) *
                          problem.grid.xy_resolution_m,
                  problem.projection.geometry.origin_m.y +
                      (static_cast<double>(terminal.iy) + 0.5) *
                          problem.grid.xy_resolution_m,
                  std::clamp(
                      catalog.capability().preferred_body_height_m,
                      terminal.reachable_z.min_m,
                      terminal.reachable_z.max_m),
              },
          .yaw_rad =
              static_cast<double>(terminal.iyaw) *
              (2.0 * std::numbers::pi /
               static_cast<double>(problem.grid.yaw_bin_count)),
      };
  plan.target_linear_velocity_mps = 0.0;
  plan.target_yaw_rate_radps = 0.0;
  return plan;
}

}  // namespace lunar::planning::v3
