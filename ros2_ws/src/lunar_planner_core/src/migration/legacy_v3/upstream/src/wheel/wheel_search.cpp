#include "lunar_path_planner/v3/wheel/wheel_search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace lunar::planning::v3 {
namespace {

constexpr double kStoppedTolerance = 1.0e-9;

[[nodiscard]] Error Failure(ErrorCode code,
                            std::string field_path,
                            std::string message) {
  return {
      .code = code,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool FiniteState(
    const WheeledOrLeggedState& state) noexcept {
  return std::isfinite(state.position_m.x) &&
         std::isfinite(state.position_m.y) &&
         std::isfinite(state.position_m.z) &&
         std::isfinite(state.yaw_rad) &&
         std::isfinite(state.linear_velocity_mps.x) &&
         std::isfinite(state.linear_velocity_mps.y) &&
         std::isfinite(state.linear_velocity_mps.z) &&
         std::isfinite(state.yaw_rate_radps);
}

[[nodiscard]] WheelMotionMode InitialMode(
    const WheeledOrLeggedState& state) noexcept {
  const double cosine = std::cos(state.yaw_rad);
  const double sine = std::sin(state.yaw_rad);
  const double body_forward_speed =
      cosine * state.linear_velocity_mps.x +
      sine * state.linear_velocity_mps.y;
  const double linear_speed =
      std::hypot(state.linear_velocity_mps.x,
                 state.linear_velocity_mps.y);
  if (linear_speed <= kStoppedTolerance &&
      std::abs(state.yaw_rate_radps) <= kStoppedTolerance) {
    return WheelMotionMode::kStart;
  }
  if (linear_speed <= kStoppedTolerance) {
    return state.yaw_rate_radps < 0.0
               ? WheelMotionMode::kSpinClockwise
               : WheelMotionMode::kSpinCounterClockwise;
  }
  return body_forward_speed < 0.0 ? WheelMotionMode::kReverse
                                  : WheelMotionMode::kForward;
}

[[nodiscard]] bool ActiveMode(WheelMotionMode mode) noexcept {
  return mode != WheelMotionMode::kStart;
}

[[nodiscard]] bool CheckedAdd(
    DurationNanoseconds value,
    DurationNanoseconds& accumulator) noexcept {
  const auto lhs = accumulator.value.count();
  const auto rhs = value.value.count();
  if (rhs < 0 ||
      lhs > std::numeric_limits<std::int64_t>::max() - rhs) {
    return false;
  }
  accumulator.value += value.value;
  return true;
}

[[nodiscard]] Result<std::vector<WheelDiscreteSegment>>
SegmentPath(std::span<const WheelLatticeEdge> edges) {
  std::vector<WheelDiscreteSegment> segments;
  WheelDiscreteSegment* active_segment = nullptr;
  for (const WheelLatticeEdge& edge : edges) {
    if (edge.primitive_kind == WheelPrimitiveKind::kModeSwitch) {
      if (edge.source.motion_mode == WheelMotionMode::kStart &&
          ActiveMode(edge.target.motion_mode)) {
        segments.push_back(
            {.mode = edge.target.motion_mode});
        active_segment = &segments.back();
      } else if (ActiveMode(edge.source.motion_mode) &&
                 edge.target.motion_mode ==
                     WheelMotionMode::kStart) {
        if (active_segment == nullptr ||
            active_segment->mode != edge.source.motion_mode) {
          return Failure(
              ErrorCode::kInvalidArgument,
              "/discrete_plan/edges",
              "INVALID_WHEEL_MODE_SWITCH_SEQUENCE");
        }
      } else {
        return Failure(ErrorCode::kInvalidArgument,
                       "/discrete_plan/edges",
                       "INVALID_WHEEL_MODE_SWITCH_SEQUENCE");
      }
    } else if (!ActiveMode(edge.target.motion_mode) ||
               edge.source.motion_mode != edge.target.motion_mode) {
      return Failure(ErrorCode::kInvalidArgument,
                     "/discrete_plan/edges",
                     "INVALID_WHEEL_ACTIVE_EDGE");
    } else if (active_segment == nullptr ||
               active_segment->mode != edge.target.motion_mode) {
      segments.push_back({.mode = edge.target.motion_mode});
      active_segment = &segments.back();
    }

    if (active_segment == nullptr ||
        !CheckedAdd(edge.transition_time,
                    active_segment->expected_time)) {
      return Failure(ErrorCode::kInvalidArgument,
                     "/discrete_plan/expected_time",
                     "WHEEL_SEGMENT_TIME_OVERFLOW");
    }
    active_segment->edges.push_back(edge);
    active_segment->secondary_costs += edge.secondary_costs;
  }
  return segments;
}

}  // namespace

Result<WheelDiscretePlan> PlanWheelDiscrete(
    const WheelPlanningProblem& problem,
    const WheelPrimitiveCatalog& primitives,
    const AraStarConfig& search_config) {
  if (!FiniteState(problem.current_state)) {
    return Failure(ErrorCode::kInvalidArgument,
                   "/current_state",
                   "INVALID_WHEEL_STATE");
  }
  if (problem.terminals.kind != TerminalKind::kGoal &&
      problem.terminals.kind != TerminalKind::kSafeFrontier) {
    return Failure(ErrorCode::kNoKnownSafeRoute,
                   "/resolved_terminal_set",
                   "NO_WHEEL_TERMINAL_SET");
  }
  if (problem.terminals.candidates.empty()) {
    return Failure(ErrorCode::kNoKnownSafeRoute,
                   "/resolved_terminal_set/candidates",
                   "NO_WHEEL_TERMINAL_CANDIDATE");
  }
  if (problem.algorithm.wheeled.state_lattice.xy_resolution_m <=
          0.0 ||
      problem.algorithm.wheeled.state_lattice.yaw_bin_count == 0U ||
      problem.terminals.candidates.size() >
          problem.algorithm.wheeled.state_lattice
              .maximum_terminal_candidates) {
    return Failure(ErrorCode::kInvalidArgument,
                   "/algorithm_config/wheeled/state_lattice",
                   "INVALID_WHEEL_LATTICE_CONFIG");
  }
  if (primitives.content_ref() !=
      problem.capability.content_ref) {
    return Failure(ErrorCode::kInvalidArgument,
                   "/platform_capability",
                   "WHEEL_PRIMITIVE_CAPABILITY_MISMATCH");
  }

  const auto capability_result =
      WheelCapabilityView::Create(problem.capability);
  if (!IsOk(capability_result)) {
    return std::get<Error>(capability_result);
  }
  WheelLatticeAdapter adapter{
      primitives,
      problem.projection,
      std::get<WheelCapabilityView>(capability_result),
      problem.terminals.candidates,
      problem.algorithm.wheeled.state_lattice,
  };
  const WheelLatticeState start = adapter.Quantize(
      {.position_m = problem.current_state.position_m,
       .yaw_rad = problem.current_state.yaw_rad},
      InitialMode(problem.current_state));
  if (!problem.projection.HardFeasible({start.ix, start.iy})) {
    return Failure(ErrorCode::kNoKnownSafeRoute,
                   "/current_state/position_m",
                   "WHEEL_START_NOT_HARD_FEASIBLE");
  }

  const SearchProblem<WheelLatticeState> search_problem{
      .start = start,
      .limits = search_config.resource_caps,
  };
  const auto search_result =
      RunAraStar<WheelLatticeState, WheelLatticeEdge>(
          adapter, search_problem, search_config);
  if (search_result.candidates.empty()) {
    return Failure(
        search_result.status == SearchStatus::kResourceLimit
            ? ErrorCode::kResourceLimit
            : ErrorCode::kNoKnownSafeRoute,
        "/wheel_search",
        search_result.status == SearchStatus::kResourceLimit
            ? "WHEEL_SEARCH_RESOURCE_LIMIT_WITHOUT_INCUMBENT"
            : "NO_KNOWN_SAFE_WHEEL_ROUTE");
  }

  std::vector<CandidateScore> scores;
  scores.reserve(search_result.candidates.size());
  for (const auto& candidate : search_result.candidates) {
    scores.push_back(
        {
            .stable_candidate_id = candidate.stable_path_id,
            .total_time = candidate.total_time,
            .energy = candidate.secondary_costs.energy,
            .risk = candidate.secondary_costs.risk,
            .smoothness = candidate.secondary_costs.smoothness,
            .fully_hard_validated =
                candidate.fully_hard_validated,
        });
  }
  const auto pool_result =
      CandidateRanker::BuildTimeEquivalentPool(
          scores, problem.algorithm.time_equivalence_tolerance);
  if (!IsOk(pool_result)) {
    return std::get<Error>(pool_result);
  }
  const auto& pool = std::get<TimeEquivalentPool>(pool_result);
  const std::string& selected_id =
      pool.ordered_candidates.front().stable_candidate_id;
  const auto selected = std::ranges::find_if(
      search_result.candidates,
      [&selected_id](const auto& candidate) {
        return candidate.stable_path_id == selected_id;
      });
  if (selected == search_result.candidates.end() ||
      selected->states.empty() ||
      selected->states.back().motion_mode !=
          WheelMotionMode::kStart) {
    return Failure(ErrorCode::kNoKnownSafeRoute,
                   "/wheel_search/candidates",
                   "WHEEL_TERMINAL_NOT_STOPPED");
  }

  auto segments_result = SegmentPath(selected->edges);
  if (!IsOk(segments_result)) {
    return std::get<Error>(segments_result);
  }
  WheeledOrLeggedState stop_state{
      .position_m = adapter.Pose(selected->states.back()).position_m,
      .yaw_rad = adapter.Pose(selected->states.back()).yaw_rad,
      .linear_velocity_mps = {},
      .yaw_rate_radps = 0.0,
      .error_bounds =
          std::get<WheeledCapability>(problem.capability.content)
              .certified_state_error_bounds,
  };
  const auto anchor_result = ResolveSafeStopAnchor(
      problem.projection, stop_state, problem.capability);
  if (!IsOk(anchor_result)) {
    return Failure(ErrorCode::kNoKnownSafeRoute,
                   "/safe_stop_anchor",
                   "NO_CERTIFIED_WHEEL_SAFE_STOP_ANCHOR");
  }

  return WheelDiscretePlan{
      .selected_candidate_id = selected_id,
      .segments =
          std::get<std::vector<WheelDiscreteSegment>>(
              std::move(segments_result)),
      .expected_time = selected->total_time,
      .total_secondary_costs = selected->secondary_costs,
      .safe_stop_anchor = std::get<SafeStopAnchor>(anchor_result),
      .diagnostics =
          {
              .status = search_result.status,
              .final_epsilon = search_result.achieved_epsilon,
              .expanded_states = search_result.expansions,
              .generated_states = search_result.generated_states,
              .candidate_count = search_result.candidates.size(),
              .resource_limit_hit =
                  search_result.status ==
                  SearchStatus::kResourceLimit,
              .termination_reason = search_result.reason_code,
          },
  };
}

}  // namespace lunar::planning::v3
