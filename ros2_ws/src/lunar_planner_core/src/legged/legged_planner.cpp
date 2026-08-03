#include "legged/legged_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "legged/legged_lattice.hpp"
#include "legged/legged_spline_optimizer.hpp"
#include "legged/legged_terrain.hpp"
#include "legged/legged_timing.hpp"
#include "shared/ara_star.hpp"
#include "shared/convex_corridor.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"

namespace lunar::planning::legged {
namespace {

constexpr std::string_view kPlannerName = "cpp_v3_native_legged";

[[nodiscard]] PlannerOutput Failure(
    const PlanningOutcome outcome,
    const ExecutionDirective directive,
    std::string reason_code,
    const std::chrono::steady_clock::time_point started,
    const std::uint64_t expanded_states = 0U,
    std::optional<double> best_cost = std::nullopt,
    std::vector<std::string> warning_codes = {}) {
  return PlannerOutput{
      .outcome = outcome,
      .directive = directive,
      .reason_code = std::move(reason_code),
      .reference = std::nullopt,
      .diagnostics = PlannerDiagnostics{
          .planner_name = std::string{kPlannerName},
          .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started),
          .expanded_states = expanded_states,
          .best_cost = best_cost,
          .warning_codes = std::move(warning_codes),
      },
  };
}

[[nodiscard]] PlannerOutput Canceled(
    const std::chrono::steady_clock::time_point started,
    const std::uint64_t expanded_states = 0U) {
  return Failure(
      PlanningOutcome::kCanceled,
      ExecutionDirective::kHoldPosition,
      "REQUEST_CANCELED", started, expanded_states);
}

[[nodiscard]] bool HasFeasibleGoalPosition(
    const GoalRegion& goal,
    const shared::SafeProjection& projection,
    const LeggedCapability& capability,
    const std::stop_token stop_token) {
  if (projection.source_map() == nullptr) {
    return false;
  }
  GoalRegion position_goal = goal;
  position_goal.yaw_rad.reset();
  for (std::size_t y = 0U; y < projection.source_map()->height(); ++y) {
    for (std::size_t x = 0U; x < projection.source_map()->width(); ++x) {
      if (stop_token.stop_requested()) {
        return false;
      }
      const shared::GridCell cell{
          .x = static_cast<std::int32_t>(x),
          .y = static_cast<std::int32_t>(y),
      };
      const LeggedTerrainEvaluation terrain = EvaluateLeggedTerrainCell(
          projection, capability, cell, stop_token);
      if (!terrain.hard_feasible) {
        continue;
      }
      Vec3 position = projection.source_map()->CellCenter(cell);
      position.z = 0.5 *
          (terrain.body_height_m.lower + terrain.body_height_m.upper);
      if (GoalContainsBodyPose(
              position_goal,
              LeggedPose{.position_m = position, .yaw_rad = 0.0})) {
        return true;
      }
    }
  }
  return false;
}

[[nodiscard]] std::vector<Vec2> Centerline(
    const std::vector<LeggedTransition>& transitions) {
  std::vector<Vec2> result;
  if (transitions.empty()) {
    return result;
  }
  result.reserve(transitions.size() + 1U);
  result.push_back(Vec2{
      .x = transitions.front().source_pose.position_m.x,
      .y = transitions.front().source_pose.position_m.y,
  });
  for (const LeggedTransition& transition : transitions) {
    result.push_back(Vec2{
        .x = transition.target_pose.position_m.x,
        .y = transition.target_pose.position_m.y,
    });
  }
  return result;
}

[[nodiscard]] TrajectoryReference StationaryTrajectory(
    const LeggedState& state) {
  return TrajectoryReference{
      .semantics = TrajectorySemantics::kLeggedBodyReference,
      .points = {
          TrajectoryPoint{
              .time_from_start = std::chrono::nanoseconds{0},
              .pose = state.body_pose,
              .velocity = {},
          },
          TrajectoryPoint{
              .time_from_start = std::chrono::milliseconds{1},
              .pose = state.body_pose,
              .velocity = {},
          },
      },
  };
}

[[nodiscard]] bool ValidateTransitions(
    const std::vector<LeggedTransition>& transitions,
    const Interval& start_interval,
    const shared::SafeProjection& projection,
    const LeggedCapability& capability,
    const PlannerConfig& config,
    const std::stop_token stop_token) {
  Interval source_interval = start_interval;
  for (const LeggedTransition& transition : transitions) {
    if (transition.source_pose.position_m.z <
            source_interval.lower - 1.0e-9 ||
        transition.source_pose.position_m.z >
            source_interval.upper + 1.0e-9) {
      return false;
    }
    const LeggedSweepResult sweep = ValidateLeggedBodySweep(
        transition.source_pose, transition.target_pose, source_interval,
        transition.nominal_duration, projection, capability,
        std::min(
            config.legged.continuous_validation_maximum_subdivisions,
            config.legged.maximum_height_interval_splits),
        stop_token);
    if (!sweep.valid) {
      return false;
    }
    if (transition.target_pose.position_m.z <
            sweep.reachable_body_z_m.lower - 1.0e-9 ||
        transition.target_pose.position_m.z >
            sweep.reachable_body_z_m.upper + 1.0e-9) {
      return false;
    }
    source_interval = transition.target_body_z_m;
  }
  return true;
}

}  // namespace

PlannerOutput LeggedPlanner::Plan(const PlannerInput& input) const {
  const auto started = std::chrono::steady_clock::now();
  if (input.stop_token.stop_requested()) {
    return Canceled(started);
  }
  const auto* current_state = std::get_if<LeggedState>(&input.current_state);
  const auto* capability = std::get_if<LeggedCapability>(&input.capability);
  if (current_state == nullptr || capability == nullptr) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        "LEGGED_PLATFORM_TYPE_MISMATCH", started);
  }
  const shared::MapSnapshotBuildResult map =
      shared::MapSnapshot::Create(input.world.local_map);
  if (!map.ok()) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        map.reason_code, started);
  }
  const shared::SafeProjectionBuildResult projection =
      shared::BuildSafeProjection(
          map.snapshot, input.capability, input.config.map_safety,
          input.stop_token);
  if (!projection.ok()) {
    if (projection.reason_code == "REQUEST_CANCELED") {
      return Canceled(started);
    }
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        projection.reason_code, started);
  }
  if (!HasFeasibleGoalPosition(
          input.goal, *projection.projection, *capability,
          input.stop_token)) {
    if (input.stop_token.stop_requested()) {
      return Canceled(started);
    }
    return Failure(
        PlanningOutcome::kGoalInfeasible,
        ExecutionDirective::kHoldPosition,
        "LEGGED_GOAL_INFEASIBLE", started);
  }

  LeggedLatticeBuildResult lattice = BuildLeggedLattice(
      *current_state, input.goal, *projection.projection, *capability,
      input.config, input.stop_token);
  if (!lattice.ok()) {
    switch (lattice.status) {
      case LeggedLatticeStatus::kCanceled:
        return Canceled(started);
      case LeggedLatticeStatus::kResourceExhausted:
        return Failure(
            PlanningOutcome::kResourceExhausted,
            ExecutionDirective::kNoSafeReference,
            lattice.reason_code, started);
      case LeggedLatticeStatus::kInvalidRequest:
        if (lattice.reason_code == "LEGGED_START_NOT_SAFE") {
          return Failure(
              PlanningOutcome::kNoKnownSafeRoute,
              ExecutionDirective::kNoSafeReference,
              lattice.reason_code, started);
        }
        return Failure(
            PlanningOutcome::kInvalidRequest,
            ExecutionDirective::kNoSafeReference,
            lattice.reason_code, started);
      case LeggedLatticeStatus::kReady:
        break;
    }
  }
  if (!lattice.graph.has_value()) {
    return Failure(
        PlanningOutcome::kNumericalFailure,
        ExecutionDirective::kHoldPosition,
        "LEGGED_LATTICE_RESULT_INVALID", started);
  }
  if (std::ranges::none_of(
          lattice.graph->search_problem.goal_mask,
          [](const std::uint8_t value) { return value != 0U; })) {
    return Failure(
        PlanningOutcome::kNoKnownSafeRoute,
        ExecutionDirective::kNoSafeReference,
        "LEGGED_NO_KNOWN_SAFE_ROUTE", started);
  }
  const shared::AraStarResult search = shared::SearchAraStar(
      lattice.graph->search_problem, input.stop_token);
  switch (search.status) {
    case shared::AraStarStatus::kCanceled:
      return Canceled(started, search.expanded_states);
    case shared::AraStarStatus::kResourceExhausted:
      return Failure(
          PlanningOutcome::kResourceExhausted,
          ExecutionDirective::kNoSafeReference,
          search.reason_code, started, search.expanded_states);
    case shared::AraStarStatus::kNoPath:
      return Failure(
          PlanningOutcome::kNoKnownSafeRoute,
          ExecutionDirective::kNoSafeReference,
          "LEGGED_NO_KNOWN_SAFE_ROUTE", started, search.expanded_states);
    case shared::AraStarStatus::kInvalidProblem:
      return Failure(
          PlanningOutcome::kNumericalFailure,
          ExecutionDirective::kHoldPosition,
          search.reason_code, started, search.expanded_states);
    case shared::AraStarStatus::kSolved:
      break;
  }
  const auto discrete = ResolveLeggedPlan(*lattice.graph, search);
  if (!discrete.has_value()) {
    return Failure(
        PlanningOutcome::kNumericalFailure,
        ExecutionDirective::kHoldPosition,
        "LEGGED_SEARCH_RESULT_INVALID", started, search.expanded_states);
  }

  std::vector<std::string> warnings{"LEGGED_BODY_REFERENCE_ONLY"};
  TrajectoryReference trajectory;
  if (discrete->transitions.empty()) {
    trajectory = StationaryTrajectory(*current_state);
  } else {
    const double support_radius = std::hypot(
        capability->body_half_extent_m.x,
        capability->body_half_extent_m.y);
    const shared::CorridorResult corridor = shared::BuildConvexCorridor(
        *projection.projection, Centerline(discrete->transitions),
        shared::CorridorTightening{
            .footprint_support_radius_m = support_radius,
            .tracking_error_bound_m = 0.0,
            .additional_margin_m = 0.0,
        },
        input.config.corridor, input.stop_token);
    if (corridor.status == shared::CorridorStatus::kCanceled) {
      return Canceled(started, search.expanded_states);
    }
    if (corridor.status != shared::CorridorStatus::kCertified) {
      warnings.push_back(corridor.reason_code);
    }
    LeggedOptimizationResult optimized = OptimizeLeggedBodySpline(
        discrete->transitions, corridor, input.config.optimization,
        input.stop_token);
    if (optimized.canceled) {
      return Canceled(started, search.expanded_states);
    }
    if (!optimized.optimized &&
        optimized.reason_code != "LEGGED_OPTIMIZATION_NOT_NEEDED") {
      warnings.push_back(optimized.reason_code);
    }
    const auto start_cell = projection.projection->source_map()->PositionToCell(
        Vec2{
            .x = current_state->body_pose.position_m.x,
            .y = current_state->body_pose.position_m.y,
        });
    const LeggedTerrainEvaluation start_terrain =
        EvaluateLeggedTerrainCell(
            *projection.projection, *capability, *start_cell,
            input.stop_token);
    std::vector<LeggedTransition> selected = std::move(optimized.transitions);
    if (!ValidateTransitions(
            selected, start_terrain.body_height_m,
            *projection.projection, *capability, input.config,
            input.stop_token)) {
      if (input.stop_token.stop_requested()) {
        return Canceled(started, search.expanded_states);
      }
      selected = discrete->transitions;
      warnings.emplace_back("LEGGED_OPTIMIZATION_SWEEP_FALLBACK");
    }
    if (!ValidateTransitions(
            selected, start_terrain.body_height_m,
            *projection.projection, *capability, input.config,
            input.stop_token)) {
      return Failure(
          PlanningOutcome::kNumericalFailure,
          ExecutionDirective::kHoldPosition,
          "LEGGED_VALIDATED_PATH_LOST", started, search.expanded_states,
          discrete->cost, std::move(warnings));
    }
    LeggedTimingResult timed = ParameterizeLeggedBodyTiming(
        selected, *capability, input.stop_token);
    if (timed.canceled) {
      return Canceled(started, search.expanded_states);
    }
    if (!timed.ok()) {
      return Failure(
          PlanningOutcome::kNumericalFailure,
          ExecutionDirective::kHoldPosition,
          timed.reason_code, started, search.expanded_states,
          discrete->cost, std::move(warnings));
    }
    trajectory = std::move(*timed.trajectory);
  }

  return PlannerOutput{
      .outcome = PlanningOutcome::kNewReferenceAvailable,
      .directive = ExecutionDirective::kActivateNewReference,
      .reason_code = "LEGGED_BODY_PLAN_AVAILABLE",
      .reference = MotionReference{
          .plan_id = "legged/" + input.request_id,
          .platform_type = PlatformType::kLegged,
          .input_time = input.state_time,
          .data = std::move(trajectory),
      },
      .diagnostics = PlannerDiagnostics{
          .planner_name = std::string{kPlannerName},
          .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started),
          .expanded_states = search.expanded_states,
          .best_cost = discrete->cost,
          .warning_codes = std::move(warnings),
      },
  };
}

}  // namespace lunar::planning::legged
