#include "legged/legged_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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

[[nodiscard]] double PlanarGoalError(const GoalRegion& goal,
                                     const Pose3& pose) noexcept {
  if (const auto* point = std::get_if<PointGoal>(&goal.target)) {
    return std::hypot(pose.position_m.x - point->position_m.x,
                      pose.position_m.y - point->position_m.y);
  }
  const auto* region = std::get_if<PlanarRegionGoal>(&goal.target);
  if (region == nullptr || region->boundary_m.empty()) {
    return 0.0;
  }
  GoalRegion position_goal = goal;
  position_goal.yaw_rad.reset();
  if (GoalContainsBodyPose(position_goal,
                           LeggedPose{.position_m = pose.position_m})) {
    return 0.0;
  }
  double error = std::numeric_limits<double>::infinity();
  for (const Vec3& vertex : region->boundary_m) {
    error = std::min(error, std::hypot(pose.position_m.x - vertex.x,
                                      pose.position_m.y - vertex.y));
  }
  return error;
}

[[nodiscard]] double PositionError(const Pose3& lhs,
                                   const Pose3& rhs) noexcept {
  return std::hypot(
      std::hypot(lhs.position_m.x - rhs.position_m.x,
                 lhs.position_m.y - rhs.position_m.y),
      lhs.position_m.z - rhs.position_m.z);
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
    source_interval = sweep.reachable_body_z_m;
  }
  return true;
}

}  // namespace

PlannerOutput LeggedPlanner::Plan(
    const hierarchical::LocalPlanningProblem& problem) const {
  const auto started = std::chrono::steady_clock::now();
  if (problem.stop_token.stop_requested()) {
    return Canceled(started);
  }
  const auto* current_state = std::get_if<LeggedState>(&problem.current_state);
  const auto* capability = std::get_if<LeggedCapability>(&problem.capability);
  if (current_state == nullptr || capability == nullptr) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        "LEGGED_PLATFORM_TYPE_MISMATCH", started);
  }
  const shared::MapSnapshotBuildResult map =
      shared::MapSnapshot::Create(problem.local_map_view);
  if (!map.ok()) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        map.reason_code, started);
  }
  const shared::SafeProjectionBuildResult projection =
      shared::BuildSafeProjection(
          map.snapshot, problem.capability, problem.config.map_safety,
          problem.stop_token);
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
          problem.goal_odom, *projection.projection, *capability,
          problem.stop_token)) {
    if (problem.stop_token.stop_requested()) {
      return Canceled(started);
    }
    return Failure(
        PlanningOutcome::kGoalInfeasible,
        ExecutionDirective::kHoldPosition,
        "LEGGED_GOAL_INFEASIBLE", started);
  }

  LeggedLatticeBuildResult lattice = BuildLeggedLattice(
      *current_state, problem.goal_odom, *projection.projection, *capability,
      problem.config, problem.stop_token);
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
        if (lattice.reason_code == "LEGGED_START_NOT_SAFE" ||
            lattice.reason_code == "LEGGED_START_CONNECTOR_INFEASIBLE") {
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
      lattice.graph->search_problem, problem.stop_token);
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
  TrajectoryMode trajectory_mode = TrajectoryMode::kStationary;
  CollisionValidation collision_validation =
      CollisionValidation::kNotApplicable;
  std::chrono::nanoseconds smoothing_elapsed{};
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
        problem.config.corridor, problem.stop_token);
    if (corridor.status == shared::CorridorStatus::kCanceled) {
      return Canceled(started, search.expanded_states);
    }
    if (corridor.status != shared::CorridorStatus::kCertified) {
      warnings.push_back(corridor.reason_code);
    }
    const auto smoothing_started = std::chrono::steady_clock::now();
    LeggedOptimizationResult optimized = OptimizeLeggedBodySpline(
        discrete->transitions, corridor, problem.config.optimization,
        problem.stop_token);
    smoothing_elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - smoothing_started);
    if (optimized.canceled) {
      return Canceled(started, search.expanded_states);
    }
    bool used_discrete_fallback =
        corridor.status != shared::CorridorStatus::kCertified;
    if (!optimized.optimized &&
        optimized.reason_code != "LEGGED_OPTIMIZATION_NOT_NEEDED") {
      warnings.push_back(optimized.reason_code);
      used_discrete_fallback = true;
    }
    const Interval true_start_height{
        .lower = current_state->body_pose.position_m.z,
        .upper = current_state->body_pose.position_m.z,
    };
    std::vector<LeggedTransition> selected = std::move(optimized.transitions);
    if (!ValidateTransitions(
            selected, true_start_height,
            *projection.projection, *capability, problem.config,
            problem.stop_token)) {
      if (problem.stop_token.stop_requested()) {
        return Canceled(started, search.expanded_states);
      }
      selected = discrete->transitions;
      warnings.emplace_back("LEGGED_OPTIMIZATION_SWEEP_FALLBACK");
      used_discrete_fallback = true;
    }
    if (!ValidateTransitions(
            selected, true_start_height,
            *projection.projection, *capability, problem.config,
            problem.stop_token)) {
      return Failure(
          PlanningOutcome::kNumericalFailure,
          ExecutionDirective::kHoldPosition,
          "LEGGED_VALIDATED_PATH_LOST", started, search.expanded_states,
          discrete->cost, std::move(warnings));
    }
    if (used_discrete_fallback &&
        std::ranges::find(
            warnings, "LEGGED_OPTIMIZATION_DISCRETE_FALLBACK") ==
            warnings.end()) {
      warnings.emplace_back("LEGGED_OPTIMIZATION_DISCRETE_FALLBACK");
    }
    if (used_discrete_fallback &&
        problem.config.optimization.require_smoothed_execution) {
      return Failure(
          PlanningOutcome::kNoKnownSafeRoute,
          ExecutionDirective::kNoSafeReference,
          "LEGGED_SMOOTHED_EXECUTION_REQUIRED", started,
          search.expanded_states, discrete->cost, std::move(warnings));
    }
    trajectory_mode = used_discrete_fallback
        ? TrajectoryMode::kDiscreteFallback
        : TrajectoryMode::kOptimized;
    collision_validation = CollisionValidation::kCertified;
    LeggedTimingResult timed = ParameterizeLeggedBodyTiming(
        selected, *capability,
        std::min(problem.config.optimization.maximum_smoothing_samples,
                 std::size_t{512U}),
        problem.stop_token);
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

  const LocalTrajectoryDiagnostics local_diagnostics{
      .trajectory_mode = trajectory_mode,
      .start_anchor_error_m = PositionError(
          trajectory.points.front().pose, current_state->body_pose),
      .endpoint_error_m =
          PlanarGoalError(problem.goal_odom, trajectory.points.back().pose),
      .maximum_curvature_per_m = 0.0,
      .collision_validation = collision_validation,
      .smoothing_elapsed_s =
          std::chrono::duration<double>(smoothing_elapsed).count(),
      .landing_field_elapsed_s = 0.0,
  };
  if (!std::isfinite(local_diagnostics.start_anchor_error_m) ||
      !std::isfinite(local_diagnostics.endpoint_error_m) ||
      !std::isfinite(local_diagnostics.smoothing_elapsed_s)) {
    return Failure(PlanningOutcome::kNumericalFailure,
                   ExecutionDirective::kNoSafeReference,
                   "LEGGED_TRAJECTORY_DIAGNOSTICS_NONFINITE", started,
                   search.expanded_states, discrete->cost,
                   std::move(warnings));
  }

  return PlannerOutput{
      .outcome = PlanningOutcome::kNewReferenceAvailable,
      .directive = ExecutionDirective::kActivateNewReference,
      .reason_code = "LEGGED_BODY_PLAN_AVAILABLE",
      .reference = MotionReference{
          .plan_id = "legged/" + problem.request_id,
          .platform_type = PlatformType::kLegged,
          .input_time = problem.state_time,
          .data = std::move(trajectory),
      },
      .diagnostics = PlannerDiagnostics{
          .planner_name = std::string{kPlannerName},
          .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started),
          .expanded_states = search.expanded_states,
          .best_cost = discrete->cost,
          .warning_codes = std::move(warnings),
          .local_trajectory = local_diagnostics,
      },
  };
}

}  // namespace lunar::planning::legged
