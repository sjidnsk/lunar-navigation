#include "wheel/wheel_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "shared/convex_corridor.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/projection_cache.hpp"
#include "shared/safe_projection.hpp"
#include "wheel/wheel_lattice.hpp"
#include "wheel/wheel_spline_optimizer.hpp"
#include "wheel/wheel_sweep_validator.hpp"
#include "wheel/wheel_timing.hpp"
#include "wheel/wheel_types.hpp"

namespace lunar::planning::wheel {
namespace {

constexpr std::string_view kPlannerName = "cpp_v3_native_wheel";

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
    const shared::SafeProjection& projection) {
  if (projection.source_map() == nullptr) {
    return false;
  }
  if (const auto* point = std::get_if<PointGoal>(&goal.target)) {
    const auto cell = projection.source_map()->PositionToCell(Vec2{
        .x = point->position_m.x,
        .y = point->position_m.y,
    });
    return cell.has_value() && projection.HardFeasible(*cell);
  }
  GoalRegion position_goal = goal;
  position_goal.yaw_rad.reset();
  for (std::size_t y = 0U; y < projection.source_map()->height(); ++y) {
    for (std::size_t x = 0U; x < projection.source_map()->width(); ++x) {
      const shared::GridCell cell{
          .x = static_cast<std::int32_t>(x),
          .y = static_cast<std::int32_t>(y),
      };
      if (!projection.HardFeasible(cell)) {
        continue;
      }
      if (GoalContainsPose(
              position_goal,
              WheelPose{
                  .position_m = projection.source_map()->CellCenter(cell),
                  .yaw_rad = 0.0,
              })) {
        return true;
      }
    }
  }
  return false;
}

[[nodiscard]] std::vector<Vec2> Centerline(
    const std::vector<WheelTransition>& transitions) {
  std::vector<Vec2> result;
  if (transitions.empty()) {
    return result;
  }
  result.reserve(transitions.size() + 1U);
  result.push_back(Vec2{
      .x = transitions.front().source_pose.position_m.x,
      .y = transitions.front().source_pose.position_m.y,
  });
  for (const WheelTransition& transition : transitions) {
    result.push_back(Vec2{
        .x = transition.target_pose.position_m.x,
        .y = transition.target_pose.position_m.y,
    });
  }
  return result;
}

[[nodiscard]] double FootprintSupportRadius(
    const WheeledCapability& capability) noexcept {
  double radius = 0.0;
  for (const Vec2& vertex : capability.footprint_xy_m) {
    radius = std::max(radius, std::hypot(vertex.x, vertex.y));
  }
  return radius;
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
  if (GoalContainsPose(position_goal,
                       WheelPose{.position_m = pose.position_m})) {
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
    const WheeledState& state) {
  return TrajectoryReference{
      .semantics = TrajectorySemantics::kWheeledBase,
      .points = {
          TrajectoryPoint{
              .time_from_start = std::chrono::nanoseconds{0},
              .pose = state.pose,
              .velocity = {},
          },
          TrajectoryPoint{
              .time_from_start = std::chrono::milliseconds{1},
              .pose = state.pose,
              .velocity = {},
          },
      },
  };
}

}  // namespace

namespace {

[[nodiscard]] PlannerOutput PlanRankedOutput(
    const std::span<const hierarchical::LocalPlanningProblem> problems,
    std::optional<std::size_t>& selected_problem_index,
    bool& projection_cache_hit,
    shared::ProjectionCache& projection_cache) {
  const auto started = std::chrono::steady_clock::now();
  if (problems.empty()) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        "WHEEL_RANKED_PROBLEMS_EMPTY", started);
  }
  const hierarchical::LocalPlanningProblem& common = problems.front();
  if (common.stop_token.stop_requested()) {
    return Canceled(started);
  }
  const auto* current_state = std::get_if<WheeledState>(&common.current_state);
  const auto* capability = std::get_if<WheeledCapability>(&common.capability);
  if (current_state == nullptr || capability == nullptr) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        "WHEEL_PLATFORM_TYPE_MISMATCH", started);
  }

  const shared::ProjectionContextResult projection =
      projection_cache.GetOrBuild(
          shared::MakeProjectionCacheKey(
              common.local_map_generation, common.platform_id,
              common.capability_version, common.local_map,
              common.capability, common.config.map_safety),
          common.local_map, common.capability, common.config.map_safety,
          common.stop_token);
  if (!projection.ok()) {
    if (projection.reason_code == "REQUEST_CANCELED") {
      return Canceled(started);
    }
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        projection.reason_code, started);
  }
  projection_cache_hit = projection.cache_hit;
  const shared::SafeProjection& safe_projection =
      *projection.context->projection;
  std::vector<GoalRegion> ranked_goals;
  ranked_goals.reserve(problems.size());
  bool has_feasible_goal = false;
  for (const hierarchical::LocalPlanningProblem& problem : problems) {
    ranked_goals.push_back(problem.goal_odom);
    has_feasible_goal = has_feasible_goal ||
        HasFeasibleGoalPosition(problem.goal_odom, safe_projection);
  }
  if (!has_feasible_goal) {
    return Failure(
        PlanningOutcome::kGoalInfeasible,
        ExecutionDirective::kHoldPosition,
        "WHEEL_GOAL_INFEASIBLE", started);
  }

  WheelLatticeSearchResult search = SearchWheelLatticeRanked(
      *current_state, ranked_goals, safe_projection, common.search_domain,
      *capability,
      common.config, common.stop_token);
  if (!search.ok()) {
    const std::size_t expanded_states = search.plan.has_value()
        ? search.plan->expanded_states
        : 0U;
    switch (search.status) {
      case WheelLatticeStatus::kCanceled:
        return Canceled(started, expanded_states);
      case WheelLatticeStatus::kResourceExhausted:
        return Failure(
            PlanningOutcome::kResourceExhausted,
            ExecutionDirective::kNoSafeReference,
            search.reason_code, started, expanded_states);
      case WheelLatticeStatus::kInvalidRequest:
        if (search.reason_code == "WHEEL_START_NOT_SAFE" ||
            search.reason_code == "WHEEL_START_CONNECTOR_INFEASIBLE") {
          return Failure(
              PlanningOutcome::kNoKnownSafeRoute,
              ExecutionDirective::kNoSafeReference,
              search.reason_code, started, expanded_states);
        }
        return Failure(
            PlanningOutcome::kInvalidRequest,
            ExecutionDirective::kNoSafeReference,
            search.reason_code, started, expanded_states);
      case WheelLatticeStatus::kNoPath:
        return Failure(
            PlanningOutcome::kNoKnownSafeRoute,
            ExecutionDirective::kNoSafeReference,
            search.reason_code, started, expanded_states);
      case WheelLatticeStatus::kSolved:
        break;
    }
  }
  if (!search.plan.has_value()) {
    return Failure(
        PlanningOutcome::kNumericalFailure,
        ExecutionDirective::kHoldPosition,
        "WHEEL_SEARCH_RESULT_INVALID", started);
  }
  if (!search.selected_goal_index.has_value() ||
      *search.selected_goal_index >= problems.size()) {
    return Failure(
        PlanningOutcome::kNumericalFailure,
        ExecutionDirective::kHoldPosition,
        "WHEEL_SELECTED_FRONTIER_INVALID", started,
        search.plan->expanded_states);
  }
  selected_problem_index = search.selected_goal_index;
  const hierarchical::LocalPlanningProblem& problem =
      problems[*selected_problem_index];
  const WheelDiscretePlan& discrete = *search.plan;

  std::vector<std::string> warnings;
  TrajectoryReference trajectory;
  TrajectoryMode trajectory_mode = TrajectoryMode::kStationary;
  CollisionValidation collision_validation =
      CollisionValidation::kNotApplicable;
  std::chrono::nanoseconds smoothing_elapsed{};
  double maximum_curvature_per_m = 0.0;
  if (discrete.transitions.empty()) {
    trajectory = StationaryTrajectory(*current_state);
  } else {
    const shared::CorridorResult corridor = shared::BuildConvexCorridor(
        safe_projection, Centerline(discrete.transitions),
        shared::CorridorTightening{
            .footprint_support_radius_m =
                FootprintSupportRadius(*capability),
            .tracking_error_bound_m = 0.0,
            .additional_margin_m = 0.0,
        },
        problem.config.corridor, problem.stop_token);
    if (corridor.status == shared::CorridorStatus::kCanceled) {
      return Canceled(started, discrete.expanded_states);
    }
    if (corridor.status != shared::CorridorStatus::kCertified) {
      warnings.push_back(corridor.reason_code);
    }
    const auto smoothing_started = std::chrono::steady_clock::now();
    WheelOptimizationResult optimized = OptimizeWheelSpline(
        discrete.transitions, corridor, problem.config.optimization,
        problem.stop_token);
    smoothing_elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - smoothing_started);
    if (optimized.canceled) {
      return Canceled(started, discrete.expanded_states);
    }
    bool used_discrete_fallback =
        corridor.status != shared::CorridorStatus::kCertified;
    if (!optimized.optimized &&
        optimized.reason_code != "WHEEL_OPTIMIZATION_NOT_NEEDED") {
      warnings.push_back(optimized.reason_code);
      used_discrete_fallback = true;
    }
    std::vector<WheelTransition> selected = std::move(optimized.transitions);
    WheelSweepValidator validator{
        safe_projection, *capability, common.search_domain};
    const bool optimized_valid = std::ranges::all_of(
        selected, [&](const WheelTransition& transition) {
          return validator.Validate(transition, problem.stop_token).valid;
        });
    if (problem.stop_token.stop_requested()) {
      return Canceled(started, discrete.expanded_states);
    }
    if (!optimized_valid) {
      selected = discrete.transitions;
      warnings.emplace_back("WHEEL_OPTIMIZATION_SWEEP_FALLBACK");
      used_discrete_fallback = true;
    }
    const bool discrete_valid = std::ranges::all_of(
        selected, [&](const WheelTransition& transition) {
          return validator.Validate(transition, problem.stop_token).valid;
        });
    if (problem.stop_token.stop_requested()) {
      return Canceled(started, discrete.expanded_states);
    }
    if (!discrete_valid) {
      return Failure(
          PlanningOutcome::kNumericalFailure,
          ExecutionDirective::kHoldPosition,
          "WHEEL_VALIDATED_PATH_LOST", started, discrete.expanded_states,
          discrete.cost, std::move(warnings));
    }
    if (used_discrete_fallback &&
        std::ranges::find(
            warnings, "WHEEL_OPTIMIZATION_DISCRETE_FALLBACK") ==
            warnings.end()) {
      warnings.emplace_back("WHEEL_OPTIMIZATION_DISCRETE_FALLBACK");
    }
    if (used_discrete_fallback &&
        problem.config.optimization.require_smoothed_execution) {
      return Failure(
          PlanningOutcome::kNoKnownSafeRoute,
          ExecutionDirective::kNoSafeReference,
          "WHEEL_SMOOTHED_EXECUTION_REQUIRED", started,
          discrete.expanded_states, discrete.cost, std::move(warnings));
    }
    trajectory_mode = used_discrete_fallback
        ? TrajectoryMode::kDiscreteFallback
        : TrajectoryMode::kOptimized;
    collision_validation = CollisionValidation::kCertified;
    for (const WheelTransition& transition : selected) {
      maximum_curvature_per_m =
          std::max(maximum_curvature_per_m,
                   std::abs(transition.curvature_per_m));
    }
    WheelTimingResult timed = ParameterizeWheelTiming(
        selected, *capability, current_state->velocity,
        std::min(problem.config.optimization.maximum_smoothing_samples,
                 std::size_t{512U}),
        problem.stop_token);
    if (timed.canceled) {
      return Canceled(started, discrete.expanded_states);
    }
    if (!timed.ok()) {
      return Failure(
          PlanningOutcome::kNumericalFailure,
          ExecutionDirective::kHoldPosition,
          timed.reason_code, started, discrete.expanded_states,
          discrete.cost, std::move(warnings));
    }
    trajectory = std::move(*timed.trajectory);
  }

  const LocalTrajectoryDiagnostics local_diagnostics{
      .trajectory_mode = trajectory_mode,
      .start_anchor_error_m =
          PositionError(trajectory.points.front().pose, current_state->pose),
      .endpoint_error_m =
          PlanarGoalError(problem.goal_odom, trajectory.points.back().pose),
      .maximum_curvature_per_m = maximum_curvature_per_m,
      .collision_validation = collision_validation,
      .smoothing_elapsed_s =
          std::chrono::duration<double>(smoothing_elapsed).count(),
      .landing_field_elapsed_s = 0.0,
  };
  if (!std::isfinite(local_diagnostics.start_anchor_error_m) ||
      !std::isfinite(local_diagnostics.endpoint_error_m) ||
      !std::isfinite(local_diagnostics.maximum_curvature_per_m) ||
      !std::isfinite(local_diagnostics.smoothing_elapsed_s)) {
    return Failure(PlanningOutcome::kNumericalFailure,
                   ExecutionDirective::kNoSafeReference,
                   "WHEEL_TRAJECTORY_DIAGNOSTICS_NONFINITE", started,
                   discrete.expanded_states, discrete.cost,
                   std::move(warnings));
  }

  return PlannerOutput{
      .outcome = PlanningOutcome::kNewReferenceAvailable,
      .directive = ExecutionDirective::kActivateNewReference,
      .reason_code = "WHEEL_PLAN_AVAILABLE",
      .reference = MotionReference{
          .plan_id = "wheel/" + problem.request_id,
          .platform_type = PlatformType::kWheeled,
          .input_time = problem.state_time,
          .data = std::move(trajectory),
      },
      .diagnostics = PlannerDiagnostics{
          .planner_name = std::string{kPlannerName},
          .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started),
          .expanded_states = discrete.expanded_states,
          .best_cost = discrete.cost,
          .warning_codes = std::move(warnings),
          .local_trajectory = local_diagnostics,
      },
  };
}

}  // namespace

WheelRankedPlanResult WheelPlanner::PlanRanked(
    const std::span<const hierarchical::LocalPlanningProblem> problems) const {
  std::optional<std::size_t> selected_problem_index;
  bool projection_cache_hit = false;
  PlannerOutput output = PlanRankedOutput(
      problems, selected_problem_index, projection_cache_hit,
      projection_cache_);
  return WheelRankedPlanResult{
      .output = std::move(output),
      .selected_problem_index = selected_problem_index,
      .projection_cache_hit = projection_cache_hit,
  };
}

PlannerOutput WheelPlanner::Plan(
    const hierarchical::LocalPlanningProblem& problem) const {
  return PlanRanked(std::span{&problem, std::size_t{1U}}).output;
}

}  // namespace lunar::planning::wheel
