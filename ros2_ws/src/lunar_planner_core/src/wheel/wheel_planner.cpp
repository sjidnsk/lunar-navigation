#include "wheel/wheel_planner.hpp"

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

#include "shared/ara_star.hpp"
#include "shared/convex_corridor.hpp"
#include "shared/map_snapshot.hpp"
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

PlannerOutput WheelPlanner::Plan(
    const hierarchical::LocalPlanningProblem& problem) const {
  const auto started = std::chrono::steady_clock::now();
  if (problem.stop_token.stop_requested()) {
    return Canceled(started);
  }
  const auto* current_state = std::get_if<WheeledState>(&problem.current_state);
  const auto* capability = std::get_if<WheeledCapability>(&problem.capability);
  if (current_state == nullptr || capability == nullptr) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        "WHEEL_PLATFORM_TYPE_MISMATCH", started);
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
  if (!HasFeasibleGoalPosition(problem.goal_odom, *projection.projection)) {
    return Failure(
        PlanningOutcome::kGoalInfeasible,
        ExecutionDirective::kHoldPosition,
        "WHEEL_GOAL_INFEASIBLE", started);
  }

  WheelLatticeBuildResult lattice = BuildWheelLattice(
      *current_state, problem.goal_odom, *projection.projection, *capability,
      problem.config, problem.stop_token);
  if (!lattice.ok()) {
    switch (lattice.status) {
      case WheelLatticeStatus::kCanceled:
        return Canceled(started);
      case WheelLatticeStatus::kResourceExhausted:
        return Failure(
            PlanningOutcome::kResourceExhausted,
            ExecutionDirective::kNoSafeReference,
            lattice.reason_code, started);
      case WheelLatticeStatus::kInvalidRequest:
        if (lattice.reason_code == "WHEEL_START_NOT_SAFE" ||
            lattice.reason_code == "WHEEL_START_CONNECTOR_INFEASIBLE") {
          return Failure(
              PlanningOutcome::kNoKnownSafeRoute,
              ExecutionDirective::kNoSafeReference,
              lattice.reason_code, started);
        }
        return Failure(
            PlanningOutcome::kInvalidRequest,
            ExecutionDirective::kNoSafeReference,
            lattice.reason_code, started);
      case WheelLatticeStatus::kReady:
        break;
    }
  }
  if (!lattice.graph.has_value()) {
    return Failure(
        PlanningOutcome::kNumericalFailure,
        ExecutionDirective::kHoldPosition,
        "WHEEL_LATTICE_RESULT_INVALID", started);
  }
  if (std::ranges::none_of(
          lattice.graph->search_problem.goal_mask,
          [](const std::uint8_t value) { return value != 0U; })) {
    return Failure(
        PlanningOutcome::kNoKnownSafeRoute,
        ExecutionDirective::kNoSafeReference,
        "WHEEL_NO_KNOWN_SAFE_ROUTE", started);
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
          "WHEEL_NO_KNOWN_SAFE_ROUTE", started, search.expanded_states);
    case shared::AraStarStatus::kInvalidProblem:
      return Failure(
          PlanningOutcome::kNumericalFailure,
          ExecutionDirective::kHoldPosition,
          search.reason_code, started, search.expanded_states);
    case shared::AraStarStatus::kSolved:
      break;
  }
  const auto discrete = ResolveWheelPlan(*lattice.graph, search);
  if (!discrete.has_value()) {
    return Failure(
        PlanningOutcome::kNumericalFailure,
        ExecutionDirective::kHoldPosition,
        "WHEEL_SEARCH_RESULT_INVALID", started, search.expanded_states);
  }

  std::vector<std::string> warnings;
  TrajectoryReference trajectory;
  if (discrete->transitions.empty()) {
    trajectory = StationaryTrajectory(*current_state);
  } else {
    const shared::CorridorResult corridor = shared::BuildConvexCorridor(
        *projection.projection, Centerline(discrete->transitions),
        shared::CorridorTightening{
            .footprint_support_radius_m =
                FootprintSupportRadius(*capability),
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
    WheelOptimizationResult optimized = OptimizeWheelSpline(
        discrete->transitions, corridor, problem.config.optimization,
        problem.stop_token);
    if (optimized.canceled) {
      return Canceled(started, search.expanded_states);
    }
    if (!optimized.optimized &&
        optimized.reason_code != "WHEEL_OPTIMIZATION_NOT_NEEDED") {
      warnings.push_back(optimized.reason_code);
    }
    std::vector<WheelTransition> selected = std::move(optimized.transitions);
    WheelSweepValidator validator{
        *projection.projection, *capability,
        problem.config.wheel.continuous_validation_maximum_subdivisions};
    const bool optimized_valid = std::ranges::all_of(
        selected, [&](const WheelTransition& transition) {
          return validator.Validate(transition, problem.stop_token).valid;
        });
    if (problem.stop_token.stop_requested()) {
      return Canceled(started, search.expanded_states);
    }
    if (!optimized_valid) {
      selected = discrete->transitions;
      warnings.emplace_back("WHEEL_OPTIMIZATION_SWEEP_FALLBACK");
    }
    const bool discrete_valid = std::ranges::all_of(
        selected, [&](const WheelTransition& transition) {
          return validator.Validate(transition, problem.stop_token).valid;
        });
    if (problem.stop_token.stop_requested()) {
      return Canceled(started, search.expanded_states);
    }
    if (!discrete_valid) {
      return Failure(
          PlanningOutcome::kNumericalFailure,
          ExecutionDirective::kHoldPosition,
          "WHEEL_VALIDATED_PATH_LOST", started, search.expanded_states,
          discrete->cost, std::move(warnings));
    }
    WheelTimingResult timed = ParameterizeWheelTiming(
        selected, *capability, problem.stop_token);
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
          .expanded_states = search.expanded_states,
          .best_cost = discrete->cost,
          .warning_codes = std::move(warnings),
      },
  };
}

}  // namespace lunar::planning::wheel
