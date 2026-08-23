#include "lunar_pure_planner_core/planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "hierarchical/global_route_planner.hpp"
#include "hierarchical/reference_composer.hpp"
#include "hopper/anytime_hopper_planner.hpp"
#include "legged/anytime_legged_planner.hpp"
#include "legged/legged_types.hpp"
#include "shared/controlled_work.hpp"
#include "shared/local_terrain_projection.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/obstacle_height_estimator.hpp"
#include "wheel/anytime_wheel_planner.hpp"

namespace lunar::pure_planning {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] SteadyClock::time_point ReadNow(const NowFn& now) {
  return now ? now() : SteadyClock::now();
}

[[nodiscard]] std::chrono::nanoseconds NonNegativeElapsed(
    const SteadyClock::time_point start,
    const SteadyClock::time_point finish) noexcept {
  return finish > start
             ? std::chrono::duration_cast<std::chrono::nanoseconds>(finish -
                                                                    start)
             : 0ns;
}

[[nodiscard]] bool FinitePointGoal(const GoalRegion& goal) noexcept {
  const auto* point = std::get_if<PointGoal>(&goal.target);
  return point != nullptr && std::isfinite(point->position_m.x) &&
         std::isfinite(point->position_m.y) &&
         std::isfinite(point->tolerance_m) && point->tolerance_m >= 0.0 &&
         (!goal.yaw_rad.has_value() || std::isfinite(*goal.yaw_rad)) &&
         std::isfinite(goal.yaw_tolerance_rad) &&
         goal.yaw_tolerance_rad >= 0.0;
}

[[nodiscard]] bool MatchingPlatform(const PlanningRequest& input) noexcept {
  return input.current_state.index() == input.capability.index();
}

[[nodiscard]] PlanningResult Failure(const PlanningStatus status,
                                     std::string reason_code) {
  return PlanningResult{
      .status = status,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] PlanningResult LocalFailure(const LocalPlanStatus status) {
  switch (status) {
    case LocalPlanStatus::kNoPath:
      return Failure(PlanningStatus::kNoPath, "NO_PATH");
    case LocalPlanStatus::kTimedOut:
      return Failure(PlanningStatus::kTimedOut, "TIMEOUT");
    case LocalPlanStatus::kCanceled:
      return Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED");
    case LocalPlanStatus::kInvalidInput:
      return Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT");
    case LocalPlanStatus::kPlannerError:
    case LocalPlanStatus::kSolved:
      return Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR");
  }
  return Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR");
}

[[nodiscard]] PlanningResult GlobalFailure(const std::string& reason_code) {
  if (reason_code == "NO_PATH") {
    return Failure(PlanningStatus::kNoPath, "NO_PATH");
  }
  if (reason_code == "TIMEOUT") {
    return Failure(PlanningStatus::kTimedOut, "TIMEOUT");
  }
  if (reason_code == "REQUEST_CANCELED") {
    return Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (reason_code == "INVALID_INPUT") {
    return Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT");
  }
  return Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR");
}

[[nodiscard]] bool FinalizeTiming(
    PlanningResult& result, const SteadyClock::time_point started,
    const SteadyClock::time_point output_started,
    const NowFn& now, const PlannerCallTiming& timing,
    SteadyClock::time_point* const finished_at) noexcept {
  result.timing = timing;
  try {
    const SteadyClock::time_point finish = ReadNow(now);
    result.timing.total_elapsed = NonNegativeElapsed(started, finish);
    result.timing.output_elapsed +=
        NonNegativeElapsed(output_started, finish);
    if (finished_at != nullptr) {
      *finished_at = finish;
    }
    return true;
  } catch (...) {
    return false;
  }
}

[[nodiscard]] bool MinimalLocalMapValid(const GridMap& map) noexcept {
  if (map.frame_id.empty() || map.CellCount() == 0U ||
      !std::isfinite(map.resolution_m) || map.resolution_m <= 0.0 ||
      !std::isfinite(map.origin_m.x) || !std::isfinite(map.origin_m.y) ||
      !std::isfinite(map.origin_m.z)) {
    return false;
  }
  const auto occupancy = map.layers.find("occupancy");
  const auto elevation = map.layers.find("elevation");
  return occupancy != map.layers.end() && elevation != map.layers.end() &&
         std::holds_alternative<std::vector<float>>(occupancy->second.values) &&
         std::holds_alternative<std::vector<float>>(elevation->second.values) &&
         occupancy->second.size() == map.CellCount() &&
         elevation->second.size() == map.CellCount();
}

[[nodiscard]] LocalStageResult LocalControlledFailure(
    const std::string& reason_code) {
  if (reason_code == "TIMEOUT") {
    return {.status = LocalPlanStatus::kTimedOut, .reason_code = "TIMEOUT"};
  }
  if (reason_code == "REQUEST_CANCELED") {
    return {.status = LocalPlanStatus::kCanceled,
            .reason_code = "REQUEST_CANCELED"};
  }
  return {.status = LocalPlanStatus::kInvalidInput,
          .reason_code = "INVALID_INPUT"};
}

[[nodiscard]] Quaternion QuaternionFromYaw(const double yaw) noexcept {
  return Quaternion{.w = std::cos(yaw / 2.0), .z = std::sin(yaw / 2.0)};
}

[[nodiscard]] LocalStageResult PlanLocalDefault(
    const PlanningRequest& input, const LocalGoalSet& goals_odom,
    SearchControl control) {
  if (goals_odom.goals_odom.empty()) {
    return {.status = LocalPlanStatus::kInvalidInput,
            .reason_code = "INVALID_INPUT"};
  }
  auto snapshot = shared::MapSnapshot::Create(
      input.world.local_map, shared::MapContract::kLocalElevation, control);
  if (!snapshot.ok()) {
    return LocalControlledFailure(snapshot.reason_code);
  }
  auto projection = shared::BuildLocalTerrainProjection(
      snapshot.snapshot,
      static_cast<float>(input.config.local_occupancy_threshold), control);
  if (!projection.ok()) {
    return LocalControlledFailure(projection.reason_code);
  }
  const shared::LocalTerrainProjection& terrain = *projection.value;

  if (const auto* capability =
          std::get_if<WheeledCapability>(&input.capability)) {
    const auto* state = std::get_if<WheeledState>(&input.current_state);
    if (state == nullptr) {
      return {.status = LocalPlanStatus::kInvalidInput,
              .reason_code = "INVALID_INPUT"};
    }
    wheel::WheelPlanResult result = wheel::PlanWheel({
        .start = *state,
        .goals_odom = goals_odom,
        .terrain = &terrain,
        .capability = capability,
        .local_source_sequence = input.world.local_map_sequence,
        .control = control,
        .search = input.config.search,
    });
    return LocalStageResult{
        .status = result.status,
        .data = result.ok()
                    ? std::optional<MotionReferenceData>{TrajectoryReference{
                          .semantics = TrajectorySemantics::kWheeledBase,
                          .points = std::move(result.trajectory),
                      }}
                    : std::nullopt,
        .reason_code = std::move(result.reason_code),
        .selected_goal_index = result.selected_goal_index,
    };
  }

  const GoalRegion& goal_odom = goals_odom.goals_odom.front();

  if (const auto* capability =
          std::get_if<LeggedCapability>(&input.capability)) {
    const auto* state = std::get_if<LeggedState>(&input.current_state);
    if (state == nullptr) {
      return {.status = LocalPlanStatus::kInvalidInput,
              .reason_code = "INVALID_INPUT"};
    }
    legged::LeggedPlanResult result = legged::PlanLegged({
        .start = *state,
        .goal_odom = goal_odom,
        .terrain = &terrain,
        .capability = capability,
        .control = control,
        .search = input.config.search,
    });
    std::optional<MotionReferenceData> data;
    if (result.ok()) {
      if (const auto stopped = shared::StopReason(control);
          stopped.has_value()) {
        return LocalControlledFailure(std::string{*stopped});
      }
      TrajectoryReference trajectory{
          .semantics = TrajectorySemantics::kLeggedBodyReference,
      };
      trajectory.points.reserve(result.trajectory.size() + 1U);
      if (const auto stopped = shared::StopReason(control);
          stopped.has_value()) {
        return LocalControlledFailure(std::string{*stopped});
      }
      trajectory.points.push_back(
          {.pose = state->body_pose, .velocity = state->body_velocity});
      for (const legged::LeggedTransition& transition : result.trajectory) {
        if (const auto stopped = shared::StopReason(control);
            stopped.has_value()) {
          return LocalControlledFailure(std::string{*stopped});
        }
        trajectory.points.push_back({
            .pose = Pose3{
                .position_m = transition.target_pose.position_m,
                .orientation =
                    QuaternionFromYaw(transition.target_pose.yaw_rad),
            },
        });
      }
      if (const auto stopped = shared::StopReason(control);
          stopped.has_value()) {
        return LocalControlledFailure(std::string{*stopped});
      }
      data = std::move(trajectory);
    }
    return LocalStageResult{
        .status = result.status,
        .data = std::move(data),
        .reason_code = std::move(result.reason_code),
        .selected_goal_index = result.ok()
                                   ? std::optional<std::size_t>{0U}
                                   : std::nullopt,
    };
  }

  const auto* capability = std::get_if<HopperCapability>(&input.capability);
  const auto* state = std::get_if<HopperState>(&input.current_state);
  if (capability == nullptr || state == nullptr) {
    return {.status = LocalPlanStatus::kInvalidInput,
            .reason_code = "INVALID_INPUT"};
  }
  hopper::HopperPlanResult result = hopper::PlanHopper({
      .start = *state,
      .goal_odom = goal_odom,
      .terrain = &terrain,
      .capability = capability,
      .control = std::move(control),
      .search = input.config.search,
      .estimate_height = shared::EstimateObstacleHeight,
  });
  return LocalStageResult{
      .status = result.status,
      .data = result.ok()
                  ? std::optional<MotionReferenceData>{
                        HopReference{.segments = std::move(result.hops)}}
                  : std::nullopt,
      .reason_code = std::move(result.reason_code),
      .selected_goal_index = result.ok()
                                 ? std::optional<std::size_t>{0U}
                                 : std::nullopt,
  };
}

}  // namespace

Planner::Planner()
    : Planner(PlannerBackends{
          .global = [](const PlanningRequest& input, SearchControl control) {
            return hierarchical::PlanSurfaceGlobal(input, std::move(control));
          },
          .local = PlanLocalDefault,
      }) {}

Planner::Planner(PlannerBackends backends) : backends_(std::move(backends)) {}

PlanningResult Planner::Plan(const PlanningRequest& input) noexcept {
  PlannerCallTiming timing;
  SteadyClock::time_point started{};
  try {
    if (input.request_started_at.has_value()) {
      started = *input.request_started_at;
    } else {
      started = ReadNow(input.control.now);
    }
  } catch (...) {
    return Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR");
  }
  const RequestTimingPolicy policy = MakeRequestTimingPolicy(started);
  SteadyClock::time_point phase_started = started;

  const auto report_progress = [&](const PlannerPhase phase,
                                   const std::chrono::nanoseconds elapsed) {
    if (input.progress) {
      input.progress(PlannerProgress{.phase = phase, .elapsed = elapsed});
    }
  };

  const auto finish_phase = [&](std::chrono::nanoseconds& elapsed,
                                const PlannerPhase phase) {
    const auto phase_finished = ReadNow(input.control.now);
    elapsed += NonNegativeElapsed(phase_started, phase_finished);
    phase_started = phase_finished;
    report_progress(phase, elapsed);
  };

  const auto finish = [&](PlanningResult result) {
    SteadyClock::time_point finished_at{};
    if (!FinalizeTiming(result, started, phase_started, input.control.now, timing,
                        &finished_at)) {
      PlanningResult error =
          Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR");
      error.timing = timing;
      return error;
    }
    try {
      report_progress(PlannerPhase::kOutput, result.timing.output_elapsed);
    } catch (...) {
      PlanningResult error =
          Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR");
      error.timing = result.timing;
      return error;
    }
    if (result.status != PlanningStatus::kCanceled &&
        finished_at >= policy.hard_deadline) {
      PlanningResult timeout = Failure(PlanningStatus::kTimedOut, "TIMEOUT");
      timeout.timing = result.timing;
      return timeout;
    }
    return result;
  };

  try {
    if (input.control.stop_token.stop_requested()) {
      return finish(Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED"));
    }
    const bool known_mode =
        input.environment_mode == EnvironmentMode::kLunarSurface ||
        input.environment_mode == EnvironmentMode::kLavaTube;
    if (!known_mode || input.request_id.empty() || !FinitePointGoal(input.goal_map) ||
        !MatchingPlatform(input) || !MinimalLocalMapValid(input.world.local_map)) {
      return finish(Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT"));
    }
    finish_phase(timing.snapshot_projection_elapsed,
                 PlannerPhase::kSnapshotProjection);

    std::optional<GlobalRoute> global_route;
    LocalGoalSet local_goals;
    if (input.environment_mode == EnvironmentMode::kLunarSurface) {
      if (!input.world.global_map.has_value() || !backends_.global) {
        return finish(Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT"));
      }
      SearchControl global_control{
          .deadline = policy.hard_deadline,
          .stop_token = input.control.stop_token,
          .now = input.control.now,
      };
      GlobalStageResult global;
      SteadyClock::time_point global_finished{};
      {
        ScopedPlannerCall call(PlannerStage::kGlobal, timing,
                               input.control.now);
        global = backends_.global(input, std::move(global_control));
        if (!call.Finish(&global_finished)) {
          return finish(
              Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR"));
        }
      }
      report_progress(PlannerPhase::kGlobal, timing.global_elapsed);
      phase_started = global_finished;
      if (global.reason_code == "REQUEST_CANCELED" ||
          input.control.stop_token.stop_requested()) {
        return finish(
            Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED"));
      }
      if (global_finished >= policy.hard_deadline) {
        return finish(Failure(PlanningStatus::kTimedOut, "TIMEOUT"));
      }
      if (!global.route.has_value()) {
        return finish(GlobalFailure(global.reason_code));
      }
      global_route = std::move(global.route);
      const auto selected = hierarchical::SelectSurfaceLocalGoals(
          input, *global_route,
          SearchControl{
              .deadline = policy.hard_deadline,
              .stop_token = input.control.stop_token,
              .now = input.control.now,
          });
      if (!selected.ok()) {
        return finish(GlobalFailure(selected.reason_code));
      }
      local_goals = *selected.goals;
    } else {
      if (!hierarchical::GoalInsideLocalMap(input, input.goal_map)) {
        return finish(Failure(PlanningStatus::kGoalOutsideLocalMap,
                              "GOAL_OUTSIDE_LOCAL_MAP"));
      }
      const auto transformed =
          hierarchical::GoalMapToOdomPlanar(input, input.goal_map);
      if (!transformed.has_value()) {
        return finish(Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT"));
      }
      local_goals = LocalGoalSet{
          .goals_odom = {*transformed},
          .exact_final_goal = true,
      };
    }

    finish_phase(timing.local_goal_elapsed, PlannerPhase::kLocalGoal);

    if (!backends_.local) {
      return finish(Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR"));
    }
    SearchControl local_control{
        .deadline = policy.hard_deadline,
        .stop_token = input.control.stop_token,
        .now = input.control.now,
    };
    LocalStageResult local;
    SteadyClock::time_point local_finished{};
    {
      ScopedPlannerCall call(PlannerStage::kLocal, timing, input.control.now);
      local = backends_.local(input, local_goals, std::move(local_control));
      if (!call.Finish(&local_finished)) {
        return finish(
            Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR"));
      }
    }
    report_progress(PlannerPhase::kLocalSearch, timing.local_search_elapsed);
    phase_started = local_finished;
    if (local.status == LocalPlanStatus::kCanceled ||
        input.control.stop_token.stop_requested()) {
      return finish(
          Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED"));
    }
    if (local.status != LocalPlanStatus::kSolved &&
        local_finished >= policy.hard_deadline) {
      return finish(Failure(PlanningStatus::kTimedOut, "TIMEOUT"));
    }
    if (local.status != LocalPlanStatus::kSolved) {
      return finish(LocalFailure(local.status));
    }
    if (!local.data.has_value()) {
      return finish(Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR"));
    }

    hierarchical::ReferenceComposeResult composed =
        global_route.has_value()
            ? hierarchical::ComposeSurfaceReference(
                  input, *global_route, std::move(*local.data),
                  SearchControl{
                      .deadline = policy.hard_deadline,
                      .stop_token = input.control.stop_token,
                      .now = input.control.now,
                  })
            : hierarchical::ComposeCaveReference(input,
                                                  std::move(*local.data),
                                                  SearchControl{
                                                      .deadline = policy.hard_deadline,
                                                      .stop_token = input.control.stop_token,
                                                      .now = input.control.now,
                                                  });
    if (!composed.ok()) {
      return finish(GlobalFailure(composed.reason_code));
    }
    finish_phase(timing.certification_elapsed, PlannerPhase::kCertification);
    PlanningResult success{
        .status = PlanningStatus::kSuccess,
        .reason_code = {},
        .reference = std::move(composed.reference),
        .expanded_states = global_route.has_value()
                               ? global_route->expanded_states
                               : 0U,
        .selected_goal_index = local.selected_goal_index,
    };
    return finish(std::move(success));
  } catch (...) {
    return finish(Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR"));
  }
}

}  // namespace lunar::pure_planning
