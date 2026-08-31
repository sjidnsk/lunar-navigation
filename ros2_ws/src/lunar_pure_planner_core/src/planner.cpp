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
#include "grid_v1/grid_v1_planner.hpp"
#include "hopper/anytime_hopper_planner.hpp"
#include "legged/anytime_legged_planner.hpp"
#include "legged/legged_traversal_projection.hpp"
#include "legged/legged_types.hpp"
#include "shared/controlled_work.hpp"
#include "shared/active_planner_cache.hpp"
#include "shared/goal_distance_field.hpp"
#include "shared/local_terrain_projection.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/obstacle_height_estimator.hpp"
#include "shared/request_local_start_patch.hpp"
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

[[nodiscard]] LocalStageResult LocalStartPatchFailure(
    const std::string& reason_code) {
  if (reason_code == "WHEEL_START_SUPPORT_PLANE_UNAVAILABLE" ||
      reason_code == "WHEEL_START_SUPPORT_PLANE_INFEASIBLE") {
    return {.status = LocalPlanStatus::kNoPath,
            .reason_code = reason_code};
  }
  return LocalControlledFailure(reason_code);
}

[[nodiscard]] Quaternion QuaternionFromYaw(const double yaw) noexcept {
  return Quaternion{.w = std::cos(yaw / 2.0), .z = std::sin(yaw / 2.0)};
}

[[nodiscard]] LocalStageResult PlanLocalDefault(
    const PlanningRequest& input, const LocalGoalSet& goals_odom,
    SearchControl control, shared::ActivePlannerCache& cache) {
  if (goals_odom.goals_odom.empty()) {
    return {.status = LocalPlanStatus::kInvalidInput,
            .reason_code = "INVALID_INPUT"};
  }

  shared::RequestLocalStartPatch start_patch;
  if (const auto* capability =
          std::get_if<WheeledCapability>(&input.capability)) {
    const auto* state = std::get_if<WheeledState>(&input.current_state);
    if (state == nullptr) {
      return {.status = LocalPlanStatus::kInvalidInput,
              .reason_code = "INVALID_INPUT"};
    }
    auto analyzed = shared::AnalyzeWheelStartPatch(
        input.world.local_map, *state, *capability,
        input.config.local_occupancy_threshold, control);
    if (!analyzed.ok()) {
      return LocalStartPatchFailure(analyzed.reason_code);
    }
    start_patch = std::move(*analyzed.patch);
  }

  const auto snapshot = cache.local_snapshot().GetOrBuild(
      shared::MakeLocalSnapshotCacheKey(input.world.local_map_sequence,
                                        start_patch.identity),
      control, [&](const SearchControl& build_control) {
        shared::MapSnapshotBuildResult built;
        if (start_patch.required()) {
          auto patched = shared::BuildWheelStartPatchedMap(
              input.world.local_map, start_patch, build_control);
          if (!patched.ok()) {
            return shared::ImmutableCacheBuildResult<shared::MapSnapshot>{
                .reason_code = std::move(patched.reason_code),
            };
          }
          built = shared::MapSnapshot::Create(
              std::move(*patched.map), shared::MapContract::kLocalElevation);
        } else {
          built = shared::MapSnapshot::Create(
              input.world.local_map, shared::MapContract::kLocalElevation,
              build_control);
        }
        return shared::ImmutableCacheBuildResult<shared::MapSnapshot>{
            .value = std::move(built.snapshot),
            .reason_code = std::move(built.reason_code),
        };
      });
  if (!snapshot.ok()) {
    return LocalControlledFailure(snapshot.reason_code);
  }
  const auto local_projection_key = shared::MakeLocalProjectionCacheKey(
      input.world.local_map_sequence,
      input.config.local_occupancy_threshold, start_patch.identity);
  const auto projection = cache.local_projection().GetOrBuild(
      local_projection_key,
      control, [&](const SearchControl& build_control) {
        auto built = shared::BuildLocalTerrainProjection(
            snapshot.value,
            static_cast<float>(input.config.local_occupancy_threshold),
            build_control);
        return shared::ImmutableCacheBuildResult<
            shared::LocalTerrainProjection>{
            .value = !built.value.has_value()
                         ? nullptr
                         : std::make_shared<const shared::LocalTerrainProjection>(
                               std::move(*built.value)),
            .reason_code = std::move(built.reason_code),
        };
      });
  if (!projection.ok()) {
    LocalStageResult failure = LocalControlledFailure(projection.reason_code);
    failure.snapshot_cache_hit = snapshot.cache_hit;
    return failure;
  }
  const shared::LocalTerrainProjection& terrain = *projection.value;

  if (const auto* capability =
          std::get_if<WheeledCapability>(&input.capability)) {
    const auto* state = std::get_if<WheeledState>(&input.current_state);
    if (state == nullptr) {
      return {.status = LocalPlanStatus::kInvalidInput,
              .reason_code = "INVALID_INPUT",
              .snapshot_cache_hit = snapshot.cache_hit,
              .projection_cache_hit = projection.cache_hit};
    }
    std::vector<shared::GridCell> goal_cells;
    goal_cells.reserve(goals_odom.goals_odom.size());
    for (const GoalRegion& goal : goals_odom.goals_odom) {
      const auto* point = std::get_if<PointGoal>(&goal.target);
      if (point == nullptr) {
        continue;
      }
      const auto cell = terrain.map->PositionToCell(
          {.x = point->position_m.x, .y = point->position_m.y});
      if (cell.has_value()) {
        goal_cells.push_back(*cell);
      }
    }
    const std::uint64_t capability_fingerprint =
        shared::StableCapabilityFingerprint(input.capability);
    const auto goal_field = cache.goal_field().GetOrBuild(
        shared::MakeGoalFieldCacheKey(
            input.world.local_map_sequence,
            input.config.local_occupancy_threshold, capability_fingerprint,
            goals_odom, input.config.search, start_patch.identity),
        control, [&](const SearchControl& build_control) {
          auto built = shared::BuildGoalDistanceField(
              terrain, goal_cells, build_control);
          return shared::ImmutableCacheBuildResult<shared::GoalDistanceField>{
              .value = !built.has_value()
                           ? nullptr
                           : std::make_shared<const shared::GoalDistanceField>(
                                 std::move(*built)),
              .reason_code =
                  !built.has_value()
                      ? std::string{shared::StopReason(build_control)
                                        .value_or("INVALID_INPUT")}
                      : std::string{},
          };
        });
    if (!goal_field.ok()) {
      LocalStageResult failure = LocalControlledFailure(goal_field.reason_code);
      failure.snapshot_cache_hit = snapshot.cache_hit;
      failure.projection_cache_hit = projection.cache_hit;
      return failure;
    }
    wheel::WheelPlanResult result = wheel::PlanWheel({
        .start = *state,
        .goals_odom = goals_odom,
        .terrain = &terrain,
        .capability = capability,
        .local_source_sequence = input.world.local_map_sequence,
        .local_terrain_semantics_id =
            local_projection_key.semantic_identities.front(),
        .capability_fingerprint = capability_fingerprint,
        .goal_distance_field = goal_field.value,
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
        .snapshot_cache_hit = snapshot.cache_hit,
        .projection_cache_hit = projection.cache_hit,
        .goal_field_cache_hit = goal_field.cache_hit,
        .expanded_states = result.metrics.expanded_states,
        .best_cost = result.ok() ? std::optional<double>{result.cost}
                                 : std::nullopt,
    };
  }

  if (const auto* capability =
          std::get_if<LeggedCapability>(&input.capability)) {
    const auto* state = std::get_if<LeggedState>(&input.current_state);
    if (state == nullptr) {
      return {.status = LocalPlanStatus::kInvalidInput,
              .reason_code = "INVALID_INPUT",
              .snapshot_cache_hit = snapshot.cache_hit,
              .projection_cache_hit = projection.cache_hit};
    }
    const std::uint64_t capability_fingerprint =
        shared::StableCapabilityFingerprint(input.capability);
    const auto legged_projection =
        cache.legged_local_projection().GetOrBuild(
            shared::MakeLeggedLocalProjectionCacheKey(
                input.world.local_map_sequence, terrain.map->width(),
                terrain.map->height(), terrain.map->resolution_m(),
                input.config.local_occupancy_threshold,
                capability_fingerprint),
            control, [&](const SearchControl& build_control) {
              auto built = legged::BuildLeggedTraversalProjection(
                  projection.value, *capability, build_control);
              return shared::ImmutableCacheBuildResult<
                  legged::LeggedTraversalProjection>{
                  .value = std::move(built.value),
                  .reason_code = std::move(built.reason_code),
              };
            });
    if (!legged_projection.ok()) {
      LocalStageResult failure =
          LocalControlledFailure(legged_projection.reason_code);
      failure.snapshot_cache_hit = snapshot.cache_hit;
      failure.projection_cache_hit = projection.cache_hit;
      failure.legged_local.active = true;
      return failure;
    }
    std::vector<shared::GridCell> goal_cells;
    goal_cells.reserve(goals_odom.goals_odom.size());
    for (const GoalRegion& goal : goals_odom.goals_odom) {
      const auto* point = std::get_if<PointGoal>(&goal.target);
      if (point == nullptr) {
        return {.status = LocalPlanStatus::kInvalidInput,
                .reason_code = "INVALID_INPUT",
                .snapshot_cache_hit = snapshot.cache_hit,
                .projection_cache_hit = projection.cache_hit};
      }
      const auto cell = terrain.map->PositionToCell(
          {.x = point->position_m.x, .y = point->position_m.y});
      if (!cell.has_value()) {
        return {.status = LocalPlanStatus::kNoPath,
                .reason_code = "LEGGED_GOAL_OUTSIDE_LOCAL_MAP",
                .snapshot_cache_hit = snapshot.cache_hit,
                .projection_cache_hit = projection.cache_hit};
      }
      goal_cells.push_back(*cell);
    }
    const auto goal_field = cache.goal_field().GetOrBuild(
        shared::MakeGoalFieldCacheKey(
            input.world.local_map_sequence,
            input.config.local_occupancy_threshold, capability_fingerprint,
            goals_odom, input.config.search, start_patch.identity),
        control, [&](const SearchControl& build_control) {
          auto built = shared::BuildGoalDistanceField(
              *terrain.map,
              legged_projection.value->body_center_feasible,
              goal_cells, build_control);
          return shared::ImmutableCacheBuildResult<shared::GoalDistanceField>{
              .value = !built.has_value()
                  ? nullptr
                  : std::make_shared<const shared::GoalDistanceField>(
                        std::move(*built)),
              .reason_code = !built.has_value()
                  ? std::string{shared::StopReason(build_control)
                                    .value_or("LEGGED_NO_PATH")}
                  : std::string{},
          };
        });
    if (!goal_field.ok()) {
      LocalStageResult failure =
          goal_field.reason_code == "LEGGED_NO_PATH"
              ? LocalStageResult{.status = LocalPlanStatus::kNoPath,
                                 .reason_code = "LEGGED_NO_PATH"}
              : LocalControlledFailure(goal_field.reason_code);
      failure.snapshot_cache_hit = snapshot.cache_hit;
      failure.projection_cache_hit = projection.cache_hit;
      failure.legged_local.active = true;
      return failure;
    }
    legged::LeggedPlanResult result = legged::PlanLegged({
        .start = *state,
        .goals_odom = goals_odom,
        .terrain = &terrain,
        .traversal = legged_projection.value,
        .goal_distance_field = goal_field.value,
        .capability = capability,
        .control = control,
        .search = input.config.search,
    });
    const LeggedLocalDiagnostics legged_diagnostics{
        .active = true,
        .traversal_projection_cache_hit = legged_projection.cache_hit,
        .fast_path_accepts = result.fast_path_accepts,
        .exact_sweep_fallbacks = result.exact_sweep_fallbacks,
        .exact_sweep_cell_checks = result.exact_sweep_cell_checks,
        .edge_validation_cache_hits = result.edge_validation_cache_hits,
    };
    const auto stopped_failure = [&](const std::string& reason_code) {
      LocalStageResult failure = LocalControlledFailure(reason_code);
      failure.snapshot_cache_hit = snapshot.cache_hit;
      failure.projection_cache_hit = projection.cache_hit;
      failure.expanded_states = result.metrics.expanded_states;
      failure.legged_local = legged_diagnostics;
      return failure;
    };
    std::optional<MotionReferenceData> data;
    if (result.ok()) {
      if (const auto stopped = shared::StopReason(control);
          stopped.has_value()) {
        return stopped_failure(std::string{*stopped});
      }
      TrajectoryReference trajectory{
          .semantics = TrajectorySemantics::kLeggedBodyReference,
      };
      trajectory.points.reserve(result.trajectory.size() + 1U);
      if (const auto stopped = shared::StopReason(control);
          stopped.has_value()) {
        return stopped_failure(std::string{*stopped});
      }
      trajectory.points.push_back(
          {.pose = state->body_pose, .velocity = state->body_velocity});
      for (const legged::LeggedTransition& transition : result.trajectory) {
        if (const auto stopped = shared::StopReason(control);
            stopped.has_value()) {
          return stopped_failure(std::string{*stopped});
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
        return stopped_failure(std::string{*stopped});
      }
      data = std::move(trajectory);
    }
    return LocalStageResult{
        .status = result.status,
        .data = std::move(data),
        .reason_code = std::move(result.reason_code),
        .selected_goal_index = result.selected_goal_index,
        .snapshot_cache_hit = snapshot.cache_hit,
        .projection_cache_hit = projection.cache_hit,
        .goal_field_cache_hit = goal_field.cache_hit,
        .expanded_states = result.metrics.expanded_states,
        .best_cost = result.ok() ? std::optional<double>{result.cost}
                                 : std::nullopt,
        .legged_local = legged_diagnostics,
    };
  }

  const GoalRegion& goal_odom = goals_odom.goals_odom.front();
  const auto* capability = std::get_if<HopperCapability>(&input.capability);
  const auto* state = std::get_if<HopperState>(&input.current_state);
  if (capability == nullptr || state == nullptr) {
    return {.status = LocalPlanStatus::kInvalidInput,
            .reason_code = "INVALID_INPUT",
            .snapshot_cache_hit = snapshot.cache_hit,
            .projection_cache_hit = projection.cache_hit};
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
      .snapshot_cache_hit = snapshot.cache_hit,
      .projection_cache_hit = projection.cache_hit,
      .expanded_states = result.metrics.expanded_states,
      .best_cost = result.ok() ? std::optional<double>{result.cost}
                               : std::nullopt,
  };
}

}  // namespace

Planner::Planner() : cache_(std::make_shared<shared::ActivePlannerCache>()) {
  backends_.global =
      [cache = cache_](const PlanningRequest& input, SearchControl control) {
        return hierarchical::PlanSurfaceGlobal(input, std::move(control),
                                               *cache);
      };
  backends_.local =
      [cache = cache_](const PlanningRequest& input,
                       const LocalGoalSet& goals, SearchControl control) {
        return PlanLocalDefault(input, goals, std::move(control), *cache);
      };
}

LocalStageResult Planner::PlanLocal(const PlanningRequest& input,
                                    const LocalGoalSet& goals_odom,
                                    SearchControl control) noexcept {
  try {
    if (!backends_.local) {
      return {.status = LocalPlanStatus::kPlannerError,
              .reason_code = "PLANNER_ERROR"};
    }
    return backends_.local(input, goals_odom, std::move(control));
  } catch (...) {
    return {.status = LocalPlanStatus::kPlannerError,
            .reason_code = "PLANNER_ERROR"};
  }
}

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
  bool global_snapshot_cache_hit{};
  bool global_projection_cache_hit{};
  bool global_route_cache_hit{};
  bool local_snapshot_cache_hit{};
  bool local_projection_cache_hit{};
  bool goal_field_cache_hit{};

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
    const auto apply_cache_hits = [&](PlanningResult& output) {
      output.global_snapshot_cache_hit = global_snapshot_cache_hit;
      output.global_projection_cache_hit = global_projection_cache_hit;
      output.global_route_cache_hit = global_route_cache_hit;
      output.local_snapshot_cache_hit = local_snapshot_cache_hit;
      output.local_projection_cache_hit = local_projection_cache_hit;
      output.goal_field_cache_hit = goal_field_cache_hit;
    };
    apply_cache_hits(result);
    SteadyClock::time_point finished_at{};
    if (!FinalizeTiming(result, started, phase_started, input.control.now, timing,
                        &finished_at)) {
      PlanningResult error =
          Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR");
      error.timing = timing;
      apply_cache_hits(error);
      return error;
    }
    try {
      report_progress(PlannerPhase::kOutput, result.timing.output_elapsed);
    } catch (...) {
      PlanningResult error =
          Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR");
      error.timing = result.timing;
      apply_cache_hits(error);
      return error;
    }
    if (result.status != PlanningStatus::kCanceled &&
        finished_at >= policy.hard_deadline) {
      PlanningResult timeout = Failure(PlanningStatus::kTimedOut, "TIMEOUT");
      timeout.timing = result.timing;
      timeout.expanded_states = result.expanded_states;
      timeout.selected_goal_index = result.selected_goal_index;
      timeout.best_cost = result.best_cost;
      timeout.legged_local = result.legged_local;
      timeout.grid_v1 = result.grid_v1;
      apply_cache_hits(timeout);
      return timeout;
    }
    return result;
  };

  try {
    if (input.control.stop_token.stop_requested()) {
      return finish(Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED"));
    }
    const bool wheel_grid_v1_requested =
        input.config.wheel_planner_mode ==
        WheelPlannerMode::kGridTraversabilityV1;
    const bool use_wheel_grid_v1 =
        wheel_grid_v1_requested &&
        std::holds_alternative<WheeledState>(input.current_state) &&
        std::holds_alternative<WheeledCapability>(input.capability);
    const bool use_legged_grid_v1 =
        input.environment_mode == EnvironmentMode::kLunarSurface &&
        std::holds_alternative<LeggedState>(input.current_state) &&
        std::holds_alternative<LeggedCapability>(input.capability) &&
        input.config.legged_global_mode ==
            LeggedGlobalMode::kGridTraversabilityV1;
    const bool known_mode =
        input.environment_mode == EnvironmentMode::kLunarSurface ||
        input.environment_mode == EnvironmentMode::kLavaTube;
    if (!known_mode || input.request_id.empty() || !FinitePointGoal(input.goal_map) ||
        !MatchingPlatform(input) || !MinimalLocalMapValid(input.world.local_map) ||
        (wheel_grid_v1_requested && !use_wheel_grid_v1) ||
        ((use_wheel_grid_v1 || use_legged_grid_v1) &&
         (!input.world.traversability_snapshot ||
          !input.world.traversability_snapshot->valid()))) {
      return finish(Failure(PlanningStatus::kInvalidInput, "INVALID_INPUT"));
    }
    finish_phase(timing.snapshot_projection_elapsed,
                 PlannerPhase::kSnapshotProjection);

    if (use_wheel_grid_v1) {
      PlanningRequest grid_request = input;
      grid_request.control.deadline = policy.hard_deadline;
      PlanningResult result = grid_v1::Plan(grid_request);
      timing.global_elapsed += result.timing.global_elapsed;
      timing.global_call_count += result.timing.global_call_count;
      timing.local_search_elapsed += result.timing.local_search_elapsed;
      timing.local_elapsed += result.timing.local_elapsed;
      timing.local_call_count += result.timing.local_call_count;
      phase_started = ReadNow(input.control.now);
      report_progress(PlannerPhase::kGlobal, timing.global_elapsed);
      report_progress(PlannerPhase::kLocalSearch,
                      timing.local_search_elapsed);
      result.grid_v1.global_input_sequence =
          input.world.global_map_sequence;
      result.grid_v1.local_input_sequence = input.world.local_map_sequence;
      result.grid_v1.odometry_input_sequence =
          input.world.odometry_sequence;
      finish_phase(timing.certification_elapsed,
                   PlannerPhase::kCertification);
      return finish(std::move(result));
    }

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
      global_snapshot_cache_hit = global.snapshot_cache_hit;
      global_projection_cache_hit = global.projection_cache_hit;
      global_route_cache_hit = global.route_cache_hit;
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
    local_snapshot_cache_hit = local.snapshot_cache_hit;
    local_projection_cache_hit = local.projection_cache_hit;
    goal_field_cache_hit = local.goal_field_cache_hit;
    const auto with_local_evidence = [&](PlanningResult result) {
      result.expanded_states =
          (global_route.has_value() ? global_route->expanded_states : 0U) +
          local.expanded_states;
      result.selected_goal_index = local.selected_goal_index;
      result.best_cost = local.best_cost;
      result.legged_local = local.legged_local;
      return result;
    };
    report_progress(PlannerPhase::kLocalSearch, timing.local_search_elapsed);
    phase_started = local_finished;
    if (local.status == LocalPlanStatus::kCanceled ||
        input.control.stop_token.stop_requested()) {
      return finish(with_local_evidence(
          Failure(PlanningStatus::kCanceled, "REQUEST_CANCELED")));
    }
    if (local.status != LocalPlanStatus::kSolved &&
        local_finished >= policy.hard_deadline) {
      return finish(with_local_evidence(
          Failure(PlanningStatus::kTimedOut, "TIMEOUT")));
    }
    if (local.status != LocalPlanStatus::kSolved) {
      return finish(with_local_evidence(LocalFailure(local.status)));
    }
    if (!local.data.has_value()) {
      return finish(with_local_evidence(
          Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR")));
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
      return finish(with_local_evidence(GlobalFailure(composed.reason_code)));
    }
    finish_phase(timing.certification_elapsed, PlannerPhase::kCertification);
    PlanningResult success{
        .status = PlanningStatus::kSuccess,
        .reason_code = {},
        .reference = std::move(composed.reference),
        .expanded_states =
            (global_route.has_value() ? global_route->expanded_states : 0U) +
            local.expanded_states,
        .selected_goal_index = local.selected_goal_index,
        .best_cost = local.best_cost,
        .legged_local = local.legged_local,
    };
    return finish(std::move(success));
  } catch (...) {
    return finish(Failure(PlanningStatus::kPlannerError, "PLANNER_ERROR"));
  }
}

}  // namespace lunar::pure_planning
