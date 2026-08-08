#include "lunar_planner_core/planner.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hierarchical/global_route_planner.hpp"
#include "hierarchical/local_frontier.hpp"
#include "hierarchical/map_level.hpp"
#include "hierarchical/reference_composer.hpp"
#include "hierarchical/route_continuation.hpp"
#include "hopper/commitment_state_machine.hpp"
#include "hopper/hopper_planner.hpp"
#include "legged/legged_planner.hpp"
#include "wheel/wheel_planner.hpp"

namespace lunar::planning {
namespace {

constexpr std::array<std::string_view, 10> kRequiredMapLayers{
    "elevation",          "valid_mask",        "obstacle",
    "obstacle_height",    "observation_age_s", "observation_quality",
    "elevation_variance", "obstacle_variance", "observation_count",
    "forbidden",
};
constexpr std::string_view kPlannerName = "cpp_v3_hierarchical";

[[nodiscard]] PlannerOutput
Failure(const PlanningOutcome outcome, const ExecutionDirective directive,
        std::string reason_code,
        const std::chrono::steady_clock::time_point started,
        const std::uint64_t expanded_states = 0U,
        std::optional<double> best_cost = std::nullopt,
        std::vector<std::string> warnings = {},
        std::optional<HierarchicalPlannerMetrics> metrics = std::nullopt) {
  return PlannerOutput{
      .outcome = outcome,
      .directive = directive,
      .reason_code = std::move(reason_code),
      .reference = std::nullopt,
      .diagnostics =
          PlannerDiagnostics{
              .planner_name = std::string{kPlannerName},
              .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - started),
              .expanded_states = expanded_states,
              .best_cost = best_cost,
              .warning_codes = std::move(warnings),
              .hierarchical = std::move(metrics),
          },
  };
}

[[nodiscard]] std::string MissingLayerReason(const std::string_view layer) {
  std::string reason{"MISSING_MAP_LAYER_"};
  reason.reserve(reason.size() + layer.size());
  for (const char character : layer) {
    reason.push_back(
        static_cast<char>(std::toupper(static_cast<unsigned char>(character))));
  }
  return reason;
}

[[nodiscard]] std::string MissingRequiredLayer(const WorldSnapshot &world) {
  for (const GridMap *map : {&world.global_map, &world.local_map}) {
    for (const std::string_view layer : kRequiredMapLayers) {
      if (!map->HasLayer(layer)) {
        return MissingLayerReason(layer);
      }
    }
  }
  return {};
}

[[nodiscard]] ExecutionDirective
FailureDirective(const PlanningOutcome outcome) noexcept {
  if (outcome == PlanningOutcome::kCanceled ||
      outcome == PlanningOutcome::kGoalInfeasible) {
    return ExecutionDirective::kHoldPosition;
  }
  return ExecutionDirective::kNoSafeReference;
}

struct GroundAccumulation final {
  std::chrono::nanoseconds global_elapsed{};
  std::chrono::nanoseconds local_elapsed{};
  std::uint64_t local_expanded_states{};
  std::size_t local_frontier_attempts{};
  std::size_t local_search_runs{};
  std::size_t global_replans{};
  std::size_t global_projection_cache_hits{};
  std::size_t local_projection_cache_hits{};
};

[[nodiscard]] HierarchicalPlannerMetrics GroundMetrics(
    const PlannerInput &input,
    const hierarchical::GlobalRoutePlanResult &global,
    const GroundAccumulation &accumulated,
    const hierarchical::LocalFrontierResult *frontiers,
    const std::uint64_t, const std::size_t,
    const double frontier_distance_m,
    const bool route_reused = false, const std::size_t route_cursor = 0U,
    const std::uint64_t rolling_request_count = 1U) {
  const hierarchical::GlobalRoute *route =
      global.route.has_value() ? &*global.route : nullptr;
  return HierarchicalPlannerMetrics{
      .global_level = global.global_level.value_or(0U),
      .global_resolution_m = input.world.global_map.resolution_m,
      .global_cells = input.world.global_map.CellCount(),
      .global_elapsed = accumulated.global_elapsed,
      .local_elapsed = accumulated.local_elapsed,
      .global_expanded_states = route == nullptr ? 0U : route->expanded_states,
      .local_expanded_states = accumulated.local_expanded_states,
      .global_open_peak = route == nullptr ? 0U : route->open_peak,
      .estimated_work_memory_bytes =
          route == nullptr ? 0U : route->estimated_work_memory_bytes,
      .raw_route_points = route == nullptr ? 0U : route->raw_cells.size(),
      .simplified_route_points =
          route == nullptr ? 0U : route->simplified_cells.size(),
      .local_frontier_distance_m = frontier_distance_m,
      .local_attempts = accumulated.local_frontier_attempts,
      .local_search_runs = accumulated.local_search_runs,
      .global_replans = accumulated.global_replans,
      .global_projection_cache_hits =
          accumulated.global_projection_cache_hits,
      .local_projection_cache_hits =
          accumulated.local_projection_cache_hits,
      .corridor_width_m =
          frontiers == nullptr ? 0.0 : 2.0 * frontiers->corridor_half_width_m,
      .route_reused = route_reused,
      .route_cursor = route_cursor,
      .rolling_request_count = rolling_request_count,
  };
}

[[nodiscard]] std::string GroundRouteId(const PlatformType platform,
                                        const std::string &request_id) {
  return (platform == PlatformType::kWheeled ? "wheel-route/"
                                             : "legged-route/") +
         request_id;
}

[[nodiscard]] bool AppendNewConditionalCorridorCells(
    const hierarchical::LocalFrontierResult &frontiers,
    std::vector<shared::GridCell> &excluded_cells) {
  bool added = false;
  for (const shared::GridCell cell : frontiers.conditional_corridor_cells) {
    if (std::ranges::find(excluded_cells, cell) == excluded_cells.end()) {
      excluded_cells.push_back(cell);
      added = true;
    }
  }
  return added;
}

void ApplyValidatedGlobalMapLevel(
    PlannerOutput &output, const PlannerInput &input,
    const hierarchical::MapLevelValidationResult &levels) {
  if (!output.diagnostics.hierarchical.has_value()) {
    output.diagnostics.hierarchical.emplace();
  }
  HierarchicalPlannerMetrics &metrics = *output.diagnostics.hierarchical;
  metrics.global_level = levels.global_level.value_or(0U);
  metrics.global_resolution_m = input.world.global_map.resolution_m;
  metrics.global_cells = input.world.global_map.CellCount();
}

} // namespace

struct Planner::Impl final {
  shared::ProjectionCache global_projection_cache;
  hopper::HopperPlanner hopper_planner;
  legged::LeggedPlanner legged_planner;
  wheel::WheelPlanner wheel_planner;
};

Planner::Planner() : impl_(std::make_unique<Impl>()) {}

Planner::~Planner() = default;

Planner::Planner(Planner &&) noexcept = default;

Planner &Planner::operator=(Planner &&) noexcept = default;

PlannerOutput Planner::Plan(const PlannerInput &input) noexcept {
  return Plan(input, {});
}

PlannerOutput Planner::Plan(
    const PlannerInput &input,
    const ProvisionalRouteObserver &provisional_route_observer) noexcept {
  const auto started = std::chrono::steady_clock::now();
  try {
    if (impl_ == nullptr) {
      return Failure(PlanningOutcome::kInvalidRequest,
                     ExecutionDirective::kNoSafeReference, "PLANNER_MOVED_FROM",
                     started);
    }
    const PlatformType platform = CapabilityPlatform(input.capability);
    if (platform == PlatformType::kHopper) {
      const hopper::PlanningProtection protection =
          hopper::EvaluatePlanningProtection(input.previous_execution);
      if (protection.decision != hopper::PlanningProtectionDecision::kMayPlan) {
        PlannerOutput output = impl_->hopper_planner.Plan(input);
        output.diagnostics.planner_name = std::string{kPlannerName};
        output.diagnostics.elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started);
        return output;
      }
    }
    if (input.stop_token.stop_requested()) {
      return Failure(PlanningOutcome::kCanceled,
                     ExecutionDirective::kHoldPosition, "REQUEST_CANCELED",
                     started);
    }
    if (std::string reason = MissingRequiredLayer(input.world);
        !reason.empty()) {
      return Failure(PlanningOutcome::kInvalidRequest,
                     ExecutionDirective::kNoSafeReference, std::move(reason),
                     started);
    }
    const hierarchical::MapLevelValidationResult levels =
        hierarchical::ValidateMapLevels(input.world, input.config.global_map);
    if (!levels.ok()) {
      const PlanningOutcome outcome =
          levels.reason_code == "GLOBAL_MAP_SCALE_UNSUPPORTED"
              ? PlanningOutcome::kResourceExhausted
              : PlanningOutcome::kInvalidRequest;
      return Failure(outcome, FailureDirective(outcome), levels.reason_code,
                     started);
    }
    if (platform == PlatformType::kHopper) {
      PlannerOutput output = impl_->hopper_planner.Plan(input);
      output.diagnostics.planner_name = std::string{kPlannerName};
      output.diagnostics.elapsed =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started);
      ApplyValidatedGlobalMapLevel(output, input, levels);
      if (output.reason_code == "HOPPER_GLOBAL_RESOLUTION_INSUFFICIENT") {
        output.outcome = PlanningOutcome::kResourceExhausted;
        output.directive = ExecutionDirective::kNoSafeReference;
      }
      return output;
    }
    if (platform != PlatformType::kWheeled &&
        platform != PlatformType::kLegged) {
      return Failure(PlanningOutcome::kInvalidRequest,
                     ExecutionDirective::kNoSafeReference,
                     "PLANNER_BACKEND_NOT_CONFIGURED", started);
    }

    bool route_reused = false;
    std::size_t route_cursor = 0U;
    std::uint64_t rolling_request_count = 1U;
    GroundAccumulation accumulated;
    hierarchical::GlobalRoutePlanResult global;
    if (input.continuation != nullptr) {
      hierarchical::GroundRouteReuseResult reused =
          hierarchical::TryReuseGroundRoute(input, *input.continuation);
      if (reused.ok()) {
        route_reused = true;
        route_cursor = reused.route_cursor;
        rolling_request_count =
            input.continuation->rolling_request_count() + 1U;
        global = hierarchical::GlobalRoutePlanResult{
            .outcome = PlanningOutcome::kNewReferenceAvailable,
            .reason_code = "GLOBAL_ROUTE_AVAILABLE",
            .route = std::move(reused.route),
            .global_level = levels.global_level,
            .elapsed = {},
        };
      }
    }
    if (!route_reused) {
      global = hierarchical::PlanGroundGlobalRoute(
          input, {}, &impl_->global_projection_cache);
      accumulated.global_elapsed += global.elapsed;
      accumulated.global_projection_cache_hits +=
          global.projection_cache_hit ? 1U : 0U;
    }
    std::vector<shared::GridCell> excluded_conditional_cells;
    std::vector<std::string> conditional_retry_warnings;
    for (;;) {
      if (!global.ok()) {
        return Failure(global.outcome, FailureDirective(global.outcome),
                       global.reason_code, started, 0U, std::nullopt, {},
                       GroundMetrics(
                           input, global, accumulated, nullptr, 0U, 0U, 0.0));
      }
      if (provisional_route_observer) {
        try {
          provisional_route_observer(ProvisionalGlobalRoute{
              .request_id = input.request_id,
              .route_id = route_reused && input.continuation != nullptr
                  ? input.continuation->route_id()
                  : GroundRouteId(platform, input.request_id),
              .platform_type = platform,
              .poses_map = global.route->poses_map,
          });
        } catch (...) {
          // Visualization observers are non-authoritative and must never
          // change a planning result.
        }
      }
      hierarchical::LocalFrontierResult frontiers =
          hierarchical::BuildLocalFrontiers(input, *global.route);
      if (!frontiers.ok() && route_reused) {
        global = hierarchical::PlanGroundGlobalRoute(
            input, {}, &impl_->global_projection_cache);
        accumulated.global_elapsed += global.elapsed;
        accumulated.global_projection_cache_hits +=
            global.projection_cache_hit ? 1U : 0U;
        route_reused = false;
        route_cursor = 0U;
        rolling_request_count = 1U;
        if (global.ok()) {
          frontiers = hierarchical::BuildLocalFrontiers(input, *global.route);
        }
      }
      if (!global.ok()) {
        return Failure(global.outcome, FailureDirective(global.outcome),
                       global.reason_code, started, 0U, std::nullopt, {},
                       GroundMetrics(
                           input, global, accumulated, nullptr, 0U, 0U, 0.0,
                           route_reused, route_cursor,
                           rolling_request_count));
      }
      if (!frontiers.ok()) {
        PlanningOutcome outcome = PlanningOutcome::kInvalidRequest;
        if (frontiers.status ==
            hierarchical::LocalFrontierStatus::kCoverageInsufficient) {
          outcome = PlanningOutcome::kNoKnownSafeRoute;
        } else if (frontiers.status ==
                   hierarchical::LocalFrontierStatus::kGoalInfeasible) {
          outcome = PlanningOutcome::kGoalInfeasible;
        } else if (frontiers.status ==
                   hierarchical::LocalFrontierStatus::kCanceled) {
          outcome = PlanningOutcome::kCanceled;
        }
        return Failure(
            outcome, FailureDirective(outcome), frontiers.reason_code, started,
            global.route->expanded_states, global.route->cost, {},
            GroundMetrics(
                input, global, accumulated, &frontiers, 0U, 0U, 0.0));
      }

      std::uint64_t local_expanded = 0U;
      std::optional<std::string> specific_local_failure;
      std::vector<std::string> local_failure_reasons;
      bool retry_global_route = false;
      std::optional<wheel::WheelRankedPlanResult> ranked_wheel;
      if (platform == PlatformType::kWheeled) {
        const auto search_started = std::chrono::steady_clock::now();
        ranked_wheel = impl_->wheel_planner.PlanRanked(frontiers.problems);
        accumulated.local_elapsed +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - search_started);
        ++accumulated.local_search_runs;
        accumulated.local_projection_cache_hits +=
            ranked_wheel->projection_cache_hit ? 1U : 0U;
      }
      const std::size_t local_iterations =
          platform == PlatformType::kWheeled
              ? 1U
              : frontiers.problems.size();
      for (std::size_t iteration = 0U; iteration < local_iterations;
           ++iteration) {
        std::size_t attempt = iteration;
        PlannerOutput local;
        if (platform == PlatformType::kWheeled) {
          attempt = ranked_wheel->selected_problem_index.value_or(
              frontiers.problems.size() - 1U);
          local = std::move(ranked_wheel->output);
        } else {
          const auto search_started = std::chrono::steady_clock::now();
          local = impl_->legged_planner.Plan(frontiers.problems[attempt]);
          accumulated.local_elapsed +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - search_started);
          ++accumulated.local_search_runs;
        }
        local_expanded += local.diagnostics.expanded_states;
        accumulated.local_expanded_states +=
            local.diagnostics.expanded_states;
        const std::size_t attempts = attempt + 1U;
        accumulated.local_frontier_attempts +=
            platform == PlatformType::kWheeled ? attempts : 1U;
        if (local.outcome == PlanningOutcome::kNewReferenceAvailable ||
            local.outcome == PlanningOutcome::kSafeFrontierReferenceAvailable) {
          if (attempt > 0U && AppendNewConditionalCorridorCells(
                                  frontiers, excluded_conditional_cells)) {
            route_reused = false;
            route_cursor = 0U;
            rolling_request_count = 1U;
            if (conditional_retry_warnings.empty()) {
              conditional_retry_warnings.emplace_back(
                  "GLOBAL_CONDITIONAL_CORRIDOR_RETRY");
            }
            global = hierarchical::PlanGroundGlobalRoute(
                input, excluded_conditional_cells,
                &impl_->global_projection_cache);
            accumulated.global_elapsed += global.elapsed;
            accumulated.global_projection_cache_hits +=
                global.projection_cache_hit ? 1U : 0U;
            ++accumulated.global_replans;
            retry_global_route = true;
            break;
          }
          const PlanningOutcome outcome = local.outcome;
          const ExecutionDirective directive = local.directive;
          std::string reason_code = local.reason_code;
          std::vector<std::string> warnings = local.diagnostics.warning_codes;
          warnings.insert(warnings.end(), conditional_retry_warnings.begin(),
                          conditional_retry_warnings.end());
          std::optional<LocalTrajectoryDiagnostics> local_trajectory =
              local.diagnostics.local_trajectory;
          if (attempt > 0U) {
            warnings.emplace_back("LOCAL_FRONTIER_BACKOFF");
          }
          hierarchical::ReferenceComposeResult composed =
              hierarchical::ComposeReference(input, *global.route,
                                             std::move(local));
          if (!composed.ok()) {
            return Failure(
                PlanningOutcome::kNumericalFailure,
                ExecutionDirective::kNoSafeReference, composed.reason_code,
                started, global.route->expanded_states + local_expanded,
                global.route->cost, std::move(warnings),
                GroundMetrics(
                    input, global, accumulated, &frontiers, local_expanded,
                    attempts, frontiers.frontier_distances_m[attempt]));
          }
          const std::string reference_plan_id = composed.reference->plan_id;
          const std::string route_id =
              route_reused ? input.continuation->route_id()
                           : GroundRouteId(platform, input.request_id);
          const hierarchical::GlobalRoute &continuation_route =
              route_reused ? input.continuation->global_route() : *global.route;
          auto continuation = std::make_shared<const RouteContinuation>(
              route_id, reference_plan_id, input, continuation_route,
              std::vector<CertifiedHopPreview>{}, route_cursor,
              frontiers.corridor_half_width_m, rolling_request_count);
          return PlannerOutput{
              .outcome = outcome,
              .directive = directive,
              .reason_code = std::move(reason_code),
              .reference = std::move(composed.reference),
              .diagnostics =
                  PlannerDiagnostics{
                      .planner_name = std::string{kPlannerName},
                      .elapsed =
                          std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - started),
                      .expanded_states =
                          global.route->expanded_states + local_expanded,
                      .best_cost = global.route->cost,
                      .warning_codes = std::move(warnings),
                      .hierarchical = GroundMetrics(
                          input, global, accumulated, &frontiers,
                          local_expanded, attempts,
                          frontiers.frontier_distances_m[attempt], route_reused,
                          route_cursor,
                          rolling_request_count),
                      .local_trajectory = std::move(local_trajectory),
                  },
              .continuation = std::move(continuation),
          };
        }
        if (local.outcome == PlanningOutcome::kCanceled ||
            local.outcome == PlanningOutcome::kResourceExhausted ||
            local.outcome == PlanningOutcome::kInvalidRequest ||
            local.outcome == PlanningOutcome::kNumericalFailure ||
            local.outcome == PlanningOutcome::kActiveReferenceInvalidated) {
          return Failure(
              local.outcome, local.directive, local.reason_code, started,
              global.route->expanded_states + local_expanded,
              global.route->cost, local.diagnostics.warning_codes,
              GroundMetrics(
                  input, global, accumulated, &frontiers, local_expanded,
                  attempts, frontiers.frontier_distances_m[attempt]));
        }
        if (!local.reason_code.empty() &&
            std::find(local_failure_reasons.begin(),
                      local_failure_reasons.end(),
                      local.reason_code) == local_failure_reasons.end()) {
          local_failure_reasons.push_back(local.reason_code);
        }
        if (local.reason_code == "WHEEL_START_CONNECTOR_INFEASIBLE" ||
            local.reason_code == "LEGGED_START_CONNECTOR_INFEASIBLE" ||
            local.reason_code == "WHEEL_SMOOTHED_EXECUTION_REQUIRED" ||
            local.reason_code == "LEGGED_SMOOTHED_EXECUTION_REQUIRED") {
          specific_local_failure = local.reason_code;
        }
      }
      if (retry_global_route) {
        continue;
      }
      if (AppendNewConditionalCorridorCells(frontiers,
                                            excluded_conditional_cells)) {
        route_reused = false;
        route_cursor = 0U;
        rolling_request_count = 1U;
        if (conditional_retry_warnings.empty()) {
          conditional_retry_warnings.emplace_back(
              "GLOBAL_CONDITIONAL_CORRIDOR_RETRY");
        }
        global = hierarchical::PlanGroundGlobalRoute(
            input, excluded_conditional_cells,
            &impl_->global_projection_cache);
        accumulated.global_elapsed += global.elapsed;
        accumulated.global_projection_cache_hits +=
            global.projection_cache_hit ? 1U : 0U;
        ++accumulated.global_replans;
        continue;
      }
      local_failure_reasons.insert(local_failure_reasons.end(),
                                   conditional_retry_warnings.begin(),
                                   conditional_retry_warnings.end());
      return Failure(
          PlanningOutcome::kNoKnownSafeRoute,
          ExecutionDirective::kNoSafeReference,
          specific_local_failure.value_or("LOCAL_SEGMENT_INFEASIBLE"), started,
          global.route->expanded_states + local_expanded, global.route->cost,
          std::move(local_failure_reasons),
          GroundMetrics(
              input, global, accumulated, &frontiers, local_expanded,
              frontiers.problems.size(),
              frontiers.frontier_distances_m.back()));
    }
  } catch (const std::bad_alloc &) {
    return Failure(PlanningOutcome::kResourceExhausted,
                   ExecutionDirective::kNoSafeReference, "RESOURCE_EXHAUSTED",
                   started);
  } catch (const std::exception &) {
    return Failure(PlanningOutcome::kNumericalFailure,
                   ExecutionDirective::kNoSafeReference,
                   "INTERNAL_PLANNER_EXCEPTION", started);
  } catch (...) {
    return Failure(PlanningOutcome::kNumericalFailure,
                   ExecutionDirective::kNoSafeReference,
                   "INTERNAL_PLANNER_EXCEPTION", started);
  }
}

} // namespace lunar::planning
