#include "lunar_planner_core/planner.hpp"

#include <array>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
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

[[nodiscard]] HierarchicalPlannerMetrics
GroundMetrics(const PlannerInput &input,
              const hierarchical::GlobalRoutePlanResult &global,
              const hierarchical::LocalFrontierResult *frontiers,
              const std::uint64_t local_expanded_states,
              const std::size_t local_attempts,
              const double frontier_distance_m,
              const std::chrono::nanoseconds local_elapsed) {
  const hierarchical::GlobalRoute *route =
      global.route.has_value() ? &*global.route : nullptr;
  return HierarchicalPlannerMetrics{
      .global_level = global.global_level.value_or(0U),
      .global_resolution_m = input.world.global_map.resolution_m,
      .global_cells = input.world.global_map.CellCount(),
      .global_elapsed = global.elapsed,
      .local_elapsed = local_elapsed,
      .global_expanded_states = route == nullptr ? 0U : route->expanded_states,
      .local_expanded_states = local_expanded_states,
      .global_open_peak = route == nullptr ? 0U : route->open_peak,
      .estimated_work_memory_bytes =
          route == nullptr ? 0U : route->estimated_work_memory_bytes,
      .raw_route_points = route == nullptr ? 0U : route->raw_cells.size(),
      .simplified_route_points =
          route == nullptr ? 0U : route->simplified_cells.size(),
      .local_frontier_distance_m = frontier_distance_m,
      .local_attempts = local_attempts,
      .corridor_width_m =
          frontiers == nullptr ? 0.0 : 2.0 * frontiers->corridor_half_width_m,
  };
}

} // namespace

struct Planner::Impl final {
  hopper::HopperPlanner hopper_planner;
  legged::LeggedPlanner legged_planner;
  wheel::WheelPlanner wheel_planner;
};

Planner::Planner() : impl_(std::make_unique<Impl>()) {}

Planner::~Planner() = default;

Planner::Planner(Planner &&) noexcept = default;

Planner &Planner::operator=(Planner &&) noexcept = default;

PlannerOutput Planner::Plan(const PlannerInput &input) noexcept {
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

    const hierarchical::GlobalRoutePlanResult global =
        hierarchical::PlanGroundGlobalRoute(input);
    if (!global.ok()) {
      return Failure(global.outcome, FailureDirective(global.outcome),
                     global.reason_code, started, 0U, std::nullopt, {},
                     GroundMetrics(input, global, nullptr, 0U, 0U, 0.0, {}));
    }
    const hierarchical::LocalFrontierResult frontiers =
        hierarchical::BuildLocalFrontiers(input, *global.route);
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
      return Failure(outcome, FailureDirective(outcome), frontiers.reason_code,
                     started, global.route->expanded_states, global.route->cost,
                     {},
                     GroundMetrics(input, global, &frontiers, 0U, 0U, 0.0, {}));
    }

    const auto local_started = std::chrono::steady_clock::now();
    std::uint64_t local_expanded = 0U;
    for (std::size_t attempt = 0U; attempt < frontiers.problems.size();
         ++attempt) {
      PlannerOutput local =
          platform == PlatformType::kWheeled
              ? impl_->wheel_planner.Plan(frontiers.problems[attempt])
              : impl_->legged_planner.Plan(frontiers.problems[attempt]);
      local_expanded += local.diagnostics.expanded_states;
      const std::size_t attempts = attempt + 1U;
      const auto local_elapsed =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - local_started);
      if (local.outcome == PlanningOutcome::kNewReferenceAvailable ||
          local.outcome == PlanningOutcome::kSafeFrontierReferenceAvailable) {
        const PlanningOutcome outcome = local.outcome;
        const ExecutionDirective directive = local.directive;
        std::string reason_code = local.reason_code;
        std::vector<std::string> warnings = local.diagnostics.warning_codes;
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
              GroundMetrics(input, global, &frontiers, local_expanded, attempts,
                            frontiers.frontier_distances_m[attempt],
                            local_elapsed));
        }
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
                        input, global, &frontiers, local_expanded, attempts,
                        frontiers.frontier_distances_m[attempt], local_elapsed),
                },
        };
      }
      if (local.outcome == PlanningOutcome::kCanceled ||
          local.outcome == PlanningOutcome::kResourceExhausted ||
          local.outcome == PlanningOutcome::kInvalidRequest ||
          local.outcome == PlanningOutcome::kNumericalFailure ||
          local.outcome == PlanningOutcome::kActiveReferenceInvalidated) {
        return Failure(local.outcome, local.directive, local.reason_code,
                       started, global.route->expanded_states + local_expanded,
                       global.route->cost, local.diagnostics.warning_codes,
                       GroundMetrics(input, global, &frontiers, local_expanded,
                                     attempts,
                                     frontiers.frontier_distances_m[attempt],
                                     local_elapsed));
      }
    }
    const auto local_elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - local_started);
    return Failure(
        PlanningOutcome::kNoKnownSafeRoute,
        ExecutionDirective::kNoSafeReference, "LOCAL_SEGMENT_INFEASIBLE",
        started, global.route->expanded_states + local_expanded,
        global.route->cost, {},
        GroundMetrics(input, global, &frontiers, local_expanded,
                      frontiers.problems.size(),
                      frontiers.frontier_distances_m.back(), local_elapsed));
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
