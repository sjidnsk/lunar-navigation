#include <new>
#include <stop_token>
#include <utility>

#include "lunar_planner_training_bridge/request.hpp"

namespace lunar::planning::training {
namespace {

[[nodiscard]] PlannerOutput BridgeFailure(const PlanningOutcome outcome,
                                          const char *reason_code) noexcept {
  PlannerOutput output;
  output.outcome = outcome;
  output.directive = ExecutionDirective::kNoSafeReference;
  output.reason_code = reason_code;
  output.diagnostics.planner_name = "cpp_v3_hierarchical";
  return output;
}

}  // namespace

PlannerOutput PlannerBridge::Plan(const TrainingPlanRequest &request) noexcept {
  try {
    PlannerOutput output = planner_.Plan(PlannerInput{
        .request_id = request.request_id,
        .mission_id = request.mission_id,
        .mission_revision = request.mission_revision,
        .platform_id = request.platform_id,
        .capability_version = request.capability_version,
        .global_map_generation = request.global_map_generation,
        .local_map_generation = request.local_map_generation,
        .map_from_odom_generation = request.map_from_odom_generation,
        .state_time = request.state_time,
        .current_state = request.current_state,
        .hopper_propellant = request.hopper_propellant,
        .goal_map = request.goal,
        .world = request.world,
        .capability = request.capability,
        .config = request.config,
        .previous_execution = request.previous_execution,
        .stop_token = std::stop_token{},
        .position_uncertainty_m = request.position_uncertainty_m,
        .velocity_uncertainty_mps = request.velocity_uncertainty_mps,
    });
    return output;
  } catch (const std::bad_alloc &) {
    return BridgeFailure(PlanningOutcome::kResourceExhausted,
                         "BRIDGE_RESOURCE_EXHAUSTED");
  } catch (...) {
    return BridgeFailure(PlanningOutcome::kNumericalFailure,
                         "BRIDGE_REQUEST_CONVERSION_FAILED");
  }
}

}  // namespace lunar::planning::training
