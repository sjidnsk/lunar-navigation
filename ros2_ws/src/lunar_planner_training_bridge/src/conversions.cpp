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
  output.diagnostics.planner_name = "cpp_v3";
  return output;
}

}  // namespace

PlannerOutput PlannerBridge::Plan(const TrainingPlanRequest &request) noexcept {
  try {
    PlannerOutput output = planner_.Plan(PlannerInput{
        .request_id = request.request_id,
        .state_time = request.state_time,
        .current_state = request.current_state,
        .goal = request.goal,
        .world = request.world,
        .capability = request.capability,
        .config = request.config,
        .previous_execution = request.previous_execution,
        .stop_token = std::stop_token{},
    });
    output.diagnostics.planner_name = "cpp_v3";
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
