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

[[nodiscard]] PlannerInput ToPlannerInput(
    const TrainingPlanRequest &request) {
  return PlannerInput{
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
      .goal_map = request.goal,
      .world = request.world,
      .capability = request.capability,
      .config = request.config,
      .previous_execution = request.previous_execution,
      .stop_token = std::stop_token{},
      .position_uncertainty_m = request.position_uncertainty_m,
      .velocity_uncertainty_mps = request.velocity_uncertainty_mps,
  };
}

}  // namespace

PlannerOutput PlannerBridge::Plan(const TrainingPlanRequest &request) noexcept {
  try {
    PlannerOutput output = planner_.Plan(ToPlannerInput(request));
    return output;
  } catch (const std::bad_alloc &) {
    return BridgeFailure(PlanningOutcome::kResourceExhausted,
                         "BRIDGE_RESOURCE_EXHAUSTED");
  } catch (...) {
    return BridgeFailure(PlanningOutcome::kNumericalFailure,
                         "BRIDGE_REQUEST_CONVERSION_FAILED");
  }
}

ReachabilityProjectionResult PlannerBridge::ProjectReachability(
    const TrainingPlanRequest &request,
    const double maximum_edge_distance_m) const noexcept {
  try {
    return lunar::planning::ProjectReachability(
        ToPlannerInput(request), maximum_edge_distance_m);
  } catch (const std::bad_alloc &) {
    return ReachabilityProjectionResult{
        .projection = std::nullopt,
        .reason_code = "REACHABILITY_RESOURCE_EXHAUSTED",
    };
  } catch (...) {
    return ReachabilityProjectionResult{
        .projection = std::nullopt,
        .reason_code = "BRIDGE_REQUEST_CONVERSION_FAILED",
    };
  }
}

ReachabilityProjectionResult PlannerBridge::ProjectReachability(
    const TrainingPlanRequest &request,
    const double maximum_edge_distance_m,
    const HopperLandingEvidenceGrid &evidence) const noexcept {
  try {
    return lunar::planning::ProjectReachability(
        ToPlannerInput(request), maximum_edge_distance_m, evidence);
  } catch (const std::bad_alloc &) {
    return ReachabilityProjectionResult{
        .projection = std::nullopt,
        .reason_code = "REACHABILITY_RESOURCE_EXHAUSTED",
    };
  } catch (...) {
    return ReachabilityProjectionResult{
        .projection = std::nullopt,
        .reason_code = "BRIDGE_REQUEST_CONVERSION_FAILED",
    };
  }
}

ReachabilityProjectionResult PlannerBridge::ProjectDirectHopperReachability(
    const TrainingPlanRequest &request,
    const double maximum_edge_distance_m,
    const HopperLandingEvidenceGrid &evidence) const noexcept {
  try {
    return lunar::planning::ProjectDirectHopperReachability(
        ToPlannerInput(request), maximum_edge_distance_m, evidence);
  } catch (const std::bad_alloc &) {
    return ReachabilityProjectionResult{
        .projection = std::nullopt,
        .reason_code = "REACHABILITY_RESOURCE_EXHAUSTED",
    };
  } catch (...) {
    return ReachabilityProjectionResult{
        .projection = std::nullopt,
        .reason_code = "BRIDGE_REQUEST_CONVERSION_FAILED",
    };
  }
}

HopperLandingEvidenceProjectionResult
PlannerBridge::ProjectHopperLandingEvidence(
    const TrainingPlanRequest &request,
    const std::vector<Vec3> &target_positions_map) const noexcept {
  try {
    return lunar::planning::ProjectHopperLandingEvidence(
        ToPlannerInput(request), target_positions_map);
  } catch (const std::bad_alloc &) {
    return HopperLandingEvidenceProjectionResult{
        .projection = std::nullopt,
        .reason_code = "REACHABILITY_RESOURCE_EXHAUSTED",
    };
  } catch (...) {
    return HopperLandingEvidenceProjectionResult{
        .projection = std::nullopt,
        .reason_code = "BRIDGE_REQUEST_CONVERSION_FAILED",
    };
  }
}

PrimitiveReachabilityResult TrainingPrimitiveReachabilityEngine::Update(
    const TrainingPlanRequest& request,
    const std::optional<double> maximum_action_distance_m) noexcept {
  try {
    return engine_.Update(
        ToPlannerInput(request), maximum_action_distance_m);
  } catch (const std::bad_alloc&) {
    return PrimitiveReachabilityResult{
        .snapshot = std::nullopt,
        .reason_code = "REACHABILITY_RESOURCE_EXHAUSTED",
    };
  } catch (...) {
    return PrimitiveReachabilityResult{
        .snapshot = std::nullopt,
        .reason_code = "BRIDGE_REQUEST_CONVERSION_FAILED",
    };
  }
}

PrimitiveReachabilityResult TrainingPrimitiveReachabilityEngine::Update(
    const TrainingPlanRequest& request,
    const std::optional<double> maximum_action_distance_m,
    const HopperLandingEvidenceGrid& hopper_landing_evidence) noexcept {
  try {
    return engine_.Update(
        ToPlannerInput(request), maximum_action_distance_m,
        hopper_landing_evidence);
  } catch (const std::bad_alloc&) {
    return PrimitiveReachabilityResult{
        .snapshot = std::nullopt,
        .reason_code = "REACHABILITY_RESOURCE_EXHAUSTED",
    };
  } catch (...) {
    return PrimitiveReachabilityResult{
        .snapshot = std::nullopt,
        .reason_code = "BRIDGE_REQUEST_CONVERSION_FAILED",
    };
  }
}

void TrainingPrimitiveReachabilityEngine::Reset() noexcept {
  engine_.Reset();
}

}  // namespace lunar::planning::training
