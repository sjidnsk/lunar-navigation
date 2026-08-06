#include "hopper/commitment_state_machine.hpp"

#include <utility>

namespace lunar::planning::hopper {
namespace {

[[nodiscard]] bool NonEmpty(
    const std::optional<std::string>& value) noexcept {
  return value.has_value() && !value->empty();
}

[[nodiscard]] bool HasBothIds(const CommitmentEvent& event) noexcept {
  return NonEmpty(event.plan_id) && NonEmpty(event.segment_id);
}

[[nodiscard]] bool Matches(
    const HopperExecutionContext& current,
    const CommitmentEvent& event,
    const bool require_plan) noexcept {
  const bool segment_matches = NonEmpty(current.active_segment_id) &&
      NonEmpty(event.segment_id) &&
      current.active_segment_id == event.segment_id;
  const bool plan_matches = !require_plan ||
      (NonEmpty(current.active_plan_id) && NonEmpty(event.plan_id) &&
       current.active_plan_id == event.plan_id);
  return segment_matches && plan_matches;
}

[[nodiscard]] CommitmentTransition Reject(
    const HopperExecutionContext& current, std::string reason_code) {
  return CommitmentTransition{
      .next = current,
      .accepted = false,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] CommitmentTransition Accept(HopperExecutionContext next) {
  return CommitmentTransition{
      .next = std::move(next),
      .accepted = true,
      .reason_code = "ACCEPTED",
  };
}

}  // namespace

CommitmentTransition CommitmentStateMachine::Transition(
    const HopperExecutionContext& current,
    const CommitmentEvent& event) const {
  switch (current.state) {
    case HopperExecutionState::kGroundHold:
    case HopperExecutionState::kLandedHold:
      if (event.type == CommitmentEventType::kPublishCertifiedHop) {
        if (!HasBothIds(event)) {
          return Reject(current, "COMMITMENT_ID_REQUIRED");
        }
        return Accept(HopperExecutionContext{
            .state = HopperExecutionState::kJumpReady,
            .active_plan_id = event.plan_id,
            .active_segment_id = event.segment_id,
        });
      }
      if (event.type == CommitmentEventType::kDetectLaunch) {
        return Reject(current, "BOUNDARY_NOT_LOCKED");
      }
      return Reject(current, "EVENT_NOT_ALLOWED_IN_GROUND_HOLD");

    case HopperExecutionState::kJumpReady:
      if (event.type == CommitmentEventType::kPublishCertifiedHop) {
        if (!HasBothIds(event)) {
          return Reject(current, "COMMITMENT_ID_REQUIRED");
        }
        return Accept(HopperExecutionContext{
            .state = HopperExecutionState::kJumpReady,
            .active_plan_id = event.plan_id,
            .active_segment_id = event.segment_id,
        });
      }
      if (event.type == CommitmentEventType::kWithdrawHop) {
        if (!Matches(current, event, true)) {
          return Reject(current, "COMMITMENT_ID_MISMATCH");
        }
        return Accept(HopperExecutionContext{
            .state = HopperExecutionState::kGroundHold,
            .active_plan_id = std::nullopt,
            .active_segment_id = std::nullopt,
        });
      }
      if (event.type == CommitmentEventType::kLockJumpBoundary) {
        if (!Matches(current, event, true)) {
          return Reject(current, "COMMITMENT_ID_MISMATCH");
        }
        HopperExecutionContext next = current;
        next.state = HopperExecutionState::kJumpCommitted;
        return Accept(std::move(next));
      }
      if (event.type == CommitmentEventType::kDetectLaunch) {
        return Reject(current, "BOUNDARY_NOT_LOCKED");
      }
      return Reject(current, "EVENT_NOT_ALLOWED_WHILE_JUMP_READY");

    case HopperExecutionState::kJumpCommitted:
      if (event.type == CommitmentEventType::kPublishCertifiedHop) {
        return Reject(current, "COMMITTED_HOP_CANNOT_BE_REPLACED");
      }
      if (event.type == CommitmentEventType::kDetectLaunch) {
        if (!Matches(current, event, false)) {
          return Reject(current, "COMMITMENT_ID_MISMATCH");
        }
        HopperExecutionContext next = current;
        next.state = HopperExecutionState::kInFlight;
        return Accept(std::move(next));
      }
      if (event.type == CommitmentEventType::kInvalidateCommittedHop) {
        HopperExecutionContext next = current;
        next.state = HopperExecutionState::kEmergencyDelegated;
        return Accept(std::move(next));
      }
      return Reject(current, "EVENT_NOT_ALLOWED_AFTER_BOUNDARY_LOCK");

    case HopperExecutionState::kInFlight:
      if (event.type == CommitmentEventType::kDetectStableLanding) {
        if (!Matches(current, event, false)) {
          return Reject(current, "COMMITMENT_ID_MISMATCH");
        }
        HopperExecutionContext next = current;
        next.state = HopperExecutionState::kLandedHold;
        return Accept(std::move(next));
      }
      if (event.type == CommitmentEventType::kInvalidateCommittedHop) {
        HopperExecutionContext next = current;
        next.state = HopperExecutionState::kEmergencyDelegated;
        return Accept(std::move(next));
      }
      return Reject(current, "IN_FLIGHT_HOP_CANNOT_BE_REDIRECTED");

    case HopperExecutionState::kEmergencyDelegated:
      return Reject(current, "EMERGENCY_CONTROL_DELEGATED");
  }
  return Reject(current, "INVALID_COMMITMENT_STATE");
}

bool CommitmentStateMachine::MayPublishNewHop(
    const HopperExecutionContext& current) const noexcept {
  return current.state == HopperExecutionState::kGroundHold ||
      current.state == HopperExecutionState::kJumpReady ||
      current.state == HopperExecutionState::kLandedHold;
}

PlanningProtection EvaluatePlanningProtection(
    const std::optional<ExecutionContext>& context) {
  if (!context.has_value()) {
    return PlanningProtection{
        .decision = PlanningProtectionDecision::kMayPlan,
        .reason_code = {},
    };
  }
  const auto* hopper = std::get_if<HopperExecutionContext>(&*context);
  if (hopper == nullptr) {
    return PlanningProtection{
        .decision = PlanningProtectionDecision::kActiveReferenceInvalidated,
        .reason_code = "HOPPER_EXECUTION_CONTEXT_PLATFORM_MISMATCH",
    };
  }
  if (hopper->state == HopperExecutionState::kJumpCommitted ||
      hopper->state == HopperExecutionState::kInFlight) {
    if (!NonEmpty(hopper->active_plan_id) ||
        !NonEmpty(hopper->active_segment_id)) {
      return PlanningProtection{
          .decision = PlanningProtectionDecision::kActiveReferenceInvalidated,
          .reason_code = "COMMITTED_HOP_CONTEXT_INVALID",
      };
    }
    return PlanningProtection{
        .decision = PlanningProtectionDecision::kContinueCommittedHop,
        .reason_code = "COMMITTED_HOP_CONTINUES",
    };
  }
  if (hopper->state == HopperExecutionState::kEmergencyDelegated) {
    return PlanningProtection{
        .decision = PlanningProtectionDecision::kActiveReferenceInvalidated,
        .reason_code = "HOPPER_EMERGENCY_CONTROL_DELEGATED",
    };
  }
  return PlanningProtection{
      .decision = PlanningProtectionDecision::kMayPlan,
      .reason_code = {},
  };
}

}  // namespace lunar::planning::hopper
