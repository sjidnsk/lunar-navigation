#include "lunar_path_planner/v3/hopper/hopper_commitment_state_machine.hpp"

#include <utility>

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] bool HasIds(
    const HopperCommitmentEvent& event) {
  return event.bundle_id.has_value() &&
         !event.bundle_id->empty() &&
         event.boundary_id.has_value() &&
         !event.boundary_id->empty();
}

[[nodiscard]] bool Matches(
    const HopperCommitmentSnapshot& current,
    const HopperCommitmentEvent& event,
    const bool require_bundle) {
  const bool boundary_matches =
      current.active_boundary_id.has_value() &&
      event.boundary_id.has_value() &&
      current.active_boundary_id == event.boundary_id;
  const bool bundle_matches =
      !require_bundle ||
      (current.active_bundle_id.has_value() &&
       event.bundle_id.has_value() &&
       current.active_bundle_id == event.bundle_id);
  return boundary_matches && bundle_matches;
}

[[nodiscard]] HopperCommitmentTransition Reject(
    const HopperCommitmentSnapshot& current,
    ReasonCode reason) {
  return {
      .next = current,
      .accepted = false,
      .reason_code = std::move(reason),
  };
}

[[nodiscard]] HopperCommitmentTransition Accept(
    HopperCommitmentSnapshot next) {
  return {
      .next = std::move(next),
      .accepted = true,
      .reason_code = "ACCEPTED",
  };
}

}  // namespace

HopperCommitmentTransition
HopperCommitmentStateMachine::transition(
    const HopperCommitmentSnapshot& current,
    const HopperCommitmentEvent& event) const {
  switch (current.state) {
    case HopperCommitmentState::kGroundHold:
    case HopperCommitmentState::kLandedHold:
      if (event.type ==
          HopperCommitmentEventType::
              kPublishCertifiedCandidate) {
        if (!HasIds(event)) {
          return Reject(current, "COMMITMENT_ID_REQUIRED");
        }
        return Accept(
            {
                .state = HopperCommitmentState::kJumpReady,
                .active_bundle_id = event.bundle_id,
                .active_boundary_id = event.boundary_id,
                .commitment_scope =
                    HopperCommitmentScope::kGroundHold,
            });
      }
      if (event.type ==
          HopperCommitmentEventType::kDetectLaunch) {
        return Reject(current, "BOUNDARY_NOT_LOCKED");
      }
      return Reject(current, "EVENT_NOT_ALLOWED_IN_GROUND_HOLD");

    case HopperCommitmentState::kJumpReady:
      if (event.type ==
          HopperCommitmentEventType::
              kPublishCertifiedCandidate) {
        if (!HasIds(event)) {
          return Reject(current, "COMMITMENT_ID_REQUIRED");
        }
        return Accept(
            {
                .state = HopperCommitmentState::kJumpReady,
                .active_bundle_id = event.bundle_id,
                .active_boundary_id = event.boundary_id,
                .commitment_scope =
                    HopperCommitmentScope::kGroundHold,
            });
      }
      if (event.type ==
          HopperCommitmentEventType::kWithdrawCandidate) {
        if (!Matches(current, event, true)) {
          return Reject(current, "COMMITMENT_ID_MISMATCH");
        }
        return Accept(
            {
                .state = HopperCommitmentState::kGroundHold,
                .commitment_scope =
                    HopperCommitmentScope::kGroundHold,
            });
      }
      if (event.type ==
          HopperCommitmentEventType::kLockJumpBoundary) {
        if (!Matches(current, event, true)) {
          return Reject(current, "COMMITMENT_ID_MISMATCH");
        }
        HopperCommitmentSnapshot next = current;
        next.state = HopperCommitmentState::kJumpCommitted;
        next.commitment_scope =
            HopperCommitmentScope::kJumpBoundary;
        return Accept(std::move(next));
      }
      if (event.type ==
          HopperCommitmentEventType::kDetectLaunch) {
        return Reject(current, "BOUNDARY_NOT_LOCKED");
      }
      return Reject(current, "EVENT_NOT_ALLOWED_WHILE_JUMP_READY");

    case HopperCommitmentState::kJumpCommitted:
      if (event.type ==
          HopperCommitmentEventType::
              kPublishCertifiedCandidate) {
        return Reject(
            current, "COMMITTED_JUMP_CANNOT_BE_REPLACED");
      }
      if (event.type ==
          HopperCommitmentEventType::kDetectLaunch) {
        if (!Matches(current, event, false)) {
          return Reject(current, "COMMITMENT_ID_MISMATCH");
        }
        HopperCommitmentSnapshot next = current;
        next.state = HopperCommitmentState::kInFlight;
        return Accept(std::move(next));
      }
      if (event.type ==
          HopperCommitmentEventType::
              kInvalidateCommittedAction) {
        HopperCommitmentSnapshot next = current;
        next.state =
            HopperCommitmentState::kEmergencyDelegated;
        next.commitment_scope =
            HopperCommitmentScope::kEmergencyDelegated;
        return Accept(std::move(next));
      }
      return Reject(current, "EVENT_NOT_ALLOWED_AFTER_BOUNDARY_LOCK");

    case HopperCommitmentState::kInFlight:
      if (event.type ==
          HopperCommitmentEventType::kDetectStableLanding) {
        return Accept(
            {
                .state = HopperCommitmentState::kLandedHold,
                .commitment_scope =
                    HopperCommitmentScope::kGroundHold,
            });
      }
      if (event.type ==
          HopperCommitmentEventType::
              kInvalidateCommittedAction) {
        HopperCommitmentSnapshot next = current;
        next.state =
            HopperCommitmentState::kEmergencyDelegated;
        next.commitment_scope =
            HopperCommitmentScope::kEmergencyDelegated;
        return Accept(std::move(next));
      }
      return Reject(current, "IN_FLIGHT_JUMP_CANNOT_BE_REDIRECTED");

    case HopperCommitmentState::kEmergencyDelegated:
      return Reject(current, "EMERGENCY_CONTROL_DELEGATED");
  }
  return Reject(current, "INVALID_COMMITMENT_STATE");
}

bool HopperCommitmentStateMachine::may_publish_new_jump(
    const HopperCommitmentSnapshot& current) const noexcept {
  return current.state == HopperCommitmentState::kGroundHold ||
         current.state == HopperCommitmentState::kJumpReady ||
         current.state == HopperCommitmentState::kLandedHold;
}

}  // namespace lunar::planning::v3
