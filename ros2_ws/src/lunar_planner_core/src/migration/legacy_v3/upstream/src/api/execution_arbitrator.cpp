#include "lunar_path_planner/v3/api/execution_arbitrator.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] bool CanonicalZero(double value) noexcept {
  return value == 0.0 && !std::signbit(value);
}

[[nodiscard]] bool CanonicalZero(const Vec3& value) noexcept {
  return CanonicalZero(value.x) && CanonicalZero(value.y) &&
         CanonicalZero(value.z);
}

[[nodiscard]] bool Stationary(const PlatformState& state) noexcept {
  return std::visit(
      [](const auto& concrete) {
        using StateT = std::decay_t<decltype(concrete)>;
        if constexpr (
            std::is_same_v<StateT, WheeledOrLeggedState>) {
          return CanonicalZero(concrete.linear_velocity_mps) &&
                 CanonicalZero(concrete.yaw_rate_radps);
        } else {
          return CanonicalZero(concrete.linear_velocity_mps) &&
                 CanonicalZero(concrete.angular_velocity_radps);
        }
      },
      state);
}

[[nodiscard]] bool PlausibleContentRef(
    const ContentRef& ref) noexcept {
  if (ref.id.empty() || ref.revision == 0U ||
      ref.content_hash.size() != 64U) {
    return false;
  }
  for (const char character : ref.content_hash) {
    const bool decimal =
        character >= '0' && character <= '9';
    const bool lower_hex =
        character >= 'a' && character <= 'f';
    if (!decimal && !lower_hex) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool MatchingCommittedJump(
    const std::optional<PreviousExecutionContext>& previous,
    const PlatformState& state) noexcept {
  if (!previous.has_value() ||
      !std::holds_alternative<HopperState>(state) ||
      !PlausibleContentRef(previous->active_bundle_ref)) {
    return false;
  }
  const auto* boundary =
      std::get_if<JumpCommitBoundary>(
          &previous->commit_boundary);
  const auto* cursor =
      std::get_if<JumpExecutionCursor>(
          &previous->execution_cursor);
  if (boundary == nullptr || cursor == nullptr ||
      !boundary->locked || boundary->boundary_id.empty() ||
      !cursor->boundary_id.has_value() ||
      *cursor->boundary_id != boundary->boundary_id) {
    return false;
  }
  if (cursor->jump_state ==
      JumpExecutionState::kJumpCommitted) {
    return previous->controller_status ==
           ControllerStatus::kCommitted;
  }
  if (cursor->jump_state == JumpExecutionState::kInFlight) {
    return previous->controller_status ==
           ControllerStatus::kInFlight;
  }
  return false;
}

[[nodiscard]] bool ValidActiveGroundReference(
    const std::optional<PreviousExecutionContext>& previous,
    const PlatformState& state,
    bool context_valid) noexcept {
  if (!context_valid || !previous.has_value() ||
      !PlausibleContentRef(previous->active_bundle_ref) ||
      previous->controller_status == ControllerStatus::kFault) {
    return false;
  }
  if (std::holds_alternative<WheeledOrLeggedState>(state)) {
    return std::holds_alternative<TimeCommitBoundary>(
               previous->commit_boundary) &&
           std::holds_alternative<TimeExecutionCursor>(
               previous->execution_cursor);
  }
  const auto* boundary =
      std::get_if<JumpCommitBoundary>(
          &previous->commit_boundary);
  const auto* cursor =
      std::get_if<JumpExecutionCursor>(
          &previous->execution_cursor);
  if (boundary == nullptr || cursor == nullptr ||
      boundary->locked) {
    return false;
  }
  return cursor->jump_state == JumpExecutionState::kGroundHold ||
         cursor->jump_state == JumpExecutionState::kJumpReady ||
         cursor->jump_state == JumpExecutionState::kLandedHold;
}

[[nodiscard]] bool ReadyOutcome(PlanningOutcome outcome) noexcept {
  return outcome == PlanningOutcome::kNewReferenceReady ||
         outcome == PlanningOutcome::kSafeFrontierReferenceReady;
}

[[nodiscard]] PlanningOutcome FailureOutcome(
    const PlanningAttempt& attempt) noexcept {
  if (!ReadyOutcome(attempt.planning_outcome)) {
    return attempt.previous_execution_context_valid
               ? attempt.planning_outcome
               : PlanningOutcome::kActiveReferenceInvalidated;
  }
  if (!attempt.failure.has_value()) {
    return PlanningOutcome::kInvalidRequest;
  }
  switch (attempt.failure->code) {
    case ErrorCode::kSchemaMismatch:
    case ErrorCode::kInvalidArgument:
    case ErrorCode::kMissingRegistryObject:
      return PlanningOutcome::kInvalidRequest;
    case ErrorCode::kStaleInput:
    case ErrorCode::kInconsistentSnapshot:
      return PlanningOutcome::kStaleInput;
    case ErrorCode::kResourceLimit:
      return PlanningOutcome::kResourceLimit;
    case ErrorCode::kNumericalFailure:
      return PlanningOutcome::kNumericalFailure;
    case ErrorCode::kNoKnownSafeRoute:
      return PlanningOutcome::kNoKnownSafeRoute;
  }
  return PlanningOutcome::kInvalidRequest;
}

[[nodiscard]] RequestId ResponseRequestId(
    const PlanningAttempt& attempt) {
  return attempt.request_id.empty() ? "invalid-request"
                                    : attempt.request_id;
}

[[nodiscard]] ClockStamp ResponseTime(
    const PlanningAttempt& attempt) {
  if (!attempt.response_time.clock_id.empty()) {
    return attempt.response_time;
  }
  return {
      .clock_id = "planner-clock",
      .tick = std::chrono::nanoseconds{0},
  };
}

[[nodiscard]] ReasonCode ResponseReason(
    const PlanningAttempt& attempt,
    std::string fallback) {
  return attempt.reason_code.empty() ? std::move(fallback)
                                     : attempt.reason_code;
}

[[nodiscard]] CallDiagnostics ResponseDiagnostics(
    const PlanningAttempt& attempt,
    std::string fallback) {
  CallDiagnostics diagnostics = attempt.call_diagnostics;
  if (diagnostics.termination_reason.empty()) {
    diagnostics.termination_reason = std::move(fallback);
  }
  return diagnostics;
}

[[nodiscard]] PlanningResponse ContinueCommitted(
    const PlanningAttempt& attempt,
    const PreviousExecutionContext& previous) {
  return {
      .request_id = ResponseRequestId(attempt),
      .response_time = ResponseTime(attempt),
      .planning_outcome = FailureOutcome(attempt),
      .execution_directive =
          ExecutionDirective::kContinueCommittedJump,
      .reason_code =
          ResponseReason(attempt, "COMMITTED_JUMP_CONTINUES"),
      .active_bundle_ref = previous.active_bundle_ref,
      .call_diagnostics =
          ResponseDiagnostics(
              attempt, "COMMITTED_JUMP_CONTINUES"),
  };
}

[[nodiscard]] PlanningResponse Arbitrate(
    const PlanningAttempt& attempt,
    const std::optional<PreviousExecutionContext>& previous,
    const PlatformState& state) {
  const bool has_bundle =
      attempt.validated_new_bundle.has_value();
  const bool has_failure = attempt.failure.has_value();

  if (MatchingCommittedJump(previous, state)) {
    return ContinueCommitted(attempt, *previous);
  }

  if (has_bundle != has_failure && has_bundle &&
      ReadyOutcome(attempt.planning_outcome)) {
    return {
        .request_id = ResponseRequestId(attempt),
        .response_time = ResponseTime(attempt),
        .planning_outcome = attempt.planning_outcome,
        .execution_directive =
            ExecutionDirective::kActivateNewBundle,
        .reason_code =
            ResponseReason(attempt, "NEW_REFERENCE_READY"),
        .new_reference_bundle = attempt.validated_new_bundle,
        .call_diagnostics =
            ResponseDiagnostics(
                attempt, "NEW_REFERENCE_READY"),
    };
  }

  PlanningAttempt normalized = attempt;
  if (has_bundle == has_failure || has_bundle) {
    normalized.validated_new_bundle.reset();
    normalized.failure = Error{
        ErrorCode::kInvalidArgument,
        "planning_attempt",
        "planning attempt must carry exactly one outcome payload",
    };
    normalized.planning_outcome =
        PlanningOutcome::kInvalidRequest;
    normalized.reason_code =
        "INVALID_PLANNING_ATTEMPT";
  }

  const PlanningOutcome failure_outcome =
      FailureOutcome(normalized);
  if (ValidActiveGroundReference(
          previous, state,
          normalized.previous_execution_context_valid)) {
    return {
        .request_id = ResponseRequestId(normalized),
        .response_time = ResponseTime(normalized),
        .planning_outcome = failure_outcome,
        .execution_directive =
            ExecutionDirective::kContinueActiveBundle,
        .reason_code =
            ResponseReason(
                normalized, "CONTINUE_ACTIVE_BUNDLE"),
        .active_bundle_ref = previous->active_bundle_ref,
        .call_diagnostics =
            ResponseDiagnostics(
                normalized, "CONTINUE_ACTIVE_BUNDLE"),
    };
  }
  if (Stationary(state)) {
    return {
        .request_id = ResponseRequestId(normalized),
        .response_time = ResponseTime(normalized),
        .planning_outcome = failure_outcome,
        .execution_directive =
            ExecutionDirective::kHoldStationary,
        .reason_code =
            ResponseReason(normalized, "HOLD_STATIONARY"),
        .call_diagnostics =
            ResponseDiagnostics(
                normalized, "HOLD_STATIONARY"),
    };
  }
  return {
      .request_id = ResponseRequestId(normalized),
      .response_time = ResponseTime(normalized),
      .planning_outcome = failure_outcome,
      .execution_directive =
          ExecutionDirective::kNoSafePlannerReference,
      .reason_code =
          ResponseReason(
              normalized, "NO_SAFE_PLANNER_REFERENCE"),
      .call_diagnostics =
          ResponseDiagnostics(
              normalized, "NO_SAFE_PLANNER_REFERENCE"),
  };
}

}  // namespace

PlanningResponse ArbitratePlanningResponse(
    const PlanningAttempt& attempt,
    const std::optional<PreviousExecutionContext>& previous,
    const PlatformState& state) noexcept {
  try {
    return Arbitrate(attempt, previous, state);
  } catch (const std::exception&) {
    PlanningAttempt failed = attempt;
    failed.validated_new_bundle.reset();
    failed.failure = Error{
        ErrorCode::kNumericalFailure,
        "execution_arbitrator",
        "execution arbitration failed closed",
    };
    failed.planning_outcome =
        PlanningOutcome::kNumericalFailure;
    failed.reason_code = "ARBITRATION_FAILURE";
    failed.call_diagnostics.termination_reason =
        "ARBITRATION_FAILURE";
    return Arbitrate(failed, std::nullopt, state);
  } catch (...) {
    PlanningAttempt failed = attempt;
    failed.validated_new_bundle.reset();
    failed.failure = Error{
        ErrorCode::kNumericalFailure,
        "execution_arbitrator",
        "unknown execution arbitration failure",
    };
    failed.planning_outcome =
        PlanningOutcome::kNumericalFailure;
    failed.reason_code = "ARBITRATION_FAILURE";
    failed.call_diagnostics.termination_reason =
        "ARBITRATION_FAILURE";
    return Arbitrate(failed, std::nullopt, state);
  }
}

}  // namespace lunar::planning::v3
