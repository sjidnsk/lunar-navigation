#pragma once

#include <optional>

#include "lunar_path_planner/v3/contracts/planning_response.hpp"

namespace lunar::planning::v3 {

struct PlanningAttempt final {
  RequestId request_id;
  ClockStamp response_time;
  std::optional<ReferenceBundle> validated_new_bundle;
  std::optional<Error> failure;
  PlanningOutcome planning_outcome{PlanningOutcome::kInvalidRequest};
  ReasonCode reason_code;
  CallDiagnostics call_diagnostics;
  bool previous_execution_context_valid{true};
};

[[nodiscard]] PlanningResponse ArbitratePlanningResponse(
    const PlanningAttempt& attempt,
    const std::optional<PreviousExecutionContext>& previous,
    const PlatformState& state) noexcept;

}  // namespace lunar::planning::v3
