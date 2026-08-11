#pragma once

#include <functional>
#include <new>
#include <utility>

#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning::training::detail {

[[nodiscard]] inline PlannerOutput BridgeFailure(
    const PlanningOutcome outcome,
    const CandidateDisposition candidate_disposition,
    const char *reason_code) noexcept {
  PlannerOutput output;
  output.outcome = outcome;
  output.directive = ExecutionDirective::kNoSafeReference;
  output.candidate_disposition = candidate_disposition;
  output.reason_code = reason_code;
  output.diagnostics.planner_name = "cpp_v3_hierarchical";
  return output;
}

template <typename Callable>
[[nodiscard]] PlannerOutput TranslatePlanExceptions(
    Callable &&callable) noexcept {
  try {
    return std::invoke(std::forward<Callable>(callable));
  } catch (const std::bad_alloc &) {
    return BridgeFailure(PlanningOutcome::kResourceExhausted,
                         CandidateDisposition::kKeep,
                         "BRIDGE_RESOURCE_EXHAUSTED");
  } catch (...) {
    return BridgeFailure(PlanningOutcome::kNumericalFailure,
                         CandidateDisposition::kKeep,
                         "BRIDGE_REQUEST_CONVERSION_FAILED");
  }
}

}  // namespace lunar::planning::training::detail
