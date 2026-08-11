#include <gtest/gtest.h>

#include <new>
#include <stdexcept>
#include <string_view>

#include "plan_exception_boundary.hpp"

namespace lunar::planning::training::detail {
namespace {

void ExpectFailure(const PlannerOutput &output,
                   const PlanningOutcome outcome,
                   const std::string_view reason_code) {
  EXPECT_EQ(output.outcome, outcome);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_EQ(output.reason_code, reason_code);
  EXPECT_EQ(output.candidate_disposition, CandidateDisposition::kKeep);
}

TEST(PlanExceptionBoundary, PreservesSuccessfulPlannerOutput) {
  const PlannerOutput output = TranslatePlanExceptions([] {
    return PlannerOutput{
        .outcome = PlanningOutcome::kNewReferenceAvailable,
        .directive = ExecutionDirective::kActivateNewReference,
        .candidate_disposition = CandidateDisposition::kKeep,
        .reason_code = "BRIDGE_PLAN_AVAILABLE",
    };
  });

  EXPECT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable);
  EXPECT_EQ(output.directive, ExecutionDirective::kActivateNewReference);
  EXPECT_EQ(output.reason_code, "BRIDGE_PLAN_AVAILABLE");
  EXPECT_EQ(output.candidate_disposition, CandidateDisposition::kKeep);
}

TEST(PlanExceptionBoundary, TranslatesBadAllocAndKeepsCandidate) {
  const PlannerOutput output = TranslatePlanExceptions([]() -> PlannerOutput {
    throw std::bad_alloc{};
  });

  ExpectFailure(output, PlanningOutcome::kResourceExhausted,
                "BRIDGE_RESOURCE_EXHAUSTED");
}

TEST(PlanExceptionBoundary, TranslatesCatchAllAndKeepsCandidate) {
  const PlannerOutput output = TranslatePlanExceptions([]() -> PlannerOutput {
    throw std::runtime_error{"conversion failed"};
  });

  ExpectFailure(output, PlanningOutcome::kNumericalFailure,
                "BRIDGE_REQUEST_CONVERSION_FAILED");
}

}  // namespace
}  // namespace lunar::planning::training::detail
