#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "lunar_planner_core/types/execution_context.hpp"

namespace lunar::planning::hopper {

enum class CommitmentEventType : std::uint8_t {
  kPublishCertifiedHop,
  kWithdrawHop,
  kLockJumpBoundary,
  kDetectLaunch,
  kDetectStableLanding,
  kInvalidateCommittedHop,
};

struct CommitmentEvent final {
  CommitmentEventType type{CommitmentEventType::kPublishCertifiedHop};
  std::optional<std::string> plan_id;
  std::optional<std::string> segment_id;
};

struct CommitmentTransition final {
  HopperExecutionContext next;
  bool accepted{};
  std::string reason_code;
};

class CommitmentStateMachine final {
 public:
  [[nodiscard]] CommitmentTransition Transition(
      const HopperExecutionContext& current,
      const CommitmentEvent& event) const;

  [[nodiscard]] bool MayPublishNewHop(
      const HopperExecutionContext& current) const noexcept;
};

enum class PlanningProtectionDecision : std::uint8_t {
  kMayPlan,
  kContinueCommittedHop,
  kActiveReferenceInvalidated,
};

struct PlanningProtection final {
  PlanningProtectionDecision decision{PlanningProtectionDecision::kMayPlan};
  std::string reason_code;
};

[[nodiscard]] PlanningProtection EvaluatePlanningProtection(
    const std::optional<ExecutionContext>& context);

}  // namespace lunar::planning::hopper
