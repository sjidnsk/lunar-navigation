#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/contracts/reference_bundle.hpp"

namespace lunar::planning::v3 {

enum class PlanningOutcome {
  kNewReferenceReady,
  kSafeFrontierReferenceReady,
  kNoKnownSafeRoute,
  kGoalInfeasible,
  kInvalidRequest,
  kStaleInput,
  kNumericalFailure,
  kResourceLimit,
  kActiveReferenceInvalidated,
};

enum class ExecutionDirective {
  kActivateNewBundle,
  kContinueActiveBundle,
  kHoldStationary,
  kContinueCommittedJump,
  kNoSafePlannerReference,
};

enum class LearnedCostUsage {
  kDisabled,
  kUsedBoundedSoftCost,
  kFellBackToAnalytic,
};

struct SecondaryCosts final {
  std::optional<double> energy;
  std::optional<double> nonfatal_risk;
  std::optional<double> smoothness;
};

struct CallDiagnostics final {
  DurationNanoseconds api_latency;
  std::string termination_reason;
  std::optional<double> final_epsilon;
  std::optional<DurationNanoseconds> expected_execution_time;
  std::optional<SecondaryCosts> secondary_costs;
  std::uint64_t expanded_state_count{};
  std::uint64_t reopened_state_count{};
  std::uint64_t candidate_count{};
  bool resource_limit_hit{};
  LearnedCostUsage learned_cost_usage{LearnedCostUsage::kDisabled};
  std::optional<ContentRef> learned_cost_snapshot_ref;
  std::vector<std::string> message_codes;
};

struct PlanningResponse final {
  RequestId request_id;
  ClockStamp response_time;
  PlanningOutcome planning_outcome{};
  ExecutionDirective execution_directive{};
  ReasonCode reason_code;
  std::optional<ContentRef> active_bundle_ref;
  std::optional<ReferenceBundle> new_reference_bundle;
  CallDiagnostics call_diagnostics;
};

}  // namespace lunar::planning::v3
