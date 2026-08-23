#include "lunar_pure_planner_core/planning_timing.hpp"

namespace lunar::pure_planning {
namespace {

[[nodiscard]] SteadyClock::time_point ReadNow(const NowFn& now) {
  return now ? now() : SteadyClock::now();
}

[[nodiscard]] std::chrono::nanoseconds NonNegativeElapsed(
    SteadyClock::time_point started_at,
    SteadyClock::time_point finished_at) noexcept {
  if (finished_at <= started_at) {
    return std::chrono::nanoseconds::zero();
  }
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      finished_at - started_at);
}

}  // namespace

RequestTimingPolicy MakeRequestTimingPolicy(
    const SteadyClock::time_point started_at) noexcept {
  return {
      .started_at = started_at,
      .target_milestone = started_at + RequestTimingPolicy::kTarget,
      .sla_milestone = started_at + RequestTimingPolicy::kSla,
      .hard_deadline = started_at + RequestTimingPolicy::kHard,
  };
}

RequestLatencyClass ClassifyRequestLatency(
    const RequestTimingPolicy& policy,
    const SteadyClock::time_point finalized_at) noexcept {
  if (finalized_at >= policy.hard_deadline) {
    return RequestLatencyClass::kHardTimeout;
  }
  if (finalized_at >= policy.sla_milestone) {
    return RequestLatencyClass::kSlaMissed;
  }
  if (finalized_at >= policy.target_milestone) {
    return RequestLatencyClass::kTargetMissed;
  }
  return RequestLatencyClass::kTargetMet;
}

RequestLatencyClass ClassifyRequestLatency(
    const std::chrono::nanoseconds elapsed) noexcept {
  return ClassifyRequestLatency(
      MakeRequestTimingPolicy(SteadyClock::time_point{}),
      SteadyClock::time_point{elapsed});
}

std::string_view RequestLatencyClassName(
    const RequestLatencyClass latency_class) noexcept {
  switch (latency_class) {
    case RequestLatencyClass::kTargetMet:
      return "TARGET_MET";
    case RequestLatencyClass::kTargetMissed:
      return "TARGET_MISSED";
    case RequestLatencyClass::kSlaMissed:
      return "SLA_MISSED";
    case RequestLatencyClass::kHardTimeout:
      return "HARD_TIMEOUT";
  }
  return "HARD_TIMEOUT";
}

ScopedPlannerCall::ScopedPlannerCall(
    const PlannerStage stage, PlannerCallTiming& timing, NowFn now)
    : stage_(stage),
      timing_(&timing),
      now_(std::move(now)),
      started_at_(ReadNow(now_)) {
  if (stage_ == PlannerStage::kGlobal) {
    ++timing_->global_call_count;
  } else {
    ++timing_->local_call_count;
  }
}

bool ScopedPlannerCall::Finish(
    SteadyClock::time_point* const finished_at) noexcept {
  if (finished_) {
    return finish_succeeded_;
  }
  finished_ = true;
  SteadyClock::time_point finish = started_at_;
  try {
    finish = ReadNow(now_);
  } catch (...) {
    return false;
  }

  const auto elapsed = NonNegativeElapsed(started_at_, finish);
  if (stage_ == PlannerStage::kGlobal) {
    timing_->global_elapsed += elapsed;
  } else {
    timing_->local_search_elapsed += elapsed;
    timing_->local_elapsed += elapsed;
  }
  timing_->total_elapsed += elapsed;
  if (finished_at != nullptr) {
    *finished_at = finish;
  }
  finish_succeeded_ = true;
  return true;
}

ScopedPlannerCall::~ScopedPlannerCall() noexcept {
  if (!finished_) {
    static_cast<void>(Finish());
  }
}

}  // namespace lunar::pure_planning
