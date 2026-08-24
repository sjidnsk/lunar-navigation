#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <string_view>

#include "lunar_pure_planner_core/search_control.hpp"

namespace lunar::pure_planning {

enum class RequestLatencyClass : std::uint8_t {
  kTargetMet,
  kTargetMissed,
  kSlaMissed,
  kHardTimeout,
};

struct RequestTimingPolicy final {
  static constexpr auto kTarget = std::chrono::seconds{1};
  static constexpr auto kSla = std::chrono::seconds{2};
  static constexpr auto kHard = std::chrono::seconds{3};
  SteadyClock::time_point started_at;
  SteadyClock::time_point target_milestone;
  SteadyClock::time_point sla_milestone;
  SteadyClock::time_point hard_deadline;
};

[[nodiscard]] RequestTimingPolicy MakeRequestTimingPolicy(
    SteadyClock::time_point started_at) noexcept;
[[nodiscard]] RequestLatencyClass ClassifyRequestLatency(
    const RequestTimingPolicy& policy,
    SteadyClock::time_point finalized_at) noexcept;
[[nodiscard]] RequestLatencyClass ClassifyRequestLatency(
    std::chrono::nanoseconds elapsed) noexcept;
[[nodiscard]] std::string_view RequestLatencyClassName(
    RequestLatencyClass latency_class) noexcept;

enum class PlannerPhase : std::uint8_t {
  kSnapshotProjection,
  kGlobal,
  kLocalGoal,
  kLocalSearch,
  kCertification,
  kOutput,
};

struct PlannerProgress final {
  PlannerPhase phase{PlannerPhase::kSnapshotProjection};
  std::chrono::nanoseconds elapsed{};
};

using ProgressFn = std::function<void(const PlannerProgress&)>;

enum class PlannerStage : std::uint8_t {
  kGlobal,
  kLocal,
};

struct PlannerCallTiming final {
  std::chrono::nanoseconds snapshot_projection_elapsed{};
  std::chrono::nanoseconds global_elapsed{};
  std::uint64_t global_call_count{};
  std::chrono::nanoseconds local_goal_elapsed{};
  std::chrono::nanoseconds local_search_elapsed{};
  std::chrono::nanoseconds local_elapsed{};
  std::uint64_t local_call_count{};
  std::chrono::nanoseconds certification_elapsed{};
  std::chrono::nanoseconds output_elapsed{};
  std::chrono::nanoseconds total_elapsed{};
};

class ScopedPlannerCall final {
 public:
  ScopedPlannerCall(PlannerStage stage, PlannerCallTiming& timing, NowFn now = {});
  ScopedPlannerCall(const ScopedPlannerCall&) = delete;
  ScopedPlannerCall& operator=(const ScopedPlannerCall&) = delete;
  [[nodiscard]] bool Finish(
      SteadyClock::time_point* finished_at = nullptr) noexcept;
  ~ScopedPlannerCall() noexcept;

 private:
  PlannerStage stage_;
  PlannerCallTiming* timing_;
  NowFn now_;
  SteadyClock::time_point started_at_;
  bool finished_{};
  bool finish_succeeded_{};
};

}  // namespace lunar::pure_planning
