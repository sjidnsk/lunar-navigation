#pragma once

#include <chrono>
#include <cstdint>

#include "lunar_pure_planner_core/search_control.hpp"

namespace lunar::pure_planning {

enum class PlannerStage : std::uint8_t {
  kGlobal,
  kLocal,
};

struct PlannerCallTiming final {
  std::chrono::nanoseconds global_elapsed{};
  std::uint64_t global_call_count{};
  std::chrono::nanoseconds local_elapsed{};
  std::uint64_t local_call_count{};
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
