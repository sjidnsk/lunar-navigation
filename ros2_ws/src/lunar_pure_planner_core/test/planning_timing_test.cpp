#include <chrono>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <stop_token>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_planner_core/planning_timing.hpp"
#include "lunar_pure_planner_core/search_control.hpp"
#include "lunar_pure_planner_core/types/planning_request.hpp"

namespace lunar::pure_planning {
namespace {

using namespace std::chrono_literals;

class FakeClock final {
 public:
  explicit FakeClock(std::initializer_list<std::chrono::milliseconds> ticks)
      : ticks_(ticks) {}

  SteadyClock::time_point Now() {
    const auto tick = ticks_.at(index_++);
    return SteadyClock::time_point{tick};
  }

  NowFn NowFnValue() {
    return [this] { return Now(); };
  }

 private:
  std::vector<std::chrono::milliseconds> ticks_;
  std::size_t index_{};
};

TEST(PlanningTiming, CountsEveryEnteredCallIncludingException) {
  FakeClock clock({0ms, 7ms});
  PlannerCallTiming timing;
  {
    ScopedPlannerCall call(PlannerStage::kGlobal, timing, clock.NowFnValue());
  }

  EXPECT_EQ(timing.global_call_count, 1U);
  EXPECT_EQ(timing.global_elapsed, 7ms);
  EXPECT_EQ(timing.local_call_count, 0U);
  EXPECT_EQ(timing.total_elapsed, 7ms);
}

TEST(PlanningTiming, CountsCallWhenExceptionUnwinds) {
  FakeClock clock({0ms, 11ms});
  PlannerCallTiming timing;

  try {
    ScopedPlannerCall call(PlannerStage::kLocal, timing, clock.NowFnValue());
    throw std::runtime_error("planner backend failure");
  } catch (const std::runtime_error&) {
  }

  EXPECT_EQ(timing.local_call_count, 1U);
  EXPECT_EQ(timing.local_search_elapsed, 11ms);
  EXPECT_EQ(timing.local_elapsed, 11ms);
  EXPECT_EQ(timing.total_elapsed, 11ms);
}

TEST(PlanningTiming, ExactMilestonesBelongToTheLaterLatencyClass) {
  const auto policy =
      MakeRequestTimingPolicy(SteadyClock::time_point{100ms});
  struct Case final {
    std::chrono::milliseconds finalized;
    RequestLatencyClass latency_class;
    const char* name;
  };
  for (const Case& test_case : {
           Case{1099ms, RequestLatencyClass::kTargetMet, "TARGET_MET"},
           Case{1100ms, RequestLatencyClass::kTargetMissed, "TARGET_MISSED"},
           Case{2099ms, RequestLatencyClass::kTargetMissed, "TARGET_MISSED"},
           Case{2100ms, RequestLatencyClass::kSlaMissed, "SLA_MISSED"},
           Case{3099ms, RequestLatencyClass::kSlaMissed, "SLA_MISSED"},
           Case{3100ms, RequestLatencyClass::kHardTimeout, "HARD_TIMEOUT"},
       }) {
    const auto actual = ClassifyRequestLatency(
        policy, SteadyClock::time_point{test_case.finalized});
    EXPECT_EQ(actual, test_case.latency_class) << test_case.finalized.count();
    EXPECT_EQ(RequestLatencyClassName(actual), test_case.name);
  }
  EXPECT_EQ(policy.target_milestone, SteadyClock::time_point{1100ms});
  EXPECT_EQ(policy.sla_milestone, SteadyClock::time_point{2100ms});
  EXPECT_EQ(policy.hard_deadline, SteadyClock::time_point{3100ms});
}

TEST(PlanningTiming, ClampsClockRegressionToNonNegativeDuration) {
  FakeClock clock({10ms, 5ms});
  PlannerCallTiming timing;
  { ScopedPlannerCall call(PlannerStage::kGlobal, timing, clock.NowFnValue()); }

  EXPECT_EQ(timing.global_call_count, 1U);
  EXPECT_EQ(timing.global_elapsed, 0ns);
  EXPECT_EQ(timing.total_elapsed, 0ns);
}

TEST(SearchControl, UsesOnlyStopTokenAndSteadyDeadline) {
  std::stop_source source;
  SearchControl control;
  control.deadline = SteadyClock::time_point{10ms};
  control.now = [] { return SteadyClock::time_point{10ms}; };

  EXPECT_FALSE(control.canceled());
  EXPECT_TRUE(control.expired());

  source.request_stop();
  control.stop_token = source.get_token();
  EXPECT_TRUE(control.canceled());
}

TEST(PlanningTypes, PreserveExactEnvironmentModeValuesAndMinimalRequest) {
  EXPECT_EQ(static_cast<std::uint8_t>(EnvironmentMode::kLunarSurface), 1U);
  EXPECT_EQ(static_cast<std::uint8_t>(EnvironmentMode::kLavaTube), 2U);

  const AnytimePlannerConfig config;
  EXPECT_TRUE(config.search.stop_after_first_solution);
  EXPECT_DOUBLE_EQ(config.global_occupancy_threshold, 50.0);
  EXPECT_DOUBLE_EQ(config.local_occupancy_threshold, 0.5);
  EXPECT_EQ(config.search.epsilon_schedule,
            (std::array<double, 4>{2.5, 2.0, 1.5, 1.0}));

  PlanningRequest request;
  request.environment_mode = EnvironmentMode::kLavaTube;
  EXPECT_EQ(request.environment_mode, EnvironmentMode::kLavaTube);
  EXPECT_FALSE(request.world.global_map.has_value());
}

}  // namespace
}  // namespace lunar::pure_planning
