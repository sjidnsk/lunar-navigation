#include "lunar_pure_exploration_ros/stationary_planning_gate.hpp"

#include <chrono>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

namespace lunar::pure_exploration_ros {
namespace {

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

StationaryGateParameters TestParameters() {
  return {
      .maximum_linear_speed_mps = 0.01,
      .maximum_angular_speed_radps = 0.02,
      .confirmation_samples = 3U,
      .diagnostic_period = 5s,
  };
}

TEST(StationaryPlanningGateTest, ConfirmsAfterRequiredConsecutiveSamples) {
  StationaryPlanningGate gate{TestParameters()};
  const auto start = Clock::time_point{};
  gate.Begin(start, {.linear_speed_mps = 0.2, .angular_speed_radps = 0.1});

  EXPECT_FALSE(gate.Observe(start + 100ms, {0.0, 0.0}).confirmed);
  EXPECT_FALSE(gate.Observe(start + 200ms, {0.0, 0.0}).confirmed);
  EXPECT_TRUE(gate.Observe(start + 300ms, {0.0, 0.0}).confirmed);
  EXPECT_FALSE(gate.waiting());
}

TEST(StationaryPlanningGateTest, MovingSampleResetsConsecutiveSamples) {
  StationaryPlanningGate gate{TestParameters()};
  const auto start = Clock::time_point{};
  gate.Begin(start, {});

  EXPECT_EQ(gate.Observe(start + 100ms, {0.0, 0.0}).consecutive_samples,
            1U);
  EXPECT_EQ(gate.Observe(start + 200ms, {0.0, 0.0}).consecutive_samples,
            2U);
  EXPECT_EQ(gate.Observe(start + 300ms, {0.02, 0.0}).consecutive_samples,
            0U);
  EXPECT_FALSE(gate.Observe(start + 400ms, {0.0, 0.0}).confirmed);
  EXPECT_FALSE(gate.Observe(start + 500ms, {0.0, 0.0}).confirmed);
  EXPECT_TRUE(gate.Observe(start + 600ms, {0.0, 0.0}).confirmed);
}

TEST(StationaryPlanningGateTest, ThresholdValuesAreStationary) {
  auto parameters = TestParameters();
  parameters.confirmation_samples = 1U;
  StationaryPlanningGate gate{parameters};
  const auto start = Clock::time_point{};
  gate.Begin(start, {});

  const auto update = gate.Observe(
      start + 1ms, {.linear_speed_mps = 0.01, .angular_speed_radps = 0.02});

  EXPECT_TRUE(update.confirmed);
  EXPECT_EQ(update.consecutive_samples, 1U);
}

TEST(StationaryPlanningGateTest, NonFiniteSamplesResetAndCannotConfirm) {
  StationaryPlanningGate gate{TestParameters()};
  const auto start = Clock::time_point{};
  gate.Begin(start, {});

  EXPECT_EQ(gate.Observe(start + 100ms, {0.0, 0.0}).consecutive_samples,
            1U);
  EXPECT_EQ(gate.Observe(start + 200ms,
                          {std::numeric_limits<double>::infinity(), 0.0})
                .consecutive_samples,
            0U);
  EXPECT_EQ(gate.Observe(start + 300ms,
                          {0.0, std::numeric_limits<double>::quiet_NaN()})
                .consecutive_samples,
            0U);
  EXPECT_FALSE(gate.Observe(start + 400ms, {0.0, 0.0}).confirmed);
  EXPECT_FALSE(gate.Observe(start + 500ms, {0.0, 0.0}).confirmed);
  EXPECT_TRUE(gate.Observe(start + 600ms, {0.0, 0.0}).confirmed);
}

TEST(StationaryPlanningGateTest, DiagnosticDueBeginsAtPeriodAndAdvancesOnce) {
  StationaryPlanningGate gate{TestParameters()};
  const auto start = Clock::time_point{};
  gate.Begin(start, {});

  EXPECT_FALSE(gate.Observe(start + 4999ms, {0.2, 0.0}).diagnostic_due);
  EXPECT_TRUE(gate.Observe(start + 5s, {0.2, 0.0}).diagnostic_due);
  EXPECT_FALSE(gate.Observe(start + 5s, {0.2, 0.0}).diagnostic_due);
  EXPECT_TRUE(gate.Observe(start + 10s, {0.2, 0.0}).diagnostic_due);
}

TEST(StationaryPlanningGateTest, CancelClearsWaitingAndCounters) {
  StationaryPlanningGate gate{TestParameters()};
  const auto start = Clock::time_point{};
  gate.Begin(start, {.linear_speed_mps = 0.2, .angular_speed_radps = 0.1});
  EXPECT_EQ(gate.Observe(start + 1ms, {0.0, 0.0}).consecutive_samples, 1U);

  gate.Cancel();
  const auto update = gate.Observe(start + 2ms, {0.0, 0.0});

  EXPECT_FALSE(gate.waiting());
  EXPECT_FALSE(update.confirmed);
  EXPECT_EQ(update.consecutive_samples, 0U);
  EXPECT_DOUBLE_EQ(gate.entry_speed().linear_speed_mps, 0.0);
  EXPECT_DOUBLE_EQ(gate.entry_speed().angular_speed_radps, 0.0);
}

TEST(StationaryPlanningGateTest, InvalidParametersThrow) {
  auto negative_linear = TestParameters();
  negative_linear.maximum_linear_speed_mps = -0.01;
  EXPECT_THROW(StationaryPlanningGate{negative_linear}, std::invalid_argument);

  auto negative_angular = TestParameters();
  negative_angular.maximum_angular_speed_radps = -0.02;
  EXPECT_THROW(StationaryPlanningGate{negative_angular}, std::invalid_argument);

  auto no_samples = TestParameters();
  no_samples.confirmation_samples = 0U;
  EXPECT_THROW(StationaryPlanningGate{no_samples}, std::invalid_argument);

  auto no_period = TestParameters();
  no_period.diagnostic_period = std::chrono::steady_clock::duration::zero();
  EXPECT_THROW(StationaryPlanningGate{no_period}, std::invalid_argument);
}

}  // namespace
}  // namespace lunar::pure_exploration_ros
