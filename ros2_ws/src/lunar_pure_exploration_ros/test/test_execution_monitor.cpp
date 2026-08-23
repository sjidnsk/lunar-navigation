#include "lunar_pure_exploration_ros/exploration_node.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

namespace lunar::pure_exploration_ros {
namespace {

using lunar::pure_exploration::Pose2;
using lunar::pure_exploration::Vec2;
using namespace std::chrono_literals;

TEST(ExecutionMonitorTest, ArrivalUsesLiteralPositionToleranceBoundaries) {
  struct Fixture {
    double resolution_m;
    double platform_width_m;
    double tolerance_m;
  };
  // The first fixture is resolution-won; the second is width-won.
  const std::vector<Fixture> fixtures{{0.8, 1.0, 0.4}, {0.4, 1.2, 0.3}};
  constexpr double kYawTolerance = std::numbers::pi / 16.0;

  for (const auto& fixture : fixtures) {
    ExecutionMonitor monitor({.maximum_executable_path_points = 8U,
                              .position_tolerance_m = fixture.tolerance_m,
                              .yaw_tolerance_rad = kYawTolerance,
                              .stuck_window = 30s,
                              .minimum_progress_m = 0.2});
    const Pose2 target{0.0, 0.0, 0.0};
    EXPECT_TRUE(monitor.ReachedFinalGoal(
        {fixture.tolerance_m, 0.0, 0.0},
        target));
    EXPECT_FALSE(monitor.ReachedFinalGoal(
        {std::nextafter(fixture.tolerance_m,
                        std::numeric_limits<double>::infinity()),
         0.0, 0.0},
        target));
    EXPECT_TRUE(monitor.ReachedFinalGoal(
        {0.0, 0.0, kYawTolerance}, target));
    EXPECT_FALSE(monitor.ReachedFinalGoal(
        {0.0, 0.0, std::nextafter(kYawTolerance,
                                  std::numeric_limits<double>::infinity())},
        target));
    EXPECT_FALSE(monitor.ReachedFinalGoal(
        {fixture.tolerance_m, 0.0,
         std::nextafter(kYawTolerance,
                        std::numeric_limits<double>::infinity())},
        target));
    EXPECT_FALSE(monitor.ReachedFinalGoal(
        {std::nextafter(fixture.tolerance_m,
                        std::numeric_limits<double>::infinity()),
         0.0, 0.0},
        target));
    EXPECT_TRUE(monitor.ReachedFinalGoal(
        {0.0, 0.0, -std::numbers::pi + 0.01},
        {0.0, 0.0, std::numbers::pi - 0.01}));
  }
}

TEST(ExecutionMonitorTest, ProgressWindowUsesAcceptedSegmentAndSteadyClock) {
  ExecutionMonitor monitor({.maximum_executable_path_points = 8U,
                            .position_tolerance_m = 0.3,
                            .yaw_tolerance_rad = std::numbers::pi / 16.0,
                            .stuck_window = 30s,
                            .minimum_progress_m = 0.2});
  const auto start = std::chrono::steady_clock::time_point{};
  const std::vector<Vec2> segment{{0.0, 0.0}, {1.0, 0.0}};
  monitor.ResetProgress(start, segment, {0.0, 0.0});
  EXPECT_FALSE(monitor.UpdateProgress(start + 29s, {0.0, 0.0}, false));
  EXPECT_FALSE(monitor.UpdateProgress(start + 30s, {0.2, 0.0}, false));
  EXPECT_FALSE(monitor.UpdateProgress(
      start + 59s,
      {std::nextafter(0.4, -std::numeric_limits<double>::infinity()), 0.0},
      false));
  EXPECT_FALSE(monitor.UpdateProgress(start + 60s, {0.4, 0.0}, false));
  EXPECT_TRUE(monitor.UpdateProgress(start + 90s, {0.4, 0.0}, false));
}

}  // namespace
}  // namespace lunar::pure_exploration_ros
