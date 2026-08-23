#include "lunar_pure_planner_ros/request_diagnostics.hpp"

#include <chrono>
#include <set>
#include <string>

#include <gtest/gtest.h>

namespace lunar::pure_planner_ros {
namespace {

using namespace std::chrono_literals;

TEST(RequestDiagnostics, EmitsOneExactTenKeyStatusAndTiming) {
  const lunar::pure_planning::PlanningResult result{
      .status = lunar::pure_planning::PlanningStatus::kNoPath,
      .reason_code = "ignored",
      .timing = {
      .global_elapsed = 1250us,
      .global_call_count = 2U,
      .local_elapsed = 3500us,
      .local_call_count = 3U,
      .total_elapsed = 6ms,
  }};
  const auto diagnostics = MakeRequestDiagnostics(
      "request-17", lunar::pure_planning::PlatformType::kLegged,
      lunar::pure_planning::EnvironmentMode::kLavaTube, result);

  ASSERT_EQ(diagnostics.status.size(), 1U);
  const auto& status = diagnostics.status.front();
  EXPECT_EQ(status.level, diagnostic_msgs::msg::DiagnosticStatus::WARN);

  std::set<std::string> keys;
  for (const auto& value : status.values) {
    keys.insert(value.key);
  }
  EXPECT_EQ(keys, (std::set<std::string>{
      "request_id", "platform_type", "environment_mode", "planning_outcome",
      "reason_code", "global_elapsed_ms", "global_call_count", "local_elapsed_ms",
      "local_call_count", "total_elapsed_ms"}));
  EXPECT_EQ(status.values.size(), 10U);
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "global_elapsed_ms"), "1.25");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "local_elapsed_ms"), "3.5");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "total_elapsed_ms"), "6");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "global_call_count"), "2");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "local_call_count"), "3");
}

TEST(RequestDiagnostics, AssignsLevelForEveryTypedStatus) {
  const lunar::pure_planning::PlannerCallTiming timing{
      .global_elapsed = 1ms,
      .global_call_count = 1U,
      .local_elapsed = 0ns,
      .local_call_count = 0U,
      .total_elapsed = 2ms,
  };
  struct Case final {
    lunar::pure_planning::PlanningStatus status;
    std::uint8_t level;
  };
  for (const auto expected : std::initializer_list<Case>{
           {lunar::pure_planning::PlanningStatus::kSuccess, diagnostic_msgs::msg::DiagnosticStatus::OK},
           {lunar::pure_planning::PlanningStatus::kGoalOutsideLocalMap, diagnostic_msgs::msg::DiagnosticStatus::WARN},
           {lunar::pure_planning::PlanningStatus::kNoPath, diagnostic_msgs::msg::DiagnosticStatus::WARN},
           {lunar::pure_planning::PlanningStatus::kTimedOut, diagnostic_msgs::msg::DiagnosticStatus::WARN},
           {lunar::pure_planning::PlanningStatus::kCanceled, diagnostic_msgs::msg::DiagnosticStatus::WARN},
           {lunar::pure_planning::PlanningStatus::kInvalidInput, diagnostic_msgs::msg::DiagnosticStatus::ERROR},
           {lunar::pure_planning::PlanningStatus::kPlannerError, diagnostic_msgs::msg::DiagnosticStatus::ERROR}}) {
    const lunar::pure_planning::PlanningResult result{
        .status = expected.status, .reason_code = "ignored", .timing = timing};
    const auto diagnostics = MakeRequestDiagnostics(
        "request", lunar::pure_planning::PlatformType::kHopper,
        lunar::pure_planning::EnvironmentMode::kLunarSurface, result);
    ASSERT_EQ(diagnostics.status.size(), 1U);
    EXPECT_EQ(diagnostics.status.front().level, expected.level);
    EXPECT_EQ(diagnostics.status.front().values.size(), 10U);
  }
}

}  // namespace
}  // namespace lunar::pure_planner_ros
