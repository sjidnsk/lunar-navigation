#include "lunar_pure_planner_ros/request_diagnostics.hpp"

#include <chrono>
#include <set>
#include <string>

#include <gtest/gtest.h>

namespace lunar::pure_planner_ros {
namespace {

using namespace std::chrono_literals;

TEST(RequestDiagnostics, EmitsStatusLatencyClassAndEveryPhaseTiming) {
  const lunar::pure_planning::PlanningResult result{
      .status = lunar::pure_planning::PlanningStatus::kNoPath,
      .reason_code = "ignored",
      .timing = {
      .snapshot_projection_elapsed = 250us,
      .global_elapsed = 1250us,
      .global_call_count = 2U,
      .local_goal_elapsed = 750us,
      .local_search_elapsed = 3500us,
      .local_elapsed = 3500us,
      .local_call_count = 3U,
      .certification_elapsed = 400us,
      .output_elapsed = 100us,
      .total_elapsed = 6ms,
  },
      .expanded_states = 47U,
      .best_cost = 12.25};
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
      "reason_code", "expanded_states", "has_best_cost", "best_cost",
      "latency_class", "snapshot_projection_elapsed_ms",
      "global_elapsed_ms", "global_call_count", "local_goal_elapsed_ms",
      "local_search_elapsed_ms", "local_elapsed_ms", "local_call_count",
      "certification_elapsed_ms", "output_elapsed_ms", "total_elapsed_ms",
      "wheel_metrics_available"}));
  EXPECT_EQ(status.values.size(), 20U);
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "expanded_states"), "47");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "has_best_cost"), "true");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "best_cost"), "12.25");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "latency_class"), "TARGET_MET");
  EXPECT_EQ(FindDiagnosticValue(diagnostics,
                                "snapshot_projection_elapsed_ms"),
            "0.25");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "global_elapsed_ms"), "1.25");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "local_goal_elapsed_ms"),
            "0.75");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "local_search_elapsed_ms"),
            "3.5");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "local_elapsed_ms"), "3.5");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "certification_elapsed_ms"),
            "0.4");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "output_elapsed_ms"), "0.1");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "total_elapsed_ms"), "6");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "global_call_count"), "2");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "local_call_count"), "3");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "wheel_metrics_available"), "false");
}

TEST(RequestDiagnostics, EmitsConditionalWheelMetricsWhenAvailable) {
  const lunar::pure_planning::PlanningResult result{
      .status = lunar::pure_planning::PlanningStatus::kTimedOut,
      .wheel_metrics = lunar::pure_planning::WheelPlanningMetrics{
          .expanded_states = 4U,
          .edge_validation_evaluations = 11U,
          .sweep_cell_checks = 29U,
          .used_narrow_resolution = true,
          .finest_xy_key_resolution_m = 0.125,
          .cost_components = {1.0, 2.0, 3.0, 4.0, 5.0},
      },
  };
  const auto diagnostics = MakeRequestDiagnostics(
      "wheel", lunar::pure_planning::PlatformType::kWheeled,
      lunar::pure_planning::EnvironmentMode::kLunarSurface, result);

  EXPECT_EQ(FindDiagnosticValue(diagnostics, "wheel_metrics_available"), "true");
  EXPECT_EQ(FindDiagnosticValue(
                diagnostics, "wheel_edge_validation_evaluations"), "11");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "wheel_sweep_cell_checks"), "29");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "wheel_used_narrow_resolution"),
            "true");
  EXPECT_EQ(FindDiagnosticValue(diagnostics,
                                "wheel_finest_xy_key_resolution_m"),
            "0.125");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "wheel_cost_component_4"), "5");
}

TEST(RequestDiagnostics, OmitsConditionalWheelMetricsBeforeGraphCreation) {
  const auto diagnostics = MakeRequestDiagnostics(
      "invalid", lunar::pure_planning::PlatformType::kWheeled,
      lunar::pure_planning::EnvironmentMode::kLunarSurface,
      lunar::pure_planning::PlanningResult{
          .status = lunar::pure_planning::PlanningStatus::kInvalidInput});

  EXPECT_EQ(FindDiagnosticValue(diagnostics, "wheel_metrics_available"), "false");
  EXPECT_TRUE(FindDiagnosticValue(
      diagnostics, "wheel_edge_validation_evaluations").empty());
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
    EXPECT_EQ(diagnostics.status.front().values.size(), 20U);
  }
}

TEST(RequestDiagnostics, UsesLaterLatencyClassAtEveryExactMilestone) {
  struct Case final {
    std::chrono::milliseconds elapsed;
    const char* latency_class;
  };
  for (const Case& test_case : {
           Case{999ms, "TARGET_MET"}, Case{1000ms, "TARGET_MISSED"},
           Case{1999ms, "TARGET_MISSED"}, Case{2000ms, "SLA_MISSED"},
           Case{2999ms, "SLA_MISSED"}, Case{3000ms, "HARD_TIMEOUT"},
       }) {
    const lunar::pure_planning::PlanningResult result{
        .status = lunar::pure_planning::PlanningStatus::kSuccess,
        .timing = {.total_elapsed = test_case.elapsed},
    };
    const auto diagnostics = MakeRequestDiagnostics(
        "request", lunar::pure_planning::PlatformType::kWheeled,
        lunar::pure_planning::EnvironmentMode::kLunarSurface, result);
    EXPECT_EQ(FindDiagnosticValue(diagnostics, "latency_class"),
              test_case.latency_class)
        << test_case.elapsed.count();
  }
}

}  // namespace
}  // namespace lunar::pure_planner_ros
