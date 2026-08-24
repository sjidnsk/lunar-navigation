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
      "reason_code", "expanded_states", "has_best_cost", "best_cost", "has_reference",
      "latency_class", "snapshot_projection_elapsed_ms",
      "global_elapsed_ms", "global_call_count", "local_goal_elapsed_ms",
      "local_search_elapsed_ms", "local_elapsed_ms", "local_call_count",
      "certification_elapsed_ms", "output_elapsed_ms", "total_elapsed_ms"}));
  EXPECT_EQ(status.values.size(), 20U);
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "expanded_states"), "47");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "has_best_cost"), "true");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "has_reference"), "false");
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
}

TEST(RequestDiagnostics, EmitsEveryGridV1DiagnosticFromPlanningResult) {
  const lunar::pure_planning::PlanningResult result{
      .status = lunar::pure_planning::PlanningStatus::kSuccess,
      .reference = lunar::pure_planning::MotionReference{},
      .grid_v1 = {
          .active = true,
          .global_input_sequence = 10U,
          .local_input_sequence = 11U,
          .odometry_input_sequence = 12U,
          .traversability_revision = 13U,
          .publish_check_revision = 14U,
          .profile_hash = 15U,
          .canonical_resolution_m = 0.2,
          .map_origin_m = {.x = 1.5, .y = -2.5, .z = 3.5},
          .allocated_tiles = 16U,
          .estimated_map_bytes = 17U,
          .updated_cells = 18U,
          .dirty_tiles = 19U,
          .halo_recomputed_cells = 20U,
          .free_cells = 21U,
          .blocked_cells = 22U,
          .unknown_cells = 23U,
          .prior_conflicts = 24U,
          .global_route_reused = true,
          .global_expanded_states = 25U,
          .global_open_peak = 26U,
          .local_expanded_states = 27U,
          .local_open_peak = 28U,
          .local_candidate_count = 29U,
          .local_attempt_count = 30U,
          .selected_candidate_index = 31U,
          .raw_path_points = 32U,
          .shortcut_path_points = 33U,
          .resampled_path_points = 34U,
          .final_trajectory_points = 35U,
          .direction = "REVERSE",
          .forward_cost = 36.5,
          .reverse_cost = 37.5,
          .final_supercover_cells = 38U,
          .postprocess_mode = "RAW_GRID_FALLBACK",
          .map_fusion_elapsed = 1500us,
          .traversability_elapsed = 2500us,
          .postprocess_elapsed = 3500us,
      }};
  const auto diagnostics = MakeRequestDiagnostics(
      "v1-request", lunar::pure_planning::PlatformType::kWheeled,
      lunar::pure_planning::EnvironmentMode::kLunarSurface, result);

  ASSERT_EQ(diagnostics.status.size(), 1U);
  std::set<std::string> keys;
  for (const auto& value : diagnostics.status.front().values) {
    keys.insert(value.key);
  }
  EXPECT_EQ(keys, (std::set<std::string>{
      "request_id", "platform_type", "environment_mode", "planning_outcome",
      "reason_code", "expanded_states", "has_best_cost", "best_cost",
      "has_reference", "latency_class", "snapshot_projection_elapsed_ms",
      "global_elapsed_ms", "global_call_count", "local_goal_elapsed_ms",
      "local_search_elapsed_ms", "local_elapsed_ms", "local_call_count",
      "certification_elapsed_ms", "output_elapsed_ms", "total_elapsed_ms",
      "grid_v1_active", "global_input_sequence", "local_input_sequence",
      "odometry_input_sequence", "traversability_revision",
      "publish_check_revision", "profile_hash", "canonical_resolution_m",
      "map_origin_x_m", "map_origin_y_m", "map_origin_z_m", "allocated_tiles",
      "estimated_map_bytes", "updated_cells", "dirty_tiles",
      "halo_recomputed_cells", "free_cells", "blocked_cells", "unknown_cells",
      "prior_conflicts", "global_route_reused", "global_expanded_states",
      "global_open_peak", "local_expanded_states", "local_open_peak",
      "local_candidate_count", "local_attempt_count", "selected_candidate_index",
      "raw_path_points", "shortcut_path_points", "resampled_path_points",
      "final_trajectory_points", "direction", "forward_cost", "reverse_cost",
      "final_supercover_cells", "postprocess_mode", "map_fusion_elapsed_ms",
      "traversability_elapsed_ms", "postprocess_elapsed_ms"}));
  EXPECT_EQ(diagnostics.status.front().values.size(), 60U);
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "has_reference"), "true");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "global_input_sequence"), "10");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "canonical_resolution_m"), "0.2");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "map_origin_y_m"), "-2.5");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "prior_conflicts"), "24");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "local_open_peak"), "28");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "direction"), "REVERSE");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "final_supercover_cells"), "38");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "postprocess_mode"),
            "RAW_GRID_FALLBACK");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "map_fusion_elapsed_ms"), "1.5");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "traversability_elapsed_ms"), "2.5");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "postprocess_elapsed_ms"), "3.5");
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
