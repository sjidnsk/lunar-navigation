#include <chrono>
#include <string>

#include <gtest/gtest.h>

#include "lunar_pure_planner_ros/lunar_surface_reporter.hpp"

namespace lunar::pure_planner_ros {
namespace {

nav_msgs::msg::Path ThreeFourFivePath() {
  nav_msgs::msg::Path path;
  path.header.frame_id = "map";
  path.poses.resize(3U);
  path.poses[0].pose.position.x = 0.0;
  path.poses[0].pose.position.y = 0.0;
  path.poses[1].pose.position.x = 3.0;
  path.poses[1].pose.position.y = 4.0;
  path.poses[2].pose.position.x = 6.0;
  path.poses[2].pose.position.y = 8.0;
  return path;
}

TEST(LunarSurfaceReporter, ComputesGeometricPathLength) {
  EXPECT_DOUBLE_EQ(PathLengthMetres(ThreeFourFivePath()), 10.0);
  EXPECT_DOUBLE_EQ(PathLengthMetres(nav_msgs::msg::Path{}), 0.0);
}

TEST(LunarSurfaceReporter, FormatsOneConciseSuccessfulPlanningLine) {
  PlanningSummary summary;
  summary.sequence = 3U;
  summary.start_x_m = -349.5;
  summary.start_y_m = 0.5;
  summary.goal_x_m = 350.5;
  summary.goal_y_m = 0.5;
  summary.planning_time = std::chrono::duration<double, std::milli>{1284.6};
  summary.global_path_length_m = 712.3;
  summary.local_path_length_m = 12.1;
  summary.reason_code = "PLAN_FOUND";
  summary.success = true;

  EXPECT_EQ(
      FormatPlanningSummary(summary),
      "----------------------------\n"
      "[PLAN 003] OK start=(-349.50,0.50) goal=(350.50,0.50) "
      "time=1284.6 ms path=712.3 m local=12.1 m reason=PLAN_FOUND");
}

TEST(LunarSurfaceReporter, UsesNotAvailableLengthsForFailure) {
  PlanningSummary summary;
  summary.sequence = 4U;
  summary.start_x_m = -20.0;
  summary.start_y_m = 15.0;
  summary.goal_x_m = 84.0;
  summary.goal_y_m = 91.0;
  summary.planning_time = std::chrono::duration<double, std::milli>{3000.0};
  summary.reason_code = "TIMEOUT";
  summary.success = false;

  EXPECT_EQ(
      FormatPlanningSummary(summary),
      "----------------------------\n"
      "[PLAN 004] FAIL start=(-20.00,15.00) goal=(84.00,91.00) "
      "time=3000.0 ms path=N/A local=N/A reason=TIMEOUT");
}

TEST(LunarSurfaceReporter, WaitsForPathsWhenDiagnosticsArriveFirst) {
  PlanningReportState state;
  state.Begin(-1.0, 2.0, 30.0, 40.0);
  state.SetResult(true, "PLAN_FOUND",
                  std::chrono::duration<double, std::milli>{25.0});
  EXPECT_FALSE(state.TakeReadySummary().has_value());

  state.SetGlobalPath(ThreeFourFivePath());
  EXPECT_FALSE(state.TakeReadySummary().has_value());
  state.SetLocalPath(ThreeFourFivePath());
  const auto summary = state.TakeReadySummary();

  ASSERT_TRUE(summary.has_value());
  EXPECT_TRUE(summary->success);
  EXPECT_DOUBLE_EQ(*summary->global_path_length_m, 10.0);
  EXPECT_DOUBLE_EQ(*summary->local_path_length_m, 10.0);
  EXPECT_FALSE(state.TakeReadySummary().has_value());
}

}  // namespace
}  // namespace lunar::pure_planner_ros
