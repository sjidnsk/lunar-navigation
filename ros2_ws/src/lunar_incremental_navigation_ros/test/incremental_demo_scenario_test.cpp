#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_ros/incremental_demo_scenario.hpp"
#include "lunar_incremental_navigation_ros/input_store.hpp"
#include "lunar_incremental_navigation_ros/map_adapters.hpp"

namespace lunar::incremental_navigation_ros {
namespace {

[[nodiscard]] InputSnapshot SnapshotFor(
    grid_map_msgs::msg::GridMap observation) {
  InputSnapshot input;
  input.local_map =
      std::make_shared<grid_map_msgs::msg::GridMap>(std::move(observation));
  input.map_from_odom = geometry_msgs::msg::TransformStamped{};
  input.map_from_odom->header.frame_id = "map";
  input.map_from_odom->child_frame_id = "odom";
  input.map_from_odom->transform.rotation.w = 1.0;
  return input;
}

[[nodiscard]] float ElevationAt(
    const OwnedElevationEvidence& evidence, const double x_m,
    const double y_m) {
  const auto& geometry = evidence.geometry;
  const auto x = static_cast<std::size_t>(std::floor(
      (x_m - geometry.origin_m.x) / geometry.resolution_m));
  const auto y = static_cast<std::size_t>(std::floor(
      (y_m - geometry.origin_m.y) / geometry.resolution_m));
  return evidence.elevation_m[y * geometry.width + x];
}

TEST(IncrementalDemoScenario,
     PublishesOnlyTheTenMeterOneHundredTwentyDegreeObservationSector) {
  // This catches a regression to a full square input map, a 90 degree sensor,
  // or a range that silently extends beyond the documented 10 m demo sensor.
  IncrementalDemoScenario scenario({.fine_resolution_m = 0.2,
                                    .local_window_size_m = 24.0,
                                    .task_size_m = 24.0});
  const auto adapted = AdaptLocalElevation(
      SnapshotFor(scenario.MakeLocalObservation({.x_m = 0.0,
                                                 .y_m = 0.0,
                                                 .yaw_rad = 0.0})));

  ASSERT_TRUE(adapted.value) << adapted.reason_code;
  const auto& evidence = *adapted.value;
  EXPECT_TRUE(std::isfinite(ElevationAt(evidence, 4.1, 5.9)));
  EXPECT_TRUE(std::isnan(ElevationAt(evidence, 0.1, 4.9)));
  EXPECT_TRUE(std::isnan(ElevationAt(evidence, 10.1, 0.1)));

  const auto local_window =
      scenario.MakeLocalWindow({.x_m = 0.0, .y_m = 0.0, .yaw_rad = 0.0});
  ASSERT_EQ(local_window.markers.size(), 1U);
  const auto& marker = local_window.markers.front();
  ASSERT_GE(marker.points.size(), 3U);
  EXPECT_DOUBLE_EQ(marker.points.front().x, 0.0);
  EXPECT_DOUBLE_EQ(marker.points.front().y, 0.0);
  EXPECT_NEAR(marker.points[1U].x, 5.0, 1.0e-6);
  EXPECT_NEAR(marker.points[1U].y, -8.6602540378, 1.0e-6);
  EXPECT_DOUBLE_EQ(marker.points.back().x, 0.0);
  EXPECT_DOUBLE_EQ(marker.points.back().y, 0.0);
}

}  // namespace
}  // namespace lunar::incremental_navigation_ros
