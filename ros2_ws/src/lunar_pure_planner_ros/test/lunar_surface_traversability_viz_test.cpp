#include <algorithm>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>
#include <grid_map_msgs/msg/grid_map.hpp>

#include "lunar_pure_planner_ros/map_adapters.hpp"
#include "lunar_pure_planner_ros/lunar_surface_traversability_viz.hpp"

namespace lunar::pure_planner_ros {
namespace {

grid_map_msgs::msg::GridMap MakeTwoByTwoTraversability() {
  grid_map_msgs::msg::GridMap input;
  input.header.frame_id = "odom";
  input.info.resolution = 1.0;
  input.info.length_x = 2.0;
  input.info.length_y = 2.0;
  input.info.pose.position.x = 10.0;
  input.info.pose.position.y = 20.0;
  input.info.pose.orientation.w = 1.0;
  input.layers = {"traversability"};
  input.basic_layers = input.layers;
  input.data.resize(1U);
  auto& layer = input.data.front();
  layer.layout.dim.resize(2U);
  layer.layout.dim[0].label = "column_index";
  layer.layout.dim[0].size = 2U;
  layer.layout.dim[0].stride = 4U;
  layer.layout.dim[1].label = "row_index";
  layer.layout.dim[1].size = 2U;
  layer.layout.dim[1].stride = 2U;
  // GridMap's physical buffer is reversed in both axes.  Logical rows are:
  //   [1.0, 0.0]
  //   [NaN, 0.25]
  layer.data = {0.25F, std::numeric_limits<float>::quiet_NaN(), 0.0F, 1.0F};
  return input;
}

TEST(LunarSurfaceTraversabilityViz,
     ConvertsPassabilityToCostWhilePreservingUnknownAndGeometry) {
  const auto output = MakeTraversabilityCostmap(MakeTwoByTwoTraversability());

  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->header.frame_id, "odom");
  EXPECT_EQ(output->info.width, 2U);
  EXPECT_EQ(output->info.height, 2U);
  EXPECT_FLOAT_EQ(output->info.resolution, 1.0F);
  EXPECT_DOUBLE_EQ(output->info.origin.position.x, 9.0);
  EXPECT_DOUBLE_EQ(output->info.origin.position.y, 19.0);
  EXPECT_GT(output->info.origin.position.z, 0.0);
  EXPECT_EQ(output->data, (std::vector<std::int8_t>{0, 100, -1, 75}));
}

TEST(LunarSurfaceTraversabilityViz, RejectsMissingTraversabilityLayer) {
  auto input = MakeTwoByTwoTraversability();
  input.layers.front() = "roughness";

  EXPECT_FALSE(MakeTraversabilityCostmap(input).has_value());
}

TEST(LunarSurfaceTraversabilityViz,
     BuildsAOneKilometreGlobalInputAcceptedByTraversabilityCore) {
  const auto scenario = BuildLunarSurfaceScenario(20260823U);

  const auto input = MakeGlobalTraversabilityInput(scenario);
  const auto adapted = AdaptLocal(input);

  EXPECT_EQ(input.header.frame_id, "odom");
  EXPECT_DOUBLE_EQ(input.info.length_x, 1000.0);
  EXPECT_DOUBLE_EQ(input.info.length_y, 1000.0);
  EXPECT_DOUBLE_EQ(input.info.resolution, 1.0);
  ASSERT_TRUE(adapted.value.has_value());
  EXPECT_EQ(adapted.value->width, 1000U);
  EXPECT_EQ(adapted.value->height, 1000U);
}

TEST(LunarSurfaceTraversabilityViz,
     BuildsContinuousFreeBackgroundAndSparseClassicOverlay) {
  nav_msgs::msg::OccupancyGrid obstacles;
  obstacles.header.frame_id = "odom";
  obstacles.info.resolution = 1.0F;
  obstacles.info.width = 2U;
  obstacles.info.height = 2U;
  obstacles.info.origin.position.x = 9.0;
  obstacles.info.origin.position.y = 19.0;
  obstacles.info.origin.orientation.w = 1.0;
  obstacles.data = {0, 100, -1, 0};
  const auto costmap = MakeTraversabilityCostmap(MakeTwoByTwoTraversability());
  ASSERT_TRUE(costmap.has_value());

  const auto markers = MakeClassicLocalOverlayMarkers(
      obstacles, *costmap, 9.5, 19.5);

  ASSERT_TRUE(markers.has_value());
  ASSERT_EQ(markers->markers.size(), 5U);
  const auto& free_background = markers->markers[0];
  const auto& risk_marker = markers->markers[1];
  const auto& unknown_marker = markers->markers[2];
  const auto& disconnected_marker = markers->markers[3];
  const auto& obstacle_marker = markers->markers[4];
  EXPECT_EQ(free_background.type, visualization_msgs::msg::Marker::CUBE);
  EXPECT_GT(free_background.color.g, free_background.color.r);
  EXPECT_TRUE(free_background.points.empty());
  EXPECT_EQ(obstacle_marker.header.frame_id, "map");
  EXPECT_EQ(obstacle_marker.type,
            visualization_msgs::msg::Marker::CUBE_LIST);
  ASSERT_EQ(obstacle_marker.points.size(), 1U);
  EXPECT_DOUBLE_EQ(obstacle_marker.points[0].x, 10.5);
  EXPECT_DOUBLE_EQ(obstacle_marker.points[0].y, 19.5);
  EXPECT_FLOAT_EQ(obstacle_marker.color.r, 0.05F);
  ASSERT_EQ(risk_marker.points.size(), 1U);
  ASSERT_EQ(risk_marker.colors.size(), 1U);
  EXPECT_DOUBLE_EQ(risk_marker.points[0].x, 10.5);
  EXPECT_DOUBLE_EQ(risk_marker.points[0].y, 20.5);
  EXPECT_GT(risk_marker.colors[0].r, risk_marker.colors[0].g);
  ASSERT_EQ(unknown_marker.points.size(), 1U);
  EXPECT_DOUBLE_EQ(unknown_marker.points[0].x, 9.5);
  EXPECT_DOUBLE_EQ(unknown_marker.points[0].y, 20.5);
  EXPECT_TRUE(disconnected_marker.points.empty());
}

TEST(LunarSurfaceTraversabilityViz,
     BuildsOneContinuousGlobalBackgroundAndOnlyTilesObstacles) {
  const auto scenario = BuildLunarSurfaceScenario(20260823U);
  const auto markers = MakeClassicGlobalObstacleMarkers(scenario);

  ASSERT_EQ(markers.markers.size(), 2U);
  EXPECT_EQ(markers.markers[0].type, visualization_msgs::msg::Marker::CUBE);
  EXPECT_DOUBLE_EQ(markers.markers[0].scale.x, 1000.0);
  EXPECT_DOUBLE_EQ(markers.markers[0].scale.y, 1000.0);
  EXPECT_EQ(markers.markers[1].type,
            visualization_msgs::msg::Marker::CUBE_LIST);
  const auto occupied = static_cast<std::size_t>(std::count_if(
      scenario.occupancy.begin(), scenario.occupancy.end(),
      [](const std::int8_t value) { return value >= 50; }));
  EXPECT_EQ(markers.markers[1].points.size(), occupied);
  EXPECT_LT(markers.markers[1].points.size(), scenario.occupancy.size());
}

TEST(LunarSurfaceTraversabilityViz,
     BuildsGlobalTraversabilityWithTheSamePaletteAsTheLocalOverlay) {
  LunarSurfaceScenario scenario;
  scenario.width = 2U;
  scenario.height = 2U;
  scenario.resolution_m = 1.0;
  scenario.origin_x_m = 9.0;
  scenario.origin_y_m = 19.0;
  scenario.occupancy = {0, 100, 0, 0};
  scenario.elevation_m = {-1.0F, 0.0F, 1.0F, 2.0F};

  nav_msgs::msg::OccupancyGrid costmap;
  costmap.header.frame_id = "map";
  costmap.info.resolution = 1.0F;
  costmap.info.width = 2U;
  costmap.info.height = 2U;
  costmap.info.origin.position.x = 9.0;
  costmap.info.origin.position.y = 19.0;
  costmap.info.origin.orientation.w = 1.0;
  costmap.data = {0, 100, -1, 50};

  nav_msgs::msg::OccupancyGrid local_obstacles = costmap;
  local_obstacles.data = {0, 100, -1, 0};
  const double map_base_z = ClassicMapBaseZ(scenario);
  const auto global = MakeClassicGlobalTraversabilityMarkers(
      scenario, costmap, 9.5, 19.5);
  const auto local = MakeClassicLocalOverlayMarkers(
      local_obstacles, costmap, 9.5, 19.5, map_base_z);

  ASSERT_TRUE(global.has_value());
  ASSERT_TRUE(local.has_value());
  ASSERT_EQ(global->markers.size(), 5U);
  ASSERT_EQ(local->markers.size(), 5U);
  const auto& global_background = global->markers[0];
  const auto& global_risk = global->markers[1];
  const auto& global_unknown = global->markers[2];
  const auto& global_disconnected = global->markers[3];
  const auto& global_obstacles = global->markers[4];
  const auto& local_risk = local->markers[1];
  const auto& local_unknown = local->markers[2];
  const auto& local_disconnected = local->markers[3];

  EXPECT_GT(global_background.color.g, global_background.color.r);
  ASSERT_EQ(global_risk.points.size(), 1U);
  ASSERT_EQ(global_risk.colors.size(), 1U);
  ASSERT_EQ(local_risk.colors.size(), 1U);
  EXPECT_EQ(global_risk.colors[0], local_risk.colors[0]);
  EXPECT_EQ(global_unknown.color, local_unknown.color);
  EXPECT_EQ(global_disconnected.color, local_disconnected.color);
  EXPECT_EQ(global_unknown.points.size(), 1U);
  EXPECT_EQ(global_obstacles.points.size(), 1U);

  const double minimum_elevation = -1.0;
  const auto marker_top_z = [](const visualization_msgs::msg::Marker& marker) {
    if (marker.points.empty()) {
      return marker.pose.position.z + 0.5 * marker.scale.z;
    }
    double top_z = -std::numeric_limits<double>::infinity();
    for (const auto& point : marker.points) {
      top_z = std::max(top_z, marker.pose.position.z + point.z +
                                  0.5 * marker.scale.z);
    }
    return top_z;
  };
  for (const auto& marker : global->markers) {
    EXPECT_LT(marker_top_z(marker), minimum_elevation);
  }
  for (const auto& marker : local->markers) {
    EXPECT_LT(marker_top_z(marker), minimum_elevation);
  }
}

TEST(LunarSurfaceTraversabilityViz,
     MarksSafeCellsOutsideTheSeedConnectedComponentAsUnreachable) {
  LunarSurfaceScenario scenario;
  scenario.width = 3U;
  scenario.height = 3U;
  scenario.resolution_m = 1.0;
  scenario.origin_x_m = 0.0;
  scenario.origin_y_m = 0.0;
  scenario.occupancy = {
      0, 100, 0,
      0, 100, 0,
      0, 100, 0,
  };
  scenario.elevation_m.assign(9U, 0.0F);

  nav_msgs::msg::OccupancyGrid costmap;
  costmap.header.frame_id = "map";
  costmap.info.resolution = 1.0F;
  costmap.info.width = 3U;
  costmap.info.height = 3U;
  costmap.info.origin.orientation.w = 1.0;
  costmap.data = {
      0, 100, 0,
      0, 100, 0,
      0, 100, 0,
  };
  nav_msgs::msg::OccupancyGrid obstacles = costmap;

  const auto global = MakeClassicGlobalTraversabilityMarkers(
      scenario, costmap, 0.5, 1.5);
  const auto local = MakeClassicLocalOverlayMarkers(
      obstacles, costmap, 0.5, 1.5, ClassicMapBaseZ(scenario));

  ASSERT_TRUE(global.has_value());
  ASSERT_TRUE(local.has_value());
  ASSERT_EQ(global->markers.size(), 5U);
  ASSERT_EQ(local->markers.size(), 5U);
  const auto& global_disconnected = global->markers[3];
  const auto& local_disconnected = local->markers[3];
  EXPECT_EQ(global_disconnected.text, "safe but unreachable");
  EXPECT_EQ(local_disconnected.text, "safe but unreachable");
  ASSERT_EQ(global_disconnected.points.size(), 3U);
  ASSERT_EQ(local_disconnected.points.size(), 3U);
  EXPECT_EQ(global_disconnected.color, local_disconnected.color);
  for (const auto& point : global_disconnected.points) {
    EXPECT_DOUBLE_EQ(point.x, 2.5);
  }
}

TEST(LunarSurfaceTraversabilityViz,
     UsesTheMajorityConnectivityClassAsTheContinuousBackground) {
  LunarSurfaceScenario scenario;
  scenario.width = 4U;
  scenario.height = 3U;
  scenario.resolution_m = 1.0;
  scenario.origin_x_m = 0.0;
  scenario.origin_y_m = 0.0;
  scenario.occupancy = {
      0, 100, 0, 0,
      0, 100, 0, 0,
      0, 100, 0, 0,
  };
  scenario.elevation_m.assign(12U, 0.0F);

  nav_msgs::msg::OccupancyGrid costmap;
  costmap.header.frame_id = "map";
  costmap.info.resolution = 1.0F;
  costmap.info.width = 4U;
  costmap.info.height = 3U;
  costmap.info.origin.orientation.w = 1.0;
  costmap.data = {
      0, 100, 0, 0,
      0, 100, 0, 0,
      0, 100, 0, 0,
  };

  const auto markers = MakeClassicGlobalTraversabilityMarkers(
      scenario, costmap, 0.5, 1.5);

  ASSERT_TRUE(markers.has_value());
  ASSERT_EQ(markers->markers.size(), 5U);
  const auto& background = markers->markers[0];
  const auto& connectivity_exceptions = markers->markers[3];
  EXPECT_FLOAT_EQ(background.color.r, 0.50F);
  EXPECT_FLOAT_EQ(background.color.g, 0.57F);
  EXPECT_EQ(connectivity_exceptions.text, "reachable safe");
  EXPECT_EQ(connectivity_exceptions.points.size(), 3U);
  EXPECT_LT(connectivity_exceptions.points.size(), 6U);
  EXPECT_GT(connectivity_exceptions.color.g,
            connectivity_exceptions.color.r);
}

}  // namespace
}  // namespace lunar::pure_planner_ros
