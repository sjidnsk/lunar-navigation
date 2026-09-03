#include "lunar_pure_exploration_ros/task_map_marker_builder.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include <lunar_pure_exploration_core/occupancy_grid.hpp>
#include <lunar_pure_exploration_core/task_raster.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace lunar::pure_exploration_ros {
namespace {

using lunar::pure_exploration::GridGeometry;
using lunar::pure_exploration::OccupancyGridView;
using lunar::pure_exploration::Polygon2;
using lunar::pure_exploration::TaskRaster;

const visualization_msgs::msg::Marker& FindMarker(
    const visualization_msgs::msg::MarkerArray& array, const char* ns) {
  const auto found = std::find_if(
      array.markers.begin(), array.markers.end(),
      [ns](const auto& marker) { return marker.ns == ns; });
  if (found == array.markers.end()) {
    throw std::logic_error{"missing task-map marker namespace"};
  }
  return *found;
}

TEST(TaskMapMarkerBuilder, UsesSpecifiedColorsForTaskCellsOnly) {
  const GridGeometry geometry{3U, 1U, 1.0, 0.0, 0.0, 0.0};
  const std::vector<std::int8_t> data{0, 100, -1};
  const OccupancyGridView map(geometry, data, 50);
  const TaskRaster raster = TaskRaster::Build(
      map, Polygon2{{{0.0, 0.0}, {3.0, 0.0}, {3.0, 1.0}, {0.0, 1.0}}});

  TaskMapMarkerBuilder builder;
  const auto markers = builder.Build(raster, "map");

  ASSERT_EQ(markers.markers.size(), 3U);
  const auto& known = FindMarker(markers, "task_known");
  const auto& blocked = FindMarker(markers, "task_blocked");
  const auto& unknown = FindMarker(markers, "task_unknown");
  EXPECT_EQ(known.type, visualization_msgs::msg::Marker::CUBE_LIST);
  EXPECT_EQ(blocked.type, visualization_msgs::msg::Marker::CUBE_LIST);
  EXPECT_EQ(unknown.type, visualization_msgs::msg::Marker::CUBE_LIST);
  EXPECT_EQ(known.points.size(), 1U);
  EXPECT_EQ(blocked.points.size(), 1U);
  EXPECT_EQ(unknown.points.size(), 1U);
  EXPECT_FLOAT_EQ(known.color.r, 176.0F / 255.0F);
  EXPECT_FLOAT_EQ(known.color.g, 176.0F / 255.0F);
  EXPECT_FLOAT_EQ(known.color.b, 176.0F / 255.0F);
  EXPECT_FLOAT_EQ(blocked.color.r, 0.0F);
  EXPECT_FLOAT_EQ(blocked.color.g, 0.0F);
  EXPECT_FLOAT_EQ(blocked.color.b, 0.0F);
  EXPECT_FLOAT_EQ(unknown.color.r, 74.0F / 255.0F);
  EXPECT_FLOAT_EQ(unknown.color.g, 74.0F / 255.0F);
  EXPECT_FLOAT_EQ(unknown.color.b, 74.0F / 255.0F);
}

}  // namespace
}  // namespace lunar::pure_exploration_ros
