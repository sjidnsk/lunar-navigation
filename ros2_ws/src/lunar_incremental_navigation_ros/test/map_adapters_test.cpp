#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_ros/input_store.hpp"
#include "lunar_incremental_navigation_ros/map_adapters.hpp"

namespace lunar::incremental_navigation_ros {
namespace {

[[nodiscard]] std_msgs::msg::Float32MultiArray Layer(
    std::vector<float> data) {
  std_msgs::msg::Float32MultiArray layer;
  std_msgs::msg::MultiArrayDimension outer;
  outer.label = "column_index";
  outer.size = 2U;
  outer.stride = 6U;
  std_msgs::msg::MultiArrayDimension inner;
  inner.label = "row_index";
  inner.size = 3U;
  inner.stride = 3U;
  layer.layout.dim = {outer, inner};
  layer.data = std::move(data);
  return layer;
}

[[nodiscard]] grid_map_msgs::msg::GridMap FourLayerMap() {
  grid_map_msgs::msg::GridMap map;
  map.header.frame_id = "odom";
  map.info.resolution = 0.2;
  map.info.length_x = 0.6;
  map.info.length_y = 0.4;
  map.info.pose.position.x = 3.0;
  map.info.pose.position.y = 4.0;
  map.info.pose.orientation.w = 1.0;
  map.layers = {"occupancy", "semantic_id", "elevation", "roughness"};
  map.data = {
      Layer({0.F, 1.F, 2.F, 3.F, 4.F, 5.F}),
      Layer({10.F, 11.F, 12.F, 13.F, 14.F, 15.F}),
      Layer({100.F, 101.F, 102.F, 103.F, 104.F, 105.F}),
      Layer({20.F, 21.F, 22.F, 23.F, 24.F, 25.F}),
  };
  map.outer_start_index = 1U;
  map.inner_start_index = 1U;
  return map;
}

[[nodiscard]] InputSnapshot SnapshotFor(
    grid_map_msgs::msg::GridMap message) {
  InputSnapshot input;
  input.local_map =
      std::make_shared<grid_map_msgs::msg::GridMap>(std::move(message));
  input.map_from_odom = geometry_msgs::msg::TransformStamped{};
  input.map_from_odom->header.frame_id = "map";
  input.map_from_odom->child_frame_id = "odom";
  input.map_from_odom->transform.translation.x = 7.0;
  input.map_from_odom->transform.rotation.w = 1.0;
  return input;
}

TEST(MapAdaptersTest,
     OwnsOnlyElevationAndUnwrapsCircularBufferIntoWorldEvidence) {
  auto map = FourLayerMap();
  map.header.stamp.sec = 123;
  auto input = SnapshotFor(std::move(map));
  input.map_from_odom->header.stamp.sec = 456;

  const auto adapted = AdaptLocalElevation(input);

  ASSERT_TRUE(adapted.value) << adapted.reason_code;
  EXPECT_EQ(adapted.value->geometry.frame_id, "odom");
  EXPECT_DOUBLE_EQ(adapted.value->geometry.resolution_m, 0.2);
  EXPECT_EQ(adapted.value->elevation_m,
            (std::vector<float>{100.F, 102.F, 101.F,
                                103.F, 105.F, 104.F}));
  EXPECT_EQ(adapted.value->map_from_source.parent_frame, "map");
  EXPECT_EQ(adapted.value->map_from_source.child_frame, "odom");
  EXPECT_DOUBLE_EQ(adapted.value->map_from_source.translation_m.x, 7.0);
  EXPECT_EQ(adapted.value->map_from_source.stamp.nanoseconds_since_epoch, 0);
  EXPECT_EQ(adapted.value->View().elevation_m.data(),
            adapted.value->elevation_m.data());

  lunar::incremental_navigation::PersistentElevationMap persistent;
  const auto update = persistent.Apply(adapted.value->View());
  ASSERT_EQ(update.status,
            lunar::incremental_navigation::ElevationUpdateResult::Status::
                kApplied);
  const auto projected = persistent.Snapshot();
  ASSERT_TRUE(projected);
  const auto first_cell = projected->ElevationAtWorld(9.8, 3.9);
  ASSERT_TRUE(first_cell);
  EXPECT_FLOAT_EQ(*first_cell, 100.F);
}

TEST(MapAdaptersTest, IgnoresMalformedNonElevationLayers) {
  auto map = FourLayerMap();
  map.data[0].layout.dim.clear();
  map.data[1].data.clear();
  map.data[3].layout.data_offset = 999U;

  const auto adapted = AdaptLocalElevation(SnapshotFor(std::move(map)));

  ASSERT_TRUE(adapted.value) << adapted.reason_code;
  EXPECT_EQ(adapted.value->elevation_m.size(), 6U);
}

TEST(MapAdaptersTest, RejectsMissingOrMalformedElevationLayer) {
  auto missing = FourLayerMap();
  missing.layers.erase(missing.layers.begin() + 2);
  missing.data.erase(missing.data.begin() + 2);
  EXPECT_FALSE(AdaptLocalElevation(SnapshotFor(std::move(missing))).value);

  auto malformed = FourLayerMap();
  malformed.data[2].layout.dim[1].stride = 0U;
  const auto adapted = AdaptLocalElevation(SnapshotFor(std::move(malformed)));
  EXPECT_FALSE(adapted.value);
  EXPECT_EQ(adapted.reason_code, "INVALID_INPUT");
}

TEST(MapAdaptersTest, RejectsMissingDirectTransformOrWrongSourceFrame) {
  auto no_transform = SnapshotFor(FourLayerMap());
  no_transform.map_from_odom.reset();
  EXPECT_FALSE(AdaptLocalElevation(no_transform).value);

  auto wrong_frame = SnapshotFor(FourLayerMap());
  auto wrong_frame_message = FourLayerMap();
  wrong_frame_message.header.frame_id = "map";
  wrong_frame = SnapshotFor(std::move(wrong_frame_message));
  EXPECT_FALSE(AdaptLocalElevation(wrong_frame).value);
}

TEST(MapAdaptersTest, PreservesNanAsUnknownElevationEvidence) {
  auto map = FourLayerMap();
  map.data[2].data[3] = std::numeric_limits<float>::quiet_NaN();

  const auto adapted = AdaptLocalElevation(SnapshotFor(std::move(map)));

  ASSERT_TRUE(adapted.value) << adapted.reason_code;
  EXPECT_TRUE(std::any_of(adapted.value->elevation_m.begin(),
                          adapted.value->elevation_m.end(),
                          [](const float value) { return std::isnan(value); }));
}

}  // namespace
}  // namespace lunar::incremental_navigation_ros
