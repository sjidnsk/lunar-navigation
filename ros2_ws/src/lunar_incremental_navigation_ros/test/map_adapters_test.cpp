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


[[nodiscard]] grid_map_msgs::msg::GridMap TerrainMap() {
  auto map = FourLayerMap();
  map.layers.insert(map.layers.end(), {"terrain_relief", "terrain_complete",
      "terrain_slope", "terrain_positive_rise"});
  map.data.insert(map.data.end(), {Layer({0, 1, 2, 3, 4, 5}),
      Layer({1, 1, 1, 1, 1, 1}), Layer({0, .1F, .2F, .3F, .4F, .5F}),
      Layer({0, .01F, .02F, .03F, .04F, .05F})});
  return map;
}

TEST(MapAdaptersTest, TerrainGroupUnwrapsWithElevationAndSupportsQuarterTurn) {
  auto input = SnapshotFor(TerrainMap());
  input.map_from_odom->transform.rotation.w = std::sqrt(0.5);
  input.map_from_odom->transform.rotation.z = std::sqrt(0.5);
  auto adapted = AdaptLocalElevation(input);
  ASSERT_TRUE(adapted.value);
  ASSERT_EQ(adapted.value->terrain_measurements.size(), 6U);
  const std::vector<double> expected{0, 2, 1, 3, 5, 4};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_TRUE(adapted.value->terrain_measurements[i].center_known);
    EXPECT_TRUE(adapted.value->terrain_measurements[i].neighborhood_complete);
    EXPECT_DOUBLE_EQ(adapted.value->terrain_measurements[i].relief_m, expected[i]);
  }
  lunar::incremental_navigation::PersistentElevationMap persistent;
  ASSERT_EQ(persistent.Apply(adapted.value->View()).status,
      lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
  const auto stored = persistent.Snapshot()->TerrainMeasurementsAt({-1, 0});
  ASSERT_TRUE(stored);
  EXPECT_DOUBLE_EQ(stored->relief_m, 0);
}

TEST(MapAdaptersTest, TerrainGroupIsAtomicAndHeightOnlyStillSupported) {
  auto partial = TerrainMap();
  partial.layers.pop_back(); partial.data.pop_back();
  EXPECT_FALSE(AdaptLocalElevation(SnapshotFor(partial)).value);
  auto malformed = TerrainMap();
  malformed.data.back().layout.dim.clear();
  EXPECT_FALSE(AdaptLocalElevation(SnapshotFor(malformed)).value);
  auto bad_value = TerrainMap();
  bad_value.data[5].data[0] = .5F;
  EXPECT_FALSE(AdaptLocalElevation(SnapshotFor(bad_value)).value);
  auto input = SnapshotFor(TerrainMap());
  input.map_from_odom->transform.rotation.w = std::cos(.2);
  input.map_from_odom->transform.rotation.x = std::sin(.2);
  EXPECT_FALSE(AdaptLocalElevation(input).value);
  input.local_map = std::make_shared<grid_map_msgs::msg::GridMap>(FourLayerMap());
  auto height_only = AdaptLocalElevation(input);
  ASSERT_TRUE(height_only.value);
  EXPECT_TRUE(height_only.value->terrain_measurements.empty());
  lunar::incremental_navigation::PersistentElevationMap persistent;
  EXPECT_EQ(persistent.Apply(height_only.value->View()).status,
      lunar::incremental_navigation::ElevationUpdateResult::Status::kApplied);
}


TEST(MapAdaptersTest, TerrainLayersRespectIndependentLayoutsAndAbsentCells) {
  auto map = TerrainMap();
  // Same logical relief layer in row-major layout with a nonzero data offset.
  auto& relief = map.data[4];
  relief.layout.dim[0].label = "row_index";
  relief.layout.dim[0].size = 3;
  relief.layout.dim[1].label = "column_index";
  relief.layout.dim[1].size = 2;
  relief.layout.dim[1].stride = 2;
  relief.layout.data_offset = 1;
  relief.data = {99, 0, 3, 1, 4, 2, 5};
  auto adapted = AdaptLocalElevation(SnapshotFor(map));
  ASSERT_TRUE(adapted.value);
  EXPECT_DOUBLE_EQ(adapted.value->terrain_measurements[1].relief_m, 2);
  EXPECT_DOUBLE_EQ(adapted.value->terrain_measurements[4].relief_m, 5);

  map = TerrainMap();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  map.data[2].data[0] = nan;
  for (std::size_t layer = 4; layer < 8; ++layer) map.data[layer].data[0] = nan;
  map.data[5].data[1] = 0;  // incomplete measured neighborhood is still evidence
  adapted = AdaptLocalElevation(SnapshotFor(map));
  ASSERT_TRUE(adapted.value);
  EXPECT_FALSE(adapted.value->terrain_measurements[0].center_known);
  EXPECT_TRUE(adapted.value->terrain_measurements[2].center_known);
  EXPECT_FALSE(adapted.value->terrain_measurements[2].neighborhood_complete);
  map.data[4].data[0] = 0;  // a partial scalar tuple must not be accepted
  EXPECT_FALSE(AdaptLocalElevation(SnapshotFor(map)).value);
  map = TerrainMap();
  map.layers.push_back("terrain_slope"); map.data.push_back(map.data[6]);
  EXPECT_FALSE(AdaptLocalElevation(SnapshotFor(map)).value);
}

}  // namespace
}  // namespace lunar::incremental_navigation_ros
