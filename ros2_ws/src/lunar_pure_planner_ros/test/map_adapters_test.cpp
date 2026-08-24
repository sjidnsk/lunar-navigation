#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_planner_ros/input_store.hpp"
#include "lunar_pure_planner_ros/map_adapters.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planner_ros {
namespace {

std_msgs::msg::Float32MultiArray Layer(std::vector<float> data) {
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

grid_map_msgs::msg::GridMap FourLayerMap() {
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

std_msgs::msg::Float32MultiArray Standard320Layer() {
  constexpr std::size_t kWidth = 320U;
  constexpr std::size_t kHeight = 320U;
  constexpr std::size_t kOffset = 7U;
  std_msgs::msg::Float32MultiArray layer;
  std_msgs::msg::MultiArrayDimension outer;
  outer.label = "column_index";
  outer.size = kHeight;
  outer.stride = kWidth * kHeight;
  std_msgs::msg::MultiArrayDimension inner;
  inner.label = "row_index";
  inner.size = kWidth;
  inner.stride = kWidth;
  layer.layout.dim = {outer, inner};
  layer.layout.data_offset = kOffset;
  layer.data.assign(kOffset + kWidth * kHeight, -999.F);
  return layer;
}

std::size_t StandardSourceIndex(const std::size_t x, const std::size_t y,
                                const std::size_t outer_start,
                                const std::size_t inner_start) {
  constexpr std::size_t kWidth = 320U;
  constexpr std::size_t kHeight = 320U;
  const std::size_t physical_row = (kWidth - 1U - x + outer_start) % kWidth;
  const std::size_t physical_column =
      (kHeight - 1U - y + inner_start) % kHeight;
  return 7U + physical_column * kWidth + physical_row;
}

grid_map_msgs::msg::GridMap Standard320Map() {
  constexpr std::size_t kWidth = 320U;
  constexpr std::size_t kHeight = 320U;
  grid_map_msgs::msg::GridMap map;
  map.header.frame_id = "odom";
  map.info.resolution = 0.2;
  map.info.length_x = static_cast<double>(kWidth) * map.info.resolution;
  map.info.length_y = static_cast<double>(kHeight) * map.info.resolution;
  map.info.pose.orientation.w = 1.0;
  map.layers = {"occupancy", "semantic_id", "elevation", "roughness"};
  map.data = {Standard320Layer(), Standard320Layer(), Standard320Layer(),
              Standard320Layer()};
  map.outer_start_index = 23U;
  map.inner_start_index = 41U;
  const auto set = [&](const std::size_t x, const std::size_t y,
                       const float occupancy, const float elevation) {
    const std::size_t index = StandardSourceIndex(
        x, y, map.outer_start_index, map.inner_start_index);
    map.data[0].data[index] = occupancy;
    map.data[2].data[index] = elevation;
  };
  set(0U, 0U, 0.11F, 11.F);
  set(7U, 31U, 0.22F, 22.F);
  set(319U, 319U, 0.33F, 33.F);
  return map;
}

TEST(MapAdapters, PreservesNativeGlobalOccupancyValues) {
  nav_msgs::msg::OccupancyGrid message;
  message.header.frame_id = "map";
  message.info.width = 5U;
  message.info.height = 1U;
  message.info.resolution = 1.0F;
  message.info.origin.orientation.w = 1.0;
  message.data = {-1, 0, 49, 50, 100};

  const auto adapted = AdaptGlobal(message);

  ASSERT_TRUE(adapted.value.has_value()) << adapted.reason_code;
  const auto& values = std::get<std::vector<std::int8_t>>(
      adapted.value->layers.at("occupancy").values);
  EXPECT_EQ(values, message.data);
}

TEST(MapAdapters, UnwrapsCircularLocalLayersAndIgnoresSemanticAndRoughness) {
  const auto adapted = AdaptLocal(FourLayerMap());

  ASSERT_TRUE(adapted.value.has_value()) << adapted.reason_code;
  EXPECT_EQ(adapted.value->frame_id, "odom");
  EXPECT_FALSE(adapted.value->HasLayer("semantic_id"));
  EXPECT_FALSE(adapted.value->HasLayer("roughness"));
  const auto& occupancy = std::get<std::vector<float>>(
      adapted.value->layers.at("occupancy").values);
  const auto& elevation = std::get<std::vector<float>>(
      adapted.value->layers.at("elevation").values);
  EXPECT_EQ(occupancy, (std::vector<float>{0.F, 2.F, 1.F, 3.F, 5.F, 4.F}));
  EXPECT_EQ(elevation,
            (std::vector<float>{100.F, 102.F, 101.F, 103.F, 105.F, 104.F}));
}

TEST(MapAdapters, UnwrapsStandardEigenCircular320MapIntoCoreRowMajorIndices) {
  const auto adapted = AdaptLocal(Standard320Map());

  ASSERT_TRUE(adapted.value.has_value()) << adapted.reason_code;
  ASSERT_EQ(adapted.value->width, 320U);
  ASSERT_EQ(adapted.value->height, 320U);
  const auto built = lunar::pure_planning::shared::MapSnapshot::Create(
      *adapted.value, lunar::pure_planning::shared::MapContract::kLocalElevation);
  ASSERT_TRUE(built.ok()) << built.reason_code;
  const auto occupancy = built.snapshot->FloatLayer("occupancy");
  const auto elevation = built.snapshot->FloatLayer("elevation");
  for (const auto [x, y, expected_occupancy, expected_elevation] : {
           std::tuple{0, 0, 0.11F, 11.F},
           std::tuple{7, 31, 0.22F, 22.F},
           std::tuple{319, 319, 0.33F, 33.F},
       }) {
    const std::size_t index = built.snapshot->Index({.x = x, .y = y});
    ASSERT_LT(index, occupancy.size());
    EXPECT_FLOAT_EQ(occupancy[index], expected_occupancy);
    EXPECT_FLOAT_EQ(elevation[index], expected_elevation);
  }
}

TEST(MapAdapters, RejectsUnindexableLocalLayoutButNotIndividualNanCells) {
  auto map = FourLayerMap();
  map.data[0].data[3] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(AdaptLocal(map).value.has_value());

  map.data[0].layout.dim[1].stride = 0U;
  const auto malformed = AdaptLocal(map);
  EXPECT_FALSE(malformed.value.has_value());
  EXPECT_EQ(malformed.reason_code, "INVALID_INPUT");
}

TEST(MapAdapters, RejectsNonstandardOrMalformedGridMapLayouts) {
  auto map = Standard320Map();
  map.data[0].layout.dim[0].stride = 320U;
  EXPECT_FALSE(AdaptLocal(map).value.has_value());

  map = Standard320Map();
  map.data[2].layout.dim[1].label = "inner";
  EXPECT_FALSE(AdaptLocal(map).value.has_value());
}

TEST(MapAdapters, RejectsWrongMapFramesAndNonAxisAlignedMapOrientation) {
  nav_msgs::msg::OccupancyGrid global;
  global.header.frame_id = "odom";
  global.info.width = 1U;
  global.info.height = 1U;
  global.info.resolution = 1.0F;
  global.info.origin.orientation.w = 1.0;
  global.data = {0};
  EXPECT_FALSE(AdaptGlobal(global).value.has_value());

  auto local = Standard320Map();
  local.header.frame_id = "map";
  EXPECT_FALSE(AdaptLocal(local).value.has_value());
  local = Standard320Map();
  local.info.pose.orientation.z = 0.1;
  EXPECT_FALSE(AdaptLocal(local).value.has_value());
}

TEST(MapAdapters, SurfaceRequiresGlobalButLavaTubeDoesNotReadIt) {
  InputSnapshot snapshot;
  snapshot.global_sequence = 11U;
  snapshot.local_sequence = 12U;
  snapshot.odometry_sequence = 13U;
  snapshot.tf_sequence = 14U;
  snapshot.local_map = std::make_shared<grid_map_msgs::msg::GridMap>(FourLayerMap());
  auto odometry = std::make_shared<nav_msgs::msg::Odometry>();
  odometry->header.frame_id = "odom";
  odometry->child_frame_id = "base_link";
  odometry->pose.pose.orientation.w = 1.0;
  snapshot.odometry = odometry;
  snapshot.map_from_odom = geometry_msgs::msg::TransformStamped{};
  snapshot.map_from_odom->header.frame_id = "map";
  snapshot.map_from_odom->child_frame_id = "odom";
  snapshot.map_from_odom->transform.rotation.w = 1.0;

  EXPECT_FALSE(AdaptSnapshot(lunar::pure_planning::EnvironmentMode::kLunarSurface,
                             snapshot)
                   .value.has_value());
  const auto lava = AdaptSnapshot(
      lunar::pure_planning::EnvironmentMode::kLavaTube, snapshot);
  ASSERT_TRUE(lava.value.has_value());
  EXPECT_EQ(lava.value->global_map_sequence, 11U);
  EXPECT_EQ(lava.value->local_map_sequence, 12U);
  EXPECT_EQ(lava.value->odometry_sequence, 13U);
  EXPECT_EQ(lava.value->tf_sequence, 14U);

  odometry->pose.pose.orientation.w = 0.0;
  EXPECT_FALSE(AdaptSnapshot(lunar::pure_planning::EnvironmentMode::kLavaTube,
                             snapshot)
                   .value.has_value());
}

}  // namespace
}  // namespace lunar::pure_planner_ros
