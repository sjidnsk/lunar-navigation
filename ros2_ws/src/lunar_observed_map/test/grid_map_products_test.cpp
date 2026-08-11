#include "lunar_observed_map/grid_map_products.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <lunar_planner_ros/grid_map_adapter.hpp>

#include <gtest/gtest.h>

namespace lunar::observed_map {
namespace {

using lunar::planning::ros::GridMapAdapter;

ObservedPatch RowPatch(
    const std::int64_t x, const std::int64_t y,
    const std::vector<double>& elevations,
    const std::vector<std::uint8_t>& valid = {}) {
  return ObservedPatch{
      .session_id = "session-a",
      .simulation_time_ns = 1'000'000'000LL,
      .cell_zero = {x, y},
      .width = elevations.size(),
      .height = 1U,
      .elevation_m = elevations,
      .valid = valid.empty()
          ? std::vector<std::uint8_t>(elevations.size(), 1U)
          : valid,
  };
}

SparseObservedMap OneCellMap(
    const GridCellIndex cell = {0, 0}, const double elevation = 5.0) {
  SparseObservedMap map;
  map.ResetSession("session-a");
  EXPECT_TRUE(map.Fuse(RowPatch(cell.x, cell.y, {elevation})).accepted);
  return map;
}

ProductRequest Request(
    const MapPoint2D robot_map = {0.1, 0.1},
    const std::optional<MapPoint2D> goal = std::nullopt) {
  return ProductRequest{
      .robot_map = robot_map,
      .robot_odom = robot_map,
      .l0_origin_map = {},
      .goal_map = goal,
      .simulation_time_ns = 2'000'000'000LL,
  };
}

const std::vector<float>& FloatLayer(
    const lunar::planning::GridMap& map, const std::string& name) {
  return std::get<std::vector<float>>(map.layers.at(name).values);
}

const std::vector<std::uint8_t>& ByteLayer(
    const lunar::planning::GridMap& map, const std::string& name) {
  return std::get<std::vector<std::uint8_t>>(map.layers.at(name).values);
}

TEST(GridMapProducts, BuildsThreeHundredTwentySquareLocalMapWithTenLayers) {
  SparseObservedMap map = OneCellMap();
  const MapProductResult product = GridMapProducts{}.BuildLocal(map, Request());

  ASSERT_TRUE(product.ok()) << product.reason_code;
  EXPECT_EQ(product.message->header.frame_id, "odom");
  EXPECT_DOUBLE_EQ(product.message->info.resolution, 0.2);
  EXPECT_DOUBLE_EQ(product.message->info.length_x, 64.0);
  EXPECT_DOUBLE_EQ(product.message->info.length_y, 64.0);
  EXPECT_EQ(product.message->layers.size(), 10U);
  const auto adapted = GridMapAdapter{}.Adapt(*product.message, "odom");
  ASSERT_TRUE(adapted.ok())
      << (adapted.error ? adapted.error->reason_code : "unknown");
  EXPECT_EQ(adapted.map->width, 320U);
  EXPECT_EQ(adapted.map->height, 320U);
}

TEST(GridMapProducts, NeverEmitsUnobservedTruth) {
  SparseObservedMap map = OneCellMap({0, 0}, 123.0);
  const MapProductResult product = GridMapProducts{}.BuildLocal(map, Request());
  ASSERT_TRUE(product.ok());
  const auto adapted = GridMapAdapter{}.Adapt(*product.message, "odom");
  ASSERT_TRUE(adapted.ok());
  const auto& valid = ByteLayer(*adapted.map, "valid_mask");
  const auto& elevation = FloatLayer(*adapted.map, "elevation");
  ASSERT_EQ(valid.size(), elevation.size());
  std::size_t observed = 0U;
  for (std::size_t index = 0U; index < valid.size(); ++index) {
    if (valid[index] != 0U) {
      ++observed;
      EXPECT_FLOAT_EQ(elevation[index], 123.0F);
    } else {
      EXPECT_FLOAT_EQ(elevation[index], 0.0F);
    }
  }
  EXPECT_EQ(observed, 1U);
}

TEST(GridMapProducts, CropsAcrossSparseTileBoundaries) {
  SparseObservedMap map;
  map.ResetSession("session-a");
  ASSERT_TRUE(map.Fuse(RowPatch(255, 0, {7.0, 8.0})).accepted);
  const ProductRequest request = Request({51.1, 0.1});

  const MapProductResult product = GridMapProducts{}.BuildLocal(map, request);

  ASSERT_TRUE(product.ok());
  const auto adapted = GridMapAdapter{}.Adapt(*product.message, "odom");
  ASSERT_TRUE(adapted.ok());
  const auto& elevation = FloatLayer(*adapted.map, "elevation");
  const auto& valid = ByteLayer(*adapted.map, "valid_mask");
  const std::size_t row = 160U;
  EXPECT_EQ(valid[row * 320U + 160U], 1U);
  EXPECT_EQ(valid[row * 320U + 161U], 1U);
  EXPECT_FLOAT_EQ(elevation[row * 320U + 160U], 7.0F);
  EXPECT_FLOAT_EQ(elevation[row * 320U + 161U], 8.0F);
}

TEST(GridMapProducts, SelectsSmallestAdmissibleGlobalLevel) {
  const GridMapProducts products;
  EXPECT_EQ(products.SelectGlobalLevel(256U, 256U), 1U);
  EXPECT_EQ(products.SelectGlobalLevel(257U, 256U), 2U);
  EXPECT_EQ(products.SelectGlobalLevel(1024U, 512U), 4U);
  EXPECT_EQ(products.SelectGlobalLevel(5120U, 5120U), 20U);
  EXPECT_EQ(products.SelectGlobalLevel(5121U, 1U), std::nullopt);
}

TEST(GridMapProducts, AppliesConservativeAndOrMaxMinAggregation) {
  const ProductCell children[] = {
      ProductCell{0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.1F, 0.0F, 5.0F, 0.0F},
      ProductCell{2.0F, 1.0F, 1.0F, 0.4F, 2.0F, 0.9F, 0.2F, 0.3F, 4.0F, 0.0F},
      ProductCell{4.0F, 1.0F, 0.0F, 0.0F, 3.0F, 0.8F, 0.3F, 0.0F, 3.0F, 1.0F},
      ProductCell{6.0F, 1.0F, 0.0F, 0.0F, 4.0F, 0.7F, 0.4F, 0.0F, 2.0F, 0.0F},
  };

  const ProductCell parent = GridMapProducts::Aggregate(children, 4U, 4U);

  EXPECT_FLOAT_EQ(parent.valid_mask, 1.0F);
  EXPECT_FLOAT_EQ(parent.elevation, 3.0F);
  EXPECT_FLOAT_EQ(parent.obstacle, 1.0F);
  EXPECT_FLOAT_EQ(parent.forbidden, 1.0F);
  EXPECT_FLOAT_EQ(parent.obstacle_height, 0.4F);
  EXPECT_FLOAT_EQ(parent.observation_age_s, 4.0F);
  EXPECT_FLOAT_EQ(parent.observation_quality, 0.7F);
  EXPECT_FLOAT_EQ(parent.observation_count, 2.0F);
  EXPECT_FLOAT_EQ(parent.obstacle_variance, 0.3F);
  EXPECT_FLOAT_EQ(parent.elevation_variance, 5.4F);

  const ProductCell edge = GridMapProducts::Aggregate(children, 3U, 4U);
  EXPECT_FLOAT_EQ(edge.valid_mask, 0.0F);
  EXPECT_FLOAT_EQ(edge.forbidden, 1.0F);
  EXPECT_FLOAT_EQ(edge.elevation, 0.0F);
}

TEST(GridMapProducts, IncludesRobotGoalAndObservedCorridor) {
  SparseObservedMap map;
  map.ResetSession("session-a");
  ASSERT_TRUE(map.Fuse(RowPatch(0, 0, std::vector<double>(11U, 0.0))).accepted);
  const MapProductResult product = GridMapProducts{}.BuildGlobal(
      map, Request({0.1, 0.1}, MapPoint2D{2.1, 0.1}));

  ASSERT_TRUE(product.ok()) << product.reason_code;
  const auto adapted = GridMapAdapter{}.Adapt(*product.message, "map");
  ASSERT_TRUE(adapted.ok());
  const double minimum_x = adapted.map->origin_m.x;
  const double maximum_x = minimum_x +
      adapted.map->resolution_m * static_cast<double>(adapted.map->width);
  EXPECT_LE(minimum_x, 0.1);
  EXPECT_GE(maximum_x, 2.1);
}

TEST(GridMapProducts, RejectsGoalOutsideObservedConnectedEvidence) {
  SparseObservedMap map;
  map.ResetSession("session-a");
  ASSERT_TRUE(map.Fuse(RowPatch(
      0, 0, {0.0, 0.0, 0.0}, {1U, 0U, 1U})).accepted);

  const MapProductResult product = GridMapProducts{}.BuildGlobal(
      map, Request({0.1, 0.1}, MapPoint2D{0.5, 0.1}));

  EXPECT_FALSE(product.ok());
  EXPECT_EQ(product.reason_code, "GOAL_NOT_CONNECTED");
}

}  // namespace
}  // namespace lunar::observed_map
