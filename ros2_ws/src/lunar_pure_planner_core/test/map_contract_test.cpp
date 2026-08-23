#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::shared {
namespace {

[[nodiscard]] GridMap BaseMap(const std::size_t width = 3U,
                              const std::size_t height = 2U) {
  return GridMap{
      .frame_id = "map",
      .stamp = TimePoint{.nanoseconds_since_epoch = 1},
      .width = width,
      .height = height,
      .resolution_m = 1.0,
      .origin_m = Vec3{},
      .layers = {},
  };
}

[[nodiscard]] GridMap GlobalMap(std::vector<std::int8_t> occupancy) {
  GridMap map = BaseMap();
  map.layers.emplace(
      "occupancy", GridLayer{.values = std::move(occupancy)});
  return map;
}

[[nodiscard]] GridMap LocalMap() {
  GridMap map = BaseMap();
  map.frame_id = "odom";
  map.layers.emplace(
      "occupancy",
      GridLayer{.values = std::vector<float>{
                    0.0F, std::numeric_limits<float>::quiet_NaN(), -0.1F,
                    0.49F, 0.5F, 1.1F}});
  map.layers.emplace(
      "elevation",
      GridLayer{.values = std::vector<float>{
                    0.0F, 1.0F, std::numeric_limits<float>::quiet_NaN(),
                    3.0F, 4.0F, 5.0F}});
  return map;
}

TEST(MapContract, AcceptsNativeGlobalOccupancyWithoutLegacyLayers) {
  const std::vector<std::int8_t> values{-1, 0, 49, 50, 100, 101};

  const auto result =
      MapSnapshot::Create(GlobalMap(values), MapContract::kGlobalOccupancy);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const auto occupancy = result.snapshot->Int8Layer("occupancy");
  ASSERT_EQ(occupancy.size(), values.size());
  EXPECT_TRUE(std::ranges::equal(occupancy, values));
  EXPECT_TRUE(result.snapshot->FloatLayer("elevation").empty());
}

TEST(MapContract, AcceptsMinimalLocalLayersWithCellLevelFaults) {
  const auto result =
      MapSnapshot::Create(LocalMap(), MapContract::kLocalElevation);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const auto occupancy = result.snapshot->FloatLayer("occupancy");
  const auto elevation = result.snapshot->FloatLayer("elevation");
  ASSERT_EQ(occupancy.size(), 6U);
  ASSERT_EQ(elevation.size(), 6U);
  EXPECT_TRUE(std::isnan(occupancy[1]));
  EXPECT_FLOAT_EQ(occupancy[2], -0.1F);
  EXPECT_FLOAT_EQ(occupancy[5], 1.1F);
  EXPECT_TRUE(std::isnan(elevation[2]));
}

TEST(MapContract, MinimalGlobalCellCenterUsesOriginHeight) {
  GridMap map = GlobalMap({-1, 0, 49, 50, 100, 101});
  map.origin_m = Vec3{.x = 10.0, .y = -4.0, .z = 7.25};
  map.layers.emplace(
      "elevation", GridLayer{.values = std::vector<float>(6U, 99.0F)});
  const auto result =
      MapSnapshot::Create(std::move(map), MapContract::kGlobalOccupancy);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const Vec3 center = result.snapshot->CellCenter(GridCell{.x = 1, .y = 1});
  EXPECT_DOUBLE_EQ(center.x, 11.5);
  EXPECT_DOUBLE_EQ(center.y, -2.5);
  EXPECT_DOUBLE_EQ(center.z, 7.25);
}

TEST(MapContract, MinimalGlobalElevationSamplingFailsClosed) {
  const auto result = MapSnapshot::Create(
      GlobalMap({-1, 0, 49, 50, 100, 101}),
      MapContract::kGlobalOccupancy);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_FALSE(
      result.snapshot->SampleElevationBilinear(Vec2{.x = 1.5, .y = 0.5})
          .has_value());
}

TEST(MapContract, LocalElevationSamplingUsesFiniteElevationOnly) {
  const auto result =
      MapSnapshot::Create(LocalMap(), MapContract::kLocalElevation);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_TRUE(
      result.snapshot->SampleElevationBilinear(Vec2{.x = 0.5, .y = 0.5})
          .has_value());
  EXPECT_DOUBLE_EQ(
      *result.snapshot->SampleElevationBilinear(Vec2{.x = 0.5, .y = 0.5}),
      0.0);
}

TEST(MapContract, MinimalLocalCellCenterPreservesElevation) {
  const auto result =
      MapSnapshot::Create(LocalMap(), MapContract::kLocalElevation);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const Vec3 center = result.snapshot->CellCenter(GridCell{.x = 1, .y = 0});
  EXPECT_DOUBLE_EQ(center.x, 1.5);
  EXPECT_DOUBLE_EQ(center.y, 0.5);
  EXPECT_DOUBLE_EQ(center.z, 1.0);
}

TEST(MapContract, MinimalLocalCellCenterFallsBackForNonfiniteElevation) {
  const auto result =
      MapSnapshot::Create(LocalMap(), MapContract::kLocalElevation);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const Vec3 center = result.snapshot->CellCenter(GridCell{.x = 2, .y = 0});
  EXPECT_DOUBLE_EQ(center.z, 0.0);
}

TEST(MapContract, RejectsMalformedGeometryLayerSizeAndTypeUniformly) {
  struct Case final {
    std::string name;
    GridMap map;
    MapContract contract;
  };
  std::vector<Case> cases;

  GridMap zero_width = GlobalMap({-1, 0, 49, 50, 100, 101});
  zero_width.width = 0U;
  cases.push_back({"zero width", std::move(zero_width),
                   MapContract::kGlobalOccupancy});

  GridMap zero_resolution = GlobalMap({-1, 0, 49, 50, 100, 101});
  zero_resolution.resolution_m = 0.0;
  cases.push_back({"zero resolution", std::move(zero_resolution),
                   MapContract::kGlobalOccupancy});

  GridMap negative_resolution = LocalMap();
  negative_resolution.resolution_m = -1.0;
  cases.push_back({"negative resolution", std::move(negative_resolution),
                   MapContract::kLocalElevation});

  cases.push_back({"global wrong size", GlobalMap({-1, 0, 49}),
                   MapContract::kGlobalOccupancy});

  GridMap global_wrong_type = BaseMap();
  global_wrong_type.layers.emplace(
      "occupancy", GridLayer{.values = std::vector<float>(6U, 0.0F)});
  cases.push_back({"global wrong type", std::move(global_wrong_type),
                   MapContract::kGlobalOccupancy});

  GridMap local_wrong_size = LocalMap();
  local_wrong_size.layers.at("elevation") =
      GridLayer{.values = std::vector<float>(5U, 0.0F)};
  cases.push_back({"local wrong size", std::move(local_wrong_size),
                   MapContract::kLocalElevation});

  GridMap local_wrong_type = LocalMap();
  local_wrong_type.layers.at("occupancy") =
      GridLayer{.values = std::vector<std::uint8_t>(6U, 0U)};
  cases.push_back({"local wrong type", std::move(local_wrong_type),
                   MapContract::kLocalElevation});

  for (auto& test_case : cases) {
    const auto result =
        MapSnapshot::Create(std::move(test_case.map), test_case.contract);
    EXPECT_FALSE(result.ok()) << test_case.name;
    EXPECT_EQ(result.reason_code, "MAP_FORMAT_INVALID") << test_case.name;
  }
}

}  // namespace
}  // namespace lunar::pure_planning::shared
