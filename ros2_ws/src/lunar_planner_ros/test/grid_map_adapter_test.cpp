#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_planner_ros/grid_map_adapter.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning::ros {
namespace {

TEST(GridMapAdapter, UnwrapsCircularBufferIntoTypedRowMajorLayers) {
  const auto message = test::MakeGridMap("odom", 1U, 1U);

  const GridMapAdaptResult result = GridMapAdapter{}.Adapt(message, "odom");

  ASSERT_TRUE(result.ok());
  ASSERT_TRUE(result.map.has_value());
  EXPECT_EQ(result.map->frame_id, "odom");
  EXPECT_EQ(result.map->stamp.nanoseconds_since_epoch, 10'000'000'000LL);
  EXPECT_EQ(result.map->width, test::kMapWidth);
  EXPECT_EQ(result.map->height, test::kMapHeight);
  EXPECT_DOUBLE_EQ(result.map->origin_m.x, 0.0);
  EXPECT_DOUBLE_EQ(result.map->origin_m.y, 0.0);
  EXPECT_EQ(
      std::get<std::vector<float>>(result.map->layers.at("elevation").values),
      (std::vector<float>{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F}));
  EXPECT_EQ(
      std::get<std::vector<std::uint8_t>>(
          result.map->layers.at("valid_mask").values),
      (std::vector<std::uint8_t>(test::kMapWidth * test::kMapHeight, 1U)));
  EXPECT_EQ(
      std::get<std::vector<std::uint32_t>>(
          result.map->layers.at("observation_count").values),
      (std::vector<std::uint32_t>(test::kMapWidth * test::kMapHeight, 2U)));
}

TEST(GridMapAdapter, RejectsDuplicateAndMissingRequiredLayers) {
  auto duplicate = test::MakeGridMap();
  duplicate.layers.push_back(duplicate.layers.front());
  duplicate.data.push_back(duplicate.data.front());
  const GridMapAdaptResult duplicate_result =
      GridMapAdapter{}.Adapt(duplicate, "odom");
  ASSERT_TRUE(duplicate_result.error.has_value());
  EXPECT_EQ(duplicate_result.error->code, GridMapErrorCode::kDuplicateLayer);

  auto missing = test::MakeGridMap();
  const std::size_t missing_index = test::LayerIndex(missing, "forbidden");
  missing.layers.erase(missing.layers.begin() + missing_index);
  missing.data.erase(missing.data.begin() + missing_index);
  const GridMapAdaptResult missing_result =
      GridMapAdapter{}.Adapt(missing, "odom");
  ASSERT_TRUE(missing_result.error.has_value());
  EXPECT_EQ(missing_result.error->code, GridMapErrorCode::kMissingLayer);
  EXPECT_EQ(missing_result.error->layer, "forbidden");
}

TEST(GridMapAdapter, RejectsNanAndIllegalTypedRanges) {
  auto nonfinite = test::MakeGridMap();
  nonfinite.data[test::LayerIndex(nonfinite, "elevation")].data[0] =
      std::numeric_limits<float>::quiet_NaN();
  const GridMapAdaptResult nonfinite_result =
      GridMapAdapter{}.Adapt(nonfinite, "odom");
  ASSERT_TRUE(nonfinite_result.error.has_value());
  EXPECT_EQ(nonfinite_result.error->code, GridMapErrorCode::kNonFiniteValue);

  auto quality = test::MakeGridMap();
  quality.data[test::LayerIndex(quality, "observation_quality")].data[0] =
      1.1F;
  const GridMapAdaptResult quality_result =
      GridMapAdapter{}.Adapt(quality, "odom");
  ASSERT_TRUE(quality_result.error.has_value());
  EXPECT_EQ(quality_result.error->code, GridMapErrorCode::kOutOfRangeValue);

  auto binary = test::MakeGridMap();
  binary.data[test::LayerIndex(binary, "obstacle")].data[0] = 0.5F;
  const GridMapAdaptResult binary_result =
      GridMapAdapter{}.Adapt(binary, "odom");
  ASSERT_TRUE(binary_result.error.has_value());
  EXPECT_EQ(binary_result.error->code, GridMapErrorCode::kOutOfRangeValue);
}

TEST(GridMapAdapter, RejectsWrongFrameMalformedGeometryAndLayout) {
  const GridMapAdaptResult wrong_frame =
      GridMapAdapter{}.Adapt(test::MakeGridMap("map"), "odom");
  ASSERT_TRUE(wrong_frame.error.has_value());
  EXPECT_EQ(wrong_frame.error->code, GridMapErrorCode::kWrongFrame);

  auto geometry = test::MakeGridMap();
  geometry.info.resolution = 0.0;
  const GridMapAdaptResult geometry_result =
      GridMapAdapter{}.Adapt(geometry, "odom");
  ASSERT_TRUE(geometry_result.error.has_value());
  EXPECT_EQ(geometry_result.error->code, GridMapErrorCode::kInvalidGeometry);

  auto layout = test::MakeGridMap();
  layout.data.front().layout.dim[0].label = "unexpected";
  const GridMapAdaptResult layout_result =
      GridMapAdapter{}.Adapt(layout, "odom");
  ASSERT_TRUE(layout_result.error.has_value());
  EXPECT_EQ(layout_result.error->code, GridMapErrorCode::kMalformedLayer);

  auto start = test::MakeGridMap();
  start.outer_start_index = static_cast<std::uint16_t>(test::kMapWidth);
  const GridMapAdaptResult start_result =
      GridMapAdapter{}.Adapt(start, "odom");
  ASSERT_TRUE(start_result.error.has_value());
  EXPECT_EQ(start_result.error->code, GridMapErrorCode::kInvalidStartIndex);
}

}  // namespace
}  // namespace lunar::planning::ros
