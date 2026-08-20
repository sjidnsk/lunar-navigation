#include "luna_t3_map_adapter/conservative_aggregation.hpp"

#include <gtest/gtest.h>

#include <array>
#include <limits>

namespace luna::task3 {
namespace {

FineCell TraversableCell(const double elevation) {
  return FineCell{
      .elevation_m = elevation,
      .valid_mask = true,
      .obstacle = false,
      .obstacle_height_m = 0.2,
      .observation_age_s = 1.0,
      .observation_quality = 1.0,
      .elevation_variance_m2 = 0.1,
      .obstacle_variance_m2 = 0.2,
      .observation_count = 5U,
      .forbidden = false,
  };
}

FineGrid TwoByTwoGrid(std::array<FineCell, 4U> cells) {
  return FineGrid{
      .resolution_m = 0.2,
      .width = 2U,
      .height = 2U,
      .cells = {cells.begin(), cells.end()},
  };
}

TEST(ConservativeAggregationTest, PropagatesWorstCaseSafetyValues) {
  auto cells = std::array<FineCell, 4U>{
      TraversableCell(10.0), TraversableCell(12.0),
      TraversableCell(14.0), TraversableCell(16.0)};
  cells[1].obstacle = true;
  cells[1].obstacle_height_m = 2.0;
  cells[2].forbidden = true;
  cells[2].observation_age_s = 8.0;
  cells[3].observation_quality = 0.2;
  cells[3].observation_count = 3U;
  cells[3].obstacle_variance_m2 = 0.9;

  const auto aggregated = AggregateConservatively(
      TwoByTwoGrid(cells), SelectedGlobalLevel{.level = 1U,
                                                .width = 1U,
                                                .height = 1U,
                                                .resolution_m = 0.4});

  ASSERT_TRUE(aggregated.ok());
  ASSERT_EQ(aggregated.value->cells.size(), 1U);
  const FineCell& parent = aggregated.value->cells.front();
  EXPECT_TRUE(parent.valid_mask);
  EXPECT_TRUE(parent.obstacle);
  EXPECT_TRUE(parent.forbidden);
  EXPECT_DOUBLE_EQ(parent.obstacle_height_m, 2.0);
  EXPECT_DOUBLE_EQ(parent.observation_age_s, 8.0);
  EXPECT_DOUBLE_EQ(parent.observation_quality, 0.2);
  EXPECT_EQ(parent.observation_count, 3U);
  EXPECT_DOUBLE_EQ(parent.obstacle_variance_m2, 0.9);
  EXPECT_DOUBLE_EQ(parent.elevation_m, 13.0);
  EXPECT_DOUBLE_EQ(parent.elevation_variance_m2, 5.1);
}

TEST(ConservativeAggregationTest, MakesInvalidAndIncompleteParentsForbidden) {
  auto cells = std::array<FineCell, 4U>{
      TraversableCell(10.0), TraversableCell(10.0),
      TraversableCell(10.0), TraversableCell(10.0)};
  cells[0].valid_mask = false;

  const auto invalid_child = AggregateConservatively(
      TwoByTwoGrid(cells), SelectedGlobalLevel{.level = 1U,
                                                .width = 1U,
                                                .height = 1U,
                                                .resolution_m = 0.4});
  ASSERT_TRUE(invalid_child.ok());
  EXPECT_FALSE(invalid_child.value->cells.front().valid_mask);
  EXPECT_TRUE(invalid_child.value->cells.front().forbidden);

  FineGrid incomplete{
      .resolution_m = 0.2,
      .width = 3U,
      .height = 2U,
      .cells = {TraversableCell(10.0), TraversableCell(10.0),
                TraversableCell(10.0), TraversableCell(10.0),
                TraversableCell(10.0), TraversableCell(10.0)},
  };
  const auto boundary = AggregateConservatively(
      incomplete, SelectedGlobalLevel{.level = 1U,
                                      .width = 2U,
                                      .height = 1U,
                                      .resolution_m = 0.4});
  ASSERT_TRUE(boundary.ok());
  EXPECT_FALSE(boundary.value->cells[1].valid_mask);
  EXPECT_TRUE(boundary.value->cells[1].forbidden);
}

TEST(ConservativeAggregationTest, RejectsMalformedSourceAndTargetShapes) {
  const FineGrid malformed{
      .resolution_m = 0.2,
      .width = 2U,
      .height = 2U,
      .cells = {TraversableCell(1.0)},
  };
  EXPECT_EQ(AggregateConservatively(
                malformed, SelectedGlobalLevel{1U, 1U, 1U, 0.4})
                .reason_code,
            "TASK3_FINE_GRID_SHAPE_INVALID");

  const auto wrong_target = AggregateConservatively(
      TwoByTwoGrid({TraversableCell(1.0), TraversableCell(1.0),
                    TraversableCell(1.0), TraversableCell(1.0)}),
      SelectedGlobalLevel{1U, 2U, 1U, 0.4});
  EXPECT_EQ(wrong_target.reason_code, "TASK3_AGGREGATION_TARGET_INVALID");
}

}  // namespace
}  // namespace luna::task3
