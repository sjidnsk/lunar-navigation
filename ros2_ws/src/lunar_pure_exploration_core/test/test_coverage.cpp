#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "detail/coverage_statistics.hpp"
#include "lunar_pure_exploration_core/coverage.hpp"
#include "lunar_pure_exploration_core/occupancy_grid.hpp"
#include "lunar_pure_exploration_core/task_raster.hpp"

namespace lunar::pure_exploration {
namespace {

GridGeometry UnitGeometry(std::uint32_t width, std::uint32_t height) {
  return GridGeometry{
      .width = width,
      .height = height,
      .resolution = 1.0,
      .origin_x = 0.0,
      .origin_y = 0.0,
      .origin_yaw = 0.0,
  };
}

TEST(CoverageTest, ReportsShoelaceAreaAndMixedRasterStateAreas) {
  const std::array<std::int8_t, 3> data{0, 100, -1};
  const OccupancyGridView map(UnitGeometry(3U, 1U), data, 50);
  const Polygon2 polygon{{
      Vec2{-0.9, 0.1}, Vec2{3.9, 0.1},
      Vec2{3.9, 0.9}, Vec2{-0.9, 0.9},
  }};
  const TaskRaster raster = TaskRaster::Build(map, polygon);

  ASSERT_EQ(raster.task_cells().size(), 5U);
  const CoverageStats stats = CalculateCoverage(raster);

  EXPECT_DOUBLE_EQ(stats.polygon_area_m2, raster.polygon_area_m2());
  EXPECT_DOUBLE_EQ(stats.task_raster_area_m2, 5.0);
  EXPECT_DOUBLE_EQ(stats.known_free_area_m2, 1.0);
  EXPECT_DOUBLE_EQ(stats.known_occupied_area_m2, 1.0);
  EXPECT_DOUBLE_EQ(stats.unknown_area_m2, 1.0);
  EXPECT_DOUBLE_EQ(stats.outside_map_area_m2, 2.0);
  EXPECT_DOUBLE_EQ(stats.coverage_ratio, 2.0 / 5.0);
}

TEST(CoverageTest, NonzeroPolygonWithNoCellCenterHasPositiveZeroStatistics) {
  const std::array<std::int8_t, 1> data{0};
  const OccupancyGridView map(UnitGeometry(1U, 1U), data, 50);
  const Polygon2 polygon{{
      Vec2{0.1, 0.1}, Vec2{0.2, 0.1}, Vec2{0.1, 0.2},
  }};
  const TaskRaster raster = TaskRaster::Build(map, polygon);

  ASSERT_TRUE(raster.task_cells().empty());
  ASSERT_GT(raster.polygon_area_m2(), 0.0);
  const CoverageStats stats = CalculateCoverage(raster);

  EXPECT_DOUBLE_EQ(stats.polygon_area_m2, raster.polygon_area_m2());
  for (double value : {stats.task_raster_area_m2,
                       stats.known_free_area_m2,
                       stats.known_occupied_area_m2,
                       stats.unknown_area_m2,
                       stats.outside_map_area_m2,
                       stats.coverage_ratio}) {
    EXPECT_DOUBLE_EQ(value, 0.0);
    EXPECT_FALSE(std::signbit(value));
  }
}

TEST(CoverageTest, FullyKnownAndPartialTasksRemainPlainStatistics) {
  const std::array<std::int8_t, 2> data{0, 100};
  const OccupancyGridView map(UnitGeometry(2U, 1U), data, 50);
  const Polygon2 full_polygon{{
      Vec2{0.0, 0.0}, Vec2{2.0, 0.0},
      Vec2{2.0, 1.0}, Vec2{0.0, 1.0},
  }};
  const Polygon2 partial_polygon{{
      Vec2{0.0, 0.0}, Vec2{3.0, 0.0},
      Vec2{3.0, 1.0}, Vec2{0.0, 1.0},
  }};

  const CoverageStats full =
      CalculateCoverage(TaskRaster::Build(map, full_polygon));
  const CoverageStats partial =
      CalculateCoverage(TaskRaster::Build(map, partial_polygon));

  EXPECT_DOUBLE_EQ(full.coverage_ratio, 1.0);
  EXPECT_DOUBLE_EQ(partial.coverage_ratio, 2.0 / 3.0);
}

TEST(CoverageDetailTest, AccumulatesTheFourAdmittedStatesExactly) {
  detail::CoverageCounts counts{};
  detail::AccumulateCoverageState(CellState::kFree, counts);
  detail::AccumulateCoverageState(CellState::kOccupied, counts);
  detail::AccumulateCoverageState(CellState::kUnknown, counts);
  detail::AccumulateCoverageState(CellState::kOutsideMap, counts);

  EXPECT_EQ(counts.free, 1U);
  EXPECT_EQ(counts.occupied, 1U);
  EXPECT_EQ(counts.unknown, 1U);
  EXPECT_EQ(counts.outside_map, 1U);
}

TEST(CoverageDetailTest, OutsideTaskIsAnInvariantViolation) {
  detail::CoverageCounts counts{};
  EXPECT_THROW(
      detail::AccumulateCoverageState(CellState::kOutsideTask, counts),
      std::logic_error);
  EXPECT_EQ(counts.free, 0U);
  EXPECT_EQ(counts.occupied, 0U);
  EXPECT_EQ(counts.unknown, 0U);
  EXPECT_EQ(counts.outside_map, 0U);
}

TEST(CoverageDetailTest, EveryStateCounterFailsBeforeWrapping) {
  constexpr std::size_t kMaximum = std::numeric_limits<std::size_t>::max();
  for (CellState state : {CellState::kFree, CellState::kOccupied,
                          CellState::kUnknown, CellState::kOutsideMap}) {
    detail::CoverageCounts counts{};
    std::size_t* selected = nullptr;
    switch (state) {
      case CellState::kFree:
        selected = &counts.free;
        break;
      case CellState::kOccupied:
        selected = &counts.occupied;
        break;
      case CellState::kUnknown:
        selected = &counts.unknown;
        break;
      case CellState::kOutsideMap:
        selected = &counts.outside_map;
        break;
      case CellState::kOutsideTask:
        FAIL() << "outside-task is not an admitted counter";
        break;
    }
    ASSERT_NE(selected, nullptr);
    *selected = kMaximum;
    EXPECT_THROW(detail::AccumulateCoverageState(state, counts),
                 std::overflow_error);
    EXPECT_EQ(*selected, kMaximum);
  }
}

TEST(CoverageDetailTest, RejectsPositiveAreasNotRepresentableAsDouble) {
  const double underflow_resolution =
      std::numeric_limits<double>::denorm_min();
  ASSERT_GT(static_cast<long double>(underflow_resolution) *
                static_cast<long double>(underflow_resolution),
            0.0L);
  EXPECT_THROW(detail::CheckedCoverageArea(1U, underflow_resolution),
               std::overflow_error);
  EXPECT_THROW(
      detail::CheckedCoverageArea(1U, std::numeric_limits<double>::max()),
      std::overflow_error);
}

TEST(CoverageDetailTest, FinalizeScalesEveryClassByCellArea) {
  const detail::CoverageCounts counts{
      .free = 1U,
      .occupied = 2U,
      .unknown = 3U,
      .outside_map = 4U,
  };

  const CoverageStats stats = detail::FinalizeCoverage(12.345, 0.2, counts);

  EXPECT_DOUBLE_EQ(stats.polygon_area_m2, 12.345);
  EXPECT_NEAR(stats.task_raster_area_m2, 0.4, 1.0e-15);
  EXPECT_NEAR(stats.known_free_area_m2, 0.04, 1.0e-15);
  EXPECT_NEAR(stats.known_occupied_area_m2, 0.08, 1.0e-15);
  EXPECT_NEAR(stats.unknown_area_m2, 0.12, 1.0e-15);
  EXPECT_NEAR(stats.outside_map_area_m2, 0.16, 1.0e-15);
  EXPECT_DOUBLE_EQ(stats.coverage_ratio, 3.0 / 10.0);
}

TEST(CoverageDetailTest, RejectsInvalidResolutionAndAggregateCountOverflow) {
  EXPECT_THROW(detail::CheckedCoverageArea(1U, 0.0),
               std::invalid_argument);
  EXPECT_THROW(detail::CheckedCoverageArea(
                   1U, std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);

  const detail::CoverageCounts overflowing{
      .free = std::numeric_limits<std::size_t>::max(),
      .occupied = 1U,
      .unknown = 0U,
      .outside_map = 0U,
  };
  EXPECT_THROW(detail::FinalizeCoverage(1.0, 1.0, overflowing),
               std::overflow_error);
}

}  // namespace
}  // namespace lunar::pure_exploration
