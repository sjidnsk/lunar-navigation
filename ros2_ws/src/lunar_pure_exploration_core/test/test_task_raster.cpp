#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "lunar_pure_exploration_core/occupancy_grid.hpp"
#include "lunar_pure_exploration_core/task_raster.hpp"

namespace lunar::pure_exploration {
namespace {

constexpr double kPi = 3.14159265358979323846;

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

TEST(OccupancyGridViewTest, ClassifiesEverySignedInt8UsingNativeSemantics) {
  std::vector<std::int8_t> data;
  data.reserve(256U);
  for (int value = -128; value <= 127; ++value) {
    data.push_back(static_cast<std::int8_t>(value));
  }
  const OccupancyGridView map(UnitGeometry(256U, 1U), data, 50);

  for (int value = -128; value <= 127; ++value) {
    const auto actual = map.Classify(GridIndex{value + 128, 0});
    const auto expected = value >= 0 && value <= 49
                              ? CellState::kFree
                              : value >= 50 && value <= 100
                                    ? CellState::kOccupied
                                    : CellState::kUnknown;
    SCOPED_TRACE(value);
    EXPECT_EQ(actual, expected);
  }

  EXPECT_EQ(map.Classify(GridIndex{127, 0}), CellState::kUnknown);  // -1
  EXPECT_EQ(map.Classify(GridIndex{128, 0}), CellState::kFree);     // 0
  EXPECT_EQ(map.Classify(GridIndex{177, 0}), CellState::kFree);     // 49
  EXPECT_EQ(map.Classify(GridIndex{178, 0}), CellState::kOccupied); // 50
  EXPECT_EQ(map.Classify(GridIndex{228, 0}), CellState::kOccupied); // 100
  EXPECT_EQ(map.Classify(GridIndex{229, 0}), CellState::kUnknown);  // 101
}

TEST(OccupancyGridViewTest, ReportsIndicesOutsideTheMapWithoutArrayAccess) {
  const std::array<std::int8_t, 1> data{0};
  const OccupancyGridView map(UnitGeometry(1U, 1U), data, 50);

  EXPECT_TRUE(map.Contains(GridIndex{0, 0}));
  EXPECT_FALSE(map.Contains(GridIndex{-1, 0}));
  EXPECT_FALSE(map.Contains(GridIndex{0, -1}));
  EXPECT_FALSE(map.Contains(GridIndex{1, 0}));
  EXPECT_EQ(map.Classify(GridIndex{-1, 0}), CellState::kOutsideMap);
  EXPECT_EQ(map.Classify(GridIndex{1, 0}), CellState::kOutsideMap);
}

TEST(OccupancyGridViewTest, OwnsACopyOfNativeOccupancyData) {
  std::vector<std::int8_t> source{0, 100};
  const OccupancyGridView map(UnitGeometry(2U, 1U), source, 50);
  source[0] = 100;
  source[1] = 0;

  EXPECT_EQ(map.Classify(GridIndex{0, 0}), CellState::kFree);
  EXPECT_EQ(map.Classify(GridIndex{1, 0}), CellState::kOccupied);
}

TEST(OccupancyGridViewTest, ConvertsWorldAndCellCoordinatesForRotatedOrigin) {
  const std::array<std::int8_t, 4> data{0, 0, 0, 0};
  const GridGeometry geometry{
      .width = 2U,
      .height = 2U,
      .resolution = 2.0,
      .origin_x = 10.0,
      .origin_y = -3.0,
      .origin_yaw = kPi / 2.0,
  };
  const OccupancyGridView map(geometry, data, 50);

  const Vec2 center_00 = map.CellCenter(GridIndex{0, 0});
  EXPECT_NEAR(center_00.x, 9.0, 1.0e-12);
  EXPECT_NEAR(center_00.y, -2.0, 1.0e-12);
  EXPECT_EQ(map.WorldToCell(Vec2{9.0, -2.0}),
            (std::optional<GridIndex>{GridIndex{0, 0}}));
  EXPECT_EQ(map.WorldToCell(Vec2{9.0, 0.0}),
            (std::optional<GridIndex>{GridIndex{1, 0}}));
  EXPECT_EQ(map.WorldToCell(Vec2{7.0, -2.0}),
            (std::optional<GridIndex>{GridIndex{0, 1}}));
  EXPECT_EQ(map.WorldToCell(Vec2{11.0, -2.0}),
            (std::optional<GridIndex>{GridIndex{0, -1}}));
  EXPECT_FALSE(map.WorldToCell(
      Vec2{std::numeric_limits<double>::quiet_NaN(), 0.0}).has_value());
}

TEST(OccupancyGridViewTest, ConvertsNonCardinalRotationBeyondMapBounds) {
  const std::array<std::int8_t, 1> data{0};
  const GridGeometry geometry{
      .width = 1U,
      .height = 1U,
      .resolution = 2.0,
      .origin_x = 2.0,
      .origin_y = -1.0,
      .origin_yaw = kPi / 4.0,
  };
  const OccupancyGridView map(geometry, data, 50);

  const Vec2 center = map.CellCenter(GridIndex{-2, 3});
  EXPECT_NEAR(center.x, -5.0710678118654755, 1.0e-12);
  EXPECT_NEAR(center.y, 1.8284271247461898, 1.0e-12);
  EXPECT_EQ(map.WorldToCell(center),
            (std::optional<GridIndex>{GridIndex{-2, 3}}));
  EXPECT_FALSE(map.Contains(GridIndex{-2, 3}));
}

TEST(OccupancyGridViewTest, ConvertsGridIndexExtremesAndRejectsOverflow) {
  const std::array<std::int8_t, 1> data{0};
  const OccupancyGridView map(UnitGeometry(1U, 1U), data, 50);
  constexpr auto kMin = std::numeric_limits<std::int32_t>::min();
  constexpr auto kMax = std::numeric_limits<std::int32_t>::max();

  EXPECT_EQ(map.WorldToCell(Vec2{static_cast<double>(kMin) + 0.5, 0.5}),
            (std::optional<GridIndex>{GridIndex{kMin, 0}}));
  EXPECT_EQ(map.WorldToCell(Vec2{static_cast<double>(kMax) + 0.5, 0.5}),
            (std::optional<GridIndex>{GridIndex{kMax, 0}}));
  EXPECT_FALSE(
      map.WorldToCell(Vec2{static_cast<double>(kMin) - 1.0, 0.5})
          .has_value());
  EXPECT_FALSE(
      map.WorldToCell(Vec2{static_cast<double>(kMax) + 1.0, 0.5})
          .has_value());
}

TEST(OccupancyGridViewTest, RejectsInvalidGeometryAndStorage) {
  const std::array<std::int8_t, 1> one_cell{0};
  auto geometry = UnitGeometry(1U, 1U);

  geometry.resolution = 0.0;
  EXPECT_THROW(OccupancyGridView(geometry, one_cell, 50), std::invalid_argument);
  geometry.resolution = -1.0;
  EXPECT_THROW(OccupancyGridView(geometry, one_cell, 50), std::invalid_argument);
  geometry.resolution = std::numeric_limits<double>::infinity();
  EXPECT_THROW(OccupancyGridView(geometry, one_cell, 50), std::invalid_argument);

  geometry = UnitGeometry(1U, 1U);
  geometry.origin_x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(OccupancyGridView(geometry, one_cell, 50), std::invalid_argument);
  geometry = UnitGeometry(1U, 1U);
  geometry.origin_y = std::numeric_limits<double>::infinity();
  EXPECT_THROW(OccupancyGridView(geometry, one_cell, 50), std::invalid_argument);
  geometry = UnitGeometry(1U, 1U);
  geometry.origin_yaw = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(OccupancyGridView(geometry, one_cell, 50), std::invalid_argument);

  EXPECT_THROW(OccupancyGridView(UnitGeometry(2U, 1U), one_cell, 50),
               std::invalid_argument);
  EXPECT_THROW(
      OccupancyGridView(
          UnitGeometry(std::numeric_limits<std::uint32_t>::max(),
                       std::numeric_limits<std::uint32_t>::max()),
          std::span<const std::int8_t>{}, 50),
      std::overflow_error);
  EXPECT_THROW(OccupancyGridView(UnitGeometry(1U, 1U), one_cell, -1),
               std::invalid_argument);
  EXPECT_THROW(OccupancyGridView(UnitGeometry(1U, 1U), one_cell, 101),
               std::invalid_argument);
}

TEST(TaskRasterTest, RasterizesConcaveTaskAcrossSignedMapBoundary) {
  const std::array<std::int8_t, 9> data{
      0,  -1, 100,
      49, 50, 101,
      0,  0,  0,
  };
  const OccupancyGridView map(UnitGeometry(3U, 3U), data, 50);
  const Polygon2 polygon{{
      Vec2{-1.0, -1.0}, Vec2{4.0, -1.0}, Vec2{4.0, 4.0},
      Vec2{2.0, 4.0}, Vec2{2.0, 1.0}, Vec2{-1.0, 1.0},
  }};

  const TaskRaster raster = TaskRaster::Build(map, polygon);

  EXPECT_DOUBLE_EQ(raster.polygon_area_m2(), 16.0);
  EXPECT_EQ(raster.task_cells().size(), 16U);
  EXPECT_EQ(raster.map_backed_cells().size(), 5U);
  EXPECT_TRUE(raster.ContainsCellCenter(GridIndex{-1, 0}));
  EXPECT_FALSE(raster.IsMapBacked(GridIndex{-1, 0}));
  EXPECT_EQ(raster.Classify(GridIndex{-1, 0}), CellState::kOutsideMap);
  EXPECT_EQ(raster.Classify(GridIndex{3, 2}), CellState::kOutsideMap);
  EXPECT_EQ(raster.Classify(GridIndex{1, 0}), CellState::kUnknown);
  EXPECT_EQ(raster.Classify(GridIndex{0, 0}), CellState::kFree);
  EXPECT_EQ(raster.Classify(GridIndex{2, 0}), CellState::kOccupied);
  EXPECT_EQ(raster.Classify(GridIndex{1, 1}), CellState::kOutsideTask);
  EXPECT_EQ(raster.Classify(GridIndex{0, 2}), CellState::kOutsideTask);
}

TEST(TaskRasterTest, IncludesCellCentersOnPolygonBoundary) {
  const std::array<std::int8_t, 4> data{0, 0, 0, 0};
  const OccupancyGridView map(UnitGeometry(2U, 2U), data, 50);
  const Polygon2 polygon{{
      Vec2{0.5, 0.5}, Vec2{1.5, 0.5},
      Vec2{1.5, 1.5}, Vec2{0.5, 1.5},
  }};

  const TaskRaster raster = TaskRaster::Build(map, polygon);

  EXPECT_EQ(raster.task_cells().size(), 4U);
  EXPECT_TRUE(raster.ContainsCellCenter(GridIndex{0, 0}));
  EXPECT_TRUE(raster.ContainsCellCenter(GridIndex{1, 0}));
  EXPECT_TRUE(raster.ContainsCellCenter(GridIndex{1, 1}));
  EXPECT_TRUE(raster.ContainsCellCenter(GridIndex{0, 1}));
}

TEST(TaskRasterTest, BoundaryToleranceIsInvariantUnderLargeTranslation) {
  constexpr double kOffset = 1.0e15;
  const std::array<std::int8_t, 16> data{};
  const GridGeometry geometry{
      .width = 4U,
      .height = 4U,
      .resolution = 1.0,
      .origin_x = kOffset,
      .origin_y = kOffset,
      .origin_yaw = 0.0,
  };
  const OccupancyGridView map(geometry, data, 50);
  const Polygon2 polygon{{
      Vec2{kOffset + 1.5, kOffset + 1.5},
      Vec2{kOffset + 3.5, kOffset + 1.5},
      Vec2{kOffset + 3.5, kOffset + 3.5},
      Vec2{kOffset + 0.5, kOffset + 3.5},
      Vec2{kOffset + 0.5, kOffset + 2.5},
      Vec2{kOffset + 1.5, kOffset + 2.5},
  }};

  const TaskRaster raster = TaskRaster::Build(map, polygon);

  EXPECT_EQ(raster.Classify(GridIndex{0, 1}), CellState::kOutsideTask);
  EXPECT_TRUE(raster.ContainsCellCenter(GridIndex{2, 1}));
  EXPECT_TRUE(raster.ContainsCellCenter(GridIndex{1, 1}));
}

TEST(TaskRasterTest, WindingAndRepeatedBuildPreserveRowMajorCellOrder) {
  const std::array<std::int8_t, 4> data{0, -1, 50, 100};
  const OccupancyGridView map(UnitGeometry(2U, 2U), data, 50);
  const Polygon2 counterclockwise{{
      Vec2{0.0, 0.0}, Vec2{2.0, 0.0},
      Vec2{2.0, 2.0}, Vec2{0.0, 2.0},
  }};
  const Polygon2 clockwise{{
      Vec2{0.0, 2.0}, Vec2{2.0, 2.0},
      Vec2{2.0, 0.0}, Vec2{0.0, 0.0},
  }};

  const TaskRaster first = TaskRaster::Build(map, counterclockwise);
  const TaskRaster reversed = TaskRaster::Build(map, clockwise);
  const TaskRaster repeated = TaskRaster::Build(map, counterclockwise);
  const std::vector<GridIndex> expected{
      GridIndex{0, 0}, GridIndex{1, 0}, GridIndex{0, 1}, GridIndex{1, 1}};

  EXPECT_EQ(std::vector<GridIndex>(first.task_cells().begin(),
                                   first.task_cells().end()), expected);
  EXPECT_EQ(std::vector<GridIndex>(reversed.task_cells().begin(),
                                   reversed.task_cells().end()), expected);
  EXPECT_EQ(std::vector<GridIndex>(repeated.task_cells().begin(),
                                   repeated.task_cells().end()), expected);
  EXPECT_EQ(std::vector<GridIndex>(first.map_backed_cells().begin(),
                                   first.map_backed_cells().end()), expected);
  for (const GridIndex index : expected) {
    EXPECT_EQ(first.Classify(index), reversed.Classify(index));
    EXPECT_EQ(first.Classify(index), repeated.Classify(index));
  }
}

TEST(TaskRasterTest, UsesMapRotationForLogicalCellCenters) {
  const std::array<std::int8_t, 1> data{0};
  const GridGeometry geometry{
      .width = 1U,
      .height = 1U,
      .resolution = 2.0,
      .origin_x = 10.0,
      .origin_y = -3.0,
      .origin_yaw = kPi / 2.0,
  };
  const OccupancyGridView map(geometry, data, 50);
  const Polygon2 polygon{{
      Vec2{8.5, -2.5}, Vec2{9.5, -2.5},
      Vec2{9.5, -1.5}, Vec2{8.5, -1.5},
  }};

  const TaskRaster raster = TaskRaster::Build(map, polygon);

  EXPECT_TRUE(raster.ContainsCellCenter(GridIndex{0, 0}));
  EXPECT_EQ(raster.Classify(GridIndex{0, 0}), CellState::kFree);
  const Vec2 center = raster.CellCenter(GridIndex{0, 0});
  EXPECT_NEAR(center.x, 9.0, 1.0e-12);
  EXPECT_NEAR(center.y, -2.0, 1.0e-12);
  EXPECT_DOUBLE_EQ(raster.geometry().origin_yaw, kPi / 2.0);
}

TEST(TaskRasterTest, OwnsRasterStatesAfterSourceMapIsDestroyed) {
  auto make_raster = [] {
    const std::array<std::int8_t, 1> temporary{100};
    const OccupancyGridView map(UnitGeometry(1U, 1U), temporary, 50);
    return TaskRaster::Build(
        map, Polygon2{{Vec2{0.0, 0.0}, Vec2{1.0, 0.0},
                       Vec2{1.0, 1.0}, Vec2{0.0, 1.0}}});
  };

  const TaskRaster raster = make_raster();

  EXPECT_EQ(raster.Classify(GridIndex{0, 0}), CellState::kOccupied);
}

TEST(TaskRasterTest, RejectsZeroRasterCellLimit) {
  const std::array<std::int8_t, 1> data{0};
  const OccupancyGridView map(UnitGeometry(1U, 1U), data, 50);
  const Polygon2 polygon{{
      Vec2{0.0, 0.0}, Vec2{1.0, 0.0},
      Vec2{1.0, 1.0}, Vec2{0.0, 1.0},
  }};

  EXPECT_THROW(TaskRaster::Build(
                   map, polygon,
                   TaskRaster::Limits{.maximum_raster_cell_count = 0U}),
               std::invalid_argument);
}

TEST(TaskRasterTest, EnforcesCustomRasterCellLimitBeforeWork) {
  const std::array<std::int8_t, 4> data{};
  const OccupancyGridView map(UnitGeometry(2U, 2U), data, 50);
  const Polygon2 polygon{{
      Vec2{0.0, 0.0}, Vec2{2.0, 0.0},
      Vec2{2.0, 2.0}, Vec2{0.0, 2.0},
  }};

  EXPECT_THROW(TaskRaster::Build(
                   map, polygon,
                   TaskRaster::Limits{.maximum_raster_cell_count = 3U}),
               std::length_error);
  const TaskRaster raster = TaskRaster::Build(
      map, polygon, TaskRaster::Limits{.maximum_raster_cell_count = 4U});
  EXPECT_EQ(raster.task_cells().size(), 4U);
}

TEST(TaskRasterTest, RejectsLargeRepresentableRasterBeforeAllocation) {
  const std::array<std::int8_t, 1> data{0};
  const OccupancyGridView map(UnitGeometry(1U, 1U), data, 50);
  const Polygon2 polygon{{
      Vec2{0.0, 0.0}, Vec2{100000.0, 0.0},
      Vec2{100000.0, 100000.0}, Vec2{0.0, 100000.0},
  }};

  EXPECT_EQ(TaskRaster::Limits{}.maximum_raster_cell_count, 1048576U);
  EXPECT_THROW(TaskRaster::Build(map, polygon, TaskRaster::Limits{}),
               std::length_error);
}

TEST(TaskRasterTest, RejectsFewerThanThreeDistinctVertices) {
  const std::array<std::int8_t, 1> data{0};
  const OccupancyGridView map(UnitGeometry(1U, 1U), data, 50);

  EXPECT_THROW(TaskRaster::Build(map, Polygon2{{Vec2{0.0, 0.0},
                                                Vec2{1.0, 0.0},
                                                Vec2{0.0, 0.0}}}),
               std::invalid_argument);
}

TEST(TaskRasterTest, RejectsSelfIntersectionWithNonzeroShoelaceArea) {
  const std::array<std::int8_t, 1> data{0};
  const OccupancyGridView map(UnitGeometry(1U, 1U), data, 50);
  const Polygon2 crossing{{Vec2{0.0, 0.0}, Vec2{3.0, 0.0},
                           Vec2{0.0, 2.0}, Vec2{2.0, 2.0}}};

  EXPECT_THROW(TaskRaster::Build(map, crossing), std::invalid_argument);
}

TEST(TaskRasterTest, RejectsRepeatedVerticesAndCollinearOverlap) {
  const std::array<std::int8_t, 1> data{0};
  const OccupancyGridView map(UnitGeometry(1U, 1U), data, 50);
  const Polygon2 repeated{{
      Vec2{0.0, 0.0}, Vec2{2.0, 0.0}, Vec2{2.0, 2.0},
      Vec2{0.0, 2.0}, Vec2{2.0, 0.0},
  }};
  const Polygon2 overlapping{{
      Vec2{0.0, 0.0}, Vec2{2.0, 0.0}, Vec2{1.0, 0.0},
      Vec2{1.0, 1.0}, Vec2{0.0, 1.0},
  }};

  EXPECT_THROW(TaskRaster::Build(map, repeated), std::invalid_argument);
  EXPECT_THROW(TaskRaster::Build(map, overlapping), std::invalid_argument);
}

TEST(TaskRasterTest, RejectsZeroAreaAndNonfiniteCoordinates) {
  const std::array<std::int8_t, 1> data{0};
  const OccupancyGridView map(UnitGeometry(1U, 1U), data, 50);
  const Polygon2 collinear{{
      Vec2{0.0, 0.0}, Vec2{1.0, 1.0}, Vec2{2.0, 2.0},
  }};
  const Polygon2 nonfinite{{
      Vec2{0.0, 0.0},
      Vec2{1.0, std::numeric_limits<double>::infinity()},
      Vec2{0.0, 1.0},
  }};

  EXPECT_THROW(TaskRaster::Build(map, collinear), std::invalid_argument);
  EXPECT_THROW(TaskRaster::Build(map, nonfinite), std::invalid_argument);
}

TEST(TaskRasterTest, RejectsLogicalRasterBoundsOutsideGridIndexRange) {
  const std::array<std::int8_t, 1> data{0};
  const OccupancyGridView map(UnitGeometry(1U, 1U), data, 50);
  const Polygon2 enormous{{
      Vec2{1.0e20, 1.0e20}, Vec2{1.0e20 + 1.0e10, 1.0e20},
      Vec2{1.0e20, 1.0e20 + 1.0e10},
  }};

  EXPECT_THROW(TaskRaster::Build(map, enormous), std::overflow_error);
}

TEST(TaskRasterCoordinateTest, DelegatesEveryCoordinatePrimitiveBitForBit) {
  const GridGeometry geometry{
      .width = 4U,
      .height = 3U,
      .resolution = 0.2,
      .origin_x = 1.0e9,
      .origin_y = -1.0e9,
      .origin_yaw = kPi / 6.0,
  };
  const std::array<std::int8_t, 12> data{};
  const OccupancyGridView map(geometry, data, 50);
  const Polygon2 polygon{{
      *OccupancyGridView::GridToWorld(geometry, Vec2{0.0, 0.0}),
      *OccupancyGridView::GridToWorld(geometry, Vec2{4.0, 0.0}),
      *OccupancyGridView::GridToWorld(geometry, Vec2{4.0, 3.0}),
      *OccupancyGridView::GridToWorld(geometry, Vec2{0.0, 3.0}),
  }};
  const TaskRaster raster = TaskRaster::Build(map, polygon);

  const Vec2 world =
      *OccupancyGridView::GridToWorld(geometry, Vec2{-0.25, 2.75});
  const auto raster_grid = raster.WorldToGrid(world);
  const auto shared_grid = OccupancyGridView::WorldToGrid(geometry, world);
  ASSERT_TRUE(raster_grid.has_value());
  ASSERT_TRUE(shared_grid.has_value());
  EXPECT_DOUBLE_EQ(raster_grid->x, shared_grid->x);
  EXPECT_DOUBLE_EQ(raster_grid->y, shared_grid->y);
  EXPECT_EQ(raster.WorldToCell(world),
            OccupancyGridView::WorldToCell(geometry, world));

  const auto raster_world = raster.GridToWorld(Vec2{-5.0, 8.0});
  const auto shared_world =
      OccupancyGridView::GridToWorld(geometry, Vec2{-5.0, 8.0});
  ASSERT_TRUE(raster_world.has_value());
  ASSERT_TRUE(shared_world.has_value());
  EXPECT_DOUBLE_EQ(raster_world->x, shared_world->x);
  EXPECT_DOUBLE_EQ(raster_world->y, shared_world->y);

  const auto raster_corners = raster.CellCornersInGrid(GridIndex{-3, 5});
  const auto shared_corners =
      OccupancyGridView::CellCornersInGrid(GridIndex{-3, 5});
  for (std::size_t index = 0U; index < raster_corners.size(); ++index) {
    EXPECT_DOUBLE_EQ(raster_corners[index].x, shared_corners[index].x);
    EXPECT_DOUBLE_EQ(raster_corners[index].y, shared_corners[index].y);
  }
  for (const GridIndex index : {
           GridIndex{0, 0}, GridIndex{-1, 4},
           GridIndex{std::numeric_limits<std::int32_t>::min(), 0},
           GridIndex{std::numeric_limits<std::int32_t>::max(), -1}}) {
    const Vec2 expected = map.CellCenter(index);
    const Vec2 actual = raster.CellCenter(index);
    EXPECT_DOUBLE_EQ(actual.x, expected.x);
    EXPECT_DOUBLE_EQ(actual.y, expected.y);
  }
}

}  // namespace
}  // namespace lunar::pure_exploration
