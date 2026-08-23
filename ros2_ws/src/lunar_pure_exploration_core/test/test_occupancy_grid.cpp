#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

#include "lunar_pure_exploration_core/occupancy_grid.hpp"

namespace lunar::pure_exploration {
namespace {

constexpr double kPi = 3.14159265358979323846;

GridGeometry Geometry(double yaw = 0.0) {
  return GridGeometry{10U, 10U, 0.2, 1.25, -3.5, yaw};
}

TEST(OccupancyGridCoordinateTest, StaticTransformsRoundTripRotatedTranslatedGeometry) {
  for (const double yaw : {0.0, kPi / 6.0, kPi / 2.0}) {
    const GridGeometry geometry = Geometry(yaw);
    for (const Vec2 grid : {Vec2{-2.25, 4.75}, Vec2{0.0, 0.0},
                            Vec2{8.125, -3.5}}) {
      const auto world = OccupancyGridView::GridToWorld(geometry, grid);
      ASSERT_TRUE(world.has_value());
      const auto rebuilt = OccupancyGridView::WorldToGrid(geometry, *world);
      ASSERT_TRUE(rebuilt.has_value());
      EXPECT_NEAR(rebuilt->x, grid.x, 2.0e-12);
      EXPECT_NEAR(rebuilt->y, grid.y, 2.0e-12);
    }
  }
}

TEST(OccupancyGridCoordinateTest, WorldToCellFloorsSignedExteriorCoordinates) {
  const GridGeometry geometry = Geometry(kPi / 6.0);
  const auto exterior_world =
      OccupancyGridView::GridToWorld(geometry, Vec2{-0.001, -2.001});
  ASSERT_TRUE(exterior_world.has_value());
  EXPECT_EQ(OccupancyGridView::WorldToCell(geometry, *exterior_world),
            (std::optional<GridIndex>{GridIndex{-1, -3}}));

  const auto min_world = OccupancyGridView::GridToWorld(
      geometry, Vec2{static_cast<double>(std::numeric_limits<std::int32_t>::min()) + 0.5,
                     0.5});
  ASSERT_TRUE(min_world.has_value());
  EXPECT_EQ(OccupancyGridView::WorldToCell(geometry, *min_world),
            (std::optional<GridIndex>{GridIndex{
                std::numeric_limits<std::int32_t>::min(), 0}}));
}

TEST(OccupancyGridCoordinateTest, CellCornersAreTheClosedLogicalSquare) {
  const auto corners = OccupancyGridView::CellCornersInGrid(GridIndex{-3, 5});
  const std::array<Vec2, 4> expected{Vec2{-3.0, 5.0}, Vec2{-2.0, 5.0},
                                     Vec2{-2.0, 6.0}, Vec2{-3.0, 6.0}};
  for (std::size_t index = 0U; index < corners.size(); ++index) {
    EXPECT_DOUBLE_EQ(corners[index].x, expected[index].x);
    EXPECT_DOUBLE_EQ(corners[index].y, expected[index].y);
  }
}

TEST(OccupancyGridCoordinateTest, RejectsNonFiniteAndNonrepresentableInputs) {
  const GridGeometry geometry = Geometry();
  EXPECT_FALSE(OccupancyGridView::WorldToGrid(
                   geometry, Vec2{std::numeric_limits<double>::quiet_NaN(), 0.0})
                   .has_value());
  EXPECT_FALSE(OccupancyGridView::GridToWorld(
                   geometry, Vec2{std::numeric_limits<double>::infinity(), 0.0})
                   .has_value());
  EXPECT_FALSE(OccupancyGridView::WorldToCell(
                   geometry, Vec2{1.0e300, 0.0})
                   .has_value());
}

TEST(OccupancyGridCoordinateTest, InstanceMethodsDelegateToStaticGeometryPrimitives) {
  std::array<std::int8_t, 100> data{};
  const GridGeometry geometry = Geometry(kPi / 2.0);
  const OccupancyGridView map(geometry, data, 50);
  const Vec2 world{-1.25, -3.2};
  const auto instance_grid = map.WorldToGrid(world);
  const auto static_grid = OccupancyGridView::WorldToGrid(geometry, world);
  ASSERT_TRUE(instance_grid.has_value());
  ASSERT_TRUE(static_grid.has_value());
  EXPECT_DOUBLE_EQ(instance_grid->x, static_grid->x);
  EXPECT_DOUBLE_EQ(instance_grid->y, static_grid->y);
  EXPECT_EQ(map.WorldToCell(world), OccupancyGridView::WorldToCell(geometry, world));
  const auto instance_world = map.GridToWorld(Vec2{2.0, 3.0});
  const auto static_world = OccupancyGridView::GridToWorld(geometry, Vec2{2.0, 3.0});
  ASSERT_TRUE(instance_world.has_value());
  ASSERT_TRUE(static_world.has_value());
  EXPECT_DOUBLE_EQ(instance_world->x, static_world->x);
  EXPECT_DOUBLE_EQ(instance_world->y, static_world->y);
  const auto instance_corners = map.CellCornersInGrid(GridIndex{-2, 4});
  const auto static_corners = OccupancyGridView::CellCornersInGrid(GridIndex{-2, 4});
  for (std::size_t index = 0U; index < instance_corners.size(); ++index) {
    EXPECT_DOUBLE_EQ(instance_corners[index].x, static_corners[index].x);
    EXPECT_DOUBLE_EQ(instance_corners[index].y, static_corners[index].y);
  }

  const Vec2 expected = *OccupancyGridView::GridToWorld(
      geometry, Vec2{static_cast<double>(std::numeric_limits<std::int32_t>::max()) + 0.5,
                     static_cast<double>(std::numeric_limits<std::int32_t>::min()) + 0.5});
  const Vec2 actual = map.CellCenter(GridIndex{
      std::numeric_limits<std::int32_t>::max(),
      std::numeric_limits<std::int32_t>::min()});
  EXPECT_DOUBLE_EQ(actual.x, expected.x);
  EXPECT_DOUBLE_EQ(actual.y, expected.y);
}

}  // namespace
}  // namespace lunar::pure_exploration
