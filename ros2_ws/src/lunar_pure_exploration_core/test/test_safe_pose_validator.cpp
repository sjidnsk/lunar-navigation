#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "lunar_pure_exploration_core/safe_pose_validator.hpp"

namespace lunar::pure_exploration {
namespace {

constexpr double kPi = 3.14159265358979323846;

PlatformGeometry SquarePlatform(double half_extent = 0.5,
                                double clearance = 0.0) {
  return PlatformGeometry{
      .platform_id = "validator-wheel",
      .platform_type = "WHEELED",
      .base_frame_id = "base_footprint",
      .footprint_vertices = {{half_extent, half_extent},
                             {half_extent, -half_extent},
                             {-half_extent, -half_extent},
                             {-half_extent, half_extent}},
      .minimum_clearance_m = clearance,
  };
}

PlatformGeometry TrianglePlatform() {
  return PlatformGeometry{
      .platform_id = "validator-triangle",
      .platform_type = "WHEELED",
      .base_frame_id = "base_footprint",
      .footprint_vertices = {{-0.5, -0.5}, {0.5, -0.5}, {0.0, 0.5}},
      .minimum_clearance_m = 0.0,
  };
}

TaskRaster BuildTaskRaster(const OccupancyGridView& map,
                           const GridGeometry& geometry,
                           double task_min_x, double task_min_y,
                           double task_max_x, double task_max_y) {
  const auto world = [&geometry](double x, double y) {
    return *OccupancyGridView::GridToWorld(geometry, Vec2{x, y});
  };
  return TaskRaster::Build(
      map, Polygon2{{world(task_min_x, task_min_y), world(task_max_x, task_min_y),
                     world(task_max_x, task_max_y), world(task_min_x, task_max_y)}});
}

TEST(SafePoseValidatorTest, MapFreeCanPassOutsideTaskWhileTaskFreeFails) {
  const GridGeometry geometry{12U, 12U, 1.0, 0.0, 0.0, 0.0};
  const OccupancyGridView map(geometry, std::vector<std::int8_t>(144U, 0), 50);
  const TaskRaster raster = BuildTaskRaster(map, geometry, 0.0, 0.0, 12.0, 6.0);
  const Pose2 pose{6.5, 8.5, 0.0};
  const SafePoseValidator validator(SquarePlatform(0.1), 100U);

  std::size_t map_work = 0U;
  std::size_t task_work = 0U;
  EXPECT_TRUE(validator.IsMapFree(map, pose, map_work));
  EXPECT_FALSE(validator.IsTaskFree(raster, pose, task_work));
  EXPECT_GT(map_work, 0U);
  EXPECT_GT(task_work, 0U);
}

TEST(SafePoseValidatorTest, RejectsUnknownOccupiedAndOutsideMapForBothClassifiers) {
  const GridGeometry geometry{12U, 12U, 1.0, 0.0, 0.0, 0.0};
  for (const std::int8_t state : {std::int8_t{-1}, std::int8_t{100}}) {
    std::vector<std::int8_t> data(144U, 0);
    data[6U * 12U + 6U] = state;
    const OccupancyGridView map(geometry, data, 50);
    const TaskRaster raster = BuildTaskRaster(map, geometry, 0.0, 0.0, 12.0, 12.0);
    const SafePoseValidator validator(SquarePlatform(0.1), 100U);
    std::size_t map_work = 0U;
    std::size_t task_work = 0U;
    EXPECT_FALSE(validator.IsMapFree(map, Pose2{6.5, 6.5, 0.0}, map_work));
    EXPECT_FALSE(validator.IsTaskFree(raster, Pose2{6.5, 6.5, 0.0}, task_work));
  }

  const OccupancyGridView map(geometry, std::vector<std::int8_t>(144U, 0), 50);
  const TaskRaster raster = BuildTaskRaster(map, geometry, 0.0, 0.0, 12.0, 12.0);
  const SafePoseValidator validator(SquarePlatform(0.5), 100U);
  std::size_t map_work = 0U;
  std::size_t task_work = 0U;
  EXPECT_FALSE(validator.IsMapFree(map, Pose2{0.5, 6.5, 0.0}, map_work));
  EXPECT_FALSE(validator.IsTaskFree(raster, Pose2{0.5, 6.5, 0.0}, task_work));
}

TEST(SafePoseValidatorTest, RejectsClosedTaskBoundaryTangency) {
  const GridGeometry geometry{12U, 12U, 1.0, 0.0, 0.0, 0.0};
  const OccupancyGridView map(geometry, std::vector<std::int8_t>(144U, 0), 50);
  const TaskRaster raster = BuildTaskRaster(map, geometry, 0.0, 0.0, 6.0, 12.0);
  const SafePoseValidator validator(SquarePlatform(0.5), 100U);
  std::size_t consumed_work = 0U;

  EXPECT_FALSE(validator.IsTaskFree(raster, Pose2{5.5, 6.5, 0.0}, consumed_work));
}

TEST(SafePoseValidatorTest, RotatedFootprintAndClearanceMatchClosedTangencyFixture) {
  constexpr double kRelativeYaw = kPi / 4.0;
  constexpr double kMapYaw = kPi / 6.0;
  const GridGeometry geometry{12U, 12U, 1.0, 100.0, -50.0, kMapYaw};
  const auto world = [&geometry](double x, double y) {
    return *OccupancyGridView::GridToWorld(geometry, Vec2{x, y});
  };
  const double final_phase = kPi / 8.0;
  const double clearance = 1.5 -
      0.5 * (std::cos(final_phase) + std::sin(final_phase));
  const double base_phase = final_phase - kRelativeYaw;
  std::vector<Vec2> footprint;
  for (const Vec2 vertex : {Vec2{0.5, 0.5}, Vec2{0.5, -0.5},
                            Vec2{-0.5, -0.5}, Vec2{-0.5, 0.5}}) {
    footprint.push_back(Vec2{std::cos(base_phase) * vertex.x -
                                 std::sin(base_phase) * vertex.y,
                             std::sin(base_phase) * vertex.x +
                                 std::cos(base_phase) * vertex.y});
  }
  PlatformGeometry platform = SquarePlatform();
  platform.footprint_vertices = footprint;
  platform.minimum_clearance_m = clearance;

  const OccupancyGridView free_map(geometry, std::vector<std::int8_t>(144U, 0), 50);
  const TaskRaster free_raster = BuildTaskRaster(free_map, geometry, 0.0, 0.0, 12.0, 12.0);
  const SafePoseValidator validator(platform, 1000U);
  std::size_t free_work = 0U;
  EXPECT_TRUE(validator.IsTaskFree(free_raster,
                                   Pose2{world(6.5, 5.5).x, world(6.5, 5.5).y,
                                         kMapYaw + kRelativeYaw}, free_work));

  std::vector<std::int8_t> occupied(144U, 0);
  occupied[7U * 12U + 6U] = 100;
  const OccupancyGridView occupied_map(geometry, occupied, 50);
  const TaskRaster occupied_raster =
      BuildTaskRaster(occupied_map, geometry, 0.0, 0.0, 12.0, 12.0);
  std::size_t occupied_work = 0U;
  EXPECT_FALSE(validator.IsTaskFree(occupied_raster,
                                    Pose2{world(6.5, 5.5).x, world(6.5, 5.5).y,
                                          kMapYaw + kRelativeYaw}, occupied_work));
}

TEST(SafePoseValidatorTest, ChargesCumulativeWorkExactlyAtLimit) {
  const GridGeometry geometry{20U, 20U, 1.0, 0.0, 0.0, 0.0};
  const OccupancyGridView map(geometry, std::vector<std::int8_t>(400U, 0), 50);
  const Pose2 pose{10.5, 10.5, 0.0};

  const SafePoseValidator exact(TrianglePlatform(), 12U);
  std::size_t exact_work = 0U;
  EXPECT_TRUE(exact.IsMapFree(map, pose, exact_work));
  EXPECT_EQ(exact_work, 12U);

  const SafePoseValidator one_less(TrianglePlatform(), 11U);
  std::size_t one_less_work = 0U;
  EXPECT_THROW(one_less.IsMapFree(map, pose, one_less_work), std::length_error);
}

TEST(SafePoseValidatorTest, RejectsNonAdjacentEndpointContactAsSelfIntersection) {
  PlatformGeometry footprint = SquarePlatform();
  footprint.footprint_vertices = {
      {0.0, 0.0}, {2.0, 1.0}, {1.0, 1.0}, {3.0, 1.0}, {0.0, 3.0}};

  try {
    (void)SafePoseValidator(std::move(footprint), 20U);
    FAIL() << "non-adjacent edge endpoint contact must be rejected";
  } catch (const std::invalid_argument& error) {
    EXPECT_STREQ(error.what(), "candidate footprint self-intersects");
  }
}

}  // namespace
}  // namespace lunar::pure_exploration
