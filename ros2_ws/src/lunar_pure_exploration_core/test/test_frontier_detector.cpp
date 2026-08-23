#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "lunar_pure_exploration_core/frontier_detector.hpp"

namespace lunar::pure_exploration {
namespace {

constexpr double kPi = 3.14159265358979323846;

Vec2 GridCornerToWorld(const GridGeometry& geometry, double x, double y) {
  const double cosine = std::cos(geometry.origin_yaw);
  const double sine = std::sin(geometry.origin_yaw);
  const double local_x = x * geometry.resolution;
  const double local_y = y * geometry.resolution;
  return Vec2{
      geometry.origin_x + cosine * local_x - sine * local_y,
      geometry.origin_y + sine * local_x + cosine * local_y,
  };
}

Polygon2 WholeMapPolygon(const GridGeometry& geometry) {
  return Polygon2{{
      GridCornerToWorld(geometry, 0.0, 0.0),
      GridCornerToWorld(geometry, geometry.width, 0.0),
      GridCornerToWorld(geometry, geometry.width, geometry.height),
      GridCornerToWorld(geometry, 0.0, geometry.height),
  }};
}

TaskRaster BuildRaster(std::uint32_t width, std::uint32_t height,
                       std::vector<std::int8_t> data,
                       double resolution = 1.0, double origin_x = 0.0,
                       double origin_y = 0.0, double origin_yaw = 0.0) {
  const GridGeometry geometry{
      .width = width,
      .height = height,
      .resolution = resolution,
      .origin_x = origin_x,
      .origin_y = origin_y,
      .origin_yaw = origin_yaw,
  };
  const OccupancyGridView map(geometry, data, 50);
  return TaskRaster::Build(map, WholeMapPolygon(geometry));
}

TEST(FrontierDetectorTest, FindsOneCorridorFrontierWithExactInterfaceIdentity) {
  const TaskRaster raster = BuildRaster(
      5U, 3U,
      {
          100, 100, 100, 100, 100,
          0,   0,   0,   -1,  -1,
          100, 100, 100, 100, 100,
      });

  const FrontierDetection detection =
      FrontierDetector({1.0}).Detect(raster, GridIndex{0, 1});

  ASSERT_EQ(detection.reason, FrontierDetectionReason::kOk);
  EXPECT_TRUE(detection.has_reachable_free_start);
  EXPECT_EQ(detection.reachable_free_cell_count, 3U);
  ASSERT_EQ(detection.clusters.size(), 1U);
  const FrontierCluster& cluster = detection.clusters.front();
  EXPECT_EQ(cluster.cells, (std::vector<GridIndex>{{2, 1}}));
  ASSERT_EQ(cluster.interface_edges.size(), 1U);
  EXPECT_EQ(cluster.interface_edges[0].free_cell, (GridIndex{2, 1}));
  EXPECT_EQ(cluster.interface_edges[0].unknown_cell, (GridIndex{3, 1}));
  EXPECT_EQ(cluster.interface_edges[0].direction, 0U);
  EXPECT_DOUBLE_EQ(cluster.interface_edges[0].midpoint.x, 3.0);
  EXPECT_DOUBLE_EQ(cluster.interface_edges[0].midpoint.y, 1.5);
  EXPECT_DOUBLE_EQ(cluster.centroid.x, 3.0);
  EXPECT_DOUBLE_EQ(cluster.centroid.y, 1.5);
  EXPECT_DOUBLE_EQ(cluster.length_m, 1.0);
  EXPECT_EQ(cluster.canonical_key,
            (std::vector<std::int64_t>{3000, 1500, 0}));
  EXPECT_EQ(cluster.id, UINT64_C(1330758108225781293));
}

TEST(FrontierDetectorTest, FreeWavefrontCannotCrossAnObstacleCorner) {
  const TaskRaster raster = BuildRaster(
      3U, 3U,
      {
          0,   100, 100,
          100, 0,   -1,
          100, 100, 100,
      });

  const FrontierDetection detection =
      FrontierDetector({0.0}).Detect(raster, GridIndex{0, 0});

  EXPECT_EQ(detection.reachable_free_cell_count, 1U);
  EXPECT_TRUE(detection.clusters.empty());
}

TEST(FrontierDetectorTest, EightNeighborClusteringMergesDiagonalFrontierCells) {
  const TaskRaster raster = BuildRaster(
      3U, 3U,
      {
          0,   0,   -1,
          0,   100, 100,
          -1,  100, 100,
      });

  const FrontierDetection detection =
      FrontierDetector({0.0}).Detect(raster, GridIndex{0, 0});

  ASSERT_EQ(detection.clusters.size(), 1U);
  EXPECT_EQ(detection.clusters[0].cells,
            (std::vector<GridIndex>{{0, 1}, {1, 0}}));
  EXPECT_EQ(detection.clusters[0].interface_edges.size(), 2U);
}

TEST(FrontierDetectorTest, EmitsOneDirectedEdgeForEveryUnknownNeighbor) {
  const TaskRaster raster = BuildRaster(
      3U, 3U,
      {
          100, -1, 100,
          -1,  0,  -1,
          100, -1, 100,
      });

  const FrontierDetection detection =
      FrontierDetector({0.0}).Detect(raster, GridIndex{1, 1});

  ASSERT_EQ(detection.clusters.size(), 1U);
  const FrontierCluster& cluster = detection.clusters[0];
  ASSERT_EQ(cluster.interface_edges.size(), 4U);
  EXPECT_EQ(cluster.interface_edges[0].direction, 0U);
  EXPECT_EQ(cluster.interface_edges[1].direction, 1U);
  EXPECT_EQ(cluster.interface_edges[2].direction, 2U);
  EXPECT_EQ(cluster.interface_edges[3].direction, 3U);
  EXPECT_DOUBLE_EQ(cluster.length_m, 4.0);
  EXPECT_DOUBLE_EQ(cluster.centroid.x, 1.5);
  EXPECT_DOUBLE_EQ(cluster.centroid.y, 1.5);
}

TEST(FrontierDetectorTest, KeepsDisconnectedUnknownBoundariesAsSeparateClusters) {
  const TaskRaster raster = BuildRaster(
      8U, 3U,
      {
          100, 100, 100, 100, 100, 100, 100, 100,
          0,   0,   0,   0,   0,   0,   0,   0,
          100, 100, -1,  100, 100, -1,  100, 100,
      });

  const FrontierDetection detection =
      FrontierDetector({0.0}).Detect(raster, GridIndex{0, 1});

  ASSERT_EQ(detection.clusters.size(), 2U);
  EXPECT_EQ(detection.clusters[0].id, UINT64_C(6329560424189663974));
  EXPECT_EQ(detection.clusters[0].cells,
            (std::vector<GridIndex>{{5, 1}}));
  EXPECT_EQ(detection.clusters[1].id, UINT64_C(17549903202327538458));
  EXPECT_EQ(detection.clusters[1].cells,
            (std::vector<GridIndex>{{2, 1}}));
  for (std::size_t index = 1U; index < detection.clusters.size(); ++index) {
    const FrontierCluster& previous = detection.clusters[index - 1U];
    const FrontierCluster& current = detection.clusters[index];
    EXPECT_LT(std::tie(previous.id, previous.canonical_key, previous.cells),
              std::tie(current.id, current.canonical_key, current.cells));
  }
}

TEST(FrontierDetectorTest, IgnoresUnknownIslandBehindOccupiedWall) {
  const TaskRaster raster = BuildRaster(
      5U, 3U,
      {
          100, 100, 100, 100, 100,
          0,   0,   100, 0,   -1,
          100, 100, 100, 100, 100,
      });

  const FrontierDetection detection =
      FrontierDetector({0.0}).Detect(raster, GridIndex{0, 1});

  EXPECT_EQ(detection.reachable_free_cell_count, 2U);
  EXPECT_TRUE(detection.clusters.empty());
}

TEST(FrontierDetectorTest, ConcaveTaskNotchDoesNotCreateAFrontier) {
  const GridGeometry geometry{
      .width = 3U,
      .height = 3U,
      .resolution = 1.0,
      .origin_x = 0.0,
      .origin_y = 0.0,
      .origin_yaw = 0.0,
  };
  const std::array<std::int8_t, 9> data{
      0, 0, 100,
      0, 0, -1,
      0, 0, -1,
  };
  const OccupancyGridView map(geometry, data, 50);
  const Polygon2 concave{{
      Vec2{0.0, 0.0}, Vec2{3.0, 0.0}, Vec2{3.0, 1.0},
      Vec2{2.0, 1.0}, Vec2{2.0, 3.0}, Vec2{0.0, 3.0},
  }};
  const TaskRaster raster = TaskRaster::Build(map, concave);

  const FrontierDetection detection =
      FrontierDetector({0.0}).Detect(raster, GridIndex{0, 0});

  EXPECT_EQ(raster.Classify(GridIndex{2, 1}), CellState::kOutsideTask);
  EXPECT_TRUE(detection.clusters.empty());
}

TEST(FrontierDetectorTest, OutsideMapCellsNeverCreateFrontiers) {
  const GridGeometry geometry{
      .width = 2U,
      .height = 1U,
      .resolution = 1.0,
      .origin_x = 0.0,
      .origin_y = 0.0,
      .origin_yaw = 0.0,
  };
  const std::array<std::int8_t, 2> data{0, 0};
  const OccupancyGridView map(geometry, data, 50);
  const Polygon2 wider_task{{
      Vec2{0.0, 0.0}, Vec2{3.0, 0.0},
      Vec2{3.0, 1.0}, Vec2{0.0, 1.0},
  }};
  const TaskRaster raster = TaskRaster::Build(map, wider_task);

  const FrontierDetection detection =
      FrontierDetector({0.0}).Detect(raster, GridIndex{0, 0});

  EXPECT_EQ(raster.Classify(GridIndex{2, 0}), CellState::kOutsideMap);
  EXPECT_TRUE(detection.clusters.empty());
}

TEST(FrontierDetectorTest, FullyKnownTaskHasNoFrontier) {
  const TaskRaster raster = BuildRaster(3U, 1U, {0, 0, 0});

  const FrontierDetection detection =
      FrontierDetector({0.0}).Detect(raster, GridIndex{1, 0});

  EXPECT_EQ(detection.reason, FrontierDetectionReason::kOk);
  EXPECT_TRUE(detection.has_reachable_free_start);
  EXPECT_EQ(detection.reachable_free_cell_count, 3U);
  EXPECT_TRUE(detection.clusters.empty());
}

TEST(FrontierDetectorTest, RotatedMapFallbackChoosesWorldLexicographicNeighbor) {
  const TaskRaster raster = BuildRaster(
      4U, 4U,
      {
          100, 0,   100, 100,
          0,   100, 0,   100,
          100, 0,   100, 100,
          100, -1,  100, 100,
      },
      1.0, 10.0, -4.0, kPi / 2.0);

  const FrontierDetection detection =
      FrontierDetector({0.0}).Detect(raster, GridIndex{1, 1});

  EXPECT_EQ(detection.reachable_free_cell_count, 1U);
  ASSERT_EQ(detection.clusters.size(), 1U);
  EXPECT_EQ(detection.clusters[0].cells,
            (std::vector<GridIndex>{{1, 2}}));
}

TEST(FrontierDetectorTest, ReportsNoReachableFreeStartWithoutThrowing) {
  const TaskRaster raster = BuildRaster(
      3U, 3U,
      {
          100, 100, 100,
          100, -1,  100,
          100, 100, 100,
      });

  const FrontierDetection detection =
      FrontierDetector({0.0}).Detect(raster, GridIndex{1, 1});

  EXPECT_EQ(detection.reason,
            FrontierDetectionReason::kNoReachableFreeStart);
  EXPECT_FALSE(detection.has_reachable_free_start);
  EXPECT_EQ(detection.reachable_free_cell_count, 0U);
  EXPECT_TRUE(detection.clusters.empty());
}

TEST(FrontierDetectorTest, AppliesPhysicalMinimumLengthWithoutRoundingCells) {
  constexpr double kPlatformWidthM = 0.818;
  constexpr double kResolutionM = 0.2;
  const double minimum_length =
      std::max(kPlatformWidthM, 2.0 * kResolutionM);
  ASSERT_DOUBLE_EQ(minimum_length, 0.818);
  const TaskRaster raster = BuildRaster(2U, 1U, {0, -1}, kResolutionM);

  EXPECT_TRUE(FrontierDetector({minimum_length})
                  .Detect(raster, GridIndex{0, 0})
                  .clusters.empty());
  ASSERT_EQ(FrontierDetector({kResolutionM})
                .Detect(raster, GridIndex{0, 0})
                .clusters.size(),
            1U);
}

TEST(FrontierDetectorTest, DiscoveryOrderAndIndependentBuildDoNotChangeIdentity) {
  const std::vector<std::int8_t> data{
      100, -1,  -1,  100,
      0,   0,   0,   0,
      100, 100, 100, 100,
  };
  const TaskRaster first = BuildRaster(4U, 3U, data);
  const TaskRaster second = BuildRaster(4U, 3U, data);
  const FrontierDetector detector({0.0});

  const FrontierDetection left_to_right =
      detector.Detect(first, GridIndex{0, 1});
  const FrontierDetection right_to_left =
      detector.Detect(second, GridIndex{3, 1});

  ASSERT_EQ(left_to_right.clusters.size(), 1U);
  ASSERT_EQ(right_to_left.clusters.size(), 1U);
  EXPECT_EQ(left_to_right.clusters[0].id, right_to_left.clusters[0].id);
  EXPECT_EQ(left_to_right.clusters[0].canonical_key,
            right_to_left.clusters[0].canonical_key);
  EXPECT_EQ(left_to_right.clusters[0].cells,
            right_to_left.clusters[0].cells);
}

TEST(FrontierDetectorTest, CanonicalSortUsesWorldCoordinatesOnRotatedMap) {
  const TaskRaster raster = BuildRaster(
      3U, 3U,
      {
          0,   0,   -1,
          0,   100, 100,
          -1,  100, 100,
      },
      1.0, 0.0, 0.0, -kPi / 2.0);

  const FrontierDetection detection =
      FrontierDetector({0.0}).Detect(raster, GridIndex{0, 0});

  ASSERT_EQ(detection.clusters.size(), 1U);
  EXPECT_EQ(detection.clusters[0].cells,
            (std::vector<GridIndex>{{1, 0}, {0, 1}}));
  ASSERT_EQ(detection.clusters[0].interface_edges.size(), 2U);
  EXPECT_EQ(detection.clusters[0].interface_edges[0].free_cell,
            (GridIndex{1, 0}));
  EXPECT_EQ(detection.clusters[0].interface_edges[1].free_cell,
            (GridIndex{0, 1}));
}

TEST(FrontierDetectorTest, MillimeterQuantizationUsesHalfAwayFromZero) {
  const FrontierCluster positive =
      FrontierDetector({0.0})
          .Detect(BuildRaster(2U, 1U, {0, -1}, 0.001, 0.0, 0.0),
                  GridIndex{0, 0})
          .clusters.at(0);
  const FrontierCluster negative =
      FrontierDetector({0.0})
          .Detect(BuildRaster(2U, 1U, {0, -1}, 0.001, 0.0, -0.001),
                  GridIndex{0, 0})
          .clusters.at(0);

  EXPECT_EQ(positive.canonical_key,
            (std::vector<std::int64_t>{1, 1, 0}));
  EXPECT_EQ(negative.canonical_key,
            (std::vector<std::int64_t>{1, -1, 0}));
  EXPECT_EQ(positive.id, UINT64_C(10560200489033160511));
  EXPECT_EQ(negative.id, UINT64_C(10485969403947550996));
}

TEST(FrontierDetectorTest, RejectsInvalidParametersAndQuantizationOverflow) {
  EXPECT_THROW(FrontierDetector({-0.1}), std::invalid_argument);
  EXPECT_THROW(
      FrontierDetector({std::numeric_limits<double>::quiet_NaN()}),
      std::invalid_argument);
  EXPECT_THROW(
      FrontierDetector({std::numeric_limits<double>::infinity()}),
      std::invalid_argument);

  const TaskRaster raster =
      BuildRaster(2U, 1U, {0, -1}, 4.0, 1.0e16, 0.0);
  EXPECT_THROW(FrontierDetector({0.0}).Detect(raster, GridIndex{0, 0}),
               std::overflow_error);
}

TEST(FrontierDetectorTest, FullCanonicalKeyCanRejectDisplayIdCollision) {
  FrontierCluster first{
      .id = 42U,
      .cells = {},
      .interface_edges = {},
      .canonical_key = {1, 2, 0},
      .centroid = {},
      .length_m = 0.0,
  };
  FrontierCluster second{
      .id = 42U,
      .cells = {},
      .interface_edges = {},
      .canonical_key = {1, 3, 0},
      .centroid = {},
      .length_m = 0.0,
  };

  EXPECT_EQ(first.id, second.id);
  EXPECT_NE(first.canonical_key, second.canonical_key);
}

}  // namespace
}  // namespace lunar::pure_exploration
