#include "lunar_pure_exploration_core/information_gain.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace lunar::pure_exploration {
namespace {

struct RasterFixture {
  std::vector<std::int8_t> data;
  TaskRaster raster;
};

RasterFixture MakeRaster(double resolution, std::uint32_t width,
                         std::uint32_t height, double origin_x,
                         double origin_y, double origin_yaw,
                         std::vector<std::int8_t> data,
                         Polygon2 polygon) {
  OccupancyGridView map(
      GridGeometry{width, height, resolution, origin_x, origin_y, origin_yaw},
      data, 50);
  return RasterFixture{std::move(data), TaskRaster::Build(map, polygon)};
}

Polygon2 Rectangle(double min_x, double min_y, double max_x, double max_y) {
  return Polygon2{{{min_x, min_y}, {max_x, min_y}, {max_x, max_y},
                    {min_x, max_y}}};
}

CandidateView Candidate(double x, double y, double yaw) {
  return CandidateView{1U, 2U, 0U, CandidateKey{0, 0, 0},
                       Pose2{x, y, yaw}, 0.0};
}

InformationGainEvaluator Evaluator(
    double range = 10.0,
    double fov = std::numbers::pi / 2.0,
    std::size_t budget = 1000000U) {
  return InformationGainEvaluator({range, fov}, {budget});
}

TEST(InformationGain, ValidatesConstructionAndCandidate) {
  for (double value : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(InformationGainEvaluator({value, 1.0}, {1U}),
                 std::invalid_argument);
    EXPECT_THROW(InformationGainEvaluator({1.0, value}, {1U}),
                 std::invalid_argument);
  }
  EXPECT_THROW(
      InformationGainEvaluator(
          {1.0, std::nextafter(2.0 * std::numbers::pi,
                               std::numeric_limits<double>::infinity())},
          {1U}),
      std::invalid_argument);
  EXPECT_NO_THROW(
      InformationGainEvaluator({1.0, 2.0 * std::numbers::pi}, {1U}));
  EXPECT_THROW(InformationGainEvaluator({1.0, 1.0}, {0U}),
               std::invalid_argument);

  auto fixture = MakeRaster(1.0, 3, 3, 0.0, 0.0, 0.0,
                            std::vector<std::int8_t>(9U, -1),
                            Rectangle(0.0, 0.0, 3.0, 3.0));
  for (const Pose2 pose :
       {Pose2{NAN, 1.5, 0.0}, Pose2{1.5, INFINITY, 0.0},
        Pose2{1.5, 1.5, NAN}}) {
    EXPECT_THROW(Evaluator().Evaluate(fixture.raster,
                                      Candidate(pose.x, pose.y, pose.yaw)),
                 std::invalid_argument);
  }
}

TEST(InformationGain, IncludesExactRangeAndFovWithoutEpsilon) {
  const auto run = [](double target_x, double target_y, double range,
                      double yaw, double fov) {
    auto fixture = MakeRaster(1.0, 25, 25, -12.5, -12.5, 0.0,
                              std::vector<std::int8_t>(625U, 0),
                              Rectangle(-12.5, -12.5, 12.5, 12.5));
    auto& data = fixture.data;
    (void)data;
    // Rebuild with the one target cell unknown.
    std::vector<std::int8_t> cells(625U, 0);
    const int x = static_cast<int>(std::floor(target_x + 12.5));
    const int y = static_cast<int>(std::floor(target_y + 12.5));
    cells[static_cast<std::size_t>(y * 25 + x)] = -1;
    fixture = MakeRaster(1.0, 25, 25, -12.5, -12.5, 0.0,
                         std::move(cells),
                         Rectangle(-12.5, -12.5, 12.5, 12.5));
    return Evaluator(range, fov).Evaluate(fixture.raster,
                                          Candidate(0.0, 0.0, yaw));
  };

  EXPECT_EQ(run(10.0, 0.0, std::nextafter(10.0, 0.0), 0.0,
                std::numbers::pi / 2.0)
                .visible_unknown_cells,
            0U);
  EXPECT_EQ(run(10.0, 0.0, 10.0, 0.0, std::numbers::pi / 2.0)
                .visible_unknown_cells,
            1U);
  EXPECT_EQ(run(10.0, 0.0,
                std::nextafter(10.0,
                               std::numeric_limits<double>::infinity()),
                0.0, std::numbers::pi / 2.0)
                .visible_unknown_cells,
            1U);

  const double exact = std::numbers::pi / 4.0;
  EXPECT_EQ(run(5.0, 5.0, 10.0, 0.0, 2.0 * exact)
                .visible_unknown_cells,
            1U);
  EXPECT_EQ(run(5.0, -5.0, 10.0, 0.0, 2.0 * exact)
                .visible_unknown_cells,
            1U);
  const double outward = std::nextafter(exact,
                                        std::numeric_limits<double>::infinity());
  EXPECT_EQ(run(5.0, 5.0, 10.0,
                exact - outward, 2.0 * exact)
                .visible_unknown_cells,
            0U);
  EXPECT_EQ(run(-5.0, 0.0, 10.0, std::numbers::pi,
                2.0 * std::numbers::pi)
                .visible_unknown_cells,
            1U);
}

TEST(InformationGain, IncludesFiftyAndOneHundredCellRangeBoundaries) {
  const auto run = [](double resolution, std::int32_t target_x) {
    const std::uint32_t width = static_cast<std::uint32_t>(target_x + 2);
    std::vector<std::int8_t> data(width, 0);
    data[static_cast<std::size_t>(target_x)] = -1;
    auto fixture = MakeRaster(resolution, width, 1, 0.0, 0.0, 0.0,
                              std::move(data),
                              Rectangle(0.0, 0.0, width * resolution,
                                        resolution));
    return Evaluator().Evaluate(
        fixture.raster, Candidate(resolution / 2.0, resolution / 2.0, 0.0));
  };
  EXPECT_EQ(run(0.2, 50).visible_unknown_cells, 1U);
  EXPECT_EQ(run(0.1, 100).visible_unknown_cells, 1U);
  EXPECT_EQ(run(0.2, 51).visible_unknown_cells, 0U);
  EXPECT_EQ(run(0.1, 101).visible_unknown_cells, 0U);
}

TEST(InformationGain, WrapsYawAcrossPiAndTreatsZeroDistanceAsInFov) {
  std::vector<std::int8_t> data(5U, 0);
  data[0U] = -1;
  data[2U] = -1;
  auto fixture = MakeRaster(1.0, 5, 1, -2.5, -0.5, 0.0,
                            std::move(data),
                            Rectangle(-2.5, -0.5, 2.5, 0.5));
  const auto positive = Evaluator(3.0, 0.2)
                            .Evaluate(fixture.raster,
                                      Candidate(0.0, 0.0,
                                                std::numbers::pi));
  const auto negative = Evaluator(3.0, 0.2)
                            .Evaluate(fixture.raster,
                                      Candidate(0.0, 0.0,
                                                -std::numbers::pi));
  EXPECT_EQ(positive.visible_unknown_cells, 2U);
  EXPECT_EQ(negative.visible_unknown_cells, positive.visible_unknown_cells);
}

TEST(InformationGain, UsesGridYawAndIsTranslationAndRotationInvariant) {
  const auto evaluate = [](double origin_x, double origin_y,
                           double origin_yaw) {
    constexpr std::uint32_t size = 9U;
    std::vector<std::int8_t> data(size * size, 0);
    data[4U * size + 6U] = -1;
    const GridGeometry geometry{size, size, 1.0, origin_x, origin_y,
                                origin_yaw};
    const auto world = OccupancyGridView::GridToWorld(geometry, {4.5, 4.5});
    const auto corners = std::vector<Vec2>{
        *OccupancyGridView::GridToWorld(geometry, {0.0, 0.0}),
        *OccupancyGridView::GridToWorld(geometry, {9.0, 0.0}),
        *OccupancyGridView::GridToWorld(geometry, {9.0, 9.0}),
        *OccupancyGridView::GridToWorld(geometry, {0.0, 9.0})};
    auto fixture = MakeRaster(1.0, size, size, origin_x, origin_y, origin_yaw,
                              std::move(data), Polygon2{corners});
    return Evaluator().Evaluate(
        fixture.raster,
        Candidate(world->x, world->y, origin_yaw));
  };
  const auto baseline = evaluate(0.0, 0.0, 0.0);
  EXPECT_EQ(evaluate(0.0, 0.0, std::numbers::pi / 6.0)
                .visible_unknown_cells,
            baseline.visible_unknown_cells);
  EXPECT_EQ(evaluate(0.0, 0.0, std::numbers::pi / 2.0)
                .visible_unknown_cells,
            baseline.visible_unknown_cells);
  EXPECT_EQ(evaluate(1e9, -1e9, std::numbers::pi / 6.0)
                .visible_unknown_cells,
            baseline.visible_unknown_cells);
}

TEST(InformationGain, PreservesNativeStatesAndCountsOnlyTargets) {
  const std::vector<std::int8_t> native{-1, 0, 49, 50, 100, 101, -128};
  auto classified = MakeRaster(1.0, 7, 1, 0.0, 0.0, 0.0, native,
                               Rectangle(0.0, 0.0, 7.0, 1.0));
  EXPECT_EQ(classified.raster.Classify({0, 0}), CellState::kUnknown);
  EXPECT_EQ(classified.raster.Classify({1, 0}), CellState::kFree);
  EXPECT_EQ(classified.raster.Classify({2, 0}), CellState::kFree);
  EXPECT_EQ(classified.raster.Classify({3, 0}), CellState::kOccupied);
  EXPECT_EQ(classified.raster.Classify({4, 0}), CellState::kOccupied);
  EXPECT_EQ(classified.raster.Classify({5, 0}), CellState::kUnknown);
  EXPECT_EQ(classified.raster.Classify({6, 0}), CellState::kUnknown);

  const auto target_gain = [](std::int8_t target, bool outside_map) {
    auto fixture = MakeRaster(
        1.0, outside_map ? 1U : 2U, 1, 0.0, 0.0, 0.0,
        outside_map ? std::vector<std::int8_t>{0}
                    : std::vector<std::int8_t>{0, target},
        Rectangle(0.0, 0.0, 2.0, 1.0));
    return Evaluator(2.0, 2.0 * std::numbers::pi)
        .Evaluate(fixture.raster, Candidate(0.5, 0.5, 0.0))
        .visible_unknown_cells;
  };
  EXPECT_EQ(target_gain(-1, false), 1U);
  EXPECT_EQ(target_gain(-128, false), 1U);
  EXPECT_EQ(target_gain(0, false), 0U);
  EXPECT_EQ(target_gain(49, false), 0U);
  EXPECT_EQ(target_gain(50, false), 0U);
  EXPECT_EQ(target_gain(100, false), 0U);
  EXPECT_EQ(target_gain(101, false), 1U);
  EXPECT_EQ(target_gain(0, true), 1U);
}

TEST(InformationGain, BlockersAndTargetExclusionFollowMatrix) {
  const auto run = [](std::int8_t middle, std::int8_t target) {
    std::vector<std::int8_t> data{0, middle, target};
    auto fixture = MakeRaster(1.0, 3, 1, 0.0, 0.0, 0.0, std::move(data),
                              Rectangle(0.0, 0.0, 3.0, 1.0));
    return Evaluator(3.0, 0.1)
        .Evaluate(fixture.raster, Candidate(0.5, 0.5, 0.0));
  };
  EXPECT_EQ(run(50, -1).visible_unknown_cells, 0U);
  EXPECT_EQ(run(0, -1).visible_unknown_cells, 1U);
  EXPECT_EQ(run(-1, -1).visible_unknown_cells, 2U);
  EXPECT_EQ(run(0, 50).visible_unknown_cells, 0U);
}

TEST(InformationGain, OccupiedTargetConsumesNoRayWork) {
  std::vector<std::int8_t> data{0, 0, 50};
  auto fixture = MakeRaster(1.0, 3, 1, 0.0, 0.0, 0.0, std::move(data),
                            Rectangle(0.0, 0.0, 3.0, 1.0));
  const auto result = Evaluator(3.0, 2.0 * std::numbers::pi, 49U)
                          .Evaluate(fixture.raster,
                                    Candidate(0.5, 0.5, 0.0));
  EXPECT_EQ(result.visible_unknown_cells, 0U);
}

TEST(InformationGain, CornerAndGridLineBlockersUseSupercover) {
  std::vector<std::int8_t> diagonal(9U, 0);
  diagonal[8] = -1;
  diagonal[1] = 50;
  auto blocked = MakeRaster(1.0, 3, 3, 0.0, 0.0, 0.0,
                            std::move(diagonal),
                            Rectangle(0.0, 0.0, 3.0, 3.0));
  EXPECT_EQ(Evaluator(5.0, 2.0 * std::numbers::pi)
                .Evaluate(blocked.raster, Candidate(0.5, 0.5, 0.0))
                .visible_unknown_cells,
            0U);

}

TEST(InformationGain, ExactBinaryCornerBlockerOccludesUnknownTarget) {
  constexpr std::uint32_t width = 40U;
  constexpr std::uint32_t height = 28U;
  std::vector<std::int8_t> data(width * height, 0);
  data[1U * width + 3U] = 50;
  data[9U * width + 15U] = -1;
  auto fixture = MakeRaster(0.5, width, height, 0.0, 0.0, 0.0,
                            std::move(data),
                            Rectangle(0.0, 0.0, 20.0, 14.0));
  const auto gain = Evaluator(10.0, 2.0 * std::numbers::pi, 5000U)
                        .Evaluate(fixture.raster,
                                  Candidate(0.25, 0.25, 0.0));
  EXPECT_EQ(gain.visible_unknown_cells, 0U);
  EXPECT_DOUBLE_EQ(gain.visible_unknown_area_m2, 0.0);
}

TEST(InformationGain, ConcaveOutsideTaskBlocksReentry) {
  std::vector<std::int8_t> data(25U, 0);
  data[2U * 5U + 4U] = -1;
  const Polygon2 concave{{{0, 0}, {5, 0}, {5, 5}, {3, 5},
                           {3, 2}, {2, 2}, {2, 5}, {0, 5}}};
  auto fixture = MakeRaster(1.0, 5, 5, 0.0, 0.0, 0.0, std::move(data),
                            concave);
  EXPECT_EQ(Evaluator(10.0, 2.0 * std::numbers::pi)
                .Evaluate(fixture.raster, Candidate(0.5, 2.5, 0.0))
                .visible_unknown_cells,
            0U);
}

TEST(InformationGain, FiveYawResultsAreOrderIndependentAndKeepZero) {
  std::vector<std::int8_t> data(11U * 11U, 0);
  for (const auto& [x, y] :
       {std::pair{8, 5}, std::pair{8, 8}, std::pair{5, 8},
        std::pair{2, 8}, std::pair{2, 5}}) {
    data[static_cast<std::size_t>(y * 11 + x)] = -1;
  }
  auto fixture = MakeRaster(1.0, 11, 11, 0.0, 0.0, 0.0,
                            std::move(data),
                            Rectangle(0.0, 0.0, 11.0, 11.0));
  const std::vector<double> yaws{-std::numbers::pi / 4.0,
                                 -std::numbers::pi / 8.0, 0.0,
                                 std::numbers::pi / 8.0,
                                 std::numbers::pi / 4.0};
  std::vector<std::uint32_t> first;
  for (double yaw : yaws) {
    first.push_back(Evaluator().Evaluate(fixture.raster,
                                         Candidate(5.5, 5.5, yaw))
                        .visible_unknown_cells);
  }
  std::vector<std::uint32_t> reverse;
  for (auto it = yaws.rbegin(); it != yaws.rend(); ++it) {
    reverse.push_back(Evaluator().Evaluate(fixture.raster,
                                           Candidate(5.5, 5.5, *it))
                          .visible_unknown_cells);
  }
  std::reverse(reverse.begin(), reverse.end());
  EXPECT_EQ(first, reverse);
  std::sort(first.begin(), first.end());
  EXPECT_GE(std::unique(first.begin(), first.end()) - first.begin(), 3);

  auto free = MakeRaster(1.0, 3, 3, 0.0, 0.0, 0.0,
                         std::vector<std::int8_t>(9U, 0),
                         Rectangle(0.0, 0.0, 3.0, 3.0));
  const auto zero = Evaluator().Evaluate(free.raster,
                                         Candidate(1.5, 1.5, 0.0));
  EXPECT_EQ(zero.visible_unknown_cells, 0U);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(zero.visible_unknown_area_m2),
            std::bit_cast<std::uint64_t>(0.0));
}

TEST(InformationGain, ResolutionGoldensAndFrozenAreaExpression) {
  const auto sector = [](double resolution) {
    const std::uint32_t radius =
        static_cast<std::uint32_t>(10.0 / resolution);
    const std::uint32_t width = radius + 1U;
    const std::uint32_t height = radius + 1U;
    std::vector<std::int8_t> data(width * height, -1);
    data[0U] = 0;
    auto fixture = MakeRaster(resolution, width, height, 0.0, 0.0, 0.0,
                              std::move(data),
                              Rectangle(0.0,
                                        -static_cast<double>(radius) * resolution,
                                        width * resolution,
                                        height * resolution));
    return Evaluator().Evaluate(fixture.raster,
                                Candidate(resolution / 2.0,
                                          resolution / 2.0, 0.0));
  };
  const auto coarse = sector(0.2);
  const auto fine = sector(0.1);
  EXPECT_EQ(coarse.visible_unknown_cells, 1996U);
  EXPECT_DOUBLE_EQ(coarse.visible_unknown_area_m2, 79.84);
  EXPECT_EQ(fine.visible_unknown_cells, 7924U);
  EXPECT_DOUBLE_EQ(fine.visible_unknown_area_m2, 79.24);
  EXPECT_DOUBLE_EQ(
      coarse.visible_unknown_area_m2,
      static_cast<double>(coarse.visible_unknown_cells) * (0.2 * 0.2));
}

TEST(InformationGain, AlignedRectangleHasTwentyAndEightyCellsAtSameArea) {
  const auto rectangle_gain = [](double resolution) {
    const std::uint32_t width =
        static_cast<std::uint32_t>(5.0 / resolution);
    const std::uint32_t height =
        static_cast<std::uint32_t>(2.0 / resolution);
    std::vector<std::int8_t> data(width * height, 0);
    for (std::uint32_t y = 0U; y < height; ++y) {
      const double center_y = -1.0 + (static_cast<double>(y) + 0.5) * resolution;
      for (std::uint32_t x = 0U; x < width; ++x) {
        const double center_x = -1.0 +
                                (static_cast<double>(x) + 0.5) * resolution;
        if (center_x >= 2.0 && center_x < 3.0 &&
            center_y >= -0.4 && center_y < 0.4) {
          data[static_cast<std::size_t>(y) * width + x] = -1;
        }
      }
    }
    auto fixture = MakeRaster(resolution, width, height, -1.0, -1.0, 0.0,
                              std::move(data),
                              Rectangle(-1.0, -1.0, 4.0, 1.0));
    return Evaluator().Evaluate(fixture.raster, Candidate(0.0, 0.0, 0.0));
  };
  const auto coarse = rectangle_gain(0.2);
  const auto fine = rectangle_gain(0.1);
  EXPECT_EQ(coarse.visible_unknown_cells, 20U);
  EXPECT_EQ(fine.visible_unknown_cells, 80U);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(coarse.visible_unknown_area_m2),
            std::bit_cast<std::uint64_t>(fine.visible_unknown_area_m2));
  EXPECT_DOUBLE_EQ(coarse.visible_unknown_area_m2, 0.8);
}

TEST(InformationGain, RejectsUnrepresentableCandidateAndOutputArea) {
  const GridGeometry rotated_geometry{1, 1, 1.0, 0.0, 0.0,
                                      std::numbers::pi / 4.0};
  const Polygon2 rotated_polygon{{
      *OccupancyGridView::GridToWorld(rotated_geometry, {0.0, 0.0}),
      *OccupancyGridView::GridToWorld(rotated_geometry, {1.0, 0.0}),
      *OccupancyGridView::GridToWorld(rotated_geometry, {1.0, 1.0}),
      *OccupancyGridView::GridToWorld(rotated_geometry, {0.0, 1.0}),
  }};
  auto ordinary = MakeRaster(1.0, 1, 1, 0.0, 0.0,
                             std::numbers::pi / 4.0,
                             std::vector<std::int8_t>{-1}, rotated_polygon);
  EXPECT_THROW(Evaluator().Evaluate(
                   ordinary.raster,
                   Candidate(std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::max(), 0.0)),
               std::overflow_error);

  constexpr double huge_resolution = 1.35e154;
  const double low = huge_resolution * 0.1;
  const double high = huge_resolution * 0.9;
  auto huge = MakeRaster(huge_resolution, 1, 1, 0.0, 0.0, 0.0,
                         std::vector<std::int8_t>{-1},
                         Rectangle(low, low, high, high));
  EXPECT_THROW(InformationGainEvaluator({huge_resolution, 1.0}, {10U})
                   .Evaluate(huge.raster,
                             Candidate(huge_resolution * 0.5,
                                       huge_resolution * 0.5, 0.0)),
               std::overflow_error);
}

TEST(InformationGain, HandlesTaskCellsAtBothInt32ExtremesDeterministically) {
  const auto run = [](std::int32_t index) {
    const double x = static_cast<double>(index);
    auto fixture = MakeRaster(1.0, 1, 1, 0.0, 0.0, 0.0,
                              std::vector<std::int8_t>{0},
                              Rectangle(x, 0.0, x + 1.0, 1.0));
    const auto evaluator = Evaluator(1.0, 2.0 * std::numbers::pi, 20U);
    const auto first = evaluator.Evaluate(
        fixture.raster, Candidate(x + 0.5, 0.5, 0.0));
    const auto second = evaluator.Evaluate(
        fixture.raster, Candidate(x + 0.5, 0.5, 0.0));
    EXPECT_EQ(first.visible_unknown_cells, second.visible_unknown_cells);
  };
  run(std::numeric_limits<std::int32_t>::min());
  run(std::numeric_limits<std::int32_t>::max());
}

TEST(InformationGain, RepeatsBoundedEvaluationAtTaskRasterCellCap) {
  constexpr std::uint32_t side = 1024U;
  static_assert(static_cast<std::size_t>(side) * side == 1048576U);
  auto fixture = MakeRaster(1.0, side, side, 0.0, 0.0, 0.0,
                            std::vector<std::int8_t>(
                                static_cast<std::size_t>(side) * side, 0),
                            Rectangle(0.0, 0.0, side, side));
  const auto evaluator = Evaluator(1.0, 2.0 * std::numbers::pi, 20U);
  const CandidateView candidate = Candidate(512.5, 512.5, 0.0);
  const auto first = evaluator.Evaluate(fixture.raster, candidate);
  const auto second = evaluator.Evaluate(fixture.raster, candidate);
  EXPECT_EQ(first.visible_unknown_cells, 0U);
  EXPECT_EQ(first.visible_unknown_cells, second.visible_unknown_cells);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.visible_unknown_area_m2),
            std::bit_cast<std::uint64_t>(second.visible_unknown_area_m2));
}

TEST(InformationGain, TightAabbAndRayHaveExactElevenUnitBudget) {
  auto fixture = MakeRaster(10.0, 3, 3, -15.0, -15.0, 0.0,
                            std::vector<std::int8_t>(9U, 0),
                            Rectangle(-15.0, -15.0, 15.0, 15.0));
  // Adjacent cell center at (10,0) is the sole unknown target.
  std::vector<std::int8_t> data(9U, 0);
  data[5U] = -1;
  fixture = MakeRaster(10.0, 3, 3, -15.0, -15.0, 0.0,
                       std::move(data),
                       Rectangle(-15.0, -15.0, 15.0, 15.0));
  EXPECT_EQ(Evaluator(10.0, 2.0 * std::numbers::pi, 11U)
                .Evaluate(fixture.raster, Candidate(0.0, 0.0, 0.0))
                .visible_unknown_cells,
            1U);
  EXPECT_THROW(Evaluator(10.0, 2.0 * std::numbers::pi, 10U)
                   .Evaluate(fixture.raster, Candidate(0.0, 0.0, 0.0)),
               std::length_error);
}

TEST(InformationGain, PreflightsTheInt32ClippedAabbArea) {
  const double x =
      static_cast<double>(std::numeric_limits<std::int32_t>::max());
  auto fixture = MakeRaster(1.0, 1, 1, 0.0, 0.0, 0.0,
                            std::vector<std::int8_t>{0},
                            Rectangle(x, 0.0, x + 1.0, 1.0));
  const auto gain = Evaluator(1.0, 2.0 * std::numbers::pi, 7U)
                        .Evaluate(fixture.raster,
                                  Candidate(x + 0.5, 0.5, 0.0));
  EXPECT_EQ(gain.visible_unknown_cells, 1U);
  EXPECT_DOUBLE_EQ(gain.visible_unknown_area_m2, 1.0);
}

TEST(InformationGain, PreflightsTinyResolutionAabbBeforeTraversal) {
  auto fixture = MakeRaster(1.0, 1, 1, 0.0, 0.0, 0.0,
                            std::vector<std::int8_t>{-1},
                            Rectangle(0.0, 0.0, 1.0, 1.0));
  EXPECT_THROW(InformationGainEvaluator({10.0, 1.0}, {100U})
                   .Evaluate(fixture.raster,
                             Candidate(0.5, 0.5, 0.0)),
               std::length_error);
}

}  // namespace
}  // namespace lunar::pure_exploration
