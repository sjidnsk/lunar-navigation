#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "lunar_planner_training_bridge/visibility.hpp"

namespace lunar::planning::training {
namespace {

[[nodiscard]] std::size_t Index(const GridShape shape, const int row,
                                const int column) {
  return static_cast<std::size_t>(row) * shape.width +
         static_cast<std::size_t>(column);
}

TEST(VisibilityKernel, RejectsInvalidGeometry) {
  EXPECT_THROW((VisibilityKernel{0.0, 30.0}), std::invalid_argument);
  EXPECT_THROW((VisibilityKernel{0.2, 0.0}), std::invalid_argument);
  EXPECT_THROW(
      (VisibilityKernel{std::numeric_limits<double>::infinity(), 30.0}),
      std::invalid_argument);
}

TEST(VisibilityKernel, PrecomputesStableRowMajorOffsetsAcrossAllOctants) {
  const VisibilityKernel kernel{1.0, 3.0};
  const auto offsets = kernel.endpoint_offsets();

  ASSERT_FALSE(offsets.empty());
  EXPECT_TRUE(std::ranges::is_sorted(offsets, {}, [](const GridCell cell) {
    return std::pair{cell.row, cell.column};
  }));
  EXPECT_EQ(std::ranges::count(offsets, GridCell{-3, 0}), 1);
  EXPECT_EQ(std::ranges::count(offsets, GridCell{3, 0}), 1);
  EXPECT_EQ(std::ranges::count(offsets, GridCell{0, -3}), 1);
  EXPECT_EQ(std::ranges::count(offsets, GridCell{0, 3}), 1);
  EXPECT_EQ(std::ranges::count(offsets, GridCell{-2, -2}), 1);
  EXPECT_EQ(std::ranges::count(offsets, GridCell{-2, 2}), 1);
  EXPECT_EQ(std::ranges::count(offsets, GridCell{2, -2}), 1);
  EXPECT_EQ(std::ranges::count(offsets, GridCell{2, 2}), 1);
}

TEST(VisibilityKernel, RevealsFreeDiskAtMapEdgeDeterministically) {
  const VisibilityKernel kernel{1.0, 3.0};
  constexpr GridShape shape{.height = 7U, .width = 7U};
  const std::vector<float> obstacles(shape.height * shape.width, 0.0F);

  const auto first =
      kernel.RevealFromPose(shape, GridCell{0, 0}, obstacles);
  const auto second =
      kernel.RevealFromPose(shape, GridCell{0, 0}, obstacles);

  EXPECT_EQ(first, second);
  EXPECT_EQ(std::ranges::count(first, std::uint8_t{1}), 11);
  EXPECT_EQ(first[Index(shape, 0, 3)], 1U);
  EXPECT_EQ(first[Index(shape, 2, 2)], 1U);
  EXPECT_EQ(first[Index(shape, 3, 3)], 0U);
}

TEST(VisibilityKernel, RevealsFirstPositiveObstacleButNotCellsBehindIt) {
  const VisibilityKernel kernel{1.0, 3.0};
  constexpr GridShape shape{.height = 7U, .width = 7U};
  std::vector<float> obstacles(shape.height * shape.width, 0.0F);
  obstacles[Index(shape, 3, 5)] = 0.001F;

  const auto visible =
      kernel.RevealFromPose(shape, GridCell{3, 3}, obstacles);

  EXPECT_EQ(visible[Index(shape, 3, 4)], 1U);
  EXPECT_EQ(visible[Index(shape, 3, 5)], 1U);
  EXPECT_EQ(visible[Index(shape, 3, 6)], 0U);
}

TEST(VisibilityKernel, ForbiddenValuesCannotAffectReveal) {
  const VisibilityKernel kernel{1.0, 3.0};
  constexpr GridShape shape{.height = 7U, .width = 7U};
  const std::vector<float> obstacles(shape.height * shape.width, 0.0F);
  std::vector<std::uint8_t> forbidden(shape.height * shape.width, 0U);
  const auto first =
      kernel.RevealFromPose(shape, GridCell{3, 3}, obstacles);
  std::ranges::fill(forbidden, 1U);
  const auto second =
      kernel.RevealFromPose(shape, GridCell{3, 3}, obstacles);

  EXPECT_EQ(first, second);
  EXPECT_TRUE(std::ranges::all_of(forbidden,
                                  [](const auto value) { return value == 1U; }));
}

TEST(VisibilityKernel, EstimatesObservedOnlyWeightedCandidateGainsInOneBatch) {
  const VisibilityKernel kernel{1.0, 3.0};
  constexpr GridShape shape{.height = 7U, .width = 7U};
  std::vector<std::uint8_t> observed(shape.height * shape.width, 0U);
  std::vector<float> obstacles(shape.height * shape.width, 0.0F);
  std::vector<float> roi(shape.height * shape.width, 0.0F);
  std::vector<float> priority(shape.height * shape.width, 0.0F);
  observed[Index(shape, 3, 3)] = 1U;
  observed[Index(shape, 3, 4)] = 1U;
  observed[Index(shape, 1, 1)] = 1U;
  roi[Index(shape, 3, 5)] = 0.25F;
  priority[Index(shape, 3, 5)] = 0.75F;
  roi[Index(shape, 3, 6)] = 1.0F;
  priority[Index(shape, 3, 6)] = 1.0F;

  const std::vector<GridCell> candidates{{3, 3}, {1, 1}};
  const auto gains = kernel.EstimateCandidateGains(
      shape, observed, obstacles, roi, priority, candidates);

  ASSERT_EQ(gains.size(), candidates.size());
  EXPECT_FLOAT_EQ(gains[0].roi, 0.25F);
  EXPECT_FLOAT_EQ(gains[0].priority, 0.75F);
  EXPECT_FLOAT_EQ(gains[1].roi, 0.0F);
  EXPECT_FLOAT_EQ(gains[1].priority, 0.0F);

  obstacles[Index(shape, 3, 4)] = 0.01F;
  const auto blocked = kernel.EstimateCandidateGains(
      shape, observed, obstacles, roi, priority, candidates);
  EXPECT_FLOAT_EQ(blocked[0].roi, 0.0F);
  EXPECT_FLOAT_EQ(blocked[0].priority, 0.0F);
}

TEST(VisibilityKernel, RejectsMismatchedStorageAndOutOfBoundsCandidates) {
  const VisibilityKernel kernel{1.0, 3.0};
  constexpr GridShape shape{.height = 3U, .width = 3U};
  const std::vector<std::uint8_t> observed(9U, 1U);
  const std::vector<float> values(9U, 0.0F);

  EXPECT_THROW(
      kernel.EstimateCandidateGains(
          shape, std::span<const std::uint8_t>{observed}.first(8U), values,
          values, values, std::vector<GridCell>{{1, 1}}),
      std::invalid_argument);
  EXPECT_THROW(
      kernel.EstimateCandidateGains(
          shape, observed, values, values, values,
          std::vector<GridCell>{{3, 1}}),
      std::out_of_range);
  EXPECT_THROW(kernel.RevealFromPose(shape, GridCell{-1, 0}, values),
               std::out_of_range);
}

}  // namespace
}  // namespace lunar::planning::training
