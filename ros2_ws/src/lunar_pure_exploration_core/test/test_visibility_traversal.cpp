#include "detail/visibility_traversal.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lunar::pure_exploration::detail {
namespace {

struct SeenGroup {
  long double t;
  std::vector<TraceCell> cells;
};

std::vector<SeenGroup> Trace(Vec2 start, Vec2 end, std::size_t limit,
                             TraceSummary* summary = nullptr) {
  VisibilityWorkBudget budget{limit, 0U};
  std::vector<SeenGroup> groups;
  const TraceSummary result = TraceClosedSegment(
      start, end, budget,
      [&groups](long double t, std::span<const TraceCell> cells) {
        groups.push_back(SeenGroup{t, {cells.begin(), cells.end()}});
        return TraceControl::kContinue;
      });
  EXPECT_EQ(result.visited_cell_count, budget.used);
  if (summary != nullptr) {
    *summary = result;
  }
  return groups;
}

std::set<TraceCell> Flatten(const std::vector<SeenGroup>& groups) {
  std::set<TraceCell> cells;
  for (const SeenGroup& group : groups) {
    cells.insert(group.cells.begin(), group.cells.end());
  }
  return cells;
}

constexpr std::int64_t kReferenceScale = INT64_C(1) << 20U;
constexpr std::int64_t kReferenceMaximumTerm = 32 * kReferenceScale;
static_assert(kReferenceMaximumTerm <=
              std::numeric_limits<std::int64_t>::max() /
                  kReferenceMaximumTerm);

struct ReferenceFraction {
  std::int64_t numerator;
  std::int64_t denominator;
};

std::int64_t CheckedReferenceProduct(std::int64_t left,
                                     std::int64_t right) {
  if (std::abs(left) > kReferenceMaximumTerm ||
      std::abs(right) > kReferenceMaximumTerm) {
    throw std::overflow_error("reference fixture exceeds exact int64 bound");
  }
  return left * right;
}

struct ReferenceFractionLess {
  bool operator()(const ReferenceFraction& left,
                  const ReferenceFraction& right) const {
    return CheckedReferenceProduct(left.numerator, right.denominator) <
           CheckedReferenceProduct(right.numerator, left.denominator);
  }
};

int CompareReferenceFraction(const ReferenceFraction& left,
                             const ReferenceFraction& right) {
  const std::int64_t left_product =
      CheckedReferenceProduct(left.numerator, right.denominator);
  const std::int64_t right_product =
      CheckedReferenceProduct(right.numerator, left.denominator);
  return left_product < right_product ? -1 : left_product > right_product ? 1
                                                                       : 0;
}

ReferenceFraction Maximum(ReferenceFraction left,
                          const ReferenceFraction& right) {
  return CompareReferenceFraction(left, right) < 0 ? right : left;
}

ReferenceFraction Minimum(ReferenceFraction left,
                          const ReferenceFraction& right) {
  return CompareReferenceFraction(left, right) > 0 ? right : left;
}

std::int64_t FloorScaled(std::int64_t numerator, std::int64_t scale) {
  std::int64_t quotient = numerator / scale;
  if (numerator % scale < 0) {
    --quotient;
  }
  return quotient;
}

bool RestrictReferenceAxis(std::int64_t start, std::int64_t delta,
                           std::int64_t cell, std::int64_t scale,
                           ReferenceFraction& lower,
                           ReferenceFraction& upper) {
  const std::int64_t cell_low = cell * scale;
  const std::int64_t cell_high = (cell + 1) * scale;
  if (delta == 0) {
    return cell_low <= start && start <= cell_high;
  }
  ReferenceFraction axis_lower;
  ReferenceFraction axis_upper;
  if (delta > 0) {
    axis_lower = {cell_low - start, delta};
    axis_upper = {cell_high - start, delta};
  } else {
    axis_lower = {start - cell_high, -delta};
    axis_upper = {start - cell_low, -delta};
  }
  lower = Maximum(std::move(lower), axis_lower);
  upper = Minimum(std::move(upper), axis_upper);
  return CompareReferenceFraction(lower, upper) <= 0;
}

std::vector<std::vector<TraceCell>> ReferenceClosedGroups(
    std::array<std::int64_t, 2> start,
    std::array<std::int64_t, 2> end, std::int64_t scale) {
  const std::int64_t minimum_x_numerator = std::min(start[0], end[0]);
  const std::int64_t minimum_y_numerator = std::min(start[1], end[1]);
  std::int64_t minimum_x = FloorScaled(minimum_x_numerator, scale);
  std::int64_t minimum_y = FloorScaled(minimum_y_numerator, scale);
  if (minimum_x_numerator % scale == 0) {
    --minimum_x;
  }
  if (minimum_y_numerator % scale == 0) {
    --minimum_y;
  }
  const std::int64_t maximum_x =
      FloorScaled(std::max(start[0], end[0]), scale);
  const std::int64_t maximum_y =
      FloorScaled(std::max(start[1], end[1]), scale);
  const std::int64_t delta_x = end[0] - start[0];
  const std::int64_t delta_y = end[1] - start[1];

  std::map<ReferenceFraction, std::vector<TraceCell>, ReferenceFractionLess>
      groups;
  for (std::int64_t y = minimum_y; y <= maximum_y; ++y) {
    for (std::int64_t x = minimum_x; x <= maximum_x; ++x) {
      ReferenceFraction lower{0, 1};
      ReferenceFraction upper{1, 1};
      if (!RestrictReferenceAxis(start[0], delta_x, x, scale, lower, upper) ||
          !RestrictReferenceAxis(start[1], delta_y, y, scale, lower, upper)) {
        continue;
      }
      groups[lower].push_back({x, y});
    }
  }

  std::vector<std::vector<TraceCell>> result;
  for (auto& [time, cells] : groups) {
    (void)time;
    std::sort(cells.begin(), cells.end(),
              [](const TraceCell& left, const TraceCell& right) {
                return left.x < right.x ||
                       (left.x == right.x && left.y < right.y);
              });
    result.push_back(std::move(cells));
  }
  return result;
}

std::vector<std::vector<TraceCell>> ProductionClosedGroups(
    std::array<std::int64_t, 2> start,
    std::array<std::int64_t, 2> end, std::int64_t scale) {
  const Vec2 start_point{static_cast<double>(start[0]) / scale,
                         static_cast<double>(start[1]) / scale};
  const Vec2 end_point{static_cast<double>(end[0]) / scale,
                       static_cast<double>(end[1]) / scale};
  const auto groups = Trace(start_point, end_point, 256U);
  std::vector<std::vector<TraceCell>> result;
  for (const SeenGroup& group : groups) {
    result.push_back(group.cells);
  }
  return result;
}

TEST(VisibilityTraversal, RejectsZeroBudgetAndConsumesExactly) {
  VisibilityWorkBudget zero{0U, 0U};
  EXPECT_THROW(ConsumeVisibilityWork(zero), std::length_error);
  VisibilityWorkBudget one{1U, 0U};
  EXPECT_NO_THROW(ConsumeVisibilityWork(one));
  EXPECT_EQ(one.used, 1U);
  EXPECT_THROW(ConsumeVisibilityWork(one), std::length_error);
}

TEST(VisibilityTraversal, EmitsOneTwoAndFourCellsAtStart) {
  const auto ordinary = Trace({0.25, 0.25}, {0.25, 0.25}, 1U);
  ASSERT_EQ(ordinary.size(), 1U);
  EXPECT_EQ(ordinary[0].cells, (std::vector<TraceCell>{{0, 0}}));

  const auto line = Trace({0.25, 1.0}, {0.25, 1.0}, 2U);
  ASSERT_EQ(line.size(), 1U);
  EXPECT_EQ(line[0].cells,
            (std::vector<TraceCell>{{0, 0}, {0, 1}}));

  const auto corner = Trace({1.0, 1.0}, {1.0, 1.0}, 4U);
  ASSERT_EQ(corner.size(), 1U);
  EXPECT_EQ(corner[0].cells,
            (std::vector<TraceCell>{{0, 0}, {0, 1}, {1, 0}, {1, 1}}));
}

TEST(VisibilityTraversal, CornerCrossingProducesCanonicalThreeCellTie) {
  TraceSummary summary{};
  const auto groups = Trace({0.5, 0.5}, {2.5, 2.5}, 7U, &summary);
  ASSERT_EQ(groups.size(), 3U);
  EXPECT_EQ(groups[0].cells, (std::vector<TraceCell>{{0, 0}}));
  EXPECT_EQ(groups[1].cells,
            (std::vector<TraceCell>{{0, 1}, {1, 0}, {1, 1}}));
  EXPECT_EQ(groups[2].cells,
            (std::vector<TraceCell>{{1, 2}, {2, 1}, {2, 2}}));
  EXPECT_LT(groups[0].t, groups[1].t);
  EXPECT_LT(groups[1].t, groups[2].t);
  EXPECT_EQ(summary.visited_cell_count, 7U);

  const auto repeated = Trace({0.5, 0.5}, {2.5, 2.5}, 7U);
  ASSERT_EQ(repeated.size(), groups.size());
  for (std::size_t index = 0U; index < groups.size(); ++index) {
    EXPECT_EQ(repeated[index].t, groups[index].t);
    EXPECT_EQ(repeated[index].cells, groups[index].cells);
  }
}

TEST(VisibilityTraversal, ExactBinaryCornerEventEmitsOneCompleteTieGroup) {
  const auto groups = Trace({0.5, 0.5}, {15.5, 9.5}, 28U);
  const std::vector<TraceCell> expected{{2, 2}, {3, 1}, {3, 2}};
  const auto tie = std::find_if(
      groups.begin(), groups.end(), [&expected](const SeenGroup& group) {
        return group.cells == expected;
      });
  ASSERT_NE(tie, groups.end());
  EXPECT_EQ(std::count_if(groups.begin(), groups.end(),
                          [](const SeenGroup& group) {
                            return std::ranges::find(group.cells,
                                                     TraceCell{2, 2}) !=
                                   group.cells.end() ||
                                   std::ranges::find(group.cells,
                                                     TraceCell{3, 1}) !=
                                       group.cells.end() ||
                                   std::ranges::find(group.cells,
                                                     TraceCell{3, 2}) !=
                                       group.cells.end();
                          }),
            1);
}

TEST(VisibilityTraversal, ClosedEndpointsAreDirectionSymmetric) {
  const std::set<TraceCell> horizontal_expected{
      {0, 0}, {1, 0}, {2, 0}, {3, 0},
      {4, 0}, {5, 0}, {6, 0}, {7, 0}};
  const auto horizontal_forward = Trace({0.1, 0.3}, {7.0, 0.3}, 8U);
  const auto horizontal_reverse = Trace({7.0, 0.3}, {0.1, 0.3}, 8U);
  EXPECT_EQ(Flatten(horizontal_forward), horizontal_expected);
  EXPECT_EQ(Flatten(horizontal_reverse), horizontal_expected);

  const std::set<TraceCell> corner_expected{
      {0, 0}, {0, 1}, {1, 0}, {1, 1}, {1, 2}, {2, 1},
      {2, 2}, {2, 3}, {3, 2}, {3, 3}, {3, 4}, {4, 3},
      {4, 4}, {4, 5}, {5, 4}, {5, 5}, {5, 6}, {6, 5},
      {6, 6}, {6, 7}, {7, 6}, {7, 7}};
  const auto corner_forward = Trace({0.1, 0.1}, {7.0, 7.0}, 22U);
  const auto corner_reverse = Trace({7.0, 7.0}, {0.1, 0.1}, 22U);
  EXPECT_EQ(Flatten(corner_forward), corner_expected);
  EXPECT_EQ(Flatten(corner_reverse), corner_expected);
}

TEST(VisibilityTraversal, ExactDyadicEventsCoverSubnormalAndInt64FloorDomain) {
  const double denormal = std::numeric_limits<double>::denorm_min();
  const std::set<TraceCell> subnormal_expected{{0, 0}, {1, 0}};
  EXPECT_EQ(Flatten(Trace({denormal, 0.25}, {1.0, 0.25}, 2U)),
            subnormal_expected);
  EXPECT_EQ(Flatten(Trace({1.0, 0.25}, {denormal, 0.25}, 2U)),
            subnormal_expected);

  const double negative_limit = -std::ldexp(1.0, 63);
  const double first_inside =
      std::nextafter(negative_limit, std::numeric_limits<double>::infinity());
  const double second_inside =
      std::nextafter(first_inside, std::numeric_limits<double>::infinity());
  const auto forward = Trace({first_inside, 0.25}, {second_inside, 0.25},
                             1026U);
  const auto reverse = Trace({second_inside, 0.25}, {first_inside, 0.25},
                             1026U);
  EXPECT_EQ(Flatten(forward), Flatten(reverse));
  EXPECT_EQ(Flatten(forward).size(), 1026U);
  EXPECT_TRUE(Flatten(forward).contains(
      TraceCell{std::numeric_limits<std::int64_t>::min() + 1023, 0}));

  VisibilityWorkBudget edge_budget{4U, 0U};
  EXPECT_THROW(TraceClosedSegment(
                   {negative_limit, 0.25}, {first_inside, 0.25}, edge_budget,
                   [](long double, std::span<const TraceCell>) {
                     return TraceControl::kContinue;
                   }),
               std::overflow_error);
}

TEST(VisibilityTraversal, ExactEventComparisonHandlesDenseMultiwordProducts) {
  using traversal_internal::AddShifted64;
  using traversal_internal::AbsoluteDifference;
  using traversal_internal::CheckedAdd;
  using traversal_internal::CheckedMultiply;
  using traversal_internal::Compare;
  using traversal_internal::CompareEventRatios;
  using traversal_internal::ExactFromDouble;
  using traversal_internal::ExactUnsigned;

  const auto zero = ExactFromDouble(0.0);
  const auto denormal = ExactFromDouble(
      std::numeric_limits<double>::denorm_min());
  const auto one = AbsoluteDifference(ExactFromDouble(1.0), zero);
  const auto dense = AbsoluteDifference(ExactFromDouble(1.0), denormal);
  EXPECT_LT(Compare(CheckedMultiply(dense, dense),
                    CheckedMultiply(one, one)),
            0);
  EXPECT_EQ(CompareEventRatios(dense, dense, one, one), 0);
  EXPECT_LT(CompareEventRatios(dense, one, one, dense), 0);

  ExactUnsigned<1U> full{{std::numeric_limits<std::uint32_t>::max()}, 1U,
                         0U};
  ExactUnsigned<1U> low_one{{1U}, 1U, 0U};
  EXPECT_THROW(CheckedAdd(full, low_one), std::overflow_error);
  ExactUnsigned<1U> too_narrow{};
  EXPECT_THROW(AddShifted64(too_narrow, 1U, 32U), std::overflow_error);

  ExactUnsigned<traversal_internal::kExactCoordinateWords> carry_left{};
  ExactUnsigned<traversal_internal::kExactCoordinateWords> carry_right{};
  carry_left.words[0U] = UINT32_C(0xffffffff);
  carry_left.words[1U] = UINT32_C(0xfffffffe);
  carry_left.words[2U] = UINT32_C(0x80000001);
  carry_left.words[3U] = UINT32_C(0x12345678);
  carry_left.used_words = 4U;
  carry_left.first_word = 0U;
  carry_right.words[0U] = UINT32_C(0xffffffff);
  carry_right.words[1U] = UINT32_C(0x7fffffff);
  carry_right.words[2U] = UINT32_C(0xdeadbeef);
  carry_right.words[3U] = UINT32_C(0xffffffff);
  carry_right.used_words = 4U;
  carry_right.first_word = 0U;
  const auto carry_product = CheckedMultiply(carry_left, carry_right);
  const std::array<std::uint32_t, 8U> independent_golden{{
      UINT32_C(0x00000001), UINT32_C(0x80000001), UINT32_C(0x2152410e),
      UINT32_C(0x0f1dea98), UINT32_C(0x7d5b7dde), UINT32_C(0x4e92d4ba),
      UINT32_C(0x7da16778), UINT32_C(0x12345678),
  }};
  EXPECT_TRUE(std::equal(independent_golden.begin(), independent_golden.end(),
                         carry_product.words.begin()));
  EXPECT_TRUE(std::all_of(carry_product.words.begin() +
                              independent_golden.size(),
                          carry_product.words.end(),
                          [](std::uint32_t word) { return word == 0U; }));
}

TEST(VisibilityTraversal, IndependentRationalOracleCoversClosedTopology) {
  constexpr std::int64_t scale = kReferenceScale;
  const auto scaled = [](std::int64_t whole, std::int64_t numerator,
                         std::int64_t denominator) {
    return whole * scale + numerator * (scale / denominator);
  };
  struct Segment {
    std::array<std::int64_t, 2> start;
    std::array<std::int64_t, 2> end;
  };
  const std::vector<Segment> segments{
      Segment{{scaled(0, 1, 4), scaled(0, 1, 4)},
              {scaled(2, 1, 2), scaled(5, 1, 2)}},
      Segment{{scaled(2, 1, 2), scaled(5, 1, 2)},
              {scaled(0, 1, 4), scaled(0, 1, 4)}},
      Segment{{-scaled(0, 1, 4), scaled(0, 1, 4)},
              {-scaled(2, 1, 2), scaled(5, 1, 2)}},
      Segment{{scaled(0, 1, 4), -scaled(0, 1, 4)},
              {scaled(2, 1, 2), -scaled(5, 1, 2)}},
      Segment{{-scaled(0, 1, 4), -scaled(0, 1, 4)},
              {-scaled(2, 1, 2), -scaled(5, 1, 2)}},
      Segment{{scaled(0, 1, 2), scaled(0, 1, 2)},
              {scaled(15, 1, 2), scaled(9, 1, 2)}},
      Segment{{scaled(0, 1, 2), scaled(0, 1, 2)},
              {scaled(15, 1, 2), scaled(9, 1, 2) + 1}},
      Segment{{scaled(0, 1, 2), scaled(0, 1, 2)},
              {scaled(15, 1, 2), scaled(9, 1, 2) - 1}},
      Segment{{scaled(0, 1, 2), scaled(1, 0, 1)},
              {scaled(2, 1, 2), scaled(1, 0, 1)}},
      Segment{{scaled(1, 0, 1), scaled(0, 1, 2)},
              {scaled(1, 0, 1), scaled(2, 1, 2)}},
      Segment{{scaled(1, 0, 1), scaled(1, 0, 1)},
              {scaled(2, 1, 2), scaled(3, 1, 2)}},
      Segment{{scaled(3, 0, 1), scaled(2, 1, 4)},
              {scaled(0, 1, 4), scaled(0, 1, 4)}},
  };
  for (const Segment& segment : segments) {
    EXPECT_EQ(ProductionClosedGroups(segment.start, segment.end, scale),
              ReferenceClosedGroups(segment.start, segment.end, scale));
  }
}

TEST(VisibilityTraversal, ExactProductsAreInitializedOnceThenIncremented) {
  VisibilityWorkBudget budget{256U, 0U};
  TraceInstrumentation instrumentation{};
  const auto summary = TraceClosedSegment(
      {0.25, 0.25}, {15.5, 9.5}, budget,
      [](long double, std::span<const TraceCell>) {
        return TraceControl::kContinue;
      },
      &instrumentation);
  EXPECT_GT(summary.visited_cell_count, 20U);
  EXPECT_EQ(instrumentation.full_product_multiplications, 2U);
  EXPECT_GT(instrumentation.incremental_product_additions, 20U);
}

TEST(VisibilityTraversal, ShortBudgetFailsBeforeVisitedMutation) {
  VisibilityWorkBudget budget{3U, 0U};
  TraceInstrumentation instrumentation{};
  std::size_t visitor_calls = 0U;
  EXPECT_THROW(TraceClosedSegment(
                   {1.0, 1.0}, {1.0, 1.0}, budget,
                   [&visitor_calls](long double,
                                    std::span<const TraceCell>) {
                     ++visitor_calls;
                     return TraceControl::kContinue;
                   },
                   &instrumentation),
               std::length_error);
  EXPECT_EQ(budget.used, 0U);
  EXPECT_EQ(visitor_calls, 0U);
  EXPECT_EQ(instrumentation.visited_insertions, 0U);
}

TEST(VisibilityTraversal, IntegerGridLinesEnumerateBothClosedSides) {
  const auto horizontal = Trace({0.5, 1.0}, {2.5, 1.0}, 6U);
  ASSERT_EQ(horizontal.size(), 3U);
  EXPECT_EQ(horizontal[0].cells,
            (std::vector<TraceCell>{{0, 0}, {0, 1}}));
  EXPECT_EQ(horizontal[1].cells,
            (std::vector<TraceCell>{{1, 0}, {1, 1}}));
  EXPECT_EQ(horizontal[2].cells,
            (std::vector<TraceCell>{{2, 0}, {2, 1}}));

  const auto vertical = Trace({1.0, 0.5}, {1.0, 2.5}, 6U);
  ASSERT_EQ(vertical.size(), 3U);
  EXPECT_EQ(vertical[0].cells,
            (std::vector<TraceCell>{{0, 0}, {1, 0}}));
  EXPECT_EQ(vertical[1].cells,
            (std::vector<TraceCell>{{0, 1}, {1, 1}}));
  EXPECT_EQ(vertical[2].cells,
            (std::vector<TraceCell>{{0, 2}, {1, 2}}));
}

TEST(VisibilityTraversal, StopConsumesCompleteTieAndNothingAfter) {
  VisibilityWorkBudget budget{4U, 0U};
  std::vector<std::vector<TraceCell>> groups;
  const auto summary = TraceClosedSegment(
      {0.5, 0.5}, {2.5, 2.5}, budget,
      [&groups](long double, std::span<const TraceCell> cells) {
        groups.emplace_back(cells.begin(), cells.end());
        return groups.size() == 2U ? TraceControl::kStop
                                   : TraceControl::kContinue;
      });
  ASSERT_EQ(groups.size(), 2U);
  EXPECT_EQ(groups[1].size(), 3U);
  EXPECT_EQ(budget.used, 4U);
  EXPECT_EQ(summary.visited_cell_count, 4U);

  VisibilityWorkBudget short_budget{3U, 0U};
  std::size_t short_groups = 0U;
  EXPECT_THROW(
      TraceClosedSegment({0.5, 0.5}, {2.5, 2.5}, short_budget,
                         [&short_groups](long double,
                                         std::span<const TraceCell>) {
                           ++short_groups;
                           return short_groups == 2U ? TraceControl::kStop
                                                     : TraceControl::kContinue;
                         }),
      std::length_error);
}

TEST(VisibilityTraversal, BlockerGroupChargesAllCellsBeforeStopping) {
  const auto run = [](std::size_t limit) {
    VisibilityWorkBudget budget{limit, 0U};
    std::size_t group_number = 0U;
    return TraceClosedSegment(
        {0.5, 0.5}, {3.5, 3.5}, budget,
        [&group_number](long double, std::span<const TraceCell> group) {
          ++group_number;
          if (group_number == 2U) {
            EXPECT_EQ(group.size(), 3U);
            return TraceControl::kStop;
          }
          return TraceControl::kContinue;
        });
  };
  EXPECT_EQ(run(4U).visited_cell_count, 4U);
  EXPECT_THROW(run(3U), std::length_error);
}

TEST(VisibilityTraversal, SameTieAppliesOutsideTaskBeforeOccupied) {
  const GridIndex target{2, 2};
  const std::array<std::pair<TraceCell, CellState>, 3> contacts{{
      {{0, 1}, CellState::kOccupied},
      {{1, 0}, CellState::kOutsideTask},
      {{1, 1}, CellState::kFree},
  }};
  bool outside = false;
  bool occupied = false;
  for (const auto& [cell, state] : contacts) {
    const GridIndex visited{static_cast<std::int32_t>(cell.x),
                            static_cast<std::int32_t>(cell.y)};
    const auto contact = ClassifyPreTargetContact(state, visited, target);
    outside = outside || contact == PreTargetContact::kOutsideTask;
    occupied = occupied || contact == PreTargetContact::kOccupied;
  }
  ASSERT_TRUE(outside);
  ASSERT_TRUE(occupied);
  const PreTargetContact group_result =
      outside ? PreTargetContact::kOutsideTask
              : (occupied ? PreTargetContact::kOccupied
                          : PreTargetContact::kPass);
  EXPECT_EQ(group_result, PreTargetContact::kOutsideTask);
}

TEST(VisibilityTraversal, RejectsNonFiniteAndRepresentsVirtualInt32Sides) {
  VisibilityWorkBudget budget{10U, 0U};
  EXPECT_THROW(
      TraceClosedSegment({std::numeric_limits<double>::infinity(), 0.0},
                         {0.0, 0.0}, budget,
                         [](long double, std::span<const TraceCell>) {
                           return TraceControl::kContinue;
                         }),
      std::invalid_argument);
  std::vector<TraceCell> cells;
  EXPECT_NO_THROW(TraceClosedSegment(
      {static_cast<double>(std::numeric_limits<std::int32_t>::max()) + 1.0,
       0.5},
      {static_cast<double>(std::numeric_limits<std::int32_t>::max()) + 2.0,
       0.5},
      budget, [&cells](long double, std::span<const TraceCell> group) {
        cells.insert(cells.end(), group.begin(), group.end());
        return TraceControl::kContinue;
      }));
  ASSERT_FALSE(cells.empty());
  EXPECT_GT(cells.back().x,
            static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()));
}

TEST(VisibilityTraversal, ContactAndCountHelpersPreserveTargetSemantics) {
  const GridIndex target{4, 5};
  for (const CellState state :
       {CellState::kOutsideTask, CellState::kOutsideMap,
        CellState::kUnknown, CellState::kFree, CellState::kOccupied}) {
    EXPECT_EQ(ClassifyPreTargetContact(state, target, target),
              PreTargetContact::kPass);
  }
  EXPECT_EQ(ClassifyPreTargetContact(CellState::kOutsideTask, {3, 5}, target),
            PreTargetContact::kOutsideTask);
  EXPECT_EQ(ClassifyPreTargetContact(CellState::kOccupied, {3, 5}, target),
            PreTargetContact::kOccupied);
  EXPECT_EQ(ClassifyPreTargetContact(CellState::kUnknown, {3, 5}, target),
            PreTargetContact::kPass);

  std::uint32_t count = std::numeric_limits<std::uint32_t>::max() - 1U;
  EXPECT_NO_THROW(CheckedVisibleIncrement(count));
  EXPECT_EQ(count, std::numeric_limits<std::uint32_t>::max());
  EXPECT_THROW(CheckedVisibleIncrement(count), std::overflow_error);
}

}  // namespace
}  // namespace lunar::pure_exploration::detail
