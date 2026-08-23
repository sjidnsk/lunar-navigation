#include "lunar_pure_exploration_core/candidate_ranker.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lunar::pure_exploration {
namespace {

FrontierCluster Frontier(std::vector<std::int64_t> canonical_key,
                         std::uint64_t display_id = 17U) {
  return FrontierCluster{display_id, {}, {}, std::move(canonical_key),
                         {0.0, 0.0}, 1.0};
}

CandidateView Candidate(std::size_t frontier_index, CandidateKey key,
                        Pose2 pose, std::uint64_t candidate_display_id = 31U,
                        std::uint64_t frontier_display_id = 17U) {
  return CandidateView{candidate_display_id, frontier_display_id,
                       frontier_index, key, pose, 0.0};
}

std::vector<std::size_t> Indices(
    const std::vector<RankedCandidate>& ranked) {
  std::vector<std::size_t> result;
  for (const RankedCandidate& row : ranked) {
    result.push_back(row.candidate_index);
  }
  return result;
}

std::vector<RankedCandidate> CoarseFromList(
    const CandidateRanker& ranker,
    std::span<const CandidateView> candidates,
    std::span<const FrontierCluster> frontiers,
    std::initializer_list<CandidateGain> gains,
    Pose2 robot_pose = {}) {
  return ranker.CoarseRank(candidates, frontiers,
                           std::span<const CandidateGain>(gains.begin(),
                                                          gains.size()),
                           robot_pose);
}

std::vector<RankedCandidate> FinalFromLists(
    const CandidateRanker& ranker,
    std::span<const CandidateView> candidates,
    std::span<const FrontierCluster> frontiers,
    std::span<const CandidateGain> gains,
    std::initializer_list<PlannedCandidate> planned,
    Pose2 robot_pose,
    std::initializer_list<Vec2> goals,
    double resolution_m) {
  return ranker.FinalRank(
      candidates, frontiers, gains,
      std::span<const PlannedCandidate>(planned.begin(), planned.size()),
      robot_pose, std::span<const Vec2>(goals.begin(), goals.size()),
      resolution_m);
}

TEST(CandidateRanker, RejectsInvalidConstructorDomains) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  EXPECT_THROW(CandidateRanker({}, 0.0), std::invalid_argument);
  EXPECT_THROW(CandidateRanker({}, -1.0), std::invalid_argument);
  EXPECT_THROW(CandidateRanker({}, nan), std::invalid_argument);
  EXPECT_THROW(CandidateRanker({}, infinity), std::invalid_argument);
  EXPECT_THROW(CandidateRanker({-1.0, 0.0, 0.0, 0.0}, 1.0),
               std::invalid_argument);
  EXPECT_THROW(CandidateRanker({nan, 0.0, 0.0, 0.0}, 1.0),
               std::invalid_argument);
  EXPECT_THROW(CandidateRanker({infinity, 0.0, 0.0, 0.0}, 1.0),
               std::invalid_argument);
  EXPECT_THROW(CandidateRanker({0.0, 0.0, 0.0, 0.0}, 1.0),
               std::invalid_argument);
  EXPECT_NO_THROW(CandidateRanker({2.0, 3.0, 0.0, 0.0}, 1.0));
}

TEST(CandidateRanker, CoarseUsesExactFormulaAndFiltersOnlyExactZeroGain) {
  const std::vector<FrontierCluster> frontiers{Frontier({1, 2, 0})};
  const std::vector<CandidateView> candidates{
      Candidate(0U, {1000, 0, 0}, {1.0, 0.0, 0.0}),
      Candidate(0U, {2000, 0, 0}, {2.0, 0.0, 0.0}),
      Candidate(0U, {3000, 0, 0}, {3.0, 0.0, 0.0}),
  };
  const double smallest_positive = std::numeric_limits<double>::denorm_min();
  const std::vector<CandidateGain> gains{{2U, smallest_positive},
                                         {0U, 2.0},
                                         {1U, 0.0}};
  const auto ranked = CandidateRanker({}, 1.0).CoarseRank(
      candidates, frontiers, gains, {0.0, 0.0, 0.0});
  ASSERT_EQ(ranked.size(), 2U);
  EXPECT_EQ(ranked[0].candidate_index, 0U);
  EXPECT_DOUBLE_EQ(ranked[0].rank_value, 1.0);
  EXPECT_EQ(ranked[1].candidate_index, 2U);
  EXPECT_EQ(ranked[1].information_gain_m2, smallest_positive);
  EXPECT_EQ(ranked[1].rank_value, 0.0);
  for (const RankedCandidate& row : ranked) {
    EXPECT_TRUE(std::isfinite(row.information_gain_m2));
    EXPECT_TRUE(std::isfinite(row.euclidean_distance_m));
    EXPECT_TRUE(std::isfinite(row.path_length_m));
    EXPECT_TRUE(std::isfinite(row.heading_change_rad));
    EXPECT_TRUE(std::isfinite(row.revisit_penalty));
    EXPECT_TRUE(std::isfinite(row.rank_value));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(row.path_length_m),
              std::bit_cast<std::uint64_t>(0.0));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(row.revisit_penalty),
              std::bit_cast<std::uint64_t>(0.0));
  }
}

TEST(CandidateRanker, CoarseReturnsAllThirtyThreeOwningIndices) {
  const std::vector<FrontierCluster> frontiers{Frontier({1, 2, 0})};
  std::vector<CandidateView> candidates;
  std::vector<CandidateGain> gains;
  for (std::size_t index = 0U; index < 33U; ++index) {
    candidates.push_back(Candidate(
        0U, {static_cast<std::int64_t>(index), 0, 0},
        {static_cast<double>(index + 1U), 0.0, 0.0}));
    gains.push_back({32U - index, static_cast<double>(33U - index)});
  }
  const auto ranked = CandidateRanker({}, 1.0).CoarseRank(
      candidates, frontiers, gains, {0.0, 0.0, 0.0});
  ASSERT_EQ(ranked.size(), 33U);
  std::set<std::size_t> unique_indices;
  std::vector<std::size_t> batch_sizes;
  for (std::size_t cursor = 0U; cursor < ranked.size();) {
    const std::size_t end = std::min(cursor + 16U, ranked.size());
    batch_sizes.push_back(end - cursor);
    for (; cursor < end; ++cursor) {
      unique_indices.insert(ranked[cursor].candidate_index);
    }
  }
  EXPECT_EQ(batch_sizes, (std::vector<std::size_t>{16U, 16U, 1U}));
  EXPECT_EQ(unique_indices.size(), 33U);
  EXPECT_EQ(*unique_indices.begin(), 0U);
  EXPECT_EQ(*unique_indices.rbegin(), 32U);
}

TEST(CandidateRanker, CoarseWrapsHeadingAndCanonicalizesNegativeZero) {
  const double pi = std::numbers::pi;
  const double robot_yaw = std::nextafter(pi, 0.0);
  const std::vector<FrontierCluster> frontiers{
      Frontier({1, 0, 0}, 99U), Frontier({2, 0, 0}, 99U),
      Frontier({3, 0, 0}, 99U)};
  const std::vector<CandidateView> candidates{
      Candidate(0U, {1, 0, -1800}, {1.0, 0.0, std::nextafter(-pi, 0.0)},
                88U, 99U),
      Candidate(1U, {2, 0, 0}, {1.0, 0.0, 0.0}, 88U, 99U),
      Candidate(2U, {3, 0, 0}, {1.0, 0.0, -0.0}, 88U, 99U),
  };
  const std::vector<CandidateGain> gains{{2U, 1.0}, {0U, 1.0}, {1U, 1.0}};
  const auto ranked = CandidateRanker({}, 1.0).CoarseRank(
      candidates, frontiers, gains, {0.0, 0.0, robot_yaw});
  ASSERT_EQ(ranked.size(), 3U);
  EXPECT_EQ(ranked[0].candidate_index, 0U);
  EXPECT_LT(ranked[0].heading_change_rad, 1e-14);

  const auto zero_heading = CandidateRanker({}, 1.0).CoarseRank(
      std::span<const CandidateView>(candidates).subspan(2U, 1U),
      frontiers, std::vector<CandidateGain>{{0U, 1.0}},
      {1.0, 0.0, 0.0});
  ASSERT_EQ(zero_heading.size(), 1U);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(zero_heading[0].heading_change_rad),
            std::bit_cast<std::uint64_t>(0.0));
}

TEST(CandidateRanker, CoarseTreatsPositiveAndNegativePiAsEquivalent) {
  const std::vector<FrontierCluster> frontiers{
      Frontier({2, 0, 0}, 9U), Frontier({1, 0, 0}, 9U)};
  const std::vector<CandidateView> candidates{
      Candidate(0U, {1, 0, -1800}, {1.0, 0.0, std::numbers::pi}, 8U, 9U),
      Candidate(1U, {1, 0, -1800}, {1.0, 0.0, -std::numbers::pi}, 8U, 9U)};
  const std::vector<CandidateGain> gains{{0U, 1.0}, {1U, 1.0}};
  const auto ranked = CandidateRanker({}, 1.0).CoarseRank(
      candidates, frontiers, gains, {});
  ASSERT_EQ(ranked.size(), 2U);
  EXPECT_DOUBLE_EQ(ranked[0].heading_change_rad, std::numbers::pi);
  EXPECT_DOUBLE_EQ(ranked[1].heading_change_rad, std::numbers::pi);
  EXPECT_EQ(Indices(ranked), (std::vector<std::size_t>{1U, 0U}));
}

TEST(CandidateRanker, CoarseUsesVisibleTieChainThenFullIdentity) {
  std::vector<FrontierCluster> frontiers{
      Frontier({2, 0, 0}, 7U), Frontier({1, 0, 0}, 7U),
      Frontier({1, 0, 0}, 7U)};
  std::vector<CandidateView> candidates{
      Candidate(0U, {5, 0, 0}, {1.0, 0.0, 0.0}, 5U, 7U),
      Candidate(1U, {9, 0, 0}, {1.0, 0.0, 0.0}, 5U, 7U),
      Candidate(2U, {8, 0, 0}, {1.0, 0.0, 0.0}, 5U, 7U),
      Candidate(0U, {6, 0, 0}, {2.0, 0.0, 0.0}, 5U, 7U),
  };
  std::vector<CandidateGain> gains{{3U, 3.0}, {0U, 2.0}, {2U, 2.0},
                                   {1U, 2.0}};
  const auto first = CandidateRanker({}, 1.0).CoarseRank(
      candidates, frontiers, gains, {0.0, 0.0, 0.0});
  std::reverse(gains.begin(), gains.end());
  candidates[0].id = 999U;
  candidates[1].id = 1U;
  const auto second = CandidateRanker({}, 1.0).CoarseRank(
      candidates, frontiers, gains, {0.0, 0.0, 0.0});
  EXPECT_EQ(Indices(first), (std::vector<std::size_t>{3U, 2U, 1U, 0U}));
  EXPECT_EQ(Indices(second), Indices(first));
}

TEST(CandidateRanker, CoarseFreezesEveryVisibleTieBreakLevel) {
  const std::vector<FrontierCluster> frontiers{Frontier({1, 0, 0})};
  const CandidateRanker ranker({}, 1.0);
  const auto rank_pair = [&](std::vector<CandidateView> candidates,
                             std::vector<CandidateGain> gains) {
    return Indices(ranker.CoarseRank(candidates, frontiers, gains, {}));
  };

  EXPECT_EQ(rank_pair({Candidate(0U, {1, 0, 0}, {1.0, 0.0, 0.0}),
                       Candidate(0U, {2, 0, 0}, {0.0, 0.0, 0.0})},
                      {{0U, 2.0}, {1U, 1.0}}),
            (std::vector<std::size_t>{0U, 1U}));

  const double tiny = std::numeric_limits<double>::denorm_min();
  EXPECT_EQ(rank_pair({Candidate(0U, {1, 0, 0}, {2.0, 0.0, 0.0}),
                       Candidate(0U, {2, 0, 0}, {1.0, 0.0, 0.0})},
                      {{0U, tiny}, {1U, tiny}}),
            (std::vector<std::size_t>{1U, 0U}));

  EXPECT_EQ(rank_pair({Candidate(0U, {1, 0, 10}, {1.0, 0.0, 1.0}),
                       Candidate(0U, {2, 0, 0}, {1.0, 0.0, 0.0})},
                      {{0U, 1.0}, {1U, 1.0}}),
            (std::vector<std::size_t>{1U, 0U}));
  EXPECT_EQ(rank_pair({Candidate(0U, {1, 0, 0}, {1.0, 0.0, 0.0}),
                       Candidate(0U, {2, 0, 0}, {-1.0, 0.0, 0.0})},
                      {{0U, 1.0}, {1U, 1.0}}),
            (std::vector<std::size_t>{1U, 0U}));
  EXPECT_EQ(rank_pair({Candidate(0U, {1, 0, 0}, {0.0, 1.0, 0.0}),
                       Candidate(0U, {2, 0, 0}, {0.0, -1.0, 0.0})},
                      {{0U, 1.0}, {1U, 1.0}}),
            (std::vector<std::size_t>{1U, 0U}));
  EXPECT_EQ(rank_pair({Candidate(0U, {1, 0, 5}, {1.0, 0.0, 0.5}),
                       Candidate(0U, {2, 0, -5}, {1.0, 0.0, -0.5})},
                      {{0U, 1.0}, {1U, 1.0}}),
            (std::vector<std::size_t>{1U, 0U}));
}

TEST(CandidateRanker, RejectsIncompleteOrInvalidCoarseAuthority) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<FrontierCluster> frontiers{Frontier({1, 0, 0})};
  std::vector<CandidateView> candidates{
      Candidate(0U, {1, 0, 0}, {0.0, 0.0, 0.0}),
      Candidate(0U, {2, 0, 0}, {1.0, 0.0, 0.0})};
  const CandidateRanker ranker({}, 1.0);
  EXPECT_THROW(CoarseFromList(ranker, candidates, frontiers, {{0U, 1.0}}),
               std::invalid_argument);
  EXPECT_THROW(CoarseFromList(ranker, candidates, frontiers,
                              {{0U, 1.0}, {0U, 2.0}}),
               std::invalid_argument);
  EXPECT_THROW(CoarseFromList(ranker, candidates, frontiers,
                              {{0U, 1.0}, {2U, 2.0}}),
               std::invalid_argument);
  for (double invalid : {-1.0, nan, infinity}) {
    EXPECT_THROW(CoarseFromList(ranker, candidates, frontiers,
                                {{0U, invalid}, {1U, 1.0}}),
                 std::invalid_argument);
  }
  candidates[1].frontier_index = 1U;
  EXPECT_THROW(CoarseFromList(ranker, candidates, frontiers,
                              {{0U, 1.0}, {1U, 1.0}}),
               std::invalid_argument);
  candidates[1].frontier_index = 0U;
  candidates[1].frontier_id = 1234U;
  EXPECT_NO_THROW(CoarseFromList(ranker, candidates, frontiers,
                                 {{0U, 1.0}, {1U, 1.0}}));
  candidates[1] = candidates[0];
  EXPECT_THROW(CoarseFromList(ranker, candidates, frontiers,
                              {{0U, 1.0}, {1U, 1.0}}),
               std::invalid_argument);
  candidates[1] = Candidate(0U, {2, 0, 0}, {nan, 0.0, 0.0});
  EXPECT_THROW(CoarseFromList(ranker, candidates, frontiers,
                              {{0U, 1.0}, {1U, 1.0}}),
               std::invalid_argument);
  candidates[1] = Candidate(0U, {2, 0, 0}, {1.0, 0.0, 0.0});
  EXPECT_THROW(CoarseFromList(ranker, candidates, frontiers,
                              {{0U, 1.0}, {1U, 1.0}},
                              {0.0, infinity, 0.0}),
               std::invalid_argument);
}

TEST(CandidateRanker, RejectsNonfiniteCoarseDerivations) {
  const double maximum = std::numeric_limits<double>::max();
  const std::vector<FrontierCluster> frontiers{Frontier({1, 0, 0})};
  const std::vector<CandidateView> candidates{
      Candidate(0U, {1, 0, 0}, {maximum, 0.0, 0.0})};
  EXPECT_THROW(CoarseFromList(CandidateRanker({}, 1.0), candidates, frontiers,
                              {{0U, 1.0}}, {-maximum, 0.0, 0.0}),
               std::overflow_error);
  const std::vector<CandidateView> colocated{
      Candidate(0U, {1, 0, 0}, {0.0, 0.0, 0.0})};
  EXPECT_THROW(CoarseFromList(
                   CandidateRanker({},
                                   std::numeric_limits<double>::denorm_min()),
                   colocated, frontiers, {{0U, maximum}}),
               std::overflow_error);
}

TEST(CandidateRanker, FinalUsesGainAuthorityAndDefaultNormalization) {
  const std::vector<FrontierCluster> frontiers{Frontier({1, 0, 0})};
  const std::vector<CandidateView> candidates{
      Candidate(0U, {1, 0, 0}, {0.0, 0.0, 0.0}),
      Candidate(0U, {2, 0, -1800}, {100.0, 0.0, std::numbers::pi})};
  const std::vector<CandidateGain> gains{{1U, 5.0}, {0U, 10.0}};
  const std::vector<PlannedCandidate> planned{{1U, 10.0}, {0U, 5.0}};
  const std::vector<Vec2> completed_goals{{100.0, 0.0}};
  const auto ranked = CandidateRanker({}, 1.0).FinalRank(
      candidates, frontiers, gains, planned, {0.0, 0.0, 0.0},
      completed_goals, 0.2);
  ASSERT_EQ(ranked.size(), 2U);
  ASSERT_EQ(ranked[0].candidate_index, 0U);
  const double expected_first = static_cast<double>(
      static_cast<long double>(0.60) -
      static_cast<long double>(0.30) * 0.5L);
  const double expected_second = static_cast<double>(
      static_cast<long double>(0.60) * 0.5L -
      static_cast<long double>(0.30) - static_cast<long double>(0.05) -
      static_cast<long double>(0.05));
  EXPECT_DOUBLE_EQ(ranked[0].rank_value, expected_first);
  EXPECT_DOUBLE_EQ(ranked[1].rank_value, expected_second);
  EXPECT_DOUBLE_EQ(ranked[0].information_gain_m2, 10.0);
  EXPECT_DOUBLE_EQ(ranked[0].path_length_m, 5.0);
  EXPECT_DOUBLE_EQ(ranked[1].heading_change_rad, std::numbers::pi);
  EXPECT_DOUBLE_EQ(ranked[1].revisit_penalty, 1.0);
}

TEST(CandidateRanker, FinalHandlesEmptyAndZeroNormalizationDenominators) {
  const std::vector<FrontierCluster> frontiers{Frontier({1, 0, 0})};
  const std::vector<CandidateView> candidates{
      Candidate(0U, {1, 0, 0}, {0.0, 0.0, 0.0})};
  const std::vector<CandidateGain> gains{{0U, 0.0}};
  const CandidateRanker ranker({}, 1.0);
  EXPECT_TRUE(ranker.FinalRank(candidates, frontiers, gains, {}, {}, {}, 0.2)
                  .empty());
  EXPECT_TRUE(CandidateRanker({}, std::numeric_limits<double>::max())
                  .FinalRank(candidates, frontiers, gains, {}, {}, {}, 0.2)
                  .empty());
  const auto ranked = FinalFromLists(ranker, candidates, frontiers, gains,
                                     {{0U, 0.0}}, {}, {}, 0.2);
  ASSERT_EQ(ranked.size(), 1U);
  EXPECT_EQ(ranked[0].rank_value, 0.0);
  EXPECT_TRUE(std::isfinite(ranked[0].rank_value));
}

TEST(CandidateRanker, FinalDerivesClosedRevisitRadiusInternally) {
  const std::vector<FrontierCluster> frontiers{Frontier({1, 0, 0})};
  const std::vector<CandidateView> candidates{
      Candidate(0U, {1, 0, 0}, {0.0, 0.0, 0.0})};
  const std::vector<CandidateGain> gains{{0U, 1.0}};
  const std::vector<PlannedCandidate> planned{{0U, 1.0}};
  const std::vector<Vec2> tangent_goal{{2.0, 0.0}};
  const std::vector<Vec2> outside_goal{{
      std::nextafter(2.0, std::numeric_limits<double>::infinity()), 0.0}};
  const CandidateRanker ranker({}, 1.0);
  const auto tangent = ranker.FinalRank(candidates, frontiers, gains, planned,
                                        {}, tangent_goal, 0.2);
  const auto outside = ranker.FinalRank(candidates, frontiers, gains, planned,
                                        {}, outside_goal, 0.2);
  ASSERT_EQ(tangent.size(), 1U);
  ASSERT_EQ(outside.size(), 1U);
  EXPECT_DOUBLE_EQ(tangent[0].revisit_penalty, 1.0);
  EXPECT_DOUBLE_EQ(outside[0].revisit_penalty, 0.0);
}

TEST(CandidateRanker, FinalUsesResolutionDominatedClosedRevisitRadius) {
  const std::vector<FrontierCluster> frontiers{Frontier({1, 0, 0})};
  const std::vector<CandidateView> candidates{
      Candidate(0U, {1, 0, 0}, {0.0, 0.0, 0.0})};
  const std::vector<CandidateGain> gains{{0U, 1.0}};
  const std::vector<PlannedCandidate> planned{{0U, 1.0}};
  const std::vector<Vec2> tangent_goal{{2.0, 0.0}};
  const std::vector<Vec2> outside_goal{{
      std::nextafter(2.0, std::numeric_limits<double>::infinity()), 0.0}};
  const CandidateRanker ranker({}, 0.2);
  const auto tangent = ranker.FinalRank(candidates, frontiers, gains, planned,
                                        {}, tangent_goal, 1.0);
  const auto outside = ranker.FinalRank(candidates, frontiers, gains, planned,
                                        {}, outside_goal, 1.0);
  ASSERT_EQ(tangent.size(), 1U);
  ASSERT_EQ(outside.size(), 1U);
  EXPECT_DOUBLE_EQ(tangent[0].revisit_penalty, 1.0);
  EXPECT_DOUBLE_EQ(outside[0].revisit_penalty, 0.0);
}

TEST(CandidateRanker, FinalAcceptsNonunitWeightSumWithoutRenormalizing) {
  const std::vector<FrontierCluster> frontiers{Frontier({1, 0, 0})};
  const std::vector<CandidateView> candidates{
      Candidate(0U, {1, 0, 0}, {0.0, 0.0, 0.0})};
  const std::vector<CandidateGain> gains{{0U, 1.0}};
  const auto ranked = FinalFromLists(
      CandidateRanker({2.0, 3.0, 0.0, 0.0}, 1.0), candidates, frontiers,
      gains, {{0U, 1.0}}, {}, {}, 0.2);
  ASSERT_EQ(ranked.size(), 1U);
  EXPECT_DOUBLE_EQ(ranked[0].rank_value, -1.0);
}

TEST(CandidateRanker, FinalRejectsInvalidSubsetAndScalarDomains) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  const std::vector<FrontierCluster> frontiers{Frontier({1, 0, 0})};
  const std::vector<CandidateView> candidates{
      Candidate(0U, {1, 0, 0}, {0.0, 0.0, 0.0}),
      Candidate(0U, {2, 0, 0}, {1.0, 0.0, 0.0})};
  const std::vector<CandidateGain> gains{{0U, 1.0}, {1U, 2.0}};
  const CandidateRanker ranker({}, 1.0);
  EXPECT_THROW(FinalFromLists(ranker, candidates, frontiers, gains,
                              {{0U, 1.0}, {0U, 2.0}}, {}, {}, 0.2),
               std::invalid_argument);
  EXPECT_THROW(FinalFromLists(ranker, candidates, frontiers, gains,
                              {{2U, 1.0}}, {}, {}, 0.2),
               std::invalid_argument);
  for (double invalid : {-1.0, nan, infinity}) {
    EXPECT_THROW(FinalFromLists(ranker, candidates, frontiers, gains,
                                {{0U, invalid}}, {}, {}, 0.2),
                 std::invalid_argument);
  }
  for (double invalid : {0.0, -1.0, nan, infinity}) {
    EXPECT_THROW(ranker.FinalRank(candidates, frontiers, gains, {}, {}, {},
                                  invalid),
                 std::invalid_argument);
  }
  EXPECT_THROW(FinalFromLists(ranker, candidates, frontiers, gains,
                              {{0U, 1.0}}, {}, {{nan, 0.0}}, 0.2),
               std::invalid_argument);
  EXPECT_THROW(FinalFromLists(ranker, candidates, frontiers, gains,
                              {{0U, 1.0}}, {}, {{0.0, infinity}}, 0.2),
               std::invalid_argument);
}

TEST(CandidateRanker, FinalRejectsScoreOutsideDoubleRange) {
  const double maximum = std::numeric_limits<double>::max();
  const std::vector<FrontierCluster> frontiers{Frontier({1, 0, 0})};
  const std::vector<CandidateView> candidates{
      Candidate(0U, {1, 0, -1800}, {0.0, 0.0, std::numbers::pi})};
  const std::vector<CandidateGain> gains{{0U, 1.0}};
  EXPECT_THROW(FinalFromLists(
                   CandidateRanker({0.0, maximum, maximum, 0.0}, 1.0),
                   candidates, frontiers, gains, {{0U, 1.0}}, {}, {}, 0.2),
               std::overflow_error);
}

TEST(CandidateRanker, FinalRejectsNonzeroComponentUnderflowToDoubleZero) {
  const double smallest_positive = std::numeric_limits<double>::denorm_min();
  const std::vector<FrontierCluster> frontiers{Frontier({1, 0, 0})};
  const std::vector<CandidateView> candidates{
      Candidate(0U, {1, 0, 0}, {0.0, 0.0, 0.0}),
      Candidate(0U, {2, 0, 0}, {1.0, 0.0, 0.0})};
  const std::vector<CandidateGain> gains{{0U, 1.0}, {1U, 1.0}};
  EXPECT_THROW(FinalFromLists(
                   CandidateRanker({0.0, smallest_positive, 0.0, 0.0}, 1.0),
                   candidates, frontiers, gains, {{0U, 1.0}, {1U, 2.0}},
                   {}, {}, 0.2),
               std::overflow_error);
}

TEST(CandidateRanker, FinalTieIsPermutationInvariantAndIgnoresDisplayHashes) {
  std::vector<FrontierCluster> frontiers{Frontier({2, 0, 0}, 44U),
                                         Frontier({1, 0, 0}, 44U)};
  std::vector<CandidateView> candidates{
      Candidate(0U, {2, 0, 0}, {1.0, 1.0, 0.0}, 55U, 44U),
      Candidate(1U, {1, 0, 0}, {1.0, 1.0, 0.0}, 55U, 44U)};
  std::vector<CandidateGain> gains{{0U, 1.0}, {1U, 1.0}};
  std::vector<PlannedCandidate> planned{{0U, 1.0}, {1U, 1.0}};
  const CandidateRanker ranker({}, 1.0);
  const auto first = ranker.FinalRank(candidates, frontiers, gains, planned,
                                      {}, {}, 0.2);
  std::reverse(gains.begin(), gains.end());
  std::reverse(planned.begin(), planned.end());
  candidates[0].id = 0U;
  candidates[1].id = std::numeric_limits<std::uint64_t>::max();
  const auto second = ranker.FinalRank(candidates, frontiers, gains, planned,
                                       {}, {}, 0.2);
  EXPECT_EQ(Indices(first), (std::vector<std::size_t>{1U, 0U}));
  EXPECT_EQ(Indices(second), Indices(first));
}

TEST(CandidateRanker, FinalFreezesEveryVisibleTieBreakLevel) {
  const std::vector<FrontierCluster> frontiers{Frontier({1, 0, 0})};
  const CandidateRanker ranker({0.0, 0.0, 0.0, 1.0}, 1.0);
  const auto rank_pair = [&](std::vector<CandidateView> candidates,
                             std::vector<CandidateGain> gains,
                             std::vector<PlannedCandidate> planned) {
    return Indices(ranker.FinalRank(candidates, frontiers, gains, planned,
                                    {}, {}, 0.2));
  };

  EXPECT_EQ(rank_pair({Candidate(0U, {1, 0, 0}, {}),
                       Candidate(0U, {2, 0, 0}, {})},
                      {{0U, 1.0}, {1U, 2.0}},
                      {{0U, 1.0}, {1U, 1.0}}),
            (std::vector<std::size_t>{1U, 0U}));
  EXPECT_EQ(rank_pair({Candidate(0U, {1, 0, 0}, {}),
                       Candidate(0U, {2, 0, 0}, {})},
                      {{0U, 1.0}, {1U, 1.0}},
                      {{0U, 2.0}, {1U, 1.0}}),
            (std::vector<std::size_t>{1U, 0U}));
  EXPECT_EQ(rank_pair({Candidate(0U, {1, 0, 10}, {0.0, 0.0, 1.0}),
                       Candidate(0U, {2, 0, 0}, {})},
                      {{0U, 1.0}, {1U, 1.0}},
                      {{0U, 1.0}, {1U, 1.0}}),
            (std::vector<std::size_t>{1U, 0U}));
  EXPECT_EQ(rank_pair({Candidate(0U, {1, 0, 0}, {2.0, 0.0, 0.0}),
                       Candidate(0U, {2, 0, 0}, {1.0, 0.0, 0.0})},
                      {{0U, 1.0}, {1U, 1.0}},
                      {{0U, 1.0}, {1U, 1.0}}),
            (std::vector<std::size_t>{1U, 0U}));
  EXPECT_EQ(rank_pair({Candidate(0U, {1, 0, 0}, {0.0, 2.0, 0.0}),
                       Candidate(0U, {2, 0, 0}, {0.0, 1.0, 0.0})},
                      {{0U, 1.0}, {1U, 1.0}},
                      {{0U, 1.0}, {1U, 1.0}}),
            (std::vector<std::size_t>{1U, 0U}));
  EXPECT_EQ(rank_pair({Candidate(0U, {1, 0, 5}, {0.0, 0.0, 0.5}),
                       Candidate(0U, {2, 0, -5}, {0.0, 0.0, -0.5})},
                      {{0U, 1.0}, {1U, 1.0}},
                      {{0U, 1.0}, {1U, 1.0}}),
            (std::vector<std::size_t>{1U, 0U}));
}

std::vector<RankedCandidate> RankTemporaryOwningBatch() {
  const std::vector<FrontierCluster> frontiers{Frontier({1, 0, 0})};
  const std::vector<CandidateView> candidates{
      Candidate(0U, {1, 0, 0}, {1.0, 0.0, 0.0})};
  const std::vector<CandidateGain> gains{{0U, 1.0}};
  return CandidateRanker({}, 1.0).CoarseRank(
      candidates, frontiers, gains, {});
}

TEST(CandidateRanker, ResultOwnsOnlyIndexAndScalarSnapshot) {
  const auto ranked = RankTemporaryOwningBatch();
  ASSERT_EQ(ranked.size(), 1U);
  EXPECT_EQ(ranked[0].candidate_index, 0U);
  EXPECT_DOUBLE_EQ(ranked[0].information_gain_m2, 1.0);
  EXPECT_TRUE(std::isfinite(ranked[0].rank_value));
}

}  // namespace
}  // namespace lunar::pure_exploration
