#include <cstddef>
#include <stop_token>
#include <vector>

#include <gtest/gtest.h>

#include "shared/ara_star.hpp"
#include "shared/bounded_qp_solver.hpp"
#include "shared/candidate_ranker.hpp"

namespace lunar::planning::shared {
namespace {

template <typename Config>
concept HasFixedSearchResources = requires(Config config) {
  config.resources;
};

template <typename Config>
concept HasFixedTerminalCandidateLimit = requires(Config config) {
  config.maximum_terminal_candidates;
};

template <typename Config>
concept HasFixedValidationSubdivisionLimit = requires(Config config) {
  config.continuous_validation_maximum_subdivisions;
};

static_assert(!HasFixedSearchResources<AraStarConfig>);
static_assert(!HasFixedTerminalCandidateLimit<WheelPlannerConfig>);
static_assert(!HasFixedTerminalCandidateLimit<LeggedPlannerConfig>);
static_assert(!HasFixedValidationSubdivisionLimit<WheelPlannerConfig>);
static_assert(!HasFixedValidationSubdivisionLimit<LeggedPlannerConfig>);

AraStarProblem MakeDiamondProblem() {
  AraStarProblem problem;
  problem.state_count = 4U;
  problem.start_state = 0U;
  problem.goal_mask = {0U, 0U, 0U, 1U};
  problem.heuristic = {2.0, 1.0, 1.0, 0.0};
  problem.outgoing_edges.resize(4U);
  problem.outgoing_edges[0] = {
      GraphEdge{.target_state = 2U, .cost = 1.0, .stable_index = 20U},
      GraphEdge{.target_state = 1U, .cost = 1.0, .stable_index = 10U},
  };
  problem.outgoing_edges[1] = {
      GraphEdge{.target_state = 3U, .cost = 1.0, .stable_index = 11U},
  };
  problem.outgoing_edges[2] = {
      GraphEdge{.target_state = 3U, .cost = 1.0, .stable_index = 21U},
  };
  problem.config.initial_epsilon = 2.0;
  problem.config.epsilon_decrement = 0.5;
  problem.config.target_epsilon = 1.0;
  return problem;
}

TEST(CandidateRanker, UsesCostThenStableIndex) {
  const std::vector<CandidateScore> candidates{
      {.cost = 3.0, .stable_index = 9U, .fully_hard_validated = true},
      {.cost = 2.0, .stable_index = 8U, .fully_hard_validated = true},
      {.cost = 2.0, .stable_index = 3U, .fully_hard_validated = true},
      {.cost = 1.0, .stable_index = 1U, .fully_hard_validated = false},
  };

  const auto ranked = RankCandidates(candidates);

  ASSERT_EQ(ranked.size(), 3U);
  EXPECT_EQ(ranked[0].stable_index, 3U);
  EXPECT_EQ(ranked[1].stable_index, 8U);
  EXPECT_EQ(ranked[2].stable_index, 9U);
}

TEST(AraStar, IsDeterministicAcrossRepeatedRuns) {
  const auto first = SearchAraStar(MakeDiamondProblem(), {});
  const auto second = SearchAraStar(MakeDiamondProblem(), {});

  ASSERT_EQ(first.status, AraStarStatus::kSolved) << first.reason_code;
  ASSERT_EQ(second.status, AraStarStatus::kSolved) << second.reason_code;
  ASSERT_FALSE(first.candidates.empty());
  ASSERT_EQ(first.candidates.size(), second.candidates.size());
  EXPECT_EQ(first.candidates, second.candidates);
  EXPECT_EQ(first.candidates.front().states, (std::vector<std::size_t>{0U, 1U, 3U}));
  EXPECT_DOUBLE_EQ(first.candidates.front().cost, 2.0);
}

TEST(AraStar, RefinesEpsilonWithoutRestartingExpandedStates) {
  AraStarProblem problem;
  problem.state_count = 4U;
  problem.start_state = 0U;
  problem.goal_mask = {0U, 0U, 0U, 1U};
  problem.heuristic = {0.0, 8.0, 0.0, 0.0};
  problem.outgoing_edges.resize(4U);
  problem.outgoing_edges[0] = {
      GraphEdge{.target_state = 1U, .cost = 2.0, .stable_index = 10U},
      GraphEdge{.target_state = 2U, .cost = 1.0, .stable_index = 20U},
  };
  problem.outgoing_edges[1] = {
      GraphEdge{.target_state = 3U, .cost = 8.0, .stable_index = 11U},
  };
  problem.outgoing_edges[2] = {
      GraphEdge{.target_state = 3U, .cost = 10.0, .stable_index = 21U},
  };
  problem.config.initial_epsilon = 2.0;
  problem.config.epsilon_decrement = 0.5;
  problem.config.target_epsilon = 1.0;

  const auto result = SearchAraStar(problem, {});

  ASSERT_EQ(result.status, AraStarStatus::kSolved) << result.reason_code;
  ASSERT_EQ(result.candidates.size(), 2U);
  EXPECT_DOUBLE_EQ(result.candidates[0].cost, 10.0);
  EXPECT_DOUBLE_EQ(result.candidates[1].cost, 11.0);
  EXPECT_EQ(result.expanded_states, 3U);
}

TEST(AraStar, ChecksCancellationAtExpansionBoundary) {
  std::stop_source stop_source;
  stop_source.request_stop();

  const auto result =
      SearchAraStar(MakeDiamondProblem(), stop_source.get_token());

  EXPECT_EQ(result.status, AraStarStatus::kCanceled);
  EXPECT_EQ(result.reason_code, "REQUEST_CANCELED");
  EXPECT_EQ(result.expanded_states, 0U);
  EXPECT_TRUE(result.candidates.empty());
}

TEST(AraStar, ExhaustsFiniteGraphWithoutFormerCountCeilings) {
  constexpr std::size_t kStateCount = 257U;
  AraStarProblem problem;
  problem.state_count = kStateCount;
  problem.start_state = 0U;
  problem.goal_mask.assign(kStateCount, 0U);
  problem.goal_mask.back() = 1U;
  problem.heuristic.assign(kStateCount, 0.0);
  problem.outgoing_edges.resize(kStateCount);
  for (std::size_t state = 0U; state + 1U < kStateCount; ++state) {
    problem.outgoing_edges[state].push_back(GraphEdge{
        .target_state = state + 1U,
        .cost = 1.0,
        .stable_index = state,
    });
  }
  problem.config.initial_epsilon = 1.0;
  problem.config.target_epsilon = 1.0;

  const AraStarResult result = SearchAraStar(problem, {});

  ASSERT_EQ(result.status, AraStarStatus::kSolved) << result.reason_code;
  ASSERT_EQ(result.candidates.size(), 1U);
  EXPECT_EQ(result.candidates.front().states.size(), kStateCount);
  EXPECT_EQ(result.expanded_states, kStateCount - 1U);
}

TEST(BoundedQpSolver, ChecksCancellationAtIterationBoundary) {
  const BoundedQpProblem problem{
      .dimension = 1U,
      .hessian = {2.0},
      .gradient = {-1.0},
      .lower_bounds = {-1.0},
      .upper_bounds = {1.0},
  };
  const BoundedQpSettings settings{
      .maximum_iterations = 16U,
      .absolute_tolerance = 1.0e-10,
  };
  std::stop_source stop_source;
  stop_source.request_stop();

  const auto result =
      SolveBoundedQp(problem, settings, stop_source.get_token());

  EXPECT_EQ(result.termination, QpTermination::kCanceled);
  EXPECT_EQ(result.reason_code, "REQUEST_CANCELED");
  EXPECT_EQ(result.iterations, 0U);
}

}  // namespace
}  // namespace lunar::planning::shared
